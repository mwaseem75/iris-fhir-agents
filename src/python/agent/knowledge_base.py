"""
from config import IRIS_BASE, FHIR_AUTH, EMBEDDING_MODEL, RAG_GUIDELINES_CSV
knowledge_base.py — IRIS Vector Search RAG Knowledge Base
==========================================================
Implements Retrieval-Augmented Generation (RAG) for the clinical agents using
InterSystems IRIS as the vector database.

How it works:
  1. On startup, clinical guidelines are loaded from clinical_rag_guidelines.csv
     (50 guidelines from CDC, WHO, AHA, FDA, KDIGO and other authorities).
  2. Each guideline's content is embedded into a 1536-dimensional vector using
     OpenAI text-embedding-3-small and stored in RAG.VectorKnowledgeBase on IRIS.
  3. When an agent calls search_clinical_guidelines(), the query is embedded
     the same way and IRIS finds the most semantically similar guidelines using
     cosine similarity — so "chest tightness and sweating" retrieves AHA chest
     pain guidelines even though neither phrase matches the other exactly.
  4. If the vector table is unavailable (e.g. IRIS is restarting), a keyword
     fallback searches the in-memory CSV copy so agents are never left without
     any guidelines at all.

Why IRIS for vector search:
  Unlike a separate vector database (Pinecone, Weaviate), storing embeddings
  directly in IRIS means the guideline vectors live alongside the FHIR data in
  the same namespace. One system, one connection, no synchronisation overhead.

Why VARCHAR(8000) instead of LONGVARCHAR:
  IRIS LONGVARCHAR fields are stream objects that cannot be read back through
  the Atelier REST SQL API. VARCHAR(8000) is readable via REST and comfortably
  holds any clinical guideline paragraph.
"""

import os
import csv
import json
import httpx
from openai import OpenAI
from langchain.tools import tool
from config import IRIS_BASE, FHIR_AUTH, EMBEDDING_MODEL, RAG_GUIDELINES_CSV

# ── OpenAI client ─────────────────────────────────────────────────────────────
# text-embedding-3-small produces 1536-dimensional vectors — the same dimension
# we declared in the VECTOR(DOUBLE, 1536) column, so they align exactly.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── IRIS connection ───────────────────────────────────────────────────────────
# All SQL runs through the IRIS Atelier REST API rather than the native Python
# driver. This keeps the container image small (no IRIS client libraries needed)
# and works reliably across all IRIS versions that support Atelier.
# Connection settings imported from config.py
IRIS_AUTH = FHIR_AUTH  # Reuse the shared auth tuple

# Guidelines CSV is mounted from the host at data/guidelines/ via docker-compose.
# Keeping it outside the image means new guidelines can be added without
# rebuilding — just edit the CSV and restart the API container.
# RAG_GUIDELINES_CSV imported from config.py


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_guidelines_from_csv() -> list:
    """
    Read the clinical guidelines CSV into memory.

    The CSV has four columns: id, source, topic, content.
    All values are stripped of leading/trailing whitespace to prevent
    embedding differences caused by invisible characters — a subtle bug
    that would make identical guidelines appear dissimilar to the vector search.

    Returns an empty list (rather than raising) so callers can decide
    whether a missing CSV is fatal or recoverable.
    """
    guidelines = []
    try:
        with open(RAG_GUIDELINES_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                guidelines.append({
                    "id":      row["id"].strip(),
                    "source":  row["source"].strip(),
                    "topic":   row["topic"].strip(),
                    "content": row["content"].strip()
                })
        print(f"RAG: Loaded {len(guidelines)} guidelines from CSV")
    except FileNotFoundError:
        print(f"RAG: WARNING — CSV not found at {RAG_GUIDELINES_CSV}. No guidelines loaded.")
    except Exception as e:
        print(f"RAG: ERROR reading CSV: {e}")
    return guidelines


# ═══════════════════════════════════════════════════════════════════════════════
#  EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════════

def get_embedding(text: str) -> list:
    """
    Convert a text string into a 1536-dimensional embedding vector.

    We use text-embedding-3-small rather than the larger ada-002 or 3-large
    because it hits the right balance: fast enough for real-time query
    embedding (~100ms), cheap enough to run 50 embeddings at startup, and
    accurate enough for clinical text similarity.

    The explicit float() cast is necessary — OpenAI returns Decimal-like
    objects in some versions that IRIS rejects when passed to TO_VECTOR().
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return [float(x) for x in response.data[0].embedding]


# ═══════════════════════════════════════════════════════════════════════════════
#  IRIS SQL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def iris_sql_query(query: str, params: list = None) -> list:
    """
    Run a SELECT query against IRIS via the Atelier REST API.

    The Atelier endpoint accepts parameterised queries which protects against
    SQL injection and lets IRIS cache the execution plan across repeated calls.
    Raises on HTTP error or SQL error so callers see a clear exception rather
    than silently receiving empty results.
    """
    url = f"{IRIS_BASE}/api/atelier/v1/FHIRSERVER/action/query"
    payload = {"query": query, "parameters": params or []}
    r = httpx.post(url, json=payload, auth=IRIS_AUTH, timeout=30)
    r.raise_for_status()
    data = r.json()

    # IRIS signals SQL errors in the response body, not the HTTP status code
    errors = data.get("status", {}).get("errors", [])
    if errors:
        raise Exception(f"SQL Error: {errors[0].get('error', 'Unknown')}")

    return data.get("result", {}).get("content", [])


def iris_sql_execute(query: str, params: list = None):
    """
    Run a DDL or DML statement against IRIS (CREATE TABLE, INSERT, etc.).

    Same transport as iris_sql_query — the Atelier API handles both reads
    and writes through the same endpoint. We keep them as separate functions
    to make call sites self-documenting.
    """
    url = f"{IRIS_BASE}/api/atelier/v1/FHIRSERVER/action/query"
    payload = {"query": query, "parameters": params or []}
    r = httpx.post(url, json=payload, auth=IRIS_AUTH, timeout=30)
    r.raise_for_status()
    return r.json()


# ═══════════════════════════════════════════════════════════════════════════════
#  KEYWORD FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

def keyword_search(query: str, guidelines: list) -> list:
    """
    Simple word-overlap search against the in-memory guidelines list.

    This runs when IRIS Vector Search is unavailable — for example, if IRIS
    is still initialising when the first chat message arrives. It's intentionally
    simple: count how many query words (>3 chars) appear in each guideline's
    topic and content, then return the top 3 by score.

    The synthetic similarity score (0.5 + 0.05 per matching word, capped at 0.95)
    is lower than a real vector similarity would be, which signals to the agent
    that this result is less certain than a proper semantic match.
    """
    keywords = [w.lower() for w in query.split() if len(w) > 3]
    scored = []
    for g in guidelines:
        text = (g["topic"] + " " + g["content"]).lower()
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scored.append((score, {**g, "similarity": min(0.5 + score * 0.05, 0.95)}))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [r for _, r in scored[:3]]
    print(f"RAG: Keyword fallback found {len(results)} result(s)")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  INITIALISATION — runs once when the module is first imported
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_knowledge_base():
    """
    Bootstrap the RAG.VectorKnowledgeBase table and populate it from the CSV.

    Called automatically at the bottom of this file so the knowledge base is
    ready before the first HTTP request arrives. The logic is idempotent:
      - If the table doesn't exist, create it.
      - If it already has as many rows as the CSV, skip loading.
      - If it's partially loaded (e.g. previous startup was interrupted),
        insert only the missing rows by checking each id individually.

    This means the API container can be restarted without re-embedding
    guidelines that are already in IRIS — each embedding costs an OpenAI
    API call, so avoiding redundant work keeps startup fast and cheap.
    """
    guidelines = load_guidelines_from_csv()
    if not guidelines:
        print("RAG: No guidelines to load — check CSV path.")
        return

    try:
        # ── Create table if it doesn't exist ─────────────────────────────────
        # We catch the exception rather than using IF NOT EXISTS because some
        # IRIS versions don't support that DDL syntax via the Atelier API.
        try:
            iris_sql_execute("""
                CREATE TABLE RAG.VectorKnowledgeBase (
                    id        VARCHAR(100) PRIMARY KEY,
                    source    VARCHAR(200),
                    topic     VARCHAR(200),
                    content   VARCHAR(8000),
                    embedding VECTOR(DOUBLE, 1536)
                )
            """)
            print("RAG: Table RAG.VectorKnowledgeBase created")
        except Exception:
            print("RAG: Table already exists — skipping CREATE")

        # ── Check how many guidelines are already stored ──────────────────────
        rows = iris_sql_query("SELECT COUNT(*) AS cnt FROM RAG.VectorKnowledgeBase")
        existing_count = int(rows[0].get("cnt", 0)) if rows else 0

        if existing_count >= len(guidelines):
            print(f"RAG: Knowledge base ready — {existing_count} guidelines already in IRIS")
            return

        # ── Embed and insert any missing guidelines ───────────────────────────
        print(f"RAG: Embedding {len(guidelines)} guidelines into IRIS Vector Search...")
        loaded = 0
        for g in guidelines:
            try:
                # Skip rows that are already present (partial load recovery)
                exists = iris_sql_query(
                    "SELECT COUNT(*) AS cnt FROM RAG.VectorKnowledgeBase WHERE id = ?",
                    [g["id"]]
                )
                if int(exists[0].get("cnt", 0)) > 0:
                    continue

                # Embed the guideline content — this is what gets searched later
                embedding = get_embedding(g["content"])

                # TO_VECTOR(?, DOUBLE) tells IRIS the type explicitly.
                # Without the DOUBLE qualifier some versions infer FLOAT32
                # which silently truncates precision and corrupts similarity scores.
                iris_sql_execute("""
                    INSERT INTO RAG.VectorKnowledgeBase
                        (id, source, topic, content, embedding)
                    VALUES (?, ?, ?, ?, TO_VECTOR(?, DOUBLE))
                """, [
                    g["id"], g["source"], g["topic"], g["content"],
                    json.dumps(embedding)
                ])

                loaded += 1
                print(f"  [{loaded}/{len(guidelines)}] {g['topic']}")

            except Exception as e:
                # Log and continue — one bad guideline shouldn't abort the whole load
                print(f"  WARNING: Could not load {g['id']}: {e}")

        print(f"RAG: Initialisation complete — {loaded} new guidelines embedded and stored")

    except Exception as e:
        print(f"RAG: Initialisation error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  LANGCHAIN TOOL — exposed to all three clinical agents
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def search_clinical_guidelines(query: str) -> str:
    """
    Search clinical guidelines from CDC, WHO, AHA, FDA and other medical authorities
    using IRIS Vector Search with semantic similarity.
    ALWAYS use this tool before making any clinical recommendation, symptom assessment,
    or drug interaction check. Cite the source in your response.
    """
    try:
        print(f"RAG: Searching for '{query}'")

        # ── Primary path: IRIS Vector Search ─────────────────────────────────
        # Embed the query and find the top 3 guidelines by cosine similarity.
        # VECTOR_COSINE returns values from -1 (opposite) to 1 (identical);
        # we filter below 0.1 to exclude genuinely unrelated results that
        # happen to share a few common medical words.
        try:
            query_embedding = get_embedding(query)
            embedding_str = json.dumps(query_embedding)

            rows = iris_sql_query("""
                SELECT TOP 3
                    id, source, topic, content,
                    VECTOR_COSINE(embedding, TO_VECTOR(?, DOUBLE)) AS similarity
                FROM RAG.VectorKnowledgeBase
                ORDER BY VECTOR_COSINE(embedding, TO_VECTOR(?, DOUBLE)) DESC
            """, [embedding_str, embedding_str])

            # Discard low-confidence matches
            rows = [r for r in rows if float(r.get("similarity", 0)) > 0.1]
            print(f"RAG: Vector search returned {len(rows)} relevant result(s)")

        except Exception as e:
            print(f"RAG: Vector search failed ({e}) — falling back to keyword search")
            rows = []

        # ── Fallback: keyword search over in-memory CSV ───────────────────────
        if not rows:
            guidelines = load_guidelines_from_csv()
            rows = keyword_search(query, guidelines)

        if not rows:
            return "No relevant clinical guidelines found."

        # ── Format results for the agent ──────────────────────────────────────
        # The agent's system prompt instructs it to cite sources using this
        # exact format, which the frontend then renders as a styled citation block.
        output = []
        for row in rows:
            source    = row.get("source", "Unknown")
            topic     = row.get("topic", "")
            content   = row.get("content", "")
            relevance = round(float(row.get("similarity", 0.5)) * 100, 1)
            output.append(
                f"[{source}] (Relevance: {relevance}%)\n"
                f"Topic: {topic}\n"
                f"{content}"
            )

        print(f"RAG: Returning {len(output)} guideline(s) to agent")
        return "\n\n---\n\n".join(output)

    except Exception as e:
        print(f"RAG: Search error: {e}")
        return f"Error searching guidelines: {str(e)}"


# Run on import — the agents import this module, triggering initialisation
# before the first chat message is processed.
initialize_knowledge_base()