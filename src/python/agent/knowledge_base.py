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
     OpenAI text-embedding-3-small and stored in RAG.ClinicalGuidelines on IRIS.
  3. When an agent calls search_clinical_guidelines(), the query is embedded
     the same way and IRIS finds the most semantically similar guidelines using
     cosine similarity — so "chest tightness and sweating" retrieves AHA chest
     pain guidelines even though neither phrase matches the other exactly.
  4. If the vector table is unavailable (e.g. IRIS is restarting), a keyword
     fallback searches the in-memory CSV copy so agents are never left without
     any guidelines at all.

Load path — two strategies tried in order:
  PRIMARY:  Embedded Python (iris.sql.exec) — runs inside the IRIS process
            with direct in-process SQL access. No HTTP, no network round-trips.
            Only available when irispython is the interpreter (IRIS container).
  FALLBACK: External REST (httpx → Atelier API) — runs from the API container.
            Used when the iris module is not available.

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
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── IRIS connection (REST path) ───────────────────────────────────────────────
IRIS_AUTH = FHIR_AUTH

# ── Embedded Python availability flag ────────────────────────────────────────
# Attempt to import the iris module — only succeeds when running inside the
# IRIS process via irispython. When running in the external API container,
# this import fails and we fall back to the REST path transparently.
try:
    import iris as _iris_module
    # Verify iris.sql and iris.system are actually available.
    # The iris __init__.py can be imported outside IRIS but iris.sql
    # is a native C extension that only works inside the IRIS process.
    _ = _iris_module.sql
    _ = _iris_module.system
    EMBEDDED_PYTHON_AVAILABLE = True
    print("RAG: Embedded Python available — iris.sql confirmed (running inside IRIS process)")
except (ImportError, AttributeError):
    _iris_module = None
    EMBEDDED_PYTHON_AVAILABLE = False
    print("RAG: Embedded Python not available — using REST path (running in API container)")


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_guidelines_from_csv() -> list:
    """
    Read the clinical guidelines CSV into memory.

    The CSV has four columns: id, source, topic, content.
    All values are stripped of leading/trailing whitespace to prevent
    embedding differences caused by invisible characters.
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

    text-embedding-3-small: fast (~100ms), cheap, accurate for clinical text.
    The explicit float() cast ensures IRIS accepts the values in TO_VECTOR().
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return [float(x) for x in response.data[0].embedding]


# ═══════════════════════════════════════════════════════════════════════════════
#  IRIS SQL HELPERS — REST path (API container)
# ═══════════════════════════════════════════════════════════════════════════════

def iris_sql_query(query: str, params: list = None) -> list:
    """Run a SELECT via the Atelier REST API."""
    url = f"{IRIS_BASE}/api/atelier/v1/FHIRSERVER/action/query"
    payload = {"query": query, "parameters": params or []}
    r = httpx.post(url, json=payload, auth=IRIS_AUTH, timeout=30)
    r.raise_for_status()
    data = r.json()
    errors = data.get("status", {}).get("errors", [])
    if errors:
        raise Exception(f"SQL Error: {errors[0].get('error', 'Unknown')}")
    return data.get("result", {}).get("content", [])


def iris_sql_execute(query: str, params: list = None):
    """Run a DDL or DML statement via the Atelier REST API."""
    url = f"{IRIS_BASE}/api/atelier/v1/FHIRSERVER/action/query"
    payload = {"query": query, "parameters": params or []}
    r = httpx.post(url, json=payload, auth=IRIS_AUTH, timeout=30)
    r.raise_for_status()
    return r.json()


# ═══════════════════════════════════════════════════════════════════════════════
#  EMBEDDED PYTHON LOAD PATH — runs inside IRIS process
# ═══════════════════════════════════════════════════════════════════════════════

def _embedded_count() -> int:
    """
    Count existing guidelines using iris.sql — direct in-process SQL.
    No HTTP round-trip — the query executes inside the IRIS SQL engine directly.
    """
    try:
        rs = _iris_module.sql.exec(
            "SELECT COUNT(*) AS cnt FROM RAG.ClinicalGuidelines"
        )
        for row in rs:
            return int(row[0])
    except Exception as e:
        print(f"RAG: Embedded count error: {e}")
    return 0


def _embedded_row_exists(row_id: str) -> bool:
    """Check if a guideline ID already exists — direct iris.sql."""
    try:
        rs = _iris_module.sql.exec(
            "SELECT COUNT(*) FROM RAG.ClinicalGuidelines WHERE id = ?",
            [row_id]
        )
        for row in rs:
            return int(row[0]) > 0
    except:
        pass
    return False


def _embedded_insert(row_id: str, source: str, topic: str,
                     content: str, embedding: list) -> None:
    """
    Insert one guideline directly into IRIS using Embedded Python iris.sql.

    This is the key difference from the REST path:
      REST path:     Python → HTTP → Atelier API → IRIS SQL engine
      Embedded path: Python → iris.sql.exec() → IRIS SQL engine (in-process)

    TO_VECTOR(?, DOUBLE) converts the JSON float array into an IRIS VECTOR type.
    DOUBLE ensures 64-bit precision — without it IRIS may infer FLOAT32 and
    silently truncate precision, corrupting similarity scores.
    """
    _iris_module.sql.exec(
        "INSERT INTO RAG.ClinicalGuidelines "
        "(id, source, topic, content, embedding) "
        "VALUES (?, ?, ?, ?, TO_VECTOR(?, DOUBLE))",
        [row_id, source, topic, content, json.dumps(embedding)]
    )


def load_via_embedded_python(guidelines: list) -> int:
    """
    PRIMARY load path — Embedded Python (iris.sql).

    Runs inside the IRIS process with direct SQL engine access.
    Called when `import iris` succeeds (irispython interpreter).

    Returns the number of guidelines newly inserted.
    """
    print("RAG: PRIMARY path — Embedded Python iris.sql (in-process, no HTTP)")

    # Switch to FHIRSERVER namespace — only possible via Embedded Python
    try:
        _iris_module.system.Process.SetNamespace("FHIRSERVER")
        print("RAG: Embedded Python — switched to FHIRSERVER namespace")
    except Exception as e:
        print(f"RAG: Namespace switch warning: {e}")

    existing = _embedded_count()
    print(f"RAG: Embedded Python — {existing} / {len(guidelines)} guidelines already in IRIS")

    if existing >= len(guidelines):
        print(f"RAG: Embedded Python — knowledge base complete, skipping load")
        return 0

    loaded = 0
    for g in guidelines:
        try:
            if _embedded_row_exists(g["id"]):
                continue

            # Call OpenAI from inside IRIS — Embedded Python reaching external API
            embedding = get_embedding(g["content"])

            # Write directly to IRIS SQL engine — zero HTTP overhead
            _embedded_insert(
                g["id"], g["source"], g["topic"],
                g["content"], embedding
            )
            loaded += 1
            print(f"  Embedded [{loaded}/{len(guidelines)}] {g['source']} — {g['topic']}")

        except Exception as e:
            print(f"  RAG: Embedded WARNING — could not load {g['id']}: {e}")

    print(f"RAG: Embedded Python load complete — {loaded} guidelines inserted via iris.sql")
    return loaded


# ═══════════════════════════════════════════════════════════════════════════════
#  REST LOAD PATH — runs from API container
# ═══════════════════════════════════════════════════════════════════════════════

def load_via_rest(guidelines: list) -> int:
    """
    FALLBACK load path — external REST via Atelier API.

    Used when the iris module is not available (API container).
    Functionally identical outcome — same data in same table —
    but goes through HTTP rather than direct in-process SQL.

    Returns the number of guidelines newly inserted.
    """
    print("RAG: FALLBACK path — REST via Atelier API (HTTP from API container)")

    # Create table if needed
    try:
        iris_sql_execute("""
            CREATE TABLE RAG.ClinicalGuidelines (
                id        VARCHAR(100) PRIMARY KEY,
                source    VARCHAR(200),
                topic     VARCHAR(200),
                content   VARCHAR(8000),
                embedding VECTOR(DOUBLE, 1536)
            )
        """)
        print("RAG: Table RAG.ClinicalGuidelines created")
    except Exception:
        print("RAG: Table already exists — skipping CREATE")

    # Check existing count
    rows = iris_sql_query("SELECT COUNT(*) AS cnt FROM RAG.ClinicalGuidelines")
    existing_count = int(rows[0].get("cnt", 0)) if rows else 0

    if existing_count >= len(guidelines):
        print(f"RAG: REST — knowledge base ready, {existing_count} guidelines already in IRIS")
        return 0

    print(f"RAG: REST — embedding {len(guidelines)} guidelines into IRIS Vector Search...")
    loaded = 0
    for g in guidelines:
        try:
            exists = iris_sql_query(
                "SELECT COUNT(*) AS cnt FROM RAG.ClinicalGuidelines WHERE id = ?",
                [g["id"]]
            )
            if int(exists[0].get("cnt", 0)) > 0:
                continue

            embedding = get_embedding(g["content"])

            iris_sql_execute("""
                INSERT INTO RAG.ClinicalGuidelines
                    (id, source, topic, content, embedding)
                VALUES (?, ?, ?, ?, TO_VECTOR(?, DOUBLE))
            """, [
                g["id"], g["source"], g["topic"], g["content"],
                json.dumps(embedding)
            ])

            loaded += 1
            print(f"  REST [{loaded}/{len(guidelines)}] {g['topic']}")

        except Exception as e:
            print(f"  RAG: REST WARNING — could not load {g['id']}: {e}")

    print(f"RAG: REST load complete — {loaded} new guidelines embedded and stored")
    return loaded


# ═══════════════════════════════════════════════════════════════════════════════
#  KEYWORD FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

def keyword_search(query: str, guidelines: list) -> list:
    """
    Simple word-overlap search against the in-memory guidelines list.
    Runs when IRIS Vector Search is unavailable.
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
    Bootstrap RAG.ClinicalGuidelines using the best available path.

    Strategy:
      1. Try Embedded Python (iris.sql) — direct in-process SQL inside IRIS.
         This is the PRIMARY path and demonstrates InterSystems Embedded Python.
      2. Fall back to REST (Atelier API) — if iris module not available.

    Both paths produce identical results — 50 guidelines embedded as
    VECTOR(DOUBLE, 1536) in RAG.ClinicalGuidelines, ready for VECTOR_COSINE
    similarity search.

    The load is idempotent — existing rows are skipped so container restarts
    do not re-embed guidelines that are already in IRIS.
    """
    guidelines = load_guidelines_from_csv()
    if not guidelines:
        print("RAG: No guidelines to load — check CSV path.")
        return

    try:
        if EMBEDDED_PYTHON_AVAILABLE:
            # ── PRIMARY: Embedded Python ──────────────────────────────────────
            # iris.sql.exec() runs inside the IRIS process — no HTTP required.
            # Demonstrates InterSystems Embedded Python integration.
            load_via_embedded_python(guidelines)
        else:
            # ── FALLBACK: REST via Atelier API ────────────────────────────────
            # httpx → Atelier REST → IRIS SQL engine.
            # Used when running in the external API container.
            load_via_rest(guidelines)

        print("RAG: Initialisation complete")

    except Exception as e:
        print(f"RAG: Initialisation error: {e}")
        # Last resort — try the other path if primary failed
        if EMBEDDED_PYTHON_AVAILABLE:
            print("RAG: Embedded Python failed — trying REST fallback...")
            try:
                load_via_rest(guidelines)
            except Exception as e2:
                print(f"RAG: REST fallback also failed: {e2}")


# ═══════════════════════════════════════════════════════════════════════════════
#  LANGCHAIN TOOL — exposed to all clinical agents
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

        # ── Primary: IRIS Vector Search ───────────────────────────────────────
        try:
            query_embedding = get_embedding(query)
            embedding_str = json.dumps(query_embedding)

            rows = iris_sql_query("""
                SELECT TOP 3
                    id, source, topic, content,
                    VECTOR_COSINE(embedding, TO_VECTOR(?, DOUBLE)) AS similarity
                FROM RAG.ClinicalGuidelines
                ORDER BY VECTOR_COSINE(embedding, TO_VECTOR(?, DOUBLE)) DESC
            """, [embedding_str, embedding_str])

            rows = [r for r in rows if float(r.get("similarity", 0)) > 0.1]
            print(f"RAG: Vector search returned {len(rows)} relevant result(s)")

        except Exception as e:
            print(f"RAG: Vector search failed ({e}) — falling back to keyword search")
            rows = []

        # ── Fallback: keyword search ──────────────────────────────────────────
        if not rows:
            guidelines = load_guidelines_from_csv()
            rows = keyword_search(query, guidelines)

        if not rows:
            return "No relevant clinical guidelines found."

        # ── Format for agent ──────────────────────────────────────────────────
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


# Run on import — initialises before the first HTTP request arrives
initialize_knowledge_base()