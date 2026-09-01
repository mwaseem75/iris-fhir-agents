# Learning Guide — Agentic Architecture of *IRIS FHIR Agents*

This document has three parts:

1. **The LangChain agentic framework** — the concepts this project is built on.
2. **File-by-file walkthrough** of `src/python/`, focused on the *agentic* role each file plays.
3. **Gap analysis against a production agentic-engineering job description** (Doctolib Assistant) — what this codebase already demonstrates and what it does not.

---

## Part 1 — The LangChain Agentic Framework

### 1.1 What "agentic" means here

A program is **agentic** when an LLM — not the surrounding Python — decides *what to do next*: which function (tool) to call, with which arguments, in which order, and when the task is finished. The Python code supplies **capabilities and guardrails**; the model supplies **control flow**.

Contrast:

| Scripted | Agentic |
|---|---|
| `patient = get_patient(id); allergies = get_allergies(id); return summarize(...)` | Give the model `get_patient`, `get_allergies`, a goal, and let it choose the calls and the stopping point |
| Flow is fixed at code-writing time | Flow is decided at run time, per message, from tool results |

### 1.2 Core primitives

| Primitive | LangChain object | Role in this repo |
|---|---|---|
| **LLM** | `ChatOpenAI(model="gpt-4o-mini", temperature=…)` | The reasoning engine. One instance per agent, each with a task-tuned temperature. |
| **Tool** | `@tool`-decorated function | A capability the model may invoke. Signature + docstring become the JSON schema the model sees. In this repo: FHIR reads/writes, drug-interaction checks, RAG search, IRIS SQL. |
| **Prompt** | `ChatPromptTemplate.from_messages([...])` | Fixes the message layout: `system` rules, `MessagesPlaceholder("chat_history")`, the `human` turn, and `MessagesPlaceholder("agent_scratchpad")`. |
| **Agent** | `create_openai_tools_agent(llm, tools, prompt)` | Binds the tools' JSON schemas to the model using OpenAI's native tool-calling. Produces a *runnable* that, given state, emits either tool calls or a final answer. |
| **Executor** | `AgentExecutor(agent, tools, memory, max_iterations, handle_parsing_errors)` | The **loop**. Runs the agent, executes the tool calls it asks for, feeds results back, repeats until a final answer or `max_iterations`. |
| **Memory** | `ConversationBufferMemory(memory_key="chat_history", return_messages=True)` | Verbatim transcript of the conversation, injected on every turn so the agent remembers the patient ID, prior findings, etc. |
| **Scratchpad** | `agent_scratchpad` placeholder | *Within a single turn*, the running list of `(tool call, tool result)` pairs. This is what makes the loop "reactive" — the model sees what its previous tool calls returned before deciding the next one. |

### 1.3 The tool-calling loop (what `AgentExecutor.invoke` actually does)

```
invoke({"input": user_message}):
    scratchpad = []
    for i in range(max_iterations):           # 8–10 here
        llm_output = llm(system + chat_history + input + scratchpad)
        if llm_output is a final answer:
            memory.save(input, llm_output)
            return llm_output
        for call in llm_output.tool_calls:    # model chose these
            result = tools[call.name](**call.args)   # Python runs the capability
            scratchpad.append((call, result))
    return "stopped: max iterations"           # fail-safe ceiling
```

Key properties:

- **The model picks the tools.** The system prompt describes a *desired* workflow in English ("fetch patient data, then search guidelines, then assess"); it does not call anything itself.
- **It branches on results.** "Allergy: Aspirin (high)" in the scratchpad can make the model then call `check_drug_interactions` — a path never written in Python.
- **`max_iterations`** is a safety ceiling because loop length is unknowable in advance.
- **`handle_parsing_errors=True`** feeds malformed tool JSON back to the model as an error instead of raising.
- **Tools return strings and never raise** (repo convention) so a FHIR 404 becomes text the model can explain, not a 500.

### 1.4 RAG as a tool

Retrieval here is **not** a preprocessing step wired into every request. `search_clinical_guidelines` is a `@tool`; the agent decides *when* it needs grounding and *what* query to embed. The prompts push hard ("ALWAYS use this before any recommendation") but the call is the model's choice.

### 1.5 Multi-agent pattern

Two layers:

- **Router** (`orchestrator.route_message`): a separate `temperature=0` LLM call whose entire job is to output one word — `TRIAGE | SPECIALIST | PHARMACY | <custom name>`. This is "LLM-as-classifier": a single shot, no loop.
- **Specialist executors**: the chosen agent runs its own full tool-calling loop (§1.3).

The orchestrator also owns **cross-agent state** the individual `ConversationBufferMemory` instances cannot see: the patient ID, the turn count, the last agent used.

---

## Part 2 — File-by-File Walkthrough (`src/python/`)

```
src/python/
├── requirements.txt
├── api/
│   ├── __init__.py
│   └── main.py          ← HTTP surface, SSE, autonomous agent trigger
├── agent/
│   ├── __init__.py
│   ├── config.py        ← settings + per-agent temperatures
│   ├── orchestrator.py  ← router + session state + context injection
│   ├── triage_agent.py  ← intake agent      (temp 0.3)
│   ├── specialist_agent.py ← condition analysis (temp 0.2)
│   ├── pharmacy_agent.py   ← drug safety     (temp 0.1) + safety tables
│   ├── fhir_agent.py       ← population explorer (temp 0.2), 11 tools
│   ├── fhir_tools.py       ← shared FHIR read/write tools
│   ├── knowledge_base.py   ← RAG: embed + VECTOR_COSINE + fallback
│   └── dynamic_agent.py    ← user-defined agents at runtime
└── iris/__init__.py        ← InterSystems Embedded Python shim
```

### 2.1 `requirements.txt`

Declares the agentic stack: `langchain` + `langchain-openai` (agent constructors, `ChatOpenAI`), `langchain-community` (misc tools/loaders), `openai` (embeddings + chat), `tiktoken` (token counting for the OpenAI client), `fastapi`/`uvicorn` (HTTP), `httpx` (all FHIR + Atelier calls). No evaluation, tracing, or observability libraries are present — see Part 3.

### 2.2 `api/__init__.py` / `agent/__init__.py`

Empty package markers. No logic.

### 2.3 `api/main.py` — the HTTP surface and the *autonomous* trigger

Not an agent itself; it is where agents are **invoked**.

| Route | Agentic relevance |
|---|---|
| `POST /chat` → `chat_endpoint` (`main.py:203`) | Mints a `session_id` on turn 1, calls `orchestrate(session_id, message)`. The single entry point for the clinical multi-agent system. |
| `GET /session/{id}/new` (`main.py:230`) | Wipes the session from **all four** in-memory stores (`session_context`, `sessions`, `specialist_sessions`, `pharmacy_sessions`) — the only "memory management" in the system. |
| `POST /fhir-agent/chat` (`main.py:733`) | Invokes the standalone FHIR Server Agent (`fhir_agent.chat`), session-per-tab. |
| `POST /agents/create`, `/agents/{id}/test`, `DELETE /agents/{id}` (`main.py:121-170`) | CRUD for **dynamic agents** — writes/reads `data/custom_agents.json`, runs a one-off test invocation. |
| `GET /vitals/stream/{patient_id}` (`main.py:582`) | SSE loop. Every 2 s it generates vitals, writes 5 FHIR Observations, and — on a critical reading — schedules `trigger_ai_alert`. |
| `trigger_ai_alert` (`main.py:657`) | **Autonomous agent invocation.** Builds an alert message, then `loop.run_in_executor(None, lambda: orchestrate(alert_session_id, context_msg))` — the Triage agent runs its full tool loop with **no human in the loop**, writing its assessment to `ai_alerts_log` (in-memory, capped at 20). Blocking LLM work is pushed to a thread so other SSE streams are not stalled. |
| `GET /analytics/*`, `/fhir/metadata`, `/vitals/alerts` | Plain FHIR/REST aggregation and polling. **No LLM.** |

Agentic patterns visible here: **event-driven agent invocation**, **thread-offload of blocking LLM calls**, **fire-and-forget task with a cooldown** (`alert_cooldown = 15` ticks ≈ 30 s), **swallowed FHIR write errors** so persistence failure never breaks the stream.

### 2.4 `agent/config.py` — reasoning configuration

Centralises every knob. The agentic-relevant part is the **temperature ladder**, which encodes how much freedom each reasoning role gets:

```
TEMP_ROUTER     = 0.0   # deterministic single-word classification
TEMP_PHARMACY   = 0.1   # drug safety is close to binary
TEMP_SPECIALIST = 0.2   # multi-condition clinical reasoning
TEMP_TRIAGE     = 0.3   # patient-facing, needs a natural register
```

Also: `LLM_MODEL` (`gpt-4o-mini`), `EMBEDDING_MODEL` (`text-embedding-3-small`, must match the `VECTOR(DOUBLE, 1536)` column), FHIR/IRIS URLs and credentials. Credentials default to `_SYSTEM` / `SYS` in source — see Part 3.

### 2.5 `agent/orchestrator.py` — multi-agent coordination

The coordination layer. Responsibilities:

- **`session_context: dict`** (`orchestrator.py:91`) — per-session `{last_agent, patient_id, turn_count}`. In-memory, no TTL, no persistence.
- **`extract_patient_id(message)`** (`:98`) — regex, five lead-in patterns. Deterministic, not agentic. Runs every turn; a found ID is stored on the session forever.
- **`_build_router_prompt()`** (`:56`) — rebuilt **every turn** so newly created dynamic agents appear in the router's option list immediately, no restart.
- **`route_message(message)`** (`:129`) — the LLM router call. Normalises the reply, validates it against `{TRIAGE, SPECIALIST, PHARMACY} ∪ {custom agent names}`, and **falls back to `TRIAGE`** on anything unexpected.
- **`inject_patient_context(message, patient_id)`** (`:169`) — prepends `"[Context: Patient ID is X. Do NOT ask for patient ID again.] "` so downstream agents never re-ask. Only injected if the ID is not already in the text.
- **`orchestrate(session_id, message)`** (`:194`) — the turn pipeline:
  1. init/load session context, `turn_count += 1`
  2. extract + store patient ID
  3. route: **turn 1 is hard-coded to `TRIAGE`**; turn ≥ 2 uses `route_message`
  4. enrich: SPECIALIST/PHARMACY/custom always get the context prefix; TRIAGE gets it from turn 3
  5. dispatch to `triage_chat` / `run_specialist` / `run_pharmacy` / `run_custom_agent`
  6. if a routed custom agent was deleted mid-turn → **fall back to `TRIAGE`**
  7. return `{response, agent_used, session_id, turn}`

Agentic patterns: **hierarchical control** (deterministic router → autonomous executor), **shared blackboard state** the sub-agents cannot see, **defensive fallbacks** at both routing and dispatch.
Limits: single-shot router with no confidence score or escalation; no agent-to-agent handoff protocol; sub-agents do **not** share `ConversationBufferMemory` (the code comments call this out).

### 2.6 `agent/triage_agent.py` — the intake agent (`temp 0.3`)

- **Tools** (`triage_agent.py:74`): `get_patient`, `get_patient_conditions`, `get_patient_allergies`, `get_patient_medications`, `create_triage_observation`, `create_service_request`, `search_clinical_guidelines`.
- **System prompt** (`:89`): an 8-step workflow, hard rules ("if `[Context: Patient ID is X]` … immediately fetch, never ask"), a mandatory language rule (respond in the message's language, keep an English handoff section), and a **mandatory citation format** — `> According to [Source] (Relevance: X%) — …`.
- **Assembly**: `ChatPromptTemplate` with the four message slots → `create_openai_tools_agent` → `AgentExecutor(max_iterations=10, handle_parsing_errors=True, verbose=True)`.
- **Session store**: `sessions: dict[str, AgentExecutor]` (`:163`); `get_or_create_session` (`:195`) makes one executor + one memory per `session_id`.
- **`chat(session_id, message)`** (`:208`) — `get_or_create_session(...).invoke({"input": message})["output"]`.

This is the **canonical agent shape**; the other three clinical agents differ only in prompt, tools, and temperature. It is also the agent the autonomous vitals trigger drives.

### 2.7 `agent/specialist_agent.py` — condition analysis (`temp 0.2`)

- Same tool set as Triage.
- **`SPECIALIST_PROMPT`** (`specialist_agent.py:72`) is stricter about evidence: "for EACH active condition call `search_clinical_guidelines`", "include at least 2–3 citation blocks", "a response WITHOUT citation blocks is INCORRECT". It even ships a worked example response.
- Store: `specialist_sessions` (`:158`); `run_specialist(session_id, message)` (`:190`).
- `max_iterations=10` — sized for complex patients (fetch × 4 → RAG per condition → multiple `ServiceRequest` writes).

Illustrates **prompt-as-policy**: the guardrail ("always cite") is English text, enforced only by the model's compliance — not validated in code.

### 2.8 `agent/pharmacy_agent.py` — drug safety (`temp 0.1`)

The most safety-critical agent, and the one with the most **deterministic tools**:

- **`KNOWN_INTERACTIONS`** (`pharmacy_agent.py:68`) — hard-coded `(drug_a, drug_b) → warning` table; `check_drug_interactions` (`:80`) does O(n²) substring matching over a comma-separated med list.
- **`ALLERGY_CROSS_REACTIVITY`** (`:110`) — maps an allergen class to specific drugs (`penicillin → [amoxicillin, ampicillin, …]`); `check_allergy_medication_conflict` (`:118`) fetches `AllergyIntolerance` and matches both directions.
- **`create_medication_request`** (`:156`) — writes a FHIR `MedicationRequest` with `intent="proposal"` (an AI suggestion, explicitly not a signed order).
- **`PHARMACY_PROMPT`** (`:188`): fetch meds + allergies → `search_clinical_guidelines` → apply FDA guidance → flag `HIGH RISK` / `MODERATE` / `LOW` → cite source + relevance.
- Store: `pharmacy_sessions` (`:266`); `run_pharmacy` (`:298`).

Pattern: **agent + rule-based backstops**. The lookup tables give the model deterministic ground truth so the safety verdict does not rest solely on LLM recall; the *decision to call them* is still the model's.

### 2.9 `agent/fhir_agent.py` — the population/exploration agent (`temp 0.2`)

Standalone (not behind the orchestrator). Reachable at `POST /fhir-agent/chat`.

- **11 tools** (`fhir_agent.py:557`): `get_patient_list`, `get_conditions_for_patient`, `get_medications_for_patient`, `get_observations_for_patient`, `get_allergies_for_patient`, `get_procedures_for_patient`, `search_patients_by_condition` (fetch-all + client-side substring match), `get_fhir_statistics` (`_summary=count` per resource), `get_full_patient_summary` (5 FHIR GETs in one tool), `query_iris_sql`, `search_clinical_guidelines`.
- **`query_iris_sql`** (`:471`) — **fail-safe by allow-list**: rejects anything not starting with `SELECT` *before* hitting the Atelier API (which runs with DBA rights).
- **`SYSTEM_PROMPT`** (`:516`): "bridge between natural language and the IRIS FHIR server"; mandatory-workflow and response-format sections; cite the FHIR resource type used.
- Session store + `AgentExecutor(max_iterations=8)` in `get_or_create_session` (`:589`); `chat(session_id, message)` (`:616`).

Pattern: **open-ended read-mostly agent** with one guarded escape hatch to raw SQL.

### 2.10 `agent/fhir_tools.py` — the tool adapter layer

The **adapter** between FHIR and the agents. Design conventions (documented in the file header):

- **`fhir_get(path)`** / **`fhir_post(path, data)`** (`fhir_tools.py:40`, `:60`) — thin `httpx` wrappers; `fhir_post` tolerates an empty 201 body.
- **Read tools**: `get_patient`, `get_patient_conditions`, `get_patient_allergies`, `get_patient_medications` — each returns a **short human-readable string** (`"Allergies: Aspirin (criticality: high)"`), not JSON, so the model does not have to parse structure.
- **Write tools**: `create_triage_observation` (`:185`) — `Observation`, `category=survey`, SNOMED-coded, `status=preliminary`; `create_service_request` (`:226`) — `ServiceRequest` with `priority` mapped straight from the `urgency` argument.
- **`SYMPTOM_SNOMED_MAP`** + **`get_snomed_code`** (`:267`, `:286`) — substring lookup with a generic fallback code (`418799008`).
- **Every tool is wrapped in `try/except` and returns `"Error …: {e}"`** — tools never raise into the executor.

This is the file that most resembles a production **"tool adapter"** concern, and it is deliberately consistent.

### 2.11 `agent/knowledge_base.py` — RAG with layered fallback

- **Load paths** (chosen at import): PRIMARY = InterSystems **Embedded Python** (`iris.sql.exec`, in-process, no HTTP) when the interpreter is `irispython`; FALLBACK = **REST** via the Atelier SQL API from the API container. Both write the same `RAG.ClinicalGuidelines(id, source, topic, content, embedding VECTOR(DOUBLE,1536))` table.
- **`initialize_knowledge_base()`** (`knowledge_base.py:349`) — runs on import (module bottom, `:460`). Idempotent: skips rows already present, so container restarts don't re-embed. On failure of the primary path it retries the other path.
- **`get_embedding(text)`** (`:107`) — OpenAI `text-embedding-3-small`, explicit `float()` cast for `TO_VECTOR`.
- **`search_clinical_guidelines(query)`** (`:399`) — the `@tool`. Three-tier retrieval:
  1. embed query → `SELECT TOP 3 … VECTOR_COSINE(embedding, TO_VECTOR(?,DOUBLE)) AS similarity ORDER BY … DESC`, drop rows `< 0.1`
  2. if the SQL throws (IRIS busy/restarting) → **`keyword_search`** (`:326`), word-overlap over the in-memory CSV
  3. if still nothing → `"No relevant clinical guidelines found."`
- Output is formatted as `"[Source] (Relevance: 92.3%)\nTopic: …\n{content}"` — the `%` is the real cosine score ×100, which is what the agents quote.

Patterns: **retrieval-as-tool**, **graceful degradation** (vector → keyword → explicit null), **idempotent bootstrap**.

### 2.12 `agent/dynamic_agent.py` — agents defined at runtime

Turns JSON config into a live agent, no code change, no restart.

- **`AGENTS_FILE`** = `src/python/data/custom_agents.json` (a flat list).
- **`load_all_agents` / `save_agent` / `delete_agent`** (`dynamic_agent.py:60`–`113`) — file-backed CRUD; `save_agent` is idempotent by `id`; `delete_agent` also purges the executor cache.
- **`TOOL_REGISTRY`** (`:42`) — maps string tool names in the config to the actual tool objects, so config stays declarative (`"tools": ["get_patient", "search_clinical_guidelines"]`).
- **`_build_executor(config)`** (`:125`):
  - resolves the tool subset from the registry, force-adds `search_clinical_guidelines` if `rag_enabled`
  - **wraps the user's system prompt** with the same mandatory blocks the built-ins have (patient-ID injection rule, language rule, citation rule)
  - builds `ChatOpenAI(temperature=config["temperature"])` → `create_openai_tools_agent` → `AgentExecutor(max_iterations=config["max_iterations"])`
- **`_executor_cache`** keyed `"{agent_id}:{session_id}"` (`:122`) — same per-session pattern as the built-ins.
- **`run_custom_agent(agent_id, session_id, message)`** (`:207`) — called by the orchestrator when the router returns a custom agent name.

Pattern: **agent-as-configuration / plugin registry**. The router picks these up automatically because `_build_router_prompt` and `route_message` both call `load_all_agents()` every turn.

### 2.13 `iris/__init__.py`

InterSystems Embedded Python shim, copied out of the IRIS image (`Dockerfile.api:16`). It lets `knowledge_base.py` *detect* whether it is running in-process inside IRIS (`import iris; iris.sql`). In the API container that import fails on `iris.sql` and the REST path is used. Not agentic — an infrastructure detail that decides *how* RAG writes reach IRIS.

---

## Part 3 — Gap Analysis vs. the Job Description

> **Role:** Design, ship, and operate production-grade agentic systems powering a healthcare AI personal assistant.

Legend: ✅ demonstrated · 🟡 partial / toy-level · ❌ absent

### 3.1 "Build and improve AI agents and orchestration systems"

| Aspect | Status | Evidence |
|---|---|---|
| Multiple specialised agents | ✅ | `triage_agent.py`, `specialist_agent.py`, `pharmacy_agent.py`, `fhir_agent.py` |
| Orchestration / routing | ✅ | `orchestrator.py` — LLM router, turn-1 rule, validation, fallback |
| Runtime-extensible agents | ✅ | `dynamic_agent.py` + hot-reloaded router prompt |
| Autonomous (event-triggered) agent runs | ✅ | `main.py:657` `trigger_ai_alert` |
| Router sophistication | 🟡 | single-shot classifier; no confidence, no escalation, no re-routing on low-quality answers |
| Agent-to-agent coordination | 🟡 | orchestrator hands off, but sub-agents share **no** memory; no structured handoff / negotiation protocol |

### 3.2 "Core agentic capabilities: memory, reasoning orchestration, tool adapters, multi-agent coordination"

| Capability | Status | Notes |
|---|---|---|
| **Tool adapters** | ✅ | `fhir_tools.py` is a clean, consistent adapter layer — string returns, never raises, coded resources |
| **Memory** | 🟡 | `ConversationBufferMemory` per session + `session_context` blackboard. But: in-process dicts, no persistence, **no TTL / eviction**, **unbounded growth** (no summarisation or token budget), lost on restart, not shared across agents |
| **Reasoning orchestration** | 🟡 | `AgentExecutor` loop + per-role temperature + `max_iterations`. No planning step, no reflection / self-critique, no structured-output validation, no cost ceiling per conversation |
| **Multi-agent coordination** | 🟡 | see 3.1 |

### 3.3 "Strong observability, fallback mechanisms, fail-safe behaviors"

| Aspect | Status | Evidence / gap |
|---|---|---|
| Fallbacks | 🟡 | Good coverage of *local* failures: RAG vector→keyword→null; router unknown→TRIAGE; deleted custom agent→TRIAGE; empty FHIR 201 body guard; tools catch all exceptions |
| Fail-safes | 🟡 | `query_iris_sql` SELECT-only allow-list; vitals FHIR-write errors swallowed to protect the stream; `handle_parsing_errors=True`; `max_iterations` ceiling |
| **OpenAI outage handling** | ❌ | if the LLM call fails, the whole `/chat` request 500s — no retry, no backoff, no model fallback, no queue, no cached/degraded response |
| **FHIR/IRIS outage handling** | 🟡 | tools return an error string, but there is no circuit breaker, no retry policy, timeouts are fixed constants |
| **Observability** | ❌ | only `print()` to stdout and `verbose=True`. No structured logging, no request IDs, no tracing (LangSmith / OpenTelemetry), no metrics (latency, tokens, cost, per-tool success rate), no dashboards |
| **Loop / cost protection** | 🟡 | `max_iterations` only; no per-conversation token or dollar budget, no runaway detection beyond the ceiling |

### 3.4 "Own performance, observability and security: monitoring dashboards, SLO/SLI, security posture"

| Aspect | Status | Notes |
|---|---|---|
| Monitoring dashboards | ❌ | the `/dashboard` page is a **clinical-data** view, not a system-health view. No infra/agent metrics anywhere |
| SLO / SLI | ❌ | none defined or measured |
| Performance work | 🟡 | some deliberate choices (`_summary=count`, thread-offload of blocking LLM calls, internal Docker networking). No load testing, no latency budgets, no caching of embeddings/LLM responses |
| Security posture | ❌ | **no authentication or authorization on any endpoint** — anyone who can reach `:8000` can read every patient's data and trigger FHIR writes. `CORS allow_origins=["*"]`. Default credentials `_SYSTEM` / `SYS` in `config.py`. `.env` is committed to the repo (commit `47a1c01`). Atelier SQL runs with DBA rights. Dev server (`uvicorn --reload`). No rate limiting, no input validation, no prompt-injection defence (patient free text **and** FHIR field values flow straight into agent prompts) |

### 3.5 "Privacy and security by design, focus on medical-data sensitivity"

| Aspect | Status | Notes |
|---|---|---|
| PHI to third parties | ❌ | full patient records (names, DOB, conditions, meds, allergies) are sent to the OpenAI API. No de-identification, no BAA / data-processing-agreement handling, no data-residency control, no opt-out path |
| PHI in logs | ❌ | `verbose=True` and numerous `print()` calls emit patient data and tool payloads to container stdout |
| PHI in the browser | 🟡 | `index.html` persists the whole conversation (including clinical content) to `localStorage` in plaintext |
| Audit trail | ❌ | no record of who asked what, which patient was accessed, or what the AI wrote back |
| Consent / access model | ❌ | none |
| Data lifecycle | ❌ | in-memory session data has no retention policy; `ai_alerts_log` and vitals state live until restart |

### 3.6 "Partner with data scientists on evaluation pipelines, safety guardrails, and metrics"

| Aspect | Status | Notes |
|---|---|---|
| Evaluation pipeline | ❌ | no test suite of any kind, no golden dataset, no regression harness for agent behaviour, no LLM-as-judge, no offline eval |
| Safety guardrails | 🟡 | the "always cite a guideline" rule exists **only in the prompt** — nothing validates that the answer actually contains a citation or that the citation matches a real retrieved row. No content moderation, no clinical-safety validation of outputs, no confidence thresholds |
| Human-in-the-loop | ❌ | agents write `Observation`, `ServiceRequest`, and `MedicationRequest` resources to FHIR **autonomously**, with no review/confirmation gate |
| Metrics | ❌ | no groundedness / hallucination / helpfulness / safety metrics captured |

### 3.7 "Contribute to fullstack development for end-to-end features"

| Aspect | Status | Notes |
|---|---|---|
| End-to-end feature delivery | ✅ | five working front-end pages, a FastAPI backend, an IRIS FHIR server, RAG, SSE streaming — all wired together and demonstrably functional |
| Healthcare domain modelling | ✅ | correct use of FHIR R4 resources, SNOMED CT, LOINC, UCUM; `intent="proposal"` vs `"order"` distinction shows domain awareness |

### 3.8 Summary

**This repo is a strong **architecture and prototype** demonstration of the JD's *design* half:**

- multi-agent orchestration with an LLM router and hot-pluggable agents ✅
- a clean tool-adapter layer ✅
- retrieval-as-a-tool with graceful degradation ✅
- autonomous, event-driven agent invocation ✅
- per-role reasoning configuration ✅
- genuine end-to-end fullstack delivery in the healthcare/FHIR domain ✅

**It does *not* address the JD's *ship and operate* half:**

- ❌ observability: no logging/tracing/metrics/dashboards, no SLO/SLI
- ❌ evaluation: no eval pipeline, no tests, no regression safety net
- ❌ security: no authn/authz, committed secrets, default creds, wide-open CORS, no prompt-injection defence
- ❌ privacy: PHI to third-party LLM with no de-identification, PHI in logs, no audit trail, no retention policy
- ❌ resilience: no retry/backoff/circuit-breaker/model-fallback for LLM or FHIR outages
- ❌ enforced guardrails: safety rules are prompt-only; autonomous FHIR writes have no human-in-the-loop
- 🟡 memory: works, but unbounded, non-persistent, non-shared, no eviction

In interview terms: the codebase is a good answer to *"sketch an agentic architecture for a clinical assistant"* and a weak answer to *"operate it in production with medical data."* The productionisation gaps above are the concrete backlog that JD is asking a candidate to own.
