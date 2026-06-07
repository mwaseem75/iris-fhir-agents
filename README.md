# IRIS FHIR Agents

> A multi-agent clinical AI platform powered by InterSystems IRIS for Health. Features agents for triage, specialist consultation, and pharmacy safety, grounded by IRIS Vector Search RAG.

[![InterSystems IRIS](https://img.shields.io/badge/InterSystems-IRIS%20for%20Health-blue)](https://www.intersystems.com/products/intersystems-iris-for-health/)
[![FHIR R4](https://img.shields.io/badge/FHIR-R4-orange)](https://hl7.org/fhir/R4/)
[![LangChain](https://img.shields.io/badge/LangChain-Agents-green)](https://python.langchain.com/)
[![GPT-4o-mini](https://img.shields.io/badge/OpenAI-GPT--4o--mini-purple)](https://openai.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB)](https://www.python.org/)

---

## What Is This?

IRIS FHIR Agents orchestrates **four LangChain-powered AI agents** that work together to deliver clinical intelligence directly on top of a live FHIR R4 server:

| Agent | Role | Key Capability |
|---|---|---|
| **Triage Agent** | Patient intake | Urgency classification · FHIR Observation writes · SNOMED CT codes |
| **Specialist Agent** | Condition analysis | Comorbidity review · Referral planning · ServiceRequest writes |
| **Pharmacy Agent** | Medication safety | Drug interaction checks · Allergy conflict detection · MedicationRequest writes |
| **FHIR Server Agent** | FHIR exploration | Natural language FHIR queries · IRIS SQL · Capability explorer |

Every agent is grounded by **IRIS Vector Search RAG** — 50 clinical guidelines from CDC, AHA, FDA, WHO, and KDIGO embedded into IRIS and retrieved semantically at query time. No guideline citation means no recommendation.

---

## Features at a Glance

- **Multi-agent orchestration** — a zero-temperature LLM router classifies every message and dispatches to the correct agent automatically
- **IRIS Vector Search RAG** — guidelines stored as `VECTOR(DOUBLE, 1536)` in IRIS; queried with `VECTOR_COSINE` for semantic similarity
- **Live vitals monitoring** — SSE stream writes every reading to FHIR as a coded Observation; critical vitals auto-trigger the Triage Agent
- **FHIR Capability Explorer** — visual breakdown of what the IRIS FHIR server supports: interaction matrix, resource cards, search param charts
- **Full FHIR R4 write path** — agents create Observations, ServiceRequests, and MedicationRequests directly in IRIS
- **Four-page frontend** — consistent sidebar navigation, three themes (Dark / Light / Clinical), live agent network panel
- **10 rich demo patients** — covering CAD, HFrEF, T2DM, CKD, sepsis, and complex polypharmacy scenarios

---

## Architecture

```
<img width="1500" height="1979" alt="iris_fhir_agents_architecture_v2_white_bg" src="https://github.com/user-attachments/assets/fbacc068-9836-4195-a70a-d5af396f4c3c" />

```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **AI Agents** | LangChain · GPT-4o-mini · ConversationBufferMemory |
| **RAG / Vector Search** | InterSystems IRIS Vector Search · text-embedding-3-small · VECTOR(DOUBLE, 1536) |
| **FHIR Server** | InterSystems IRIS for Health · FHIR R4 (4.0.1) |
| **Clinical Standards** | FHIR R4 · SNOMED CT · LOINC · UCUM · HL7 |
| **Backend** | Python 3.11 · FastAPI · httpx · SSE |
| **Frontend** | Vanilla HTML/CSS/JS · Syne · JetBrains Mono · Lato |
| **Infrastructure** | Docker Compose · two containers |
| **Guidelines** | CDC · AHA/ACC · FDA · WHO · KDIGO · AAAAI · ADA |

---

## Project Structure

```
iris-fhir-template/
├── docker-compose.yml
├── .env                          ← your secrets (never commit)
├── .env.example                  ← template for new contributors
├── Dockerfile
│
├── src/python/
│   ├── api/
│   │   └── main.py               ← FastAPI server, all HTTP routes
│   │
│   ├── agent/
│   │   ├── config.py             ← centralised configuration
│   │   ├── orchestrator.py       ← LLM router + session management
│   │   ├── triage_agent.py       ← patient intake agent
│   │   ├── specialist_agent.py   ← condition analysis agent
│   │   ├── pharmacy_agent.py     ← medication safety agent
│   │   ├── fhir_agent.py         ← FHIR server exploration agent
│   │   ├── fhir_tools.py         ← shared FHIR R4 tools
│   │   └── knowledge_base.py     ← IRIS Vector Search RAG
│   │
│   └── static/
│       ├── index.html            ← Triage Chat
│       ├── dashboard.html        ← Analytics Dashboard
│       ├── vitals.html           ← Live Vitals Monitor
│       └── fhir_agent.html       ← FHIR Server Agent
│
└── data/
    ├── fhir/
    │   ├── patient_01.json       ← James Anderson — CAD + Warfarin + Aspirin allergy
    │   ├── patient_02.json       ← Sarah Thompson — T2DM complex
    │   ├── patient_03.json       ← Michael Williams — HFrEF + Digoxin danger
    │   ├── patient_04.json       ← Emily Johnson — Asthma + NSAID allergy
    │   ├── patient_05.json       ← Robert Davis — T2DM + SGLT2 + retinopathy
    │   ├── patient_06.json       ← Linda Martinez — Stroke + AFib + sub-therapeutic INR
    │   ├── patient_07.json       ← Charles Wilson — CKD3b + NSAID AKI history
    │   ├── patient_08.json       ← Patricia Taylor — HFrEF EF28% + BNP 845
    │   ├── patient_09.json       ← Kevin Garcia — Sepsis + T1DM (ICU demo)
    │   └── patient_10.json       ← Margaret Young — HFrEF + T2DM + CKD + 7 meds
    │
    └── guidelines/
        └── clinical_rag_guidelines.csv   ← 50 guidelines for RAG
```

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- [InterSystems IRIS for Health Community Edition](https://docs.intersystems.com/irisforhealth20231/csp/docbook/DocBook.UI.Page.cls?KEY=PAGE_deployment_docker) (handled by docker-compose)
- An [OpenAI API key](https://platform.openai.com/api-keys) with access to `gpt-4o-mini` and `text-embedding-3-small`

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/iris-fhir-agents.git
cd iris-fhir-agents
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-real-key-here
```

All other defaults work out of the box with the Docker setup.

### 3. Start the platform

```bash
docker-compose up -d --build
```

This starts two containers:
- `fhir-template` — InterSystems IRIS for Health on ports `32782 / 32783 / 32784`
- `fhir-triage-api` — FastAPI application on port `8000`

First startup takes 3–5 minutes while IRIS initialises and the RAG knowledge base embeds 50 guidelines into IRIS Vector Search. Watch the logs:

```bash
docker logs fhir-triage-api --tail=50 -f
```

You should see:

```
RAG: Loaded 50 guidelines from CSV
RAG: Embedding 50 guidelines into IRIS Vector Search...
RAG: Initialisation complete — 50 new guidelines embedded and stored
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4. Load demo patient data

```bash
docker exec fhir-template bash -c "cd /home/irisowner/irisdev && iris session iris < load_patients.sh"
```

Or load each FHIR bundle manually via the FHIR REST API:

```bash
curl -X POST http://localhost:32783/fhir/r4 \
  -H "Content-Type: application/fhir+json" \
  -u _SYSTEM:SYS \
  -d @data/fhir/patient_01.json
```

### 5. Open the application

| Page | URL |
|---|---|
| Triage Chat | http://localhost:8000 |
| Analytics Dashboard | http://localhost:8000/dashboard |
| Live Vitals Monitor | http://localhost:8000/vitals |
| FHIR Server Agent | http://localhost:8000/fhir-agent |

---

## Demo Walkthrough

### Triage Chat — try these patients

| Patient ID | Name | Best demo |
|---|---|---|
| `pt-001` | James Anderson | CAD + Warfarin + **Aspirin allergy** → Pharmacy agent flags conflict |
| `pt-003` | Michael Williams | HFrEF + K⁺ 3.4 LOW + **Digoxin** → dangerous interaction |
| `pt-008` | Patricia Taylor | HFrEF EF28% + BNP 845 → **EMERGENCY** escalation |
| `pt-009` | Kevin Garcia | Sepsis + BP 88/54 + lactate 4.2 → **ICU** assessment |
| `pt-010` | Margaret Young | HFrEF + T2DM + CKD + HTN + 7 meds → all three agents |

Type `My patient ID is pt-009` to start.

### Live Vitals Monitor

Select any patient from the sidebar. Vitals stream every 2 seconds. Occasional critical spikes automatically trigger the Triage Agent — watch the AI Alert Feed panel on the left for the real-time assessment.

### FHIR Server Agent

Opens on the **FHIR Capability** tab by default — a full visual breakdown of what the IRIS server supports. Switch to **AI Chat** and try:

```
Which patients have both diabetes and kidney disease?
Show me a complete clinical summary for pt-010
Run this SQL: SELECT COUNT(*) FROM HSFHIR_X0001_S.Patient
```

---

## How the RAG Works

1. On startup, `knowledge_base.py` loads `clinical_rag_guidelines.csv` (50 guidelines)
2. Each guideline is embedded with `text-embedding-3-small` → 1536-dimensional vector
3. Vectors are stored in IRIS: `INSERT INTO RAG.VectorKnowledgeBase ... TO_VECTOR(?, DOUBLE)`
4. At query time, the agent's question is embedded the same way
5. IRIS finds the closest guidelines: `VECTOR_COSINE(embedding, TO_VECTOR(?, DOUBLE))`
6. Results above 0.1 similarity are returned with relevance scores
7. If IRIS is unavailable, keyword search over the in-memory CSV provides fallback coverage

The same IRIS instance that stores FHIR patient data also stores the clinical guideline vectors — no separate vector database required.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Triage Chat page |
| `GET` | `/dashboard` | Analytics Dashboard page |
| `GET` | `/vitals` | Live Vitals Monitor page |
| `GET` | `/fhir-agent` | FHIR Server Agent page |
| `GET` | `/health` | Service health check |
| `POST` | `/chat` | Multi-agent clinical chat |
| `GET` | `/session/{id}/new` | Clear session context |
| `GET` | `/analytics/summary` | FHIR resource counts |
| `GET` | `/analytics/conditions` | Top active conditions |
| `GET` | `/analytics/observations` | AI-created triage observations |
| `GET` | `/analytics/service-requests` | AI-created service requests |
| `GET` | `/analytics/patients` | Patient roster |
| `GET` | `/vitals/stream/{patient_id}` | SSE vitals stream |
| `GET` | `/vitals/alerts` | AI-triggered critical alerts |
| `GET` | `/vitals/snapshot/{patient_id}` | Single vitals reading |
| `POST` | `/fhir-agent/chat` | FHIR Server Agent chat |
| `GET` | `/fhir-agent/status` | IRIS server status |
| `GET` | `/fhir/metadata` | FHIR CapabilityStatement proxy |

---

## Docker Commands

```bash
# Start everything
docker-compose up -d --build

# Rebuild API container only (after code changes)
docker-compose stop api && docker-compose rm -f api && docker-compose up -d --build api

# View API logs
docker logs fhir-triage-api --tail=50 -f

# View IRIS logs
docker logs fhir-template --tail=30

# Stop everything
docker-compose down

# Full reset (removes volumes — clears all FHIR data and RAG embeddings)
docker-compose down -v
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model for all agents |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for RAG |
| `FHIR_BASE_URL` | `http://fhir-template:52773/fhir/r4` | Internal IRIS FHIR endpoint |
| `IRIS_BASE_URL` | `http://fhir-template:52773` | Internal IRIS base (for SQL) |
| `FHIR_USERNAME` | `_SYSTEM` | IRIS credentials |
| `FHIR_PASSWORD` | `SYS` | IRIS credentials |
| `RAG_GUIDELINES_CSV` | `/home/irisowner/.../clinical_rag_guidelines.csv` | Path to guidelines CSV |
| `TEMP_TRIAGE` | `0.3` | Triage Agent temperature |
| `TEMP_SPECIALIST` | `0.2` | Specialist Agent temperature |
| `TEMP_PHARMACY` | `0.1` | Pharmacy Agent temperature |
| `TEMP_ROUTER` | `0.0` | Orchestrator router temperature |

---

## Troubleshooting

**RAG re-embeds every restart**
The IRIS container was reset. Embeddings are stored in IRIS — a `docker-compose down -v` clears them. Normal behaviour on first start; subsequent restarts skip embedding.

**`Failed to fetch` on FHIR Capability tab**
The `/fhir/metadata` proxy route in `main.py` is not reachable. Run `docker logs fhir-triage-api` and check for startup errors. Verify IRIS is healthy: `http://localhost:32783/fhir/r4/metadata`

**`ModuleNotFoundError: No module named 'config'`**
`config.py` is not in the `/app/agent/` directory inside the container. Verify the Dockerfile COPY path includes `src/python/agent/`.

**AI Chat not responding**
Check `docker logs fhir-triage-api` for the error. Most common causes: missing `OPENAI_API_KEY` in `.env`, or the API container was not rebuilt after a code change.

**IRIS shows `Status: Error` in sidebar**
IRIS takes 2–3 minutes to fully initialise after `docker-compose up`. Wait and refresh. Check with: `docker logs fhir-template --tail=20`

---

## Contest Information

**Contest:** [InterSystems Programming Contest: AI Agents + FHIR](https://community.intersystems.com/post/intersystems-programming-contest-ai-agents-fhir)

**Submission:** [Open Exchange — IRIS FHIR Agents](https://openexchange.intersystems.com)

**Key contest criteria addressed:**

- ✅ Uses InterSystems IRIS for Health as the primary data store
- ✅ Full FHIR R4 compliance — reads and writes via REST API
- ✅ AI Agents — four LangChain agents with tool use and memory
- ✅ IRIS-native Vector Search for RAG (not a third-party vector DB)
- ✅ Real clinical value — triage, specialist, pharmacy, and FHIR exploration
- ✅ Docker Compose deployment — one command to run

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with InterSystems IRIS for Health · LangChain · FastAPI · GPT-4o-mini*
