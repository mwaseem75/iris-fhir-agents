# IRIS FHIR Agents

> A multi-agent clinical AI platform powered by InterSystems IRIS for Health. Features agents for triage, specialist consultation, and pharmacy safety, grounded by IRIS Vector Search RAG.
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
<img width="1500" alt="iris_fhir_agents_architecture_v2_white_bg" src="https://github.com/user-attachments/assets/ca85e584-8fcc-45f2-940a-e909f4931bc2" />

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
├── Dockerfile                    ← iris 
├── Dockerfile.api                ← application
├── iris.script                   ← Setup the FHIR server
├── merge.cpf                     ← iris
├── module.xml                    ← ZPM 
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
    │   ├── demo_patients.json    ← FHIR Synthetic Data
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

First startup takes some time while IRIS initialises and the RAG knowledge base embeds 50 guidelines into IRIS Vector Search. Watch the logs:

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

### 4. Demo patient data

Synthetic FHIR data is loaded during container build from the data/fhir/demo_patients.json file.

The data can also be loaded using the following cURL command:
```bash
curl -X POST http://localhost:32783/fhir/r4 \
  -H "Content-Type: application/fhir+json" \
  -u _SYSTEM:SYS \
  -d @data/fhir/demo_patients.json
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

| Patient ID | Name | Condition | Key Demo |
|------------|------|-----------|----------|
| `demo-001` | Sarah Rahman | T2DM · Hypertension · Hypothyroidism | Specialist — HbA1c 8.2% above target, Metformin + Lisinopril interaction with Potassium supplements |
| `demo-002` | Ahmed Khan | CAD · Asthma · Atrial Fibrillation | Pharmacy — Warfarin + Aspirin HIGH RISK bleeding combo. Sub-therapeutic INR 1.6 flagged |
| `demo-003` | Mohammed Al-Farsi | HFrEF · CKD Stage 3 | Emergency — BNP 845, creatinine 2.1, weight +4kg. Digoxin toxicity risk with worsening CKD |

Type `My patient ID is pt-001` to start.

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

## License

MIT License — see [LICENSE](LICENSE) for details.

---

Thanks
