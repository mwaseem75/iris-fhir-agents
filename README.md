# IRIS FHIR Agents

> A multi-agent clinical AI platform powered by InterSystems IRIS for Health. Features agents for triage, specialist consultation, and pharmacy safety, grounded by IRIS Vector Search RAG.
IRIS FHIR Agents orchestrates **four LangChain-powered AI agents** that work together to deliver clinical intelligence directly on top of a live FHIR R4 server:

| Agent | Role | Key Capability |
|---|---|---|
| **Triage Agent** | Patient intake | Urgency classification · FHIR Observation writes · SNOMED CT codes |
| **Specialist Agent** | Condition analysis | Comorbidity review · Referral planning · ServiceRequest writes |
| **Pharmacy Agent** | Medication safety | Drug interaction checks · Allergy conflict detection · MedicationRequest writes |
| **FHIR Server Agent** | FHIR exploration | Natural language FHIR queries · IRIS SQL · Capability explorer |
| **Custom Agents** |  User-defined specialty | No-code Agent Builder · Configurable tools · 5 clinical templates · Routes automatically via orchestrator  |

Every agent is grounded by **IRIS Vector Search RAG** — 50 clinical guidelines from CDC, AHA, FDA, WHO, and KDIGO embedded into IRIS and retrieved semantically at query time. No guideline citation means no recommendation.

---

## Features at a Glance

- **Dynamic multi-agent orchestration** — a zero-temperature LLM router classifies every message and dispatches to the correct agent automatically — including user-created custom agents
- **No-code Agent Builder** — design, configure, and test custom clinical agents via a visual UI; five built-in templates (Oncology, Geriatrics, Pediatrics, Cardiology, Nutrition); deployed instantly into the orchestrator
- **IRIS Vector Search RAG** — 50 clinical guidelines stored as `VECTOR(DOUBLE, 1536)` in IRIS; queried with `VECTOR_COSINE` for semantic similarity; every agent recommendation is guideline-grounded
- **Full FHIR R4 write path** — agents create Observations, ServiceRequests, and MedicationRequests directly in IRIS with proper SNOMED CT and RxNorm coding
- **Live vitals monitoring** — SSE stream writes every reading to FHIR as a coded Observation; critical vitals auto-trigger the Triage Agent with a 30-second AI alert cooldown
- **FHIR Capability Explorer** — visual breakdown of what the IRIS FHIR server supports: interaction matrix, resource cards, donut charts, search param rankings
- **Voice input** — Web Speech API integration in Triage Chat and FHIR Agent; auto-detects language and switches voice recognition accordingly; auto-sends on final transcript
- **Multi-language support** — agents automatically detect and respond in the patient's language (English, Spanish, French, Mandarin); drug safety warnings appear in both languages; English handoff summary always included for clinical staff
- **Patient Picker** — modal browser loading live from IRIS FHIR; real-time search; clinical hints per patient; one-click session start
- **Five-page frontend** — consistent sidebar navigation, three themes (Dark / Light / Clinical), live agent network panel, language badge
- **20 rich demo patients** — covering CAD, HFrEF, T2DM, CKD, sepsis, oncology, geriatrics, paediatrics, and complex polypharmacy scenarios across demo-001 to demo-029

---

## Architecture
<img width="1141" height="1601" alt="iris_fhir_agents_architecture_updated" src="https://github.com/user-attachments/assets/68110d3d-ff95-4888-9e02-b0f226171611" />

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
│   │   ├── dynamic_agent.py      ← create custom agent
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
│       ├── fhir_agent.html       ← FHIR Server Agent
        └── agent_builder.html       ← Build Custom Agent
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
- `fhir-agents-iris` — InterSystems IRIS for Health on ports `32782 / 32783 / 32784`
- `fhir-agents-api` — FastAPI application on port `8000`

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

### 5. Open the application

| Page | URL |
|---|---|
| Triage Chat | http://localhost:8000 |
| Analytics Dashboard | http://localhost:8000/dashboard |
| Live Vitals Monitor | http://localhost:8000/vitals |
| FHIR Server Agent | http://localhost:8000/fhir-agent |
| Agent Builder | http://localhost:8000/agent-builder |

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

## Agent Builder — Create Your Own Clinical Agent

One of the platform's most powerful features is the **Agent Builder** (`/agent-builder`) — a no-code interface that lets anyone design, configure, and deploy a custom AI clinical agent without writing a single line of code. Every custom agent integrates directly into the Triage Chat orchestrator and appears in the Agent Network sidebar alongside the built-in agents.

---
### How it works
Open Agent Builder → Choose template or start blank → Write system prompt
↓
Configure tools · Set temperature · Enable RAG
↓
Test against live IRIS FHIR data → Save → Available instantly in Triage Chat

---

### Built-in Templates

Five clinical templates are provided as starting points — each pre-configured with a clinically accurate system prompt, recommended temperature, and appropriate tool set:

| Template | Specialty | Key Capabilities |
|---|---|---|
| 🎗️ **Oncology Agent** | Oncology | Chemotherapy drug interactions, platinum compound contraindications, tumour board referrals, NCCN/ASCO guideline citations |
| 👴 **Geriatrics Agent** | Geriatrics | Beers Criteria screening, anticholinergic burden, fall risk, polypharmacy review (≥5 drugs flagged) |
| 👶 **Pediatrics Agent** | Pediatrics | Weight-based dosing (mg/kg), age-appropriate normal ranges, contraindicated medications (aspirin, codeine, fluoroquinolones) |
| ❤️ **Cardiology Agent** | Cardiology | HFrEF/HFpEF management, digoxin + electrolyte danger detection, GDMT gap identification, AHA/ACC guidelines |
| 🥗 **Nutrition Agent** | Nutrition | Drug-nutrient interactions, disease-specific dietary guidance (ADA, KDOQI), warfarin + vitamin K counselling |

---

### Demo — Building an Oncology Agent

**Step 1 — Open the Agent Builder:**
http://localhost:8000/agent-builder

**Step 2 — Click the Oncology Agent template.** The system prompt auto-fills with a complete clinical prompt covering chemo drug interactions, NCCN guideline citations, and contraindication checks.

**Step 3 — Configure:**
- Temperature: `0.15` — precise and consistent for drug safety
- Tools: all FHIR read tools + `search_clinical_guidelines` + `create_service_request`
- RAG: enabled — retrieves NCCN/ASCO guidelines from IRIS Vector Search

**Step 4 — Click Test Agent.** The slide-in test panel opens. Type:
My patient ID is demo-022. Can she receive platinum-based chemotherapy?

The agent fetches Susan Lee's record from IRIS, finds her **Platinum compounds allergy** (criticality: high), and responds:
⚠ HIGH RISK: Platinum compound allergy documented for this patient.
Carboplatin and Cisplatin are CONTRAINDICATED.
According to NCCN Guidelines (Relevance: 94%) — patients with prior
platinum hypersensitivity should receive alternative regimens.
Recommend: Oncology MDT review for non-platinum alternative.
--- ENGLISH HANDOFF SUMMARY ---
Patient: Susan Lee (demo-022) | Breast cancer Stage IIB | Platinum allergy HIGH
Assessment: Platinum-based chemo CONTRAINDICATED — allergy on record
Action: ServiceRequest written to IRIS — oncology MDT referral

---
**Step 5 — Save.** The Oncology Agent is now available in:
- **Triage Chat sidebar** — appears in the Agent Network panel
- **Orchestrator** — messages about cancer and chemotherapy route to it automatically
- **API** — callable via `POST /agents/oncology-agent/test`

---

### What makes a good custom agent

| Setting | Guidance |
|---|---|
| **Temperature** | `0.1` for drug safety and strict protocols · `0.2` for clinical assessments · `0.3` for counselling and dietary advice |
| **System prompt** | Start with the specialty, list responsibilities, add clinical rules. The platform auto-appends patient ID injection, language detection, and guideline citation rules. |
| **Routing description** | One sentence telling the orchestrator when to route to this agent. Be specific: *"For cancer, chemotherapy, and oncology questions"* works better than *"For complex patients"*. |
| **Tools** | Enable `create_service_request` if the agent should write referrals. Enable `create_triage_observation` if it should record clinical findings. Disable write tools for read-only advisory agents. |
| **RAG** | Keep enabled for any clinical agent — guideline grounding prevents hallucinated recommendations. |

---

### Custom agent in action — Triage Chat routing

Once saved, the orchestrator's router prompt is updated dynamically. If a user types:
'''
My patient has breast cancer and is asking about Tamoxifen side effects
'''
The zero-temperature router recognises this as an oncology question and routes it to the **Oncology Agent** rather than the built-in Specialist Agent — without any configuration change.

The Agent Network sidebar in Triage Chat highlights the active agent in real time, showing the custom agent name, icon, colour, and call count alongside the built-in agents.

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

### Pages

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Triage Chat |
| `GET` | `/dashboard` | Analytics Dashboard |
| `GET` | `/vitals` | Live Vitals Monitor |
| `GET` | `/fhir-agent` | FHIR Server Agent |
| `GET` | `/agent-builder` | Agent Builder |
| `GET` | `/health` | Service health check |

### Clinical Chat

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Multi-agent clinical chat — routed to built-in or custom agents |
| `GET` | `/session/{id}/new` | Clear session context and memory |

### Analytics

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/analytics/summary` | FHIR resource counts |
| `GET` | `/analytics/conditions` | Top active conditions across all patients |
| `GET` | `/analytics/observations` | AI-created triage observations |
| `GET` | `/analytics/service-requests` | AI-created service requests |
| `GET` | `/analytics/patients` | Patient roster — used by Patient Picker modal |

### Live Vitals

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/vitals/stream/{patient_id}` | SSE real-time vitals stream |
| `GET` | `/vitals/alerts` | AI-triggered critical alert feed |
| `GET` | `/vitals/snapshot/{patient_id}` | Single vitals reading |

### FHIR Server Agent

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/fhir-agent/chat` | FHIR Server Agent natural language chat |
| `GET` | `/fhir-agent/status` | IRIS server connectivity check |
| `GET` | `/fhir/metadata` | FHIR CapabilityStatement proxy |

### Custom Agent Builder

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/agents` | List all custom agents |
| `POST` | `/agents/create` | Create or update a custom agent |
| `GET` | `/agents/{agent_id}` | Get a single agent config by ID |
| `DELETE` | `/agents/{agent_id}` | Delete a custom agent |
| `POST` | `/agents/{agent_id}/test` | Test a custom agent with a single message against live IRIS |

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
