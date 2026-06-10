# IRIS FHIR Agents

A multi-agent clinical AI platform powered by InterSystems IRIS for Health. Features agents for triage, specialist consultation, pharmacy safety, and FHIR server exploration — all grounded by IRIS Vector Search RAG. Includes a no-code Agent Builder that lets you design and deploy custom clinical agents without writing a single line of code.

IRIS FHIR Agents orchestrates five LangChain-powered AI agents that work together to deliver clinical intelligence directly on top of a live FHIR R4 server

<img width="1627" alt="image" src="https://github.com/user-attachments/assets/410b2ffa-3af6-43d0-944e-60e463ca8b00" />

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
<img width="1141"  alt="iris_fhir_agents_architecture_updated" src="https://github.com/user-attachments/assets/68110d3d-ff95-4888-9e02-b0f226171611" />

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
├── iris_module/
│   └── __init__.py               ← For Embedded python
│
├── src/fhirsetup/
│   └── Setup.cls                 ← Setup FHIR server
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
│       └── agent_builder.html       ← Build Custom Agent
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
git clone https://github.com/mwaseem75/iris-fhir-agents.git
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

### 3. Start the containers

```bash
docker-compose up -d --build
```

This starts two containers:
- `fhir-agents-iris` — InterSystems IRIS for Health on ports `32782 / 32783 / 32784`
- `fhir-agents-api` — FastAPI application on port `8000`

First startup takes some time while IRIS initialises and the RAG knowledge base embeds 50 guidelines into IRIS Vector Search. Watch the logs:

```bash
docker logs fhir-agents-api --tail=50 -f
```

You should see:

```
RAG: Loaded 50 guidelines from CSV
RAG: Embedding 50 guidelines into IRIS Vector Search...
RAG: Initialisation complete — 50 new guidelines embedded and stored
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4. Clinical Conversation flow

<img width="1625"  alt="image" src="https://github.com/user-attachments/assets/a246dd4c-3b35-4edf-85d1-e0d89b92e9e1" />


### 5. Open the application

| Page | URL |
|---|---|
| Triage Chat | http://localhost:8000 |
| Analytics Dashboard | http://localhost:8000/dashboard |
| Live Vitals Monitor | http://localhost:8000/vitals |
| FHIR Server Agent | http://localhost:8000/fhir-agent |
| Agent Builder | http://localhost:8000/agent-builder |

---

### Demo Patients & Use Cases

| Patient ID | Name | Conditions | Key Demo |
|---|---|---|---|
| `demo-010` | James Anderson, M/68 | CAD · Hypertension · Atrial Fibrillation | **Pharmacy** — Warfarin on board + **Aspirin allergy** (high) · Antiplatelet alternative needed |
| `demo-011` | Maria Gonzalez, F/37 | T1DM · Septicaemia · DKA | **Emergency** — BP 88/54 · Lactate 4.2 · Glucose 22.4 · pH 7.22 · ICU critical care |
| `demo-012` | Robert Davis, M/60 | T2DM · HTN · Diabetic Retinopathy · CKD3 | **Specialist** — HbA1c 7.8% · eGFR 38 · Microalbuminuria · Nephrology + ophthalmology referrals |
| `demo-013` | Patricia Taylor, F/75 | HFrEF · T2DM · CKD3 · Hypertension | **Emergency** — BNP 845 · BP 96 hypotension · eGFR 42 · EF 28% · Decompensated HF |

Click on Browse patients to select the patient OR Type `My patient ID is demo-010` to start.
<img width="1625"  alt="image" src="https://github.com/user-attachments/assets/636468fc-e15d-43c1-9725-1060de238506" />

Triage Agent reads the data from the FHIR server and responds.
<img width="1632"  alt="image" src="https://github.com/user-attachments/assets/1974fe19-cb6a-4fdc-a7d4-b54c6c9b24d5" />

The user enters the condition, and the Agent will respond accordingly.
<img width="1635"  alt="image" src="https://github.com/user-attachments/assets/9f36f573-c432-457c-bbf5-483ddeafc609" />

New FHIR resources created by the Triage Agent can be viewed in the Analytics Dashboard.
<img width="1627"  alt="image" src="https://github.com/user-attachments/assets/546d58d0-8e3d-4b17-bb69-2de0c09d55d0" />

New FHIR resource can be verified using Postman.
<img width="1477" alt="image" src="https://github.com/user-attachments/assets/e8f0af66-64d5-4a4d-99cb-1501794d5a90" />


## Analytics Dashboard

The Analytics Dashboard (`/dashboard`) provides a real-time population-level view of all clinical data in IRIS, alongside a live record of every FHIR resource written by the AI agents during triage sessions.

### Five tabs

**Overview**
The landing view — four summary stat cards showing total counts of Patients, Conditions, Medications, and Allergies queried live from IRIS FHIR R4. Beneath them, three panels sit side by side:
<img width="1632"  alt="image" src="https://github.com/user-attachments/assets/dd479df3-1432-4d2d-9bb3-a94be888403f" />
- **Top Active Conditions** — horizontal bar chart ranked by frequency across all patients. Click any bar to open a Condition Detail panel showing exactly which patients carry that condition with a direct link to open each in Triage Chat.
- **Gender Distribution** — donut chart of patient demographics.
- **Recent Triage Observations** — the last AI-written FHIR Observations from triage sessions, showing symptom, severity, and the patient they belong to.

**Patients**
A paginated roster of every patient in IRIS. Each row shows Patient ID, name, date of birth, and gender. Click any patient to expand an inline detail panel with four sub-sections: Demographics, Active Conditions, Known Allergies, and Current Medications — all fetched live from FHIR. An **Open in Triage Chat** button starts a session for that patient directly.
<img width="1632"  alt="image" src="https://github.com/user-attachments/assets/04fafbaf-b33f-4ac0-a963-d86786aa2f0d" />

**Conditions**
A full ranked bar chart of all active conditions across the population. Clicking a condition bar drills down to a list of affected patients with their IDs and names. Useful for population analytics queries such as "how many patients have both diabetes and CKD?".
<img width="1623"  alt="image" src="https://github.com/user-attachments/assets/d09e3cbb-3a81-452b-922d-64f9fe6d4da7" />

**AI Observations**
Every FHIR Observation resource written by the Triage Agent during chat sessions. Shows the symptom recorded, severity, patient reference, and timestamp. This tab is the live audit trail of AI clinical activity.
<img width="1627"  alt="image" src="https://github.com/user-attachments/assets/3f9789d8-1480-4e97-9292-6b630c848c0e" />

**Service Requests**
Every FHIR ServiceRequest written by the Triage and Specialist agents. Shows the referral type, priority (routine / urgent / asap), patient reference, and the clinical reason recorded. Demonstrates the full FHIR write pipeline end to end.
<img width="1627"  alt="image" src="https://github.com/user-attachments/assets/3e767f69-2f89-4b60-b4e1-3edf48ec7607" />

---
## Live Vitals Monitor

The Live Vitals Monitor (`/vitals`) is a real-time bedside monitoring simulation that streams patient vitals via Server-Sent Events (SSE), writes every reading to IRIS as a FHIR Observation, and automatically triggers the Triage Agent when critical values are detected — without any user action required.
<img width="1627" alt="image" src="https://github.com/user-attachments/assets/787830c2-2f22-4680-84cd-f59c2df87cc7" />

---

### What it shows

Five vital sign cards update every 2 seconds, each with a colour-coded status indicator and a mini sparkline chart showing the trend of the last 20 readings:

| Vital | Unit | Status colours |
|---|---|---|
| **Heart Rate** | beats/min | 🟢 Normal · 🟡 Warning · 🔴 Critical |
| **Blood Pressure** | systolic/diastolic mmHg | 🟢 Normal · 🟡 Warning · 🔴 Critical |
| **SpO₂** | oxygen saturation % | 🟢 Normal · 🟡 Warning · 🔴 Critical |
| **Temperature** | degrees Celsius | 🟢 Normal · 🟡 Warning · 🔴 Critical |
| **Respiratory Rate** | breaths/min | 🟢 Normal · 🟡 Warning · 🔴 Critical |

A status banner at the top reflects the overall patient state in real time:

---

### Dynamic patient selection

The patient sidebar loads **live from IRIS FHIR R4** on every page open via `GET /analytics/patients`. Whoever is in your FHIR server appears automatically — no hardcoding, no configuration needed. Add new patients to IRIS and they appear on the next page refresh.

Each patient pip in the sidebar turns amber or red as their vitals status changes, giving a at-a-glance overview of all monitored patients simultaneously.

---

### How the streaming works

Each reading arrives via `GET /vitals/stream/{patient_id}` — a long-lived SSE connection. The backend generates a realistic reading every 2 seconds with occasional simulated spikes to demonstrate critical escalation. Every reading — normal or critical — is immediately written to IRIS as a FHIR Observation.

The **FHIR Write Counter** in the sidebar footer increments with each reading, confirming live persistence to IRIS. The **FHIR Observation Writes** panel on the right shows the last 20 readings in real time with HR, BP, SpO₂, Temperature, and timestamp per row.

---

### AI auto-trigger

When any vital crosses a critical threshold, the platform automatically dispatches an assessment to the Triage Agent — no user input needed. A **30-second cooldown** prevents repeated alerts from the same sustained critical episode.

The complete flow per critical reading:
SSE reading arrives → FHIR Observation written to IRIS
↓
Critical threshold crossed → Triage Agent dispatched
↓
RAG searches IRIS Vector Knowledge Base for relevant guidelines
↓
AI assessment returned → AI Critical Alert Feed panel updated
↓
FHIR ServiceRequest written if referral is warranted

The **AI Critical Alert Feed** panel shows each automated assessment with:
- The critical vital values that triggered it
- The Triage Agent's urgency classification
- The clinical reasoning with RAG guideline citations
- Timestamp of the automated trigger

AI alert polling runs every 5 seconds via `GET /vitals/alerts` so assessments appear immediately.

---

### Vitals History table

Below the vital cards, a scrollable history table shows the last 15 readings with full detail:

| Column | Description |
|---|---|
| Time | Reading timestamp |
| Heart Rate | bpm |
| Blood Pressure | systolic/diastolic |
| SpO₂ | percentage |
| Temperature | degrees Celsius |
| Respiratory Rate | breaths/min |
| Status | Normal / Warning / Critical badge |
| FHIR | ✓ confirming write to IRIS |

Every row confirms the FHIR write — this table is the live audit trail proving that every vital sign is persisted as a structured FHIR Observation in IRIS.

---

### How to demo the Live Vitals Monitor

**Step 1** — Open `http://localhost:8000/vitals`. The sidebar loads patients directly from IRIS — no hardcoded list.

**Step 2** — Select `demo-003` (Mohammed Al-Farsi — HFrEF + BNP 845). His cardiac profile makes critical vital spikes produce particularly detailed Triage Agent assessments.

**Step 3** — Wait for a critical spike. The status banner flashes `🚨 CRITICAL — AI Alert Triggered Automatically` and within a few seconds the AI Critical Alert Feed panel updates with the Triage Agent's assessment — zero user input.

**Step 4** — Point to the FHIR write counter in the sidebar footer. Show it incrementing every 2 seconds — live proof of continuous writes to IRIS.

**Step 5** — Scroll down to the Vitals History table. Show that every reading is captured and every row has a FHIR ✓ confirmation.

**Step 6** — Switch to `pt-003` (Michael Williams — HFrEF + Digoxin + K⁺ low). Any cardiac critical reading triggers an AI alert that specifically references the Digoxin toxicity risk — demonstrating that the AI is reading and reasoning from the patient's actual FHIR record, not generating generic responses.

**Step 7** — Switch themes using the Dark / Light / Clinical button to show the UI adapts cleanly across modes.


## FHIR Server Agent

The FHIR Server Agent (`/fhir-agent`) provides two complementary ways to explore your InterSystems IRIS FHIR R4 server — a **live visual Capability Explorer** and a **natural language AI chat** interface that can query patients, run IRIS SQL, and cross-reference clinical guidelines.

---

### Two tabs

**AI Chat** (default tab)
A conversational interface powered by GPT-4o-mini with 11 FHIR tools. Ask anything about your IRIS data in plain English — the agent fetches, analyses, and synthesises responses grounded in live FHIR data.
<img width="1625"  alt="image" src="https://github.com/user-attachments/assets/b4029dd3-5f9d-4d48-9a44-724c300e9a74" />


**FHIR Capability Explorer**
A live visual breakdown of everything your IRIS FHIR server supports — loaded directly from the CapabilityStatement at startup. No static data — every chart and table reflects the actual server configuration.
<img width="1620"  alt="image" src="https://github.com/user-attachments/assets/c8863b7f-421c-4876-8553-be159da586f7" />

---

### FHIR Capability Explorer

Opens automatically when you navigate to the page. Fetches `GET /fhir/metadata` and renders the CapabilityStatement into five visual components:

**Server info cards**
Four cards showing Software name and version, FHIR version, server kind, and endpoint address.

**Summary stats row**
Five counters: Resource Types supported · Total Interactions · Supported Formats · Operations · Search Parameters.

**Interaction Coverage donut chart**
A proportional donut showing the breakdown of interaction types (read, search-type, create, update, delete, patch) across all resources.

**Resource Categories donut chart**
Groups all supported resources into Clinical, Patient, Administrative, Financial, and Infrastructure categories with proportional slicing.

**Top Search Parameters bar chart**
Horizontal bars showing which resources have the most search parameters — useful for understanding query capability.

**Interaction Matrix**
A table showing the eight key clinical resources (Patient, Condition, MedicationRequest, Observation, AllergyIntolerance, ServiceRequest, Procedure, Encounter) against all six interaction types with ✓ / · indicators and search parameter counts.

**All Supported Resources grid**
Every resource the server supports rendered as a card showing its type, interaction tags (read, search-type, create, update, delete, patch), and search parameter count. Includes:
- Real-time search filter — type any resource name to narrow the grid
- Category filter buttons — Clinical / Patient / Administrative / Financial / Infrastructure
- Click any card to expand a detail panel showing all interactions, every search parameter with type, and capability flags (ReadHistory, UpdateCreate, ConditionalCreate, ConditionalUpdate)

---

### AI Chat — 11 FHIR tools

The FHIR Server Agent has access to 11 tools that it calls automatically based on your question:

| Tool | What it does |
|---|---|
| `get_patient_list` | Fetch all patients from IRIS |
| `get_conditions` | Get conditions for a patient |
| `get_medications` | Get active medications for a patient |
| `get_observations` | Get observations and lab results |
| `get_capability` | Fetch the CapabilityStatement |
| `query_iris_sql` | Run direct SQL against IRIS via Atelier REST API |
| `get_fhir_statistics` | Resource counts across the server |
| `search_guidelines` | RAG search of the clinical knowledge base |

The sidebar **Tools Used** panel tracks which tools have been called in the current session, with a call counter and a pulsing indicator when a tool is actively running.

---

## Agent Builder
The Agent Builder (`/agent-builder`) is a no-code interface for designing, configuring, and deploying custom AI clinical agents. Every agent created here integrates directly into the Triage Chat orchestrator, appears in the Agent Network sidebar, and is callable from the same conversation interface as the built-in agents — without writing a single line of code.
<img width="1627" alt="image" src="https://github.com/user-attachments/assets/d9c7da8a-6420-4d5e-8daa-724f603f6e46" />

### Five clinical templates
| Template | Specialty | Key Capabilities |
|---|---|---|
| 🎗️ **Oncology Agent** | Oncology | Chemotherapy drug interactions, platinum compound contraindications, tumour board referrals, NCCN/ASCO guideline citations |
| 👴 **Geriatrics Agent** | Geriatrics | Beers Criteria screening, anticholinergic burden, fall risk, polypharmacy review (≥5 drugs flagged) |
| 👶 **Pediatrics Agent** | Pediatrics | Weight-based dosing (mg/kg), age-appropriate normal ranges, contraindicated medications (aspirin, codeine, fluoroquinolones) |
| ❤️ **Cardiology Agent** | Cardiology | HFrEF/HFpEF management, digoxin + electrolyte danger detection, GDMT gap identification, AHA/ACC guidelines |
| 🥗 **Nutrition Agent** | Nutrition | Drug-nutrient interactions, disease-specific dietary guidance (ADA, KDOQI), warfarin + vitamin K counselling |

---
### How it works
Open Agent Builder 


Choose template or start blank → Write system promptConfigure tools · Set temperature · Enable RAG


Test against live IRIS FHIR data → Save → Available instantly in Triage Chat

---
### Demo — Building an Oncology Agent

**Step 1 — Open the Agent Builder:**
http://localhost:8000/agent-builder
<img width="1526"  alt="image" src="https://github.com/user-attachments/assets/a101b287-0313-4112-8ba7-c230cf6ea9b2" />

**Step 2 — Click and configure Oncology Agent template.** The system prompt auto-fills with a complete clinical prompt covering chemo drug interactions, NCCN guideline citations, and contraindication checks.
<img width="1183"  alt="image" src="https://github.com/user-attachments/assets/fb5e186f-6212-421b-874d-baedf29c548d" />

**Step 3 — Save the agent and click Test Agent button.** The slide-in test panel opens. Type:
My patient ID is demo-022. Can she receive platinum-based chemotherapy?
<img width="1523" alt="image" src="https://github.com/user-attachments/assets/10e7ae5e-d7bb-4760-b472-a3a9396296e9" />

The agent fetches Susan Lee's record from IRIS, finds her **Platinum compounds allergy** (criticality: high), and responds

**The Oncology Agent is now available in:**
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
The Agent Network sidebar in Triage Chat highlights the active agent in real time, showing the custom agent name, icon, colour, and call count alongside the built-in agents.
<img width="1528" alt="image" src="https://github.com/user-attachments/assets/484bfb12-9a59-4ce7-bdfd-4a91e1dbc92a" />

---

## RAG — Clinical Guidelines with IRIS Vector Search

Every agent recommendation is grounded by **50 clinical guidelines** stored as `VECTOR(DOUBLE, 1536)` embeddings in IRIS — the same instance that holds your FHIR patient data. No external vector database required.

### How to see it working

**Step 1 — Load a patient and ask a clinical question:**
```
My patient ID is demo-012
```
Then ask the below:
"My HbA1c is 7.8% and my kidney function has been declining. What should I do?"
<img width="1621" alt="image" src="https://github.com/user-attachments/assets/ba02da39-cf0c-41d2-82ea-b83b7bab4431" />

The relevance score is the actual `VECTOR_COSINE` similarity value returned by IRIS — not a hallucinated reference.

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

Thanks
