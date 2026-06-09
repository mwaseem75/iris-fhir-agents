"""
IRIS FHIR Agents — FastAPI Application Server
=============================================
Entry point for the IRIS FHIR Agents platform.

This server ties together all components of the system:
  - Serves the four frontend pages (Triage Chat, Dashboard, Live Vitals, FHIR Server Agent)
  - Routes clinical chat messages through the multi-agent orchestrator
  - Queries InterSystems IRIS FHIR R4 for analytics and patient data
  - Streams real-time vital signs via SSE, writing every reading to FHIR as an Observation
  - Automatically escalates critical vitals to the AI Triage Agent
  - Proxies the FHIR CapabilityStatement to avoid browser CORS restrictions
  - Hosts the FHIR Server Agent, which accepts natural language queries against IRIS

Architecture note:
  All FHIR communication uses the internal Docker network address (fhir-template:52773)
  rather than the externally exposed port. This keeps traffic inside the container
  network and avoids unnecessary round-trips through the host machine.
"""

import sys, os
# Ensure the agent directory is on the path regardless of where uvicorn is invoked from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agent'))
sys.path.insert(0, '/app/agent')


import httpx
import asyncio
import random
import json
import uuid
import os
import sys
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from config import FHIR_BASE, FHIR_AUTH, FHIR_HEADERS, IRIS_BASE, APP_TITLE, APP_VERSION, APP_DESC
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

# Agent layer lives in /app/agent — add to path before importing
sys.path.insert(0, "/app/agent")
from orchestrator import orchestrate
from knowledge_base import search_clinical_guidelines  # triggers RAG initialisation on startup

# ── FHIR connection settings ─────────────────────────────────────────────────
# The base URL and credentials are injected via environment variables in
# docker-compose.yml so they never need to be hardcoded. The defaults point
# to the IRIS container on the internal Docker network.
# Connection settings imported from config.py

# ── FastAPI application ───────────────────────────────────────────────────────
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESC,
    version=APP_VERSION
)

# Allow all origins during development — the app runs locally inside Docker
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Serve static HTML/CSS/JS assets from the /app/static directory
app.mount("/static", StaticFiles(directory="/app/static"), name="static")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE ROUTES — serve the four frontend pages
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def serve_frontend():
    """Main Triage Chat — multi-agent clinical assistant."""
    return FileResponse("/app/static/index.html")

@app.get("/dashboard")
def serve_dashboard():
    """Analytics Dashboard — live FHIR resource counts and triage activity."""
    return FileResponse("/app/static/dashboard.html")

@app.get("/vitals")
def serve_vitals_monitor():
    """Live Vitals Monitor — SSE stream with AI-triggered critical alerts."""
    return FileResponse("/app/static/vitals.html")

@app.get("/agent-builder")
async def agent_builder_page():
    """Serve the custom agent builder UI."""
    return FileResponse("/app/static/agent_builder.html")


@app.get("/fhir-agent")
def serve_fhir_agent():
    """FHIR Server Agent — natural language interface to IRIS FHIR R4."""
    return FileResponse("/app/static/fhir_agent.html")


# ═══════════════════════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOM AGENT MANAGEMENT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/agents")
async def list_agents():
    """Return all custom agents."""
    from dynamic_agent import load_all_agents
    agents = load_all_agents()
    return {"agents": agents, "count": len(agents)}


@app.post("/agents/create")
async def create_agent(request: Request):
    """Create or update a custom agent."""
    import re
    from dynamic_agent import save_agent
    data = await request.json()
    if not data.get("name") or not data.get("system_prompt"):
        return JSONResponse({"error": "name and system_prompt are required"}, status_code=400)
    agent_id = data.get("id") or re.sub(r"[^a-z0-9]+", "-", data["name"].lower()).strip("-")
    config = {
        "id":                  agent_id,
        "name":                data["name"],
        "description":         data.get("description", ""),
        "specialty":           data.get("specialty", "custom"),
        "system_prompt":       data["system_prompt"],
        "temperature":         float(data.get("temperature", 0.2)),
        "max_iterations":      int(data.get("max_iterations", 10)),
        "tools":               data.get("tools", ["get_patient","get_patient_conditions","get_patient_allergies","get_patient_medications","search_clinical_guidelines"]),
        "rag_enabled":         bool(data.get("rag_enabled", True)),
        "routing_description": data.get("routing_description", f"For {data['name']} related questions"),
        "color":               data.get("color", "#a78bfa"),
        "icon":                data.get("icon", "🤖"),
        "created_at":          data.get("created_at", ""),
    }
    save_agent(config)
    print(f"Agent Builder: saved '{config['name']}' (id={agent_id})")
    return {"success": True, "agent": config}


@app.delete("/agents/{agent_id}")
async def delete_agent_route(agent_id: str):
    """Delete a custom agent."""
    from dynamic_agent import delete_agent
    if not delete_agent(agent_id):
        return JSONResponse({"error": f"Agent not found: {agent_id}"}, status_code=404)
    return {"success": True, "deleted": agent_id}


@app.post("/agents/{agent_id}/test")
async def test_agent(agent_id: str, request: Request):
    """Test a custom agent with a single message."""
    from dynamic_agent import run_custom_agent, get_agent_config, save_agent
    data = await request.json()
    # If config passed inline (not yet saved), save temporarily
    if data.get("config"):
        save_agent(data["config"])
    message = data.get("message", "Hello, please introduce yourself.")
    test_session = f"test-{agent_id}"
    response = run_custom_agent(agent_id, test_session, message)
    return {"response": response, "agent_id": agent_id}


@app.get("/health")
def health_check():
    """
    Simple liveness probe used by Docker and monitoring tools.
    Returns the names of all registered agent types so operators
    can confirm the full agent network loaded correctly.
    """
    return {
        "status": "healthy",
        "service": "IRIS FHIR Agents",
        "version": "2.0.0",
        "agents": ["triage", "specialist", "pharmacy", "fhir_server"]
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  TRIAGE CHAT — multi-agent orchestration endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    session_id: str | None = None  # Omit on first turn; server creates one
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    agent_used: str   # TRIAGE | SPECIALIST | PHARMACY
    turn: int


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint consumed by the Triage Chat frontend.

    The orchestrator examines the message and routes it to whichever
    specialist agent is most appropriate — Triage for symptoms and
    urgency assessment, Specialist for condition analysis and referrals,
    or Pharmacy for medication interactions and safety checks.

    Session IDs persist conversation context across turns so agents
    remember what was discussed earlier in the same session.
    """
    try:
        # Generate a new session ID on the first message of every conversation
        session_id = request.session_id or str(uuid.uuid4())
        result = orchestrate(session_id, request.message)
        return ChatResponse(
            session_id=session_id,
            response=result["response"],
            agent_used=result["agent_used"],
            turn=result["turn"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/new")
def new_session(session_id: str):
    """
    Clear all in-memory context for a given session.

    Called when the user clicks 'New Triage Session' in the UI.
    Removes state from every agent's session store so the next message
    starts a completely fresh conversation without prior context bleeding in.
    """
    from orchestrator import session_context
    from triage_agent import sessions
    from specialist_agent import specialist_sessions
    from pharmacy_agent import pharmacy_sessions

    for store in [session_context, sessions, specialist_sessions, pharmacy_sessions]:
        if session_id in store:
            del store[session_id]

    return {"status": "session reset", "session_id": session_id}


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYTICS — FHIR data for the Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/analytics/summary")
def get_analytics_summary():
    """
    Return resource counts across the six core FHIR resource types.

    Uses the FHIR _summary=count parameter which is far more efficient
    than fetching full bundles — IRIS returns just the total without
    transmitting any resource content.
    """
    try:
        def fhir_count(resource: str) -> int:
            url = f"{FHIR_BASE}/{resource}?_summary=count"
            r = httpx.get(url, auth=FHIR_AUTH, headers=FHIR_HEADERS, timeout=10)
            return r.json().get("total", 0) if r.status_code == 200 else 0

        return {
            "patients":         fhir_count("Patient"),
            "conditions":       fhir_count("Condition"),
            "medications":      fhir_count("MedicationRequest"),
            "allergies":        fhir_count("AllergyIntolerance"),
            "observations":     fhir_count("Observation"),
            "service_requests": fhir_count("ServiceRequest")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/conditions")
def get_conditions_breakdown():
    """
    Aggregate active conditions across all patients and return the
    top 8 by frequency. The dashboard uses this to render the
    'Top Active Conditions' bar chart.

    Filters to clinical-status=active to exclude resolved conditions
    that would skew the population health picture.
    """
    try:
        url = f"{FHIR_BASE}/Condition?_count=100&clinical-status=active"
        r = httpx.get(url, auth=FHIR_AUTH, headers=FHIR_HEADERS, timeout=10)
        entries = r.json().get("entry", [])

        condition_counts: dict[str, int] = {}
        for e in entries:
            coding = e.get("resource", {}).get("code", {}).get("coding", [{}])[0]
            display = coding.get("display", "Unknown")
            condition_counts[display] = condition_counts.get(display, 0) + 1

        # Sort descending — most prevalent conditions first
        sorted_conditions = sorted(
            condition_counts.items(), key=lambda x: x[1], reverse=True
        )[:8]
        return {"conditions": [{"name": k, "count": v} for k, v in sorted_conditions]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/observations")
def get_recent_observations():
    """
    Retrieve the 10 most recent triage Observations written by the AI agents.

    Agents write FHIR Observations tagged with category=survey to record
    symptom assessments. This endpoint surfaces those back to the dashboard
    so operators can see real triage activity rather than static demo data.
    """
    try:
        url = f"{FHIR_BASE}/Observation?_count=20&category=survey&_sort=-_lastUpdated"
        r = httpx.get(url, auth=FHIR_AUTH, headers=FHIR_HEADERS, timeout=10)
        entries = r.json().get("entry", [])

        obs = []
        for e in entries:
            resource = e.get("resource", {})
            coding = resource.get("code", {}).get("coding", [{}])[0]
            patient_ref = resource.get("subject", {}).get("reference", "").replace("Patient/", "")
            obs.append({
                "id":       resource.get("id", ""),
                "symptom":  coding.get("display", "Unknown"),
                "severity": resource.get("valueString", "unknown"),
                "patient":  patient_ref,
                "date":     resource.get("effectiveDateTime", "")
            })
        return {"observations": obs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/service-requests")
def get_service_requests():
    """
    Retrieve the 10 most recent ServiceRequests written by the AI agents.

    When the Triage Agent determines a patient needs follow-up — an urgent
    referral, an imaging order, or a specialist consult — it writes a FHIR
    ServiceRequest. This endpoint exposes those back to the dashboard.
    """
    try:
        url = f"{FHIR_BASE}/ServiceRequest?_count=50&_sort=-_lastUpdated"
        r = httpx.get(url, auth=FHIR_AUTH, headers=FHIR_HEADERS, timeout=10)
        entries = r.json().get("entry", [])

        requests = []
        for e in entries:
            resource = e.get("resource", {})
            patient_ref = resource.get("subject", {}).get("reference", "").replace("Patient/", "")
            reason = ""
            if resource.get("reasonCode"):
                reason = resource["reasonCode"][0].get("text", "")
            requests.append({
                "id":       resource.get("id", ""),
                "priority": resource.get("priority", "routine"),
                "patient":  patient_ref,
                "reason":   reason,
                "status":   resource.get("status", "")
            })
        return {"service_requests": requests}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/patients")
def get_patients_list():
    """
    Return the patient roster for the Dashboard patient browser.

    Limited to 20 patients to keep response times fast — the demo data set
    contains 10 rich patients so this ceiling is never hit in practice,
    but the limit prevents accidental full-table scans in production.
    """
    try:
        url = f"{FHIR_BASE}/Patient?_count=20"
        r = httpx.get(url, auth=FHIR_AUTH, headers=FHIR_HEADERS, timeout=10)
        entries = r.json().get("entry", [])

        patients = []
        for e in entries:
            resource = e.get("resource", {})
            name = resource.get("name", [{}])[0]
            full_name = f"{' '.join(name.get('given', []))} {name.get('family', '')}".strip()
            patients.append({
                "id":        resource.get("id", ""),
                "name":      full_name,
                "gender":    resource.get("gender", "unknown"),
                "birthDate": resource.get("birthDate", "")
            })
        return {"patients": patients}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE VITALS — SSE streaming with FHIR writes and AI escalation
# ═══════════════════════════════════════════════════════════════════════════════

# Per-patient vital state — values drift between readings to simulate a real
# bedside monitor rather than producing uncorrelated random noise each tick
vitals_state: dict = {}

# In-memory log of AI-generated critical alerts, capped at 20 entries
ai_alerts_log: list = []


def generate_vitals(patient_id: str) -> dict:
    """
    Produce one set of vital sign readings for the given patient.

    Values drift gradually from their previous reading with small random
    changes. Occasional larger 'spikes' simulate real physiological events
    (a patient coughing, moving, or genuinely deteriorating). This approach
    produces realistic-looking waveforms on the sparklines rather than the
    jagged noise you'd get from purely random values.

    Clinical thresholds used here follow standard nursing alert parameters:
      HR:   <50 or >120 = critical, <60 or >100 = warning
      BP:   systolic >180 or <90 = critical, >140 = warning (Stage 2 HTN)
      SpO2: <90% = critical (respiratory failure threshold), <95% = warning
      Temp: >39.5°C = critical (high fever), >38°C = warning
      RR:   <10 or >28 = critical (respiratory compromise), <12 or >20 = warning
    """
    # Initialise state for new patients with healthy baseline values
    if patient_id not in vitals_state:
        vitals_state[patient_id] = {
            "heart_rate": 72,
            "systolic_bp": 120,
            "diastolic_bp": 80,
            "spo2": 98.0,
            "temperature": 36.8,
            "respiratory_rate": 16
        }

    state = vitals_state[patient_id]

    def drift(val, min_v, max_v, step=2, spike_chance=0.05, spike_size=15):
        """Apply a small random walk to a vital sign, clamped to safe bounds."""
        change = random.uniform(-step, step)
        # Occasional spike simulates a real physiological event
        if random.random() < spike_chance:
            change = random.uniform(-spike_size, spike_size)
        return round(max(min_v, min(max_v, val + change)), 1)

    state["heart_rate"]       = drift(state["heart_rate"],       45,  150, 3,   0.05, 20)
    state["systolic_bp"]      = drift(state["systolic_bp"],      80,  200, 4,   0.04, 25)
    state["diastolic_bp"]     = drift(state["diastolic_bp"],     50,  130, 3,   0.04, 15)
    state["spo2"]             = drift(state["spo2"],             85,  100, 0.5, 0.03, 5)
    state["temperature"]      = drift(state["temperature"],      35.0, 40.5, 0.1, 0.03, 1.5)
    state["respiratory_rate"] = drift(state["respiratory_rate"], 8,   35,  1,   0.04, 8)

    # ── Per-vital status classification ──────────────────────────────────────
    def hr_status(v):
        if v < 50 or v > 120: return "critical"
        if v < 60 or v > 100: return "warning"
        return "normal"

    def bp_status(s, d):
        if s > 180 or s < 90 or d > 120: return "critical"
        if s > 140 or s < 100 or d > 90: return "warning"
        return "normal"

    def spo2_status(v):
        if v < 90: return "critical"   # Below this, cells start dying
        if v < 95: return "warning"
        return "normal"

    def temp_status(v):
        if v > 39.5 or v < 35.5: return "critical"
        if v > 38.0 or v < 36.0: return "warning"
        return "normal"

    def rr_status(v):
        if v < 10 or v > 28: return "critical"
        if v < 12 or v > 20: return "warning"
        return "normal"

    vitals = {
        "patient_id": patient_id,
        "timestamp":  datetime.utcnow().isoformat(),
        "heart_rate": {
            "value": state["heart_rate"], "unit": "bpm",
            "status": hr_status(state["heart_rate"])
        },
        "blood_pressure": {
            "systolic": state["systolic_bp"], "diastolic": state["diastolic_bp"],
            "unit": "mmHg", "status": bp_status(state["systolic_bp"], state["diastolic_bp"])
        },
        "spo2": {
            "value": round(state["spo2"], 1), "unit": "%",
            "status": spo2_status(state["spo2"])
        },
        "temperature": {
            "value": state["temperature"], "unit": "°C",
            "status": temp_status(state["temperature"])
        },
        "respiratory_rate": {
            "value": state["respiratory_rate"], "unit": "br/min",
            "status": rr_status(state["respiratory_rate"])
        }
    }

    # Roll up individual statuses — any critical vital makes the whole reading critical
    all_statuses = [v["status"] for k, v in vitals.items() if isinstance(v, dict) and "status" in v]
    vitals["overall_status"] = (
        "critical" if "critical" in all_statuses else
        "warning"  if "warning"  in all_statuses else
        "normal"
    )
    return vitals


async def write_vital_to_fhir(
    patient_id: str, vital_name: str, value: float,
    unit: str, loinc_code: str, loinc_display: str
):
    """
    Persist a single vital sign reading as a FHIR R4 Observation in IRIS.

    Each reading produces a properly structured Observation with:
      - category: vital-signs (standard FHIR category code)
      - code: LOINC code so the reading is machine-readable by any FHIR system
      - subject: reference back to the patient
      - valueQuantity: the numeric value with UCUM unit

    Failures are swallowed intentionally — a FHIR write error should never
    interrupt the SSE stream that the frontend is reading. The monitor
    keeps displaying live data even if persistence is temporarily unavailable.
    """
    try:
        observation = {
            "resourceType": "Observation",
            "status": "preliminary",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "vital-signs",
                    "display": "Vital Signs"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": loinc_code,
                    "display": loinc_display
                }]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": datetime.utcnow().isoformat() + "Z",
            "valueQuantity": {
                "value": value,
                "unit": unit,
                "system": "http://unitsofmeasure.org"
            }
        }
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{FHIR_BASE}/Observation",
                json=observation,
                auth=FHIR_AUTH,
                headers={
                    "Content-Type": "application/fhir+json",
                    "Accept": "application/fhir+json"
                },
                timeout=5
            )
    except Exception:
        pass  # Stream continuity takes priority over write confirmation


@app.get("/vitals/stream/{patient_id}")
async def stream_vitals(patient_id: str):
    """
    SSE endpoint that streams vital signs every 2 seconds.

    On each tick:
      1. Generate new vitals via the drift model
      2. Persist all five readings to IRIS as FHIR Observations
      3. If the overall status is critical AND we haven't fired an alert
         recently, dispatch an async AI assessment via the Triage Agent

    The 15-reading cooldown (~30 seconds) prevents alert storms when a
    patient stays in a critical state — the AI gets one chance to respond
    before the next escalation is permitted.

    SSE headers disable Nginx proxy buffering (X-Accel-Buffering: no)
    so the browser receives events in real time rather than in batches.
    """
    async def event_generator():
        last_alert_status = "normal"
        alert_cooldown = 0

        while True:
            vitals = generate_vitals(patient_id)

            # Write all five vitals to IRIS concurrently using LOINC codes
            await write_vital_to_fhir(patient_id, "heart_rate",       vitals["heart_rate"]["value"],       "beats/minute", "8867-4",  "Heart rate")
            await write_vital_to_fhir(patient_id, "spo2",             vitals["spo2"]["value"],             "%",            "59408-5", "Oxygen saturation")
            await write_vital_to_fhir(patient_id, "blood_pressure",   vitals["blood_pressure"]["systolic"],"mmHg",         "8480-6",  "Systolic blood pressure")
            await write_vital_to_fhir(patient_id, "temperature",      vitals["temperature"]["value"],      "Cel",          "8310-5",  "Body temperature")
            await write_vital_to_fhir(patient_id, "respiratory_rate", vitals["respiratory_rate"]["value"], "breaths/min",  "9279-1",  "Respiratory rate")

            # ── AI escalation logic ───────────────────────────────────────────
            if vitals["overall_status"] == "critical" and alert_cooldown <= 0:
                if last_alert_status != "critical":
                    # Identify which specific vitals crossed the threshold
                    critical_vitals = []
                    if vitals["heart_rate"]["status"]       == "critical": critical_vitals.append(f"HR {vitals['heart_rate']['value']} bpm")
                    if vitals["blood_pressure"]["status"]   == "critical": critical_vitals.append(f"BP {vitals['blood_pressure']['systolic']}/{vitals['blood_pressure']['diastolic']} mmHg")
                    if vitals["spo2"]["status"]             == "critical": critical_vitals.append(f"SpO2 {vitals['spo2']['value']}%")
                    if vitals["temperature"]["status"]      == "critical": critical_vitals.append(f"Temp {vitals['temperature']['value']}°C")
                    if vitals["respiratory_rate"]["status"] == "critical": critical_vitals.append(f"RR {vitals['respiratory_rate']['value']} br/min")

                    alert_msg = (
                        f"CRITICAL VITALS ALERT for patient {patient_id}: "
                        f"{', '.join(critical_vitals)}. "
                        f"Please assess urgency immediately based on clinical guidelines."
                    )

                    # Fire and forget — don't block the SSE stream waiting for the AI
                    asyncio.create_task(trigger_ai_alert(patient_id, alert_msg, vitals))

                    vitals["ai_alert_triggered"] = True
                    vitals["ai_alert_message"] = alert_msg
                    alert_cooldown = 15  # ~30 second cooling-off period

                last_alert_status = "critical"
            else:
                if vitals["overall_status"] != "critical":
                    last_alert_status = "normal"
                alert_cooldown = max(0, alert_cooldown - 1)

            yield f"data: {json.dumps(vitals)}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # Disable Nginx buffering for real-time delivery
        }
    )


async def trigger_ai_alert(patient_id: str, alert_msg: str, vitals: dict):
    """
    Run the Triage Agent in a thread pool when a patient's vitals go critical.

    The agent runs synchronously (LangChain blocking call) so we offload it
    to a thread executor rather than calling it directly in the async event
    loop, which would block all other SSE streams while the LLM responds.

    The alert is stored in ai_alerts_log and the frontend polls /vitals/alerts
    every 5 seconds to display the AI assessment alongside the vitals.
    """
    try:
        print(f"VITALS ALERT: Triggering AI assessment for patient {patient_id}")

        # Unique session per alert so each escalation gets a clean context
        alert_session_id = f"vitals-alert-{patient_id}-{int(datetime.utcnow().timestamp())}"

        # Inject patient context so the agent doesn't ask who the patient is
        context_msg = (
            f"[Context: Patient ID is {patient_id}. "
            f"AUTOMATED VITALS ALERT — Do NOT ask for patient ID.] "
            f"{alert_msg}"
        )

        # Offload blocking LLM call to thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: orchestrate(alert_session_id, context_msg)
        )

        alert_entry = {
            "patient_id":     patient_id,
            "timestamp":      datetime.utcnow().isoformat(),
            "vitals_summary": alert_msg,
            "ai_response":    response,
            "vitals": {
                "heart_rate":  vitals["heart_rate"]["value"],
                "bp":          f"{vitals['blood_pressure']['systolic']}/{vitals['blood_pressure']['diastolic']}",
                "spo2":        vitals["spo2"]["value"],
                "temperature": vitals["temperature"]["value"],
                "rr":          vitals["respiratory_rate"]["value"]
            }
        }

        ai_alerts_log.append(alert_entry)

        # Rolling window — keep only the 20 most recent alerts to bound memory
        if len(ai_alerts_log) > 20:
            ai_alerts_log.pop(0)

        print(f"VITALS ALERT: AI assessment complete for patient {patient_id}")

    except Exception as e:
        print(f"VITALS ALERT ERROR: {e}")


@app.get("/vitals/alerts")
def get_vitals_alerts():
    """
    Return AI-generated critical alerts for display in the Vitals Monitor.
    The frontend polls this every 5 seconds after a critical event fires.
    """
    return {"alerts": ai_alerts_log}


@app.get("/vitals/snapshot/{patient_id}")
def get_vitals_snapshot(patient_id: str):
    """Single vitals reading for a patient — useful for testing."""
    return generate_vitals(patient_id)


# ═══════════════════════════════════════════════════════════════════════════════
#  FHIR SERVER AGENT — natural language interface to IRIS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/fhir-agent/chat")
async def fhir_agent_endpoint(request: Request):
    """
    Chat endpoint for the FHIR Server Agent.

    Unlike the clinical Triage Chat, this agent is aimed at clinicians and
    developers who want to explore the FHIR server using plain English — asking
    questions like 'show me all diabetic patients' or 'what does the Patient
    resource support?' without needing to know FHIR query syntax.

    Sessions are scoped to the browser tab using a short random ID so
    multiple users can query the server independently.
    """
    try:
        body = await request.json()
        message = body.get("message", "")
        session_id = body.get("session_id", f"fhir-agent-{uuid.uuid4().hex[:8]}")

        if not message:
            raise HTTPException(status_code=400, detail="Message required")

        from fhir_agent import chat as fhir_agent_chat
        response = fhir_agent_chat(session_id, message)
        return {"response": response, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fhir-agent/status")
async def fhir_server_status():
    """
    Probe the IRIS FHIR server and return its reported identity.

    The FHIR CapabilityStatement (metadata endpoint) is the standard way
    to ask any FHIR server what it is and what it supports. We surface
    the software name and FHIR version to the sidebar status panel so
    users can confirm they're connected to a healthy IRIS instance.
    """
    try:
        r = httpx.get(
            f"{FHIR_BASE}/metadata",
            auth=FHIR_AUTH,
            headers=FHIR_HEADERS,
            timeout=5
        )
        meta = r.json()
        return {
            "status":       "online",
            "fhir_version": meta.get("fhirVersion", "R4"),
            "software":     meta.get("software", {}).get("name", "IRIS"),
            "endpoint":     FHIR_BASE
        }
    except Exception as e:
        return {"status": "offline", "error": str(e)}


@app.get("/fhir/metadata")
async def get_fhir_metadata():
    """
    Proxy the FHIR CapabilityStatement through the FastAPI server.

    Browsers enforce the Same-Origin Policy — a page served from
    localhost:8000 cannot fetch directly from localhost:32783 without
    CORS headers. Rather than configuring CORS on IRIS, we proxy the
    metadata request here so the Capability Explorer tab can load the
    full CapabilityStatement without browser security errors.
    """
    try:
        r = httpx.get(
            f"{FHIR_BASE}/metadata",
            auth=FHIR_AUTH,
            headers=FHIR_HEADERS,
            timeout=15
        )
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))