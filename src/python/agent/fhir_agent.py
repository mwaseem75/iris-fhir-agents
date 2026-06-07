"""
from config import FHIR_BASE, FHIR_AUTH, FHIR_HEADERS, IRIS_BASE, LLM_MODEL, TEMP_SPECIALIST
fhir_agent.py — IRIS FHIR Server Agent
=======================================
A LangChain agent that provides a natural language interface to the
InterSystems IRIS FHIR R4 server.

Unlike the three clinical agents (Triage, Specialist, Pharmacy) which operate
on a single patient within a guided conversation, the FHIR Server Agent is
an open-ended exploration tool. It operates across the entire patient population
and exposes capabilities that would otherwise require knowing FHIR query syntax
or IRIS SQL — neither of which should be required to ask "which patients are on
Warfarin and have a penicillin allergy?"

Who uses this agent:
  - Clinicians wanting a quick population overview without opening an EHR
  - Developers building on IRIS who want to understand what data is available
  - Contest judges exploring the breadth of the IRIS FHIR implementation

Tools provided (11 total):
  get_patient_list            — roster with demographics, filterable by FHIR params
  get_conditions_for_patient  — active diagnoses with onset dates and notes
  get_medications_for_patient — active prescriptions with dosage and indication
  get_observations_for_patient— labs and vitals, optionally filtered by category
  get_allergies_for_patient   — allergens with criticality and reaction details
  search_patients_by_condition— reverse lookup: condition name → matching patients
  get_fhir_statistics         — server-wide resource counts across 7 resource types
  get_full_patient_summary    — all of the above in one call (5 FHIR requests)
  query_iris_sql              — direct SELECT access to IRIS SQL for complex analytics
  get_procedures_for_patient  — procedure history with dates and notes
  search_clinical_guidelines  — RAG vector search against 50 clinical guidelines

Agent configuration:
  temperature=0.2 (vs 0 for the router) — this agent benefits from slightly more
  natural prose in its responses while still being factually disciplined.
  max_iterations=8 — enough for complex multi-tool queries (e.g. "summarise all
  diabetic patients and enrich with ADA guidelines") without runaway loops.
"""

import os
import httpx
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from knowledge_base import search_clinical_guidelines
from config import FHIR_BASE, FHIR_AUTH, FHIR_HEADERS, IRIS_BASE, LLM_MODEL, TEMP_SPECIALIST

# ── FHIR / IRIS connection ────────────────────────────────────────────────────
# Connection settings imported from config.py

# ── LLM ──────────────────────────────────────────────────────────────────────
# temperature=0.2 gives natural flowing responses while keeping clinical facts
# grounded. Higher values risk the agent embellishing lab values or drug names.
llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=TEMP_SPECIALIST,
    api_key=os.getenv("OPENAI_API_KEY")
)


# ═══════════════════════════════════════════════════════════════════════════════
#  PATIENT TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_patient_list(query_params: str = "") -> str:
    """
    Retrieve a list of patients from IRIS FHIR R4 server.
    query_params: FHIR search parameters e.g. 'gender=female&_count=10'
    Leave empty to get all patients.
    Returns patient names, IDs, gender, and birth dates.
    """
    try:
        # _count=50 is a safe ceiling — large enough to cover all demo patients,
        # small enough to avoid timeouts on production servers with thousands of records
        url = f"{FHIR_BASE}/Patient?_count=50"
        if query_params:
            url += f"&{query_params}"

        r = httpx.get(url, auth=FHIR_AUTH, headers=FHIR_HEADERS, timeout=15)
        bundle = r.json()
        entries = bundle.get("entry", [])

        if not entries:
            return "No patients found matching the criteria."

        results = []
        for e in entries:
            pt = e.get("resource", {})
            name = pt.get("name", [{}])[0]
            full = f"{' '.join(name.get('given', []))} {name.get('family', '')}".strip()
            results.append({
                "id":        pt.get("id"),
                "name":      full or "Unknown",
                "gender":    pt.get("gender", "unknown"),
                "birthDate": pt.get("birthDate", "unknown")
            })

        # bundle.total reflects the server-side count; len(results) is what we received
        total = bundle.get("total", len(results))
        summary = f"Found {total} patient(s):\n"
        for p in results:
            summary += f"  • {p['name']} (ID: {p['id']}) — {p['gender']}, DOB: {p['birthDate']}\n"
        return summary
    except Exception as e:
        return f"Error querying patients: {str(e)}"


@tool
def get_conditions_for_patient(patient_id: str) -> str:
    """
    Get all active medical conditions for a specific patient from IRIS FHIR R4.
    patient_id: The FHIR patient ID e.g. 'pt-001'
    Returns list of conditions with onset dates.
    """
    try:
        # clinical-status=active excludes resolved and entered-in-error conditions
        url = f"{FHIR_BASE}/Condition?patient={patient_id}&clinical-status=active&_count=50"
        r = httpx.get(url, auth=FHIR_AUTH, headers=FHIR_HEADERS, timeout=15)
        entries = r.json().get("entry", [])

        if not entries:
            return f"No active conditions found for patient {patient_id}."

        results = []
        for e in entries:
            cond   = e.get("resource", {})
            code   = cond.get("code", {})
            coding = code.get("coding", [{}])[0]
            # Prefer coding display, fall back to code.text (some IRIS resources use text only)
            display   = coding.get("display") or code.get("text", "Unknown condition")
            onset     = cond.get("onsetDateTime", "unknown onset")
            note      = cond.get("note", [{}])
            note_text = note[0].get("text", "") if note else ""
            results.append(
                f"  • {display} (onset: {onset})"
                f"{' — ' + note_text if note_text else ''}"
            )

        return f"Active conditions for patient {patient_id}:\n" + "\n".join(results)
    except Exception as e:
        return f"Error querying conditions: {str(e)}"


@tool
def get_medications_for_patient(patient_id: str) -> str:
    """
    Get all active medications for a specific patient from IRIS FHIR R4.
    patient_id: The FHIR patient ID e.g. 'pt-001'
    Returns medication names, dosages, and reasons.
    """
    try:
        url = f"{FHIR_BASE}/MedicationRequest?patient={patient_id}&status=active&_count=50"
        r = httpx.get(url, auth=FHIR_AUTH, headers=FHIR_HEADERS, timeout=15)
        entries = r.json().get("entry", [])

        if not entries:
            return f"No active medications found for patient {patient_id}."

        results = []
        for e in entries:
            med      = e.get("resource", {})
            med_code = med.get("medicationCodeableConcept", {})
            coding   = med_code.get("coding", [{}])[0]
            name     = coding.get("display") or med_code.get("text", "Unknown medication")
            # dosageInstruction is a list; we take the first (most common case)
            dosage      = med.get("dosageInstruction", [{}])[0].get("text", "")
            reason      = med.get("reasonCode", [{}])
            reason_text = reason[0].get("text", "") if reason else ""
            results.append(
                f"  • {name}"
                f"{' — ' + dosage if dosage else ''}"
                f"{' (' + reason_text + ')' if reason_text else ''}"
            )

        return f"Active medications for patient {patient_id}:\n" + "\n".join(results)
    except Exception as e:
        return f"Error querying medications: {str(e)}"


@tool
def get_observations_for_patient(patient_id: str, category: str = "") -> str:
    """
    Get clinical observations (labs, vitals) for a patient from IRIS FHIR R4.
    patient_id: The FHIR patient ID
    category: Optional — 'laboratory' or 'vital-signs'
    Returns observation values with dates.
    """
    try:
        # _sort=-date returns most recent first — most useful for clinical review
        url = f"{FHIR_BASE}/Observation?patient={patient_id}&_count=20&_sort=-date"
        if category:
            url += f"&category={category}"

        r = httpx.get(url, auth=FHIR_AUTH, headers=FHIR_HEADERS, timeout=15)
        entries = r.json().get("entry", [])

        if not entries:
            return f"No observations found for patient {patient_id}."

        results = []
        for e in entries:
            obs    = e.get("resource", {})
            code   = obs.get("code", {}).get("coding", [{}])[0].get("display", "Unknown")
            val    = obs.get("valueQuantity", {})
            # Observations can carry either a numeric valueQuantity or a free-text valueString
            val_str = (
                f"{val.get('value', '?')} {val.get('unit', '')}".strip()
                if val else obs.get("valueString", "no value")
            )
            date   = obs.get("effectiveDateTime", "unknown date")[:10]
            # Interpretation (H/L/N) is present on lab results — surface it when available
            interp = (
                obs.get("interpretation", [{}])[0].get("coding", [{}])[0].get("display", "")
                if obs.get("interpretation") else ""
            )
            results.append(
                f"  • {code}: {val_str} ({date})"
                f"{' — ' + interp if interp else ''}"
            )

        return f"Observations for patient {patient_id}:\n" + "\n".join(results)
    except Exception as e:
        return f"Error querying observations: {str(e)}"


@tool
def get_allergies_for_patient(patient_id: str) -> str:
    """
    Get all allergies and intolerances for a patient from IRIS FHIR R4.
    patient_id: The FHIR patient ID
    Returns allergen names, criticality, and reactions.
    """
    try:
        url = f"{FHIR_BASE}/AllergyIntolerance?patient={patient_id}&_count=50"
        r = httpx.get(url, auth=FHIR_AUTH, headers=FHIR_HEADERS, timeout=15)
        entries = r.json().get("entry", [])

        if not entries:
            return f"No allergies found for patient {patient_id}."

        results = []
        for e in entries:
            allergy     = e.get("resource", {})
            code        = allergy.get("code", {})
            name        = code.get("coding", [{}])[0].get("display") or code.get("text", "Unknown")
            criticality = allergy.get("criticality", "unknown")
            reactions   = allergy.get("reaction", [])
            reaction_text = ""
            if reactions:
                # Reaction manifestation describes the clinical presentation (e.g. anaphylaxis)
                manifestations = reactions[0].get("manifestation", [{}])
                reaction_text  = manifestations[0].get("coding", [{}])[0].get("display", "")

            results.append(
                f"  • {name} — criticality: {criticality}"
                f"{' — reaction: ' + reaction_text if reaction_text else ''}"
            )

        return f"Allergies for patient {patient_id}:\n" + "\n".join(results)
    except Exception as e:
        return f"Error querying allergies: {str(e)}"


@tool
def get_procedures_for_patient(patient_id: str) -> str:
    """
    Get all procedures performed for a patient from IRIS FHIR R4.
    patient_id: The FHIR patient ID
    Returns procedure names, dates, and notes.
    """
    try:
        url = f"{FHIR_BASE}/Procedure?patient={patient_id}&_count=20&_sort=-date"
        r = httpx.get(url, auth=FHIR_AUTH, headers=FHIR_HEADERS, timeout=15)
        entries = r.json().get("entry", [])

        if not entries:
            return f"No procedures found for patient {patient_id}."

        results = []
        for e in entries:
            proc      = e.get("resource", {})
            code      = proc.get("code", {}).get("coding", [{}])[0].get("display", "Unknown procedure")
            date      = proc.get("performedDateTime", "unknown date")[:10]
            note      = proc.get("note", [{}])
            # Truncate long procedure notes at 100 chars to keep the response readable
            note_text = note[0].get("text", "")[:100] if note else ""
            results.append(
                f"  • {code} ({date})"
                f"{' — ' + note_text if note_text else ''}"
            )

        return f"Procedures for patient {patient_id}:\n" + "\n".join(results)
    except Exception as e:
        return f"Error querying procedures: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
#  POPULATION TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def search_patients_by_condition(condition_name: str) -> str:
    """
    Find all patients who have a specific condition in IRIS FHIR R4.
    condition_name: Condition to search for e.g. 'diabetes', 'hypertension', 'asthma'
    Returns list of patients with that condition.
    """
    try:
        # Fetch all conditions and filter client-side — IRIS FHIR R4 doesn't support
        # full-text search on Condition.code.display via the REST API, so we pull
        # all conditions and do a substring match here.
        url = f"{FHIR_BASE}/Condition?_count=100"
        r = httpx.get(url, auth=FHIR_AUTH, headers=FHIR_HEADERS, timeout=15)
        entries = r.json().get("entry", [])

        matched_patients: dict = {}
        search_lower = condition_name.lower()

        for e in entries:
            cond    = e.get("resource", {})
            code    = cond.get("code", {})
            coding  = code.get("coding", [{}])[0]
            display = (coding.get("display") or code.get("text", "")).lower()

            if search_lower in display:
                pt_id = cond.get("subject", {}).get("reference", "").replace("Patient/", "")
                if pt_id and pt_id not in matched_patients:
                    matched_patients[pt_id] = display

        if not matched_patients:
            return f"No patients found with condition matching '{condition_name}'."

        # Resolve patient IDs to names — one GET per patient is acceptable
        # for the demo dataset (10 patients); a larger system would use _include
        results = []
        for pt_id, cond_display in matched_patients.items():
            try:
                pt_r = httpx.get(
                    f"{FHIR_BASE}/Patient/{pt_id}",
                    auth=FHIR_AUTH, headers=FHIR_HEADERS, timeout=10
                )
                pt   = pt_r.json()
                name = pt.get("name", [{}])[0]
                full = f"{' '.join(name.get('given', []))} {name.get('family', '')}".strip()
                results.append(f"  • {full} (ID: {pt_id}) — {cond_display}")
            except Exception:
                results.append(f"  • Patient {pt_id} — {cond_display}")

        return (
            f"Patients with '{condition_name}' ({len(results)} found):\n"
            + "\n".join(results)
        )
    except Exception as e:
        return f"Error searching by condition: {str(e)}"


@tool
def get_fhir_statistics() -> str:
    """
    Get comprehensive statistics about the IRIS FHIR R4 server.
    Returns counts of all resource types, demographics breakdown,
    most common conditions, and server summary.
    """
    try:
        # _summary=count is the FHIR-standard way to get a resource count
        # without fetching any content — far faster than retrieving full bundles
        resources = [
            "Patient", "Condition", "MedicationRequest",
            "Observation", "AllergyIntolerance", "ServiceRequest", "Procedure"
        ]
        stats: dict = {}
        for res in resources:
            try:
                r = httpx.get(
                    f"{FHIR_BASE}/{res}?_summary=count",
                    auth=FHIR_AUTH, headers=FHIR_HEADERS, timeout=10
                )
                stats[res] = r.json().get("total", 0)
            except Exception:
                stats[res] = "error"

        result  = "IRIS FHIR R4 Server Statistics:\n"
        result += "─" * 40 + "\n"
        for res, count in stats.items():
            result += f"  • {res}: {count}\n"
        result += f"\nFHIR Endpoint: {FHIR_BASE}\n"
        result += "IRIS Version: InterSystems IRIS for Health\n"
        return result
    except Exception as e:
        return f"Error getting statistics: {str(e)}"


@tool
def get_full_patient_summary(patient_id: str) -> str:
    """
    Get a comprehensive clinical summary for a patient including
    demographics, conditions, medications, allergies, and recent observations.
    patient_id: The FHIR patient ID e.g. 'pt-001'
    """
    try:
        sections = []

        # ── Demographics ──────────────────────────────────────────────────────
        pt = httpx.get(
            f"{FHIR_BASE}/Patient/{patient_id}",
            auth=FHIR_AUTH, headers=FHIR_HEADERS, timeout=10
        ).json()
        name      = pt.get("name", [{}])[0]
        full_name = f"{' '.join(name.get('given', []))} {name.get('family', '')}".strip()
        sections.append(f"PATIENT: {full_name} (ID: {patient_id})")
        sections.append(f"Gender: {pt.get('gender', 'unknown')} | DOB: {pt.get('birthDate', 'unknown')}")

        # ── Active conditions ─────────────────────────────────────────────────
        cond_entries = httpx.get(
            f"{FHIR_BASE}/Condition?patient={patient_id}&clinical-status=active&_count=20",
            auth=FHIR_AUTH, headers=FHIR_HEADERS, timeout=10
        ).json().get("entry", [])
        conds = [
            e["resource"].get("code", {}).get("coding", [{}])[0].get("display", "Unknown")
            for e in cond_entries
        ]
        sections.append(f"\nActive Conditions ({len(conds)}):\n" + "\n".join(f"  • {c}" for c in conds))

        # ── Active medications ────────────────────────────────────────────────
        med_entries = httpx.get(
            f"{FHIR_BASE}/MedicationRequest?patient={patient_id}&status=active&_count=20",
            auth=FHIR_AUTH, headers=FHIR_HEADERS, timeout=10
        ).json().get("entry", [])
        meds = [
            e["resource"].get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("display", "Unknown")
            for e in med_entries
        ]
        sections.append(f"\nMedications ({len(meds)}):\n" + "\n".join(f"  • {m}" for m in meds))

        # ── Allergies — highlighted with ⚠ to draw attention ─────────────────
        allergy_entries = httpx.get(
            f"{FHIR_BASE}/AllergyIntolerance?patient={patient_id}&_count=20",
            auth=FHIR_AUTH, headers=FHIR_HEADERS, timeout=10
        ).json().get("entry", [])
        allergies = [
            e["resource"].get("code", {}).get("coding", [{}])[0].get("display", "Unknown")
            for e in allergy_entries
        ]
        sections.append(f"\nAllergies ({len(allergies)}):\n" + "\n".join(f"  ⚠ {a}" for a in allergies))

        # ── Recent observations (last 5) ──────────────────────────────────────
        obs_entries = httpx.get(
            f"{FHIR_BASE}/Observation?patient={patient_id}&_count=5&_sort=-date",
            auth=FHIR_AUTH, headers=FHIR_HEADERS, timeout=10
        ).json().get("entry", [])
        obs_list = []
        for e in obs_entries:
            obs     = e["resource"]
            code    = obs.get("code", {}).get("coding", [{}])[0].get("display", "Unknown")
            val     = obs.get("valueQuantity", {})
            val_str = f"{val.get('value', '?')} {val.get('unit', '')}".strip() if val else "no value"
            obs_list.append(f"  • {code}: {val_str}")
        sections.append(f"\nRecent Observations ({len(obs_list)}):\n" + "\n".join(obs_list))

        return "\n".join(sections)
    except Exception as e:
        return f"Error getting patient summary: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
#  IRIS SQL TOOL
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def query_iris_sql(sql_query: str) -> str:
    """
    Execute a SQL query directly against the IRIS FHIR database.
    Use for complex analytics not possible with standard FHIR queries.
    Example: SELECT COUNT(*) FROM HSFHIR_X0001_S.Patient
    Only SELECT queries are allowed.
    """
    try:
        # Hard-block any non-SELECT statement — the Atelier API runs with
        # full DBA privileges so a destructive query would cause real damage
        if not sql_query.strip().upper().startswith("SELECT"):
            return "Only SELECT queries are allowed for safety."

        url     = f"{IRIS_BASE}/api/atelier/v1/FHIRSERVER/action/query"
        payload = {"query": sql_query, "parameters": []}
        r       = httpx.post(url, json=payload, auth=FHIR_AUTH, timeout=30)
        data    = r.json()

        errors = data.get("status", {}).get("errors", [])
        if errors:
            return f"SQL Error: {errors[0].get('error', 'Unknown error')}"

        content = data.get("result", {}).get("content", [])
        if not content:
            return "Query returned no results."

        # Format results as a readable pipe-delimited table
        headers = list(content[0].keys())
        result  = " | ".join(headers) + "\n"
        result += "-" * (len(result) - 1) + "\n"
        for row in content[:20]:
            result += " | ".join(str(v) for v in row.values()) + "\n"
        if len(content) > 20:
            result += f"... ({len(content)} total rows, showing first 20)"

        return result
    except Exception as e:
        return f"Error executing SQL: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are the IRIS FHIR Agent — an intelligent AI assistant with direct access to an InterSystems IRIS FHIR R4 server.

You are the bridge between natural language and the IRIS FHIR server. You help clinicians, developers, and analysts explore and understand FHIR data without needing to know FHIR query syntax.

Your capabilities:
1. PATIENT QUERIES — Find patients by name, condition, demographics, or ID
2. CLINICAL DATA — Retrieve conditions, medications, allergies, observations, procedures
3. POPULATION ANALYSIS — Analyze trends across the patient population
4. IRIS SQL — Execute direct SQL queries for complex analytics
5. RAG GUIDELINES — Enrich findings with clinical guidelines from IRIS Vector Search
6. SERVER STATISTICS — Report on FHIR server content and health
7. CAPABILITY STATEMENT — Summarize and explain what the IRIS FHIR server supports,
   including all resource types, interactions, search parameters, and operations
8. RESOURCE DETAILS — Deep-dive into any specific FHIR resource type capabilities

MANDATORY WORKFLOW:
- For any patient query: use get_patient_list or get_full_patient_summary
- For condition-based queries: use search_patients_by_condition
- For clinical analysis: use search_clinical_guidelines to enrich findings with evidence
- For statistics: use get_fhir_statistics
- Always cite the FHIR resource type you queried (Patient, Condition, Observation, etc.)

RESPONSE FORMAT:
- Be concise but comprehensive
- Always show patient IDs for traceability
- Highlight critical findings (allergies, high-risk medications)
- Cite clinical guidelines when relevant using: > According to [Source] — [recommendation]
- End with actionable insights when appropriate

IRIS FHIR SERVER: InterSystems IRIS for Health
FHIR VERSION: R4 (4.0.1)
NAMESPACE: FHIRSERVER
"""

# Tool list passed to both the agent constructor and AgentExecutor.
# Order here is cosmetic — LangChain selects tools by name, not position.
tools = [
    get_patient_list,
    get_conditions_for_patient,
    get_medications_for_patient,
    get_observations_for_patient,
    get_allergies_for_patient,
    search_patients_by_condition,
    get_fhir_statistics,
    get_full_patient_summary,
    query_iris_sql,
    get_procedures_for_patient,
    search_clinical_guidelines,   # Imported from knowledge_base.py
]

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")  # LangChain writes tool calls here
])


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

# Each browser session gets its own AgentExecutor with its own
# ConversationBufferMemory so users don't see each other's conversation history.
# Sessions are keyed by the short random ID generated in main.py.
sessions: dict = {}


def get_or_create_session(session_id: str) -> AgentExecutor:
    """
    Return the existing AgentExecutor for this session, or create a new one.

    Each executor gets its own ConversationBufferMemory so the agent remembers
    what was discussed earlier in the same session — asking "now show me their
    medications" after "show me pt-001's conditions" works correctly because
    the agent still has pt-001 in its context window.

    max_iterations=8 allows complex multi-tool queries (find patients with
    diabetes → fetch each one's medications → search RAG guidelines) without
    risking an infinite tool-call loop if the LLM gets confused.
    """
    if session_id not in sessions:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        agent  = create_openai_tools_agent(llm, tools, prompt)
        sessions[session_id] = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,           # Logs tool calls to container stdout for debugging
            max_iterations=8,
            handle_parsing_errors=True  # Recovers gracefully if the LLM produces malformed JSON
        )
    return sessions[session_id]


def chat(session_id: str, message: str) -> str:
    """
    Process one turn of conversation and return the agent's response.
    Called by main.py's /fhir-agent/chat endpoint.
    """
    executor = get_or_create_session(session_id)
    result   = executor.invoke({"input": message})
    return result["output"]