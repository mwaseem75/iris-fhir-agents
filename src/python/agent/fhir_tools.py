"""
fhir_tools.py — FHIR R4 Data Access Layer
==========================================
Provides the LangChain tools that clinical agents use to interact with
the InterSystems IRIS FHIR R4 server. Every FHIR read and write in the
Triage, Specialist, and Pharmacy agents flows through this module.

Design principles:
  - Each tool returns a plain string rather than a dict. LangChain agents
    work with string tool outputs — returning structured data would require
    the agent to parse it, adding a failure mode with no benefit.
  - Tools never raise exceptions to the agent. FHIR errors are caught and
    returned as descriptive strings so the agent can report the problem
    to the user in natural language rather than crashing.
  - Read tools (GET) and write tools (POST) use separate helper functions
    to make the intent at each call site immediately obvious.

FHIR resources used:
  Patient            — demographics (name, DOB, gender)
  Condition          — active diagnoses
  AllergyIntolerance — known allergens and criticality
  MedicationRequest  — active prescriptions
  Observation        — symptom records written by the Triage Agent
  ServiceRequest     — follow-up orders (referrals, consults) written by Triage
"""

import os
import httpx
from langchain.tools import tool
from config import FHIR_BASE, FHIR_AUTH

# ── FHIR connection ───────────────────────────────────────────────────────────
# Connection settings imported from config.py


# ═══════════════════════════════════════════════════════════════════════════════
#  HTTP HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def fhir_get(path: str) -> dict:
    """
    Make an authenticated GET request to the IRIS FHIR server.

    The Accept header is required — without it IRIS may return XML, which
    none of the parsing code downstream is prepared to handle. The 10-second
    timeout is generous enough for IRIS cold-start queries but short enough
    that a network partition doesn't stall an agent indefinitely.
    """
    url = f"{FHIR_BASE}/{path}"
    response = httpx.get(
        url,
        auth=FHIR_AUTH,
        headers={"Accept": "application/fhir+json"},
        timeout=10
    )
    response.raise_for_status()
    return response.json()


def fhir_post(path: str, data: dict) -> dict:
    """
    Create a new FHIR resource by POSTing to its type endpoint.

    IRIS returns the created resource in the response body with the
    server-assigned id. We handle an empty body gracefully — some IRIS
    versions return 201 with no body on successful creation, which would
    cause response.json() to raise a decode error.

    The 30-second timeout is higher than for reads because FHIR writes
    on IRIS can trigger business rule evaluation and indexing.
    """
    url = f"{FHIR_BASE}/{path}"
    response = httpx.post(
        url,
        auth=FHIR_AUTH,
        headers={
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json"
        },
        json=data,
        timeout=30
    )
    response.raise_for_status()
    # Guard against empty response body on 201 Created
    if response.content:
        return response.json()
    return {"id": "created", "status": "success"}


# ═══════════════════════════════════════════════════════════════════════════════
#  READ TOOLS — fetch patient data from IRIS
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_patient(patient_id: str) -> str:
    """Fetch patient demographics from FHIR server by patient ID."""
    try:
        data = fhir_get(f"Patient/{patient_id}")

        # FHIR HumanName is a list — we take the first (official) name
        name = data.get("name", [{}])[0]
        full_name = f"{' '.join(name.get('given', []))} {name.get('family', '')}".strip()
        dob    = data.get("birthDate", "Unknown")
        gender = data.get("gender",    "Unknown")

        return f"Patient: {full_name}, DOB: {dob}, Gender: {gender}"
    except Exception as e:
        return f"Error fetching patient: {str(e)}"


@tool
def get_patient_conditions(patient_id: str) -> str:
    """Get all active medical conditions for a patient."""
    try:
        # clinical-status=active excludes resolved and entered-in-error conditions
        data    = fhir_get(f"Condition?patient={patient_id}&clinical-status=active")
        entries = data.get("entry", [])

        if not entries:
            return "No active conditions found."

        conditions = []
        for e in entries:
            resource = e.get("resource", {})
            coding   = resource.get("code", {}).get("coding", [{}])[0]
            conditions.append(coding.get("display", "Unknown condition"))

        return f"Active conditions: {', '.join(conditions)}"
    except Exception as e:
        return f"Error fetching conditions: {str(e)}"


@tool
def get_patient_allergies(patient_id: str) -> str:
    """Get all known allergies for a patient."""
    try:
        data    = fhir_get(f"AllergyIntolerance?patient={patient_id}")
        entries = data.get("entry", [])

        if not entries:
            return "No known allergies."

        allergies = []
        for e in entries:
            resource    = e.get("resource", {})
            coding      = resource.get("code", {}).get("coding", [{}])[0]
            criticality = resource.get("criticality", "unknown")
            # Criticality (high/low) is clinically significant — included so
            # the Pharmacy agent can weight allergy severity in interaction checks
            allergies.append(
                f"{coding.get('display', 'Unknown')} (criticality: {criticality})"
            )

        return f"Allergies: {', '.join(allergies)}"
    except Exception as e:
        return f"Error fetching allergies: {str(e)}"


@tool
def get_patient_medications(patient_id: str) -> str:
    """Get current medications for a patient."""
    try:
        # status=active filters out stopped and on-hold prescriptions
        data    = fhir_get(f"MedicationRequest?patient={patient_id}&status=active")
        entries = data.get("entry", [])

        if not entries:
            return "No active medications found."

        meds = []
        for e in entries:
            resource = e.get("resource", {})
            coding   = resource.get("medicationCodeableConcept", {}).get("coding", [{}])[0]
            meds.append(coding.get("display", "Unknown medication"))

        return f"Current medications: {', '.join(meds)}"
    except Exception as e:
        return f"Error fetching medications: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
#  WRITE TOOLS — create FHIR resources in IRIS
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def create_triage_observation(
    patient_id: str, symptom: str, severity: str, snomed_code: str
) -> str:
    """
    Create a FHIR Observation resource for a reported symptom.
    severity should be: mild, moderate, or severe
    snomed_code should be the SNOMED CT code for the symptom
    """
    try:
        observation = {
            "resourceType": "Observation",
            "status": "preliminary",     # Preliminary — not yet reviewed by a clinician
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code":    "survey",  # 'survey' marks AI triage records vs lab/vitals
                    "display": "Survey"
                }]
            }],
            "code": {
                "coding": [{
                    "system":  "http://snomed.info/sct",
                    "code":    snomed_code,
                    "display": symptom
                }]
            },
            "subject":           {"reference": f"Patient/{patient_id}"},
            "valueString":       severity,          # Severity stored as valueString
            "effectiveDateTime": "2026-05-28T00:00:00Z"
        }
        result = fhir_post("Observation", observation)
        obs_id = result.get("id", "unknown")
        return (
            f"Observation created successfully. "
            f"ID: {obs_id}, Symptom: {symptom}, Severity: {severity}"
        )
    except Exception as e:
        return f"Error creating observation: {str(e)}"


@tool
def create_service_request(patient_id: str, urgency: str, reason: str) -> str:
    """
    Create a FHIR ServiceRequest for clinical follow-up.
    urgency should be: routine, urgent, or asap
    """
    try:
        service_request = {
            "resourceType": "ServiceRequest",
            "status":       "active",
            "intent":       "order",    # 'order' means this is a firm clinical request
            "priority":     urgency,    # Maps directly to FHIR request-priority codes
            "code": {
                "coding": [{
                    "system":  "http://snomed.info/sct",
                    "code":    "11429006",
                    "display": "Consultation"
                }]
            },
            "subject":    {"reference": f"Patient/{patient_id}"},
            "reasonCode": [{"text": reason}],
            "authoredOn": "2026-05-28T00:00:00Z"
        }
        result = fhir_post("ServiceRequest", service_request)
        req_id = result.get("id", "unknown")
        return (
            f"ServiceRequest created. "
            f"ID: {req_id}, Priority: {urgency}, Reason: {reason}"
        )
    except Exception as e:
        return f"Error creating service request: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
#  SNOMED CT MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

# Maps common symptom phrases to their SNOMED CT codes.
# This ensures Observations written to IRIS are properly coded rather than
# relying on free text, which makes them interoperable with any FHIR-compliant
# system that understands SNOMED — EHRs, analytics engines, quality dashboards.
SYMPTOM_SNOMED_MAP = {
    "chest pain":          "29857009",
    "shortness of breath": "230145002",
    "headache":            "25064002",
    "fever":               "386661006",
    "nausea":              "422587007",
    "vomiting":            "422400008",
    "dizziness":           "404640003",
    "fatigue":             "84229001",
    "cough":               "49727002",
    "abdominal pain":      "21522001",
    "back pain":           "161891005",
    "palpitations":        "80313002",
    "rash":                "271807003",
    "swelling":            "267038008",
    "sore throat":         "162397003"
}


def get_snomed_code(symptom: str) -> str:
    """
    Look up the SNOMED CT code for a symptom string.

    Checks whether any known symptom phrase appears as a substring of the
    input — so "severe chest pain radiating to the jaw" correctly returns
    the chest pain code even though it isn't an exact match.

    Falls back to 418799008 (Finding reported by subject) — a generic
    SNOMED concept that is valid FHIR and indicates the code couldn't be
    determined precisely, which is preferable to an empty or invalid code.
    """
    symptom_lower = symptom.lower()
    for key, code in SYMPTOM_SNOMED_MAP.items():
        if key in symptom_lower:
            return code
    return "418799008"  # SNOMED: Finding reported by subject (safe generic fallback)