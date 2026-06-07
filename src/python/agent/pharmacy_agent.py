"""
pharmacy_agent.py — Clinical Pharmacy Safety Agent
===================================================
The Pharmacy Agent is responsible for all medication-related safety checks
in the IRIS FHIR Agents platform. It is the most safety-critical agent in
the system — its decisions directly affect what drugs a patient should or
should not receive.

Responsibilities:
  1. Fetch the patient's current medications and known allergies from IRIS FHIR
  2. Check every pair of current medications for known drug-drug interactions
  3. Cross-check any proposed new medication against the patient's allergy profile
  4. Search the RAG knowledge base for FDA/clinical guidelines on the drugs involved
  5. Produce a risk-rated safety report (HIGH RISK / MODERATE / LOW)
  6. Write a FHIR MedicationRequest to IRIS when a new medication is proposed

Clinical design decisions:
  - The interaction database is intentionally conservative — it includes the most
    clinically significant interactions that a pharmacist would flag immediately
    (e.g. Warfarin + Aspirin, Digoxin + Amiodarone). False negatives in a safety
    system are more dangerous than false positives.
  - Allergy cross-reactivity groups handle the critical clinical reality that a
    patient allergic to penicillin may react to all beta-lactams. A naive string
    match on "penicillin" would miss amoxicillin, ampicillin, and piperacillin.
  - The agent is instructed to NEVER flag an interaction without citing the
    guideline source and relevance score — this keeps the RAG grounded and
    prevents the agent from inventing warnings not supported by evidence.
"""

import os
from config import LLM_MODEL, TEMP_PHARMACY
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool

from fhir_tools import (
    get_patient,
    get_patient_conditions,
    get_patient_allergies,
    get_patient_medications,
    create_service_request,
    fhir_post
)
from knowledge_base import search_clinical_guidelines

# ── LLM ──────────────────────────────────────────────────────────────────────
# temperature=0.1 — low enough for factual drug safety output, with just enough
# flexibility to produce readable natural-language risk summaries rather than
# raw structured data the user would need to interpret themselves
llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=TEMP_PHARMACY,
    api_key=os.getenv("OPENAI_API_KEY")
)


# ═══════════════════════════════════════════════════════════════════════════════
#  DRUG INTERACTION CHECKER
# ═══════════════════════════════════════════════════════════════════════════════

# Known clinically significant drug-drug interactions.
# Pairs are stored as (drug1, drug2) tuples with substring matching so
# "warfarin 5mg" still matches the "warfarin" key.
# This list covers the high-frequency, high-severity interactions most
# relevant to the demo patient population (cardiac, diabetic, renal patients).
KNOWN_INTERACTIONS = {
    ("warfarin",    "aspirin"):     "HIGH RISK: Increased bleeding risk — dual antiplatelet/anticoagulant effect",
    ("warfarin",    "ibuprofen"):   "HIGH RISK: Increased bleeding risk — NSAID inhibits platelet function",
    ("metformin",   "contrast"):    "HIGH RISK: Risk of lactic acidosis — hold metformin 48h before/after contrast",
    ("metformin",   "alcohol"):     "MODERATE: Risk of lactic acidosis — alcohol potentiates metformin toxicity",
    ("lisinopril",  "potassium"):   "MODERATE: Risk of hyperkalemia — ACE inhibitor + potassium supplement",
    ("simvastatin", "amiodarone"):  "HIGH RISK: Risk of myopathy/rhabdomyolysis — CYP3A4 inhibition raises statin levels",
    ("ssri",        "maoi"):        "CONTRAINDICATED: Risk of serotonin syndrome — potentially fatal combination",
    ("digoxin",     "amiodarone"):  "HIGH RISK: Increased digoxin toxicity — amiodarone raises digoxin serum levels",
}


@tool
def check_drug_interactions(medications: str) -> str:
    """
    Check for potential drug interactions between medications.
    Pass a comma-separated list of medication names.
    Returns known interaction warnings.
    """
    meds = [m.strip().lower() for m in medications.split(",")]
    warnings = []

    # Check every pair — O(n²) but medication lists are short enough
    # that this is faster than any lookup structure
    for i, med1 in enumerate(meds):
        for med2 in meds[i + 1:]:
            for (drug1, drug2), warning in KNOWN_INTERACTIONS.items():
                # Substring match handles brand names, dosage suffixes, etc.
                if (drug1 in med1 or drug1 in med2) and (drug2 in med1 or drug2 in med2):
                    warnings.append(f"{med1} + {med2}: {warning}")

    return "\n".join(warnings) if warnings else "No known critical interactions found."


# ═══════════════════════════════════════════════════════════════════════════════
#  ALLERGY-MEDICATION CONFLICT CHECKER
# ═══════════════════════════════════════════════════════════════════════════════

# Cross-reactivity groups map a broad allergen class to specific drugs that
# share the same mechanism of sensitisation. This is where a lot of clinical
# harm happens — an allergist documents "penicillin allergy" but the prescriber
# orders amoxicillin without realising it's a beta-lactam.
ALLERGY_CROSS_REACTIVITY = {
    "penicillin": ["amoxicillin", "ampicillin", "penicillin", "piperacillin"],
    "sulfa":      ["sulfamethoxazole", "trimethoprim", "bactrim"],
    "nsaid":      ["ibuprofen", "naproxen", "aspirin", "diclofenac"],
    "codeine":    ["codeine", "morphine", "oxycodone", "hydrocodone"],
}


@tool
def check_allergy_medication_conflict(patient_id: str, medication: str) -> str:
    """
    Check if a proposed medication conflicts with patient's known allergies.
    """
    from fhir_tools import fhir_get
    try:
        data     = fhir_get(f"AllergyIntolerance?patient={patient_id}")
        entries  = data.get("entry", [])
        allergens = []
        for e in entries:
            resource = e.get("resource", {})
            coding   = resource.get("code", {}).get("coding", [{}])[0]
            allergens.append(coding.get("display", "").lower())

        med_lower = medication.lower()
        conflicts = []

        for allergen in allergens:
            for group, related_meds in ALLERGY_CROSS_REACTIVITY.items():
                # Match the allergen against the group name in both directions
                # so "penicillin allergy" matches group "penicillin" and vice versa
                if allergen in group or group in allergen:
                    if any(m in med_lower for m in related_meds):
                        conflicts.append(
                            f"ALLERGY CONFLICT: Patient allergic to {allergen}, "
                            f"{medication} is contraindicated (cross-reactivity: {group} group)"
                        )

        return "\n".join(conflicts) if conflicts else f"No allergy conflicts found for {medication}"
    except Exception as e:
        return f"Error checking allergy conflicts: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
#  MEDICATION REQUEST WRITER
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def create_medication_request(patient_id: str, medication: str, dosage: str, reason: str) -> str:
    """
    Create a FHIR MedicationRequest for a patient.
    """
    try:
        med_request = {
            "resourceType": "MedicationRequest",
            "status":       "active",
            "intent":       "proposal",  # 'proposal' rather than 'order' — AI suggestion, not a signed prescription
            "medicationCodeableConcept": {
                "text": medication       # Free text; a production system would use RxNorm codes
            },
            "subject":            {"reference": f"Patient/{patient_id}"},
            "dosageInstruction":  [{"text": dosage}],
            "reasonCode":         [{"text": reason}],
            "authoredOn":         "2026-06-01T00:00:00Z"
        }
        result = fhir_post("MedicationRequest", med_request)
        req_id = result.get("id", "unknown")
        return (
            f"MedicationRequest created. "
            f"ID: {req_id}, Medication: {medication}, Dosage: {dosage}"
        )
    except Exception as e:
        return f"Error creating medication request: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

PHARMACY_PROMPT = """You are a clinical pharmacy AI assistant with expertise in medications and drug safety.

Your role is to:
1. If message contains [Context: Patient ID is X], IMMEDIATELY call get_patient, get_patient_medications, and get_patient_allergies WITHOUT asking — fetch automatically
2. Check for drug-drug interactions between all current medications
3. Check for allergy-medication conflicts
4. Review medications in context of patient conditions
5. Flag concerning medication combinations with risk levels
6. Create FHIR MedicationRequests for proposed medications
7. Provide a clear medication safety summary

CRITICAL RULES:
- NEVER ask for patient ID if it is already in the message context
- ALWAYS fetch patient data automatically when patient ID is available
- Check interactions BEFORE suggesting any new medication
- Always verify against patient allergies first

MANDATORY WORKFLOW for medication checks:
1. Call get_patient_medications to fetch current medications
2. Call get_patient_allergies to fetch known allergies
3. Call search_clinical_guidelines with the drug name or interaction query
4. Read the guidelines returned
5. Apply FDA and clinical guidelines to assess drug safety
6. Flag HIGH RISK interactions explicitly
7. Create FHIR MedicationRequest for any proposed medications
8. Cite the guideline source explicitly in response

GUIDELINE CITATION FORMAT — MANDATORY:
When you use search_clinical_guidelines results, you MUST explicitly cite them using this format:
> According to [Source] (Relevance: X%) — [key recommendation]

Examples:
> According to FDA Drug Interaction Guidelines (Relevance: 95%) — NSAIDs significantly increase bleeding risk with warfarin. This combination is HIGH RISK.
> According to AAAAI Allergy Guidelines (Relevance: 90%) — Penicillin allergy cross-reactivity: 1-10% may react to cephalosporins.

NEVER flag a drug interaction without citing the FDA or clinical guideline source and relevance score.
ALWAYS call search_clinical_guidelines before checking any medication safety.
ALWAYS show risk level: HIGH RISK, MODERATE, or LOW for every interaction found.
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

prompt = ChatPromptTemplate.from_messages([
    ("system", PHARMACY_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Full tool set for the pharmacy agent — includes both FHIR read tools
# (from fhir_tools.py) and the pharmacy-specific safety check tools above
tools = [
    get_patient,
    get_patient_conditions,
    get_patient_allergies,
    get_patient_medications,
    check_drug_interactions,
    check_allergy_medication_conflict,
    create_medication_request,
    create_service_request,
    search_clinical_guidelines
]


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

# Per-session executors preserve conversation context — important for follow-up
# questions like "what about adding metoprolol?" after a full med review
pharmacy_sessions: dict[str, AgentExecutor] = {}


def create_pharmacy_executor() -> AgentExecutor:
    """
    Build a fresh AgentExecutor with its own conversation memory.

    max_iterations=10 is higher than the other agents because pharmacy
    safety checks are genuinely multi-step: fetch meds → fetch allergies →
    check interactions → search RAG → check proposed drug → create FHIR resource.
    Each step is one iteration, so 10 gives enough headroom for the full
    workflow without allowing runaway loops.
    """
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent  = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=10,          # Higher than other agents — pharmacy workflow has more steps
        handle_parsing_errors=True
    )


def get_pharmacy_session(session_id: str) -> AgentExecutor:
    """Return existing session or create a new one for this session ID."""
    if session_id not in pharmacy_sessions:
        pharmacy_sessions[session_id] = create_pharmacy_executor()
    return pharmacy_sessions[session_id]


def run_pharmacy(session_id: str, message: str) -> str:
    """
    Process one turn of a pharmacy consultation.
    Called by orchestrator.py when a message is routed to PHARMACY.
    """
    executor = get_pharmacy_session(session_id)
    result   = executor.invoke({"input": message})
    return result["output"]