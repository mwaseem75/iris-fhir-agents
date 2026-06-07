"""
from config import LLM_MODEL, TEMP_TRIAGE
triage_agent.py — Clinical Triage Agent
========================================
The Triage Agent is the entry point for every patient interaction on the
IRIS FHIR Agents platform. It handles the intake conversation — establishing
who the patient is, what brought them in today, and how urgently they need care.

Unlike the Specialist and Pharmacy agents, which are invoked mid-conversation
after the patient context is already established, the Triage Agent is designed
to work with a patient who has just arrived. It speaks to the patient directly
in an empathetic, conversational tone rather than producing clinician-facing
structured reports.

Workflow per session:
  1. Greet the patient and request their Patient ID
  2. As soon as the ID is available (from the patient or from an orchestrator
     context prefix), immediately fetch demographics, conditions, allergies,
     and medications from IRIS FHIR — no confirmation step, no delay
  3. Ask about current symptoms in a natural, conversational way
  4. Search the RAG knowledge base for guidelines relevant to each symptom
  5. Classify urgency: EMERGENCY, URGENT, or ROUTINE
  6. Write a FHIR Observation for each symptom with the appropriate SNOMED CT code
  7. Write a FHIR ServiceRequest reflecting the urgency classification
  8. Deliver a clear clinician handoff summary with guideline citations

Urgency thresholds (follows standard triage nursing criteria):
  EMERGENCY — chest pain, breathing difficulty, stroke symptoms, anaphylaxis.
              These warrant an immediate 911 call or resuscitation response.
  URGENT    — high fever, moderate and worsening pain, acute deterioration.
              Needs same-day clinical attention.
  ROUTINE   — mild symptoms, stable chronic condition queries.
              Can be managed at a scheduled appointment.

Temperature choice:
  0.3 — the highest of the three clinical agents. Triage involves direct
  patient-facing conversation where warmth and natural phrasing matter.
  The Pharmacy Agent uses 0.1 (binary safety decisions) and the Specialist
  uses 0.2 (structured clinical reasoning). Triage benefits from the most
  conversational register because a cold or robotic intake experience
  discourages patients from disclosing important symptoms.
"""

import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from fhir_tools import (
    get_patient,
    get_patient_conditions,
    get_patient_allergies,
    get_patient_medications,
    create_triage_observation,
    create_service_request
)
from knowledge_base import search_clinical_guidelines
from config import LLM_MODEL, TEMP_TRIAGE

# ── LLM ──────────────────────────────────────────────────────────────────────
# temperature=0.3 — the most expressive of the three clinical agents.
# Triage is patient-facing; the conversation needs to feel natural and
# empathetic, not like a structured data-extraction form.
llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=TEMP_TRIAGE,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ── Tool set ─────────────────────────────────────────────────────────────────
# Four read tools cover the full intake picture.
# Two write tools let the agent document findings directly into IRIS FHIR
# rather than just producing a report that has to be manually entered later.
tools = [
    get_patient,              # Demographics — name, DOB, gender
    get_patient_conditions,   # Active diagnoses — context for symptom interpretation
    get_patient_allergies,    # Known allergens — critical before any treatment suggestion
    get_patient_medications,  # Current prescriptions — interactions with new symptoms
    create_triage_observation,# Write symptom record to FHIR with SNOMED code
    create_service_request,   # Write follow-up order to FHIR with urgency priority
    search_clinical_guidelines# RAG — ground every assessment in authoritative guidelines
]


# ═══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an AI-powered clinical triage assistant integrated with a FHIR R4 health record system.

Your role is to:
1. Greet the patient and ask for their Patient ID to look up their records
2. As soon as you have a Patient ID (either provided by patient or in [Context]), IMMEDIATELY call get_patient, get_patient_conditions, get_patient_allergies, and get_patient_medications tools WITHOUT asking permission
3. Ask about their current symptoms in a conversational, empathetic way
4. Assess urgency based on symptoms + medical history
5. For each reported symptom, create a FHIR Observation with the appropriate SNOMED code
6. Determine urgency level: EMERGENCY, URGENT, or ROUTINE
7. Create a FHIR ServiceRequest based on urgency
8. Provide a clear clinician handoff summary

CRITICAL RULES:
- If message contains [Context: Patient ID is X], use X as patient ID and IMMEDIATELY fetch all patient data
- NEVER ask for patient ID if it is already provided in context
- NEVER ask permission before fetching patient data — just do it automatically
- Always fetch patient data before asking about symptoms

Urgency guidelines:
- EMERGENCY: chest pain, difficulty breathing, severe allergic reaction, stroke symptoms
- URGENT: high fever, moderate pain, symptoms worsening rapidly
- ROUTINE: mild symptoms, chronic condition management

MANDATORY WORKFLOW for symptoms:
1. Call search_clinical_guidelines with the symptom
2. Read the guidelines returned
3. Apply guidelines to assess urgency
4. Create FHIR Observation
5. Create FHIR ServiceRequest
6. Cite the guideline source in response

GUIDELINE CITATION FORMAT — MANDATORY:
When you use search_clinical_guidelines results, you MUST explicitly cite them in your response using this exact format:
> According to [Source] (Relevance: X%) — [key recommendation from guideline]

Example of correct citation:
> According to AHA/ACC Guidelines 2021 (Relevance: 92%) — chest pain with shortness of breath is a medical emergency requiring immediate 911 call.

NEVER summarize guidelines without showing the source name and relevance score.
ALWAYS include at least one guideline citation when responding to any symptom.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")  # LangChain tool call scratchpad
])


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

# Active triage sessions keyed by session_id (UUID from main.py).
# Each session has its own ConversationBufferMemory so the agent remembers
# everything discussed — the patient's ID, their reported symptoms, the
# urgency classification — across the full multi-turn intake conversation.
sessions: dict[str, AgentExecutor] = {}


def create_agent_executor() -> AgentExecutor:
    """
    Build a fresh AgentExecutor with dedicated conversation memory.

    Each new triage session gets a clean slate — no prior patient's data
    bleeds into the new session. This is critical for patient privacy and
    for preventing the agent from confusing two patients' symptoms.

    max_iterations=10 accommodates the full triage workflow:
    fetch patient (1) → fetch conditions (2) → fetch allergies (3) →
    fetch medications (4) → search RAG (5) → create Observation (6) →
    create ServiceRequest (7), leaving headroom for follow-up symptom
    iterations within the same session.
    """
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,           # Log tool calls to container stdout for debugging
        max_iterations=10,
        handle_parsing_errors=True
    )


def get_or_create_session(session_id: str) -> AgentExecutor:
    """
    Return the existing AgentExecutor for this session, or create a new one.

    The session store is intentionally in-memory — triage conversations are
    short-lived and the overhead of persisting them to a database would add
    latency without meaningful benefit for a clinical intake workflow.
    """
    if session_id not in sessions:
        sessions[session_id] = create_agent_executor()
    return sessions[session_id]


def chat(session_id: str, message: str) -> str:
    """
    Process one message in a triage session and return the agent's response.

    Called by orchestrator.py for turn 1 of every new conversation,
    and for any subsequent turn that the router classifies as TRIAGE.
    Also called directly by the vitals alert system in main.py when a
    patient's vital signs cross a critical threshold — in that case the
    session_id is a generated alert ID and the message already contains
    the full critical vitals context, so the agent skips the greeting
    and goes straight to clinical assessment.
    """
    executor = get_or_create_session(session_id)
    result   = executor.invoke({"input": message})
    return result["output"]