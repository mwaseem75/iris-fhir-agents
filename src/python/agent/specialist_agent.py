"""
specialist_agent.py — Clinical Specialist Consultation Agent
=============================================================
The Specialist Agent provides deep clinical analysis for patients with
complex or chronic conditions. It is invoked by the orchestrator when
a message requires more than initial triage — condition complications,
multi-system disease interactions, referral planning, and monitoring
recommendations all route here.

Where the Triage Agent asks "how urgent is this?", the Specialist Agent
asks "what is the full clinical picture and what needs to happen next?"

Responsibilities:
  1. Retrieve the patient's complete clinical record from IRIS FHIR
     (demographics, active conditions, medications, allergies)
  2. Analyse conditions in depth — not in isolation, but as a system.
     A patient with Type 2 Diabetes + Hypertension + CKD has three
     conditions that each make the others harder to manage.
  3. Search the RAG knowledge base for specialty guidelines relevant to
     each condition (AHA, CDC, KDIGO, ADA, etc.)
  4. Recommend appropriate specialist referrals with specific clinical
     justifications grounded in the retrieved guidelines
  5. Suggest diagnostic tests and monitoring schedules
  6. Write FHIR ServiceRequests to IRIS for every referral recommended

Clinical design decisions:
  - temperature=0.2 allows the agent to produce nuanced, readable clinical
    assessments while staying factually grounded. The Pharmacy Agent uses
    0.1 because drug safety is binary (safe/unsafe); specialist assessments
    involve more clinical judgement and benefit from slightly more expressive prose.
  - The system prompt enforces RAG citation before any recommendation.
    This prevents the LLM from producing plausible-sounding but unsupported
    clinical guidance — every recommendation must trace back to a guideline.
  - max_iterations=10 accommodates the full workflow for complex patients:
    fetch patient → fetch conditions → fetch meds → search RAG per condition
    → create multiple ServiceRequests. Each step is one iteration.
"""

import os
from config import LLM_MODEL, TEMP_SPECIALIST
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

# ── LLM ──────────────────────────────────────────────────────────────────────
# temperature=0.2 — higher than the Pharmacy Agent (0.1) because clinical
# specialist assessments require more nuanced reasoning across interacting
# conditions, where a slightly more expressive response serves the clinician
# better than a terse factual output
llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=TEMP_SPECIALIST,
    api_key=os.getenv("OPENAI_API_KEY")
)


# ═══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

SPECIALIST_PROMPT = """You are a clinical specialist AI assistant with deep medical knowledge.

Your role is to:
1. If message contains [Context: Patient ID is X], IMMEDIATELY call get_patient, get_patient_conditions, get_patient_allergies, and get_patient_medications WITHOUT asking — just fetch automatically
2. Analyze conditions in depth and identify potential complications
3. Check for condition interactions e.g. diabetes + hypertension risk
4. Recommend appropriate specialist referrals with specific reasons
5. Suggest relevant diagnostic tests or monitoring
6. Create FHIR ServiceRequests for referrals
7. Provide a structured clinical assessment

CRITICAL RULES:
- NEVER ask for patient ID if it is already in the message context
- ALWAYS fetch patient data automatically when patient ID is available
- Be proactive — fetch first, then analyze, then respond

LANGUAGE RULE — MANDATORY:
- Detect the language of the CURRENT message and respond in THAT language
- If the patient writes in English, ALWAYS respond in English — even if previous messages were in another language
- If the patient writes in Spanish/French/Mandarin, respond in that language
- Clinical recommendations and guideline citations appear in the user's language
- Always include an English summary section at the end for clinical staff handoff

MANDATORY WORKFLOW — FOLLOW THIS EXACTLY FOR EVERY RESPONSE:
1. Fetch patient data (get_patient, get_patient_conditions, get_patient_medications, get_patient_allergies)
2. For EACH active condition, call search_clinical_guidelines with the condition name
3. READ every guideline returned — do not skip this step
4. Write your response — EVERY recommendation must be preceded by a citation block
5. Create FHIR ServiceRequest for each referral recommended
6. End with a summary

GUIDELINE CITATION FORMAT — YOU MUST INCLUDE THIS IN EVERY RESPONSE:
Before ANY clinical recommendation, you MUST write the citation in this exact format:
> According to [Source] (Relevance: X%) — [key recommendation from the guideline]

YOU MUST include at least 2-3 citation blocks in every specialist response.
If search_clinical_guidelines returns results, you MUST quote them — do not summarise without citing.

Example of a CORRECT response structure:
---
Based on your records, I found the following guidelines:

> According to ADA Diabetes Guidelines 2023 (Relevance: 91%) — HbA1c target for T2DM with CKD is 7.0–8.0%. Metformin is contraindicated below eGFR 30.

> According to KDIGO CKD Guidelines (Relevance: 84%) — eGFR should be monitored every 3 months in CKD Stage 3. ACE inhibitors are first-line for proteinuria.

Based on these guidelines:
- Your HbA1c of 7.8% is above target — endocrinology referral recommended
- Your eGFR requires nephrology follow-up

Referrals created:
- ServiceRequest: Endocrinology (reason: HbA1c above ADA target)
- ServiceRequest: Nephrology (reason: CKD Stage 3 monitoring per KDIGO)
---

A response WITHOUT citation blocks like the above is INCORRECT and must not be produced.
NEVER write recommendations without the > According to [Source] citation lines above them.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

prompt = ChatPromptTemplate.from_messages([
    ("system", SPECIALIST_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

tools = [
    get_patient,
    get_patient_conditions,
    get_patient_allergies,
    get_patient_medications,
    create_triage_observation,
    create_service_request,
    search_clinical_guidelines
]


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

specialist_sessions: dict[str, AgentExecutor] = {}


def create_specialist_executor() -> AgentExecutor:
    """
    Build a fresh AgentExecutor with dedicated conversation memory.

    max_iterations=10 handles the full workflow for a complex patient
    like pt-010 (HFrEF + T2DM + CKD + HTN + 7 medications) where the
    agent needs to fetch patient data, search RAG for each of 4 conditions,
    and create multiple ServiceRequests — each action counts as one iteration.
    Setting this too low would cut off the agent mid-assessment on complex cases.
    """
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent  = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True
    )


def get_specialist_session(session_id: str) -> AgentExecutor:
    """Return existing session or create a new one for this session ID."""
    if session_id not in specialist_sessions:
        specialist_sessions[session_id] = create_specialist_executor()
    return specialist_sessions[session_id]


def run_specialist(session_id: str, message: str) -> str:
    """
    Process one turn of a specialist consultation.
    Called by orchestrator.py when a message is routed to SPECIALIST.
    """
    executor = get_specialist_session(session_id)
    result   = executor.invoke({"input": message})
    return result["output"]