"""
dynamic_agent.py — Dynamic Custom Agent Executor
=================================================
Handles execution of user-created custom agents. Each custom agent is defined
by a JSON configuration stored in custom_agents.json and has its own:
  - System prompt (user-written)
  - Temperature setting
  - Tool subset (selected from the shared FHIR tool library)
  - RAG toggle (whether to search clinical guidelines)
  - Specialty category (for routing hints)

The agent executor is built on demand the first time a session uses a given
custom agent, then cached per session exactly like the built-in agents.
The dynamic agent architecture means zero code changes are needed when a
new custom agent is created — the orchestrator picks it up automatically
on next startup or hot-reload.
"""

import os
import json
from pathlib import Path
from config import LLM_MODEL
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
    create_service_request,
)
from knowledge_base import search_clinical_guidelines

# ── Tool registry — maps string names to tool objects ──────────────────────
# This allows the JSON config to specify tools by name rather than
# importing them directly, keeping the config file simple and portable.
TOOL_REGISTRY = {
    "get_patient":             get_patient,
    "get_patient_conditions":  get_patient_conditions,
    "get_patient_allergies":   get_patient_allergies,
    "get_patient_medications": get_patient_medications,
    "create_triage_observation": create_triage_observation,
    "create_service_request":  create_service_request,
    "search_clinical_guidelines": search_clinical_guidelines,
}

# ── Config file path ────────────────────────────────────────────────────────
AGENTS_FILE = Path(__file__).parent.parent / "data" / "custom_agents.json"


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_agents() -> list[dict]:
    """
    Load all custom agent configurations from custom_agents.json.
    Returns an empty list if the file does not exist yet — this is the normal
    state before any custom agents have been created.
    """
    if not AGENTS_FILE.exists():
        return []
    with open(AGENTS_FILE) as f:
        return json.load(f)


def save_all_agents(agents: list[dict]) -> None:
    """Persist the full agent list to disk, creating the file if needed."""
    AGENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(AGENTS_FILE, "w") as f:
        json.dump(agents, f, indent=2)


def get_agent_config(agent_id: str) -> dict | None:
    """Return the config dict for a single agent by its ID, or None if not found."""
    return next((a for a in load_all_agents() if a["id"] == agent_id), None)


def save_agent(config: dict) -> None:
    """
    Save or update a single agent config.
    If an agent with the same ID already exists it is replaced; otherwise
    the new config is appended. This makes the operation idempotent —
    saving the same agent twice produces only one entry.
    """
    agents = load_all_agents()
    agents = [a for a in agents if a["id"] != config["id"]]
    agents.append(config)
    save_all_agents(agents)


def delete_agent(agent_id: str) -> bool:
    """
    Remove an agent by ID. Returns True if found and deleted, False otherwise.
    Also clears any cached executors for this agent so the change takes effect
    immediately without requiring a container restart.
    """
    agents = load_all_agents()
    before = len(agents)
    agents = [a for a in agents if a["id"] != agent_id]
    if len(agents) == before:
        return False
    save_all_agents(agents)
    # Invalidate cached executors for this agent
    keys_to_remove = [k for k in _executor_cache if k.startswith(f"{agent_id}:")]
    for k in keys_to_remove:
        del _executor_cache[k]
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  EXECUTOR FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

# Session-scoped executor cache — key: "{agent_id}:{session_id}"
# This mirrors the per-session memory pattern used in the built-in agents.
_executor_cache: dict[str, AgentExecutor] = {}


def _build_executor(config: dict) -> AgentExecutor:
    """
    Construct an AgentExecutor from a custom agent config dict.

    The system prompt is wrapped with mandatory clinical context —
    patient ID injection rules and the language rule — so custom agents
    behave consistently with built-in agents even if the user's system
    prompt does not mention these rules.
    """
    # Resolve selected tools from registry
    selected_tools = [
        TOOL_REGISTRY[t]
        for t in config.get("tools", list(TOOL_REGISTRY.keys()))
        if t in TOOL_REGISTRY
    ]

    # Always include RAG if toggled on
    if config.get("rag_enabled", True) and search_clinical_guidelines not in selected_tools:
        selected_tools.append(search_clinical_guidelines)

    # Wrap the user's system prompt with mandatory context rules
    system_prompt = f"""{config['system_prompt']}

CRITICAL RULES — ALWAYS FOLLOW:
- If the message contains [Context: Patient ID is X], IMMEDIATELY fetch all patient data without asking
- NEVER ask for patient ID if it is already provided in context
- Always fetch patient data before analysis
- Always call search_clinical_guidelines before making clinical recommendations
- Cite every guideline source explicitly in your response

LANGUAGE RULE:
- Respond in the same language the user writes in
- Clinical warnings must appear in BOTH English and the user's language
- Always include an English summary at the end for clinical staff handoff
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=float(config.get("temperature", 0.2)),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    agent = create_openai_tools_agent(llm, selected_tools, prompt)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return AgentExecutor(
        agent=agent,
        tools=selected_tools,
        memory=memory,
        verbose=True,
        max_iterations=int(config.get("max_iterations", 10)),
        handle_parsing_errors=True,
    )


def get_executor(agent_id: str, session_id: str) -> AgentExecutor | None:
    """
    Return a cached executor for this agent+session combination.
    Creates a new executor if one does not exist yet.
    Returns None if the agent config is not found.
    """
    cache_key = f"{agent_id}:{session_id}"
    if cache_key not in _executor_cache:
        config = get_agent_config(agent_id)
        if not config:
            return None
        _executor_cache[cache_key] = _build_executor(config)
    return _executor_cache[cache_key]


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC CHAT INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def run_custom_agent(agent_id: str, session_id: str, message: str) -> str:
    """
    Process one turn of a conversation with a custom agent.
    Called by orchestrator.py when a message is routed to a custom agent.

    Returns the agent's response string, or an error message if the agent
    config is not found or the executor fails to run.
    """
    executor = get_executor(agent_id, session_id)
    if not executor:
        return (
            f"Custom agent '{agent_id}' not found. "
            "It may have been deleted. Please refresh the page."
        )
    result = executor.invoke({"input": message})
    return result["output"]