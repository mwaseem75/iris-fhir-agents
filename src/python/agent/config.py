"""
config.py — Centralised Application Configuration
==================================================
Single source of truth for all environment-specific settings.

Every value here is read from an environment variable with a sensible
default for local Docker development. To deploy to a different environment
(staging, production, cloud), only the environment variables need to change —
no source files need to be edited.

Environment variables are set in docker-compose.yml under the api service.
For production, use Docker secrets or a secrets manager instead of
plain-text environment variables.

Usage in any module:
    from config import FHIR_BASE, FHIR_AUTH, FHIR_HEADERS, IRIS_BASE, LLM_MODEL
"""

import os

# ── InterSystems IRIS FHIR R4 ─────────────────────────────────────────────────
# Internal Docker network address — used by all server-side FHIR calls.
# The browser-facing address (localhost:32783) is only used in the HTML
# frontend; all Python code uses this internal address.
FHIR_BASE = os.getenv("FHIR_BASE_URL", "http://fhir-template:52773/fhir/r4")

# IRIS Atelier REST API base — without /fhir/r4 suffix.
# Used by knowledge_base.py and fhir_agent.py for direct SQL queries
# via the /api/atelier/v1/FHIRSERVER/action/query endpoint.
IRIS_BASE = os.getenv("IRIS_BASE_URL", "http://fhir-template:52773")

# IRIS credentials — same account used for both FHIR REST and Atelier SQL
FHIR_USERNAME = os.getenv("FHIR_USERNAME", "_SYSTEM")
FHIR_PASSWORD = os.getenv("FHIR_PASSWORD", "SYS")
FHIR_AUTH     = (FHIR_USERNAME, FHIR_PASSWORD)

# Standard headers for all FHIR REST requests.
# Content-Type is only needed on POST/PUT — included here for convenience
# so callers can pass FHIR_HEADERS to both GET and POST without thinking about it.
FHIR_HEADERS = {
    "Accept":       "application/fhir+json",
    "Content-Type": "application/fhir+json"
}

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model used by all clinical agents and the orchestrator router.
# Override via environment variable to switch to gpt-4o for production
# without touching any agent code.
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Per-agent temperatures — kept here so the reasoning behind each value
# is documented in one place rather than scattered across agent files:
#   TRIAGE     0.3 — patient-facing conversation needs natural, empathetic tone
#   SPECIALIST 0.2 — clinical reasoning benefits from some expressive prose
#   PHARMACY   0.1 — drug safety decisions are binary; minimise creativity
#   ROUTER     0.0 — deterministic single-word classification; no randomness
TEMP_TRIAGE     = float(os.getenv("TEMP_TRIAGE",     "0.3"))
TEMP_SPECIALIST = float(os.getenv("TEMP_SPECIALIST", "0.2"))
TEMP_PHARMACY   = float(os.getenv("TEMP_PHARMACY",   "0.1"))
TEMP_ROUTER     = float(os.getenv("TEMP_ROUTER",     "0.0"))

# ── RAG Knowledge Base ────────────────────────────────────────────────────────
# Path to the clinical guidelines CSV inside the container.
# Mounted from host data/guidelines/ via docker-compose volume so guidelines
# can be updated without rebuilding the image.
RAG_GUIDELINES_CSV = os.getenv(
    "RAG_GUIDELINES_CSV",
    "/home/irisowner/irisdev/data/RAG/clinical_rag_guidelines.csv"
)

# OpenAI embedding model — must match the VECTOR(DOUBLE, 1536) column
# dimension in RAG.VectorKnowledgeBase. Changing this requires dropping
# and recreating the table with the new vector dimension.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ── Application ───────────────────────────────────────────────────────────────
APP_TITLE   = "IRIS FHIR Agents"
APP_VERSION = "2.0.0"
APP_DESC    = (
    "Multi-agent AI clinical platform built on InterSystems IRIS for Health. "
    "Triage, Specialist, Pharmacy, and FHIR Server agents powered by GPT-4o-mini, "
    "grounded by IRIS Vector Search RAG."
)