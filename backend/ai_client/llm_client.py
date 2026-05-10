"""
backend/ai_client/llm_client.py
================================
LLM Client — Zero-LLM Mode.

All external LLM dependencies (OpenAI, Ollama, Llama 3) have been
replaced with a deterministic RuleBasedResponder that generates
instant responses with zero GPU/network usage.

Setting USE_LOCAL_LLM in .env is no longer required.
"""

from __future__ import annotations

import os
import re
from typing import Optional

import requests

from backend.config import LLM_PROVIDER, LLM_TIMEOUT_SECONDS, LOCAL_LLM_MODEL, LOCAL_LLM_URL
from backend.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Rule-Based Responder
# ---------------------------------------------------------------------------

class RuleBasedResponder:
    """
    A deterministic, zero-dependency responder that matches user prompts
    to pre-written legal knowledge responses via keyword rules.

    This completely replaces the LLM client with instant, offline responses.
    """

    _RULES: list[tuple[re.Pattern, str]] = [
        # --- Greetings ---
        (re.compile(r"\b(hi|hello|hey|greetings|good\s*(morning|evening|afternoon))\b", re.I),
         "Hello! I'm Lex AI, your intelligent legal document assistant. "
         "Upload a contract or paste text in the sidebar to get started. "
         "I can summarize it, identify risks, classify clauses, and compare documents."),

        # --- Identity ---
        (re.compile(r"\b(who|what)\s+are\s+you\b|\byour\s+name\b|tell me about yourself", re.I),
         "I'm Lex AI — a fully offline, privacy-first legal document intelligence system. "
         "I use trained machine learning models to analyze your contracts with zero cloud dependency."),

        # --- Warranty ---
        (re.compile(r"\bwarrant(y|ies|s)?\b", re.I),
         "**Warranty** is a legally binding promise made by one party to another "
         "that certain facts or conditions are true. In contracts, warranties can be:\n\n"
         "- **Express Warranty**: Explicitly stated in writing.\n"
         "- **Implied Warranty**: Automatically granted by law even if not written.\n"
         "- **Warranty of Merchantability**: The product works as expected.\n\n"
         "⚠️ *Always consult a qualified attorney for specific legal advice.*"),

        # --- NDA ---
        (re.compile(r"\b(nda|non.?disclosure|confidentiality\s+agreement)\b", re.I),
         "**NDA (Non-Disclosure Agreement)** is a legal contract that prevents parties "
         "from sharing confidential information with third parties. Key components:\n\n"
         "- Definition of what constitutes 'confidential information'.\n"
         "- Duration of the confidentiality obligation.\n"
         "- Permitted disclosures (e.g., to attorneys, regulators).\n"
         "- Remedies for breach (injunctions, damages).\n\n"
         "⚠️ *Always consult a qualified attorney for specific legal advice.*"),

        # --- Liability ---
        (re.compile(r"\b(liabilit(y|ies)|liable)\b", re.I),
         "**Liability** refers to a party's legal responsibility for their actions or omissions. "
         "In contracts:\n\n"
         "- **Limitation of Liability**: Caps the maximum damages one party can owe.\n"
         "- **Indemnification**: One party agrees to cover losses of the other.\n"
         "- **Consequential Damages**: Indirect losses flowing from a breach.\n\n"
         "A broad liability clause with no cap is a significant **HIGH RISK** flag."),

        # --- Termination ---
        (re.compile(r"\b(terminat(e|ion|ing)|end\s+the\s+(contract|agreement)|cancell?ation)\b", re.I),
         "**Termination clauses** define the conditions under which a contract can be ended. Types:\n\n"
         "- **Termination for Cause**: One party breaches obligations.\n"
         "- **Termination for Convenience**: Either party can exit without reason.\n"
         "- **Notice Period**: Required advance warning before termination (e.g., 30 days).\n\n"
         "Always check what happens to deliverables, IP, and payments after termination."),

        # --- Indemnification ---
        (re.compile(r"\bindemni(f(y|ication)|t(y|ies))?\b", re.I),
         "**Indemnification** is an obligation by one party to compensate the other for "
         "harm, loss, or liability arising from specific events. It typically covers:\n\n"
         "- Third-party claims and lawsuits.\n"
         "- IP infringement claims.\n"
         "- Negligence by either party.\n\n"
         "Broad, one-sided indemnification clauses with no cap are HIGH RISK."),

        # --- Force Majeure ---
        (re.compile(r"\bforce\s+majeure\b|\bact\s+of\s+god\b", re.I),
         "**Force Majeure** ('superior force') excuses a party from performing contractual "
         "obligations when extraordinary events beyond their control occur:\n\n"
         "- Natural disasters, earthquakes, floods.\n"
         "- Wars, pandemics, government actions.\n"
         "- Strikes or labor disputes.\n\n"
         "Check whether the clause is too broad — some parties use it to avoid any difficult obligation."),

        # --- Governing Law ---
        (re.compile(r"\b(governing\s+law|jurisdiction|arbitration|dispute\s+resolution)\b", re.I),
         "**Governing Law / Dispute Resolution** clauses specify:\n\n"
         "- Which country or state's laws apply to the contract.\n"
         "- Where disputes must be filed (court jurisdiction).\n"
         "- Whether disputes go to arbitration (private) or litigation (court).\n\n"
         "⚠️ A clause requiring disputes to be filed in a foreign jurisdiction is a significant risk."),

        # --- Non-Compete ---
        (re.compile(r"\b(non.?compete|non.?solicitation|restraint\s+of\s+trade)\b", re.I),
         "**Non-Compete clauses** restrict one party from working with competitors "
         "after the contract ends. Enforceability depends on:\n\n"
         "- **Geographic scope**: City, country, or worldwide?\n"
         "- **Duration**: 6 months is standard; 3+ years is often unenforceable.\n"
         "- **Scope**: Does it cover the entire industry or just direct competitors?\n\n"
         "Many jurisdictions (like California) ban non-competes entirely for employees."),

        # --- What can you do ---
        (re.compile(r"\bwhat\s+can\s+you\s+do\b|\byour\s+capabilities\b|\bhelp\s*me\b", re.I),
         "I can perform the following document analysis tasks:\n\n"
         "- 📄 **Summarize**: *'Summarize this contract'*\n"
         "- ⚠️ **Risk Analysis**: *'Find risks in this document'*\n"
         "- 🗂️ **Clause Mapping**: *'List all clauses in this contract'*\n"
         "- 🔍 **Compare**: *'Compare these two documents'*\n\n"
         "Upload a PDF or DOCX using the sidebar to get started!"),

        # --- Intellectual Property ---
        (re.compile(r"\b(intellectual\s+property|ip\s+rights|copyright|patent|trademark)\b", re.I),
         "**Intellectual Property (IP)** clauses govern ownership of creative work:\n\n"
         "- **Work-for-Hire**: IP created during the contract belongs to the client.\n"
         "- **License**: One party grants permission to use IP without transferring ownership.\n"
         "- **Assignment**: Full ownership of IP is transferred permanently.\n\n"
         "⚠️ Always clarify who owns IP created *during* and *after* the contract period."),

        # --- Payment ---
        (re.compile(r"\b(payment|invoice|compensation|fee|rate|salary)\b", re.I),
         "**Payment Terms** define when and how money changes hands:\n\n"
         "- **Net 30/60/90**: Payment due within 30, 60, or 90 days of invoice.\n"
         "- **Milestone-based**: Payment tied to project deliverable completion.\n"
         "- **Late Payment**: Many contracts include interest on overdue payments.\n\n"
         "Check for automatic price escalation clauses that can increase rates without explicit consent."),
    ]

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Match prompt against rules and return the best matching response."""
        for pattern, response in self._RULES:
            if pattern.search(prompt):
                logger.info("RuleBasedResponder: matched pattern '%s'", pattern.pattern[:40])
                return response

        # Default fallback
        logger.info("RuleBasedResponder: no rule matched, returning fallback.")
        return (
            "I'm currently operating in **offline, document-analysis mode**. "
            "I can analyze documents you upload — try:\n\n"
            "- *'Summarize this contract'*\n"
            "- *'Find legal risks'*\n"
            "- *'List all clauses'*\n\n"
            "For general legal advice, please consult a qualified attorney."
        )


# ---------------------------------------------------------------------------
# Singleton accessor (drop-in compatible with existing code)
# ---------------------------------------------------------------------------

_responder: Optional[RuleBasedResponder] = None


class LocalHTTPResponder:
    """OpenAI-compatible local model server responder."""

    def __init__(
        self,
        base_url: str = LOCAL_LLM_URL,
        model: str = LOCAL_LLM_MODEL,
        timeout: float = LLM_TIMEOUT_SECONDS,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 256,
                "temperature": 0.3,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


class LLMClient:
    """Drop-in compatible wrapper around the configured responder."""

    def __init__(self, api_key=None, model=None):
        if LLM_PROVIDER == "local_http":
            self._responder = LocalHTTPResponder()
            logger.info("LLMClient initialised with local_http provider at %s.", LOCAL_LLM_URL)
        else:
            self._responder = RuleBasedResponder()
            logger.info("LLMClient initialised with rule provider.")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            return self._responder.generate(prompt, system_prompt)
        except Exception as exc:  # noqa: BLE001
            if isinstance(self._responder, RuleBasedResponder):
                raise
            logger.warning("Configured LLM provider failed (%s); falling back to rule responder.", exc)
            return RuleBasedResponder().generate(prompt, system_prompt)


def get_llm_client() -> LLMClient:
    """Return a singleton LLMClient instance."""
    global _responder
    if _responder is None:
        _responder = LLMClient()
    return _responder
