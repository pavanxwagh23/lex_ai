"""
backend/services/chat_service.py
================================
General conversational service — Zero-LLM Mode.

Uses the deterministic RuleBasedResponder from llm_client.py.
Maintains session history for context-aware follow-up responses.
"""

from __future__ import annotations

import re
from typing import Dict, Any

from backend.services.session_manager import get_session, add_message
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Conversational Knowledge Base (instant, offline, no LLM required)
# ---------------------------------------------------------------------------

_KNOWLEDGE_BASE: list[tuple[re.Pattern, str]] = [
    # Greetings
    (re.compile(r"\b(hi|hello|hey|greetings)\b", re.I),
     "Hello! I'm **Lex AI** — your offline legal document assistant. "
     "Upload a contract or paste text in the sidebar, then ask me to summarize it, find risks, or list all clauses!"),

    # Identity
    (re.compile(r"\bwho\s+are\s+you\b|\bwhat\s+are\s+you\b|your\s+name", re.I),
     "I'm **Lex AI**, a fully offline AI legal assistant powered by trained machine learning models. "
     "I analyze contracts, identify clause types, detect legal risks, and compare documents — "
     "all without sending your data anywhere."),

    # Capabilities
    (re.compile(r"\bwhat\s+can\s+you\s+(do|help)\b|your\s+features|capabilities", re.I),
     "Here's what I can do:\n\n"
     "- 📄 **Summarize** a contract → *'Summarize this'*\n"
     "- ⚠️ **Risk Analysis** → *'Find risks'* or *'Are there any red flags?'*\n"
     "- 🗂️ **Clause Mapping** → *'List all clauses'* or *'What types of sections are in this?'*\n"
     "- 🔍 **Compare** two documents → *'Compare these contracts'*\n\n"
     "Upload a PDF or DOCX in the sidebar to begin!"),

    # Warranty
    (re.compile(r"\bwarrant(y|ies)?\b", re.I),
     "A **warranty** is a legally binding promise that certain facts are true. Types include:\n\n"
     "- **Express**: Explicitly written in the contract.\n"
     "- **Implied**: Automatically assumed by law.\n"
     "- **Fitness for Purpose**: Product/service will meet a specific need.\n\n"
     "*Upload your contract and I'll identify warranty clauses for you automatically.*"),

    # NDA / Confidentiality
    (re.compile(r"\b(nda|non.?disclosure|confidentiality)\b", re.I),
     "An **NDA (Non-Disclosure Agreement)** legally prevents sharing of confidential information. "
     "Key elements to check:\n\n"
     "- What counts as 'confidential information'?\n"
     "- How long does the obligation last?\n"
     "- What are the penalties for breach?\n\n"
     "*Upload an NDA to detect its confidentiality clauses instantly.*"),

    # Liability
    (re.compile(r"\b(liabilit(y|ies)|liable|indemnif)\b", re.I),
     "**Liability clauses** determine who pays when things go wrong:\n\n"
     "- **Limitation of Liability**: Caps total damages (e.g., 'not to exceed 3 months of fees').\n"
     "- **Mutual vs One-sided**: Both parties capped, or only one?\n"
     "- **Exclusions**: Consequential, indirect, and special damages often excluded.\n\n"
     "⚠️ *No liability cap is a HIGH RISK flag. Upload your contract to check.*"),

    # Termination
    (re.compile(r"\btermination?\b|cancel+ation?\b|end\s+the\s+(deal|agreement|contract)\b", re.I),
     "**Termination clauses** define when and how a contract can be ended:\n\n"
     "- **For Cause**: After a material breach.\n"
     "- **For Convenience**: Either party can exit without reason.\n"
     "- **Notice Period**: Usually 30–90 days written notice required.\n\n"
     "*I can identify termination clauses automatically when you upload a contract.*"),

    # Force Majeure
    (re.compile(r"\bforce\s+majeure\b|\bact\s+of\s+god\b", re.I),
     "**Force Majeure** excuses performance obligations due to extraordinary events "
     "(natural disasters, pandemics, wars). Watch out for:\n\n"
     "- Overly broad definitions that let a party exit too easily.\n"
     "- No obligation to resume performance after the event ends.\n"
     "- Missing notice requirements for claiming force majeure."),

    # Governing law
    (re.compile(r"\b(governing\s+law|jurisdiction|arbitration|dispute)\b", re.I),
     "**Dispute Resolution** clauses determine where and how conflicts are settled:\n\n"
     "- **Arbitration**: Private process, typically faster and cheaper than courts.\n"
     "- **Litigation**: Through public courts.\n"
     "- **Governing Law**: Which state/country's laws apply.\n\n"
     "⚠️ *Foreign jurisdiction clauses can dramatically increase enforcement costs.*"),

    # Non-compete
    (re.compile(r"\b(non.?compete|non.?solicitation)\b", re.I),
     "**Non-Compete clauses** restrict working with competitors after the contract ends:\n\n"
     "- **Duration**: 6 mo–1 year is typical; 3+ years is often unenforceable.\n"
     "- **Geography**: Local vs. global scope.\n"
     "- **Scope**: Full industry vs. direct competitors only.\n\n"
     "⚠️ *California, North Dakota, and some other US states ban non-competes entirely.*"),

    # Intellectual Property
    (re.compile(r"\b(ip|intellectual\s+property|copyright|patent|ownership)\b", re.I),
     "**IP clauses** govern who owns creative work produced:\n\n"
     "- **Work-for-Hire**: Client owns everything created.\n"
     "- **License**: Creator grants usage rights but retains ownership.\n"
     "- **Assignment**: Permanent ownership transfer.\n\n"
     "*Upload your contract and I'll identify IP clauses automatically.*"),

    # Payment
    (re.compile(r"\b(payment|invoice|fees?|salary|compensation)\b", re.I),
     "**Payment Terms** define the financial obligations:\n\n"
     "- **Net 30/60/90**: Days to pay after invoice date.\n"
     "- **Milestone Payments**: Tied to deliverables.\n"
     "- **Late Fees**: Interest on overdue amounts.\n"
     "- **Auto-escalation**: Annual % increases — always check for these!"),

    # Thank you
    (re.compile(r"\b(thank(s|\s*you)?|thx|ty)\b", re.I),
     "You're welcome! Let me know if there's anything else I can help analyze. "
     "Remember: for binding legal decisions, always consult a qualified attorney."),

    # How are you
    (re.compile(r"\bhow\s+are\s+you\b", re.I),
     "I'm running at full capacity — all ML models loaded and ready! "
     "Upload a contract and let's get analyzing. 📄"),
]


def _match_knowledge_base(message: str) -> str | None:
    """Match a message against the knowledge base. Returns None if no match."""
    for pattern, response in _KNOWLEDGE_BASE:
        if pattern.search(message):
            return response
    return None


def generate_chat_response(message: str, session_id: str = None) -> Dict[str, Any]:
    """Generate an instant offline conversational response.

    Parameters
    ----------
    message : str
        User's natural language message.
    session_id : str, optional
        Session ID for context tracking.

    Returns
    -------
    dict
        ``{"response": str, "data": None}``
    """
    if not message or not message.strip():
        raise ValueError("Cannot generate a response for an empty message.")

    logger.info("chat_service: processing message via KnowledgeBaseResponder")

    if session_id:
        add_message(session_id, "user", message)

    response = _match_knowledge_base(message)

    if response is None:
        response = (
            "I'm operating in **document-analysis mode**. I don't have a pre-built answer for that, "
            "but I can analyze any contract you upload!\n\n"
            "Try:\n"
            "- *'Summarize this contract'*\n"
            "- *'Find all risks'*\n"
            "- *'List all clause types'*\n\n"
            "Upload a PDF or DOCX in the sidebar to begin. For general legal advice, "
            "please consult a qualified attorney."
        )

    if session_id:
        add_message(session_id, "assistant", response)

    return {"response": response, "data": None}
