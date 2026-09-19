"""
backend/services/chat_service.py
================================
General conversational service - Zero-LLM Mode.

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
# Landmark cases / doctrine (offline snippets - matches Judgments tab + chat)
# Checked before the general knowledge base so “explain Bachan Singh…” works.
# ---------------------------------------------------------------------------

_LANDMARK_CASES: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"Bachan\s+Singh|1980\s+SCC\s*\(\s*2\s*\)\s*684|rarest\s+of\s+rare",
            re.I,
        ),
        (
            "**Bachan Singh v. State of Punjab** (1980 SCC (2) 684) - what practitioners remember:\n\n"
            "- Upheld that the **death penalty is constitutional** in India, but imposed a **high constitutional threshold** for its use.\n"
            "- Created the **“rarest of rare”** framing: capital punishment should be reserved for cases where the alternative (life) is "
            "**unquestionably foreclosed** and the crime shocks the collective conscience.\n"
            "- Judges must weigh **aggravating and mitigating** circumstances individually; sentencing is not mechanical.\n"
            "- Later benches treat it as the **lodestar for capital sentencing**; appellate courts scrutinize whether the test was applied.\n"
            "- **Criminal practitioners** use it in sentencing arguments, mercy petitions, and review of death references.\n"
            "- **Civil/commercial lawyers** rarely cite it directly, but it is core teaching in **constitutional criminal law**.\n\n"
            "_Educational summary only - always read the full judgment and current precedents before filing._"
        ),
    ),
    (
        re.compile(r"Arnesh\s+Kumar|\(2014\)\s*8\s*SCC\s*273", re.I),
        (
            "**Arnesh Kumar v. State of Bihar** ((2014) 8 SCC 273) - key takeaways:\n\n"
            "- Addresses **arrest in matrimonial / IPC 498A**-type complaints where immediate custody was routine.\n"
            "- Police must **apply the law strictly**: arrest is not automatic; **reasonable satisfaction** and **necessity** matter.\n"
            "- Emphasizes **bail** and **de-escalation** of criminal process where complaints may be misused.\n"
            "- Often read with later guidelines on **mediation**, **dowry harassment** procedure, and **judicial oversight** of arrest.\n"
            "- **Practitioners** cite it in bail applications, quashing strategy, and instructions to investigating officers.\n\n"
            "_Educational summary only - verify against the latest SC and High Court orders on your facts._"
        ),
    ),
    (
        re.compile(r"Puttaswamy|privacy.*fundamental|10\s*SCC\s*1.*2017", re.I),
        (
            "**K.S. Puttaswamy v. Union of India** ((2017) 10 SCC 1) - key takeaways:\n\n"
            "- **Nine-judge bench**: **privacy** is a **fundamental right** under Part III, linked to dignity and autonomy.\n"
            "- Enables **proportionality review** of state action (surveillance, identity programs, data collection).\n"
            "- Informs **Aadhaar** reasoning, **data protection** debates, and limits on **intrusive investigation**.\n"
            "- **Practitioners** use it in writs, constitutional challenges, and arguments on **fair procedure** alongside Article 21.\n\n"
            "_Educational summary only - pair with the D.P.D.P. Act and sector rules for compliance work._"
        ),
    ),
    (
        re.compile(r"Vishaka|POSH|sexual\s+harassment.*workplace", re.I),
        (
            "**Vishaka v. State of Rajasthan** ((1997) 6 SCC 241) - key takeaways:\n\n"
            "- Filled a **legislative gap** by issuing **binding guidelines** on workplace sexual harassment until a statute arrived.\n"
            "- Shaped **employer duties**, **complaints committees**, and **preventive** obligations.\n"
            "- **Superseded in form** by the **POSH Act, 2013**, but remains the **historical and doctrinal anchor**.\n"
            "- **HR and employment counsel** still cite it for policy framing and training narratives.\n\n"
            "_Educational summary only - use the POSH Act and rules for compliance today._"
        ),
    ),
    (
        re.compile(r"Carlill|Carbolic\s+Smoke\s+Ball", re.I),
        (
            "**Carlill v. Carbolic Smoke Ball Co.** ([1893] 1 QB 256) - key takeaways:\n\n"
            "- Classic **unilateral offer**: performance of conditions (using the smoke ball as directed) **is acceptance**.\n"
            "- **Consideration** found in the **discomfort/inconvenience** of the user and the company’s marketing bargain.\n"
            "- Taught everywhere for **intention to be bound** and **advertisement-as-offer** problems.\n"
            "- **Indian contract courses** map it to **offer, acceptance, and consideration** under the Contract Act, 1872.\n\n"
            "_Common-law teaching case - check Indian precedents for analogous marketing/offers._"
        ),
    ),
    (
        re.compile(r"Salomon|Salomon\s+v\.|veil\s+of\s+incorporation", re.I),
        (
            "**Salomon v. Salomon & Co. Ltd.** ([1897] AC 22) - key takeaways:\n\n"
            "- **Separate legal personality**: a valid company is distinct from its shareholders.\n"
            "- **Limited liability** flows from that separation.\n"
            "- **Lifting the corporate veil** remains **exceptional**-fraud, sham, agency, or statute-specific pierce tests.\n"
            "- **Corporate lawyers** use it as the baseline for group structures, lending, and shareholder risk.\n\n"
            "_Educational summary only - Indian company law has its own veil-piercing and related-party rules._"
        ),
    ),
    (
        re.compile(r"Donoghue|Stevenson|neighbour\s+principle", re.I),
        (
            "**Donoghue v. Stevenson** ([1932] AC 562) - key takeaways:\n\n"
            "- **Duty of care** in negligence: manufacturers owe consumers a duty where harm is **foreseeable**.\n"
            "- Famous **“neighbour”** formulation for who is owed a duty.\n"
            "- Foundation of **modern tort** teaching in common-law systems.\n"
            "- **Indian law** uses different statutory and common-law strands for product liability-use this case comparatively, not as binding IP.\n\n"
            "_Educational summary only._"
        ),
    ),
    (
        re.compile(r"\bArticle\s*14\b|equality\s+before\s+the\s+law|equal\s+protection", re.I),
        (
            "**Article 14** (Constitution of India) - in practice:\n\n"
            "- Guarantees **equality before the law** and **equal protection of the laws**.\n"
            "- Courts read in **non-arbitrariness** and **reasonable classification** tests.\n"
            "- **Not a single “case card”**: hundreds of judgments refine state action, taxation, licensing, and personal liberty.\n"
            "- **Practitioners** pair text + leading SC lines for writs and discrimination arguments.\n\n"
            "_Educational summary only._"
        ),
    ),
    (
        re.compile(r"\bArticle\s*21\b|life\s+and\s+personal\s+liberty", re.I),
        (
            "**Article 21** (Constitution of India) - in practice:\n\n"
            "- **Life and personal liberty** cannot be deprived except according to **procedure established by law**.\n"
            "- Interpreted to require **fair, just, and reasonable** procedure (not any statute on paper).\n"
            "- Anchor for **privacy (Puttaswamy)**, health, livelihood, and environmental rights in various decisions.\n\n"
            "_Educational summary only._"
        ),
    ),
    (
        re.compile(r"53A|part\s+performance|Transfer\s+of\s+Property", re.I),
        (
            "**TPA Section 53A** (part performance) - key takeaways:\n\n"
            "- **Limited equity** where a **written contract** exists and the transferee has **possessed** or performed **acts in furtherance**.\n"
            "- Bars the transferor from **denying the agreement** in defined circumstances-**not a substitute for registration** where mandatory.\n"
            "- **Property disputes** turn on possession, pleadings, and specific performance / declaratory strategy.\n\n"
            "_Educational summary only - verify against current precedents on your transaction._"
        ),
    ),
]


def _match_landmark_case(message: str) -> str | None:
    for pattern, response in _LANDMARK_CASES:
        if pattern.search(message):
            return response
    return None


# ---------------------------------------------------------------------------
# Conversational Knowledge Base (instant, offline, no LLM required)
# ---------------------------------------------------------------------------

_KNOWLEDGE_BASE: list[tuple[re.Pattern, str]] = [
    # Greetings
    (re.compile(r"\b(hi|hello|hey|greetings)\b", re.I),
     "Hello! I'm **Lex AI** - your offline legal document assistant. "
     "Upload a contract or paste text in the sidebar, then ask me to summarize it, find risks, or list all clauses!"),

    # Identity
    (re.compile(r"\bwho\s+are\s+you\b|\bwhat\s+are\s+you\b|your\s+name", re.I),
     "I'm **Lex AI**, a fully offline AI legal assistant powered by trained machine learning models. "
     "I analyze contracts, identify clause types, detect legal risks, and compare documents - "
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
     "- **Notice Period**: Usually 30-90 days written notice required.\n\n"
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
     "- **Duration**: 6 mo-1 year is typical; 3+ years is often unenforceable.\n"
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
     "- **Auto-escalation**: Annual % increases - always check for these!"),

    # Thank you
    (re.compile(r"\b(thank(s|\s*you)?|thx|ty)\b", re.I),
     "You're welcome! Let me know if there's anything else I can help analyze. "
     "Remember: for binding legal decisions, always consult a qualified attorney."),

    # How are you
    (re.compile(r"\bhow\s+are\s+you\b", re.I),
     "I'm running at full capacity - all ML models loaded and ready! "
     "Upload a contract and let's get analyzing. 📄"),
]


def _match_knowledge_base(message: str) -> str | None:
    """Match a message against the knowledge base. Returns None if no match."""
    for pattern, response in _KNOWLEDGE_BASE:
        if pattern.search(message):
            return response
    return None


def match_offline_snippet(message: str) -> str | None:
    """
    Return a curated offline reply (landmark case briefing or knowledge-base hit).

    Used by ``/chat`` and by ``RuleBasedResponder`` in ``llm_client`` so all
    zero-LLM paths share the same landmark answers.
    """
    r = _match_landmark_case(message)
    if r is not None:
        return r
    return _match_knowledge_base(message)


OFFLINE_CHAT_FALLBACK = (
    "I'm strongest when you **upload a contract** (or paste text) and ask me to summarize, "
    "map clauses, or scan risks.\n\n"
    "For **named judgments** in the **Judgments** tab, use **Ask in Assistant** - I have short "
    "briefings for those landmarks. For other questions I don't have a stored briefing yet.\n\n"
    "Try:\n"
    "- *'Summarize this contract'*\n"
    "- *'Find all risks'*\n"
    "- *'List all clause types'*\n\n"
    "For binding advice or novel issues, consult a **qualified attorney**."
)


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

    response = match_offline_snippet(message)
    if response is None:
        response = OFFLINE_CHAT_FALLBACK

    if session_id:
        add_message(session_id, "assistant", response)

    return {"response": response, "data": None}
