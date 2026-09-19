"""
backend/services/intent_detector.py
====================================
Rule-based intent detection for the AI assistant chat layer.

Detects user intent from a natural language message and returns a
canonical intent label used by the router to dispatch to the correct
service.

Supported Intents
-----------------
- ``CLAUSE_MAP`` — list / map / classify clause types in a contract
- ``SUMMARY``  — user wants a contract summarized
- ``RISK``     — user wants a risk analysis
- ``COMPARE``  — user wants to compare two contracts
- ``DRAFT``    — user wants to draft / generate contract text (placeholder)
- ``GENERAL``  — fallback for general legal questions or small talk
"""

from __future__ import annotations

import re
from typing import List, Tuple

from backend.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Intent keyword mapping
# ---------------------------------------------------------------------------
# Each intent is associated with a list of keyword patterns (case-insensitive).
# The patterns are compiled once at import time for performance.

_INTENT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # ── CLAUSE_MAP (before SUMMARY: SUMMARY regex used to match bare "break down") ─
    ("CLAUSE_MAP", re.compile(
        r"(list|map|identify|show|find|detect)\s+(all\s+)?(the\s+)?"
        r"(clause|section|provision|article|part)(s|(\s+type))?"
        r"|what\s+(type|kind)s?\s+of\s+(clause|section|provision)"
        r"|classify\s+(the\s+)?(contract|document|clauses?)"
        r"|break\s*down\s+(the\s+)?(clause|section|contract|clauses?)"
        r"|clause\s+map|map\s+(the\s+)?clauses?",
        re.IGNORECASE,
    )),

    # ── SUMMARY ─────────────────────────────────────────────────────────
    ("SUMMARY", re.compile(
        r"summari[sz]e|summary|overview|brief|what\s+does\s+this\s+(contract|agreement)\s+(say|cover)"
        r"|give\s+me\s+a\s+summary|key\s+points|main\s+points|extract\s+summary|tldr|tl;dr"
        r"|explain\s+this\s+(contract|document|agreement)\b"
        r"|break\s+down\s+(this\s+)?(contract|agreement|document|for\s+me)\b",
        re.IGNORECASE,
    )),

    # ── RISK ────────────────────────────────────────────────────────────
    ("RISK", re.compile(
        r"\brisk[s]?\b|danger|red\s*flag|flag|vulnerabilit|concern|problematic"
        r"|liability|indemnit|penalty|risky|warning|issue[s]?"
        r"|what\s+(are|could\s+be)\s+the\s+(risks?|issues?|problems?)"
        r"|anything\s+dangerous|safe\s+to\s+sign|should\s+i\s+(sign|worry)",
        re.IGNORECASE,
    )),

    # ── COMPARE ─────────────────────────────────────────────────────────
    ("COMPARE", re.compile(
        r"\bcompar[ei]|differ|diff\b|contrast|versus|vs\.?\b|changes?\s+between"
        r"|what('s|\s+is)\s+(different|changed|new)|side\s+by\s+side|match",
        re.IGNORECASE,
    )),

    # ── DRAFT ───────────────────────────────────────────────────────────
    ("DRAFT", re.compile(
        r"\bdraft\b|generate\s+(a\s+)?(contract|clause|agreement|nda|terms)"
        r"|write\s+(a\s+)?(contract|clause|section)|create\s+(a\s+)?(contract|agreement)"
        r"|template|boilerplate",
        re.IGNORECASE,
    )),

]



def detect_intent_with_llm(message: str) -> str:
    """Fallback to LLM for intent classification when regex fails.
    
    Returns
    -------
    str
        One of SUMMARY, RISK, COMPARE, DRAFT, GENERAL.
    """
    logger.info("Using LLM for intent classification.")
    prompt = f"Classify this legal request into EXACTLY ONE of the following tags: SUMMARY, RISK, COMPARE, DRAFT, GENERAL.\n\nRequest: {message}\n\nTag:"
    
    client = get_llm_client()
    system_prompt = "You are a precise classifier. Output ONLY the exact tag string."
    response = client.generate(prompt=prompt, system_prompt=system_prompt)
    
    # Clean up the response in case the LLM returned extra text
    clean_resp = response.strip().upper()
    valid_intents = {"SUMMARY", "RISK", "COMPARE", "DRAFT", "GENERAL"}
    
    for vi in valid_intents:
        if vi in clean_resp:
            return vi
            
    return "GENERAL"


def detect_intent(message: str) -> Tuple[str, float]:
    """Detect user intent from a natural language message.

    The function iterates over compiled regex patterns. If a pattern matches,
    it returns the intent with high confidence (1.0). If no pattern matches, 
    confidence is 0.0 and it falls back to LLM for classification.

    Parameters
    ----------
    message : str
        Raw user message text.

    Returns
    -------
    tuple[str, float]
        (intent_string, confidence_score)
    """
    if not message or not message.strip():
        logger.warning("Empty message received — defaulting to GENERAL.")
        return "GENERAL", 1.0

    cleaned = message.strip()
    
    intent = "GENERAL"
    confidence = 0.0

    # 1. Rule-based detection (regex)
    for intent_label, pattern in _INTENT_PATTERNS:
        if pattern.search(cleaned):
            logger.info("Regex Intent detected: %s", intent_label)
            intent = intent_label
            confidence = 1.0
            break

    # 2. If no regex match -> instantly default to GENERAL
    if confidence < 0.6:
        intent = "GENERAL"
        confidence = 0.5  # Assumed default intent

    logger.info("Final Intent: %s (confidence: %.2f) for message: %r", intent, confidence, cleaned[:80])
    return intent, confidence
