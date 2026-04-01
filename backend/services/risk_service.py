"""
backend/services/risk_service.py
=================================
Service wrapper for the legal risk detection engine.

Delegates to ``backend.ai_client.ai_pipeline.detect_risks`` and
reshapes the output into a chat-friendly dict.
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.ai_client import ai_pipeline
from backend.ai_client.llm_client import get_llm_client
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def analyze_risk(paragraphs: List[str]) -> Dict[str, Any]:
    """Run risk analysis on a list of contract paragraphs.

    Parameters
    ----------
    paragraphs : list[str]
        Pre-split contract paragraphs.

    Returns
    -------
    dict
        ``{"risk_score": float, "risk_level": str, "findings": list[dict],
        "total_paragraphs": int, "risky_count": int}``

    Raises
    ------
    ValueError
        If *paragraphs* is empty.
    """
    if not paragraphs:
        raise ValueError("Cannot analyze risk on an empty paragraph list.")

    logger.info("risk_service: analyzing %d paragraphs", len(paragraphs))

    # Join paragraphs into a single text block because ai_pipeline.detect_risks
    # accepts raw text and internally re-splits via paragraph_splitter.
    text = "\n\n".join(paragraphs)
    raw: Dict[str, Any] = ai_pipeline.detect_risks(text)

    risk_score: float = raw.get("risk_score", 0.0)
    risks: List[Dict[str, Any]] = raw.get("risks", [])

    # Determine human-readable risk level
    if risk_score < 3:
        risk_level = "Low Risk"
    elif risk_score < 6:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    risky_count = sum(1 for r in risks if r.get("severity") in ("high", "HIGH"))

    # Zero-LLM: deterministic explanation from risk category templates
    _RISK_EXPLANATIONS = {
        "liability": "⚠️ A **Liability** clause was detected. Uncapped liability exposes a party to unlimited financial losses. Look for a clear liability cap (e.g., 'not to exceed total fees paid').",
        "termination": "⚠️ A **Termination** clause was detected. Ensure notice periods are fair and check whether either party can exit 'for convenience' without cause.",
        "confidentiality": "⚠️ A **Confidentiality** clause was detected. Verify the definition of confidential information is not overly broad and check the duration of the obligation.",
        "indemnification": "⚠️ An **Indemnification** clause was detected. One-sided indemnification without a liability cap creates significant financial exposure.",
        "dispute_resolution": "⚠️ A **Dispute Resolution** clause was detected. Mandatory foreign arbitration or jurisdiction clauses can dramatically increase enforcement costs.",
        "non_compete": "⚠️ A **Non-Compete** clause was detected. Review the duration, geographic scope, and industry scope. Overly broad non-competes may be unenforceable.",
        "intellectual_property": "⚠️ An **Intellectual Property** clause was detected. Confirm who owns IP created during and after the contract, including pre-existing IP.",
        "force_majeure": "⚠️ A **Force Majeure** clause was detected. Check if it's too broadly defined, allowing a party to exit obligations too easily.",
        "payment_terms": "⚠️ A **Payment Terms** clause was detected. Watch for hidden auto-escalation, late payment penalties, or unfavorable invoice windows.",
        "governing_law": "⚠️ A **Governing Law** clause was detected. Foreign jurisdiction increases legal costs and complexity significantly.",
        "warranties": "⚠️ A **Warranties** clause was detected. Ensure warranties are mutual and check for any blanket disclaimer of all implied warranties.",
    }

    if risks:
        parts = []
        for r in risks:
            cat = r.get("category", r.get("type", "")).lower()
            # Find the best matching template
            matched_template = next(
                (v for k, v in _RISK_EXPLANATIONS.items() if k in cat), None
            )
            if matched_template:
                parts.append(matched_template)
            else:
                parts.append(f"⚠️ A legal risk was detected in category **{cat}**. Review carefully.")
        explanation = "\n\n".join(dict.fromkeys(parts))  # Deduplicate
    else:
        explanation = "✅ No significant legal risks were identified in the provided text."

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "findings": risks,
        "total_paragraphs": len(paragraphs),
        "risky_count": risky_count,
        "explanation": explanation
    }


def analyze_risk_from_text(text: str) -> Dict[str, Any]:
    """Convenience wrapper that accepts raw text instead of paragraphs.

    Splits the text on double newlines, then delegates to :func:`analyze_risk`.

    Parameters
    ----------
    text : str
        Raw contract text.

    Returns
    -------
    dict
        Same structure as :func:`analyze_risk`.
    """
    if not text or not text.strip():
        raise ValueError("Cannot analyze risk on empty text.")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return analyze_risk(paragraphs)
