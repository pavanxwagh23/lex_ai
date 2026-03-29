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

    # Issue 5: LLM Augmented Response
    logger.info("risk_service: generating LLM explanation of the risks")
    if risks:
        risk_descriptions = [r.get("description", "Unknown risk") for r in risks]
        prompt = f"Explain these legal risks in simple terms:\n{risk_descriptions}"
        client = get_llm_client()
        explanation = client.generate(
            prompt=prompt, 
            system_prompt="You are a legal advisor explaining risks simply. Do not use complex jargon."
        )
    else:
        explanation = "No significant legal risks were identified in the provided text."

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
