"""
backend/services/summary_service.py
=====================================
Service wrapper for the legal document summarization engine.

Delegates to ``backend.ai_client.ai_pipeline.generate_summary`` and
reshapes the output into a chat-friendly dict.
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.ai_client import ai_pipeline
from backend.ai_client.llm_client import get_llm_client
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def generate_summary(text: str) -> Dict[str, Any]:
    """Generate a structured legal summary from contract text.

    Parameters
    ----------
    text : str
        Full contract text (or paragraph-joined text) to summarize.

    Returns
    -------
    dict
        ``{"summary": str, "key_points": list[str], "obligations": list[str],
        "payment_terms": list[str], "termination": list[str], "risks": list[str]}``

    Raises
    ------
    ValueError
        If *text* is empty or whitespace-only.
    """
    if not text or not text.strip():
        raise ValueError("Cannot summarize empty text.")

    logger.info("summary_service: generating summary (%d chars)", len(text))

    raw: Dict[str, Any] = ai_pipeline.generate_summary(text)

    # The ai_pipeline returns {"summary": list[str], "full_summary": str}
    full_summary: str = raw.get("full_summary", "")
    key_points: List[str] = raw.get("summary", [])

    final_summary_text = full_summary or " ".join(key_points)
    
    # Issue 5: LLM Augmented Response
    logger.info("summary_service: generating LLM explanation of the summary")
    prompt = f"Provide a brief, simplified explanation of this contract summary for a non-lawyer:\n{final_summary_text}"
    client = get_llm_client()
    explanation = client.generate(
        prompt=prompt, 
        system_prompt="You are a legal expert translating legal jargon into plain English."
    )

    return {
        "summary": final_summary_text,
        "key_points": key_points,
        "explanation": explanation
    }
