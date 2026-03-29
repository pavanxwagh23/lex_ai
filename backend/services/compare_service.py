"""
backend/services/compare_service.py
=====================================
Service wrapper for the semantic contract comparison engine.

Delegates to ``backend.ai_client.ai_pipeline.compare_contracts`` and
reshapes the output into a chat-friendly dict.
"""

from __future__ import annotations

from typing import Any, Dict

from backend.ai_client import ai_pipeline
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def compare_contracts(text_a: str, text_b: str) -> Dict[str, Any]:
    """Semantically compare two contract texts.

    Parameters
    ----------
    text_a : str
        Full text of the original / baseline contract.
    text_b : str
        Full text of the revised / second contract.

    Returns
    -------
    dict
        ``{"added_clauses": list, "removed_clauses": list,
        "modified_clauses": list, "similar_clauses": list,
        "metadata": dict, "change_summary": str}``

    Raises
    ------
    ValueError
        If either text is empty.
    """
    if not text_a or not text_a.strip():
        raise ValueError("Original contract text (text_a) cannot be empty.")
    if not text_b or not text_b.strip():
        raise ValueError("Revised contract text (text_b) cannot be empty.")

    logger.info(
        "compare_service: comparing contracts (A=%d chars, B=%d chars)",
        len(text_a),
        len(text_b),
    )

    raw: Dict[str, Any] = ai_pipeline.compare_contracts(text_a, text_b)

    # Build a human-readable change summary
    added    = raw.get("added_clauses", [])
    removed  = raw.get("removed_clauses", [])
    modified = raw.get("modified_clauses", [])
    similar  = raw.get("similar_clauses", [])

    parts = []
    if added:
        parts.append(f"{len(added)} clause(s) added")
    if removed:
        parts.append(f"{len(removed)} clause(s) removed")
    if modified:
        parts.append(f"{len(modified)} clause(s) modified")
    if similar:
        parts.append(f"{len(similar)} clause(s) unchanged or similar")
    change_summary = ". ".join(parts) + "." if parts else "No differences detected."

    return {
        **raw,
        "change_summary": change_summary,
    }
