"""
backend/services/clause_service.py
====================================
Clause mapping service using the custom-trained Scikit-Learn classifier.

Splits contract text into paragraphs and runs batch inference through
the trained TF-IDF + Logistic Regression pipeline to classify every
clause type with a confidence score.

Zero LLM dependency — entirely powered by your trained model.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Human-readable labels for the clause taxonomy
# ---------------------------------------------------------------------------

_CLAUSE_DESCRIPTIONS = {
    "termination":          "Termination",
    "payment_terms":        "Payment Terms",
    "liability":            "Liability / Limitation",
    "confidentiality":      "Confidentiality / NDA",
    "governing_law":        "Governing Law",
    "indemnification":      "Indemnification",
    "intellectual_property":"Intellectual Property",
    "dispute_resolution":   "Dispute Resolution",
    "force_majeure":        "Force Majeure",
    "non_compete":          "Non-Compete",
    "warranties":           "Warranties",
    "other":                "General / Other",
}


def _split_paragraphs(text: str) -> List[str]:
    """Split raw contract text into clean paragraphs."""
    paragraphs = re.split(r"\n{2,}", text)
    cleaned = []
    for p in paragraphs:
        p = p.strip()
        if len(p) > 30:   # Skip very short fragments
            cleaned.append(p)
    return cleaned


def map_clauses(text: str) -> Dict[str, Any]:
    """
    Run the trained Scikit-Learn clause classifier on contract text.

    Parameters
    ----------
    text : str
        Raw contract text.

    Returns
    -------
    dict
        Structured clause mapping with groups, counts, and confidence scores.
    """
    if not text or not text.strip():
        return {
            "total_paragraphs": 0,
            "clause_groups": {},
            "all_results": [],
            "error": "No text provided for clause mapping."
        }

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return {
            "total_paragraphs": 0,
            "clause_groups": {},
            "all_results": [],
            "error": "Could not extract meaningful paragraphs from this document."
        }

    logger.info("clause_service: classifying %d paragraphs", len(paragraphs))

    # --- Run the trained sklearn model ---
    try:
        from ai_engine.clause_classifier.predict_sklearn import predict_batch_sklearn
        results = predict_batch_sklearn(paragraphs)
        logger.info("clause_service: sklearn inference complete for %d paragraphs", len(results))
    except FileNotFoundError:
        logger.warning("clause_service: Sklearn model not found — running train_sklearn.py first is required.")
        return {
            "total_paragraphs": len(paragraphs),
            "clause_groups": {},
            "all_results": [],
            "error": (
                "The clause classification model has not been trained yet. "
                "Run: `python ai_engine/clause_classifier/train_sklearn.py` to train it first."
            )
        }
    except Exception as exc:
        logger.error("clause_service: inference error — %s", exc)
        return {
            "total_paragraphs": len(paragraphs),
            "clause_groups": {},
            "all_results": [],
            "error": f"Inference error: {str(exc)}"
        }

    # --- Group results by clause type ---
    clause_groups: Dict[str, List[Dict]] = {}
    all_results = []

    for paragraph, result in zip(paragraphs, results):
        clause_type = result.get("clause_type", "other")
        confidence = result.get("confidence", 0.0)

        record = {
            "clause_type": clause_type,
            "label": _CLAUSE_DESCRIPTIONS.get(clause_type, clause_type.replace("_", " ").title()),
            "confidence": round(confidence, 4),
            "text_snippet": paragraph[:200] + "..." if len(paragraph) > 200 else paragraph,
        }
        all_results.append(record)

        if clause_type not in clause_groups:
            clause_groups[clause_type] = []
        clause_groups[clause_type].append(record)

    # --- Build summary stats per clause type ---
    summary = {}
    for ctype, items in clause_groups.items():
        avg_conf = round(sum(i["confidence"] for i in items) / len(items), 4)
        summary[ctype] = {
            "label": _CLAUSE_DESCRIPTIONS.get(ctype, ctype.replace("_", " ").title()),
            "count": len(items),
            "avg_confidence": avg_conf,
            "items": items,
        }

    # Sort by count descending
    summary = dict(sorted(summary.items(), key=lambda x: x[1]["count"], reverse=True))

    logger.info(
        "clause_service: found %d unique clause types in %d paragraphs",
        len(summary), len(paragraphs)
    )

    return {
        "total_paragraphs": len(paragraphs),
        "unique_clause_types": len(summary),
        "clause_groups": summary,
        "all_results": all_results,
    }
