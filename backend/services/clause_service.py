"""
backend/services/clause_service.py
====================================
Clause mapping: splits contract text into chunks, then classifies each chunk.

Uses the trained sklearn pipeline when ``models/sklearn_clause_classifier/model.pkl``
exists; otherwise falls back to the built-in rule engine in ``ai_engine.clause_detector``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# sklearn / legacy taxonomy (trained model labels)
_CLAUSE_DESCRIPTIONS: Dict[str, str] = {
    "termination": "Termination",
    "payment_terms": "Payment Terms",
    "liability": "Liability / Limitation",
    "confidentiality": "Confidentiality / NDA",
    "governing_law": "Governing Law",
    "indemnification": "Indemnification",
    "intellectual_property": "Intellectual Property",
    "dispute_resolution": "Dispute Resolution",
    "force_majeure": "Force Majeure",
    "non_compete": "Non-Compete",
    "warranties": "Warranties",
    "other": "General / Other",
}

# rule-based engine (clause_detector.py) labels
_RULE_LABELS: Dict[str, str] = {
    "definitions": "Definitions",
    "parties": "Parties",
    "term_duration": "Term / Duration",
    "termination": "Termination",
    "payment_fees": "Payment / Fees",
    "intellectual_property": "Intellectual Property",
    "confidentiality": "Confidentiality / NDA",
    "liability_indemnity": "Liability / Indemnity",
    "warranties": "Warranties",
    "dispute_resolution": "Dispute Resolution",
    "governing_law": "Governing Law",
    "miscellaneous": "Miscellaneous",
    "other": "General / Other",
}


def _label_for_type(clause_type: str) -> str:
    return (
        _CLAUSE_DESCRIPTIONS.get(clause_type)
        or _RULE_LABELS.get(clause_type)
        or clause_type.replace("_", " ").title()
    )


def _split_paragraphs(text: str) -> List[str]:
    """
    Split raw contract text into chunks suitable for classification.

    PDF/DOCX extraction often uses single newlines or one long block; we fall
    back from blank-line paragraphs → line breaks → windowed chunks.
    """
    text = text.strip()
    if not text:
        return []

    def _min_len_chunks(parts: List[str]) -> List[str]:
        return [p.strip() for p in parts if len(p.strip()) > 30]

    # 1) Blank-line paragraphs (typical structured contracts)
    chunks = _min_len_chunks(re.split(r"\n{2,}", text))
    if len(chunks) >= 2:
        return chunks

    # 2) Single newlines (common in extracted PDFs)
    line_chunks = _min_len_chunks(re.split(r"\n+", text))
    if len(line_chunks) >= 2:
        return line_chunks

    # 3) One or few large blobs — window into overlapping segments
    blob = chunks[0] if chunks else text
    if len(blob) <= 30:
        return []

    def _window_chunks(s: str, max_len: int = 900) -> List[str]:
        s = s.strip()
        if len(s) <= max_len:
            return [s]
        pieces: List[str] = []
        i = 0
        n = len(s)
        while i < n:
            j = min(i + max_len, n)
            piece = s[i:j].strip()
            if j < n:
                cut = piece.rfind(" ")
                if cut > max_len // 2:
                    piece = s[i : i + cut].strip()
                    i = i + cut + 1
                else:
                    i = j
            else:
                i = j
            if len(piece) > 30:
                pieces.append(piece)
        return pieces

    if len(blob) > 900:
        return _window_chunks(blob, 900)

    return [blob]


def _classify_rule_based(paragraphs: List[str]) -> List[Dict[str, Any]]:
    from ai_engine.clause_detector import detect_clause

    out: List[Dict[str, Any]] = []
    for p in paragraphs:
        det = detect_clause(p)
        ctype = det.get("label", "other")
        conf = float(det.get("confidence", 0.0))
        out.append({"clause_type": ctype, "confidence": round(conf, 4)})
    return out


def map_clauses(text: str) -> Dict[str, Any]:
    """
    Classify contract chunks into clause types and return grouped results.

    Returns
    -------
    dict
        ``total_paragraphs``, ``unique_clause_types``, ``clause_groups``,
        ``all_results``, optional ``classifier`` (``sklearn`` | ``rules``),
        optional ``note`` for UX.
    """
    if not text or not text.strip():
        return {
            "total_paragraphs": 0,
            "clause_groups": {},
            "all_results": [],
            "error": "No text provided for clause mapping.",
        }

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return {
            "total_paragraphs": 0,
            "clause_groups": {},
            "all_results": [],
            "error": "Could not extract meaningful paragraphs from this document.",
        }

    logger.info("clause_service: classifying %d chunks", len(paragraphs))

    classifier = "rules"
    note: str | None = None
    results: List[Dict[str, Any]]

    try:
        from ai_engine.clause_classifier.predict_sklearn import (
            predict_batch_sklearn,
            sklearn_model_available,
        )

        if sklearn_model_available():
            results = predict_batch_sklearn(paragraphs)
            classifier = "sklearn"
            # Model file present but pipeline failed to load → all "other" zeros
            if all(
                r.get("clause_type") == "other" and float(r.get("confidence", 0)) == 0.0
                for r in results
            ):
                logger.warning(
                    "clause_service: sklearn returned no signal — falling back to rules"
                )
                results = _classify_rule_based(paragraphs)
                classifier = "rules"
                note = (
                    "Sklearn classifier did not produce usable scores; "
                    "showing rule-based clause map instead."
                )
        else:
            results = _classify_rule_based(paragraphs)
            note = (
                "Trained sklearn model not found — using built-in rule-based clause detection. "
                "Train with: python ai_engine/clause_classifier/train_sklearn.py"
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("clause_service: inference error — %s", exc)
        results = _classify_rule_based(paragraphs)
        classifier = "rules"
        note = f"Classifier error ({exc!r}); showing rule-based clause map instead."

    clause_groups: Dict[str, List[Dict[str, Any]]] = {}
    all_results: List[Dict[str, Any]] = []

    for paragraph, result in zip(paragraphs, results):
        clause_type = result.get("clause_type", "other")
        confidence = float(result.get("confidence", 0.0))

        record = {
            "clause_type": clause_type,
            "label": _label_for_type(clause_type),
            "confidence": round(confidence, 4),
            "text_snippet": paragraph[:200] + "..." if len(paragraph) > 200 else paragraph,
        }
        all_results.append(record)

        clause_groups.setdefault(clause_type, []).append(record)

    summary: Dict[str, Any] = {}
    for ctype, items in clause_groups.items():
        avg_conf = round(sum(i["confidence"] for i in items) / len(items), 4)
        summary[ctype] = {
            "label": _label_for_type(ctype),
            "count": len(items),
            "avg_confidence": avg_conf,
            "items": items,
        }

    summary = dict(sorted(summary.items(), key=lambda x: x[1]["count"], reverse=True))

    logger.info(
        "clause_service: %s — %d types in %d chunks",
        classifier,
        len(summary),
        len(paragraphs),
    )

    out: Dict[str, Any] = {
        "total_paragraphs": len(paragraphs),
        "unique_clause_types": len(summary),
        "clause_groups": summary,
        "all_results": all_results,
        "classifier": classifier,
    }
    if note:
        out["note"] = note
    return out
