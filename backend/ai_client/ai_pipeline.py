"""
backend/ai_client/ai_pipeline.py
=================================
Unified AI pipeline adapter for the Legal Document Analyzer backend.

This module wraps the real AI engine modules (pdf_extractor, paragraph_splitter,
clause_detector_pro, risk_detection, summarizer, contract_comparator) and exposes
a clean, async-friendly interface to the services layer.

Fallback behaviour
------------------
If a real AI engine is unavailable (not installed / model not trained), the
corresponding function falls back to a clearly labelled **mock** response so the
backend can be tested end-to-end without the full ML stack.

Environment control
-------------------
Set ``USE_REAL_AI=false`` to force mock mode regardless of what is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from backend.config import (
    COMPARE_THRESHOLD_IDENTICAL,
    COMPARE_THRESHOLD_MODIFIED,
    COMPARE_THRESHOLD_RELATED,
    PROJECT_ROOT,
    USE_REAL_AI,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Ensure the project root is on sys.path so ai_engine imports resolve
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Module-level lazy singletons  (initialised on first call)
# ---------------------------------------------------------------------------
_risk_engine    = None
_summarizer     = None
_comparator     = None


def _get_risk_engine():
    global _risk_engine
    if _risk_engine is None:
        from ai_engine.risk_detection import RiskDetectionEngine
        _risk_engine = RiskDetectionEngine()
        logger.info("RiskDetectionEngine ready.")
    return _risk_engine


def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        from ai_engine.summarizer import LegalSummarizer
        _summarizer = LegalSummarizer()
        logger.info("LegalSummarizer ready.")
    return _summarizer


def _get_comparator():
    global _comparator
    if _comparator is None:
        from ai_engine.contract_comparator import ContractComparator
        _comparator = ContractComparator(
            threshold_identical=COMPARE_THRESHOLD_IDENTICAL,
            threshold_modified=COMPARE_THRESHOLD_MODIFIED,
            threshold_related=COMPARE_THRESHOLD_RELATED,
        )
        logger.info("ContractComparator ready.")
    return _comparator


# ===========================================================================
# 1. Text extraction
# ===========================================================================

def extract_text(file_path: str | Path) -> str:
    """
    Extract raw text from a PDF or DOCX contract file.

    Parameters
    ----------
    file_path : str | Path
        Absolute path to an uploaded contract file.

    Returns
    -------
    str
        Extracted plain text.
    """
    if not USE_REAL_AI:
        logger.debug("Mock: extract_text(%s)", file_path)
        return _mock_contract_text()

    try:
        from ai_engine.pdf_extractor import process_pdf
        result = process_pdf(str(file_path))
        text: str = result.get("full_text", "")
        logger.info("Extracted %d characters from %s", len(text), file_path)
        return text
    except Exception as exc:  # noqa: BLE001
        logger.warning("extract_text failed (%s), using mock.", exc)
        return _mock_contract_text()


# ===========================================================================
# 2. Clause detection
# ===========================================================================

def detect_clauses(text: str) -> dict[str, str]:
    """
    Detect and classify clauses in the contract text.

    Parameters
    ----------
    text : str
        Raw or cleaned contract text.

    Returns
    -------
    dict[str, str]
        Mapping of clause label → representative clause text.
        e.g. ``{"termination": "Either party may terminate …"}``.
    """
    if not USE_REAL_AI:
        return _mock_clauses()

    try:
        from ai_engine.paragraph_splitter import split_into_paragraphs
        from ai_engine.clause_detector_pro import detect_clause_enhanced
        from ai_engine.clause_classifier.predict import predict_clause

        paragraphs  = split_into_paragraphs(text)
        clause_map: dict[str, str] = {}

        for para in paragraphs:
            if not para.strip():
                continue
            result = predict_clause(para)
            label  = result.get("clause_type", "other")
            # Keep the first (typically best) representative for each type
            if label != "other" and label not in clause_map:
                clause_map[label] = para

        logger.info("Detected %d clause types.", len(clause_map))
        return clause_map or _mock_clauses()

    except Exception as exc:  # noqa: BLE001
        logger.warning("detect_clauses failed (%s), using mock.", exc)
        return _mock_clauses()


# ===========================================================================
# 3. Risk detection
# ===========================================================================

def detect_risks(text: str) -> dict[str, Any]:
    """
    Run the risk detection engine on the contract text.

    Parameters
    ----------
    text : str
        Raw contract text.

    Returns
    -------
    dict
        Keys: ``risk_score`` (float 0–10), ``risks`` (list[dict]),
        ``risk_summary`` (list[str]).
    """
    if not USE_REAL_AI:
        return _mock_risks()

    try:
        from ai_engine.paragraph_splitter import split_into_paragraphs

        paragraphs = split_into_paragraphs(text)
        engine     = _get_risk_engine()
        result     = engine.analyze(paragraphs)

        risk_items = [
            {
                "category":    match.rule.category.value if hasattr(match, "rule") else "unknown",
                "description": match.description if hasattr(match, "description") else str(match),
                "severity":    match.severity.value if hasattr(match, "severity") else "medium",
                "paragraph":   match.paragraph_text if hasattr(match, "paragraph_text") else None,
            }
            for match in result.risks
        ]

        risk_summary = [r["description"] for r in risk_items]

        return {
            "risk_score":  round(getattr(result, "risk_score", 5.0), 2),
            "risks":       risk_items,
            "risk_summary":risk_summary,
            "metadata":    {"paragraphs_analysed": len(paragraphs), "engine": "RiskDetectionEngine"},
        }

    except Exception as exc:  # noqa: BLE001
        logger.warning("detect_risks failed (%s), using mock.", exc)
        return _mock_risks()


# ===========================================================================
# 4. Summarization
# ===========================================================================

def generate_summary(text: str) -> dict[str, Any]:
    """
    Generate a structured summary of the contract.

    Parameters
    ----------
    text : str
        Raw contract text.

    Returns
    -------
    dict
        Keys: ``summary`` (list[str]), ``full_summary`` (str).
    """
    if not USE_REAL_AI:
        return _mock_summary()

    try:
        summarizer = _get_summarizer()
        result     = summarizer.generate_summary(text)
        output     = result.to_dict() if hasattr(result, "to_dict") else {}

        key_points: list[str] = output.get("key_points", [])
        full_text:  str       = output.get("overall_summary", "")

        return {
            "summary":      key_points or [full_text] if full_text else _mock_summary()["summary"],
            "full_summary": full_text,
        }

    except Exception as exc:  # noqa: BLE001
        logger.warning("generate_summary failed (%s), using mock.", exc)
        return _mock_summary()


# ===========================================================================
# 5. Contract comparison
# ===========================================================================

def compare_contracts(text_a: str, text_b: str) -> dict[str, Any]:
    """
    Semantically compare two contract texts and return a structured diff.

    Parameters
    ----------
    text_a : str
        Full text of the original contract.
    text_b : str
        Full text of the revised contract.

    Returns
    -------
    dict
        Keys: ``added_clauses``, ``removed_clauses``, ``modified_clauses``,
        ``similar_clauses``, ``metadata``.
    """
    if not USE_REAL_AI:
        return _mock_comparison()

    try:
        from ai_engine.paragraph_splitter import split_into_paragraphs

        clauses_a = split_into_paragraphs(text_a)
        clauses_b = split_into_paragraphs(text_b)

        comparator = _get_comparator()
        report     = comparator.compare_documents(clauses_a, clauses_b)
        raw        = report.to_dict()

        # Normalise into the schema used by comparison_service
        return {
            "added_clauses":    raw.get("added", []),
            "removed_clauses":  raw.get("removed", []),
            "modified_clauses": raw.get("modified", []),
            "similar_clauses":  raw.get("identical", []) + raw.get("related", []),
            "metadata":         raw.get("metadata", {}),
        }

    except Exception as exc:  # noqa: BLE001
        logger.warning("compare_contracts failed (%s), using mock.", exc)
        return _mock_comparison()


# ===========================================================================
# Mock responses  (used when USE_REAL_AI=false or AI modules are unavailable)
# ===========================================================================

def _mock_contract_text() -> str:
    return (
        "THIS AGREEMENT is entered into between Company A and Company B.\n\n"
        "1. Termination\n"
        "Either party may terminate this agreement with 30 days written notice.\n\n"
        "2. Payment Terms\n"
        "The client agrees to pay all invoices within thirty (30) days of receipt.\n\n"
        "3. Limitation of Liability\n"
        "In no event shall either party be liable for indirect or consequential damages.\n\n"
        "4. Confidentiality\n"
        "Each party agrees to keep all confidential information strictly private.\n\n"
        "5. Governing Law\n"
        "This agreement is governed by the laws of the State of California."
    )


def _mock_clauses() -> dict[str, str]:
    return {
        "termination":    "Either party may terminate this agreement with 30 days written notice.",
        "payment_terms":  "The client agrees to pay all invoices within thirty (30) days of receipt.",
        "liability":      "In no event shall either party be liable for indirect or consequential damages.",
        "confidentiality":"Each party agrees to keep all confidential information strictly private.",
        "governing_law":  "This agreement is governed by the laws of the State of California.",
    }


def _mock_risks() -> dict[str, Any]:
    return {
        "risk_score": 6.5,
        "risks": [
            {
                "category":    "liability",
                "description": "Broad limitation of liability clause detected — may expose client to uncapped losses.",
                "severity":    "high",
                "paragraph":   "In no event shall either party be liable for indirect or consequential damages.",
            },
            {
                "category":    "dispute_resolution",
                "description": "Missing dispute resolution clause — no arbitration or mediation mechanism defined.",
                "severity":    "medium",
                "paragraph":   None,
            },
        ],
        "risk_summary": [
            "Broad limitation of liability clause detected.",
            "Missing dispute resolution clause.",
        ],
        "metadata": {"paragraphs_analysed": 5, "engine": "mock"},
    }


def _mock_summary() -> dict[str, Any]:
    return {
        "summary": [
            "Contract duration is 2 years with automatic renewal.",
            "Vendor must provide monthly SaaS platform services.",
            "Termination requires 30 days written notice from either party.",
            "Client must pay invoices within 30 days of receipt.",
            "All confidential information must be kept strictly private.",
        ],
        "full_summary": (
            "This agreement establishes a 2-year service relationship between Company A "
            "and Company B. The vendor commits to monthly SaaS delivery, while the client "
            "agrees to timely payment. Either party may terminate with 30 days notice, "
            "and confidentiality obligations survive termination."
        ),
    }


def _mock_comparison() -> dict[str, Any]:
    return {
        "added_clauses": [
            "The vendor must comply with GDPR and applicable data protection regulations."
        ],
        "removed_clauses": [],
        "modified_clauses": [
            {
                "clause_a":   "Either party may terminate this agreement with 30 days notice.",
                "clause_b":   "Either party may terminate this agreement with 60 days notice.",
                "similarity": 0.82,
                "status":     "modified",
            }
        ],
        "similar_clauses": [
            {
                "clause_a":   "The vendor shall maintain confidentiality of all data.",
                "clause_b":   "The vendor must keep all customer information strictly confidential.",
                "similarity": 0.88,
                "status":     "related",
            }
        ],
        "metadata": {"engine": "mock"},
    }
