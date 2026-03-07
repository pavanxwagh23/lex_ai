"""
backend/services/analysis_service.py
========================================
Business logic for contract analysis and summarisation.

This service orchestrates calls to the AI pipeline and structures
the results into the Pydantic schemas returned by the API routes.
"""

from __future__ import annotations

from backend.ai_client import ai_pipeline
from backend.schemas.analysis_schema import AnalysisResponse, RiskItem, SummaryResponse
from backend.services.document_service import get_file_text
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def run_analysis(contract_id: str) -> AnalysisResponse:
    """
    Run the full AI analysis pipeline on an uploaded contract.

    Steps
    -----
    1. Locate the file on disk via the in-memory registry.
    2. Extract plain text.
    3. Detect clause types.
    4. Detect legal risks.
    5. Assemble and return :class:`~backend.schemas.analysis_schema.AnalysisResponse`.

    Parameters
    ----------
    contract_id : str
        UUID hex string of the uploaded contract.

    Returns
    -------
    AnalysisResponse

    Raises
    ------
    KeyError
        If ``contract_id`` is not in the in-memory registry.
    FileNotFoundError
        If the contract file is missing from disk.
    """
    logger.info("Starting analysis for contract_id=%s", contract_id)

    # Step 1 — locate file
    file_path_str, _ = get_file_text(contract_id)

    # Step 2 — extract text
    text = ai_pipeline.extract_text(file_path_str)

    # Step 3 — detect clauses
    clauses = ai_pipeline.detect_clauses(text)

    # Step 4 — detect risks
    risk_data = ai_pipeline.detect_risks(text)

    # Step 5 — assemble response
    risk_items = [
        RiskItem(
            category=r.get("category", "unknown"),
            description=r.get("description", ""),
            severity=r.get("severity", "medium"),
            paragraph=r.get("paragraph"),
        )
        for r in risk_data.get("risks", [])
    ]

    logger.info(
        "Analysis complete: contract_id=%s  risk_score=%.2f  clauses=%d  risks=%d",
        contract_id,
        risk_data.get("risk_score", 0.0),
        len(clauses),
        len(risk_items),
    )

    return AnalysisResponse(
        contract_id  = contract_id,
        risk_score   = risk_data.get("risk_score", 0.0),
        clauses      = clauses,
        risks        = risk_items,
        risk_summary = risk_data.get("risk_summary", []),
        metadata     = risk_data.get("metadata", {}),
    )


def run_summary(contract_id: str) -> SummaryResponse:
    """
    Generate a structured natural-language summary of an uploaded contract.

    Parameters
    ----------
    contract_id : str
        UUID hex string of the uploaded contract.

    Returns
    -------
    SummaryResponse

    Raises
    ------
    KeyError
        If ``contract_id`` is not in the in-memory registry.
    FileNotFoundError
        If the contract file is missing from disk.
    """
    logger.info("Starting summarisation for contract_id=%s", contract_id)

    file_path_str, _ = get_file_text(contract_id)
    text              = ai_pipeline.extract_text(file_path_str)
    summary_data      = ai_pipeline.generate_summary(text)

    logger.info(
        "Summary complete: contract_id=%s  bullet_points=%d",
        contract_id,
        len(summary_data.get("summary", [])),
    )

    return SummaryResponse(
        contract_id  = contract_id,
        summary      = summary_data.get("summary", []),
        full_summary = summary_data.get("full_summary"),
    )
