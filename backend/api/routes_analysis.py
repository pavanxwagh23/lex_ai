"""
backend/api/routes_analysis.py
================================
FastAPI router for contract analysis and summarisation endpoints.

Routes
------
POST /contracts/{contract_id}/analyze  — Run full AI analysis.
POST /contracts/{contract_id}/summary  — Generate contract summary.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from backend.dependencies import get_contract_or_404
from backend.schemas.analysis_schema import AnalysisResponse, SummaryResponse
from backend.services import analysis_service
from backend.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/contracts", tags=["Analysis"])


# ---------------------------------------------------------------------------
# POST /contracts/{contract_id}/analyze
# ---------------------------------------------------------------------------

@router.post(
    "/{contract_id}/analyze",
    response_model=AnalysisResponse,
    summary="Analyse a contract",
    description=(
        "Run the full AI pipeline on an uploaded contract: "
        "text extraction → clause detection → risk analysis. "
        "Returns a risk score, detected clause map, and a list of risk findings."
    ),
)
async def analyze_contract(
    contract_id: str,
    _contract: dict = Depends(get_contract_or_404),   # ensures 404 guard runs
) -> AnalysisResponse:
    """
    Analyse an uploaded contract.

    **Pipeline:**
    1. Extract text from the stored file.
    2. Detect clause types using the clause classifier.
    3. Run risk detection rules.
    4. Return structured JSON.

    The ``contract_id`` must be a UUID returned by ``POST /contracts/upload``.
    """
    logger.info("Analysis request: contract_id=%s", contract_id)

    try:
        return analysis_service.run_analysis(contract_id)

    except FileNotFoundError as exc:
        logger.error("Analysis — file not found: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:     # noqa: BLE001
        logger.exception("Analysis pipeline error for contract_id=%s", contract_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {exc}",
        )


# ---------------------------------------------------------------------------
# POST /contracts/{contract_id}/summary
# ---------------------------------------------------------------------------

@router.post(
    "/{contract_id}/summary",
    response_model=SummaryResponse,
    summary="Summarise a contract",
    description=(
        "Generate a concise, bullet-point summary of the contract using the "
        "BART-based legal summarisation pipeline. "
        "Returns a list of key points and an optional full prose summary."
    ),
)
async def summarise_contract(
    contract_id: str,
    _contract: dict = Depends(get_contract_or_404),
) -> SummaryResponse:
    """
    Summarise an uploaded contract.

    **Pipeline:**
    1. Extract text from the stored file.
    2. Chunk and summarise using the LegalSummarizer model.
    3. Return structured bullet points.
    """
    logger.info("Summary request: contract_id=%s", contract_id)

    try:
        return analysis_service.run_summary(contract_id)

    except FileNotFoundError as exc:
        logger.error("Summary — file not found: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:     # noqa: BLE001
        logger.exception("Summary pipeline error for contract_id=%s", contract_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarisation failed: {exc}",
        )
