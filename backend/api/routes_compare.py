"""
backend/api/routes_compare.py
================================
FastAPI router for semantic contract comparison.

Routes
------
POST /contracts/compare  — Compare two uploaded contracts.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.schemas.analysis_schema import (
    CompareContractsRequest,
    CompareContractsResponse,
)
from backend.services import comparison_service
from backend.services.document_service import get_contract
from backend.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/contracts", tags=["Comparison"])


# ---------------------------------------------------------------------------
# POST /contracts/compare
# ---------------------------------------------------------------------------

@router.post(
    "/compare",
    response_model=CompareContractsResponse,
    summary="Compare two contracts",
    description=(
        "Semantically compare two uploaded contracts using sentence embeddings "
        "and FAISS vector search. Returns clauses that were added, removed, "
        "modified, or remain semantically identical — even when wording changes."
    ),
)
async def compare_contracts(
    request: CompareContractsRequest,
) -> CompareContractsResponse:
    """
    Compare ``contract_a`` and ``contract_b`` using the ContractComparator engine.

    **Request body:**
    ```json
    {
        "contract_a": "<uuid-of-original>",
        "contract_b": "<uuid-of-revised>"
    }
    ```

    **Algorithm:**
    1. Load and extract text from both contracts.
    2. Split into clauses.
    3. Embed with ``all-MiniLM-L6-v2`` → FAISS top-1 search.
    4. Classify each pair: identical / modified / related / removed.
    5. Clauses in B with no match → added.
    """
    logger.info(
        "Compare request: contract_a=%s  contract_b=%s",
        request.contract_a, request.contract_b,
    )

    # --- verify both contracts exist in the registry ---
    for cid in (request.contract_a, request.contract_b):
        try:
            get_contract(cid)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Contract '{cid}' not found. "
                    "Upload it first via POST /contracts/upload."
                ),
            )

    # --- prevent comparing a contract with itself ---
    if request.contract_a == request.contract_b:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="contract_a and contract_b must be different contracts.",
        )

    try:
        return comparison_service.run_comparison(request.contract_a, request.contract_b)

    except FileNotFoundError as exc:
        logger.error("Compare — file not found: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:     # noqa: BLE001
        logger.exception(
            "Comparison pipeline error: a=%s b=%s",
            request.contract_a, request.contract_b,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {exc}",
        )
