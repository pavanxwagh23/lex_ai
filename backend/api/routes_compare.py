"""
backend/api/routes_compare.py
================================
FastAPI router for semantic contract comparison.

The comparison endpoint now dispatches an **async Celery task** and returns
HTTP 202 with a ``task_id`` for polling.

Routes
------
POST /contracts/compare  — Enqueue a comparison task.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backend.schemas.analysis_schema import CompareContractsRequest
from backend.schemas.task_schema import TaskEnqueuedResponse
from backend.services.document_service import get_contract
from backend.utils.logger import get_logger
from backend.worker.tasks import task_compare_contracts

logger = get_logger(__name__)

router = APIRouter(prefix="/contracts", tags=["Comparison"])


# ---------------------------------------------------------------------------
# POST /contracts/compare
# ---------------------------------------------------------------------------

@router.post(
    "/compare",
    response_model=TaskEnqueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Compare two contracts (async)",
    description=(
        "Semantically compare two uploaded contracts using sentence embeddings "
        "and FAISS vector search.\n\n"
        "Returns a ``task_id`` immediately — poll ``GET /tasks/{task_id}`` "
        "for the result with added, removed, modified, and similar clauses."
    ),
)
async def compare_contracts(
    request: CompareContractsRequest,
) -> TaskEnqueuedResponse:
    """
    Enqueue a semantic comparison task between ``contract_a`` and ``contract_b``.

    **Request body:**
    ```json
    { "contract_a": "<uuid-original>", "contract_b": "<uuid-revised>" }
    ```

    **Background pipeline:**
    Extract text → Split clauses → Embed (MiniLM-L6) → FAISS search → Classify diff
    """
    logger.info(
        "Enqueuing compare task: a=%s  b=%s",
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

    # --- prevent self-comparison ---
    if request.contract_a == request.contract_b:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="contract_a and contract_b must be different contracts.",
        )

    try:
        task = task_compare_contracts.delay(request.contract_a, request.contract_b)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Failed to enqueue compare task: a=%s b=%s",
            request.contract_a, request.contract_b,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Failed to enqueue comparison task: {exc}. "
                "Is the Celery worker running?"
            ),
        )

    logger.info(
        "Enqueued compare task_id=%s  a=%s  b=%s",
        task.id, request.contract_a, request.contract_b,
    )

    return TaskEnqueuedResponse(
        task_id     = task.id,
        status      = "pending",
        contract_id = request.contract_a,
        task_type   = "compare",
        poll_url    = f"/tasks/{task.id}",
    )
