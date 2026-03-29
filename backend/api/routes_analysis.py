"""
backend/api/routes_analysis.py
================================
FastAPI router for contract analysis and summarisation endpoints.

Both routes now dispatch **async Celery tasks** and return immediately with a
``task_id``.  Clients poll ``GET /tasks/{task_id}`` for the result.

Routes
------
POST /contracts/{contract_id}/analyze  — Enqueue analysis task.
POST /contracts/{contract_id}/summary  — Enqueue summarisation task.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from backend.dependencies import get_contract_or_404
from backend.schemas.task_schema import TaskEnqueuedResponse
from backend.utils.logger import get_logger
from backend.worker.tasks import task_analyze_contract, task_summarize_contract

logger = get_logger(__name__)

router = APIRouter(prefix="/contracts", tags=["Analysis"])


# ---------------------------------------------------------------------------
# POST /contracts/{contract_id}/analyze
# ---------------------------------------------------------------------------

@router.post(
    "/{contract_id}/analyze",
    response_model=TaskEnqueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Analyse a contract (async)",
    description=(
        "Enqueue an AI analysis job for the uploaded contract. "
        "Returns a ``task_id`` immediately — poll ``GET /tasks/{task_id}`` "
        "every 2–3 seconds until ``status`` is ``'done'``.\n\n"
        "**Pipeline (runs in background worker):**\n"
        "Text extraction → Clause detection → Risk analysis"
    ),
)
async def analyze_contract(
    contract_id: str,
    _contract: dict = Depends(get_contract_or_404),
) -> TaskEnqueuedResponse:
    """
    Dispatch an async analysis task and return the task ID for polling.

    The ``contract_id`` must be a UUID returned by ``POST /contracts/upload``.
    """
    logger.info("Enqueuing analyze task: contract_id=%s", contract_id)

    try:
        task = task_analyze_contract.delay(contract_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to enqueue analyze task for contract_id=%s", contract_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Failed to enqueue analysis task: {exc}. "
                "Is the Celery worker running? "
                "Start it with: celery -A backend.worker.celery_app worker --loglevel=info"
            ),
        )

    logger.info("Enqueued analyze task_id=%s  contract_id=%s", task.id, contract_id)

    return TaskEnqueuedResponse(
        task_id     = task.id,
        status      = "pending",
        contract_id = contract_id,
        task_type   = "analyze",
        poll_url    = f"/tasks/{task.id}",
    )


# ---------------------------------------------------------------------------
# POST /contracts/{contract_id}/summary
# ---------------------------------------------------------------------------

@router.post(
    "/{contract_id}/summary",
    response_model=TaskEnqueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Summarise a contract (async)",
    description=(
        "Enqueue a summarisation job. "
        "Returns a ``task_id`` immediately — poll ``GET /tasks/{task_id}`` "
        "for the bullet-point result.\n\n"
        "**Pipeline (runs in background worker):**\n"
        "Text extraction → BART-based chunked summarisation"
    ),
)
async def summarise_contract(
    contract_id: str,
    _contract: dict = Depends(get_contract_or_404),
) -> TaskEnqueuedResponse:
    """Dispatch an async summarisation task and return the task ID for polling."""
    logger.info("Enqueuing summarize task: contract_id=%s", contract_id)

    try:
        task = task_summarize_contract.delay(contract_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to enqueue summarize task for contract_id=%s", contract_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Failed to enqueue summarisation task: {exc}. "
                "Is the Celery worker running?"
            ),
        )

    logger.info("Enqueued summarize task_id=%s  contract_id=%s", task.id, contract_id)

    return TaskEnqueuedResponse(
        task_id     = task.id,
        status      = "pending",
        contract_id = contract_id,
        task_type   = "summarize",
        poll_url    = f"/tasks/{task.id}",
    )
