"""
backend/api/routes_tasks.py
==============================
FastAPI router for task status polling.

Routes
------
GET  /tasks/{task_id}         — Poll the status of any queued task.
DELETE /tasks/{task_id}       — Cancel / revoke a pending or started task.
GET  /tasks/{task_id}/result  — Retrieve the raw result of a completed task.

Polling pattern (client-side)
------------------------------
::

    # 1. Submit work
    POST /contracts/{id}/analyze
    # → { "task_id": "abc123", "status": "pending", "poll_url": "/tasks/abc123" }

    # 2. Poll every 2 seconds until status is "done" or "failed"
    GET /tasks/abc123
    # → { "task_id": "abc123", "status": "started", "progress": 45 }
    # → { "task_id": "abc123", "status": "done",    "result": {...} }
"""

from __future__ import annotations

try:
    from celery.result import AsyncResult
    from backend.worker.celery_app import celery_app
    _CELERY_AVAILABLE = True
except Exception:  # noqa: BLE001
    _CELERY_AVAILABLE = False
    celery_app = None
    AsyncResult = None

from fastapi import APIRouter, HTTPException, status

from backend.schemas.task_schema import TaskStatus, TaskStatusResponse
from backend.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/tasks", tags=["Tasks"])


def _celery_state_to_task_status(celery_state: str) -> TaskStatus:
    """
    Map a Celery state string to our :data:`~backend.schemas.task_schema.TaskStatus` literal.

    Celery states include: ``PENDING``, ``STARTED``, ``SUCCESS``,
    ``FAILURE``, ``RETRY``, ``REVOKED``.
    """
    mapping: dict[str, TaskStatus] = {
        "PENDING": "pending",
        "STARTED": "started",
        "RETRY":   "started",    # still in progress
        "SUCCESS": "done",
        "FAILURE": "failed",
        "REVOKED": "revoked",
    }
    return mapping.get(celery_state.upper(), "pending")


# ---------------------------------------------------------------------------
# GET /tasks/{task_id}
# ---------------------------------------------------------------------------

@router.get(
    "/{task_id}",
    response_model=TaskStatusResponse,
    summary="Poll task status",
    description=(
        "Poll the status of an asynchronous AI task. "
        "Call this endpoint every 2–3 seconds after submitting an "
        "``/analyze``, ``/summary``, or ``/compare`` request until "
        "``status`` is ``'done'`` or ``'failed'``."
    ),
)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """
    Return current status and result (if complete) for the given Celery task.

    **Status lifecycle:**
    ``pending`` → ``started`` → ``done`` | ``failed`` | ``revoked``
    """
    result: AsyncResult = AsyncResult(task_id, app=celery_app)
    state  = result.state

    logger.debug("Polled task_id=%s  state=%s", task_id, state)

    task_status = _celery_state_to_task_status(state)

    # --- Successful completion ---
    if state == "SUCCESS":
        return TaskStatusResponse(
            task_id   = task_id,
            status    = "done",
            result    = result.result,
            progress  = 100,
        )

    # --- Failure ---
    if state == "FAILURE":
        exc = result.result  # Celery stores the exception as result on failure
        return TaskStatusResponse(
            task_id = task_id,
            status  = "failed",
            error   = str(exc),
        )

    # --- Revoked ---
    if state == "REVOKED":
        return TaskStatusResponse(
            task_id = task_id,
            status  = "revoked",
        )

    # --- In-progress (PENDING or STARTED) ---
    meta = result.info or {}   # info = dict passed to update_state(meta=...)
    progress = meta.get("progress", 0) if isinstance(meta, dict) else 0

    return TaskStatusResponse(
        task_id  = task_id,
        status   = task_status,
        progress = progress,
    )


# ---------------------------------------------------------------------------
# DELETE /tasks/{task_id}  — revoke/cancel a queued or running task
# ---------------------------------------------------------------------------

@router.delete(
    "/{task_id}",
    summary="Cancel a task",
    description=(
        "Revoke a pending or running task. "
        "If the task has already completed, revocation has no effect. "
        "A running task will be terminated only if ``terminate=True`` is set "
        "on the Celery worker — this is off by default for safety."
    ),
    status_code=status.HTTP_200_OK,
)
async def revoke_task(task_id: str) -> dict:
    """Cancel / revoke a queued Celery task."""
    result: AsyncResult = AsyncResult(task_id, app=celery_app)

    if result.state in ("SUCCESS", "FAILURE"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Task '{task_id}' has already finished with state '{result.state}'. Cannot revoke.",
        )

    celery_app.control.revoke(task_id, terminate=False)
    logger.info("Revoked task_id=%s", task_id)

    return {"task_id": task_id, "status": "revoked", "message": "Task revocation requested."}


# ---------------------------------------------------------------------------
# GET /tasks/{task_id}/result  — fetch raw result of a completed task
# ---------------------------------------------------------------------------

@router.get(
    "/{task_id}/result",
    summary="Get task result",
    description="Fetch the raw AI result of a completed task. Returns 404 if the task is not yet done.",
)
async def get_task_result(task_id: str) -> dict:
    """Return the raw result dict of a completed Celery task."""
    result: AsyncResult = AsyncResult(task_id, app=celery_app)

    if result.state == "SUCCESS":
        return {"task_id": task_id, "status": "done", "result": result.result}

    if result.state == "FAILURE":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Task '{task_id}' failed: {result.result}",
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Task '{task_id}' result not available yet (state: {result.state}).",
    )
