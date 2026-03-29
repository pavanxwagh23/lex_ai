"""
backend/schemas/task_schema.py
================================
Pydantic schemas for async task status and result endpoints.

Every long-running AI operation (analyze, summarize, compare) returns a
``TaskResponse`` immediately.  Clients poll ``GET /tasks/{task_id}`` until
``status`` is ``"done"`` or ``"failed"``.

Task lifecycle
--------------
::

    PENDING   → task is queued, not yet picked up by a worker
    STARTED   → worker has begun processing
    DONE      → completed successfully; ``result`` is populated
    FAILED    → terminated with an error; ``error`` is populated
    REVOKED   → task was cancelled before execution
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Status literals
# ---------------------------------------------------------------------------

TaskStatus = Literal["pending", "started", "done", "failed", "revoked"]


# ---------------------------------------------------------------------------
# Immediate response returned when a task is queued
# ---------------------------------------------------------------------------

class TaskEnqueuedResponse(BaseModel):
    """
    Returned immediately by POST endpoints after a task is queued.

    The client must poll ``GET /tasks/{task_id}`` for the result.
    """

    task_id:     str        = Field(..., description="Celery task UUID")
    status:      TaskStatus = Field(default="pending")
    contract_id: str        = Field(..., description="Associated contract UUID")
    task_type:   str        = Field(..., description="'analyze' | 'summarize' | 'compare'")
    poll_url:    str        = Field(..., description="Relative URL to poll for result")

    model_config = {"json_schema_extra": {
        "example": {
            "task_id":     "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "status":      "pending",
            "contract_id": "8b31f9a2c14b4e0d9a2b3c4d5e6f7a8b",
            "task_type":   "analyze",
            "poll_url":    "/tasks/3fa85f64-5717-4562-b3fc-2c963f66afa6",
        }
    }}


# ---------------------------------------------------------------------------
# Polling response returned by GET /tasks/{task_id}
# ---------------------------------------------------------------------------

class TaskStatusResponse(BaseModel):
    """
    Returned by ``GET /tasks/{task_id}`` while polling for a result.
    """

    task_id:    str           = Field(..., description="Celery task UUID")
    status:     TaskStatus    = Field(..., description="Current task lifecycle status")
    task_type:  Optional[str] = Field(None, description="'analyze' | 'summarize' | 'compare'")
    result:     Optional[Any] = Field(
        None,
        description="AI pipeline result (populated when status='done')",
    )
    error:      Optional[str] = Field(
        None,
        description="Error message (populated when status='failed')",
    )
    progress:   Optional[int] = Field(
        None,
        description="Percentage complete 0–100 (if reported by task)",
        ge=0, le=100,
    )

    model_config = {"json_schema_extra": {
        "example": {
            "task_id":   "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "status":    "done",
            "task_type": "analyze",
            "result": {
                "contract_id": "8b31f9a2c14b4e0d9a2b3c4d5e6f7a8b",
                "risk_score":  7.2,
                "clauses":     {"termination": "Either party may terminate …"},
                "risks":       [{"category": "liability", "description": "…", "severity": "high"}],
            },
            "error":    None,
            "progress": 100,
        }
    }}
