"""
backend/services/task_manager.py
================================
In-memory task manager for tracking background processing.

Used to track the state of heavy ML tasks (e.g. summarization,
contract comparison) so that the user can poll for results.
"""

from __future__ import annotations

import uuid
from typing import Dict, Any, Optional

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Global in-memory store
# Map of task_id -> Dict containing status, result, error
_TASK_STORE: Dict[str, Dict[str, Any]] = {}


def create_task() -> str:
    """Create a new pending task and return its task_id.

    Returns
    -------
    str
        Unique UUID representing the background task.
    """
    task_id = str(uuid.uuid4())
    _TASK_STORE[task_id] = {
        "status": "processing",
        "result": None,
        "error": None
    }
    logger.info("Created background task: %s", task_id)
    return task_id


def update_task(
    task_id: str,
    result: Optional[Dict[str, Any]] = None,
    status: str = "completed",
    error: Optional[str] = None
) -> None:
    """Update a task's status, result, or error message.

    Parameters
    ----------
    task_id : str
        The unique UUID of the task.
    result : dict, optional
        The successful output payload of the task.
    status : str
        The current status (e.g., "completed", "failed").
    error : str, optional
        Error message if the task failed.
    """
    if task_id in _TASK_STORE:
        _TASK_STORE[task_id]["status"] = status
        _TASK_STORE[task_id]["result"] = result
        _TASK_STORE[task_id]["error"] = error
        logger.info("Updated task %s -> status=%s", task_id, status)
    else:
        logger.warning("Attempted to update non-existent task %s", task_id)


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    """Get the state of a specific task.

    Parameters
    ----------
    task_id : str
        The unique UUID of the task.

    Returns
    -------
    dict or None
        The task record containing 'status', 'result', and 'error',
        or None if not found.
    """
    return _TASK_STORE.get(task_id)
