"""
backend/worker/tasks.py
========================
Celery task definitions for all long-running AI pipeline operations.

Each task is a thin wrapper that:
1. Calls the relevant service function (which calls the AI pipeline).
2. Returns a JSON-serialisable dict as the task result.
3. Updates task state to ``STARTED`` as soon as processing begins so the
   polling endpoint can show meaningful progress.
4. Retries on transient failures (network, model loading) using exponential
   backoff.

Retry behaviour
---------------
Tasks retry up to ``CELERY_MAX_RETRIES`` times with exponential backoff
starting at ``CELERY_RETRY_BACKOFF`` seconds.  Permanent errors (file not
found, validation) are **not** retried — they raise ``Ignore`` immediately.

Task naming convention
----------------------
All tasks are named ``legal_ai.<action>`` to avoid collisions in a shared
Celery cluster.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when the worker process runs standalone
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from celery import Task
from celery.exceptions import Ignore

from backend.config import CELERY_MAX_RETRIES, CELERY_RETRY_BACKOFF
from backend.services import analysis_service, comparison_service
from backend.utils.logger import get_logger
from backend.worker.celery_app import celery_app

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Base task class — adds structured error logging
# ---------------------------------------------------------------------------

class _LoggedTask(Task):
    """Base class that logs task start, success, and failure uniformly."""

    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(
            "Task FAILED  id=%s  name=%s  error=%s",
            task_id, self.name, exc,
        )

    def on_success(self, retval, task_id, args, kwargs):
        logger.info("Task SUCCESS id=%s  name=%s", task_id, self.name)

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        logger.warning(
            "Task RETRY   id=%s  name=%s  error=%s  countdown=%s",
            task_id, self.name, exc, self.request.retries,
        )


# ===========================================================================
# Task 1 — Contract Analysis
# ===========================================================================

@celery_app.task(
    bind=True,
    base=_LoggedTask,
    name="legal_ai.analyze_contract",
    max_retries=CELERY_MAX_RETRIES,
    default_retry_delay=CELERY_RETRY_BACKOFF,
    acks_late=True,
)
def task_analyze_contract(self, contract_id: str) -> dict:
    """
    Celery task: run the full AI analysis pipeline on a contract.

    Parameters
    ----------
    contract_id : str
        UUID of the uploaded contract (must be in the in-memory registry).

    Returns
    -------
    dict
        JSON-serialisable :class:`~backend.schemas.analysis_schema.AnalysisResponse`
        produced by :func:`~backend.services.analysis_service.run_analysis`.

    Raises
    ------
    Ignore
        For permanent errors (contract not found, file missing).
    self.retry
        For transient errors (model loading race condition, I/O errors).
    """
    self.update_state(state="STARTED", meta={"progress": 0, "step": "starting"})
    logger.info("Task analyze_contract STARTED  contract_id=%s", contract_id)

    try:
        self.update_state(state="STARTED", meta={"progress": 10, "step": "extracting_text"})
        response = analysis_service.run_analysis(contract_id)

        self.update_state(state="STARTED", meta={"progress": 100, "step": "done"})
        return response.model_dump()

    except (KeyError, FileNotFoundError) as exc:
        # Permanent error — do not retry
        logger.error("analyze_contract permanent error: %s", exc)
        raise Ignore() from exc

    except Exception as exc:  # noqa: BLE001
        # Transient error — retry with exponential backoff
        logger.warning("analyze_contract transient error (retry %d): %s", self.request.retries, exc)
        raise self.retry(
            exc=exc,
            countdown=CELERY_RETRY_BACKOFF * (2 ** self.request.retries),
        )


# ===========================================================================
# Task 2 — Contract Summarisation
# ===========================================================================

@celery_app.task(
    bind=True,
    base=_LoggedTask,
    name="legal_ai.summarize_contract",
    max_retries=CELERY_MAX_RETRIES,
    default_retry_delay=CELERY_RETRY_BACKOFF,
    acks_late=True,
)
def task_summarize_contract(self, contract_id: str) -> dict:
    """
    Celery task: generate a structured summary of a contract.

    Parameters
    ----------
    contract_id : str
        UUID of the uploaded contract.

    Returns
    -------
    dict
        JSON-serialisable :class:`~backend.schemas.analysis_schema.SummaryResponse`.
    """
    self.update_state(state="STARTED", meta={"progress": 0, "step": "starting"})
    logger.info("Task summarize_contract STARTED  contract_id=%s", contract_id)

    try:
        self.update_state(state="STARTED", meta={"progress": 10, "step": "extracting_text"})
        response = analysis_service.run_summary(contract_id)

        self.update_state(state="STARTED", meta={"progress": 100, "step": "done"})
        return response.model_dump()

    except (KeyError, FileNotFoundError) as exc:
        logger.error("summarize_contract permanent error: %s", exc)
        raise Ignore() from exc

    except Exception as exc:  # noqa: BLE001
        logger.warning("summarize_contract transient error (retry %d): %s", self.request.retries, exc)
        raise self.retry(
            exc=exc,
            countdown=CELERY_RETRY_BACKOFF * (2 ** self.request.retries),
        )


# ===========================================================================
# Task 3 — Contract Comparison
# ===========================================================================

@celery_app.task(
    bind=True,
    base=_LoggedTask,
    name="legal_ai.compare_contracts",
    max_retries=CELERY_MAX_RETRIES,
    default_retry_delay=CELERY_RETRY_BACKOFF,
    acks_late=True,
)
def task_compare_contracts(self, contract_id_a: str, contract_id_b: str) -> dict:
    """
    Celery task: semantically compare two uploaded contracts.

    Parameters
    ----------
    contract_id_a : str
        UUID of the original contract.
    contract_id_b : str
        UUID of the revised contract.

    Returns
    -------
    dict
        JSON-serialisable
        :class:`~backend.schemas.analysis_schema.CompareContractsResponse`.
    """
    self.update_state(state="STARTED", meta={"progress": 0, "step": "starting"})
    logger.info(
        "Task compare_contracts STARTED  a=%s  b=%s",
        contract_id_a, contract_id_b,
    )

    try:
        self.update_state(state="STARTED", meta={"progress": 10, "step": "extracting_texts"})
        response = comparison_service.run_comparison(contract_id_a, contract_id_b)

        self.update_state(state="STARTED", meta={"progress": 100, "step": "done"})
        return response.model_dump()

    except (KeyError, FileNotFoundError) as exc:
        logger.error("compare_contracts permanent error: %s", exc)
        raise Ignore() from exc

    except Exception as exc:  # noqa: BLE001
        logger.warning("compare_contracts transient error (retry %d): %s", self.request.retries, exc)
        raise self.retry(
            exc=exc,
            countdown=CELERY_RETRY_BACKOFF * (2 ** self.request.retries),
        )
