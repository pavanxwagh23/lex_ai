"""
backend/worker/tasks.py
========================
Celery task definitions. Tasks are only registered when Celery is
installed and available. Without Celery the module loads cleanly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when the worker runs standalone
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from celery import Task
    from celery.exceptions import Ignore
    from backend.worker.celery_app import celery_app
    _CELERY_AVAILABLE = True
except Exception:  # noqa: BLE001
    Task = object           # type: ignore[assignment,misc]
    class Ignore(Exception): pass  # type: ignore[no-redef]  # noqa: E701
    celery_app = None       # type: ignore[assignment]
    _CELERY_AVAILABLE = False

from backend.config import CELERY_MAX_RETRIES, CELERY_RETRY_BACKOFF
from backend.services import analysis_service, comparison_service
from backend.utils.logger import get_logger

logger = get_logger(__name__)

if not _CELERY_AVAILABLE:
    logger.warning(
        "Celery not installed — async task queue disabled. "
        "Install with: pip install celery redis"
    )


# ---------------------------------------------------------------------------
# Only register tasks when Celery is present
# ---------------------------------------------------------------------------
if _CELERY_AVAILABLE:

    class _LoggedTask(Task):  # type: ignore[misc]
        """Base class that logs task start, success, and failure."""
        abstract = True

        def on_failure(self, exc, task_id, args, kwargs, einfo):
            logger.error("Task FAILED  id=%s  name=%s  error=%s", task_id, self.name, exc)

        def on_success(self, retval, task_id, args, kwargs):
            logger.info("Task SUCCESS id=%s  name=%s", task_id, self.name)

        def on_retry(self, exc, task_id, args, kwargs, einfo):
            logger.warning("Task RETRY   id=%s  name=%s  error=%s", task_id, self.name, exc)

    @celery_app.task(bind=True, base=_LoggedTask, name="legal_ai.analyze_contract",
                     max_retries=CELERY_MAX_RETRIES, default_retry_delay=CELERY_RETRY_BACKOFF, acks_late=True)
    def task_analyze_contract(self, contract_id: str) -> dict:
        """Celery task: run the full AI analysis pipeline on a contract."""
        self.update_state(state="STARTED", meta={"progress": 0})
        logger.info("Task analyze_contract STARTED  contract_id=%s", contract_id)
        try:
            self.update_state(state="STARTED", meta={"progress": 10})
            response = analysis_service.run_analysis(contract_id)
            return response.model_dump()
        except (KeyError, FileNotFoundError) as exc:
            logger.error("analyze_contract permanent error: %s", exc)
            raise Ignore() from exc
        except Exception as exc:  # noqa: BLE001
            logger.warning("analyze_contract transient error (retry %d): %s", self.request.retries, exc)
            raise self.retry(exc=exc, countdown=CELERY_RETRY_BACKOFF * (2 ** self.request.retries))

    @celery_app.task(bind=True, base=_LoggedTask, name="legal_ai.summarize_contract",
                     max_retries=CELERY_MAX_RETRIES, default_retry_delay=CELERY_RETRY_BACKOFF, acks_late=True)
    def task_summarize_contract(self, contract_id: str) -> dict:
        """Celery task: generate a structured summary of a contract."""
        self.update_state(state="STARTED", meta={"progress": 0})
        logger.info("Task summarize_contract STARTED  contract_id=%s", contract_id)
        try:
            self.update_state(state="STARTED", meta={"progress": 10})
            response = analysis_service.run_summary(contract_id)
            return response.model_dump()
        except (KeyError, FileNotFoundError) as exc:
            logger.error("summarize_contract permanent error: %s", exc)
            raise Ignore() from exc
        except Exception as exc:  # noqa: BLE001
            logger.warning("summarize_contract transient error (retry %d): %s", self.request.retries, exc)
            raise self.retry(exc=exc, countdown=CELERY_RETRY_BACKOFF * (2 ** self.request.retries))

    @celery_app.task(bind=True, base=_LoggedTask, name="legal_ai.compare_contracts",
                     max_retries=CELERY_MAX_RETRIES, default_retry_delay=CELERY_RETRY_BACKOFF, acks_late=True)
    def task_compare_contracts(self, contract_id_a: str, contract_id_b: str) -> dict:
        """Celery task: semantically compare two uploaded contracts."""
        self.update_state(state="STARTED", meta={"progress": 0})
        logger.info("Task compare_contracts STARTED  a=%s  b=%s", contract_id_a, contract_id_b)
        try:
            self.update_state(state="STARTED", meta={"progress": 10})
            response = comparison_service.run_comparison(contract_id_a, contract_id_b)
            return response.model_dump()
        except (KeyError, FileNotFoundError) as exc:
            logger.error("compare_contracts permanent error: %s", exc)
            raise Ignore() from exc
        except Exception as exc:  # noqa: BLE001
            logger.warning("compare_contracts transient error (retry %d): %s", self.request.retries, exc)
            raise self.retry(exc=exc, countdown=CELERY_RETRY_BACKOFF * (2 ** self.request.retries))
else:
    class _MockTask:
        def delay(self, *args, **kwargs):
            raise Exception("Celery is not installed or unavailable.")

    task_analyze_contract = _MockTask()
    task_summarize_contract = _MockTask()
    task_compare_contracts = _MockTask()
