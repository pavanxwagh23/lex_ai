"""
backend/worker/celery_app.py
=============================
Celery application factory for the AI Legal Document Analyzer.

This module creates and configures the shared Celery instance that both
the API (task producer) and the worker process (task consumer) import.

Starting the worker
-------------------
::

    # From the project root
    celery -A backend.worker.celery_app worker --loglevel=info --concurrency=2

Monitoring (Flower UI — http://localhost:5555)
----------------------------------------------
::

    celery -A backend.worker.celery_app flower --port=5555

Environment variables
---------------------
``REDIS_URL``            — Redis connection URL (default: redis://localhost:6379/0)
``CELERY_RESULT_TTL``    — Seconds to keep results in Redis (default: 86400 = 24 h)
``CELERY_MAX_RETRIES``   — Max retry attempts on failure (default: 3)
``CELERY_RETRY_BACKOFF`` — Base seconds between retries (default: 5)
"""

from __future__ import annotations

from celery import Celery

from backend.config import (
    CELERY_BROKER_URL,
    CELERY_MAX_RETRIES,
    CELERY_RESULT_BACKEND,
    CELERY_RESULT_TTL,
    CELERY_RETRY_BACKOFF,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Create the Celery application
# ---------------------------------------------------------------------------

celery_app = Celery(
    main="legal_ai_analyzer",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        "backend.worker.tasks",  # auto-discover task functions
    ],
)

# ---------------------------------------------------------------------------
# Celery configuration
# ---------------------------------------------------------------------------

celery_app.conf.update(
    # Serialisation
    task_serializer          = "json",
    result_serializer        = "json",
    accept_content           = ["json"],

    # Result expiry
    result_expires           = CELERY_RESULT_TTL,

    # Reliability
    task_acks_late           = True,       # re-queue on worker crash
    task_reject_on_worker_lost = True,

    # Retry defaults (overridden per-task where needed)
    task_max_retries         = CELERY_MAX_RETRIES,
    task_default_retry_delay = CELERY_RETRY_BACKOFF,

    # Performance — prefetch one task at a time per worker thread
    # (important for long-running AI inference tasks)
    worker_prefetch_multiplier = 1,

    # Timezone
    timezone                 = "UTC",
    enable_utc               = True,
)

logger.info(
    "Celery configured — broker=%s  backend=%s  result_ttl=%ds",
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
    CELERY_RESULT_TTL,
)
