"""
backend/config.py
=================
Central configuration for the AI Legal Document Analyzer backend.

All tuneable constants, paths, and feature flags live here.
Other modules import from this file — never hardcode values elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — two levels up from this file (backend/ → project root)
# ---------------------------------------------------------------------------
BACKEND_DIR: Path  = Path(__file__).resolve().parent
PROJECT_ROOT: Path = BACKEND_DIR.parent

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

#: Directory where uploaded contract files are persisted.
STORAGE_DIR: Path = PROJECT_ROOT / "storage" / "contracts"

#: Allowed MIME types / extensions for uploaded contracts.
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx"})
ALLOWED_MIME_TYPES: frozenset[str] = frozenset({
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
})

#: Maximum upload size in bytes (default 50 MB).
MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", 50 * 1024 * 1024))

# ---------------------------------------------------------------------------
# API metadata
# ---------------------------------------------------------------------------

API_TITLE:       str = "AI Legal Document Analyzer"
API_DESCRIPTION: str = (
    "Backend API for uploading legal contracts, running AI clause detection, "
    "risk analysis, summarization, and semantic contract comparison."
)
API_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# AI pipeline feature flags
# ---------------------------------------------------------------------------

#: If True, the ai_pipeline module attempts to use the real AI engines.
#: If False (or engines unavailable), mock responses are returned.
USE_REAL_AI: bool = os.getenv("USE_REAL_AI", "true").lower() == "true"

#: Similarity thresholds for ContractComparator  (cosine, range [0,1])
COMPARE_THRESHOLD_IDENTICAL: float = 0.90
COMPARE_THRESHOLD_MODIFIED:  float = 0.75
COMPARE_THRESHOLD_RELATED:   float = 0.60

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL:  str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT: str = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Redis & Celery  (Phase 1 — Async task queue)
# ---------------------------------------------------------------------------

#: Redis connection URL.  Override via env var in production.
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

#: Celery broker — tasks are pushed here by the API.
CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", REDIS_URL)

#: Celery result backend — completed results are stored here.
CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

#: How long (seconds) Celery keeps task results in Redis before expiry.
CELERY_RESULT_TTL: int = int(os.getenv("CELERY_RESULT_TTL", 86_400))  # 24 h

#: Maximum retries for a failed AI task before giving up.
CELERY_MAX_RETRIES: int = int(os.getenv("CELERY_MAX_RETRIES", 3))

#: Seconds between retry attempts (exponential backoff applied on top).
CELERY_RETRY_BACKOFF: int = int(os.getenv("CELERY_RETRY_BACKOFF", 5))
