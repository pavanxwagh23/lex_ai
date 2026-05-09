"""
backend/main.py
================
FastAPI application entrypoint for the AI Legal Document Analyzer backend.

Startup sequence
----------------
1. Configure centralised logging.
2. Ensure the contract storage directory exists.
3. Create the FastAPI application with metadata.
4. Register CORS middleware.
5. Mount API routers.
6. Expose health-check and root endpoints.

Running
-------
::

    # From the project root
    uvicorn backend.main:app --reload --port 8000

OpenAPI docs are available at:

    http://localhost:8000/docs
    http://localhost:8000/redoc
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from pathlib import Path as _Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes_analysis import router as analysis_router
from backend.api.routes_compare import router as compare_router
from backend.api.routes_tasks import router as tasks_router
from backend.api.routes_upload import router as upload_router
from backend.api.chat_routes import router as chat_router
from backend.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    CORS_ALLOWED_ORIGINS,
    LOG_LEVEL,
    STORAGE_DIR,
)
from backend.utils.file_storage import ensure_storage_dir
from backend.utils.logger import configure_logging, get_logger
from backend.utils.error_handler import AppError, app_error_handler

# ---------------------------------------------------------------------------
# Logging — configure before anything else emits records
# ---------------------------------------------------------------------------
configure_logging(level=LOG_LEVEL)
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan event handler (replaces deprecated on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Handle application startup and shutdown events.

    Startup
    -------
    - Ensure the storage directory exists.
    - Log a ready banner.

    Shutdown
    --------
    - Log a goodbye message.
    """
    # --- STARTUP ---
    logger.info("=" * 60)
    logger.info("  %s  v%s", API_TITLE, API_VERSION)
    logger.info("=" * 60)

    ensure_storage_dir()
    logger.info("Storage directory: %s", STORAGE_DIR)
    logger.info("API ready. Docs: http://localhost:8000/docs")

    yield  # application runs here

    # --- SHUTDOWN ---
    logger.info("Server shutting down. Goodbye.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_exception_handler(AppError, app_error_handler)

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

#  /contracts/upload            ← routes_upload
#  /contracts                   ← routes_upload (list / get metadata)
#  /contracts/{id}/analyze      ← routes_analysis
#  /contracts/{id}/summary      ← routes_analysis
#  /contracts/compare           ← routes_compare

# NOTE: routes_compare must be registered BEFORE routes_analysis so that
# the literal path "/contracts/compare" is matched before the parametric
# path "/contracts/{contract_id}/...".
app.include_router(compare_router)
app.include_router(upload_router)
app.include_router(analysis_router)
app.include_router(tasks_router)
app.include_router(chat_router)

# ---------------------------------------------------------------------------
# Serve the premium HTML/CSS/JS frontend
# ---------------------------------------------------------------------------
_FRONTEND_DIR = _Path(__file__).resolve().parent.parent / "frontend"

if _FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
    logger.info("Frontend served at http://localhost:8000/app")
else:
    logger.warning("Frontend directory not found: %s", _FRONTEND_DIR)


# ---------------------------------------------------------------------------
# Health check & root
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"], summary="API root / health check")
async def root() -> JSONResponse:
    """
    Confirm the API is running.

    Returns
    -------
    JSON
        ``{"status": "ok", "version": "1.0.0", "docs": "/docs"}``
    """
    return JSONResponse({
        "status":  "ok",
        "service": API_TITLE,
        "version": API_VERSION,
        "docs":    "/docs",
    })


@app.get("/health", tags=["Health"], summary="Health probe")
async def health() -> JSONResponse:
    """Kubernetes / load-balancer health probe endpoint."""
    return JSONResponse({"status": "healthy"})


# ---------------------------------------------------------------------------
# Development entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=LOG_LEVEL.lower(),
    )
