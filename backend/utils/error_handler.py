"""
backend/utils/error_handler.py
==============================
Structured error handling for the AI Assistant.

Provides the `AppError` exception and the global FastAPI exception
handler that maps it into the standard ChatResponse JSON schema.
"""

from fastapi import Request
from fastapi.responses import JSONResponse

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class AppError(Exception):
    """Custom application exception that maps to HTTP responses.
    
    Attributes
    ----------
    code : int
        Internal error code (often matches HTTP status).
    message : str
        Human-readable error explanation.
    status_code : int
        HTTP status code to return.
    """
    
    def __init__(self, code: int, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Global handler for catching AppErrors and returning standard JSON."""
    logger.error("AppError [%s]: %s", exc.code, exc.message)
    import time
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "intent": "ERROR",
            "message": exc.message,
            "data": None,
            "meta": {
                "confidence": 0.0,
                "timestamp": int(time.time()),
                "error_code": exc.code,
                "status": "failed"
            }
        }
    )
