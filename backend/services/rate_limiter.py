"""
backend/services/rate_limiter.py
================================
Simple in-memory IP rate limiter to protect conversational APIs.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

from fastapi import Request

from backend.utils.error_handler import AppError

# In-memory store: IP -> (request_count, window_start_time)
_RATE_STORE: Dict[str, Tuple[int, float]] = {}

RATE_LIMIT = 50
WINDOW_SECONDS = 60


def rate_limit(request: Request) -> None:
    """FastAPI dependency that enforces rate limits per IP.
    
    Raises
    ------
    AppError
        If the IP exceeds 50 requests per rolling minute.
    """
    client_ip = request.client.host if request.client else "unknown_ip"
    now = time.time()
    
    if client_ip not in _RATE_STORE:
        _RATE_STORE[client_ip] = (1, now)
        return
        
    count, window_start = _RATE_STORE[client_ip]
    
    # If the window has expired, reset it
    if now - window_start > WINDOW_SECONDS:
        _RATE_STORE[client_ip] = (1, now)
    else:
        # Still in the active window
        if count >= RATE_LIMIT:
            raise AppError(code=429, message="Too many requests", status_code=429)
        _RATE_STORE[client_ip] = (count + 1, window_start)
