"""
backend/utils/logger.py
=======================
Centralised logging configuration for the AI Legal Document Analyzer backend.

Usage
-----
Import the factory anywhere in the backend::

    from backend.utils.logger import get_logger
    logger = get_logger(__name__)

Call ``configure_logging()`` once at application startup (done in main.py).
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """
    Configure the root logger for the entire application.

    This should be called **once** at startup in ``main.py`` before any
    other module emits log records.  Subsequent calls are safe (idempotent).

    Parameters
    ----------
    level : str
        Log level string — ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, etc.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Avoid adding duplicate handlers if called multiple times
    root = logging.getLogger()
    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root.setLevel(log_level)
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)

import json

def log_event(event_data: dict) -> None:
    """
    Log structured event data as JSON.
    
    Expected keys: intent, execution_time, module, status
    """
    logger = logging.getLogger("event_logger")
    try:
        # Convert to string to avoid complex non-serializable objects breaking the logger
        structured_msg = json.dumps(event_data)
        logger.info(structured_msg)
    except Exception as e:
        logger.error(f"Failed to log event: {e}")

