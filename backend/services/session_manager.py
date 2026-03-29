"""
backend/services/session_manager.py
===================================
In-memory session state manager for maintaining conversational context.

This tracks the last uploaded / referenced contract text for a user
so they don't have to keep supplying it in every message.
"""

from __future__ import annotations

import time
from typing import Dict, Any, List

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Global in-memory store
# Map of session_id -> dict of session data
_SESSION_STORE: Dict[str, Dict[str, Any]] = {}

SESSION_TTL_SECONDS = 30 * 60  # 30 minutes


def cleanup_sessions() -> None:
    """Remove expired sessions from the in-memory store."""
    now = time.time()
    expired = []
    for sid, data in _SESSION_STORE.items():
        if now - data.get("updated_at", now) > SESSION_TTL_SECONDS:
            expired.append(sid)
            
    for sid in expired:
        del _SESSION_STORE[sid]
        
    if expired:
        logger.info("Cleaned up %d expired sessions.", len(expired))


def get_session(session_id: str) -> Dict[str, Any]:
    """Retrieve session data by session_id.
    
    If the session does not exist, an empty session dict is created and returned.
    Also updates the access timestamp.
    """
    cleanup_sessions()
    
    if session_id not in _SESSION_STORE:
        logger.info("Initializing new session: %s", session_id)
        _SESSION_STORE[session_id] = {
            "messages": [],
            "updated_at": time.time()
        }
    else:
        _SESSION_STORE[session_id]["updated_at"] = time.time()
        
    return _SESSION_STORE[session_id]


def update_session(session_id: str, data: Dict[str, Any]) -> None:
    """Update session data with new key-value pairs."""
    session = get_session(session_id)
    session.update(data)
    session["updated_at"] = time.time()
    logger.debug("Updated session %s with keys: %s", session_id, list(data.keys()))


def add_message(session_id: str, role: str, message: str) -> None:
    """Store recent messages in session context, keeping only the last 3 user messages."""
    session = get_session(session_id)
    messages: List[Dict[str, str]] = session.setdefault("messages", [])
    
    messages.append({"role": role, "content": message})
    
    # Keep up to the last 6 messages total (3 turns)
    if len(messages) > 6:
        session["messages"] = messages[-6:]


