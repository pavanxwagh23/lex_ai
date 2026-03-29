"""
backend/services/chat_service.py
================================
Service for handling general conversational intents.

Delegates to the LLM client to generate natural language responses.
"""

from __future__ import annotations

from typing import Dict, Any

from backend.ai_client.llm_client import get_llm_client
from backend.services.session_manager import get_session, add_message
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def generate_chat_response(message: str, session_id: str = None) -> Dict[str, Any]:
    """Generate a conversational response using the LLM.

    Parameters
    ----------
    message : str
        User's natural language message.
    session_id : str, optional
        Session ID to maintain conversational continuity.

    Returns
    -------
    dict
        ``{"response": str, "data": None}``

    Raises
    ------
    ValueError
        If message is empty.
    """
    if not message or not message.strip():
        raise ValueError("Cannot generate a response for an empty message.")

    logger.info("chat_service: generating response for message")
    
    # Store user message and build context for LLM
    context = ""
    if session_id:
        add_message(session_id, "user", message)
        session = get_session(session_id)
        history = session.get("messages", [])
        
        # Don't include the message we just added in the 'past' history block
        past_msgs = history[:-1] if history else []
        if past_msgs:
            history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in past_msgs])
            context = f"\n\n--- Conversation History ---\n{history_str}\n----------------------------"

    system_prompt = (
        "You are Lex AI, a helpful and knowledgeable legal document assistant. "
        "You can summarize contracts, identify legal risks, and compare documents. "
        "Provide clear, concise, and professional answers to legal questions. "
        "Remind the user to consult a qualified attorney for specific legal advice."
    )
    
    client = get_llm_client()
    final_prompt = message + context
    response_text = client.generate(prompt=final_prompt, system_prompt=system_prompt)
    
    # Store assistant response in session
    if session_id:
        add_message(session_id, "assistant", response_text)
    
    return {
        "response": response_text,
        "data": None
    }
