"""
backend/services/router.py
==========================
Central routing logic for the AI assistant chat layer.

Takes an intent and payload, then dispatches the request to the
appropriate specific service (summary, risk, compare, or general chat).
"""

from __future__ import annotations

from typing import Dict, Any

from backend.services.summary_service import generate_summary
from backend.services.risk_service import analyze_risk_from_text
from backend.services.compare_service import compare_contracts
from backend.services.chat_service import generate_chat_response
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def route_request(intent: str, message: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Route a chat request to the correct internal service.

    Parameters
    ----------
    intent : str
        Detected intent label (e.g., "SUMMARY", "RISK").
    message : str
        Original user message.
    payload : dict
        Context payload. Must contain "text" for SUMMARY/RISK.
        Must contain "text_a" and "text_b" for COMPARE.

    Returns
    -------
    dict
        Structured output specific to the called service.
        Format: ``{"response": str, "data": dict|None}``

    Raises
    ------
    ValueError
        If required payload data is missing for the detected intent.
    """
    logger.info("Routing request -> intent: %s", intent)

    if intent == "SUMMARY":
        text = payload.get("text")
        if not text:
            return {
                "response": "Please upload or provide a contract to summarize.",
                "data": None
            }
            
        data = generate_summary(text)
        return {
            "response": f"Here is the summary of your contract. Key points included.",
            "data": data
        }

    elif intent == "RISK":
        text = payload.get("text")
        if not text:
            return {
                "response": "Please upload or provide a contract for risk analysis.",
                "data": None
            }
            
        data = analyze_risk_from_text(text)
        level = data.get("risk_level", "Unknown")
        return {
            "response": f"Risk analysis complete. This contract has a {level}.",
            "data": data
        }

    elif intent == "COMPARE":
        text_a = payload.get("text_a")
        text_b = payload.get("text_b")
        if not text_a or not text_b:
            return {
                "response": "Two contract texts (text_a, text_b) are required for comparison.",
                "data": None
            }
            
        data = compare_contracts(text_a, text_b)
        summary = data.get("change_summary", "Comparison finished.")
        return {
            "response": f"Contract comparison complete. {summary}",
            "data": data
        }

    elif intent == "DRAFT":
        return {
            "response": "Drafting contracts is currently a beta feature. Please use a legally reviewed template.",
            "data": None
        }

    else:
        # Fallback to general LLM response
        session_id = payload.get("session_id")
        return generate_chat_response(message, session_id=session_id)
