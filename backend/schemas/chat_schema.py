"""
backend/schemas/chat_schema.py
==============================
Pydantic models for the ``POST /chat`` conversational AI endpoint.

These schemas define the request and response contract for the central
AI assistant layer that routes user messages to the appropriate internal
service (summarization, risk analysis, contract comparison, or LLM chat).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat message from the user.

    Attributes
    ----------
    message : str
        Free-form natural language message (e.g. "Summarize this contract").
    contract_id : str, optional
        UUID of a previously uploaded contract.  When provided, the assistant
        can pull the stored document for analysis without re-uploading.
    extra_context : dict, optional
        Arbitrary key-value context the frontend may pass (e.g. selected
        paragraphs, a second contract for comparison, raw text, etc.).
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Natural language message from the user.",
        examples=["Summarize this contract", "What are the risks?"],
    )
    contract_id: Optional[str] = Field(
        default=None,
        description="UUID of an uploaded contract (optional).",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="UUID of the chat session to track context across messages.",
    )
    extra_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context (paragraphs, second contract text, etc.).",
    )


class ResponseMeta(BaseModel):
    """Enforce strict types for operational tracing hooks."""
    confidence: float = Field(..., description="Intent detection confidence score.")
    timestamp: int = Field(..., description="Unix timestamp at execution.")
    status: str = Field(..., description="E.g., processing, completed, failed.")
    session_id: Optional[str] = Field(default=None, description="Conversation session UUID.")
    request_id: Optional[str] = Field(default=None, description="Observability trace ID.")
    task_id: Optional[str] = Field(default=None, description="Background task UUID.")
    error_code: Optional[int] = Field(default=None, description="Error code if failed.")


class ChatResponse(BaseModel):
    """Structured response returned by the assistant.

    Attributes
    ----------
    intent : str
        Detected intent label (SUMMARY, RISK, COMPARE, DRAFT, GENERAL, ERROR).
    message : str
        Human-readable response message.
    data : dict, optional
        Structured data payload from the underlying service.
    meta : ResponseMeta
        Metadata including confidence score and timestamp.
    """

    intent: str = Field(
        ...,
        description="Detected intent (SUMMARY, RISK, COMPARE, DRAFT, GENERAL).",
    )
    message: str = Field(
        ...,
        description="Human-readable response from the assistant.",
    )
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured output from the called service.",
    )
    meta: ResponseMeta = Field(
        ...,
        description="Metadata including observability tags constraints and NLP confidence.",
    )
