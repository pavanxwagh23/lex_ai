"""
backend/api/chat_routes.py
==========================
API endpoints for the AI assistant chat interface.

Exposes a central `/chat` endpoint for processing natural language
requests and routing them to the correct backend service, along with
a `/chat/result/{task_id}` endpoint for polling async tasks.
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, Any

from fastapi import APIRouter, status, BackgroundTasks, Depends, Request

from backend.schemas.chat_schema import ChatRequest, ChatResponse
from backend.services.intent_detector import detect_intent
from backend.services.router import route_request
from backend.services.document_service import get_contract_text
from backend.services.session_manager import get_session, update_session
from backend.services.rate_limiter import rate_limit
from backend.services.task_manager import create_task, update_task, get_task
from backend.utils.error_handler import AppError
from backend.utils.logger import get_logger, log_event

logger = get_logger(__name__)

router = APIRouter(tags=["Chat Assistant"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(rate_limit)],
    summary="Process a chat request",
    description=(
        "Send a natural language message to the AI assistant. "
        "The system will detect the intent, extract necessary text from either "
        "an uploaded contract ID, session context, or the provided context, and "
        "route to the appropriate AI pipeline. Heavy tasks are sent to the background."
    ),
    status_code=status.HTTP_200_OK,
)
async def process_chat(
    chat_request: ChatRequest, 
    background_tasks: BackgroundTasks, 
    auth_req: Request
) -> ChatResponse:
    """Handle incoming messages to the central AI assistant.

    Parameters
    ----------
    chat_request : ChatRequest
        Pydantic model containing the user's message, optional contract ID,
        and optional context.
    background_tasks : BackgroundTasks
        FastAPI dependency for running heavy tasks natively.
    auth_req : Request
        FastAPI native request block used for dependency injection (Rate Limiting).
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    session_id = chat_request.session_id or str(uuid.uuid4())
    
    # Issue 5: Input Validation & Security
    msg = chat_request.message.strip()
    if not msg:
        raise AppError(code=400, message="Message cannot be empty.", status_code=400)
    if len(msg) > 2000:
        raise AppError(code=400, message="Message exceeds maximum length of 2000 characters.", status_code=400)

    logger.info("[%s] Received chat request: %r", request_id, msg[:50])

    try:
        # Step 1 — Detect the user's intent 
        intent, confidence = detect_intent(msg)

        # Step 2 — Construct payload and handle Session Memory
        payload: Dict[str, Any] = {}
        payload["session_id"] = session_id  # Passed downstream to services requiring context
        
        session = get_session(session_id)
        if "last_contract_text" in session:
            payload["text"] = session["last_contract_text"]

        if chat_request.extra_context:
            payload.update(chat_request.extra_context)

        text_provided = payload.get("text")
        
        # Validate extreme lengths (Issue 5 boundary)
        if text_provided and len(text_provided) > 50000:
            raise AppError(code=400, message="Contract text exceeds maximum length of 50000 characters.", status_code=400)

        # Flow: Document Context Resolution
        fetched_text = None
        if chat_request.contract_id:
            fetched_text = get_contract_text(chat_request.contract_id)
            payload["text"] = fetched_text
            update_session(session_id, {"last_contract_text": fetched_text})
        elif text_provided:
            update_session(session_id, {"last_contract_text": text_provided})
        elif intent in ("SUMMARY", "RISK", "COMPARE"): 
            raise AppError(code=400, message="Please provide contract text or contract_id", status_code=400)

        # Unified response metadata structure
        meta = {
            "confidence": confidence,
            "timestamp": int(start_time),
            "session_id": session_id,
            "request_id": request_id
        }

        # Step 3 — Route Request (Async Task Mapping — Issue 1)
        if intent in ("SUMMARY", "COMPARE"):
            logger.info("[%s] Offloading Computation-Heavy Intent %s to background task.", request_id, intent)
            task_id = create_task()
            meta["task_id"] = task_id
            
            def run_and_update(tid: str, i: str, m: str, p: dict):
                try:
                    res = route_request(intent=i, message=m, payload=p)
                    update_task(tid, result=res, status="completed")
                except Exception as e:
                    logger.error("Background task %s failed: %s", tid, e)
                    update_task(tid, status="failed", error=str(e))

            background_tasks.add_task(run_and_update, task_id, intent, msg, payload)
            
            log_event({
                "request_id": request_id,
                "session_id": session_id,
                "task_id": task_id,
                "intent": intent,
                "execution_time": time.time() - start_time,
                "module": "chat_routes",
                "status": "processing_async"
            })
            
            return ChatResponse(
                intent=intent,
                message="Processing request...",
                data=None,
                meta={**meta, "status": "processing"}
            )
            
        else:
            # Synchronous processing for lighter ML models and LLM translations
            result = route_request(intent=intent, message=msg, payload=payload)
            
            log_event({
                "request_id": request_id,
                "session_id": session_id,
                "intent": intent,
                "execution_time": time.time() - start_time,
                "module": "chat_routes",
                "status": "success"
            })

            return ChatResponse(
                intent=intent,
                message=result.get("response", "Processing complete."),
                data=result.get("data"),
                meta={**meta, "status": "completed"}
            )

    except AppError:
        # Validated exceptions bounce seamlessly to the global exception handler
        raise
    except Exception as exc:
        log_event({
            "request_id": request_id,
            "session_id": session_id,
            "intent": "ERROR",
            "execution_time": time.time() - start_time,
            "module": "chat_routes",
            "status": "error",
            "error_detail": str(exc)
        })
        logger.exception("[%s] Unexpected error in /chat endpoint: %s", request_id, exc)
        raise AppError(code=500, message="An unexpected error occurred while processing the request.", status_code=500)


@router.get(
    "/chat/result/{task_id}",
    summary="Retrieve background task result tracking metadata",
    status_code=status.HTTP_200_OK,
)
async def get_chat_result(task_id: str) -> Dict[str, Any]:
    """Poll for the result of a heavy ML background task.

    Parameters
    ----------
    task_id : str
        The background task tracker UUID.
    """
    task = get_task(task_id)
    if not task:
        raise AppError(code=404, message=f"Task {task_id} not found.", status_code=404)
        
    return task
