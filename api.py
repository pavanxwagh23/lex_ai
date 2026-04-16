import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import logging
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Setup logging early so logger is available before optional imports
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the ai_engine directory to the Python path
sys.path.append(str(Path(__file__).parent / "ai_engine"))

from clause_detector_pro import analyze_contract
from pdf_extractor import process_document, extract_text_with_ocr

# Transformer-based clause classifier (fine-tuned Legal-BERT)
try:
    from ai_engine.clause_classifier.predict import predict_clause, predict_batch_clauses
    _CLASSIFIER_AVAILABLE = True
except Exception as _clf_import_err:  # noqa: BLE001
    logger.warning(
        "Clause classifier could not be loaded: %s. "
        "Train the model first with: python ai_engine/clause_classifier/train.py",
        _clf_import_err,
    )
    _CLASSIFIER_AVAILABLE = False

# Semantic contract comparison engine (sentence-transformers + FAISS)
try:
    from ai_engine.contract_comparator import compare_contracts
    _COMPARATOR_AVAILABLE = True
except Exception as _cmp_import_err:  # noqa: BLE001
    _COMPARATOR_AVAILABLE = False

# Rule-based risk detection engine
try:
    from risk_detection import RiskDetectionEngine as _RiskEngine
    _risk_engine = _RiskEngine()
    _RISK_AVAILABLE = True
except Exception as _risk_import_err:  # noqa: BLE001
    logger.warning("Risk detection engine could not be loaded: %s", _risk_import_err)
    _RISK_AVAILABLE = False

# Legal document summarizer (requires transformers + torch)
try:
    from summarizer import LegalSummarizer as _LegalSummarizer
    _SUMMARIZER_AVAILABLE = True
except Exception as _sum_import_err:  # noqa: BLE001
    logger.warning(
        "Summarizer could not be loaded: %s. "
        "Install with: pip install transformers torch",
        _sum_import_err,
    )
    _SUMMARIZER_AVAILABLE = False


app = FastAPI(title="Legal AI Analyzer API")

# Setup CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextAnalysisRequest(BaseModel):
    text: str
    use_preprocessing: bool = True
    use_hybrid: bool = False


class ClassifyRequest(BaseModel):
    """Request model for single-paragraph transformer classification."""
    paragraph: str


class ClassifyBatchRequest(BaseModel):
    """Request model for batch transformer classification."""
    paragraphs: List[str]


class RiskAnalysisRequest(BaseModel):
    """Request model for the /analyze_risks endpoint."""
    paragraphs: List[str]


class SummarizeRequest(BaseModel):
    """Request model for the /summarize endpoint."""
    text: Optional[str] = None
    paragraphs: Optional[List[str]] = None

@app.get("/")
async def root():
    return {"message": "Legal AI Analyzer API is running"}

@app.post("/analyze_text")
async def analyze_text_endpoint(request: TextAnalysisRequest):
    """
    Analyze raw text for legal clauses.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    paragraphs = [p.strip() for p in request.text.split("\n\n") if p.strip()]
    
    if not paragraphs:
        # Fallback if double newline split didn't yield results
        paragraphs = [p.strip() for p in request.text.split("\n") if p.strip()]
        
    try:
        results = analyze_contract(
            paragraphs,
            use_preprocessing=request.use_preprocessing,
            use_hybrid=request.use_hybrid
        )
        return results
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze_pdf")
async def analyze_pdf_endpoint(
    file: UploadFile = File(...),
    use_ocr: bool = Form(False),
    use_preprocessing: bool = Form(True),
    use_hybrid: bool = Form(False)
):
    """
    Upload a PDF, extract text, and analyze legal clauses.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
        
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        content = await file.read()
        temp_pdf.write(content)
        temp_pdf_path = temp_pdf.name
        
    try:
        # Step 1: Extract Text
        if use_ocr:
            # OCR path: use extract_text_with_ocr for scanned PDFs
            full_text = extract_text_with_ocr(temp_pdf_path)
            num_pages = 0  # page count not available separately for OCR path
        else:
            extracted_data = process_document(temp_pdf_path)
            # process_document returns: {raw_text, clean_text, paragraphs, num_pages, source_type}
            full_text = extracted_data["clean_text"]
            num_pages = extracted_data.get("num_pages", 0)
            
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF. Try enabling OCR.")
            
        # Segment into paragraphs
        paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
        if len(paragraphs) < 5:
            # If too few paragraphs, try single newlines
            paragraphs = [p.strip() for p in full_text.split("\n") if p.strip()]
             
        # Step 2: Analyze Clauses
        analysis_results = analyze_contract(
            paragraphs,
            use_preprocessing=use_preprocessing,
            use_hybrid=use_hybrid
        )
        
        return {
            "filename": file.filename,
            "extraction_info": {
                "total_pages": num_pages,
                "used_ocr": use_ocr
            },
            "analysis": analysis_results
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")
    finally:
        # Cleanup
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

# ---------------------------------------------------------------------------
# Transformer-based Clause Classification endpoints
# ---------------------------------------------------------------------------

@app.post("/classify_clause")
async def classify_clause_endpoint(request: ClassifyRequest):
    """
    Classify a **single** legal paragraph using the fine-tuned Legal-BERT model.

    Request body
    ------------
    - ``paragraph`` : raw legal paragraph string

    Response
    --------
    - ``clause_type``  : predicted label (e.g. ``"termination"``)
    - ``confidence``   : softmax confidence score in [0, 1]

    Example
    -------
    ::

        POST /classify_clause
        {"paragraph": "Either party may terminate with 30 days notice."}

        → {"clause_type": "termination", "confidence": 0.9312}
    """
    if not _CLASSIFIER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Clause classifier model is not available. "
                   "Train it first: python ai_engine/clause_classifier/train.py",
        )

    if not request.paragraph or not request.paragraph.strip():
        raise HTTPException(status_code=400, detail="'paragraph' field cannot be empty.")

    try:
        result = predict_clause(request.paragraph)
        return result
    except Exception as exc:
        logger.error("Clause classification error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Classification failed: {exc}")


@app.post("/classify_clauses")
async def classify_clauses_endpoint(request: ClassifyBatchRequest):
    """
    Classify **multiple** legal paragraphs in a single request.

    Ideal for processing a fully-split contract document returned by the
    ``/analyze_pdf`` pipeline.

    Request body
    ------------
    - ``paragraphs`` : list of raw legal paragraph strings

    Response
    --------
    List of objects, one per input paragraph, each with:
    - ``clause_type``  : predicted label
    - ``confidence``   : softmax confidence score

    Example
    -------
    ::

        POST /classify_clauses
        {
          "paragraphs": [
            "Either party may terminate with 30 days notice.",
            "All invoices must be paid within 30 days."
          ]
        }

        → [
            {"clause_type": "termination",   "confidence": 0.9312},
            {"clause_type": "payment_terms", "confidence": 0.8921}
          ]
    """
    if not _CLASSIFIER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Clause classifier model is not available. "
                   "Train it first: python ai_engine/clause_classifier/train.py",
        )

    paragraphs = [p.strip() for p in request.paragraphs if p and p.strip()]
    if not paragraphs:
        raise HTTPException(status_code=400, detail="'paragraphs' list cannot be empty.")

    try:
        results = predict_batch_clauses(paragraphs)
        return results
    except Exception as exc:
        logger.error("Batch clause classification error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {exc}")


# ---------------------------------------------------------------------------
# Contract Comparison endpoint
# ---------------------------------------------------------------------------

class CompareContractsRequest(BaseModel):
    """
    Request body for the /compare_contracts endpoint.

    Fields
    ------
    doc_a_clauses : list[str]
        Clauses / paragraphs from the **original** contract.
    doc_b_clauses : list[str]
        Clauses / paragraphs from the **revised** contract.
    model_name : str
        Sentence-transformer model to use for embedding.
        Defaults to ``sentence-transformers/all-MiniLM-L6-v2``.
    """
    doc_a_clauses: List[str]
    doc_b_clauses: List[str]
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


@app.post("/compare_contracts")
async def compare_contracts_endpoint(request: CompareContractsRequest):
    """
    Semantically compare two lists of legal clauses and return a structured diff.

    Detects clauses that were **added**, **removed**, **modified**, or remain
    **semantically identical** between Document A and Document B — even when
    wording changes but meaning stays the same.

    Request body
    ------------
    - ``doc_a_clauses`` : list[str] — clauses from the original contract
    - ``doc_b_clauses`` : list[str] — clauses from the revised contract
    - ``model_name``    : str (optional) — embedding model to use

    Response
    --------
    ::

        {
          "added":     ["The vendor must comply with GDPR regulations."],
          "removed":   [],
          "modified":  [{"clause_a": "...30 days...", "clause_b": "...60 days...", ...}],
          "related":   [...],
          "identical": [{"clause_a": "...", "clause_b": "...", ...}],
          "metadata":  {"doc_a_clauses": 3, "doc_b_clauses": 4, ...}
        }
    """
    if not _COMPARATOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Contract comparator is unavailable. "
                   "Install dependencies: pip install sentence-transformers faiss-cpu",
        )

    doc_a = [c.strip() for c in request.doc_a_clauses if c and c.strip()]
    doc_b = [c.strip() for c in request.doc_b_clauses if c and c.strip()]

    if not doc_a:
        raise HTTPException(status_code=400, detail="'doc_a_clauses' cannot be empty.")
    if not doc_b:
        raise HTTPException(status_code=400, detail="'doc_b_clauses' cannot be empty.")

    try:
        result = compare_contracts(doc_a, doc_b, model_name=request.model_name)
        return result
    except Exception as exc:
        logger.error("Contract comparison error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {exc}")


# ---------------------------------------------------------------------------
# Risk Analysis endpoint
# ---------------------------------------------------------------------------

@app.post("/analyze_risks")
async def analyze_risks_endpoint(request: RiskAnalysisRequest):
    """
    Scan contract paragraphs for legal risks using the rule-based engine.

    Detects 8 risk categories (Unlimited Liability, One-Sided Indemnity,
    IP Ownership Risk, Payment Risk, Jurisdiction Risk, etc.) and returns
    a normalised overall risk score (0–10) plus per-finding details.

    Request body
    ------------
    - ``paragraphs`` : list[str] — raw contract paragraphs

    Response
    --------
    ::

        {
          "risk_score": 6.67,              # 0–10
          "total_paragraphs": 5,
          "risky_paragraph_count": 3,
          "risks_detected": [
            {
              "type": "One-sided Indemnity",
              "severity": "HIGH",
              "text": "The vendor shall indemnify...",
              "explanation": "...",
              "matched_phrase": "shall indemnify",
              "paragraph_index": 2
            }
          ]
        }
    """
    if not _RISK_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Risk detection engine is not available.",
        )

    paragraphs = [p.strip() for p in request.paragraphs if p and p.strip()]
    if not paragraphs:
        raise HTTPException(status_code=400, detail="'paragraphs' list cannot be empty.")

    try:
        result = _risk_engine.analyze(paragraphs)
        return result.to_dict()
    except Exception as exc:
        logger.error("Risk analysis error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Risk analysis failed: {exc}")


# ---------------------------------------------------------------------------
# Summarization endpoint
# ---------------------------------------------------------------------------

@app.post("/summarize")
async def summarize_endpoint(request: SummarizeRequest):
    """
    Generate a structured legal summary of a contract using BART-large-CNN.

    Splits long contracts into overlapping chunks, summarises each chunk,
    then consolidates into a final report with labelled sections.

    Request body
    ------------
    - ``text``       : str (optional) — raw contract text
    - ``paragraphs`` : list[str] (optional) — pre-split paragraphs
    Provide at least one of the two.

    Response
    --------
    ::

        {
          "overall_summary": "This agreement establishes...",
          "key_points": ["...", "..."],
          "obligations": ["..."],
          "payment_terms": ["..."],
          "termination": ["..."],
          "risks": ["..."]
        }

    Note
    ----
    First call downloads the ``facebook/bart-large-cnn`` model (~1.6 GB).
    Subsequent calls use the local HuggingFace cache.
    """
    if not _SUMMARIZER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "Summarizer is not available. "
                "Install dependencies: pip install transformers torch"
            ),
        )

    if not request.text and not request.paragraphs:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'text' or 'paragraphs' in the request body.",
        )

    try:
        summarizer = _LegalSummarizer()
        if request.paragraphs:
            result = summarizer.generate_summary("", paragraphs=request.paragraphs)
        else:
            result = summarizer.generate_summary(request.text)
        return result.to_dict()
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Summarizer dependencies missing: {exc}. Install: pip install transformers torch",
        )
    except Exception as exc:
        logger.error("Summarization error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Summarization failed: {exc}")



# ---------------------------------------------------------------------------
# Chat endpoint — routes messages through the fine-tuned Lex AI legal model
# ---------------------------------------------------------------------------

import uuid as _uuid
import re as _re

# In-memory session store: {session_id: {"messages": [...], "context": str}}
_sessions: Dict[str, Dict] = {}
_tasks: Dict[str, Any] = {}

LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:11435/v1")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "lex-ai-legal")
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"

import httpx as _httpx

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    contract_id: Optional[str] = None
    extra_context: Optional[Dict[str, Any]] = {}


def _detect_intent(message: str) -> str:
    msg = message.lower()
    if any(w in msg for w in ["risk", "danger", "flag", "problematic", "liability", "dangerous", "issue"]):
        return "RISK"
    if any(w in msg for w in ["summarize", "summary", "summarise", "key points", "overview", "tldr", "brief"]):
        return "SUMMARY"
    if any(w in msg for w in ["clause", "classify", "identify clause", "clause type"]):
        return "CLAUSE_MAP"
    return "GENERAL"


def _build_system_prompt(intent: str, context: str) -> str:
    base = (
        "You are Lex AI, an expert legal assistant specializing in contract analysis, "
        "risk identification, and legal document summarization. Answer clearly and professionally."
    )
    if context:
        base += f"\n\nThe user has provided the following contract text for analysis:\n\n---\n{context[:3000]}\n---"
    if intent == "RISK":
        base += "\n\nFocus on identifying legal risks, liability issues, and problematic clauses."
    elif intent == "SUMMARY":
        base += "\n\nFocus on providing a concise, structured summary with key points."
    return base


async def _process_chat_task(task_id: str, session_id: str, llm_messages: list, intent: str):
    """Background task to fetch LLM response and update task status."""
    try:
        async with _httpx.AsyncClient(timeout=None) as client:
            llm_resp = await client.post(
                f"{LOCAL_LLM_URL}/chat/completions",
                json={
                    "model": LOCAL_LLM_MODEL,
                    "messages": llm_messages,
                    "max_tokens": 512,
                    "temperature": 0.7,
                },
            )
            if llm_resp.status_code == 200:
                llm_data = llm_resp.json()
                reply = llm_data["choices"][0]["message"]["content"].strip()
                result = {
                    "message": reply,
                    "intent": intent,
                    "data": {},
                    "meta": {
                        "status": "completed",
                        "intent": intent,
                        "confidence": 1.0,
                        "task_id": task_id
                    }
                }
                _sessions[session_id]["messages"].append({"role": "assistant", "content": reply})
                _tasks[task_id] = {"status": "completed", "result": result}
            else:
                _tasks[task_id] = {"status": "failed", "error": f"LLM API returned {llm_resp.status_code}"}
    except Exception as exc:
        logger.warning("Background LLM call failed: %s. Falling back to mock response.", exc)
        reply = "I am operating in offline mock mode since the local LLM server is unavailable. You can upload a contract or paste text in the sidebar, and I will analyze it using my local ML models for risks, summaries, and clause mapping!"
        result = {
            "message": reply,
            "intent": intent,
            "data": {},
            "meta": {
                "status": "completed",
                "intent": intent,
                "confidence": 1.0,
                "task_id": task_id
            }
        }
        _sessions[session_id]["messages"].append({"role": "assistant", "content": reply})
        _tasks[task_id] = {"status": "completed", "result": result}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Main conversational chat endpoint.
    Routes through the fine-tuned Lex AI legal model running on port 11435 via background polling.
    """
    session_id = request.session_id or str(_uuid.uuid4())

    # Init or retrieve session
    if session_id not in _sessions:
        _sessions[session_id] = {"messages": [], "context": ""}

    session = _sessions[session_id]

    # Absorb any new context from this request
    extra = request.extra_context or {}
    if "text" in extra and extra["text"]:
        session["context"] = extra["text"]

    intent = _detect_intent(request.message)
    system_prompt = _build_system_prompt(intent, session["context"])

    # Build messages list for the LLM
    llm_messages = [{"role": "system", "content": system_prompt}]
    # Include last 6 messages of history for context window efficiency
    for m in session["messages"][-6:]:
        llm_messages.append({"role": m["role"], "content": m["content"]})
    llm_messages.append({"role": "user", "content": request.message})

    # Save user message to session history
    session["messages"].append({"role": "user", "content": request.message})

    task_id = str(_uuid.uuid4())
    _tasks[task_id] = {"status": "processing"}
    background_tasks.add_task(_process_chat_task, task_id, session_id, llm_messages, intent)

    return {
        "message": "Processing your request...",
        "intent": intent,
        "data": {},
        "meta": {
            "status": "processing",
            "intent": intent,
            "confidence": 1.0,
            "task_id": task_id
        }
    }

@app.get("/chat/result/{task_id}")
async def get_chat_result(task_id: str):
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return _tasks[task_id]


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
