import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import logging
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Add the ai_engine directory to the Python path
sys.path.append(str(Path(__file__).parent / "ai_engine"))

from clause_detector_pro import analyze_contract
from pdf_extractor import process_pdf, process_pdf_with_ocr

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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            extracted_data = process_pdf_with_ocr(temp_pdf_path)
            # Extracted data structure: {pages: [{page_num, text}], full_text}
            full_text = extracted_data["full_text"]
        else:
            extracted_data = process_pdf(temp_pdf_path)
            full_text = extracted_data["full_text"]
            
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF. Try enabling OCR.")
            
        # Segment into paragraphs (basic split for now, robust extraction could be better)
         # Try split by double newline first, then single
        paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
        if len(paragraphs) < 5: 
             # If too few paragraphs, maybe it's just single newlines
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
                "total_pages": len(extracted_data.get("pages", [])),
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


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
