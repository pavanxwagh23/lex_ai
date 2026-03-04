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

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
