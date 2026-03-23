from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import torch

app = FastAPI(title="LexAI Ultra Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load model once at startup ───────────────────────────────────────────────
# Uses google/flan-t5-base — small, fast, runs on CPU, no API key needed
print("Loading AI model... (first run may take a minute to download)")
try:
    ai_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1,  # CPU
        max_new_tokens=512,
    )
    print("✅ AI model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    ai_pipeline = None


class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    result: str


@app.get("/")
def root():
    return {"status": "LexAI Ultra Chat API is running"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    if ai_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="AI model failed to load. Please restart the server.",
        )

    prompt = (
        "You are a professional legal expert. "
        "Answer the following legal question clearly and accurately.\n\n"
        f"Question: {request.text.strip()}\n\nAnswer:"
    )

    try:
        outputs = ai_pipeline(prompt, max_new_tokens=512, do_sample=False)
        result_text = outputs[0]["generated_text"].strip()
        if not result_text:
            raise HTTPException(status_code=500, detail="Model returned an empty response.")
        return AnalyzeResponse(result=result_text)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference error: {str(e)}")
