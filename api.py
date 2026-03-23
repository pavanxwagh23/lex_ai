from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

app = FastAPI(title="LexAI Ultra Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Uses HuggingFace Inference API (free, no API key for public models) ──────
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HF_HEADERS = {"Content-Type": "application/json"}


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

    prompt = (
        "You are a professional legal expert. "
        "Answer the following legal question clearly and accurately.\n\n"
        f"Question: {request.text.strip()}\n\nAnswer:"
    )

    try:
        response = requests.post(
            HF_API_URL,
            headers=HF_HEADERS,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 300}},
            timeout=60,
        )

        if response.status_code == 503:
            # Model is loading on HuggingFace servers
            raise HTTPException(
                status_code=503,
                detail="AI model is warming up on HuggingFace servers. Please try again in 20 seconds.",
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"HuggingFace API error: {response.text}",
            )

        data = response.json()

        # flan-t5-base returns list of dicts: [{"generated_text": "..."}]
        if isinstance(data, list) and len(data) > 0:
            result_text = data[0].get("generated_text", "").strip()
        elif isinstance(data, dict):
            result_text = data.get("generated_text", "").strip()
        else:
            result_text = ""

        if not result_text:
            raise HTTPException(status_code=500, detail="AI returned an empty response. Please try again.")

        return AnalyzeResponse(result=result_text)

    except HTTPException:
        raise
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="No internet connection. Cannot reach HuggingFace API.")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request timed out. Please try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
