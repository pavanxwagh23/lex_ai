from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json

app = FastAPI(title="LexAI Ultra Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"


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
        "You are a professional legal expert named LexAI. "
        "Give clear, simple, structured and correct legal answers. "
        "Be concise but thorough.\n\n"
        f"User: {request.text.strip()}\n\nLexAI:"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        result_text = data.get("response", "").strip()
        if not result_text:
            raise HTTPException(status_code=500, detail="Ollama returned an empty response.")
        return AnalyzeResponse(result=result_text)

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Please start Ollama with: ollama run llama3",
        )
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Ollama timed out. Please try again.")
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Ollama HTTP error: {str(e)}")
    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Ollama response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
