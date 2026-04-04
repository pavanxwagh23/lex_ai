"""
fine_tuning/serve.py
=====================
Serves your fine-tuned Lex AI legal model via an OpenAI-compatible
REST API on port 11435.

This makes your custom model a drop-in replacement for Ollama!
To activate it in Lex AI, update your .env:

    USE_LOCAL_LLM=true
    LOCAL_LLM_URL=http://localhost:11435/v1
    LOCAL_LLM_MODEL=lex-ai-legal

Requirements:
    pip install fastapi uvicorn transformers peft torch

Usage:
    python fine_tuning/serve.py
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "lex_ai_custom_llm"

sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Pydantic models — must be at module level for FastAPI to validate correctly
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "lex-ai-legal"
    messages: List[Message]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7


class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"


class ChatUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: ChatUsage


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """Load the fine-tuned model from disk."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch

    base_model_name = "microsoft/Phi-3-mini-4k-instruct"

    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        trust_remote_code=False,
        attn_implementation="eager",  # Required to avoid DynamicCache incompatibility
    )
    model = PeftModel.from_pretrained(base_model, str(MODEL_DIR))
    model.eval()

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"✅ Lex AI legal model loaded on {device}")
    return tokenizer, model


def generate_response(tokenizer, model, messages: List[Message], max_new_tokens: int = 256) -> str:
    """Generate a response from the fine-tuned model."""
    import torch

    # Build a clean prompt from all messages
    prompt_parts = []
    for msg in messages:
        role = msg.role.lower()
        if role == "system":
            prompt_parts.append(f"### System:\n{msg.content}")
        elif role == "user":
            prompt_parts.append(f"### Instruction:\n{msg.content}")
        elif role == "assistant":
            prompt_parts.append(f"### Response:\n{msg.content}")

    # The model was trained with this exact format
    formatted = "\n\n".join(prompt_parts) + "\n\n### Response:\n"

    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return reply if reply else "I'm unable to generate a response at this time."


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(tokenizer, model):
    """Build FastAPI OpenAI-compatible server."""

    app = FastAPI(title="Lex AI Custom Legal Model Server", version="1.0.0")

    @app.get("/")
    def root():
        return {"status": "Lex AI Custom Legal Model is running!", "model": "lex-ai-legal"}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": "lex-ai-legal", "object": "model", "owned_by": "lex-ai"}]
        }

    @app.post("/v1/chat/completions", response_model=ChatResponse)
    def chat_completions(request: ChatRequest):
        response_text = generate_response(
            tokenizer, model, request.messages,
            max_new_tokens=request.max_tokens or 256
        )
        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[ChatChoice(
                index=0,
                message=Message(role="assistant", content=response_text)
            )],
            usage=ChatUsage()
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not MODEL_DIR.exists():
        print(f"❌ Model not found at {MODEL_DIR}")
        print("   Run: python fine_tuning/finetune.py  to train your model first.")
        sys.exit(1)

    import uvicorn

    print("=" * 60)
    print("  Lex AI — Custom Legal Model Server")
    print(f"  Model: {MODEL_DIR}")
    print("  API  : http://localhost:11435/v1")
    print("=" * 60)

    tokenizer, model = load_model()
    app = create_app(tokenizer, model)
    uvicorn.run(app, host="0.0.0.0", port=11435)


if __name__ == "__main__":
    main()
