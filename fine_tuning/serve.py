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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "lex_ai_custom_llm"

sys.path.insert(0, str(PROJECT_ROOT))


def load_model():
    """Load the fine-tuned model from disk."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch

    base_model_name = "microsoft/phi-2"

    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(MODEL_DIR))
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"✅ Lex AI legal model loaded on {device}")
    return tokenizer, model


def generate_response(tokenizer, model, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a response from the fine-tuned model."""
    import torch

    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    if "### Response:" in full_output:
        return full_output.split("### Response:")[-1].strip()
    return full_output.strip()


def create_app(tokenizer, model):
    """Build FastAPI OpenAI-compatible server."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="Lex AI Custom Legal Model Server", version="1.0.0")

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
        # Combine all messages into a single prompt
        prompt = "\n".join(
            f"{msg.role.capitalize()}: {msg.content}"
            for msg in request.messages
        )
        response_text = generate_response(
            tokenizer, model, prompt,
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
