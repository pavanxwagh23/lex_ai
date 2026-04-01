# Fine-Tuning Pipeline — Lex AI Legal Model

This directory contains everything needed to create and serve **your own custom legal AI model**, fully trained on your data with zero dependency on OpenAI, Ollama, or any third-party service.

---

## Architecture

```
fine_tuning/
├── generate_dataset.py   # Step 1: Generate legal Q&A training data
├── finetune.py           # Step 2: Fine-tune phi-2 on your legal data
├── serve.py              # Step 3: Serve your model as an OpenAI-compatible API
└── data/
    └── legal_qa_dataset.jsonl   # Auto-generated training data
```

The output model is saved to `models/lex_ai_custom_llm/` and is fully yours.

---

## Step-by-Step Guide

### Prerequisites
Install the required packages:
```bash
pip install transformers peft datasets accelerate bitsandbytes torch
```

### Step 1: Generate the Training Dataset
```bash
python fine_tuning/generate_dataset.py
```
This produces 54+ labeled legal Q&A pairs in `fine_tuning/data/legal_qa_dataset.jsonl`.

### Step 2: Fine-Tune the Model
```bash
python fine_tuning/finetune.py
```

**Training time estimate:**
| Hardware | Estimated Time |
|---|---|
| CPU Only | 2–4 hours (not recommended) |
| NVIDIA GPU (8GB VRAM) | 15–30 minutes |
| NVIDIA GPU (16GB+ VRAM) | 8–15 minutes |

The model will be saved to `models/lex_ai_custom_llm/`.

### Step 3: Serve Your Model
```bash
python fine_tuning/serve.py
```
Your model will start serving on `http://localhost:11435/v1` using an OpenAI-compatible API.

### Step 4: Connect Lex AI to Your Model
Update your `.env` file:
```env
USE_LOCAL_LLM=true
LOCAL_LLM_URL=http://localhost:11435/v1
LOCAL_LLM_MODEL=lex-ai-legal
```
Then restart the backend:
```bash
python -m uvicorn backend.main:app --env-file .env --port 8000
```

---

## How It Works

The fine-tuning process uses **LoRA (Low-Rank Adaptation)** — an extremely efficient technique that:
- Freezes the original `phi-2` model weights (no expensive full retraining).
- Trains only a tiny set of "adapter" layers (0.1% of total parameters).
- Merges the adapters back onto the base model after training.

The result is a model that has **your legal knowledge built in** but required only a fraction of the compute of training from scratch.

---

## Expanding the Dataset
You can add your own Q&A pairs to `fine_tuning/generate_dataset.py` by adding entries to the `_SEED_QA` list:
```python
{"prompt": "What is a limitation of liability?",
 "completion": "A limitation of liability clause caps the maximum..."},
```
Then re-run `generate_dataset.py` and `finetune.py` to retrain.
