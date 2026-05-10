# ⚖️ Lex AI — Legal Document Analyzer

An end-to-end AI-powered platform for analyzing legal contracts. Lex AI automatically detects clause types, flags legal risks, extracts text from PDFs and DOCX files, compares contract versions semantically, and summarizes documents through a browser SaaS dashboard backed by a modular FastAPI service.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Clause Detection** | Identifies legal clause types (termination, payment, confidentiality, indemnity, governing law, etc.) with confidence scoring |
| **Risk Detection** | Rule-based engine that flags high/medium/low-risk passages (unlimited liability, one-sided termination, missing IP clauses, etc.) |
| **OCR Pipeline** | Extracts text from both normal and scanned/image-based PDFs |
| **Legal-BERT Classifier** | Fine-tuned transformer model for clause classification (trainable on your own dataset) |
| **Contract Comparator** | Semantically compares two contract versions using sentence-transformers + FAISS, identifying added, removed, modified, and identical clauses |
| **Summarizer** | Generates concise summaries of long legal documents |
| **Hybrid Mode** | Combines fast rule-based classification with ML fallback for low-confidence cases |
| **spaCy Preprocessing** | Optional lemmatization and text normalization for improved accuracy |

---

## 🏗️ Project Structure

```
lex_ai/
│
├── app.py                        # Deprecated Streamlit prototype UI
├── api.py                        # Deprecated single-file FastAPI prototype
│
├── ai_engine/                    # Core AI/NLP engine
│   ├── clause_detector.py        # Base rule-based clause detector
│   ├── clause_detector_pro.py    # Enhanced detector (spaCy, heading detection, hybrid)
│   ├── contract_comparator.py    # Semantic contract diff (sentence-transformers + FAISS)
│   ├── legal_text_preprocessor.py# Text cleaning and normalization
│   ├── ocr_pipeline.py           # OCR support for scanned PDFs
│   ├── paragraph_splitter.py     # Smart contract paragraph segmentation
│   ├── pdf_extractor.py          # PDF text extraction
│   ├── summarizer.py             # Legal document summarizer
│   │
│   ├── clause_classifier/        # Fine-tunable Legal-BERT classifier
│   │   ├── train.py              # Training pipeline (Hugging Face Trainer API)
│   │   ├── predict.py            # Inference (single + batch)
│   │   ├── dataset_loader.py     # Dataset loading & tokenization
│   │   ├── config.py             # Model hyperparameters & label map
│   │   └── utils.py              # Metrics, device resolution, reporting
│   │
│   └── risk_detection/           # Risk analysis module
│       ├── risk_engine.py        # Main RiskDetectionEngine class
│       ├── risk_rules.py         # Rule definitions (keywords, categories)
│       ├── risk_types.py         # Data types (RiskMatch, RiskSeverity, etc.)
│       └── risk_utils.py         # Helpers (keyword matching, scoring, dedup)
│
├── backend/                      # Service layer
│   ├── main.py                   # Backend app entry point
│   ├── config.py                 # Configuration settings
│   ├── dependencies.py           # Dependency injection
│   ├── schemas/                  # Pydantic request/response schemas
│   ├── services/                 # Business logic services
│   │   ├── analysis_service.py
│   │   ├── comparison_service.py
│   │   └── document_service.py
│   ├── api/                      # Additional route handlers
│   └── utils/                    # Backend utilities
│
└── data/
    └── clause_dataset.csv        # Training data for the clause classifier
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9+
- pip

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

For the full ML/OCR feature set, install the optional ML profile too:

```bash
pip install -r requirements.txt -r requirements-ml.txt
```

For development and tests:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

Install the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

> **Note:** `faiss-cpu` and `sentence-transformers` are only required for the contract comparison feature. `pytesseract` requires the separate [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) system executable for scanned PDF support. If it is not on `PATH`, set `TESSERACT_CMD` in `.env`.

### 3. Run the API Server

```bash
uvicorn backend.main:app --reload --port 8000
```

The FastAPI server starts at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive Swagger UI.

`backend.main:app` is the supported backend entry point. `api.py` is deprecated and kept only as a migration reference.

### 4. Open the SaaS Dashboard

The modular backend serves the browser frontend directly:

`http://localhost:8000/app`

The old Streamlit prototype in `app.py` is deprecated. Do not add new product features there.

### 5. Run Celery Workers for Heavy AI Jobs

Analysis, summary, and comparison routes enqueue Celery tasks. For those routes,
start Redis and then run a worker in a separate terminal:

```bash
celery -A backend.worker.celery_app worker --loglevel=info
```

---

## 🔌 Modular API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/health` | Health probe |
| `POST` | `/contracts/upload` | Upload a PDF or DOCX contract |
| `GET` | `/contracts` | List uploaded contracts in the current registry |
| `POST` | `/contracts/{contract_id}/analyze` | Enqueue full analysis task |
| `POST` | `/contracts/{contract_id}/summary` | Enqueue summary task |
| `POST` | `/contracts/compare` | Enqueue semantic comparison task |
| `GET` | `/tasks/{task_id}` | Poll task status/result |
| `POST` | `/chat` | Chat interface used by the SaaS dashboard |

Legacy endpoints in `api.py` such as `/analyze_text`, `/analyze_pdf`, and
`/classify_clause` are deprecated. Migrate any missing behavior into `backend/`
instead of extending `api.py`.

### Example: Analyze Text

```bash
curl -X POST http://localhost:8000/analyze_text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Either party may terminate this Agreement with 30 days written notice.\n\nClient shall pay all invoices within 30 days.",
    "use_preprocessing": true,
    "use_hybrid": false
  }'
```

### Example: Compare Contracts

```bash
curl -X POST http://localhost:8000/compare_contracts \
  -H "Content-Type: application/json" \
  -d '{
    "doc_a_clauses": ["Either party may terminate with 30 days notice."],
    "doc_b_clauses": ["Either party may terminate with 60 days written notice."]
  }'
```

---

## 🧠 Training the Clause Classifier

The Legal-BERT classifier can be fine-tuned on your own labeled data.

### 1. Prepare the Dataset

Place a CSV file at `data/clause_dataset.csv` with at least two columns:

| text | label |
|---|---|
| `Either party may terminate with 30 days notice.` | `termination` |
| `All invoices must be paid within 30 days.` | `payment_terms` |

### 2. Run Training

```bash
python ai_engine/clause_classifier/train.py
```

The script will:
1. Resolve the best available model (`nlpaueb/legal-bert-base-uncased` → `bert-base-uncased` fallback)
2. Tokenize and split data 80/20 train/validation
3. Fine-tune with the Hugging Face `Trainer` API (with early stopping)
4. Print a full per-class classification report
5. Save the model to `models/clause_classifier_model/`

Once trained, the `/classify_clause` and `/classify_clauses` API endpoints become active automatically.

---

## 🛡️ Risk Detection

The `RiskDetectionEngine` scans contract paragraphs against a rule library and returns a normalized risk score (0–10) along with detailed findings.

**Risk categories include:**
- Unlimited liability clauses
- One-sided termination rights
- Missing indemnification limits
- Ambiguous IP ownership
- Non-compete / non-solicitation overreach
- Auto-renewal traps

**Severity levels:** `LOW` · `MEDIUM` · `HIGH`

```python
from ai_engine.risk_detection import RiskDetectionEngine

engine = RiskDetectionEngine()
result = engine.analyze([
    "Company shall have unlimited liability for any damages caused.",
    "Either party may terminate the agreement with 30 days notice.",
])

print(result.risk_score)       # e.g. 3.33
print(result.risks_detected)   # List of RiskMatch objects
```

---

## 🖥️ UI Overview

The Streamlit frontend (`app.py`) provides:

- **Input modes:** Upload PDF or paste raw text
- **Sidebar options:** Toggle spaCy preprocessing and Hybrid classifier mode
- **OCR toggle:** Enable for scanned/image-based PDFs
- **Results dashboard:**
  - Metrics: total paragraphs analyzed, average confidence, clause types found
  - Tabbed view: "All Extracted" + individual tabs per clause type
  - Color-coded confidence: 🟢 High (≥0.75) · 🟡 Medium (≥0.50) · 🔴 Low (<0.50)
  - Raw JSON output expander for developers

---

## ⚙️ Configuration

Key settings in `backend/config.py` and `ai_engine/clause_classifier/config.py`:

| Setting | Default | Description |
|---|---|---|
| `API_URL` (app.py) | `http://localhost:8000` | FastAPI server URL |
| `PRIMARY_MODEL_NAME` | `nlpaueb/legal-bert-base-uncased` | Base model for fine-tuning |
| `FALLBACK_MODEL_NAME` | `bert-base-uncased` | Fallback if legal-bert unavailable |
| `EPOCHS` | (see config.py) | Training epochs |
| `BATCH_SIZE` | (see config.py) | Training batch size |
| `hybrid_threshold` | `0.75` | Rule confidence threshold before ML fallback |

### Runtime Safety Flags

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | Set to `production` for deployed environments |
| `USE_REAL_AI` | `true` | Attempts to use the real extraction/classification/risk engines |
| `ALLOW_MOCK_AI` | `true` outside production | Allows labelled demo responses when optional ML dependencies are missing |
| `LLM_PROVIDER` | `rule` | Conversational provider: `rule` or `local_http` |
| `LOCAL_LLM_URL` | `http://localhost:11435/v1` | OpenAI-compatible local LLM server URL |
| `LOCAL_LLM_MODEL` | `lex-ai-legal` | Local fine-tuned model id |
| `CORS_ALLOWED_ORIGINS` | `*` | Comma-separated frontend origins; restrict this in production |

The local fine-tuned LLM is experimental. Run
`python fine_tuning/validate_dataset.py` before retraining, and keep
`LLM_PROVIDER=rule` until the dataset and behavior tests are clean.

---

## 🔧 Optional Dependencies

| Package | Feature |
|---|---|
| `spacy` + `en_core_web_sm` | Advanced lemmatization preprocessing |
| `sentence-transformers` + `faiss-cpu` | Contract comparison endpoint |
| `pytesseract` + `Tesseract` | OCR for scanned PDFs |
| `transformers` + `datasets` + `torch` | Legal-BERT fine-tuning & inference |

---

## 📄 License

This project is provided for educational and research purposes.
