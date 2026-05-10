# Lex AI Architecture

Lex AI is being consolidated around one production architecture:

- Frontend: `frontend/`, served by FastAPI at `/app`
- Backend: `backend/main.py`, started with `uvicorn backend.main:app`
- Worker: Celery tasks in `backend/worker/tasks.py`
- AI engine: reusable extraction, risk, summary, clause, and comparison modules in `ai_engine/`

## Deprecated Entry Points

The following files are legacy prototype entry points:

- `app.py`: Streamlit prototype UI
- `api.py`: single-file FastAPI prototype API

Do not add new features to these files. If a feature exists only in a legacy
entry point, migrate it into the modular backend under `backend/` and expose it
through the browser frontend in `frontend/`.

## Request Flow

The preferred production flow is:

1. Browser frontend calls the modular FastAPI backend.
2. FastAPI validates input, stores uploads, and returns either a direct response
   for lightweight requests or a task id for heavy requests.
3. Celery workers execute heavy AI work:
   - PDF/DOCX extraction
   - OCR
   - summarization
   - risk detection
   - clause classification
   - contract comparison
4. The frontend polls `/tasks/{task_id}` or a chat-specific result endpoint
   until the task is complete.

## Performance Strategy

- Prefer hybrid clause detection: run regex/spaCy first, use transformer models
  only for low-confidence or unknown clauses.
- Use batch inference for transformer classification.
- Load heavy models once per worker process and reuse them through module-level
  singletons or a model registry.
- Keep the FastAPI request cycle focused on validation, storage, routing, and
  task creation.

## Current Transition State

The modular backend already owns the SaaS dashboard, upload routes, contract
analysis routes, task polling, chat routes, and comparison routes. Some chat
flows still use an internal lightweight background task manager rather than
Celery. Those flows should be migrated incrementally after the current routes
and frontend behavior are covered by tests.
