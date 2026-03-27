# ⚖️ LexAI Ultra Chat

A **ChatGPT-style AI legal assistant** built with Streamlit + FastAPI.  
Ask questions about contracts, tenant rights, employment, police rights, bail, fraud, divorce, copyright, immigration, criminal law, tax law, and more.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Add OpenAI Key

Copy `.env.example` → `.env` and add your OpenAI API key for GPT-3.5 responses.  
**The app works perfectly without any API key** using the built-in legal AI.

```bash
copy .env.example .env
# Then edit .env and add your key
```

### 3. Start the Backend

```bash
uvicorn api:app --reload
```

Backend runs at: `http://localhost:8000`  
Check status: `http://localhost:8000/health`

### 4. Start the Frontend (new terminal)

```bash
streamlit run app.py
```

Frontend opens at: `http://localhost:8501`

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 Chat with history | Full ChatGPT-style conversation sidebar |
| 🧠 Built-in Legal AI | 15 legal topic areas, no API key needed |
| 🤖 OpenAI Integration | GPT-3.5 Turbo (if API key provided) |
| 📄 PDF Upload | Analyses contracts and legal documents |
| 🌙 Dark / ☀️ Light theme | Toggle from sidebar |
| ⚡ Prompt Chips | Quick-start suggested questions |
| 🗑 Delete Conversations | Manage chat history |

---

## 📁 Project Structure

```
legal_ai_analyzer/
├── app.py              ← Streamlit frontend
├── api.py              ← FastAPI backend
├── requirements.txt    ← Python dependencies
├── .env.example        ← API key template
└── ai_engine/          ← Legal AI modules
```

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server status + AI mode |
| POST | `/chat` | Chat with message history |
| POST | `/analyze` | Single-message analysis |
| POST | `/upload` | PDF document analysis |
| GET | `/sessions/{id}` | Get session history |
| DELETE | `/sessions/{id}` | Clear session |

---

## 📋 Legal Topics Covered

Contracts · Tenant Rights · Employment Law · Police Rights · Bail ·  
Fraud · Consumer Rights · Divorce · Family Law · Copyright · IP ·  
Lawsuits · Immigration · Criminal Law · Tax Law · Privacy Rights
