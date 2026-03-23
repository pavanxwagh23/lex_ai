import streamlit as st
import requests
import time

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexAI Ultra Chat",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://localhost:8000/analyze"

# ─── Premium CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App Background ── */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 40%, #0f0620 70%, #0a0a0f 100%);
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a 0%, #111128 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.25);
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* ── Hide default Streamlit elements ── */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {
        padding-top: 1rem;
        padding-bottom: 6rem;
        max-width: 860px;
    }

    /* ── App title in sidebar ── */
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
        text-align: center;
    }
    .sidebar-subtitle {
        font-size: 0.75rem;
        color: #64748b !important;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* ── Chat header ── */
    .chat-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .chat-header h1 {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }
    .chat-header p {
        color: #64748b;
        font-size: 0.875rem;
        margin: 0.25rem 0 0 0;
    }

    /* ── Message bubbles ── */
    .msg-row {
        display: flex;
        margin: 0.6rem 0;
        animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .msg-row.user { justify-content: flex-end; }
    .msg-row.ai   { justify-content: flex-start; }

    .bubble {
        max-width: 72%;
        padding: 0.75rem 1.1rem;
        border-radius: 18px;
        font-size: 0.92rem;
        line-height: 1.6;
        word-wrap: break-word;
        white-space: pre-wrap;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    .bubble.user {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: #f1f5f9;
        border-bottom-right-radius: 4px;
        border: 1px solid rgba(139, 92, 246, 0.4);
    }
    .bubble.ai {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        color: #e2e8f0;
        border-bottom-left-radius: 4px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .avatar {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
        margin: 0 0.5rem;
        align-self: flex-end;
    }
    .avatar.user {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        order: 1;
    }
    .avatar.ai {
        background: linear-gradient(135deg, #0f172a, #1e1b4b);
        border: 1px solid rgba(139,92,246,0.4);
        color: #a78bfa;
        order: -1;
    }

    /* ── Error bubble ── */
    .bubble.error {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #fca5a5;
        border-radius: 18px;
        border-bottom-left-radius: 4px;
    }

    /* ── Typing animation ── */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 4px;
        padding: 0.6rem 1rem;
    }
    .dot {
        width: 8px; height: 8px;
        background: #a78bfa;
        border-radius: 50%;
        animation: bounce 1.2s infinite;
    }
    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.6; }
        30%            { transform: translateY(-6px); opacity: 1; }
    }

    /* ── Input area ── */
    .stChatInput > div {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(139,92,246,0.35) !important;
        border-radius: 14px !important;
    }
    .stChatInput textarea {
        color: #e2e8f0 !important;
        background: transparent !important;
    }
    .stChatInput button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        border-radius: 10px !important;
        border: none !important;
    }

    /* ── New Chat button ── */
    div[data-testid="stButton"] button {
        width: 100%;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1rem;
        font-weight: 600;
        font-size: 0.875rem;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    div[data-testid="stButton"] button:hover { opacity: 0.88; }

    /* ── Divider ── */
    hr { border-color: rgba(139,92,246,0.2) !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.3); border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Session State ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_loading" not in st.session_state:
    st.session_state.is_loading = False

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚖️ LexAI Ultra</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">AI Legal Assistant · Powered by Llama 3</div>', unsafe_allow_html=True)
    st.markdown("---")

    if st.button("✦ New Chat", key="new_chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div style="color:#475569;font-size:0.78rem;line-height:1.6;">
        <b style="color:#a78bfa;">How to use:</b><br>
        Ask any legal question and LexAI will provide a clear, professional answer.<br><br>
        <b style="color:#a78bfa;">Running locally on:</b><br>
        🔗 Ollama · llama3<br>
        🔗 FastAPI backend<br>
        🔗 Streamlit UI
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Main Header ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="chat-header">
        <h1>⚖️ LexAI Ultra Chat</h1>
        <p>Your AI-powered legal assistant — ask anything about law, contracts, or rights.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Chat History ─────────────────────────────────────────────────────────────
def render_message(role: str, content: str, is_error: bool = False):
    bubble_class = "error" if is_error else role
    avatar_icon = "👤" if role == "user" else "⚖️"

    if role == "user":
        st.markdown(
            f"""
            <div class="msg-row user">
                <div class="bubble user">{content}</div>
                <div class="avatar user">{avatar_icon}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="msg-row ai">
                <div class="avatar ai">{avatar_icon}</div>
                <div class="bubble {bubble_class}">{content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"], msg.get("is_error", False))

# ─── Typing Indicator ─────────────────────────────────────────────────────────
typing_placeholder = st.empty()

if st.session_state.is_loading:
    typing_placeholder.markdown(
        """
        <div class="msg-row ai">
            <div class="avatar ai">⚖️</div>
            <div class="bubble ai">
                <div class="typing-indicator">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Input Box ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a legal question…", disabled=st.session_state.is_loading)

if user_input:
    user_text = user_input.strip()
    if not user_text:
        st.warning("Please enter a message before sending.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_text})
        st.session_state.is_loading = True
        st.rerun()

# ─── Backend Call (runs on rerun when is_loading=True) ────────────────────────
if st.session_state.is_loading and st.session_state.messages:
    last_msg = st.session_state.messages[-1]
    if last_msg["role"] == "user":
        try:
            response = requests.post(
                BACKEND_URL,
                json={"text": last_msg["content"]},
                timeout=120,
            )
            if response.status_code == 200:
                ai_text = response.json().get("result", "No response received.")
                st.session_state.messages.append({"role": "ai", "content": ai_text})
            elif response.status_code == 503:
                st.session_state.messages.append({
                    "role": "ai",
                    "content": "⚠️ Ollama is not running. Please start it with: `ollama run llama3`",
                    "is_error": True,
                })
            else:
                detail = response.json().get("detail", "Unknown error from backend.")
                st.session_state.messages.append({
                    "role": "ai",
                    "content": f"⚠️ Backend error: {detail}",
                    "is_error": True,
                })
        except requests.exceptions.ConnectionError:
            st.session_state.messages.append({
                "role": "ai",
                "content": "⚠️ Backend not responding. Make sure you ran: `python -m uvicorn api:app --reload`",
                "is_error": True,
            })
        except requests.exceptions.Timeout:
            st.session_state.messages.append({
                "role": "ai",
                "content": "⚠️ Request timed out. Ollama may be slow — please try again.",
                "is_error": True,
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "ai",
                "content": f"⚠️ Unexpected error: {str(e)}",
                "is_error": True,
            })
        finally:
            st.session_state.is_loading = False
            st.rerun()
