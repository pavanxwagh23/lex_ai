import streamlit as st
import requests
import uuid
import time

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexAI Ultra Chat",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://localhost:8000"

# ─── Session State Init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "conversations": {},        # { conv_id: { title, messages: [ {role, content, is_error} ] } }
        "active_conv": None,        # current conversation id
        "loading": False,
        "theme": "dark",
        "ai_mode": "built-in",      # updated from /health
        "health_checked": False,
        "uploaded_doc_summary": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── Check Backend Health ──────────────────────────────────────────────────────
if not st.session_state.health_checked:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if r.status_code == 200:
            st.session_state.ai_mode = r.json().get("ai_mode", "built-in")
    except Exception:
        pass
    st.session_state.health_checked = True

# ─── Helpers ──────────────────────────────────────────────────────────────────
def new_conversation():
    cid = str(uuid.uuid4())
    st.session_state.conversations[cid] = {"title": "New Chat", "messages": []}
    st.session_state.active_conv = cid
    st.session_state.loading = False
    st.session_state.uploaded_doc_summary = None
    return cid

def get_active_messages():
    if st.session_state.active_conv and st.session_state.active_conv in st.session_state.conversations:
        return st.session_state.conversations[st.session_state.active_conv]["messages"]
    return []

def add_message(role, content, is_error=False):
    cid = st.session_state.active_conv
    if not cid:
        cid = new_conversation()
    msgs = st.session_state.conversations[cid]["messages"]
    msgs.append({"role": role, "content": content, "is_error": is_error})
    # Auto-title from first user message
    if role == "user" and st.session_state.conversations[cid]["title"] == "New Chat":
        st.session_state.conversations[cid]["title"] = content[:40] + ("…" if len(content) > 40 else "")

# Init a default conversation if empty
if not st.session_state.conversations:
    new_conversation()

# ─── CSS ─────────────────────────────────────────────────────────────────────
is_dark = st.session_state.theme == "dark"

# Theme colors
if is_dark:
    BG             = "linear-gradient(160deg, #060612 0%, #0c0c1e 50%, #100820 100%)"
    SIDEBAR_BG     = "#0a0a1a"
    SIDEBAR_BORDER = "rgba(139,92,246,0.18)"
    TEXT_PRIMARY   = "#e2e8f0"
    TEXT_MUTED     = "#6b7280"
    CONV_BG        = "rgba(255,255,255,0.04)"
    CONV_ACTIVE    = "rgba(139,92,246,0.18)"
    CONV_BORDER    = "rgba(255,255,255,0.06)"
    AI_BUBBLE_BG   = "rgba(255,255,255,0.04)"
    AI_BUBBLE_CLR  = "#e2e8f0"
    AI_BUBBLE_BRD  = "rgba(255,255,255,0.09)"
    AI_AV_BG       = "linear-gradient(135deg,#0f172a,#1e1b4b)"
    INPUT_BG       = "rgba(255,255,255,0.04)"
    INPUT_CLR      = "#e2e8f0"
    HR_CLR         = "rgba(139,92,246,0.12)"
    EMPTY_CLR      = "#6b7280"
    CHIP_BG        = "rgba(139,92,246,0.1)"
    CHIP_BRD       = "rgba(139,92,246,0.25)"
    CHIP_CLR       = "#a78bfa"
    BADGE_BG       = "rgba(139,92,246,0.15)"
    BADGE_CLR      = "#a78bfa"
    SCROLLBAR      = "rgba(139,92,246,0.2)"
else:
    BG             = "linear-gradient(160deg, #f0f4ff 0%, #ffffff 50%, #f5f0ff 100%)"
    SIDEBAR_BG     = "#f8f7ff"
    SIDEBAR_BORDER = "rgba(109,40,217,0.15)"
    TEXT_PRIMARY   = "#1e1b4b"
    TEXT_MUTED     = "#6b7280"
    CONV_BG        = "rgba(0,0,0,0.03)"
    CONV_ACTIVE    = "rgba(109,40,217,0.1)"
    CONV_BORDER    = "rgba(0,0,0,0.06)"
    AI_BUBBLE_BG   = "#ffffff"
    AI_BUBBLE_CLR  = "#1e1b4b"
    AI_BUBBLE_BRD  = "rgba(109,40,217,0.12)"
    AI_AV_BG       = "linear-gradient(135deg,#ede9fe,#ddd6fe)"
    INPUT_BG       = "rgba(255,255,255,0.85)"
    INPUT_CLR      = "#1e1b4b"
    HR_CLR         = "rgba(109,40,217,0.12)"
    EMPTY_CLR      = "#9ca3af"
    CHIP_BG        = "rgba(109,40,217,0.07)"
    CHIP_BRD       = "rgba(109,40,217,0.2)"
    CHIP_CLR       = "#6d28d9"
    BADGE_BG       = "rgba(109,40,217,0.1)"
    BADGE_CLR      = "#6d28d9"
    SCROLLBAR      = "rgba(109,40,217,0.18)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

/* ── App Background ── */
.stApp {{
    background: {BG};
    min-height: 100vh;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {SIDEBAR_BG} !important;
    border-right: 1px solid {SIDEBAR_BORDER} !important;
    width: 280px !important;
}}
section[data-testid="stSidebar"] * {{ color: {TEXT_PRIMARY} !important; }}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span {{
    color: {TEXT_PRIMARY} !important;
}}

/* ── Hide defaults ── */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{
    padding-top: 0 !important;
    padding-bottom: 8rem !important;
    max-width: 860px;
    margin: 0 auto;
}}

/* ── Logo ── */
.lex-logo {{ text-align: center; padding: 1.5rem 0 1rem 0; }}
.lex-logo-icon {{ font-size: 2.8rem; line-height: 1; }}
.lex-logo-name {{
    font-size: 1.35rem; font-weight: 700; margin-top: 0.35rem;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}}
.lex-logo-tag {{
    font-size: 0.7rem; color: {TEXT_MUTED} !important;
    margin-top: 0.1rem; letter-spacing: 0.04em;
}}

/* ── Sidebar Buttons ── */
div[data-testid="stButton"] button {{
    width: 100%;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white !important;
    border: none; border-radius: 10px;
    padding: 0.5rem 1rem; font-weight: 600; font-size: 0.82rem;
    cursor: pointer; transition: all 0.2s; margin-bottom: 0.1rem;
}}
div[data-testid="stButton"] button:hover {{
    opacity: 0.88; transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(79,70,229,0.35);
}}

/* ── Conversation list items ── */
.conv-item {{
    background: {CONV_BG};
    border: 1px solid {CONV_BORDER};
    border-radius: 8px;
    padding: 0.55rem 0.8rem;
    margin-bottom: 0.35rem;
    font-size: 0.8rem;
    color: {TEXT_PRIMARY};
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.conv-item:hover {{ background: {CONV_ACTIVE}; border-color: rgba(139,92,246,0.3); }}
.conv-item.active {{
    background: {CONV_ACTIVE};
    border-color: rgba(139,92,246,0.4);
    font-weight: 600;
}}

/* ── AI Mode Badge ── */
.mode-badge {{
    display: inline-block;
    background: {BADGE_BG};
    color: {BADGE_CLR};
    border: 1px solid {BADGE_CLR}44;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.72rem; font-weight: 600;
    margin-bottom: 0.5rem;
}}

/* ── Page Header ── */
.page-header {{
    text-align: center;
    padding: 2rem 0 1.2rem 0;
}}
.page-header h1 {{
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin: 0;
}}
.page-header p {{
    color: {TEXT_MUTED}; font-size: 0.85rem; margin: 0.35rem 0 0.5rem 0;
}}

/* ── Empty State ── */
.empty-state {{ text-align: center; padding: 3.5rem 1rem; }}
.empty-state-icon {{ font-size: 3.5rem; margin-bottom: 0.6rem; animation: float 3s ease-in-out infinite; }}
@keyframes float {{
    0%, 100% {{ transform: translateY(0px); }}
    50%       {{ transform: translateY(-8px); }}
}}
.empty-state h3 {{ font-size: 1.15rem; color: {EMPTY_CLR}; font-weight: 500; margin: 0 0 0.3rem 0; }}
.empty-state p  {{ font-size: 0.8rem; color: {TEXT_MUTED}; margin: 0; }}
.prompt-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    margin-top: 1.8rem;
    max-width: 520px;
    margin-left: auto; margin-right: auto;
}}
.prompt-chip {{
    background: {CHIP_BG};
    border: 1px solid {CHIP_BRD};
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    font-size: 0.78rem; color: {CHIP_CLR};
    text-align: left; cursor: default;
    transition: all 0.15s;
    line-height: 1.4;
}}
.prompt-chip:hover {{
    background: {CHIP_BRD};
    transform: translateY(-1px);
}}

/* ── Messages ── */
.msg-row {{
    display: flex;
    margin: 0.7rem 0;
    animation: fadein 0.25s ease;
    align-items: flex-end;
    gap: 0.5rem;
}}
@keyframes fadein {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.msg-row.user {{ justify-content: flex-end; }}
.msg-row.ai   {{ justify-content: flex-start; }}

/* ── Avatars ── */
.av {{
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem; flex-shrink: 0;
}}
.av.user {{
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    box-shadow: 0 2px 10px rgba(79,70,229,0.4);
}}
.av.ai {{
    background: {AI_AV_BG};
    border: 1px solid rgba(139,92,246,0.3);
}}

/* ── Chat Bubbles ── */
.bubble {{
    max-width: 72%;
    padding: 0.75rem 1.05rem;
    border-radius: 18px;
    font-size: 0.875rem;
    line-height: 1.7;
    word-wrap: break-word;
    white-space: pre-wrap;
}}
.bubble.user {{
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: #f1f5f9;
    border-bottom-right-radius: 4px;
    border: 1px solid rgba(139,92,246,0.3);
    box-shadow: 0 4px 20px rgba(79,70,229,0.28);
}}
.bubble.ai {{
    background: {AI_BUBBLE_BG};
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    color: {AI_BUBBLE_CLR};
    border-bottom-left-radius: 4px;
    border: 1px solid {AI_BUBBLE_BRD};
    box-shadow: 0 4px 20px rgba(0,0,0,0.07);
}}
.bubble.error {{
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.28);
    color: #f87171;
    border-radius: 18px;
    border-bottom-left-radius: 4px;
}}

/* ── Source label ── */
.src-label {{
    font-size: 0.68rem;
    color: {TEXT_MUTED};
    margin-top: 0.25rem;
    padding-left: 0.3rem;
}}

/* ── Typing dots ── */
.typing {{ display: flex; gap: 5px; align-items: center; padding: 0.2rem 0; }}
.dot {{
    width: 8px; height: 8px;
    background: linear-gradient(135deg, #7c3aed, #60a5fa);
    border-radius: 50%;
    animation: bounce 1.1s infinite;
}}
.dot:nth-child(2) {{ animation-delay: 0.18s; }}
.dot:nth-child(3) {{ animation-delay: 0.36s; }}
@keyframes bounce {{
    0%, 60%, 100% {{ transform: translateY(0); opacity: 0.5; }}
    30%            {{ transform: translateY(-6px); opacity: 1; }}
}}

/* ── Chat Input ── */
.stChatInput > div {{
    background: {INPUT_BG} !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(8px);
}}
.stChatInput textarea {{ color: {INPUT_CLR} !important; background: transparent !important; }}
.stChatInput button {{
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border-radius: 10px !important; border: none !important;
}}

/* ── Divider ── */
hr {{ border-color: {HR_CLR} !important; margin: 0.6rem 0 !important; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {SCROLLBAR}; border-radius: 4px; }}

/* ── Welcome screen example buttons (card style override) ── */
div[data-testid=\"stHorizontalBlock\"] div[data-testid=\"stButton\"].welcome-ex button,
.welcome-ex-btn button {{
    background: {CONV_BG} !important;
    border: 1px solid {CONV_BORDER} !important;
    color: {TEXT_PRIMARY} !important;
    border-radius: 10px !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    padding: 0.65rem 0.85rem !important;
    text-align: center !important;
    box-shadow: none !important;
    transition: all 0.18s !important;
}}
.welcome-ex-btn button:hover {{
    background: {CHIP_BG} !important;
    border-color: {CHIP_BRD} !important;
    color: {CHIP_CLR} !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 14px rgba(139,92,246,0.15) !important;
}}

/* ── Radio ── */
div[data-testid="stRadio"] label {{ font-size: 0.82rem !important; }}

/* ── File uploader ── */
.stFileUploader {{
    background: {CONV_BG};
    border: 1px dashed rgba(139,92,246,0.3) !important;
    border-radius: 10px !important;
    padding: 0.4rem !important;
}}

/* ── Sidebar section labels ── */
.sidebar-section-label {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TEXT_MUTED} !important;
    margin: 0.8rem 0 0.4rem 0;
    padding-left: 0.1rem;
}}

/* ── Upload result ── */
.upload-result {{
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.2);
    border-radius: 8px;
    padding: 0.5rem 0.7rem;
    font-size: 0.75rem;
    color: #34d399;
    margin-top: 0.4rem;
}}

/* ── Welcome Screen (ChatGPT-style) ── */
.welcome-screen {{
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem 0 1rem 0;
}}
.welcome-title {{
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(90deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin-bottom: 0.2rem;
}}
.welcome-subtitle {{
    font-size: 0.85rem;
    color: {TEXT_MUTED};
    margin-bottom: 2rem;
}}
.welcome-cols {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1.1rem;
    width: 100%;
    max-width: 820px;
}}
.welcome-col {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.55rem;
}}
.welcome-col-header {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.35rem;
    margin-bottom: 0.3rem;
}}
.welcome-col-icon {{
    font-size: 1.6rem;
    opacity: 0.85;
}}
.welcome-col-title {{
    font-size: 0.92rem;
    font-weight: 600;
    color: {TEXT_PRIMARY};
}}
.welcome-card {{
    background: {CONV_BG};
    border: 1px solid {CONV_BORDER};
    border-radius: 10px;
    padding: 0.65rem 0.85rem;
    font-size: 0.78rem;
    color: {TEXT_PRIMARY};
    text-align: center;
    width: 100%;
    line-height: 1.5;
    transition: all 0.18s;
}}
.welcome-card.clickable {{
    cursor: pointer;
}}
.welcome-card.clickable:hover {{
    background: {CHIP_BG};
    border-color: {CHIP_BRD};
    color: {CHIP_CLR};
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(139,92,246,0.15);
}}
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="lex-logo">
        <div class="lex-logo-icon">⚖️</div>
        <div class="lex-logo-name">LexAI Ultra</div>
        <div class="lex-logo-tag">AI LEGAL ASSISTANT</div>
    </div>
    """, unsafe_allow_html=True)

    # AI Mode Badge
    mode_emoji = "🤖" if st.session_state.ai_mode == "openai" else "🧠"
    mode_label = "GPT-3.5 Turbo" if st.session_state.ai_mode == "openai" else "Built-in Legal AI"
    st.markdown(f'<div style="text-align:center"><span class="mode-badge">{mode_emoji} {mode_label}</span></div>', unsafe_allow_html=True)

    st.markdown("---")

    # New Chat Button
    if st.button("＋  New Chat", key="btn_new_chat"):
        new_conversation()
        st.rerun()

    # Conversation History List
    conversations = st.session_state.conversations
    if conversations:
        st.markdown('<div class="sidebar-section-label">Recent Chats</div>', unsafe_allow_html=True)
        conv_ids = list(conversations.keys())[:15]  # Show up to 15
        for cid in reversed(conv_ids):
            conv = conversations[cid]
            title = conv.get("title", "New Chat")
            msg_count = len(conv.get("messages", []))
            is_active = cid == st.session_state.active_conv
            active_cls = "active" if is_active else ""
            # Use button for click interaction
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(
                    f"{'💬' if msg_count > 0 else '🔘'} {title}",
                    key=f"conv_{cid}",
                    use_container_width=True,
                ):
                    st.session_state.active_conv = cid
                    st.rerun()
            with col2:
                if not is_active:
                    if st.button("🗑", key=f"del_{cid}"):
                        del st.session_state.conversations[cid]
                        if st.session_state.active_conv == cid:
                            remaining = list(st.session_state.conversations.keys())
                            st.session_state.active_conv = remaining[-1] if remaining else new_conversation()
                        st.rerun()

    st.markdown("---")

    # PDF Upload
    st.markdown('<div class="sidebar-section-label">📎 Upload Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a contract or legal PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        if st.button("🔍 Analyse Document", key="btn_analyse_pdf"):
            with st.spinner("Reading document…"):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        summary = data.get("summary", "")
                        pages = data.get("pages_read", "?")
                        st.session_state.uploaded_doc_summary = summary
                        if not st.session_state.active_conv:
                            new_conversation()
                        add_message("user", f"📄 Analysed document: **{uploaded_file.name}** ({pages} pages)")
                        add_message("ai", summary)
                        st.rerun()
                    else:
                        detail = resp.json().get("detail", "Upload failed.")
                        st.error(f"⚠️ {detail}")
                except requests.exceptions.ConnectionError:
                    st.error("⚠️ Backend not running. Start it first.")
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")

    st.markdown("---")

    # Theme Toggle
    st.markdown('<div class="sidebar-section-label">🎨 Appearance</div>', unsafe_allow_html=True)
    selected_theme = st.radio(
        "theme",
        options=["dark", "light"],
        format_func=lambda x: "🌙 Dark" if x == "dark" else "☀️ Light",
        index=0 if st.session_state.theme == "dark" else 1,
        key="theme_radio",
        label_visibility="collapsed",
        horizontal=True,
    )
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()

    st.markdown("---")

    # Topic Reference
    st.markdown(f"""
    <div style="font-size:0.75rem; color:{TEXT_MUTED}; line-height:1.8;">
        <strong style="color:{TEXT_PRIMARY};">Topics I can help with:</strong><br>
        Contracts • Tenant Rights<br>
        Employment Law • Police Rights<br>
        Bail • Fraud • Consumer Rights<br>
        Divorce & Family Law<br>
        Copyright & IP • Lawsuits<br>
        Immigration • Criminal Law<br>
        Tax Law • Privacy Rights
    </div>
    """, unsafe_allow_html=True)

# ─── MAIN AREA ────────────────────────────────────────────────────────────────
active_messages = get_active_messages()

# Header
st.markdown("""
<div class="page-header">
    <h1>⚖️ LexAI Ultra Chat</h1>
    <p>Your AI-powered legal assistant — instant answers on law, rights & contracts</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── Empty State (ChatGPT-style 3-column welcome) ────────────────────────────
EXAMPLE_PROMPTS = [
    ('"What are my rights as a tenant?" →', "What are my tenant rights?"),
    ('"Can my employer fire me without warning?" →', "Can my employer fire me without warning?"),
    ('"How do I file a consumer complaint?" →', "How do I file a consumer complaint?"),
]

if not active_messages:
    welcome_html = (
        '<div class="welcome-screen">'
        f'<div class="welcome-title">⚖️ LexAI Ultra</div>'
        f'<div class="welcome-subtitle">Your AI-powered legal assistant — ask anything about law, rights &amp; contracts</div>'
        '<div class="welcome-cols">'

        '<div class="welcome-col">'
        '<div class="welcome-col-header">'
        '<div class="welcome-col-icon">☀️</div>'
        '<div class="welcome-col-title">Examples</div>'
        '</div>'
        '<div class="welcome-card">&quot;What are my rights as a tenant?&quot; →</div>'
        '<div class="welcome-card">&quot;Can my employer fire me without warning?&quot; →</div>'
        '<div class="welcome-card">&quot;How do I file a consumer complaint?&quot; →</div>'
        '</div>'

        '<div class="welcome-col">'
        '<div class="welcome-col-header">'
        '<div class="welcome-col-icon">⚡</div>'
        '<div class="welcome-col-title">Capabilities</div>'
        '</div>'
        '<div class="welcome-card">Remembers context within the conversation</div>'
        '<div class="welcome-card">Allows follow-up questions &amp; corrections</div>'
        '<div class="welcome-card">Analyses uploaded legal PDFs &amp; contracts</div>'
        '</div>'

        '<div class="welcome-col">'
        '<div class="welcome-col-header">'
        '<div class="welcome-col-icon">⚠️</div>'
        '<div class="welcome-col-title">Limitations</div>'
        '</div>'
        '<div class="welcome-card">May occasionally generate imprecise legal info</div>'
        '<div class="welcome-card">Not a substitute for a qualified lawyer</div>'
        '<div class="welcome-card">Knowledge may not cover the latest case law</div>'
        '</div>'

        '</div>'
        '</div>'
    )
    st.markdown(welcome_html, unsafe_allow_html=True)

    # Hidden clickable buttons for the Example cards (visually replaced by welcome cards above)
    st.markdown('<div style="display:none">', unsafe_allow_html=True)
    ex_cols = st.columns(3)
    for i, (label, prompt) in enumerate(EXAMPLE_PROMPTS):
        with ex_cols[i]:
            if st.button(label, key=f"ex_btn_{i}", use_container_width=True):
                if not st.session_state.active_conv:
                    new_conversation()
                add_message("user", prompt)
                st.session_state.loading = True
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Spacer
    st.markdown("<br>", unsafe_allow_html=True)

# ─── Render Messages ──────────────────────────────────────────────────────────
def render_message(role, content, is_error=False, source=None):
    icon = "👤" if role == "user" else "⚖️"
    bubble_cls = "error" if is_error else role

    if role == "user":
        st.markdown(f"""
        <div class="msg-row user">
            <div class="bubble user">{content}</div>
            <div class="av user">{icon}</div>
        </div>""", unsafe_allow_html=True)
    else:
        # AI message — render markdown inside bubble via st.markdown for proper formatting
        col_av, col_content = st.columns([0.06, 0.94])
        with col_av:
            st.markdown(f'<div class="av ai" style="margin-top:0.5rem">{icon}</div>', unsafe_allow_html=True)
        with col_content:
            with st.container():
                if is_error:
                    st.markdown(f'<div class="bubble error">{content}</div>', unsafe_allow_html=True)
                else:
                    # Use a styled container for glassmorphism effect, render markdown inside
                    st.markdown(f"""
                    <div class="bubble ai" style="max-width:100%">
                    {content}
                    </div>""", unsafe_allow_html=True)


for msg in active_messages:
    render_message(msg["role"], msg["content"], msg.get("is_error", False))

# ─── Typing Indicator ─────────────────────────────────────────────────────────
typing_slot = st.empty()
if st.session_state.loading:
    typing_slot.markdown("""
    <div class="msg-row ai" style="gap:0.5rem; align-items:flex-end;">
        <div class="av ai">⚖️</div>
        <div class="bubble ai" style="padding:0.7rem 1rem">
            <div class="typing">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

# ─── Chat Input ───────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Ask a legal question… (e.g. 'What are my rights as a tenant?')",
    disabled=st.session_state.loading,
)

if user_input:
    text = user_input.strip()
    if not text:
        st.warning("⚠️ Please type a question before sending.")
    else:
        if not st.session_state.active_conv:
            new_conversation()
        add_message("user", text)
        st.session_state.loading = True
        st.rerun()

# ─── API Call ─────────────────────────────────────────────────────────────────
if st.session_state.loading and active_messages:
    last = active_messages[-1]
    if last["role"] == "user":
        try:
            # Build messages payload for /chat
            payload_messages = [
                {"role": m["role"] if m["role"] != "ai" else "assistant", "content": m["content"]}
                for m in active_messages
                if not m.get("is_error")
            ]

            resp = requests.post(
                f"{BACKEND_URL}/chat",
                json={
                    "session_id": st.session_state.active_conv,
                    "messages": payload_messages,
                },
                timeout=30,
            )

            if resp.status_code == 200:
                data = resp.json()
                reply = data.get("reply", "")
                src = data.get("source", "fallback")
                add_message("ai", reply)

            elif resp.status_code == 400:
                add_message("ai", "⚠️ Please enter a valid legal question.", is_error=True)
            else:
                detail = resp.json().get("detail", "Unknown backend error.")
                add_message("ai", f"⚠️ Backend error: {detail}", is_error=True)

        except requests.exceptions.ConnectionError:
            add_message("ai", (
                "⚠️ **Backend not responding.**\n\n"
                "Please start the backend server:\n"
                "```\nuvicorn api:app --reload\n```"
            ), is_error=True)
        except requests.exceptions.Timeout:
            add_message("ai", "⚠️ Request timed out. Please try again.", is_error=True)
        except Exception as e:
            add_message("ai", f"⚠️ Unexpected error: {str(e)}", is_error=True)
        finally:
            st.session_state.loading = False
            st.rerun()
