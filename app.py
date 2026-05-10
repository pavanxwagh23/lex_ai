"""
Deprecated Streamlit prototype UI.

The supported frontend is the static SaaS dashboard in ``frontend/``, served by
the modular backend at ``http://localhost:8000/app``.
"""

import uuid
import time
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Lex AI Assistant",
    page_icon="⚖️",
    layout="wide"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .sev-high   { background: #e74c3c; color: white; padding: 2px 8px; border-radius: 8px; font-weight: bold; font-size: 0.8rem;}
    .sev-medium { background: #f39c12; color: white; padding: 2px 8px; border-radius: 8px; font-weight: bold; font-size: 0.8rem;}
    .sev-low    { background: #3498db; color: white; padding: 2px 8px; border-radius: 8px; font-weight: bold; font-size: 0.8rem;}
    .meta-tag   { font-size: 0.75rem; color: #7f8c8d; font-family: monospace; display: block; margin-top: 10px; }
    .block-container { max-width: 1000px; padding-top: 2rem; }
    .data-card  { border-left: 5px solid #2ecc71; padding-left: 15px; margin: 10px 0; background: #f9f9f9; border-radius: 4px; }
    .risk-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    .risk-table th { background: #f2f2f2; text-align: left; padding: 12px; }
    .risk-table td { padding: 12px; border-bottom: 1px solid #eee; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ──────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to Lex AI. Please upload a contract or paste one using the sidebar to begin checking for risks, summarizing, or analyzing!"}
    ]
if "context_text" not in st.session_state:
    st.session_state.context_text = None
if "contract_id" not in st.session_state:
    st.session_state.contract_id = None

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Context Setup")

st.sidebar.subheader("1. Upload Document")
uploaded_file = st.sidebar.file_uploader("Support: PDF, DOCX", type=["pdf", "docx"], label_visibility="collapsed")
if uploaded_file and st.sidebar.button("📤 Upload Document", use_container_width=True):
    with st.spinner("Uploading and analyzing..."):
        file_tuple = (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")
        resp = requests.post(f"{API_URL}/contracts/upload", files={"file": file_tuple})
        if resp.status_code == 201:
            cid = resp.json().get("contract_id")
            st.session_state.contract_id = cid
            st.session_state.context_text = None  # Wipes typed text if an upload replaces it
            st.sidebar.success(f"Successfully processed: {uploaded_file.name}")
            st.session_state.messages.append({"role": "user", "content": f"I have uploaded `{uploaded_file.name}`. Let's begin."})
            st.session_state.messages.append({"role": "assistant", "content": "Document parsed, vectorized, and cached into context! What would you like to do? I can summarize the key points or look for legal risks."})
        else:
            err = resp.json().get("detail", "Upload failed")
            st.sidebar.error(f"Error: {err}")

st.sidebar.markdown("---")
st.sidebar.subheader("2. OR Paste Text")
text_input = st.sidebar.text_area("Paste Contract Text:", height=200, 
                                  placeholder="Paste the raw text of the agreement here...", label_visibility="collapsed")

if st.sidebar.button("📄 Load Context to Memory", type="primary", use_container_width=True):
    if text_input.strip():
        st.session_state.context_text = text_input.strip()
        st.session_state.contract_id = None  # Wipes uploaded document if raw text replaces it
        st.sidebar.success("Contract loaded into Session Memory!")
        st.session_state.messages.append({"role": "user", "content": "I have pasted a new document context. Let's begin."})
        st.session_state.messages.append({"role": "assistant", "content": "Text context received and cached. What would you like to know?"})
    else:
        st.sidebar.error("Please paste text first.")

if st.session_state.contract_id:
    st.sidebar.info(f"**🟢 Active Document ID:**\n`{st.session_state.contract_id}`")
    
st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.context_text = None
    st.session_state.contract_id = None
    st.rerun()

# ── Main Chat Interface ─────────────────────────────────────────────────────
st.title("⚖️ Lex AI — Intelligent Assistant")
st.caption("Powered by Hybrid NLP Engines & Background Tasks")

# Render existing chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if "data" in msg and msg["data"]:
            with st.expander("View Structured Data Details", expanded=False):
                st.json(msg["data"])
        if "meta" in msg and msg["meta"]:
            m = msg["meta"]
            st.markdown(f"<span class='meta-tag'>⚡ Intent: {m.get('intent', 'UNKNOWN')} | Confidence: {m.get('confidence', 0.0):.1%} | Task ID: {m.get('task_id', 'None')}</span>", unsafe_allow_html=True)


# ── Chat Input Hook ─────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask a legal question (e.g., 'Summarize this' or 'Find risks')..."):
    # Render user message instantly
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Construct API Payload
    payload = {
        "message": user_input,
        "session_id": st.session_state.session_id,
        "extra_context": {}
    }
    
    if st.session_state.contract_id:
        payload["contract_id"] = st.session_state.contract_id
    
    # Attach context text if it's the very first hit for this session
    if st.session_state.context_text:
        payload["extra_context"]["text"] = st.session_state.context_text
        # Clear the payload text after the first send because the backend SessionManager persists it
        st.session_state.context_text = None

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🧠 Analyzing...")
        
        try:
            # Increased timeout to 300s (5mins) to securely handle offline Local LLM processing
            resp = requests.post(f"{API_URL}/chat", json=payload, timeout=300)
            
            if resp.status_code != 200:
                err_text = resp.json().get("message", "Unknown error")
                message_placeholder.markdown(f"**Error:** {err_text}")
                st.session_state.messages.append({"role": "assistant", "content": f"**Error:** {err_text}"})
            else:
                data = resp.json()
                meta = data.get("meta", {})
                status = meta.get("status", "completed")
                intent = data.get("intent", "GENERAL")
                
                # ── ASYNC POLLING LOGIC (Issue 1) ───────────────────────────
                if status == "processing" and "task_id" in meta:
                    task_id = meta["task_id"]
                    message_placeholder.markdown(f"⏳ **Background Task started.**\n\nTask ID: `{task_id}`\n\n_Polling for results (this may take 1-3 minutes if running heavy transformers)_...")
                    
                    # Poll loop - Optimized to 1 second pings for highly responsive feedback loop
                    poll_result = None
                    max_retries = 300 # 5 minutes max (1s polling)
                    for i in range(max_retries):
                        time.sleep(1.0)
                        poll_resp = requests.get(f"{API_URL}/chat/result/{task_id}", timeout=10)
                        if poll_resp.status_code == 200:
                            poll_data = poll_resp.json()
                            if poll_data.get("status") == "completed":
                                poll_result = poll_data.get("result", {})
                                break
                            elif poll_data.get("status") == "failed":
                                message_placeholder.markdown(f"❌ **Task Failed:** {poll_data.get('error')}")
                                st.stop()
                    
                    if not poll_result:
                        message_placeholder.markdown("⚠️ **Task Timed Out via polling boundary.**")
                        st.stop()
                        
                    # Overwrite data block with the fetched background result
                    data = poll_result
                    data["meta"] = {"intent": intent, "confidence": meta.get("confidence", 1.0), "task_id": task_id}
                else:
                    data["meta"]["intent"] = intent
                
                # ── RICH RESPONSE RENDERING ──────────────────────────────────
                final_content = data.get("message")
                structured_data = data.get("data", {})
                
                if structured_data:
                    # Append rich Markdown if it's RISK data
                    if intent == "RISK":
                        explanation = structured_data.get("explanation", "")
                        final_content = f"{final_content}\n\n{explanation}\n\n"
                        findings = structured_data.get("findings", [])
                        for f in findings:
                            sev = f.get('severity', 'low').lower()
                            category = f.get('category', 'Risk Detected').replace('_', ' ').title()
                            final_content += f"- <span class='sev-{sev}'>{sev.upper()}</span> **{category}**: {f.get('description', '')}\n"

                    # Append rich Markdown if it's SUMMARY data
                    elif intent == "SUMMARY":
                        explanation = structured_data.get("explanation", "")
                        points = structured_data.get("key_points", [])
                        final_content = f"{final_content}\n\n{explanation}\n\n**Key Points:**\n"
                        for p in points:
                            final_content += f"- {p}\n"

                message_placeholder.markdown(final_content, unsafe_allow_html=True)
                
                if structured_data:
                    with st.expander("🔍 View Structured Findings Table", expanded=False):
                        if intent == "RISK":
                            import pandas as pd
                            findings = structured_data.get("findings", [])
                            if findings:
                                df = pd.DataFrame(findings)
                                # Clean up column names for display
                                df.columns = [col.replace('_', ' ').title() for col in df.columns]
                                st.table(df)
                            else:
                                st.write("No specific risks granularly indexed.")
                        elif intent == "SUMMARY":
                            st.info("**Summary Breakdown:**")
                            st.write(structured_data.get("explanation", "No detailed summary provided."))
                            if "key_points" in structured_data:
                                for point in structured_data["key_points"]:
                                    st.markdown(f"✅ {point}")
                        elif intent == "CLAUSE_MAP":
                            import pandas as pd
                            clause_groups = structured_data.get("clause_groups", {})
                            if clause_groups:
                                st.markdown(f"**🤖 Model:** Custom Trained Scikit-Learn (TF-IDF + Logistic Regression)  \n"
                                            f"**📊 Total paragraphs analyzed:** {structured_data.get('total_paragraphs', 0)}  \n"
                                            f"**🗂️ Unique clause types found:** {structured_data.get('unique_clause_types', 0)}")
                                st.markdown("---")
                                for ctype, info in clause_groups.items():
                                    label = info.get("label", ctype)
                                    count = info.get("count", 0)
                                    avg_conf = info.get("avg_confidence", 0)
                                    items = info.get("items", [])
                                    conf_pct = f"{avg_conf * 100:.1f}%"
                                    with st.expander(f"📌 **{label}** — {count} paragraph(s) • Avg confidence: {conf_pct}"):
                                        rows = [{"Confidence": f"{i['confidence']*100:.1f}%",
                                                 "Text Snippet": i['text_snippet']} for i in items]
                                        if rows:
                                            st.table(pd.DataFrame(rows))
                            else:
                                st.warning("No clause groups found. Ensure the model is trained first.")
                        else:
                            st.json(structured_data)
                
                m = data.get("meta", {})
                st.markdown(f"<span class='meta-tag'>⚡ Intent: {m.get('intent', 'UNKNOWN')} | Confidence: {m.get('confidence', 0.0):.1%} | Task ID: {m.get('task_id', 'None')}</span>", unsafe_allow_html=True)
                
                # Append to memory
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_content,
                    "data": structured_data,
                    "meta": m
                })

        except requests.exceptions.ConnectionError:
            message_placeholder.markdown("❌ **Error:** Cannot connect to backend running on port 8000.")
            st.session_state.messages.append({"role": "assistant", "content": "Failed to connect to AI server."})
