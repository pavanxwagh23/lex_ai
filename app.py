import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Legal AI Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Risk score gauge  */
    .risk-gauge { font-size: 2.5rem; font-weight: 800; }
    .risk-low    { color: #27ae60; }
    .risk-medium { color: #f39c12; }
    .risk-high   { color: #c0392b; }

    /* Severity badges */
    .sev-badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.78rem; font-weight: 700; color: #fff; margin-right: 6px;
    }
    .sev-HIGH   { background: #e74c3c; }
    .sev-MEDIUM { background: #f39c12; }
    .sev-LOW    { background: #3498db; }

    /* General polish */
    .block-container { max-width: 1100px; }
    div[data-testid="stExpander"] summary { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ─────────────────────────────────────────────────────────────────

CONF_EMOJI = {True: "🟢", False: "🟡"}   # ≥0.75 vs <0.75
SEV_ICON   = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}


def confidence_indicator(score):
    if score >= 0.75:
        return "🟢 High"
    elif score >= 0.50:
        return "🟡 Medium"
    return "🔴 Low"


def render_clause_card(clause):
    """Render one clause as a native Streamlit expander."""
    ctype  = clause["clause_type"].replace("_", " ").title()
    conf   = clause["confidence"]
    is_hdg = clause.get("is_heading", False)

    badge  = " 🏷️ _Heading_" if is_hdg else ""
    title  = f"{confidence_indicator(conf)}  **{ctype}**  —  Confidence {conf:.0%}{badge}"

    with st.expander(title, expanded=False):
        st.markdown(clause["text"])
        if clause.get("scores_per_class"):
            # Show top-3 competing scores for transparency
            scores = clause["scores_per_class"]
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = "  ·  ".join(f"`{k}` {v}" for k, v in top if v > 0)
            if top_str:
                st.caption(f"Pattern scores → {top_str}")


def render_risk_card(risk):
    """Render one risk finding."""
    sev   = risk.get("severity", "LOW")
    icon  = SEV_ICON.get(sev, "🔵")
    rtype = risk.get("type", "Unknown")

    st.markdown(f"""
{icon} **{rtype}** — `{sev}`  
**Matched phrase:** `{risk.get('matched_phrase', '—')}`  
{risk.get('explanation', '')}  
""")
    with st.expander("Show source paragraph", expanded=False):
        st.caption(risk.get("text", ""))
    st.markdown("---")


def render_summary(data):
    """Display the structured summary."""
    st.subheader("📄 Overall Summary")
    st.info(data.get("overall_summary", "No summary generated."))

    sections = [
        ("🔑 Key Points",         data.get("key_points", [])),
        ("📋 Obligations",        data.get("obligations", [])),
        ("💰 Payment Terms",      data.get("payment_terms", [])),
        ("🚪 Termination",        data.get("termination", [])),
        ("⚠️ Risks & Liability",  data.get("risks", [])),
    ]

    for title, items in sections:
        if items:
            st.markdown(f"**{title}**")
            for item in items:
                st.markdown(f"- {item}")
            st.write("")


# ── Main App ────────────────────────────────────────────────────────────────
st.title("⚖️ Legal AI Analyzer Pro")
st.markdown("Upload a contract or paste text to detect legal clauses, assess risks, and generate a summary.")

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
use_preprocessing = st.sidebar.checkbox("Use spaCy Preprocessing", value=True,
                                        help="Applies lemmatization and advanced normalization.")
use_hybrid = st.sidebar.checkbox("Use Hybrid Classifier", value=False,
                                  help="Fall back to ML models for low-confidence rules.")

mode = st.radio("Select Input Mode:", ["Upload PDF", "Paste Text"], horizontal=True)
st.markdown("---")

# ── Session state ───────────────────────────────────────────────────────────
for key in ["analysis_results", "risk_results", "summary_results", "paragraphs", "input_text"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ── INPUT ───────────────────────────────────────────────────────────────────
if mode == "Upload PDF":
    uploaded_file = st.file_uploader("Choose a PDF Contract", type="pdf")
    use_ocr = st.checkbox("Enable OCR (for scanned PDFs)", value=False)

    if st.button("🔍 Analyze PDF", type="primary", key="btn_pdf"):
        if uploaded_file is not None:
            with st.spinner("Extracting and analyzing your contract…"):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    data = {"use_ocr": use_ocr, "use_preprocessing": use_preprocessing,
                            "use_hybrid": use_hybrid}
                    resp = requests.post(f"{API_URL}/analyze_pdf", files=files, data=data, timeout=120)
                    if resp.status_code == 200:
                        body = resp.json()
                        st.session_state.analysis_results = body["analysis"]
                        st.session_state.paragraphs = [
                            c["text"] for c in body["analysis"]["structured_clauses"] if c.get("text")
                        ]
                        st.session_state.input_text = "\n\n".join(st.session_state.paragraphs)
                        st.session_state.risk_results = None
                        st.session_state.summary_results = None
                        st.success("✅ Analysis Complete!")
                    else:
                        st.error(f"Server error: {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the API — is FastAPI running on port 8000?")
        else:
            st.warning("Please upload a file first.")

else:
    text_input = st.text_area("Paste contract text here:", height=300,
                               placeholder="1. DEFINITIONS\n\nFor the purposes of this Agreement…")
    if st.button("🔍 Analyze Text", type="primary", key="btn_text"):
        if text_input.strip():
            with st.spinner("Analyzing text…"):
                try:
                    payload = {"text": text_input, "use_preprocessing": use_preprocessing,
                               "use_hybrid": use_hybrid}
                    resp = requests.post(f"{API_URL}/analyze_text", json=payload, timeout=120)
                    if resp.status_code == 200:
                        st.session_state.analysis_results = resp.json()
                        st.session_state.paragraphs = [
                            c["text"] for c in resp.json()["structured_clauses"] if c.get("text")
                        ]
                        st.session_state.input_text = text_input
                        st.session_state.risk_results = None
                        st.session_state.summary_results = None
                        st.success("✅ Analysis Complete!")
                    else:
                        st.error(f"Server error: {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the API — is FastAPI running on port 8000?")
        else:
            st.warning("Please enter some text to analyze.")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CLAUSE DETECTION
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    stats   = results["statistics"]
    grouped = results["grouped_clauses"]
    all_clauses = results["structured_clauses"]

    st.markdown("---")
    st.header("📑 Clause Detection Results")

    # ── Metrics row ─────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Paragraphs Analyzed", stats["total_paragraphs"])
    m2.metric("Average Confidence", f"{stats['average_confidence']:.0%}")
    detected = sum(1 for v in stats["clause_counts"].values() if v > 0)
    m3.metric("Clause Types Found", detected)

    # ── Clause type summary table ───────────────────────────────────────
    if stats["clause_counts"]:
        st.markdown("**Detected clause types:**")
        type_cols = st.columns(min(len(stats["clause_counts"]), 6))
        for idx, (ctype, count) in enumerate(stats["clause_counts"].items()):
            if count > 0:
                type_cols[idx % len(type_cols)].markdown(
                    f"• **{ctype.replace('_', ' ').title()}** — {count}"
                )

    st.write("")

    # ── Active categories as tabs ───────────────────────────────────────
    active = {k: v for k, v in grouped.items() if v}

    if not active:
        st.info("No specific legal clauses recognized — everything classified as 'Other'.")
    else:
        tab_labels = ["📋 All Clauses"] + [
            f"{k.replace('_', ' ').title()} ({len(v)})" for k, v in active.items()
        ]
        tabs = st.tabs(tab_labels)

        # Tab 0 — all clauses (exclude 'other')
        with tabs[0]:
            display = sorted(
                [c for c in all_clauses if c["clause_type"] != "other" or c.get("is_heading")],
                key=lambda c: c["confidence"], reverse=True,
            )
            st.subheader(f"All Extracted Clauses ({len(display)})")
            for clause in display[:30]:
                render_clause_card(clause)
            if len(display) > 30:
                st.info(f"…and {len(display) - 30} more clauses not shown.")

        # Per-type tabs
        for i, (cat, cat_clauses) in enumerate(active.items()):
            with tabs[i + 1]:
                st.subheader(f"{cat.replace('_', ' ').title()} ({len(cat_clauses)})")
                for clause in sorted(cat_clauses, key=lambda c: c["confidence"], reverse=True):
                    render_clause_card(clause)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RISK ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.analysis_results and st.session_state.paragraphs:
    st.markdown("---")
    st.header("🛡️ Risk Analysis")
    st.caption("Scans for 8 legal risk categories: Unlimited Liability, One-Sided Indemnity, "
               "IP Ownership, Payment Risk, Jurisdiction Risk, and more.  "
               "_Fast rule-based engine — no model download required._")

    if st.button("🔍 Run Risk Analysis", key="btn_risk", type="primary"):
        with st.spinner("Scanning for legal risks…"):
            try:
                resp = requests.post(f"{API_URL}/analyze_risks",
                                     json={"paragraphs": st.session_state.paragraphs}, timeout=30)
                if resp.status_code == 200:
                    st.session_state.risk_results = resp.json()
                elif resp.status_code == 503:
                    st.warning(resp.json().get("detail", "Risk engine unavailable."))
                else:
                    st.error(resp.text)
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API server.")

    if st.session_state.risk_results:
        risk = st.session_state.risk_results
        score   = risk.get("risk_score", 0)
        total   = risk.get("total_paragraphs", 0)
        flagged = risk.get("risky_paragraph_count", 0)
        findings = risk.get("risks_detected", [])

        # Risk gauge
        if score < 3:
            css, label = "risk-low",    "Low Risk"
        elif score < 6:
            css, label = "risk-medium", "Medium Risk"
        else:
            css, label = "risk-high",   "High Risk"

        g1, g2, g3, g4 = st.columns(4)
        g1.markdown(f'<div class="risk-gauge {css}">{score:.1f}<span style="font-size:1rem"> / 10</span></div>', unsafe_allow_html=True)
        g2.metric("Risk Level", label)
        g3.metric("Risky Paragraphs", f"{flagged} / {total}")
        g4.metric("Findings", len(findings))

        st.write("")

        if not findings:
            st.success("✅ No significant legal risks detected.")
        else:
            high   = [f for f in findings if f.get("severity") == "HIGH"]
            medium = [f for f in findings if f.get("severity") == "MEDIUM"]
            low    = [f for f in findings if f.get("severity") == "LOW"]

            if high:
                st.subheader(f"🔴 High Severity ({len(high)})")
                for f in high:
                    render_risk_card(f)
            if medium:
                st.subheader(f"🟡 Medium Severity ({len(medium)})")
                for f in medium:
                    render_risk_card(f)
            if low:
                st.subheader(f"🔵 Low Severity ({len(low)})")
                for f in low:
                    render_risk_card(f)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SUMMARY
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.analysis_results and st.session_state.paragraphs:
    st.markdown("---")
    st.header("📝 Contract Summary")
    st.caption("Uses **facebook/bart-large-cnn** to generate an AI summary with labelled sections.  "
               "⏳ _First run downloads ~1.6 GB model. Requires_ `pip install transformers torch`.")

    if st.button("📝 Generate Summary", key="btn_summary", type="primary"):
        with st.spinner("Generating legal summary — this may take 1-3 minutes on first run…"):
            try:
                resp = requests.post(f"{API_URL}/summarize",
                                     json={"paragraphs": st.session_state.paragraphs}, timeout=300)
                if resp.status_code == 200:
                    st.session_state.summary_results = resp.json()
                elif resp.status_code == 503:
                    st.warning(resp.json().get("detail", "Summarizer unavailable."))
                    st.code("pip install transformers torch", language="bash")
                else:
                    st.error(resp.text)
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API server.")
            except requests.exceptions.Timeout:
                st.error("Timed out — model may still be downloading. Try again shortly.")

    if st.session_state.summary_results:
        render_summary(st.session_state.summary_results)
