import streamlit as st
import requests
import pandas as pd
import json

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Legal AI Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# --- Define custom CSS for the app ---
def local_css():
    st.markdown("""
    <style>
        .clause-box {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border-left: 5px solid;
            background-color: var(--background-color-secondary);
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        }
        .clause-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        .clause-title {
            font-size: 1.2rem;
            text-transform: uppercase;
        }
        .confidence-high { color: #2ecc71; border-left-color: #2ecc71; }
        .confidence-medium { color: #f1c40f; border-left-color: #f1c40f; }
        .confidence-low { color: #e74c3c; border-left-color: #e74c3c; }
        .clause-text {
            border-top: 1px solid rgba(128,128,128,0.2);
            padding-top: 0.8rem;
            font-size: 1rem;
            line-height: 1.5;
        }
        .badge {
            padding: 0.3rem 0.6rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            font-weight: bold;
            display: inline-block;
        }
        .badge-heading {
            background-color: #3498db;
            color: white;
            margin-left: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
local_css()

# --- Utility Functions ---
def get_confidence_class(score):
    if score >= 0.75: return "confidence-high"
    elif score >= 0.5: return "confidence-medium"
    return "confidence-low"

def display_clause(clause):
    conf_class = get_confidence_class(clause['confidence'])
    heading_badge = '<span class="badge badge-heading">HEADING</span>' if clause.get('is_heading') else ''
    
    st.markdown(f"""
    <div class="clause-box {conf_class}">
        <div class="clause-header">
            <div>
                <span class="clause-title">{clause['clause_type']}</span>
                {heading_badge}
            </div>
            <span>Confidence: {clause['confidence']:.2f}</span>
        </div>
        <div class="clause-text">
            {clause['text']}
        </div>
    </div>
    """, unsafe_allow_html=True)


# --- Main App ---
st.title("⚖️ Legal AI Analyzer Pro")
st.markdown("Upload a contract or paste text to detect legal clauses automatically.")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")
use_preprocessing = st.sidebar.checkbox("Use spaCy Preprocessing", value=True, help="Applies lemmatization and advanced normalization.")
use_hybrid = st.sidebar.checkbox("Use Hybrid Classifier", value=False, help="Fall back to ML models for low-confidence rules (if configured).")

mode = st.radio("Select Input Mode:", ["Upload PDF", "Paste Text"], horizontal=True)

st.markdown("---")

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# --- Upload PDF Mode ---
if mode == "Upload PDF":
    uploaded_file = st.file_uploader("Choose a PDF Contract", type="pdf")
    use_ocr = st.checkbox("Enable OCR (for scanned PDFs)", value=False)
    
    if st.button("Analyze PDF", type="primary"):
        if uploaded_file is not None:
            with st.spinner("Analyzing document... This may take a moment."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    data = {
                        "use_ocr": use_ocr,
                        "use_preprocessing": use_preprocessing,
                        "use_hybrid": use_hybrid
                    }
                    response = requests.post(f"{API_URL}/analyze_pdf", files=files, data=data)
                    
                    if response.status_code == 200:
                        st.session_state.analysis_results = response.json()["analysis"]
                        st.success("Analysis Complete!")
                    else:
                        st.error(f"Error from server: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to the backend server. Is FastAPI running on port 8000?")
        else:
            st.warning("Please upload a file first.")

# --- Paste Text Mode ---
else:
    text_input = st.text_area("Paste contract text here:", height=300, placeholder="1. DEFINITIONS\n\nFor the purposes of this Agreement...")
    
    if st.button("Analyze Text", type="primary"):
        if text_input.strip():
            with st.spinner("Analyzing text..."):
                try:
                    payload = {
                        "text": text_input,
                        "use_preprocessing": use_preprocessing,
                        "use_hybrid": use_hybrid
                    }
                    response = requests.post(f"{API_URL}/analyze_text", json=payload)
                    
                    if response.status_code == 200:
                        st.session_state.analysis_results = response.json()
                        st.success("Analysis Complete!")
                    else:
                        st.error(f"Error from server: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to the backend server. Is FastAPI running on port 8000?")
        else:
            st.warning("Please enter some text to analyze.")


# --- Display Results ---
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    stats = results["statistics"]
    grouped_clauses = results["grouped_clauses"]
    all_clauses = results["structured_clauses"]
    
    st.markdown("---")
    st.header("Results Dashboard")
    
    # Overview Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Paragraphs Checked", stats["total_paragraphs"])
    col2.metric("Average Confidence", f"{stats['average_confidence']:.2f}")
    detected_types = sum(1 for v in stats['clause_counts'].values() if v > 0)
    col3.metric("Clause Types Found", detected_types)
    
    st.write("")
    
    # Filter non-empty categories
    active_categories = {k: v for k, v in grouped_clauses.items() if v}
    
    if not active_categories:
        st.info("No specific legal clauses were recognized. Everything was classified as 'Other'.")
    else:
        # Create tabs for structured viewing
        tab_names = ["All Extracted"] + [k.replace('_', ' ').title() for k in active_categories.keys()]
        tabs = st.tabs(tab_names)
        
        # Tab 0: All Clauses (grouped, easy scanning)
        with tabs[0]:
            # Filter out the "other" category for the main view to reduce noise, unless it's only 'other'
            display_list = [c for c in all_clauses if c['clause_type'] != 'other' or c.get('is_heading')]
            
            # Show high confidence first
            display_list.sort(key=lambda x: x['confidence'], reverse=True)
            
            st.subheader(f"Top Extracted Clauses ({len(display_list)})")
            # Only show top 20 to avoid freezing on huge docs
            for clause in display_list[:20]:
                display_clause(clause)
            if len(display_list) > 20:
                st.info(f"...and {len(display_list) - 20} more clauses below threshold.")
        
        # Subsequent Tabs: Individual Categories
        for i, (cat_name, cat_clauses) in enumerate(active_categories.items()):
            # i+1 because Tab 0 is "All"
            with tabs[i+1]:
                st.subheader(f"{cat_name.replace('_', ' ').title()} Clauses ({len(cat_clauses)})")
                
                # Sort by confidence
                cat_clauses.sort(key=lambda x: x['confidence'], reverse=True)
                
                for clause in cat_clauses:
                    display_clause(clause)

    # Raw Json option
    with st.expander("View Raw JSON Output"):
        st.json(results)
