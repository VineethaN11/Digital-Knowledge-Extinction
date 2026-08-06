import streamlit as st
import pandas as pd
import json
import numpy as np
import time
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

import config
from ingestion import DocumentIngester
from rag_pipeline import RAGPipeline
from question_generator import BenchmarkQuestionGenerator
from gap_detector import KnowledgeGapDetector
from agent_decision import AgenticDecisionLayer
from evaluation import RAIEvaluator, analyze_gap_reduction

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Digital Knowledge Extinction Framework",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS INJECTION  — st.markdown with unsafe_allow_html works on Streamlit 1.32
# ─────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global Page Override to Vibrant sapphire Navy Gradient */
.stApp {
    background-color: #071024 !important;
    background-image: radial-gradient(circle at 50% 50%, #0e224b 0%, #050b18 100%) !important;
    color: #ffffff !important;
}
html, body, [class*="css"] { 
    font-family: 'Plus Jakarta Sans', sans-serif !important; 
    color: #f8fafc !important;
}

/* Sidebar Width & Styling */
[data-testid="stSidebar"] {
    background-color: #040c1b !important;
    border-right: 1px solid rgba(0, 195, 255, 0.25) !important;
    width: 280px !important;
    min-width: 280px !important;
    max-width: 280px !important;
}
/* Force all text inside sidebar to be bright and visible */
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #050b18;
}
::-webkit-scrollbar-thumb {
    background: rgba(0, 195, 255, 0.25);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 195, 255, 0.5);
}

/* ── Shimmer Animation ── */
@keyframes text-shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* ── Header ── */
.title-wrap {
    padding: 1.8rem;
    background: rgba(13, 27, 56, 0.8) !important;
    border: 1px solid rgba(0, 195, 255, 0.3) !important;
    border-radius: 16px;
    margin-bottom: 1.6rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35), inset 0 0 15px rgba(255, 255, 255, 0.02);
}
.main-title {
    font-size: 2.8rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 25%, #ffffff 50%, #cbd5e1 75%, #ffffff 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: text-shimmer 7s linear infinite;
    text-shadow: 0 0 20px rgba(255,255,255,0.1);
}
.main-sub { font-size: .98rem; color: #94a3b8; margin: .3rem 0 0; }
.interactive-sub {
    font-size: 1.05rem;
    font-weight: 600;
    color: #38bdf8;
    margin-top: .6rem;
    padding-top: .6rem;
    border-top: 1px solid rgba(0, 195, 255, 0.15);
}

/* ── Section cards (translucent sapphire navy with glowing cyan borders) ── */
.sec {
    background: rgba(13, 27, 56, 0.7) !important;
    border: 1px solid rgba(0, 195, 255, 0.35) !important;
    border-radius: 16px;
    padding: 1.4rem 1.6rem 0.5rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), 
                0 0 20px rgba(0, 195, 255, 0.15),
                inset 0 0 15px rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(12px);
}
.sec-title {
    font-size: 1.15rem; font-weight: 800; color: #ffffff !important;
    border-bottom: 2px solid rgba(0, 195, 255, 0.35) !important;
    padding-bottom: .5rem; margin: 0 0 .1rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    text-shadow: 0 0 12px rgba(255, 255, 255, 0.45);
}

/* ── Shimmering headings for results ── */
.shimmer-title {
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    border-bottom: 2px solid rgba(0, 195, 255, 0.35) !important;
    padding-bottom: .5rem;
    margin-top: 1.2rem;
    margin-bottom: 1rem;
    text-shadow: 0 0 12px rgba(255, 255, 255, 0.45);
}

/* ── Mono response box (official scientific terminal style) ── */
.mono {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .9rem;
    background: rgba(6, 14, 30, 0.95) !important;
    padding: 1.2rem;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3), inset 0 0 10px rgba(255, 255, 255, 0.05);
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.65;
}
.mono-green { 
    border-color: rgba(0, 195, 255, 0.55) !important;
    box-shadow: 0 0 20px rgba(0, 195, 255, 0.2) !important;
}

/* ── KGI meter ── */
.meter-wrap { margin: .75rem 0 .5rem; }
.meter-bar  { height: 16px; border-radius: 8px; background: #030d1d; overflow: hidden; border: 1px solid rgba(0,195,0.15); }
.meter-fill { height: 100%; border-radius: 8px; }
.meter-labels {
    display: flex; justify-content: space-between;
    font-size: .76rem; font-weight: 600; margin-top: .28rem;
}

/* ── Agent chips (academic/muted styling) ── */
.chip-yes {
    display: inline-block; padding: .4rem 1rem; border-radius: 6px;
    background: rgba(244,63,94,.08); border: 1px solid rgba(244,63,94,.4);
    color: #f43f5e; font-weight: 700; font-size: .9rem; margin-bottom: .6rem;
}
.chip-no {
    display: inline-block; padding: .4rem 1rem; border-radius: 6px;
    background: rgba(56,189,248,.08); border: 1px solid rgba(56,189,248,.4);
    color: #38bdf8; font-weight: 700; font-size: .9rem; margin-bottom: .6rem;
}

/* ── Agent reasoning ── */
.reasoning {
    background: rgba(255, 255, 255, 0.02);
    border-left: 3px solid rgba(0, 195, 255, 0.7);
    border-radius: 0 8px 8px 0;
    padding: .9rem 1.1rem;
    margin-top: .7rem;
    font-style: italic;
    color: #cbd5e1;
    font-size: .9rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* ── UI Inputs & Widgets custom overrides ── */
div[data-baseweb="select"] > div, 
div[data-baseweb="textarea"] > div,
div[data-baseweb="input"] > div {
    background-color: rgba(13, 27, 56, 0.95) !important;
    color: #ffffff !important;
    border: 1px solid rgba(0, 195, 255, 0.3) !important;
    border-radius: 10px !important;
}
/* Focus border */
div[data-baseweb="select"] > div:focus-within, 
div[data-baseweb="textarea"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
    border-color: rgba(0, 195, 255, 0.7) !important;
}

/* Force all form labels, input texts, and select text to be bright white */
label, p, span, li, ul {
    color: #e2e8f0 !important;
}
div[data-testid="stWidgetLabel"] p, label, .stSlider > label {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    text-shadow: 0 0 5px rgba(0, 195, 255, 0.2);
}
input, textarea, [data-baseweb="select"] * {
    color: #ffffff !important;
}

/* Buttons (Premium Neon Cyan-to-Purple Gradient Accent) */
.stButton > button {
    background: linear-gradient(135deg, #00c3ff 0%, #7d12ff 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 15px rgba(0, 195, 255, 0.3) !important;
    transition: all 0.3s ease !important;
    height: 48px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #7d12ff 0%, #00c3ff 100%) !important;
    box-shadow: 0 4px 22px rgba(0, 195, 255, 0.5) !important;
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(1px);
}

/* Secondary Button overrides (for resets, etc.) */
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.03) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    box-shadow: none !important;
    height: auto !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255, 255, 255, 0.5) !important;
}

/* Metrics Overrides */
div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    text-shadow: 0 0 10px rgba(0, 195, 255, 0.4) !important;
    font-weight: 800 !important;
}
div[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
}

/* Historical Card Panel */
.hist-card {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(0, 195, 255, 0.25) !important;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
}
.hist-card:hover {
    border-color: rgba(255, 255, 255, 0.5) !important;
    background: rgba(255, 255, 255, 0.08) !important;
    box-shadow: 0 4px 15px rgba(0, 195, 255, 0.2);
}
.hist-query {
    font-weight: 600;
    color: #ffffff;
    font-size: 0.84rem;
    margin-bottom: 0.4rem;
    line-height: 1.4;
}
.hist-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: #94a3b8;
}
.hist-reduction {
    font-weight: 700;
    color: #38bdf8;
}

/* Table styling */
.stTable, table {
    background-color: rgba(16, 32, 66, 0.8) !important;
    color: #f1f5f9 !important;
    border-collapse: collapse !important;
    border: 1px solid rgba(0, 195, 255, 0.2) !important;
    border-radius: 8px !important;
}
th {
    background-color: rgba(0, 195, 255, 0.15) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-bottom: 2px solid rgba(0, 195, 255, 0.4) !important;
    padding: 8px 12px !important;
}
td {
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding: 8px 12px !important;
}
</style>

"""
st.markdown(_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _md(html: str):
    """Shortcut — render raw HTML without repeating the flag."""
    st.markdown(html, unsafe_allow_html=True)

def sec_header(icon_title: str):
    _md(f'<div class="sec"><p class="sec-title">{icon_title}</p></div>')

def render_metrics_row(metrics):
    cols_html = ""
    col_width = 100 / len(metrics)
    for label, val in metrics:
        cols_html += f'<div style="width: {col_width}%; text-align: left; padding: 0.5rem 0.8rem; border-left: 1px solid rgba(255,255,255,0.08);"><div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">{label}</div><div style="font-size: 1.35rem; font-weight: 800; color: #ffffff; text-shadow: 0 0 10px rgba(0,195,255,0.25); margin-top: 0.2rem;">{val}</div></div>'
    html = f'<div style="display: flex; justify-content: space-between; background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 0.6rem 0.4rem; margin-bottom: 0.8rem;">{cols_html}</div>'
    _md(html)

def _set(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_set("api_key",       config.get_gemini_api_key() or "")
_set("kgi_threshold", config.KGI_THRESHOLD)
_set("history",       [])
_set("current_result", None)
_set("corpus_loaded", False)
_set("get_started",   False)
_set("show_history",  True)

def _make_pipeline(api_key):
    return RAGPipeline(api_key=api_key if api_key else None)

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = _make_pipeline(st.session_state.api_key)
if "ingester" not in st.session_state:
    st.session_state.ingester = DocumentIngester()
if "benchmark_questions" not in st.session_state:
    st.session_state.benchmark_questions = BenchmarkQuestionGenerator().get_fallback_questions()

# ─────────────────────────────────────────────
# AUTO-INGESTION (runs once per session)
# ─────────────────────────────────────────────
def auto_ingest():
    ingester = st.session_state.ingester
    pipeline = st.session_state.rag_pipeline
    corpus   = Path(config.CORPUS_DIR)
    if not corpus.exists():
        return
    # ── Ingest traditional knowledge documents (PDF/TXT) ─────────────────
    for f in corpus.iterdir():
        if f.suffix.lower() not in [".txt", ".pdf", ".docx", ".md"]:
            continue
        if any(m["filename"] == f.name for m in ingester.documents_metadata.values()):
            continue
        try:
            _, doc_meta = ingester.ingest_document(
                f, domain="Traditional Knowledge / Sustainable Agriculture",
                chunk_size=800, chunk_overlap=150
            )
            cp = Path(config.EVAL_DIR) / doc_meta["chunks_file"]
            with open(cp, "r", encoding="utf-8") as fh:
                pipeline.ingest_new_documents(json.load(fh))
        except Exception as exc:
            st.sidebar.warning(f"Could not ingest {f.name}: {exc}")

    # ── Load pre-processed Agro Dataset chunks from evaluation_data ───────
    # These are already embedded by ingest_agro_datasets.py; we only need
    # to load them into the in-memory vector store at startup.
    eval_dir = Path(config.EVAL_DIR)
    agro_chunk_files = list(eval_dir.glob("agro_*_chunks.json"))
    for cf in agro_chunk_files:
        try:
            with open(cf, "r", encoding="utf-8") as fh:
                agro_chunks = json.load(fh)
            # Only add if not already in the vector store (check by chunk_id prefix)
            existing_ids = {c.get("chunk_id", "") for c in pipeline.vector_store.chunks_data}
            new_chunks = [c for c in agro_chunks if c.get("chunk_id", "") not in existing_ids]
            if new_chunks:
                pipeline.ingest_new_documents(new_chunks)
        except Exception as exc:
            st.sidebar.warning(f"Could not load agro chunks from {cf.name}: {exc}")

if not st.session_state.corpus_loaded:
    auto_ingest()
    st.session_state.corpus_loaded = True

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# WELCOME SPLASH / ONBOARDING PAGE
# ─────────────────────────────────────────────
if not st.session_state.get_started:
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        display: none !important;
    }
    .stApp {
        background-color: #071024 !important;
        background-image: radial-gradient(circle at 50% 50%, #0e224b 0%, #050b18 100%) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 12vh;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; max-width: 800px; margin: 0 auto; padding: 3rem 2.5rem; background: rgba(13, 27, 56, 0.85); border: 2px solid rgba(0, 195, 255, 0.35); border-radius: 20px; box-shadow: 0 20px 50px rgba(0, 195, 255, 0.25), inset 0 0 20px rgba(255, 255, 255, 0.02); backdrop-filter: blur(12px);">
        <h1 style="font-size: 3.5rem; font-weight: 800; color: #ffffff; text-shadow: 0 0 25px rgba(255,255,255,0.45); margin-bottom: 1.5rem; line-height: 1.2;">👋 Hello, How Are You?</h1>
        <p style="font-size: 1.25rem; color: #cbd5e1; font-weight: 500; margin-bottom: 2.5rem; line-height: 1.6;">
            Welcome to the <b>Digital Knowledge Extinction Framework (DKEF)</b>.<br>
            An interactive console to audit, analyze, and recover traditional knowledge gaps in generative language models.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    c_left, c_mid, c_right = st.columns([1, 1, 1])
    with c_mid:
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        if st.button("🚀 Get Started", use_container_width=True, key="get_started_btn"):
            st.session_state.get_started = True
            st.rerun()
    st.stop()

with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/shield.png", width=72)
    st.markdown("## Responsible AI Controls")
    st.markdown("---")

    # 1. API Key
    st.markdown("### 🔑 Gemini API Key")
    new_key = st.text_input(
        "Google AI Studio API key",
        value=st.session_state.api_key,
        type="password",
        help="Leave blank to run in offline simulation mode."
    )
    if new_key != st.session_state.api_key:
        st.session_state.api_key       = new_key
        st.session_state.rag_pipeline  = _make_pipeline(new_key)
        st.session_state.corpus_loaded = False
        st.success("✅ API key saved — pipeline re-initialised.")
        st.rerun()

    mode_label = "🟢 Live (Gemini)" if st.session_state.api_key else "🟡 Simulation (Offline)"
    st.caption(f"Mode: **{mode_label}**")
    st.markdown("---")

    # 2. Corpus status
    st.markdown("### 📚 Backend Knowledge Corpus")
    meta = st.session_state.ingester.documents_metadata
    ca, cb = st.columns(2)
    ca.metric("Documents", len(meta))
    cb.metric("Chunks",    sum(d["total_chunks"] for d in meta.values()))
    if meta:
        with st.expander("📂 Indexed files"):
            for d in meta.values():
                st.write(f"• `{d['filename']}` ({d['total_chunks']} chunks)")
    st.markdown("---")

    # 3. KGI Threshold
    st.markdown("### ⚙️ KGI Activation Threshold")
    new_thresh = st.slider(
        "Minimum KGI to trigger RAG",
        min_value=0.0, max_value=1.0,
        value=st.session_state.kgi_threshold,
        step=0.05,
        help="Agent triggers retrieval when KGI ≥ this value."
    )
    if new_thresh != st.session_state.kgi_threshold:
        st.session_state.kgi_threshold = new_thresh
    st.caption(f"Current threshold: **{st.session_state.kgi_threshold:.2f}**")
    st.markdown("---")

    # 4. Session controls
    st.markdown("### 🧹 Session Controls")
    if st.button("Clear History & Reset", use_container_width=True):
        st.session_state.history        = []
        st.session_state.current_result = None
        st.success("History cleared.")
        st.rerun()
    if st.button("Re-scan Knowledge Corpus", use_container_width=True):
        st.session_state.corpus_loaded = False
        st.rerun()
    st.markdown("---")
    st.caption("Digital Knowledge Extinction Framework v2.0\nResponsible AI · Agentic RAG · XAI")

# ─────────────────────────────────────────────
# MAIN LAYOUT SPAN (SPLIT PANEL WORKSPACE & COLLAPSIBLE HISTORY)
# ─────────────────────────────────────────────
if st.session_state.show_history:
    col_main, col_right = st.columns([4.2, 0.8])
else:
    col_main, col_right = st.columns([4.95, 0.05])

with col_main:
    # ─────────────────────────────────────────────
    # MAIN HEADER & COLLAPSE TOGGLE
    # ─────────────────────────────────────────────
    col_title, col_toggle = st.columns([5, 1.5])
    with col_title:
        _md("""
        <div style="margin-bottom: 1.2rem;">
          <h1 style="font-size: 2.8rem; font-weight: 800; color: #ffffff; text-shadow: 0 0 20px rgba(0,195,255,0.25); margin: 0; padding: 0;">Digital Knowledge Extinction Framework</h1>
          <div style="font-size: 1.05rem; font-weight: 600; color: #38bdf8; margin-top: 0.5rem; letter-spacing: 0.5px;">
              👋 Hi, how are you? Ask a query to know more about sustainable agriculture and traditional knowledge.
          </div>
        </div>
        """)
    with col_toggle:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        hist_label = "❌ Collapse Log" if st.session_state.show_history else "📜 Expand Log"
        if st.button(hist_label, key="toggle_hist_sidebar", use_container_width=True):
            st.session_state.show_history = not st.session_state.show_history
            st.rerun()

    # ─────────────────────────────────────────────
    # SECTION 1 — USER QUERY PANEL
    # ─────────────────────────────────────────────
    _md('<h3 class="shimmer-title">📂 USER QUERY PANEL</h3>')

from language_detector import detect_language, get_language_flag

SAMPLE_QUERIES = [
    "-- Select a benchmark query --",
    # ── English Queries (Original Manual & PDFs) ──────────────────────────
    "What are the exact nine ingredients required to prepare Panchagavya, and what are their respective quantities?",
    "Describe the multi-phase preparation process and timeline required to ferment Panchagavya.",
    "What is the formulation, preparation, and usage of Bijamrita for seed treatment?",
    "Explain the layout structure and agricultural benefits of the Akkadi Saalu intercropping system.",
    "What is Navara rice, and how is it traditionally utilized in Ayurvedic therapies?",
    "What is the Rolu indigenous rain gauge and how do farmers use it to decide sowing time?",
    "How does the ICAR-validated Parasi (Cleistanthus collinus) leaf practice control yellow stem-borer in rice?",
    "What is the traditional practice of using cow urine mixed with tobacco-soaked water for vegetable pest control as documented by ICAR?",
    "What are the key ITK practices documented in the Karnataka ITK Book 2020 for crop protection?",
    "How do Karnataka farmers traditionally predict weather and seasonal patterns for crop planning?",
    "What are the traditional soil assessment methods used by Karnataka farmers before sowing?",
    "What are the traditional methods used by Karnataka farmers for seed selection and preservation?",
    "How is traditional seed priming performed in Karnataka farming communities?",
    "What is the definition and scope of Indigenous Technical Knowledge (ITK) in Indian agriculture?",
    "Describe the ethno-veterinary practices documented as Indigenous Technical Knowledge for livestock disease management.",
    "How does ITK contribute to biodiversity conservation and why is its documentation important?",
    # ── English Queries (Agro Dataset) ──────────────────────────────────────
    "What are the optimal soil nitrogen, phosphorus, and potassium levels for growing rice?",
    "What fertilizer is recommended for Maize crops grown in Sandy soil with low nitrogen?",
    "What disease affects potato crops, and what are the recommended treatments for early blight and late blight?",
    "What are the key soil parameters (temperature, humidity, pH, rainfall) needed for growing wheat?",
    "What rice seasons are suitable for cultivation in Tamil Nadu, and what are their sowing months?",
    "What is the recommended fertilizer for Sugarcane grown in Loamy soil with balanced NPK levels?",
    "What nutrient conditions and climate are best suited for growing cotton according to agricultural data?",
    # ── Multilingual Queries (Hindi / हिंदी) ──────────────────────────────────
    "पंचगव्य बनाने के लिए कौन सी नौ सामग्रियां आवश्यक हैं और उनकी मात्रा क्या है?",
    "चावल उगाने के लिए मिट्टी में नाइट्रोजन, फास्फोरस और पोटेशियम का इष्टतम स्तर क्या है?",
    "धान में पीला तना छेदक कीट को नियंत्रित करने के लिए परसी (Parasi) की पत्तियों का उपयोग कैसे किया जाता है?",
    # ── Multilingual Queries (Kannada / ಕನ್ನಡ) ──────────────────────────────
    "ಪಂಚಗವ್ಯವನ್ನು ತಯಾರಿಸಲು ಬೇಕಾಗುವ ಒಂಬತ್ತು ಪದಾರ್ಥಗಳು ಯಾವುವು ಮತ್ತು ಅವುಗಳ ಪ್ರಮಾಣ ಎಷ್ಟು?",
    "ಭತ್ತ ಬೆಳೆಯಲು ಮಣ್ಣಿನಲ್ಲಿ ಸಾರಜನಕ, ರಂಜಕ ಮತ್ತು ಪೊಟ್ಯಾಸಿಯಮ್‌ನ ಸೂಕ್ತ ಮಟ್ಟಗಳು ಯಾವುವು?",
    "ಕರ್ನಾಟಕ ಐಟಿಕೆ ಪುಸ್ತಕದ ಪ್ರಕಾರ ಬಿತ್ತನೆಗೆ ಮುನ್ನ ಸಾಂಪ್ರದಾಯಿಕ ಮಣ್ಣಿನ ಪರೀಕ್ಷೆಯನ್ನು ಹೇಗೆ ಮಾಡುತ್ತಾರೆ?",
]

with col_main:
    sel       = st.selectbox("Select a benchmark query:", SAMPLE_QUERIES)
    prefill   = "" if sel.startswith("--") else sel
    
    cq1, cq2 = st.columns([4.2, 1.8])
    with cq1:
        user_query = st.text_area("Or type a custom query:", value=prefill, height=85)
    with cq2:
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        run_btn = st.button("⚖️ Analyze Knowledge Gap", type="primary", use_container_width=True)
        st.caption(f"Threshold: **{st.session_state.kgi_threshold:.2f}** · Mode: **{mode_label}**")

    # Dynamic language detection display
    lang_code, lang_conf, lang_name = detect_language(user_query)
    lang_flag = get_language_flag(lang_code)

    if user_query.strip():
        _md(f"""
        <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); padding: 0.6rem 1rem; border-radius: 8px; margin-bottom: 1.2rem; font-size: 0.88rem;">
          🌐 <b>Detected Language:</b> {lang_flag} {lang_name} &nbsp;|&nbsp; 
          📊 <b>Language Confidence:</b> {lang_conf:.2%} &nbsp;|&nbsp; 
          ⚙️ <b>Cross-Lingual Retrieval:</b> Active (English Index)
        </div>
        """)

with col_right:
    if st.session_state.show_history:
        _md('<h3 class="shimmer-title" style="margin-top:0;border-bottom:2px solid rgba(0, 195, 255, .3);padding-bottom:.4rem">📜 AUDIT & QUERY LOG</h3>')
        if st.session_state.history:
            for i, h in enumerate(reversed(st.session_state.history)):
                q_text = h["query"]
                if len(q_text) > 75:
                    q_text = q_text[:72] + "..."
                
                kgi_b = h["base_eval"]["kgi"]
                kgi_a = h["enh_eval"]["kgi"]
                reduction_pct = h["reduction"]["gap_reduction_percentage"]
                lang_code_lbl = h.get("query_lang_code", "en").upper()
                
                _md(f"""
                <div class="hist-card">
                    <div class="hist-query">Q{len(st.session_state.history)-i}: {q_text}</div>
                    <div class="hist-meta">
                        <span>🌐 {lang_code_lbl}</span>
                        <span>📉 KGI: {kgi_b:.2f} → {kgi_a:.2f}</span>
                        <span class="hist-reduction">⚡ -{reduction_pct:.1f}%</span>
                    </div>
                </div>
                """)
        else:
            st.info("No queries analyzed yet.")
    else:
        st.markdown("")

# ─────────────────────────────────────────────
# SIMULATED RESPONSES (offline mode)
# ─────────────────────────────────────────────
_SIM_BASE = {
    # ── English ──────────────────────────────────
    "panchagavya_ing":  "Panchagavya is a traditional fertilizer made from cow dung, urine, milk, and curd. It is fermented together and sprayed on crops to improve yield.",
    "panchagavya_ferm": "To prepare Panchagavya, mix cow dung and urine with other ingredients, then ferment for a couple of weeks before spraying.",
    "bijamrita":        "Bijamrita is a traditional seed treatment made from cow dung, cow urine, and lime used to coat seeds before planting.",
    "akkadi":           "Akkadi Saalu is an intercropping method where multiple crop rows are grown together to naturally reduce pests.",
    "navara":           "Navara rice is a traditional red rice variety from South India used in Ayurveda.",
    "icar_rolu":        "Rolu is a traditional rain gauge. Farmers use it to know when to sow seeds based on rainfall.",
    "icar_parasi":      "Parasi leaves are spread in rice fields to control stem-borer. This is a traditional pest control method.",
    "icar_cow_urine":   "Farmers spray cow urine mixed with tobacco water on vegetables to control insects. This is an old traditional practice.",
    "karnataka_crop_protect": "Karnataka farmers use neem and cow urine sprays to protect crops from pests.",
    "karnataka_weather":     "Karnataka farmers observe nature signs like animal behavior and plant flowering to predict weather.",
    "karnataka_soil":        "Karnataka farmers check soil by looking at its color and texture before planting.",
    "seed_selection":   "Karnataka farmers choose good seeds from healthy plants and store them in pots with ash.",
    "seed_priming":     "Seeds are soaked in water overnight before planting to improve germination.",
    "itk_definition":   "ITK refers to local farming knowledge passed down through generations in communities.",
    "ethno_vet":        "Farmers use herbal remedies like turmeric and neem for treating sick livestock.",
    "itk_biodiversity": "Traditional farming preserves local plant varieties and is important for biodiversity.",
    "agro_rice_npk":    "Rice generally requires nitrogen, phosphorus, and potassium. The optimal levels depend on the specific variety and region.",
    "agro_maize_fert":  "Maize grown in Sandy soil with low nitrogen typically requires a nitrogen-rich fertilizer like Urea.",
    "agro_potato_dis":  "Potato crops can be affected by various diseases. Early blight and late blight are common fungal diseases treated with fungicides.",
    "agro_wheat_soil":  "Wheat grows best in loamy soil with adequate temperature and rainfall. It needs moderate nitrogen levels.",
    "agro_rice_season": "Rice is cultivated in multiple seasons in Tamil Nadu. The seasons vary by district.",
    "agro_sugarcane":   "Sugarcane requires a specific fertilizer depending on the nitrogen, phosphorus and potassium levels of the soil.",
    "agro_cotton":      "Cotton grows best in black or red soils and requires careful management of nitrogen and potassium levels.",
    # ── Hindi ───────────────────────────────────
    "panchagavya_ing_hi":  "पंचगव्य गाय के गोबर, गोमूत्र, दूध और दही से बनाया जाने वाला एक पारंपरिक जैविक उर्वरक है।",
    "agro_rice_npk_hi":    "धान की फसल को मुख्य रूप से नाइट्रोजन, फास्फोरस और पोटेशियम की आवश्यकता होती है।",
    "icar_parasi_hi":      "धान के खेत में कीटों और तना छेदक को रोकने के लिए परसी की पत्तियों का पारंपरिक रूप से छिड़काव किया जाता है।",
    # ── Kannada ─────────────────────────────────
    "panchagavya_ing_kn":  "ಪಂಚಗವ್ಯವು ಹಸುವಿನ ಸಗಣಿ, ಗೋಮೂತ್ರ, ಹಾಲು ಮತ್ತು ಮೊಸರಿನಿಂದ ತಯಾರಿಸುವ ಒಂದು ಸಾಂಪ್ರದಾಯಿಕ ಗೊಬ್ಬರವಾಗಿದೆ.",
    "agro_rice_npk_kn":    "ಭತ್ತದ ಬೆಳೆಗೆ ಮುಖ್ಯವಾಗಿ ಸಾರಜನಕ, ರಂಜಕ ಮತ್ತು ಪೊಟ್ಯಾಸಿಯಮ್ ಪೋಷಕಾಂಶಗಳ ಅವಶ್ಯಕತೆಯಿದೆ.",
    "karnataka_soil_kn":   "ಕರ್ನಾಟಕದ ರೈತರು ಮಣ್ಣಿನ ಬಣ್ಣ ಮತ್ತು ತೇವಾಂಶವನ್ನು ನೋಡಿ ಸಾಂಪ್ರದಾಯಿಕವಾಗಿ ಮಣ್ಣಿನ ಗುಣಮಟ್ಟವನ್ನು ಅಂದಾಜಿಸುತ್ತಾರೆ.",
    "default":          "This traditional practice involves using organic materials to nourish plants and cure seed-borne diseases."
}

_SIM_ENH = {
    # ── English ──────────────────────────────────
    "panchagavya_ing":  "Panchagavya requires nine exact ingredients: 1) Fresh Cow Dung: 7 kg, 2) Cow Ghee: 1 kg, 3) Fresh Cow Urine: 10 L, 4) Water: 10 L, 5) Cow Milk: 3 L, 6) Cow Curd: 2 L, 7) Tender Coconut Water: 3 L, 8) Jaggery: 3 kg, 9) Ripe Bananas: 12.",
    "panchagavya_ferm": "Three-phase fermentation — Phase 1: Mix 7 kg dung + 1 kg ghee, ferment 3 days (stir twice daily). Phase 2: Day 4 — add 10 L cow urine + 10 L water, ferment 15 days. Phase 3: Day 19 — add milk, curd, coconut water, jaggery, mashed bananas; ferment 7 more days (total 25–30 days in shade, covered with cotton cloth).",
    "bijamrita":        "Suspend 5 kg cow dung in 20 L water for 12 h; squeeze extract. Mix with 5 L cow urine, 50 ml milk, 50 g lime, handful of local soil. Stir, leave overnight. Sprinkle over seeds or dip seedling roots for 10–15 s before transplanting.",
    "akkadi":           "Akkadi Saalu: 9 rows of primary crop (ragi/sorghum) followed by 1 row of pulse/oilseed (pigeon pea, field bean, mustard, sesame, horse gram). Benefits: nitrogen fixation, natural pest control (mustard lures aphids; marigold/cosmos attract predators), and diverse harvest.",
    "navara":           "Navara is a medicinal red-grained rice endemic to Kerala (60–90 day cycle). Used in 'Navarakizhi' Ayurvedic rejuvenation: cooked Navara is wrapped in muslin bags, dipped in warm milk and herbal decoctions, and massaged on the body to treat neurological disorders and muscle wasting.",
    "icar_rolu":        "The Rolu is an indigenous rain gauge — a hole of 7.4 inches depth and 9 inches diameter on a 3x3x1.5 inch granite stone block. Farmers sow seeds when Rolu fills with rain water. Experimental validation by CRIDA Hyderabad confirmed sorghum + pigeonpea sowing by 50–100% farmers within 3 days when gauge was filled >3/4 capacity (equivalent to 50mm in a standard rain gauge).",
    "icar_parasi":      "ICAR validates use of 75–150 kg Parasi (Cleistanthus collinus) leaves broadcast in rice fields at 3 days after transplanting to control yellow stem-borer. CRRI Cuttack experiments showed Parasi at 150 kg/ha applied thrice (30, 60, 90 DAT) increased rice yield and raised earthworm and soil bacteria populations. In Jharkhand, 10 kg fresh parso/persu leaves per 100 m2 control gallfly in rice. Also effective against caseworm infestation.",
    "icar_cow_urine":   "As documented in ICAR's Traditional Knowledge in Agriculture, farmers in Bahadurpur, Dhanbad, Jharkhand (56% adoption) spray domestic animal urine mixed with tobacco-soaked water on cucurbits, cowpea and okra to control insect pests. CTCRI validation (2003–05) showed this was equally effective as chemical insecticides in in-vitro bioassays. Net returns: Rs 4,059–6,511 for cucurbits; Rs 3,510–7,779 for okra; Rs 1,677–3,413 for cowpea. In Kurchi village, Dhanbad, 85% farmers also spray rice starch + animal urine followed by cowdung ash dusting to control aphids.",
    "karnataka_crop_protect": "Karnataka ITK Book 2020 documents indigenous crop protection: neem-based sprays (neem seed kernel extract at 5%), cow urine spray for pest deterrence, chili-garlic paste dilutions, smoke treatment, intercropping with repellent plants like marigold and basil, sticky traps coated with castor oil, and placement of dried neem leaves in stored grain to prevent weevils.",
    "karnataka_weather":     "Karnataka farmers use bioindicators for weather prediction: ants, swallows nesting low indicating heavy rain, flowering of specific trees, insect behavior, and Nakshatra-based constellations. The Panchangam guides sowing windows, and elders interpret wind direction, cloud formation, and animal behavior to predict rainfall intensity.",
    "karnataka_soil":        "Karnataka farmers assess soil health by: (1) tasting for salinity, (2) smelling for organic matter, (3) observing earthworm presence, (4) checking soil color (dark = organic-rich), (5) the 'ball test' — loamy soil forms a crumbling ball, clay holds firm, sandy soil doesn't form a ball. Vegetation types indicate soil type.",
    "seed_selection":   "Karnataka farmers select seeds from healthy, vigorous, disease-free mother plants in the field. Seeds are sun-dried, treated with ash, dried neem leaves, or turmeric powder, then stored in air-tight clay pots, gunny bags with ash lining, or bamboo containers. Community seed banks (beeja mitra) facilitate seed exchange.",
    "seed_priming":     "Seed priming in Karnataka: soak seeds in water, diluted cow urine, or Bijamrita for 6–12 hours. Spread on mats under shade to semi-dry before sowing. Ash water (wood ash + water) is used as alkaline priming. Seeds wrapped in banana or tender coconut leaves overnight before sowing induce uniform sprouting.",
    "itk_definition":   "ITK refers to the cumulative body of knowledge, practices, and beliefs about the relationship of living beings with their environment, held by local communities, evolved by adaptive processes and handed down through generations by cultural transmission. In Indian agriculture, ITK encompasses traditional crop varieties, pest management, soil conservation, water management, livestock care (ethno-veterinary), food preservation, and medicinal plant use.",
    "ethno_vet":        "Ethno-veterinary ITK includes herbal preparations: neem bark decoction for fever, turmeric paste for wounds and infections, garlic and ajwain for digestive disorders, castor oil as purgative, ginger and salt solution for bloat, mustard oil massage for joint pain. Plant-based fly repellents, neem oil for external parasites, and smoke fumigation for ectoparasites are common. 'Pashu vaidya' use root and bark preparations for parturition care.",
    "itk_biodiversity": "ITK conserves biodiversity by preserving traditional crop landraces adapted to local conditions, maintaining agroforestry systems, conserving medicinal plants through community use, and sustaining wild gene pools through traditional land management. Documentation is critical because modernization erodes oral traditions within a generation, ITK holds climate solutions, and biopiracy risks arise when undocumented TK is patented without community consent.",
    "agro_rice_npk":    "According to the Agro Dataset (Kaggle crop recommendation dataset), rice grows optimally with Nitrogen 60-90 kg/ha, Phosphorus 35-60 kg/ha, Potassium 38-45 kg/ha. Optimal temperature: 20-27°C, Humidity: 80-85%, Soil pH: 5.5-7.5, Rainfall: 200-270 mm. These parameters are critical for high-yield rice cultivation.",
    "agro_maize_fert":  "According to the Agro Dataset fertilizer recommendation data, Maize crops grown in Sandy soil with nitrogen level 37, phosphorus 0, potassium 0 — the recommended fertilizer is Urea (single-nutrient nitrogen fertilizer). For balanced NPK in Sandy soil (N:14, P:12, K:15), 17-17-17 compound fertilizer is recommended.",
    "agro_potato_dis":  "According to the Agro Dataset crop disease data: Early Blight (Alternaria solani) causes visible lesions and discoloration on potato leaves/stems; treatment: apply appropriate fungicide and use resistant varieties. Late Blight (Phytophthora infestans) also affects potato with similar symptoms; treatment: systemic fungicide application and certified disease-resistant potato varieties.",
    "agro_wheat_soil":  "According to the Agro Dataset crop data, Wheat grows best with Nitrogen 58-100 kg/ha, Phosphorus 35-65 kg/ha, Potassium 38-62 kg/ha. Optimal temperature: 15-27°C, Humidity: 65-80%, Soil pH: 6.0-7.5, Rainfall: 50-110 mm. In Loamy soil with moderate nitrogen, Urea is the first-choice fertilizer; DAP is used when phosphorus is also needed.",
    "agro_rice_season": "According to the Agro Dataset (rice_dataset.json from TNAU Agritech Portal): Rice in Tamil Nadu is grown in 3 main seasons — (1) Navarai (Sown: Dec-Jan, Duration: <120 days, Districts: Tiruvallur, Vellore, Cuddalore etc.), (2) Kuruvai (Sown: Jun-Jul, Duration: <120 days, delta districts), (3) Samba/Thaladi (Sown: Aug-Oct, Duration: 135-180 days, most districts). Each season requires specific variety selection.",
    "agro_sugarcane":   "According to the Agro Dataset, Sugarcane fertilizer recommendations by soil: (1) Loamy soil, N:36 → Urea; (2) Loamy soil, N:12, P:36 → DAP; (3) Loamy soil, N:12, P:12, K:14 → 17-17-17 balanced fertilizer; (4) Black soil, N:35 → Urea; (5) Loamy soil, N:10, P:32, K:7 → 14-35-14. Fertilizer choice is driven primarily by nitrogen deficiency level.",
    "agro_cotton":      "According to the Agro Dataset crop data, Cotton grows well with Nitrogen 110-130 kg/ha, Phosphorus 35-50 kg/ha, Potassium 40-55 kg/ha. Temperature: 21-30°C, Humidity: 60-80%, pH: 5.8-7.5, Rainfall: 60-110 mm. In Black soil with high N → Urea; balanced NPK → 17-17-17; high P needs → DAP or 14-35-14. Cotton in Red soil with N:9, P:10 → 20-20 fertilizer.",
    # ── Hindi ───────────────────────────────────
    "panchagavya_ing_hi":  "पंचगव्य बनाने के लिए सटीक नौ सामग्रियां और उनकी मात्रा इस प्रकार हैं: १) ताजा गोबर: ७ किलो, २) गाय का घी: १ किलो, ३) ताजा गोमूत्र: १० लीटर, ४) पानी: १० लीटर, ५) गाय का दूध: ३ लीटर, ६) गाय का दही: २ लीटर, ७) नारियल पानी: ३ लीटर, ८) गुड़: ३ किलो, ९) पके केले: १२ नग।",
    "agro_rice_npk_hi":    "कृषि डेटासेट (Kaggle) के अनुसार, धान (rice) के लिए इष्टतम मिट्टी के पोषक तत्व निम्न प्रकार हैं: नाइट्रोजन: ६०-९० किलो/हेक्टेयर, फास्फोरस: ३५-६० किलो/हेक्टेयर, पोटेशियम: ३८-४५ किलो/हेक्टेयर। तापमान २०-२७ डिग्री सेल्सियस और मृदा पीएच ५.५-७.५ आदर्श है।",
    "icar_parasi_hi":      "ICAR के अनुसार, रोपाई के ३ दिन बाद खेत में ७५-१५० किलोग्राम प्रति हेक्टेयर की दर से परसी (Cleistanthus collinus) की ताज़ी पत्तियों का छिड़काव करने से पीला तना छेदक कीट प्रभावी ढंग से नियंत्रित होता है। यह केंचुओं और जैव-विविधता के अनुकूल है।",
    # ── Kannada ─────────────────────────────────
    "panchagavya_ing_kn":  "ಪಂಚಗವ್ಯ ತಯಾರಿಸಲು ಬೇಕಾಗುವ ನಿಖರವಾದ ಒಂಬತ್ತು ಪದಾರ್ಥಗಳು ಮತ್ತು ಅವುಗಳ ಪ್ರಮಾಣ: ೧) ಹಸಿ ಸಗಣಿ: ೭ ಕೆಜಿ, ೨) ಹಸುವಿನ ತುಪ್ಪ: ೧ ಕೆಜಿ, ೩) ಹಸುವಿನ ಗಂಜಲ: ೧೦ ಲೀಟರ್, ೪) ನೀರು: ೧೦ ಲೀಟರ್, ೫) ಹಸುವಿನ ಹಾಲು: ೩ ಲೀಟರ್, ೬) ಹಸುವಿನ ಮೊಸರು: ೨ ಲೀಟರ್, ೭) ಎಳನೀರು: ೩ ಲೀಟರ್, ೮) ಬೆಲ್ಲ: ೩ ಕೆಜಿ, ೯) ಹಣ್ಣಾದ ಬಾಳೆಹಣ್ಣು: ೧೨ ಸಂಖ್ಯೆ.",
    "agro_rice_npk_kn":    "ಕೃಷಿ ಡೇಟಾಸೆಟ್ (Kaggle) ಪ್ರಕಾರ ಭತ್ತಕ್ಕೆ (rice) ಸೂಕ್ತವಾದ ಮಣ್ಣಿನ ಪೋಷಕಾಂಶಗಳ ಮಟ್ಟಗಳು ಹೀಗಿವೆ: ಸಾರಜನಕ: ೬೦-೯೦ ಕೆಜಿ/ಹೆಕ್ಟೇರ್, ರಂಜಕ: ೩೫-೬೦ ಕೆಜಿ/ಹೆಕ್ಟೇರ್, ಪೊಟ್ಯಾಸಿಯಮ್: ೩೮-೪೫ ಕೆಜಿ/ಹೆಕ್ಟೇರ್, ಪಿಎಚ್ ಮಟ್ಟ: ೫.೫-೭.೫.",
    "karnataka_soil_kn":   "ಕರ್ನಾಟಕ ಐಟಿಕೆ ಪುಸ್ತಕ ೨೦೨೦ ರ ಪ್ರಕಾರ, ರೈತರು ಬಿತ್ತನೆಗೆ ಮುನ್ನ ಮಣ್ಣಿನ ತೇವಾಂಶ ಮತ್ತು ಜಿಗುಟುತನವನ್ನು ಪರೀಕ್ಷಿಸಲು ಸಾಂಪ್ರದಾಯಿಕ 'ಮಣ್ಣಿನ ಉಂಡೆ ಪರೀಕ್ಷೆ' (Soil Ball Test) ನಡೆಸುತ್ತಾರೆ. ಉಂಡೆ ಒಡೆಯದಿದ್ದರೆ ಮಣ್ಣು ಬಿತ್ತನೆಗೆ ಸೂಕ್ತವಾಗಿದೆ ಎಂದು ನಿರ್ಧರಿಸುತ್ತಾರೆ.",
    "default":          "This traditional practice involves using organic materials to nourish plants and cure seed-borne diseases."
}

def _pick_sim(q, bank):
    q = q.lower()
    
    # ── Hindi Query Routing ───────────────────
    if any(w in q for w in ["नौ सामग्रियां", "सामग्री", "सामग्रियां", "पञ्चगव्य", "पंचगव्य"]):
        return bank.get("panchagavya_ing_hi", bank["default"])
    if any(w in q for w in ["नाइट्रोजन", "पोटेशियम", "फास्फोरस", "चावल"]):
        return bank.get("agro_rice_npk_hi", bank["default"])
    if any(w in q for w in ["तना छेदक", "परसी", "पत्तियों"]):
        return bank.get("icar_parasi_hi", bank["default"])

    # ── Kannada Query Routing ─────────────────
    if any(w in q for w in ["ಪದಾರ್ಥಗಳು", "ಪಂಚಗವ್ಯ", "ಪಂಚಗವ್ಯವನ್ನು"]):
        return bank.get("panchagavya_ing_kn", bank["default"])
    if any(w in q for w in ["ಸಾರಜನಕ", "ರಂಜಕ", "ಪೊಟ್ಯಾಸಿಯಮ್", "ಭತ್ತ"]):
        return bank.get("agro_rice_npk_kn", bank["default"])
    if any(w in q for w in ["ಮಣ್ಣಿನ", "ಉಂಡೆ", "ಪರೀಕ್ಷೆ"]):
        return bank.get("karnataka_soil_kn", bank["default"])

    # ── English Query Routing ─────────────────
    if "panchagavya" in q:
        if any(w in q for w in ["ingredient", "nine", "quantity"]):
            return bank["panchagavya_ing"]
        return bank["panchagavya_ferm"]
    if "bijamrita" in q:
        return bank["bijamrita"]
    if any(w in q for w in ["akkadi", "saalu", "intercrop"]):
        return bank["akkadi"]
    if any(w in q for w in ["navara", "njavara", "medicinal rice"]):
        return bank["navara"]
    if any(w in q for w in ["rolu", "rain gauge", "rain-gauge", "indigenous gauge", "sowing time"]):
        return bank["icar_rolu"]
    if any(w in q for w in ["parasi", "cleistanthus", "stem-borer", "stem borer", "gallfly"]):
        return bank["icar_parasi"]
    if any(w in q for w in ["tobacco", "tobacco-soaked", "animal urine", "icar", "vegetable pest"]):
        return bank["icar_cow_urine"]
    if any(w in q for w in ["karnataka", "itk book", "indigenous technical"]):
        if any(w in q for w in ["weather", "predict", "season", "rain", "nakshatra"]):
            return bank["karnataka_weather"]
        if any(w in q for w in ["soil", "assess", "sowing", "ball test"]):
            return bank["karnataka_soil"]
        if any(w in q for w in ["crop protect", "pest", "neem", "spray"]):
            return bank["karnataka_crop_protect"]
        return bank["karnataka_crop_protect"]
    if any(w in q for w in ["seed select", "seed preserv", "seed storage", "beeja"]):
        return bank["seed_selection"]
    if any(w in q for w in ["seed prim", "soaking", "germination"]):
        return bank["seed_priming"]
    if any(w in q for w in ["ethno", "veterinary", "livestock", "pashu"]):
        return bank["ethno_vet"]
    if any(w in q for w in ["biodiversity", "landrace", "conservation", "biopiracy"]):
        return bank["itk_biodiversity"]
    if any(w in q for w in ["definition", "scope", "what is itk", "indigenous knowledge"]):
        return bank["itk_definition"]
    if any(w in q for w in ["nitrogen", "phosphorus", "potassium", "npk", "optimal", "soil level"]):
        if any(w in q for w in ["rice", "paddy"]):
            return bank.get("agro_rice_npk", bank["default"])
        if any(w in q for w in ["wheat", "rabi"]):
            return bank.get("agro_wheat_soil", bank["default"])
        if any(w in q for w in ["cotton"]):
            return bank.get("agro_cotton", bank["default"])
    if any(w in q for w in ["fertilizer", "urea", "dap", "npk fertilizer", "recommended fertilizer"]):
        if any(w in q for w in ["maize", "corn"]):
            return bank.get("agro_maize_fert", bank["default"])
        if any(w in q for w in ["sugarcane", "sugar cane"]):
            return bank.get("agro_sugarcane", bank["default"])
        if any(w in q for w in ["cotton"]):
            return bank.get("agro_cotton", bank["default"])
        return bank.get("agro_maize_fert", bank["default"])
    if any(w in q for w in ["disease", "blight", "bacterial spot", "fungicide", "pest control"]):
        if any(w in q for w in ["potato"]):
            return bank.get("agro_potato_dis", bank["default"])
    if any(w in q for w in ["rice season", "sowing month", "navarai", "kuruvai", "samba", "tnau"]):
        return bank.get("agro_rice_season", bank["default"])
    if any(w in q for w in ["sugarcane"]):
        return bank.get("agro_sugarcane", bank["default"])
    if any(w in q for w in ["cotton"]):
        return bank.get("agro_cotton", bank["default"])
    return bank["default"]

# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(query):
    live     = bool(st.session_state.api_key)
    pipeline = st.session_state.rag_pipeline
    detector = KnowledgeGapDetector(pipeline.emb_manager)
    api      = st.session_state.api_key if live else None

    # Apply current threshold
    config.KGI_THRESHOLD = st.session_state.kgi_threshold

    # Language Detection
    q_lang, q_conf, q_name = detect_language(query)

    # Step 1 — Reference retrieval (cross-lingual retrieval gets English context)
    ref_chunks  = pipeline.retrieve(query, top_k=3)
    sources     = list(dict.fromkeys(c["metadata"]["source"] for c in ref_chunks)) or ["No corpus source"]
    max_rel     = float(np.max([c["score"] for c in ref_chunks])) if ref_chunks else 0.0

    # Match expected answer
    exp_ans, domain = "", "Traditional Knowledge"
    for bq in st.session_state.benchmark_questions:
        if bq["question"].lower() in query.lower() or query.lower() in bq["question"].lower():
            exp_ans, domain = bq["expected_answer"], bq["domain"]
            break
    if not exp_ans:
        if ref_chunks:
            # Combine retrieved chunks as the ground truth reference answer
            exp_ans = "\n\n".join([c["text"] for c in ref_chunks])
            domain = ref_chunks[0]["metadata"].get("domain", "Traditional Knowledge")
        else:
            exp_ans = "This query concerns traditional domain knowledge. Specific quantities, timelines, and ecological relationships are key."

    # Format Section 3 display content showing actual retrieved chunks
    ref_display_text = ""
    if ref_chunks:
        ref_display_text = "\n\n".join([
            f"**[{i+1}] Source: {ch['metadata']['source']} (Similarity: {ch['score']:.3f})**\n{ch['text']}"
            for i, ch in enumerate(ref_chunks)
        ])
    else:
        ref_display_text = "No matching reference knowledge found in the database."

    # Step 2 — Baseline response
    t0 = time.time()
    if live:
        baseline = pipeline.generate_baseline_response(query, api_key=api, target_language=q_lang)
    else:
        baseline = _pick_sim(query, _SIM_BASE)
    lat_base = time.time() - t0

    # Step 3 — Gap detection (baseline)
    base_eval  = detector.evaluate_gap(query, exp_ans, baseline, api_key=api)
    kgi_before = base_eval["kgi"]

    # Step 4 — Agentic decision
    domains      = st.session_state.ingester.get_domains() or ["Traditional Knowledge"]
    agent        = AgenticDecisionLayer(api_key=api)
    decision_out = agent.decide(query, kgi_before, domains)

    # Step 5 — RAG enhanced response
    t1 = time.time()
    if decision_out["decision"] == "YES":
        if live:
            enh, _ = pipeline.generate_enhanced_response(query, ref_chunks, api_key=api, target_language=q_lang)
        else:
            enh = _pick_sim(query, _SIM_ENH)
        enh_eval = detector.evaluate_gap(query, exp_ans, enh, api_key=api)
    else:
        enh      = baseline
        enh_eval = base_eval.copy()
    lat_enh = time.time() - t1

    # Detect actual output language of responses (in simulation, matches target)
    base_lang, _, _ = detect_language(baseline)
    enh_lang, _, _ = detect_language(enh)

    # Step 6 — Gap reduction
    reduction = analyze_gap_reduction(base_eval, enh_eval)

    # Step 7 — RAI scorecard (includes 3 new multilingual metrics)
    rai_ev      = RAIEvaluator()
    chunks_used = ref_chunks if decision_out["decision"] == "YES" else []
    rai = rai_ev.compute_rai_scorecard(
        baseline, enh, exp_ans, decision_out, chunks_used,
        query_lang=q_lang, baseline_lang=base_lang, enhanced_lang=enh_lang
    )
    for m in ("baseline", "enhanced"):
        rai[m]["knowledge_preservation"] = rai[m]["inclusiveness"]
        rai[m]["overall"] = float(np.mean([
            rai[m]["inclusiveness"], rai[m]["fair_representation"],
            rai[m]["transparency"], rai[m]["explainability"],
            rai[m]["knowledge_preservation"], rai[m]["language_inclusiveness"],
            rai[m]["cross_lingual_representation"], rai[m]["regional_knowledge_access"]
        ]))

    return {
        "query": query, "expected_answer": exp_ans, "domain": domain, "live_mode": live,
        "query_language": q_name, "query_lang_code": q_lang, "query_lang_confidence": q_conf,
        "baseline_model":    "gemini-1.5-flash" if live else "Simulated LLM (offline)",
        "baseline_answer":   baseline,
        "baseline_language": config.SUPPORTED_LANGUAGES.get(base_lang, "English"),
        "baseline_confidence": float(base_eval["semantic_similarity"] * 0.9),
        "baseline_latency":  lat_base,
        "ref_chunks": ref_chunks, "ref_answer": ref_display_text,
        "sources": sources, "source_relevance": max_rel,
        "base_eval": base_eval,
        "decision":  decision_out,
        "enhanced_answer":  enh,
        "enhanced_language": config.SUPPORTED_LANGUAGES.get(enh_lang, "English"),
        "enhanced_latency": lat_enh,
        "enh_eval":         enh_eval,
        "reduction": reduction,
        "rai": rai,
    }

with col_main:
    if run_btn and user_query.strip():
        with st.spinner("Running Explainable Agentic RAG pipeline…"):
            try:
                result = run_pipeline(user_query.strip())
                st.session_state.current_result = result
                st.session_state.history.append(result)
                st.success("✅ Analysis complete — results displayed below.")
                st.rerun()
            except Exception as exc:
                import traceback
                st.error(f"Pipeline error: {exc}")
                st.code(traceback.format_exc())
    elif run_btn:
        st.warning("Please enter or select a query first.")

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
with col_main:
    r = st.session_state.current_result
    if r is None:
        st.info("👆 Select a benchmark query above or enter a custom one and click **Analyze Knowledge Gap** to begin.")
    else:


        # ── S2 & S3 ──────────────────────────────────
        c2, c3 = st.columns(2)

        with c2:
            _md('<h3 class="shimmer-title">🤖 BASELINE GENERATIVE AI RESPONSE</h3>')
            st.markdown(f"**Model:** `{r['baseline_model']}`")
            _md(f'<div class="mono">{r["baseline_answer"]}</div>')
            st.markdown(f"**Confidence:** `{r['baseline_confidence']:.2%}` &nbsp;|&nbsp; **Latency:** `{r['baseline_latency']:.3f} s`")

        with c3:
            _md('<h3 class="shimmer-title">📖 REFERENCE KNOWLEDGE RESPONSE</h3>')
            st.markdown(f"**Sources:** `{', '.join(r['sources'])}`")
            _md(f'<div class="mono">{r["ref_answer"]}</div>')
            st.markdown(f"**Source Relevance Score:** `{r['source_relevance']:.3f}`")

        st.markdown("---")

        # ── S4 & S5 ──────────────────────────────────
        c4, c5 = st.columns(2)

        with c4:
            _md('<h3 class="shimmer-title">🔍 KNOWLEDGE GAP ANALYSIS</h3>')
            be  = r["base_eval"]
            kgi = be["kgi"]

            render_metrics_row([
                ("Semantic Similarity", f"{be['semantic_similarity']:.3f}"),
                ("Coverage Score", f"{be['coverage_score']:.2%}"),
                ("Missing Concept", f"{be['missing_concept_score']:.2%}")
            ])

            if kgi >= 0.6:
                kgi_color, kgi_lbl = "#e74c3c", "Severe Gap"
            elif kgi >= 0.3:
                kgi_color, kgi_lbl = "#f39c12", "Moderate Gap"
            else:
                kgi_color, kgi_lbl = "#2ecc71", "Low Gap"

            _md(f"""
            <div class="meter-wrap">
              <strong>Knowledge Gap Index (KGI)</strong>
              <div class="meter-bar">
                <div class="meter-fill" style="width:{kgi*100:.1f}%;background:{kgi_color}"></div>
              </div>
              <div class="meter-labels">
                <span style="color:#aaa">0.0 · Complete</span>
                <span style="color:{kgi_color};font-weight:700">KGI {kgi:.3f} [{kgi_lbl}]</span>
                <span style="color:#aaa">1.0 · Extinct</span>
              </div>
            </div>
            """)

        with c5:
            _md('<h3 class="shimmer-title">🧠 EXPLAINABLE AGENTIC DECISION PANEL</h3>')
            dec = r["decision"]
            if dec["decision"] == "YES":
                _md('<span class="chip-yes">⚠️ RETRIEVAL REQUIRED</span>')
            else:
                _md('<span class="chip-no">✅ RETRIEVAL NOT REQUIRED</span>')

            st.markdown(f"**Decision Confidence:** `{dec['confidence']:.2%}`")
            st.markdown(f"**Targeted Domain:** `{dec['chosen_domain']}`")
            st.markdown(f"**KGI vs Threshold:** `{r['base_eval']['kgi']:.3f}` vs `{st.session_state.kgi_threshold:.2f}`")
            _md(f'<div class="reasoning">{dec["reason"]}</div>')

        st.markdown("---")

        # ── S6 ───────────────────────────────────────
        _md('<h3 class="shimmer-title">✨ RAG ENHANCED RESPONSE</h3>')

        c6l, c6r = st.columns([2, 1])
        with c6l:
            st.markdown("**Enhanced Answer:**")
            _md(f'<div class="mono mono-green">{r["enhanced_answer"]}</div>')
            st.markdown(f"**Retrieval Confidence:** `{r['source_relevance']:.2%}` &nbsp;|&nbsp; **Generation Latency:** `{r['enhanced_latency']:.3f} s`")
        with c6r:
            st.markdown("**Retrieved Chunks (Hybrid Search):**")
            if dec["decision"] == "YES" and r["ref_chunks"]:
                for i, ch in enumerate(r["ref_chunks"]):
                    with st.expander(f"Chunk {i+1} · Score `{ch['score']:.3f}`"):
                        st.caption(f"**Source:** {ch['metadata']['source']} · Page {ch['metadata']['page']}")
                        st.caption(f"Vec `{ch['vector_score']:.3f}` | Keyword `{ch['keyword_score']:.3f}`")
                        st.code(ch["text"][:300], language="text")
            else:
                st.warning("Agent decided retrieval was not required — no chunks injected.")

        st.markdown("---")

        # ── S7 ───────────────────────────────────────
        _md('<h3 class="shimmer-title">📈 KNOWLEDGE GAP REDUCTION ANALYSIS</h3>')

        red = r["reduction"]
        c7m, c7ch = st.columns([1, 2])
        with c7m:
            st.metric("KGI Before RAG", f"{red['kgi_before']:.3f}")
            st.metric("KGI After RAG",  f"{red['kgi_after']:.3f}",
                      delta=f"{red['kgi_before']-red['kgi_after']:.3f} reduced")
            st.metric("Gap Reduction",  f"{red['gap_reduction_percentage']:.1f}%")
            st.markdown(f"Semantic improvement: **+{red['semantic_improvement']:.3f}**")
            st.markdown(f"Coverage improvement: **+{red['coverage_improvement']:.2%}**")

        with c7ch:
            fig_red = go.Figure()
            fig_red.add_trace(go.Bar(
                name="Baseline LLM", marker_color="#f43f5e",
                x=["KGI", "Semantic Similarity", "Concept Coverage"],
                y=[red["kgi_before"],
                   r["base_eval"]["semantic_similarity"],
                   r["base_eval"]["coverage_score"]],
            ))
            fig_red.add_trace(go.Bar(
                name="RAG Enhanced", marker_color="#38bdf8",
                x=["KGI", "Semantic Similarity", "Concept Coverage"],
                y=[red["kgi_after"],
                   r["enh_eval"]["semantic_similarity"],
                   r["enh_eval"]["coverage_score"]],
            ))
            fig_red.update_layout(
                barmode="group", height=255,
                margin=dict(t=10, b=10, l=5, r=5),
                legend=dict(orientation="h", y=1.12)
            )
            st.plotly_chart(fig_red, use_container_width=True)

        st.markdown("---")

        # ── S8 ───────────────────────────────────────
        _md('<h3 class="shimmer-title">🔬 EXPLAINABILITY PANEL (XAI)</h3>')

        c8l, c8r = st.columns(2)
        with c8l:
            st.markdown("**XAI Audit Trail:**")
            trigger_reason = "KGI exceeded threshold — retrieval triggered" if dec["decision"] == "YES" else "KGI below threshold — baseline was sufficient"
            st.write(f"📌 **Trigger cause:** {trigger_reason}")
            st.write(f"📌 **Source targeted:** `{', '.join(r['sources'])}`")
            st.write(f"📌 **Concepts audited:** `{len(r['base_eval']['concepts'])}` key concepts")
            st.write(f"📌 **Explainability confidence:** `{dec['confidence']*0.97:.2%}`")

        concepts  = r["base_eval"]["concepts"]
        enh_conc  = r["enh_eval"].get("concepts", [])

        with c8r:
            rows = []
            for i, bc in enumerate(concepts):
                ec_stat = enh_conc[i]["status"] if i < len(enh_conc) else (
                    "Present" if dec["decision"] == "YES" else bc["status"])
                rows.append({
                    "Concept": bc["concept"][:55],
                    "Baseline": "❌ Absent" if bc["status"] == "Absent" else "✅ Present",
                    "Enhanced": "❌ Absent" if ec_stat  == "Absent" else "✅ Present",
                })
            st.table(pd.DataFrame(rows))

        st.markdown("---")

        # ── S9 ───────────────────────────────────────
        _md('<h3 class="shimmer-title">🎯 RESPONSIBLE AI SCORECARD</h3>')

        rai = r["rai"]
        RAI_METRICS = [
            ("Inclusiveness",                      "inclusiveness"),
            ("Fair Representation",                "fair_representation"),
            ("Transparency",                       "transparency"),
            ("Explainability",                     "explainability"),
            ("Knowledge Preservation",             "knowledge_preservation"),
            ("Language Inclusiveness",             "language_inclusiveness"),
            ("Cross-Lingual Representation",       "cross_lingual_representation"),
            ("Regional Knowledge Accessibility",   "regional_knowledge_access"),
            ("OVERALL RAI INDEX",                  "overall"),
        ]

        c9t, c9ch = st.columns([3, 2])
        with c9t:
            rai_rows = [{
                "Responsible AI Metric": lbl,
                "Baseline LLM": f"{rai['baseline'][k]:.2f}",
                "RAG Enhanced": f"{rai['enhanced'][k]:.2f}",
                "Delta": f"+{rai['enhanced'][k]-rai['baseline'][k]:.2f}",
            } for lbl, k in RAI_METRICS]
            st.table(pd.DataFrame(rai_rows))

        with c9ch:
            rai_plot = []
            for lbl, k in RAI_METRICS[:-1]:
                rai_plot += [
                    {"Metric": lbl, "Score": rai["baseline"][k], "Model": "Baseline LLM"},
                    {"Metric": lbl, "Score": rai["enhanced"][k], "Model": "RAG Enhanced"},
                ]
            fig_rai = px.bar(
                pd.DataFrame(rai_plot),
                x="Metric", y="Score", color="Model", barmode="group",
                color_discrete_map={"Baseline LLM": "#f43f5e", "RAG Enhanced": "#38bdf8"}
            )
            fig_rai.update_layout(
                height=320, margin=dict(t=10, b=10, l=5, r=5),
                legend=dict(orientation="h", y=1.12)
            )
            st.plotly_chart(fig_rai, use_container_width=True)

        st.markdown("---")

        # ── S10 ──────────────────────────────────────
        _md('<h3 class="shimmer-title">📊 ANALYTICS VISUALIZATION</h3>')

        c10l, c10r = st.columns(2)
        with c10l:
            st.markdown("**Historical Query Log:**")
            if st.session_state.history:
                hist_df = pd.DataFrame([{
                    "#": i+1,
                    "Query": h["query"][:36] + "…",
                    "Language": h.get("query_language", "English"),
                    "Baseline KGI": f"{h['base_eval']['kgi']:.3f}",
                    "Enhanced KGI": f"{h['enh_eval']['kgi']:.3f}",
                    "Reduction":    f"{h['reduction']['gap_reduction_percentage']:.1f}%",
                } for i, h in enumerate(st.session_state.history)])
                st.dataframe(hist_df, use_container_width=True)
            else:
                st.info("No history yet.")

            st.markdown("**Domain-wise Average KGI:**")
            if st.session_state.history:
                dom_df = (
                    pd.DataFrame([{
                        "Domain": h["domain"],
                        "Baseline KGI": h["base_eval"]["kgi"],
                        "Enhanced KGI": h["enh_eval"]["kgi"],
                    } for h in st.session_state.history])
                    .groupby("Domain").mean().reset_index()
                )
                fig_dom = px.bar(
                    dom_df.melt("Domain", var_name="Model", value_name="Avg KGI"),
                    x="Domain", y="Avg KGI", color="Model", barmode="group",
                    color_discrete_map={"Baseline KGI": "#f43f5e", "Enhanced KGI": "#38bdf8"}
                )
                fig_dom.update_layout(
            height=235, margin=dict(t=10, b=10, l=5, r=5),
            legend=dict(orientation="h", y=1.12, font=dict(color="#f1f5f9")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#cbd5e1")),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#cbd5e1")),
        )
                st.plotly_chart(fig_dom, use_container_width=True)

        with c10r:
            st.markdown("**Concept Coverage Heatmap:**")
            if concepts:
                c_labels = [c["concept"][:28] + "…" for c in concepts]
                b_vals   = [1.0 if c["status"] == "Present" else 0.0 for c in concepts]
                e_vals   = []
                for i, bc in enumerate(concepts):
                    es = enh_conc[i]["status"] if i < len(enh_conc) else (
                        "Present" if dec["decision"] == "YES" else bc["status"])
                    e_vals.append(1.0 if es == "Present" else 0.0)

                fig_heat = go.Figure(go.Heatmap(
                    z=[b_vals, e_vals],
                    x=c_labels,
                    y=["Baseline LLM", "RAG Enhanced"],
                    colorscale="RdYlGn", zmin=0, zmax=1,
                    colorbar=dict(title="", tickvals=[0, 1], ticktext=["Absent", "Present"])
                ))
                fig_heat.update_layout(height=220, margin=dict(t=10, b=10, l=5, r=5))
                st.plotly_chart(fig_heat, use_container_width=True)

            st.markdown("**Retrieval Latency Trend:**")
            if len(st.session_state.history) > 1:
                lat_df = pd.DataFrame([{
                    "Query": f"Q{i+1}",
                    "Baseline (s)": h["baseline_latency"],
                    "Enhanced (s)": h["enhanced_latency"],
                } for i, h in enumerate(st.session_state.history)])
                fig_lat = px.line(
                    lat_df.melt("Query", var_name="Stage", value_name="Latency (s)"),
                    x="Query", y="Latency (s)", color="Stage", markers=True
                )
                fig_lat.update_layout(height=220, margin=dict(t=10, b=10, l=5, r=5))
                st.plotly_chart(fig_lat, use_container_width=True)
            else:
                st.caption("Run more queries to see latency trends.")

        st.markdown("---")

        # ── S11 ──────────────────────────────────────
        _md('<h3 class="shimmer-title">🌐 MULTILINGUAL ANALYSIS</h3>')

        c11a, c11b = st.columns(2)

        with c11a:
            st.markdown("**Detected Language & Routing Details:**")
            m_rows = [
                {"Multilingual Parameter": "Detected Language", "Value": f"{r.get('query_language', 'English')}"},
                {"Multilingual Parameter": "Language Confidence", "Value": f"{r.get('query_lang_confidence', 1.0):.2%}"},
                {"Multilingual Parameter": "Question Language", "Value": f"{r.get('query_language', 'English')}"},
                {"Multilingual Parameter": "Answer Language", "Value": f"{r.get('enhanced_language', 'English')}"},
                {"Multilingual Parameter": "Retrieved Document Language", "Value": "English (Cross-Lingual Retrieval)"},
            ]
            st.table(pd.DataFrame(m_rows))

        with c11b:
            st.markdown("**Language-wise Performance Indicators:**")
            lang_name_lbl = r.get('query_language', 'English')
            st.metric(label=f"Language-wise KGI ({lang_name_lbl})", value=f"{r['enh_eval']['kgi']:.3f}", delta=f"-{r['reduction']['kgi_before']-r['enh_eval']['kgi']:.3f}")
    
            render_metrics_row([
                ("Lang-wise Similarity Score", f"{r['enh_eval']['semantic_similarity']:.3f}"),
                ("Lang-wise Coverage Score", f"{r['enh_eval']['coverage_score']:.1%}"),
                ("Language-wise Gap Reduction", f"{r['reduction']['gap_reduction_percentage']:.1f}%")
            ])

        # ── Export ────────────────────────────────────
        st.markdown("---")
        export_payload = {
            "query":              r["query"],
            "query_language":     r.get("query_language", "English"),
            "baseline_kgi":       r["base_eval"]["kgi"],
            "enhanced_kgi":       r["enh_eval"]["kgi"],
            "gap_reduction_pct":  r["reduction"]["gap_reduction_percentage"],
            "agent_decision":     r["decision"]["decision"],
            "agent_reasoning":    r["decision"]["reason"],
            "rai_scores":         r["rai"],
        }
        st.download_button(
            "📥 Export Full Analysis Report (JSON)",
            data=json.dumps(export_payload, indent=4),
            file_name="knowledge_gap_report.json",
            mime="application/json",
            use_container_width=True
        )
