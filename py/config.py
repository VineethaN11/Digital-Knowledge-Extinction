import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

EVAL_DIR      = PROJECT_ROOT / 'evaluation_data'
VECTOR_DB_DIR = PROJECT_ROOT / 'vector_db'
CORPUS_DIR    = PROJECT_ROOT / 'knowledge_corpus'

for directory in [EVAL_DIR, VECTOR_DB_DIR, CORPUS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ── Model Settings ──────────────────────────────────────────────────────────
DEFAULT_GEMINI_MODEL       = 'gemini-1.5-flash'
GEMINI_EMBEDDING_MODEL     = 'models/text-embedding-004'

# Multilingual embedding model (BAAI/bge-m3) — primary
MULTILINGUAL_EMBEDDING_MODEL = 'BAAI/bge-m3'
MULTILINGUAL_EMBEDDING_DIM   = 1024

# Legacy fallback (384-dim)
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# ── Supported languages ─────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'kn': 'Kannada',
}

# ── KGI Weights (must sum to 1.0) ────────────────────────────────────────────
KGI_WEIGHTS = {
    'semantic': 0.4,
    'coverage': 0.3,
    'concept':  0.3,
}

KGI_THRESHOLD = 0.40

# ── Responsible AI Weights (7 metrics, each ~0.143) ─────────────────────────
RAI_WEIGHTS = {
    'inclusiveness':                 0.15,
    'fair_representation':           0.15,
    'transparency':                  0.15,
    'explainability':                0.15,
    'knowledge_preservation':        0.15,
    'language_inclusiveness':        0.125,
    'cross_lingual_representation':  0.05,
    'regional_knowledge_access':     0.075,
}

def get_gemini_api_key():
    key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    return key