"""
Central configuration — paths, model names, and search queries used to build
the knowledge base. Keeping these in one place means the ingestion script,
the RAG graph, and the API all reference the exact same settings.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = PROJECT_ROOT / "data" / "papers"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "chroma_db"

# Search queries used to pull a focused, relevant paper set from arXiv.
# Deliberately scoped to RAG / agentic AI / LLM topics — this is what makes
# the knowledge base "about" a coherent subject rather than a random dump.
ARXIV_SEARCH_QUERIES = [
    "retrieval augmented generation",
    "agentic large language models",
    "LLM agent reasoning",
    "corrective RAG",
]
PAPERS_PER_QUERY = 6  # keep this small — 2-day project, not a research corpus

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Groq model — fast and capable, good fit for a multi-call agentic loop
LLM_MODEL_NAME = "llama-3.3-70b-versatile"

RETRIEVAL_TOP_K = 10  # bumped from 4 — at k=4, the CRAG paper's own chunk
# was getting pushed just outside the top results, which meant a question
# about CRAG itself hit weak retrieval and triggered a needless web search.

# --- Stage 3: LangGraph self-correction loop settings ---
WEB_SEARCH_MAX_RESULTS = 3
MAX_GENERATION_RETRIES = 2  # caps the verify -> regenerate loop so a
# stubborn question can't spin forever on repeated LLM calls
