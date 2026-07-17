"""
Centralized LLM client for the RAG graph. Groq matters here specifically
because the self-correction loop makes several LLM calls per question
(grade -> maybe rewrite -> generate -> verify) — a slow model would make
that loop painful to demo.
"""

from langchain_groq import ChatGroq

from src.ingestion.config import LLM_MODEL_NAME


def get_llm(temperature: float = 0.0) -> ChatGroq:
    """
    Returns a configured Groq chat model. Reads GROQ_API_KEY from the
    environment (same variable set up in Stage 1).
    """
    return ChatGroq(model=LLM_MODEL_NAME, temperature=temperature)
