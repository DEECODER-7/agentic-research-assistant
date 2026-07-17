"""
Loads the Chroma vector store persisted by Stage 1's ingestion pipeline and
exposes it as a LangChain retriever for the graph's retrieve node.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.ingestion.config import EMBEDDING_MODEL_NAME, RETRIEVAL_TOP_K, VECTOR_STORE_DIR

_retriever = None  # lazy singleton — avoids reloading the embedding model on every call


def get_retriever():
    """
    Returns a retriever over the Stage 1 vector store, built once and
    reused across questions.
    """
    global _retriever
    if _retriever is None:
        if not VECTOR_STORE_DIR.exists():
            raise FileNotFoundError(
                f"No vector store found at {VECTOR_STORE_DIR}. "
                "Run `python -m src.ingestion.build_index` first (Stage 1)."
            )
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vectorstore = Chroma(
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=embeddings,
        )
        _retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
    return _retriever