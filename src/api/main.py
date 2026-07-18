"""
Thin FastAPI wrapper around the Stage 3 graph.

Usage:
    uvicorn src.api.main:app --reload

Then POST to http://localhost:8000/ask with {"question": "..."}.
"""
from src.graph.retriever import get_retriever
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import AskRequest, AskResponse
from src.graph.graph import ask

app = FastAPI(
    title="Agentic Research Assistant API",
    description="Self-correcting RAG over AI/ML papers, with a live web search fallback.",
    version="1.0.0",
)

# Permissive CORS for local dev — the Streamlit UI (Stage 5) runs on a
# different port and needs to be able to call this from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
def preload_retriever() -> None:
    """
    Loads the embedding model and vector store once at boot instead of
    lazily on the first request. Without this, the first /ask call pays
    that load cost on top of its own LLM calls — on a resource-constrained
    host that's slow enough to starve the health check and trigger an
    unnecessary restart mid-request.
    """
    get_retriever()

@app.get("/health")
def health() -> dict:
    """Simple liveness check — doesn't touch the LLM or vector store."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """Runs a question through the full CRAG graph and returns a grounded answer."""
    try:
        result = ask(request.question)
    except FileNotFoundError as e:
        # Vector store hasn't been built yet (Stage 1 not run)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {e}")

    sources = [
        {
            "title": doc.metadata.get("title", "Untitled"),
            "url": doc.metadata.get("url", ""),
        }
        for doc in result["documents"]
    ]

    return AskResponse(
        answer=result["generation"],
        web_search_used=result["web_search_used"],
        generation_retries=result["generation_retries"],
        sources=sources,
    )
