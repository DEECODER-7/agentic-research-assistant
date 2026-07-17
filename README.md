# Agentic Research Assistant — Self-Correcting RAG over AI/ML Papers

A research assistant that answers questions about RAG, LLMs, and agentic AI
by retrieving from real arXiv papers — and knows when its own retrieval is
weak, automatically falling back to a live web search instead of guessing.

Built with LangGraph (agent orchestration), LangChain (RAG plumbing), Groq
(fast free-tier LLM inference), and a local embedding model (no embedding
API costs).

## Why this isn't "just a RAG chatbot"

Most RAG demos retrieve top-k chunks and generate an answer, full stop — if
retrieval pulls irrelevant chunks, the answer is confidently wrong anyway.
This system implements a **Corrective RAG (CRAG) + self-verification loop**:

1. **Retrieve** — pull candidate chunks from the vector store
2. **Grade retrieval** — an LLM call checks whether the retrieved chunks are
   actually relevant to the question
3. **Correct, if needed** — if retrieval was weak, rewrite the query and fall
   back to a live web search instead of answering from bad context
4. **Generate** — produce a grounded, cited answer
5. **Verify** — a final LLM check catches answers that drift beyond what the
   sources actually support, looping back to regenerate if needed

This is a real, recognized architecture pattern (Corrective RAG / Self-RAG),
not an invented workflow — the kind of design decision an interviewer will
recognize and want to discuss in depth.

## Roadmap

- [x] Stage 1: Project setup + document ingestion pipeline
- [x] Stage 2: Baseline RAG chain (retrieve + generate, no correction yet)
- [x] Stage 3: LangGraph state machine — grading, correction, verification loop
- [x] Stage 4: FastAPI wrapper around the graph
- [x] Stage 5: Streamlit chat UI
- [ ] Stage 6: Deployment (Render + Streamlit Cloud)
- [ ] Stage 7: README/GitHub polish for portfolio presentation

## Stage 1 setup

1. Create and activate a virtual environment (same process as your last project):
   ```bash
   python -m venv venv
   venv\Scripts\Activate.ps1   # Windows PowerShell
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Get a **free Groq API key**: go to [console.groq.com/keys](https://console.groq.com/keys),
   sign up (free, no card), create an API key.
4. Set it as an environment variable (same pattern as your Kaggle token earlier):
   ```powershell
   $env:GROQ_API_KEY = "your_key_here"
   ```
5. Run the ingestion script to pull real papers and build the vector store:
   ```bash
   python -m src.ingestion.build_index
   ```

## Stage 3: running the self-correcting graph

With Stage 1's vector store already built and `GROQ_API_KEY` set:

```bash
python -m src.graph.graph "What is Corrective RAG (CRAG)?"
```

Or import it directly:

```python
from src.graph.graph import ask

result = ask("How does agentic reasoning differ from a standard LLM call?")
print(result["generation"])
print("used web search:", result["web_search_used"])
```

The graph logs each stage (`RETRIEVE`, `GRADE DOCUMENTS`, `TRANSFORM QUERY`,
`WEB SEARCH`, `GENERATE`, `VERIFY`) as it runs, so you can watch the
self-correction loop decide, in real time, whether local retrieval was good
enough or whether it needed to rewrite the query and fall back to a live
web search — and whether the final answer actually held up against the
sources it was given.

Run the structural smoke tests (no API key or vector store needed):

```bash
pytest tests/test_graph.py
```

## Stage 4 & 5: running the API + chat UI

Two terminals, both from the project root, with `GROQ_API_KEY` set and the
Stage 1 vector store already built:

```bash
# Terminal 1 — the API
uvicorn src.api.main:app --reload
```
```bash
# Terminal 2 — the chat UI
streamlit run ui/app.py
```

Streamlit opens at `http://localhost:8501` and talks to the API at
`http://localhost:8000` (override with the `API_BASE_URL` env var if the
API runs elsewhere). Each answer shows its sources and flags whether it
needed the live web search fallback.

You can also hit the API directly:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Corrective RAG (CRAG)?"}'
```

## Why these specific tool choices

- **Local embeddings (sentence-transformers) instead of an embeddings API**:
  zero cost, zero rate limits, and it's a legitimate production pattern —
  many real systems self-host embeddings specifically to avoid per-call
  costs at scale.
- **ChromaDB**: runs locally, no account/server needed, perfect for a
  portfolio project; swapping it for a hosted vector DB (Pinecone, Weaviate)
  later is a small, well-understood change if you ever need to scale this.
- **Groq over OpenAI/Anthropic APIs for this project**: genuinely free tier,
  and notably fast inference — which matters a lot for an agentic loop that
  makes multiple LLM calls per question (grading, correcting, generating,
  verifying). A slow LLM would make the self-correction loop painfully slow
  to demo.
