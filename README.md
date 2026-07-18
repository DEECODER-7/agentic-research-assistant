# Agentic Research Assistant — Self-Correcting RAG over AI/ML Papers

A research assistant that answers questions about RAG, LLMs, and agentic AI
by retrieving from real arXiv papers — and knows when its own retrieval is
weak, automatically falling back to a live web search instead of guessing.

**🔗 Live demo:** [agentic-research-assistant-ak7tedecm7lewrfzftuahm.streamlit.app](https://agentic-research-assistant-ak7tedecm7lewrfzftuahm.streamlit.app/)
**🔗 API:** [agentic-research-assistant-api.onrender.com](https://agentic-research-assistant-api.onrender.com) ([`/health`](https://agentic-research-assistant-api.onrender.com/health))

> Both run on free-tier hosting and spin down after ~15 min idle — the
> first request after that takes 30–60s to wake back up. Not a bug.

Built with LangGraph (agent orchestration), LangChain (RAG plumbing), Groq
(fast free-tier LLM inference), and remote Hugging Face embeddings (keeps
the deployed API's memory footprint small enough for a free-tier host).

## Why this isn't "just a RAG chatbot"

Most RAG demos retrieve top-k chunks and generate an answer, full stop — if
retrieval pulls irrelevant chunks, the answer is confidently wrong anyway.
This system implements a **Corrective RAG (CRAG) + self-verification loop**:

1. **Retrieve** — pull candidate chunks from the vector store
2. **Grade retrieval** — a single batched LLM call checks whether the
   retrieved chunks, taken together, are actually relevant to the question
3. **Correct, if needed** — if retrieval was weak, rewrite the query (kept
   anchored to the AI/ML domain, so ambiguous terms like "CRAG" don't drift
   into unrelated meanings) and fall back to a live web search instead of
   answering from bad context
4. **Generate** — produce a grounded, cited answer
5. **Verify** — two more LLM checks: is the answer actually grounded in the
   sources (catches hallucination), and does it address the question that
   was asked (catches drift)? Failures loop back to regenerate or re-search,
   capped by a retry limit so a hard question can't loop forever

This is a real, recognized architecture pattern (Corrective RAG / Self-RAG),
not an invented workflow.

## Architecture
`FastAPI` wraps the compiled graph behind a `/ask` endpoint; `Streamlit`
is a thin client that calls that API and renders sources + a fallback flag.

## Roadmap

- [x] Stage 1: Project setup + document ingestion pipeline
- [x] Stage 2: Baseline RAG chain (retrieve + generate, no correction yet)
- [x] Stage 3: LangGraph state machine — grading, correction, verification loop
- [x] Stage 4: FastAPI wrapper around the graph
- [x] Stage 5: Streamlit chat UI
- [x] Stage 6: Deployment (Render + Streamlit Cloud)
- [x] Stage 7: README/GitHub polish for portfolio presentation

## Running it locally

1. Create and activate a virtual environment:
```powershell
   python -m venv venv
   venv\Scripts\Activate.ps1
```
   (If `python` isn't recognized, or `python -m venv` errors on Windows,
   use the `py` launcher instead: `py -m venv venv`.)
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Get two free API keys:
   - **Groq** (LLM calls): [console.groq.com/keys](https://console.groq.com/keys)
   - **Hugging Face** (remote embeddings — "Read" access is enough):
     [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Set them as environment variables:
```powershell
   $env:GROQ_API_KEY = "your_groq_key"
   $env:HF_TOKEN = "your_hf_token"
```
   (Or set them permanently for your user account with
   `[Environment]::SetEnvironmentVariable("GROQ_API_KEY", "...", "User")`,
   restart the terminal, and skip this step in future sessions.)
5. Build the vector store from real arXiv papers:
```bash
   python -m src.ingestion.build_index
```

## Running the graph directly

```bash
python -m src.graph.graph "What is Corrective RAG (CRAG)?"
```

Or import it:

```python
from src.graph.graph import ask

result = ask("How does agentic reasoning differ from a standard LLM call?")
print(result["generation"])
print("used web search:", result["web_search_used"])
```

The graph logs each stage (`RETRIEVE`, `GRADE DOCUMENTS`, `TRANSFORM QUERY`,
`WEB SEARCH`, `GENERATE`, `VERIFY`) as it runs, so you can watch the
self-correction loop decide in real time whether local retrieval was good
enough, or whether it needed to rewrite the query, fall back to a live web
search, and verify the result actually held up against its sources.

Structural smoke tests (no API key or vector store needed):
```bash
pytest tests/test_graph.py
```

## Running the API + UI locally

Two terminals, both from the project root, with both env vars set and the
vector store already built:

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

Or hit the API directly:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Corrective RAG (CRAG)?"}'
```

## Deployment

The API deploys to **Render** (free tier), the UI to **Streamlit Community
Cloud** (free) — both straight from this GitHub repo.

### 1. Push to GitHub — including the vector store

Render's filesystem doesn't persist across deploys, and re-scraping arXiv
on every deploy is slow and depends on arXiv being reachable at that exact
moment. So instead of rebuilding the index at deploy time, the already-built
`data/chroma_db/` is committed straight into the repo (raw PDFs stay
gitignored — large and easy to regenerate; the index built from them is
small and is what actually needs to ship).

```bash
git add -f data/chroma_db
git add .
git commit -m "Deployment config"
git push
```

### 2. Deploy the API to Render

1. [render.com](https://render.com) → sign up / log in with GitHub.
2. **New → Blueprint** → select this repo. Render reads `render.yaml`
   automatically and configures the service.
3. In the service's **Environment** tab, add both `GROQ_API_KEY` and
   `HF_TOKEN` (deliberately left unset in `render.yaml` — never commit keys).
4. Confirm it's alive: `curl https://<your-service>.onrender.com/health`

### 3. Deploy the UI to Streamlit Community Cloud

1. [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub.
2. **New app** → this repo, branch `main`, main file path `ui/app.py`.
3. **Advanced settings → Secrets**:
```toml
   API_BASE_URL = "https://<your-render-service>.onrender.com"
```
4. Deploy.

### 4. Verify

Ask an in-corpus question ("What is retrieval-augmented generation?") and
an out-of-corpus one ("What is Corrective RAG?") — confirm sources show up
on both and the web-search fallback flag appears only on the second.

## Why these specific tool choices

- **Remote Hugging Face embeddings instead of a local model**: the original
  design used local `sentence-transformers`, but that plus `torch` need
  500MB+ RAM just to load — more than Render's free-tier 512MB limit.
  Calling the same model via HF's Inference API keeps the deployed process
  lightweight at the cost of a network round-trip per embedding call, a
  worthwhile trade for a low-traffic deployment. This was a real production
  constraint discovered during deployment, not a day-one design choice.
- **ChromaDB**: runs embedded in-process, no separate server/account needed
  — good fit for a project this size; swapping it for a hosted vector DB
  (Pinecone, Weaviate) later is a small, well-understood change.
- **Groq over OpenAI/Anthropic for this project**: genuinely free tier, and
  fast inference — which matters for an agentic loop making several LLM
  calls per question (grade, correct, generate, verify). A slow LLM would
  make the self-correction loop painfully slow to demo.
- **Startup-time model preloading**: the retriever is loaded once at API
  startup rather than lazily on the first request — otherwise the first
  `/ask` call pays that load cost on top of its own LLM calls, which on a
  resource-constrained host was slow enough to starve the platform's health
  check and trigger a restart mid-request. Another real failure encountered
  and fixed during deployment, not anticipated in advance.

## Known limitations

- No persistent conversation memory — each question runs through the graph
  independently; follow-up questions don't carry prior context.
- Evaluation is inline/online (the verify step, per request) rather than an
  offline regression test suite against a labeled question set.
- Free-tier hosting means cold starts (~30-60s) after idle periods on both
  the API and, less commonly, the UI.