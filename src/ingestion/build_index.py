"""
Fetches real papers from arXiv on RAG/agentic AI topics, extracts their text,
chunks it, embeds it via a remote embedding API, and persists it all into a
local ChromaDB vector store.

Usage:
    python -m src.ingestion.build_index
"""

from urllib.request import urlretrieve

import arxiv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.ingestion.config import (
    ARXIV_SEARCH_QUERIES,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    PAPERS_PER_QUERY,
    PDF_DIR,
    VECTOR_STORE_DIR,
)
from src.ingestion.embeddings import get_embeddings


def fetch_papers() -> list[arxiv.Result]:
    """
    Pulls a focused set of real papers from arXiv across our search queries.
    Deduplicates by paper ID since different queries can surface the same
    paper more than once.
    """
    client = arxiv.Client()
    seen_ids = set()
    papers = []

    for query in ARXIV_SEARCH_QUERIES:
        print(f"Searching arXiv for: '{query}'")
        search = arxiv.Search(
            query=query,
            max_results=PAPERS_PER_QUERY,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for result in client.results(search):
            if result.entry_id not in seen_ids:
                seen_ids.add(result.entry_id)
                papers.append(result)

    print(f"Found {len(papers)} unique papers total.")
    return papers


def download_papers(papers: list[arxiv.Result]) -> list[dict]:
    """Downloads each paper's PDF and returns metadata alongside the local file path."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for i, paper in enumerate(papers, 1):
        safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in paper.title)[:80]
        filename = f"{paper.get_short_id().replace('/', '_')}.pdf"
        filepath = PDF_DIR / filename

        if not filepath.exists():
            print(f"  [{i}/{len(papers)}] Downloading: {safe_title}")
            urlretrieve(paper.pdf_url, str(filepath))
        else:
            print(f"  [{i}/{len(papers)}] Already downloaded: {safe_title}")

        downloaded.append({
            "filepath": filepath,
            "title": paper.title,
            "authors": ", ".join(a.name for a in paper.authors[:3]),
            "arxiv_id": paper.get_short_id(),
            "url": paper.entry_id,
        })

    return downloaded


def load_and_chunk(downloaded: list[dict]) -> list[Document]:
    """
    Loads each PDF's text and splits it into overlapping chunks. Overlap
    matters here: without it, a sentence split across two chunks could lose
    the context needed to answer a question correctly at retrieval time.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks = []
    for paper in downloaded:
        try:
            loader = PyPDFLoader(str(paper["filepath"]))
            pages = loader.load()
        except Exception as e:
            print(f"  Skipping {paper['title']} — failed to parse PDF: {e}")
            continue

        full_text = "\n".join(p.page_content for p in pages)
        chunks = splitter.split_text(full_text)

        for chunk in chunks:
            all_chunks.append(Document(
                page_content=chunk,
                metadata={
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "arxiv_id": paper["arxiv_id"],
                    "url": paper["url"],
                },
            ))

    print(f"Created {len(all_chunks)} chunks from {len(downloaded)} papers.")
    return all_chunks


def build_vector_store(chunks: list[Document]) -> None:
    """Embeds all chunks via the remote API and persists them to ChromaDB."""
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} (via HF Inference API)...")
    embeddings = get_embeddings()

    print("Embedding chunks and building vector store (this is the slow step)...")
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_STORE_DIR),
    )
    print(f"Vector store built and saved to {VECTOR_STORE_DIR}")


def main():
    papers = fetch_papers()
    downloaded = download_papers(papers)
    chunks = load_and_chunk(downloaded)
    build_vector_store(chunks)
    print("\nIngestion complete. Ready for Stage 2 (baseline RAG chain).")


if __name__ == "__main__":
    main()