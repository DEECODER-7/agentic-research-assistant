"""
Node functions for the self-correcting RAG graph.

Flow:
    retrieve -> grade_documents -+-> generate ----------------------+
                                  |                                  |
                                  +-> transform_query -> web_search -+
                                                                      v
                                                                   verify
                                          not_supported (ungrounded) -> generate (retry)
                                          not_useful (off-target)    -> transform_query (retry)
                                          useful                     -> END

Each function takes the current GraphState and returns a dict of fields to
update (LangGraph merges this into state) — this is the standard LangGraph
node signature.
"""
from ddgs import DDGS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.graph.grading_models import GradeAnswer, GradeDocuments, GradeHallucination
from src.graph.llm import get_llm
from src.graph.retriever import get_retriever
from src.ingestion.config import MAX_GENERATION_RETRIES, WEB_SEARCH_MAX_RESULTS


def _format_documents(documents: list[Document]) -> str:
    return "\n\n".join(f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(documents))


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------

def retrieve(state: dict) -> dict:
    """Pulls candidate chunks from the vector store for the current question."""
    print(f"---RETRIEVE--- ({state['question']!r})")
    documents = get_retriever().invoke(state["question"])
    print(f"  got {len(documents)} chunks")
    return {"documents": documents}


# ---------------------------------------------------------------------------
# Grade documents (single batch call, not per-chunk — cheaper and the real
# question is "is this retrieval good enough overall?" anyway)
# ---------------------------------------------------------------------------

_GRADE_DOCS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a grader assessing whether a set of retrieved document excerpts, "
        "taken together, contain enough relevant information to answer a user's "
        "question. Grade generously: partial but genuinely relevant coverage is a "
        "'yes'. Grade 'no' only if the excerpts are largely off-topic or unrelated "
        "to the question.",
    ),
    ("human", "Question: {question}\n\nRetrieved excerpts:\n{documents}"),
])


def grade_documents(state: dict) -> dict:
    """One LLM call grades all retrieved chunks together as a batch."""
    print("---GRADE DOCUMENTS---")
    documents = state["documents"]

    grader = get_llm().with_structured_output(GradeDocuments)
    result = grader.invoke(
        _GRADE_DOCS_PROMPT.format_messages(
            question=state["question"], documents=_format_documents(documents)
        )
    )
    print(f"  grade: {result.binary_score} ({result.reasoning})")

    return {"documents": documents if result.binary_score == "yes" else []}


def decide_to_generate(state: dict) -> str:
    """Routes straight to generate, or to transform_query if retrieval was weak."""
    return "generate" if state["documents"] else "transform_query"


# ---------------------------------------------------------------------------
# Transform query (the "correct" step)
# ---------------------------------------------------------------------------

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You rewrite questions into better search queries. Local retrieval for "
        "this question came back weak or off-topic. Rewrite it to surface its "
        "underlying intent as a clearer, more specific web search query.\n\n"
        "IMPORTANT: this question is always about AI/ML research — retrieval-"
        "augmented generation (RAG), LLMs, or agentic AI systems. Keep the "
        "rewrite anchored to that domain even if a term could mean something "
        "else in another field (e.g. 'CRAG' here always means Corrective RAG, "
        "never a project-management or geography term). Return only the "
        "rewritten query, nothing else.",
    ),
    ("human", "Original question: {question}"),
])


def transform_query(state: dict) -> dict:
    """Rewrites the question so the web search fallback has a better shot at it."""
    print("---TRANSFORM QUERY---")
    chain = _REWRITE_PROMPT | get_llm() | StrOutputParser()
    rewritten = chain.invoke({"question": state["question"]}).strip()
    print(f"  rewritten -> {rewritten!r}")
    return {"question": rewritten}


# ---------------------------------------------------------------------------
# Web search (fallback, not primary path)
# ---------------------------------------------------------------------------

def web_search(state: dict) -> dict:
    """
    Falls back to a live web search when local retrieval was insufficient,
    rather than generating a confident-sounding answer from weak chunks.
    """
    print("---WEB SEARCH---")
    with DDGS() as ddgs:
        results = list(ddgs.text(state["question"], max_results=WEB_SEARCH_MAX_RESULTS))

    documents = [
        Document(
            page_content=r.get("body", ""),
            metadata={"title": r.get("title", "Web result"), "url": r.get("href", "")},
        )
        for r in results
    ]
    print(f"  got {len(documents)} web results")
    return {"documents": documents, "web_search_used": True}


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

_GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research assistant. Answer the question using ONLY the "
        "provided context. Cite sources inline using the [n] markers matching "
        "the numbered excerpts. If the context is insufficient to fully answer, "
        "say so plainly instead of filling gaps with outside knowledge.",
    ),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])


def generate(state: dict) -> dict:
    """Produces a grounded, cited answer from whatever documents are currently in state."""
    print("---GENERATE---")
    chain = _GENERATE_PROMPT | get_llm() | StrOutputParser()
    generation = chain.invoke(
        {
            "question": state["original_question"],
            "context": _format_documents(state["documents"]),
        }
    )
    return {
        "generation": generation,
        "generation_retries": state.get("generation_retries", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Verify (self-check)
# ---------------------------------------------------------------------------

_HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You check whether an answer is actually grounded in the provided source "
        "documents, or whether it drifts into claims the sources don't support. "
        "Grade 'yes' only if every substantive claim is backed by the documents.",
    ),
    ("human", "Documents:\n{documents}\n\nAnswer:\n{generation}"),
])

_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Does this answer actually address the question that was asked? "
        "Grade 'yes' or 'no'.",
    ),
    ("human", "Question: {question}\n\nAnswer:\n{generation}"),
])


def verify(state: dict) -> dict:
    """Pass-through node — the actual check + routing happens in grade_generation below."""
    return {}


def grade_generation(state: dict) -> str:
    """
    Final self-correction check. Two distinct things can go wrong even with
    good retrieval: the model can hallucinate beyond the sources (loop back
    to regenerate), or answer something adjacent to but not quite the
    question (loop back to search again). The retry cap stops a stubborn
    question from looping forever.
    """
    print("---VERIFY---")
    if state.get("generation_retries", 0) >= MAX_GENERATION_RETRIES:
        print("  retry cap reached — returning best-effort answer")
        return "useful"

    hallucination_grader = get_llm().with_structured_output(GradeHallucination)
    grounded = hallucination_grader.invoke(
        _HALLUCINATION_PROMPT.format_messages(
            documents=_format_documents(state["documents"]),
            generation=state["generation"],
        )
    )
    if grounded.binary_score != "yes":
        print("  not grounded in sources — regenerating")
        return "not_supported"

    answer_grader = get_llm().with_structured_output(GradeAnswer)
    addresses = answer_grader.invoke(
        _ANSWER_PROMPT.format_messages(
            question=state["original_question"], generation=state["generation"]
        )
    )
    if addresses.binary_score == "yes":
        print("  useful — done")
        return "useful"

    print("  doesn't address the question — retrying search")
    return "not_useful"
