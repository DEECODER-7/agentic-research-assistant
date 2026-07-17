"""
Shared state passed between every node in the self-correction graph.
"""

from typing import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str             # current question — gets rewritten by transform_query
    original_question: str    # the user's actual question, kept for generate/verify
    documents: list[Document]  # chunks currently in play (vector store or web results)
    generation: str           # latest generated answer
    web_search_used: bool     # whether this answer needed the live web fallback
    generation_retries: int   # guards the verify -> regenerate loop
