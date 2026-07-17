"""
Wires the node functions from nodes.py into the LangGraph state machine
implementing the Corrective RAG + self-verification loop:

    retrieve -> grade_documents -+-> generate ----------------------+
                                  |                                  |
                                  +-> transform_query -> web_search -+
                                                                      v
                                                                   verify
                                          not_supported -> generate (retry)
                                          not_useful    -> transform_query (retry)
                                          useful         -> END

Usage:
    python -m src.graph.graph "What is Corrective RAG?"
"""

from langgraph.graph import END, StateGraph

from src.graph.nodes import (
    decide_to_generate,
    generate,
    grade_documents,
    grade_generation,
    retrieve,
    transform_query,
    verify,
    web_search,
)
from src.graph.state import GraphState


def build_graph():
    """Compiles the CRAG state machine. Call once and reuse across questions."""
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_node("verify", verify)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"generate": "generate", "transform_query": "transform_query"},
    )

    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "verify")

    workflow.add_conditional_edges(
        "verify",
        grade_generation,
        {
            "useful": END,
            "not_supported": "generate",
            "not_useful": "transform_query",
        },
    )

    return workflow.compile()


def ask(question: str) -> dict:
    """Convenience entry point — runs a question through the full graph."""
    graph = build_graph()
    initial_state = {
        "question": question,
        "original_question": question,
        "documents": [],
        "generation": "",
        "web_search_used": False,
        "generation_retries": 0,
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    import sys

    user_question = " ".join(sys.argv[1:]) or "What is Corrective RAG (CRAG)?"
    result = ask(user_question)

    print("\n" + "=" * 70)
    print(f"Q: {user_question}")
    print("=" * 70)
    print(result["generation"])
    print("=" * 70)
    print(f"web search used: {result['web_search_used']}")
    print(f"generation retries: {result['generation_retries']}")
