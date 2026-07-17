"""
Structural smoke tests for the Stage 3 graph — checks the state machine
compiles and wires together correctly. Doesn't call the LLM or vector
store, so it runs without a GROQ_API_KEY or a built index.
"""

from src.graph.graph import build_graph


def test_graph_compiles():
    graph = build_graph()
    assert graph is not None


def test_graph_has_expected_nodes():
    graph = build_graph()
    node_names = set(graph.get_graph().nodes.keys())
    expected = {
        "retrieve",
        "grade_documents",
        "transform_query",
        "web_search",
        "generate",
        "verify",
    }
    assert expected.issubset(node_names)
