"""
Streamlit chat UI for the research assistant. Talks to the Stage 4 FastAPI
backend over HTTP rather than importing the graph directly — keeps the UI
a real client of the API, the way it'd work in production.

Usage:
    1. In one terminal: uvicorn src.api.main:app --reload
    2. In another:       streamlit run ui/app.py
"""

import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Agentic Research Assistant", page_icon="🔎")
st.title("🔎 Agentic Research Assistant")
st.caption(
    "Self-correcting RAG over real arXiv papers on RAG, LLMs, and agentic AI — "
    "falls back to a live web search when local retrieval is weak, instead of guessing."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(
                f"Sources{' (web search used)' if message.get('web_search_used') else ''}"
            ):
                for source in message["sources"]:
                    if source["url"]:
                        st.markdown(f"- [{source['title']}]({source['url']})")
                    else:
                        st.markdown(f"- {source['title']}")

if question := st.chat_input("Ask about RAG, LLMs, or agentic AI..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving, grading, and verifying..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/ask", json={"question": question}, timeout=120
                )
                response.raise_for_status()
                data = response.json()

                st.markdown(data["answer"])
                if data["web_search_used"]:
                    st.caption("⚠️ Local retrieval was weak — this answer used a live web search fallback.")

                if data["sources"]:
                    with st.expander("Sources"):
                        for source in data["sources"]:
                            if source["url"]:
                                st.markdown(f"- [{source['title']}]({source['url']})")
                            else:
                                st.markdown(f"- {source['title']}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data["sources"],
                        "web_search_used": data["web_search_used"],
                    }
                )
            except requests.exceptions.ConnectionError:
                error_msg = (
                    f"Can't reach the API at {API_BASE_URL}. "
                    "Make sure it's running: `uvicorn src.api.main:app --reload`"
                )
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except requests.exceptions.HTTPError as e:
                error_msg = f"API error: {e.response.json().get('detail', str(e))}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

with st.sidebar:
    st.header("About")
    st.markdown(
        "This assistant implements a **Corrective RAG (CRAG)** loop:\n\n"
        "1. Retrieve chunks from a vector store of real arXiv papers\n"
        "2. Grade whether they're actually relevant\n"
        "3. If weak, rewrite the query and fall back to a live web search\n"
        "4. Generate a grounded, cited answer\n"
        "5. Verify the answer is actually supported before returning it"
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
