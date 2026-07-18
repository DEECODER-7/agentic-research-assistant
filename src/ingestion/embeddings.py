"""
Shared embedding client — used identically by both the ingestion pipeline
(build_index.py) and the retriever (src/graph/retriever.py), so vectors are
computed the same way on both sides.

Uses Hugging Face's hosted Inference API instead of loading model weights
locally. This is a deliberate memory tradeoff: local sentence-transformers +
torch need 500MB+ RAM just to load the model, which blows past Render's
free-tier 512MB limit. Calling the same model remotely keeps the running
process lightweight, at the cost of a network round-trip per embedding call
— a fine trade for a low-traffic portfolio deployment.
"""

import os

from langchain_huggingface import HuggingFaceEndpointEmbeddings

from src.ingestion.config import EMBEDDING_MODEL_NAME


def get_embeddings() -> HuggingFaceEndpointEmbeddings:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise OSError(
            "HF_TOKEN environment variable is required. Get a free token at "
            "https://huggingface.co/settings/tokens (read access is enough), "
            "then set it via `$env:HF_TOKEN = \"...\"` locally or as a "
            "secret in the Render dashboard for the deployed API."
        )
    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        huggingfacehub_api_token=hf_token,
    )