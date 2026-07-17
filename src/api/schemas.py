"""
Request/response schemas for the API. Kept separate from main.py so the
contract is easy to scan on its own, and reusable if a second endpoint
needs the same shapes later.
"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")


class AskResponse(BaseModel):
    answer: str
    web_search_used: bool
    generation_retries: int
    sources: list[dict]
