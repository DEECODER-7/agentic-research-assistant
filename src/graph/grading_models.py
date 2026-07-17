"""
Structured output schemas for the LLM grading calls. Using structured
output instead of parsing free text keeps the routing logic after each
grading call reliable — no guessing whether the model said "yes" or "Yes."
or "the documents are relevant."
"""

from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """Grades whether the retrieved documents, as a set, are sufficient to answer the question."""

    binary_score: str = Field(
        description="'yes' if the retrieved documents are sufficient to answer "
        "the question, 'no' otherwise"
    )
    reasoning: str = Field(description="One-sentence explanation for the score")


class GradeHallucination(BaseModel):
    """Grades whether a generated answer is grounded in the retrieved documents."""

    binary_score: str = Field(
        description="'yes' if every substantive claim in the answer is backed by "
        "the documents, 'no' if it drifts into claims the documents don't support"
    )


class GradeAnswer(BaseModel):
    """Grades whether a generated answer actually addresses the question asked."""

    binary_score: str = Field(
        description="'yes' if the answer addresses the question asked, 'no' otherwise"
    )
