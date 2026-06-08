"""Pydantic models that cross the LLM boundary.

Per project convention, nothing that comes from or goes to an LLM is passed as a
raw dict — it is validated through one of these models first.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low"]


class Claim(BaseModel):
    """A single sourced claim extracted from a page by the Reader."""

    claim: str = Field(..., description="A concise factual statement.")
    evidence_quote: str = Field(
        ..., description="Verbatim text from the source supporting the claim."
    )
    source_url: str = Field(..., description="URL of the page the claim came from.")
    confidence: Confidence = Field(
        ..., description="How well the evidence supports the claim."
    )


class ExtractedClaims(BaseModel):
    """The Reader's full response for one page: a list of claims."""

    claims: list[Claim] = Field(default_factory=list)
