"""LangGraph state for the research workflow.

A single Pydantic model threaded through every node. `trace` uses an additive
reducer so each node appends one transition record without clobbering others;
all other fields are overwritten by the node that owns them.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from pydantic import BaseModel, Field

from polymath.models.schemas import Claim, SlideDeck


class GraphState(BaseModel):
    topic: str
    max_iterations: int = 3

    # Mutated as the graph runs:
    iteration: int = 0
    subtasks: list[str] = Field(default_factory=list)  # current search queries
    pending_urls: list[str] = Field(default_factory=list)  # URLs awaiting the Reader
    claims: list[Claim] = Field(default_factory=list)  # snapshot of the store

    # Critic output:
    decision: str = ""
    new_subtasks: list[str] = Field(default_factory=list)
    reason: str = ""

    # Final output:
    report: str = ""
    deck: SlideDeck | None = None

    # Append-only transition log (one entry per node visit).
    trace: Annotated[list[dict[str, Any]], operator.add] = Field(default_factory=list)
