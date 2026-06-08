"""Shared base class for all agents — consistent model routing + tracing.

Every agent declares its `role`; the base resolves the role to a model via the
router and logs each LLM call with the agent name. Agents never hardcode model
ids or build OpenRouter requests themselves.
"""

from __future__ import annotations

from typing import Any

import structlog

from polymath.models.openrouter import chat_completion
from polymath.models.router import Role, model_for


class BaseAgent:
    #: Subclasses set these.
    role: Role
    name: str = "agent"

    def __init__(self) -> None:
        self.log = structlog.get_logger(self.name)

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict:
        """Run a chat completion using this agent's routed model."""
        model = model_for(self.role)
        self.log.info("agent.complete", agent=self.name, model=model)
        return await chat_completion(model, messages, **kwargs)
