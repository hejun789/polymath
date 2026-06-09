"""PlannerAgent — decomposes a topic into focused search subtasks.

Parse->validate->retry like the other agents. On repeated failure it falls back
to researching the raw topic so the workflow can always make progress.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from polymath.agents.base import BaseAgent
from polymath.models.openrouter import chat_completion
from polymath.models.parsing import parse_json
from polymath.models.router import Role, models_for
from polymath.models.schemas import ResearchPlan

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "planner.md"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")


class PlannerAgent(BaseAgent):
    role = Role.PLANNER
    name = "planner"

    async def plan(self, topic: str, max_retries: int = 2) -> ResearchPlan:
        prompt = _PROMPT_TEMPLATE.replace("{{TOPIC}}", topic)
        messages: list[dict] = [{"role": "user", "content": prompt}]
        model = models_for(self.role)

        for attempt in range(1, max_retries + 2):
            message = await chat_completion(model, messages, temperature=0.3)
            content = message.get("content") or ""
            try:
                plan = ResearchPlan.model_validate(parse_json(content))
                if plan.subtasks:
                    self.log.info("planner.plan", n_subtasks=len(plan.subtasks), attempt=attempt)
                    return plan
                raise ValueError("empty subtasks")
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                self.log.warning("planner.parse_failed", attempt=attempt, error=str(exc)[:200])
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That was not valid. Error: {exc}. Return ONLY "
                            '{"subtasks": ["...", "..."]} — no prose, no fences.'
                        ),
                    }
                )

        self.log.warning("planner.gave_up", topic=topic)
        return ResearchPlan(subtasks=[topic])
