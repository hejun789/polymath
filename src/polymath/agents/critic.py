"""CriticAgent — reads accumulated claims and decides continue-vs-stop.

This is the first agent that reasons about other agents' output: it looks for
gaps and contradictions and either proposes new subtasks (continue) or declares
coverage sufficient (stop). On repeated parse failure it defaults to `stop` so
the research loop can never spin forever on a malformed response.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from polymath.agents.base import BaseAgent
from polymath.models.openrouter import chat_completion
from polymath.models.parsing import parse_json
from polymath.models.router import Role, models_for
from polymath.models.schemas import Claim, CriticDecision

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "critic.md"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")

_SCHEMA_HINT = (
    '{"decision": "continue"|"stop", "new_subtasks": [str, ...], "reason": str}'
)


def _format_claims(claims: list[Claim]) -> str:
    if not claims:
        return "(no claims gathered yet)"
    return "\n".join(f"{i}. {c.claim}" for i, c in enumerate(claims, 1))


class CriticAgent(BaseAgent):
    role = Role.CRITIC
    name = "critic"

    async def review(
        self,
        *,
        topic: str,
        claims: list[Claim],
        iteration: int = 1,
        max_iterations: int = 3,
        max_retries: int = 2,
    ) -> CriticDecision:
        prompt = (
            _PROMPT_TEMPLATE.replace("{{TOPIC}}", topic)
            .replace("{{ITERATION}}", str(iteration))
            .replace("{{MAX_ITERATIONS}}", str(max_iterations))
            .replace("{{CLAIMS}}", _format_claims(claims))
        )
        messages: list[dict] = [{"role": "user", "content": prompt}]
        model = models_for(self.role)

        for attempt in range(1, max_retries + 2):
            message = await chat_completion(model, messages, temperature=0)
            content = message.get("content") or ""
            try:
                decision = CriticDecision.model_validate(parse_json(content))
                self.log.info(
                    "critic.decision",
                    decision=decision.decision,
                    n_subtasks=len(decision.new_subtasks),
                    attempt=attempt,
                )
                return decision
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                self.log.warning("critic.parse_failed", attempt=attempt, error=str(exc)[:200])
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That response was not valid. Error: {exc}. "
                            f"Return ONLY a JSON object of the form {_SCHEMA_HINT}. "
                            "No prose, no markdown fences."
                        ),
                    }
                )

        self.log.warning("critic.gave_up", topic=topic)
        return CriticDecision(
            decision="stop", reason="critic failed to produce a valid decision"
        )
