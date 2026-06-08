"""WriterAgent — synthesizes a cited markdown report from accumulated claims.

Unlike the other agents the Writer returns prose, not JSON, so there is no
parse/validate/retry loop — it formats the claims into the prompt and returns
the model's markdown.
"""

from __future__ import annotations

from pathlib import Path

from polymath.agents.base import BaseAgent
from polymath.models.openrouter import chat_completion
from polymath.models.router import Role, model_for
from polymath.models.schemas import Claim

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "writer.md"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")


def _format_claims(claims: list[Claim]) -> str:
    if not claims:
        return "(no claims available)"
    lines = []
    for i, c in enumerate(claims, 1):
        lines.append(
            f'{i}. {c.claim} [confidence: {c.confidence}] '
            f'(source: {c.source_url}) — "{c.evidence_quote}"'
        )
    return "\n".join(lines)


class WriterAgent(BaseAgent):
    role = Role.WRITER
    name = "writer"

    async def write(self, *, topic: str, claims: list[Claim]) -> str:
        prompt = _PROMPT_TEMPLATE.replace("{{TOPIC}}", topic).replace(
            "{{CLAIMS}}", _format_claims(claims)
        )
        messages = [{"role": "user", "content": prompt}]
        model = model_for(self.role)
        self.log.info("writer.write", n_claims=len(claims))
        message = await chat_completion(model, messages, temperature=0.3)
        return (message.get("content") or "").strip()
