"""WriterAgent — synthesizes a cited markdown report from accumulated claims.

Unlike the other agents the Writer returns prose, not JSON, so there is no
parse/validate/retry loop — it formats the claims into the prompt and returns
the model's markdown.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from polymath.agents.base import BaseAgent
from polymath.models.openrouter import chat_completion
from polymath.models.parsing import parse_json
from polymath.models.router import Role, model_for
from polymath.models.schemas import Claim, Slide, SlideDeck

_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
_PROMPT_TEMPLATE = (_PROMPT_DIR / "writer.md").read_text(encoding="utf-8")
_DECK_TEMPLATE = (_PROMPT_DIR / "slides.md").read_text(encoding="utf-8")


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

    async def build_deck(
        self, *, topic: str, claims: list[Claim], max_retries: int = 2
    ) -> SlideDeck:
        """Turn claims into a structured SlideDeck (parse/validate/retry)."""
        prompt = _DECK_TEMPLATE.replace("{{TOPIC}}", topic).replace(
            "{{CLAIMS}}", _format_claims(claims)
        )
        messages: list[dict] = [{"role": "user", "content": prompt}]
        model = model_for(self.role)

        for attempt in range(1, max_retries + 2):
            message = await chat_completion(model, messages, temperature=0.3)
            content = message.get("content") or ""
            try:
                deck = SlideDeck.model_validate(parse_json(content))
                if deck.slides:
                    self.log.info("writer.deck", n_slides=len(deck.slides), attempt=attempt)
                    return deck
                raise ValueError("empty slides")
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                self.log.warning("writer.deck_parse_failed", attempt=attempt, error=str(exc)[:200])
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That was not valid. Error: {exc}. Return ONLY the JSON "
                            "deck object — no prose, no markdown fences."
                        ),
                    }
                )

        self.log.warning("writer.deck_fallback", topic=topic)
        return _fallback_deck(topic, claims)


def _fallback_deck(topic: str, claims: list[Claim], max_slides: int = 5) -> SlideDeck:
    """Minimal deck built directly from claims when the LLM can't produce one."""
    slides = [
        Slide(title=c.claim[:80], bullets=[c.evidence_quote[:200]] if c.evidence_quote else [], source_url=c.source_url)
        for c in claims[:max_slides]
    ]
    if not slides:
        slides = [Slide(title=topic, bullets=[], source_url="")]
    return SlideDeck(title=topic, slides=slides)
