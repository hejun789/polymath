"""ReaderAgent — extracts structured Claims from page text.

Prompts the LLM for JSON, parses it, validates against the `ExtractedClaims`
schema, and retries (feeding the error back) up to `max_retries` times. The
page's source_url is authoritative: it is stamped onto every claim regardless
of what the model returns, so a claim can never cite the wrong page.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from polymath.agents.base import BaseAgent
from polymath.models.openrouter import chat_completion
from polymath.models.parsing import parse_json
from polymath.models.router import Role, model_for
from polymath.models.schemas import Claim, ExtractedClaims

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "reader.md"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")

_SCHEMA_HINT = (
    '{"claims": [{"claim": str, "evidence_quote": str, '
    '"source_url": str, "confidence": "high"|"medium"|"low"}]}'
)


@dataclass
class ReaderResult:
    claims: list[Claim] = field(default_factory=list)
    attempts: int = 0
    validated: bool = False


class ReaderAgent(BaseAgent):
    role = Role.READER
    name = "reader"

    async def extract(
        self, *, text: str, source_url: str, max_retries: int = 2
    ) -> ReaderResult:
        prompt = _PROMPT_TEMPLATE.replace("{{SOURCE_URL}}", source_url).replace(
            "{{TEXT}}", text
        )
        messages: list[dict] = [{"role": "user", "content": prompt}]
        model = model_for(self.role)

        attempts = 0
        for attempt in range(1, max_retries + 2):  # attempts 1..(max_retries+1)
            attempts = attempt
            message = await chat_completion(model, messages, temperature=0)
            content = message.get("content") or ""
            try:
                data = parse_json(content)
                for c in data.get("claims", []):
                    if isinstance(c, dict):
                        c["source_url"] = source_url  # authoritative
                extracted = ExtractedClaims.model_validate(data)
                self.log.info(
                    "reader.extracted",
                    source_url=source_url,
                    n_claims=len(extracted.claims),
                    attempt=attempt,
                )
                return ReaderResult(
                    claims=extracted.claims, attempts=attempt, validated=True
                )
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                self.log.warning(
                    "reader.parse_failed", attempt=attempt, error=str(exc)[:200]
                )
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

        self.log.warning(
            "reader.gave_up", source_url=source_url, attempts=attempts
        )
        return ReaderResult(claims=[], attempts=attempts, validated=False)
