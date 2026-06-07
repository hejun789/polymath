"""Minimal async OpenRouter chat client (OpenAI-compatible /chat/completions).

Raw httpx per project convention (no SDK). Returns the first choice's `message`
dict so callers can inspect `content` and `tool_calls` directly.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from polymath.config import settings

log = structlog.get_logger(__name__)


async def chat_completion(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
    temperature: float = 0.2,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Call OpenRouter and return the assistant `message` dict from choice 0."""
    if not settings.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to .env (key at openrouter.ai)."
        )

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "HTTP-Referer": "https://github.com/polymath",
        "X-Title": "Polymath",
    }

    log.info("openrouter.request", model=model, n_messages=len(messages))
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{settings.openrouter_base_url}/chat/completions",
            json=body,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"OpenRouter returned no choices: {data}")

    return data["choices"][0]["message"]
