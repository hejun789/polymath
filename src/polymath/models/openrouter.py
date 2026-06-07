"""Minimal async OpenRouter chat client (OpenAI-compatible /chat/completions).

Raw httpx per project convention (no SDK). Returns the first choice's `message`
dict so callers can inspect `content` and `tool_calls` directly.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

from polymath.config import settings

log = structlog.get_logger(__name__)

# Free-tier models get briefly throttled upstream (HTTP 429) or hit transient
# 5xx. Retry a few times, honoring Retry-After when present.
_RETRY_STATUSES = {429, 502, 503, 504}
_MAX_RETRIES = 5
_DEFAULT_BACKOFF = 3.0


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
    url = f"{settings.openrouter_base_url}/chat/completions"

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, _MAX_RETRIES + 1):
            resp = await client.post(url, json=body, headers=headers)

            if resp.status_code in _RETRY_STATUSES and attempt < _MAX_RETRIES:
                wait = _retry_after_seconds(resp, attempt)
                log.warning(
                    "openrouter.retry",
                    status=resp.status_code,
                    attempt=attempt,
                    wait=wait,
                )
                await asyncio.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            break

    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"OpenRouter returned no choices: {data}")

    return data["choices"][0]["message"]


def _retry_after_seconds(resp: httpx.Response, attempt: int) -> float:
    """Seconds to wait before retrying — prefer server hint, else backoff."""
    header = resp.headers.get("Retry-After")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    try:
        meta = resp.json().get("error", {}).get("metadata", {})
        if (hinted := meta.get("retry_after_seconds")) is not None:
            return float(hinted)
    except Exception:  # noqa: BLE001 - best-effort hint parsing
        pass
    return _DEFAULT_BACKOFF * attempt  # linear backoff: 3s, 6s, 9s, ...
