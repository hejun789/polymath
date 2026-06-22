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

# Free-tier models get briefly throttled (429) or hit transient 5xx; they also
# get RETIRED (404 "No endpoints found") since the free lineup churns. In all
# these cases, rotate to the next model in the chain immediately. Only throttles
# (429/5xx) warrant a short capped wait between full rounds — a 404 is permanent,
# so there's no point sleeping on it.
_THROTTLE_STATUSES = {429, 502, 503, 504}
_ROTATE_STATUSES = _THROTTLE_STATUSES | {404}
_MAX_ROUNDS = 3
_MAX_WAIT_CAP = 8.0  # never wait longer than this on a throttle


async def chat_completion(
    model: str | list[str],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
    temperature: float = 0.2,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Call OpenRouter and return the assistant `message` dict from choice 0.

    `model` may be a single id or a fallback chain (list). On a rate-limit/5xx
    the next model in the chain is tried immediately; a short capped wait is only
    inserted after a full round through the chain has all been throttled.
    """
    if not settings.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to .env (key at openrouter.ai)."
        )

    models = [model] if isinstance(model, str) else list(model)
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "HTTP-Referer": "https://github.com/polymath",
        "X-Title": "Polymath",
    }
    url = f"{settings.openrouter_base_url}/chat/completions"

    log.info("openrouter.request", models=models, n_messages=len(messages))
    last_throttle: httpx.Response | None = None

    async with httpx.AsyncClient(timeout=timeout) as client:
        for round_no in range(1, _MAX_ROUNDS + 1):
            for m in models:
                body: dict[str, Any] = {"model": m, "messages": messages, "temperature": temperature}
                if tools:
                    body["tools"] = tools
                    body["tool_choice"] = tool_choice

                resp = await client.post(url, json=body, headers=headers)
                if resp.status_code in _ROTATE_STATUSES:
                    log.warning("openrouter.skip_model", model=m, status=resp.status_code)
                    if resp.status_code in _THROTTLE_STATUSES:
                        last_throttle = resp  # only throttles justify a between-round wait
                    continue  # try the next model immediately

                resp.raise_for_status()  # raise on non-retryable errors (4xx/etc.)
                data = resp.json()
                if "choices" not in data or not data["choices"]:
                    raise RuntimeError(f"OpenRouter returned no choices: {data}")
                log.info("openrouter.ok", model=m, round=round_no)
                return data["choices"][0]["message"]

            # Whole chain throttled this round — brief capped wait, then retry.
            if round_no < _MAX_ROUNDS and last_throttle is not None:
                wait = min(_retry_after_seconds(last_throttle, round_no), _MAX_WAIT_CAP)
                log.warning("openrouter.round_throttled", round=round_no, wait=wait)
                await asyncio.sleep(wait)

    raise RuntimeError(
        f"All models throttled after {_MAX_ROUNDS} rounds: {models}"
    )


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
    return float(attempt)  # tiny default backoff; capped by caller anyway
