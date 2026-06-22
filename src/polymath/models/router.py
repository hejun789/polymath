"""Single source of truth for which model each agent role uses.

No model IDs anywhere else in the codebase — agents/scripts ask for a role,
not a model string. Swapping a model is a one-line change here. Any value can
be overridden via an env var like POLYMATH_MODEL_SEARCH=...
"""

from __future__ import annotations

import os
from enum import Enum


class Role(str, Enum):
    PLANNER = "planner"
    SEARCH = "search"
    READER = "reader"
    CRITIC = "critic"
    WRITER = "writer"


# Free-tier OpenRouter variants. NOTE: the model IDs in PROJECT_SPEC.md §4
# (deepseek-r1:free, gemini-2.0-flash-exp:free) were retired from OpenRouter's
# free tier; these are current equivalents verified against the live /models
# list on 2026-06-07. Roles that call tools (search/reader) use an instruct
# model with tool support; reasoning-heavy roles use a reasoning-capable model.
# NOTE: free models churn constantly. glm-4.5-air:free was retired by 2026-06-22.
# These are verified live against the /models list on 2026-06-22. The fallback
# chain (models_for) + 404 rotation in openrouter.py mean a future retirement is
# skipped automatically rather than crashing a run.
_DEFAULTS: dict[Role, str] = {
    Role.PLANNER: "openai/gpt-oss-120b:free",
    Role.SEARCH: "openai/gpt-oss-120b:free",
    Role.READER: "openai/gpt-oss-120b:free",
    Role.CRITIC: "openai/gpt-oss-120b:free",
    Role.WRITER: "openai/gpt-oss-120b:free",
}


# Reliable free chat/JSON models, verified available on OpenRouter. Used as a
# fallback chain: if the primary model is rate-limited (429), the client rotates
# to the next one immediately instead of waiting out a long Retry-After.
_FALLBACK_POOL: list[str] = [
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-coder:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]


def model_for(role: Role) -> str:
    """Return the primary model id for a role, allowing env override."""
    env_key = f"POLYMATH_MODEL_{role.value.upper()}"
    return os.environ.get(env_key) or _DEFAULTS[role]


def models_for(role: Role) -> list[str]:
    """Return [primary, ...fallbacks] for a role, de-duplicated, primary first.

    Lets a caller try the next free model the instant the current one is
    throttled, rather than sleeping on a single rate-limited endpoint.
    """
    primary = model_for(role)
    chain = [primary]
    for model in _FALLBACK_POOL:
        if model not in chain:
            chain.append(model)
    return chain
