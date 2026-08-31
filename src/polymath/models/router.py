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
# NOTE: OpenRouter's free lineup churns hard — models get retired (404), moved
# behind payment (402), gated (403), or silently become unusably slow. Every id
# below was verified by actually CALLING it with a representative prompt on
# 2026-08-31, not just by appearing in the /models listing (the listing lies).
# laguna-s-2.1 handled both JSON extraction and long-form writing in ~15s.
_DEFAULTS: dict[Role, str] = {
    Role.PLANNER: "poolside/laguna-s-2.1:free",
    Role.SEARCH: "poolside/laguna-s-2.1:free",
    Role.READER: "poolside/laguna-s-2.1:free",
    Role.CRITIC: "poolside/laguna-s-2.1:free",
    Role.WRITER: "poolside/laguna-s-2.1:free",
}


# Fallback chain, ordered fastest-first. nemotron-3.5-lightning works but took
# ~370s on a long writing prompt (past the client timeout), so it sits last —
# the transport-error rotation in openrouter.py handles it gracefully.
_FALLBACK_POOL: list[str] = [
    "poolside/laguna-s-2.1:free",
    "poolside/laguna-xs-2.1:free",
    "nvidia/nemotron-3.5-lightning:free",
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
