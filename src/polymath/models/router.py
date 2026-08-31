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
# NOTE: OpenRouter's free lineup is highly volatile — models get retired (404),
# moved behind payment (402), gated (403), throttled (429), or become unusably
# slow, and they flip between these states within minutes (models that returned
# 402 one hour were serving fine the next). So the strategy is NOT to bet on one
# "best" model but to keep a DEEP chain of verified-capable ones and let
# openrouter.py rotate through it.
#
# Every id below was verified on 2026-08-31 by actually calling it on both real
# workloads — JSON claim extraction AND long-form report writing — not by a
# "say ok" ping and not by trusting the /models listing (which lists models that
# then refuse to serve). Excluded despite being listed: nemotron-3-ultra-550b
# (writes reports with zero citation links), nemotron-3.5-content-safety (can't
# produce valid JSON), nemotron-3.5-lightning (~370s on long prompts).
_DEFAULTS: dict[Role, str] = {
    Role.PLANNER: "minimax/minimax-m3:free",
    Role.SEARCH: "minimax/minimax-m3:free",
    Role.READER: "minimax/minimax-m3:free",
    Role.CRITIC: "minimax/minimax-m3:free",
    Role.WRITER: "minimax/minimax-m3:free",
}


# Fallback chain, ordered by measured speed on the JSON-extraction workload.
# Timings are extraction / long-form writing.
_FALLBACK_POOL: list[str] = [
    "minimax/minimax-m3:free",                  # 3.2s / 16.9s — best quality
    "inclusionai/ling-3.0-flash-fin:free",      # 3.7s / 23.7s
    "cohere/north-mini-code:free",              # 3.7s / 16.3s
    "nvidia/nemotron-3-super-120b-a12b:free",   # 8.8s / 15.8s
    "poolside/laguna-s-2.1:free",               # ~15s / ~15s
    "dots-studio/dots-3-note-preview:free",     # 26.9s / 28.5s
    "minimax/minimax-m2.7:free",                # 18.7s / 75.8s — last resort
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
