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


# Free-tier OpenRouter variants (see PROJECT_SPEC.md §4).
_DEFAULTS: dict[Role, str] = {
    Role.PLANNER: "deepseek/deepseek-r1:free",
    Role.SEARCH: "google/gemini-2.0-flash-exp:free",
    Role.READER: "google/gemini-2.0-flash-exp:free",
    Role.CRITIC: "deepseek/deepseek-r1:free",
    Role.WRITER: "meta-llama/llama-3.3-70b-instruct:free",
}


def model_for(role: Role) -> str:
    """Return the model id for a role, allowing env override."""
    env_key = f"POLYMATH_MODEL_{role.value.upper()}"
    return os.environ.get(env_key) or _DEFAULTS[role]
