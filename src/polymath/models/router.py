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
_DEFAULTS: dict[Role, str] = {
    Role.PLANNER: "z-ai/glm-4.5-air:free",
    Role.SEARCH: "openai/gpt-oss-120b:free",
    Role.READER: "openai/gpt-oss-120b:free",
    Role.CRITIC: "z-ai/glm-4.5-air:free",
    # llama-3.3-70b:free is frequently rate-limited (429) upstream; gpt-oss-120b
    # has reliable free capacity and handles long-form synthesis well.
    Role.WRITER: "openai/gpt-oss-120b:free",
}


def model_for(role: Role) -> str:
    """Return the model id for a role, allowing env override."""
    env_key = f"POLYMATH_MODEL_{role.value.upper()}"
    return os.environ.get(env_key) or _DEFAULTS[role]
