"""Helpers for pulling JSON out of LLM text output.

Models often wrap JSON in markdown fences or surround it with prose; these
helpers tolerate that. Shared by the Reader and Critic agents.
"""

from __future__ import annotations

import json
import re


def strip_to_json(content: str) -> str:
    """Remove markdown code fences and surrounding whitespace."""
    s = content.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    return s


def parse_json(content: str) -> dict:
    """Parse a JSON object from model output, tolerating a prose wrapper.

    Raises json.JSONDecodeError if no JSON object can be found.
    """
    s = strip_to_json(content)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise
