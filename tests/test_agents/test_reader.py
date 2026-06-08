"""Tests for ReaderAgent: parse -> validate -> retry (<=2 retries)."""

import json

import pytest

from polymath.agents import reader as reader_mod
from polymath.agents.reader import ReaderAgent

SRC = "https://example.com/page"


def _msg(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _claims_json(**overrides) -> str:
    claim = {
        "claim": "Solid-state batteries are safer.",
        "evidence_quote": "Solid electrolytes are non-flammable.",
        "source_url": "https://model-put-this.example",
        "confidence": "high",
    }
    claim.update(overrides)
    return json.dumps({"claims": [claim]})


def _patch_responses(monkeypatch, responses: list[str]):
    """Make chat_completion return the queued response strings in order."""
    queue = list(responses)

    async def fake_complete(model, messages, **kwargs):
        return _msg(queue.pop(0))

    monkeypatch.setattr(reader_mod, "chat_completion", fake_complete)


async def test_clean_json_validates_first_try(monkeypatch):
    _patch_responses(monkeypatch, [_claims_json()])
    result = await ReaderAgent().extract(text="...", source_url=SRC)
    assert result.validated is True
    assert result.attempts == 1
    assert len(result.claims) == 1


async def test_source_url_is_forced_to_given_url(monkeypatch):
    # Model returns a different/wrong source_url; reader must override it.
    _patch_responses(monkeypatch, [_claims_json(source_url="https://wrong.example")])
    result = await ReaderAgent().extract(text="...", source_url=SRC)
    assert result.claims[0].source_url == SRC


async def test_strips_code_fences(monkeypatch):
    fenced = "```json\n" + _claims_json() + "\n```"
    _patch_responses(monkeypatch, [fenced])
    result = await ReaderAgent().extract(text="...", source_url=SRC)
    assert result.validated is True
    assert result.attempts == 1


async def test_retries_after_bad_then_succeeds(monkeypatch):
    _patch_responses(monkeypatch, ["this is not json at all", _claims_json()])
    result = await ReaderAgent().extract(text="...", source_url=SRC)
    assert result.validated is True
    assert result.attempts == 2


async def test_exhausts_retries_and_reports_failure(monkeypatch):
    _patch_responses(monkeypatch, ["nope", "still nope", "nope again"])
    result = await ReaderAgent().extract(text="...", source_url=SRC)
    assert result.validated is False
    assert result.attempts == 3
    assert result.claims == []
