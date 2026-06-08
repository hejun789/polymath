"""Tests for WriterAgent: synthesizes a markdown report from claims."""

import json

from polymath.agents import writer as writer_mod
from polymath.agents.writer import WriterAgent
from polymath.models.schemas import Claim


def _claims():
    return [
        Claim(
            claim="Solid-state batteries are safer.",
            evidence_quote="non-flammable",
            source_url="https://example.com/ssb",
            confidence="high",
        )
    ]


def _patch(monkeypatch, content: str):
    async def fake_complete(model, messages, **kwargs):
        return {"role": "assistant", "content": content}

    monkeypatch.setattr(writer_mod, "chat_completion", fake_complete)


def _patch_seq(monkeypatch, contents):
    queue = list(contents)

    async def fake_complete(model, messages, **kwargs):
        return {"role": "assistant", "content": queue.pop(0)}

    monkeypatch.setattr(writer_mod, "chat_completion", fake_complete)


async def test_returns_report_markdown(monkeypatch):
    _patch(monkeypatch, "# Report\n\nSolid-state batteries are safer ([src](https://example.com/ssb)).")
    report = await WriterAgent().write(topic="solid-state batteries", claims=_claims())
    assert report.startswith("# Report")
    assert "https://example.com/ssb" in report


async def test_strips_surrounding_whitespace(monkeypatch):
    _patch(monkeypatch, "\n\n  # Report body  \n\n")
    report = await WriterAgent().write(topic="t", claims=_claims())
    assert report == "# Report body"


async def test_build_deck_parses(monkeypatch):
    payload = json.dumps(
        {
            "title": "Solid-State Batteries",
            "slides": [
                {"title": "Safety", "bullets": ["non-flammable"], "source_url": "https://example.com/ssb"}
            ],
        }
    )
    _patch(monkeypatch, payload)
    deck = await WriterAgent().build_deck(topic="solid-state batteries", claims=_claims())
    assert deck.title == "Solid-State Batteries"
    assert deck.slides[0].title == "Safety"


async def test_build_deck_retries_then_succeeds(monkeypatch):
    good = json.dumps({"title": "T", "slides": [{"title": "A", "bullets": [], "source_url": "u"}]})
    _patch_seq(monkeypatch, ["not json", good])
    deck = await WriterAgent().build_deck(topic="t", claims=_claims())
    assert len(deck.slides) == 1


async def test_build_deck_falls_back_to_claims(monkeypatch):
    _patch_seq(monkeypatch, ["nope", "still nope", "nope again"])
    deck = await WriterAgent().build_deck(topic="my topic", claims=_claims())
    assert deck.title  # non-empty fallback title
    assert len(deck.slides) >= 1  # built from the claims
