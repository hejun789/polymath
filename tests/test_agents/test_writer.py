"""Tests for WriterAgent: synthesizes a markdown report from claims."""

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


async def test_returns_report_markdown(monkeypatch):
    _patch(monkeypatch, "# Report\n\nSolid-state batteries are safer ([src](https://example.com/ssb)).")
    report = await WriterAgent().write(topic="solid-state batteries", claims=_claims())
    assert report.startswith("# Report")
    assert "https://example.com/ssb" in report


async def test_strips_surrounding_whitespace(monkeypatch):
    _patch(monkeypatch, "\n\n  # Report body  \n\n")
    report = await WriterAgent().write(topic="t", claims=_claims())
    assert report == "# Report body"
