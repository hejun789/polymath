"""Tests for CriticAgent: gap analysis -> continue/stop decision with retries."""

import json

from polymath.agents import critic as critic_mod
from polymath.agents.critic import CriticAgent
from polymath.models.schemas import Claim


def _msg(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _claims() -> list[Claim]:
    return [
        Claim(claim="EVs have long range.", evidence_quote="q", source_url="u", confidence="high"),
        Claim(claim="EVs charge fast.", evidence_quote="q", source_url="u", confidence="high"),
    ]


def _patch(monkeypatch, responses: list[str]):
    queue = list(responses)

    async def fake_complete(model, messages, **kwargs):
        return _msg(queue.pop(0))

    monkeypatch.setattr(critic_mod, "chat_completion", fake_complete)


async def test_parses_continue_decision(monkeypatch):
    payload = json.dumps(
        {"decision": "continue", "new_subtasks": ["EV battery recycling", "EV emissions"]}
    )
    _patch(monkeypatch, [payload])
    d = await CriticAgent().review(topic="electric vehicles", claims=_claims())
    assert d.decision == "continue"
    assert "EV battery recycling" in d.new_subtasks


async def test_parses_stop_decision(monkeypatch):
    payload = json.dumps({"decision": "stop", "reason": "coverage is comprehensive"})
    _patch(monkeypatch, [payload])
    d = await CriticAgent().review(topic="electric vehicles", claims=_claims())
    assert d.decision == "stop"
    assert "comprehensive" in d.reason


async def test_retries_bad_then_succeeds(monkeypatch):
    good = json.dumps({"decision": "continue", "new_subtasks": ["x"]})
    _patch(monkeypatch, ["not json", good])
    d = await CriticAgent().review(topic="t", claims=_claims())
    assert d.decision == "continue"


async def test_exhausts_retries_defaults_to_stop(monkeypatch):
    _patch(monkeypatch, ["nope", "still nope", "nope again"])
    d = await CriticAgent().review(topic="t", claims=_claims())
    assert d.decision == "stop"  # fail-safe: never loop forever
