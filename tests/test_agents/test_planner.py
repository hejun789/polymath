"""Tests for PlannerAgent: topic -> ResearchPlan(subtasks), with retries."""

import json

from polymath.agents import planner as planner_mod
from polymath.agents.planner import PlannerAgent


def _msg(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _patch(monkeypatch, responses):
    queue = list(responses)

    async def fake_complete(model, messages, **kwargs):
        return _msg(queue.pop(0))

    monkeypatch.setattr(planner_mod, "chat_completion", fake_complete)


async def test_plans_subtasks(monkeypatch):
    _patch(monkeypatch, [json.dumps({"subtasks": ["history of X", "current state of X"]})])
    plan = await PlannerAgent().plan("X")
    assert plan.subtasks == ["history of X", "current state of X"]


async def test_retries_bad_then_succeeds(monkeypatch):
    _patch(monkeypatch, ["not json", json.dumps({"subtasks": ["a"]})])
    plan = await PlannerAgent().plan("X")
    assert plan.subtasks == ["a"]


async def test_falls_back_to_topic_on_failure(monkeypatch):
    _patch(monkeypatch, ["nope", "still nope", "nope again"])
    plan = await PlannerAgent().plan("my topic")
    assert plan.subtasks == ["my topic"]  # fail-safe: research the raw topic
