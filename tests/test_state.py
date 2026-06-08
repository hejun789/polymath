"""Tests for the LangGraph GraphState model."""

from polymath.graph.state import GraphState


def test_defaults():
    s = GraphState(topic="x")
    assert s.topic == "x"
    assert s.max_iterations == 3
    assert s.iteration == 0
    assert s.subtasks == []
    assert s.pending_urls == []
    assert s.claims == []
    assert s.decision == ""
    assert s.report == ""
    assert s.trace == []


def test_accepts_overrides():
    s = GraphState(topic="x", max_iterations=5, subtasks=["q1", "q2"])
    assert s.max_iterations == 5
    assert s.subtasks == ["q1", "q2"]
