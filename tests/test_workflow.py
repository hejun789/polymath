"""Tests for the LangGraph workflow: routing logic + full graph run with fakes."""

from polymath.graph.state import GraphState
from polymath.graph.workflow import WorkflowDeps, build_workflow, route_after_critic
from polymath.models.schemas import Claim, CriticDecision, ResearchPlan


# ---- route_after_critic (pure logic) ----

def _state(**kw) -> GraphState:
    base = dict(topic="t", max_iterations=3, iteration=1, decision="continue", new_subtasks=["q"])
    base.update(kw)
    return GraphState(**base)


def test_route_stop_goes_to_writer():
    assert route_after_critic(_state(decision="stop")) == "writer"


def test_route_continue_goes_to_search():
    assert route_after_critic(_state(decision="continue", iteration=1)) == "search"


def test_route_max_iterations_goes_to_writer():
    assert route_after_critic(_state(decision="continue", iteration=3, max_iterations=3)) == "writer"


def test_route_no_new_subtasks_goes_to_writer():
    assert route_after_critic(_state(decision="continue", new_subtasks=[])) == "writer"


# ---- full graph integration with fakes (no network) ----

class FakeStore:
    def __init__(self):
        self._by_id = {}

    def add_claims(self, claims):
        for c in claims:
            self._by_id[(c.claim, c.source_url)] = c

    def all_claims(self):
        return list(self._by_id.values())


class ReaderResultLike:
    def __init__(self, claims):
        self.claims = claims
        self.attempts = 1
        self.validated = True


def _make_deps(review_fn):
    async def plan_fn(topic):
        return ResearchPlan(subtasks=["q1"])

    async def search_fn(query, max_results):
        return [{"url": f"https://ex/{query}", "title": "", "content": ""}]

    async def fetch_fn(url):
        return f"text for {url}"

    async def extract_fn(*, text, source_url):
        return ReaderResultLike(
            [Claim(claim=f"claim from {source_url}", evidence_quote="q", source_url=source_url, confidence="high")]
        )

    async def write_fn(*, topic, claims):
        return f"# Report on {topic}\n\n{len(claims)} claims."

    return WorkflowDeps(
        plan_fn=plan_fn,
        search_fn=search_fn,
        fetch_fn=fetch_fn,
        extract_fn=extract_fn,
        review_fn=review_fn,
        write_fn=write_fn,
        store=FakeStore(),
    )


async def test_graph_stops_immediately():
    async def review_fn(*, topic, claims, iteration, max_iterations):
        return CriticDecision(decision="stop", reason="done")

    graph = build_workflow(_make_deps(review_fn))
    result = await graph.ainvoke(GraphState(topic="batteries"))

    assert result["report"].startswith("# Report on batteries")
    assert result["iteration"] == 1
    assert len(result["claims"]) >= 1
    nodes = [t["node"] for t in result["trace"]]
    assert nodes == ["planner", "search", "reader", "critic", "writer"]


async def test_graph_loops_once_then_stops():
    async def review_fn(*, topic, claims, iteration, max_iterations):
        if iteration < 2:
            return CriticDecision(decision="continue", new_subtasks=["q2"])
        return CriticDecision(decision="stop", reason="enough")

    graph = build_workflow(_make_deps(review_fn))
    result = await graph.ainvoke(GraphState(topic="batteries", max_iterations=3))

    assert result["iteration"] == 2
    nodes = [t["node"] for t in result["trace"]]
    assert nodes.count("search") == 2  # looped back through search once
    assert nodes.count("critic") == 2
    assert result["report"] != ""


async def test_astream_values_yields_progress_and_final():
    """The app streams stream_mode='values'; verify the contract it relies on:
    intermediate states grow the trace, and the final state has the report."""
    async def review_fn(*, topic, claims, iteration, max_iterations):
        return CriticDecision(decision="stop", reason="done")

    graph = build_workflow(_make_deps(review_fn))
    states = []
    async for state in graph.astream(GraphState(topic="t"), stream_mode="values"):
        states.append(state)

    # Trace grows monotonically across streamed states.
    trace_lens = [len(s.get("trace", [])) for s in states]
    assert trace_lens == sorted(trace_lens)
    final = states[-1]
    assert final["report"] != ""
    assert [t["node"] for t in final["trace"]] == ["planner", "search", "reader", "critic", "writer"]
