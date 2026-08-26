"""Tests for the LangGraph workflow: routing logic + full graph run with fakes."""

import asyncio

from polymath.graph.state import GraphState
from polymath.graph.workflow import (
    WorkflowDeps,
    build_workflow,
    route_after_critic,
    select_urls,
)
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


# ---- URL selection: round-robin across subtasks + skip already-read pages ----

def test_select_urls_takes_one_from_each_subtask_before_seconds():
    """The Planner's whole point is covering different angles. Truncating a
    sequentially-built list starved every subtask but the first."""
    per_subtask = [
        ["https://a/1", "https://a/2"],
        ["https://b/1", "https://b/2"],
        ["https://c/1", "https://c/2"],
    ]
    chosen = select_urls(per_subtask, seen_urls=[], limit=3)
    assert chosen == ["https://a/1", "https://b/1", "https://c/1"]


def test_select_urls_wraps_to_second_round_when_limit_allows():
    per_subtask = [["https://a/1", "https://a/2"], ["https://b/1", "https://b/2"]]
    assert select_urls(per_subtask, seen_urls=[], limit=4) == [
        "https://a/1", "https://b/1", "https://a/2", "https://b/2",
    ]


def test_select_urls_skips_already_read_pages():
    per_subtask = [["https://a/1", "https://a/2"], ["https://b/1"]]
    chosen = select_urls(per_subtask, seen_urls=["https://a/1"], limit=2)
    assert "https://a/1" not in chosen
    assert chosen == ["https://b/1", "https://a/2"]


def test_select_urls_dedupes_same_url_from_two_subtasks():
    per_subtask = [["https://dup"], ["https://dup"], ["https://c/1"]]
    assert select_urls(per_subtask, seen_urls=[], limit=5) == ["https://dup", "https://c/1"]


# ---- resilience + concurrency ----

async def test_reader_survives_one_failing_page():
    """A single bad page (or a throttled extraction) must not kill the whole run."""
    async def plan_fn(topic):
        return ResearchPlan(subtasks=["q1", "q2"])

    async def search_fn(query, max_results):
        return [{"url": f"https://ex/{query}", "title": "", "content": ""}]

    async def fetch_fn(url):
        return f"text for {url}"

    async def extract_fn(*, text, source_url):
        if source_url.endswith("q1"):
            raise RuntimeError("all models throttled for this page")
        return ReaderResultLike(
            [Claim(claim="survived", evidence_quote="q", source_url=source_url, confidence="high")]
        )

    async def review_fn(*, topic, claims, iteration, max_iterations):
        return CriticDecision(decision="stop", reason="done")

    async def write_fn(*, topic, claims):
        return f"# Report ({len(claims)} claims)"

    deps = WorkflowDeps(
        plan_fn=plan_fn, search_fn=search_fn, fetch_fn=fetch_fn, extract_fn=extract_fn,
        review_fn=review_fn, write_fn=write_fn, store=FakeStore(),
    )
    result = await build_workflow(deps).ainvoke(GraphState(topic="t"))

    assert result["report"] != ""            # run completed despite the failure
    assert len(result["claims"]) == 1        # the healthy page still contributed
    assert result["claims"][0].claim == "survived"


async def test_search_queries_run_concurrently():
    in_flight = 0
    peak = 0

    async def search_fn(query, max_results):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return [{"url": f"https://ex/{query}", "title": "", "content": ""}]

    async def plan_fn(topic):
        return ResearchPlan(subtasks=["q1", "q2", "q3"])

    async def fetch_fn(url):
        return "text"

    async def extract_fn(*, text, source_url):
        return ReaderResultLike([])

    async def review_fn(*, topic, claims, iteration, max_iterations):
        return CriticDecision(decision="stop", reason="done")

    async def write_fn(*, topic, claims):
        return "# Report"

    deps = WorkflowDeps(
        plan_fn=plan_fn, search_fn=search_fn, fetch_fn=fetch_fn, extract_fn=extract_fn,
        review_fn=review_fn, write_fn=write_fn, store=FakeStore(),
    )
    await build_workflow(deps).ainvoke(GraphState(topic="t"))

    assert peak > 1, "subtask searches should overlap, not run one after another"


async def test_reader_backfills_past_pages_that_fail_to_fetch():
    """~44% of real search results are blocked/paywalled and return empty text.
    The reader should skip those and still spend its extraction budget on
    `max_pages_per_iter` pages that actually have content."""
    extracted: list[str] = []

    async def plan_fn(topic):
        return ResearchPlan(subtasks=["q1"])

    async def search_fn(query, max_results):
        return [{"url": f"https://ex/{i}", "title": "", "content": ""} for i in range(6)]

    async def fetch_fn(url):
        return "" if url in ("https://ex/0", "https://ex/1") else f"text {url}"

    async def extract_fn(*, text, source_url):
        extracted.append(source_url)
        return ReaderResultLike(
            [Claim(claim=f"c {source_url}", evidence_quote="q", source_url=source_url, confidence="high")]
        )

    async def review_fn(*, topic, claims, iteration, max_iterations):
        return CriticDecision(decision="stop", reason="done")

    async def write_fn(*, topic, claims):
        return "# Report"

    deps = WorkflowDeps(
        plan_fn=plan_fn, search_fn=search_fn, fetch_fn=fetch_fn, extract_fn=extract_fn,
        review_fn=review_fn, write_fn=write_fn, store=FakeStore(), max_pages_per_iter=2,
    )
    result = await build_workflow(deps).ainvoke(GraphState(topic="t"))

    # Blocked pages skipped; the two healthy ones extracted instead.
    assert extracted == ["https://ex/2", "https://ex/3"]
    assert len(result["claims"]) == 2
    # Dead links are remembered so the next iteration doesn't retry them.
    assert "https://ex/0" in result["seen_urls"]
