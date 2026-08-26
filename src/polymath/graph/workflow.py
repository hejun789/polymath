"""LangGraph state machine wiring the full research pipeline.

    planner -> search -> reader -> critic -> (search | writer) -> END

Nodes are thin orchestration over the already-tested agents and tools. All
collaborators are injected via WorkflowDeps so the graph can be tested with
fakes; `run_workflow()` / `make_direct_workflow()` wire the real agents + tools.

Run end to end:
    python -m polymath.graph.workflow --topic "current state of solid-state batteries"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import structlog
from langgraph.graph import END, StateGraph

from polymath.config import settings
from polymath.graph.state import GraphState

log = structlog.get_logger("workflow")


@dataclass
class WorkflowDeps:
    """Collaborators the graph nodes call. Real ones in production, fakes in tests."""

    plan_fn: Callable[[str], Awaitable[Any]]
    search_fn: Callable[..., Awaitable[list[dict]]]
    fetch_fn: Callable[[str], Awaitable[str]]
    extract_fn: Callable[..., Awaitable[Any]]
    review_fn: Callable[..., Awaitable[Any]]
    write_fn: Callable[..., Awaitable[str]]
    store: Any
    deck_fn: Callable[..., Awaitable[Any]] | None = None  # build the slide deck
    results_per_subtask: int = 3
    # Pages are fetched+extracted concurrently, so a higher cap costs little
    # wall-clock time but lets more of the Planner's angles actually get read.
    max_pages_per_iter: int = 4
    # ~44% of real search results are blocked/paywalled and fetch empty, so we
    # shortlist this many times the budget and keep the first pages that work.
    fetch_oversample: int = 2


def route_after_critic(state: GraphState) -> str:
    """Decide whether to loop back to search or proceed to the writer."""
    if (
        state.decision == "stop"
        or state.iteration >= state.max_iterations
        or not state.new_subtasks
    ):
        return "writer"
    return "search"


def select_urls(
    per_subtask: list[list[str]], seen_urls: list[str], limit: int
) -> list[str]:
    """Pick up to `limit` URLs, round-robin across subtasks.

    Taking one URL from each subtask before any subtask's second keeps every
    angle the Planner identified represented. Concatenating then truncating
    starved all but the first subtask. Already-read URLs are skipped so a later
    iteration never re-fetches (and re-extracts) a page we already paid for.
    """
    chosen: list[str] = []
    skip = set(seen_urls)
    depth = max((len(urls) for urls in per_subtask), default=0)
    for i in range(depth):
        for urls in per_subtask:
            if len(chosen) >= limit:
                return chosen
            if i < len(urls) and urls[i] not in skip:
                chosen.append(urls[i])
                skip.add(urls[i])
    return chosen


def build_workflow(deps: WorkflowDeps):
    """Build and compile the StateGraph from injected dependencies."""

    async def planner_node(state: GraphState) -> dict:
        log.info("node.planner.enter", topic=state.topic)
        plan = await deps.plan_fn(state.topic)
        log.info("node.planner.exit", subtasks=plan.subtasks)
        return {"subtasks": plan.subtasks, "trace": [{"node": "planner", "subtasks": plan.subtasks}]}

    async def search_node(state: GraphState) -> dict:
        log.info("node.search.enter", subtasks=state.subtasks)
        # Each subtask is an independent network call — query them concurrently.
        raw = await asyncio.gather(
            *(deps.search_fn(q, deps.results_per_subtask) for q in state.subtasks),
            return_exceptions=True,
        )
        per_subtask: list[list[str]] = []
        for result in raw:
            if isinstance(result, BaseException):
                log.warning("node.search.query_failed", error=str(result)[:200])
                per_subtask.append([])
                continue
            per_subtask.append([hit["url"] for hit in result if hit.get("url")])

        budget = deps.max_pages_per_iter * deps.fetch_oversample
        urls = select_urls(per_subtask, state.seen_urls, budget)
        log.info("node.search.exit", n_urls=len(urls), n_subtasks=len(state.subtasks))
        return {
            "pending_urls": urls,
            "seen_urls": state.seen_urls + urls,
            "trace": [{"node": "search", "n_urls": len(urls)}],
        }

    async def reader_node(state: GraphState) -> dict:
        log.info("node.reader.enter", n_candidates=len(state.pending_urls))

        # Phase 1 — fetch candidates concurrently. This is cheap (no LLM), and
        # many real results come back empty (bot-blocked, paywalled, JS-only).
        texts = await asyncio.gather(
            *(deps.fetch_fn(u) for u in state.pending_urls), return_exceptions=True
        )
        usable: list[tuple[str, str]] = []
        for url, text in zip(state.pending_urls, texts):
            if isinstance(text, BaseException):
                log.warning("node.reader.fetch_failed", url=url, error=str(text)[:200])
                continue
            if text:
                usable.append((url, text))
        usable = usable[: deps.max_pages_per_iter]

        # Phase 2 — spend the expensive extraction budget only on pages that
        # actually have content. One failure must not kill the run.
        results = await asyncio.gather(
            *(deps.extract_fn(text=t, source_url=u) for u, t in usable),
            return_exceptions=True,
        )
        for (url, _), result in zip(usable, results):
            if isinstance(result, BaseException):
                log.warning("node.reader.extract_failed", url=url, error=str(result)[:200])
                continue
            if result is not None:
                deps.store.add_claims(result.claims)

        claims = deps.store.all_claims()
        new_iteration = state.iteration + 1
        log.info(
            "node.reader.exit",
            pages_read=len(usable),
            candidates=len(state.pending_urls),
            claims_total=len(claims),
            iteration=new_iteration,
        )
        return {
            "claims": claims,
            "iteration": new_iteration,
            "pending_urls": [],
            "trace": [
                {
                    "node": "reader",
                    "claims_total": len(claims),
                    "pages_read": len(usable),
                    "iteration": new_iteration,
                }
            ],
        }

    async def critic_node(state: GraphState) -> dict:
        # On the final allowed iteration the verdict can't change routing (we go
        # to the writer regardless), so skip the LLM call entirely.
        if state.iteration >= state.max_iterations:
            log.info("node.critic.skip", iteration=state.iteration)
            return {
                "decision": "stop",
                "new_subtasks": [],
                "reason": "max iterations reached",
                "subtasks": [],
                "trace": [{"node": "critic", "decision": "stop", "skipped": True}],
            }

        log.info("node.critic.enter", claims_total=len(state.claims), iteration=state.iteration)
        decision = await deps.review_fn(
            topic=state.topic,
            claims=state.claims,
            iteration=state.iteration,
            max_iterations=state.max_iterations,
        )
        log.info("node.critic.exit", decision=decision.decision, n_subtasks=len(decision.new_subtasks))
        return {
            "decision": decision.decision,
            "new_subtasks": decision.new_subtasks,
            "reason": decision.reason,
            "subtasks": decision.new_subtasks,  # queries for the next loop, if any
            "trace": [{"node": "critic", "decision": decision.decision, "new_subtasks": decision.new_subtasks}],
        }

    async def writer_node(state: GraphState) -> dict:
        log.info("node.writer.enter", claims_total=len(state.claims))
        # Report and deck are independent — build them concurrently.
        if deps.deck_fn is not None:
            report, deck = await asyncio.gather(
                deps.write_fn(topic=state.topic, claims=state.claims),
                deps.deck_fn(topic=state.topic, claims=state.claims),
            )
        else:
            report = await deps.write_fn(topic=state.topic, claims=state.claims)
            deck = None
        log.info("node.writer.exit", report_chars=len(report))
        out = {"report": report, "trace": [{"node": "writer", "report_chars": len(report)}]}
        if deck is not None:
            out["deck"] = deck
        return out

    g = StateGraph(GraphState)
    g.add_node("planner", planner_node)
    g.add_node("search", search_node)
    g.add_node("reader", reader_node)
    g.add_node("critic", critic_node)
    g.add_node("writer", writer_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "search")
    g.add_edge("search", "reader")
    g.add_edge("reader", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"search": "search", "writer": "writer"})
    g.add_edge("writer", END)
    return g.compile()


def _assemble_deps(search_fn, fetch_fn, store, writer) -> WorkflowDeps:
    """Bundle the real agents with the given search/fetch functions."""
    from polymath.agents.critic import CriticAgent
    from polymath.agents.planner import PlannerAgent
    from polymath.agents.reader import ReaderAgent

    return WorkflowDeps(
        plan_fn=PlannerAgent().plan,
        search_fn=search_fn,
        fetch_fn=fetch_fn,
        extract_fn=ReaderAgent().extract,
        review_fn=CriticAgent().review,
        write_fn=writer.write,
        deck_fn=writer.build_deck,
        store=store,
    )


def make_direct_workflow(persist_dir: str | None = None):
    """Graph wired with direct (non-MCP) tools + an ephemeral store by default.

    Returns (compiled_graph, writer). Used by the Streamlit app, which streams
    node updates and builds the deck from the final claims.
    """
    from polymath.agents.writer import WriterAgent
    from polymath.memory.vector_store import ClaimStore
    from polymath.tools.page_fetch import page_fetch
    from polymath.tools.web_search import web_search

    store = ClaimStore(persist_dir=persist_dir)
    store.reset()
    writer = WriterAgent()
    return build_workflow(_assemble_deps(web_search, page_fetch, store, writer)), writer


async def run_workflow(topic: str, max_iterations: int = 3, use_mcp: bool = True) -> dict:
    """Run the full pipeline and synthesize a slide deck.

    use_mcp=True routes web_search/page_fetch through the MCP server (CLI default);
    use_mcp=False calls the tools directly (used by the app). Returns the final
    state plus a "deck" key.
    """
    from polymath.agents.writer import WriterAgent
    from polymath.memory.vector_store import ClaimStore

    writer = WriterAgent()
    initial = GraphState(topic=topic, max_iterations=max_iterations)

    if use_mcp:
        from polymath.tools.mcp_client import mcp_tools

        async with mcp_tools() as tools:  # spawns the MCP tool server over stdio
            store = ClaimStore(persist_dir=settings.chroma_persist_dir)
            store.reset()
            graph = build_workflow(_assemble_deps(tools.web_search, tools.page_fetch, store, writer))
            result = await graph.ainvoke(initial)
    else:
        from polymath.tools.page_fetch import page_fetch
        from polymath.tools.web_search import web_search

        store = ClaimStore(persist_dir=settings.chroma_persist_dir)
        store.reset()
        graph = build_workflow(_assemble_deps(web_search, page_fetch, store, writer))
        result = await graph.ainvoke(initial)

    # The deck is built concurrently with the report inside the writer node.
    return result


# ---- CLI entrypoint ----

def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "topic"


def _configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(level),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymath research workflow (LangGraph).")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--max-iterations", type=int, default=3)
    args = parser.parse_args()

    _configure_logging()
    if not settings.openrouter_api_key or not settings.tavily_api_key:
        sys.exit("Both OPENROUTER_API_KEY and TAVILY_API_KEY must be set in .env.")

    result = asyncio.run(run_workflow(args.topic, args.max_iterations))

    from polymath.outputs.slides import render_deck

    out_dir = Path(__file__).resolve().parents[3] / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    stem = f"{_slugify(args.topic)}-{ts}"

    report_path = out_dir / f"{stem}.md"
    report_path.write_text(result["report"] + "\n", encoding="utf-8")
    deck_path = render_deck(result["deck"], out_dir / f"{stem}.pptx")

    print(f"\nNode transitions: {' -> '.join(t['node'] for t in result['trace'])}")
    print(f"Iterations: {result['iteration']} | claims: {len(result['claims'])} | slides: {len(result['deck'].slides)}")
    print(f"Report written to: {report_path}")
    print(f"Deck written to:   {deck_path}")


if __name__ == "__main__":
    main()
