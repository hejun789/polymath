"""LangGraph state machine wiring the full research pipeline.

    planner -> search -> reader -> critic -> (search | writer) -> END

Nodes are thin orchestration over the already-tested agents and tools. All
collaborators are injected via WorkflowDeps so the graph can be tested with
fakes; `make_default_workflow()` wires the real agents + tools + Chroma store.

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
    results_per_subtask: int = 2
    max_pages_per_iter: int = 3


def route_after_critic(state: GraphState) -> str:
    """Decide whether to loop back to search or proceed to the writer."""
    if (
        state.decision == "stop"
        or state.iteration >= state.max_iterations
        or not state.new_subtasks
    ):
        return "writer"
    return "search"


def build_workflow(deps: WorkflowDeps):
    """Build and compile the StateGraph from injected dependencies."""

    async def planner_node(state: GraphState) -> dict:
        log.info("node.planner.enter", topic=state.topic)
        plan = await deps.plan_fn(state.topic)
        log.info("node.planner.exit", subtasks=plan.subtasks)
        return {"subtasks": plan.subtasks, "trace": [{"node": "planner", "subtasks": plan.subtasks}]}

    async def search_node(state: GraphState) -> dict:
        log.info("node.search.enter", subtasks=state.subtasks)
        urls: list[str] = []
        for query in state.subtasks:
            for r in await deps.search_fn(query, deps.results_per_subtask):
                url = r.get("url")
                if url and url not in urls:
                    urls.append(url)
        urls = urls[: deps.max_pages_per_iter]
        log.info("node.search.exit", n_urls=len(urls))
        return {"pending_urls": urls, "trace": [{"node": "search", "n_urls": len(urls)}]}

    async def reader_node(state: GraphState) -> dict:
        log.info("node.reader.enter", n_urls=len(state.pending_urls))
        for url in state.pending_urls:
            text = await deps.fetch_fn(url)
            if not text:
                continue
            result = await deps.extract_fn(text=text, source_url=url)
            deps.store.add_claims(result.claims)
        claims = deps.store.all_claims()
        new_iteration = state.iteration + 1
        log.info("node.reader.exit", claims_total=len(claims), iteration=new_iteration)
        return {
            "claims": claims,
            "iteration": new_iteration,
            "pending_urls": [],
            "trace": [{"node": "reader", "claims_total": len(claims), "iteration": new_iteration}],
        }

    async def critic_node(state: GraphState) -> dict:
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
        report = await deps.write_fn(topic=state.topic, claims=state.claims)
        log.info("node.writer.exit", report_chars=len(report))
        return {"report": report, "trace": [{"node": "writer", "report_chars": len(report)}]}

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


def make_default_workflow():
    """Wire the real agents, tools, and a fresh Chroma store."""
    from polymath.agents.critic import CriticAgent
    from polymath.agents.planner import PlannerAgent
    from polymath.agents.reader import ReaderAgent
    from polymath.agents.writer import WriterAgent
    from polymath.memory.vector_store import ClaimStore
    from polymath.tools.page_fetch import page_fetch
    from polymath.tools.web_search import web_search

    store = ClaimStore(persist_dir=settings.chroma_persist_dir)
    store.reset()  # fresh per-session memory

    deps = WorkflowDeps(
        plan_fn=PlannerAgent().plan,
        search_fn=web_search,
        fetch_fn=page_fetch,
        extract_fn=ReaderAgent().extract,
        review_fn=CriticAgent().review,
        write_fn=WriterAgent().write,
        store=store,
    )
    return build_workflow(deps)


async def run_workflow(topic: str, max_iterations: int = 3) -> dict:
    """Run the full pipeline with web_search/page_fetch flowing through MCP, then
    synthesize a slide deck. Returns the final state plus a "deck" key.
    """
    from polymath.agents.critic import CriticAgent
    from polymath.agents.planner import PlannerAgent
    from polymath.agents.reader import ReaderAgent
    from polymath.agents.writer import WriterAgent
    from polymath.memory.vector_store import ClaimStore
    from polymath.tools.mcp_client import mcp_tools

    writer = WriterAgent()

    async with mcp_tools() as tools:  # spawns the MCP tool server over stdio
        store = ClaimStore(persist_dir=settings.chroma_persist_dir)
        store.reset()
        deps = WorkflowDeps(
            plan_fn=PlannerAgent().plan,
            search_fn=tools.web_search,  # via MCP
            fetch_fn=tools.page_fetch,  # via MCP
            extract_fn=ReaderAgent().extract,
            review_fn=CriticAgent().review,
            write_fn=writer.write,
            store=store,
        )
        graph = build_workflow(deps)
        result = await graph.ainvoke(GraphState(topic=topic, max_iterations=max_iterations))

    # Second artifact: a slide deck synthesized from the gathered claims.
    result["deck"] = await writer.build_deck(topic=topic, claims=result["claims"])
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
