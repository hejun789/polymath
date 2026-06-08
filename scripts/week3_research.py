"""Week 3: iterative research loop with memory + Critic.

Flow per PROJECT_SPEC.md Week 3:
    subtasks = [topic]
    repeat up to max_iterations:
        for each subtask: search -> fetch -> Reader -> store claims in Chroma
        Critic reads ALL stored claims -> {continue, new_subtasks} | {stop, reason}
        if stop / no new subtasks / last iteration: break
        subtasks = new_subtasks

Writes outputs/week3-<topic>.json with the final claims and the decision trace.

Usage:
    uv run python scripts/week3_research.py "electric vehicles" --max-iterations 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import structlog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from polymath.agents.critic import CriticAgent  # noqa: E402
from polymath.config import settings  # noqa: E402
from polymath.memory.vector_store import ClaimStore  # noqa: E402
from week2_reader import gather_claims  # noqa: E402

log = structlog.get_logger("week3")
OUTPUT_DIR = ROOT / "outputs"


async def research(topic: str, max_iterations: int = 3, pages: int = 2) -> dict:
    # Fresh per-session memory: reset the persisted collection at run start.
    store = ClaimStore(persist_dir=settings.chroma_persist_dir)
    store.reset()
    critic = CriticAgent()

    subtasks = [topic]
    trace: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        log.info("research.iteration", n=iteration, subtasks=subtasks)
        for subtask in subtasks:
            for result in await gather_claims(subtask, n_pages=pages):
                store.add_claims(result.claims)

        all_claims = store.all_claims()
        decision = await critic.review(
            topic=topic,
            claims=all_claims,
            iteration=iteration,
            max_iterations=max_iterations,
        )
        trace.append(
            {
                "iteration": iteration,
                "subtasks": subtasks,
                "claims_total": len(all_claims),
                "decision": decision.decision,
                "new_subtasks": decision.new_subtasks,
                "reason": decision.reason,
            }
        )
        log.info(
            "research.decision",
            iteration=iteration,
            decision=decision.decision,
            claims_total=len(all_claims),
        )

        if decision.decision == "stop" or not decision.new_subtasks:
            break
        subtasks = decision.new_subtasks

    return {
        "topic": topic,
        "iterations_run": len(trace),
        "claims": [c.model_dump() for c in store.all_claims()],
        "trace": trace,
    }


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
    parser = argparse.ArgumentParser(description="Polymath Week 3 iterative research.")
    parser.add_argument("topic")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--pages", type=int, default=2)
    args = parser.parse_args()

    _configure_logging()
    report = asyncio.run(research(args.topic, args.max_iterations, args.pages))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = OUTPUT_DIR / f"week3-{_slugify(args.topic)}-{ts}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nIterations run: {report['iterations_run']} | total claims: {len(report['claims'])}")
    for step in report["trace"]:
        print(
            f"  iter {step['iteration']}: {step['decision']} "
            f"(claims={step['claims_total']}, new_subtasks={len(step['new_subtasks'])})"
        )
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
