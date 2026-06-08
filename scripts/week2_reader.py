"""Week 2 demo: topic -> search -> fetch pages -> Reader extracts structured Claims.

Usage:
    uv run python scripts/week2_reader.py "solid-state batteries" --pages 3

Writes outputs/week2-<topic>.json with the validated claims. This is still a flat
pipeline (LangGraph orchestration arrives in Week 4); it exists to exercise the
ReaderAgent end to end and to be reused by eval/run_eval.py.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polymath.agents.reader import ReaderAgent, ReaderResult  # noqa: E402
from polymath.config import settings  # noqa: E402
from polymath.tools.page_fetch import page_fetch  # noqa: E402
from polymath.tools.web_search import web_search  # noqa: E402

log = structlog.get_logger("week2")
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


async def gather_claims(
    topic: str, n_pages: int = 3, max_results: int = 6
) -> list[ReaderResult]:
    """Search the topic, fetch the top pages, run the Reader on each.

    Returns one ReaderResult per successfully-fetched page (so callers can
    inspect per-page attempts/validated for metrics).
    """
    results = await web_search(topic, max_results=max_results)
    reader = ReaderAgent()

    out: list[ReaderResult] = []
    fetched = 0
    for r in results:
        if fetched >= n_pages:
            break
        url = r.get("url")
        if not url:
            continue
        text = await page_fetch(url)
        if not text:
            continue
        fetched += 1
        out.append(await reader.extract(text=text, source_url=url))
    return out


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "topic"


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
    parser = argparse.ArgumentParser(description="Polymath Week 2 Reader demo.")
    parser.add_argument("topic")
    parser.add_argument("--pages", type=int, default=3, help="Pages to read (default 3).")
    args = parser.parse_args()

    _configure_logging()
    results = asyncio.run(gather_claims(args.topic, n_pages=args.pages))

    all_claims = [c for res in results for c in res.claims]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = OUTPUT_DIR / f"week2-{_slugify(args.topic)}-{ts}.json"
    out_path.write_text(
        json.dumps(
            {
                "topic": args.topic,
                "generated_utc": ts,
                "claims": [c.model_dump() for c in all_claims],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    n_pages = len(results)
    n_valid_pages = sum(1 for r in results if r.validated)
    print(f"\nPages read: {n_pages} | pages validated: {n_valid_pages} | claims: {len(all_claims)}")
    print(f"Claims written to: {out_path}")


if __name__ == "__main__":
    main()
