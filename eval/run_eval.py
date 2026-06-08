"""Week 2 acceptance eval: structured-extraction validation rates.

Runs the Reader pipeline over N topics and measures, at the extraction level
(one extraction = one page's JSON validated against the schema):

    first-try rate   = extractions that validated on attempt 1 / total
    within-2 rate    = extractions that validated within 2 retries / total

Acceptance (PROJECT_SPEC.md Week 2): first-try >= 90%, within-2 >= 99%.

We measure per extraction because a claim only exists once its extraction has
validated — so a claim-level "within-2" rate would be trivially 100%. The
extraction is the unit that either passes Pydantic validation or doesn't.

Usage:
    uv run python eval/run_eval.py --topics 5 --pages 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import structlog
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from polymath.config import settings  # noqa: E402
from week2_reader import gather_claims  # noqa: E402

TOPICS_FILE = ROOT / "eval" / "test_topics.yaml"
RESULTS_DIR = ROOT / "eval" / "results"

FIRST_TRY_THRESHOLD = 0.90
WITHIN_2_THRESHOLD = 0.99


def _configure_logging() -> None:
    # Eval prints a human report; keep agent logs at WARNING to reduce noise.
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )


async def _eval(n_topics: int, pages: int) -> dict:
    topics = yaml.safe_load(TOPICS_FILE.read_text(encoding="utf-8"))["topics"][:n_topics]

    per_topic = []
    total = first_try = within_2 = total_claims = 0

    for topic in topics:
        results = await gather_claims(topic, n_pages=pages)
        t_total = len(results)
        t_first = sum(1 for r in results if r.validated and r.attempts == 1)
        t_within = sum(1 for r in results if r.validated)
        t_claims = sum(len(r.claims) for r in results)

        total += t_total
        first_try += t_first
        within_2 += t_within
        total_claims += t_claims

        per_topic.append(
            {
                "topic": topic,
                "extractions": t_total,
                "first_try_valid": t_first,
                "within_2_valid": t_within,
                "claims": t_claims,
            }
        )
        print(
            f"  {topic[:45]:<45} extractions={t_total} "
            f"first-try={t_first} within-2={t_within} claims={t_claims}"
        )

    first_rate = first_try / total if total else 0.0
    within_rate = within_2 / total if total else 0.0
    passed = first_rate >= FIRST_TRY_THRESHOLD and within_rate >= WITHIN_2_THRESHOLD

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_topics": len(topics),
        "pages_per_topic": pages,
        "total_extractions": total,
        "total_claims": total_claims,
        "first_try_rate": round(first_rate, 4),
        "within_2_rate": round(within_rate, 4),
        "thresholds": {"first_try": FIRST_TRY_THRESHOLD, "within_2": WITHIN_2_THRESHOLD},
        "passed": passed,
        "per_topic": per_topic,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 2 acceptance eval.")
    parser.add_argument("--topics", type=int, default=5)
    parser.add_argument("--pages", type=int, default=2)
    args = parser.parse_args()

    _configure_logging()
    if not settings.tavily_api_key or not settings.openrouter_api_key:
        sys.exit("Both OPENROUTER_API_KEY and TAVILY_API_KEY must be set in .env.")

    print(f"Running Week 2 eval: {args.topics} topics x {args.pages} pages\n")
    report = asyncio.run(_eval(args.topics, args.pages))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"week2-{ts}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Extractions: {report['total_extractions']} | claims: {report['total_claims']}")
    print(f"First-try valid: {report['first_try_rate']:.1%} (need >= {FIRST_TRY_THRESHOLD:.0%})")
    print(f"Within-2 valid:  {report['within_2_rate']:.1%} (need >= {WITHIN_2_THRESHOLD:.0%})")
    print(f"RESULT: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
