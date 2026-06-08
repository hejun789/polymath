"""Week 3 acceptance eval: does the Critic spot a deliberately-omitted aspect?

For each hand-built case (claims that omit one major aspect), the Critic must
decide "continue" AND mention the omitted aspect (one of expected_keywords) in
its new_subtasks/reason.

Acceptance (PROJECT_SPEC.md Week 3): correct on >= 80% of 5 cases (>= 4/5).

Usage:
    uv run python eval/run_critic_eval.py
"""

from __future__ import annotations

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

from polymath.agents.critic import CriticAgent  # noqa: E402
from polymath.config import settings  # noqa: E402
from polymath.models.schemas import Claim  # noqa: E402

CASES_FILE = ROOT / "eval" / "critic_cases.yaml"
RESULTS_DIR = ROOT / "eval" / "results"
THRESHOLD = 0.80


def _configure_logging() -> None:
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        processors=[structlog.processors.add_log_level, structlog.processors.JSONRenderer()],
    )


def _as_claims(topic: str, claim_texts: list[str]) -> list[Claim]:
    url = f"https://example.com/{topic.replace(' ', '-')}"
    return [
        Claim(claim=t, evidence_quote=t, source_url=url, confidence="medium")
        for t in claim_texts
    ]


async def _eval() -> dict:
    cases = yaml.safe_load(CASES_FILE.read_text(encoding="utf-8"))["cases"]
    critic = CriticAgent()

    per_case = []
    correct = 0
    for case in cases:
        topic = case["topic"]
        decision = await critic.review(
            topic=topic, claims=_as_claims(topic, case["claims"])
        )
        haystack = " ".join(decision.new_subtasks + [decision.reason]).lower()
        matched_kw = next(
            (kw for kw in case["expected_keywords"] if kw.lower() in haystack), None
        )
        ok = decision.decision == "continue" and matched_kw is not None
        correct += int(ok)
        per_case.append(
            {
                "topic": topic,
                "omitted_aspect": case["omitted_aspect"],
                "decision": decision.decision,
                "matched_keyword": matched_kw,
                "new_subtasks": decision.new_subtasks,
                "correct": ok,
            }
        )
        mark = "OK " if ok else "MISS"
        print(f"  [{mark}] {topic[:40]:<40} decision={decision.decision} kw={matched_kw}")

    rate = correct / len(cases) if cases else 0.0
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_cases": len(cases),
        "correct": correct,
        "rate": round(rate, 4),
        "threshold": THRESHOLD,
        "passed": rate >= THRESHOLD,
        "per_case": per_case,
    }


def main() -> None:
    _configure_logging()
    if not settings.openrouter_api_key:
        sys.exit("OPENROUTER_API_KEY must be set in .env.")

    print("Running Week 3 Critic eval...\n")
    report = asyncio.run(_eval())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"week3-critic-{ts}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Correct: {report['correct']}/{report['n_cases']} = {report['rate']:.0%} (need >= {THRESHOLD:.0%})")
    print(f"RESULT: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
