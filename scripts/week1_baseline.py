"""Week 1 baseline: single LLM + web_search + page_fetch (function-calling style).

Usage:
    uv run python scripts/week1_baseline.py "current state of solid-state batteries"

Produces a ~1-page cited markdown summary in outputs/. This is intentionally a
flat script (not MCP, not LangGraph) — it proves the LLM+tools loop works end to
end before we build the agent graph in later phases.
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

# Make `import polymath...` work when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polymath.config import settings  # noqa: E402
from polymath.models.openrouter import chat_completion  # noqa: E402
from polymath.models.router import Role, model_for  # noqa: E402
from polymath.tools.page_fetch import page_fetch  # noqa: E402
from polymath.tools.web_search import web_search  # noqa: E402

log = structlog.get_logger("week1")

MAX_TOOL_ITERATIONS = 10
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

SYSTEM_PROMPT = """\
You are a research assistant. Given a topic, research it using the provided tools
and write a concise, ~1-page markdown summary.

Process:
1. Use `web_search` to find relevant, authoritative sources.
2. Use `page_fetch` to read the most promising pages for detail and exact facts.
3. Synthesize at least 5 substantive claims about the topic.

Rules:
- EVERY claim must cite a real source URL that you actually retrieved via the tools.
- NEVER invent or guess a URL. Only cite URLs returned by web_search / page_fetch.
- Use inline markdown citations like: claim text ([source](https://real-url)).
- End with a "## Sources" section listing every URL you cited.
- Keep it to roughly one page. Be specific and factual, not vague.

When you are done researching, output ONLY the final markdown report (no tool calls).
"""

# ---- Tool schemas exposed to the model (OpenAI function-calling format) ----

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for a query. Returns title, url, and a content snippet for each result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "How many results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "page_fetch",
            "description": "Fetch a URL and return its cleaned main-content text. Use on URLs from web_search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch."},
                },
                "required": ["url"],
            },
        },
    },
]


async def _dispatch_tool(name: str, args: dict) -> str:
    """Run a tool call and return its result as a string for the model."""
    if name == "web_search":
        results = await web_search(
            query=args["query"], max_results=int(args.get("max_results", 5))
        )
        return json.dumps(results, ensure_ascii=False)
    if name == "page_fetch":
        return await page_fetch(url=args["url"])
    return f"ERROR: unknown tool {name!r}"


async def run(topic: str) -> str:
    """Run the research loop and return the final markdown report."""
    model = model_for(Role.SEARCH)  # tool-calling-capable model for Week 1
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Research this topic: {topic}"},
    ]

    for i in range(MAX_TOOL_ITERATIONS):
        log.info("loop.iteration", n=i + 1, model=model)
        message = await chat_completion(model, messages, tools=TOOLS)
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            content = message.get("content") or ""
            log.info("loop.final", n_chars=len(content))
            return content

        for call in tool_calls:
            fn = call["function"]
            name = fn["name"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            log.info("tool.call", tool=name, args=args)
            try:
                result = await _dispatch_tool(name, args)
            except Exception as exc:  # noqa: BLE001 - report to model, keep loop alive
                log.warning("tool.error", tool=name, error=str(exc))
                result = f"ERROR running {name}: {exc}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result,
                }
            )

    raise RuntimeError(
        f"Hit MAX_TOOL_ITERATIONS ({MAX_TOOL_ITERATIONS}) without a final answer."
    )


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "report"


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
    parser = argparse.ArgumentParser(description="Polymath Week 1 baseline researcher.")
    parser.add_argument("topic", help="Research topic to summarize.")
    args = parser.parse_args()

    _configure_logging()
    report = asyncio.run(run(args.topic))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = OUTPUT_DIR / f"week1-{_slugify(args.topic)}-{ts}.md"
    header = f"# Research summary: {args.topic}\n\n*Generated {ts} UTC by Polymath Week 1 baseline.*\n\n"
    out_path.write_text(header + report.strip() + "\n", encoding="utf-8")

    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
