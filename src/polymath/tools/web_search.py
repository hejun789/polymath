"""web_search tool — thin async wrapper over the Tavily Search API."""

from __future__ import annotations

import httpx
import structlog

from polymath.config import settings

log = structlog.get_logger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"


async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web via Tavily.

    Returns a list of {title, url, content} dicts (content is Tavily's snippet).
    Raises if no API key is configured or the request fails.
    """
    if not settings.tavily_api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Add it to .env (free key at tavily.com)."
        )

    payload = {
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
    }
    headers = {"Authorization": f"Bearer {settings.tavily_api_key}"}

    log.info("web_search.request", query=query, max_results=max_results)
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(_TAVILY_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in data.get("results", [])
    ]
    log.info("web_search.response", query=query, n_results=len(results))
    return results
