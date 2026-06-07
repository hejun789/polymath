"""page_fetch tool — fetch a URL and extract clean main-content text.

httpx fetches the HTML (async, per project convention); trafilatura does the
extraction (best-in-class boilerplate removal).
"""

from __future__ import annotations

import httpx
import structlog
import trafilatura

from polymath.config import settings  # noqa: F401  (kept for symmetry / future config)

log = structlog.get_logger(__name__)

_DEFAULT_MAX_CHARS = 8000
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; PolymathBot/0.1; +research agent)"
    )
}


async def page_fetch(url: str, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
    """Fetch `url` and return cleaned main text, truncated to `max_chars`.

    Returns an empty string if the page can't be fetched or no text is found,
    so a single bad URL never crashes the run.
    """
    log.info("page_fetch.request", url=url)
    try:
        async with httpx.AsyncClient(
            timeout=30.0, follow_redirects=True, headers=_HEADERS
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
    except httpx.HTTPError as exc:
        log.warning("page_fetch.fetch_failed", url=url, error=str(exc))
        return ""

    text = trafilatura.extract(html, include_comments=False, include_tables=False)
    if not text:
        log.warning("page_fetch.no_text", url=url)
        return ""

    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[...truncated...]"
    log.info("page_fetch.response", url=url, n_chars=len(text))
    return text
