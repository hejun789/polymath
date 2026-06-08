"""MCP server exposing Polymath's tools over the Model Context Protocol.

Tools: web_search, page_fetch, claim_extract. Run as a stdio server:
    python -m polymath.mcp_server.server

Tool bodies are plain async impl functions (registered with FastMCP) so they can
be unit-tested in-process without the transport.
"""

from __future__ import annotations

import sys

import structlog
from mcp.server.fastmcp import FastMCP

from polymath.tools.page_fetch import page_fetch as _page_fetch
from polymath.tools.web_search import web_search as _web_search

# Lazily constructed so importing this module (e.g. in tests) is cheap.
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        from polymath.agents.reader import ReaderAgent

        _reader = ReaderAgent()
    return _reader


async def web_search_impl(query: str, max_results: int = 5) -> list[dict]:
    """Search the web; returns [{title, url, content}, ...]."""
    return await _web_search(query, max_results=max_results)


async def page_fetch_impl(url: str) -> str:
    """Fetch a URL and return cleaned main-content text."""
    return await _page_fetch(url)


async def claim_extract_impl(text: str, source_url: str) -> list[dict]:
    """Extract structured claims from page text; returns a list of claim dicts."""
    result = await _get_reader().extract(text=text, source_url=source_url)
    return [c.model_dump() for c in result.claims]


mcp = FastMCP("polymath-tools")
mcp.tool(name="web_search")(web_search_impl)
mcp.tool(name="page_fetch")(page_fetch_impl)
mcp.tool(name="claim_extract")(claim_extract_impl)


def main() -> None:
    # stdio transport uses STDOUT for JSON-RPC, so all logging MUST go to STDERR
    # or it corrupts the protocol stream.
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
