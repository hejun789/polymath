"""MCP client: spawn the Polymath tool server over stdio and call its tools.

`mcp_tools()` yields an object whose async `web_search`/`page_fetch` methods have
the same signatures the workflow already expects, so the graph can call tools
over MCP transparently.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

log = structlog.get_logger(__name__)

# Repo root (src/polymath/tools/mcp_client.py -> parents[3]); used as the server
# subprocess cwd so it finds .env for API keys.
_ROOT = Path(__file__).resolve().parents[3]


def _payload(result: Any) -> Any:
    """Unwrap an MCP CallToolResult into the plain Python value the tool returned."""
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        # FastMCP wraps non-dict returns as {"result": <value>}.
        if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
            return structured["result"]
        return structured
    texts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
    joined = "\n".join(texts)
    try:
        return json.loads(joined)
    except (json.JSONDecodeError, ValueError):
        return joined


class MCPTools:
    def __init__(self, session: ClientSession):
        self._session = session

    async def web_search(self, query: str, max_results: int = 5) -> list[dict]:
        result = await self._session.call_tool(
            "web_search", {"query": query, "max_results": max_results}
        )
        payload = _payload(result)
        return payload if isinstance(payload, list) else []

    async def page_fetch(self, url: str) -> str:
        result = await self._session.call_tool("page_fetch", {"url": url})
        payload = _payload(result)
        return payload if isinstance(payload, str) else ""


@asynccontextmanager
async def mcp_tools():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "polymath.mcp_server.server"],
        cwd=str(_ROOT),
        env=os.environ.copy(),
    )
    log.info("mcp_client.connect")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            log.info("mcp_client.ready", tools=[t.name for t in tools.tools])
            yield MCPTools(session)
