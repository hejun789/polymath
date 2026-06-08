"""Offline tests for the MCP tool handlers (no transport, no network)."""

from polymath.mcp_server import server
from polymath.models.schemas import Claim


async def test_web_search_handler(monkeypatch):
    async def fake(query, max_results=5):
        return [{"title": "t", "url": "https://x", "content": "c"}]

    monkeypatch.setattr(server, "_web_search", fake)
    out = await server.web_search_impl("solid-state batteries", max_results=3)
    assert out == [{"title": "t", "url": "https://x", "content": "c"}]


async def test_page_fetch_handler(monkeypatch):
    async def fake(url):
        return "clean text"

    monkeypatch.setattr(server, "_page_fetch", fake)
    assert await server.page_fetch_impl("https://x") == "clean text"


async def test_claim_extract_handler(monkeypatch):
    class FakeResult:
        claims = [Claim(claim="c", evidence_quote="q", source_url="https://x", confidence="high")]

    class FakeReader:
        async def extract(self, *, text, source_url):
            return FakeResult()

    monkeypatch.setattr(server, "_get_reader", lambda: FakeReader())
    out = await server.claim_extract_impl(text="...", source_url="https://x")
    assert out == [
        {"claim": "c", "evidence_quote": "q", "source_url": "https://x", "confidence": "high"}
    ]
