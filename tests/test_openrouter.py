"""Tests for chat_completion's model-rotation behavior (httpx mocked)."""

import httpx
import pytest

from polymath.models import openrouter


class FakeResp:
    def __init__(self, status_code, json_data=None, headers=None):
        self.status_code = status_code
        self._json = json_data or {}
        self.headers = headers or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)


class FakeClient:
    """Returns a queued response per POST, recording which model was requested."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.models_tried = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None):
        self.models_tried.append(json["model"])
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setattr(openrouter.settings, "openrouter_api_key", "test-key")


def _ok(text="hi"):
    return FakeResp(200, {"choices": [{"message": {"role": "assistant", "content": text}}]})


async def test_rotates_to_next_model_on_429(monkeypatch):
    client = FakeClient([FakeResp(429), _ok("from second")])
    monkeypatch.setattr(openrouter.httpx, "AsyncClient", lambda **kw: client)

    msg = await openrouter.chat_completion(["model-a", "model-b"], [{"role": "user", "content": "x"}])

    assert msg["content"] == "from second"
    assert client.models_tried == ["model-a", "model-b"]  # rotated, no wait on model-a


async def test_rotates_past_retired_model_404(monkeypatch):
    # A retired model returns 404 ("No endpoints found"); must skip to the next,
    # not crash. This is the bug that broke the live app when glm-4.5-air was retired.
    client = FakeClient([FakeResp(404), _ok("from live model")])
    monkeypatch.setattr(openrouter.httpx, "AsyncClient", lambda **kw: client)

    msg = await openrouter.chat_completion(["retired-model", "live-model"], [{"role": "user", "content": "x"}])

    assert msg["content"] == "from live model"
    assert client.models_tried == ["retired-model", "live-model"]


async def test_returns_first_model_when_it_succeeds(monkeypatch):
    client = FakeClient([_ok("from first")])
    monkeypatch.setattr(openrouter.httpx, "AsyncClient", lambda **kw: client)

    msg = await openrouter.chat_completion(["model-a", "model-b"], [{"role": "user", "content": "x"}])
    assert msg["content"] == "from first"
    assert client.models_tried == ["model-a"]  # never needed the fallback


async def test_raises_when_all_models_throttled(monkeypatch):
    # 2 models x 3 rounds = 6 throttles; capped waits make this fast.
    monkeypatch.setattr(openrouter, "_MAX_ROUNDS", 2)

    async def _no_sleep(_):
        return None

    monkeypatch.setattr(openrouter.asyncio, "sleep", _no_sleep)
    client = FakeClient([FakeResp(429)] * 4)
    monkeypatch.setattr(openrouter.httpx, "AsyncClient", lambda **kw: client)

    with pytest.raises(RuntimeError, match="No model in"):
        await openrouter.chat_completion(["a", "b"], [{"role": "user", "content": "x"}])


async def test_rotates_when_200_body_carries_upstream_error(monkeypatch):
    """OpenRouter answers HTTP 200 with an error payload when the upstream
    provider is overloaded. That must rotate to the next model, not abort
    the run (this crashed a live run: 'Upstream error from Nvidia')."""
    overloaded = FakeResp(200, {"error": {"message": "Service temporarily overloaded", "code": 502}})
    client = FakeClient([overloaded, _ok("from healthy model")])
    monkeypatch.setattr(openrouter.httpx, "AsyncClient", lambda **kw: client)

    msg = await openrouter.chat_completion(["overloaded", "healthy"], [{"role": "user", "content": "x"}])

    assert msg["content"] == "from healthy model"
    assert client.models_tried == ["overloaded", "healthy"]


async def test_raises_when_every_model_returns_error_body(monkeypatch):
    monkeypatch.setattr(openrouter, "_MAX_ROUNDS", 1)
    client = FakeClient([FakeResp(200, {"error": {"message": "boom"}})] * 2)
    monkeypatch.setattr(openrouter.httpx, "AsyncClient", lambda **kw: client)

    with pytest.raises(RuntimeError):
        await openrouter.chat_completion(["a", "b"], [{"role": "user", "content": "x"}])


async def test_rotates_when_a_model_times_out(monkeypatch):
    """A slow model raises httpx.ReadTimeout rather than returning a status.
    That must rotate to the next model, not crash the run (a 370s writer call
    blew the client timeout and killed a live run)."""
    client = FakeClient([httpx.ReadTimeout("too slow"), _ok("from fast model")])
    monkeypatch.setattr(openrouter.httpx, "AsyncClient", lambda **kw: client)

    msg = await openrouter.chat_completion(["slow", "fast"], [{"role": "user", "content": "x"}])

    assert msg["content"] == "from fast model"
    assert client.models_tried == ["slow", "fast"]


async def test_raises_when_every_model_times_out(monkeypatch):
    monkeypatch.setattr(openrouter, "_MAX_ROUNDS", 1)
    client = FakeClient([httpx.ConnectError("down")] * 2)
    monkeypatch.setattr(openrouter.httpx, "AsyncClient", lambda **kw: client)

    with pytest.raises(RuntimeError, match="No model in"):
        await openrouter.chat_completion(["a", "b"], [{"role": "user", "content": "x"}])
