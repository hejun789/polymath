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
        return self._responses.pop(0)


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

    with pytest.raises(RuntimeError, match="throttled"):
        await openrouter.chat_completion(["a", "b"], [{"role": "user", "content": "x"}])
