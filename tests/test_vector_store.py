"""Tests for the Chroma-backed ClaimStore.

A deterministic fake embedding function keeps these tests offline and fast
(no ONNX model download), while still exercising real Chroma add/query/dedup.
"""

import uuid

import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from polymath.memory.vector_store import ClaimStore
from polymath.models.schemas import Claim


class FakeEmbed(EmbeddingFunction):
    """Maps text to a fixed 16-dim vector deterministically (identical text -> identical vector)."""

    def __call__(self, input: Documents) -> Embeddings:
        out = []
        for text in input:
            vec = [0.0] * 16
            for byte in text.encode("utf-8"):
                vec[byte % 16] += 1.0
            out.append(vec)
        return out

    @staticmethod
    def name() -> str:
        return "fake-embed"


def _claim(text: str, url: str = "https://example.com/a", conf: str = "high") -> Claim:
    return Claim(claim=text, evidence_quote=f"quote for {text}", source_url=url, confidence=conf)


@pytest.fixture
def store():
    # Unique collection per test: EphemeralClient shares one backend per process.
    name = f"claims_test_{uuid.uuid4().hex}"
    return ClaimStore(persist_dir=None, embedding_function=FakeEmbed(), collection_name=name)


def test_add_then_all_claims_roundtrip(store):
    claims = [_claim("A is true"), _claim("B is false")]
    store.add_claims(claims)
    texts = {c.claim for c in store.all_claims()}
    assert texts == {"A is true", "B is false"}


def test_adding_duplicate_is_a_noop(store):
    store.add_claims([_claim("dup claim")])
    store.add_claims([_claim("dup claim")])  # same claim+url -> same id
    assert store.count() == 1


def test_count_reflects_distinct_claims(store):
    store.add_claims([_claim("one"), _claim("two"), _claim("three")])
    assert store.count() == 3


def test_query_returns_relevant_claim(store):
    store.add_claims([_claim("solid state batteries are safe"), _claim("the sky is blue")])
    results = store.query("solid state batteries are safe", k=1)
    assert len(results) == 1
    assert results[0].claim == "solid state batteries are safe"


def test_reset_clears_store(store):
    store.add_claims([_claim("x"), _claim("y")])
    store.reset()
    assert store.count() == 0
