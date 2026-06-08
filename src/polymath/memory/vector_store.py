"""Chroma-backed store of extracted Claims — the agents' shared working memory.

Each Claim is embedded (default: all-MiniLM-L6-v2 via Chroma's ONNX function)
and upserted under a deterministic id derived from its text + source URL, so
re-adding the same claim is a no-op. The store is the substrate the Critic
reads from to reason about gaps and contradictions.
"""

from __future__ import annotations

import hashlib

import chromadb
import structlog

from polymath.config import settings
from polymath.models.schemas import Claim, Confidence

log = structlog.get_logger(__name__)

_COLLECTION = "claims"


def _claim_id(claim: Claim) -> str:
    digest = hashlib.sha256(f"{claim.claim}|{claim.source_url}".encode("utf-8"))
    return digest.hexdigest()[:16]


class ClaimStore:
    """Add, retrieve, and semantically query Claims."""

    def __init__(
        self,
        persist_dir: str | None = None,
        embedding_function=None,
        collection_name: str = _COLLECTION,
    ):
        # persist_dir=None -> in-memory (tests / ephemeral sessions). Note that
        # EphemeralClient shares one in-memory backend per process, so distinct
        # logical stores must use distinct collection_names.
        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.EphemeralClient()
        self._embedding_function = embedding_function
        self._collection_name = collection_name
        self._collection = self._get_or_create()

    def _get_or_create(self):
        kwargs = {"name": self._collection_name}
        if self._embedding_function is not None:
            kwargs["embedding_function"] = self._embedding_function
        return self._client.get_or_create_collection(**kwargs)

    def add_claims(self, claims: list[Claim]) -> None:
        if not claims:
            return
        # De-dupe within the batch by id, then upsert (idempotent across calls).
        by_id: dict[str, Claim] = {_claim_id(c): c for c in claims}
        self._collection.upsert(
            ids=list(by_id.keys()),
            documents=[c.claim for c in by_id.values()],
            metadatas=[
                {
                    "source_url": c.source_url,
                    "evidence_quote": c.evidence_quote,
                    "confidence": c.confidence,
                }
                for c in by_id.values()
            ],
        )
        log.info("claimstore.add", n=len(by_id), total=self.count())

    def _to_claim(self, document: str, metadata: dict) -> Claim:
        confidence: Confidence = metadata.get("confidence", "low")  # type: ignore[assignment]
        return Claim(
            claim=document,
            evidence_quote=metadata.get("evidence_quote", ""),
            source_url=metadata.get("source_url", ""),
            confidence=confidence,
        )

    def all_claims(self) -> list[Claim]:
        data = self._collection.get(include=["documents", "metadatas"])
        return [
            self._to_claim(doc, meta)
            for doc, meta in zip(data["documents"], data["metadatas"])
        ]

    def query(self, text: str, k: int = 5) -> list[Claim]:
        n = min(k, self.count()) or 1
        res = self._collection.query(
            query_texts=[text], n_results=n, include=["documents", "metadatas"]
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        return [self._to_claim(doc, meta) for doc, meta in zip(docs, metas)]

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._get_or_create()
