"""Tests for the Pydantic schemas that cross the LLM boundary."""

import pytest
from pydantic import ValidationError

from polymath.models.schemas import Claim, ExtractedClaims


def test_valid_claim_constructs():
    c = Claim(
        claim="Solid-state batteries are safer.",
        evidence_quote="Solid electrolytes are non-flammable.",
        source_url="https://example.com/ssb",
        confidence="high",
    )
    assert c.confidence == "high"
    assert c.source_url == "https://example.com/ssb"


def test_invalid_confidence_rejected():
    with pytest.raises(ValidationError):
        Claim(
            claim="x",
            evidence_quote="y",
            source_url="https://example.com",
            confidence="very-high",  # not in the allowed literal set
        )


def test_missing_required_field_rejected():
    with pytest.raises(ValidationError):
        Claim(
            claim="x",
            evidence_quote="y",
            confidence="low",
        )  # missing source_url


def test_extracted_claims_parses_list():
    payload = {
        "claims": [
            {
                "claim": "a",
                "evidence_quote": "q",
                "source_url": "https://example.com/1",
                "confidence": "medium",
            },
            {
                "claim": "b",
                "evidence_quote": "q2",
                "source_url": "https://example.com/2",
                "confidence": "low",
            },
        ]
    }
    parsed = ExtractedClaims.model_validate(payload)
    assert len(parsed.claims) == 2
    assert parsed.claims[0].claim == "a"


def test_extracted_claims_rejects_bad_inner_claim():
    with pytest.raises(ValidationError):
        ExtractedClaims.model_validate(
            {"claims": [{"claim": "a", "evidence_quote": "q", "source_url": "u", "confidence": "bogus"}]}
        )
