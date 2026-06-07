"""Tests for the Week 1 citation-provenance verifier.

The verifier is the guardrail behind the "no hallucinated URLs" acceptance
criterion, so its URL normalization must be correct (and not produce false
positives on real citation styles like gpt-oss's 【n†url】 brackets).
"""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "week1_baseline.py"
_spec = importlib.util.spec_from_file_location("week1_baseline", _SCRIPT)
week1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(week1)


def test_cjk_bracket_citation_is_not_a_false_positive():
    seen = {"https://en.wikipedia.org/wiki/Solid-state_battery"}
    report = "Claim text 【1†https://en.wikipedia.org/wiki/Solid-state_battery】."
    n_cited, hallucinated = week1.verify_citations(report, seen)
    assert n_cited == 1
    assert hallucinated == []


def test_trailing_punctuation_is_trimmed():
    seen = {"https://example.com/page"}
    report = "See (https://example.com/page), and also https://example.com/page."
    _, hallucinated = week1.verify_citations(report, seen)
    assert hallucinated == []


def test_real_hallucination_is_caught():
    seen = {"https://example.com/real"}
    report = "Real ([x](https://example.com/real)) but fake (https://made-up.test/x)."
    n_cited, hallucinated = week1.verify_citations(report, seen)
    assert n_cited == 2
    assert hallucinated == ["https://made-up.test/x"]
