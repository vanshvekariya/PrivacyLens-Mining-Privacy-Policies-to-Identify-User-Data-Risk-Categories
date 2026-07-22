"""Tests for the explanation templates."""

import pytest

from privacylens.templates import (
    DISCLAIMER,
    MAX_EVIDENCE_LEN,
    explain,
    explain_all,
    sanitize_evidence,
)


def test_explain_is_cautious():
    text = explain("Third-Party Sharing")
    assert "appears to" in text.lower()


def test_explain_no_evidence_has_no_evidence_clause():
    assert "influenced" not in explain("Data Collection")
    assert "influenced" not in explain("Data Collection", None)
    assert "influenced" not in explain("Data Collection", [])


def test_explain_with_evidence_includes_it():
    text = explain("Third-Party Sharing", ["we may share"])
    assert "we may share" in text


def test_sanitize_dedupes_and_trims():
    spans = sanitize_evidence(["  share  ", "share", "SHARE"])
    assert spans == ["share"]


def test_sanitize_truncates_long_evidence():
    long = "x" * 500
    spans = sanitize_evidence([long])
    assert len(spans) == 1
    assert len(spans[0]) <= MAX_EVIDENCE_LEN + 1  # +1 for the ellipsis char


def test_sanitize_limits_item_count():
    spans = sanitize_evidence([f"phrase {i}" for i in range(20)])
    assert len(spans) <= 3


def test_html_like_evidence_kept_literal():
    # No HTML escaping happens in this module (that is the UI layer's job).
    text = explain("Data Security", ["<b>secure</b> & safe"])
    assert "<b>secure</b> & safe" in text
    assert "&lt;" not in text
    assert "&amp;" not in text


def test_explain_all_covers_every_label_dominant_first():
    out = explain_all(["Data Collection", "Third-Party Sharing"])
    cats = [o["category"] for o in out]
    assert set(cats) == {"Data Collection", "Third-Party Sharing"}
    assert cats[0] == "Third-Party Sharing"  # High-tier dominant first


def test_explain_all_empty():
    assert explain_all([]) == []


def test_explain_unknown_category_raises():
    with pytest.raises(ValueError):
        explain("Marketing")


def test_disclaimer_mentions_negation():
    assert "do not" in DISCLAIMER
