"""Tests for the attention framework and multi-label profile."""

import pytest

from privacylens.attention import attention_for, attention_profile, framework_table
from privacylens.config import CATEGORIES, canonicalize_category


def test_framework_covers_all_categories():
    rows = framework_table()
    assert {r["category"] for r in rows} == set(CATEGORIES)


def test_tiers_and_weights():
    assert attention_for("Third-Party Sharing")["attention_tier"] == "High"
    assert attention_for("Third-Party Sharing")["weight"] == 4
    assert attention_for("Data Security")["attention_tier"] == "Medium"  # not Low
    other = attention_for("Other/Unclear")
    assert other["attention_tier"] == "Unknown"
    assert other["review_flag"] is True


def test_canonicalize_variants():
    assert canonicalize_category("user control and deletion") == "User Control & Deletion"
    assert canonicalize_category(" THIRD party  sharing ") == "Third-Party Sharing"
    assert canonicalize_category("other") == "Other/Unclear"


def test_canonicalize_unknown_raises():
    with pytest.raises(ValueError):
        canonicalize_category("Marketing Emails")
    with pytest.raises(ValueError):
        canonicalize_category(123)


def test_empty_profile():
    prof = attention_profile([])
    assert prof["dominant_category"] is None
    assert prof["dominant_tier"] == "Unknown"
    assert prof["all_categories"] == []


def test_protective_does_not_cancel_high():
    # High (Third-Party) + Medium (User Control) -> dominant stays High.
    prof = attention_profile(["User Control & Deletion", "Third-Party Sharing"],
                             {"User Control & Deletion": 0.9,
                              "Third-Party Sharing": 0.6})
    assert prof["dominant_category"] == "Third-Party Sharing"
    assert prof["dominant_tier"] == "High"
    assert prof["tier_counts"]["high"] == 1
    assert prof["tier_counts"]["medium"] == 1


def test_dedupe_keeps_max_confidence():
    prof = attention_profile(["Data Collection", "data collection"],
                             {"Data Collection": 0.3, "data collection": 0.9})
    assert prof["all_categories"] == ["Data Collection"]


def test_tie_break_is_deterministic():
    # Two Medium categories with equal confidence -> canonical order wins.
    a = attention_profile(["User Control & Deletion", "Data Collection"],
                          {"User Control & Deletion": 0.7, "Data Collection": 0.7})
    b = attention_profile(["Data Collection", "User Control & Deletion"],
                          {"User Control & Deletion": 0.7, "Data Collection": 0.7})
    assert a["dominant_category"] == b["dominant_category"] == "Data Collection"
