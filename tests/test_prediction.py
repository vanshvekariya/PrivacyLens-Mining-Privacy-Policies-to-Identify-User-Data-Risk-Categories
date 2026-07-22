"""Tests for the centralized label-decision logic."""

import math

import pytest

from privacylens.prediction import decide_labels, validate_confidence


def test_validate_confidence_accepts_valid():
    assert validate_confidence(0.0) == 0.0
    assert validate_confidence(1.0) == 1.0
    assert validate_confidence(0.5) == 0.5


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"),
                                 -0.01, 1.01, "0.5", True])
def test_validate_confidence_rejects_malformed(bad):
    with pytest.raises(ValueError):
        validate_confidence(bad)


def test_validate_confidence_none_allowed():
    assert validate_confidence(None, allow_none=True) is None


def test_decide_labels_basic_threshold():
    out = decide_labels({"Data Collection": 0.8, "Data Security": 0.2})
    assert out["predicted_categories"] == ["Data Collection"]
    assert not out["abstained"]


def test_decide_labels_abstention():
    out = decide_labels({"Data Collection": 0.2, "Data Security": 0.1})
    assert out["predicted_categories"] == []
    assert out["abstained"] is True


def test_high_attention_near_miss():
    # Third-Party Sharing (High) just below its threshold, Collection passes.
    out = decide_labels(
        {"Data Collection": 0.80, "Third-Party Sharing": 0.48},
        {"Data Collection": 0.50, "Third-Party Sharing": 0.50})
    assert out["predicted_categories"] == ["Data Collection"]
    assert out["high_attention_near_misses"] == ["Third-Party Sharing"]
    assert not out["abstained"]


def test_near_threshold_both_sides():
    out = decide_labels({"Data Collection": 0.52}, {"Data Collection": 0.50})
    assert "Data Collection" in out["near_threshold_categories"]


def test_decide_labels_canonicalizes_and_dedupes_max():
    # naming variant + duplicate -> single canonical key with the max value
    out = decide_labels(
        {"third party sharing": 0.9, "Third-Party Sharing": 0.4},
        {"Third-Party Sharing": 0.5})
    assert out["predicted_categories"] == ["Third-Party Sharing"]


def test_decide_labels_unknown_category_raises():
    with pytest.raises(ValueError):
        decide_labels({"Nonexistent Practice": 0.9})
