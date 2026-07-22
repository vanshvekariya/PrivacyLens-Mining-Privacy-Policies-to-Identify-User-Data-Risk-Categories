"""Tests for the reading-priority rule."""

import pytest

from privacylens.priority import reading_priority


def test_high_confidence_sharing_tops_out():
    r = reading_priority({"Third-Party Sharing": 0.95, "Data Collection": 0.9})
    assert r["stars"] == 5
    assert r["dominant_category"] == "Third-Party Sharing"
    assert r["attention_tier"] == "High"


def test_abstention_state():
    r = reading_priority({"Data Collection": 0.2, "Data Security": 0.1})
    assert r["stars"] == 3
    assert r["dominant_category"] is None
    assert r["attention_tier"] == "Unknown"
    assert r["confidence"] is None
    assert r["review_flag"] is True
    assert r["all_categories"] == []


def test_other_unclear_alone_is_flagged_unknown():
    r = reading_priority({"Other/Unclear": 0.9})
    assert r["attention_tier"] == "Unknown"
    assert r["review_flag"] is True


def test_confidence_boundary_070_adds_star():
    # threshold below 0.70 so the label is predicted; 0.70 hits the high band.
    r = reading_priority({"Data Security": 0.70}, {"Data Security": 0.30})
    assert "high confidence" in r["reason"]
    assert r["stars"] == 4  # Medium base 3 + high-confidence 1


def test_confidence_boundary_040_no_low_behavior():
    # 0.40 is NOT < LOW_CONF(0.40); predicted via low threshold.
    r = reading_priority({"Data Security": 0.40}, {"Data Security": 0.30})
    assert r["stars"] == 3
    assert r["confidence_band"] == "Medium"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.5, None])
def test_malformed_confidence_raises(bad):
    with pytest.raises(ValueError):
        reading_priority({"Data Collection": bad})


def test_high_plus_mixed_not_cancelled():
    r = reading_priority({"Third-Party Sharing": 0.8,
                          "User Control & Deletion": 0.9})
    assert r["attention_tier"] == "High"


def test_combination_bonus_requires_two_confident_high_medium():
    # Only one confident label -> no bonus.
    one = reading_priority({"Data Collection": 0.9})
    assert "combination bonus" not in one["reason"]
    # Two confident high/medium labels clearing threshold+margin -> bonus.
    two = reading_priority({"Data Collection": 0.9, "Third-Party Sharing": 0.9})
    assert "combination bonus" in two["reason"]


def test_combination_bonus_not_for_barely_above_threshold():
    # Both just above 0.5 (within the 0.05 bonus margin) -> no bonus.
    r = reading_priority({"Data Collection": 0.52, "Third-Party Sharing": 0.53},
                         {"Data Collection": 0.5, "Third-Party Sharing": 0.5})
    assert "combination bonus" not in r["reason"]


def test_high_attention_near_miss_flags_even_when_not_abstaining():
    r = reading_priority(
        {"Data Collection": 0.85, "Third-Party Sharing": 0.48},
        {"Data Collection": 0.5, "Third-Party Sharing": 0.5})
    assert r["dominant_category"] == "Data Collection"  # did not abstain
    assert r["review_flag"] is True
    assert any("just below threshold" in x for x in r["review_reasons"])


def test_duplicate_and_variant_keys_are_merged():
    r = reading_priority({"data collection": 0.9, "Data Collection": 0.4})
    assert r["dominant_category"] == "Data Collection"
    assert r["confidence"] == pytest.approx(0.9)


def test_moderate_confidence_does_not_autoflag():
    # Lone Medium label at moderate confidence, comfortably away from its
    # threshold -> should NOT be flagged (avoids over-flagging).
    r = reading_priority({"Data Collection": 0.62}, {"Data Collection": 0.5})
    assert r["review_flag"] is False
    assert r["review_required"] is False
    assert r["review_types"] == []
    assert r["primary_review_reason"] is None
    assert r["review_severity"] is None


def test_score_components_explain_final_stars():
    r = reading_priority({"Third-Party Sharing": 0.95})
    sc = r["score_components"]
    assert sc["base"] == 4 and sc["confidence_adjustment"] == 1
    assert sc["combination_bonus"] == 0 and sc["final"] == 5
    assert r["base_stars"] == 4
    assert r["stars"] == sc["final"]
    # clamped components still reconcile to final via clamp of the sum.
    assert r["stars"] == max(1, min(5, sc["base"] + sc["confidence_adjustment"]
                                    + sc["combination_bonus"]))


def test_priority_band_labels():
    assert reading_priority({"Third-Party Sharing": 0.95})["priority_band"] \
        == "Very important"
    assert reading_priority({"Data Security": 0.70},
                            {"Data Security": 0.30})["priority_band"] \
        == "Important"
    assert reading_priority({"Data Security": 0.55},
                            {"Data Security": 0.30})["priority_band"] == "Review"


def test_abstention_vs_identified_medium_subtypes_differ():
    ab = reading_priority({"Data Collection": 0.2, "Data Security": 0.1})
    med = reading_priority({"Data Security": 0.55}, {"Data Security": 0.30})
    assert ab["stars"] == med["stars"] == 3
    assert ab["priority_subtype"] == "unknown_abstention"
    assert med["priority_subtype"] == "identified_medium"
    # abstention is model-driven review, high severity.
    assert ab["review_types"] == ["abstention"]
    assert ab["review_severity"] == "high"
    assert ab["model_uncertainty_flag"] is True
    assert ab["content_unclear_flag"] is False


def test_review_types_ordered_and_primary_is_first():
    # Dominant Other/Unclear at low confidence: content_unclear + low_confidence.
    r = reading_priority({"Other/Unclear": 0.35}, {"Other/Unclear": 0.30})
    assert r["content_unclear_flag"] is True
    assert set(r["review_types"]) >= {"low_confidence", "content_unclear"}
    # low_confidence precedes content_unclear in REVIEW_REASON_PRIORITY.
    assert r["review_types"].index("low_confidence") \
        < r["review_types"].index("content_unclear")
    assert r["primary_review_reason"] == r["review_types"][0]


def test_content_unclear_role_only_dominant_secondary():
    only = reading_priority({"Other/Unclear": 0.9})
    assert only["content_unclear_role"] == "only"
    # Other/Unclear dominant but with a secondary predicted practice.
    dom = reading_priority({"Other/Unclear": 0.9, "Data Collection": 0.55},
                           {"Other/Unclear": 0.5, "Data Collection": 0.5})
    assert dom["content_unclear_role"] == "dominant"
    # Meaningful practice dominates; Other/Unclear is secondary.
    sec = reading_priority({"Third-Party Sharing": 0.9, "Other/Unclear": 0.55},
                           {"Third-Party Sharing": 0.5, "Other/Unclear": 0.5})
    assert sec["dominant_category"] == "Third-Party Sharing"
    assert sec["content_unclear_role"] == "secondary"
    # secondary Other/Unclear does NOT set the content_unclear review flag.
    assert sec["content_unclear_flag"] is False
