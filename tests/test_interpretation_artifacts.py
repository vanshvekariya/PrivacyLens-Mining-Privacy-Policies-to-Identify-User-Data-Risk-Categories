"""Tests for the artifact-generation helper functions (no model fitting).

These exercise the pure numeric/aggregation functions on small synthetic
out-of-fold predictions, so they are fast and deterministic.
"""

import numpy as np
import pytest

import make_interpretation_artifacts as mia
from privacylens.config import CATEGORIES, DEFAULT_THRESHOLD, MIN_THRESHOLD_SUPPORT

RNG = np.random.default_rng(0)


def _synthetic(n=200):
    """Build (P, Y) where each category's probability correlates with label."""
    Y = (RNG.random((n, len(CATEGORIES))) < 0.3).astype(int)
    # Make Data Retention (index 1) rare to trigger the min-support fallback.
    Y[:, 1] = 0
    Y[:5, 1] = 1
    noise = RNG.normal(0, 0.15, (n, len(CATEGORIES)))
    P = np.clip(0.25 + 0.5 * Y + noise, 0, 1)
    return P, Y


def test_select_thresholds_min_support_fallback():
    P, Y = _synthetic()
    thresholds, details = mia.select_thresholds(P, Y)
    assert set(thresholds) == set(CATEGORIES)
    # Data Retention has < MIN_THRESHOLD_SUPPORT positives -> default kept.
    assert Y[:, 1].sum() < MIN_THRESHOLD_SUPPORT
    assert thresholds["Data Retention"] == DEFAULT_THRESHOLD
    row = details[details.category == "Data Retention"].iloc[0]
    assert row["rule"] == "min_support_fallback"


def test_select_thresholds_deterministic():
    P, Y = _synthetic()
    t1, _ = mia.select_thresholds(P, Y)
    t2, _ = mia.select_thresholds(P, Y)
    assert t1 == t2


def test_calibration_summary_shape_and_range():
    P, Y = _synthetic()
    cal = mia.calibration_summary(P, Y)
    assert set(cal.category) == set(CATEGORIES)
    assert (cal.brier >= 0).all() and (cal.brier <= 1).all()


def test_flag_and_abstention_stats():
    P, Y = _synthetic()
    thresholds, _ = mia.select_thresholds(P, Y)
    priorities = mia.compute_priorities(P, thresholds)
    fs = mia.flag_stats(priorities)
    assert set(fs.scope) >= {"overall", "driver", "review_type",
                             "primary_reason", "other_unclear_role",
                             "review_severity"}
    assert (fs["rate"] >= 0).all() and (fs["rate"] <= 1).all()
    overall = fs[fs.scope == "overall"].iloc[0]
    assert overall.key == "review_required"

    # content-only + model-only + both == review_required (partition check)
    drv = fs[fs.scope == "driver"].set_index("key")["count"]
    assert drv["content_only"] + drv["model_only"] + drv["both_content_and_model"] \
        == overall["count"]

    # deterministic primary_review_reason is a partition of ALL segments
    pr = fs[fs.scope == "primary_reason"]
    assert pr["count"].sum() == len(priorities)
    flagged_primary = pr[pr.key != "not_flagged"]["count"].sum()
    assert flagged_primary == overall["count"]

    label_sets = [[CATEGORIES[j] for j in range(len(CATEGORIES)) if Y[i, j]]
                  for i in range(len(Y))]
    groups = np.array([f"p{i % 5}" for i in range(len(Y))])
    fold_of_policy = {f"p{k}": k for k in range(5)}
    ab = mia.abstention_stats(priorities, label_sets, groups, fold_of_policy)
    assert (ab.abstention_rate >= 0).all() and (ab.abstention_rate <= 1).all()


def test_review_diagnostics_and_distribution():
    P, Y = _synthetic()
    thresholds, _ = mia.select_thresholds(P, Y)
    priorities = mia.compute_priorities(P, thresholds)
    predicted_sets = [p["all_categories"] for p in priorities]
    gold_sets = [[CATEGORIES[j] for j in range(len(CATEGORIES)) if Y[i, j]]
                 for i in range(len(Y))]
    diag = mia.review_diagnostics(priorities, predicted_sets, gold_sets)
    assert set(diag.flag) >= {"review_required", "model_uncertainty",
                              "content_unclear"}
    rr = diag[diag.flag == "review_required"].iloc[0]
    # rates are valid probabilities where defined
    for col in ["flagged_error_rate", "not_flagged_error_rate",
                "error_recall", "flag_precision"]:
        v = rr[col]
        assert np.isnan(v) or (0.0 <= v <= 1.0)

    dist = mia.priority_distribution(priorities)
    assert dist["count"].sum() == len(priorities)
    assert {"stars", "priority_band", "priority_subtype"} <= set(dist.columns)


def test_priority_sensitivity_rows():
    P, Y = _synthetic()
    thresholds, _ = mia.select_thresholds(P, Y)
    sens = mia.priority_sensitivity(P, thresholds)
    assert len(sens) == len(mia.SENSITIVITY_BANDS)
    for _, r in sens.iterrows():
        assert r["1_star"] + r["2_star"] + r["3_star"] + r["4_star"] \
            + r["5_star"] == len(P)
