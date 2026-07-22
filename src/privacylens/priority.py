"""Reading-priority rule (proposal section 7.3).

Turns the classifier into a triage tool by assigning each segment a 1-5 star
reading priority, while keeping THREE things separate and explicit:

  reading stars / priority_band : how important the content may be to a reader.
  review_required + review_types: whether, and WHY, the prediction needs a
                                  human check. This distinguishes CONTENT-based
                                  review (the segment is Other/Unclear) from
                                  MODEL-based review (low confidence, borderline
                                  threshold, high-attention near-miss, or
                                  abstention). One Boolean is kept for the UI,
                                  but each reason is preserved separately.
  score_components              : base / confidence_adjustment / combination_bonus
                                  / final, so any score is explainable.

A low-confidence prediction may deserve MORE human attention, not less, so low
confidence never lowers the reading priority of a High/Unknown-tier segment; it
sets a review reason instead. The rule is deterministic (no learned params).

Note on the star range: with the v1 tiers every category is High/Medium/Unknown
(base >= 3), so only 3-5 stars are reachable. This is an intended consequence
of the framework, not a bug; ``priority_band`` gives each star a readable name.
"""

from .attention import attention_profile
from .config import (
    BONUS_MARGIN,
    DEFAULT_THRESHOLD,
    FRAMEWORK_VERSION,
    HIGH_CONF,
    LOW_CONF,
    canonicalize_category,
)
from .prediction import decide_labels, validate_confidence

_BASE_STARS = {"High": 4, "Medium": 3, "Low": 2, "Unknown": 3}

STAR_LABELS = {1: "Minimal", 2: "Low", 3: "Review",
               4: "Important", 5: "Very important"}

# Deterministic ordering so overlapping reasons collapse to one primary reason
# and per-reason percentages stay unambiguous.
REVIEW_REASON_PRIORITY = [
    "abstention",
    "high_attention_near_miss",
    "low_confidence",
    "near_threshold",
    "content_unclear",
]
_REVIEW_SEVERITY = {
    "abstention": "high",
    "high_attention_near_miss": "high",
    "low_confidence": "medium",
    "content_unclear": "medium",
    "near_threshold": "low",
}


def _band(conf, low=LOW_CONF, high=HIGH_CONF):
    if conf is None:
        return None
    if conf >= high:
        return "High"
    if conf < low:
        return "Low"
    return "Medium"


def _clamp_stars(n):
    return max(1, min(5, int(n)))


def _order_reasons(types):
    return [t for t in REVIEW_REASON_PRIORITY if t in types]


def _abstention():
    return {
        "stars": 3,
        "priority_band": STAR_LABELS[3],
        "priority_subtype": "unknown_abstention",
        "base_stars": 3,
        "score_components": {"base": 3, "confidence_adjustment": 0,
                             "combination_bonus": 0, "final": 3},
        "dominant_category": None,
        "attention_tier": "Unknown",
        "valence": "Unknown",
        "confidence": None,
        "confidence_band": None,
        "all_categories": [],
        "tier_counts": {"high": 0, "medium": 0, "low": 0, "unknown": 0},
        # review: model-based (the model could not classify), highest severity
        "review_required": True,
        "review_flag": True,               # kept for backwards compatibility
        "review_types": ["abstention"],
        "primary_review_reason": "abstention",
        "review_severity": "high",
        "content_unclear_flag": False,
        "model_uncertainty_flag": True,
        "high_attention_near_miss_flag": False,
        "content_unclear_role": None,
        "review_reasons": ["No category exceeded its prediction threshold"],
        "reason": ("The model could not confidently assign this segment to a "
                   "mapped category."),
        "framework_version": FRAMEWORK_VERSION,
    }


def reading_priority(probabilities, thresholds=None, bands=None):
    """Compute the reading priority for one segment.

    Parameters
    ----------
    probabilities : dict[str, float]
        Per-category model probability (confidence). Keys are canonicalized;
        values are validated and must be finite in [0, 1] (None or malformed
        raises ValueError - a missing confidence is a bug, not a 0).
    thresholds : dict[str, float] | None
        Per-category decision thresholds; missing use DEFAULT_THRESHOLD.
    bands : tuple[float, float] | None
        Optional (low_conf, high_conf) override for the sensitivity analysis.

    Returns
    -------
    dict following the documented output schema (see _abstention() for keys).
    """
    low_conf, high_conf = bands if bands else (LOW_CONF, HIGH_CONF)

    # Validate all provided confidences up front (fail loud on None/NaN/range).
    probs = {}
    for raw, val in probabilities.items():
        c = canonicalize_category(raw)
        v = validate_confidence(val)          # raises on None/malformed
        probs[c] = max(probs.get(c, 0.0), v)

    thr = {canonicalize_category(k): v for k, v in (thresholds or {}).items()}
    decision = decide_labels(probs, thr)

    if decision["abstained"]:
        return _abstention()

    predicted = decision["predicted_categories"]
    profile = attention_profile(predicted, probs)
    dominant = profile["dominant_category"]
    tier = profile["dominant_tier"]
    conf = probs[dominant]

    # ---- score components -------------------------------------------------
    base = _BASE_STARS[tier]
    reason_bits = [f"{dominant} sets the base to {base} stars ({tier} tier)"]

    conf_adj = 0
    if conf >= high_conf:
        conf_adj = 1
        reason_bits.append(f"high confidence {conf:.2f} (+1)")
    elif conf < low_conf:
        if tier == "Low" and len(predicted) < 2:
            conf_adj = -1
            reason_bits.append(f"low confidence {conf:.2f} on a lone "
                               f"low-tier label (-1)")
        else:
            reason_bits.append(f"low confidence {conf:.2f} (no reduction for "
                               f"{tier}-tier; flagged for review instead)")
    else:
        reason_bits.append(f"moderate confidence {conf:.2f} (no change)")

    # Combination bonus: >=2 unique High/Medium labels that cleared their
    # threshold with margin (avoids two barely-above-threshold labels making a
    # 5-star result).
    eligible = []
    for c in predicted:
        c_tier = attention_profile([c])["dominant_tier"]
        if c_tier in ("High", "Medium"):
            cutoff = min(1.0, thr.get(c, DEFAULT_THRESHOLD) + BONUS_MARGIN)
            if probs[c] >= cutoff:
                eligible.append(c)
    bonus = 1 if len(eligible) >= 2 else 0
    if bonus:
        reason_bits.append(f"combination bonus for {len(eligible)} confident "
                           f"high/medium practices (+1)")

    stars = _clamp_stars(base + conf_adj + bonus)
    score_components = {"base": base, "confidence_adjustment": conf_adj,
                        "combination_bonus": bonus, "final": stars}

    # ---- review signal (first-class, two meanings kept separate) ----------
    borderline_predicted = [c for c in predicted
                            if c in decision["near_threshold_categories"]]
    high_near_miss = list(decision["high_attention_near_misses"])

    review_types, reason_text = [], {}
    if high_near_miss:
        review_types.append("high_attention_near_miss")
        reason_text["high_attention_near_miss"] = (
            "High-attention practice(s) just below threshold: "
            + ", ".join(high_near_miss))
    if conf < low_conf:
        review_types.append("low_confidence")
        reason_text["low_confidence"] = ("Model confidence is below the "
                                         "low-confidence band")
    if borderline_predicted:
        review_types.append("near_threshold")
        reason_text["near_threshold"] = (
            "Borderline predicted label(s) near threshold: "
            + ", ".join(borderline_predicted))
    if tier == "Unknown":
        review_types.append("content_unclear")
        reason_text["content_unclear"] = ("The segment's dominant prediction "
                                          "is Other/Unclear")

    review_types = _order_reasons(review_types)
    review_reasons = [reason_text[t] for t in review_types]
    review_required = bool(review_types)
    primary = review_types[0] if review_types else None
    severity = _REVIEW_SEVERITY.get(primary)

    content_unclear_flag = tier == "Unknown"
    high_near_miss_flag = bool(high_near_miss)
    model_uncertainty_flag = bool(high_near_miss) or (conf < low_conf) \
        or bool(borderline_predicted)

    # Record the ROLE of Other/Unclear (for the flag-composition diagnostics),
    # without changing the approved "dominant Unknown -> review" behavior.
    if "Other/Unclear" in predicted:
        if len(predicted) == 1:
            content_role = "only"
        elif dominant == "Other/Unclear":
            content_role = "dominant"
        else:
            content_role = "secondary"
    else:
        content_role = None

    subtype = ("unknown_identified" if tier == "Unknown"
               else f"identified_{tier.lower()}")

    reason = "; ".join(reason_bits) + "."
    if review_required:
        reason += " Flagged for review: " + review_reasons[0].lower() + "."

    return {
        "stars": stars,
        "priority_band": STAR_LABELS[stars],
        "priority_subtype": subtype,
        "base_stars": base,
        "score_components": score_components,
        "dominant_category": dominant,
        "attention_tier": tier,
        "valence": profile["dominant_valence"],
        "confidence": conf,
        "confidence_band": _band(conf, low_conf, high_conf),
        "all_categories": profile["all_categories"],
        "tier_counts": profile["tier_counts"],
        "review_required": review_required,
        "review_flag": review_required,        # backwards-compatible alias
        "review_types": review_types,
        "primary_review_reason": primary,
        "review_severity": severity,
        "content_unclear_flag": content_unclear_flag,
        "model_uncertainty_flag": model_uncertainty_flag,
        "high_attention_near_miss_flag": high_near_miss_flag,
        "content_unclear_role": content_role,
        "review_reasons": review_reasons,
        "reason": reason,
        "framework_version": FRAMEWORK_VERSION,
    }
