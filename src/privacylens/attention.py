"""Attention-tier framework and multi-label attention profile (proposal s6).

``attention_for`` looks up one category's tier/valence/weight. ``attention_profile``
combines the predicted categories of a segment into a single headline plus a
full profile, with three deliberate properties:

  - protective/Mixed labels NEVER cancel a higher-attention label (the
    headline is the highest-attention practice present);
  - duplicate / naming-variant categories are canonicalized and deduped,
    keeping the maximum confidence;
  - tie-breaking is deterministic: highest weight, then highest confidence,
    then fixed canonical category order.
"""

from .config import (
    ATTENTION_COLORS,
    ATTENTION_FRAMEWORK,
    CATEGORIES,
    TIER_WEIGHT,
    canonicalize_category,
)

_CANON_INDEX = {c: i for i, c in enumerate(CATEGORIES)}


def attention_for(category: str) -> dict:
    """Return the framework entry for a category, plus weight and colour."""
    c = canonicalize_category(category)
    info = ATTENTION_FRAMEWORK[c]
    tier = info["attention_tier"]
    return {
        "category": c,
        "attention_tier": tier,
        "default_valence": info["default_valence"],
        "weight": TIER_WEIGHT[tier],
        "review_flag": info["review_flag"],
        "color": ATTENTION_COLORS[tier],
        "rationale": info["rationale"],
        "gdpr": list(info["gdpr"]),
        "sources": list(info["sources"]),
    }


def attention_profile(categories, confidences=None) -> dict:
    """Combine predicted categories into a headline + full profile.

    Parameters
    ----------
    categories : iterable[str]
        Predicted categories (may contain duplicates / naming variants).
    confidences : dict[str, float] | None
        Per-category confidence, used only for deterministic tie-breaking.
        Missing entries are treated as 0.0 here (priority.py enforces
        presence for predicted labels).
    """
    confidences = confidences or {}

    # Canonicalize + dedupe, keeping the max confidence per category.
    conf = {}
    for raw in categories:
        c = canonicalize_category(raw)
        cval = float(confidences.get(raw, confidences.get(c, 0.0)))
        conf[c] = max(conf.get(c, 0.0), cval)

    unique = sorted(conf, key=lambda c: _CANON_INDEX[c])

    tier_counts = {"High": 0, "Medium": 0, "Low": 0, "Unknown": 0}
    for c in unique:
        tier_counts[attention_for(c)["attention_tier"]] += 1

    if not unique:
        return {
            "dominant_category": None,
            "dominant_tier": "Unknown",
            "dominant_valence": "Unknown",
            "all_categories": [],
            "tier_counts": {k.lower(): v for k, v in tier_counts.items()},
        }

    # Deterministic dominance: weight desc, confidence desc, canonical order.
    dominant = max(
        unique,
        key=lambda c: (TIER_WEIGHT[attention_for(c)["attention_tier"]],
                       conf[c],
                       -_CANON_INDEX[c]),
    )
    dom = attention_for(dominant)
    return {
        "dominant_category": dominant,
        "dominant_tier": dom["attention_tier"],
        "dominant_valence": dom["default_valence"],
        "all_categories": unique,
        "tier_counts": {k.lower(): v for k, v in tier_counts.items()},
    }


def framework_table():
    """Report-ready rows describing the framework (CATEGORIES order)."""
    rows = []
    for c in CATEGORIES:
        info = ATTENTION_FRAMEWORK[c]
        tier = info["attention_tier"]
        rows.append({
            "category": c,
            "attention_tier": tier,
            "default_valence": info["default_valence"],
            "weight": TIER_WEIGHT[tier],
            "review_flag": info["review_flag"],
            "gdpr": "; ".join(info["gdpr"]),
            "rationale": info["rationale"],
            "sources": "; ".join(info["sources"]),
        })
    return rows
