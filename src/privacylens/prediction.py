"""Centralized label decision, threshold loading, and fold loading.

This is the single place that turns per-category probabilities into a set of
predicted labels. The interpretation modules (attention, priority) and the
artifact script all call ``decide_labels`` so they never disagree about which
labels "passed", what counts as borderline, and when a segment abstains.

Key subtlety (review): a high-attention label just BELOW its threshold is a
``high_attention_near_miss``. It must still trigger human review even when
another label passed and the segment therefore does not abstain.
"""

import json
import math

from .config import (
    CATEGORIES,
    DEFAULT_THRESHOLD,
    FRAMEWORK_VERSION,
    FOLDS_CSV,
    MODEL_NAME,
    N_FOLDS,
    NEAR_MARGIN,
    THRESHOLDS_JSON,
    canonicalize_category,
    tier_of,
)


def validate_confidence(value, *, allow_none=False):
    """Fail loudly on malformed confidence; return the float otherwise.

    Malformed model output (NaN, inf, out of [0, 1], non-numeric) is a bug we
    want surfaced, not silently clamped. Star clamping happens elsewhere and
    only for well-formed inputs.
    """
    if value is None:
        if allow_none:
            return None
        raise ValueError("confidence is None")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"confidence must be a real number, got {value!r}")
    v = float(value)
    if not math.isfinite(v) or not 0.0 <= v <= 1.0:
        raise ValueError(f"confidence must be finite in [0, 1], got {value!r}")
    return v


def _canon_map(mapping):
    """Canonicalize the keys of a {category: value} mapping (max on clash)."""
    out = {}
    for k, v in mapping.items():
        c = canonicalize_category(k)
        if c in out:
            out[c] = max(out[c], v)
        else:
            out[c] = v
    return out


def decide_labels(probabilities, thresholds=None, near_margin=NEAR_MARGIN):
    """Decide predicted / borderline / near-miss labels from probabilities.

    Parameters
    ----------
    probabilities : dict[str, float]
        Per-category model probability. Keys are canonicalized; values are
        validated (raise on malformed).
    thresholds : dict[str, float] | None
        Per-category decision threshold. Missing categories use
        DEFAULT_THRESHOLD. None uses DEFAULT_THRESHOLD for all.
    near_margin : float
        Half-width of the "borderline" band around a threshold.

    Returns
    -------
    dict with:
      predicted_categories          : list[str]  (p >= threshold)
      near_threshold_categories     : list[str]  (|p - threshold| <= margin)
      high_attention_near_misses    : list[str]  (High tier, thr-margin <= p < thr)
      abstained                     : bool       (no category predicted)
    All lists are in canonical CATEGORIES order for determinism.
    """
    probs = _canon_map(probabilities)
    for c, p in probs.items():
        validate_confidence(p)
    thr = _canon_map(thresholds) if thresholds else {}

    predicted, near, near_miss = [], [], []
    for c in CATEGORIES:
        if c not in probs:
            continue
        p = probs[c]
        t = thr.get(c, DEFAULT_THRESHOLD)
        if p >= t:
            predicted.append(c)
        if abs(p - t) <= near_margin:
            near.append(c)
        if tier_of(c) == "High" and (t - near_margin) <= p < t:
            near_miss.append(c)

    return {
        "predicted_categories": predicted,
        "near_threshold_categories": near,
        "high_attention_near_misses": near_miss,
        "abstained": len(predicted) == 0,
    }


def load_thresholds(path=THRESHOLDS_JSON):
    """Load tuned thresholds for INFERENCE, validating provenance.

    Rejects a stale file whose framework version, model, or category set does
    not match the current configuration. Never called during artifact
    generation (which always computes thresholds fresh).
    """
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)

    if blob.get("framework_version") != FRAMEWORK_VERSION:
        raise ValueError(
            f"threshold file framework_version {blob.get('framework_version')!r}"
            f" != current {FRAMEWORK_VERSION!r}")
    if blob.get("model_name") != MODEL_NAME:
        raise ValueError(
            f"threshold file model_name {blob.get('model_name')!r}"
            f" != current {MODEL_NAME!r}")

    thresholds = _canon_map(blob.get("thresholds", {}))
    if set(thresholds) != set(CATEGORIES):
        missing = set(CATEGORIES) - set(thresholds)
        extra = set(thresholds) - set(CATEGORIES)
        raise ValueError(
            f"threshold file category set mismatch (missing={missing}, "
            f"extra={extra})")
    return thresholds


def load_folds(path=FOLDS_CSV):
    """Return {policy_id: fold_id} from the canonical fold-assignment file."""
    import pandas as pd
    df = pd.read_csv(path)
    return dict(zip(df["policy_id"].astype(str), df["fold_id"].astype(int)))


def build_fold_assignment(groups, n_splits=N_FOLDS):
    """Deterministically map each policy to its validation fold via GroupKFold.

    Splitting depends only on ``groups`` (GroupKFold ignores X and y), so this
    reproduces exactly the folds used in run_baselines.py.
    """
    import numpy as np
    from sklearn.model_selection import GroupKFold

    groups = np.asarray([str(g) for g in groups])
    X = np.zeros((len(groups), 1))
    gkf = GroupKFold(n_splits=n_splits)
    fold_of_policy = {}
    for fold, (_, val_idx) in enumerate(gkf.split(X, X, groups)):
        for i in val_idx:
            fold_of_policy[groups[i]] = fold
    return fold_of_policy


def load_or_build_folds(groups, path=FOLDS_CSV, n_splits=N_FOLDS):
    """Load the canonical fold file if present, else build and persist it.

    Single source of CV folds shared by run_baselines.py and the artifact
    script, so every model, figure, and threshold uses identical folds.
    """
    if path.exists():
        return load_folds(path)
    import pandas as pd
    fold_of_policy = build_fold_assignment(groups, n_splits)
    path.parent.mkdir(parents=True, exist_ok=True)
    (pd.DataFrame(sorted(fold_of_policy.items()),
                  columns=["policy_id", "fold_id"])
       .to_csv(path, index=False))
    return fold_of_policy


def folds_to_indices(groups, fold_of_policy, n_splits=N_FOLDS):
    """Yield (train_idx, val_idx) per fold from a {policy: fold} mapping."""
    import numpy as np
    groups = np.asarray([str(g) for g in groups])
    fold_ids = np.array([fold_of_policy[g] for g in groups])
    for fold in range(n_splits):
        val_idx = np.where(fold_ids == fold)[0]
        tr_idx = np.where(fold_ids != fold)[0]
        yield tr_idx, val_idx
