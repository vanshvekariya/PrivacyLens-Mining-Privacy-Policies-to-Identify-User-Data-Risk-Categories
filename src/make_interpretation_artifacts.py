"""Generate the interpretation-layer artifacts for the PrivacyLens report.

Everything here is derived from POLICY-GROUPED OUT-OF-FOLD predictions on the
TRAIN split only (never in-sample; the held-out test split is untouched and
reserved for the final report). Logistic Regression (the exact run_baselines
configuration) provides the per-category probabilities.

Outputs (all under results/):
  cv_fold_assignments.csv        shared folds (written by run_baselines too)
  fold_support.csv               positive support per category per fold
  category_thresholds.json       FRESH per-category thresholds + provenance
  calibration_summary.csv        per-category Brier + reliability summary
  priority_sensitivity.csv        star distribution under nearby conf bands
  flag_stats.csv                 review composition: content- vs model-driven,
                                 overlap, per-type + deterministic primary reason
  review_diagnostics.csv         does the flag concentrate errors (rate/recall/
                                 precision, high-confidence errors missed)
  priority_distribution.csv       stars x band x subtype counts (abstention split)
  abstention_stats.csv           abstention count/rate, by gold cat, by fold
  attention_framework.csv        the framework table (report section 6)
  explanation_templates.csv      the explanation templates
  priority_examples.csv          worked OOF examples spanning the star range
  figures/fig_priority_distribution.png
  interpretation_metadata.json   run-level provenance

These threshold artifacts are DEVELOPMENT-SET / exploratory: thresholds are
tuned on the same OOF predictions they are then applied to, so they are not an
unbiased estimate of deployed performance. The fold-mean CV table in
run_baselines.py remains the primary model comparison, and a single thresholded
test-set evaluation is deferred to the final report.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from privacylens import templates
from privacylens.attention import framework_table
from privacylens.config import (
    CATEGORIES,
    DEFAULT_THRESHOLD,
    FRAMEWORK_VERSION,
    HIGH_CONF,
    INTERP_METADATA_JSON,
    LOW_CONF,
    LR_PARAMS,
    MIN_THRESHOLD_SUPPORT,
    MODEL_NAME,
    N_FOLDS,
    RESULTS,
    SEED,
    SEGMENTS_CSV,
    SPLIT_ID,
    TFIDF_PARAMS,
    THRESHOLDS_JSON,
    model_version,
    tier_of,
)
from privacylens.prediction import folds_to_indices, load_or_build_folds
from privacylens.priority import reading_priority

FIGS = RESULTS / "figures"
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)  # excludes 0 and 1
SENSITIVITY_BANDS = [(0.35, 0.65), (0.40, 0.70), (0.45, 0.75)]

# Figure palette (matches src/make_figures.py).
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, AQUA = "#2a78d6", "#1baf7a"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8, "axes.titlesize": 11, "axes.titlecolor": INK,
    "font.size": 9,
})


# --------------------------------------------------------------------------
# Out-of-fold probabilities
# --------------------------------------------------------------------------
def compute_oof(texts, Y, groups, fold_of_policy):
    """Return pooled OOF probability matrix (TF-IDF fit inside each fold)."""
    P = np.zeros_like(Y, dtype=float)
    for tr_idx, va_idx in folds_to_indices(groups, fold_of_policy):
        pipe = make_pipeline(
            TfidfVectorizer(**TFIDF_PARAMS),
            OneVsRestClassifier(LogisticRegression(**LR_PARAMS)))
        pipe.fit([texts[i] for i in tr_idx], Y[tr_idx])
        proba = np.asarray(pipe.predict_proba([texts[i] for i in va_idx]))
        P[va_idx] = proba
    return P


def fold_support_table(Y, groups, fold_of_policy):
    """Positive support per category per fold (in the fold-TRAINING set)."""
    rows, warnings = [], []
    for fold, (tr_idx, va_idx) in enumerate(
            folds_to_indices(groups, fold_of_policy)):
        row = {"fold": fold, "train_segments": len(tr_idx),
               "val_segments": len(va_idx)}
        for j, c in enumerate(CATEGORIES):
            tr_pos = int(Y[tr_idx, j].sum())
            va_pos = int(Y[va_idx, j].sum())
            row[c] = tr_pos
            if tr_pos == 0:
                warnings.append(f"fold {fold}: 0 positive train examples for {c}")
            elif tr_pos < 10:
                warnings.append(f"fold {fold}: only {tr_pos} train examples for {c}")
            if va_pos > 0 and tr_pos == 0:
                warnings.append(
                    f"fold {fold}: {c} present in validation but absent in training")
        rows.append(row)
    return pd.DataFrame(rows), warnings


# --------------------------------------------------------------------------
# Threshold selection (fresh, deterministic)
# --------------------------------------------------------------------------
def select_thresholds(P, Y):
    """Per-category threshold by maximizing OOF F1, with deterministic rules.

    - support < MIN_THRESHOLD_SUPPORT  -> keep DEFAULT_THRESHOLD (0.5)
    - else maximize F1 over a fixed grid (0.05..0.95)
    - ties: High-tier -> lowest threshold (max recall); others -> closest to
      0.5; final tie -> lower threshold.
    """
    thresholds, details = {}, []
    for j, c in enumerate(CATEGORIES):
        y = Y[:, j]
        support = int(y.sum())
        if support < MIN_THRESHOLD_SUPPORT:
            thresholds[c] = DEFAULT_THRESHOLD
            details.append({"category": c, "support": support,
                            "threshold": DEFAULT_THRESHOLD,
                            "rule": "min_support_fallback", "f1": np.nan})
            continue
        f1s = np.array([f1_score(y, (P[:, j] >= t).astype(int),
                                 zero_division=0) for t in THRESHOLD_GRID])
        best = f1s.max()
        cand = THRESHOLD_GRID[f1s >= best - 1e-12]
        if tier_of(c) == "High":
            t = float(min(cand))
        else:
            t = float(min(cand, key=lambda x: (abs(x - 0.5), x)))
        thresholds[c] = t
        details.append({"category": c, "support": support, "threshold": t,
                        "rule": "maximize_f1", "f1": float(best)})
    return thresholds, pd.DataFrame(details)


def write_thresholds(thresholds):
    blob = {
        "framework_version": FRAMEWORK_VERSION,
        "model_name": MODEL_NAME,
        "model_version": model_version(),
        "split_id": SPLIT_ID,
        "prediction_source": "train_oof",
        "threshold_method": "maximize_f1",
        "thresholds": thresholds,
    }
    THRESHOLDS_JSON.write_text(json.dumps(blob, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def calibration_summary(P, Y):
    rows = []
    for j, c in enumerate(CATEGORIES):
        y = Y[:, j]
        support = int(y.sum())
        brier = float(brier_score_loss(y, P[:, j])) if 0 < support < len(y) \
            else float(np.mean((P[:, j] - y) ** 2))
        rows.append({
            "category": c, "support": support,
            "base_rate": float(y.mean()),
            "mean_pred_prob": float(P[:, j].mean()),
            "mean_pred_prob_pos": float(P[y == 1, j].mean()) if support else np.nan,
            "mean_pred_prob_neg": float(P[y == 0, j].mean()) if support < len(y) else np.nan,
            "brier": brier,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Priorities, flags, abstention
# --------------------------------------------------------------------------
def probs_dict(row):
    return {c: float(row[j]) for j, c in enumerate(CATEGORIES)}


def compute_priorities(P, thresholds, bands=None):
    return [reading_priority(probs_dict(P[i]), thresholds, bands=bands)
            for i in range(len(P))]


REVIEW_TYPES = ["abstention", "high_attention_near_miss", "low_confidence",
                "near_threshold", "content_unclear"]


def flag_stats(priorities):
    """Review-flag composition, separating the TWO meanings of the flag.

    Content-based review (predicted Other/Unclear) and model-based review
    (low confidence / borderline threshold / high-attention near-miss /
    abstention) are reported separately, with their overlap, plus a
    deterministic non-overlapping primary_review_reason breakdown so the
    per-reason percentages add up cleanly.
    """
    n = len(priorities)

    def rate(cnt):
        return round(cnt / n, 4)

    rows = []

    # --- overall + the two drivers and their overlap --------------------
    content = sum(p["content_unclear_flag"] for p in priorities)
    model = sum(p["model_uncertainty_flag"] for p in priorities)
    both = sum(p["content_unclear_flag"] and p["model_uncertainty_flag"]
               for p in priorities)
    review = sum(p["review_required"] for p in priorities)
    rows.append({"scope": "overall", "key": "review_required",
                 "count": review, "rate": rate(review)})
    rows.append({"scope": "driver", "key": "content_unclear",
                 "count": content, "rate": rate(content)})
    rows.append({"scope": "driver", "key": "model_uncertainty",
                 "count": model, "rate": rate(model)})
    rows.append({"scope": "driver", "key": "both_content_and_model",
                 "count": both, "rate": rate(both)})
    rows.append({"scope": "driver", "key": "content_only",
                 "count": content - both, "rate": rate(content - both)})
    rows.append({"scope": "driver", "key": "model_only",
                 "count": model - both, "rate": rate(model - both)})

    # --- per review-type presence rate (overlapping, for transparency) --
    for t in REVIEW_TYPES:
        cnt = sum(t in p["review_types"] for p in priorities)
        rows.append({"scope": "review_type", "key": t,
                     "count": cnt, "rate": rate(cnt)})

    # --- deterministic primary_review_reason (non-overlapping) ----------
    for t in REVIEW_TYPES + [None]:
        cnt = sum(p["primary_review_reason"] == t for p in priorities)
        rows.append({"scope": "primary_reason",
                     "key": t if t is not None else "not_flagged",
                     "count": cnt, "rate": rate(cnt)})

    # --- Other/Unclear role breakdown (only / dominant / secondary) -----
    for role in ["only", "dominant", "secondary"]:
        cnt = sum(p["content_unclear_role"] == role for p in priorities)
        rows.append({"scope": "other_unclear_role", "key": role,
                     "count": cnt, "rate": rate(cnt)})

    # --- review severity breakdown --------------------------------------
    for sev in ["high", "medium", "low"]:
        cnt = sum(p["review_severity"] == sev for p in priorities)
        rows.append({"scope": "review_severity", "key": sev,
                     "count": cnt, "rate": rate(cnt)})

    return pd.DataFrame(rows)


def review_diagnostics(priorities, predicted_sets, gold_sets):
    """Is the review flag actually concentrating errors?

    A segment is an ERROR when its thresholded predicted label set differs
    from the gold set (exact multi-label match). For each flag we report the
    error rate inside vs. outside the flagged group, how many of all errors
    the flag captures (recall), how many flagged segments are truly wrong
    (precision), and how many high-confidence errors the flag missed.
    """
    n = len(priorities)
    errors = np.array([set(predicted_sets[i]) != set(gold_sets[i])
                       for i in range(n)])
    total_err = int(errors.sum())
    rows = []
    for scope, key in [("review_required", "review_required"),
                       ("model_uncertainty", "model_uncertainty_flag"),
                       ("content_unclear", "content_unclear_flag")]:
        flag = np.array([bool(p[key]) for p in priorities])
        n_f, n_nf = int(flag.sum()), int((~flag).sum())
        err_f = int(errors[flag].sum()) if n_f else 0
        err_nf = int(errors[~flag].sum()) if n_nf else 0
        hi_conf_missed = sum(
            errors[i] and not flag[i]
            and priorities[i]["confidence"] is not None
            and priorities[i]["confidence"] >= HIGH_CONF
            for i in range(n))
        rows.append({
            "flag": scope,
            "n_flagged": n_f,
            "n_not_flagged": n_nf,
            "flagged_error_rate": round(err_f / n_f, 4) if n_f else np.nan,
            "not_flagged_error_rate": round(err_nf / n_nf, 4) if n_nf else np.nan,
            "error_recall": round(err_f / total_err, 4) if total_err else np.nan,
            "flag_precision": round(err_f / n_f, 4) if n_f else np.nan,
            "high_conf_errors_unflagged": int(hi_conf_missed),
        })
    # context row
    rows.append({"flag": "(corpus)", "n_flagged": total_err,
                 "n_not_flagged": n - total_err,
                 "flagged_error_rate": round(total_err / n, 4),
                 "not_flagged_error_rate": np.nan, "error_recall": np.nan,
                 "flag_precision": np.nan, "high_conf_errors_unflagged": np.nan})
    return pd.DataFrame(rows)


def priority_distribution(priorities):
    """Star distribution with abstention/subtype kept distinct from ordinary
    3-star results (identified medium-attention vs unknown / model abstained)."""
    rows = []
    combos = {}
    for p in priorities:
        combos.setdefault((p["stars"], p["priority_band"],
                           p["priority_subtype"]), 0)
        combos[(p["stars"], p["priority_band"], p["priority_subtype"])] += 1
    n = len(priorities)
    for (stars, band, subtype), cnt in sorted(combos.items()):
        rows.append({"stars": stars, "priority_band": band,
                     "priority_subtype": subtype, "count": cnt,
                     "rate": round(cnt / n, 4)})
    return pd.DataFrame(rows)


def abstention_stats(priorities, label_sets, groups, fold_of_policy):
    abstained = np.array([p["dominant_category"] is None for p in priorities])
    n = len(priorities)
    rows = [{"scope": "overall", "key": "all", "n": n,
             "abstained": int(abstained.sum()),
             "abstention_rate": round(float(abstained.mean()), 4)}]
    # by gold category
    for c in CATEGORIES:
        mask = np.array([c in s for s in label_sets])
        tot = int(mask.sum())
        if tot:
            ab = int(abstained[mask].sum())
            rows.append({"scope": "by_gold_category", "key": c, "n": tot,
                         "abstained": ab,
                         "abstention_rate": round(ab / tot, 4)})
    # by fold
    fold_ids = np.array([fold_of_policy[str(g)] for g in groups])
    for fold in sorted(set(fold_ids)):
        mask = fold_ids == fold
        tot = int(mask.sum())
        if not tot:
            continue
        ab = int(abstained[mask].sum())
        rows.append({"scope": "by_fold", "key": f"fold_{fold}", "n": tot,
                     "abstained": ab, "abstention_rate": round(ab / tot, 4)})
    return pd.DataFrame(rows)


def priority_sensitivity(P, thresholds):
    rows = []
    for low, high in SENSITIVITY_BANDS:
        prios = compute_priorities(P, thresholds, bands=(low, high))
        dist = {s: 0 for s in range(1, 6)}
        for p in prios:
            dist[p["stars"]] += 1
        rows.append({"low_band": low, "high_band": high, **{
            f"{s}_star": dist[s] for s in range(1, 6)},
            "flagged": sum(p["review_flag"] for p in prios)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Examples + figure
# --------------------------------------------------------------------------
def priority_examples(P, thresholds, texts, label_sets, priorities, k_per_star=3):
    picked, by_star = [], {s: 0 for s in range(1, 6)}
    order = sorted(range(len(priorities)),
                   key=lambda i: priorities[i]["stars"], reverse=True)
    # ensure at least one high-attention near-miss (flagged) example
    near_miss_idx = next(
        (i for i in order if priorities[i]["review_flag"] and any(
            "just below threshold" in r for r in priorities[i]["review_reasons"])),
        None)
    forced = {near_miss_idx} if near_miss_idx is not None else set()

    for i in list(forced) + [i for i in order if i not in forced]:
        p = priorities[i]
        s = p["stars"]
        if i not in forced and by_star[s] >= k_per_star:
            continue
        by_star[s] += 1
        pr = probs_dict(P[i])
        pred = "; ".join(f"{c} ({pr[c]:.2f})"
                         for c in p["all_categories"]) or "(none)"
        picked.append({
            "stars": s,
            "priority_band": p["priority_band"],
            "priority_subtype": p["priority_subtype"],
            "dominant_category": p["dominant_category"] or "(abstained)",
            "attention_tier": p["attention_tier"],
            "confidence": p["confidence"],
            "review_required": p["review_required"],
            "primary_review_reason": p["primary_review_reason"] or "",
            "review_severity": p["review_severity"] or "",
            "review_types": " | ".join(p["review_types"]),
            "review_reasons": " | ".join(p["review_reasons"]),
            "gold_labels": "; ".join(sorted(label_sets[i])),
            "predicted": pred,
            "reason": p["reason"],
            "text": " ".join(texts[i].split())[:300],
            "prediction_source": "train_oof",
        })
        if len(picked) >= 15:
            break
    return pd.DataFrame(sorted(picked, key=lambda r: -r["stars"]))


def fig_priority_distribution(priorities):
    """Star distribution, separating identified-Medium from unknown/abstained.

    3-star (and other) bars are split so classifier failures (abstained or
    Unknown dominant) do not hide inside genuine medium-attention content.
    """
    identified = {s: 0 for s in range(1, 6)}
    unknown = {s: 0 for s in range(1, 6)}
    for p in priorities:
        s = p["stars"]
        if p["dominant_category"] is None or p["attention_tier"] == "Unknown":
            unknown[s] += 1
        else:
            identified[s] += 1

    stars = list(range(1, 6))
    id_vals = [identified[s] for s in stars]
    un_vals = [unknown[s] for s in stars]
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    ax.bar(stars, id_vals, color=BLUE, width=0.62,
           label="Identified practice")
    ax.bar(stars, un_vals, bottom=id_vals, color="#7a5cc0", width=0.62,
           label="Unknown / model abstained")
    for x, (a, b) in enumerate(zip(id_vals, un_vals), start=1):
        if a + b:
            ax.text(x, a + b + max(id_vals + un_vals) * 0.01, f"{a + b:,}",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(stars)
    ax.set_xlabel("Reading priority (stars)")
    ax.set_ylabel("Segments (train, OOF)")
    ax.set_title("Reading-priority distribution (Logistic Regression, OOF)",
                 loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8.5)
    ax.tick_params(length=0)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_priority_distribution.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    RESULTS.mkdir(exist_ok=True)
    df = pd.read_csv(SEGMENTS_CSV)
    train = df[df.split == "train"].reset_index(drop=True)
    texts = train["text"].fillna("").tolist()
    label_sets = [s.split("|") for s in train["labels"]]
    groups = train["policy_stem"].astype(str).values

    mlb = MultiLabelBinarizer(classes=CATEGORIES)
    Y = mlb.fit_transform(label_sets)

    fold_of_policy = load_or_build_folds(groups)

    # Fold support (rare-category safety) ---------------------------------
    support_df, warnings = fold_support_table(Y, groups, fold_of_policy)
    support_df.to_csv(RESULTS / "fold_support.csv", index=False)
    for w in warnings:
        print("WARNING:", w)

    # OOF probabilities ---------------------------------------------------
    P = compute_oof(texts, Y, groups, fold_of_policy)

    # Fresh thresholds ----------------------------------------------------
    thresholds, thr_details = select_thresholds(P, Y)
    write_thresholds(thresholds)
    thr_details.to_csv(RESULTS / "threshold_selection.csv", index=False)

    # Calibration ---------------------------------------------------------
    calibration_summary(P, Y).to_csv(
        RESULTS / "calibration_summary.csv", index=False)

    # Priorities (default bands) -> flags, abstention, examples, figure ---
    priorities = compute_priorities(P, thresholds)
    predicted_sets = [p["all_categories"] for p in priorities]
    flag_stats(priorities).to_csv(RESULTS / "flag_stats.csv", index=False)
    review_diagnostics(priorities, predicted_sets, label_sets).to_csv(
        RESULTS / "review_diagnostics.csv", index=False)
    priority_distribution(priorities).to_csv(
        RESULTS / "priority_distribution.csv", index=False)
    abstention_stats(priorities, label_sets, groups, fold_of_policy).to_csv(
        RESULTS / "abstention_stats.csv", index=False)
    priority_sensitivity(P, thresholds).to_csv(
        RESULTS / "priority_sensitivity.csv", index=False)
    priority_examples(P, thresholds, texts, label_sets, priorities).to_csv(
        RESULTS / "priority_examples.csv", index=False)
    fig_priority_distribution(priorities)

    # Framework + templates tables ---------------------------------------
    pd.DataFrame(framework_table()).to_csv(
        RESULTS / "attention_framework.csv", index=False)
    pd.DataFrame([{"category": c, "template": templates.explain(c)}
                  for c in CATEGORIES]).to_csv(
        RESULTS / "explanation_templates.csv", index=False)

    # Run-level provenance ------------------------------------------------
    INTERP_METADATA_JSON.write_text(json.dumps({
        "framework_version": FRAMEWORK_VERSION,
        "model_name": MODEL_NAME,
        "model_version": model_version(),
        "split_id": SPLIT_ID,
        "prediction_source": "train_oof",
        "threshold_source": "train_oof_f1",
        "threshold_note": ("development-set thresholds; not an unbiased "
                           "estimate of deployed performance"),
        "cv_protocol": f"{N_FOLDS}-fold policy-grouped",
        "seed": SEED,
        "confidence_bands": {"low": LOW_CONF, "high": HIGH_CONF},
    }, indent=2), encoding="utf-8")

    n = len(priorities)
    n_flag = sum(p["review_required"] for p in priorities)
    n_content = sum(p["content_unclear_flag"] for p in priorities)
    n_model = sum(p["model_uncertainty_flag"] for p in priorities)
    n_abs = sum(p["dominant_category"] is None for p in priorities)
    print(f"segments (train OOF): {n}")
    print("thresholds:", {c: round(t, 2) for c, t in thresholds.items()})
    print(f"review_required: {n_flag} ({n_flag / n:.1%})")
    print(f"  content-driven: {n_content} ({n_content / n:.1%}); "
          f"model-driven: {n_model} ({n_model / n:.1%})")
    print(f"abstained: {n_abs} ({n_abs / n:.1%})")
    print("wrote interpretation artifacts to", RESULTS)


if __name__ == "__main__":
    main()
