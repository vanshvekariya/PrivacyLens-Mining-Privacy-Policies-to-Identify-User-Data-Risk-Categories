"""Generalization evaluation on the modern-policy set (RQ3).

Compares in-distribution (OPP-115) vs out-of-distribution (modern policies)
per-category performance for the two strongest classical models.

FREEZE DISCIPLINE (see the labeling guide): this script must be run only AFTER
the model configs and thresholds are frozen, and only ONCE. It does not tune
anything on the modern labels.

Models:
  - Logistic Regression: the FROZEN persisted prototype pipeline, with the
    frozen tuned thresholds applied to predict_proba.
  - Linear SVM: trained on the full OPP-115 train split with the reported
    run_baselines configuration, using its fixed decision rule (predict). SVM
    scores are NOT fed into the interpretation layer.

In-distribution reference: policy-grouped out-of-fold results already computed
by run_baselines.py (results/per_category_f1.csv). The untouched policy-disjoint
OPP-115 TEST split remains reserved for the final report and is not read here.

Low support is reported honestly: a category with zero positive examples in the
modern gold gets F1 = N/A (not 0), and macro-F1 is averaged only over categories
that have modern support. Policy-level bootstrap CIs communicate uncertainty.

Outputs (results/):
  generalization_support.csv     per-category positive/negative support (modern)
  generalization_eval.csv        per-category F1 in-dist vs modern (+ degradation)
  generalization_headline.csv    macro/micro F1 in-dist vs modern + bootstrap CIs
  figures/fig_generalization.png per-category F1, in-dist vs modern
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from privacylens.config import (
    CATEGORIES,
    DATA,
    RESULTS,
    SEED,
    SEGMENTS_CSV,
    TFIDF_PARAMS,
)
from privacylens.pipeline import load_prototype_model

GEN = DATA / "generalization"
FIGS = RESULTS / "figures"
N_BOOTSTRAP = 1000


# --------------------------------------------------------------------------
def load_modern():
    """Join adjudicated labels with segment text + provenance."""
    adj_path = GEN / "segments_adjudicated.csv"
    seg_path = GEN / "segments_unlabeled.csv"
    if not adj_path.exists() or not seg_path.exists():
        raise FileNotFoundError(
            "Need segments_adjudicated.csv and segments_unlabeled.csv. Run "
            "build_generalization_set.py and merge_generalization_labels.py.")
    adj = pd.read_csv(adj_path, dtype=str).fillna("")
    seg = pd.read_csv(seg_path, dtype=str).fillna("")
    df = seg.merge(adj[["segment_id", "labels", "needs_adjudication",
                        "source"]], on="segment_id", how="inner")
    unresolved = df["needs_adjudication"].astype(str).str.lower() == "true"
    if unresolved.any():
        print(f"WARNING: excluding {int(unresolved.sum())} unresolved segment(s) "
              f"pending adjudication.")
        df = df[~unresolved].reset_index(drop=True)
    df["label_set"] = df["labels"].apply(
        lambda s: [x for x in s.split("|") if x])
    return df


def train_svm(train_texts, Y):
    pipe = make_pipeline(
        TfidfVectorizer(**TFIDF_PARAMS),
        OneVsRestClassifier(LinearSVC(class_weight="balanced", C=0.5,
                                      random_state=SEED)))
    pipe.fit(train_texts, Y)
    return pipe


def lr_predict(model, texts):
    """Frozen LR predict_proba -> frozen thresholds -> binary matrix."""
    proba = np.asarray(model["pipeline"].predict_proba(list(texts)))
    thr = np.array([model["thresholds"][c] for c in CATEGORIES])
    return (proba >= thr).astype(int)


def per_category_ood(Y_true, Y_pred):
    """Per-category P/R/F1 with N/A (NaN) where modern support is 0."""
    p, r, f, s = precision_recall_fscore_support(
        Y_true, Y_pred, labels=range(len(CATEGORIES)), zero_division=0)
    rows = []
    for j, c in enumerate(CATEGORIES):
        support = int(s[j]) if s is not None else int(Y_true[:, j].sum())
        support = int(Y_true[:, j].sum())
        if support == 0:
            rows.append({"category": c, "modern_support": 0,
                         "precision": np.nan, "recall": np.nan, "f1": np.nan})
        else:
            rows.append({"category": c, "modern_support": support,
                         "precision": round(float(p[j]), 4),
                         "recall": round(float(r[j]), 4),
                         "f1": round(float(f[j]), 4)})
    return pd.DataFrame(rows)


def macro_micro(Y_true, Y_pred, supported):
    """Macro over supported categories only; micro over all."""
    cols = [j for j, c in enumerate(CATEGORIES) if c in supported]
    macro = f1_score(Y_true[:, cols], Y_pred[:, cols],
                     average="macro", zero_division=0) if cols else np.nan
    micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    return float(macro), float(micro)


def bootstrap_ci(Y_true, Y_pred, groups, supported, n=N_BOOTSTRAP, seed=SEED):
    """Policy-level bootstrap of macro/micro F1 (resample policies)."""
    rng = np.random.default_rng(seed)
    policies = np.array(sorted(set(groups)))
    idx_by_policy = {p: np.where(groups == p)[0] for p in policies}
    macros, micros = [], []
    for _ in range(n):
        sample = rng.choice(policies, size=len(policies), replace=True)
        idx = np.concatenate([idx_by_policy[p] for p in sample])
        ma, mi = macro_micro(Y_true[idx], Y_pred[idx], supported)
        if not np.isnan(ma):
            macros.append(ma)
        micros.append(mi)

    def ci(vals):
        if not vals:
            return (np.nan, np.nan)
        return (round(float(np.percentile(vals, 2.5)), 4),
                round(float(np.percentile(vals, 97.5)), 4))
    return ci(macros), ci(micros)


def indist_reference():
    """In-distribution per-category F1 from run_baselines OOF results."""
    path = RESULTS / "per_category_f1.csv"
    if not path.exists():
        print("WARNING: results/per_category_f1.csv not found; run "
              "run_baselines.py for the in-distribution reference.")
        return {}
    df = pd.read_csv(path)
    ref = {}
    for model in df["model"].unique():
        sub = df[df.model == model]
        ref[model] = dict(zip(sub["category"], sub["f1"]))
    return ref


def fig_generalization(eval_df, model_label):
    sub = eval_df[eval_df.model == model_label].dropna(subset=["modern_f1"])
    if sub.empty:
        return
    cats = sub["category"].tolist()
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.bar(x - 0.2, sub["indist_f1"].fillna(0), width=0.4, label="In-dist (OOF)",
           color="#2a78d6")
    ax.bar(x + 0.2, sub["modern_f1"], width=0.4, label="Modern (OOD)",
           color="#d9922a")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("F1")
    ax.set_title(f"In-distribution vs modern-policy F1 ({model_label})",
                 loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig_generalization.png", dpi=200)
    plt.close(fig)


def main():
    print("NOTE: run this once, after models + thresholds are frozen.\n")
    modern = load_modern()
    if modern.empty:
        print("No resolved modern segments to evaluate.")
        return

    mlb = MultiLabelBinarizer(classes=CATEGORIES)
    Y_true = mlb.fit_transform(modern["label_set"])
    groups = modern["policy_id"].values
    texts = modern["text"].tolist()
    supported = {c for j, c in enumerate(CATEGORIES) if Y_true[:, j].sum() > 0}

    # ---- train / load models -------------------------------------------
    train = pd.read_csv(SEGMENTS_CSV)
    train = train[train.split == "train"].reset_index(drop=True)
    Y_train = MultiLabelBinarizer(classes=CATEGORIES).fit_transform(
        [s.split("|") for s in train["labels"]])
    train_texts = train["text"].fillna("").tolist()

    lr_model = load_prototype_model()
    svm = train_svm(train_texts, Y_train)

    preds = {
        "Logistic Regression": lr_predict(lr_model, texts),
        "Linear SVM": svm.predict(texts),
    }

    ref = indist_reference()

    # ---- support table --------------------------------------------------
    support_rows = []
    for j, c in enumerate(CATEGORIES):
        pos = int(Y_true[:, j].sum())
        support_rows.append({"category": c, "positive_support": pos,
                             "negative_support": int(len(Y_true) - pos)})
    pd.DataFrame(support_rows).to_csv(
        RESULTS / "generalization_support.csv", index=False)

    # ---- per-category eval + headline ----------------------------------
    eval_rows, headline_rows = [], []
    for model_label, Y_pred in preds.items():
        cat_df = per_category_ood(Y_true, Y_pred)
        model_ref = ref.get(model_label, {})
        for _, row in cat_df.iterrows():
            indist = model_ref.get(row["category"], np.nan)
            modern_f1 = row["f1"]
            degradation = (round(indist - modern_f1, 4)
                           if not (np.isnan(modern_f1) or pd.isna(indist))
                           else np.nan)
            eval_rows.append({
                "model": model_label, "category": row["category"],
                "modern_support": row["modern_support"],
                "indist_f1": round(float(indist), 4) if not pd.isna(indist)
                else np.nan,
                "modern_precision": row["precision"],
                "modern_recall": row["recall"],
                "modern_f1": modern_f1,
                "degradation": degradation,
            })
        macro, micro = macro_micro(Y_true, Y_pred, supported)
        (ma_ci, mi_ci) = bootstrap_ci(Y_true, Y_pred, groups, supported)
        headline_rows.append({
            "model": model_label,
            "modern_macro_f1": round(macro, 4),
            "modern_macro_f1_ci_low": ma_ci[0], "modern_macro_f1_ci_high": ma_ci[1],
            "modern_micro_f1": round(micro, 4),
            "modern_micro_f1_ci_low": mi_ci[0], "modern_micro_f1_ci_high": mi_ci[1],
            "n_supported_categories": len(supported),
            "n_segments": int(len(Y_true)),
            "n_policies": int(len(set(groups))),
        })

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(RESULTS / "generalization_eval.csv", index=False)
    pd.DataFrame(headline_rows).to_csv(
        RESULTS / "generalization_headline.csv", index=False)

    fig_generalization(eval_df, "Logistic Regression")

    print(f"modern segments: {len(Y_true)} from {len(set(groups))} policies")
    print(f"categories with modern support: {sorted(supported)}")
    print(pd.DataFrame(headline_rows).to_string(index=False))
    print("\nwrote generalization_eval.csv, generalization_headline.csv, "
          "generalization_support.csv, figures/fig_generalization.png")


if __name__ == "__main__":
    main()
