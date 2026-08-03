import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

from privacylens.config import (CATEGORIES, SEED, TRANSFORMER_OOF_CSV)

# Data and results paths
ROOT = Path(__file__).resolve().parent.parent
SEGMENTS_CSV = ROOT / "data" / "segments.csv"
RESULTS = ROOT / "results"
MODEL_COMPARISON_CSV = RESULTS / "model_comparison.csv"
PER_CATEGORY_F1_CSV = RESULTS / "per_category_f1.csv"
ERROR_EXAMPLES_FILE = RESULTS / "error_examples_distilbert.txt"

MODEL_LABEL = "DistilBERT"


def load_oof():
    # Load transformer out-of-fold predictions
    return pd.read_csv(TRANSFORMER_OOF_CSV)


def to_multi_hot(label_strings):
    # Convert pipe-separated label strings into a fixed-order multi-hot matrix
    label_sets = [s.split("|") if isinstance(s, str) and s else [] for s in label_strings]
    mlb = MultiLabelBinarizer(classes=CATEGORIES)
    return mlb.fit_transform(label_sets).astype(int)


def compute_fold_scores(oof_df):
    # Compute macro and micro f1 per fold
    fold_scores = []
    for fold_id, sub in oof_df.groupby("fold_id"):
        gold = to_multi_hot(sub["gold_labels"])
        pred = to_multi_hot(sub["predicted_labels_at_0_5"])
        fold_scores.append({
            "fold_id": int(fold_id),
            "macro_f1": f1_score(gold, pred, average="macro", zero_division=0),
            "micro_f1": f1_score(gold, pred, average="micro", zero_division=0),
        })
    return pd.DataFrame(fold_scores).sort_values("fold_id").reset_index(drop=True)


def compute_per_category(oof_df):
    # Compute per-category precision, recall, f1 pooled across folds
    gold = to_multi_hot(oof_df["gold_labels"])
    pred = to_multi_hot(oof_df["predicted_labels_at_0_5"])
    p, r, f, s = precision_recall_fscore_support(gold, pred, zero_division=0)
    rows = []
    for j, cat in enumerate(CATEGORIES):
        rows.append({
            "model": MODEL_LABEL,
            "category": cat,
            "support": int(s[j]),
            "precision": float(p[j]),
            "recall": float(r[j]),
            "f1": float(f[j]),
        })
    return pd.DataFrame(rows)


def update_model_comparison(fold_scores):
    # Append or replace the DistilBERT row in model_comparison.csv
    row = {
        "model": MODEL_LABEL,
        "macro_f1_mean": float(fold_scores["macro_f1"].mean()),
        "macro_f1_std": float(fold_scores["macro_f1"].std()),
        "micro_f1_mean": float(fold_scores["micro_f1"].mean()),
        "micro_f1_std": float(fold_scores["micro_f1"].std()),
    }
    if MODEL_COMPARISON_CSV.exists():
        existing = pd.read_csv(MODEL_COMPARISON_CSV)
        existing = existing[existing["model"] != MODEL_LABEL]
        combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        combined = pd.DataFrame([row])
    combined = combined.sort_values("macro_f1_mean").reset_index(drop=True)
    combined.to_csv(MODEL_COMPARISON_CSV, index=False)
    return combined


def update_per_category_f1(per_cat):
    # Append or replace DistilBERT rows in per_category_f1.csv
    if PER_CATEGORY_F1_CSV.exists():
        existing = pd.read_csv(PER_CATEGORY_F1_CSV)
        existing = existing[existing["model"] != MODEL_LABEL]
        combined = pd.concat([existing, per_cat], ignore_index=True)
    else:
        combined = per_cat.copy()
    combined.to_csv(PER_CATEGORY_F1_CSV, index=False)
    return combined


def write_error_examples(oof_df, n_examples=12):
    # Sample misclassified segments and write text file with gold, pred, and text
    segments = pd.read_csv(SEGMENTS_CSV)[["policy_stem", "segment_id", "text"]]
    merged = oof_df.merge(segments, on=["policy_stem", "segment_id"], how="left")

    gold_sets = [set(s.split("|")) if isinstance(s, str) and s else set()
                 for s in merged["gold_labels"]]
    pred_sets = [set(s.split("|")) if isinstance(s, str) and s else set()
                 for s in merged["predicted_labels_at_0_5"]]
    wrong_idx = [i for i in range(len(merged)) if gold_sets[i] != pred_sets[i]]

    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(wrong_idx),
                      size=min(n_examples, len(wrong_idx)),
                      replace=False)

    lines = [f"Model: {MODEL_LABEL}. {len(wrong_idx)}/{len(merged)} segments with "
             f"imperfect label sets in out-of-fold predictions.", ""]
    for k in pick:
        i = wrong_idx[k]
        text = re.sub(r"\s+", " ", str(merged.iloc[i]["text"]))[:400]
        lines += [
            f"GOLD: {sorted(gold_sets[i])}",
            f"PRED: {sorted(pred_sets[i]) or '(none)'}",
            f"TEXT: {text}",
            "-" * 80,
        ]
    ERROR_EXAMPLES_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    # Load out-of-fold predictions
    print('Loading transformer OOF predictions')
    oof = load_oof()
    print(f'Loaded {len(oof)} predicted segments across '
          f'{oof["fold_id"].nunique()} folds')

    # Compute per-fold macro / micro f1
    print('Computing per-fold scores')
    fold_scores = compute_fold_scores(oof)
    print(fold_scores.to_string(index=False))

    # Update model comparison table
    print('Updating model comparison table')
    comparison = update_model_comparison(fold_scores)
    print(comparison.to_string(index=False))

    # Compute and update per-category metrics
    print('Computing per-category precision, recall, f1')
    per_cat = compute_per_category(oof)
    update_per_category_f1(per_cat)
    print(per_cat.sort_values("f1", ascending=False).to_string(index=False))

    # Write sample error examples
    print('Writing error examples')
    write_error_examples(oof)
    print(f'Wrote {ERROR_EXAMPLES_FILE.name}')


# Call the main function
main()
