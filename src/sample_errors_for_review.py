import re
from pathlib import Path

import numpy as np
import pandas as pd

from privacylens.config import CATEGORIES, SEED

# Data and results paths
ROOT = Path(__file__).resolve().parent.parent
SEGMENTS_CSV = ROOT / "data" / "segments.csv"
RESULTS = ROOT / "results"
ERROR_ANALYSIS_DIR = RESULTS / "error_analysis"

# Sampling parameters
ERRORS_PER_CATEGORY = 15
NEAR_THRESHOLD_MARGIN = 0.05
TAXONOMY_VERSION = "1.0"

# Model slug to OOF predictions filename
MODEL_OOF_FILES = {
    "naive_bayes": "oof_predictions_naive_bayes.csv",
    "logistic_regression": "oof_predictions_logistic_regression.csv",
    "linear_svm": "oof_predictions_linear_svm.csv",
    "distilbert": "transformer_oof_predictions.csv",
}


def sanitize_col(category):
    # Convert a category name into the corresponding probability column name
    return "prob_" + (category.replace("&", "and").replace("/", "_")
                              .replace("-", "_").replace(" ", "_"))


PROB_COLS = [sanitize_col(c) for c in CATEGORIES]


def load_predictions(model_slug):
    # Load the OOF predictions csv for a given model
    path = RESULTS / MODEL_OOF_FILES[model_slug]
    return pd.read_csv(path)


def load_segment_text():
    # Load segment text so error rows can include the original passage
    return pd.read_csv(SEGMENTS_CSV)[["policy_stem", "segment_id", "text"]]


def to_label_set(label_str):
    # Convert a pipe-separated label string into a set
    if not isinstance(label_str, str) or not label_str:
        return set()
    return set(label_str.split("|"))


def error_type_for_row(row):
    # Classify each error into a coarse pool for stratified sampling
    gold = to_label_set(row["gold_labels"])
    pred = to_label_set(row["predicted_labels_at_0_5"])
    false_negatives = gold - pred
    false_positives = pred - gold
    if false_negatives and false_positives:
        return "both"
    if false_negatives:
        return "false_negative"
    if false_positives:
        return "false_positive"
    return "correct"


def near_threshold(row):
    # True when any category probability sits within the near-threshold band of 0.5
    return any(abs(row[col] - 0.5) <= NEAR_THRESHOLD_MARGIN for col in PROB_COLS)


def build_error_pool(oof_df):
    # Return an errors-only dataframe augmented with helper columns
    errors = oof_df.copy()
    errors["error_type"] = errors.apply(error_type_for_row, axis=1)
    errors = errors[errors["error_type"] != "correct"].reset_index(drop=True)
    errors["near_threshold"] = errors.apply(near_threshold, axis=1)
    return errors


def stratified_sample(errors, per_category, rng):
    # Sample ~per_category errors per gold-or-predicted category, balancing
    # FP and FN, biased half toward near-threshold rows
    picks = []
    for cat in CATEGORIES:
        involved = errors[
            errors["gold_labels"].fillna("").str.contains(re.escape(cat)) |
            errors["predicted_labels_at_0_5"].fillna("").str.contains(re.escape(cat))
        ]
        if involved.empty:
            continue
        near = involved[involved["near_threshold"]]
        far = involved[~involved["near_threshold"]]
        n_near = min(len(near), per_category // 2)
        n_far = min(len(far), per_category - n_near)
        if n_near > 0:
            picks.append(near.sample(n=n_near, random_state=rng.integers(1 << 30)))
        if n_far > 0:
            picks.append(far.sample(n=n_far, random_state=rng.integers(1 << 30)))
    if not picks:
        return errors.head(0)
    combined = pd.concat(picks, ignore_index=True)
    combined = combined.drop_duplicates(subset=["policy_stem", "segment_id"]).reset_index(drop=True)
    return combined


def build_annotation_frame(sample, segments_df, model_slug):
    # Join with segment text and shape into the annotator-facing csv layout
    joined = sample.merge(segments_df, on=["policy_stem", "segment_id"], how="left")
    joined = joined.rename(columns={"predicted_labels_at_0_5": "predicted_labels"})
    joined["model"] = model_slug
    joined["failure_type"] = ""
    joined["notes"] = ""
    joined["taxonomy_version"] = TAXONOMY_VERSION
    cols = [
        "model", "policy_stem", "segment_id", "fold_id", "gold_labels",
        "predicted_labels", "error_type", "near_threshold", "text",
        "failure_type", "notes", "taxonomy_version",
    ]
    return joined[cols]


def write_annotation_csv(model_slug, annotation_df):
    # Write the per-model errors_to_annotate csv, skip if annotations already exist
    ERROR_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    path = ERROR_ANALYSIS_DIR / f"errors_to_annotate_{model_slug}.csv"
    if path.exists():
        prior = pd.read_csv(path)
        if "failure_type" in prior.columns:
            filled = prior["failure_type"].astype(str).str.strip() != ""
            if filled.any():
                return None
    annotation_df.to_csv(path, index=False)
    return path


def process_model(model_slug, segments_df, rng):
    # Full pipeline for one model: load, filter, sample, write
    print(f'\n{model_slug}:')
    oof = load_predictions(model_slug)
    errors = build_error_pool(oof)
    print(f'  {len(errors)} errors of {len(oof)} predicted segments')
    sample = stratified_sample(errors, ERRORS_PER_CATEGORY, rng)
    annotation = build_annotation_frame(sample, segments_df, model_slug)
    path = write_annotation_csv(model_slug, annotation)
    if path is None:
        print(f'  skipping write, existing errors_to_annotate_{model_slug}.csv has annotations')
    else:
        print(f'  wrote {len(annotation)} rows to {path.name}')


def main():
    # Load segment text once for all models
    print('Loading segment text')
    segments_df = load_segment_text()

    # Deterministic sampler
    rng = np.random.default_rng(SEED)

    # Sample errors for each model whose OOF predictions exist on disk
    for model_slug in MODEL_OOF_FILES:
        path = RESULTS / MODEL_OOF_FILES[model_slug]
        if not path.exists():
            print(f'\n{model_slug}: skipping ({path.name} not found)')
            continue
        process_model(model_slug, segments_df, rng)


# Call the main function
main()
