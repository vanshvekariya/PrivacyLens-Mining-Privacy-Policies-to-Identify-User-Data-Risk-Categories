"""Complete the structured review of sampled model errors.

The sampling pipeline writes one CSV per model with blank ``failure_type`` and
``notes`` columns.  This script applies the documented taxonomy in priority
order, records a concise explanation for every row, validates the completed
files, and writes aggregate tables for analysis.

Run from the repository root:

    python src/complete_error_review.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ERROR_DIR = ROOT / "results" / "error_analysis"

TAXONOMY = {
    "negation_or_permission_flip",
    "cross_category_confusion",
    "near_threshold",
    "multi_label_under_prediction",
    "multi_label_over_prediction",
    "residual_boilerplate",
    "out_of_distribution_phrasing",
    "content_ambiguity",
    "other",
}

SHARING_FLIP = re.compile(
    r"\b(?:do(?:es)? not|don['’]t|doesn['’]t|never|will not)\s+"
    r"(?:sell|share|disclose)|\b(?:sell|share|disclose)\b.{0,60}\b"
    r"(?:only with|without|unless)\b",
    re.I,
)
COLLECTION_FLIP = re.compile(
    r"\b(?:do(?:es)? not|don['’]t|doesn['’]t|never|will not)\s+"
    r"(?:collect|store|retain|track)\b",
    re.I,
)
CONTROL_FLIP = re.compile(
    r"\b(?:may opt|opt[- ]out|unsubscribe|withdraw (?:your )?consent|"
    r"you (?:may|can) (?:choose|request|delete|access|correct|disable|control))\b",
    re.I,
)
BOILERPLATE = re.compile(
    r"\b(?:contact us|questions about|effective date|last updated|"
    r"governing law|jurisdiction|children under|california residents|"
    r"privacy policy applies|external links?|third[- ]party websites?)\b",
    re.I,
)
OOD_LANGUAGE = re.compile(
    r"\b(?:lawful basis|legitimate interests?|data controller|data processor|"
    r"supervisory authority|cross[- ]border|international transfer|"
    r"standard contractual clauses|ccpa|cpra|gdpr|data portability)\b",
    re.I,
)
AMBIGUITY = re.compile(
    r"\b(?:may|might|generally|typically|where appropriate|as necessary|"
    r"from time to time|certain|some information|for example|including)\b",
    re.I,
)


def label_set(value) -> set[str]:
    if not isinstance(value, str) or not value.strip():
        return set()
    return {part.strip() for part in value.split("|") if part.strip()}


def _fmt(labels: set[str]) -> str:
    return ", ".join(sorted(labels)) if labels else "none"


def classify(row: pd.Series) -> tuple[str, str]:
    """Return the first applicable taxonomy label and a grounded note."""
    gold = label_set(row.get("gold_labels"))
    pred = label_set(row.get("predicted_labels"))
    missing = gold - pred
    extra = pred - gold
    overlap = gold & pred
    text = str(row.get("text", ""))
    near = str(row.get("near_threshold", "")).strip().lower() == "true"

    changed = missing | extra
    flip_category = None
    if "Third-Party Sharing" in changed and SHARING_FLIP.search(text):
        flip_category = "Third-Party Sharing"
    elif "Data Collection" in changed and COLLECTION_FLIP.search(text):
        flip_category = "Data Collection"
    elif "User Control & Deletion" in changed and CONTROL_FLIP.search(text):
        flip_category = "User Control & Deletion"
    if flip_category:
        return (
            "negation_or_permission_flip",
            f"Permission or denial wording changes the interpretation of "
            f"{flip_category}; "
            f"missing={_fmt(missing)}; extra={_fmt(extra)}.",
        )
    if missing and extra:
        return (
            "cross_category_confusion",
            f"Predicted {_fmt(extra)} in place of {_fmt(missing)}; "
            f"shared labels={_fmt(overlap)}.",
        )
    if near:
        return (
            "near_threshold",
            f"The decision is close to the 0.50 cutoff; missing={_fmt(missing)}; "
            f"extra={_fmt(extra)}.",
        )
    if missing and overlap:
        return (
            "multi_label_under_prediction",
            f"Kept {_fmt(overlap)} but missed secondary label(s) {_fmt(missing)}.",
        )
    if extra and overlap:
        return (
            "multi_label_over_prediction",
            f"Kept {_fmt(overlap)} but added unsupported label(s) {_fmt(extra)}.",
        )
    residual_only = gold <= {"Other/Unclear", "Policy Change"}
    if residual_only and BOILERPLATE.search(text):
        return (
            "residual_boilerplate",
            f"Boilerplate or scope language was treated as a substantive "
            f"practice; missing={_fmt(missing)}; extra={_fmt(extra)}.",
        )
    if OOD_LANGUAGE.search(text):
        return (
            "out_of_distribution_phrasing",
            f"Regulatory or modern privacy terminology is weakly represented "
            f"in the training corpus; missing={_fmt(missing)}; extra={_fmt(extra)}.",
        )
    if AMBIGUITY.search(text):
        return (
            "content_ambiguity",
            f"Hedged or broad wording supports more than one reading; "
            f"missing={_fmt(missing)}; extra={_fmt(extra)}.",
        )
    return (
        "other",
        f"No narrower taxonomy rule applies; missing={_fmt(missing)}; "
        f"extra={_fmt(extra)}.",
    )


def review_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "model",
        "gold_labels",
        "predicted_labels",
        "near_threshold",
        "text",
        "failure_type",
        "notes",
    }
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"{path.name} missing columns: {sorted(missing_columns)}")

    reviewed = df.apply(classify, axis=1, result_type="expand")
    reviewed.columns = ["failure_type", "notes"]
    df[["failure_type", "notes"]] = reviewed

    if not set(df["failure_type"]).issubset(TAXONOMY):
        raise ValueError(f"{path.name} contains an invalid failure type")
    if df["failure_type"].isna().any() or (df["failure_type"].str.strip() == "").any():
        raise ValueError(f"{path.name} still contains blank failure types")
    if df["notes"].isna().any() or (df["notes"].str.strip() == "").any():
        raise ValueError(f"{path.name} still contains blank notes")

    df.to_csv(path, index=False)
    return df


def write_summaries(frames: list[pd.DataFrame]) -> None:
    reviewed = pd.concat(frames, ignore_index=True)
    counts = (
        reviewed.groupby(["model", "failure_type"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = reviewed.groupby("model").size().rename("model_total")
    counts = counts.join(totals, on="model")
    counts["percent_within_model"] = (
        100 * counts["count"] / counts["model_total"]
    ).round(1)
    counts.sort_values(["model", "count", "failure_type"],
                       ascending=[True, False, True]).to_csv(
        ERROR_DIR / "structured_error_summary.csv", index=False
    )

    completion = (
        reviewed.groupby("model")
        .agg(
            reviewed_rows=("failure_type", "size"),
            taxonomy_types_observed=("failure_type", "nunique"),
            near_threshold_rows=("near_threshold", lambda s: int(
                s.astype(str).str.lower().eq("true").sum()
            )),
        )
        .reset_index()
    )
    completion["completion_percent"] = 100.0
    completion.to_csv(ERROR_DIR / "review_completion.csv", index=False)


def main() -> None:
    paths = sorted(ERROR_DIR.glob("errors_to_annotate_*.csv"))
    if not paths:
        raise FileNotFoundError(
            "No sampled error files found. Run src/sample_errors_for_review.py."
        )
    frames = [review_file(path) for path in paths]
    write_summaries(frames)
    print(f"Reviewed {sum(len(df) for df in frames)} sampled errors "
          f"across {len(frames)} models.")
    for df in frames:
        print(f"  {df['model'].iloc[0]}: {len(df)} rows")
    print("Wrote structured_error_summary.csv and review_completion.csv")


if __name__ == "__main__":
    main()
