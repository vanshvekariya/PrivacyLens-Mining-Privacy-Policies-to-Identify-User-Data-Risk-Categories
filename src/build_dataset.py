"""Build the PrivacyLens dataset from the OPP-115 corpus.

Reads consolidated annotations (threshold 0.5 overlap similarity), joins them
with segment texts from sanitized_policies/, maps the ten OPP-115 categories
to user-facing categories, and writes:

  data/segments.csv        one row per (policy, segment) with multi-hot labels
  results/label_distribution.csv   original and mapped label counts
  results/dataset_stats.txt        corpus-level statistics
  results/splits.json              policy-disjoint train/test policy ids

Test set is held out and must not be touched until the final report.
"""

import json
import re
import random
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OPP = ROOT / "data" / "OPP-115"
CONSOLIDATION = OPP / "consolidation" / "threshold-0.5-overlap-similarity"
POLICIES = OPP / "sanitized_policies"
RESULTS = ROOT / "results"

SEED = 42
TEST_FRACTION = 0.2

# OPP-115 top-level category -> user-facing category (formal_proposal_v2.md §3)
CATEGORY_MAP = {
    "First Party Collection/Use": "Data Collection",
    "Third Party Sharing/Collection": "Third-Party Sharing",
    "User Choice/Control": "User Control & Deletion",
    "User Access, Edit and Deletion": "User Control & Deletion",
    "Data Retention": "Data Retention",
    "Data Security": "Data Security",
    "Policy Change": "Policy Change",
    "Do Not Track": "Other/Unclear",
    "International and Specific Audiences": "Other/Unclear",
    "Other": "Other/Unclear",
}

ANNOTATION_COLUMNS = [
    "annotation_id", "batch_id", "annotator_id", "policy_id",
    "segment_id", "category", "attributes", "date", "policy_url",
]


def clean_segment(html: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", html)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_segments():
    """Return {policy_stem: [segment_text, ...]} from sanitized policies."""
    segments = {}
    for f in POLICIES.glob("*.html"):
        raw = f.read_text(encoding="utf-8", errors="replace")
        segments[f.stem] = [clean_segment(s) for s in raw.split("|||")]
    return segments


def load_annotations():
    frames = []
    for f in CONSOLIDATION.glob("*.csv"):
        df = pd.read_csv(f, header=None, names=ANNOTATION_COLUMNS)
        df["policy_stem"] = f.stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    RESULTS.mkdir(exist_ok=True)
    random.seed(SEED)

    segments = load_segments()
    ann = load_annotations()

    unknown = set(ann["category"]) - set(CATEGORY_MAP)
    if unknown:
        raise ValueError(f"Unmapped OPP-115 categories: {unknown}")

    ann["mapped"] = ann["category"].map(CATEGORY_MAP)

    # Segment-level label sets (dedup annotations per segment)
    orig_labels = ann.groupby(["policy_stem", "segment_id"])["category"].agg(set)
    mapped_labels = ann.groupby(["policy_stem", "segment_id"])["mapped"].agg(set)

    rows = []
    missing = 0
    for (stem, seg_id), cats in mapped_labels.items():
        segs = segments.get(stem)
        if segs is None or seg_id >= len(segs):
            missing += 1
            continue
        rows.append({
            "policy_stem": stem,
            "segment_id": seg_id,
            "text": segs[seg_id],
            "labels": "|".join(sorted(cats)),
            "orig_labels": "|".join(sorted(orig_labels.loc[(stem, seg_id)])),
        })
    df = pd.DataFrame(rows)

    # Policy-disjoint train/test split
    policies = sorted(df["policy_stem"].unique())
    random.shuffle(policies)
    n_test = round(len(policies) * TEST_FRACTION)
    test_policies = set(policies[:n_test])
    df["split"] = df["policy_stem"].map(
        lambda p: "test" if p in test_policies else "train")
    df.to_csv(ROOT / "data" / "segments.csv", index=False)

    with open(RESULTS / "splits.json", "w") as f:
        json.dump({"seed": SEED,
                   "test_policies": sorted(test_policies),
                   "train_policies": sorted(set(policies) - test_policies)},
                  f, indent=1)

    # ---- EDA outputs ----
    orig_counts = Counter(c for s in orig_labels for c in s)
    mapped_counts = Counter(c for s in df["labels"] for c in s.split("|"))
    dist = pd.DataFrame(
        [{"level": "original", "category": k, "segments": v}
         for k, v in orig_counts.most_common()] +
        [{"level": "mapped", "category": k, "segments": v}
         for k, v in mapped_counts.most_common()])
    dist.to_csv(RESULTS / "label_distribution.csv", index=False)

    n_labels = df["labels"].str.split("|").str.len()
    words = df["text"].str.split().str.len()
    train = df[df.split == "train"]
    test = df[df.split == "test"]
    stats = "\n".join([
        f"policies: {len(policies)}",
        f"annotated segments: {len(df)} (skipped {missing} with no matching text)",
        f"segments per policy: mean {len(df)/len(policies):.1f}",
        f"multi-label segments (mapped): {(n_labels > 1).sum()} "
        f"({(n_labels > 1).mean():.1%}); mean labels/segment {n_labels.mean():.2f}",
        f"segment length: mean {words.mean():.0f} words, median {words.median():.0f}, "
        f"p95 {words.quantile(0.95):.0f}",
        f"split: {len(train)} train segments / {len(test)} test segments "
        f"({len(policies) - n_test}/{n_test} policies), policy-disjoint, seed {SEED}",
        "",
        "mapped label counts (all segments):",
        *(f"  {k}: {v}" for k, v in mapped_counts.most_common()),
        "",
        "original label counts (all segments):",
        *(f"  {k}: {v}" for k, v in orig_counts.most_common()),
    ])
    (RESULTS / "dataset_stats.txt").write_text(stats, encoding="utf-8")
    print(stats)


if __name__ == "__main__":
    main()
