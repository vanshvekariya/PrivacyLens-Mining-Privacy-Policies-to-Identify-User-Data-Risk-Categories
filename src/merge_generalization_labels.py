"""Validate, measure agreement on, and adjudicate modern-policy annotations.

Input:  data/generalization/annotations_raw.csv
        columns: segment_id, annotator, labels (pipe-separated), notes,
                 guide_version
Optional override: data/generalization/adjudication.csv
        columns: segment_id, final_labels (pipe-separated) - resolves the
        disagreements a human adjudicated.

Outputs (data/generalization/):
  agreement_by_category.csv     per-category Cohen's kappa + pos/neg agreement
                                on the double-labeled overlap subset
  annotation_disagreements.csv  the specific category-level disagreements
  segments_adjudicated.csv      one final label set per segment (+ provenance),
                                with needs_adjudication flag for unresolved ones

Design choices (see generalization_labeling_guide.md):
  - Labels are validated against the canonical category set; unknown labels are
    a hard error (never silently dropped or invented).
  - Agreement is reported PER BINARY CATEGORY (Cohen's kappa + simple
    positive/negative agreement + positive count), not as one exact-match
    number, because the task is multi-label. When >2 annotators overlap on a
    segment, all annotator pairs are pooled per category (documented
    approximation for a small study).
  - Both raw annotations and the final adjudicated labels are retained.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from privacylens.config import CATEGORIES, DATA, canonicalize_category

GEN = DATA / "generalization"
RAW = GEN / "annotations_raw.csv"
ADJUDICATION = GEN / "adjudication.csv"


def _parse_labels(cell):
    if not isinstance(cell, str) or not cell.strip():
        return set()
    out = set()
    for part in cell.split("|"):
        part = part.strip()
        if part:
            out.add(canonicalize_category(part))     # raises on unknown label
    return out


def _load_annotations():
    if not RAW.exists():
        raise FileNotFoundError(
            f"{RAW} not found. Collect annotations into it (see the labeling "
            f"guide), or run build_generalization_set.py to create a template.")
    df = pd.read_csv(RAW, dtype=str).fillna("")
    required = {"segment_id", "annotator", "labels"}
    if not required.issubset(df.columns):
        raise ValueError(f"annotations_raw.csv needs columns {required}")
    invalid = []
    parsed = []
    for _, r in df.iterrows():
        try:
            labels = _parse_labels(r["labels"])
        except ValueError as e:
            invalid.append((r["segment_id"], r["annotator"], r["labels"], str(e)))
            labels = set()
        parsed.append(labels)
    if invalid:
        lines = "\n".join(f"  {sid} / {ann}: {lab!r} ({err})"
                          for sid, ann, lab, err in invalid)
        raise ValueError(f"invalid labels in annotations_raw.csv:\n{lines}")
    df["label_set"] = parsed
    return df


def agreement_by_category(df):
    """Per-category kappa + pos/neg agreement on the overlap subset."""
    # segments annotated by >=2 annotators
    ann_by_seg = df.groupby("segment_id")["annotator"].nunique()
    overlap_ids = ann_by_seg[ann_by_seg >= 2].index.tolist()

    rows = []
    n_pairs_total = 0
    per_cat_pairs = {c: ([], []) for c in CATEGORIES}   # (rater_a, rater_b)
    for sid in overlap_ids:
        sub = df[df.segment_id == sid]
        sets = list(sub["label_set"])
        annos = list(sub["annotator"])
        for i, j in combinations(range(len(sets)), 2):
            n_pairs_total += 1
            for c in CATEGORIES:
                per_cat_pairs[c][0].append(int(c in sets[i]))
                per_cat_pairs[c][1].append(int(c in sets[j]))

    for c in CATEGORIES:
        a = np.array(per_cat_pairs[c][0])
        b = np.array(per_cat_pairs[c][1])
        n = len(a)
        pos = int(((a == 1) | (b == 1)).sum())
        if n == 0:
            kappa = np.nan
            pos_agree = np.nan
            neg_agree = np.nan
        else:
            agree = a == b
            # positive agreement = both positive / (at least one positive)
            both_pos = int(((a == 1) & (b == 1)).sum())
            any_pos = int(((a == 1) | (b == 1)).sum())
            both_neg = int(((a == 0) & (b == 0)).sum())
            any_neg = int(((a == 0) | (b == 0)).sum())
            pos_agree = round(both_pos / any_pos, 4) if any_pos else np.nan
            neg_agree = round(both_neg / any_neg, 4) if any_neg else np.nan
            # kappa undefined when a category never varies across raters
            if a.std() == 0 and b.std() == 0:
                kappa = np.nan
            else:
                kappa = round(float(cohen_kappa_score(a, b)), 4)
        rows.append({"category": c, "n_overlap_segments": len(overlap_ids),
                     "n_pairs": n, "positive_count": pos,
                     "cohen_kappa": kappa, "positive_agreement": pos_agree,
                     "negative_agreement": neg_agree})
    return pd.DataFrame(rows), overlap_ids


def disagreements(df, overlap_ids):
    rows = []
    for sid in overlap_ids:
        sub = df[df.segment_id == sid]
        sets = {r["annotator"]: r["label_set"] for _, r in sub.iterrows()}
        union = set().union(*sets.values())
        inter = set.intersection(*sets.values()) if sets else set()
        for c in sorted(union - inter):
            pos = [a for a, s in sets.items() if c in s]
            neg = [a for a, s in sets.items() if c not in s]
            rows.append({"segment_id": sid, "category": c,
                         "annotators_positive": "|".join(sorted(pos)),
                         "annotators_negative": "|".join(sorted(neg))})
    return pd.DataFrame(rows)


def adjudicate(df, overlap_ids):
    override = {}
    if ADJUDICATION.exists():
        adj = pd.read_csv(ADJUDICATION, dtype=str).fillna("")
        for _, r in adj.iterrows():
            override[r["segment_id"]] = _parse_labels(r.get("final_labels", ""))

    rows = []
    for sid, sub in df.groupby("segment_id"):
        sets = list(sub["label_set"])
        n_ann = sub["annotator"].nunique()
        if sid in override:
            final, source, needs = override[sid], "adjudicated", False
        elif n_ann < 2:
            final, source, needs = sets[0], "single", False
        else:
            inter = set.intersection(*sets)
            union = set().union(*sets)
            if inter == union:
                final, source, needs = union, "agreed", False
            else:
                # keep a best-effort union but flag as unresolved
                final, source, needs = union, "union_unresolved", True
        rows.append({"segment_id": sid,
                     "labels": "|".join(sorted(final)),
                     "n_annotators": int(n_ann),
                     "source": source,
                     "needs_adjudication": needs})
    return pd.DataFrame(rows).sort_values("segment_id")


def main():
    df = _load_annotations()
    agree, overlap_ids = agreement_by_category(df)
    dis = disagreements(df, overlap_ids)
    adj = adjudicate(df, overlap_ids)

    agree.to_csv(GEN / "agreement_by_category.csv", index=False)
    dis.to_csv(GEN / "annotation_disagreements.csv", index=False)
    adj.to_csv(GEN / "segments_adjudicated.csv", index=False)

    n_unresolved = int(adj["needs_adjudication"].sum())
    print(f"annotators: {df['annotator'].nunique()}; "
          f"annotated segments: {df['segment_id'].nunique()}; "
          f"overlap (double-labeled): {len(overlap_ids)}")
    print(agree.to_string(index=False))
    if n_unresolved:
        print(f"\n{n_unresolved} segment(s) NEED ADJUDICATION. Resolve them in "
              f"{ADJUDICATION.name} (segment_id, final_labels) and re-run.")
    else:
        print("\nAll segments resolved (no pending adjudication).")


if __name__ == "__main__":
    main()
