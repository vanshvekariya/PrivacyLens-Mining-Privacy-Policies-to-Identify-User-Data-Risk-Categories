"""Build the modern-policy generalization set from fixed policy snapshots.

Reads a provenance manifest and the untouched raw policy texts, then cleans and
segments each policy with the SAME length-aware segmentation the prototype uses
(``privacylens.pipeline.segment_policy``), so modern segments match the length
range the model was trained on. Emits:

  data/generalization/segments_unlabeled.csv   one row per segment, to be labeled
  data/generalization/annotation_template.csv  blank labels column to fill in
  data/generalization/manifest_resolved.csv    manifest + content hash + counts
  data/generalization/build_report.txt         human-readable summary

Raw files under data/generalization/raw/ are NEVER modified; all cleaning is
in-memory and written to separate derived files. See generalization_manifest.md
and generalization_labeling_guide.md.

Usage:
  1. python src/collect_generalization_policies.py
  2. python src/build_generalization_set.py
"""

import hashlib
import re
from pathlib import Path

import pandas as pd

from privacylens.config import DATA
from privacylens.pipeline import segment_policy

GEN = DATA / "generalization"
RAW = GEN / "raw"
MANIFEST = GEN / "manifest.csv"

MANIFEST_COLUMNS = ["policy_id", "service", "sector", "shift_type",
                    "source_url", "retrieval_date", "effective_date",
                    "raw_file", "notes"]

_TAG = re.compile(r"<[^>]+>")


def _strip_html(text):
    """Light HTML de-tagging in case a policy was pasted with markup."""
    if "<" in text and ">" in text:
        text = _TAG.sub(" ", text)
    return text


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"{MANIFEST} not found. Run "
            "python src/collect_generalization_policies.py first."
        )

    manifest = pd.read_csv(MANIFEST, dtype=str).fillna("")
    missing_cols = set(MANIFEST_COLUMNS) - set(manifest.columns)
    if missing_cols:
        raise ValueError(f"manifest.csv missing columns: {sorted(missing_cols)}")

    rows, resolved = [], []
    for _, m in manifest.iterrows():
        raw_path = RAW / m["raw_file"]
        if not raw_path.exists():
            print(f"WARNING: raw file not found, skipping: {raw_path}")
            continue
        raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
        sha = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        segments = segment_policy(_strip_html(raw_text))
        for s in segments:
            rows.append({
                "policy_id": m["policy_id"],
                "service": m["service"],
                "sector": m["sector"],
                "shift_type": m["shift_type"],
                "segment_id": f'{m["policy_id"]}-{s["segment_id"]}',
                "segment_index": s["segment_index"],
                "paragraph_index": s["paragraph_index"],
                "word_count": s["word_count"],
                "text": s["text"],
            })
        resolved.append({**m.to_dict(), "content_sha256": sha,
                         "n_segments": len(segments)})

    if not rows:
        print("No segments produced. Add raw files and fill the manifest.")
        return

    seg_df = pd.DataFrame(rows)
    seg_df.to_csv(GEN / "segments_unlabeled.csv", index=False)

    # ready-to-fill annotation template (labels/notes/guide_version blank)
    template = seg_df[["segment_id", "text"]].copy()
    template["annotator"] = ""
    template["labels"] = ""
    template["notes"] = ""
    template["guide_version"] = ""
    template.to_csv(GEN / "annotation_template.csv", index=False)

    pd.DataFrame(resolved).to_csv(GEN / "manifest_resolved.csv", index=False)

    n_pol = seg_df["policy_id"].nunique()
    by_shift = seg_df.groupby("shift_type")["segment_id"].count().to_dict()
    report = "\n".join([
        f"policies: {n_pol}",
        f"segments: {len(seg_df)}",
        f"segments by shift_type: {by_shift}",
        f"mean segments/policy: {len(seg_df)/n_pol:.1f}",
        f"word_count: mean {seg_df.word_count.mean():.0f}, "
        f"median {seg_df.word_count.median():.0f}, max {seg_df.word_count.max()}",
        "",
        "Next: run src/label_generalization_set.py, then "
        "src/merge_generalization_labels.py.",
    ])
    (GEN / "build_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
