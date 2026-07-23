"""Build the modern-policy generalization set from manually collected policies.

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
  1. python src/build_generalization_set.py   # first run scaffolds the manifest
  2. paste policy text into data/generalization/raw/<raw_file>.txt
  3. fill data/generalization/manifest.csv
  4. python src/build_generalization_set.py   # builds segments + templates
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


def _scaffold_manifest():
    RAW.mkdir(parents=True, exist_ok=True)
    example = pd.DataFrame([
        {"policy_id": "example_service_2026", "service": "Example Service",
         "sector": "example", "shift_type": "service",
         "source_url": "https://example.com/privacy",
         "retrieval_date": "2026-07-22", "effective_date": "2026-01-01",
         "raw_file": "example_service_2026.txt",
         "notes": "EXAMPLE ROW - replace with real policies"},
    ], columns=MANIFEST_COLUMNS)
    example.to_csv(MANIFEST, index=False)
    # a clearly-synthetic example raw file so the pipeline is runnable end-to-end
    (RAW / "example_service_2026.txt").write_text(
        "Your Privacy Choices\n\n"
        "We collect the information you provide when you create an account, "
        "including your name and email address, and we automatically collect "
        "device identifiers and cookies when you use the service.\n\n"
        "We may share your personal information with third-party advertising "
        "partners to personalize the ads you see.\n\n"
        "You can access, correct, or delete your personal data at any time in "
        "your account settings.\n\n"
        "We retain your information for as long as your account is active.\n\n"
        "We may update this policy and will post material changes on this page.\n",
        encoding="utf-8")
    print(f"Scaffolded {MANIFEST} with an EXAMPLE row and raw file.")
    print("Replace the example with real policies, then re-run.")


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    if not MANIFEST.exists():
        _scaffold_manifest()
        return

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
        "Next: distribute annotation_template.csv per the labeling guide "
        "(calibration round, 10-20% double-labeled overlap, then split), "
        "collect into annotations_raw.csv, and run "
        "src/merge_generalization_labels.py.",
    ])
    (GEN / "build_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
