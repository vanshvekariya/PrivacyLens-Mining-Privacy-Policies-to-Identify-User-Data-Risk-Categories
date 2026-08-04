"""Apply the frozen v1.1 category guide to the fixed modern-policy set.

The labels below correspond to the segment identifiers produced by
``collect_generalization_policies.py`` and ``build_generalization_set.py``.
The script validates that the snapshot contains exactly the reviewed segment
set before writing ``annotations_raw.csv``; this prevents labels from silently
drifting onto changed source text.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "data" / "generalization"
SEGMENTS = GEN / "segments_unlabeled.csv"
MANIFEST = GEN / "manifest_resolved.csv"
OUT = GEN / "annotations_raw.csv"

EXPECTED_HASHES = {
    "reddit_2026": "f55042d5a9bdd849a5d98a775f7d3ce747c0578d850b83d655bc19008df1776d",
    "openai_2026": "3c03894303615dd23e6b4e6fea9e8833c3552da5481949eb0bc177627736ed36",
    "tiktok_2026": "a3d2baa847a48af5e5472bcf8340d7b43e047cec3bba96131115c999fda9628a",
    "github_2026": "bc16563ad097d14e48b5f9676ed9c00f901397558d82a7c9d7b7813cbeb32112",
    "firefox_2026": "764904ced379298026756f3baafabbcadfe18b5190df8499c84f108909262e73",
    "spotify_2026": "4a69784f1668a9a08a42883e7507cec8c548349a66ebe2d8bbb5d12f5afe8b23",
    "dropbox_2025": "1cdea56d916f7da6fcda5e0c34cdb6315ba4f0feead3bb21395bf843b88c1eab",
    "shopify_2026": "0389fe313b83f6436af64b9a16c069320614566e087d9b9e721a0d7430cf993f",
}

L = {
    # Reddit
    "reddit_2026-seg-0000": "Policy Change",
    "reddit_2026-seg-0001": "Other/Unclear",
    "reddit_2026-seg-0002": "Data Collection",
    "reddit_2026-seg-0003": "Data Collection",
    "reddit_2026-seg-0004": "Data Collection",
    "reddit_2026-seg-0005": "Third-Party Sharing",
    "reddit_2026-seg-0006": "User Control & Deletion",
    "reddit_2026-seg-0007": "Other/Unclear",
    "reddit_2026-seg-0008": "Data Collection",
    "reddit_2026-seg-0009": "Data Security",
    "reddit_2026-seg-0010": "Data Collection",
    "reddit_2026-seg-0011": "Data Collection",
    "reddit_2026-seg-0012": "User Control & Deletion",
    "reddit_2026-seg-0013": "Data Retention|Data Security|Third-Party Sharing",
    "reddit_2026-seg-0014": "Other/Unclear",
    # OpenAI
    "openai_2026-seg-0000": "Other/Unclear",
    "openai_2026-seg-0001": "Data Collection",
    "openai_2026-seg-0002": "Data Collection|Third-Party Sharing",
    "openai_2026-seg-0003": "Data Collection",
    "openai_2026-seg-0004": "Data Collection",
    "openai_2026-seg-0005": "Data Collection",
    "openai_2026-seg-0006": "Third-Party Sharing",
    "openai_2026-seg-0007": "Third-Party Sharing|User Control & Deletion",
    "openai_2026-seg-0008": "Third-Party Sharing|User Control & Deletion",
    "openai_2026-seg-0009": "Data Retention|Data Security",
    "openai_2026-seg-0010": "Data Retention|Data Security|User Control & Deletion",
    "openai_2026-seg-0011": "Data Retention",
    "openai_2026-seg-0012": "User Control & Deletion",
    "openai_2026-seg-0013": "Data Collection|User Control & Deletion",
    "openai_2026-seg-0014": "User Control & Deletion",
    "openai_2026-seg-0015": "User Control & Deletion",
    "openai_2026-seg-0016": "Other/Unclear",
    "openai_2026-seg-0017": "Other/Unclear",
    # TikTok
    "tiktok_2026-seg-0000": "Other/Unclear",
    "tiktok_2026-seg-0001": "Data Collection",
    "tiktok_2026-seg-0002": "Data Collection",
    "tiktok_2026-seg-0003": "Data Collection",
    "tiktok_2026-seg-0004": "Data Collection",
    "tiktok_2026-seg-0005": "Data Collection|Third-Party Sharing",
    "tiktok_2026-seg-0006": "Data Collection",
    "tiktok_2026-seg-0007": "Data Security",
    "tiktok_2026-seg-0008": "Data Collection",
    "tiktok_2026-seg-0009": "Third-Party Sharing",
    "tiktok_2026-seg-0010": "Third-Party Sharing",
    "tiktok_2026-seg-0011": "Third-Party Sharing|User Control & Deletion",
    "tiktok_2026-seg-0012": "Other/Unclear|User Control & Deletion",
    "tiktok_2026-seg-0013": "Other/Unclear|Third-Party Sharing",
    "tiktok_2026-seg-0014": "Third-Party Sharing|User Control & Deletion",
    "tiktok_2026-seg-0015": "Other/Unclear",
    # GitHub
    "github_2026-seg-0000": "Other/Unclear",
    "github_2026-seg-0001": "Data Collection",
    "github_2026-seg-0002": "Data Collection|Third-Party Sharing|User Control & Deletion",
    "github_2026-seg-0003": "Data Collection|Data Security",
    "github_2026-seg-0004": "Data Security|Third-Party Sharing",
    "github_2026-seg-0005": "Other/Unclear",
    "github_2026-seg-0006": "User Control & Deletion",
    "github_2026-seg-0007": "Other/Unclear",
    "github_2026-seg-0008": "Other/Unclear",
    "github_2026-seg-0009": "Data Collection|Third-Party Sharing",
    "github_2026-seg-0010": "User Control & Deletion",
    "github_2026-seg-0011": "Third-Party Sharing",
    "github_2026-seg-0012": "Other/Unclear",
    # Firefox
    "firefox_2026-seg-0000": "Other/Unclear",
    "firefox_2026-seg-0001": "Data Collection|Data Retention",
    "firefox_2026-seg-0002": "User Control & Deletion",
    "firefox_2026-seg-0003": "Data Collection",
    "firefox_2026-seg-0004": "Third-Party Sharing|User Control & Deletion",
    "firefox_2026-seg-0005": "Other/Unclear|User Control & Deletion",
    "firefox_2026-seg-0006": "Data Collection|Data Security|Third-Party Sharing",
    "firefox_2026-seg-0007": "Data Collection|Third-Party Sharing|User Control & Deletion",
    "firefox_2026-seg-0008": "Other/Unclear",
    "firefox_2026-seg-0009": "Data Security|Other/Unclear|Third-Party Sharing",
    "firefox_2026-seg-0010": "User Control & Deletion",
    "firefox_2026-seg-0011": "Data Collection|User Control & Deletion",
    "firefox_2026-seg-0012": "Policy Change",
    "firefox_2026-seg-0013": "Third-Party Sharing|User Control & Deletion",
    "firefox_2026-seg-0014": "Data Collection",
    "firefox_2026-seg-0015": "Other/Unclear",
    # Spotify
    "spotify_2026-seg-0000": "User Control & Deletion",
    "spotify_2026-seg-0001": "User Control & Deletion",
    "spotify_2026-seg-0002": "User Control & Deletion",
    "spotify_2026-seg-0003": "Data Collection",
    "spotify_2026-seg-0004": "Data Collection|Third-Party Sharing",
    "spotify_2026-seg-0005": "Data Collection",
    "spotify_2026-seg-0006": "Third-Party Sharing",
    "spotify_2026-seg-0007": "Other/Unclear",
    "spotify_2026-seg-0008": "Data Collection",
    "spotify_2026-seg-0009": "Third-Party Sharing",
    "spotify_2026-seg-0010": "Data Security|Third-Party Sharing",
    "spotify_2026-seg-0011": "Third-Party Sharing",
    "spotify_2026-seg-0012": "Data Retention",
    # Dropbox
    "dropbox_2025-seg-0000": "Other/Unclear",
    "dropbox_2025-seg-0001": "Data Collection",
    "dropbox_2025-seg-0002": "Data Collection|Data Retention",
    "dropbox_2025-seg-0003": "Data Collection|User Control & Deletion",
    "dropbox_2025-seg-0004": "User Control & Deletion",
    "dropbox_2025-seg-0005": "Data Collection|User Control & Deletion",
    "dropbox_2025-seg-0006": "Third-Party Sharing",
    "dropbox_2025-seg-0007": "Third-Party Sharing|User Control & Deletion",
    "dropbox_2025-seg-0008": "Data Security",
    "dropbox_2025-seg-0009": "Data Security",
    "dropbox_2025-seg-0010": "Third-Party Sharing",
    "dropbox_2025-seg-0011": "Data Security|Other/Unclear|Third-Party Sharing",
    "dropbox_2025-seg-0012": "User Control & Deletion",
    "dropbox_2025-seg-0013": "User Control & Deletion",
    "dropbox_2025-seg-0014": "Other/Unclear",
    "dropbox_2025-seg-0015": "Other/Unclear",
    # Shopify
    "shopify_2026-seg-0000": "Other/Unclear",
    "shopify_2026-seg-0001": "Data Collection",
    "shopify_2026-seg-0002": "Data Collection|Third-Party Sharing",
    "shopify_2026-seg-0003": "Data Collection",
    "shopify_2026-seg-0004": "User Control & Deletion",
    "shopify_2026-seg-0005": "Data Collection|Policy Change",
    "shopify_2026-seg-0006": "Data Collection|Third-Party Sharing",
    "shopify_2026-seg-0007": "Third-Party Sharing",
    "shopify_2026-seg-0008": "Third-Party Sharing|User Control & Deletion",
    "shopify_2026-seg-0009": "User Control & Deletion",
    "shopify_2026-seg-0010": "Third-Party Sharing|User Control & Deletion",
    "shopify_2026-seg-0011": "User Control & Deletion",
    "shopify_2026-seg-0012": "Data Retention",
    "shopify_2026-seg-0013": "User Control & Deletion",
    "shopify_2026-seg-0014": "Data Collection",
    "shopify_2026-seg-0015": "User Control & Deletion",
}


def main() -> None:
    segments = pd.read_csv(SEGMENTS, dtype=str).fillna("")
    manifest = pd.read_csv(MANIFEST, dtype=str).fillna("")
    actual_hashes = dict(
        zip(manifest["policy_id"], manifest["content_sha256"])
    )
    if actual_hashes != EXPECTED_HASHES:
        raise ValueError(
            "Policy snapshot hash mismatch. Re-review labels before evaluating "
            "changed source text."
        )
    actual = set(segments["segment_id"])
    expected = set(L)
    if actual != expected:
        raise ValueError(
            "Reviewed-label snapshot mismatch. "
            f"missing labels={sorted(actual - expected)}; "
            f"stale labels={sorted(expected - actual)}"
        )
    rows = [
        {
            "segment_id": segment_id,
            "annotator": "primary_review",
            "labels": L[segment_id],
            "notes": "",
            "guide_version": "1.1",
        }
        for segment_id in segments["segment_id"]
    ]
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {len(rows)} reviewed labels to {OUT}")


if __name__ == "__main__":
    main()
