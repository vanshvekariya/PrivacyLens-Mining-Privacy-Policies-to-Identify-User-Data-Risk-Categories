"""Collect fixed modern-policy excerpts and write the provenance manifest.

The source pages are official privacy-policy pages.  For each policy, the
collector extracts substantive headings/paragraphs/list items and selects a
deterministic, document-spanning sample.  The resulting raw excerpt files are
then processed by ``build_generalization_set.py`` using the same segmentation
logic as the prototype.

Run from the repository root:

    python src/collect_generalization_policies.py
    python src/build_generalization_set.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "data" / "generalization"
RAW = GEN / "raw"
MANIFEST = GEN / "manifest.csv"
SEGMENTS_PER_POLICY = 16

SOURCES = [
    {
        "policy_id": "reddit_2026",
        "service": "Reddit",
        "sector": "social",
        "shift_type": "temporal",
        "source_url": (
            "https://www.reddit.com/policies/privacy-policy"
            "?builder=true&builder_id=5e7f1e7914a309eaa3b59957ce53641c"
        ),
        "fallback_url": (
            "https://web.archive.org/web/20260702000000id_/"
            "https://www.reddit.com/en-us/policies/privacy-policy"
        ),
        "effective_date": "2026-07-01",
    },
    {
        "policy_id": "openai_2026",
        "service": "OpenAI",
        "sector": "artificial-intelligence",
        "shift_type": "service",
        "source_url": "https://openai.com/policies/privacy-policy/",
        "effective_date": "2026-05-18",
    },
    {
        "policy_id": "tiktok_2026",
        "service": "TikTok",
        "sector": "social",
        "shift_type": "service",
        "source_url": "https://t.tiktok.com/legal/page/us/privacy-policy/en?lang=en",
        "fallback_url": (
            "https://web.archive.org/web/20260716000000id_/"
            "https://www.tiktok.com/legal/page/us/privacy-policy/en"
        ),
        "effective_date": "2026-07-15",
    },
    {
        "policy_id": "github_2026",
        "service": "GitHub",
        "sector": "software-development",
        "shift_type": "service",
        "source_url": (
            "https://docs.github.com/en/site-policy/privacy-policies/"
            "github-general-privacy-statement"
        ),
        "effective_date": "2026-04-27",
    },
    {
        "policy_id": "firefox_2026",
        "service": "Firefox",
        "sector": "web-browser",
        "shift_type": "service",
        "source_url": "https://www.mozilla.org/en-US/privacy/firefox/",
        "effective_date": "2026-05-04",
    },
    {
        "policy_id": "spotify_2026",
        "service": "Spotify",
        "sector": "streaming",
        "shift_type": "service",
        "source_url": "https://www.spotify.com/us/legal/privacy-policy/",
        "effective_date": "2026-04-13",
    },
    {
        "policy_id": "dropbox_2025",
        "service": "Dropbox",
        "sector": "cloud-storage",
        "shift_type": "service",
        "source_url": "https://www.dropbox.com/privacy",
        "effective_date": "2025-05-30",
    },
    {
        "policy_id": "shopify_2026",
        "service": "Shopify",
        "sector": "commerce",
        "shift_type": "service",
        "source_url": "https://www.shopify.com/legal/privacy/consumers",
        "effective_date": "2026-03-02",
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/126 Safari/537.36"
    )
}


def extract_blocks(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "form", "nav", "footer"]):
        tag.decompose()
    container = soup.find("article") or soup.find("main") or soup.body or soup
    blocks, seen = [], set()
    for tag in container.find_all(["h1", "h2", "h3", "p", "li"]):
        text = " ".join(tag.get_text(" ", strip=True).split())
        key = text.casefold()
        if len(text) < 40 or len(text) > 2200 or key in seen:
            continue
        if text.lower().startswith(("skip to ", "select language", "copy as ")):
            continue
        seen.add(key)
        blocks.append(text)
    return blocks


def document_spanning_sample(blocks: list[str], n: int) -> list[str]:
    if len(blocks) <= n:
        return blocks
    # Evenly sample the full page, always retaining the first and last block.
    positions = [round(i * (len(blocks) - 1) / (n - 1)) for i in range(n)]
    return [blocks[i] for i in positions]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in SOURCES:
        response = requests.get(source["source_url"], headers=HEADERS, timeout=45)
        response.raise_for_status()
        blocks = extract_blocks(response.content.decode("utf-8", errors="replace"))
        fetched_url = source["source_url"]
        if len(blocks) < SEGMENTS_PER_POLICY and source.get("fallback_url"):
            response = requests.get(
                source["fallback_url"], headers=HEADERS, timeout=60
            )
            response.raise_for_status()
            blocks = extract_blocks(
                response.content.decode("utf-8", errors="replace")
            )
            fetched_url = source["fallback_url"]
        if len(blocks) < SEGMENTS_PER_POLICY:
            raise ValueError(
                f"{source['service']} yielded only {len(blocks)} substantive "
                "blocks; inspect the page extractor before proceeding"
            )
        selected = document_spanning_sample(blocks, SEGMENTS_PER_POLICY)
        raw_file = f"{source['policy_id']}.txt"
        (RAW / raw_file).write_text(
            "\n\n".join(selected) + "\n", encoding="utf-8"
        )
        rows.append({
            **{k: v for k, v in source.items() if k != "fallback_url"},
            "source_url": fetched_url,
            "retrieval_date": date.today().isoformat(),
            "raw_file": raw_file,
            "notes": (
                f"Deterministic document-spanning excerpt: "
                f"{len(selected)} of {len(blocks)} substantive blocks."
            ),
        })
        print(f"{source['service']}: selected {len(selected)}/{len(blocks)} blocks")

    columns = [
        "policy_id", "service", "sector", "shift_type", "source_url",
        "retrieval_date", "effective_date", "raw_file", "notes",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(MANIFEST, index=False)
    print(f"Wrote {MANIFEST} and {len(rows)} raw policy excerpts.")


if __name__ == "__main__":
    main()
