"""Tests for model-agnostic transformer word-occlusion evidence."""

import numpy as np
import pytest

from privacylens.config import CATEGORIES
from privacylens.transformer_evidence import (
    transformer_evidence_by_category,
    word_occlusion_evidence,
)


def keyword_predictor(texts):
    rows = []
    for text in texts:
        low = text.lower()
        values = np.full(len(CATEGORIES), 0.1, dtype=float)
        values[CATEGORIES.index("Third-Party Sharing")] = (
            0.9 if "share" in low else 0.2
        )
        values[CATEGORIES.index("Data Collection")] = (
            0.8 if "collect" in low else 0.3
        )
        rows.append(values)
    return np.vstack(rows)


def test_word_occlusion_finds_supporting_word_and_offsets():
    text = "We collect account data and share it with partners."
    result = word_occlusion_evidence(
        keyword_predictor, text, "Third-Party Sharing", k=2
    )
    assert result["method"] == "word_occlusion"
    assert result["phrases"] == ["share"]
    item = result["spans"][0]
    assert text[item["start"]:item["end"]] == "share"
    assert item["contribution"] == pytest.approx(0.7)


def test_transformer_evidence_multiple_categories():
    text = "We collect account data and share it."
    result = transformer_evidence_by_category(
        keyword_predictor,
        text,
        ["Data Collection", "Third-Party Sharing"],
    )
    assert result["Data Collection"]["phrases"] == ["collect"]
    assert result["Third-Party Sharing"]["phrases"] == ["share"]


def test_empty_text_and_zero_k():
    empty = word_occlusion_evidence(
        keyword_predictor, "", "Data Collection"
    )
    assert empty["spans"] == []
    zero = word_occlusion_evidence(
        keyword_predictor, "We collect data.", "Data Collection", k=0
    )
    assert zero["spans"] == []


def test_bad_predictor_shape_rejected():
    with pytest.raises(ValueError):
        word_occlusion_evidence(
            lambda texts: np.zeros((len(texts), 2)),
            "We share data.",
            "Third-Party Sharing",
        )
