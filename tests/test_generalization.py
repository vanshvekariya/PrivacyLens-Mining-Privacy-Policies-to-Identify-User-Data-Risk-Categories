"""Tests for the generalization-study scripts (validation, agreement,
adjudication, and HTML stripping). These exercise the pure functions; the full
evaluation is an integration script covered by manual runs."""

import pandas as pd
import pytest

from build_generalization_set import _strip_html
from merge_generalization_labels import (
    _parse_labels,
    adjudicate,
    agreement_by_category,
    disagreements,
)


def _df(rows):
    """rows: list of (segment_id, annotator, label_set)."""
    return pd.DataFrame([{"segment_id": s, "annotator": a, "label_set": ls}
                         for s, a, ls in rows])


# ---- label validation -----------------------------------------------------
def test_parse_labels_canonicalizes():
    assert _parse_labels("Third Party Sharing|data collection") == {
        "Third-Party Sharing", "Data Collection"}


def test_parse_labels_rejects_unknown():
    with pytest.raises(ValueError):
        _parse_labels("Nonexistent Category")


def test_parse_labels_empty():
    assert _parse_labels("") == set()
    assert _parse_labels(None) == set()


# ---- agreement -------------------------------------------------------------
def test_agreement_perfect_and_disagreement():
    df = _df([
        ("s1", "Vansh", {"Data Collection"}),
        ("s1", "Lakshita", {"Data Collection"}),
        ("s2", "Vansh", {"Third-Party Sharing"}),
        ("s2", "Lakshita", {"Third-Party Sharing", "Other/Unclear"}),
    ])
    agree, overlap = agreement_by_category(df)
    assert set(overlap) == {"s1", "s2"}
    row = agree.set_index("category")
    # Data Collection: both positive on s1, both negative on s2 -> perfect
    assert row.loc["Data Collection", "positive_agreement"] == 1.0
    # Other/Unclear: one positive, one negative on s2 -> disagreement
    assert row.loc["Other/Unclear", "positive_agreement"] == 0.0


def test_disagreements_lists_disputed_category():
    df = _df([
        ("s2", "Vansh", {"Third-Party Sharing"}),
        ("s2", "Lakshita", {"Third-Party Sharing", "Other/Unclear"}),
    ])
    dis = disagreements(df, ["s2"])
    assert list(dis["category"]) == ["Other/Unclear"]
    assert dis.iloc[0]["annotators_positive"] == "Lakshita"


# ---- adjudication ----------------------------------------------------------
def test_adjudicate_single_and_agreed_and_unresolved():
    df = _df([
        ("single", "Vansh", {"Data Retention"}),
        ("agree", "Vansh", {"Data Security"}),
        ("agree", "Lakshita", {"Data Security"}),
        ("conflict", "Vansh", {"Policy Change"}),
        ("conflict", "Lakshita", {"Policy Change", "Other/Unclear"}),
    ])
    adj = adjudicate(df, ["agree", "conflict"]).set_index("segment_id")
    assert adj.loc["single", "source"] == "single"
    assert adj.loc["single", "needs_adjudication"] == False  # noqa: E712
    assert adj.loc["agree", "source"] == "agreed"
    assert adj.loc["conflict", "needs_adjudication"] == True  # noqa: E712
    # unresolved keeps a best-effort union
    assert "Other/Unclear" in adj.loc["conflict", "labels"]


# ---- html stripping --------------------------------------------------------
def test_strip_html():
    assert "<p>" not in _strip_html("<p>We collect <b>data</b>.</p>")
    assert "collect" in _strip_html("<p>We collect data.</p>")
    # plain text is untouched
    assert _strip_html("no markup here") == "no markup here"
