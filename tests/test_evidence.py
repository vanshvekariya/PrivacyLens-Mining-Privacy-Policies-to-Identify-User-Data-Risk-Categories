"""Tests for local n-gram evidence extraction (privacylens.evidence).

These build a tiny in-memory TF-IDF + one-vs-rest Logistic Regression model so
the tests are fast and deterministic and do not depend on the full corpus.
"""

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from privacylens import evidence
from privacylens.evidence import (
    binary_estimator_for,
    evidence_by_category,
    top_evidence,
)

CATS = ["Data Collection", "Third-Party Sharing", "User Control & Deletion"]

TRAIN = [
    ("we collect the information you provide to us", ["Data Collection"]),
    ("we automatically collect device information", ["Data Collection"]),
    ("we may share your information with third party partners",
     ["Third-Party Sharing"]),
    ("we disclose personal information to third party advertisers",
     ["Third-Party Sharing"]),
    ("you can delete your account and opt out at any time",
     ["User Control & Deletion"]),
    ("you may access update or delete your personal information",
     ["User Control & Deletion"]),
    ("we collect and share your information with third party vendors",
     ["Data Collection", "Third-Party Sharing"]),
]


def _fit(category_order=CATS):
    texts = [t for t, _ in TRAIN]
    labels = [ls for _, ls in TRAIN]
    mlb = MultiLabelBinarizer(classes=category_order)
    Y = mlb.fit_transform(labels)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(texts)
    ovr = OneVsRestClassifier(
        LogisticRegression(class_weight="balanced", max_iter=2000)).fit(X, Y)
    return vec, ovr, list(mlb.classes_)


def test_spans_are_substrings_at_reported_offsets():
    vec, ovr, order = _fit()
    text = "We may SHARE your information with a Third-Party partner."
    res = top_evidence(vec, ovr, text, "Third-Party Sharing",
                       category_order=order)
    assert res["status"] == "ok"
    assert res["spans"], "expected at least one evidence span"
    for s in res["spans"]:
        # offsets must index the exact displayed text
        assert text[s["start"]:s["end"]] == s["text"]
        assert s["contribution"] > 0
        assert abs(s["contribution"]
                   - s["tfidf_value"] * s["coefficient"]) < 1e-9


def test_original_capitalization_preserved():
    vec, ovr, order = _fit()
    text = "We SHARE data with a Third-Party."
    res = top_evidence(vec, ovr, text, "Third-Party Sharing",
                       category_order=order)
    # matched text keeps the segment's original casing, not the lowercase
    # feature string.
    assert any(sp["text"] != sp["text"].lower() for sp in res["spans"])


def test_bigram_matches_across_punctuation_and_whitespace():
    vec, ovr, order = _fit()
    text = "shared with third-party   partners and third  party vendors"
    res = top_evidence(vec, ovr, text, "Third-Party Sharing",
                       category_order=order)
    joined = " ".join(sp["feature"] for sp in res["spans"])
    # the 'third party' bigram feature should locate despite hyphen / spaces
    assert "third party" in joined


def test_spans_do_not_overlap():
    vec, ovr, order = _fit()
    text = "we collect and share your information with third party vendors"
    res = evidence_by_category(vec, ovr, text,
                               ["Data Collection", "Third-Party Sharing"],
                               k=5, category_order=order)
    for cat_res in res.values():
        spans = cat_res["spans"]
        for a, b in zip(spans, spans[1:]):
            assert a["end"] <= b["start"], "spans must be non-overlapping"


def test_spans_ranked_by_contribution_and_capped():
    vec, ovr, order = _fit()
    text = "we collect and share your information with third party vendors"
    res = top_evidence(vec, ovr, text, "Third-Party Sharing", k=2,
                       category_order=order)
    assert len(res["spans"]) <= 2


def test_no_positive_evidence_fallback_is_honest():
    vec, ovr, order = _fit()
    # text with no meaningful signal for the queried category
    res = top_evidence(vec, ovr, "the the the the", "Third-Party Sharing",
                       category_order=order)
    assert res["status"] in ("no_positive_evidence", "no_locatable_evidence")
    assert res["spans"] == []
    assert res["phrases"] == []


def test_evidence_survives_category_order_permutation():
    """Evidence must come from the correct binary classifier regardless of the
    column order the model was trained with."""
    text = "we may share your information with third party partners"

    vec_a, ovr_a, order_a = _fit(CATS)
    res_a = top_evidence(vec_a, ovr_a, text, "Third-Party Sharing",
                         category_order=order_a)

    permuted = ["User Control & Deletion", "Third-Party Sharing",
                "Data Collection"]
    vec_b, ovr_b, order_b = _fit(permuted)
    res_b = top_evidence(vec_b, ovr_b, text, "Third-Party Sharing",
                         category_order=order_b)

    feats_a = {sp["feature"] for sp in res_a["spans"]}
    feats_b = {sp["feature"] for sp in res_b["spans"]}
    assert feats_a == feats_b and feats_a, (
        "same category must yield the same evidence under a column permutation")


def test_binary_estimator_length_mismatch_raises():
    vec, ovr, order = _fit()
    with pytest.raises(ValueError):
        binary_estimator_for(ovr, "Data Collection",
                             category_order=order + ["Data Security"])


def test_disclaimer_always_present():
    vec, ovr, order = _fit()
    res = top_evidence(vec, ovr, "we share with third party", "Third-Party Sharing",
                       category_order=order)
    assert "do not share" in res["disclaimer"]
