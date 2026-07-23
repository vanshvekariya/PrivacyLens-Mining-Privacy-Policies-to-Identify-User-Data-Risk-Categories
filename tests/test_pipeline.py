"""Tests for the UI-agnostic prototype core (privacylens.pipeline).

Segmentation tests are pure. Analysis tests build a tiny self-contained
7-category model so they do not depend on the persisted artifact or the corpus.
"""

import json

import joblib
import pytest
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from privacylens import pipeline as pl
from privacylens.config import CATEGORIES, FRAMEWORK_VERSION, MODEL_NAME
from privacylens.pipeline import (
    MAX_SEGMENT_WORDS,
    MIN_SEGMENT_WORDS,
    analyze_policy,
    analyze_segment,
    load_prototype_model,
    segment_policy,
)

# --------------------------------------------------------------------------
# Synthetic training data covering all 7 user-facing categories.
# --------------------------------------------------------------------------
_TRAIN = [
    ("we collect the information you provide and device data",
     ["Data Collection"]),
    ("we automatically collect cookies and log your ip address",
     ["Data Collection"]),
    ("we may share your information with third party advertisers",
     ["Third-Party Sharing"]),
    ("we disclose personal data to third party partners", ["Third-Party Sharing"]),
    ("you can access update or delete your account information",
     ["User Control & Deletion"]),
    ("you may opt out and control your privacy choices",
     ["User Control & Deletion"]),
    ("we retain your data for as long as necessary", ["Data Retention"]),
    ("data is stored and kept for a retention period", ["Data Retention"]),
    ("we protect your data using encryption and security measures",
     ["Data Security"]),
    ("no method of storage is completely secure", ["Data Security"]),
    ("we may update this policy and notify you of changes", ["Policy Change"]),
    ("changes to this policy take effect on the effective date",
     ["Policy Change"]),
    ("please contact us with questions about california residents",
     ["Other/Unclear"]),
    ("do not track signals and international audiences", ["Other/Unclear"]),
]


def _tiny_pipeline():
    texts = [t for t, _ in _TRAIN]
    labels = [ls for _, ls in _TRAIN]
    mlb = MultiLabelBinarizer(classes=CATEGORIES)
    Y = mlb.fit_transform(labels)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("classifier", OneVsRestClassifier(
            LogisticRegression(class_weight="balanced", max_iter=2000))),
    ])
    pipe.fit(texts, Y)
    return pipe


@pytest.fixture(scope="module")
def model():
    pipe = _tiny_pipeline()
    return {
        "pipeline": pipe,
        "vectorizer": pipe.named_steps["tfidf"],
        "classifier": pipe.named_steps["classifier"],
        "thresholds": {c: 0.5 for c in CATEGORIES},
        "metadata": {"model_name": MODEL_NAME},
    }


# ==========================================================================
# Segmentation
# ==========================================================================
def test_empty_input_returns_no_segments():
    assert segment_policy("") == []
    assert segment_policy("   \n\n  ") == []


def test_blank_line_split():
    segs = segment_policy(
        "First paragraph about data collection here.\n\n"
        "Second paragraph about third party sharing here.")
    assert len(segs) == 2
    assert [s["segment_index"] for s in segs] == [0, 1]
    assert all(s["segment_id"].startswith("seg-") for s in segs)


def test_near_empty_fragments_dropped():
    segs = segment_policy(
        "ok\n\nThis paragraph is clearly long enough to be kept as content.")
    # "ok" is a 1-word non-heading fragment -> dropped by MIN_SEGMENT_WORDS
    assert len(segs) == 1
    assert all(s["word_count"] >= MIN_SEGMENT_WORDS for s in segs)


def test_heading_attached_to_following_paragraph():
    segs = segment_policy(
        "Your Choices\n\n"
        "You can opt out of marketing emails at any time using the link.")
    assert len(segs) == 1
    assert "Your Choices" in segs[0]["text"]
    assert "opt out" in segs[0]["text"]


def test_long_paragraph_split_by_sentences_respects_max():
    sentence = "We collect and process your personal information for services. "
    para = sentence * 60           # ~540 words, far above MAX
    segs = segment_policy(para)
    assert len(segs) > 1
    assert all(s["word_count"] <= MAX_SEGMENT_WORDS for s in segs)


def test_giant_single_sentence_is_word_chunked():
    para = "data " * 500           # one 500-word "sentence", no boundaries
    segs = segment_policy(para)
    assert len(segs) > 1
    assert all(s["word_count"] <= MAX_SEGMENT_WORDS for s in segs)


# ==========================================================================
# Analysis schema
# ==========================================================================
def test_analyze_segment_full_schema(model):
    r = analyze_segment("we may share your information with third party partners",
                        model, {"segment_id": "seg-0000", "segment_index": 0,
                                "paragraph_index": 0, "word_count": 9})
    for key in ("text", "probabilities", "thresholds", "predicted_categories",
                "near_threshold_categories", "high_attention_near_misses",
                "abstained", "attention_profile", "priority",
                "evidence_by_category", "explanations", "model_name",
                "framework_version"):
        assert key in r, f"missing {key}"
    assert set(r["probabilities"]) == set(CATEGORIES)
    assert r["framework_version"] == FRAMEWORK_VERSION
    assert "Third-Party Sharing" in r["predicted_categories"]


def test_analyze_policy_document_order_and_summary(model):
    policy = ("We collect the information you provide when you register.\n\n"
              "We may share your data with third party advertisers.\n\n"
              "You can delete your account at any time.")
    out = analyze_policy(policy, model)
    assert [s["segment_index"] for s in out["segments"]] == [0, 1, 2]
    summ = out["summary"]
    assert summ["n_segments"] == 3
    assert sum(summ["stars"].values()) == 3


def test_predicted_categories_have_explanations(model):
    r = analyze_segment("we protect your data using encryption", model)
    cats = {e["category"] for e in r["explanations"]}
    assert set(r["predicted_categories"]).issubset(cats)


# ==========================================================================
# Loader validation
# ==========================================================================
def _write_thresholds(path):
    path.write_text(json.dumps({
        "framework_version": FRAMEWORK_VERSION,
        "model_name": MODEL_NAME,
        "thresholds": {c: 0.5 for c in CATEGORIES},
    }), encoding="utf-8")


def test_loader_rejects_wrong_framework_version(tmp_path):
    model_path = tmp_path / "m.joblib"
    meta_path = tmp_path / "meta.json"
    thr_path = tmp_path / "thr.json"
    joblib.dump(_tiny_pipeline(), model_path)
    meta_path.write_text(json.dumps({
        "framework_version": "0.0-wrong", "model_name": MODEL_NAME,
        "category_order": CATEGORIES}), encoding="utf-8")
    _write_thresholds(thr_path)
    with pytest.raises(ValueError):
        load_prototype_model(model_path, meta_path, thr_path)


def test_loader_happy_path(tmp_path):
    model_path = tmp_path / "m.joblib"
    meta_path = tmp_path / "meta.json"
    thr_path = tmp_path / "thr.json"
    joblib.dump(_tiny_pipeline(), model_path)
    meta_path.write_text(json.dumps({
        "framework_version": FRAMEWORK_VERSION, "model_name": MODEL_NAME,
        "category_order": CATEGORIES}), encoding="utf-8")
    _write_thresholds(thr_path)
    bundle = load_prototype_model(model_path, meta_path, thr_path)
    assert set(bundle["thresholds"]) == set(CATEGORIES)
    assert len(bundle["classifier"].estimators_) == len(CATEGORIES)
