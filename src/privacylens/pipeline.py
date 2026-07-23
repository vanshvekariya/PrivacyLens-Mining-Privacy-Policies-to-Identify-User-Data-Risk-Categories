"""UI-agnostic prototype core: segment -> classify -> interpret -> explain.

This module deliberately imports NOTHING from the presentation/UI layer so it
can be unit-tested headlessly and reused by any front end (the Streamlit app is
a thin renderer on top of it).

Pipeline for a pasted policy:

  raw text
    -> segment_policy()      length-aware, sentence-boundary segmentation
    -> per segment:
         pipeline.predict_proba  (independent one-vs-rest probabilities)
         decide_labels           predicted / borderline / near-miss / abstain
         attention_profile       dominant tier + valence
         reading_priority        1-5 stars + review flag (two meanings kept)
         evidence.top_evidence   local (tfidf x coef) highlighted spans
         templates.explain_all   cautious per-category explanations

The segment text that is classified is the SAME cleaned string returned to the
caller, so evidence character offsets always align with the displayed text (no
raw->cleaned offset mapping needed).

Model loading validates the persisted model against the threshold file so a
stale or mismatched artifact fails loudly instead of silently misbehaving.
"""

import json
import re

import joblib

from . import templates
from .attention import attention_profile
from .config import (
    CATEGORIES,
    FRAMEWORK_VERSION,
    MODEL_NAME,
    PROTOTYPE_METADATA_JSON,
    PROTOTYPE_MODEL_JOBLIB,
    THRESHOLDS_JSON,
)
from .evidence import evidence_by_category
from .prediction import decide_labels, load_thresholds
from .priority import reading_priority

# --------------------------------------------------------------------------
# Segmentation parameters. Chosen from the OPP-115 training segment-length
# distribution (mean ~70, median ~59, p95 ~163, p99 ~218 words), so modern
# segments are chunked to a range the model actually saw in training. If a
# modern paragraph is far longer than any training segment, degradation could
# otherwise be a segmentation artifact rather than a genuine temporal shift.
# --------------------------------------------------------------------------
MAX_SEGMENT_WORDS = 180
MIN_SEGMENT_WORDS = 4
SENTENCE_OVERLAP = 1
MAX_HEADING_WORDS = 8          # short, punctuation-free line treated as heading

# Guardrail for pasted input (the UI enforces its own upload caps too).
MAX_POLICY_CHARS = 200_000

_PARA_SPLIT = re.compile(r"\n\s*\n+")
_WS = re.compile(r"\s+")
# Sentence boundary: end punctuation followed by space and a capital / opener.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


# ==========================================================================
# Segmentation
# ==========================================================================
def _clean(text):
    return _WS.sub(" ", text).strip()


def _word_count(text):
    return len(text.split())


def _is_heading(text):
    """A short line with no terminal sentence punctuation reads as a heading."""
    return (_word_count(text) <= MAX_HEADING_WORDS
            and not text.rstrip().endswith((".", "!", "?", ":", ";")))


def _sentences(text):
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts or [text]


def _split_long_text(text, max_words=MAX_SEGMENT_WORDS,
                     overlap=SENTENCE_OVERLAP):
    """Split an over-long block at sentence boundaries with small overlap.

    A single sentence longer than ``max_words`` is further chunked by words so
    no output segment exceeds the model's training length range.
    """
    sentences = _sentences(text)
    chunks, cur = [], []
    cur_wc = 0
    for sent in sentences:
        wc = _word_count(sent)
        if wc > max_words:
            # flush what we have, then hard-split the giant sentence by words
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_wc = [], 0
            words = sent.split()
            step = max(1, max_words - overlap)
            for i in range(0, len(words), step):
                chunks.append(" ".join(words[i:i + max_words]))
                if i + max_words >= len(words):
                    break
            continue
        if cur_wc + wc > max_words and cur:
            chunks.append(" ".join(cur))
            # carry the trailing ``overlap`` sentences into the next chunk
            cur = cur[-overlap:] if overlap else []
            cur_wc = sum(_word_count(s) for s in cur)
        cur.append(sent)
        cur_wc += wc
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [text]


def segment_policy(raw_text):
    """Split a pasted policy into model-sized segments.

    Returns a list of dicts (document order):
      {segment_id, segment_index, paragraph_index, text, word_count}

    Rules (in order): split on blank lines; drop empties; attach a heading to
    the paragraph that follows it; split over-long paragraphs at sentence
    boundaries with a small overlap; drop trailing near-empty fragments.
    """
    if not raw_text or not raw_text.strip():
        return []

    paragraphs = [_clean(p) for p in _PARA_SPLIT.split(raw_text)]
    paragraphs = [p for p in paragraphs if p]

    # Attach headings to the following paragraph where practical.
    merged, pending_heading = [], None
    for para in paragraphs:
        if _is_heading(para):
            pending_heading = (pending_heading + " " + para
                               if pending_heading else para)
            continue
        if pending_heading:
            para = pending_heading + " \u2014 " + para
            pending_heading = None
        merged.append(para)
    if pending_heading:                       # trailing heading with no body
        merged.append(pending_heading)

    segments = []
    for p_idx, para in enumerate(merged):
        blocks = ([para] if _word_count(para) <= MAX_SEGMENT_WORDS
                  else _split_long_text(para))
        for block in blocks:
            block = _clean(block)
            if _word_count(block) < MIN_SEGMENT_WORDS:
                continue
            idx = len(segments)
            segments.append({
                "segment_id": f"seg-{idx:04d}",
                "segment_index": idx,
                "paragraph_index": p_idx,
                "text": block,
                "word_count": _word_count(block),
            })
    return segments


# ==========================================================================
# Model loading (fail loudly on a stale / mismatched artifact)
# ==========================================================================
def load_prototype_model(model_path=PROTOTYPE_MODEL_JOBLIB,
                         metadata_path=PROTOTYPE_METADATA_JSON,
                         thresholds_path=THRESHOLDS_JSON):
    """Load the persisted pipeline + thresholds, validating the contract.

    Verifies that: the metadata framework version and model name match the
    current configuration; the persisted category order equals CATEGORIES; the
    classifier has exactly one binary estimator per category; and the threshold
    file (validated by ``load_thresholds``) was generated for this Logistic
    Regression model, this framework version, and has exactly one threshold per
    category.
    """
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    if metadata.get("framework_version") != FRAMEWORK_VERSION:
        raise ValueError(
            f"model metadata framework_version {metadata.get('framework_version')!r}"
            f" != current {FRAMEWORK_VERSION!r}")
    if metadata.get("model_name") != MODEL_NAME:
        raise ValueError(
            f"model metadata model_name {metadata.get('model_name')!r}"
            f" != current {MODEL_NAME!r}")
    if metadata.get("category_order") != CATEGORIES:
        raise ValueError("model metadata category_order does not match "
                         "config.CATEGORIES; retrain the prototype model")

    pipeline = joblib.load(model_path)
    classifier = pipeline.named_steps["classifier"]
    vectorizer = pipeline.named_steps["tfidf"]
    if len(classifier.estimators_) != len(CATEGORIES):
        raise ValueError(
            f"classifier has {len(classifier.estimators_)} estimators for "
            f"{len(CATEGORIES)} categories")

    # load_thresholds enforces: framework version, model_name == LR, and
    # exactly the CATEGORIES set (one threshold per category).
    thresholds = load_thresholds(thresholds_path)

    return {
        "pipeline": pipeline,
        "vectorizer": vectorizer,
        "classifier": classifier,
        "thresholds": thresholds,
        "metadata": metadata,
    }


# ==========================================================================
# Analysis
# ==========================================================================
def _probs_for(pipeline, text):
    """Independent one-vs-rest probabilities as {category: prob}."""
    proba = pipeline.predict_proba([text])[0]
    return {c: float(proba[j]) for j, c in enumerate(CATEGORIES)}


def analyze_segment(text, model, segment_meta=None):
    """Run the full interpretation stack on one segment; return the schema."""
    bundle = model
    probs = _probs_for(bundle["pipeline"], text)
    thresholds = bundle["thresholds"]

    decision = decide_labels(probs, thresholds)
    predicted = decision["predicted_categories"]
    profile = attention_profile(predicted, probs)
    priority = reading_priority(probs, thresholds)

    ev = evidence_by_category(bundle["vectorizer"], bundle["classifier"],
                              text, predicted) if predicted else {}
    evidence_phrases = {c: ev[c]["phrases"] for c in ev}
    explanations = templates.explain_all(predicted, evidence_phrases)

    result = {
        "text": text,
        "probabilities": probs,
        "thresholds": {c: thresholds.get(c) for c in CATEGORIES},
        "predicted_categories": predicted,
        "near_threshold_categories": decision["near_threshold_categories"],
        "high_attention_near_misses": decision["high_attention_near_misses"],
        "abstained": decision["abstained"],
        "attention_profile": profile,
        "priority": priority,
        "evidence_by_category": ev,
        "explanations": explanations,
        "model_name": MODEL_NAME,
        "framework_version": FRAMEWORK_VERSION,
    }
    if segment_meta:
        result.update({k: segment_meta[k] for k in
                       ("segment_id", "segment_index", "paragraph_index",
                        "word_count") if k in segment_meta})
    return result


def _summary(results):
    """Corpus-level counts for an optional 'nutrition label' overview."""
    n = len(results)
    stars = {s: 0 for s in range(1, 6)}
    category_counts = {c: 0 for c in CATEGORIES}
    review = 0
    abstained = 0
    for r in results:
        stars[r["priority"]["stars"]] += 1
        review += int(r["priority"]["review_required"])
        abstained += int(r["abstained"])
        for c in r["predicted_categories"]:
            category_counts[c] += 1
    return {
        "n_segments": n,
        "stars": stars,
        "category_counts": category_counts,
        "review_required": review,
        "abstained": abstained,
    }


def analyze_policy(raw_text, model):
    """Segment and analyze a whole policy.

    Returns a dict with:
      segments   list of segment results in DOCUMENT order (each carries its
                 segment_index, so the UI can also render a triage order by
                 sorting on priority stars without losing document position).
      summary    corpus-level counts (segments, stars, categories, review).
    """
    segments = segment_policy(raw_text)
    results = [analyze_segment(s["text"], model, s) for s in segments]
    return {"segments": results, "summary": _summary(results)}
