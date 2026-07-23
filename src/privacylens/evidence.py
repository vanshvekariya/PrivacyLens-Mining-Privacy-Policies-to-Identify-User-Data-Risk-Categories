"""Local n-gram evidence for a linear model's per-category prediction.

Evidence is ranked by **local contribution** (``tfidf_value * coefficient``)
for the specific segment being explained, not by global coefficient magnitude.
A feature can carry a large model coefficient yet a tiny TF-IDF value in this
segment; ranking by the product lets us truthfully say

    "these phrases pushed THIS prediction up"

rather than the weaker

    "these are globally important phrases for this category".

Design points (all deliberate):

  - **Structured, character-offset spans.** Each span records its position in
    the segment text, the underlying TF-IDF feature, and the three numbers
    behind it (tfidf_value, coefficient, contribution), so the presentation/UI
    layer never has to rediscover phrase positions.
  - **Offsets refer to the cleaned text the model actually classified.** The
    prototype classifies and highlights the *identical* string, so offsets
    always align (no raw->cleaned mapping needed). Callers must pass that same
    string here.
  - **Safe class lookup.** With a multilabel ``OneVsRestClassifier`` fitted on
    an indicator matrix, ``ovr.classes_`` is ``[0, 1, ...]`` (column indices),
    NOT category names. The category -> binary-estimator mapping therefore goes
    through an explicit ``category_order`` (the MultiLabelBinarizer order,
    i.e. ``config.CATEGORIES``), with a length check.
  - **Overlap resolution.** Overlapping unigram/bigram evidence
    ("share" / "share information") is resolved to the highest-contribution
    non-overlapping spans so highlighting does not look cluttered.
  - **Honest fallback.** If no feature has positive contribution, we return an
    empty span list with an explicit status rather than inventing evidence.

HTML escaping is intentionally NOT done here; it belongs in the
presentation/UI layer so exports stay readable and nothing is double-escaped.
"""

import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .config import CATEGORIES, canonicalize_category
from .templates import DISCLAIMER

# Default number of highlighted spans (matches templates.MAX_EVIDENCE_ITEMS).
DEFAULT_K = 3


def _is_stopword_unigram(feature):
    """A single token that is an English stopword carries no interpretable
    signal to a reader even if it has a small positive contribution. We drop
    only PURE stopword unigrams; bigrams (even those containing a stopword,
    e.g. "share your") and content unigrams are always kept, so this trims
    noise like "and"/"the"/"can" without inventing or hiding real evidence."""
    return " " not in feature and feature.lower() in ENGLISH_STOP_WORDS


def binary_estimator_for(ovr, category, category_order=CATEGORIES):
    """Return (estimator, column_index) for one category's OvR binary model.

    ``category_order`` is the source of truth for column alignment: column j of
    the indicator matrix (and therefore ``ovr.estimators_[j]``) corresponds to
    ``category_order[j]``. We validate the estimator count against it instead
    of trusting positional assumptions.
    """
    c = canonicalize_category(category)
    order = [canonicalize_category(x) for x in category_order]
    if len(ovr.estimators_) != len(order):
        raise ValueError(
            f"estimator count {len(ovr.estimators_)} != category_order length "
            f"{len(order)}; cannot align categories to binary classifiers")
    idx = order.index(c)
    return ovr.estimators_[idx], idx


def _feature_contributions(vectorizer, estimator, text):
    """Positive local contributions for the features present in ``text``.

    Returns a list of dicts sorted by contribution (desc), one per nonzero
    TF-IDF feature whose contribution (tfidf * coef) is strictly positive.
    """
    x = vectorizer.transform([text])          # 1 x V sparse (CSR)
    coef = np.ravel(estimator.coef_)          # (V,)
    names = vectorizer.get_feature_names_out()

    x = x.tocsr()
    contribs = []
    for col, tfidf_val in zip(x.indices, x.data):
        contribution = float(tfidf_val) * float(coef[col])
        if contribution > 0.0:
            contribs.append({
                "feature": str(names[col]),
                "tfidf_value": float(tfidf_val),
                "coefficient": float(coef[col]),
                "contribution": contribution,
            })
    # contribution desc, then feature name for a deterministic tie-break.
    contribs.sort(key=lambda d: (-d["contribution"], d["feature"]))
    return contribs


def _locate(feature, text):
    """Find every occurrence of a TF-IDF ``feature`` in ``text``.

    Features are lowercased, accent-stripped n-grams of ``\\w\\w+`` tokens. We
    match them back tolerantly: case-insensitive, allowing arbitrary
    whitespace/punctuation between tokens (so "third party" also matches
    "Third-Party" and "third  party"), with word boundaries at the ends.

    Returns a list of (start, end) offsets into ``text``.
    """
    tokens = feature.split()
    if not tokens:
        return []
    pattern = r"\b" + r"\W+".join(re.escape(t) for t in tokens) + r"\b"
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        return []
    return [(m.start(), m.end()) for m in rx.finditer(text)]


def _select_non_overlapping(occurrences, k):
    """Greedily keep the highest-contribution non-overlapping spans.

    ``occurrences`` is a list of span dicts with ``start``/``end``/
    ``contribution``. Ties break to earlier start, then longer span, then
    feature name, so selection is fully deterministic.
    """
    ordered = sorted(
        occurrences,
        key=lambda s: (-s["contribution"], s["start"],
                       -(s["end"] - s["start"]), s["feature"]),
    )
    chosen = []
    for span in ordered:
        if any(span["start"] < c["end"] and c["start"] < span["end"]
               for c in chosen):
            continue
        chosen.append(span)
        if len(chosen) >= k:
            break
    # Return in reading order for stable, left-to-right highlighting.
    return sorted(chosen, key=lambda s: s["start"])


def top_evidence(vectorizer, ovr, text, category, k=DEFAULT_K,
                 category_order=CATEGORIES, drop_stopword_unigrams=True):
    """Structured, offset-aware evidence spans for one predicted category.

    Parameters
    ----------
    vectorizer : fitted TfidfVectorizer
    ovr : fitted OneVsRestClassifier (linear estimators with ``coef_``)
    text : str
        The exact cleaned text that was (or will be) classified; offsets refer
        to this string.
    category : str
        Predicted category to explain (canonicalized).
    k : int
        Maximum number of highlighted spans to return.
    category_order : list[str]
        Column order used when the model was trained (``config.CATEGORIES``).

    Returns
    -------
    dict with keys:
      category      canonical category name
      status        "ok" | "no_positive_evidence" | "no_locatable_evidence"
      spans         list of {text, start, end, feature, tfidf_value,
                             coefficient, contribution} (reading order)
      phrases       matched original-case phrases (for templates.explain)
      disclaimer    the standing negation/uncertainty disclaimer
    """
    c = canonicalize_category(category)
    estimator, _ = binary_estimator_for(ovr, c, category_order)

    contribs = _feature_contributions(vectorizer, estimator, text)
    if drop_stopword_unigrams:
        contribs = [con for con in contribs
                    if not _is_stopword_unigram(con["feature"])]
    if not contribs:
        return {"category": c, "status": "no_positive_evidence",
                "spans": [], "phrases": [], "disclaimer": DISCLAIMER}

    occurrences = []
    for con in contribs:
        for start, end in _locate(con["feature"], text):
            occurrences.append({
                "text": text[start:end],       # original capitalization
                "start": start,
                "end": end,
                "feature": con["feature"],
                "tfidf_value": con["tfidf_value"],
                "coefficient": con["coefficient"],
                "contribution": con["contribution"],
            })

    if not occurrences:
        # Positive contributions existed but none could be located in the text
        # (e.g. accent-stripping mismatch). Do not invent spans.
        return {"category": c, "status": "no_locatable_evidence",
                "spans": [], "phrases": [], "disclaimer": DISCLAIMER}

    spans = _select_non_overlapping(occurrences, k)
    phrases, seen = [], set()
    for s in spans:
        key = s["text"].lower()
        if key not in seen:
            seen.add(key)
            phrases.append(s["text"])
    return {"category": c, "status": "ok", "spans": spans,
            "phrases": phrases, "disclaimer": DISCLAIMER}


def evidence_by_category(vectorizer, ovr, text, categories, k=DEFAULT_K,
                         category_order=CATEGORIES,
                         drop_stopword_unigrams=True):
    """Evidence for several predicted categories: {category: top_evidence}."""
    out = {}
    for cat in categories:
        c = canonicalize_category(cat)
        out[c] = top_evidence(vectorizer, ovr, text, c, k=k,
                              category_order=category_order,
                              drop_stopword_unigrams=drop_stopword_unigrams)
    return out
