# PrivacyLens Attention-Tier, Valence, and Reading-Priority Framework

Framework version: 1.0. This document is the source text for Section 6 of the
final report and the design record for the interpretation layer implemented in
`src/privacylens/`. Machine-readable versions are emitted to
`results/attention_framework.csv` and `results/interpretation_metadata.json`.

## 1. What this layer does (and does not) claim

The classifier assigns each policy segment one or more **topics** (privacy
practice categories). This layer adds three user-facing signals on top:

- an **attention tier** (how much a reader should pay attention to a topic),
- a **default valence** (whether the topic tends to be favourable, harmful, or
  neither), and
- a **1-5 star reading priority** plus a separate **review flag**.

Two honesty boundaries are built into the design:

1. **Topic is not valence.** A "Third-Party Sharing" segment can be harmful
   ("we share data with advertisers") or protective ("we do not sell or share
   your information"). The model detects the topic, not negation, permission,
   or denial. We therefore call the concept an *attention tier*, not a *risk
   tier*, and attach a `default_valence` that is mostly `Mixed` in this version.
2. **Reading importance is not model uncertainty.** A low-confidence
   prediction may deserve *more* human attention, not less. We keep reading
   priority and review need as two separate outputs.

These tiers are user-friendly interpretation and are **not legal advice** and
**not a GDPR compliance determination**.

## 2. Which model powers this layer

The evaluation in the progress report keeps **Linear SVM** as the headline
classical model for RQ2 (it had the numerically highest cross-validated
macro-F1, essentially tied with Logistic Regression). The interpretation layer
instead uses **Logistic Regression**, for one concrete reason: `LinearSVC` does
not provide `predict_proba`, and reading priority and thresholding need
per-category probabilities. Logistic Regression provides them directly and was
essentially tied with the SVM in cross-validation, so nothing is lost.

To guarantee the interpretation-layer model is the *same* reported Logistic
Regression rather than a slightly different unreported one, the exact
configuration lives in `src/privacylens/config.py` and is imported by both
`src/run_baselines.py` and the artifact script:

- TF-IDF: `ngram_range=(1, 2)`, `min_df=2`, `max_df=1.0` (no `max_features`),
  `sublinear_tf=True`, `strip_accents="unicode"`.
- Classifier: `OneVsRestClassifier(LogisticRegression(C=1.0,
  class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=42))`.

One-vs-rest probabilities are independent per category and are **model
confidence estimates, not calibrated real-world probabilities**.

## 3. Attention tiers and valence

The framework is a **transparent, project-specific heuristic informed by**
usable-privacy research (Kelley et al.'s privacy "nutrition label"), ToS;DR
community assessments, and GDPR privacy principles. The cited sources motivate
*which practices deserve a reader's attention*; the numerical weights and the
1-5 star mapping are **PrivacyLens design choices**, not established severity
levels validated by those sources.

| Category | Attention tier | Default valence | Weight |
| --- | --- | --- | --- |
| Third-Party Sharing | High | Mixed | 4 |
| Data Collection | Medium | Mixed | 3 |
| Data Retention | Medium | Mixed | 3 |
| Policy Change | Medium | Mixed | 3 |
| User Control & Deletion | Medium | Mixed | 3 |
| Data Security | Medium | Mixed | 3 |
| Other/Unclear | Unknown | Unknown | 3 |

Design notes:

- **User Control & Deletion is Medium, not "positive."** The topic alone does
  not reveal whether a right is strong, limited, or effectively absent.
- **Data Security is Medium, not Low.** "We use encryption" and "no method of
  storage is completely secure" are both Data Security; v1 cannot tell a strong
  measure from a weak disclaimer, so Low would be unsafe.
- **Other/Unclear is Unknown with `review_flag=True`, never "Low."** "Not
  understood" is not the same as "low risk." It is always distinguished from
  genuinely low-attention content.

Valence values are `Potentially Harmful`, `Potentially Protective`, `Mixed`,
`Unknown`. They are mostly `Mixed` here because negation/permission detection is
future work; refining valence (and re-introducing Low-tier content) would let
the star scale use its full range (see Section 6).

## 4. GDPR principle mapping (informative, not compliance)

| Category | Relevant GDPR principles |
| --- | --- |
| Data Collection | Art. 5(1)(a) lawfulness, fairness, transparency; Art. 5(1)(c) data minimisation |
| Third-Party Sharing | transparency, purpose limitation, lawful processing, information about recipients (not only Art. 5(1)(b)) |
| Data Retention | Art. 5(1)(e) storage limitation |
| Data Security | Art. 5(1)(f) integrity and confidentiality; Art. 32 security of processing |
| User Control & Deletion | Arts. 15-22 data-subject rights |
| Policy Change | transparency and notice obligations (context-dependent) |
| Other/Unclear | no direct mapping without examining the actual practice |

The tier is an attention heuristic informed by these principles, not a finding
of compliance or non-compliance.

## 5. Reading-priority rule

Deterministic, documented, no learned parameters:

1. Base stars from the dominant attention tier: High=4, Medium=3, Low=2,
   Unknown=3. The dominant tier is the highest-attention practice present; a
   protective/Mixed label never cancels a higher-attention label. Tie-breaking
   is deterministic (highest weight, then highest confidence, then canonical
   category order).
2. Confidence adjustment on the dominant practice: `>= 0.70` adds one star;
   `< 0.40` never *subtracts* from a High/Unknown segment (it triggers review
   instead); a lone Low-tier label may drop one star.
3. Combination bonus: +1 when at least two distinct High/Medium labels clear
   their threshold with margin (`p >= threshold + 0.05`), so two
   barely-above-threshold labels cannot manufacture a 5-star result. The bonus
   cutoff is `min(1.0, threshold + 0.05)`.
4. Clamp to 1-5 (only the star total is clamped; malformed model confidence -
   NaN, infinity, out of [0, 1] - raises an error rather than being silently
   clamped).

Every score is explainable through `score_components` (`base`,
`confidence_adjustment`, `combination_bonus`, `final`) so it is always obvious
why a segment received its stars. Each star also carries a readable
`priority_band` (`STAR_LABELS`: 3 = *Review*, 4 = *Important*, 5 = *Very
important*; 1-2 = *Minimal*/*Low* but unreachable in v1) and a
`priority_subtype`. The subtype is the key that keeps two very different 3-star
outcomes from looking identical: `identified_medium` (a genuine medium-attention
practice) versus `unknown_abstention` (the model could not classify) versus
`unknown_identified` (a confident Other/Unclear). `results/priority_distribution.csv`
and the distribution figure split the bars on this subtype.

**Abstention.** If no category clears its threshold, the segment is not forced
into Other/Unclear. It returns an explicit abstention state (3 stars, dominant
category `None`, tier `Unknown`, `review_flag=True`, reason "the model could
not confidently assign this segment to a mapped category").

**Review flag (separate from reading stars, and itself two signals).** Moderate
confidence *alone* does not flag a segment; that would flag most of the corpus
and make the flag useless. A segment is flagged only when there is a genuine
verification need. Crucially, `review_required` keeps a single Boolean for the
UI but the code and statistics preserve **two distinct meanings** separately:

- **Content-based review** (`content_unclear_flag`): the dominant prediction is
  Other/Unclear - the segment's *content* is ambiguous.
- **Model-based review** (`model_uncertainty_flag`): the *model* is uncertain -
  a high-attention practice sits just below its threshold (`high_attention_near_miss`,
  a near miss that would otherwise be silently dropped even though another label
  passed), a predicted label sits on its decision boundary (`near_threshold`),
  confidence is below the low band (`low_confidence`), or no category cleared
  its threshold at all (`abstention`).

The full ordered list is exposed as `review_types`, with human-readable
`review_reasons`. Because one segment can satisfy several conditions, a
deterministic `primary_review_reason` is chosen by a fixed priority order
(`abstention` > `high_attention_near_miss` > `low_confidence` > `near_threshold`
> `content_unclear`) so per-reason percentages partition cleanly, and a
`review_severity` (`high`/`medium`/`low`) follows from that primary reason.
`content_unclear_role` additionally records whether Other/Unclear was the *only*,
the *dominant*, or merely a *secondary* predicted label; the approved rule flags
only the only/dominant cases, but recording the secondary role lets us measure
how much of the content-driven rate is genuinely "nothing else identified."

**Does the flag concentrate difficulty?** `results/flag_stats.csv` reports the
overall rate, the two driver rates and their overlap, per-type rates, the
deterministic primary-reason partition, and the Other/Unclear-role breakdown.
`results/review_diagnostics.csv` then checks whether the flag is *useful*: it
compares the error rate inside vs. outside each flag, how many of all errors the
flag captures (recall), how many flagged segments are actually wrong
(precision), and how many high-confidence errors slipped through unflagged.
Empirically the model-uncertainty flag concentrates errors sharply (flagged
error rate well above the non-flagged rate), whereas the content-unclear flag
does *not* - which is exactly why the two meanings must not be collapsed: one
tracks model correctness, the other tracks content ambiguity. (Errors here use
the strict exact-match multi-label criterion, so the absolute rates are
conservative.)

**Confidence bands are heuristic.** `LOW_CONF=0.40` and `HIGH_CONF=0.70` are
provisional. `results/priority_sensitivity.csv` reports the star distribution
under nearby bands (0.35/0.65, 0.40/0.70, 0.45/0.75), and
`results/calibration_summary.csv` reports a per-category Brier score and
reliability summary. We do not claim calibrated probabilities.

## 6. Thresholds, out-of-fold evaluation, and what is *not* claimed

Per-category decision thresholds are selected by maximizing F1 on
**policy-grouped out-of-fold predictions** on the training split (TF-IDF is fit
inside each fold; folds are the single shared assignment in
`results/cv_fold_assignments.csv`, identical to `run_baselines.py`). Selection
is deterministic: below `MIN_THRESHOLD_SUPPORT` positive examples a category
keeps 0.5; otherwise F1 is maximized over a fixed 0.05-0.95 grid; ties break to
the lowest threshold for High-tier categories (favouring recall) and to the
threshold nearest 0.5 otherwise.

These thresholds are **development-set / exploratory artifacts**: they are
tuned on the same out-of-fold predictions they are then applied to, so the
resulting numbers are not an unbiased estimate of deployed performance. The
fold-mean cross-validation table in `run_baselines.py` remains the primary
model comparison, and a single thresholded evaluation on the untouched,
policy-disjoint **test set is deferred to the final report**.

"Running the rule on real OPP-115 segments" is **implementation validation /
sanity checking / sensitivity analysis / demonstration**, not evidence that the
stars match what users or privacy experts consider important. On the current
framework every category is High/Medium/Unknown, so the reachable star range is
3-5 rather than the full 1-5; this is a direct, acknowledged consequence of the
tier choices (no Low-tier content in v1), not a bug. Actual validation of star
importance would require expert-assigned priorities, user ratings, or
inter-annotator agreement, which are out of scope.

## 7. Explanation templates and evidence

Explanations are static, auditable, and cautious ("This passage appears to
describe..."). `explain_all` returns an explanation for every predicted label
(dominant first), never suppressing the others. Evidence phrases are trimmed,
deduped, and length-capped in `templates.py`; HTML escaping is intentionally
left to the presentation (Jinja/UI) layer so CSV and report exports stay
readable and nothing is double-escaped. A standing disclaimer accompanies
highlighted evidence, noting that a highlighted phrase such as "share personal
information" can appear inside a negation like "we do not share personal
information."

## 8. Reproducing the artifacts

```
python src/run_baselines.py               # writes shared folds + CV results
python src/make_interpretation_artifacts.py
python -m pytest tests/
```

Outputs: `results/attention_framework.csv`, `category_thresholds.json`
(+ provenance), `threshold_selection.csv`, `calibration_summary.csv`,
`priority_sensitivity.csv`, `flag_stats.csv`, `review_diagnostics.csv`,
`priority_distribution.csv`, `abstention_stats.csv`, `fold_support.csv`,
`explanation_templates.csv`, `priority_examples.csv`,
`interpretation_metadata.json`, and
`results/figures/fig_priority_distribution.png`.

## References

- P. G. Kelley, J. Bresee, L. F. Cranor, and R. W. Reeder. A "Nutrition Label"
  for Privacy. In Proceedings of SOUPS, 2009.
- Terms of Service; Didn't Read (ToS;DR). Community ratings of terms-of-service
  and privacy-policy points. https://tosdr.org.
- Regulation (EU) 2016/679 (General Data Protection Regulation), esp. Art. 5
  (principles), Art. 32 (security of processing), and Arts. 15-22 (data-subject
  rights).
- S. Wilson et al. The Creation and Analysis of a Website Privacy Policy
  Corpus (OPP-115). In Proceedings of ACL, 2016.
