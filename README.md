# PrivacyLens

PrivacyLens is a multi-label privacy-policy clause classifier and reading
assistant. It maps OPP-115 annotations into seven user-facing categories,
compares classical and transformer models under policy-disjoint evaluation,
and turns predictions into attention tiers, reading-priority scores,
explanations, and highlighted supporting evidence.

## What is included

- OPP-115 parsing, cleaning, category mapping, and policy-disjoint splits
- Majority and keyword baselines
- TF-IDF with Naive Bayes, Logistic Regression, and Linear SVM
- DistilBERT multi-label fine-tuning and out-of-fold evaluation
- Frozen per-category thresholds and confidence-aware review flags
- Attention tiers, 1–5 star reading priority, and explanation templates
- TF-IDF contribution and transformer word-occlusion evidence
- Structured error-review workflow and aggregate failure analysis
- Modern-policy generalization evaluation
- Streamlit prototype
- Automated unit and integration tests

## Environment

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Data

Download OPP-115 v1.0 from the
[Usable Privacy Project](https://usableprivacy.org/data), extract it under
`data/OPP-115/`, and retain these directories:

- `consolidation/`
- `sanitized_policies/`
- `annotations/`
- `documentation/`

The `data/` directory is intentionally ignored because the source corpus and
derived datasets are large and reproducible.

Build the mapped dataset:

```powershell
python src/build_dataset.py
```

This creates `data/segments.csv`, the deterministic 92/23-policy train/test
split, label-distribution statistics, and split metadata.

## Reproduce the experiments

Run the stages from the repository root:

```powershell
# Classical baselines and policy-grouped five-fold CV
python src/run_baselines.py

# DistilBERT policy-grouped five-fold training
python src/train_transformer.py
python src/eval_transformer.py

# Freeze thresholds and create interpretation artifacts
python src/make_interpretation_artifacts.py

# Fit the deployed Logistic Regression pipeline on the full training split
python src/train_prototype_model.py

# Sample and complete the structured error analysis
python src/sample_errors_for_review.py
python src/complete_error_review.py

# Evaluate frozen models on the modern-policy set
python src/collect_generalization_policies.py
python src/build_generalization_set.py
python src/label_generalization_set.py
python src/merge_generalization_labels.py
python src/evaluate_generalization.py
```

The final OPP-115 test evaluation is guarded against accidental repeated
access:

```powershell
python src/evaluate_test_set.py
```

After a successful run, `results/test_evaluation_metadata.json` acts as a
completion sentinel and the evaluator refuses to run again.

## Run the prototype

The trained model artifact is included in `results/`. Start the application
with:

```powershell
streamlit run app.py
```

Users can paste or upload privacy-policy text and sort the analyzed clauses by
reading priority. Each clause includes predicted categories, attention tier,
priority stars, review flags, explanations, and highlighted evidence.

## Tests

```powershell
python -m pytest -q
```

The tests cover category normalization, threshold decisions, attention and
priority rules, evidence extraction, safe highlighting, segmentation,
end-to-end analysis, artifact generation, generalization-label merging, and
transformer word occlusion.

## Main outputs

| Output | Purpose |
| --- | --- |
| `results/model_comparison.csv` | Cross-validated model comparison |
| `results/per_category_f1.csv` | Pooled out-of-fold category metrics |
| `results/transformer_oof_predictions.csv` | DistilBERT out-of-fold predictions |
| `results/error_analysis/structured_error_summary.csv` | Reviewed failure-type counts |
| `results/generalization_headline.csv` | Modern-policy headline metrics |
| `results/generalization_eval.csv` | Modern-policy per-category metrics |
| `results/test_set_headline.csv` | Final held-out OPP-115 metrics |
| `results/test_set_per_category.csv` | Final held-out category metrics |
| `results/prototype_model.joblib` | Deployed Logistic Regression pipeline |

## Interpretation boundary

PrivacyLens identifies the topic of a clause and recommends what deserves
attention. It does not determine whether a clause is legally compliant,
favourable, or harmful, and it is not legal advice.
