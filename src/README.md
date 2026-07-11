# PrivacyLens — reproducing the initial experiments

Requirements: Python 3.11+, `pandas`, `scikit-learn`, `numpy`, `matplotlib`.
Dataset: OPP-115 v1.0 (https://usableprivacy.org/static/data/OPP-115_v1_0.zip),
extracted to `data/OPP-115/` (only `consolidation/`, `sanitized_policies/`,
`annotations/`, `documentation/` are needed; `original_policies/` contains
filenames that exceed the Windows path limit and is not used).

Run from the project root, in order:

```
python src/build_dataset.py    # parse + map labels + policy-disjoint split
python src/run_baselines.py    # baselines, NB, LogReg, Linear SVM (grouped 5-fold CV)
python src/make_figures.py     # PNGs for the report
```

All randomness is seeded (seed 42). The held-out test split
(`results/splits.json`) must not be evaluated until the final report.

Outputs land in `results/`:
- `dataset_stats.txt`, `label_distribution.csv` — corpus statistics (Table 1)
- `model_comparison.csv`, `per_category_f1.csv` — experiment results (Table 2)
- `cv_predictions.csv`, `error_examples.txt` — error-analysis inputs
- `figures/*.png` — report figures
