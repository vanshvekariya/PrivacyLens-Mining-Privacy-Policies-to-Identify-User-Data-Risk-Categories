"""Initial PrivacyLens experiments: baselines and classical models.

Models: majority-class, keyword rules, and TF-IDF (word 1-2 grams) with
one-vs-rest Naive Bayes, Logistic Regression, and Linear SVM.

Protocol: 5-fold cross-validation on the TRAIN split only, with folds grouped
by policy so no policy appears in both a CV-train and CV-validation fold.
The held-out test split is never read here.

Outputs:
  results/model_comparison.csv     macro/micro F1 per model (mean +- std over folds)
  results/per_category_f1.csv      per-category P/R/F1 per model
  results/error_examples.txt       sample misclassified segments (best model)
  results/cv_predictions.csv       out-of-fold predictions (best model), for figures
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from privacylens.config import CATEGORIES, LR_PARAMS, N_FOLDS, SEED, TFIDF_PARAMS
from privacylens.prediction import folds_to_indices, load_or_build_folds

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Models whose out-of-fold predictions are persisted for downstream error
# analysis, and their file-name slugs
SHARED_OOF_MODELS = {
    "Naive Bayes": "naive_bayes",
    "Logistic Regression": "logistic_regression",
    "Linear SVM": "linear_svm",
}


def sanitize_col(category):
    # Convert a category name into a csv-safe probability column
    return "prob_" + (category.replace("&", "and").replace("/", "_")
                              .replace("-", "_").replace(" ", "_"))


PROB_COLS = [sanitize_col(c) for c in CATEGORIES]


def get_proba(model, X):
    # Return per-category probabilities, using sigmoid(decision_function) as
    # a fallback for models like LinearSVC that lack predict_proba
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    raise ValueError(f"model {type(model).__name__} has no predict_proba or decision_function")


def write_oof_predictions_csv(model_slug, oof_probs, fold_of_row, train_df, classes):
    # Write out-of-fold predictions in the shared schema (matches train_transformer.py)
    kept = fold_of_row >= 0
    out = train_df.loc[kept, ["policy_stem", "segment_id", "labels"]].copy()
    out.rename(columns={"labels": "gold_labels"}, inplace=True)
    out["fold_id"] = fold_of_row[kept]
    kept_probs = oof_probs[kept]
    for j, col in enumerate(PROB_COLS):
        out[col] = kept_probs[:, j]
    preds = (kept_probs >= 0.5).astype(int)
    out["predicted_labels_at_0_5"] = [
        "|".join(c for j, c in enumerate(classes) if preds[i, j])
        for i in range(len(preds))
    ]
    path = RESULTS / f"oof_predictions_{model_slug}.csv"
    out.to_csv(path, index=False)
    return path

KEYWORDS = {
    "Data Collection": [
        "we collect", "collects", "information we collect", "you provide",
        "automatically", "cookies", "ip address", "log", "register",
    ],
    "Third-Party Sharing": [
        "third part", "share", "disclose", "partners", "affiliates",
        "advertisers", "service providers", "sell",
    ],
    "User Control & Deletion": [
        "opt out", "opt-out", "unsubscribe", "delete", "access", "update",
        "correct", "choices", "preferences", "control",
    ],
    "Data Retention": [
        "retain", "retention", "keep your", "store", "as long as",
    ],
    "Data Security": [
        "security", "secure", "encrypt", "protect", "safeguard", "ssl",
    ],
    "Policy Change": [
        "changes to this", "update this policy", "modify this", "revise",
        "effective date", "notify you of",
    ],
    "Other/Unclear": [
        "contact us", "questions", "children", "california", "residents",
    ],
}


def keyword_predict(texts, classes):
    Y = np.zeros((len(texts), len(classes)), dtype=int)
    for i, t in enumerate(texts):
        low = t.lower()
        for j, c in enumerate(classes):
            if any(k in low for k in KEYWORDS[c]):
                Y[i, j] = 1
        if Y[i].sum() == 0:                       # fall back to majority class
            Y[i, classes.index("Other/Unclear")] = 1
    return Y


def make_models():
    # TF-IDF and Logistic Regression configs come from privacylens.config so
    # the interpretation-layer LR is provably the same model reported here.
    tfidf = lambda: TfidfVectorizer(**TFIDF_PARAMS)
    return {
        "Naive Bayes": make_pipeline(
            tfidf(), OneVsRestClassifier(MultinomialNB())),
        "Logistic Regression": make_pipeline(
            tfidf(), OneVsRestClassifier(LogisticRegression(**LR_PARAMS))),
        "Linear SVM": make_pipeline(
            tfidf(), OneVsRestClassifier(
                LinearSVC(class_weight="balanced", C=0.5, random_state=SEED))),
    }


def main():
    df = pd.read_csv(ROOT / "data" / "segments.csv")
    train = df[df.split == "train"].reset_index(drop=True)
    texts = train["text"].fillna("").tolist()
    label_sets = [s.split("|") for s in train["labels"]]
    groups = train["policy_stem"].values

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(label_sets)
    classes = list(mlb.classes_)
    majority = classes.index("Other/Unclear")

    oof = {name: np.zeros_like(Y) for name in
           ["Majority", "Keyword", *make_models()]}

    # Per-model out-of-fold probabilities for shared-schema CSV export
    oof_probs = {name: np.zeros((len(texts), len(classes)), dtype=np.float32)
                 for name in SHARED_OOF_MODELS}
    fold_of_row = np.full(len(texts), -1, dtype=np.int64)

    # Shared, persisted policy-grouped folds (single source for every model,
    # figure, and threshold artifact).
    fold_of_policy = load_or_build_folds(groups)
    fold_scores = {name: [] for name in oof}
    for fold_id, (tr_idx, va_idx) in enumerate(folds_to_indices(groups, fold_of_policy)):
        X_tr = [texts[i] for i in tr_idx]
        X_va = [texts[i] for i in va_idx]
        fold_of_row[va_idx] = fold_id

        preds = {"Majority": np.zeros((len(va_idx), len(classes)), dtype=int),
                 "Keyword": keyword_predict(X_va, classes)}
        preds["Majority"][:, majority] = 1
        for name, model in make_models().items():
            model.fit(X_tr, Y[tr_idx])
            preds[name] = model.predict(X_va)
            if name in SHARED_OOF_MODELS:
                oof_probs[name][va_idx] = get_proba(model, X_va)

        for name, P in preds.items():
            oof[name][va_idx] = P
            fold_scores[name].append((
                f1_score(Y[va_idx], P, average="macro", zero_division=0),
                f1_score(Y[va_idx], P, average="micro", zero_division=0)))

    # ---- model comparison table ----
    rows = []
    for name, scores in fold_scores.items():
        arr = np.array(scores)
        rows.append({
            "model": name,
            "macro_f1_mean": arr[:, 0].mean(), "macro_f1_std": arr[:, 0].std(),
            "micro_f1_mean": arr[:, 1].mean(), "micro_f1_std": arr[:, 1].std(),
        })
    comparison = pd.DataFrame(rows).sort_values("macro_f1_mean")
    comparison.to_csv(RESULTS / "model_comparison.csv", index=False)

    # ---- per-category P/R/F1 (on pooled out-of-fold predictions) ----
    per_cat = []
    for name, P in oof.items():
        p, r, f, s = precision_recall_fscore_support(Y, P, zero_division=0)
        for j, c in enumerate(classes):
            per_cat.append({"model": name, "category": c, "support": int(s[j]),
                            "precision": p[j], "recall": r[j], "f1": f[j]})
    per_cat = pd.DataFrame(per_cat)
    per_cat.to_csv(RESULTS / "per_category_f1.csv", index=False)

    # ---- best model: save OOF predictions and error examples ----
    best = comparison.iloc[-1]["model"]
    P = oof[best]
    pred_labels = ["|".join(c for j, c in enumerate(classes) if P[i, j])
                   for i in range(len(P))]
    out = train[["policy_stem", "segment_id", "labels"]].copy()
    out["predicted"] = pred_labels
    out.to_csv(RESULTS / "cv_predictions.csv", index=False)

    wrong = [(i, set(label_sets[i]), set(pred_labels[i].split("|")) - {""})
             for i in range(len(P))
             if set(pred_labels[i].split("|")) - {""} != set(label_sets[i])]
    rng = np.random.default_rng(SEED)
    lines = [f"Best model: {best}. {len(wrong)}/{len(P)} segments with "
             f"imperfect label sets in out-of-fold CV predictions.", ""]
    for i, gold, pred in (wrong[k] for k in
                          rng.choice(len(wrong), 12, replace=False)):
        snippet = re.sub(r"\s+", " ", texts[i])[:400]
        lines += [f"GOLD: {sorted(gold)}", f"PRED: {sorted(pred) or '(none)'}",
                  f"TEXT: {snippet}", "-" * 80]
    (RESULTS / "error_examples.txt").write_text("\n".join(lines),
                                                encoding="utf-8")

    # Per-model OOF predictions in the shared schema for downstream analysis
    for name, slug in SHARED_OOF_MODELS.items():
        path = write_oof_predictions_csv(slug, oof_probs[name], fold_of_row,
                                          train, classes)
        print(f"Wrote {path.name}")

    print(comparison.to_string(index=False))
    print()
    print(per_cat[per_cat.model == best]
          .sort_values("f1", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
