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
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SEED = 42
N_FOLDS = 5

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
    tfidf = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                    sublinear_tf=True, strip_accents="unicode")
    return {
        "Naive Bayes": make_pipeline(
            tfidf(), OneVsRestClassifier(MultinomialNB())),
        "Logistic Regression": make_pipeline(
            tfidf(), OneVsRestClassifier(
                LogisticRegression(max_iter=2000, class_weight="balanced",
                                   C=1.0, random_state=SEED))),
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

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_scores = {name: [] for name in oof}
    for tr_idx, va_idx in gkf.split(texts, Y, groups):
        X_tr = [texts[i] for i in tr_idx]
        X_va = [texts[i] for i in va_idx]

        preds = {"Majority": np.zeros((len(va_idx), len(classes)), dtype=int),
                 "Keyword": keyword_predict(X_va, classes)}
        preds["Majority"][:, majority] = 1
        for name, model in make_models().items():
            model.fit(X_tr, Y[tr_idx])
            preds[name] = model.predict(X_va)

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

    print(comparison.to_string(index=False))
    print()
    print(per_cat[per_cat.model == best]
          .sort_values("f1", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
