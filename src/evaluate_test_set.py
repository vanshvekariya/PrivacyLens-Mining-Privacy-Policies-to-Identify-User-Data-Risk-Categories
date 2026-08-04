"""One-time evaluation on the held-out OPP-115 policy split.

The two strongest models were selected using policy-grouped cross-validation:
Logistic Regression and Linear SVM.  This script evaluates their frozen
configurations once on the untouched test policies and then writes a completion
sentinel.  It refuses to run again unless the existing result artifacts are
deliberately removed.

Outputs (results/):
  test_set_headline.csv
  test_set_per_category.csv
  test_set_predictions.csv
  test_evaluation_metadata.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from privacylens.config import (
    CATEGORIES,
    FRAMEWORK_VERSION,
    RESULTS,
    SEED,
    SEGMENTS_CSV,
    TFIDF_PARAMS,
)
from privacylens.pipeline import load_prototype_model

HEADLINE = RESULTS / "test_set_headline.csv"
PER_CATEGORY = RESULTS / "test_set_per_category.csv"
PREDICTIONS = RESULTS / "test_set_predictions.csv"
METADATA = RESULTS / "test_evaluation_metadata.json"
N_BOOTSTRAP = 1000


def multi_hot(label_strings) -> np.ndarray:
    sets = [
        value.split("|") if isinstance(value, str) and value else []
        for value in label_strings
    ]
    return MultiLabelBinarizer(classes=CATEGORIES).fit_transform(sets)


def train_svm(texts, targets):
    model = make_pipeline(
        TfidfVectorizer(**TFIDF_PARAMS),
        OneVsRestClassifier(
            LinearSVC(class_weight="balanced", C=0.5, random_state=SEED)
        ),
    )
    model.fit(texts, targets)
    return model


def lr_predict(bundle, texts) -> np.ndarray:
    probabilities = np.asarray(bundle["pipeline"].predict_proba(texts))
    thresholds = np.array([bundle["thresholds"][c] for c in CATEGORIES])
    return (probabilities >= thresholds).astype(int), probabilities


def sigmoid(values):
    values = np.clip(np.asarray(values), -40, 40)
    return 1.0 / (1.0 + np.exp(-values))


def policy_bootstrap(gold, pred, groups, n=N_BOOTSTRAP):
    rng = np.random.default_rng(SEED)
    policies = np.array(sorted(set(groups)))
    by_policy = {p: np.flatnonzero(groups == p) for p in policies}
    macro, micro = [], []
    for _ in range(n):
        sampled = rng.choice(policies, size=len(policies), replace=True)
        idx = np.concatenate([by_policy[p] for p in sampled])
        macro.append(f1_score(gold[idx], pred[idx], average="macro",
                              zero_division=0))
        micro.append(f1_score(gold[idx], pred[idx], average="micro",
                              zero_division=0))
    return {
        "macro_f1_ci_low": float(np.percentile(macro, 2.5)),
        "macro_f1_ci_high": float(np.percentile(macro, 97.5)),
        "micro_f1_ci_low": float(np.percentile(micro, 2.5)),
        "micro_f1_ci_high": float(np.percentile(micro, 97.5)),
    }


def labels_from_matrix(matrix) -> list[str]:
    return [
        "|".join(c for j, c in enumerate(CATEGORIES) if row[j])
        for row in matrix
    ]


def evaluate_model(name, gold, pred, groups):
    headline = {
        "model": name,
        "macro_f1": f1_score(gold, pred, average="macro", zero_division=0),
        "micro_f1": f1_score(gold, pred, average="micro", zero_division=0),
        "subset_accuracy": accuracy_score(gold, pred),
        "n_segments": int(len(gold)),
        "n_policies": int(len(set(groups))),
    }
    headline.update(policy_bootstrap(gold, pred, groups))

    precision, recall, f1, support = precision_recall_fscore_support(
        gold, pred, zero_division=0
    )
    per_category = [
        {
            "model": name,
            "category": category,
            "support": int(support[j]),
            "precision": float(precision[j]),
            "recall": float(recall[j]),
            "f1": float(f1[j]),
        }
        for j, category in enumerate(CATEGORIES)
    ]
    return headline, per_category


def main() -> None:
    if METADATA.exists():
        raise RuntimeError(
            f"Held-out evaluation is already complete ({METADATA.name}). "
            "Refusing to touch the test set again."
        )

    df = pd.read_csv(SEGMENTS_CSV)
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("segments.csv must contain non-empty train and test splits")

    train_policies = set(train["policy_stem"])
    test_policies = set(test["policy_stem"])
    overlap = train_policies & test_policies
    if overlap:
        raise ValueError(f"Policy leakage between train/test: {sorted(overlap)}")

    train_texts = train["text"].fillna("").tolist()
    test_texts = test["text"].fillna("").tolist()
    y_train = multi_hot(train["labels"])
    y_test = multi_hot(test["labels"])
    groups = test["policy_stem"].to_numpy()

    lr = load_prototype_model()
    lr_pred, lr_prob = lr_predict(lr, test_texts)

    svm = train_svm(train_texts, y_train)
    svm_pred = svm.predict(test_texts)
    svm_prob = sigmoid(svm.decision_function(test_texts))

    predictions = {
        "Logistic Regression": (lr_pred, lr_prob),
        "Linear SVM": (svm_pred, svm_prob),
    }
    headline_rows, category_rows = [], []
    output = test[["policy_stem", "segment_id", "labels"]].copy()
    output.rename(columns={"labels": "gold_labels"}, inplace=True)

    for name, (pred, scores) in predictions.items():
        headline, per_category = evaluate_model(
            name, y_test, pred, groups
        )
        headline_rows.append(headline)
        category_rows.extend(per_category)
        slug = name.lower().replace(" ", "_")
        output[f"{slug}_predicted_labels"] = labels_from_matrix(pred)
        for j, category in enumerate(CATEGORIES):
            col = (
                "score_" + category.lower().replace("&", "and")
                .replace("/", "_").replace("-", "_").replace(" ", "_")
            )
            output[f"{slug}_{col}"] = scores[:, j]

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(headline_rows).to_csv(HEADLINE, index=False)
    pd.DataFrame(category_rows).to_csv(PER_CATEGORY, index=False)
    output.to_csv(PREDICTIONS, index=False)
    metadata = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "framework_version": FRAMEWORK_VERSION,
        "selection_basis": "policy-grouped five-fold CV macro-F1",
        "models": list(predictions),
        "n_train_segments": int(len(train)),
        "n_test_segments": int(len(test)),
        "n_train_policies": int(len(train_policies)),
        "n_test_policies": int(len(test_policies)),
        "policy_overlap": 0,
        "bootstrap_resamples": N_BOOTSTRAP,
        "test_set_access": "single final evaluation",
    }
    METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(pd.DataFrame(headline_rows).to_string(index=False))
    print(f"Wrote {HEADLINE.name}, {PER_CATEGORY.name}, "
          f"{PREDICTIONS.name}, and {METADATA.name}")


if __name__ == "__main__":
    main()
