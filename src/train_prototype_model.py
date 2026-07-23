"""Train and persist the prototype model (the exact reported Logistic Regression).

The prototype needs per-category probabilities (``predict_proba``) for
thresholding, reading priority, and evidence, so it uses Logistic Regression -
the same configuration reported for RQ2 and used by the interpretation-layer
artifacts (see attention_framework.md s2). Linear SVM is the headline model in
the comparison table but does not expose calibrated probabilities, so it does
NOT power this layer.

We fit the COMPLETE pipeline on the full training split and persist it as a
single object, which is safer than pickling the vectorizer and classifier
separately (they can never drift out of sync). A sidecar metadata file records
provenance and lets the loader fail loudly on a stale or mismatched model.

Outputs:
  results/prototype_model.joblib           full fitted Pipeline
  results/prototype_model_metadata.json    provenance + load-time contract

The held-out OPP-115 test split is NOT read here.
"""

import json
from datetime import datetime, timezone

import joblib
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from privacylens.config import (
    CATEGORIES,
    FRAMEWORK_VERSION,
    LR_PARAMS,
    MODEL_NAME,
    PROTOTYPE_METADATA_JSON,
    PROTOTYPE_MODEL_JOBLIB,
    RESULTS,
    SEGMENTS_CSV,
    SPLIT_ID,
    TFIDF_PARAMS,
    THRESHOLDS_JSON,
    model_version,
)


def build_pipeline():
    """The exact reported pipeline, named for stable step access."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("classifier", OneVsRestClassifier(LogisticRegression(**LR_PARAMS))),
    ])


def main():
    RESULTS.mkdir(exist_ok=True)
    df = pd.read_csv(SEGMENTS_CSV)
    train = df[df.split == "train"].reset_index(drop=True)
    texts = train["text"].fillna("").tolist()
    label_sets = [s.split("|") for s in train["labels"]]

    # Column order fixed to CATEGORIES so probabilities, thresholds, and
    # evidence all align on the same category order.
    mlb = MultiLabelBinarizer(classes=CATEGORIES)
    Y = mlb.fit_transform(label_sets)
    assert list(mlb.classes_) == CATEGORIES, "MLB order must equal CATEGORIES"

    pipe = build_pipeline()
    pipe.fit(texts, Y)

    joblib.dump(pipe, PROTOTYPE_MODEL_JOBLIB)

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": model_version(),
        "category_order": CATEGORIES,
        "tfidf_params": {k: list(v) if isinstance(v, tuple) else v
                         for k, v in TFIDF_PARAMS.items()},
        "lr_params": LR_PARAMS,
        "training_split": SPLIT_ID,
        "n_train_segments": int(len(train)),
        "threshold_file": THRESHOLDS_JSON.name,
        "framework_version": FRAMEWORK_VERSION,
        "sklearn_version": sklearn.__version__,
        "model_created_at": datetime.now(timezone.utc).isoformat(),
    }
    PROTOTYPE_METADATA_JSON.write_text(json.dumps(metadata, indent=2),
                                       encoding="utf-8")

    print(f"trained on {len(train)} segments; {len(CATEGORIES)} categories")
    print("wrote", PROTOTYPE_MODEL_JOBLIB)
    print("wrote", PROTOTYPE_METADATA_JSON)


if __name__ == "__main__":
    main()
