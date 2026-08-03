"""Shared constants and framework schema for the PrivacyLens interpretation layer.

Single source of truth for: the user-facing category set, category-name
canonicalization, the attention-tier + valence framework, the model
configuration (identical to src/run_baselines.py so the interpretation-layer
model is provably the same reported model), heuristic confidence bands, and
paths.

Attention vs risk: we deliberately call the formal concept an "attention
tier", not a "risk tier". The classifier detects the TOPIC of a segment, not
whether the described practice is favourable or harmful (its valence). See
``default_valence`` below, which is mostly "Mixed" in this first version
because the model does not yet detect negation, permission, or denial.
"""

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SEGMENTS_CSV = DATA / "segments.csv"
FOLDS_CSV = RESULTS / "cv_fold_assignments.csv"
THRESHOLDS_JSON = RESULTS / "category_thresholds.json"
INTERP_METADATA_JSON = RESULTS / "interpretation_metadata.json"
# Persisted prototype model (full fitted pipeline) + its provenance metadata.
PROTOTYPE_MODEL_JOBLIB = RESULTS / "prototype_model.joblib"
PROTOTYPE_METADATA_JSON = RESULTS / "prototype_model_metadata.json"

# Transformer artifacts
TRANSFORMER_OOF_CSV = RESULTS / "transformer_oof_predictions.csv"
TRANSFORMER_METADATA_JSON = RESULTS / "transformer_model_metadata.json"

# ---------------------------------------------------------------------------
# Framework identity / provenance
# ---------------------------------------------------------------------------
FRAMEWORK_VERSION = "1.0"
MODEL_NAME = "logistic_regression_ovr"
SPLIT_ID = "policy_split_seed_42"
SEED = 42
N_FOLDS = 5

# ---------------------------------------------------------------------------
# Model configuration (MUST match src/run_baselines.py Logistic Regression)
# so the interpretation-layer model is the same model reported for RQ2.
# run_baselines.py imports these to guarantee a single source.
# ---------------------------------------------------------------------------
TFIDF_PARAMS = dict(
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    strip_accents="unicode",
)
LR_PARAMS = dict(
    C=1.0,
    class_weight="balanced",
    max_iter=2000,
    solver="lbfgs",
    random_state=SEED,
)

# Transformer hyperparameters
TRANSFORMER_MODEL_NAME = "distilbert_base_uncased_ovr"
TRANSFORMER_CHECKPOINT = "distilbert-base-uncased"
MAX_SEQ_LEN = 256
POS_WEIGHT_CAP = 20.0
TRANSFORMER_PARAMS = dict(
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    seed=SEED,
)
TRANSFORMER_FOLDS_TO_RUN = None


def model_version() -> str:
    """Deterministic model version string (no external infra needed)."""
    import sklearn
    ngr = "".join(str(x) for x in TFIDF_PARAMS["ngram_range"])
    return (f"{MODEL_NAME}|tfidf{ngr}_mindf{TFIDF_PARAMS['min_df']}"
            f"|C{LR_PARAMS['C']}_seed{SEED}|sklearn{sklearn.__version__}")


# ---------------------------------------------------------------------------
# Category set (alphabetical, matching sklearn MultiLabelBinarizer order)
# ---------------------------------------------------------------------------
CATEGORIES = [
    "Data Collection",
    "Data Retention",
    "Data Security",
    "Other/Unclear",
    "Policy Change",
    "Third-Party Sharing",
    "User Control & Deletion",
]
MAJORITY_CATEGORY = "Other/Unclear"

# ---------------------------------------------------------------------------
# Confidence bands (documented HEURISTIC values; sensitivity is analyzed in
# make_interpretation_artifacts.py). These are model confidence estimates,
# not calibrated real-world probabilities.
# ---------------------------------------------------------------------------
LOW_CONF = 0.40
HIGH_CONF = 0.70
NEAR_MARGIN = 0.05          # "near a threshold" band for review flagging
BONUS_MARGIN = 0.05         # extra margin required to count toward the bonus
DEFAULT_THRESHOLD = 0.50    # provisional threshold before tuning

# Minimum positive out-of-fold support required to tune a category threshold;
# below this we keep DEFAULT_THRESHOLD rather than optimize over an unstable
# sample. All seven categories exceed this on OPP-115, but rare categories in
# future data may not.
MIN_THRESHOLD_SUPPORT = 30

# ---------------------------------------------------------------------------
# Attention-tier + valence framework (proposal section 6).
#
# This is a transparent, project-specific heuristic INFORMED BY usable-privacy
# research (Kelley et al.), ToS;DR community assessments, and GDPR principles.
# The cited sources motivate which practices deserve attention; the numerical
# weights and the 1-5 star mapping are PrivacyLens design choices, not
# established severity levels. Tiers are user-friendly interpretation and are
# explicitly NOT legal advice or GDPR compliance determinations.
#
# attention_tier : High | Medium | Low | Unknown
# default_valence: Potentially Harmful | Potentially Protective | Mixed | Unknown
#   (mostly Mixed here because v1 cannot detect negation/permission/denial)
# ---------------------------------------------------------------------------
TIER_WEIGHT = {"High": 4, "Medium": 3, "Low": 2, "Unknown": 3}

ATTENTION_FRAMEWORK = {
    "Third-Party Sharing": {
        "attention_tier": "High",
        "default_valence": "Mixed",
        "review_flag": False,
        "gdpr": ["transparency", "purpose limitation",
                 "lawful processing", "information about recipients"],
        "rationale": (
            "Data may leave the first party. A missed sharing clause is the "
            "costliest error for a user, so sharing topics get the highest "
            "attention regardless of whether the specific clause is harmful "
            "or protective."
        ),
        "sources": ["Kelley et al. 2009", "ToS;DR", "GDPR"],
    },
    "Data Collection": {
        "attention_tier": "Medium",
        "default_valence": "Mixed",
        "review_flag": False,
        "gdpr": ["Art. 5(1)(a) lawfulness/fairness/transparency",
                 "Art. 5(1)(c) data minimisation"],
        "rationale": (
            "Collection is expected, but scope matters; a headline item on "
            "the privacy nutrition label. Users should see what is collected."
        ),
        "sources": ["Kelley et al. 2009", "GDPR"],
    },
    "Data Retention": {
        "attention_tier": "Medium",
        "default_valence": "Mixed",
        "review_flag": False,
        "gdpr": ["Art. 5(1)(e) storage limitation"],
        "rationale": (
            "Relates to the storage-limitation principle; indefinite or "
            "unspecified retention is a documented user concern."
        ),
        "sources": ["GDPR", "ToS;DR"],
    },
    "Policy Change": {
        "attention_tier": "Medium",
        "default_valence": "Mixed",
        "review_flag": False,
        "gdpr": ["transparency / notice obligations (context-dependent)"],
        "rationale": (
            "Terms can change, sometimes with limited notice; notice "
            "requirements vary, so change clauses deserve a look."
        ),
        "sources": ["ToS;DR", "GDPR"],
    },
    "User Control & Deletion": {
        "attention_tier": "Medium",
        "default_valence": "Mixed",
        "review_flag": False,
        "gdpr": ["Arts. 15-22 data-subject rights"],
        "rationale": (
            "Concerns user rights (access, opt-out, deletion). Kept at Medium "
            "rather than 'positive' because the topic alone does not reveal "
            "whether the right is strong, limited, or effectively absent."
        ),
        "sources": ["GDPR", "Kelley et al. 2009", "ToS;DR"],
    },
    "Data Security": {
        "attention_tier": "Medium",
        "default_valence": "Mixed",
        "review_flag": False,
        "gdpr": ["Art. 5(1)(f) integrity/confidentiality",
                 "Art. 32 security of processing"],
        "rationale": (
            "Security statements can be strong ('we use encryption') or weak "
            "disclaimers ('no method is completely secure'). Because v1 cannot "
            "tell them apart, Medium is safer than Low."
        ),
        "sources": ["GDPR"],
    },
    "Other/Unclear": {
        "attention_tier": "Unknown",
        "default_valence": "Unknown",
        "review_flag": True,
        "gdpr": ["no direct mapping without examining the actual practice"],
        "rationale": (
            "Residual category: boilerplate, rare merged practices, ambiguous "
            "wording, mapping artifacts, or a genuine practice the classifier "
            "cannot place. 'Not understood' is not the same as 'low risk', so "
            "this is Unknown and always flagged for review."
        ),
        "sources": [],
    },
}

# Colours for the UI, reusing the report's validated palette. Unknown has its
# own colour so abstained/residual content is visually distinct.
ATTENTION_COLORS = {
    "High": "#d64545",       # red
    "Medium": "#d9922a",     # amber
    "Low": "#6b7280",        # grey
    "Unknown": "#7a5cc0",    # purple (distinct from all attention tiers)
}


# ---------------------------------------------------------------------------
# Category-name canonicalization (point 10): normalize harmless variants to
# the canonical label, or fail loudly. Never silently invent a category.
# ---------------------------------------------------------------------------
def _norm(name: str) -> str:
    s = name.strip().lower()
    s = s.replace("&", "and").replace("-", " ")
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s+", " ", s)
    return s


_CANON = {_norm(c): c for c in CATEGORIES}
_CANON.update({
    _norm("Other"): "Other/Unclear",
    _norm("Unclear"): "Other/Unclear",
    _norm("Third Party Sharing"): "Third-Party Sharing",
    _norm("User Control and Deletion"): "User Control & Deletion",
})


def canonicalize_category(name) -> str:
    """Return the canonical category name for a possibly-messy input.

    Accepts case/spacing/'&'-vs-'and'/hyphen variants. Raises ValueError for
    anything not recognized, so unknown strings never leak through as new
    categories.
    """
    if not isinstance(name, str):
        raise ValueError(f"Category must be a string, got {type(name)!r}")
    key = _norm(name)
    if key in _CANON:
        return _CANON[key]
    raise ValueError(f"Unknown category: {name!r}")


def tier_of(category: str) -> str:
    return ATTENTION_FRAMEWORK[canonicalize_category(category)]["attention_tier"]


def weight_of(category: str) -> int:
    return TIER_WEIGHT[tier_of(category)]
