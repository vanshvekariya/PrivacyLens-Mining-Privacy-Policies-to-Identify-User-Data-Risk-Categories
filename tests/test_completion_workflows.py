"""Focused tests for the completion workflows added after model training."""

import pandas as pd

from collect_generalization_policies import (
    document_spanning_sample,
    extract_blocks,
)
from complete_error_review import TAXONOMY, classify
from label_generalization_set import EXPECTED_HASHES, L
from privacylens.config import CATEGORIES


def row(**updates):
    base = {
        "gold_labels": "Data Collection",
        "predicted_labels": "",
        "near_threshold": False,
        "text": "We collect account information.",
    }
    base.update(updates)
    return pd.Series(base)


def test_error_review_category_specific_negation():
    failure, _ = classify(row(
        gold_labels="Other/Unclear",
        predicted_labels="Third-Party Sharing",
        text="We do not share your personal information.",
    ))
    assert failure == "negation_or_permission_flip"


def test_error_review_taxonomy_priority():
    failure, _ = classify(row(
        gold_labels="Data Collection|Data Retention",
        predicted_labels="Data Collection",
        near_threshold=True,
        text="We retain account information.",
    ))
    assert failure == "near_threshold"
    assert failure in TAXONOMY


def test_error_review_multilabel_underprediction():
    failure, _ = classify(row(
        gold_labels="Data Collection|Data Retention",
        predicted_labels="Data Collection",
        text="We collect and retain account information.",
    ))
    assert failure == "multi_label_under_prediction"


def test_policy_html_extraction_and_sampling():
    html = """
    <html><body><nav>Navigation link</nav><main>
      <h1>Privacy policy for the example service and website</h1>
      <p>We collect account information when you register for the service.</p>
      <p>We share account information with processors that host the service.</p>
      <p>You can request deletion through your account settings at any time.</p>
    </main></body></html>
    """
    blocks = extract_blocks(html)
    assert len(blocks) == 4
    assert all("Navigation" not in block for block in blocks)
    assert document_spanning_sample(blocks, 3) == [
        blocks[0], blocks[2], blocks[3]
    ]


def test_fixed_generalization_mapping_is_complete_and_canonical():
    assert len(L) == 123
    assert len(EXPECTED_HASHES) == 8
    for labels in L.values():
        assert set(labels.split("|")) <= set(CATEGORIES)
