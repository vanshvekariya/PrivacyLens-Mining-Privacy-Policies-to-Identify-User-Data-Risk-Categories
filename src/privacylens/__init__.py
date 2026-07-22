"""PrivacyLens interpretation layer.

Turns the classical multi-label privacy-practice classifier into a
user-facing triage layer, as specified in formal_proposal_v2.md
(sections 6-8):

  segment -> predicted practice categories (with confidence, via a
             centralized thresholding step in ``prediction``)
          -> attention tier + valence (``attention``)
          -> 1-5 star reading priority + review flag (``priority``)
          -> cautious plain-English explanation (``templates``)

Modules are small and independent so each maps to a specific proposal
section and is easy to unit-test:

  config      framework schema, category set, model/threshold constants
  prediction  decide_labels() + threshold/fold loading (single source)
  attention   attention-tier framework and multi-label profile
  priority    reading-priority rule (reading importance vs review need)
  templates   cautious explanation templates

Terminology note: the formal name of the framework is "attention", not
"risk". The word "risk" appears only in user-facing wording such as
"potential privacy risk".
"""

from .attention import attention_for, attention_profile, framework_table
from .config import ATTENTION_FRAMEWORK, CATEGORIES, FRAMEWORK_VERSION
from .prediction import decide_labels
from .priority import reading_priority

__all__ = [
    "ATTENTION_FRAMEWORK",
    "CATEGORIES",
    "FRAMEWORK_VERSION",
    "attention_for",
    "attention_profile",
    "framework_table",
    "decide_labels",
    "reading_priority",
]
