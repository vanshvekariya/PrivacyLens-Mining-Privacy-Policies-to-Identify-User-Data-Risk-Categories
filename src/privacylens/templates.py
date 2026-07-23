"""Cautious plain-English explanation templates (proposal section 7.4).

Static, auditable strings (no generation) so they are defensible and easy to
cite. Two safeguards baked in:

  - Cautious phrasing ("This passage appears to describe...") because the
    classifier can be wrong and detects topic, not favourability.
  - Evidence is only trimmed / deduped / length-capped here. HTML ESCAPING IS
    NOT DONE IN THIS MODULE - it belongs in the presentation/UI layer, so CSV and
    report exports stay readable and nothing is double-escaped.

The negation/uncertainty DISCLAIMER is exported for the UI to show alongside
highlighted evidence.
"""

import re

from .attention import attention_profile
from .config import canonicalize_category

MAX_EVIDENCE_LEN = 60
MAX_EVIDENCE_ITEMS = 3

DISCLAIMER = (
    "Highlighted words indicate text that influenced the model. They do not "
    "prove that the policy performs the practice, or performs it in a harmful "
    "or unlawful way. A phrase such as \"share personal information\" can also "
    "appear inside a negation like \"we do not share personal information\"."
)

_TEMPLATES = {
    "Data Collection": (
        "This passage appears to describe information the service collects "
        "about you. The scope matters: which data, and whether it is provided "
        "by you or gathered automatically. Check what categories of data are "
        "listed."
    ),
    "Data Retention": (
        "This passage appears to relate to how long your data is kept. Check "
        "whether a specific retention period is given or whether data may be "
        "kept indefinitely."
    ),
    "Data Security": (
        "This passage appears to discuss how your data is protected. Note "
        "whether it describes concrete measures (for example, encryption) or "
        "is mainly a disclaimer that no system is fully secure."
    ),
    "Other/Unclear": (
        "The system could not confidently place this passage in a specific "
        "privacy practice. It may be boilerplate, jurisdiction-specific text, "
        "or wording the model does not handle well. Read it yourself if the "
        "surrounding context seems important."
    ),
    "Policy Change": (
        "This passage appears to describe how and when the policy itself can "
        "change. Check whether you would be notified of material changes and "
        "how."
    ),
    "Third-Party Sharing": (
        "This passage appears to describe information being shared with, or "
        "collected by, other organizations. Because sharing can send data "
        "beyond the first party, it is worth reading carefully. Check who the "
        "recipients are and for what purpose."
    ),
    "User Control & Deletion": (
        "This passage appears to concern your controls over your data, such "
        "as access, correction, opt-out, or deletion. Check whether the "
        "control is clearly granted, limited, or hard to exercise."
    ),
}


def sanitize_evidence(evidence):
    """Trim, dedupe, and length-cap evidence spans. No HTML escaping.

    Handles None, empty lists, non-string entries, whitespace, duplicates
    (case-insensitive), very long spans (e.g. a whole segment), and caps the
    number of spans returned.
    """
    if not evidence:
        return []
    out, seen = [], set()
    for e in evidence:
        if not isinstance(e, str):
            continue
        e = re.sub(r"\s+", " ", e).strip()
        if not e:
            continue
        if len(e) > MAX_EVIDENCE_LEN:
            e = e[:MAX_EVIDENCE_LEN].rstrip() + "\u2026"  # ellipsis
        key = e.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
        if len(out) >= MAX_EVIDENCE_ITEMS:
            break
    return out


def explain(category, evidence=None) -> str:
    """Return a cautious explanation for one predicted category.

    ``evidence`` is an optional list of supporting phrases; a single string is
    also accepted for convenience.
    """
    c = canonicalize_category(category)
    text = _TEMPLATES[c]
    if isinstance(evidence, str):
        evidence = [evidence]
    spans = sanitize_evidence(evidence)
    if spans:
        joined = "; ".join(f"\"{s}\"" for s in spans)
        text += f" (text that influenced this: {joined})"
    return text


def explain_all(categories, evidence_by_category=None) -> list:
    """Explain every predicted category, dominant first, others after.

    Returns a list of {category, explanation} dicts; the dominant category is
    ordered first but no category is suppressed.
    """
    evidence_by_category = evidence_by_category or {}
    if not categories:
        return []

    canon = []
    for raw in categories:
        c = canonicalize_category(raw)
        if c not in canon:
            canon.append(c)

    dominant = attention_profile(canon)["dominant_category"]
    ordered = ([dominant] + [c for c in canon if c != dominant]
               if dominant else canon)

    result = []
    for c in ordered:
        ev = evidence_by_category.get(c)
        if ev is None:
            # tolerate raw-key evidence dicts
            for raw, v in evidence_by_category.items():
                try:
                    if canonicalize_category(raw) == c:
                        ev = v
                        break
                except ValueError:
                    continue
        result.append({"category": c, "explanation": explain(c, ev)})
    return result
