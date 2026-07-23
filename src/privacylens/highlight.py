"""Evidence -> escaped HTML highlighting (presentation/UI helper).

Kept as a pure module (no Streamlit import) so the escaping logic is
unit-testable. Two guarantees matter here:

  1. **Everything is HTML-escaped.** Each text slice (highlighted or not) is
     passed through ``html.escape`` before being placed in the output, so a
     policy that contains ``<script>`` or ``&`` cannot inject markup. Offsets
     refer to the exact (unescaped) segment text, and we escape *slices* rather
     than the whole string, so the offsets stay valid.
  2. **Only our own controlled tags/attributes are injected** (a ``<mark>`` with
     a colour from the fixed ATTENTION_COLORS palette and an escaped title), so
     enabling ``unsafe_allow_html`` on the result is safe.
"""

import html

from .config import ATTENTION_COLORS, tier_of


def merge_evidence_spans(evidence_by_category):
    """Flatten per-category evidence into non-overlapping, coloured spans.

    Different categories can produce spans that overlap in the text (e.g.
    "delete" for both Data Retention and User Control). We keep the highest
    local-contribution span and drop overlaps, so highlighting stays readable.
    Each returned span carries its ``category`` and tier ``color``.
    """
    spans = []
    for category, res in evidence_by_category.items():
        color = ATTENTION_COLORS[tier_of(category)]
        for s in res.get("spans", []):
            spans.append({**s, "category": category, "color": color})

    spans.sort(key=lambda s: (-s["contribution"], s["start"]))
    chosen = []
    for sp in spans:
        if any(sp["start"] < c["end"] and c["start"] < sp["end"]
               for c in chosen):
            continue
        chosen.append(sp)
    return sorted(chosen, key=lambda s: s["start"])


def highlight_html(text, spans):
    """Return HTML for ``text`` with ``spans`` wrapped in coloured <mark>s.

    All literal text is HTML-escaped; only fixed-palette colours and escaped
    titles are injected.
    """
    out, cursor = [], 0
    for sp in sorted(spans, key=lambda s: s["start"]):
        if sp["start"] < cursor:               # safety: skip any overlap
            continue
        out.append(html.escape(text[cursor:sp["start"]]))
        title = f'{sp["category"]} (contribution {sp["contribution"]:.2f})'
        out.append(
            f'<mark style="background:{sp["color"]}40;color:inherit;'
            f'border-bottom:2px solid {sp["color"]};border-radius:3px;'
            f'padding:0 3px;" '
            f'title="{html.escape(title)}">'
            f'{html.escape(text[sp["start"]:sp["end"]])}</mark>')
        cursor = sp["end"]
    out.append(html.escape(text[cursor:]))
    return "".join(out)


def render_segment(text, evidence_by_category):
    """Convenience: merge a segment's evidence and return highlighted HTML."""
    return highlight_html(text, merge_evidence_spans(evidence_by_category))
