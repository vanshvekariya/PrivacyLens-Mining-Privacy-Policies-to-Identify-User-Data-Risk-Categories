"""Tests for evidence -> escaped HTML highlighting (privacylens.highlight)."""

from privacylens.highlight import (
    highlight_html,
    merge_evidence_spans,
    render_segment,
)


def _span(text, start, end, contribution, feature=None):
    return {"text": text, "start": start, "end": end,
            "feature": feature or text, "tfidf_value": 0.1,
            "coefficient": contribution / 0.1, "contribution": contribution}


def test_all_literal_text_is_escaped():
    text = "we <b>share</b> data & more"
    # highlight the word 'share' (chars 6..11)
    spans = [{**_span("share", 6, 11, 0.5), "category": "Third-Party Sharing",
              "color": "#d64545"}]
    out = highlight_html(text, spans)
    # raw angle brackets / ampersands from the policy must be escaped
    assert "<b>" not in out
    assert "&lt;b&gt;" in out
    assert "&amp; more" in out
    # our own mark tag is present around the highlighted word
    assert "<mark" in out and ">share</mark>" in out


def test_offsets_index_original_text():
    text = "please delete your account"
    spans = [{**_span("delete", 7, 13, 0.9),
              "category": "User Control & Deletion", "color": "#d9922a"}]
    out = highlight_html(text, spans)
    assert ">delete</mark>" in out
    # text before/after the mark is preserved
    assert out.startswith("please ")
    assert out.endswith(" your account")


def test_overlapping_cross_category_spans_resolved():
    ev = {
        "Data Retention": {"spans": [
            {**_span("delete", 0, 6, 0.3)}]},
        "User Control & Deletion": {"spans": [
            {**_span("delete", 0, 6, 0.9)}]},
    }
    merged = merge_evidence_spans(ev)
    # the two spans overlap exactly; only the higher-contribution one survives
    assert len(merged) == 1
    assert merged[0]["category"] == "User Control & Deletion"


def test_render_segment_no_spans_is_plain_escaped_text():
    ev = {"Data Security": {"spans": []}}
    out = render_segment("no & evidence here", ev)
    assert out == "no &amp; evidence here"
    assert "<mark" not in out
