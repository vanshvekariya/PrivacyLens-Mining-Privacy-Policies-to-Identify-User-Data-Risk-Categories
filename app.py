"""PrivacyLens prototype - a simple web demo (formal_proposal_v2.md s7).

Paste (or upload) a privacy policy; for each segment the app shows the
predicted privacy-practice categories, the attention tier, a 1-5 star reading
priority, whether the segment needs human review (and why), a cautious
plain-English explanation, and the highlighted text that supported the
prediction.

This file is a THIN renderer: all analysis lives in ``privacylens.pipeline``
and all highlighting/escaping in ``privacylens.highlight``. Run with:

    python src/train_prototype_model.py     # once, to build the model artifact
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from privacylens.config import ATTENTION_COLORS, tier_of                # noqa: E402
from privacylens.highlight import render_segment                        # noqa: E402
from privacylens.pipeline import (                                      # noqa: E402
    MAX_POLICY_CHARS,
    analyze_policy,
    load_prototype_model,
)
from privacylens.templates import DISCLAIMER                            # noqa: E402

MAX_UPLOAD_MB = 2
MIN_USEFUL_CHARS = 40

SAMPLE_POLICY = """\
Your Privacy Choices

We collect the information you provide when you create an account, such as your
name, email address, and phone number. We also automatically collect device
identifiers, IP address, and cookies when you use our services.

We may share your personal information with third-party advertising partners and
analytics providers in order to measure and personalize the ads you see.

We retain your personal data for as long as your account is active and as needed
to comply with our legal obligations.

You can access, update, or delete your personal information at any time from your
account settings, and you may opt out of marketing communications.

We use encryption to protect your data in transit. However, no method of
electronic storage is completely secure.

We may update this Privacy Policy from time to time. We will notify you of
material changes by posting the new policy on this page.
"""


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_prototype_model()


def stars_text(n):
    return "\u2605" * n + "\u2606" * (5 - n)


def category_badges_html(profile):
    import html as _html
    chips = []
    for c in profile["all_categories"]:
        color = ATTENTION_COLORS[tier_of(c)]
        chips.append(
            f'<span style="background:{color}33;border:1px solid {color};'
            f'color:inherit;border-radius:10px;padding:1px 8px;'
            f'margin-right:6px;font-size:0.85em;">{_html.escape(c)} '
            f'<span style="opacity:.7">({tier_of(c)})</span></span>')
    return "".join(chips) or '<span style="color:#888">no category above '\
        'threshold</span>'


def render_card(r, total, show_context, doc_segments):
    prio = r["priority"]
    idx = r.get("segment_index", 0)
    st.markdown(f"**Segment {idx + 1} of {total}** &nbsp; "
                f"{stars_text(prio['stars'])} &nbsp; "
                f"*{prio['priority_band']}*", unsafe_allow_html=True)

    # highlighted segment text (all escaped inside render_segment)
    st.markdown(
        f'<div style="line-height:1.6;padding:6px 0;">'
        f'{render_segment(r["text"], r["evidence_by_category"])}</div>',
        unsafe_allow_html=True)

    st.markdown(category_badges_html(r["attention_profile"]),
                unsafe_allow_html=True)

    if prio["review_required"]:
        reason = prio["review_reasons"][0] if prio["review_reasons"] else ""
        st.warning(f"Needs review ({prio['review_severity']}): {reason}")

    for e in r["explanations"]:
        st.caption(f"**{e['category']}** - {e['explanation']}")

    if show_context:
        prev_t = (doc_segments[idx - 1]["text"][:140] + "..."
                  if idx > 0 else "(start of policy)")
        next_t = (doc_segments[idx + 1]["text"][:140] + "..."
                  if idx + 1 < len(doc_segments) else "(end of policy)")
        with st.expander("Surrounding context"):
            st.caption(f"Previous: {prev_t}")
            st.caption(f"Next: {next_t}")
    st.divider()


def main():
    st.set_page_config(page_title="PrivacyLens", layout="wide")
    st.title("PrivacyLens - privacy-policy reading assistant")
    st.caption("Paste a privacy policy to see which parts deserve attention "
               "first. This is a research prototype, not legal advice.")

    try:
        model = get_model()
    except FileNotFoundError:
        st.error("Model artifact not found. Run "
                 "`python src/train_prototype_model.py` first.")
        st.stop()
    except Exception as exc:                       # stale/mismatched artifact
        st.error(f"Could not load the model: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Input")
        uploaded = st.file_uploader("Upload a .txt policy", type=["txt"])
        view = st.radio("Order", ["Priority (most important first)",
                                  "Document order"])
        show_context = st.checkbox("Show surrounding context", value=False)
        # Set the text-area's session state BEFORE the widget is created, so a
        # click here survives the rerun triggered by the Analyze button.
        if st.button("Load sample policy"):
            st.session_state.policy_text = SAMPLE_POLICY

    if "policy_text" not in st.session_state:
        st.session_state.policy_text = ""

    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File too large (max {MAX_UPLOAD_MB} MB).")
            st.stop()
        try:
            st.session_state.policy_text = uploaded.read().decode(
                "utf-8", errors="replace")
        except Exception:
            st.error("Could not read the uploaded file as UTF-8 text.")
            st.stop()

    text = st.text_area("Privacy-policy text", key="policy_text", height=260,
                        placeholder="Paste privacy-policy text here...")

    if not st.button("Analyze", type="primary"):
        st.stop()

    if not text or len(text.strip()) < MIN_USEFUL_CHARS:
        st.info("Please paste a longer passage of privacy-policy text "
                f"(at least {MIN_USEFUL_CHARS} characters).")
        st.stop()
    if len(text) > MAX_POLICY_CHARS:
        st.error(f"Text too long (max {MAX_POLICY_CHARS:,} characters). "
                 "Please analyze a shorter excerpt.")
        st.stop()

    out = analyze_policy(text, model)
    results = out["segments"]
    summary = out["summary"]
    if not results:
        st.info("No analyzable segments were found in that text.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segments", summary["n_segments"])
    c2.metric("Very important (5*)", summary["stars"][5])
    c3.metric("Needs review", summary["review_required"])
    c4.metric("Model abstained", summary["abstained"])

    st.info(DISCLAIMER)

    ordered = (sorted(results, key=lambda r: (-r["priority"]["stars"],
                                              r["segment_index"]))
               if view.startswith("Priority") else results)

    for r in ordered:
        render_card(r, summary["n_segments"], show_context, results)


if __name__ == "__main__":
    main()
