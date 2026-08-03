# Error Taxonomy

Pick one `failure_type` from this list for each row. If more than one fits,
pick the one with the lowest number.

| # | failure_type | when to use it |
| --- | --- | --- |
| 1 | `negation_or_permission_flip` | The text is negated or permission-granting ("we do **not** share...", "you **may** opt out") and the model missed the flip. |
| 2 | `cross_category_confusion` | The model predicted a related but wrong category (e.g. Data Retention instead of Data Security). |
| 3 | `near_threshold` | At least one gold label sits just above or below the model's decision threshold. |
| 4 | `multi_label_under_prediction` | Dominant label is right, but one or more secondary gold labels are missing. |
| 5 | `multi_label_over_prediction` | Dominant label is right, but one or more spurious extra labels were predicted. |
| 6 | `residual_boilerplate` | Contact info, jurisdiction, cookie banner, or similar filler that was mislabeled. |
| 7 | `out_of_distribution_phrasing` | Legal jargon, non-US regulatory phrasing, or unusual wording rarely seen in training. |
| 8 | `content_ambiguity` | The segment is genuinely ambiguous; two reasonable annotators could disagree. |
| 9 | `other` | None of the above. Add a short note in the `notes` column. |
