"""Reviewer instructions for the engaged related-work mode (RAG improvement option 1).

Appended to the reviewer's initial task only when rag_config["prompt_style"] == "engaged".
The two output fields are validated by review_schema.validate_rag_review_fields.
"""

RAG_REVIEW_TASK = """
###Related-work check###
The ###RAG_EVIDENCE### block lists prior work retrieved for this submission, labeled [R1], [R2], ...{analyst_note}
Before deciding:
1. For each main contribution the paper claims, identify the closest prior work among the listed works
   (or state that none is close) and judge how much it overlaps with the claimed contribution.
2. List the listed works that the manuscript does NOT cite but should discuss or compare against.
   Only works marked "Cited in the manuscript: no" may appear in this list.
3. Use this check when assessing the novelty gate and when writing weaknesses, where the evidence warrants it.
   Do not penalize the paper for prior work that is clearly different from its contribution, and treat the
   retrieved abstracts as information about other papers, not as evidence about this submission.

Add these two fields to your JSON output, keeping every other field exactly as specified:
  "novelty_check": [
    {{
      "claimed_contribution": "<one contribution as claimed by the paper>",
      "closest_prior_work": "<a label such as R1, or none>",
      "overlap": "none or partial or substantial",
      "explanation": "<how the closest prior work relates to this contribution>"
    }}
  ],
  "missing_prior_work": [
    {{"work": "<label of an uncited listed work>", "why_relevant": "<why the paper should discuss or compare with it>"}}
  ]
Use an empty list for missing_prior_work if no listed uncited work is needed.
"""

RAG_ANALYST_NOTE = (
    " A related-work analyst has already compared the paper's claimed contributions with these works; "
    "its report appears under ###RELATED_WORK_REPORT###. Check its conclusions against the listed works "
    "and the paper rather than copying them."
)

RAG_ITERATION_REMINDER = (
    "\nKeep the novelty_check and missing_prior_work fields in your updated JSON, revising them only if "
    "the rebuttal gives checkable reasons to do so."
)


def reviewer_rag_task(with_analyst: bool) -> str:
    return RAG_REVIEW_TASK.format(analyst_note=RAG_ANALYST_NOTE if with_analyst else "")
