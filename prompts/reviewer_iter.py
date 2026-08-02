reviewer_iteration = """Update your review after considering the author's rebuttal and, when present, the review-style analysis.

###Rebuttal assessment rules###
Evaluate the paper as submitted, not as the author promises to revise it.
For each concern in your previous review, determine whether it is resolved,
partially resolved, or unresolved.

A concern is resolved only when the rebuttal:
- identifies specific evidence already present in the submitted paper;
- gives a logically complete clarification that can be checked against the
  submitted paper; or
- demonstrates that the concern resulted from a reviewer misunderstanding.

The following do NOT by themselves resolve a concern:
- agreeing with the reviewer, apologizing, or expressing willingness to revise;
- promising to clarify wording, add discussion, cite work, reframe a claim, or
  make another future edit;
- promising new experiments, analyses, proofs, or results that are not included
  in the material available to you; or
- sounding confident, cooperative, persuasive, or professional.

If an acknowledged issue is minor and fixable without substantial new work,
keep it classified as minor. Do not raise a score merely because the author
agrees to fix it. Such an issue also should not cause rejection unless it
materially affects an acceptance gate.

Change a criterion score or decision only when checkable rebuttal information
changes your substantive assessment of the submitted paper. Preserve the prior
score when the response merely promises a future change. If you change any
score or acceptance-gate result, explain the evidence-based reason in the
corresponding justification or summary comment.

###Review-style analysis rules###
The review-style analysis is editorial guidance only. Use it to make the review
more specific, concise, natural, constructive, and consistent with human
conference reviews. It must not alter technical judgments, scores, weakness
severity, acceptance gates, or the final decision. Do not add fake uncertainty,
personal anecdotes, errors, or unsupported paper details to appear human.

Maintain exactly the same JSON output format as your initial review.
You MUST always return the complete JSON object again, in exactly the same format as your
initial review — even if your decision, scores, strengths, and weaknesses are all unchanged.
Never reply with plain-text confirmations such as "my review remains unchanged"; always
re-emit the full JSON.
"""
