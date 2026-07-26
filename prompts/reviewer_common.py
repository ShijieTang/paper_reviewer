TASK = """###Task###
You will receive the FULL paper below. Your task is to review the paper and produce a decision.
"""

EVAL_CRIT = """###Evaluation criteria###
Score each from 1–5:
  Novelty — Has this idea or approach already been done before?
  Soundness — Are the logic, assumptions, and mathematical reasoning correct?
  Significance — Does this result meaningfully advance the field?
  Evaluation — Are the experiments, data, and comparisons convincing and fair?
  Clarity — Is the paper clearly written and easy to understand?

Score interpretation:
  1 = very poor
  2 = weak
  3 = acceptable
  4 = strong
  5 = excellent
"""

DECISION_STANDARD = """###Decision standard###
Evaluate the submission using the standards of a selective top-tier machine
learning conference.
Do not assume that the paper should be accepted or rejected.

First assess each acceptance gate independently:

1. Soundness: Are the central claims technically supported?
2. Novelty: Is the contribution sufficiently distinct from prior work,
   based only on the evidence available?
3. Significance: Is the demonstrated contribution meaningful for a top-tier
   machine learning conference?
4. Evaluation: Do the experiments, proofs, or analyses adequately support
   the central claims?
5. Presentation: Is the work sufficiently clear to permit reliable assessment?

Decision:
- ACCEPT when the paper clearly meets the overall top-tier standard and no
  failed gate materially undermines its central contribution.
- REJECT when one or more major weaknesses prevent the paper from meeting that
  standard.
- Do not reject for minor flaws, missing polish, optional experiments, or
  improvements that would strengthen—but are not necessary to validate—the work.
- Do not accept based only on an interesting idea, potential future impact,
  persuasive writing, or author promises.
- Uncertainty is not automatically a reason to reject. State what is uncertain
  and decide whether it is central enough to change the recommendation.
- Consider strengths and weaknesses jointly; do not mechanically average scores.
"""

EVAL_CRIT += DECISION_STANDARD

REVIEW_REQUIRE = """###Review requirements###
You must:
1. Provide a score for each criterion.
2. Provide AT LEAST three strengths of the paper.
3. Provide AT LEAST two weaknesses or concerns.
4. Keep each item in "weaknesses" as a plain text string. For every weakness,
   provide a corresponding object in "weakness_details" at the same list index
   reporting:
   - severity: "minor" or "major"
   - evidence from the paper
   - affected claim
   - whether it is fixable without substantial new work
5. Before deciding, state whether each acceptance gate passes or fails and
   briefly justify the assessment.
6. Provide a concise summary comment explaining your reasoning.
7. Provide a final decision: "Accept" or "Reject". The final decision must
   follow the acceptance-gate results.
"""

OUTPUT_REQUIRE = """###Output requirements###
- Output MUST be valid JSON.
- Do NOT include explanations outside the JSON.
- Do NOT include markdown or additional text.
"""

OUTPUT_FORMAT = """\
###Output format###
Return the JSON in exactly the following format:
{
  "reviewer": "<reviewer name>",
  "decision": "Accept or Reject",
  "scores": {
    "novelty": <integer 1-5>,
    "soundness": <integer 1-5>,
    "significance": <integer 1-5>,
    "evaluation": <integer 1-5>,
    "clarity": <integer 1-5>
  },
  "strengths": [
    "...",
    "...",
    "..."
  ],
  "weaknesses": [
    "...",
    "..."
  ],
  "weakness_details": [
    {
      "weakness": "<exact text from weaknesses[0]>",
      "severity": "minor or major",
      "evidence": "...",
      "affected_claim": "...",
      "fixable_without_substantial_new_work": <true or false>
    },
    {
      "weakness": "<exact text from weaknesses[1]>",
      "severity": "minor or major",
      "evidence": "...",
      "affected_claim": "...",
      "fixable_without_substantial_new_work": <true or false>
    }
  ],
  "acceptance_gates": {
    "soundness": {
      "result": "pass or fail",
      "justification": "..."
    },
    "novelty": {
      "result": "pass or fail",
      "justification": "..."
    },
    "significance": {
      "result": "pass or fail",
      "justification": "..."
    },
    "evaluation": {
      "result": "pass or fail",
      "justification": "..."
    },
    "presentation": {
      "result": "pass or fail",
      "justification": "..."
    }
  ],
  "summary_comment": "..."
}
"""
