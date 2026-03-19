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

REVIEW_REQUIRE = """###Review requirements###
You must:
1. Provide a score for each criterion.
2. Provide AT LEAST three strengths of the paper.
3. Provide AT LEAST two weaknesses or concerns.
4. Provide a concise summary comment explaining your reasoning.
5. Provide a final decision: "Accept" or "Reject".
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
  "summary_comment": "..."
}
"""
