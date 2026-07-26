from unittest.mock import patch

from agents import Reviewer
from mas_loop import _normalize_review_weaknesses
from prompts.ai_detector import ai_detector
from prompts.author import author
from prompts.reviewer_a import reviewer_a
from prompts.reviewer_iter import reviewer_iteration


class _FakeClient:
    def complete(self, system_prompt, messages):
        return "{}"


def test_weaknesses_remain_plain_strings_with_parallel_details():
    assert '"weaknesses": [\n    "...",\n    "..."' in reviewer_a
    assert '"weakness_details": [' in reviewer_a
    assert '"weakness": "<exact text from weaknesses[0]>"' in reviewer_a


def test_reviewer_system_prompt_uses_general_top_tier_standard():
    with patch("agents.create_llm_client", return_value=_FakeClient()):
        reviewer = Reviewer(
            paper="# Paper",
            reviewer_type="reviewer_a",
            topic="NLP",
            api_key="unused",
        )

    assert "###Target Conference###" not in reviewer.persona
    assert "selective top-tier machine" in reviewer.persona
    assert "learning conference" in reviewer.persona


def test_old_object_weaknesses_are_normalized_for_evaluation():
    review = {
        "weaknesses": [
            {
                "concern": "The central comparison omits the strongest baseline.",
                "severity": "major",
                "evidence": "Table 2",
                "affected_claim": "state-of-the-art performance",
                "fixable_without_substantial_new_work": False,
            }
        ]
    }

    normalized = _normalize_review_weaknesses(review)

    assert normalized["weaknesses"] == [
        "The central comparison omits the strongest baseline."
    ]
    assert normalized["weakness_details"][0]["severity"] == "major"


def test_rebuttal_rules_do_not_reward_promised_fixes():
    assert "Evaluate the paper as submitted" in reviewer_iteration
    assert "Do not raise a score merely because the author" in reviewer_iteration
    assert "promises a future change" in reviewer_iteration
    assert "editorial guidance only" in reviewer_iteration
    assert "must not alter technical judgments" in reviewer_iteration


def test_author_separates_existing_evidence_from_future_revisions():
    assert '"supporting_evidence_in_submission"' in author
    assert '"proposed_future_revisions"' in author
    assert "not part of the current submission" in author


def test_ai_detector_is_a_non_substantive_conference_style_evaluator():
    assert "CONFERENCE REVIEW STYLE EVALUATOR" in ai_detector
    assert '"conference_review_fit_score"' in ai_detector
    assert '"conference_style_mismatches"' in ai_detector
    assert "must not recommend raising or lowering any score" in ai_detector
    assert "Do not penalize these structural" in ai_detector
    assert "Preserve technical meaning" in ai_detector
