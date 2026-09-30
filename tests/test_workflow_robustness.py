import copy
import json
from unittest.mock import patch

import pytest

import agents
import mas_loop
from review_schema import REVIEW_SCORE_KEYS, parse_json_object, review_is_valid, workflow_is_complete


REVIEW = {"decision": "reject", "scores": {key: 3 for key in REVIEW_SCORE_KEYS},
          "strengths": ["Clear objective"], "weaknesses": ["Missing baseline"]}
REBUTTAL = {"responses": [{"reviewer": "Reviewer 1 (Neutral)", "response": "Section 3 reports the baseline.",
            "main_issues_identified": ["Missing baseline"], "supporting_evidence_in_submission": [],
            "proposed_future_revisions": []}]}
CONFERENCE = {venue: {"fit_score": 3} for venue in ("ICML", "ICLR", "NeurIPS")}


class Replies:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def complete(self, persona, messages):
        self.requests.append((persona, copy.deepcopy(messages)))
        value = next(self.replies)
        if isinstance(value, BaseException):
            raise value
        return value if isinstance(value, str) else json.dumps(value)


def run_workflow(reviewer, author, conference):
    with patch("agents.create_llm_client", side_effect=[reviewer, author, conference]), \
         patch("agents.time.sleep"):
        return mas_loop.main("# Paper", n_iter=3, reviewer_types=["reviewer_nopersona"],
                             run_citation_check=False, enable_ai_detector=False)


def test_schema_retry_recovers_same_prompt_without_polluting_conversation():
    reviewer = Replies([REVIEW, REVIEW, REVIEW])
    author = Replies([{}, REBUTTAL, REBUTTAL])
    conference = Replies([CONFERENCE])
    result = run_workflow(reviewer, author, conference)
    assert workflow_is_complete(result, 1, 3)
    assert result["turn_failures"] == []
    first_author = next(item for item in result["turn_outcomes"] if item["role"] == "author")
    assert [item["status"] for item in first_author["attempts"]] == ["invalid_output", "valid"]
    assert author.requests[0] == author.requests[1]
    assert all(message["content"] != "{}" for message in author.requests[-1][1])
    assert len(result["iterations"]) == 3
    assert result["iterations"][1]["reviewers"][0]["review"] == REVIEW
    assert json.loads(result["iterations"][1]["reviewers"][0]["author_response"]) == REBUTTAL


def test_author_ablation_has_no_required_author_turns():
    reviewer, author, conference = Replies([REVIEW] * 3), Replies([]), Replies([CONFERENCE])
    with patch("agents.create_llm_client", side_effect=[reviewer, author, conference]):
        result = mas_loop.main("# Paper", n_iter=3, reviewer_types=["reviewer_nopersona"],
                               enable_author_rebuttal=False, run_citation_check=False)
    assert not author.requests
    assert workflow_is_complete(result, 1, 3, enable_author_rebuttal=False)
    assert not workflow_is_complete(result, 1, 3)  # Cannot masquerade as a trigger run.
    assert len(result["iterations"]) == 3
    assert all(turn["role"] != "author" for turn in result["turn_outcomes"])
    assert "No rebuttal" in result["iterations"][1]["reviewers"][0]["author_response"]


def test_trailing_comma_tolerance_preserves_string_contents():
    raw = '```json\n{"text": "literal ,} and ,]", "items": [1,2,],}\n```'
    expected = {"text": "literal ,} and ,]", "items": [1, 2]}
    assert parse_json_object(raw) == expected
    assert agents._parse_required_json_object(raw) == expected
    assert mas_loop._parse_json(raw) == expected


def test_agent_and_pipeline_accept_same_trailing_comma_syntax():
    raw = json.dumps(REVIEW)[:-1] + ',}'
    reviewer, author, conference = Replies([raw] * 3), Replies([REBUTTAL] * 2), Replies([CONFERENCE])
    result = run_workflow(reviewer, author, conference)
    assert workflow_is_complete(result, 1, 3)
    assert len(reviewer.requests) == 3


def test_exhausted_middle_rebuttal_stops_before_any_later_review():
    reviewer = Replies([REVIEW, REVIEW, REVIEW])
    author = Replies([{}] * 4)
    conference = Replies([CONFERENCE])
    result = run_workflow(reviewer, author, conference)
    assert result["workflow_status"] == "failed"
    assert len(reviewer.requests) == 1
    assert len(author.requests) == 4
    assert not conference.requests
    assert len(result["turn_failures"]) == 1
    failure = result["turn_failures"][0]
    assert (failure["role"], failure["iteration"], failure["reviewer_index"]) == ("author", 2, 0)
    assert len(failure["attempts"]) == 4
    # Even fabricated valid final reviews cannot make its trajectory complete.
    result["reviewers"] = [REVIEW]
    result["conference"] = CONFERENCE
    assert not workflow_is_complete(result, 1, 3)


def test_missing_or_misaddressed_rebuttal_is_retried_with_original_context():
    wrong = copy.deepcopy(REBUTTAL)
    wrong["responses"][0]["reviewer"] = "Reviewer 2"
    reviewer, author, conference = Replies([REVIEW] * 3), Replies([wrong, REBUTTAL, REBUTTAL]), Replies([CONFERENCE])
    result = run_workflow(reviewer, author, conference)
    assert workflow_is_complete(result, 1, 3)
    assert author.requests[0] == author.requests[1]


def test_transient_timeout_uses_fixed_budget_but_auth_failure_is_terminal():
    reviewer, author, conference = Replies([TimeoutError(), REVIEW, REVIEW, REVIEW]), Replies([REBUTTAL] * 2), Replies([CONFERENCE])
    assert workflow_is_complete(run_workflow(reviewer, author, conference), 1, 3)
    assert reviewer.requests[0] == reviewer.requests[1]
    reviewer = Replies([RuntimeError("auth failure")])
    result = run_workflow(reviewer, Replies([]), Replies([]))
    assert len(reviewer.requests) == 1
    assert result["turn_failures"][0]["attempts"][0]["error_type"] == "RuntimeError"


@pytest.mark.parametrize("score", [True, float("nan"), float("inf"), 0, 6])
def test_shared_review_schema_rejects_unusable_scores(score):
    review = copy.deepcopy(REVIEW)
    review["scores"]["evaluation"] = score
    assert not review_is_valid(review)


def test_shared_schema_does_not_silently_repair_valuation_typo():
    review = copy.deepcopy(REVIEW)
    review["scores"]["valuation"] = review["scores"].pop("evaluation")
    assert not review_is_valid(review)
