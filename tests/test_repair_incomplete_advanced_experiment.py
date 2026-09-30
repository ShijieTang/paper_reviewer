import json
from pathlib import Path
from unittest.mock import patch

from scripts.repair_incomplete_advanced_experiment import (
    audit_summary,
    is_valid_review,
    repair_summary,
    result_is_complete,
)


def _review(name="Reviewer - No Persona"):
    return {
        "reviewer": name,
        "decision": "accept",
        "scores": {key: 4 for key in ("novelty", "soundness", "significance", "evaluation", "clarity")},
        "strengths": ["clear"],
        "weaknesses": ["limited"],
    }


def _summary(result):
    return {
        "timestamp": "old",
        "provider": "openrouter",
        "model": "model/name",
        "conditions": {
            "C1": {
                "desc": "one reviewer",
                "agents": ["reviewer_nopersona"],
                "n_iter": 1,
                "agenttype": "N",
                "enable_rag": False,
            }
        },
        "papers": [
            {
                "paper_id": "paper-1",
                "conditions": {
                    "C1": {
                        "result_file": "old.txt",
                        "result": result,
                    }
                },
            }
        ],
    }


def test_valid_review_requires_evaluation_fields():
    assert is_valid_review(_review())
    assert not is_valid_review({"parse_error": True, "raw": "..."})
    assert not is_valid_review({**_review(), "decision": "maybe"})
    assert not is_valid_review({**_review(), "strengths": "clear"})


def test_audit_reports_invalid_or_missing_reviews():
    summary = _summary({"reviewers": [{"parse_error": True}]})
    issues = audit_summary(summary)
    assert issues == [
        {
            "paper_id": "paper-1",
            "condition_id": "C1",
            "expected_reviews": 1,
            "valid_reviews": 0,
            "stored_reviews": 1,
        }
    ]
    assert result_is_complete({"reviewers": [_review()]}, summary["conditions"]["C1"])


def test_repair_reruns_only_incomplete_condition_and_saves_replacement(tmp_path):
    summary = _summary({"reviewers": [{"parse_error": True}]})
    papers = [
        {
            "paper_id": "paper-1",
            "paper_dir": "data/pdf/paper-1.pdf",
            "topic": "NLP",
        }
    ]
    replacement = {"reviewers": [_review()], "rag_package": None}

    with patch(
        "scripts.repair_incomplete_advanced_experiment.load_markdown",
        return_value="paper text",
    ), patch(
        "scripts.repair_incomplete_advanced_experiment.run_condition",
        return_value=replacement,
    ) as run, patch(
        "scripts.repair_incomplete_advanced_experiment.save_result",
        side_effect=lambda **kwargs: str(tmp_path / "replacement.txt"),
    ) as save:
        repaired, unresolved = repair_summary(
            summary,
            papers,
            api_key="key",
            output_dir=str(tmp_path),
            md_dir="data/md",
        )

    assert unresolved == []
    assert run.call_count == 1
    assert save.call_count == 1
    entry = repaired["papers"][0]["conditions"]["C1"]
    assert entry["repaired"] is True
    assert entry["replaces_result_file"] == "old.txt"
    assert entry["result"] == replacement
    assert repaired["repair"]["replacement_count"] == 1


def test_repair_retries_incomplete_condition_until_complete(tmp_path):
    summary = _summary({"reviewers": []})
    papers = [{"paper_id": "paper-1", "paper_dir": "paper-1.pdf", "topic": "NLP"}]
    incomplete = {"reviewers": []}
    complete = {"reviewers": [_review()]}

    with patch(
        "scripts.repair_incomplete_advanced_experiment.load_markdown",
        return_value="paper text",
    ), patch(
        "scripts.repair_incomplete_advanced_experiment.run_condition",
        side_effect=[incomplete, complete],
    ) as run, patch(
        "scripts.repair_incomplete_advanced_experiment.save_result",
        return_value=str(tmp_path / "replacement.txt"),
    ):
        repaired, unresolved = repair_summary(
            summary,
            papers,
            api_key="key",
            output_dir=str(tmp_path),
            md_dir="data/md",
            max_attempts=2,
        )

    assert run.call_count == 2
    assert unresolved == []
    assert repaired["papers"][0]["conditions"]["C1"]["result"] == complete
