import json

from scripts.audit_experiment_invalid_json import (
    audit_summary_data,
    classify_review,
    discover_summaries,
)


def _valid_review():
    return {
        "reviewer": "Reviewer",
        "decision": "accept",
        "scores": {key: 4 for key in ("novelty", "soundness", "significance", "evaluation", "clarity")},
        "strengths": ["Clear method."],
        "weaknesses": ["Limited evidence."],
    }


def test_classify_review_separates_parse_and_schema_errors():
    assert classify_review(_valid_review()) == "valid"
    assert classify_review({"raw": "bad", "parse_error": True}) == "parse_error"
    assert classify_review({}) == "invalid_schema"
    assert classify_review("not an object") == "not_json_object"


def test_audit_counts_invalid_and_missing_reviews():
    summary = {
        "timestamp": "test",
        "provider": "openrouter",
        "model": "model",
        "conditions": {
            "C1": {"agents": ["reviewer_nopersona"]},
            "C3": {"agents": ["reviewer_nopersona"] * 3},
        },
        "papers": [
            {
                "paper_id": "paper-1",
                "conditions": {
                    "C1": {"result": {"reviewers": [_valid_review()]}},
                    "C3": {
                        "result": {
                            "reviewers": [
                                _valid_review(),
                                {"raw": "bad", "parse_error": True},
                                {},
                            ]
                        }
                    },
                },
            }
        ],
    }

    report = audit_summary_data(summary, "summary.json")
    assert report["stored_reviews"] == 4
    assert report["valid_reviews"] == 2
    assert report["parse_errors"] == 1
    assert report["schema_errors"] == 1
    assert report["missing_reviews"] == 2
    assert report["affected_pairs"] == 1
    assert report["expected_reviews"] == 4
    assert report["repair_runs"] == 1
    assert report["by_condition"]["C3"]["affected_pairs"] == 1
    assert report["issues"][0]["condition_id"] == "C3"


def test_audit_supports_legacy_single_condition_summary():
    summary = {
        "timestamp": "legacy",
        "condition": {
            "nagent": 3,
            "reviewers": ["reviewer_a", "reviewer_b", "reviewer_c"],
        },
        "papers": [
            {
                "paper_id": "paper-1",
                "result": {
                    "reviewers": [
                        _valid_review(),
                        _valid_review(),
                        {"raw": "bad", "parse_error": True},
                    ]
                },
            }
        ],
    }

    report = audit_summary_data(summary, "legacy.json")
    assert report["audit_format"] == "single_condition"
    assert report["condition_count"] == 1
    assert report["expected_reviews"] == 3
    assert report["valid_reviews"] == 2
    assert report["parse_errors"] == 1
    assert report["affected_pairs"] == 1


def test_discover_summaries_supports_recursive_directories(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    summary = nested / "experiment_advanced_summary_test.json"
    summary.write_text(json.dumps({}), encoding="utf-8")
    (nested / "other.json").write_text("{}", encoding="utf-8")

    assert discover_summaries([str(tmp_path)], recursive=False) == []
    assert discover_summaries([str(tmp_path)], recursive=True) == [summary.resolve()]
