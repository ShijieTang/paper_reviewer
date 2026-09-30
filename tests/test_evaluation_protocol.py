import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from eval import evaluation
from eval.evaluation_protocol import EMBED_MODEL, EMBED_REVISION
from review_schema import review_is_valid
from scripts.audit_experiment_invalid_json import classify_review
from scripts.repair_incomplete_advanced_experiment import is_valid_review


def review():
    return {
        "reviewer": "Reviewer 1 (Neutral)", "decision": "accept",
        "scores": {name: 4 for name in evaluation._OUR_REVIEW_SCORE_FIELDS},
        "strengths": ["Clear method"], "weaknesses": ["Limited evidence"],
    }


@pytest.mark.parametrize("defect", ["typo", "missing", "boolean", "nan", "out_of_range", "empty"])
def test_audit_repair_and_evaluation_share_review_validity(defect):
    value = review()
    assert review_is_valid(value) and is_valid_review(value) and classify_review(value) == "valid"
    if defect == "typo":
        value["scores"]["valuation"] = value["scores"].pop("evaluation")
    elif defect == "missing":
        value["scores"].pop("evaluation")
    elif defect == "boolean":
        value["scores"]["evaluation"] = True
    elif defect == "nan":
        value["scores"]["evaluation"] = float("nan")
    elif defect == "out_of_range":
        value["scores"]["evaluation"] = 6
    else:
        value["weaknesses"] = []
    assert not review_is_valid(value)
    assert not is_valid_review(value)
    assert classify_review(value) == "invalid_schema"
    assert evaluation._score_from_reviewers([value]) is None
    assert evaluation._collected_reviewers([value]) == []


def frozen_summary():
    return {
        "experiment": "trigger_rag_screening",
        "expected_evaluator_sha256": hashlib.sha256(Path(evaluation.__file__).read_bytes()).hexdigest(),
        "expected_src_sha256": hashlib.sha256(Path(evaluation.__file__).with_name("SRC.py").read_bytes()).hexdigest(),
        "expected_embed_model": EMBED_MODEL,
        "expected_embed_revision": EMBED_REVISION,
    }


@pytest.mark.parametrize("field", ["expected_evaluator_sha256", "expected_src_sha256", "expected_embed_model", "expected_embed_revision"])
def test_changed_metric_rejected_before_model_load(tmp_path, field):
    summary = frozen_summary()
    summary[field] = "changed"
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    with patch("eval.evaluation.load_model") as load:
        with pytest.raises(ValueError, match=field):
            evaluation.run_evaluation("unused.json", None, None, str(tmp_path), exp_summary_path=str(path))
    load.assert_not_called()


def test_fixed_metric_accepts_frozen_model_identity():
    evaluation._validate_frozen_metric(frozen_summary(), EMBED_MODEL, EMBED_REVISION)


def test_incomplete_condition_is_reported_not_scored(tmp_path):
    papers = {"p": {"title": "Paper", "conference": "ICLR", "score": 7,
                    "accept_or_not": "accept", "reviews": [{"strengths": ["S"], "weaknesses": ["W"]}]}}
    summary = {"conditions": {"C3": {"agents": ["N"] * 3}},
               "papers": [{"paper_id": "p", "conditions": {"C3": {"result": {"reviewers": [review()]}}}}]}
    papers_path, summary_path = tmp_path / "papers.json", tmp_path / "summary.json"
    papers_path.write_text(json.dumps(papers))
    summary_path.write_text(json.dumps(summary))
    with patch("eval.evaluation.load_model", return_value=object()), patch("eval.evaluation.compute_src_both") as compute:
        result = evaluation.run_evaluation(str(papers_path), None, None, str(tmp_path / "out"), exp_summary_path=str(summary_path))
    assert not result["papers"][0]["systems"]
    assert result["invalid_conditions"][0]["expected_reviews"] == 3
    compute.assert_not_called()
    assert evaluation._advanced_decision_from_reviewers([review()], 3) is None
