import json
import hashlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import analyze_trigger_screening as analysis
from eval import prepare_trigger_evidence_audit as evidence


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _artifacts(tmp_path):
    labels = {}
    paper_specs = []
    index = 1
    for conference in ("ICLR", "ICML", "NEURIPS"):
        for label in ("accept", "reject"):
            for _ in range(6):
                paper_id = f"trigger_val_{index:03d}"
                labels[paper_id] = {
                    "accept_or_not": label,
                    "score": 7 if label == "accept" else 3,
                    "conference": conference,
                }
                paper_specs.append((paper_id, conference, label, index <= 18))
                index += 1
    labels_path = _write(tmp_path / "labels.json", labels)
    selection_path = _write(
        tmp_path / "selection.json",
        {
            "labels_manifest_sha256": analysis._content_sha256(labels),
            "evaluation_manifest_sha256": "evaluation-manifest",
        },
    )

    pairs = []
    for repeat_id in (1, 2, 3):
        summary_papers = []
        evaluation_papers = []
        for paper_id, conference, label, gate_passed in paper_specs:
            package = {"paper_metadata": [{"title": f"Prior work {paper_id}",
                        "abstract": "Verified source abstract", "url": "https://example.test/source"}]}
            package_hash = analysis._content_sha256(package)
            expected_source = "T1" if gate_passed else "T0"
            metrics = {}
            conditions = {}
            for arm in analysis.ARMS:
                if arm == "T0":
                    src = 0.40
                elif arm == "T1":
                    src = 0.42 if gate_passed else 0.39
                else:
                    src = 0.42 if gate_passed else 0.40
                decision = label
                score = 4.0 if label == "accept" else 2.0
                metrics[analysis.SYSTEM_BY_ARM[arm]] = {
                    "system": analysis.SYSTEM_BY_ARM[arm],
                    "decision": decision,
                    "score": score,
                    "decision_match": True,
                    "conference_check": True,
                    "norm_score": 0.75 if label == "accept" else 0.25,
                    "norm_gt_score": 0.75 if label == "accept" else 0.25,
                    "src_strengths": src,
                    "src_weaknesses": src,
                    "src_overall": src,
                }
                provenance = {
                    "paper_id": paper_id,
                    "repeat_id": repeat_id,
                    "arm": arm,
                    "package_sha256": package_hash,
                }
                reviewers = [
                    {
                        "decision": decision,
                        "strengths": ["Clear method"],
                        "weaknesses": ["Limited evaluation"],
                        "scores": {
                            "novelty": score,
                            "soundness": score,
                            "significance": score,
                            "evaluation": score,
                            "clarity": score,
                        },
                    }
                    for _ in range(3)
                ]
                conditions[arm] = {
                    "selected_source_arm": expected_source if arm == "T2" else None,
                    "result": {
                        "trigger_provenance": provenance, "reviewers": reviewers,
                        "rag_package": package if arm == "T1" or (arm == "T2" and gate_passed) else None,
                        "workflow_schema_version": 1, "workflow_status": "complete",
                        "workflow_config": {"n_iter": 3, "reviewer_count": 3}, "turn_failures": [],
                        "turn_outcomes": [
                            {"role": role, "iteration": iteration, "reviewer_index": i, "required": True, "status": "complete"}
                            for role, start in (("reviewer", 1), ("author", 2))
                            for iteration in range(start, 4) for i in range(3)
                        ] + [{"role": "conference", "iteration": 3, "reviewer_index": None, "required": True, "status": "complete"}],
                    },
                }
            summary_papers.append(
                {
                    "paper_id": paper_id,
                    "package_sha256": package_hash,
                    "paper_sha256": "a" * 64,
                    "gate": {"passed": gate_passed},
                    "conditions": conditions,
                }
            )
            evaluation_papers.append(
                {
                    "paper_id": paper_id,
                    "conference": conference,
                    "ground_truth": {"accept_or_not": label, "score": labels[paper_id]["score"]},
                    "systems": metrics,
                }
            )
        summary = {
            "experiment": "trigger_rag_screening",
            "repeat_id": repeat_id,
            "provider": "openrouter",
            "model": "model",
            "rag_config": {"cutoff_date": "2024-12-31"},
            "gate_version": "integrity_v1",
            "prompt_bundle_sha256": "prompt",
            "code_bundle_sha256": "code",
            "selected_paper_ids_sha256": "papers",
            "condition_order_seed": 11766,
            "invalid_result_policy": "agent_fixed_json_retries_then_seal_without_resampling",
            "expected_evaluator_sha256": "evaluation-code",
            "expected_src_sha256": "src-code",
            "expected_embed_model": "mock-embedding-v1",
            "expected_embed_revision": "frozen-revision",
            "papers": summary_papers,
        }
        evaluation = {"papers": evaluation_papers}
        summary_path = _write(tmp_path / f"summary-{repeat_id}.json", summary)
        evaluation["embed_model"] = "mock-embedding-v1"
        evaluation["embed_revision"] = "frozen-revision"
        evaluation["evaluation_runtime"] = {
            "evaluation_code_sha256": "evaluation-code",
            "src_code_sha256": "src-code",
            "sentence_transformers_version": "test-version",
        }
        evaluation["input_artifacts"] = {
            "papers": {
                "path": "evaluation-manifest.json",
                "sha256": "evaluation-manifest-bytes",
                "json_content_sha256": "evaluation-manifest",
            },
            "exp_summary": {
                "path": str(summary_path),
                "sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                "repeat_id": repeat_id,
                "experiment": "trigger_rag_screening",
                "code_bundle_sha256": "code",
                "prompt_bundle_sha256": "prompt",
            },
        }
        evaluation_path = _write(tmp_path / f"evaluation-{repeat_id}.json", evaluation)
        pairs.append((summary_path, evaluation_path))
    return labels_path, selection_path, pairs


def _audit_document(identity, *, status="PASS", t1_error=False):
    records = [
        {"paper_id": f"trigger_val_{index:03d}", "repeat_id": repeat,
         "rag_usefulness_outcome": "win", "verification_context_checked": True,
         "unsupported_claims": {"T0": 1, "T1": 0},
         "major_error": {"T0": False, "T1": t1_error}}
        for index in range(1, 19) for repeat in (1, 2, 3)
    ]
    return {**identity, "schema_version": 2, "status": status, "auditor": "blind-auditor",
            "blinded_to_labels_and_arm_identity": True, "decoded_records": records,
            "audit_diagnostics": {"decoded_records_sha256": analysis._content_sha256(records)},
            **evidence.summarize_decoded_records(records)}


def test_quantitative_pass_is_provisional_without_blinded_evidence_review(tmp_path):
    labels, selection, pairs = _artifacts(tmp_path)
    report = analysis.analyze(
        pairs,
        labels_path=labels,
        selection_metadata_path=selection,
        evidence_review_path=None,
        bootstrap_samples=200,
        seed=11766,
    )
    assert report["decision"] == "PROVISIONAL_NEEDS_BLINDED_EVIDENCE_REVIEW"
    assert report["failed_quantitative_checks"] == []
    assert report["gate"]["pass_count"] == 18
    assert report["comparisons"]["T2_minus_T0"]["src_overall"] == pytest.approx(0.01)


def test_valid_blinded_evidence_review_allows_go(tmp_path):
    labels, selection, pairs = _artifacts(tmp_path)
    provisional = analysis.analyze(
        pairs,
        labels_path=labels,
        selection_metadata_path=selection,
        evidence_review_path=None,
        bootstrap_samples=50,
        seed=11766,
    )
    identity = provisional["evidence_review"]["expected_identity"]
    review_path = _write(
        tmp_path / "evidence.json",
        _audit_document(identity),
    )
    report = analysis.analyze(
        pairs,
        labels_path=labels,
        selection_metadata_path=selection,
        evidence_review_path=review_path,
        bootstrap_samples=200,
        seed=11766,
    )
    assert report["decision"] == "GO_GATED_RAG_CONFIRMATORY_HOLDOUT"


def test_changed_package_across_repetitions_is_rejected(tmp_path):
    labels, selection, pairs = _artifacts(tmp_path)
    summary = json.loads(pairs[1][0].read_text(encoding="utf-8"))
    summary["papers"][0]["package_sha256"] = "changed"
    summary["papers"][0]["conditions"]["T0"]["result"]["trigger_provenance"][
        "package_sha256"
    ] = "changed"
    summary["papers"][0]["conditions"]["T1"]["result"]["trigger_provenance"][
        "package_sha256"
    ] = "changed"
    summary["papers"][0]["conditions"]["T2"]["result"]["trigger_provenance"][
        "package_sha256"
    ] = "changed"
    _write(pairs[1][0], summary)
    evaluation = json.loads(pairs[1][1].read_text(encoding="utf-8"))
    evaluation["input_artifacts"]["exp_summary"]["sha256"] = hashlib.sha256(
        pairs[1][0].read_bytes()
    ).hexdigest()
    _write(pairs[1][1], evaluation)
    with pytest.raises(analysis.AnalysisInputError, match="changed across repeats"):
        analysis.analyze(
            pairs,
            labels_path=labels,
            selection_metadata_path=selection,
            evidence_review_path=None,
            bootstrap_samples=10,
            seed=11766,
        )


def test_duplicate_or_unbound_evaluation_is_rejected(tmp_path):
    labels, selection, pairs = _artifacts(tmp_path)
    duplicate_pairs = [pairs[0], (pairs[1][0], pairs[0][1]), pairs[2]]
    with pytest.raises(analysis.AnalysisInputError, match="duplicate evaluation artifact|not bound"):
        analysis.analyze(
            duplicate_pairs,
            labels_path=labels,
            selection_metadata_path=selection,
            evidence_review_path=None,
            bootstrap_samples=10,
            seed=11766,
        )


def test_decision_match_is_recomputed_from_private_label(tmp_path):
    labels, selection, pairs = _artifacts(tmp_path)
    evaluation = json.loads(pairs[0][1].read_text(encoding="utf-8"))
    evaluation["papers"][0]["systems"]["our_T0"]["decision_match"] = False
    _write(pairs[0][1], evaluation)
    with pytest.raises(analysis.AnalysisInputError, match="disagrees with private label"):
        analysis.analyze(
            pairs,
            labels_path=labels,
            selection_metadata_path=selection,
            evidence_review_path=None,
            bootstrap_samples=10,
            seed=11766,
        )


def test_negative_gate_discrimination_forces_no_go(tmp_path):
    labels, selection, pairs = _artifacts(tmp_path)
    for summary_path, evaluation_path in pairs:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        failed_ids = {paper["paper_id"] for paper in summary["papers"] if not paper["gate"]["passed"]}
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        for paper in evaluation["papers"]:
            if paper["paper_id"] in failed_ids:
                paper["systems"]["our_T1"]["src_strengths"] = 0.43
                paper["systems"]["our_T1"]["src_weaknesses"] = 0.43
                paper["systems"]["our_T1"]["src_overall"] = 0.43
        _write(evaluation_path, evaluation)
    report = analysis.analyze(
        pairs,
        labels_path=labels,
        selection_metadata_path=selection,
        evidence_review_path=None,
        bootstrap_samples=50,
        seed=11766,
    )
    assert report["decision"] == "NO_GO_ABANDON_RAG"
    assert "gate_src_discrimination" in report["failed_quantitative_checks"]


def test_evidence_review_must_bind_exact_result_pairs(tmp_path):
    labels, selection, pairs = _artifacts(tmp_path)
    provisional = analysis.analyze(
        pairs,
        labels_path=labels,
        selection_metadata_path=selection,
        evidence_review_path=None,
        bootstrap_samples=20,
        seed=11766,
    )
    identity = dict(provisional["evidence_review"]["expected_identity"])
    identity["audited_result_pair_set_sha256"] = "old-run"
    review_path = _write(
        tmp_path / "wrong-evidence.json",
        {
            **identity,
            "status": "PASS",
            "blinded_to_labels_and_arm_identity": True,
            "major_error_rate": 0.0,
            "paired_usefulness": {"rag_wins": 36, "ties": 18, "rag_losses": 0},
            "unsupported_claims": {"T0": 2, "T1": 1},
        },
    )
    report = analysis.analyze(
        pairs,
        labels_path=labels,
        selection_metadata_path=selection,
        evidence_review_path=review_path,
        bootstrap_samples=20,
        seed=11766,
    )
    assert report["decision"] == "INVALID_EVIDENCE_REVIEW"
    assert report["evidence_review"]["passed"] is False


@pytest.mark.parametrize("status,t1_error", [("FAIL", False), ("PASS", True)])
def test_completed_failed_audit_is_no_go_not_provisional(tmp_path, status, t1_error):
    labels, selection, pairs = _artifacts(tmp_path)
    kwargs = dict(labels_path=labels, selection_metadata_path=selection, bootstrap_samples=10, seed=11766)
    identity = analysis.analyze(pairs, evidence_review_path=None, **kwargs)["evidence_review"]["expected_identity"]
    audit = _write(tmp_path / "evidence.json", _audit_document(identity, status=status, t1_error=t1_error))
    report = analysis.analyze(pairs, evidence_review_path=audit, **kwargs)
    assert report["decision"] == "NO_GO_ABANDON_RAG"
    assert report["evidence_review"]["status"] == "FAIL"


def test_missing_middle_turn_rejected_even_with_valid_final_reviews(tmp_path):
    labels, selection, pairs = _artifacts(tmp_path)
    summary = json.loads(pairs[0][0].read_text())
    summary["papers"][0]["conditions"]["T0"]["result"]["turn_outcomes"].pop()
    _write(pairs[0][0], summary)
    evaluation = json.loads(pairs[0][1].read_text())
    evaluation["input_artifacts"]["exp_summary"]["sha256"] = hashlib.sha256(pairs[0][0].read_bytes()).hexdigest()
    _write(pairs[0][1], evaluation)
    with pytest.raises(analysis.AnalysisInputError, match="incomplete required workflow"):
        analysis.analyze(pairs, labels_path=labels, selection_metadata_path=selection,
                         evidence_review_path=None, bootstrap_samples=10, seed=11766)
