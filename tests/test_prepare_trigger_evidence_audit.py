import json
import hashlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import prepare_trigger_evidence_audit as evidence


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _result(paper_id, repeat_id, arm):
    return {
        "reviewers": [
            {
                "decision": "accept",
                "scores": {
                    "novelty": 4,
                    "soundness": 4,
                    "significance": 4,
                    "evaluation": 4,
                    "clarity": 4,
                },
                "strengths": ["Useful strength"],
                "weaknesses": ["Useful weakness"],
            }
            for _ in range(3)
        ],
        "conference": {},
        "rag_package": {"private": arm},
        "rag_warnings": [],
        "cutoff_report": {},
        "turn_outcomes": [{"raw_reply": "T1 retry diagnostics must remain private"}],
        "trigger_provenance": {
            "paper_id": paper_id,
            "repeat_id": repeat_id,
            "arm": arm,
            "package_sha256": f"package-{paper_id}",
        },
    }


def _summaries(tmp_path):
    markdown = "# Target paper\n\nMethod and experiments.\n"
    (tmp_path / "opaque-pass.md").write_text(markdown)
    _write(tmp_path / "manifest.json", {"opaque-pass": {"paper_dir": "opaque-pass.pdf", "title": "Target paper"}})
    package = {"target_paper_summary": {"title": "Target paper"},
               "paper_metadata": [{"title": "Prior baseline", "abstract": "Source abstract",
                "url": "https://example.test/source"}]}
    package_hash = evidence.content_sha256(package)
    paths = []
    for repeat_id in (1, 2, 3):
        papers = []
        for paper_id, gate_passed in (("opaque-pass", True), ("opaque-fail", False)):
            papers.append(
                {
                    "paper_id": paper_id,
                    "package_sha256": f"package-{paper_id}",
                    "paper_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
                    "gate": {"passed": gate_passed},
                    "conditions": {
                        arm: {"result": _result(paper_id, repeat_id, arm)}
                        for arm in ("T0", "T1")
                    },
                }
            )
            if gate_passed:
                papers[-1]["package_sha256"] = package_hash
                for arm in ("T0", "T1"):
                    result = papers[-1]["conditions"][arm]["result"]
                    result["trigger_provenance"]["package_sha256"] = package_hash
                    result["rag_package"] = package if arm == "T1" else None
        paths.append(
            _write(
                tmp_path / f"summary-{repeat_id}.json",
                {
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
                    "expected_embed_model": "embedding",
                    "expected_embed_revision": "revision",
                    "papers": papers,
                },
            )
        )
    return paths


def _prepare(tmp_path):
    return evidence.prepare_audit(_summaries(tmp_path), output_dir=tmp_path / "audit",
                                  manifest_path=tmp_path / "manifest.json", md_dir=tmp_path)


def test_prepare_is_blinded_and_finalize_decodes_exact_pairs(tmp_path):
    audit_dir = tmp_path / "audit"
    paths = _prepare(tmp_path)
    blinded = json.loads(paths["blinded"].read_text(encoding="utf-8"))
    private_key = json.loads(paths["key"].read_text(encoding="utf-8"))
    judgments = json.loads(paths["judgments"].read_text(encoding="utf-8"))

    assert len(blinded["pairs"]) == 3
    assert "T0" not in json.dumps(blinded["pairs"])
    assert "T1" not in json.dumps(blinded["pairs"])
    assert "rag_package" not in json.dumps(blinded["pairs"])
    assert "turn_outcomes" not in json.dumps(blinded["pairs"])
    assert "seed" not in blinded and "seed" not in judgments
    assert blinded["pairs"][0]["verification_context"]["target_paper"]["markdown"].startswith("# Target")
    assignments = {item["pair_id"]: item for item in private_key["assignments"]}

    judgments["auditor"] = "blind-auditor"
    judgments["blinded_to_labels_and_arm_identity"] = True
    for item in judgments["judgments"]:
        assignment = assignments[item["pair_id"]]
        item["usefulness_winner"] = "A" if assignment["A_arm"] == "T1" else "B"
        item["unsupported_claims_A"] = 0 if assignment["A_arm"] == "T1" else 1
        item["unsupported_claims_B"] = 0 if assignment["B_arm"] == "T1" else 1
        item["major_error_A"] = False
        item["major_error_B"] = False
        item["verification_context_checked"] = True
    _write(paths["judgments"], judgments)

    result = evidence.finalize_audit(
        blinded_path=paths["blinded"],
        private_key_path=paths["key"],
        judgments_path=paths["judgments"],
        output_path=tmp_path / "evidence-review.json",
    )
    assert result["status"] == "PASS"
    assert result["paired_usefulness"] == {"rag_wins": 3, "ties": 0, "rag_losses": 0}
    assert result["unsupported_claims"] == {"T0": 3, "T1": 0}
    assert result["audited_result_pair_count"] == 3


def test_finalize_rejects_modified_blinded_output(tmp_path):
    paths = _prepare(tmp_path)
    blinded = json.loads(paths["blinded"].read_text(encoding="utf-8"))
    blinded["pairs"][0]["review_set_A"]["reviewers"][0]["strengths"] = ["tampered"]
    _write(paths["blinded"], blinded)
    with pytest.raises(evidence.EvidenceAuditError, match="hash does not match"):
        evidence.finalize_audit(
            blinded_path=paths["blinded"],
            private_key_path=paths["key"],
            judgments_path=paths["judgments"],
            output_path=tmp_path / "evidence-review.json",
        )


def test_prepare_rejects_changed_target_before_creating_audit(tmp_path):
    paths = _summaries(tmp_path)
    (tmp_path / "opaque-pass.md").write_text("Changed paper")
    with pytest.raises(evidence.EvidenceAuditError, match="target text differs"):
        evidence.prepare_audit(paths, output_dir=tmp_path / "audit",
                               manifest_path=tmp_path / "manifest.json", md_dir=tmp_path)
    assert not (tmp_path / "audit" / "blinded_pairs.json").exists()


def test_missing_verification_sources_are_not_accepted():
    with pytest.raises(evidence.EvidenceAuditError, match="abstract is missing"):
        evidence.verification_source_panel({"paper_metadata": [{"title": "Source", "url": "https://example.test/"}]})
