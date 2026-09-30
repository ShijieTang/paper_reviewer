#!/usr/bin/env python3
"""Analyze the pre-registered final quality-gated RAG screening experiment.

The unit of analysis is the paper, not the individual model call.  Metrics are
first averaged across repetitions for each paper, then compared with a paired,
conference/label-stratified bootstrap.  T2 is the pre-registered primary policy;
T1 (always RAG) is diagnostic only and cannot replace T2 after seeing results.

Pass one summary/evaluation pair per repetition.  The evaluator output is made
by ``eval/evaluation.py`` after all model runs are sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from review_schema import REVIEW_SCORE_KEYS, validate_review_schema, workflow_is_complete
from eval.prepare_trigger_evidence_audit import (
    EvidenceAuditError, RUBRIC_VERSION, evidence_metrics_pass,
    summarize_decoded_records, verification_identity,
)


ARMS = ("T0", "T1", "T2")
SYSTEM_BY_ARM = {arm: f"our_{arm}" for arm in ARMS}

# Freeze these before the first formal model call.  They deliberately describe
# a screening gate, not a powered confirmatory trial.
SCREENING_CRITERIA = {
    "expected_papers": 36,
    "expected_repeats": 3,
    "min_gate_pass_papers": 8,
    "max_gate_pass_papers": 28,
    "min_papers_per_label_in_each_gate_group": 3,
    "primary_arm": "T2",
    "baseline_arm": "T0",
    # RAG gets one final chance if automatic SRC is non-inferior; its added
    # value must then come from the separately blinded evidence-usefulness audit.
    "min_src_delta": -0.005,
    "min_balanced_accuracy_delta": -0.02,
    "min_accept_recall": 0.60,
    "min_reject_recall": 0.60,
    "min_pass_subset_t1_minus_t0_src": 0.0,
    "min_src_gate_discrimination": 0.0,
    "min_blinded_evidence_usefulness": 0.60,
    "max_evidence_major_error_rate": 0.10,
    "bootstrap_confidence": 0.80,
}


class AnalysisInputError(ValueError):
    """Raised when sealed experiment artifacts are inconsistent or incomplete."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisInputError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisInputError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise AnalysisInputError(f"Could not hash {path}: {exc}") from exc


def _content_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalise_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if label not in {"accept", "reject"}:
        raise AnalysisInputError(f"Invalid decision label: {value!r}")
    return label


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisInputError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise AnalysisInputError(f"{name} must be finite")
    return number


def _expected_decision_and_score(result: dict[str, Any], context: str) -> tuple[str, float]:
    """Reconstruct evaluator inputs so each evaluation is bound to its summary."""
    reviewers = result.get("reviewers")
    if not isinstance(reviewers, list) or len(reviewers) != 3:
        raise AnalysisInputError(f"{context}: expected exactly three reviewers")
    votes: list[str] = []
    reviewer_scores: list[float] = []
    for reviewer in reviewers:
        errors = validate_review_schema(reviewer)
        if errors:
            raise AnalysisInputError(f"{context}: invalid reviewer result: {errors}")
        votes.append(_normalise_label(reviewer.get("decision")))
        scores = reviewer.get("scores")
        reviewer_scores.append(
            mean(_finite_number(scores[key], f"{context} {key}") for key in REVIEW_SCORE_KEYS)
        )
    decision = "accept" if votes.count("accept") >= 2 else "reject"
    return decision, round(mean(reviewer_scores), 4)


def _index_papers(document: dict[str, Any], path: Path) -> dict[str, dict[str, Any]]:
    papers = document.get("papers")
    if not isinstance(papers, list):
        raise AnalysisInputError(f"{path}: papers must be a list")
    index: dict[str, dict[str, Any]] = {}
    for item in papers:
        if not isinstance(item, dict) or not str(item.get("paper_id") or ""):
            raise AnalysisInputError(f"{path}: every paper needs a paper_id")
        paper_id = str(item["paper_id"])
        if paper_id in index:
            raise AnalysisInputError(f"{path}: duplicate paper_id {paper_id}")
        index[paper_id] = item
    return index


def _provenance_signature(summary: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "experiment",
        "provider",
        "model",
        "rag_config",
        "gate_version",
        "prompt_bundle_sha256",
        "code_bundle_sha256",
        "selected_paper_ids_sha256",
        "condition_order_seed",
        "invalid_result_policy",
        "expected_evaluator_sha256",
        "expected_src_sha256",
        "expected_embed_model",
        "expected_embed_revision",
    )
    return {field: summary.get(field) for field in fields}


def _validate_condition_result(
    condition: dict[str, Any],
    *,
    arm: str,
    paper_id: str,
    repeat_id: int,
    package_sha256: str,
) -> None:
    result = condition.get("result")
    if not isinstance(result, dict):
        raise AnalysisInputError(f"rep {repeat_id} {paper_id} {arm}: missing result")
    if not workflow_is_complete(result, 3, 3):
        raise AnalysisInputError(f"rep {repeat_id} {paper_id} {arm}: incomplete required workflow")
    provenance = result.get("trigger_provenance")
    if not isinstance(provenance, dict):
        raise AnalysisInputError(f"rep {repeat_id} {paper_id} {arm}: missing trigger provenance")
    expected = {
        "paper_id": paper_id,
        "repeat_id": repeat_id,
        "arm": arm,
        "package_sha256": package_sha256,
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise AnalysisInputError(
                f"rep {repeat_id} {paper_id} {arm}: provenance {field} mismatch"
            )


def collect_paper_records(
    pairs: list[tuple[Path, Path]],
    labels_path: Path,
    selection_metadata_path: Path,
    *,
    expected_repeats: int,
    expected_papers: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load and fail closed on any mismatch across summaries/evaluations."""
    labels_document = _load_object(labels_path)
    selection_metadata = _load_object(selection_metadata_path)
    if selection_metadata.get("labels_manifest_sha256") != _content_sha256(labels_document):
        raise AnalysisInputError("private labels do not match the frozen selection metadata")
    expected_evaluation_manifest_sha256 = selection_metadata.get("evaluation_manifest_sha256")
    if not expected_evaluation_manifest_sha256:
        raise AnalysisInputError("selection metadata lacks evaluation_manifest_sha256")
    if len(labels_document) != expected_papers:
        raise AnalysisInputError(
            f"labels contain {len(labels_document)} papers; expected {expected_papers}"
        )
    labels: dict[str, dict[str, Any]] = {}
    for paper_id, item in labels_document.items():
        if not isinstance(item, dict):
            raise AnalysisInputError(f"label entry {paper_id} must be an object")
        labels[paper_id] = {
            "label": _normalise_label(item.get("accept_or_not")),
            "conference": str(item.get("conference") or "").strip().upper(),
            "score": _finite_number(item.get("score"), f"label entry {paper_id} score"),
        }
        if not labels[paper_id]["conference"]:
            raise AnalysisInputError(f"label entry {paper_id} has no conference")
    strata = Counter((item["conference"], item["label"]) for item in labels.values())
    expected_strata = {
        (conference, label): 6
        for conference in ("ICLR", "ICML", "NEURIPS")
        for label in ("accept", "reject")
    }
    if dict(strata) != expected_strata:
        raise AnalysisInputError(
            f"private labels are not the frozen 3-conference x 2-label x 6 design: {dict(strata)}"
        )

    if len(pairs) != expected_repeats:
        raise AnalysisInputError(f"received {len(pairs)} repetitions; expected {expected_repeats}")

    records: dict[str, dict[str, Any]] = {
        paper_id: {
            **label,
            "gate_passed": None,
            "package_sha256": None,
            "by_arm": {arm: [] for arm in ARMS},
        }
        for paper_id, label in labels.items()
    }
    seen_repeats: set[int] = set()
    common_signature: dict[str, Any] | None = None
    common_evaluation_signature: dict[str, Any] | None = None
    seen_evaluation_hashes: set[str] = set()
    artifact_pairs: list[dict[str, Any]] = []

    for summary_path, evaluation_path in pairs:
        summary = _load_object(summary_path)
        evaluation = _load_object(evaluation_path)
        summary_sha256 = _file_sha256(summary_path)
        evaluation_sha256 = _file_sha256(evaluation_path)
        if evaluation_sha256 in seen_evaluation_hashes:
            raise AnalysisInputError(f"duplicate evaluation artifact: {evaluation_path}")
        seen_evaluation_hashes.add(evaluation_sha256)
        repeat_id_raw = summary.get("repeat_id")
        if isinstance(repeat_id_raw, bool) or not isinstance(repeat_id_raw, int):
            raise AnalysisInputError(f"{summary_path}: repeat_id must be an integer")
        repeat_id = repeat_id_raw
        if repeat_id in seen_repeats:
            raise AnalysisInputError(f"duplicate repeat_id {repeat_id}")
        seen_repeats.add(repeat_id)

        input_artifacts = evaluation.get("input_artifacts")
        evaluation_runtime = evaluation.get("evaluation_runtime")
        exp_artifact = input_artifacts.get("exp_summary") if isinstance(input_artifacts, dict) else None
        papers_artifact = input_artifacts.get("papers") if isinstance(input_artifacts, dict) else None
        if not isinstance(exp_artifact, dict) or not isinstance(papers_artifact, dict):
            raise AnalysisInputError(f"{evaluation_path}: missing sealed input-artifact provenance")
        if exp_artifact.get("sha256") != summary_sha256 or exp_artifact.get("repeat_id") != repeat_id:
            raise AnalysisInputError(f"{evaluation_path}: not bound to {summary_path}")
        if exp_artifact.get("experiment") != "trigger_rag_screening":
            raise AnalysisInputError(f"{evaluation_path}: evaluated the wrong experiment type")
        evaluation_signature = {
            "embed_model": evaluation.get("embed_model"),
            "embed_revision": evaluation.get("embed_revision"),
            "evaluation_runtime": evaluation_runtime,
            "papers_sha256": papers_artifact.get("sha256"),
        }
        if (
            not evaluation_signature["embed_model"]
            or not isinstance(evaluation_runtime, dict)
            or not evaluation_runtime.get("evaluation_code_sha256")
            or not evaluation_runtime.get("src_code_sha256")
            or not evaluation_runtime.get("sentence_transformers_version")
            or not evaluation_signature["embed_revision"]
            or not evaluation_signature["papers_sha256"]
        ):
            raise AnalysisInputError(f"{evaluation_path}: incomplete evaluator/model provenance")
        if papers_artifact.get("json_content_sha256") != expected_evaluation_manifest_sha256:
            raise AnalysisInputError(
                f"{evaluation_path}: ground-truth manifest does not match frozen selection metadata"
            )
        if common_evaluation_signature is None:
            common_evaluation_signature = evaluation_signature
        elif evaluation_signature != common_evaluation_signature:
            raise AnalysisInputError(f"{evaluation_path}: evaluator or embedding model differs across repeats")

        signature = _provenance_signature(summary)
        if signature["experiment"] != "trigger_rag_screening":
            raise AnalysisInputError(f"{summary_path}: not a trigger_rag_screening summary")
        for field in (
            "provider",
            "model",
            "gate_version",
            "prompt_bundle_sha256",
            "code_bundle_sha256",
            "selected_paper_ids_sha256",
            "invalid_result_policy",
            "expected_evaluator_sha256",
            "expected_src_sha256",
            "expected_embed_model",
            "expected_embed_revision",
        ):
            if not signature[field]:
                raise AnalysisInputError(f"{summary_path}: missing provenance field {field}")
        for expected_field, actual in (
            ("expected_evaluator_sha256", evaluation_runtime["evaluation_code_sha256"]),
            ("expected_src_sha256", evaluation_runtime["src_code_sha256"]),
            ("expected_embed_model", evaluation_signature["embed_model"]),
            ("expected_embed_revision", evaluation_signature["embed_revision"]),
        ):
            if signature[expected_field] != actual:
                raise AnalysisInputError(f"{evaluation_path}: evaluator differs from frozen {expected_field}")
        if common_signature is None:
            common_signature = signature
        elif signature != common_signature:
            raise AnalysisInputError(f"{summary_path}: experiment provenance differs across repeats")
        if (
            exp_artifact.get("code_bundle_sha256") != signature["code_bundle_sha256"]
            or exp_artifact.get("prompt_bundle_sha256") != signature["prompt_bundle_sha256"]
        ):
            raise AnalysisInputError(f"{evaluation_path}: embedded summary provenance mismatch")

        summary_index = _index_papers(summary, summary_path)
        evaluation_index = _index_papers(evaluation, evaluation_path)
        expected_ids = set(labels)
        if set(summary_index) != expected_ids:
            raise AnalysisInputError(f"{summary_path}: paper IDs do not match private labels")
        if set(evaluation_index) != expected_ids:
            raise AnalysisInputError(f"{evaluation_path}: paper IDs do not match private labels")

        for paper_id in sorted(expected_ids):
            summary_paper = summary_index[paper_id]
            evaluated_paper = evaluation_index[paper_id]
            ground_truth = evaluated_paper.get("ground_truth")
            if not isinstance(ground_truth, dict):
                raise AnalysisInputError(f"{evaluation_path}: {paper_id} lacks ground truth")
            if _normalise_label(ground_truth.get("accept_or_not")) != labels[paper_id]["label"]:
                raise AnalysisInputError(f"{evaluation_path}: {paper_id} ground-truth label mismatch")
            if abs(
                _finite_number(ground_truth.get("score"), f"{evaluation_path}: {paper_id} GT score")
                - labels[paper_id]["score"]
            ) > 1e-9:
                raise AnalysisInputError(f"{evaluation_path}: {paper_id} ground-truth score mismatch")
            if str(evaluated_paper.get("conference") or "").strip().upper() != labels[paper_id]["conference"]:
                raise AnalysisInputError(f"{evaluation_path}: {paper_id} conference mismatch")

            package_sha256 = str(summary_paper.get("package_sha256") or "")
            gate = summary_paper.get("gate")
            if not package_sha256 or not isinstance(gate, dict) or not isinstance(gate.get("passed"), bool):
                raise AnalysisInputError(f"{summary_path}: {paper_id} has invalid package/gate data")
            gate_passed = gate["passed"]
            if records[paper_id]["gate_passed"] is None:
                records[paper_id]["gate_passed"] = gate_passed
                records[paper_id]["package_sha256"] = package_sha256
            elif (
                records[paper_id]["gate_passed"] != gate_passed
                or records[paper_id]["package_sha256"] != package_sha256
            ):
                raise AnalysisInputError(f"{paper_id}: frozen package or gate changed across repeats")

            conditions = summary_paper.get("conditions")
            systems = evaluated_paper.get("systems")
            if not isinstance(conditions, dict) or not isinstance(systems, dict):
                raise AnalysisInputError(f"rep {repeat_id} {paper_id}: missing conditions/systems")
            if gate_passed:
                try:
                    package = conditions["T1"]["result"].get("rag_package")
                    if _content_sha256(package) != package_sha256:
                        raise EvidenceAuditError("T1 package content does not match sealed package hash")
                    context = verification_identity(paper_id, str(summary_paper.get("paper_sha256") or ""), package)
                except (KeyError, EvidenceAuditError) as exc:
                    raise AnalysisInputError(f"{paper_id}: invalid verification context: {exc}") from exc
                previous = records[paper_id].setdefault("verification_identity", context)
                if previous != context:
                    raise AnalysisInputError(f"{paper_id}: verification context changed across repeats")
            expected_source = "T1" if gate_passed else "T0"
            t2_condition = conditions.get("T2")
            if not isinstance(t2_condition, dict) or t2_condition.get("selected_source_arm") != expected_source:
                raise AnalysisInputError(f"rep {repeat_id} {paper_id}: T2 routing mismatch")

            for arm in ARMS:
                condition = conditions.get(arm)
                metric = systems.get(SYSTEM_BY_ARM[arm])
                if not isinstance(condition, dict) or not isinstance(metric, dict):
                    raise AnalysisInputError(f"rep {repeat_id} {paper_id}: missing {arm}")
                _validate_condition_result(
                    condition,
                    arm=arm,
                    paper_id=paper_id,
                    repeat_id=repeat_id,
                    package_sha256=package_sha256,
                )
                expected_decision, expected_score = _expected_decision_and_score(
                    condition["result"], f"rep {repeat_id} {paper_id} {arm}"
                )
                decision = _normalise_label(metric.get("decision"))
                evaluated_score = _finite_number(
                    metric.get("score"), f"rep {repeat_id} {paper_id} {arm} raw score"
                )
                if decision != expected_decision or abs(evaluated_score - expected_score) > 1e-9:
                    raise AnalysisInputError(
                        f"rep {repeat_id} {paper_id} {arm}: evaluation does not match its summary"
                    )
                decision_match = metric.get("decision_match")
                if not isinstance(decision_match, bool):
                    raise AnalysisInputError(
                        f"rep {repeat_id} {paper_id} {arm}: decision_match must be boolean"
                    )
                recomputed_match = decision == labels[paper_id]["label"]
                if decision_match != recomputed_match:
                    raise AnalysisInputError(
                        f"rep {repeat_id} {paper_id} {arm}: decision_match disagrees with private label"
                    )
                src_overall = _finite_number(
                    metric.get("src_overall"), f"rep {repeat_id} {paper_id} {arm} SRC"
                )
                norm_score = _finite_number(
                    metric.get("norm_score"), f"rep {repeat_id} {paper_id} {arm} score"
                )
                norm_gt_score = _finite_number(
                    metric.get("norm_gt_score"), f"rep {repeat_id} {paper_id} GT score"
                )
                if not all(0.0 <= value <= 1.0 for value in (src_overall, norm_score, norm_gt_score)):
                    raise AnalysisInputError(
                        f"rep {repeat_id} {paper_id} {arm}: normalized metric outside [0,1]"
                    )
                record = {
                    "repeat_id": repeat_id,
                    "result_sha256": _content_sha256(condition["result"]),
                    "decision": decision,
                    "decision_correct": float(recomputed_match),
                    "predicted_accept": float(decision == "accept"),
                    "src_overall": src_overall,
                    "norm_score": norm_score,
                    "norm_gt_score": norm_gt_score,
                }
                records[paper_id]["by_arm"][arm].append(record)

            selected = systems[SYSTEM_BY_ARM[expected_source]]
            derived = systems[SYSTEM_BY_ARM["T2"]]
            for field in ("decision", "score", "src_strengths", "src_weaknesses", "src_overall"):
                if derived.get(field) != selected.get(field):
                    raise AnalysisInputError(
                        f"rep {repeat_id} {paper_id}: evaluated T2 differs from routed {expected_source}"
                    )

        artifact_pairs.append(
            {
                "repeat_id": repeat_id,
                "summary": str(summary_path),
                "evaluation": str(evaluation_path),
            }
        )

    if seen_repeats != set(range(1, expected_repeats + 1)):
        raise AnalysisInputError(
            f"repeat IDs are {sorted(seen_repeats)}; expected 1..{expected_repeats}"
        )
    for paper_id, paper in records.items():
        for arm in ARMS:
            arm_repeats = {item["repeat_id"] for item in paper["by_arm"][arm]}
            if arm_repeats != seen_repeats:
                raise AnalysisInputError(f"{paper_id} {arm}: incomplete repetitions")

    return records, {
        "labels_path": str(labels_path),
        "selection_metadata_path": str(selection_metadata_path),
        "selection_metadata_sha256": _file_sha256(selection_metadata_path),
        "pairs": sorted(artifact_pairs, key=lambda item: item["repeat_id"]),
        "experiment_signature": common_signature or {},
        "evaluation_signature": common_evaluation_signature or {},
    }


def aggregate_by_paper(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    aggregated = []
    for paper_id in sorted(records):
        source = records[paper_id]
        item = {
            "paper_id": paper_id,
            "label": source["label"],
            "conference": source["conference"],
            "gate_passed": source["gate_passed"],
            "package_sha256": source["package_sha256"],
            "arms": {},
        }
        for arm in ARMS:
            repetitions = source["by_arm"][arm]
            item["arms"][arm] = {
                key: mean(record[key] for record in repetitions)
                for key in (
                    "decision_correct",
                    "predicted_accept",
                    "src_overall",
                    "norm_score",
                    "norm_gt_score",
                )
            }
        aggregated.append(item)
    return aggregated


def _safe_spearman(rows: list[dict[str, Any]], arm: str) -> float | None:
    if len(rows) < 3:
        return None
    predicted = [row["arms"][arm]["norm_score"] for row in rows]
    truth = [row["arms"][arm]["norm_gt_score"] for row in rows]
    # Index access works across both older ``SpearmanrResult`` and newer
    # SciPy result-object APIs.
    rho = float(spearmanr(predicted, truth)[0])
    return rho if math.isfinite(rho) else None


def arm_metrics(
    rows: list[dict[str, Any]],
    arm: str,
    *,
    include_spearman: bool = True,
) -> dict[str, Any]:
    if not rows:
        raise AnalysisInputError("cannot calculate metrics for an empty paper set")
    label_recall = {
        label: mean(row["arms"][arm]["decision_correct"] for row in rows if row["label"] == label)
        for label in ("accept", "reject")
    }
    return {
        "n_papers": len(rows),
        "decision_accuracy": mean(row["arms"][arm]["decision_correct"] for row in rows),
        "balanced_accuracy": mean(label_recall.values()),
        "accept_recall": label_recall["accept"],
        "reject_recall": label_recall["reject"],
        "accept_rate": mean(row["arms"][arm]["predicted_accept"] for row in rows),
        "src_overall": mean(row["arms"][arm]["src_overall"] for row in rows),
        "score_spearman": _safe_spearman(rows, arm) if include_spearman else None,
    }


def comparison_metrics(rows: list[dict[str, Any]], arm: str, baseline: str) -> dict[str, float | None]:
    left = arm_metrics(rows, arm)
    right = arm_metrics(rows, baseline)
    fields = (
        "decision_accuracy",
        "balanced_accuracy",
        "accept_recall",
        "reject_recall",
        "accept_rate",
        "src_overall",
        "score_spearman",
    )
    return {
        field: None if left[field] is None or right[field] is None else left[field] - right[field]
        for field in fields
    }


def _interaction_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for gate_value, name in ((True, "gate_pass"), (False, "gate_fail")):
        subset = [row for row in rows if row["gate_passed"] is gate_value]
        output[name] = {
            "n_papers": len(subset),
            "t1_minus_t0_src": mean(
                row["arms"]["T1"]["src_overall"] - row["arms"]["T0"]["src_overall"]
                for row in subset
            ) if subset else None,
            "t1_minus_t0_decision_accuracy": mean(
                row["arms"]["T1"]["decision_correct"]
                - row["arms"]["T0"]["decision_correct"]
                for row in subset
            ) if subset else None,
        }
    pass_src = output["gate_pass"]["t1_minus_t0_src"]
    fail_src = output["gate_fail"]["t1_minus_t0_src"]
    output["src_gate_discrimination"] = (
        pass_src - fail_src if pass_src is not None and fail_src is not None else None
    )
    return output


def _quantile(values: list[float], probability: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    position = (len(finite) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def _stratified_resample(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(row["conference"], row["label"])].append(row)
    sample: list[dict[str, Any]] = []
    for key in sorted(strata):
        members = strata[key]
        sample.extend(rng.choice(members) for _ in members)
    return sample


def bootstrap_intervals(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    if samples <= 0:
        raise AnalysisInputError("bootstrap_samples must be positive")
    rng = random.Random(seed)
    alpha = (1.0 - confidence) / 2.0
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        sample = _stratified_resample(rows, rng)
        sample_arm_metrics = {
            arm: arm_metrics(sample, arm, include_spearman=False) for arm in ARMS
        }
        for comparison_name, arm, baseline in (
            ("T2_minus_T0", "T2", "T0"),
            ("T2_minus_T1", "T2", "T1"),
            ("T1_minus_T0", "T1", "T0"),
        ):
            for metric in (
                "decision_accuracy",
                "balanced_accuracy",
                "accept_recall",
                "reject_recall",
                "accept_rate",
                "src_overall",
            ):
                value = sample_arm_metrics[arm][metric] - sample_arm_metrics[baseline][metric]
                if value is not None and math.isfinite(value):
                    draws[f"{comparison_name}.{metric}"].append(value)
        interaction = _interaction_metrics(sample)
        for group in ("gate_pass", "gate_fail"):
            for metric in ("t1_minus_t0_src", "t1_minus_t0_decision_accuracy"):
                value = interaction[group][metric]
                if value is not None and math.isfinite(value):
                    draws[f"interaction.{group}.{metric}"].append(value)
        discrimination = interaction["src_gate_discrimination"]
        if discrimination is not None and math.isfinite(discrimination):
            draws["interaction.src_gate_discrimination"].append(discrimination)

    return {
        key: {
            "confidence": confidence,
            "lower": _quantile(values, alpha),
            "upper": _quantile(values, 1.0 - alpha),
            "valid_bootstrap_samples": len(values),
        }
        for key, values in sorted(draws.items())
    }


def _load_evidence_review(
    path: Path | None,
    gate_pass_packages: list[dict[str, str]],
    gate_pass_result_pairs: list[dict[str, Any]],
    experiment_signature: dict[str, Any],
    verification_contexts: list[dict[str, str]],
) -> dict[str, Any]:
    gate_pass_count = len(gate_pass_packages)
    result_pair_count = len(gate_pass_result_pairs)
    expected_package_set_sha256 = _content_sha256(gate_pass_packages)
    expected_identity = {
        "rubric_version": RUBRIC_VERSION,
        "gate_version": experiment_signature.get("gate_version"),
        "selected_paper_ids_sha256": experiment_signature.get("selected_paper_ids_sha256"),
        "audited_package_count": gate_pass_count,
        "audited_package_set_sha256": expected_package_set_sha256,
        "audited_result_pair_count": result_pair_count,
        "audited_result_pair_set_sha256": _content_sha256(gate_pass_result_pairs),
        "verification_context_set_sha256": _content_sha256(verification_contexts),
    }
    if path is None:
        return {
            "status": "MISSING",
            "passed": False,
            "reason": "A blinded evidence-quality review JSON was not supplied.",
            "expected_identity": expected_identity,
        }
    base = {"path": str(path), "passed": False, "expected_identity": expected_identity}
    try:
        document = _load_object(path)
        if any(document.get(field) != value for field, value in expected_identity.items()):
            raise EvidenceAuditError("audit identity does not match sealed packages, result pairs and verification context")
        if document.get("schema_version") != 2:
            raise EvidenceAuditError("unsupported audit schema; use verified paired_evidence_v2")
        status = str(document.get("status") or "").upper()
        if status == "PENDING":
            if document.get("decoded_records"):
                raise EvidenceAuditError("completed judgments cannot be relabeled PENDING")
            return {**base, "status": "PENDING", "reason": "The bound blinded evidence review has not been completed."}
        if status not in {"PASS", "FAIL"}:
            raise EvidenceAuditError("status must be PASS, FAIL, or PENDING")
        if document.get("blinded_to_labels_and_arm_identity") is not True or not str(document.get("auditor") or "").strip():
            raise EvidenceAuditError("completed audit lacks named auditor or blind certification")
        decoded = document.get("decoded_records")
        if not isinstance(decoded, list) or len(decoded) != result_pair_count:
            raise EvidenceAuditError("decoded judgments must cover every PASS paper-repeat pair")
        expected_pairs = {(item["paper_id"], item["repeat_id"]) for item in gate_pass_result_pairs}
        actual_pairs = {(item["paper_id"], item["repeat_id"]) for item in decoded}
        if actual_pairs != expected_pairs or len(actual_pairs) != len(decoded):
            raise EvidenceAuditError("decoded judgment paper/repetition set mismatch")
        diagnostics = document.get("audit_diagnostics", {})
        if diagnostics.get("decoded_records_sha256") != _content_sha256(decoded):
            raise EvidenceAuditError("decoded judgment checksum mismatch")
        metrics = summarize_decoded_records(decoded)
        for field, value in metrics.items():
            if document.get(field) != value:
                raise EvidenceAuditError(f"reported {field} differs from decoded judgments")
        passed = evidence_metrics_pass(metrics)
        # A declared FAIL is terminal, even if the auditor applied stricter criteria.
        status = "FAIL" if status == "FAIL" or not passed else "PASS"
        return {
            **base, **metrics, "status": status, "passed": status == "PASS",
            "reason": "Blinded evidence-quality review passed." if status == "PASS" else
                      "Completed evidence review failed; do not resample judgments or repeat the audit to obtain PASS.",
        }
    except (AnalysisInputError, EvidenceAuditError, KeyError, TypeError, ValueError) as exc:
        return {**base, "status": "INVALID", "reason": f"Invalid audit artifact: {exc}. Restore the sealed artifact; do not collect replacement judgments."}


def screening_checks(
    *,
    arm_results: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    interaction: dict[str, Any],
    gate_pass_count: int,
    gate_label_counts: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    criteria = SCREENING_CRITERIA
    primary = arm_results[criteria["primary_arm"]]
    primary_comparison = comparisons["T2_minus_T0"]
    raw_checks = [
        (
            "gate_coverage",
            criteria["min_gate_pass_papers"] <= gate_pass_count <= criteria["max_gate_pass_papers"],
            gate_pass_count,
            f"{criteria['min_gate_pass_papers']}..{criteria['max_gate_pass_papers']} PASS papers",
        ),
        (
            "gate_groups_have_both_labels",
            all(
                gate_label_counts[group][label] >= criteria["min_papers_per_label_in_each_gate_group"]
                for group in ("pass", "fail")
                for label in ("accept", "reject")
            ),
            gate_label_counts,
            f">= {criteria['min_papers_per_label_in_each_gate_group']} papers per label in each gate group",
        ),
        (
            "t2_src_noninferiority",
            primary_comparison["src_overall"] >= criteria["min_src_delta"],
            primary_comparison["src_overall"],
            f">= {criteria['min_src_delta']}",
        ),
        (
            "t2_balanced_accuracy_noninferiority",
            primary_comparison["balanced_accuracy"] >= criteria["min_balanced_accuracy_delta"],
            primary_comparison["balanced_accuracy"],
            f">= {criteria['min_balanced_accuracy_delta']}",
        ),
        (
            "t2_accept_recall_floor",
            primary["accept_recall"] >= criteria["min_accept_recall"],
            primary["accept_recall"],
            f">= {criteria['min_accept_recall']}",
        ),
        (
            "t2_reject_recall_floor",
            primary["reject_recall"] >= criteria["min_reject_recall"],
            primary["reject_recall"],
            f">= {criteria['min_reject_recall']}",
        ),
        (
            "gate_pass_subset_direction",
            interaction["gate_pass"]["t1_minus_t0_src"] is not None
            and interaction["gate_pass"]["t1_minus_t0_src"]
            > criteria["min_pass_subset_t1_minus_t0_src"],
            interaction["gate_pass"]["t1_minus_t0_src"],
            f"> {criteria['min_pass_subset_t1_minus_t0_src']}",
        ),
        (
            "gate_src_discrimination",
            interaction["src_gate_discrimination"] is not None
            and interaction["src_gate_discrimination"] > criteria["min_src_gate_discrimination"],
            interaction["src_gate_discrimination"],
            f"> {criteria['min_src_gate_discrimination']}",
        ),
    ]
    return [
        {"name": name, "passed": bool(passed), "observed": observed, "criterion": criterion}
        for name, passed, observed, criterion in raw_checks
    ]


def analyze(
    pairs: list[tuple[Path, Path]],
    *,
    labels_path: Path,
    selection_metadata_path: Path,
    evidence_review_path: Path | None,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    criteria = SCREENING_CRITERIA
    records, provenance = collect_paper_records(
        pairs,
        labels_path,
        selection_metadata_path,
        expected_repeats=criteria["expected_repeats"],
        expected_papers=criteria["expected_papers"],
    )
    rows = aggregate_by_paper(records)
    arms = {arm: arm_metrics(rows, arm) for arm in ARMS}
    comparisons = {
        "T2_minus_T0": comparison_metrics(rows, "T2", "T0"),
        "T2_minus_T1": comparison_metrics(rows, "T2", "T1"),
        "T1_minus_T0": comparison_metrics(rows, "T1", "T0"),
    }
    interaction = _interaction_metrics(rows)
    intervals = bootstrap_intervals(
        rows,
        samples=bootstrap_samples,
        seed=seed,
        confidence=criteria["bootstrap_confidence"],
    )
    gate_pass_count = sum(row["gate_passed"] is True for row in rows)
    gate_label_counts = {
        group: {
            label: sum(
                row["gate_passed"] is gate_value and row["label"] == label for row in rows
            )
            for label in ("accept", "reject")
        }
        for group, gate_value in (("pass", True), ("fail", False))
    }
    quantitative_checks = screening_checks(
        arm_results=arms,
        comparisons=comparisons,
        interaction=interaction,
        gate_pass_count=gate_pass_count,
        gate_label_counts=gate_label_counts,
    )
    gate_pass_packages = sorted(
        [
            {"paper_id": row["paper_id"], "package_sha256": row["package_sha256"]}
            for row in rows
            if row["gate_passed"] is True
        ],
        key=lambda item: item["paper_id"],
    )
    gate_pass_result_pairs = []
    for paper_id in sorted(records):
        if records[paper_id]["gate_passed"] is not True:
            continue
        by_repeat = {
            arm: {item["repeat_id"]: item["result_sha256"] for item in records[paper_id]["by_arm"][arm]}
            for arm in ("T0", "T1")
        }
        for repeat_id in sorted(by_repeat["T0"]):
            gate_pass_result_pairs.append(
                {
                    "paper_id": paper_id,
                    "repeat_id": repeat_id,
                    "package_sha256": records[paper_id]["package_sha256"],
                    "T0_result_sha256": by_repeat["T0"][repeat_id],
                    "T1_result_sha256": by_repeat["T1"][repeat_id],
                }
            )
    evidence_review = _load_evidence_review(
        evidence_review_path,
        gate_pass_packages,
        gate_pass_result_pairs,
        provenance["experiment_signature"],
        [records[paper_id]["verification_identity"] for paper_id in sorted(records) if records[paper_id]["gate_passed"]],
    )

    failed_quantitative = [check["name"] for check in quantitative_checks if not check["passed"]]
    if failed_quantitative:
        decision = "NO_GO_ABANDON_RAG"
        reason = "One or more pre-registered quantitative screening checks failed."
    elif evidence_review["status"] == "FAIL":
        decision = "NO_GO_ABANDON_RAG"
        reason = "The completed blinded evidence review failed. This is a terminal screening result."
    elif evidence_review["status"] == "INVALID":
        decision = "INVALID_EVIDENCE_REVIEW"
        reason = "Evidence audit provenance or content is invalid; analysis is blocked, not pending a new audit."
    elif not evidence_review["passed"]:
        decision = "PROVISIONAL_NEEDS_BLINDED_EVIDENCE_REVIEW"
        reason = "Quantitative checks passed, but a valid blinded evidence-quality review is still required."
    else:
        decision = "GO_GATED_RAG_CONFIRMATORY_HOLDOUT"
        reason = "All quantitative checks and the blinded evidence-quality check passed."

    return {
        "schema_version": 1,
        "analysis": "trigger_rag_screening",
        "unit_of_analysis": "paper (metrics averaged across repetitions before inference)",
        "decision": decision,
        "decision_reason": reason,
        "screening_only": True,
        "criteria": criteria,
        "quantitative_checks": quantitative_checks,
        "failed_quantitative_checks": failed_quantitative,
        "evidence_review": evidence_review,
        "gate": {
            "pass_count": gate_pass_count,
            "fail_count": len(rows) - gate_pass_count,
            "coverage": gate_pass_count / len(rows),
            "label_counts": gate_label_counts,
            "pass_package_set_sha256": _content_sha256(gate_pass_packages),
            "pass_result_pair_count": len(gate_pass_result_pairs),
            "pass_result_pair_set_sha256": _content_sha256(gate_pass_result_pairs),
        },
        "arms": arms,
        "comparisons": comparisons,
        "gate_interaction": interaction,
        "bootstrap_intervals": intervals,
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "strata": "conference x ground-truth label",
            "confidence": criteria["bootstrap_confidence"],
        },
        "provenance": provenance,
        "paper_level": rows,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Trigger-RAG screening result",
        "",
        f"Decision: **{report['decision']}**",
        "",
        report["decision_reason"],
        "",
        "| Arm | Balanced accuracy | Accept recall | Reject recall | Accept rate | SRC | Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        metrics = report["arms"][arm]
        lines.append(
            f"| {arm} | {_fmt(metrics['balanced_accuracy'])} | {_fmt(metrics['accept_recall'])} | "
            f"{_fmt(metrics['reject_recall'])} | {_fmt(metrics['accept_rate'])} | "
            f"{_fmt(metrics['src_overall'])} | {_fmt(metrics['score_spearman'])} |"
        )
    lines.extend(["", "## Pre-registered checks", ""])
    for check in report["quantitative_checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(
            f"- {mark} — `{check['name']}`: observed {_fmt(check['observed'])}; criterion {check['criterion']}"
        )
    lines.extend(
        [
            "",
            f"Evidence review: **{report['evidence_review']['status']}** — {report['evidence_review']['reason']}",
            "",
            "This is a screening result. A GO authorizes a separate confirmatory holdout run; it is not itself confirmatory evidence.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        metavar=("SUMMARY_JSON", "EVALUATION_JSON"),
        required=True,
        help="Repeat exactly three times, once for each sealed repetition.",
    )
    parser.add_argument("--labels", type=Path, default=Path("eval/trigger_validation_36.labels.json"))
    parser.add_argument(
        "--selection-metadata",
        type=Path,
        default=Path("eval/trigger_validation_36.selection.json"),
    )
    parser.add_argument("--evidence-review", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=11766)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiment_artifacts/trigger/screening/analysis.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("experiment_artifacts/trigger/screening/analysis.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = analyze(
            [(Path(summary), Path(evaluation)) for summary, evaluation in args.pair],
            labels_path=args.labels,
            selection_metadata_path=args.selection_metadata,
            evidence_review_path=args.evidence_review,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    except AnalysisInputError as exc:
        raise SystemExit(f"Trigger analysis refused inconsistent/incomplete inputs: {exc}") from exc
    _write_json(args.output, report)
    _write_markdown(args.markdown_output, report)
    print(f"Decision: {report['decision']}")
    print(f"Gate PASS: {report['gate']['pass_count']}/{report['gate']['pass_count'] + report['gate']['fail_count']}")
    print(f"T2-T0 SRC: {report['comparisons']['T2_minus_T0']['src_overall']:.4f}")
    print(f"T2-T0 balanced accuracy: {report['comparisons']['T2_minus_T0']['balanced_accuracy']:.4f}")
    print(f"JSON report: {args.output}")
    print(f"Markdown report: {args.markdown_output}")


if __name__ == "__main__":
    main()
