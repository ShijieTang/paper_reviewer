#!/usr/bin/env python3
"""Prepare and finalize the blinded evidence audit for trigger-RAG screening.

``prepare`` consumes the three sealed trigger experiment summaries.  Only
gate-PASS paper/repetition pairs are included.  T0 and T1 are assigned to
anonymous A/B slots with a secret, repeat-block-balanced randomization. The
command writes three artifacts:

* ``blinded_pairs.json`` -- safe to give to the evidence auditor;
* ``judgments.json`` -- the auditor fills this without seeing the key; and
* ``private_key.json`` -- must remain private until judgments are sealed.

``finalize`` verifies every artifact binding, decodes A/B, and writes the
``paired_evidence_v2`` JSON accepted by ``analyze_trigger_screening.py``.
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import math
import random
import re
import secrets
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


SCHEMA_VERSION = 2
RUBRIC_VERSION = "paired_evidence_v2"
EXPECTED_REPEATS = {1, 2, 3}
REMOVED_RESULT_FIELDS = {
    "trigger_provenance",
    "rag_package",
    "rag_warnings",
    "cutoff_report",
    "trajectory",
    "turn_failures",
    "turn_outcomes",
    "execution_metadata",
}
RUBRIC = (
    "For each anonymous pair, choose the review set with more useful, correct, "
    "source-grounded literature evidence (A, B, or tie). Count material unsupported "
    "external-literature claims in A and B separately, and flag whether either full "
    "review set contains a major evidence error. Use the SAME verification_context "
    "for A and B: target paper plus raw source metadata/abstracts/links. Abstracts "
    "alone do not substantiate missing numerical or methodological claims; consult "
    "the source links and record claim-level support or missing support in notes. "
    "Confirm verification_context_checked for every pair. Do not infer arm identity."
)


def verification_source_panel(package: Any) -> list[dict[str, Any]]:
    """Expose raw source facts, never generated RAG summaries or ranking hints."""
    metadata = package.get("paper_metadata") if isinstance(package, dict) else None
    if not isinstance(metadata, list) or not metadata:
        raise EvidenceAuditError("Frozen RAG package lacks verification source metadata")
    fields = ("title", "authors", "year", "publication_date", "venue", "url", "doi", "arxiv_id", "abstract")
    sources = []
    for source in metadata:
        if not isinstance(source, dict):
            raise EvidenceAuditError("Verification source metadata is not an object")
        panel = {key: copy.deepcopy(source.get(key)) for key in fields}
        if not isinstance(panel["title"], str) or not panel["title"].strip():
            raise EvidenceAuditError("Verification source title is missing")
        if not isinstance(panel["abstract"], str) or not panel["abstract"].strip():
            raise EvidenceAuditError("Verification source abstract is missing")
        links = []
        url = str(panel.get("url") or "").strip()
        if url.startswith(("https://", "http://")):
            links.append(url)
        doi = str(panel.get("doi") or "").strip()
        if doi:
            links.append(doi if doi.startswith("https://") else f"https://doi.org/{doi}")
        arxiv_id = str(panel.get("arxiv_id") or "").strip()
        if arxiv_id:
            links.append(f"https://arxiv.org/abs/{arxiv_id}")
        if not links:
            raise EvidenceAuditError("Verification source has no usable URL/DOI/arXiv link")
        panel["verification_urls"] = sorted(set(links))
        sources.append(panel)
    return sorted(sources, key=canonical_json)


def verification_identity(paper_id: str, paper_sha256: str, package: Any) -> dict[str, str]:
    if not re.fullmatch(r"[a-f0-9]{64}", paper_sha256):
        raise EvidenceAuditError(f"{paper_id}: missing sealed target paper hash")
    return {
        "paper_id": paper_id,
        "paper_sha256": paper_sha256,
        "source_panel_sha256": content_sha256(verification_source_panel(package)),
    }


class EvidenceAuditError(ValueError):
    """Raised when source or audit artifacts fail closed validation."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise EvidenceAuditError(f"Could not hash {path}: {exc}") from exc


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceAuditError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceAuditError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise EvidenceAuditError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _private_write_json(path: Path, value: dict[str, Any], *, overwrite: bool = False) -> None:
    _write_json(path, value, overwrite=overwrite)
    try:
        path.chmod(0o600)
    except OSError as exc:
        raise EvidenceAuditError(f"Could not restrict permissions on private key {path}: {exc}") from exc


def _index_papers(summary: dict[str, Any], path: Path) -> dict[str, dict[str, Any]]:
    papers = summary.get("papers")
    if not isinstance(papers, list):
        raise EvidenceAuditError(f"{path}: papers must be a list")
    output: dict[str, dict[str, Any]] = {}
    for item in papers:
        if not isinstance(item, dict) or not str(item.get("paper_id") or ""):
            raise EvidenceAuditError(f"{path}: every paper must have an opaque paper_id")
        paper_id = str(item["paper_id"])
        if paper_id in output:
            raise EvidenceAuditError(f"{path}: duplicate paper_id {paper_id}")
        output[paper_id] = item
    return output


def _strip_private_result_fields(value: Any) -> Any:
    """Deep-copy a result while removing provenance and all direct RAG payloads."""
    if isinstance(value, dict):
        return {
            key: _strip_private_result_fields(item)
            for key, item in value.items()
            if key not in REMOVED_RESULT_FIELDS
        }
    if isinstance(value, list):
        return [_strip_private_result_fields(item) for item in value]
    return copy.deepcopy(value)


def _summary_signature(summary: dict[str, Any]) -> dict[str, Any]:
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


def _condition_result(
    paper: dict[str, Any],
    arm: str,
    *,
    context: str,
) -> dict[str, Any]:
    conditions = paper.get("conditions")
    condition = conditions.get(arm) if isinstance(conditions, dict) else None
    result = condition.get("result") if isinstance(condition, dict) else None
    if not isinstance(result, dict):
        raise EvidenceAuditError(f"{context}: missing {arm} result")
    provenance = result.get("trigger_provenance")
    if not isinstance(provenance, dict) or provenance.get("arm") != arm:
        raise EvidenceAuditError(f"{context}: invalid {arm} trigger provenance")
    return result


def collect_gate_pass_pairs(
    summary_paths: list[Path],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    """Validate three summaries and reproduce the analyzer's exact identity sets."""
    if len(summary_paths) != 3:
        raise EvidenceAuditError(f"Expected exactly 3 trigger summaries, got {len(summary_paths)}")

    common_signature: dict[str, Any] | None = None
    expected_paper_ids: set[str] | None = None
    seen_repeats: set[int] = set()
    package_by_paper: dict[str, str] = {}
    gate_by_paper: dict[str, bool] = {}
    source_pairs: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, Any]] = []
    verification_by_paper: dict[str, dict[str, str]] = {}

    for path in summary_paths:
        summary = _load_object(path)
        repeat_id = summary.get("repeat_id")
        if isinstance(repeat_id, bool) or not isinstance(repeat_id, int):
            raise EvidenceAuditError(f"{path}: repeat_id must be an integer")
        if repeat_id in seen_repeats:
            raise EvidenceAuditError(f"Duplicate repeat_id {repeat_id}")
        seen_repeats.add(repeat_id)

        signature = _summary_signature(summary)
        if signature.get("experiment") != "trigger_rag_screening":
            raise EvidenceAuditError(f"{path}: not a trigger_rag_screening summary")
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
            if not signature.get(field):
                raise EvidenceAuditError(f"{path}: missing provenance field {field}")
        if common_signature is None:
            common_signature = signature
        elif signature != common_signature:
            raise EvidenceAuditError(f"{path}: experiment provenance differs across repeats")

        paper_index = _index_papers(summary, path)
        paper_ids = set(paper_index)
        if expected_paper_ids is None:
            expected_paper_ids = paper_ids
        elif paper_ids != expected_paper_ids:
            raise EvidenceAuditError(f"{path}: paper set differs across repeats")

        for paper_id in sorted(paper_index):
            paper = paper_index[paper_id]
            package_hash = str(paper.get("package_sha256") or "")
            gate = paper.get("gate")
            if not package_hash or not isinstance(gate, dict) or not isinstance(gate.get("passed"), bool):
                raise EvidenceAuditError(f"{path}: {paper_id} has invalid package/gate data")
            gate_passed = gate["passed"]
            if paper_id not in package_by_paper:
                package_by_paper[paper_id] = package_hash
                gate_by_paper[paper_id] = gate_passed
            elif package_by_paper[paper_id] != package_hash or gate_by_paper[paper_id] != gate_passed:
                raise EvidenceAuditError(f"{paper_id}: package or gate changed across repeats")
            if not gate_passed:
                continue

            context = f"rep {repeat_id} {paper_id}"
            t0_result = _condition_result(paper, "T0", context=context)
            t1_result = _condition_result(paper, "T1", context=context)
            package = t1_result.get("rag_package")
            if not isinstance(package, dict) or content_sha256(package) != package_hash:
                raise EvidenceAuditError(f"{context}: T1 verification package differs from sealed package")
            verification = verification_identity(paper_id, str(paper.get("paper_sha256") or ""), package)
            previous = verification_by_paper.setdefault(paper_id, verification)
            if previous != verification:
                raise EvidenceAuditError(f"{context}: verification context changed across repetitions")
            for arm, result in (("T0", t0_result), ("T1", t1_result)):
                provenance = result["trigger_provenance"]
                expected = {
                    "paper_id": paper_id,
                    "repeat_id": repeat_id,
                    "package_sha256": package_hash,
                    "arm": arm,
                }
                if any(provenance.get(field) != value for field, value in expected.items()):
                    raise EvidenceAuditError(f"{context}: {arm} provenance mismatch")
            source_pairs.append(
                {
                    "paper_id": paper_id,
                    "repeat_id": repeat_id,
                    "package_sha256": package_hash,
                    "T0_result_sha256": content_sha256(t0_result),
                    "T1_result_sha256": content_sha256(t1_result),
                    "T0_result": t0_result,
                    "T1_result": t1_result,
                    "verification_identity": verification,
                }
            )
        source_artifacts.append(
            {
                "repeat_id": repeat_id,
                "summary_file": str(path),
                "summary_file_sha256": file_sha256(path),
            }
        )

    if seen_repeats != EXPECTED_REPEATS:
        raise EvidenceAuditError(
            f"Repeat IDs are {sorted(seen_repeats)}; expected {sorted(EXPECTED_REPEATS)}"
        )
    source_pairs.sort(key=lambda item: (item["paper_id"], item["repeat_id"]))
    gate_pass_packages = sorted(
        [
            {"paper_id": paper_id, "package_sha256": package_hash}
            for paper_id, package_hash in package_by_paper.items()
            if gate_by_paper[paper_id]
        ],
        key=lambda item: item["paper_id"],
    )
    analyzer_pair_set = [
        {key: item[key] for key in (
            "paper_id",
            "repeat_id",
            "package_sha256",
            "T0_result_sha256",
            "T1_result_sha256",
        )}
        for item in source_pairs
    ]
    identity = {
        "rubric_version": RUBRIC_VERSION,
        "gate_version": common_signature["gate_version"],
        "selected_paper_ids_sha256": common_signature["selected_paper_ids_sha256"],
        "audited_package_count": len(gate_pass_packages),
        "audited_package_set_sha256": content_sha256(gate_pass_packages),
        "audited_result_pair_count": len(analyzer_pair_set),
        "audited_result_pair_set_sha256": content_sha256(analyzer_pair_set),
        "verification_context_set_sha256": content_sha256(
            [verification_by_paper[paper_id] for paper_id in sorted(verification_by_paper)]
        ),
    }
    return identity, source_pairs, gate_pass_packages, {
        "experiment_signature": common_signature,
        "source_artifacts": sorted(source_artifacts, key=lambda item: item["repeat_id"]),
    }


def _balanced_assignments(source_pairs: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Randomize within each repeat while keeping A/B arm counts balanced."""
    rng = random.Random(seed)
    raw_assignments: list[dict[str, Any]] = []
    extra_a_for_t1 = bool(rng.getrandbits(1))
    for repeat_id in sorted(EXPECTED_REPEATS):
        block = [item for item in source_pairs if item["repeat_id"] == repeat_id]
        rng.shuffle(block)
        t1_in_a = len(block) // 2 + (1 if len(block) % 2 and extra_a_for_t1 else 0)
        extra_a_for_t1 = not extra_a_for_t1
        for index, item in enumerate(block):
            a_arm = "T1" if index < t1_in_a else "T0"
            b_arm = "T0" if a_arm == "T1" else "T1"
            raw_assignments.append({"source": item, "A_arm": a_arm, "B_arm": b_arm})

    rng.shuffle(raw_assignments)
    assignments = []
    for item in raw_assignments:
        source = item["source"]
        pair_id = "pair_" + secrets.token_hex(16)
        a_arm = item["A_arm"]
        b_arm = item["B_arm"]
        assignments.append(
            {
                "pair_id": pair_id,
                "paper_id": source["paper_id"],
                "repeat_id": source["repeat_id"],
                "package_sha256": source["package_sha256"],
                "A_arm": a_arm,
                "B_arm": b_arm,
                "A_result_sha256": source[f"{a_arm}_result_sha256"],
                "B_result_sha256": source[f"{b_arm}_result_sha256"],
                "A_blinded_result": _strip_private_result_fields(source[f"{a_arm}_result"]),
                "B_blinded_result": _strip_private_result_fields(source[f"{b_arm}_result"]),
                "verification_identity": source["verification_identity"],
            }
        )
    return assignments


def prepare_audit(
    summary_paths: list[Path],
    *,
    output_dir: Path,
    manifest_path: Path = Path("eval/trigger_validation_36.blinded.json"),
    md_dir: Path = Path("data/marker_md"),
    overwrite: bool = False,
) -> dict[str, Path]:
    identity, source_pairs, _packages, source_provenance = collect_gate_pass_pairs(summary_paths)
    if not source_pairs:
        raise EvidenceAuditError("No gate-PASS paper/repetition pairs are available to audit")
    seed = secrets.randbits(256)
    assignments = _balanced_assignments(source_pairs, seed)
    manifest = _load_object(manifest_path)
    contexts = {}
    for source in source_pairs:
        paper_id = source["paper_id"]
        if paper_id in contexts:
            continue
        metadata = manifest.get(paper_id)
        if not isinstance(metadata, dict) or not metadata.get("title") or not metadata.get("paper_dir"):
            raise EvidenceAuditError(f"{paper_id}: target is missing from blinded manifest")
        if set(metadata) & {"accept_or_not", "reviews", "score", "human_reviews", "decision"}:
            raise EvidenceAuditError("Use the model-input-blinded manifest, not private evaluation labels")
        frozen_title = source["T1_result"]["rag_package"].get("target_paper_summary", {}).get("title")
        if metadata["title"] != frozen_title:
            raise EvidenceAuditError(f"{paper_id}: target title differs from the frozen RAG package")
        target_path = md_dir / f"{Path(str(metadata['paper_dir'])).stem}.md"
        try:
            target_text = target_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise EvidenceAuditError(f"Missing target verification text: {target_path}") from exc
        target_hash = hashlib.sha256(target_text.encode("utf-8")).hexdigest()
        if target_hash != source["verification_identity"]["paper_sha256"] or not target_text.strip():
            raise EvidenceAuditError(f"{paper_id}: target text differs from sealed experiment input")
        contexts[paper_id] = {
            "target_paper": {"title": metadata["title"], "markdown": target_text},
            "sources": verification_source_panel(source["T1_result"]["rag_package"]),
        }

    key_assignments = []
    blind_pairs = []
    for item in assignments:
        a_blind = item["A_blinded_result"]
        b_blind = item["B_blinded_result"]
        key_assignments.append(
            {
                key: item[key]
                for key in (
                    "pair_id",
                    "paper_id",
                    "repeat_id",
                    "package_sha256",
                    "A_arm",
                    "B_arm",
                    "A_result_sha256",
                    "B_result_sha256",
                )
            }
            | {
                "A_blinded_result_sha256": content_sha256(a_blind),
                "B_blinded_result_sha256": content_sha256(b_blind),
                "verification_identity": item["verification_identity"],
                "verification_context_sha256": content_sha256(contexts[item["paper_id"]]),
            }
        )
        blind_pairs.append(
            {
                "pair_id": item["pair_id"],
                "review_set_A": a_blind,
                "review_set_B": b_blind,
                "verification_context": contexts[item["paper_id"]],
            }
        )

    key_id = content_sha256({"audit_identity": identity, "assignments": key_assignments})
    blinded = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "trigger_evidence_blinded_pairs",
        "rubric_version": RUBRIC_VERSION,
        "key_id": key_id,
        "audit_identity": identity,
        "blinded_to_labels_and_arm_identity": True,
        "rubric": RUBRIC,
        "pairs": blind_pairs,
    }
    blinded_hash = content_sha256(blinded)
    private_key = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "trigger_evidence_private_key",
        "key_id": key_id,
        "audit_identity": identity,
        "blinded_artifact_sha256": blinded_hash,
        "randomization": {
            "seed": seed,
            "method": "repeat-block-balanced A/B assignment",
        },
        "source_provenance": source_provenance,
        "assignments": key_assignments,
    }
    judgments = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "trigger_evidence_judgments",
        "rubric_version": RUBRIC_VERSION,
        "key_id": key_id,
        "audit_identity": identity,
        "blinded_artifact_sha256": blinded_hash,
        "auditor": "",
        "blinded_to_labels_and_arm_identity": None,
        "rubric": RUBRIC,
        "allowed_usefulness_winner_values": ["A", "B", "tie"],
        "judgments": [
            {
                "pair_id": item["pair_id"],
                "usefulness_winner": None,
                "unsupported_claims_A": None,
                "unsupported_claims_B": None,
                "major_error_A": None,
                "major_error_B": None,
                "verification_context_checked": None,
                "notes": "",
            }
            for item in assignments
        ],
    }

    paths = {
        "blinded": output_dir / "blinded_pairs.json",
        "key": output_dir / "private_key.json",
        "judgments": output_dir / "judgments.json",
    }
    if not overwrite:
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise EvidenceAuditError(
                "Refusing to create a partial audit over existing artifacts: " + ", ".join(existing)
            )
    _write_json(paths["blinded"], blinded, overwrite=overwrite)
    _private_write_json(paths["key"], private_key, overwrite=overwrite)
    _write_json(paths["judgments"], judgments, overwrite=overwrite)
    return paths


def _require_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceAuditError(f"{name} must be a non-negative integer")
    return value


def _reconstruct_identity_sets(assignments: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    packages_by_paper: dict[str, str] = {}
    pairs = []
    for item in assignments:
        paper_id = str(item.get("paper_id") or "")
        package_hash = str(item.get("package_sha256") or "")
        a_arm, b_arm = item.get("A_arm"), item.get("B_arm")
        if {a_arm, b_arm} != {"T0", "T1"}:
            raise EvidenceAuditError(f"{item.get('pair_id')}: private key does not map A/B to T0/T1")
        if not paper_id or not package_hash:
            raise EvidenceAuditError("Private assignment lacks paper/package identity")
        previous = packages_by_paper.setdefault(paper_id, package_hash)
        if previous != package_hash:
            raise EvidenceAuditError(f"{paper_id}: package hash differs within private key")
        arm_hashes = {
            a_arm: item.get("A_result_sha256"),
            b_arm: item.get("B_result_sha256"),
        }
        pairs.append(
            {
                "paper_id": paper_id,
                "repeat_id": item.get("repeat_id"),
                "package_sha256": package_hash,
                "T0_result_sha256": arm_hashes["T0"],
                "T1_result_sha256": arm_hashes["T1"],
            }
        )
    packages = sorted(
        [
            {"paper_id": paper_id, "package_sha256": package_hash}
            for paper_id, package_hash in packages_by_paper.items()
        ],
        key=lambda item: item["paper_id"],
    )
    pairs.sort(key=lambda item: (item["paper_id"], item["repeat_id"]))
    return packages, pairs


def _validate_artifact_bindings(
    blinded: dict[str, Any],
    private_key: dict[str, Any],
    judgments: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    for document in (blinded, private_key, judgments):
        if document.get("schema_version") != SCHEMA_VERSION:
            raise EvidenceAuditError("Unsupported audit schema; verification context is required")
    identity = private_key.get("audit_identity")
    assignments = private_key.get("assignments")
    if not isinstance(identity, dict) or not isinstance(assignments, list) or not assignments:
        raise EvidenceAuditError("Private key lacks audit identity or assignments")
    key_id = content_sha256({"audit_identity": identity, "assignments": assignments})
    if private_key.get("key_id") != key_id:
        raise EvidenceAuditError("Private key content does not match key_id")
    if content_sha256(blinded) != private_key.get("blinded_artifact_sha256"):
        raise EvidenceAuditError("Blinded artifact hash does not match private key")
    for document, name in ((blinded, "blinded artifact"), (judgments, "judgments")):
        if document.get("key_id") != key_id or document.get("audit_identity") != identity:
            raise EvidenceAuditError(f"{name} is bound to a different key/identity")
    if judgments.get("blinded_artifact_sha256") != private_key.get("blinded_artifact_sha256"):
        raise EvidenceAuditError("Judgments are bound to a different blinded artifact")

    packages, result_pairs = _reconstruct_identity_sets(assignments)
    expected = {
        "audited_package_count": len(packages),
        "audited_package_set_sha256": content_sha256(packages),
        "audited_result_pair_count": len(result_pairs),
        "audited_result_pair_set_sha256": content_sha256(result_pairs),
    }
    for field, value in expected.items():
        if identity.get(field) != value:
            raise EvidenceAuditError(f"Private key {field} does not match its assignments")

    blind_items = blinded.get("pairs")
    if not isinstance(blind_items, list):
        raise EvidenceAuditError("Blinded artifact pairs must be a list")
    blind_by_id: dict[str, dict[str, Any]] = {}
    for item in blind_items:
        pair_id = str(item.get("pair_id") or "") if isinstance(item, dict) else ""
        if not pair_id or pair_id in blind_by_id:
            raise EvidenceAuditError("Blinded artifact contains a missing/duplicate pair_id")
        blind_by_id[pair_id] = item

    assignment_by_id: dict[str, dict[str, Any]] = {}
    verification_by_paper = {}
    for item in assignments:
        pair_id = str(item.get("pair_id") or "") if isinstance(item, dict) else ""
        if not pair_id or pair_id in assignment_by_id:
            raise EvidenceAuditError("Private key contains a missing/duplicate pair_id")
        assignment_by_id[pair_id] = item
        blind_item = blind_by_id.get(pair_id)
        if blind_item is None:
            raise EvidenceAuditError(f"Blinded artifact is missing {pair_id}")
        context = blind_item.get("verification_context")
        if not isinstance(context, dict) or content_sha256(context) != item.get("verification_context_sha256"):
            raise EvidenceAuditError(f"{pair_id}: verification context missing or changed")
        target = context.get("target_paper", {})
        target_text = target.get("markdown") if isinstance(target, dict) else None
        if not isinstance(target_text, str) or not target_text.strip():
            raise EvidenceAuditError(f"{pair_id}: verification target text missing")
        verified = {
            "paper_id": item["paper_id"],
            "paper_sha256": hashlib.sha256(target_text.encode("utf-8")).hexdigest(),
            "source_panel_sha256": content_sha256(context.get("sources")),
        }
        if verified != item.get("verification_identity") or not context.get("sources"):
            raise EvidenceAuditError(f"{pair_id}: verification identity mismatch")
        old = verification_by_paper.setdefault(item["paper_id"], verified)
        if old != verified:
            raise EvidenceAuditError("Verification context differs across repetitions")
        for slot in ("A", "B"):
            if content_sha256(blind_item.get(f"review_set_{slot}")) != item.get(
                f"{slot}_blinded_result_sha256"
            ):
                raise EvidenceAuditError(f"{pair_id}: blinded review set {slot} was modified")
    if set(blind_by_id) != set(assignment_by_id):
        raise EvidenceAuditError("Blinded artifact and private key pair sets differ")
    if identity.get("verification_context_set_sha256") != content_sha256(
        [verification_by_paper[key] for key in sorted(verification_by_paper)]
    ):
        raise EvidenceAuditError("Verification context set hash differs from sealed identity")
    return identity, assignment_by_id, result_pairs


def summarize_decoded_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Use each paper once; the three dependent repetitions are descriptive."""
    by_paper = defaultdict(list)
    usefulness = {"rag_wins": 0, "ties": 0, "rag_losses": 0}
    unsupported = {"T0": 0, "T1": 0}
    major_errors = {"T0": 0, "T1": 0}
    for record in records:
        outcome = record.get("rag_usefulness_outcome")
        if outcome not in {"win", "tie", "loss"}:
            raise EvidenceAuditError("Invalid decoded usefulness outcome")
        usefulness[{"win": "rag_wins", "tie": "ties", "loss": "rag_losses"}[outcome]] += 1
        for arm in ("T0", "T1"):
            unsupported[arm] += _require_nonnegative_int(record.get("unsupported_claims", {}).get(arm), "unsupported claims")
            error = record.get("major_error", {}).get(arm)
            if not isinstance(error, bool):
                raise EvidenceAuditError("Decoded major-error flags must be boolean")
            major_errors[arm] += int(error)
        if record.get("verification_context_checked") is not True:
            raise EvidenceAuditError("Every judgment must certify verification context was checked")
        by_paper[record["paper_id"]].append(record)
    if not by_paper:
        raise EvidenceAuditError("No decoded judgments")
    papers = []
    for paper_id, group in sorted(by_paper.items()):
        if len(group) != 3 or {item["repeat_id"] for item in group} != EXPECTED_REPEATS:
            raise EvidenceAuditError(f"{paper_id}: expected one judgment for each of three repetitions")
        papers.append({
            "paper_id": paper_id,
            "usefulness_score": mean({"win": 1.0, "tie": 0.5, "loss": 0.0}[item["rag_usefulness_outcome"]] for item in group),
            "unsupported_claims": {arm: mean(item["unsupported_claims"][arm] for item in group) for arm in ("T0", "T1")},
            "major_error_rate": {arm: mean(int(item["major_error"][arm]) for item in group) for arm in ("T0", "T1")},
        })
    return {
        "paired_usefulness": usefulness,
        "unsupported_claims": unsupported,
        "major_error_counts": major_errors,
        "paper_balanced": {
            "paper_count": len(papers),
            "usefulness_score": mean(item["usefulness_score"] for item in papers),
            "unsupported_claims": {arm: mean(item["unsupported_claims"][arm] for item in papers) for arm in ("T0", "T1")},
            "major_error_rate": {arm: mean(item["major_error_rate"][arm] for item in papers) for arm in ("T0", "T1")},
        },
        "per_paper": papers,
    }


def evidence_metrics_pass(metrics: dict[str, Any]) -> bool:
    values = metrics["paper_balanced"]
    rates = values["major_error_rate"]
    return (values["usefulness_score"] >= 0.60
            and values["unsupported_claims"]["T1"] <= values["unsupported_claims"]["T0"]
            and rates["T1"] <= 0.10
            and rates["T1"] <= rates["T0"])


def finalize_audit(
    *,
    blinded_path: Path,
    private_key_path: Path,
    judgments_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    blinded = _load_object(blinded_path)
    private_key = _load_object(private_key_path)
    judgments = _load_object(judgments_path)
    identity, assignment_by_id, _result_pairs = _validate_artifact_bindings(
        blinded, private_key, judgments
    )

    auditor = str(judgments.get("auditor") or "").strip()
    if not auditor:
        raise EvidenceAuditError("judgments.auditor must be filled")
    if judgments.get("blinded_to_labels_and_arm_identity") is not True:
        raise EvidenceAuditError(
            "Auditor must certify blinded_to_labels_and_arm_identity=true before finalization"
        )
    judgment_items = judgments.get("judgments")
    if not isinstance(judgment_items, list):
        raise EvidenceAuditError("judgments.judgments must be a list")
    judgment_by_id: dict[str, dict[str, Any]] = {}
    for item in judgment_items:
        pair_id = str(item.get("pair_id") or "") if isinstance(item, dict) else ""
        if not pair_id or pair_id in judgment_by_id:
            raise EvidenceAuditError("Judgments contain a missing/duplicate pair_id")
        judgment_by_id[pair_id] = item
    if set(judgment_by_id) != set(assignment_by_id):
        raise EvidenceAuditError("Judgment and private-key pair sets differ")

    decoded_records = []
    for pair_id in sorted(assignment_by_id):
        assignment = assignment_by_id[pair_id]
        judgment = judgment_by_id[pair_id]
        winner_raw = str(judgment.get("usefulness_winner") or "").strip()
        winner = winner_raw.upper() if winner_raw.lower() != "tie" else "tie"
        if winner not in {"A", "B", "tie"}:
            raise EvidenceAuditError(f"{pair_id}: usefulness_winner must be A, B, or tie")
        claims_by_slot = {
            slot: _require_nonnegative_int(
                judgment.get(f"unsupported_claims_{slot}"),
                f"{pair_id}.unsupported_claims_{slot}",
            )
            for slot in ("A", "B")
        }
        errors_by_slot = {}
        for slot in ("A", "B"):
            value = judgment.get(f"major_error_{slot}")
            if not isinstance(value, bool):
                raise EvidenceAuditError(f"{pair_id}.major_error_{slot} must be boolean")
            errors_by_slot[slot] = value

        if winner == "tie":
            decoded_winner = "tie"
        else:
            decoded_winner = assignment[f"{winner}_arm"]
        decoded_records.append(
            {
                "pair_id": pair_id,
                "paper_id": assignment["paper_id"],
                "repeat_id": assignment["repeat_id"],
                "verification_context_checked": judgment.get("verification_context_checked"),
                "rag_usefulness_outcome": (
                    "win" if decoded_winner == "T1" else "loss" if decoded_winner == "T0" else "tie"
                ),
                "unsupported_claims": {
                    assignment["A_arm"]: claims_by_slot["A"],
                    assignment["B_arm"]: claims_by_slot["B"],
                },
                "major_error": {
                    assignment["A_arm"]: errors_by_slot["A"],
                    assignment["B_arm"]: errors_by_slot["B"],
                },
            }
        )

    metrics = summarize_decoded_records(decoded_records)
    passed = evidence_metrics_pass(metrics)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if passed else "FAIL",
        **identity,
        "blinded_to_labels_and_arm_identity": True,
        "auditor": auditor,
        **metrics,
        "decoded_records": decoded_records,
        "rubric": RUBRIC,
        "notes": (
            "Finalized from hash-bound blinded A/B judgments with verified target/source context. "
            "Criteria use paper means; repetition counts are descriptive, not independent samples."
        ),
        "audit_diagnostics": {
            "key_id": private_key["key_id"],
            "blinded_artifact_sha256": private_key["blinded_artifact_sha256"],
            "decoded_records_sha256": content_sha256(decoded_records),
            "judgments_sha256": content_sha256(judgments),
        },
    }
    _write_json(output_path, result, overwrite=overwrite)
    return result


def _expand_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = sorted(glob.glob(value))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(value))
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create blinded pairs, private key, and judgments template.")
    prepare.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Trigger summary path or quoted glob; repeat for multiple paths (exactly 3 after expansion).",
    )
    prepare.add_argument("--output-dir", type=Path, default=Path("experiment_artifacts/trigger/evidence_audit"))
    prepare.add_argument("--manifest", type=Path, default=Path("eval/trigger_validation_36.blinded.json"))
    prepare.add_argument("--md-dir", type=Path, default=Path("data/marker_md"))
    prepare.add_argument("--overwrite", action="store_true")

    finalize = subparsers.add_parser("finalize", help="Verify and decode completed blinded judgments.")
    finalize.add_argument("--audit-dir", type=Path, default=Path("experiment_artifacts/trigger/evidence_audit"))
    finalize.add_argument("--blinded", type=Path, default=None)
    finalize.add_argument("--private-key", type=Path, default=None)
    finalize.add_argument("--judgments", type=Path, default=None)
    finalize.add_argument("--output", type=Path, default=Path("experiment_artifacts/trigger/evidence_audit/evidence_review.json"))
    finalize.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        if args.command == "prepare":
            paths = prepare_audit(
                _expand_paths(args.summary),
                output_dir=args.output_dir,
                manifest_path=args.manifest,
                md_dir=args.md_dir,
                overwrite=args.overwrite,
            )
            print(f"Blinded pairs: {paths['blinded']}")
            print(f"Judgments template: {paths['judgments']}")
            print(f"PRIVATE key (do not give to auditor): {paths['key']}")
            return

        audit_dir = args.audit_dir
        result = finalize_audit(
            blinded_path=args.blinded or audit_dir / "blinded_pairs.json",
            private_key_path=args.private_key or audit_dir / "private_key.json",
            judgments_path=args.judgments or audit_dir / "judgments.json",
            output_path=args.output,
            overwrite=args.overwrite,
        )
        print(f"Evidence audit status: {result['status']}")
        print(f"Analyzer-compatible evidence review: {args.output}")
    except EvidenceAuditError as exc:
        raise SystemExit(f"Evidence audit error: {exc}") from exc


if __name__ == "__main__":
    main()
