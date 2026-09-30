#!/usr/bin/env python3
"""Run the final quality-gated related-work RAG screening experiment.

The expensive model calls are only T0 (no RAG) and T1 (always RAG).  T2 is a
deterministic policy result: it selects T1 when the frozen package passes the
pre-registered integrity gate, otherwise it selects T0.

Typical workflow:
  1. --prepare-rag-only   build each package once and freeze it, including FAILs
  2. --audit-rag-only     report gate coverage without regenerating packages
  3. normal execution     run T0/T1 and derive T2 for one repetition
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import VALID_TOPICS
from mas_loop import main as mas_main
from rag import build_rag_package, format_rag_prompt_block
from rag.config import RAGConfig
from rag.security import prompt_injection_warnings
from review_schema import review_is_valid, validate_conference_schema, workflow_is_complete
from eval.prepare_trigger_evidence_audit import EvidenceAuditError, verification_source_panel
from eval.evaluation_protocol import EMBED_MODEL, EMBED_REVISION


DEFAULT_PROVIDER = "openrouter"
DEFAULT_MODELS = {
    "deepseek": "deepseek-v4-flash",
    "openrouter": "deepseek/deepseek-chat-v3.1",
}
GATE_VERSION = "integrity_v1"
PACKAGE_SCHEMA_VERSION = 2
SUMMARY_SCHEMA_VERSION = 1
REVIEWER_TYPES = ["reviewer_nopersona"] * 3
N_ITER = 3
RUN_CITATION_CHECK = False
ENABLE_AI_DETECTOR = False
INVALID_RESULT_POLICY = "fixed_role_schema_retries_all_required_turns_then_seal_without_resampling"
CONDITIONS = {
    "T0": {
        "desc": "No RAG; 3 neutral reviewers, 3 iterations + Author",
        "agents": REVIEWER_TYPES,
        "n_iter": N_ITER,
        "agenttype": "NNN",
        "enable_rag": False,
        "derived_policy": False,
    },
    "T1": {
        "desc": "Always frozen related-work RAG; 3 neutral reviewers, 3 iterations + Author",
        "agents": REVIEWER_TYPES,
        "n_iter": N_ITER,
        "agenttype": "NNN",
        "enable_rag": True,
        "derived_policy": False,
    },
    "T2": {
        "desc": "Derived gated-RAG policy: T1 on PASS, otherwise T0",
        "agents": REVIEWER_TYPES,
        "n_iter": N_ITER,
        "agenttype": "NNN",
        "enable_rag": "gated",
        "derived_policy": True,
    },
}
SENSITIVE_MANIFEST_FIELDS = {
    "accept_or_not",
    "score",
    "reviews",
    "strengths",
    "weaknesses",
    "ground_truth",
    "collection_decision_category",
}


class FrozenPackageError(RuntimeError):
    """Raised when a frozen package is absent or fails provenance checks."""


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


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenPackageError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FrozenPackageError(f"{path} must contain a JSON object.")
    return value


def normalize_topic(topic: Any) -> str:
    text = str(topic or "").strip()
    for valid in VALID_TOPICS:
        if text.casefold() == valid.casefold():
            return valid
    return "Others"


def normalize_title(value: Any) -> str:
    text = str(value or "").casefold()
    for apostrophe in ("'", "’", "‘", "ʼ", "`"):
        text = text.replace(apostrophe, "")
    decomposed = unicodedata.normalize("NFKD", text)
    separated = "".join(
        " " if unicodedata.category(character)[:1] in {"P", "S"} else character
        for character in decomposed
    )
    ascii_text = separated.encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text))


def _title_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize_title(left).split())
    right_tokens = set(normalize_title(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def load_papers(path: Path, *, allow_sensitive: bool = False) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read paper manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a JSON object keyed by opaque paper ID.")
    papers = []
    for paper_id, metadata in data.items():
        if not isinstance(metadata, dict):
            raise SystemExit(f"Paper {paper_id!r} is not a JSON object.")
        sensitive = sorted(set(metadata) & SENSITIVE_MANIFEST_FIELDS)
        if sensitive and not allow_sensitive:
            raise SystemExit(
                f"{path} contains sensitive experiment fields for {paper_id}: {sensitive}. "
                "Use the blinded trigger manifest. --allow-sensitive-manifest is for development smoke tests only."
            )
        papers.append({"paper_id": paper_id, **metadata})
    return papers


def load_markdown(paper_meta: dict[str, Any], md_dir: Path) -> str:
    # paper_dir is used only to locate the cached file.  It is never forwarded
    # to the gate, RAG builder, or reviewer prompts.
    paper_dir = str(paper_meta.get("paper_dir") or "")
    if not paper_dir:
        raise FileNotFoundError(f"{paper_meta['paper_id']}: paper_dir is missing")
    markdown_path = md_dir / f"{Path(paper_dir).stem}.md"
    if not markdown_path.is_file():
        raise FileNotFoundError(f"Cached Markdown not found: {markdown_path}")
    return markdown_path.read_text(encoding="utf-8")


def _safe_artifact_stem(paper_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", paper_id).strip("-._") or "paper"
    suffix = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:80]}-{suffix}"


def package_path(package_dir: Path, paper_id: str) -> Path:
    return package_dir / f"{_safe_artifact_stem(paper_id)}.rag.json"


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _has_traceable_identifier(metadata: dict[str, Any]) -> bool:
    if any(str(metadata.get(key) or "").strip() for key in ("url", "doi", "arxiv_id")):
        return True
    source_ids = metadata.get("source_ids")
    return isinstance(source_ids, dict) and any(str(value or "").strip() for value in source_ids.values())


def evaluate_integrity_gate(
    package: Any,
    expected_cutoff: str,
    authoritative_target_title: str = "",
) -> dict[str, Any]:
    """Return a deterministic, fail-closed integrity gate decision."""
    reasons: list[str] = []
    checks: dict[str, bool] = {}
    diagnostics: dict[str, Any] = {"expected_cutoff": expected_cutoff}

    def fail(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    if not isinstance(package, dict):
        return {
            "version": GATE_VERSION,
            "passed": False,
            "reason_codes": ["PACKAGE_NOT_OBJECT"],
            "checks": {"package_object": False},
            "diagnostics": diagnostics,
        }
    checks["package_object"] = True

    package_id = str(package.get("rag_package_id") or "").strip()
    checks["package_id_present"] = bool(package_id)
    if not package_id:
        fail("PACKAGE_ID_MISSING")

    warnings = package.get("warnings")
    checks["warnings_list"] = isinstance(warnings, list)
    checks["warnings_empty"] = isinstance(warnings, list) and len(warnings) == 0
    diagnostics["warning_count"] = len(warnings) if isinstance(warnings, list) else None
    if not isinstance(warnings, list):
        fail("WARNINGS_INVALID")
    elif warnings:
        fail("WARNINGS_PRESENT")

    summary = package.get("related_work_summary")
    checks["summary_nonempty"] = isinstance(summary, str) and bool(summary.strip())
    if not checks["summary_nonempty"]:
        fail("SUMMARY_EMPTY")

    cutoff = package.get("cutoff_report")
    cutoff_valid = isinstance(cutoff, dict)
    checks["cutoff_report_object"] = cutoff_valid
    if not cutoff_valid:
        fail("CUTOFF_INVALID")
        cutoff = {}
    reported_cutoff = str(cutoff.get("cutoff_date") or "")
    checks["cutoff_matches"] = reported_cutoff == expected_cutoff
    if reported_cutoff != expected_cutoff:
        fail("CUTOFF_MISMATCH")
    num_used = cutoff.get("num_used")
    positive_num_used = isinstance(num_used, int) and not isinstance(num_used, bool) and num_used > 0
    checks["num_used_positive"] = positive_num_used
    diagnostics["num_used"] = num_used
    if not positive_num_used:
        fail("NO_EVIDENCE")

    metadata_list = package.get("paper_metadata")
    metadata_valid = isinstance(metadata_list, list) and bool(metadata_list) and all(
        isinstance(item, dict) for item in metadata_list
    )
    checks["metadata_valid"] = metadata_valid
    if not metadata_valid:
        fail("METADATA_INVALID")
        metadata_list = []
    try:
        verification_source_panel(package)
        checks["verification_materials_available"] = True
    except EvidenceAuditError:
        checks["verification_materials_available"] = False
        fail("VERIFICATION_MATERIALS_MISSING")

    reranked = package.get("reranking_results")
    rerank_valid = isinstance(reranked, list) and bool(reranked) and all(isinstance(item, dict) for item in reranked)
    checks["reranking_valid"] = rerank_valid
    if not rerank_valid:
        fail("RERANKING_INVALID")
        reranked = []

    metadata_by_id: dict[str, dict[str, Any]] = {}
    duplicate_metadata_ids = []
    for item in metadata_list:
        paper_id = str(item.get("paper_id") or "").strip()
        if not paper_id:
            fail("METADATA_ID_MISSING")
            continue
        if paper_id in metadata_by_id:
            duplicate_metadata_ids.append(paper_id)
            fail("DUPLICATE_METADATA_ID")
        metadata_by_id[paper_id] = item

    reranked_ids = [str(item.get("paper_id") or "").strip() for item in reranked]
    if any(not paper_id for paper_id in reranked_ids):
        fail("RERANK_ID_MISSING")
    if len(set(reranked_ids)) != len(reranked_ids):
        fail("DUPLICATE_RERANK_ID")
    unresolved_ids = sorted({paper_id for paper_id in reranked_ids if paper_id not in metadata_by_id})
    if unresolved_ids:
        fail("UNRESOLVED_RERANK_ID")
    checks["reranked_ids_resolve"] = bool(reranked_ids) and not unresolved_ids
    diagnostics["unresolved_rerank_ids"] = unresolved_ids
    diagnostics["duplicate_metadata_ids"] = sorted(set(duplicate_metadata_ids))

    if positive_num_used and metadata_list and num_used != len(metadata_list):
        fail("NUM_USED_MISMATCH")
    checks["num_used_matches_metadata"] = positive_num_used and num_used == len(metadata_list)

    target_summary = package.get("target_paper_summary")
    parsed_target_title = target_summary.get("title", "") if isinstance(target_summary, dict) else ""
    target_title = str(authoritative_target_title or "").strip() or parsed_target_title
    checks["target_title_present"] = bool(normalize_title(target_title))
    if not checks["target_title_present"]:
        fail("TARGET_TITLE_MISSING")
    title_matches_authority = not authoritative_target_title or (
        normalize_title(parsed_target_title) == normalize_title(authoritative_target_title)
    )
    checks["target_title_matches_authority"] = title_matches_authority
    if not title_matches_authority:
        fail("TARGET_TITLE_MISMATCH")

    cutoff_date = _parse_date(expected_cutoff)
    for evidence_id in sorted(set(reranked_ids)):
        item = metadata_by_id.get(evidence_id)
        if not item:
            continue
        if not str(item.get("title") or "").strip():
            fail("SOURCE_TITLE_MISSING")
        if not _has_traceable_identifier(item):
            fail("SOURCE_UNTRACEABLE")

        publication_date_raw = str(item.get("publication_date") or "").strip()
        publication_date = _parse_date(publication_date_raw)
        year = item.get("year")
        year_valid = isinstance(year, int) and not isinstance(year, bool) and 1 <= year <= 9999
        if publication_date_raw and publication_date is None:
            fail("SOURCE_DATE_INVALID")
        if publication_date is None and not year_valid:
            fail("SOURCE_DATE_INVALID")
        if cutoff_date is not None:
            if publication_date is not None and publication_date > cutoff_date:
                fail("POST_CUTOFF_SOURCE")
            if publication_date is None and year_valid and year > cutoff_date.year:
                fail("POST_CUTOFF_SOURCE")

        if target_title and _title_jaccard(target_title, str(item.get("title") or "")) >= 0.90:
            fail("TARGET_DUPLICATE")

        target_abstract = target_summary.get("abstract", "") if isinstance(target_summary, dict) else ""
        evidence_abstract = str(item.get("abstract") or "")
        target_abstract_tokens = normalize_title(target_abstract).split()
        evidence_abstract_tokens = normalize_title(evidence_abstract).split()
        if len(target_abstract_tokens) >= 40 and len(evidence_abstract_tokens) >= 40:
            abstract_overlap = len(set(target_abstract_tokens) & set(evidence_abstract_tokens)) / len(
                set(target_abstract_tokens) | set(evidence_abstract_tokens)
            )
            if abstract_overlap >= 0.80:
                fail("TARGET_ABSTRACT_DUPLICATE")

        security_text = "\n".join(
            str(item.get(key) or "") for key in ("title", "abstract")
        )
        if prompt_injection_warnings(security_text, evidence_id):
            fail("PROMPT_INJECTION")

    review_memory = package.get("review_memory")
    # Current related-work-only packages omit this removed subsystem. Older
    # packages remain admissible only when they explicitly disabled it.
    review_memory_disabled = "review_memory" not in package or (
        isinstance(review_memory, dict) and review_memory.get("status") == "disabled")
    checks["review_memory_disabled"] = review_memory_disabled
    if not review_memory_disabled:
        fail("REVIEW_MEMORY_NOT_DISABLED")

    checks["sources_traceable"] = "SOURCE_UNTRACEABLE" not in reasons
    checks["source_dates_valid"] = not ({"SOURCE_DATE_INVALID", "POST_CUTOFF_SOURCE"} & set(reasons))
    checks["target_duplicate_absent"] = not (
        {"TARGET_DUPLICATE", "TARGET_ABSTRACT_DUPLICATE"} & set(reasons)
    )
    checks["prompt_injection_absent"] = "PROMPT_INJECTION" not in reasons
    diagnostics["metadata_count"] = len(metadata_list)
    diagnostics["reranked_count"] = len(reranked)
    diagnostics["target_title"] = target_title
    diagnostics["parsed_target_title"] = parsed_target_title

    return {
        "version": GATE_VERSION,
        "passed": len(reasons) == 0,
        "reason_codes": reasons,
        "checks": checks,
        "diagnostics": diagnostics,
    }


def _prompt_bundle_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted((REPO_ROOT / "prompts").glob("*.py")):
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _code_bundle_sha256() -> str:
    paths = [
        Path(__file__),
        REPO_ROOT / "eval" / "analyze_trigger_screening.py",
        REPO_ROOT / "eval" / "prepare_trigger_evidence_audit.py",
        REPO_ROOT / "eval" / "trigger_evidence_review.template.json",
        REPO_ROOT / "eval" / "evaluation.py",
        REPO_ROOT / "eval" / "SRC.py",
        REPO_ROOT / "eval" / "evaluation_protocol.py",
        REPO_ROOT / "review_schema.py",
        REPO_ROOT / "mas_loop.py",
        REPO_ROOT / "agents.py",
        REPO_ROOT / "config.py",
        REPO_ROOT / "requirements.txt",
    ]
    paths.extend(sorted((REPO_ROOT / "rag").rglob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_revision() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "tracked_worktree_dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def rag_config(cutoff_date: str) -> RAGConfig:
    if _parse_date(cutoff_date) is None or len(str(cutoff_date)) != 10:
        raise ValueError(f"cutoff_date must be ISO YYYY-MM-DD, got {cutoff_date!r}")
    return RAGConfig(
        enable_rag=True,
        enable_related_work_rag=True,
        cutoff_date=cutoff_date,
        allow_undated_evidence=False,
    )


def _expected_package_context(
    paper_meta: dict[str, Any],
    paper_text: str,
    provider: str,
    model: str,
    config: RAGConfig,
) -> dict[str, Any]:
    config_dict = asdict(config)
    manifest_title = str(paper_meta.get("title") or "").strip()
    if not manifest_title:
        raise FrozenPackageError(f"{paper_meta['paper_id']}: authoritative manifest title is missing")
    return {
        "manifest_paper_id": paper_meta["paper_id"],
        "manifest_title": manifest_title,
        "topic": normalize_topic(paper_meta.get("topic")),
        "paper_sha256": text_sha256(paper_text),
        "provider": provider,
        "model": model,
        "rag_config": config_dict,
        "rag_config_sha256": content_sha256(config_dict),
        # A partially completed prepare step may be resumed, so bind every
        # envelope to the exact source bundle that generated its package.
        # Mixing packages built by different code revisions would invalidate
        # the paired trigger experiment even if provider/model/config match.
        "generation_code_bundle_sha256": _code_bundle_sha256(),
    }


def validate_frozen_envelope(
    envelope: dict[str, Any],
    expected: dict[str, Any],
    expected_cutoff: str,
) -> dict[str, Any]:
    if envelope.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise FrozenPackageError("Frozen package schema_version mismatch.")
    for field in (
        "manifest_paper_id",
        "manifest_title",
        "topic",
        "paper_sha256",
        "provider",
        "model",
        "rag_config_sha256",
        "generation_code_bundle_sha256",
    ):
        if envelope.get(field) != expected.get(field):
            raise FrozenPackageError(f"Frozen package {field} mismatch.")
    if envelope.get("rag_config") != expected.get("rag_config"):
        raise FrozenPackageError("Frozen package rag_config mismatch.")
    package = envelope.get("package")
    if not isinstance(package, dict):
        raise FrozenPackageError("Frozen package payload is missing or invalid.")
    actual_hash = content_sha256(package)
    if envelope.get("package_sha256") != actual_hash:
        raise FrozenPackageError("Frozen package payload hash mismatch.")
    prompt_block = format_rag_prompt_block(package)
    if envelope.get("rag_prompt_block_sha256") != text_sha256(prompt_block):
        raise FrozenPackageError("Frozen RAG prompt-block hash mismatch.")
    current_gate = evaluate_integrity_gate(package, expected_cutoff, expected["manifest_title"])
    if envelope.get("gate") != current_gate:
        raise FrozenPackageError("Stored gate decision does not match the frozen payload.")
    return envelope


def prepare_or_load_package(
    paper_meta: dict[str, Any],
    paper_text: str,
    *,
    package_dir: Path,
    provider: str,
    model: str,
    api_key: str,
    config: RAGConfig,
    overwrite: bool = False,
) -> tuple[dict[str, Any], bool]:
    path = package_path(package_dir, paper_meta["paper_id"])
    expected = _expected_package_context(paper_meta, paper_text, provider, model, config)
    if path.exists() and not overwrite:
        return validate_frozen_envelope(_load_json_object(path), expected, config.cutoff_date), True

    package = build_rag_package(
        paper=paper_text,
        topic=expected["topic"],
        target_title=expected["manifest_title"],
        provider=provider,
        model=model,
        api_key=api_key,
        config=config,
    )
    if not isinstance(package, dict):
        raise FrozenPackageError(f"RAG builder returned {type(package).__name__}, expected object.")
    prompt_block = format_rag_prompt_block(package)
    envelope = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        **expected,
        "created_at": datetime.now().astimezone().isoformat(),
        "package_sha256": content_sha256(package),
        "rag_prompt_block_sha256": text_sha256(prompt_block),
        "gate": evaluate_integrity_gate(package, config.cutoff_date, expected["manifest_title"]),
        "package": package,
    }
    _atomic_write_json(path, envelope)
    return envelope, False


def load_required_package(
    paper_meta: dict[str, Any],
    paper_text: str,
    *,
    package_dir: Path,
    provider: str,
    model: str,
    config: RAGConfig,
) -> dict[str, Any]:
    path = package_path(package_dir, paper_meta["paper_id"])
    if not path.is_file():
        raise FrozenPackageError(
            f"Missing frozen package for {paper_meta['paper_id']}: {path}. Run --prepare-rag-only first."
        )
    expected = _expected_package_context(paper_meta, paper_text, provider, model, config)
    return validate_frozen_envelope(_load_json_object(path), expected, config.cutoff_date)


def _valid_review(review: Any) -> bool:
    return review_is_valid(review)


def _conference_result_is_complete(value: Any) -> bool:
    return not validate_conference_schema(value)


def result_is_complete(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    reviewers = result.get("reviewers")
    return (
        isinstance(reviewers, list)
        and len(reviewers) == len(REVIEWER_TYPES)
        and all(_valid_review(review) for review in reviewers)
        and _conference_result_is_complete(result.get("conference"))
        and workflow_is_complete(result, len(REVIEWER_TYPES), N_ITER)
    )


def _arm_input_text(paper_text: str, arm: str, package: dict[str, Any]) -> str:
    if arm == "T0":
        return paper_text
    block = format_rag_prompt_block(package)
    return paper_text + "\n\n" + block if block else paper_text


def _result_path(output_dir: Path, paper_id: str, repeat_id: int, arm: str) -> Path:
    return output_dir / f"{_safe_artifact_stem(paper_id)}.rep-{repeat_id}.{arm}.json"


def balanced_condition_orders(
    paper_ids: list[str],
    *,
    seed: int,
    repeat_id: int,
) -> dict[str, list[str]]:
    """Assign T0-first/T1-first orders with counts differing by at most one."""
    shuffled = sorted(paper_ids)
    random.Random(f"{seed}:{repeat_id}:balanced-order").shuffle(shuffled)
    t0_first = set(shuffled[: (len(shuffled) + 1) // 2])
    return {
        paper_id: (["T0", "T1"] if paper_id in t0_first else ["T1", "T0"])
        for paper_id in paper_ids
    }


def _provenance(
    *,
    paper_meta: dict[str, Any],
    paper_text: str,
    arm: str,
    repeat_id: int,
    execution_order: int | None,
    provider: str,
    model: str,
    envelope: dict[str, Any],
    prompt_sha256: str,
    code_sha256: str,
    git_revision: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "trigger_rag_screening",
        "paper_id": paper_meta["paper_id"],
        "repeat_id": repeat_id,
        "arm": arm,
        "execution_order": execution_order,
        "provider": provider,
        "model": model,
        "paper_sha256": text_sha256(paper_text),
        "package_sha256": envelope["package_sha256"],
        "rag_config_sha256": envelope["rag_config_sha256"],
        "gate_version": envelope["gate"]["version"],
        "gate_passed": envelope["gate"]["passed"],
        "prompt_bundle_sha256": prompt_sha256,
        "code_bundle_sha256": code_sha256,
        "git": git_revision,
        "reviewer_input_sha256": text_sha256(_arm_input_text(paper_text, arm, envelope["package"])),
        "derived_policy": arm == "T2",
        "invalid_result_policy": INVALID_RESULT_POLICY,
        **_frozen_evaluation_protocol(),
    }


def _frozen_evaluation_protocol() -> dict[str, str]:
    return {
        "expected_evaluator_sha256": hashlib.sha256((REPO_ROOT / "eval" / "evaluation.py").read_bytes()).hexdigest(),
        "expected_src_sha256": hashlib.sha256((REPO_ROOT / "eval" / "SRC.py").read_bytes()).hexdigest(),
        "expected_embed_model": EMBED_MODEL,
        "expected_embed_revision": EMBED_REVISION,
    }


def _resume_result(path: Path, expected_provenance: dict[str, Any], arm: str, package_hash: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        result = _load_json_object(path)
    except FrozenPackageError:
        return None
    provenance = result.get("trigger_provenance")
    if not isinstance(provenance, dict):
        return None
    for key, value in expected_provenance.items():
        if provenance.get(key) != value:
            return None
    if not result_is_complete(result):
        return None
    rag_payload = result.get("rag_package")
    if arm == "T1":
        if not isinstance(rag_payload, dict) or content_sha256(rag_payload) != package_hash:
            return None
    elif arm == "T0" and rag_payload is not None:
        return None
    return result


def run_actual_arm(
    *,
    paper_meta: dict[str, Any],
    paper_text: str,
    arm: str,
    repeat_id: int,
    execution_order: int,
    provider: str,
    model: str,
    api_key: str,
    envelope: dict[str, Any],
    output_dir: Path,
    config: RAGConfig,
    prompt_sha256: str,
    code_sha256: str,
    git_revision: dict[str, Any],
) -> tuple[dict[str, Any], Path, bool]:
    if arm not in {"T0", "T1"}:
        raise ValueError(f"Only T0/T1 are actual model-call arms, got {arm}")
    provenance = _provenance(
        paper_meta=paper_meta,
        paper_text=paper_text,
        arm=arm,
        repeat_id=repeat_id,
        execution_order=execution_order,
        provider=provider,
        model=model,
        envelope=envelope,
        prompt_sha256=prompt_sha256,
        code_sha256=code_sha256,
        git_revision=git_revision,
    )
    path = _result_path(output_dir, paper_meta["paper_id"], repeat_id, arm)
    resumed = _resume_result(path, provenance, arm, envelope["package_sha256"])
    if resumed is not None:
        return resumed, path, True
    if path.exists():
        # The Agent layer already performs its fixed JSON retry schedule.  Once
        # an arm output is written, never draw a replacement sample based on its
        # content; doing so would selectively resample unfavorable/invalid runs.
        raise RuntimeError(
            f"{paper_meta['paper_id']} {arm}: an existing result is invalid or has mismatched "
            f"provenance at {path}. It is sealed and will not be resampled."
        )

    # Reserve before the first paid call. After an interrupted process (including
    # SIGKILL), its partially sampled arm must not silently be drawn again.
    path.parent.mkdir(parents=True, exist_ok=True)
    reservation = {"trigger_provenance": provenance, "workflow_status": "in_progress",
                   "interruption_policy": "seal_started_arm_without_resampling"}
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(reservation, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RuntimeError(f"{path}: another process reserved this arm; no model calls made") from exc

    enable_rag = arm == "T1"
    try:
        result = mas_main(
            paper=paper_text,
            topic=normalize_topic(paper_meta.get("topic")),
            n_iter=N_ITER,
            reviewer_types=list(REVIEWER_TYPES),
            api_key=api_key,
            provider=provider,
            model=model,
            run_citation_check=RUN_CITATION_CHECK,
            enable_ai_detector=ENABLE_AI_DETECTOR,
            enable_rag=enable_rag,
            precomputed_rag_package=envelope["package"] if enable_rag else None,
            rag_config=asdict(config),
        )
        if not isinstance(result, dict):
            raise TypeError("mas_loop returned a non-object result")
    except BaseException as exc:
        # Persist failures and keyboard interruption, then preserve cancellation
        # semantics. A new experiment requires a declared new protocol/run.
        _atomic_write_json(path, {**reservation, "workflow_status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                  "terminal_error_type": type(exc).__name__})
        raise
    result["trigger_provenance"] = provenance
    _atomic_write_json(path, result)
    if not result_is_complete(result):
        raise RuntimeError(
            f"{paper_meta['paper_id']} {arm}: incomplete/invalid result sealed at {path}. "
            "Inspect turn_failures. This screening cannot yield GO; do not delete the file or selectively rerun the arm."
        )
    if arm == "T1":
        returned_package = result.get("rag_package")
        if not isinstance(returned_package, dict) or content_sha256(returned_package) != envelope["package_sha256"]:
            raise RuntimeError(f"{paper_meta['paper_id']} {arm}: returned RAG package differs from frozen package")
    return result, path, False


def derive_t2(
    *,
    paper_meta: dict[str, Any],
    paper_text: str,
    repeat_id: int,
    envelope: dict[str, Any],
    source_results: dict[str, dict[str, Any]],
    source_paths: dict[str, Path],
    output_dir: Path,
    provider: str,
    model: str,
    prompt_sha256: str,
    code_sha256: str,
    git_revision: dict[str, Any],
) -> tuple[dict[str, Any], Path, str]:
    gate_passed = envelope["gate"].get("passed") is True
    source_arm = "T1" if gate_passed else "T0"
    selected = source_results[source_arm]
    derived = copy.deepcopy(selected)
    provenance = _provenance(
        paper_meta=paper_meta,
        paper_text=paper_text,
        arm="T2",
        repeat_id=repeat_id,
        execution_order=None,
        provider=provider,
        model=model,
        envelope=envelope,
        prompt_sha256=prompt_sha256,
        code_sha256=code_sha256,
        git_revision=git_revision,
    )
    provenance.update(
        {
            "selected_source_arm": source_arm,
            "selected_source_result_file": source_paths[source_arm].name,
            "selected_source_result_sha256": content_sha256(selected),
            "reviewer_input_sha256": selected["trigger_provenance"]["reviewer_input_sha256"],
        }
    )
    derived["trigger_provenance"] = provenance
    path = _result_path(output_dir, paper_meta["paper_id"], repeat_id, "T2")
    _atomic_write_json(path, derived)
    return derived, path, source_arm


def audit_packages(
    papers: list[dict[str, Any]],
    *,
    md_dir: Path,
    package_dir: Path,
    provider: str,
    model: str,
    config: RAGConfig,
    output_path: Path,
) -> dict[str, Any]:
    entries = []
    reason_counts: Counter[str] = Counter()
    invalid = []
    pass_count = 0
    for paper_meta in papers:
        paper_id = paper_meta["paper_id"]
        try:
            paper_text = load_markdown(paper_meta, md_dir)
            envelope = load_required_package(
                paper_meta,
                paper_text,
                package_dir=package_dir,
                provider=provider,
                model=model,
                config=config,
            )
            gate = envelope["gate"]
            pass_count += int(gate["passed"] is True)
            reason_counts.update(gate["reason_codes"])
            entries.append(
                {
                    "paper_id": paper_id,
                    "package_file": package_path(package_dir, paper_id).name,
                    "package_sha256": envelope["package_sha256"],
                    "gate": gate,
                }
            )
        except (FileNotFoundError, FrozenPackageError) as exc:
            invalid.append({"paper_id": paper_id, "error": str(exc)})

    report = {
        "schema_version": 1,
        "gate_version": GATE_VERSION,
        "invalid_result_policy": INVALID_RESULT_POLICY,
        "provider": provider,
        "model": model,
        "rag_config": asdict(config),
        "paper_count": len(papers),
        "valid_package_count": len(entries),
        "invalid_package_count": len(invalid),
        "gate_pass_count": pass_count,
        "gate_fail_count": len(entries) - pass_count,
        "gate_coverage": pass_count / len(entries) if entries else None,
        "reason_counts": dict(sorted(reason_counts.items())),
        "invalid_packages": invalid,
        "papers": entries,
    }
    _atomic_write_json(output_path, report)
    return report


def run_experiment(
    papers: list[dict[str, Any]],
    *,
    api_key: str,
    output_dir: Path,
    md_dir: Path,
    package_dir: Path,
    provider: str,
    model: str,
    repeat_id: int,
    condition_order_seed: int,
    config: RAGConfig,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_sha256 = _prompt_bundle_sha256()
    code_sha256 = _code_bundle_sha256()
    git_revision = _git_revision()

    # Fail before the first paid review call if any frozen artifact is absent or
    # mismatched.  This prevents a partial experiment with silently rebuilt RAG.
    prepared: dict[str, tuple[str, dict[str, Any]]] = {}
    for paper_meta in papers:
        paper_text = load_markdown(paper_meta, md_dir)
        envelope = load_required_package(
            paper_meta,
            paper_text,
            package_dir=package_dir,
            provider=provider,
            model=model,
            config=config,
        )
        prepared[paper_meta["paper_id"]] = (paper_text, envelope)

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "trigger_rag_screening",
        "created_at": datetime.now().astimezone().isoformat(),
        "repeat_id": repeat_id,
        "condition_order_seed": condition_order_seed,
        "selected_paper_ids_sha256": content_sha256(sorted(paper["paper_id"] for paper in papers)),
        "provider": provider,
        "model": model,
        "rag_config": asdict(config),
        "gate_version": GATE_VERSION,
        "invalid_result_policy": INVALID_RESULT_POLICY,
        "prompt_bundle_sha256": prompt_sha256,
        "code_bundle_sha256": code_sha256,
        **_frozen_evaluation_protocol(),
        "git": git_revision,
        "conditions": copy.deepcopy(CONDITIONS),
        "papers": [],
    }
    summary_path = output_dir / f"experiment_trigger_summary_rep-{repeat_id}.json"
    order_by_paper = balanced_condition_orders(
        [paper["paper_id"] for paper in papers],
        seed=condition_order_seed,
        repeat_id=repeat_id,
    )

    for paper_index, paper_meta in enumerate(papers, 1):
        paper_id = paper_meta["paper_id"]
        paper_text, envelope = prepared[paper_id]
        arms = order_by_paper[paper_id]
        print(f"\n[{paper_index}/{len(papers)}] {paper_id}: order={','.join(arms)} gate={'PASS' if envelope['gate']['passed'] else 'FAIL'}")

        source_results: dict[str, dict[str, Any]] = {}
        source_paths: dict[str, Path] = {}
        condition_entries: dict[str, Any] = {}
        for order, arm in enumerate(arms, 1):
            result, path, reused = run_actual_arm(
                paper_meta=paper_meta,
                paper_text=paper_text,
                arm=arm,
                repeat_id=repeat_id,
                execution_order=order,
                provider=provider,
                model=model,
                api_key=api_key,
                envelope=envelope,
                output_dir=output_dir,
                config=config,
                prompt_sha256=prompt_sha256,
                code_sha256=code_sha256,
                git_revision=git_revision,
            )
            source_results[arm] = result
            source_paths[arm] = path
            condition_entries[arm] = {
                "desc": CONDITIONS[arm]["desc"],
                "result_file": path.name,
                "reused_existing": reused,
                "result": result,
            }

        t2_result, t2_path, selected_source_arm = derive_t2(
            paper_meta=paper_meta,
            paper_text=paper_text,
            repeat_id=repeat_id,
            envelope=envelope,
            source_results=source_results,
            source_paths=source_paths,
            output_dir=output_dir,
            provider=provider,
            model=model,
            prompt_sha256=prompt_sha256,
            code_sha256=code_sha256,
            git_revision=git_revision,
        )
        condition_entries["T2"] = {
            "desc": CONDITIONS["T2"]["desc"],
            "result_file": t2_path.name,
            "reused_existing": False,
            "derived_policy": True,
            "selected_source_arm": selected_source_arm,
            "result": t2_result,
        }
        summary["papers"].append(
            {
                "paper_id": paper_id,
                "conference": paper_meta.get("conference", ""),
                "topic": normalize_topic(paper_meta.get("topic")),
                "paper_sha256": envelope["paper_sha256"],
                "package_file": package_path(package_dir, paper_id).name,
                "package_sha256": envelope["package_sha256"],
                "gate": envelope["gate"],
                "execution_order": arms,
                "conditions": condition_entries,
            }
        )
        _atomic_write_json(summary_path, summary)

    print(f"\nTrigger experiment summary: {summary_path}")
    return summary


def _api_key(args: argparse.Namespace, required: bool) -> str:
    if args.api_key:
        return args.api_key
    env_name = "OPENROUTER_API_KEY" if args.provider == "openrouter" else "DEEPSEEK_API_KEY"
    value = os.environ.get(env_name, "")
    if required and not value:
        raise SystemExit(f"API key required: pass --api_key or set {env_name}.")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json_file", type=Path, default=Path("eval/trigger_validation_36.blinded.json"))
    parser.add_argument("--md_dir", type=Path, default=Path("data/marker_md"))
    parser.add_argument("--rag_package_dir", type=Path, default=Path("experiment_artifacts/trigger/rag_packages/integrity_v1"))
    parser.add_argument("--output_dir", type=Path, default=Path("experiment_artifacts/trigger/runs/rep1"))
    parser.add_argument("--provider", choices=sorted(DEFAULT_MODELS), default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--paper_id", default=None, help="Optional opaque paper ID for a smoke test.")
    parser.add_argument(
        "--allow-sensitive-manifest",
        action="store_true",
        help="Allow labels/reviews in a development-only smoke manifest; never use for formal screening.",
    )
    parser.add_argument("--repeat_id", type=int, default=1)
    parser.add_argument("--condition_order_seed", type=int, default=11766)
    parser.add_argument("--rag_cutoff_date", default="2024-12-31")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-rag-only", action="store_true")
    modes.add_argument("--audit-rag-only", action="store_true")
    parser.add_argument(
        "--overwrite-rag-package",
        action="store_true",
        help="Explicitly rebuild frozen packages. Never use this after screening starts.",
    )
    parser.add_argument("--audit_output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeat_id < 1:
        raise SystemExit("--repeat_id must be at least 1")
    model = args.model or DEFAULT_MODELS[args.provider]
    papers = load_papers(args.json_file, allow_sensitive=args.allow_sensitive_manifest)
    if args.paper_id:
        papers = [paper for paper in papers if paper["paper_id"] == args.paper_id]
        if not papers:
            raise SystemExit(f"paper_id {args.paper_id!r} was not found in {args.json_file}")
    if not papers:
        raise SystemExit("No papers selected.")
    try:
        config = rag_config(args.rag_cutoff_date)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.prepare_rag_only:
        api_key = _api_key(args, required=True)
        reused_count = 0
        pass_count = 0
        for index, paper_meta in enumerate(papers, 1):
            print(f"[{index}/{len(papers)}] preparing {paper_meta['paper_id']}")
            paper_text = load_markdown(paper_meta, args.md_dir)
            envelope, reused = prepare_or_load_package(
                paper_meta,
                paper_text,
                package_dir=args.rag_package_dir,
                provider=args.provider,
                model=model,
                api_key=api_key,
                config=config,
                overwrite=args.overwrite_rag_package,
            )
            reused_count += int(reused)
            pass_count += int(envelope["gate"]["passed"] is True)
            print(
                f"  {'reused' if reused else 'frozen'}; gate="
                f"{'PASS' if envelope['gate']['passed'] else 'FAIL'} "
                f"reasons={envelope['gate']['reason_codes']}"
            )
        print(f"Prepared {len(papers)} package(s); reused={reused_count}; gate_pass={pass_count}.")
        return

    if args.audit_rag_only:
        audit_output = args.audit_output or args.rag_package_dir / f"audit_{GATE_VERSION}.json"
        report = audit_packages(
            papers,
            md_dir=args.md_dir,
            package_dir=args.rag_package_dir,
            provider=args.provider,
            model=model,
            config=config,
            output_path=audit_output,
        )
        print(
            f"Frozen packages: valid={report['valid_package_count']}/{report['paper_count']}; "
            f"gate PASS={report['gate_pass_count']}, FAIL={report['gate_fail_count']}, "
            f"coverage={report['gate_coverage']}"
        )
        print(f"Reason counts: {report['reason_counts']}")
        print(f"Audit report: {audit_output}")
        if report["invalid_package_count"]:
            raise SystemExit(1)
        return

    api_key = _api_key(args, required=True)
    run_experiment(
        papers,
        api_key=api_key,
        output_dir=args.output_dir,
        md_dir=args.md_dir,
        package_dir=args.rag_package_dir,
        provider=args.provider,
        model=model,
        repeat_id=args.repeat_id,
        condition_order_seed=args.condition_order_seed,
        config=config,
    )


if __name__ == "__main__":
    main()
