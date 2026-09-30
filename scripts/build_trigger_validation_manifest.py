#!/usr/bin/env python3
"""Build a model-input-blinded, stratified trigger-RAG validation split.

The run manifest intentionally removes labels, scores, and human reviews.  A
separate evaluation manifest retains the selected records under the same opaque
IDs so ground truth can be joined only after model outputs are frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "eval/openreview_2025_264_qwen.balanced_by_conference_under10mb.json"
DEFAULT_EXCLUDE = "eval/papers.json"
DEFAULT_RUN_OUTPUT = "eval/trigger_validation_36.blinded.json"
DEFAULT_EVALUATION_OUTPUT = "eval/trigger_validation_36.evaluation.json"
DEFAULT_LABELS_OUTPUT = "eval/trigger_validation_36.labels.json"
DEFAULT_METADATA_OUTPUT = "eval/trigger_validation_36.selection.json"
DEFAULT_HISTORICAL_GIT_SPECS = (
    "85138df:eval/exp_results_60/experiment_summary_2607222040.json",
)
CONFERENCES = ("ICLR", "ICML", "NeurIPS")
LABELS = ("accept", "reject")
RUN_FIELD_ALLOWLIST = ("title", "paper_dir", "conference", "topic", "year")


def _load_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read JSON object {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a JSON object keyed by paper ID.")
    return data


def _load_git_object(spec: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["git", "show", spec],
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not load historical experiment JSON from git spec {spec!r}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Historical git spec {spec!r} must contain a JSON object.")
    return data


def merge_historical_exclusions(
    source: dict[str, Any],
    development: dict[str, Any],
    historical_summaries: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Merge historical experiment paper IDs into the title exclusion set."""
    combined = dict(development)
    historical_ids = set()
    for summary in historical_summaries:
        papers = summary.get("papers")
        if not isinstance(papers, list):
            raise ValueError("historical experiment summary is missing a papers list")
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            paper_id = str(paper.get("paper_id") or paper.get("paper_name") or "").strip()
            if paper_id:
                historical_ids.add(paper_id)

    present = []
    missing = []
    for paper_id in sorted(historical_ids):
        record = source.get(paper_id)
        if not isinstance(record, dict) or not normalize_title(record.get("title")):
            missing.append(paper_id)
            continue
        present.append(paper_id)
        combined[f"historical::{paper_id}"] = {"title": record["title"]}
    return combined, present, missing


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def normalize_title(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    folded = value.casefold()
    for apostrophe in ("'", "’", "‘", "ʼ", "`"):
        folded = folded.replace(apostrophe, "")
    decomposed = unicodedata.normalize("NFKD", folded)
    separated = "".join(
        " " if unicodedata.category(character)[:1] in {"P", "S"} else character
        for character in decomposed
    )
    ascii_text = separated.encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text))


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if text.startswith("accept"):
        return "accept"
    if text.startswith("reject"):
        return "reject"
    return ""


def _segment_has_content(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return any(str(item or "").strip() for item in value)


def hard_invalid_reasons(record: Any) -> list[str]:
    if not isinstance(record, dict):
        return ["record_not_object"]
    reasons = []
    if not normalize_title(record.get("title")):
        reasons.append("missing_title")
    if not str(record.get("paper_dir") or "").strip():
        reasons.append("missing_paper_dir")
    if str(record.get("conference") or "").strip() not in CONFERENCES:
        reasons.append("invalid_conference")
    if normalize_label(record.get("accept_or_not")) not in LABELS:
        reasons.append("invalid_label")
    reviews = record.get("reviews")
    if not isinstance(reviews, list) or not reviews:
        reasons.append("missing_reviews")
    else:
        for index, review in enumerate(reviews):
            if not isinstance(review, dict):
                reasons.append(f"review_{index}_not_object")
                continue
            strengths_empty = not _segment_has_content(review.get("strengths"))
            weaknesses_empty = not _segment_has_content(review.get("weaknesses"))
            if strengths_empty and weaknesses_empty:
                reasons.append(f"review_{index}_both_segments_empty")
    return reasons


def _run_record(record: dict[str, Any]) -> dict[str, Any]:
    return {field: record.get(field) for field in RUN_FIELD_ALLOWLIST if field in record}


def _assert_outputs_available(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(existing)
        raise SystemExit(f"Refusing to overwrite frozen output(s): {joined}. Pass --overwrite explicitly.")


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_manifests(
    source: dict[str, Any],
    development: dict[str, Any],
    *,
    per_stratum: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if per_stratum < 1:
        raise ValueError("per_stratum must be at least 1")

    development_titles = {
        title
        for record in development.values()
        if isinstance(record, dict)
        for title in [normalize_title(record.get("title"))]
        if title
    }
    invalid: dict[str, list[str]] = {}
    overlap_ids: list[str] = []
    normalized_to_ids: defaultdict[str, list[str]] = defaultdict(list)

    for paper_id in sorted(source):
        record = source[paper_id]
        reasons = hard_invalid_reasons(record)
        if reasons:
            invalid[paper_id] = reasons
            continue
        title = normalize_title(record.get("title"))
        if title in development_titles:
            overlap_ids.append(paper_id)
            continue
        normalized_to_ids[title].append(paper_id)

    duplicate_titles = {
        title: ids for title, ids in normalized_to_ids.items() if len(ids) > 1
    }
    duplicate_ids = {paper_id for ids in duplicate_titles.values() for paper_id in ids}

    strata: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for _, ids in sorted(normalized_to_ids.items()):
        for paper_id in ids:
            if paper_id in duplicate_ids:
                continue
            record = source[paper_id]
            strata[(record["conference"], normalize_label(record["accept_or_not"]))].append(paper_id)

    selected_source_ids: list[str] = []
    eligible_counts = {}
    for conference in CONFERENCES:
        for label in LABELS:
            key = (conference, label)
            candidates = sorted(strata.get(key, []))
            eligible_counts[f"{conference}:{label}"] = len(candidates)
            if len(candidates) < per_stratum:
                raise ValueError(
                    f"Not enough eligible records in {conference}/{label}: "
                    f"need {per_stratum}, found {len(candidates)}"
                )
            rng = random.Random(f"{seed}:{conference}:{label}")
            rng.shuffle(candidates)
            selected_source_ids.extend(candidates[:per_stratum])

    final_rng = random.Random(seed)
    final_rng.shuffle(selected_source_ids)

    run_manifest: dict[str, Any] = {}
    evaluation_manifest: dict[str, Any] = {}
    labels_manifest: dict[str, Any] = {}
    mapping: dict[str, Any] = {}
    for index, source_id in enumerate(selected_source_ids, 1):
        opaque_id = f"trigger_val_{index:03d}"
        record = source[source_id]
        run_manifest[opaque_id] = _run_record(record)
        evaluation_record = json.loads(json.dumps(record, ensure_ascii=False))
        evaluation_record["source_paper_id"] = source_id
        evaluation_manifest[opaque_id] = evaluation_record
        labels_manifest[opaque_id] = {
            "accept_or_not": normalize_label(record.get("accept_or_not")),
            "score": record.get("score"),
            "conference": record.get("conference"),
        }
        mapping[opaque_id] = {
            "source_paper_id": source_id,
            "conference": record.get("conference"),
            "accept_or_not": normalize_label(record.get("accept_or_not")),
            "normalized_title": normalize_title(record.get("title")),
        }

    selected_counts = Counter(
        (item["conference"], item["accept_or_not"]) for item in mapping.values()
    )
    metadata = {
        "schema_version": 1,
        "seed": seed,
        "per_stratum": per_stratum,
        "selection_count": len(run_manifest),
        "blinding_scope": (
            "Model-input-blinded: labels, scores, and reviews are absent from the run manifest. "
            "paper_dir remains label-revealing to an operator but is used only by the local Markdown loader."
        ),
        "source_count": len(source),
        "development_count": len(development),
        "development_normalized_title_count": len(development_titles),
        "hard_invalid_count": len(invalid),
        "hard_invalid": invalid,
        "development_overlap_count": len(overlap_ids),
        "development_overlap_ids": overlap_ids,
        "duplicate_title_group_count": len(duplicate_titles),
        "duplicate_titles": duplicate_titles,
        "eligible_counts": eligible_counts,
        "selected_counts": {
            f"{conference}:{label}": selected_counts[(conference, label)]
            for conference in CONFERENCES
            for label in LABELS
        },
        "remaining_eligible_after_selection": sum(eligible_counts.values()) - len(run_manifest),
        "opaque_mapping": mapping,
        "run_manifest_sha256": content_sha256(run_manifest),
        "evaluation_manifest_sha256": content_sha256(evaluation_manifest),
        "labels_manifest_sha256": content_sha256(labels_manifest),
    }
    return run_manifest, evaluation_manifest, labels_manifest, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--exclude", type=Path, default=Path(DEFAULT_EXCLUDE))
    parser.add_argument("--run-output", type=Path, default=Path(DEFAULT_RUN_OUTPUT))
    parser.add_argument("--evaluation-output", type=Path, default=Path(DEFAULT_EVALUATION_OUTPUT))
    parser.add_argument("--labels-output", type=Path, default=Path(DEFAULT_LABELS_OUTPUT))
    parser.add_argument("--metadata-output", type=Path, default=Path(DEFAULT_METADATA_OUTPUT))
    parser.add_argument(
        "--historical-git-spec",
        action="append",
        default=None,
        help="Repeatable git object spec for an earlier experiment summary whose papers must be excluded.",
    )
    parser.add_argument(
        "--no-default-historical-exclusions",
        action="store_true",
        help="Disable the repository's default historical-60 exclusion (development/debugging only).",
    )
    parser.add_argument("--per-stratum", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11766)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = [args.run_output, args.evaluation_output, args.labels_output, args.metadata_output]
    _assert_outputs_available(outputs, args.overwrite)
    source = _load_object(args.input)
    development_source = _load_object(args.exclude)
    historical_specs = [] if args.no_default_historical_exclusions else list(DEFAULT_HISTORICAL_GIT_SPECS)
    historical_specs.extend(args.historical_git_spec or [])
    # Preserve order while avoiding redundant reads when a caller explicitly
    # repeats one of the repository defaults.
    historical_specs = list(dict.fromkeys(historical_specs))
    historical_summaries = [_load_git_object(spec) for spec in historical_specs]
    try:
        development, historical_present, historical_missing = merge_historical_exclusions(
            source,
            development_source,
            historical_summaries,
        )
        run_manifest, evaluation_manifest, labels_manifest, metadata = build_manifests(
            source,
            development,
            per_stratum=args.per_stratum,
            seed=args.seed,
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Could not build trigger validation split: {exc}") from exc

    metadata["source_path"] = str(args.input)
    metadata["source_sha256"] = content_sha256(source)
    metadata["development_path"] = str(args.exclude)
    metadata["development_sha256"] = content_sha256(development_source)
    metadata["historical_git_specs"] = historical_specs
    metadata["historical_present_count"] = len(historical_present)
    metadata["historical_present_ids"] = historical_present
    metadata["historical_missing_count"] = len(historical_missing)
    metadata["historical_missing_ids"] = historical_missing
    metadata["run_output"] = str(args.run_output)
    metadata["evaluation_output"] = str(args.evaluation_output)
    metadata["labels_output"] = str(args.labels_output)

    _write_json_atomic(args.run_output, run_manifest)
    _write_json_atomic(args.evaluation_output, evaluation_manifest)
    _write_json_atomic(args.labels_output, labels_manifest)
    _write_json_atomic(args.metadata_output, metadata)

    print(f"Selected {len(run_manifest)} papers ({args.per_stratum} per conference/label cell).")
    print(f"Hard-invalid records excluded: {metadata['hard_invalid_count']}")
    print(f"Prior-experiment title overlaps excluded: {metadata['development_overlap_count']}")
    print(
        f"Historical-60 IDs matched to this source: {metadata['historical_present_count']} "
        f"(absent from source: {metadata['historical_missing_count']})"
    )
    print(f"Eligible papers remaining after this validation split: {metadata['remaining_eligible_after_selection']}")
    print(f"Run manifest: {args.run_output}")
    print(f"Evaluation manifest: {args.evaluation_output}")
    print(f"Private labels: {args.labels_output}")
    print(f"Selection metadata: {args.metadata_output}")


if __name__ == "__main__":
    main()
