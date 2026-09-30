"""Repair incomplete C1-C5 advanced-experiment results.

The script reads an existing experiment summary, checks that every paper has
the configured number of valid structured reviewer returns for every
condition, and reruns only incomplete paper-condition pairs. A whole condition
is rerun so multi-round reviewer/author context remains internally coherent.

Complete source files and the original summary are never overwritten. Each
successful replacement is saved as a new raw result, followed by a new
repaired summary.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.experiment_advanced import (  # noqa: E402
    CONDITIONS,
    load_markdown,
    load_papers,
    normalize_topic,
    run_condition,
    save_result,
)
from review_schema import review_is_valid, workflow_is_complete


def is_valid_review(review: Any) -> bool:
    """Return whether a reviewer output is usable by the evaluation."""
    return review_is_valid(review)


def valid_reviews(result: Any) -> list[dict]:
    if not isinstance(result, dict):
        return []
    reviews = result.get("reviewers", [])
    if not isinstance(reviews, list):
        return []
    return [review for review in reviews if is_valid_review(review)]


def expected_review_count(condition_config: dict) -> int:
    agents = condition_config.get("agents", [])
    if not isinstance(agents, list) or not agents:
        raise ValueError("condition configuration must contain a non-empty agents list")
    return len(agents)


def result_is_complete(result: Any, condition_config: dict) -> bool:
    expected = expected_review_count(condition_config)
    if (not isinstance(result, dict) or not isinstance(result.get("reviewers"), list)
            or len(result["reviewers"]) != expected or len(valid_reviews(result)) != expected
            or result.get("turn_failures")
            or result.get("workflow_status") in {"failed", "interrupted", "in_progress", "invalid"}):
        return False
    if "workflow_schema_version" in result:
        return workflow_is_complete(result, expected, condition_config.get("n_iter", 1))
    return True  # Legacy outputs cannot prove intermediate-turn integrity.


def audit_summary(summary: dict) -> list[dict]:
    """Describe every missing or invalid paper-condition result."""
    configs = summary.get("conditions", {})
    issues = []
    for paper in summary.get("papers", []):
        for condition_id, config in configs.items():
            entry = paper.get("conditions", {}).get(condition_id, {})
            result = entry.get("result")
            expected = expected_review_count(config)
            collected = len(valid_reviews(result))
            if not result_is_complete(result, config):
                stored = len(result.get("reviewers", [])) if isinstance(result, dict) and isinstance(
                    result.get("reviewers"), list
                ) else 0
                issues.append(
                    {
                        "paper_id": paper.get("paper_id"),
                        "condition_id": condition_id,
                        "expected_reviews": expected,
                        "valid_reviews": collected,
                        "stored_reviews": stored,
                    }
                )
    return issues


def _runtime_condition(condition_id: str, config: dict) -> dict:
    templates = {condition["id"]: condition for condition in CONDITIONS}
    if condition_id not in templates:
        raise ValueError(
            f"Unsupported condition {condition_id!r}; expected one of {sorted(templates)}"
        )
    condition = copy.deepcopy(templates[condition_id])
    for key in ("agents", "n_iter", "agenttype", "enable_rag", "desc"):
        if key in config:
            condition[key] = copy.deepcopy(config[key])
    return condition


def _shared_rag_package(paper_entry: dict) -> dict | None:
    for condition_entry in paper_entry.get("conditions", {}).values():
        result = condition_entry.get("result", {})
        package = result.get("rag_package") if isinstance(result, dict) else None
        if isinstance(package, dict):
            return package
    return None


def repair_summary(
    summary: dict,
    papers: list[dict],
    *,
    api_key: str,
    output_dir: str,
    md_dir: str,
    max_attempts: int = 3,
    paper_ids: set[str] | None = None,
    condition_ids: set[str] | None = None,
) -> tuple[dict, list[dict]]:
    """Rerun incomplete pairs and return the new summary and unresolved issues."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if summary.get("experiment") == "trigger_rag_screening":
        raise ValueError("Sealed trigger runs must not be repaired or resampled")

    repaired = copy.deepcopy(summary)
    provider = str(summary.get("provider", "")).strip()
    model = str(summary.get("model", "")).strip()
    if not provider or not model:
        raise ValueError("summary must contain provider and model")

    manifest = {paper["paper_id"]: paper for paper in papers}
    repair_timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    replacements = []

    for paper_entry in repaired.get("papers", []):
        paper_id = paper_entry.get("paper_id")
        if paper_ids and paper_id not in paper_ids:
            continue
        if paper_id not in manifest:
            raise KeyError(f"Paper {paper_id!r} is missing from the paper manifest")

        paper_meta = manifest[paper_id]
        paper_text = None
        rag_package = _shared_rag_package(paper_entry)

        for condition_id, config in repaired.get("conditions", {}).items():
            if condition_ids and condition_id not in condition_ids:
                continue
            old_entry = paper_entry.get("conditions", {}).get(condition_id, {})
            if result_is_complete(old_entry.get("result"), config):
                continue

            condition = _runtime_condition(condition_id, config)
            if paper_text is None:
                paper_text = load_markdown(paper_meta["paper_dir"], md_dir)

            expected = expected_review_count(config)
            print(
                f"Repairing {paper_id}/{condition_id}: expected {expected} valid review(s), "
                f"found {len(valid_reviews(old_entry.get('result')))}."
            )

            replacement = None
            for attempt in range(1, max_attempts + 1):
                print(f"  condition attempt {attempt}/{max_attempts}")
                candidate = run_condition(
                    paper_text=paper_text,
                    topic=normalize_topic(paper_meta.get("topic", "")),
                    cond=condition,
                    api_key=api_key,
                    provider=provider,
                    model=model,
                    rag_package=rag_package if condition["enable_rag"] else None,
                )
                if result_is_complete(candidate, config):
                    replacement = candidate
                    break
                print(
                    f"  incomplete return: collected {len(valid_reviews(candidate))}/"
                    f"{expected} valid reviews"
                )

            if replacement is None:
                print(f"  unresolved after {max_attempts} condition attempt(s)")
                continue

            paper_name = Path(paper_meta["paper_dir"]).stem
            result_path = save_result(
                result=replacement,
                paper_name=paper_name,
                cond=condition,
                output_dir=output_dir,
                timestamp=repair_timestamp,
                provider=provider,
                model=model,
            )
            paper_entry.setdefault("conditions", {})[condition_id] = {
                "desc": config.get("desc", condition.get("desc", "")),
                "result_file": Path(result_path).name,
                "reused_existing": False,
                "repaired": True,
                "replaces_result_file": old_entry.get("result_file"),
                "result": replacement,
            }
            replacements.append(
                {
                    "paper_id": paper_id,
                    "condition_id": condition_id,
                    "result_file": Path(result_path).name,
                }
            )
            if condition["enable_rag"] and isinstance(
                replacement.get("rag_package"), dict
            ):
                rag_package = replacement["rag_package"]
            print(f"  repaired: {result_path}")

    unresolved = audit_summary(repaired)
    repaired["timestamp"] = repair_timestamp
    repaired["repair"] = {
        "source_timestamp": summary.get("timestamp"),
        "replacement_count": len(replacements),
        "replacements": replacements,
        "unresolved_count": len(unresolved),
        "unresolved": unresolved,
    }
    return repaired, unresolved


def _api_key(provider: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    env_names = {
        "openrouter": ("OPENROUTER_API_KEY", "OPENAI_API_KEY"),
        "deepseek": ("DEEPSEEK_API_KEY",),
    }.get(provider, ())
    for env_name in env_names:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    raise ValueError(
        f"No API key provided. Use --api_key or set one of: {', '.join(env_names) or 'provider API key'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun incomplete advanced-experiment paper/condition pairs so every "
            "paper has the configured number of valid reviewer results."
        )
    )
    parser.add_argument("--summary", required=True, help="Existing advanced summary JSON")
    parser.add_argument("--json_file", default="eval/papers.json")
    parser.add_argument("--md_dir", default="data/md")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Defaults to the source summary's directory",
    )
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--paper_id", action="append", default=None)
    parser.add_argument("--condition", action="append", default=None)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only list incomplete pairs; do not call the API or write files",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    issues = audit_summary(summary)
    print(f"Found {len(issues)} incomplete paper-condition pair(s).")
    for issue in issues:
        print(
            f"  {issue['paper_id']}/{issue['condition_id']}: "
            f"{issue['valid_reviews']}/{issue['expected_reviews']} valid "
            f"({issue['stored_reviews']} stored)"
        )

    if args.dry_run or not issues:
        return

    output_dir = args.output_dir or str(summary_path.parent)
    try:
        api_key = _api_key(str(summary.get("provider", "")), args.api_key)
        repaired, unresolved = repair_summary(
            summary,
            load_papers(args.json_file),
            api_key=api_key,
            output_dir=output_dir,
            md_dir=args.md_dir,
            max_attempts=args.max_attempts,
            paper_ids=set(args.paper_id) if args.paper_id else None,
            condition_ids=set(args.condition) if args.condition else None,
        )
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))

    suffix = "repaired" if not unresolved else "repair_partial"
    repaired_path = Path(output_dir) / (
        f"experiment_advanced_summary_{repaired['timestamp']}_{suffix}.json"
    )
    repaired_path.write_text(
        json.dumps(repaired, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nRepair summary saved: {repaired_path}")
    if unresolved:
        print(f"Unresolved incomplete pairs: {len(unresolved)}")
        raise SystemExit(1)
    print("All papers now have the expected number of valid results in every condition.")


if __name__ == "__main__":
    main()
