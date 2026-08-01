"""Audit experiment summaries for invalid structured reviewer returns.

This script is read-only. It detects the old ``{"raw": ..., "parse_error":
true}`` records, JSON objects that do not satisfy the reviewer schema, and
paper-condition pairs with fewer valid reviews than configured.

Examples:
    python scripts/audit_experiment_invalid_json.py \
        eval/advanced_exp_results/experiment_advanced_summary_2607290028.json

    python scripts/audit_experiment_invalid_json.py --recursive eval
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_REVIEW_FIELDS = {
    "reviewer",
    "decision",
    "scores",
    "strengths",
    "weaknesses",
}


def classify_review(review: Any) -> str:
    """Classify a stored reviewer return for evaluation readiness."""
    if not isinstance(review, dict):
        return "not_json_object"
    if review.get("parse_error"):
        return "parse_error"
    if not REQUIRED_REVIEW_FIELDS.issubset(review):
        return "invalid_schema"
    if str(review.get("decision", "")).strip().lower() not in {"accept", "reject"}:
        return "invalid_schema"
    scores = review.get("scores")
    if not isinstance(scores, dict) or not scores:
        return "invalid_schema"
    if not all(isinstance(value, (int, float)) for value in scores.values()):
        return "invalid_schema"
    if not isinstance(review.get("strengths"), list):
        return "invalid_schema"
    if not isinstance(review.get("weaknesses"), list):
        return "invalid_schema"
    return "valid"


def _expected_count(config: Any) -> int | None:
    if not isinstance(config, dict):
        return None
    agents = config.get("agents")
    return len(agents) if isinstance(agents, list) and agents else None


def audit_summary_data(data: dict, source: str = "") -> dict:
    """Return a machine-readable audit for one experiment summary."""
    conditions = data.get("conditions", {})
    papers = data.get("papers", [])
    audit_format = "multi_condition"
    if not isinstance(conditions, dict) or not conditions:
        legacy_condition = data.get("condition")
        if isinstance(legacy_condition, dict):
            audit_format = "single_condition"
            condition_id = str(
                legacy_condition.get("id")
                or legacy_condition.get("label")
                or legacy_condition.get("agenttype")
                or "condition"
            )
            expected = legacy_condition.get("nagent")
            if not isinstance(expected, int) or expected < 1:
                reviewers = legacy_condition.get("reviewers")
                expected = len(reviewers) if isinstance(reviewers, list) and reviewers else 1
            conditions = {condition_id: {"agents": [None] * expected}}
            papers = [
                {
                    **paper,
                    "conditions": {
                        condition_id: {
                            "result": paper.get("result"),
                            "result_file": paper.get("result_file"),
                        }
                    },
                }
                for paper in papers
                if isinstance(paper, dict)
            ]
    issues = []
    totals: Counter[str] = Counter()
    by_condition: dict[str, Counter[str]] = {
        condition_id: Counter() for condition_id in conditions
    }
    total_pairs = 0

    if not isinstance(conditions, dict):
        conditions = {}
    if not isinstance(papers, list):
        papers = []

    for paper in papers:
        if not isinstance(paper, dict):
            continue
        paper_id = paper.get("paper_id", "<unknown>")
        paper_conditions = paper.get("conditions", {})
        if not isinstance(paper_conditions, dict):
            paper_conditions = {}

        condition_ids = conditions.keys() or paper_conditions.keys()
        for condition_id in condition_ids:
            total_pairs += 1
            config = conditions.get(condition_id, {})
            entry = paper_conditions.get(condition_id)
            expected = _expected_count(config)
            condition_totals = by_condition.setdefault(condition_id, Counter())
            condition_totals["paper_condition_pairs"] += 1
            condition_totals["expected_reviews"] += expected or 0
            totals["expected_reviews"] += expected or 0
            classifications: Counter[str] = Counter()

            if not isinstance(entry, dict):
                issue = {
                    "paper_id": paper_id,
                    "condition_id": condition_id,
                    "expected_reviews": expected,
                    "stored_reviews": 0,
                    "valid_reviews": 0,
                    "parse_errors": 0,
                    "schema_errors": 0,
                    "other_invalid": 0,
                    "missing_condition": True,
                }
                issues.append(issue)
                totals["missing_conditions"] += 1
                condition_totals["missing_conditions"] += 1
                condition_totals["affected_pairs"] += 1
                if expected:
                    totals["missing_reviews"] += expected
                    condition_totals["missing_reviews"] += expected
                continue

            result = entry.get("result")
            reviews = result.get("reviewers", []) if isinstance(result, dict) else []
            if not isinstance(reviews, list):
                reviews = []
            for review in reviews:
                classifications[classify_review(review)] += 1

            valid = classifications["valid"]
            parse_errors = classifications["parse_error"]
            schema_errors = classifications["invalid_schema"]
            other_invalid = classifications["not_json_object"]
            missing = max((expected or 0) - valid, 0)
            totals["stored_reviews"] += len(reviews)
            totals["valid_reviews"] += valid
            totals["parse_errors"] += parse_errors
            totals["schema_errors"] += schema_errors
            totals["other_invalid"] += other_invalid
            totals["missing_reviews"] += missing
            condition_totals["stored_reviews"] += len(reviews)
            condition_totals["valid_reviews"] += valid
            condition_totals["parse_errors"] += parse_errors
            condition_totals["schema_errors"] += schema_errors
            condition_totals["other_invalid"] += other_invalid
            condition_totals["missing_reviews"] += missing

            if parse_errors or schema_errors or other_invalid or missing:
                condition_totals["affected_pairs"] += 1
                issues.append(
                    {
                        "paper_id": paper_id,
                        "condition_id": condition_id,
                        "expected_reviews": expected,
                        "stored_reviews": len(reviews),
                        "valid_reviews": valid,
                        "parse_errors": parse_errors,
                        "schema_errors": schema_errors,
                        "other_invalid": other_invalid,
                        "missing_condition": False,
                    }
                )

    return {
        "source": source,
        "timestamp": data.get("timestamp"),
        "provider": data.get("provider"),
        "model": data.get("model"),
        "audit_format": audit_format,
        "paper_count": len(papers),
        "condition_count": len(conditions),
        "paper_condition_pairs": total_pairs,
        "expected_reviews": totals["expected_reviews"],
        "stored_reviews": totals["stored_reviews"],
        "valid_reviews": totals["valid_reviews"],
        "parse_errors": totals["parse_errors"],
        "schema_errors": totals["schema_errors"],
        "other_invalid": totals["other_invalid"],
        "missing_reviews": totals["missing_reviews"],
        "missing_conditions": totals["missing_conditions"],
        "affected_pairs": len(issues),
        "repair_runs": len(issues),
        "by_condition": {
            condition_id: dict(condition_totals)
            for condition_id, condition_totals in by_condition.items()
        },
        "issues": issues,
    }


def audit_summary_file(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "source": str(path),
            "file_error": str(exc),
            "affected_pairs": 0,
            "parse_errors": 0,
            "schema_errors": 0,
            "other_invalid": 0,
            "missing_reviews": 0,
        }
    if not isinstance(data, dict):
        return {
            "source": str(path),
            "file_error": "summary root is not a JSON object",
            "affected_pairs": 0,
            "parse_errors": 0,
            "schema_errors": 0,
            "other_invalid": 0,
            "missing_reviews": 0,
        }
    return audit_summary_data(data, str(path))


def discover_summaries(paths: list[str], recursive: bool = False) -> list[Path]:
    found = set()
    pattern = "**/experiment*summary*.json" if recursive else "experiment*summary*.json"
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            found.add(path.resolve())
        elif path.is_dir():
            found.update(candidate.resolve() for candidate in path.glob(pattern))
        else:
            raise FileNotFoundError(f"Path not found: {path}")
    return sorted(found, key=lambda path: (path.stat().st_mtime, str(path)), reverse=True)


def _print_report_details(report: dict) -> None:
    print(f"\nDETAILS: {report['source']}")
    if report.get("file_error"):
        print(f"  FILE ERROR: {report['file_error']}")
        return
    print("  By condition:")
    for condition_id, condition in report["by_condition"].items():
        invalid = (
            condition.get("parse_errors", 0)
            + condition.get("schema_errors", 0)
            + condition.get("other_invalid", 0)
        )
        print(
            f"    {condition_id:<10} valid {condition.get('valid_reviews', 0):>3}/"
            f"{condition.get('expected_reviews', 0):<3}  invalid {invalid:>3}  "
            f"repair runs {condition.get('affected_pairs', 0):>2}"
        )
    print("  Affected paper/condition pairs:")
    for issue in report["issues"]:
        print(
            f"    {issue['paper_id']}/{issue['condition_id']}: "
            f"valid={issue['valid_reviews']}/{issue['expected_reviews']} "
            f"parse={issue['parse_errors']} schema={issue['schema_errors']} "
            f"other={issue['other_invalid']}"
        )


def _run_label(report: dict) -> str:
    timestamp = str(report.get("timestamp") or "unknown")
    parent = Path(report["source"]).parent.name
    aliases = {
        "advanced_exp_results_openrouter_v31_promptv2_rep2": "promptv2-rep2",
        "advanced_exp_results_openrouter_v31_fresh_20260730_231615": "openrouter-v31-fresh",
        "advanced_exp_results_openrouter_v31": "openrouter-v31",
        "advanced_exp_results": "deepseek",
        "exp_results": "legacy",
        "exp_baseline_results": "baseline",
    }
    if parent and parent != ".":
        tag = aliases.get(parent, parent)
        if len(tag) > 22:
            tag = tag[:19] + "..."
        return f"{timestamp} [{tag}]"
    return timestamp


def _print_final_summary(reports: list[dict], totals: dict) -> None:
    width = 108
    print("\n" + "=" * width)
    print("FINAL AUDIT SUMMARY")
    print("=" * width)
    header = (
        f"{'RUN':<38} {'PAPERS':>6} {'VALID REVIEWS':>13} {'COVERAGE':>9} "
        f"{'INVALID':>8} {'REPAIR RUNS':>11} {'STATUS':>9}"
    )
    print(header)
    print("-" * width)
    for report in reports:
        if report.get("file_error"):
            print(
                f"{_run_label(report):<38} {'-':>6} {'-':>13} {'-':>9} "
                f"{'-':>8} {'-':>11} {'ERROR':>9}"
            )
            continue
        invalid = (
            report["parse_errors"] + report["schema_errors"] + report["other_invalid"]
        )
        status = "CLEAN" if not report["affected_pairs"] else "REPAIR"
        valid_cell = f"{report['valid_reviews']}/{report['expected_reviews']}"
        coverage = (
            100 * report["valid_reviews"] / report["expected_reviews"]
            if report["expected_reviews"] else 0.0
        )
        print(
            f"{_run_label(report):<38} {report['paper_count']:>6} {valid_cell:>13} "
            f"{coverage:>8.1f}% {invalid:>8} {report['repair_runs']:>11} {status:>9}"
        )
    print("-" * width)
    print(
        f"Runs audited: {totals['summary_count']} | "
        f"Clean: {totals['clean_summaries']} | "
        f"Need repair: {totals['summaries_with_result_problems']} | "
        f"Unreadable: {totals['summaries_with_file_errors']}"
    )
    print(
        f"Invalid returns: {totals['invalid_returns']} "
        f"(JSON parse: {totals['parse_errors']}, schema: {totals['schema_errors']}, "
        f"other: {totals['other_invalid']}) | "
        f"Paper-condition repair runs needed: {totals['repair_runs']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit experiment summaries for invalid JSON and unusable reviewer returns."
    )
    parser.add_argument("paths", nargs="+", help="Summary JSON file(s) or directories")
    parser.add_argument(
        "--recursive", action="store_true", help="Recursively scan supplied directories"
    )
    parser.add_argument(
        "--show_issues", action="store_true", help="List every affected paper/condition"
    )
    parser.add_argument("--json_output", default=None, help="Optional audit report JSON")
    parser.add_argument(
        "--fail_on_invalid",
        action="store_true",
        help="Exit with status 1 if any invalid or missing result is found",
    )
    args = parser.parse_args()

    try:
        summaries = discover_summaries(args.paths, recursive=args.recursive)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not summaries:
        parser.error("No experiment summary JSON files found")

    reports = [audit_summary_file(path) for path in summaries]
    if args.show_issues:
        for report in reports:
            if report.get("file_error") or report.get("affected_pairs"):
                _print_report_details(report)

    totals = {
        "summary_count": len(reports),
        "summaries_with_file_errors": sum(bool(report.get("file_error")) for report in reports),
        "summaries_with_result_problems": sum(
            bool(
                report.get("parse_errors")
                or report.get("schema_errors")
                or report.get("other_invalid")
                or report.get("missing_reviews")
                or report.get("missing_conditions")
            )
            for report in reports
        ),
        "parse_errors": sum(report.get("parse_errors", 0) for report in reports),
        "schema_errors": sum(report.get("schema_errors", 0) for report in reports),
        "other_invalid": sum(report.get("other_invalid", 0) for report in reports),
        "missing_reviews": sum(report.get("missing_reviews", 0) for report in reports),
        "repair_runs": sum(report.get("repair_runs", 0) for report in reports),
    }
    totals["clean_summaries"] = (
        totals["summary_count"]
        - totals["summaries_with_result_problems"]
        - totals["summaries_with_file_errors"]
    )
    totals["invalid_returns"] = (
        totals["parse_errors"] + totals["schema_errors"] + totals["other_invalid"]
    )
    _print_final_summary(reports, totals)

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"totals": totals, "summaries": reports}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Audit JSON saved: {output_path}")

    has_problem = bool(
        totals["summaries_with_file_errors"] or totals["summaries_with_result_problems"]
    )
    if args.fail_on_invalid and has_problem:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
