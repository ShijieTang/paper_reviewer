"""Shared, dependency-free validation for generated review workflow outputs.

Validators return field-level errors instead of silently repairing model output.
The same reviewer contract is used at generation, audit, repair and evaluation.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

REVIEW_SCORE_KEYS = ("novelty", "soundness", "significance", "evaluation", "clarity")
WORKFLOW_SCHEMA_VERSION = 1


_PERSONA_REVIEWER_ALIASES = {
    "neutral": {"reviewer - no persona", "reviewer (neutral)", "neutral reviewer"},
    "novelty": {"reviewer a - novelty focused", "reviewer (novelty)"},
    "rigor": {"reviewer b - rigor focused", "reviewer (rigor)"},
    "practical": {"reviewer c - practicality focused", "reviewer (practical)"},
}


def parse_json_object(text: str) -> dict:
    """Accept fences and trailing commas without changing quoted string values.

    Syntax tolerance preserves the remote parser's behavior; role validators
    still reject missing fields, invalid scores and wrong role outputs.
    """
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    text = re.sub(r'(\}\s*)\],(\s*"summary_comment"\s*:)', r'\1},\2', text)
    text = re.sub(r'"response:\s+"', '"response": "', text)
    text = re.sub(r'"(?:\\.|[^"\\])*"|,(?=\s*[}\]])',
                  lambda match: '' if match.group(0) == ',' else match.group(0), text)
    parsed = json.loads(text)
    if (isinstance(parsed, list) and parsed
            and all(isinstance(item, dict) and "reviewer" in item and "response" in item for item in parsed)):
        parsed = {"responses": parsed}
    if not isinstance(parsed, dict):
        raise ValueError("structured return must be a JSON object")
    return parsed


def _reviewer_matches_expected(actual: Any, expected: str) -> bool:
    if actual == expected:
        return True
    if not isinstance(actual, str):
        return False
    actual_norm = actual.strip().casefold()
    expected_norm = expected.strip().casefold()
    if actual_norm == expected_norm:
        return True
    match = re.search(r"\(([^()]+)\)\s*$", expected_norm)
    if match:
        persona_key = match.group(1).strip()
        if actual_norm in _PERSONA_REVIEWER_ALIASES.get(persona_key, set()):
            return True
    return False


def _number(value: Any, low: float, high: float) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) and low <= value <= high)


def _text_list(value: Any, *, nonempty: bool = False) -> bool:
    return (isinstance(value, list) and (bool(value) or not nonempty)
            and all(isinstance(item, str) and bool(item.strip()) for item in value))


def validate_review_schema(value: Any) -> list[str]:
    if not isinstance(value, dict) or value.get("parse_error"):
        return ["review must be an object without parse_error"]
    errors = []
    if str(value.get("decision", "")).strip().casefold() not in {"accept", "reject"}:
        errors.append("decision must be accept or reject")
    scores = value.get("scores")
    for key in REVIEW_SCORE_KEYS:
        if not isinstance(scores, dict) or not _number(scores.get(key), 1, 5):
            errors.append(f"scores.{key} must be a finite number in [1, 5]")
    for key in ("strengths", "weaknesses"):
        if not _text_list(value.get(key), nonempty=True):
            errors.append(f"{key} must be a nonempty list of nonempty strings")
    return errors


def review_is_valid(value: Any) -> bool:
    return not validate_review_schema(value)


def validate_author_schema(value: Any, expected_reviewer: str | None = None) -> list[str]:
    if not isinstance(value, dict) or not isinstance(value.get("responses"), list) or not value["responses"]:
        return ["responses must be a nonempty list"]
    errors = []
    if expected_reviewer is not None and len(value["responses"]) != 1:
        errors.append("a rebuttal turn must address exactly its requested reviewer")
    for index, response in enumerate(value["responses"]):
        prefix = f"responses[{index}]"
        if not isinstance(response, dict):
            errors.append(f"{prefix} must be an object")
            continue
        for key in ("reviewer", "response"):
            if not isinstance(response.get(key), str) or not response[key].strip():
                errors.append(f"{prefix}.{key} must be nonempty text")
        if expected_reviewer is not None:
            if _reviewer_matches_expected(response.get("reviewer"), expected_reviewer):
                response["reviewer"] = expected_reviewer
            else:
                errors.append(f"{prefix}.reviewer must match requested reviewer")
        if not _text_list(response.get("main_issues_identified")):
            errors.append(f"{prefix}.main_issues_identified must be a string list")
        for key, fields in (
            ("supporting_evidence_in_submission", ("concern", "location", "explanation")),
            ("proposed_future_revisions", ("concern", "proposed_change")),
        ):
            items = response.get(key)
            if not isinstance(items, list):
                errors.append(f"{prefix}.{key} must be a list (empty is allowed)")
                continue
            for item in items:
                if not isinstance(item, dict) or any(
                    not isinstance(item.get(field), str) or not item[field].strip() for field in fields
                ):
                    errors.append(f"{prefix}.{key} contains an invalid evidence/revision object")
                elif key == "proposed_future_revisions" and not isinstance(
                    item.get("requires_new_experiment_or_analysis"), bool
                ):
                    errors.append(f"{prefix}.{key} requires a boolean experiment flag")
    return errors


def validate_conference_schema(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["conference recommendation must be an object"]
    errors = []
    for venue in ("ICLR", "ICML", "NeurIPS"):
        item = value.get(venue)
        if not isinstance(item, dict) or not _number(item.get("fit_score"), 1, 10):
            errors.append(f"{venue}.fit_score must be a finite number in [1, 10]")
    return errors


def validate_role_output(value: Any, role: str, expected_reviewer: str | None = None) -> list[str]:
    if role == "reviewer":
        return validate_review_schema(value)
    if role == "author":
        return validate_author_schema(value, expected_reviewer)
    if role == "conference":
        return validate_conference_schema(value)
    return [] if isinstance(value, dict) and value else ["output must be a nonempty JSON object"]


def workflow_is_complete(result: Any, reviewer_count: int, n_iter: int,
                         *, enable_author_rebuttal: bool = True) -> bool:
    """Require every scheduled turn, rather than trusting only final reviews."""
    if (not isinstance(result, dict)
            or result.get("workflow_schema_version") != WORKFLOW_SCHEMA_VERSION
            or result.get("workflow_status") != "complete"
            or result.get("turn_failures") != []):
        return False
    config = result.get("workflow_config")
    if (not isinstance(config, dict)
            or config.get("reviewer_count") != reviewer_count
            or config.get("n_iter") != n_iter
            or config.get("enable_author_rebuttal", True) is not enable_author_rebuttal):
        return False
    outcomes = result.get("turn_outcomes")
    if not isinstance(outcomes, list) or not all(isinstance(item, dict) for item in outcomes):
        return False
    roles = (("reviewer", 1), ("author", 2)) if enable_author_rebuttal else (("reviewer", 1),)
    expected = {(role, iteration, index)
                for role, start in roles
                for iteration in range(start, n_iter + 1) for index in range(reviewer_count)}
    expected.add(("conference", n_iter, None))
    actual = []
    for item in outcomes:
        if item.get("required") is True:
            if item.get("status") != "complete":
                return False
            actual.append((item.get("role"), item.get("iteration"), item.get("reviewer_index")))
        elif item.get("role") in {"reviewer", "author", "conference"}:
            return False
    return len(actual) == len(expected) and set(actual) == expected


# ── RAG improvements: related-work fields (only checked when the engaged RAG mode is on) ──

RAG_OVERLAP_LEVELS = ("none", "partial", "substantial")


def _rag_label(value: Any) -> str:
    """Accept 'R3', '[R3]' or 'r3'; return the canonical 'R3' (or '' if not a label)."""
    if not isinstance(value, str):
        return ""
    match = re.fullmatch(r"\s*\[?\s*[Rr]\s*(\d+)\s*\]?\s*", value)
    return f"R{int(match.group(1))}" if match else ""


def validate_rag_review_fields(value: Any, valid_labels: set[str] | list[str],
                               uncited_labels: set[str] | list[str] | None = None) -> list[str]:
    """Check the reviewer's novelty_check and missing_prior_work fields against the shown works.

    uncited_labels: labels of works the manuscript does not cite. When given, missing_prior_work
    may only point at those (a cited work is not missing). None means citation status is unknown.
    """
    valid = set(valid_labels)
    uncited = set(uncited_labels) if uncited_labels is not None else None
    errors: list[str] = []
    if not isinstance(value, dict):
        return ["review must be an object"]
    checks = value.get("novelty_check")
    if not isinstance(checks, list) or not checks:
        errors.append("novelty_check must be a nonempty list (one entry per claimed contribution)")
    else:
        for i, item in enumerate(checks):
            prefix = f"novelty_check[{i}]"
            if not isinstance(item, dict):
                errors.append(f"{prefix} must be an object")
                continue
            if not isinstance(item.get("claimed_contribution"), str) or not item["claimed_contribution"].strip():
                errors.append(f"{prefix}.claimed_contribution must be nonempty text")
            closest = item.get("closest_prior_work")
            label = _rag_label(closest)
            if isinstance(closest, str) and closest.strip().lower() == "none":
                item["closest_prior_work"] = "none"
            elif label and label in valid:
                item["closest_prior_work"] = label
            else:
                errors.append(f"{prefix}.closest_prior_work must be one of {sorted(valid)} or 'none'")
            if str(item.get("overlap", "")).strip().lower() not in RAG_OVERLAP_LEVELS:
                errors.append(f"{prefix}.overlap must be one of {list(RAG_OVERLAP_LEVELS)}")
            else:
                item["overlap"] = str(item["overlap"]).strip().lower()
            if not isinstance(item.get("explanation"), str) or not item["explanation"].strip():
                errors.append(f"{prefix}.explanation must be nonempty text")
    missing = value.get("missing_prior_work")
    if not isinstance(missing, list):
        errors.append("missing_prior_work must be a list (empty is allowed)")
    else:
        for i, item in enumerate(missing):
            prefix = f"missing_prior_work[{i}]"
            if not isinstance(item, dict):
                errors.append(f"{prefix} must be an object")
                continue
            label = _rag_label(item.get("work"))
            if not label or label not in valid:
                errors.append(f"{prefix}.work must be one of {sorted(valid)}")
            elif uncited is not None and label not in uncited:
                errors.append(f"{prefix}.work {label} is already cited in the manuscript; "
                              "missing_prior_work may list only works marked as not cited")
            else:
                item["work"] = label
            if not isinstance(item.get("why_relevant"), str) or not item["why_relevant"].strip():
                errors.append(f"{prefix}.why_relevant must be nonempty text")
    return errors


def validate_rw_analyst_report(value: Any, valid_labels: set[str] | list[str],
                               uncited_labels: set[str] | list[str] | None = None) -> list[str]:
    """Check the related-work analyst's report."""
    valid = set(valid_labels)
    uncited = set(uncited_labels) if uncited_labels is not None else None
    if not isinstance(value, dict):
        return ["report must be an object"]
    errors: list[str] = []
    overlaps = value.get("contribution_overlap")
    if not isinstance(overlaps, list) or not overlaps:
        errors.append("contribution_overlap must be a nonempty list")
    else:
        for i, item in enumerate(overlaps):
            prefix = f"contribution_overlap[{i}]"
            if not isinstance(item, dict) or not str(item.get("claimed_contribution", "")).strip():
                errors.append(f"{prefix}.claimed_contribution must be nonempty text")
                continue
            works = item.get("closest_prior_work")
            works = works if isinstance(works, list) else [works]
            canon = []
            for work in works:
                label = _rag_label(work)
                if isinstance(work, str) and work.strip().lower() == "none":
                    continue
                if not label or label not in valid:
                    errors.append(f"{prefix}.closest_prior_work entries must be among {sorted(valid)} or 'none'")
                else:
                    canon.append(label)
            item["closest_prior_work"] = canon
            if str(item.get("overlap", "")).strip().lower() not in RAG_OVERLAP_LEVELS:
                errors.append(f"{prefix}.overlap must be one of {list(RAG_OVERLAP_LEVELS)}")
            else:
                item["overlap"] = str(item["overlap"]).strip().lower()
            if not str(item.get("explanation", "")).strip():
                errors.append(f"{prefix}.explanation must be nonempty text")
    missing = value.get("possibly_missing_references")
    if not isinstance(missing, list):
        errors.append("possibly_missing_references must be a list (empty is allowed)")
    else:
        for i, item in enumerate(missing):
            label = _rag_label(item.get("work") if isinstance(item, dict) else None)
            if not label or label not in valid:
                errors.append(f"possibly_missing_references[{i}].work must be one of {sorted(valid)}")
            elif uncited is not None and label not in uncited:
                errors.append(f"possibly_missing_references[{i}].work {label} is already cited in the manuscript")
            elif not str(item.get("why_relevant", "")).strip():
                errors.append(f"possibly_missing_references[{i}].why_relevant must be nonempty text")
            else:
                item["work"] = label
    if not isinstance(value.get("summary"), str) or not value["summary"].strip():
        errors.append("summary must be nonempty text")
    return errors
