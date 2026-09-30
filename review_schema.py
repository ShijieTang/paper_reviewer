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


def parse_json_object(text: str) -> dict:
    """Accept fences and trailing commas without changing quoted string values.

    Syntax tolerance preserves the remote parser's behavior; role validators
    still reject missing fields, invalid scores and wrong role outputs.
    """
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    text = re.sub(r'"(?:\\.|[^"\\])*"|,(?=\s*[}\]])',
                  lambda match: '' if match.group(0) == ',' else match.group(0), text)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("structured return must be a JSON object")
    return parsed


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
        if expected_reviewer is not None and response.get("reviewer") != expected_reviewer:
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
