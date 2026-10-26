"""Possibly missing references for authors (RAG improvement option 4).

Combines three sources, all restricted to works the manuscript does not cite:
  * retrieval: relevant retrieved works (relevance >= threshold) not found in the reference list;
  * analyst: works the related-work analyst flagged;
  * reviewers: works listed in reviewers' missing_prior_work fields.
"""
from __future__ import annotations

from typing import Any

from review_schema import _rag_label


def _entry(work: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": work.get("label"),
        "title": work.get("title", ""),
        "authors": list(work.get("authors") or [])[:6],
        "year": work.get("year"),
        "venue": work.get("venue", ""),
        "url": work.get("url") or (f"https://doi.org/{work['doi']}" if work.get("doi") else ""),
        "relevance_score": work.get("relevance_score"),
        "sources": [],
        "reasons": [],
    }


def build_missing_references(package: dict[str, Any], analyst_report: dict | None,
                             reviews: list | None, min_relevance: float = 0.65) -> list[dict[str, Any]]:
    candidates = {w.get("paper_id"): w for w in package.get("candidate_works", []) or []}
    by_label = {w.get("label"): w for w in package.get("shown_works", []) or [] if w.get("label")}
    merged: dict[str, dict[str, Any]] = {}

    def add(work: dict[str, Any] | None, source: str, reason: str = "") -> None:
        if not work or work.get("cited_in_manuscript") is not False or work.get("is_submission"):
            return    # only works known to be uncited, never the submission itself
        key = work.get("paper_id") or work.get("title", "")
        entry = merged.setdefault(key, _entry({**candidates.get(work.get("paper_id"), {}), **work}))
        if source not in entry["sources"]:
            entry["sources"].append(source)
        if reason and reason not in entry["reasons"]:
            entry["reasons"].append(reason)

    for work in candidates.values():
        if float(work.get("relevance_score") or 0) >= min_relevance:
            add(work, "retrieval", work.get("rationale", ""))
    for item in (analyst_report or {}).get("possibly_missing_references", []) or []:
        add(by_label.get(_rag_label(item.get("work"))), "analyst", item.get("why_relevant", ""))
    for index, review in enumerate(reviews or [], 1):
        if not isinstance(review, dict):
            continue
        name = review.get("reviewer") or f"reviewer {index}"
        for item in review.get("missing_prior_work", []) or []:
            if isinstance(item, dict):
                add(by_label.get(_rag_label(item.get("work"))), name, item.get("why_relevant", ""))

    return sorted(merged.values(), key=lambda e: (-len(e["sources"]), -(e["relevance_score"] or 0)))
