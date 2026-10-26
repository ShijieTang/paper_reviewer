"""Post-processing applied to every related-work package (freshly built or loaded from a cache).

It implements the retrieval-side improvements that do not need new searches:
  * strict self-exclusion: drop candidates that are the submission itself
    (near-identical title, or same arXiv ID / DOI as the submission);
  * cited tagging: mark each candidate as cited / not cited in the manuscript's
    reference list;
  * selection: drop low-relevance candidates and choose the works shown to
    reviewers, reserving slots for works the manuscript does not cite;
  * labels: give the shown works stable [R1]..[Rk] labels that reviewers and the
    analyst can cite.

Because it also runs on cached packages, old caches (built before the exact-title
self filter existed) are cleaned at run time.
"""
from __future__ import annotations

import copy
import difflib
import re
from typing import Any

from .config import RAGConfig

_ARXIV_RE = re.compile(r"(?:arxiv[:\s/]*|abs/)(\d{4}\.\d{4,5})", re.IGNORECASE)
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.IGNORECASE)


def norm_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (title or "").lower())).strip()


def title_similarity(a: str, b: str) -> float:
    a, b = norm_title(a), norm_title(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _arxiv_id(value: str) -> str:
    value = (value or "").strip().lower()
    match = re.search(r"(\d{4}\.\d{4,5})", value)
    return match.group(1) if match else ""


def extract_reference_section(paper_text: str) -> str:
    match = re.search(r"(?im)^#+\s*(references|bibliography)\s*$", paper_text or "")
    if not match:
        return ""
    start = match.end()
    nxt = re.search(r"(?m)^#\s+\S", paper_text[start:])   # stop at the next top-level heading (appendix)
    end = start + nxt.start() if nxt else len(paper_text)
    return paper_text[start:end]


class ManuscriptReferences:
    """The manuscript's reference list, prepared for 'is this work cited?' checks."""

    def __init__(self, paper_text: str):
        self.section = extract_reference_section(paper_text)
        self.norm_section = norm_title(self.section)
        self.arxiv_ids = {m.group(1) for m in _ARXIV_RE.finditer(self.section)}
        self.dois = {m.group(1).rstrip(".,;)").lower() for m in _DOI_RE.finditer(self.section)}
        self.titles: list[str] = []
        if self.section:
            try:
                from citation_checker.parser import parse_references
                self.titles = [r.title for r in parse_references(self.section) if r.title]
            except Exception:
                self.titles = []

    def __bool__(self) -> bool:
        return bool(self.section.strip())

    def cites(self, work: dict[str, Any]) -> bool:
        title = norm_title(work.get("title", ""))
        if title and len(title.split()) >= 4 and title in self.norm_section:
            return True
        arxiv = _arxiv_id(work.get("arxiv_id", ""))
        if arxiv and arxiv in self.arxiv_ids:
            return True
        doi = (work.get("doi") or "").lower().replace("https://doi.org/", "")
        if doi and doi in self.dois:
            return True
        return any(title_similarity(work.get("title", ""), t) >= 0.85 for t in self.titles)


def is_submission_itself(work: dict[str, Any], target_title: str, threshold: float,
                         target_arxiv_id: str = "", target_doi: str = "") -> bool:
    if title_similarity(work.get("title", ""), target_title) >= threshold:
        return True
    if target_arxiv_id and _arxiv_id(work.get("arxiv_id", "")) == _arxiv_id(target_arxiv_id):
        return True
    doi = (work.get("doi") or "").lower().replace("https://doi.org/", "")
    return bool(target_doi and doi and doi == target_doi.lower())


def refine_package(package: dict[str, Any], paper_text: str, config: RAGConfig) -> dict[str, Any]:
    """Return a copy of the package with self-exclusion, cited tags and the shown-work selection applied."""
    if not isinstance(package, dict) or not package:
        return package
    pkg = copy.deepcopy(package)
    target = pkg.get("target_paper_summary") or {}
    target_title = target.get("title", "")
    meta = {m.get("paper_id"): m for m in pkg.get("paper_metadata", []) if isinstance(m, dict)}
    ranked = sorted((r for r in pkg.get("reranking_results", []) if isinstance(r, dict)),
                    key=lambda r: r.get("rank", 10**6))

    removed_self, removed_low = [], []
    refs = ManuscriptReferences(paper_text) if (config.tag_cited or config.min_uncited_shown
                                                 or config.missing_references_report) else None
    candidates = []
    for item in ranked:
        work = meta.get(item.get("paper_id"))
        if not work:
            continue
        is_self = is_submission_itself(work, target_title, config.self_match_threshold)
        if config.strict_self_exclusion and is_self:
            removed_self.append(work.get("title", ""))
            continue
        score = float(item.get("relevance_score") or 0.0)
        if score < config.min_relevance:
            removed_low.append(work.get("title", ""))
            continue
        cited = refs.cites(work) if refs else None
        candidates.append({
            "paper_id": work.get("paper_id"),
            "title": work.get("title", ""),
            "authors": list(work.get("authors") or []),
            "year": work.get("year"),
            "publication_date": work.get("publication_date"),
            "venue": work.get("venue", ""),
            "url": work.get("url", ""),
            "doi": work.get("doi", ""),
            "arxiv_id": work.get("arxiv_id", ""),
            "abstract": work.get("abstract", ""),
            "relevance_score": score,
            "rationale": item.get("rationale", ""),
            "evidence_summary": item.get("evidence_summary", ""),
            "matched_query_groups": list(work.get("matched_query_groups") or []),
            "cited_in_manuscript": cited,
            "is_submission": is_self,   # kept only when strict_self_exclusion is off; never reported as missing
        })

    shown = select_shown(candidates, config.max_shown, config.min_uncited_shown)
    for i, work in enumerate(shown, 1):
        work["label"] = f"R{i}"

    pkg["shown_works"] = shown
    pkg["candidate_works"] = candidates
    pkg["refinement"] = {
        "strict_self_exclusion": config.strict_self_exclusion,
        "removed_as_submission": removed_self,
        "min_relevance": config.min_relevance,
        "removed_low_relevance": removed_low,
        "max_shown": config.max_shown,
        "min_uncited_shown": config.min_uncited_shown,
        "manuscript_references_found": bool(refs) if refs is not None else None,
        "num_candidates": len(candidates),
        "num_shown": len(shown),
        "num_shown_uncited": sum(1 for w in shown if w.get("cited_in_manuscript") is False),
    }
    return pkg


def select_shown(candidates: list[dict], max_shown: int, min_uncited: int) -> list[dict]:
    """Top-ranked works, keeping at least `min_uncited` uncited ones when available (rank order preserved)."""
    max_shown = max(0, int(max_shown))
    if min_uncited <= 0:
        return [dict(w) for w in candidates[:max_shown]]
    uncited = [w for w in candidates if w.get("cited_in_manuscript") is False]
    reserved = {id(w) for w in uncited[:min(min_uncited, max_shown)]}
    chosen = [w for w in candidates if id(w) in reserved]
    for work in candidates:
        if len(chosen) >= max_shown:
            break
        if id(work) not in reserved:
            chosen.append(work)
    order = {id(w): i for i, w in enumerate(candidates)}
    return [dict(w) for w in sorted(chosen, key=lambda w: order[id(w)])]


def retrieval_missing_references(package: dict[str, Any], min_relevance: float) -> list[dict]:
    """Relevant retrieved works that the manuscript does not cite (deterministic part of the report)."""
    out = []
    for work in package.get("candidate_works", []) or []:
        if (work.get("cited_in_manuscript") is False and not work.get("is_submission")
                and float(work.get("relevance_score") or 0) >= min_relevance):
            out.append(work)
    return out
