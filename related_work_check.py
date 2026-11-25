"""Standalone related-work check for authors.

Runs retrieval and the related-work analyst only: no reviewers, no scores, no decision. The output
is for the author's own information and is never compared with human reviews.

Steps:
  1. Build (or reuse) a related-work package: LLM search queries plus one query per claimed
     contribution, OpenAlex and arXiv search, reference expansion from the manuscript's own
     references, removal of the submission itself, LLM relevance ranking.
  2. Refine it: tag each work as cited / not cited in the manuscript, drop works below the
     relevance threshold, keep the top works and label them [R1], [R2], ...
  3. The related-work analyst (a short-context call that sees the title, abstract, claimed
     contributions and the labeled works, not the full paper) reports, per claimed contribution,
     the closest prior work and the overlap (none / partial / substantial), plus uncited works the
     authors should discuss. A work is reported as possibly missing only if the analyst also named
     it as overlapping a claimed contribution.

Retrieval is not limited by a cutoff date here (cutoff_mode "today"): an author wants all prior work.

Usage (from the repository root):
    python related_work_check.py --paper data/md/my_paper.md --provider openrouter \
        --model deepseek/deepseek-chat-v3.1 [--output results/related_work_check.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Retrieval and refinement settings for the standalone check. The refinement settings are those of
# experiment condition 11 (analyst, no uncited quota, missing references must overlap a claim).
STANDALONE_RAG_CONFIG: dict[str, Any] = {
    "enable_rag": True,
    "cutoff_mode": "today",
    "strict_self_exclusion": True,
    "claim_queries": 4,
    "reference_expansion": True,
    "provider_top_k": 50,
    "rerank_top_k": 30,
    "rerank_batch_size": 10,
    "tag_cited": True,
    "min_relevance": 0.5,
    "max_shown": 5,
    "min_uncited_shown": 0,
    "prompt_style": "engaged",
    "related_work_analyst": True,
    "missing_references_report": True,
    "missing_requires_overlap": True,
    "missing_reference_min_relevance": 0.65,
}
OTHER_UNCITED_MAX = 5     # retrieval-only uncited works listed (not judged by the analyst)


def _work_summary(work: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": work.get("label"),
        "title": work.get("title", ""),
        "authors": list(work.get("authors") or [])[:6],
        "year": work.get("year"),
        "venue": work.get("venue", ""),
        "url": work.get("url") or (f"https://doi.org/{work['doi']}" if work.get("doi") else ""),
        "relevance_score": work.get("relevance_score"),
        "cited_in_manuscript": work.get("cited_in_manuscript"),
    }


def run_related_work_check(
    paper: str,
    provider: str = "openrouter",
    model: str = "",
    api_key: str = "",
    rag_config: dict[str, Any] | None = None,
    precomputed_rag_package: dict[str, Any] | None = None,
    on_event: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Related-work check for one manuscript (Markdown text). Returns a JSON-serializable dict.

    status is "complete", "no_relevant_work" (nothing passed the relevance filter, so the analyst
    was not called) or "failed" (retrieval or the analyst call failed; see "errors").
    """
    from agents import RelatedWorkAnalyst
    from mas_loop import _parse_json
    from prompts.reviewer_rag import ANALYST_OVERLAP_RULE
    from rag import build_rag_package
    from rag.config import rag_config_from_dict
    from rag.missing_references import build_missing_references
    from rag.prompt_views import format_analyst_input
    from rag.refine import refine_package
    from review_schema import normalize_rag_review_fields, validate_rw_analyst_report

    def emit(message: str) -> None:
        if on_event:
            on_event(message)

    settings = {**STANDALONE_RAG_CONFIG, **(rag_config or {})}
    cfg = rag_config_from_dict(settings)
    result: dict[str, Any] = {
        "status": "failed",
        "settings": {k: settings[k] for k in sorted(STANDALONE_RAG_CONFIG) if k in settings},
        "shown_works": [],
        "related_work_report": None,
        "possibly_missing_references": [],
        "other_relevant_uncited": [],
        "warnings": [],
        "errors": [],
    }

    # 1. Retrieval
    if precomputed_rag_package:
        package = precomputed_rag_package
        emit("Using the related-work package that was already built.")
    else:
        emit("Retrieving related work (search queries, OpenAlex and arXiv, reference expansion, ranking)...")
        try:
            package = build_rag_package(paper=paper, topic="", provider=provider, model=model,
                                        api_key=api_key, config=settings, progress=emit)
        except Exception as exc:
            result["errors"].append(f"Retrieval failed: {exc}")
            return result
    result["warnings"] += list(package.get("warnings") or [])
    result["cutoff_report"] = package.get("cutoff_report", {})
    result["reranking_source"] = (package.get("reranking") or {}).get("source")
    # "mixed" = LLM ranking with candidates the LLM omitted appended at low scores, which is normal.
    if result["reranking_source"] not in (None, "llm", "mixed", "none"):
        result["warnings"].append(f"Relevance ranking used the keyword fallback ({result['reranking_source']}), "
                                  "so relevance scores are less reliable.")

    # 2. Refinement
    package = refine_package(package, paper, cfg)
    shown = package.get("shown_works") or []
    refinement = package.get("refinement") or {}
    result["shown_works"] = [_work_summary(w) for w in shown]
    result["refinement"] = {k: refinement.get(k) for k in (
        "num_candidates", "num_shown", "num_shown_uncited", "manuscript_references_found",
        "removed_as_submission")}
    if refinement.get("manuscript_references_found") is False:
        result["warnings"].append("The manuscript's reference list was not found, so cited / not cited "
                                  "could not be determined and no work is reported as missing.")
    emit(f"{len(shown)} retrieved work(s) passed the relevance filter "
         f"({refinement.get('num_shown_uncited', 0)} not cited in the manuscript).")
    if not shown:
        result["status"] = "no_relevant_work"
        result["related_work_report"] = {
            "contribution_overlap": [], "possibly_missing_references": [],
            "summary": "No retrieved prior work passed the relevance filter for this manuscript.",
        }
        return result

    # 3. Related-work analyst
    labels = frozenset(w["label"] for w in shown)
    uncited = (frozenset(w["label"] for w in shown if w.get("cited_in_manuscript") is False)
               if any(w.get("cited_in_manuscript") is not None for w in shown) else None)
    analyst = RelatedWorkAnalyst(analyst_input=format_analyst_input(package),
                                 api_key=api_key, provider=provider, model=model)
    analyst.extra_output_validator = (
        lambda parsed: validate_rw_analyst_report(parsed, labels, uncited))
    emit("Related-work analyst comparing the claimed contributions with the retrieved work...")
    prompt = ("Compare the submission's claimed contributions with the retrieved works and return the "
              "JSON report." + (ANALYST_OVERLAP_RULE if cfg.missing_requires_overlap else ""))
    try:
        raw = analyst.call(prompt)
    except Exception as exc:
        raw = None
        result["errors"].append(f"Related-work analyst failed: {type(exc).__name__}")
    if not raw:
        if not result["errors"]:
            result["errors"].append("Related-work analyst returned no valid report after its retries.")
        return result
    report = normalize_rag_review_fields(_parse_json(raw), uncited, key="possibly_missing_references",
                                         require_overlap=cfg.missing_requires_overlap)
    result["related_work_report"] = report

    # Analyst-backed entries are the reported missing references. Works the analyst was shown but did
    # not flag are left out (it judged them). Uncited works it was not shown are listed separately by
    # retrieval relevance only, capped, since no overlap with a claim was established for them.
    merged = build_missing_references(package, report, [], cfg.missing_reference_min_relevance)
    result["possibly_missing_references"] = [m for m in merged if "analyst" in m["sources"]]
    result["other_relevant_uncited"] = [m for m in merged
                                        if "analyst" not in m["sources"] and not m.get("label")][:OTHER_UNCITED_MAX]
    result["status"] = "complete"
    emit(f"Related-work check done: {len(report.get('contribution_overlap') or [])} contribution(s) compared, "
         f"{len(result['possibly_missing_references'])} possibly missing reference(s).")
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description="Standalone related-work check (no review, no scores).")
    parser.add_argument("--paper", required=True, help="Manuscript in Markdown (e.g. data/md/<name>.md).")
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--model", default="")
    parser.add_argument("--api_key", default="", help="Defaults to the provider's environment variable.")
    parser.add_argument("--rag_package", default="",
                        help="Optional saved package (JSON) to reuse instead of retrieving again.")
    parser.add_argument("--output", default="", help="Defaults to results/<time>_related_work_check_<paper>.json")
    args = parser.parse_args()

    paper_path = Path(args.paper)
    paper = paper_path.read_text(encoding="utf-8")
    package = json.loads(Path(args.rag_package).read_text(encoding="utf-8")) if args.rag_package else None

    # A line every 30 s while retrieval or the analyst call is still running.
    start, stop = time.time(), threading.Event()

    def heartbeat() -> None:
        while not stop.wait(30):
            print(f"  ... still running ({time.time() - start:.0f} s)", flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        result = run_related_work_check(paper, provider=args.provider, model=args.model,
                                        api_key=args.api_key, precomputed_rag_package=package,
                                        on_event=lambda m: print(f"[{time.time() - start:5.0f} s] {m}", flush=True))
    finally:
        stop.set()

    out = Path(args.output) if args.output else (
        ROOT / "results" / f"{datetime.now():%y%m%d%H%M}_related_work_check_{paper_path.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"status: {result['status']}; wrote {out}", flush=True)
    for error in result["errors"]:
        print(f"ERROR: {error}", flush=True)


if __name__ == "__main__":
    _main()
