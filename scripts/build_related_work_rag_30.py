#!/usr/bin/env python3
"""Turn eval/openalex_related_30.json into cached RAG packages for the 30-paper set.

For each of the 30 papers (see eval/paper_set_30.json):
  1. Load the OpenAlex-retrieved related-work candidates.
  2. Backfill missing abstracts via an arXiv title search (only ~26 candidates
     lack one; OpenAlex often omits abstracts for policy reasons).
  3. Dedupe/cutoff-filter the candidates the same way build_related_work_rag does.
  4. Rerank with an LLM and produce a related_work_summary, reusing
     rag/related_work.py's _rerank_with_llm so the output has the same shape
     as a live build_related_work_rag() call.

The result is a cache: later small experiments on this fixed 30-paper set can
read eval/related_work_rag_30.json instead of re-hitting OpenAlex/arXiv/the
LLM every run.

Usage:
    python scripts/build_related_work_rag_30.py \
        --input eval/openalex_related_30.json \
        --output eval/related_work_rag_30.json \
        --provider openrouter --model deepseek/deepseek-chat-v3.1
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import RAGConfig
from rag.date_filter import filter_by_cutoff
from rag.llm import RAGLLMAgent
from rag.models import PaperMetadata, RelatedWorkQuery
from rag.providers.arxiv import ArxivProvider
from rag.related_work import _dedupe_papers, _rerank_with_llm
from rag.security import prompt_injection_warnings
from rag.target_parser import summarize_target_paper


def to_paper_metadata(raw: dict) -> PaperMetadata:
    return PaperMetadata(**raw)


# stdout is shared by every worker thread, so progress lines are emitted under a lock.
_print_lock = threading.Lock()


def _log(paper_id: str, message: str) -> None:
    with _print_lock:
        print(f"[{paper_id}] {message}", flush=True)


def backfill_missing_abstracts(paper_id: str, papers: list[PaperMetadata], arxiv: ArxivProvider) -> None:
    for paper in papers:
        if paper.abstract.strip():
            continue
        _log(paper_id, f"backfill: querying arXiv for {paper.title[:70]!r}")
        query = RelatedWorkQuery(group="abstract_backfill", query=paper.title)
        result = arxiv.search([query], limit=1)
        if not result.papers:
            _log(paper_id, "backfill: no arXiv match")
            continue
        match = result.papers[0]
        if match.title.strip().lower() == paper.title.strip().lower() and match.abstract.strip():
            paper.abstract = match.abstract
            if not paper.arxiv_id:
                paper.arxiv_id = match.arxiv_id
            paper.sources = sorted(set(paper.sources + ["arXiv"]))
            _log(paper_id, "backfill: matched, abstract filled")
        else:
            _log(paper_id, "backfill: closest match title didn't match exactly, skipped")


def build_rag_package(paper_id: str, entry: dict, llm_agent: RAGLLMAgent, config: RAGConfig, arxiv: ArxivProvider) -> dict:
    markdown = Path(entry["paper_dir"]).read_text(encoding="utf-8", errors="replace")
    target = summarize_target_paper(markdown)
    warnings = prompt_injection_warnings(markdown, "target paper")

    candidates = [to_paper_metadata(raw) for raw in entry["related_works"]]
    backfill_missing_abstracts(paper_id, candidates, arxiv)

    deduped = _dedupe_papers(candidates)
    cutoff_valid, cutoff_report = filter_by_cutoff(
        deduped, cutoff_date=config.cutoff_date, allow_undated=config.allow_undated_evidence
    )
    filtered = cutoff_valid[: config.rerank_top_k]
    cutoff_report["num_cutoff_valid"] = len(cutoff_valid)
    cutoff_report["candidate_cap"] = config.rerank_top_k
    cutoff_report["num_removed_by_candidate_cap"] = max(0, len(cutoff_valid) - len(filtered))
    cutoff_report["num_used"] = len(filtered)

    for paper_meta in filtered:
        warnings.extend(prompt_injection_warnings(paper_meta.abstract, paper_meta.paper_id))

    _log(paper_id, f"reranking {len(filtered)} candidates with LLM...")
    query = RelatedWorkQuery(group="related_work", query=entry["title"])
    reranked, summary, rerank_source = _rerank_with_llm(
        target=target,
        queries=[query],
        papers=filtered,
        llm_agent=llm_agent,
        top_k=len(filtered),
        warnings=warnings,
    )

    return {
        "paper_id": target.paper_id,
        "conference": entry["conference"],
        "year": entry["year"],
        "decision": entry["decision"],
        "paper_dir": entry["paper_dir"],
        "target_paper_summary": target.to_dict(),
        "paper_metadata": [paper.to_dict() for paper in filtered],
        "reranking_results": [item.to_dict() for item in reranked],
        "reranking": {"source": rerank_source},
        "related_work_summary": summary,
        "warnings": warnings,
        "cutoff_report": cutoff_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("eval/openalex_related_30.json"))
    parser.add_argument("--output", type=Path, default=Path("eval/related_work_rag_30.json"))
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--model", default="deepseek/deepseek-chat-v3.1")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Papers processed in parallel (default: 8; use 1 for sequential).")
    args = parser.parse_args()

    dataset = json.loads(args.input.read_text(encoding="utf-8"))
    config = RAGConfig()
    llm_agent = RAGLLMAgent(provider=args.provider, api_key=args.api_key, model=args.model)
    llm_agent.client.client = llm_agent.client.client.with_options(timeout=90.0)
    arxiv = ArxivProvider(config.rag_cache_dir)

    output: dict[str, dict] = {}
    if args.output.exists():
        output = json.loads(args.output.read_text(encoding="utf-8"))
        if output:
            print(f"Resuming: {len(output)} paper(s) already in {args.output}, skipping those.", flush=True)

    remaining = {pid: entry for pid, entry in dataset.items() if pid not in output}
    concurrency = max(1, args.concurrency)
    output_lock = threading.Lock()

    def _process(paper_id: str, entry: dict) -> tuple[str, dict | None]:
        _log(paper_id, "starting")
        try:
            package = build_rag_package(paper_id, entry, llm_agent, config, arxiv)
        except Exception as exc:
            _log(paper_id, f"FAILED: {exc}")
            return paper_id, None
        _log(
            paper_id,
            f"{entry['conference']} {entry['decision']}: "
            f"{len(package['reranking_results'])} reranked "
            f"(source={package['reranking']['source']})",
        )
        return paper_id, package

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_process, pid, entry) for pid, entry in remaining.items()]
        for i, future in enumerate(as_completed(futures), 1):
            paper_id, package = future.result()
            if package is None:
                continue
            with output_lock:
                output[paper_id] = package
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
            _log(paper_id, f"saved ({len(output)}/{len(dataset)} total)")

    print(f"Wrote {len(output)} RAG packages to {args.output}", flush=True)


if __name__ == "__main__":
    main()
