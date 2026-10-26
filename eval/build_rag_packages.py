#!/usr/bin/env python3
"""Build frozen related-work RAG packages with the improved retrieval options.

Each package is built once and saved, so the RAG conditions (C8, C9) read identical evidence
on every run and comparisons with the no-RAG control (C1) are fair. Needs network access
(OpenAlex, arXiv, Semantic Scholar) and an LLM for query generation and reranking.

Usage (from the repository root):
    python eval/build_rag_packages.py \\
        --dataset eval/openreview_60_module_test.json \\
        --output  eval/related_work_rag_60_v2.json \\
        --provider claude --model claude-haiku-5-5 \\
        [--claim_queries 4] [--no_reference_expansion] [--cutoff_mode venue|today] \\
        [--concurrency 3]

Re-running resumes: papers already present in --output are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import RAGConfig  # noqa: E402
from rag.llm import RAGLLMAgent  # noqa: E402
from rag.models import PaperMetadata, RelatedWorkQuery, TargetPaperSummary  # noqa: E402
from rag.related_work import _rerank_in_batches, build_related_work_rag  # noqa: E402

_lock = threading.Lock()
_budget_exhausted = threading.Event()


def openalex_refused(package: dict) -> bool:
    """True when OpenAlex refused requests (HTTP 429: daily budget used up or rate limit)."""
    texts = list(package.get("warnings") or [])
    for name, status in (package.get("provider_status") or {}).items():
        if "openalex" in name.lower() or name == "Reference expansion":
            texts += status.get("warnings") or []
    return any("429" in str(t) and ("openalex" in str(t).lower() or "OpenAlex" in str(t)) for t in texts) or \
        (package.get("provider_status", {}).get("OpenAlex", {}).get("status") == "failed")


def _log(paper_id: str, message: str) -> None:
    with _lock:
        print(f"[{paper_id}] {message}", flush=True)


def build_one(paper_id: str, meta: dict, config: RAGConfig, provider: str, model: str, api_key: str) -> dict:
    if _budget_exhausted.is_set():
        raise RuntimeError("skipped: OpenAlex budget used up earlier in this run")
    path = ROOT / meta["paper_dir"]
    paper = path.read_text(encoding="utf-8", errors="replace")
    llm = RAGLLMAgent(provider=provider, api_key=api_key, model=model)
    start = time.time()
    _log(paper_id, "started")
    package = build_related_work_rag(paper=paper, provider=provider, model=model, api_key=api_key,
                                     config=config, llm_agent=llm, target_title=meta.get("title", ""),
                                     progress=lambda msg: _log(paper_id, f"{msg} [{time.time() - start:.0f}s]"))
    package["dataset_paper_id"] = paper_id
    package["build_config"] = {k: getattr(config, k) for k in config.__dataclass_fields__}
    return package


def rerank_package(paper_id: str, package: dict, config: RAGConfig, provider: str, model: str, api_key: str) -> dict:
    """Redo only the LLM reranking of a saved package: same queries and candidates, no new searches."""
    llm = RAGLLMAgent(provider=provider, api_key=api_key, model=model)
    start = time.time()
    _log(paper_id, f"re-ranking {len(package.get('paper_metadata') or [])} saved candidates")
    target = TargetPaperSummary(**{k: v for k, v in package["target_paper_summary"].items()
                                   if k in TargetPaperSummary.__dataclass_fields__})
    queries = [RelatedWorkQuery(**{k: v for k, v in q.items() if k in RelatedWorkQuery.__dataclass_fields__})
               for q in (package.get("query_generation") or {}).get("queries") or []]
    papers = [PaperMetadata(**{k: v for k, v in m.items() if k in PaperMetadata.__dataclass_fields__})
              for m in package.get("paper_metadata") or []]
    warnings = [w for w in package.get("warnings") or [] if "rerank" not in w.lower()]
    reranked, summary, source = _rerank_in_batches(
        target=target, queries=queries, papers=papers, llm_agent=llm, batch_size=config.rerank_batch_size,
        warnings=warnings, progress=lambda msg: _log(paper_id, f"{msg} [{time.time() - start:.0f}s]"))
    out = dict(package)
    out.update({"reranking_results": [item.to_dict() for item in reranked], "reranking": {"source": source},
                "related_work_summary": summary, "warnings": warnings})
    out["build_config"] = {**(package.get("build_config") or {}), "rerank_batch_size": config.rerank_batch_size,
                           "reranked_only": True}
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="eval/openreview_60_module_test.json")
    parser.add_argument("--output", default="eval/related_work_rag_60_v2.json")
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--model", default="")
    parser.add_argument("--api_key", default="", help="Defaults to the provider's environment variable.")
    parser.add_argument("--paper_ids", nargs="*", default=None)
    parser.add_argument("--cutoff_mode", choices=["venue", "today"], default="venue")
    parser.add_argument("--cutoff_date", default="2024-12-31",
                        help="Used with --cutoff_mode venue (same default as the earlier packages).")
    parser.add_argument("--claim_queries", type=int, default=4)
    parser.add_argument("--no_reference_expansion", action="store_true")
    parser.add_argument("--rerank_top_k", type=int, default=30,
                        help="Candidates passed to the reranker (12 in the earlier packages; raised so search "
                             "results are not crowded out by reference-expansion results).")
    parser.add_argument("--provider_top_k", type=int, default=50,
                        help="Results per search provider, split across all queries (the old default of 10 "
                             "gave about one result per query).")
    parser.add_argument("--rerank_batch_size", type=int, default=10,
                        help="Candidates per rerank LLM call; larger candidate sets are split into batches.")
    parser.add_argument("--rerank_only", action="store_true",
                        help="Redo only the LLM reranking of packages already in --output (same queries and "
                             "candidates, no searches, no OpenAlex cost). Use after a run whose reranking failed.")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args(argv)

    dataset = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))
    if isinstance(dataset, list):
        dataset = {item["paper_id"]: item for item in dataset}
    ids = args.paper_ids or sorted(dataset)
    out_path = ROOT / args.output
    packages = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    todo = [pid for pid in ids if pid not in packages]
    if args.rerank_only:
        todo = [pid for pid in ids if pid in packages
                and not (packages[pid].get("build_config") or {}).get("reranked_only")]
    config = RAGConfig(
        enable_rag=True, cutoff_date=args.cutoff_date, cutoff_mode=args.cutoff_mode,
        strict_self_exclusion=True, claim_queries=args.claim_queries,
        reference_expansion=not args.no_reference_expansion, rerank_top_k=args.rerank_top_k,
        provider_top_k=args.provider_top_k, rerank_batch_size=args.rerank_batch_size,
    )
    if config.reference_expansion and not args.rerank_only:
        source = ("Semantic Scholar (key set)" if os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
                  else "OpenAlex (no Semantic Scholar key)")
        print(f"reference expansion via {source}", flush=True)
    if not os.environ.get("OPENALEX_API_KEY") and not args.rerank_only:
        print("WARNING: OPENALEX_API_KEY is not set. Since 2026 OpenAlex allows only a tiny free budget "
              "without a key, so searches will soon fail with HTTP 429. Get a free key at "
              "https://openalex.org/settings/api and export OPENALEX_API_KEY=...", flush=True)
    print(f"{len(todo)} package(s) to {'re-rank' if args.rerank_only else 'build'} ({len(ids) - len(todo)} already in {args.output})", flush=True)

    def save() -> None:
        with _lock:
            out_path.write_text(json.dumps(packages, indent=2, ensure_ascii=False), encoding="utf-8")

    failures = []
    stop = threading.Event()
    cache_dir = ROOT / config.rag_cache_dir

    def heartbeat() -> None:
        # A line every 30 s so a long network wait never looks like a hang.
        begin = time.time()
        while not stop.wait(30):
            files = sum(1 for _ in cache_dir.rglob("*") if _.is_file()) if cache_dir.exists() else 0
            with _lock:
                print(f"... still working: {time.time() - begin:.0f}s elapsed, {len(packages)} package(s) saved, "
                      f"{files} cached API responses", flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        if args.rerank_only:
            futures = {pool.submit(rerank_package, pid, packages[pid], config, args.provider, args.model,
                                   args.api_key): pid for pid in todo}
        else:
            futures = {pool.submit(build_one, pid, dataset[pid], config, args.provider, args.model,
                                   args.api_key): pid for pid in todo}
        for future in as_completed(futures):
            pid = futures[future]
            try:
                package = future.result()
            except Exception as exc:
                failures.append(pid)
                _log(pid, f"FAILED: {type(exc).__name__}: {exc}")
                continue
            if openalex_refused(package):
                # Do not save a starved package: the resume run must rebuild it.
                _budget_exhausted.set()
                failures.append(pid)
                _log(pid, "NOT saved: OpenAlex refused requests (HTTP 429, probably the daily budget). "
                          "Rerun the same command after the reset (00:00 UTC); finished papers are kept.")
                continue
            packages[pid] = package
            save()
            report = package.get("cutoff_report", {})
            status = ", ".join(f"{name}: {v.get('retrieved', 0)} ({v.get('status')})"
                               for name, v in (package.get("provider_status") or {}).items())
            _log(pid, f"sources -> {status}")
            rerank_source = (package.get("reranking") or {}).get("source")
            if rerank_source != "llm":
                _log(pid, f"WARNING: reranking source is '{rerank_source}' (LLM reranking failed for some or all "
                          "candidates; scores are lexical and most works will fall below the relevance filter)")
            if report.get("num_used", 0) < 5:
                _log(pid, f"WARNING: only {report.get('num_used', 0)} candidate(s) survived; check the warnings "
                          f"in the package (rate limits?) before running the review conditions")
            _log(pid, f"saved [{len(packages)}/{len(ids)}] ({report.get('num_used', 0)} candidates; "
                      f"{report.get('num_removed_as_target', 0)} removed as the submission itself)")
    stop.set()
    print(f"Done: {len(packages)} package(s) in {args.output}; {len(failures)} failed: {failures}", flush=True)


if __name__ == "__main__":
    main()
