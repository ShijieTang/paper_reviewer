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
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import RAGConfig  # noqa: E402
from rag.llm import RAGLLMAgent  # noqa: E402
from rag.related_work import build_related_work_rag  # noqa: E402

_lock = threading.Lock()


def _log(paper_id: str, message: str) -> None:
    with _lock:
        print(f"[{paper_id}] {message}", flush=True)


def build_one(paper_id: str, meta: dict, config: RAGConfig, provider: str, model: str, api_key: str) -> dict:
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
    parser.add_argument("--rerank_top_k", type=int, default=16,
                        help="Candidates passed to the reranker (raised from 12 to leave room for expansion results).")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args(argv)

    dataset = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))
    if isinstance(dataset, list):
        dataset = {item["paper_id"]: item for item in dataset}
    ids = args.paper_ids or sorted(dataset)
    out_path = ROOT / args.output
    packages = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    todo = [pid for pid in ids if pid not in packages]
    config = RAGConfig(
        enable_rag=True, cutoff_date=args.cutoff_date, cutoff_mode=args.cutoff_mode,
        strict_self_exclusion=True, claim_queries=args.claim_queries,
        reference_expansion=not args.no_reference_expansion, rerank_top_k=args.rerank_top_k,
    )
    print(f"{len(todo)} package(s) to build ({len(ids) - len(todo)} already in {args.output})", flush=True)

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
        futures = {pool.submit(build_one, pid, dataset[pid], config, args.provider, args.model, args.api_key): pid
                   for pid in todo}
        for future in as_completed(futures):
            pid = futures[future]
            try:
                package = future.result()
            except Exception as exc:
                failures.append(pid)
                _log(pid, f"FAILED: {type(exc).__name__}: {exc}")
                continue
            packages[pid] = package
            save()
            report = package.get("cutoff_report", {})
            _log(pid, f"saved [{len(packages)}/{len(ids)}] ({report.get('num_used', 0)} candidates; "
                      f"{report.get('num_removed_as_target', 0)} removed as the submission itself)")
    stop.set()
    print(f"Done: {len(packages)} package(s) in {args.output}; {len(failures)} failed: {failures}", flush=True)


if __name__ == "__main__":
    main()
