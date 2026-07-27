#!/usr/bin/env python3
"""
Run doc_preprocess.doc_preprocess() over every PDF in a directory in parallel.

Usage:
    python scripts/run_doc_preprocess_parallel.py \
        data/openreview_pdf data/openreview_md \
        [--workers N] [--limit N] [--overwrite]

Each worker process imports doc_preprocess once and reuses the lazily-loaded
marker model dict (the module-level _MODEL_DICT singleton) for every PDF it
handles, so the expensive model load happens once per worker, not once per PDF.

PDFs whose .md output already exists are skipped by default (resume-friendly);
pass --overwrite to reconvert them.

Note: on Apple Silicon this script uses the MPS device (see doc_preprocess.py).
Multiple marker instances sharing MPS can contend for GPU memory — if workers
crash with OOM, lower --workers (e.g. 2 or 3).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _convert_one(task: tuple[str, str, str, bool]) -> tuple[str, str, str | None]:
    """
    Worker entry point. Returns (pdf_name, status, output_path_or_error).

    The import is deferred to here so the main process (and ``--help``) does
    not need torch/marker installed. In a spawned worker process the import
    happens once, and doc_preprocess's module-level _MODEL_DICT singleton is
    then reused for every subsequent PDF this worker handles.
    """
    import doc_preprocess  # noqa: E402 - intentional lazy import in worker

    pdf_name, pdf_dir, md_dir, overwrite = task
    out_path = Path(md_dir) / Path(pdf_name).with_suffix(".md").name
    if out_path.exists() and not overwrite:
        return (pdf_name, "skipped", str(out_path))
    try:
        result = doc_preprocess.doc_preprocess(pdf_name, pdf_dir, md_dir)
        return (pdf_name, "ok", result)
    except Exception as exc:  # noqa: BLE001 - report any failure, don't kill the run
        return (pdf_name, "error", f"{type(exc).__name__}: {exc}")


def discover_pdfs(pdf_dir: Path) -> list[Path]:
    return sorted(pdf_dir.glob("*.pdf"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf_dir", help="Directory containing input PDF files (e.g. data/openreview_pdf)")
    parser.add_argument("md_dir", nargs="?", default="data/md",
                        help="Directory for output markdown files (default: data/md)")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="Only process the first N PDFs (after filtering). Useful for testing.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reconvert even if the .md output already exists.")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    md_dir = Path(args.md_dir)
    if not pdf_dir.is_dir():
        print(f"error: pdf_dir not found: {pdf_dir}", file=sys.stderr)
        return 2
    md_dir.mkdir(parents=True, exist_ok=True)

    pdfs = discover_pdfs(pdf_dir)
    if not pdfs:
        print(f"error: no .pdf files in {pdf_dir}", file=sys.stderr)
        return 2

    tasks = [(p.name, str(pdf_dir), str(md_dir), args.overwrite) for p in pdfs]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    # Pre-filter in the main process: skip PDFs whose .md already exists unless
    # --overwrite. This avoids spawning a worker (and importing torch/marker)
    # for files that are already converted — important when resuming a long run.
    n_before = len(tasks)
    tasks = [t for t in tasks
             if args.overwrite or not (Path(t[2]) / Path(t[0]).with_suffix(".md").name).exists()]
    n_skipped = n_before - len(tasks)

    n_total = len(tasks)
    n_workers = max(1, min(args.workers, n_total)) if n_total else 0
    print(f"Processing {n_total} PDF(s) with {n_workers} worker(s) "
          f"-> {pdf_dir} -> {md_dir}", flush=True)
    if n_skipped:
        print(f"Skipping {n_skipped} PDF(s) that already have .md output "
              f"in {md_dir} (use --overwrite to reconvert).", flush=True)
    if n_total == 0:
        print("Nothing to do.", flush=True)
        return 0

    # spawn avoids fork-inherited torch/marker state and forces a fresh
    # interpreter per worker, so the _MODEL_DICT singleton starts clean.
    ctx = mp.get_context("spawn")

    counts = {"ok": 0, "skipped": 0, "error": 0}
    failures: list[tuple[str, str]] = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        future_to_name = {ex.submit(_convert_one, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(future_to_name), 1):
            name, status, info = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if status == "error":
                failures.append((name, info))
                print(f"[{i}/{n_total}] ERROR  {name}: {info}", flush=True)
            elif status == "skipped":
                print(f"[{i}/{n_total}] skip   {name}", flush=True)
            else:
                print(f"[{i}/{n_total}] ok     {name} -> {info}", flush=True)

    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed:.1f}s — ok={counts['ok']} "
        f"skipped={counts['skipped']} error={counts['error']}",
        flush=True,
    )
    if failures:
        print("\nFailed PDFs:", file=sys.stderr)
        for name, info in failures:
            print(f"  - {name}: {info}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
