#!/usr/bin/env python3
"""Convert a directory of PDFs to Markdown in parallel."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_WORKER_LABEL = "Worker"


def initialize_worker(
    device: str,
    torch_threads: int,
    progress_lock,
    worker_counter,
    worker_count: int,
) -> None:
    """Load worker settings once before that process accepts PDF jobs."""
    # Let the parent handle Ctrl+C and terminate the whole pool cleanly.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Give every process a stable, labelled terminal row. Marker and Surya use
    # tqdm internally, so these defaults make their otherwise-colliding bars
    # visible at the same time.
    with worker_counter.get_lock():
        worker_position = worker_counter.value
        worker_counter.value += 1
    worker_position %= worker_count
    os.environ["TQDM_POSITION"] = str(worker_position)
    os.environ["TQDM_NROWS"] = str(worker_count + 1)
    os.environ["TQDM_MININTERVAL"] = "1.0"
    os.environ["TQDM_BAR_FORMAT"] = (
        f"[Worker {worker_position + 1}] "
        "{l_bar}{bar}{r_bar}"
    )

    from tqdm import tqdm
    tqdm.set_lock(progress_lock)

    import doc_preprocess

    global _WORKER_LABEL
    _WORKER_LABEL = f"Worker {worker_position + 1}"

    # Each process owns an independent model instance. Limit PyTorch's internal
    # CPU pool so multiple converter processes do not oversubscribe the machine.
    doc_preprocess.torch.set_num_threads(torch_threads)
    doc_preprocess.torch.set_num_interop_threads(1)
    doc_preprocess._DEVICE = device


def convert_one(pdf_path: str, output_dir: str) -> tuple[str, str, float]:
    """Convert one PDF and return its name, output path, and elapsed seconds."""
    import doc_preprocess

    pdf = Path(pdf_path)
    print(f"[{_WORKER_LABEL}] Starting: {pdf.name}", flush=True)
    started = time.monotonic()
    output = doc_preprocess.doc_preprocess(
        pdf_name=pdf.name,
        pdf_path=str(pdf.parent),
        md_path=output_dir,
    )
    return pdf.name, output, time.monotonic() - started


def convert_job(job: tuple[str, str]) -> tuple[str, str, float, str]:
    """Run one job without letting a bad PDF stop the remaining queue."""
    pdf_path, output_dir = job
    try:
        name, output, elapsed = convert_one(pdf_path, output_dir)
        return name, output, elapsed, ""
    except Exception as exc:
        return (
            Path(pdf_path).name,
            "",
            0.0,
            f"{type(exc).__name__}: {exc}",
        )


def terminate_executor(executor: ProcessPoolExecutor) -> None:
    """Terminate active workers, including on Python versions before 3.14."""
    process_map = getattr(executor, "_processes", None) or {}
    processes = list(process_map.values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    executor.shutdown(wait=True, cancel_futures=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all PDFs in a directory to Markdown in parallel."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="data/openreview_pdf",
        help="PDF directory (default: data/openreview_pdf)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="data/openreview_md",
        help="Markdown directory (default: data/openreview_md)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=2,
        help="Number of parallel conversion processes (default: 2)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Convert PDFs even when the corresponding Markdown file exists.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help=(
            "Inference device. With multiple workers, auto uses CPU to avoid "
            "unsafe multi-process accelerator sharing (default: auto)."
        ),
    )
    return parser.parse_args()


def resolve_device(requested: str, workers: int) -> tuple[str, str]:
    """Validate dependencies and select a process-safe inference device."""
    try:
        import doc_preprocess
    except ImportError as exc:
        raise RuntimeError(
            "PDF conversion dependencies are unavailable in this Python "
            "environment. Activate the project's `paper` Conda environment "
            "and try again."
        ) from exc

    torch = doc_preprocess.torch
    detected = doc_preprocess._DEVICE

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps was requested, but MPS is unavailable.")
    if requested == "mps" and workers > 1:
        raise RuntimeError(
            "MPS cannot safely run in multiple converter processes. Use "
            "`--device cpu` for parallel conversion, or `-j 1 --device mps`."
        )

    if requested != "auto":
        note = f"Using explicitly selected {requested.upper()} inference."
        if requested == "cuda" and workers > 1:
            note += " Each worker will load a separate model into GPU memory."
        return requested, note

    if workers > 1 and detected != "cpu":
        return (
            "cpu",
            f"Detected {detected.upper()}, but parallel accelerator processes "
            "are unsafe or memory-heavy; using CPU inference.",
        )
    return detected, f"Using auto-detected {detected.upper()} inference."


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.workers < 1:
        print("Error: --workers must be at least 1.", file=sys.stderr)
        return 2
    if not input_dir.is_dir():
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    all_pdfs = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not all_pdfs:
        print(f"No PDF files found in {input_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        pending = all_pdfs
    else:
        pending = [
            pdf for pdf in all_pdfs
            if not (
                (output_dir / pdf.with_suffix(".md").name).is_file()
                and (output_dir / pdf.with_suffix(".md").name).stat().st_size > 0
            )
        ]

    skipped = len(all_pdfs) - len(pending)
    effective_workers = min(args.workers, len(pending)) if pending else 0
    print(
        f"Found {len(all_pdfs)} PDFs; converting {len(pending)} with "
        f"{effective_workers} workers; skipping {skipped} nonempty outputs.",
        flush=True,
    )
    if not pending:
        return 0

    try:
        device, device_note = resolve_device(args.device, effective_workers)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    cpu_count = os.cpu_count() or effective_workers
    torch_threads = max(1, cpu_count // effective_workers)
    print(device_note, flush=True)
    if device == "cpu":
        print(
            f"Allowing {torch_threads} PyTorch CPU threads per worker.",
            flush=True,
        )

    failures: list[tuple[str, str]] = []
    completed = 0
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    progress_lock = context.RLock()
    worker_counter = context.Value("i", 0)
    executor: ProcessPoolExecutor | None = None
    try:
        executor = ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=context,
            initializer=initialize_worker,
            initargs=(
                device,
                torch_threads,
                progress_lock,
                worker_counter,
                effective_workers,
            ),
        )
        futures = {
            executor.submit(convert_job, (str(pdf), str(output_dir))): pdf
            for pdf in pending
        }
        for future in as_completed(futures):
            name, output, job_elapsed, error = future.result()
            completed += 1
            if error:
                failures.append((name, error))
                print(
                    f"[{completed}/{len(pending)}] FAILED: {name}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"[{completed}/{len(pending)}] Done: {name} "
                    f"({job_elapsed:.1f}s) -> {output}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nInterrupted; terminating workers.", file=sys.stderr, flush=True)
        if executor is not None:
            terminate_executor(executor)
        return 130
    except BrokenProcessPool as exc:
        print(
            f"\nA conversion worker exited unexpectedly: {exc}",
            file=sys.stderr,
            flush=True,
        )
        if executor is not None:
            terminate_executor(executor)
        return 1
    except BaseException:
        if executor is not None:
            terminate_executor(executor)
        raise
    else:
        assert executor is not None
        executor.shutdown(wait=True)

    elapsed = time.monotonic() - started
    succeeded = len(pending) - len(failures)
    print(
        f"Finished in {elapsed:.1f}s: {succeeded} succeeded, "
        f"{len(failures)} failed, {skipped} skipped.",
        flush=True,
    )
    if failures:
        print("Failures:", file=sys.stderr)
        for name, error in failures:
            print(f"  - {name}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
