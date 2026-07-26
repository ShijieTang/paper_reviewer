"""
experiment.py

Run five experimental conditions for each paper in papers.json and save
results for later quantitative comparison.

Condition 1 — No RAG, 1 iteration, 1 reviewer
Condition 2 — RAG,    1 iteration, 1 reviewer
Condition 3 — No RAG, 2 iterations, 1 reviewer, AI Detector, no author rebuttal
Condition 4 — No RAG, 3 iterations, 1 reviewer, author rebuttal, no AI Detector
Condition 5 — No RAG, 1 iteration, 3 reviewers
Condition 6 — No RAG, 2 iterations, 1 reviewer, no AI Detector, no author rebuttal

Conditions 3 and 6 differ only in the AI Detector, so (3 - 6) measures the
detector and (6 - 1) measures the extra iteration on its own. Note that with
n_iter=1 the rebuttal loop never runs, so enable_author_rebuttal has no effect
in conditions 1, 2 and 5.

Result files (in --output_dir):
    {timestamp}_nagent=1_niter=1_paper={name}_cond=1_no_rag_1iter_1rev.txt
    ...

Each .txt file contains the raw result dict (identical structure to the
normal webapp output):
    { "reviewers": [...], "conference": {...}, "citations": {...} }

A summary JSON is also saved:
    experiment_summary_{timestamp}.json

Usage (run from the project root):
    python eval/experiment.py \\
        --json_file  eval/openreview_60_module_test.json \\
        --api_key    YOUR_API_KEY                        \\
        --output_dir eval/exp_results                    \\
        [--paper_id  example_001]                         \\
        [--conditions 1,2,3,4,5]                          \\
        [--concurrency 5]

Papers are reviewed in parallel (default 5 at a time); --concurrency 1 restores
sequential execution. Results already present in --output_dir are reused, so an
interrupted run can simply be restarted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import VALID_TOPICS
from doc_preprocess import load_or_create_markdown
from mas_loop import main as mas_main


def normalize_topic(topic: str) -> str:
    """Return the canonical topic name, or 'Others' if not in the valid list.
    Matching is case-insensitive so e.g. 'Deep learning' → 'Deep Learning'."""
    for valid in VALID_TOPICS:
        if topic.strip().lower() == valid.lower():
            return valid
    return "Others"


# ── Condition definitions ─────────────────────────────────────────────────────
_N = ["reviewer_nopersona"]

_CONDITIONS_SPEC = [
    dict(id="1", label="no_rag_1iter_1rev",
         desc="No RAG, 1 iteration, 1 neutral reviewer",
         agents=_N, n_iter=1),
    dict(id="2", label="rag_1iter_1rev",
         desc="Related-work RAG, 1 iteration, 1 neutral reviewer",
         agents=_N, n_iter=1, enable_rag=True),
    dict(id="3", label="no_rag_2iter_aidetect_noauthor",
         desc="No RAG, 2 iterations, 1 neutral reviewer, AI Detector enabled, no author rebuttal",
         agents=_N, n_iter=2, enable_ai_detector=True, enable_author_rebuttal=False),
    dict(id="4", label="no_rag_3iter_author",
         desc="No RAG, 3 iterations, 1 neutral reviewer, author rebuttal, no AI Detector",
         agents=_N, n_iter=3),
    dict(id="5", label="no_rag_1iter_3rev",
         desc="No RAG, 1 iteration, 3 persona reviewers",
         agents=["reviewer_a", "reviewer_b", "reviewer_c"], n_iter=1),
    # Control for condition 3: identical except the AI Detector is off, so
    # (3 - 6) isolates the detector and (6 - 1) isolates the extra iteration.
    dict(id="6", label="no_rag_2iter_noaidetect_noauthor",
         desc="No RAG, 2 iterations, 1 neutral reviewer, no AI Detector, no author rebuttal",
         agents=_N, n_iter=2, enable_author_rebuttal=False),
]

_CONDITION_DEFAULTS = {
    "enable_rag":             False,
    "enable_ai_detector":     False,
    "enable_author_rebuttal": True,
}

_TYPE_CODE = {
    "reviewer_a":         "A",
    "reviewer_b":         "B",
    "reviewer_c":         "C",
    "reviewer_nopersona": "N",
}


def agenttype_code(agents: list) -> str:
    """Compact agent-type code, same convention as exp_results (e.g. 'N', 'ABC')."""
    return "".join(_TYPE_CODE.get(a, "?") for a in agents)


CONDITIONS = [
    {**_CONDITION_DEFAULTS, **spec, "agenttype": agenttype_code(spec["agents"])}
    for spec in _CONDITIONS_SPEC
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_papers(json_file: str) -> list:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [{"paper_id": pid, **meta} for pid, meta in data.items()]


def pdf_to_markdown(pdf_dir: str) -> str:
    """Load an existing markdown file when present, otherwise convert the PDF."""
    p = Path(pdf_dir)
    if p.suffix.lower() == ".md" and p.exists():
        return p.read_text(encoding="utf-8")
    return load_or_create_markdown(pdf_dir, md_path="data/md")


def run_condition(paper_text: str, topic: str, cond: dict, api_key: str,
                  model: str = "", rag_config: dict | None = None) -> dict:
    """Run one experimental condition and return the raw result dict."""
    rag_config = dict(rag_config or {})

    return mas_main(
        paper=paper_text,
        topic=topic,
        n_iter=cond["n_iter"],
        reviewer_types=cond["agents"],
        api_key=api_key,
        model=model,
        run_citation_check=False,
        enable_rag=cond.get("enable_rag", False),
        enable_ai_detector=cond.get("enable_ai_detector", False),
        enable_author_rebuttal=cond.get("enable_author_rebuttal", True),
        rag_config=rag_config,
    )


def save_result(result: dict, paper_name: str, cond: dict,
                output_dir: str, timestamp: str) -> str:
    """
    Save result in the same format as the normal webapp workflow.
    Filename encodes all metadata needed for later analysis.
    """
    fname = (
        f"{timestamp}"
        f"_nagent={len(cond['agents'])}"
        f"_niter={cond['n_iter']}"
        f"_paper={paper_name}"
        f"_cond={cond['id']}_{cond['label']}.txt"
    )
    out_path = os.path.join(output_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return out_path


def _existing_result_path(output_dir: str, paper_name: str, cond: dict) -> Path | None:
    """Return the latest saved result file for this paper/condition, if any."""
    pattern = (
        f"*_nagent={len(cond['agents'])}"
        f"_niter={cond['n_iter']}"
        f"_paper={paper_name}"
        f"_cond={cond['id']}_{cond['label']}.txt"
    )
    matches = sorted(Path(output_dir).glob(pattern))
    return matches[-1] if matches else None


def _load_existing_result(path: Path) -> dict | None:
    """Load a previously saved result file. Return None if unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ── Main experiment loop ──────────────────────────────────────────────────────

DEFAULT_CONCURRENCY = 5

# stdout is shared by every worker, so progress lines are emitted under a lock.
# The per-agent chatter printed from inside mas_loop still interleaves; the
# "[paper_id]" prefix on these lines is what makes a run traceable.
_print_lock = threading.Lock()


def _log(paper_id: str, message: str) -> None:
    with _print_lock:
        print(f"[{paper_id}] {message}", flush=True)


def _run_paper(paper_meta: dict, conditions: list, api_key: str,
               output_dir: str, timestamp: str, model: str,
               rag_config: dict | None = None) -> dict:
    """Run every condition for one paper and return its summary entry."""
    paper_id   = paper_meta["paper_id"]
    paper_name = Path(paper_meta["paper_dir"]).stem
    topic      = normalize_topic(paper_meta.get("topic", ""))

    _log(paper_id, f"Start ({paper_name})")

    existing_paths = {
        cond["id"]: _existing_result_path(output_dir, paper_name, cond)
        for cond in conditions
    }

    paper_entry = {
        "paper_id":       paper_id,
        "paper_name":     paper_name,
        "conference":     paper_meta.get("conference", ""),
        "topic":          topic,
        "ground_truth": {
            "accept_or_not":   paper_meta.get("accept_or_not"),
            "score":           paper_meta.get("score"),
            "strengths":       paper_meta.get("strengths", []),
            "weaknesses":      paper_meta.get("weaknesses", []),
            "summary":         paper_meta.get("summary", ""),
        },
        "conditions": {},
    }

    paper_text = None
    for cond in conditions:
        existing_path = existing_paths[cond["id"]]
        reused_existing = False

        if existing_path is not None:
            result = _load_existing_result(existing_path)
            if result is not None:
                out_path = str(existing_path)
                reused_existing = True
                _log(paper_id, f"Condition {cond['id']}: skipping, found {existing_path.name}")
            else:
                _log(paper_id, f"Condition {cond['id']}: existing result unreadable, rerunning")
                existing_path = None

        if existing_path is None:
            if paper_text is None:
                paper_text = pdf_to_markdown(paper_meta["paper_dir"])
            _log(paper_id, f"Condition {cond['id']}: {cond['desc']}")
            result = run_condition(paper_text, topic, cond, api_key, model=model,
                                   rag_config=rag_config)
            out_path = save_result(result, paper_name, cond, output_dir, timestamp)
            _log(paper_id, f"Condition {cond['id']}: saved {os.path.basename(out_path)}")

        paper_entry["conditions"][cond["id"]] = {
            "desc":            cond["desc"],
            "agents":          cond["agents"],
            "agenttype":       cond.get("agenttype", agenttype_code(cond["agents"])),
            "n_iter":          cond["n_iter"],
            "result_file":     os.path.basename(out_path),
            "reused_existing": reused_existing,
            "result":          result,
        }

    _log(paper_id, "Done")
    return paper_entry


def run_experiment(papers: list, api_key: str, output_dir: str,
                   conditions: list = None, model: str = "",
                   concurrency: int = DEFAULT_CONCURRENCY,
                   rag_config: dict | None = None) -> dict:
    """
    Run the given conditions (default: all of CONDITIONS) on all papers.

    Papers are processed by up to `concurrency` worker threads; the conditions
    within one paper still run in order, so they share its loaded markdown.
    Pass concurrency=1 to run everything sequentially.

    Returns a summary dict for analysis.
    """
    conditions = conditions if conditions is not None else CONDITIONS
    concurrency = max(1, concurrency)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d%H%M")

    summary = {
        "timestamp": timestamp,
        "model": model,
        "concurrency": concurrency,
        "rag_config": rag_config or {},
        "conditions": {c["id"]: {"desc": c["desc"], "agents": c["agents"],
                                  "agenttype": c.get("agenttype", agenttype_code(c["agents"])),
                                  "n_iter": c["n_iter"],
                                  "enable_rag": c.get("enable_rag", False),
                                  "enable_ai_detector": c.get("enable_ai_detector", False),
                                  "enable_author_rebuttal": c.get("enable_author_rebuttal", True)}
                       for c in conditions},
        "papers": [],
    }

    print(f"Running {len(papers)} paper(s) x {len(conditions)} condition(s) "
          f"with concurrency={concurrency}", flush=True)

    # Results are collected by index so the summary keeps the input paper order
    # regardless of the order workers happen to finish in.
    entries: dict[int, dict] = {}
    failures: list[tuple[str, Exception]] = []

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_run_paper, paper_meta, conditions, api_key,
                        output_dir, timestamp, model,
                        rag_config): (i, paper_meta["paper_id"])
            for i, paper_meta in enumerate(papers)
        }
        for future in as_completed(futures):
            i, paper_id = futures[future]
            try:
                entries[i] = future.result()
            except Exception as exc:      # keep the batch going; report at the end
                failures.append((paper_id, exc))
                _log(paper_id, f"FAILED: {type(exc).__name__}: {exc}")

    summary["papers"] = [entries[i] for i in sorted(entries)]

    if failures:
        print(f"\n{len(failures)} paper(s) failed:", flush=True)
        for paper_id, exc in failures:
            print(f"  {paper_id}: {type(exc).__name__}: {exc}", flush=True)

    # Save summary JSON for easy quantitative comparison
    summary_path = os.path.join(output_dir, f"experiment_summary_{timestamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nExperiment summary saved: {summary_path}")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run single-agent vs multi-agent experiment.")
    parser.add_argument("--json_file",  default="evaluation/papers.json",
                        help="Path to the papers JSON file.")
    parser.add_argument("--api_key",    required=True,
                        help="API key for the LLM gateway.")
    parser.add_argument("--output_dir", default="evaluation/exp_results",
                        help="Directory to save all result files.")
    parser.add_argument("--paper_id",   default=None,
                        help="Optional: run only this paper_id.")
    parser.add_argument("--conditions", default=None,
                        help="Comma-separated condition ids to run, e.g. 'A,C,D1,D2' "
                             f"(default: all of {[c['id'] for c in CONDITIONS]}).")
    parser.add_argument("--model", default="",
                        help="Model name to pass through to mas_loop (e.g. "
                             "'openai/gpt-4o-mini-2024-07-18' via OpenRouter). "
                             "Default: provider's default model.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"Number of papers reviewed in parallel "
                             f"(default: {DEFAULT_CONCURRENCY}; use 1 to run sequentially).")
    args = parser.parse_args()

    if args.concurrency < 1:
        print("Error: --concurrency must be at least 1.")
        sys.exit(1)

    papers = load_papers(args.json_file)
    if args.paper_id:
        papers = [p for p in papers if p["paper_id"] == args.paper_id]
        if not papers:
            print(f"Error: paper_id '{args.paper_id}' not found.")
            sys.exit(1)

    conditions = CONDITIONS
    if args.conditions:
        wanted = {c.strip() for c in args.conditions.split(",")}
        conditions = [c for c in CONDITIONS if c["id"] in wanted]
        missing = wanted - {c["id"] for c in conditions}
        if missing:
            print(f"Error: unknown condition id(s): {sorted(missing)}")
            sys.exit(1)

    run_experiment(papers, args.api_key, args.output_dir, conditions=conditions,
                   model=args.model, concurrency=args.concurrency)
    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
