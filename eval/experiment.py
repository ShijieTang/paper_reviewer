"""
experiment.py

Run two experimental conditions for each paper in papers.json and save
results for later quantitative comparison.

Condition A — Single agent:
    agents = [reviewer_a], n_iter = len(agents) = 1

Condition B — Multi-agent:
    agents = [reviewer_a, reviewer_b], n_iter = len(agents) = 2

Result files (in --output_dir):
    {timestamp}_nagent=1_niter=1_paper={name}_cond=A_single.txt
    {timestamp}_nagent=2_niter=2_paper={name}_cond=B_multi.txt

Each .txt file contains the raw result dict (identical structure to the
normal webapp output):
    { "reviewers": [...], "conference": {...}, "citations": {...} }

A summary JSON is also saved:
    experiment_summary_{timestamp}.json

Usage (run from the project root):
    python evaluation/experiment.py \\
        --json_file  evaluation/papers.json \\
        --api_key    YOUR_API_KEY            \\
        --output_dir evaluation/exp_results  \\
        [--paper_id  example_001]
"""

import argparse
import json
import os
import sys
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

_CONDITIONS_AGENTS = [
    ("A", "single", ["reviewer_a"]),
    ("B", "multi",  ["reviewer_a", "reviewer_b", "reviewer_c"]),
]

CONDITIONS = [
    {
        "id":     cid,
        "label":  label,
        "desc":   f"{'Single' if len(agents) == 1 else 'Multi'}-agent, {len(agents)} iteration{'s' if len(agents) > 1 else ''}",
        "agents": agents,
        "n_iter": len(agents),
    }
    for cid, label, agents in _CONDITIONS_AGENTS
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_papers(json_file: str) -> list:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [{"paper_id": pid, **meta} for pid, meta in data.items()]


def pdf_to_markdown(pdf_dir: str) -> str:
    """Load an existing markdown file when present, otherwise convert the PDF."""
    return load_or_create_markdown(pdf_dir, md_path="data/md")


def run_condition(paper_text: str, topic: str, cond: dict, api_key: str) -> dict:
    """Run one experimental condition and return the raw result dict."""
    return mas_main(
        paper=paper_text,
        topic=topic,
        n_iter=cond["n_iter"],
        reviewer_types=cond["agents"],
        api_key=api_key,
        run_citation_check=False,
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

def run_experiment(papers: list, api_key: str, output_dir: str) -> dict:
    """
    Run all conditions on all papers. Returns a summary dict for analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d%H%M")

    summary = {
        "timestamp": timestamp,
        "conditions": {c["id"]: {"desc": c["desc"], "agents": c["agents"],
                                  "n_iter": c["n_iter"]} for c in CONDITIONS},
        "papers": [],
    }

    for paper_meta in papers:
        paper_id   = paper_meta["paper_id"]
        paper_name = Path(paper_meta["paper_dir"]).stem
        topic      = normalize_topic(paper_meta.get("topic", ""))

        print(f"\n{'='*60}")
        print(f"Paper: {paper_id}  ({paper_name})")
        print(f"{'='*60}")

        existing_paths = {
            cond["id"]: _existing_result_path(output_dir, paper_name, cond)
            for cond in CONDITIONS
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
        for cond in CONDITIONS:
            print(f"\n--- Condition {cond['id']}: {cond['desc']} ---")
            existing_path = existing_paths[cond["id"]]
            reused_existing = False

            if existing_path is not None:
                result = _load_existing_result(existing_path)
                if result is not None:
                    out_path = str(existing_path)
                    reused_existing = True
                    print(f"Skipping: found existing result at {out_path}")
                else:
                    print(f"Existing result unreadable, rerunning: {existing_path}")
                    existing_path = None

            if existing_path is None:
                if paper_text is None:
                    print("Loading markdown or converting PDF...")
                    paper_text = pdf_to_markdown(paper_meta["paper_dir"])
                    print("Paper text ready.")

                result = run_condition(paper_text, topic, cond, api_key)
                out_path = save_result(result, paper_name, cond, output_dir, timestamp)
                print(f"Saved: {out_path}")

            paper_entry["conditions"][cond["id"]] = {
                "desc":            cond["desc"],
                "result_file":     os.path.basename(out_path),
                "reused_existing": reused_existing,
                "result":          result,
            }

        summary["papers"].append(paper_entry)

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
    args = parser.parse_args()

    papers = load_papers(args.json_file)
    if args.paper_id:
        papers = [p for p in papers if p["paper_id"] == args.paper_id]
        if not papers:
            print(f"Error: paper_id '{args.paper_id}' not found.")
            sys.exit(1)

    run_experiment(papers, args.api_key, args.output_dir)
    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
