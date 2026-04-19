"""
experiment_nopersona.py

Run 3 no-persona reviewers × 3 iterations for every conference paper in data/md/.

Result files (in --output_dir):
    paper={name}_niter=3_nagent=3_agenttype=NNN.txt

A summary JSON is also saved:
    experiment_nopersona_summary_{timestamp}.json

Usage (run from the project root):
    python eval/experiment_nopersona.py \\
        --api_key YOUR_API_KEY \\
        [--output_dir eval/exp_results] \\
        [--md_dir data/md] \\
        [--paper_id iclr_accept_001]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from mas_loop import main as mas_main


REVIEWER_TYPES = ["reviewer_nopersona", "reviewer_nopersona", "reviewer_nopersona"]
N_ITER = 3
N_AGENT = len(REVIEWER_TYPES)

_TYPE_CODE = {
    "reviewer_a":         "A",
    "reviewer_b":         "B",
    "reviewer_c":         "C",
    "reviewer_nopersona": "N",
}

AGENTTYPE = "".join(_TYPE_CODE.get(t, "?") for t in REVIEWER_TYPES)  # "NNN"

PAPER_PATTERN = re.compile(r"^(iclr|icml|neurips)_(accept|reject)_\d+$")


def _result_filename(paper_name: str) -> str:
    return (
        f"paper={paper_name}"
        f"_niter={N_ITER}"
        f"_nagent={N_AGENT}"
        f"_agenttype={AGENTTYPE}.txt"
    )


def load_papers(md_dir: str) -> list:
    papers = []
    for md in sorted(Path(md_dir).glob("*.md")):
        if PAPER_PATTERN.match(md.stem):
            papers.append({"paper_id": md.stem, "paper_dir": str(md)})
    return papers


def pdf_to_markdown(paper_dir: str) -> str:
    p = Path(paper_dir)
    if p.suffix.lower() == ".md" and p.exists():
        return p.read_text(encoding="utf-8")
    from doc_preprocess import load_or_create_markdown
    return load_or_create_markdown(paper_dir, md_path="data/md")


def run_experiment(papers: list, api_key: str, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d%H%M")

    summary = {
        "timestamp": timestamp,
        "condition": {
            "nagent":     N_AGENT,
            "niter":      N_ITER,
            "agenttype":  AGENTTYPE,
            "reviewers":  REVIEWER_TYPES,
        },
        "papers": [],
    }

    for paper_meta in papers:
        paper_id   = paper_meta["paper_id"]
        paper_name = Path(paper_meta["paper_dir"]).stem

        print(f"\n{'='*60}")
        print(f"Paper: {paper_name}")
        print(f"{'='*60}")

        out_path = os.path.join(output_dir, _result_filename(paper_name))
        reused = False

        if Path(out_path).exists():
            try:
                result = json.loads(Path(out_path).read_text(encoding="utf-8"))
                reused = True
                print(f"Skipping: found existing result at {out_path}")
            except (OSError, json.JSONDecodeError):
                print(f"Existing result unreadable, rerunning.")
                result = None
        else:
            result = None

        if result is None:
            print("Loading markdown...")
            paper_text = pdf_to_markdown(paper_meta["paper_dir"])
            print("Running 3 no-persona reviewers × 3 iterations...")
            result = mas_main(
                paper=paper_text,
                topic="",
                n_iter=N_ITER,
                reviewer_types=REVIEWER_TYPES,
                api_key=api_key,
                run_citation_check=False,
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Saved: {out_path}")

        summary["papers"].append({
            "paper_id":        paper_id,
            "paper_name":      paper_name,
            "result_file":     os.path.basename(out_path),
            "reused_existing": reused,
            "result":          result,
        })

    summary_path = os.path.join(
        output_dir, f"experiment_nopersona_summary_{timestamp}.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="3 no-persona agents × 3 iterations experiment.")
    parser.add_argument("--api_key",    required=True)
    parser.add_argument("--output_dir", default="eval/exp_results")
    parser.add_argument("--md_dir",     default="data/md")
    parser.add_argument("--paper_id",   default=None,
                        help="Optional: run only this paper (stem of .md file).")
    args = parser.parse_args()

    papers = load_papers(args.md_dir)
    if not papers:
        print(f"No conference papers found in {args.md_dir}")
        sys.exit(1)

    if args.paper_id:
        papers = [p for p in papers if p["paper_id"] == args.paper_id]
        if not papers:
            print(f"paper_id '{args.paper_id}' not found.")
            sys.exit(1)

    print(f"Found {len(papers)} paper(s). Running NNN × 3 iterations...")
    run_experiment(papers, args.api_key, args.output_dir)
    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
