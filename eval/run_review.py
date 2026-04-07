"""
run_review.py

Load paper metadata from a JSON file, run the full pipeline (PDF → markdown →
review), and save results in the same format as the normal webapp workflow.

Usage (run from the project root):
    python evaluation/run_review.py \\
        --json_file  evaluation/papers.json \\
        --api_key    YOUR_API_KEY \\
        --agents     reviewer_a            \\
        --output_dir results               \\
        [--paper_id  example_001]          \\
        [--n_iter    1]                    \\
        [--topic     "NLP"]

--agents accepts a comma-separated list, e.g. "reviewer_a,reviewer_b" or "both".
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from doc_preprocess import doc_preprocess
from mas_loop import main as mas_main


def load_papers(json_file: str) -> list:
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)["papers"]


def pdf_to_markdown(pdf_dir: str) -> str:
    """
    Convert a PDF to markdown (same as normal workflow).
    pdf_dir is the path to the PDF, e.g. "data/pdf/example_paper.pdf".
    Returns the markdown text.
    """
    pdf_path = Path(pdf_dir)
    md_out = doc_preprocess(
        pdf_name=pdf_path.name,
        pdf_path=str(pdf_path.parent),
        md_path="data/md",
    )
    return Path(md_out).read_text(encoding="utf-8")


def parse_agents(agents_arg: str) -> list:
    if agents_arg.strip().lower() == "both":
        return ["reviewer_a", "reviewer_b"]
    return [a.strip() for a in agents_arg.split(",")]


def run_paper(paper_meta: dict, agents: list, api_key: str,
              output_dir: str, n_iter: int, topic_override: str = None) -> dict:
    """
    Run the full pipeline for one paper and save results.
    Result structure matches the normal webapp output exactly.
    """
    paper_id   = paper_meta["paper_id"]
    topic      = topic_override or paper_meta.get("topic", "")
    paper_name = Path(paper_meta["paper_dir"]).stem

    print(f"\n{'='*60}")
    print(f"Paper : {paper_id}  ({paper_name})")
    print(f"Agents: {agents}  |  Iterations: {n_iter}")
    print(f"{'='*60}")

    # Step 1: PDF → markdown (skip manual correction)
    print("Converting PDF to markdown...")
    paper_text = pdf_to_markdown(paper_meta["paper_dir"])
    print("Conversion complete.")

    # Step 2: Run review loop (includes citation check)
    result = mas_main(
        paper=paper_text,
        topic=topic,
        n_iter=n_iter,
        reviewer_types=agents,
        api_key=api_key,
    )

    # Step 3: Save — same naming convention as webapp
    os.makedirs(output_dir, exist_ok=True)
    timestamp  = datetime.now().strftime("%y%m%d%H%M")
    fname = (
        f"{timestamp}"
        f"_nagent={len(agents)}"
        f"_niter={n_iter}"
        f"_paper={paper_name}.txt"
    )
    out_path = os.path.join(output_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved: {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run the paper reviewer from a JSON metadata file.")
    parser.add_argument("--json_file",  default="evaluation/papers.json",
                        help="Path to the papers JSON file.")
    parser.add_argument("--api_key",    required=True,
                        help="API key for the LLM gateway.")
    parser.add_argument("--agents",     default="reviewer_a",
                        help='Reviewer agent(s): "reviewer_a", "reviewer_b", '
                             '"reviewer_a,reviewer_b", or "both".')
    parser.add_argument("--output_dir", default="results",
                        help="Directory to save review results.")
    parser.add_argument("--paper_id",   default=None,
                        help="Optional: run only this paper_id.")
    parser.add_argument("--n_iter",     type=int, default=1,
                        help="Number of review iterations (1 = no rebuttal).")
    parser.add_argument("--topic",      default=None,
                        help="Override the paper topic from JSON.")
    args = parser.parse_args()

    papers = load_papers(args.json_file)
    if args.paper_id:
        papers = [p for p in papers if p["paper_id"] == args.paper_id]
        if not papers:
            print(f"Error: paper_id '{args.paper_id}' not found.")
            sys.exit(1)

    agents = parse_agents(args.agents)

    for paper_meta in papers:
        run_paper(
            paper_meta=paper_meta,
            agents=agents,
            api_key=args.api_key,
            output_dir=args.output_dir,
            n_iter=args.n_iter,
            topic_override=args.topic,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
