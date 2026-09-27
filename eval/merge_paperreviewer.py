"""
merge_paperreviewer.py

Combine the per-paper output files written by run_paperreviewer.py
(--output_dir, one <paper_id>.json each) into the single
{"papers": [...]} file eval/evaluation.py expects (--paperreviewer).

Usage:
    python eval/merge_paperreviewer.py --input_dir eval/paperreviewer_papers --output eval/paperreviewer_321.json
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_INPUT_DIR = "eval/paperreviewer_papers"
DEFAULT_OUTPUT = "eval/paperreviewer.json"


def main():
    parser = argparse.ArgumentParser(description="Merge per-paper paperreview.ai outputs into one json.")
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    paper_files = sorted(input_dir.glob("*.json"))
    if not paper_files:
        print(f"No per-paper json files found in {input_dir}.", file=sys.stderr)
        sys.exit(1)

    papers = [json.loads(p.read_text(encoding="utf-8")) for p in paper_files]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"papers": papers}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Merged {len(papers)} papers into {output_path}")


if __name__ == "__main__":
    main()
