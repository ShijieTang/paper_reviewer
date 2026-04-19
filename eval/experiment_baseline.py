"""
experiment_baseline.py

Baseline experiment: a single LLM call with no persona, no iteration.
The model receives the paper and a plain instruction prompt, then returns
one structured review in the same JSON format used by the main pipeline.

This acts as a control condition (Condition C) alongside:
    Condition A — single agent with persona, 1 iteration  (experiment.py)
    Condition B — multi-agent with personas, 3 iterations (experiment.py)

Result files (in --output_dir):
    {timestamp}_nagent=1_niter=1_paper={name}_cond=C_baseline.txt

A summary JSON is also saved:
    experiment_baseline_summary_{timestamp}.json

Usage (run from the project root):
    python eval/experiment_baseline.py \\
        --json_file  eval/papers.json  \\
        --api_key    YOUR_API_KEY       \\
        --output_dir eval/exp_results   \\
        [--paper_id  example_001]
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import VALID_TOPICS
from doc_preprocess import load_or_create_markdown


# ── Baseline prompt (no persona) ──────────────────────────────────────────────

BASELINE_SYSTEM_PROMPT = """\
You are a paper reviewer. Review the academic paper provided by the user.

Score each from 1–5:
  Novelty — Has this idea or approach already been done before?
  Soundness — Are the logic, assumptions, and mathematical reasoning correct?
  Significance — Does this result meaningfully advance the field?
  Evaluation — Are the experiments, data, and comparisons convincing and fair?
  Clarity — Is the paper clearly written and easy to understand?

Score interpretation:
  1 = very poor
  2 = weak
  3 = acceptable
  4 = strong
  5 = excellent

You must:
1. Provide a score for each criterion.
2. Provide AT LEAST three strengths of the paper.
3. Provide AT LEAST two weaknesses or concerns.
4. Provide a concise summary comment explaining your reasoning.
5. Provide a final decision: "Accept" or "Reject".

Output MUST be valid JSON in exactly the following format (no extra text):
{
  "reviewer": "Baseline Reviewer",
  "decision": "Accept or Reject",
  "scores": {
    "novelty": <integer 1-5>,
    "soundness": <integer 1-5>,
    "significance": <integer 1-5>,
    "evaluation": <integer 1-5>,
    "clarity": <integer 1-5>
  },
  "strengths": [
    "...",
    "...",
    "..."
  ],
  "weaknesses": [
    "...",
    "..."
  ],
  "summary_comment": "..."
}
"""

BASELINE_USER_PROMPT = "Here is the paper to review:\n\n{paper}\n\nNow produce your review."


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    try:
        from google.colab import userdata
        return userdata.get("API_KEY")
    except Exception:
        return os.environ.get("API_KEY", "")


def normalize_topic(topic: str) -> str:
    for valid in VALID_TOPICS:
        if topic.strip().lower() == valid.lower():
            return valid
    return "Others"


def _parse_json(text: str) -> dict:
    """Strip markdown fences and parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def load_papers(json_file: str) -> list:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [{"paper_id": pid, **meta} for pid, meta in data.items()]


def load_papers_from_md_dir(md_dir: str = "data/md") -> list:
    """Build a minimal paper list from all .md files in md_dir."""
    return [
        {
            "paper_id":     md.stem,
            "paper_dir":    str(md),
            "topic":        "",
            "conference":   "",
            "accept_or_not": None,
            "score":        None,
            "strengths":    [],
            "weaknesses":   [],
            "summary":      "",
        }
        for md in sorted(Path(md_dir).glob("*.md"))
    ]


def pdf_to_markdown(pdf_dir: str) -> str:
    return load_or_create_markdown(pdf_dir, md_path="data/md")


# ── Core baseline call ────────────────────────────────────────────────────────

def run_baseline_review(paper_text: str, api_key: str, model: str = "gpt-5") -> dict:
    """
    Single LLM call with no persona. Returns a result dict shaped like the
    main pipeline output so downstream evaluation code can consume it directly.
    """
    client = openai.OpenAI(
        api_key=api_key or _get_api_key(),
        base_url="https://ai-gateway.andrew.cmu.edu",
    )

    messages = [
        {"role": "developer", "content": BASELINE_SYSTEM_PROMPT},
        {"role": "user",      "content": BASELINE_USER_PROMPT.format(paper=paper_text)},
    ]

    print("[Baseline] Calling model (single shot, no persona)...")
    response = client.chat.completions.create(model=model, messages=messages)
    raw = response.choices[0].message.content.strip()
    print("[Baseline] Done.")

    review = _parse_json(raw)

    return {
        "reviewers": [review],
        "conference": {},
        "citations":  {},
    }


# ── File I/O ──────────────────────────────────────────────────────────────────

COND = {"id": "C", "label": "baseline"}


def _existing_result_path(output_dir: str, paper_name: str) -> Path | None:
    pattern = f"{paper_name}_baseline.txt"
    matches = sorted(Path(output_dir).glob(pattern))
    return matches[-1] if matches else None


def save_result(result: dict, paper_name: str, output_dir: str) -> str:
    fname = f"{paper_name}_baseline.txt"
    out_path = os.path.join(output_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return out_path


# ── Main experiment loop ──────────────────────────────────────────────────────

def run_experiment(papers: list, api_key: str, output_dir: str,
                   model: str = "gpt-5") -> dict:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d%H%M")

    summary = {
        "timestamp":  timestamp,
        "condition":  {"id": "C", "label": "baseline", "desc": "No-persona, single-shot"},
        "model":      model,
        "papers":     [],
    }

    for paper_meta in papers:
        paper_id   = paper_meta["paper_id"]
        paper_name = Path(paper_meta["paper_dir"]).stem
        topic      = normalize_topic(paper_meta.get("topic", ""))

        print(f"\n{'='*60}")
        print(f"Paper: {paper_id}  ({paper_name})")
        print(f"{'='*60}")

        existing_path = _existing_result_path(output_dir, paper_name)
        reused = False

        if existing_path is not None:
            try:
                result   = json.loads(existing_path.read_text(encoding="utf-8"))
                out_path = str(existing_path)
                reused   = True
                print(f"Skipping: found existing result at {out_path}")
            except (OSError, json.JSONDecodeError):
                print(f"Existing result unreadable, rerunning: {existing_path}")
                existing_path = None

        if existing_path is None:
            print("Loading markdown or converting PDF...")
            paper_text = pdf_to_markdown(paper_meta["paper_dir"])
            print("Paper text ready.")
            result   = run_baseline_review(paper_text, api_key, model)
            out_path = save_result(result, paper_name, output_dir)
            print(f"Saved: {out_path}")

        summary["papers"].append({
            "paper_id":       paper_id,
            "paper_name":     paper_name,
            "conference":     paper_meta.get("conference", ""),
            "topic":          topic,
            "ground_truth": {
                "accept_or_not": paper_meta.get("accept_or_not"),
                "score":         paper_meta.get("score"),
                "strengths":     paper_meta.get("strengths", []),
                "weaknesses":    paper_meta.get("weaknesses", []),
                "summary":       paper_meta.get("summary", ""),
            },
            "result_file":    os.path.basename(out_path),
            "reused_existing": reused,
            "result":          result,
        })

    summary_path = os.path.join(output_dir, f"experiment_baseline_summary_{timestamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nBaseline experiment summary saved: {summary_path}")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baseline experiment: no-persona, single-shot paper review.")
    parser.add_argument("--json_file",  default="eval/papers.json",
                        help="Path to papers JSON file.")
    parser.add_argument("--api_key",    required=True,
                        help="API key for the LLM gateway.")
    parser.add_argument("--output_dir", default="eval/exp_baseline_results",
                        help="Directory to save result files.")
    parser.add_argument("--paper_id",   default=None,
                        help="Optional: run only this paper_id.")
    parser.add_argument("--model",      default="gpt-5",
                        help="LLM model name (default: gpt-5).")
    parser.add_argument("--all_md",     action="store_true",
                        help="Run on all .md files in --md_dir, skipping existing results.")
    parser.add_argument("--md_dir",     default="data/md",
                        help="Directory to scan when --all_md is set (default: data/md).")
    args = parser.parse_args()

    if args.all_md:
        papers = load_papers_from_md_dir(args.md_dir)
        if not papers:
            print(f"No .md files found in {args.md_dir}")
            sys.exit(1)
        print(f"Found {len(papers)} markdown file(s) in {args.md_dir}")
    else:
        papers = load_papers(args.json_file)
        if args.paper_id:
            papers = [p for p in papers if p["paper_id"] == args.paper_id]
            if not papers:
                print(f"Error: paper_id '{args.paper_id}' not found.")
                sys.exit(1)

    run_experiment(papers, args.api_key, args.output_dir, args.model)
    print("\nBaseline experiment complete.")


if __name__ == "__main__":
    main()
