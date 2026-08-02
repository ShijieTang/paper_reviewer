"""
merge_openreviewer.py

Parse the raw OpenReviewer .review.md files (written by openreviewer/generate_review.py,
one <paper_id>.review.md each) into the single {"papers": [...]} file eval/evaluation.py
expects (--openreviewer). Mirrors merge_paperreviewer.py, but also does the text parsing
that run_paperreviewer.py's poll step already did inline for PaperReviewer.ai — the
markdown files here were never run through a parser, so this script owns that step.

Usage:
    python eval/merge_openreviewer.py --input_dir eval/openreviewer_papers --output eval/openreviewer_300.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_INPUT_DIR = "eval/openreviewer_papers"
DEFAULT_OUTPUT = "eval/openreviewer.json"


# ── Review parser (same logic as run_openreviewer.py) ────────────────────────

def _extract_section(text: str, header: str) -> str:
    pattern = rf"##\s+{re.escape(header)}\s*\n(.*?)(?=\n##\s|\Z)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _extract_numeric(section_text: str) -> int | None:
    m = re.search(r"\b([1-9]\d*)\b", section_text)
    return int(m.group(1)) if m else None


def _split_bullets(text: str) -> list[str]:
    lines = [re.sub(r"^[-*]\s*", "", l).strip() for l in text.splitlines() if l.strip()]
    return [l for l in lines if l]


def parse_review(raw: str) -> dict:
    summary     = _extract_section(raw, "Summary")
    strengths   = _split_bullets(_extract_section(raw, "Strengths"))
    weaknesses  = _split_bullets(_extract_section(raw, "Weaknesses"))
    questions   = _extract_section(raw, "Questions")
    ethics_flag = _extract_section(raw, "Flag For Ethics Review")
    ethics_det  = _extract_section(raw, "Details Of Ethics Concerns")

    soundness    = _extract_numeric(_extract_section(raw, "Soundness"))
    presentation = _extract_numeric(_extract_section(raw, "Presentation"))
    contribution = _extract_numeric(_extract_section(raw, "Contribution"))
    rating       = _extract_numeric(_extract_section(raw, "Rating"))

    numeric_scores = [s for s in [soundness, presentation, contribution] if s is not None]
    avg_score_10 = round(sum(numeric_scores) / len(numeric_scores) * (10 / 4), 2) if numeric_scores else None

    return {
        "reviewer_id": "OpenReviewer",
        "summary": summary,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "questions": questions,
        "flag_for_ethics_review": ethics_flag,
        "details_of_ethics_concerns": ethics_det,
        "scores": {
            "soundness": soundness,
            "presentation": presentation,
            "contribution": contribution,
        },
        "rating": rating,
        "average_score_10": avg_score_10,
    }


# ── Entry builder ─────────────────────────────────────────────────────────────

def build_entry(paper_id: str, title: str, review: dict) -> dict:
    """score is OpenReviewer's own "Rating" (1-10 overall recommendation) —
    kept on the same ~1-10 scale as ground-truth ratings and PaperReviewer.ai's
    numerical_score, rather than the derived average_score_10. OpenReviewer
    never states an explicit accept/reject decision, so accept_or_not stays
    None instead of guessing a threshold."""
    entry = {"paper_id": paper_id, "title": title, "accept_or_not": None, "score": None, "reviews": [review]}
    if review["rating"] is not None:
        entry["score"] = review["rating"]
    elif review["average_score_10"] is not None:
        entry["score"] = review["average_score_10"]
    return entry


def main():
    parser = argparse.ArgumentParser(description="Parse and merge OpenReviewer .review.md files into one json.")
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    md_files = sorted(input_dir.glob("*.review.md"))
    if not md_files:
        print(f"No .review.md files found in {input_dir}.", file=sys.stderr)
        sys.exit(1)

    papers = []
    for md_path in md_files:
        paper_id = md_path.name[: -len(".review.md")]
        title = paper_id.replace("_", " ").title()
        raw = md_path.read_text(encoding="utf-8")
        review = parse_review(raw)
        papers.append(build_entry(paper_id, title, review))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"papers": papers}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Merged {len(papers)} papers into {output_path}")


if __name__ == "__main__":
    main()
