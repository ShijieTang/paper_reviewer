"""
run_openreviewer.py

Read markdown files from data/md/, send each to the OpenReviewer Hugging Face
Space (https://huggingface.co/spaces/maxidl/openreviewer), parse the returned
review, and append results to eval/openreviewer.json.

Usage (run from the project root):
    python eval/run_openreviewer.py [--md_dir data/md] [--output eval/openreviewer.json]
    python eval/run_openreviewer.py --paper iclr_accept_001.md

The average_score_10 field is computed as:
    (soundness + presentation + contribution) / 3  * (10 / 4)
since each of the three scores is on a 1-4 scale.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from gradio_client import Client

# ── Constants ─────────────────────────────────────────────────────────────────

HF_SPACE = "maxidl/openreviewer"
DEFAULT_MD_DIR = "data/md"
DEFAULT_OUTPUT = "eval/openreviewer.json"
RAW_OUTPUT_DIR = Path("results/openreviewer")

# ── OpenReviewer client ───────────────────────────────────────────────────────

def call_openreviewer(paper_text: str) -> str:
    """Send paper markdown to the OpenReviewer HF Space and return raw review text."""
    client = Client(HF_SPACE)
    result = client.predict(paper_text, api_name="/generate")
    if isinstance(result, (list, tuple)):
        return result[0]
    return result


# ── Review parser ─────────────────────────────────────────────────────────────

def _extract_section(text: str, header: str) -> str:
    """Extract the content under a markdown ## header."""
    pattern = rf"##\s+{re.escape(header)}\s*\n(.*?)(?=\n##\s|\Z)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _extract_numeric(section_text: str) -> int | None:
    """Pull the first integer from a section (used for scored fields)."""
    m = re.search(r"\b([1-9]\d*)\b", section_text)
    return int(m.group(1)) if m else None


def _split_bullets(text: str) -> list[str]:
    """Turn a bullet-list block into a Python list of strings."""
    lines = [re.sub(r"^[-*]\s*", "", l).strip() for l in text.splitlines() if l.strip()]
    return [l for l in lines if l]


def parse_review(raw: str) -> dict:
    """Parse the structured OpenReviewer output into a dict."""
    summary     = _extract_section(raw, "Summary")
    strengths   = _split_bullets(_extract_section(raw, "Strengths"))
    weaknesses  = _split_bullets(_extract_section(raw, "Weaknesses"))
    questions   = _extract_section(raw, "Questions")
    ethics_flag = _extract_section(raw, "Flag For Ethics Review")
    ethics_det  = _extract_section(raw, "Details Of Ethics Concerns")

    soundness_raw    = _extract_section(raw, "Soundness")
    presentation_raw = _extract_section(raw, "Presentation")
    contribution_raw = _extract_section(raw, "Contribution")
    rating_raw       = _extract_section(raw, "Rating")

    soundness    = _extract_numeric(soundness_raw)
    presentation = _extract_numeric(presentation_raw)
    contribution = _extract_numeric(contribution_raw)
    rating       = _extract_numeric(rating_raw)

    # Average of the three 1-4 dimension scores, scaled to a 10-point scale
    numeric_scores = [s for s in [soundness, presentation, contribution] if s is not None]
    if numeric_scores:
        avg_score_10 = round(sum(numeric_scores) / len(numeric_scores) * (10 / 4), 2)
    else:
        avg_score_10 = None

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


# ── JSON helpers ──────────────────────────────────────────────────────────────

def load_output(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"papers": []}


def save_output(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def find_or_create_paper(data: dict, paper_id: str, title: str) -> dict:
    """Return the existing paper entry or insert a new one."""
    for p in data["papers"]:
        if p["paper_id"] == paper_id:
            return p
    entry = {"paper_id": paper_id, "title": title, "reviews": []}
    data["papers"].append(entry)
    return entry


# ── Per-paper pipeline ────────────────────────────────────────────────────────

def already_reviewed(paper_id: str, data: dict) -> bool:
    for p in data["papers"]:
        if p["paper_id"] == paper_id:
            return any(r.get("reviewer_id") == "OpenReviewer" for r in p.get("reviews", []))
    return False


def review_paper(md_path: Path, data: dict) -> None:
    paper_id = md_path.stem
    title    = paper_id.replace("_", " ").title()
    print(f"\n{'='*60}")
    print(f"Paper : {paper_id}")
    print(f"{'='*60}")

    if already_reviewed(paper_id, data):
        print("  Already reviewed — skipping.")
        return

    paper_text = md_path.read_text(encoding="utf-8")

    print("Calling OpenReviewer...")
    raw_review = call_openreviewer(paper_text)
    print("Response received. Saving raw output...")

    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_OUTPUT_DIR / f"{paper_id}.txt"
    raw_path.write_text(raw_review, encoding="utf-8")
    print(f"  Raw output : {raw_path}")

    print("Parsing...")

    review = parse_review(raw_review)

    entry = find_or_create_paper(data, paper_id, title)

    # Replace any existing OpenReviewer review for this paper
    entry["reviews"] = [r for r in entry["reviews"] if r.get("reviewer_id") != "OpenReviewer"]
    entry["reviews"].append(review)

    print(f"  Soundness   : {review['scores']['soundness']}/4")
    print(f"  Presentation: {review['scores']['presentation']}/4")
    print(f"  Contribution: {review['scores']['contribution']}/4")
    print(f"  Rating      : {review['rating']}/10")
    print(f"  Avg score   : {review['average_score_10']}/10")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run OpenReviewer on markdown papers.")
    parser.add_argument(
        "--md_dir", default=DEFAULT_MD_DIR,
        help="Directory containing .md paper files (default: data/md).",
    )
    parser.add_argument(
        "--paper", default=None,
        help="Optional: filename (or stem) of a single .md file to review.",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Path to the output JSON file (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()

    md_dir     = Path(args.md_dir)
    output_path = Path(args.output)

    if not md_dir.exists():
        print(f"Error: markdown directory '{md_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if args.paper:
        target = Path(args.paper)
        if not target.suffix:
            target = target.with_suffix(".md")
        if not target.is_absolute():
            target = md_dir / target.name
        md_files = [target]
    else:
        md_files = sorted(md_dir.glob("*.md"))

    if not md_files:
        print("No .md files found.", file=sys.stderr)
        sys.exit(1)

    data = load_output(output_path)

    for md_path in md_files:
        review_paper(md_path, data)
        save_output(data, output_path)  # save after each paper so progress isn't lost
        print(f"Appended to {output_path}")

    print(f"\nDone. Results saved to {output_path}")


if __name__ == "__main__":
    main()
