#!/usr/bin/env python3
"""Build a small OpenAlex related-work dataset via a scraping proxy.

Selects a balanced sample of papers already sitting in data/md/ (N accept +
N reject per conference), extracts each paper's title, and queries
OpenAlexProvider (through a proxy, since OpenAlex is IP-rate-limited) for
related work. The output feeds a later step that fetches arXiv abstracts for
each related work entry.

Example:
    python scripts/build_openalex_related_dataset.py \
        --output eval/openalex_related_30.json \
        --proxy "http://customer-c302b6:053aa72c@proxy.ipipgo.com:31212"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.models import RelatedWorkQuery
from rag.providers.openalex import OpenAlexProvider
from rag.target_parser import summarize_target_paper

DEFAULT_MD_DIR = Path("data/md")
DEFAULT_CONFERENCES = ["iclr", "icml", "neurips"]
FILENAME_RE = re.compile(r"^(?P<conf>[a-z]+)_(?P<decision>accept|reject)_2025_(?P<num>\d+)_")


def select_papers_from_json(json_file: Path) -> list[dict]:
    """Load an existing paper-list JSON (paper_id -> {title, paper_dir, conference,
    accept_or_not/decision, ...}), e.g. eval/openreview_60_module_test.json."""
    data = json.loads(json_file.read_text(encoding="utf-8"))
    selected = []
    for paper_id, meta in data.items():
        path = Path(meta["paper_dir"])
        if not path.exists():
            raise SystemExit(f"paper_dir does not exist for {paper_id}: {path}")
        selected.append(
            {
                "paper_id": paper_id,
                "conference": (meta.get("conference") or "").upper(),
                "decision": meta.get("decision") or meta.get("accept_or_not") or "",
                "year": meta.get("year", 2025),
                "path": path,
            }
        )
    return selected


def select_papers(md_dir: Path, conferences: list[str], per_category: int) -> list[dict]:
    selected = []
    for conf in conferences:
        for decision in ("accept", "reject"):
            candidates = sorted(
                path
                for path in md_dir.glob(f"{conf}_{decision}_2025_*.md")
                if FILENAME_RE.match(path.name)
            )
            if len(candidates) < per_category:
                raise SystemExit(
                    f"Not enough {conf}/{decision} papers in {md_dir}: "
                    f"need {per_category}, found {len(candidates)}."
                )
            for path in candidates[:per_category]:
                match = FILENAME_RE.match(path.name)
                selected.append(
                    {
                        "conference": conf.upper(),
                        "decision": decision,
                        "paper_num": match.group("num"),
                        "path": path,
                    }
                )
    return selected


def build_proxy_opener(proxy: str) -> None:
    if proxy:
        os.environ["RAG_HTTP_PROXY"] = proxy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md-dir", type=Path, default=DEFAULT_MD_DIR)
    parser.add_argument("--conferences", nargs="+", default=DEFAULT_CONFERENCES)
    parser.add_argument("--per-category", type=int, default=5, help="accept papers and reject papers per conference")
    parser.add_argument("--json_file", type=Path, default=None,
                        help="Load the paper list from an existing JSON file (paper_id -> "
                             "{title, paper_dir, conference, accept_or_not, ...}) instead of "
                             "scanning --md-dir, e.g. eval/openreview_60_module_test.json.")
    parser.add_argument("--limit-per-paper", type=int, default=10, help="related works to fetch per paper")
    parser.add_argument("--proxy", default=os.environ.get("RAG_HTTP_PROXY", ""), help="proxy URL, e.g. http://user:pass@host:port")
    parser.add_argument("--output", type=Path, default=Path("eval/openalex_related_30.json"))
    args = parser.parse_args()

    build_proxy_opener(args.proxy)

    if args.json_file:
        papers = select_papers_from_json(args.json_file)
        print(f"Loaded {len(papers)} papers from {args.json_file}.")
    else:
        papers = select_papers(args.md_dir, args.conferences, args.per_category)
        print(f"Selected {len(papers)} papers "
              f"({args.per_category} accept + {args.per_category} reject per conference).")

    provider = OpenAlexProvider()
    dataset: dict[str, dict] = {}
    for entry in papers:
        path = entry["path"]
        paper_id = entry.get("paper_id") or path.stem
        markdown = path.read_text(encoding="utf-8", errors="replace")
        summary = summarize_target_paper(markdown)

        query = RelatedWorkQuery(group="related_work", query=summary.title)
        result = provider.search([query], limit=args.limit_per_paper)
        if result.warnings:
            for warning in result.warnings:
                print(f"[{paper_id}] warning: {warning}", file=sys.stderr)

        dataset[paper_id] = {
            "title": summary.title,
            "conference": entry["conference"],
            "year": entry.get("year", 2025),
            "decision": entry["decision"],
            "paper_dir": str(path),
            "related_works": [paper.to_dict() for paper in result.papers],
        }
        print(f"[{paper_id}] {entry['conference']} {entry['decision']}: "
              f"{len(result.papers)} related works ({result.status})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(dataset)} papers to {args.output}")


if __name__ == "__main__":
    main()
