"""
experiment_advanced.py

Run five advanced conditions for each paper selected by papers.json.
This follows eval/experiment.py's experiment flow, but reads paper text
directly from existing Markdown files and supports DeepSeek or OpenRouter.

Conditions:
    C1 — No RAG, 1 iteration, 1 neutral reviewer
    C2 — RAG, 1 iteration, 1 neutral reviewer
    C3 — RAG, 3 iterations, 3 neutral reviewers + Author
    C4 — RAG, 3 iterations, 3 persona reviewers + Author
    C5 — No RAG, 3 iterations, 3 neutral reviewers + Author

The AI detector and citation checker are disabled in every condition.

Usage (run from the project root):
    export OPENROUTER_API_KEY="..."

    python eval/experiment_advanced.py \
        --json_file eval/papers.json \
        --md_dir data/md \
        --provider openrouter \
        --model deepseek/deepseek-chat-v3.1 \
        --api_key "$OPENROUTER_API_KEY" \
        --output_dir eval/advanced_exp_results_openrouter_v31
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import VALID_TOPICS
from mas_loop import main as mas_main


DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODELS = {
    "deepseek": "deepseek-v4-flash",
    "openrouter": "deepseek/deepseek-chat-v3.1",
}
ENABLE_AI_DETECTOR = False
RUN_CITATION_CHECK = False


# ── Condition definitions ─────────────────────────────────────────────────────

CONDITIONS = [
    {
        "id": "C1",
        "label": "no_rag_1iter_1neutral",
        "desc": "No RAG, 1 iteration, 1 neutral reviewer",
        "agents": ["reviewer_nopersona"],
        "n_iter": 1,
        "enable_rag": False,
        "agenttype": "N",
    },
    {
        "id": "C2",
        "label": "rag_1iter_1neutral",
        "desc": "RAG, 1 iteration, 1 neutral reviewer",
        "agents": ["reviewer_nopersona"],
        "n_iter": 1,
        "enable_rag": True,
        "agenttype": "N",
    },
    {
        "id": "C3",
        "label": "rag_3iter_3neutral",
        "desc": "RAG, 3 iterations, 3 neutral reviewers + Author",
        "agents": ["reviewer_nopersona"] * 3,
        "n_iter": 3,
        "enable_rag": True,
        "agenttype": "NNN",
    },
    {
        "id": "C4",
        "label": "rag_3iter_3persona",
        "desc": "RAG, 3 iterations, 3 persona reviewers + Author",
        "agents": ["reviewer_a", "reviewer_b", "reviewer_c"],
        "n_iter": 3,
        "enable_rag": True,
        "agenttype": "ABC",
    },
    {
        "id": "C5",
        "label": "no_rag_3iter_3neutral",
        "desc": "No RAG, 3 iterations, 3 neutral reviewers + Author",
        "agents": ["reviewer_nopersona"] * 3,
        "n_iter": 3,
        "enable_rag": False,
        "agenttype": "NNN",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_topic(topic: str) -> str:
    """Return the canonical topic name, or 'Others' when it is unknown."""
    for valid in VALID_TOPICS:
        if topic.strip().lower() == valid.lower():
            return valid
    return "Others"


def load_papers(json_file: str) -> list:
    """Load the paper selection and metadata exactly as experiment.py does."""
    with open(json_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [{"paper_id": paper_id, **metadata} for paper_id, metadata in data.items()]


def load_markdown(paper_dir: str, md_dir: str) -> str:
    """Read the cached Markdown matching the manifest's paper filename."""
    paper_name = Path(paper_dir).stem
    markdown_path = Path(md_dir) / f"{paper_name}.md"
    if not markdown_path.is_file():
        raise FileNotFoundError(
            f"Cached Markdown not found: {markdown_path}. "
            "This experiment does not convert PDFs."
        )
    return markdown_path.read_text(encoding="utf-8")


def run_condition(
    paper_text: str,
    topic: str,
    cond: dict,
    api_key: str,
    provider: str,
    model: str,
    rag_package: dict | None = None,
) -> dict:
    """Run one condition through the standard multi-agent pipeline."""
    return mas_main(
        paper=paper_text,
        topic=topic,
        n_iter=cond["n_iter"],
        reviewer_types=cond["agents"],
        api_key=api_key,
        provider=provider,
        model=model,
        run_citation_check=RUN_CITATION_CHECK,
        enable_ai_detector=ENABLE_AI_DETECTOR,
        enable_rag=cond["enable_rag"],
        precomputed_rag_package=rag_package,
    )


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "model"


def _result_metadata(
    paper_name: str,
    cond: dict,
    provider: str,
    model: str,
) -> str:
    """Return the standard filename metadata for one condition."""
    return (
        f"_nagent={len(cond['agents'])}"
        f"_niter={cond['n_iter']}"
        f"_agenttype={cond['agenttype']}"
        f"_rag={int(cond['enable_rag'])}"
        f"_styledetector={int(ENABLE_AI_DETECTOR)}"
        f"_provider={provider}"
        f"_model={_slug(model)}"
        f"_paper={paper_name}"
        f"_cond={cond['id']}_{cond['label']}.txt"
    )


def save_result(
    result: dict,
    paper_name: str,
    cond: dict,
    output_dir: str,
    timestamp: str,
    provider: str,
    model: str,
) -> str:
    """Save the raw mas_loop result, matching experiment.py's output format."""
    filename = (
        f"{timestamp}"
        f"{_result_metadata(paper_name, cond, provider, model)}"
    )
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return output_path


def _existing_result_path(
    output_dir: str,
    paper_name: str,
    cond: dict,
    provider: str,
    model: str,
) -> Path | None:
    """Return the latest matching result, following experiment.py's resume rule."""
    pattern = f"*{_result_metadata(paper_name, cond, provider, model)}"
    matches = sorted(Path(output_dir).glob(pattern))
    return matches[-1] if matches else None


def _load_existing_result(path: Path) -> dict | None:
    """Load an existing result, or return None when it is unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ── Main experiment loop ──────────────────────────────────────────────────────

def run_experiment(
    papers: list,
    api_key: str,
    output_dir: str,
    md_dir: str,
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
) -> dict:
    """Run all five conditions and return the comparative summary."""
    model = model or DEFAULT_MODELS[provider]
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d%H%M")

    summary = {
        "timestamp": timestamp,
        "provider": provider,
        "model": model,
        "conditions": {
            cond["id"]: {
                "desc": cond["desc"],
                "agents": cond["agents"],
                "n_iter": cond["n_iter"],
                "agenttype": cond["agenttype"],
                "enable_rag": cond["enable_rag"],
                "enable_ai_detector": ENABLE_AI_DETECTOR,
            }
            for cond in CONDITIONS
        },
        "papers": [],
    }

    for paper_meta in papers:
        paper_id = paper_meta["paper_id"]
        paper_name = Path(paper_meta["paper_dir"]).stem
        topic = normalize_topic(paper_meta.get("topic", ""))

        print(f"\n{'=' * 60}")
        print(f"Paper: {paper_id}  ({paper_name})")
        print(f"{'=' * 60}")

        existing_paths = {
            cond["id"]: _existing_result_path(
                output_dir,
                paper_name,
                cond,
                provider,
                model,
            )
            for cond in CONDITIONS
        }
        existing_results = {}
        shared_rag_package = None
        for cond in CONDITIONS:
            existing_path = existing_paths[cond["id"]]
            if existing_path is None:
                continue

            existing_result = _load_existing_result(existing_path)
            if existing_result is None:
                print(f"Existing result unreadable, rerunning: {existing_path}")
                continue

            existing_results[cond["id"]] = (existing_path, existing_result)
            if (
                cond["enable_rag"]
                and shared_rag_package is None
                and isinstance(existing_result.get("rag_package"), dict)
            ):
                shared_rag_package = existing_result["rag_package"]

        paper_entry = {
            "paper_id": paper_id,
            "paper_name": paper_name,
            "conference": paper_meta.get("conference", ""),
            "topic": topic,
            "ground_truth": {
                "accept_or_not": paper_meta.get("accept_or_not"),
                "score": paper_meta.get("score"),
                "strengths": paper_meta.get("strengths", []),
                "weaknesses": paper_meta.get("weaknesses", []),
                "summary": paper_meta.get("summary", ""),
            },
            "conditions": {},
        }

        paper_text = None
        for cond in CONDITIONS:
            print(f"\n--- Condition {cond['id']}: {cond['desc']} ---")
            existing = existing_results.get(cond["id"])
            reused_existing = existing is not None

            if existing is not None:
                existing_path, result = existing
                output_path = str(existing_path)
                print(f"Skipping: found existing result at {output_path}")
            else:
                if paper_text is None:
                    print("Loading cached Markdown...")
                    paper_text = load_markdown(paper_meta["paper_dir"], md_dir)
                    print("Paper text ready.")

                result = run_condition(
                    paper_text=paper_text,
                    topic=topic,
                    cond=cond,
                    api_key=api_key,
                    provider=provider,
                    model=model,
                    rag_package=(
                        shared_rag_package if cond["enable_rag"] else None
                    ),
                )
                output_path = save_result(
                    result=result,
                    paper_name=paper_name,
                    cond=cond,
                    output_dir=output_dir,
                    timestamp=timestamp,
                    provider=provider,
                    model=model,
                )
                print(f"Saved: {output_path}")

                if (
                    cond["enable_rag"]
                    and shared_rag_package is None
                    and isinstance(result.get("rag_package"), dict)
                ):
                    shared_rag_package = result["rag_package"]

            paper_entry["conditions"][cond["id"]] = {
                "desc": cond["desc"],
                "result_file": os.path.basename(output_path),
                "reused_existing": reused_existing,
                "result": result,
            }

        summary["papers"].append(paper_entry)

    summary_path = os.path.join(
        output_dir,
        f"experiment_advanced_summary_{timestamp}.json",
    )
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"\nExperiment summary saved: {summary_path}")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the five advanced paper-review conditions."
    )
    parser.add_argument(
        "--json_file",
        default="eval/papers.json",
        help="Path to the papers JSON file.",
    )
    parser.add_argument(
        "--md_dir",
        default="data/md",
        help="Directory containing existing Markdown files.",
    )
    parser.add_argument(
        "--api_key",
        required=True,
        help="API key for the selected provider.",
    )
    parser.add_argument(
        "--output_dir",
        default="experiment_artifacts/local/eval/advanced_exp_results",
        help="Directory for result files and the summary.",
    )
    parser.add_argument(
        "--paper_id",
        default=None,
        help="Optional exact paper_id from the JSON manifest.",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(DEFAULT_MODELS),
        default=DEFAULT_PROVIDER,
        help=f"API provider (default: {DEFAULT_PROVIDER}).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name. Defaults to the selected provider's configured "
            "model."
        ),
    )
    args = parser.parse_args()
    model = args.model or DEFAULT_MODELS[args.provider]

    papers = load_papers(args.json_file)
    if args.paper_id:
        papers = [paper for paper in papers if paper["paper_id"] == args.paper_id]
        if not papers:
            print(f"Error: paper_id '{args.paper_id}' not found.")
            sys.exit(1)

    print(f"Found {len(papers)} paper(s).")
    run_experiment(
        papers=papers,
        api_key=args.api_key,
        output_dir=args.output_dir,
        md_dir=args.md_dir,
        provider=args.provider,
        model=model,
    )
    print("\nAdvanced experiment complete.")


if __name__ == "__main__":
    main()
