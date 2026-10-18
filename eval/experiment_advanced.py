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
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import VALID_TOPICS
from mas_loop import main as mas_main
from review_schema import review_is_valid, workflow_is_complete


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


def load_papers(json_file: str, output_dir: str | None = None) -> list:
    """Load the paper selection and metadata, falling back to repeat artifacts when needed."""
    json_path = Path(json_file)
    if output_dir and json_file == "eval/papers.json":
        repeat_papers = _load_papers_from_repeat_artifacts(output_dir)
        if repeat_papers:
            return repeat_papers
    if not json_path.is_file() and output_dir:
        repeat_papers = _load_papers_from_repeat_artifacts(output_dir)
        if repeat_papers:
            return repeat_papers
    with open(json_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [{"paper_id": paper_id, **metadata} for paper_id, metadata in data.items()]


def _load_papers_from_repeat_artifacts(output_dir: str) -> list:
    """Reconstruct the 300-paper manifest from repeat_manifest.json and existing summary/eval files."""
    out = Path(output_dir)
    manifest_candidates = [out / "repeat_manifest.json", out / "run_1" / "repeat_manifest.json"]
    manifest = None
    for mc in manifest_candidates:
        if mc.is_file():
            try:
                manifest = json.loads(mc.read_text(encoding="utf-8"))
                break
            except Exception:
                pass
    if not isinstance(manifest, dict) or not isinstance(manifest.get("paper_ids"), list):
        return []

    summary_records: dict[str, dict] = {}
    for summary_file in sorted(out.glob("**/experiment_advanced_summary_*.json")):
        try:
            sdata = json.loads(summary_file.read_text(encoding="utf-8"))
            for p in sdata.get("papers", []):
                pid = p.get("paper_id")
                if pid and pid not in summary_records:
                    gt = p.get("ground_truth", {})
                    summary_records[pid] = {
                        "paper_id": pid,
                        "paper_dir": f"data/openreview_pdf/{pid}.pdf",
                        "conference": p.get("conference", ""),
                        "topic": p.get("topic", "Others"),
                        "accept_or_not": gt.get("accept_or_not"),
                        "score": gt.get("score"),
                        "strengths": gt.get("strengths", []),
                        "weaknesses": gt.get("weaknesses", []),
                        "summary": gt.get("summary", ""),
                    }
        except Exception:
            continue

    md_hashes = manifest.get("markdown_sha256", {}) if isinstance(manifest.get("markdown_sha256"), dict) else {}
    papers = []
    for pid in manifest["paper_ids"]:
        rec = dict(summary_records.get(pid, {"paper_id": pid, "paper_dir": f"data/openreview_pdf/{pid}.pdf"}))
        if pid in md_hashes:
            rec["markdown_sha256"] = md_hashes[pid]
        papers.append(rec)
    return papers


_MD_HASH_CACHE: dict[str, dict[str, Path]] = {}


def _resolve_markdown_path(
    paper_dir: str,
    md_dir: str,
    expected_sha256: str | None = None,
) -> Path:
    paper_name = Path(paper_dir).stem
    md_root = Path(md_dir)
    direct = md_root / f"{paper_name}.md"
    if direct.is_file():
        return direct

    match = re.match(r"^([a-z]+_(?:accept|reject)_\d{4}_)\d{3}_(.+)$", paper_name)
    if match:
        prefix, slug = match.groups()
        candidates = sorted(md_root.glob(f"{prefix}*_{slug}.md"))
        if candidates:
            return candidates[0]

    if expected_sha256 and md_root.is_dir():
        cache_key = str(md_root.resolve())
        if cache_key not in _MD_HASH_CACHE:
            _MD_HASH_CACHE[cache_key] = {
                hashlib.sha256(p.read_bytes()).hexdigest(): p
                for p in md_root.glob("*.md")
            }
        by_sha = _MD_HASH_CACHE[cache_key].get(expected_sha256)
        if by_sha is not None and by_sha.is_file():
            return by_sha

    return direct


def load_markdown(
    paper_dir: str,
    md_dir: str,
    expected_sha256: str | None = None,
) -> str:
    """Read the cached Markdown matching the manifest's paper filename."""
    markdown_path = _resolve_markdown_path(paper_dir, md_dir, expected_sha256=expected_sha256)
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
    resume_from: dict | None = None,
) -> dict:
    """Run one condition through the standard multi-agent pipeline."""
    kwargs = {}
    if resume_from is not None:
        kwargs["resume_from"] = resume_from
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
        **kwargs,
    )


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "model"


def _paper_slug(paper_name: str) -> str:
    """Shorten a paper stem deterministically with its 12-char sha256 prefix."""
    digest = hashlib.sha256(paper_name.encode("utf-8")).hexdigest()[:12]
    return f"{paper_name[:35]}-{digest}"


def _result_metadata(
    paper_name: str,
    cond: dict,
    provider: str,
    model: str,
    *,
    use_paper_slug: bool | None = None,
) -> str:
    """Return the standard filename metadata for one condition."""
    def _build(token: str) -> str:
        return (
            f"_nagent={len(cond['agents'])}"
            f"_niter={cond['n_iter']}"
            f"_agenttype={cond['agenttype']}"
            f"_rag={int(cond['enable_rag'])}"
            f"_styledetector={int(ENABLE_AI_DETECTOR)}"
            f"_provider={provider}"
            f"_model={_slug(model)}"
            f"_paper={token}"
            f"_cond={cond['id']}_{cond['label']}.txt"
        )

    if use_paper_slug is True:
        return _build(_paper_slug(paper_name))
    if use_paper_slug is False:
        return _build(paper_name)
    full = _build(paper_name)
    return _build(_paper_slug(paper_name)) if len(full) > 220 else full


def _result_is_complete(result: Any, cond: dict) -> bool:
    expected = len(cond.get("agents", []))
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("reviewers"), list)
        or len(result["reviewers"]) != expected
        or sum(1 for r in result["reviewers"] if review_is_valid(r)) != expected
        or result.get("turn_failures")
        or result.get("workflow_status") in {"failed", "interrupted", "in_progress", "invalid"}
    ):
        return False
    if "workflow_schema_version" in result:
        return workflow_is_complete(result, expected, cond.get("n_iter", 1))
    return True


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
    """Return the latest matching result, preferring complete results over failed ones."""
    out = Path(output_dir)
    patterns = [
        f"*{_result_metadata(paper_name, cond, provider, model, use_paper_slug=False)}",
        f"*{_result_metadata(paper_name, cond, provider, model, use_paper_slug=True)}",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(out.glob(pattern))
    matches = sorted(set(matches))
    if not matches:
        return None
    for candidate in reversed(matches):
        loaded = _load_existing_result(candidate)
        if loaded is not None and _result_is_complete(loaded, cond):
            return candidate
    return matches[-1]


def _load_existing_result(path: Path) -> dict | None:
    """Load an existing result, or return None when it is unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _terminal_provider_diagnostic(result: Any) -> dict | None:
    """Return the provider_error diagnostic if a workflow stopped due to terminal API/credit failure."""
    if not isinstance(result, dict):
        return None
    for turn in result.get("turn_failures", []):
        if not isinstance(turn, dict):
            continue
        for att in turn.get("attempts", []):
            if isinstance(att, dict) and att.get("status") == "provider_error":
                status_code = att.get("status_code")
                if status_code in {401, 402, 403}:
                    return att
    return None


def _select_conditions(condition_spec: list[str] | str | None, output_dir: str | None = None) -> list[dict]:
    """Resolve active conditions from CLI args or repeat_manifest.json."""
    by_id = {cond["id"]: cond for cond in CONDITIONS}
    if condition_spec:
        raw_items = [condition_spec] if isinstance(condition_spec, str) else list(condition_spec)
        selected_ids: list[str] = []
        for item in raw_items:
            for token in str(item).split(","):
                cid = token.strip().upper()
                if not cid:
                    continue
                if cid not in by_id:
                    raise ValueError(f"Unknown condition {cid!r}; expected one of {sorted(by_id)}")
                if cid not in selected_ids:
                    selected_ids.append(cid)
        if selected_ids:
            return [by_id[cid] for cid in selected_ids]

    if output_dir:
        for candidate in (Path(output_dir) / "repeat_manifest.json", Path(output_dir) / "run_1" / "repeat_manifest.json"):
            if candidate.is_file():
                try:
                    manifest = json.loads(candidate.read_text(encoding="utf-8"))
                    manifest_conds = manifest.get("conditions", [])
                    ids = [c.get("id") for c in manifest_conds if isinstance(c, dict) and c.get("id") in by_id]
                    if ids:
                        return [by_id[cid] for cid in ids]
                except Exception:
                    pass
    return list(CONDITIONS)


def _resolve_api_key(provider: str, explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    env_names = {
        "openrouter": ("OPENROUTER_API_KEY", "OPENAI_API_KEY"),
        "deepseek": ("DEEPSEEK_API_KEY",),
    }.get(provider, ("API_KEY",))
    for env_name in env_names:
        val = os.environ.get(env_name, "").strip()
        if val:
            return val
    raise ValueError(
        f"No API key provided for provider {provider!r}. Pass --api_key or set {' / '.join(env_names)}."
    )


# ── Main experiment loop ──────────────────────────────────────────────────────

def run_experiment(
    papers: list,
    api_key: str,
    output_dir: str,
    md_dir: str,
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    conditions: list[dict] | None = None,
    run_metadata: dict | None = None,
) -> dict:
    """Run selected conditions and return the comparative summary."""
    model = model or DEFAULT_MODELS[provider]
    active_conditions = conditions or list(CONDITIONS)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d%H%M%S%f")

    summary: dict[str, Any] = {
        "timestamp": timestamp,
        "provider": provider,
        "model": model,
    }
    if run_metadata is not None:
        summary["run_metadata"] = run_metadata
    summary["conditions"] = {
        cond["id"]: {
            "desc": cond["desc"],
            "agents": cond["agents"],
            "n_iter": cond["n_iter"],
            "agenttype": cond["agenttype"],
            "enable_rag": cond["enable_rag"],
            "enable_ai_detector": ENABLE_AI_DETECTOR,
        }
        for cond in active_conditions
    }
    summary["papers"] = []

    complete_count = 0
    stopped_early = None

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
            for cond in active_conditions
        }
        existing_complete = {}
        existing_incomplete = {}
        shared_rag_package = None
        for cond in active_conditions:
            existing_path = existing_paths[cond["id"]]
            if existing_path is None:
                continue

            existing_result = _load_existing_result(existing_path)
            if existing_result is None:
                print(f"Existing result unreadable, rerunning: {existing_path}")
                continue

            if _result_is_complete(existing_result, cond):
                existing_complete[cond["id"]] = (existing_path, existing_result)
            else:
                existing_incomplete[cond["id"]] = (existing_path, existing_result)

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
        for cond in active_conditions:
            print(f"\n--- Condition {cond['id']}: {cond['desc']} ---")
            existing = existing_complete.get(cond["id"])
            reused_existing = existing is not None

            if existing is not None:
                existing_path, result = existing
                output_path = str(existing_path)
                print(f"Skipping: found complete existing result at {output_path}")
            else:
                if paper_text is None:
                    print("Loading cached Markdown...")
                    expected_sha = (
                        paper_meta.get("markdown_sha256")
                        or (
                            run_metadata.get("markdown_sha256", {}).get(paper_id)
                            if isinstance(run_metadata, dict)
                            else None
                        )
                    )
                    if expected_sha:
                        paper_text = load_markdown(
                            paper_meta["paper_dir"], md_dir, expected_sha256=expected_sha
                        )
                    else:
                        paper_text = load_markdown(paper_meta["paper_dir"], md_dir)
                    print("Paper text ready.")

                prior_incomplete = existing_incomplete.get(cond["id"])
                resume_from = None
                if prior_incomplete is not None:
                    prior_path, prior_result = prior_incomplete
                    print(f"Resuming incomplete result from {prior_path.name}...")
                    resume_from = prior_result

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
                    resume_from=resume_from,
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

            if _result_is_complete(result, cond):
                complete_count += 1

            paper_entry["conditions"][cond["id"]] = {
                "desc": cond["desc"],
                "result_file": os.path.basename(output_path),
                "reused_existing": reused_existing,
                "result": result,
            }

            terminal_diag = _terminal_provider_diagnostic(result)
            if not reused_existing and terminal_diag is not None:
                stopped_early = {
                    "reason": "provider_access_error",
                    "paper_id": paper_id,
                    "condition": cond["id"],
                    "diagnostic": terminal_diag,
                }
                break

        summary["papers"].append(paper_entry)
        if stopped_early is not None:
            print(
                f"\nStopping early due to provider error on {paper_id}/{stopped_early['condition']}."
            )
            break

    total_expected = len(papers) * len(active_conditions)
    if stopped_early is not None:
        summary["stopped_early"] = stopped_early
    summary["completion_counts"] = {
        "complete": complete_count,
        "incomplete": max(0, total_expected - complete_count),
    }

    summary_path = os.path.join(
        output_dir,
        f"experiment_advanced_summary_{timestamp}.json",
    )
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"\nExperiment summary saved: {summary_path}")

    return summary


def _build_repeat_manifest(
    repeat_id: int,
    papers: list[dict],
    json_file: str,
    md_dir: str,
    conditions: list[dict],
    provider: str,
    model: str,
) -> dict:
    dataset_bytes = Path(json_file).read_bytes() if Path(json_file).is_file() else b""
    dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest() if dataset_bytes else ""
    markdown_sha256 = {}
    for p in papers:
        stem = Path(p["paper_dir"]).stem
        md_path = Path(md_dir) / f"{stem}.md"
        if md_path.is_file():
            markdown_sha256[p["paper_id"]] = hashlib.sha256(md_path.read_bytes()).hexdigest()
    return {
        "repeat_id": repeat_id,
        "dataset_sha256": dataset_sha256,
        "markdown_sha256": markdown_sha256,
        "paper_ids": [p["paper_id"] for p in papers],
        "conditions": conditions,
        "provider": provider,
        "model": model,
        "enable_ai_detector": ENABLE_AI_DETECTOR,
        "run_citation_check": RUN_CITATION_CHECK,
        "enable_author_rebuttal": True,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the advanced paper-review conditions."
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
        default=None,
        help="API key for the selected provider (or set OPENROUTER_API_KEY / DEEPSEEK_API_KEY).",
    )
    parser.add_argument(
        "--output_dir",
        default="experiment_artifacts/local/eval/advanced_exp_results",
        help="Directory for result files and the summary (or repeats root directory).",
    )
    parser.add_argument(
        "--paper_id",
        default=None,
        help="Optional exact paper_id from the JSON manifest.",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Optional condition IDs to run (e.g. C1 C5 or C1,C5).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Number of repeat runs (run_1..run_N) under --output_dir.",
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
    try:
        api_key = _resolve_api_key(args.provider, args.api_key)
        active_conditions = _select_conditions(args.conditions, args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))

    papers = load_papers(args.json_file, output_dir=args.output_dir)
    if args.paper_id:
        papers = [paper for paper in papers if paper["paper_id"] == args.paper_id]
        if not papers:
            print(f"Error: paper_id '{args.paper_id}' not found.")
            sys.exit(1)

    print(f"Found {len(papers)} paper(s). Conditions: {[c['id'] for c in active_conditions]}")

    out_path = Path(args.output_dir)
    is_repeats_root = (
        args.repeats is not None
        or (out_path / "run_1").is_dir()
    )
    if is_repeats_root:
        num_repeats = args.repeats or max(
            3,
            len([d for d in out_path.glob("run_*") if d.is_dir()]),
        )
        for repeat_id in range(1, num_repeats + 1):
            run_dir = out_path / f"run_{repeat_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = run_dir / "repeat_manifest.json"
            if manifest_path.is_file():
                try:
                    run_meta = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    run_meta = _build_repeat_manifest(
                        repeat_id, papers, args.json_file, args.md_dir,
                        active_conditions, args.provider, model,
                    )
            else:
                run_meta = _build_repeat_manifest(
                    repeat_id, papers, args.json_file, args.md_dir,
                    active_conditions, args.provider, model,
                )
                manifest_path.write_text(
                    json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            print(f"\n{'#' * 60}")
            print(f"# Repeat run_{repeat_id} / {num_repeats} -> {run_dir}")
            print(f"{'#' * 60}")
            summary = run_experiment(
                papers=papers,
                api_key=api_key,
                output_dir=str(run_dir),
                md_dir=args.md_dir,
                provider=args.provider,
                model=model,
                conditions=active_conditions,
                run_metadata=run_meta,
            )
            if summary.get("stopped_early"):
                print(f"\nStopped early during run_{repeat_id}; rerun the same command to continue.")
                sys.exit(1)
    else:
        manifest_path = out_path / "repeat_manifest.json"
        run_meta = None
        if manifest_path.is_file():
            try:
                run_meta = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                run_meta = None
        summary = run_experiment(
            papers=papers,
            api_key=api_key,
            output_dir=args.output_dir,
            md_dir=args.md_dir,
            provider=args.provider,
            model=model,
            conditions=active_conditions,
            run_metadata=run_meta,
        )
        if summary.get("stopped_early"):
            sys.exit(1)

    print("\nAdvanced experiment complete.")


if __name__ == "__main__":
    main()
