"""
evaluation.py

Evaluate AI paper reviewers against human ground-truth reviews from OpenReview.

Sources evaluated:
    - OpenReviewer  : results in openreviewer.json
    - PaperReviewer : results in paperreviewer.json
    - our_single    : Condition A (single-agent)  from experiment_summary_*.json
    - our_multi     : Condition B (multi-agent)   from experiment_summary_*.json
    - our_baseline  : Condition C (no-persona)    from experiment_baseline_summary_*.json

Metrics per paper per system:
    - SRC_strengths     : Semantic Review Coverage for strength statements
    - SRC_weaknesses    : Semantic Review Coverage for weakness statements
    - SRC_overall       : average of the two SRC scores
    - decision_match    : whether the system's accept/reject matches ground truth
    - conference_check  : whether the system's recommendation score fit threshold (default 6)

Usage (from project root):
    python eval/evaluation.py \\
        --papers        eval/papers.json          \\
        --openreviewer  eval/openreviewer.json    \\
        --paperreviewer eval/paperreviewer.json   \\
        [--exp_summary      eval/exp_results/experiment_summary_XXXXXX.json] \\
        [--baseline_summary eval/exp_results/experiment_baseline_summary_XXXXXX.json] \\
        [--output_dir   eval/eval_results]        \\
        [--embed_model  all-MiniLM-L6-v2]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.SRC import compute_src_both, load_model


# ── JSON loaders ──────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _papers_index(papers_path: str) -> dict:
    """Return {paper_id: paper_meta} from papers.json."""
    return _load_json(papers_path)


# ── Strength / weakness extraction ───────────────────────────────────────────

def _collect_sw_from_reviews(reviews: list) -> tuple[list, list]:
    """
    Collect all strength and weakness strings from a list of reviewer dicts
    (openreviewer.json / paperreviewer.json format).

    Each reviewer dict must have "strengths" and "weaknesses" list fields.
    All items are concatenated across reviewers — each item is one chunk.
    """
    strengths, weaknesses = [], []
    for rev in reviews:
        strengths.extend(rev.get("strengths", []))
        weaknesses.extend(rev.get("weaknesses", []))
    return strengths, weaknesses


def _collect_sw_from_gt(paper_meta: dict) -> tuple[list, list]:
    """
    Collect ground-truth strengths / weaknesses from a papers.json entry.
    Supports both the nested "reviews" format and legacy flat lists.
    """
    if "reviews" in paper_meta:
        return _collect_sw_from_reviews(paper_meta["reviews"])
    # Legacy flat format
    return (
        paper_meta.get("strengths", []),
        paper_meta.get("weaknesses", []),
    )


# ── Experiment summary loader ─────────────────────────────────────────────────

def _load_exp_summary(path: str) -> dict:
    """Load experiment_summary_*.json and index papers by paper_id."""
    data = _load_json(path)
    return {p["paper_id"]: p for p in data["papers"]}


def _load_baseline_summary(path: str) -> dict:
    """Load experiment_baseline_summary_*.json and index papers by paper_id."""
    data = _load_json(path)
    return {p["paper_id"]: p for p in data["papers"]}


def _slugify(text: str) -> str:
    """Convert arbitrary text to a compact filename-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "unknown"


def _scope_slug(selected_ids: list[str], used_all_papers: bool) -> str:
    """Build a readable description of which papers were evaluated."""
    n_papers = len(selected_ids)
    if n_papers == 0:
        return "no-papers"
    if n_papers == 1:
        return _slugify(selected_ids[0])
    if used_all_papers:
        return f"all-{n_papers}-paper{'s' if n_papers != 1 else ''}"
    if n_papers <= 3:
        return "_".join(_slugify(pid) for pid in selected_ids)
    return f"subset-{n_papers}-papers"


def _sources_slug(
    openreviewer_path: Optional[str],
    paperreviewer_path: Optional[str],
    our_results_stem: Optional[str],
    exp_summary_path: Optional[str],
    baseline_summary_path: Optional[str] = None,
) -> str:
    """Build a short readable slug describing which systems were evaluated."""
    sources = []
    if openreviewer_path:
        sources.append("openreviewer")
    if paperreviewer_path:
        sources.append("paperreviewer")
    if exp_summary_path:
        sources.append("our-experiment")
    if baseline_summary_path:
        sources.append("our-baseline")
    if our_results_stem:
        sources.append(f"our-{_slugify(our_results_stem)}")
    return "__".join(sources) if sources else "no-systems"


# ── Score / decision helpers ──────────────────────────────────────────────────

def _normalise_decision(raw: Optional[str]) -> Optional[str]:
    """Normalise accept/reject string to lowercase 'accept' or 'reject'."""
    if raw is None:
        return None
    return raw.strip().lower()


def _avg_rating(reviews: list) -> Optional[float]:
    """Average the 'rating' field across reviewer dicts."""
    ratings = [r["rating"] for r in reviews if "rating" in r]
    return sum(ratings) / len(ratings) if ratings else None


def _decision_from_reviewers(reviewers: list) -> Optional[str]:
    """Majority-vote accept/reject across all reviewers in a committee."""
    votes = [_normalise_decision(r.get("decision")) for r in reviewers]
    votes = [v for v in votes if v in ("accept", "reject")]
    if not votes:
        return None
    return "accept" if votes.count("accept") > votes.count("reject") else "reject"


def _score_from_reviewers(reviewers: list) -> Optional[float]:
    """Mean of per-reviewer average scores (novelty/soundness/… scale 1-5)."""
    per_reviewer = []
    for r in reviewers:
        vals = list(r.get("scores", {}).values())
        if vals:
            per_reviewer.append(sum(vals) / len(vals))
    return round(sum(per_reviewer) / len(per_reviewer), 4) if per_reviewer else None


# ── Per-system evaluation ─────────────────────────────────────────────────────

def _conference_check(
    gt_decision: Optional[str],
    sys_score: Optional[float],
    threshold: float,
) -> Optional[bool]:
    """
    True if the system score correctly reflects the ground-truth decision:
      - accepted paper → sys_score >= threshold
      - rejected paper → sys_score <  threshold
    Returns None when gt_decision or sys_score is unavailable.
    """
    if gt_decision is None or sys_score is None:
        return None
    if _normalise_decision(gt_decision) == "accept":
        return sys_score >= threshold
    return sys_score < threshold


def _evaluate_system(
    system_name: str,
    gt_strengths: list,
    gt_weaknesses: list,
    gt_decision: Optional[str],
    sys_strengths: list,
    sys_weaknesses: list,
    sys_decision: Optional[str],
    sys_score: Optional[float],
    model,
    conf_threshold: float = 6.0,
    run_conf_check: bool = True,
) -> dict:
    """Run all metrics for one system against ground truth for one paper."""
    src = compute_src_both(
        sys_strengths, sys_weaknesses,
        gt_strengths,  gt_weaknesses,
        model=model,
    )

    decision_match = None
    if gt_decision and sys_decision:
        decision_match = (_normalise_decision(gt_decision) ==
                          _normalise_decision(sys_decision))

    return {
        "system":            system_name,
        "decision":          sys_decision,
        "score":             sys_score,
        "decision_match":    decision_match,
        "conference_check":  _conference_check(gt_decision, sys_score, conf_threshold) if run_conf_check else None,
        "src_strengths":     src["strengths"],
        "src_weaknesses":    src["weaknesses"],
        "src_overall":       src["overall"],
        "n_strengths_gen":   len(sys_strengths),
        "n_weaknesses_gen":  len(sys_weaknesses),
        "n_strengths_gt":    len(gt_strengths),
        "n_weaknesses_gt":   len(gt_weaknesses),
    }


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_evaluation(
    papers_path:           str,
    openreviewer_path:     Optional[str],
    paperreviewer_path:    Optional[str],
    output_dir:            str,
    embed_model_name:      str = "all-MiniLM-L6-v2",
    paper_ids:             Optional[list] = None,
    conf_threshold:        float = 6.0,
    our_results:           Optional[str] = None,
    exp_summary_path:      Optional[str] = None,
    baseline_summary_path: Optional[str] = None,
) -> dict:

    print("Loading embedding model...")
    model = load_model(embed_model_name)
    print(f"Model '{embed_model_name}' ready.\n")

    gt_index = _papers_index(papers_path)
    used_all_papers = paper_ids is None

    if paper_ids:
        missing = [pid for pid in paper_ids if pid not in gt_index]
        if missing:
            print(f"Warning: paper_id(s) not found in papers.json: {missing}")
        gt_index = {pid: gt_index[pid] for pid in paper_ids if pid in gt_index}

    # Index optional sources by paper_id
    def _index(path):
        if not path:
            return {}
        data = _load_json(path)
        return {p["paper_id"]: p for p in data["papers"]}

    or_index  = _index(openreviewer_path)
    pr_index  = _index(paperreviewer_path)
    exp_index      = _load_exp_summary(exp_summary_path)         if exp_summary_path      else {}
    baseline_index = _load_baseline_summary(baseline_summary_path) if baseline_summary_path else {}

    our_results_stem = Path(our_results).stem if our_results else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp":    timestamp,
        "embed_model":  embed_model_name,
        "our_results":  our_results_stem,
        "papers": [],
    }

    all_metrics: dict[str, list] = {}  # system_name -> list of metric dicts

    for paper_id, gt_meta in gt_index.items():
        print(f"--- Evaluating paper: {paper_id} ---")

        gt_s, gt_w  = _collect_sw_from_gt(gt_meta)
        gt_decision = _normalise_decision(gt_meta.get("accept_or_not"))
        gt_score    = gt_meta.get("score")

        paper_entry = {
            "paper_id":    paper_id,
            "title":       gt_meta.get("title", ""),
            "conference":  gt_meta.get("conference", ""),
            "ground_truth": {
                "accept_or_not":    gt_decision,
                "score":            gt_score,
                "n_strengths":      len(gt_s),
                "n_weaknesses":     len(gt_w),
            },
            "systems": {},
        }

        def _add_system(name, sw_pair, decision, score, run_conf_check=True):
            sys_s, sys_w = sw_pair
            metrics = _evaluate_system(
                name, gt_s, gt_w, gt_decision,
                sys_s, sys_w, decision, score, model,
                conf_threshold=conf_threshold,
                run_conf_check=run_conf_check,
            )
            paper_entry["systems"][name] = metrics
            all_metrics.setdefault(name, []).append(metrics)
            print(
                f"  [{name}] SRC_s={metrics['src_strengths']:.4f}  "
                f"SRC_w={metrics['src_weaknesses']:.4f}  "
                f"SRC_overall={metrics['src_overall']:.4f}  "
                f"decision_match={metrics['decision_match']}"
            )

        # OpenReviewer
        if paper_id in or_index:
            p = or_index[paper_id]
            sw  = _collect_sw_from_reviews(p.get("reviews", []))
            dec = _normalise_decision(p.get("accept_or_not"))
            sc  = p.get("score")
            _add_system("openreviewer", sw, dec, sc, run_conf_check=False)

        # PaperReviewer
        if paper_id in pr_index:
            p = pr_index[paper_id]
            sw  = _collect_sw_from_reviews(p.get("reviews", []))
            dec = _normalise_decision(p.get("accept_or_not"))
            sc  = p.get("score")
            _add_system("paperreviewer", sw, dec, sc, run_conf_check=False)

        # Our system: Condition A (single-agent) and B (multi-agent)
        # Decision is majority vote across all reviewers in the committee.
        # Score is mean of per-reviewer average sub-scores (1-5 scale).
        if paper_id in exp_index:
            exp_paper = exp_index[paper_id]
            for cond_id, sys_name in [("A", "our_single"), ("B", "our_multi")]:
                cond = exp_paper.get("conditions", {}).get(cond_id, {})
                reviewers = cond.get("result", {}).get("reviewers", [])
                if reviewers:
                    sw  = _collect_sw_from_reviews(reviewers)
                    dec = _decision_from_reviewers(reviewers)
                    sc  = _score_from_reviewers(reviewers)
                    _add_system(sys_name, sw, dec, sc, run_conf_check=False)

        # Condition C: no-persona baseline (experiment_baseline_summary_*.json)
        if paper_id in baseline_index:
            reviewers = baseline_index[paper_id].get("result", {}).get("reviewers", [])
            if reviewers:
                sw  = _collect_sw_from_reviews(reviewers)
                dec = _decision_from_reviewers(reviewers)
                sc  = _score_from_reviewers(reviewers)
                _add_system("our_baseline", sw, dec, sc, run_conf_check=False)

        results["papers"].append(paper_entry)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    aggregate = {}
    for sys_name, metrics_list in all_metrics.items():
        n = len(metrics_list)

        def _mean(key):
            vals = [m[key] for m in metrics_list if m[key] is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        def _bool_acc(key):
            vals = [m[key] for m in metrics_list if m[key] is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        aggregate[sys_name] = {
            "n_papers":              n,
            "decision_accuracy":     _bool_acc("decision_match"),
            "conference_check_accuracy": _bool_acc("conference_check"),
            "src_strengths_mean":    _mean("src_strengths"),
            "src_weaknesses_mean":   _mean("src_weaknesses"),
            "src_overall_mean":      _mean("src_overall"),
        }

    results["aggregate"] = aggregate

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    scope_slug = _scope_slug(list(gt_index.keys()), used_all_papers)
    sources_slug = _sources_slug(
        openreviewer_path=openreviewer_path,
        paperreviewer_path=paperreviewer_path,
        our_results_stem=our_results_stem,
        exp_summary_path=exp_summary_path,
        baseline_summary_path=baseline_summary_path,
    )
    out_path = os.path.join(
        output_dir,
        f"evaluation_{scope_slug}__{sources_slug}__{timestamp}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Pretty summary
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    for sys_name, agg in aggregate.items():
        print(f"\n  System: {sys_name}")
        for k, v in agg.items():
            print(f"    {k:<28}: {v}")

    print(f"\nFull results saved: {out_path}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AI reviewers against OpenReview ground truth.")
    parser.add_argument("--papers",         default="eval/papers.json",
                        help="Path to papers.json (ground truth).")
    parser.add_argument("--openreviewer",   default="eval/openreviewer.json",
                        help="Path to openreviewer.json.")
    parser.add_argument("--paperreviewer",  default="eval/paperreviewer.json",
                        help="Path to paperreviewer.json.")
    parser.add_argument("--output_dir",     default="eval/eval_results",
                        help="Directory to save evaluation results.")
    parser.add_argument("--embed_model",    default="all-MiniLM-L6-v2",
                        help="sentence-transformers model for SRC computation.")
    parser.add_argument("--paper_ids",      default=None, nargs="+",
                        help="Optional: space-separated list of paper_ids to evaluate "
                             "(must exist in papers.json). Evaluates all if omitted.")
    parser.add_argument("--conf_threshold", default=6.0, type=float,
                        help="Score threshold for conference_check (default: 6.0). "
                             "Accepted papers must score >= threshold; rejected < threshold.")
    parser.add_argument("--our_results",   default=None,
                        help="Path to our model result file (from results/). "
                             "Used to label the config in the output filename.")
    parser.add_argument("--exp_summary",      default=None,
                        help="Path to experiment_summary_*.json from eval/exp_results/. "
                             "Adds our_single (Cond A) and our_multi (Cond B) to the benchmark.")
    parser.add_argument("--baseline_summary", default=None,
                        help="Path to experiment_baseline_summary_*.json from eval/exp_results/. "
                             "Adds our_baseline (Cond C) to the benchmark.")
    args = parser.parse_args()

    if not any([args.openreviewer, args.paperreviewer, args.our_results,
                args.exp_summary, args.baseline_summary]):
        print("Warning: no system sources provided. "
              "Supply at least one of --openreviewer, --paperreviewer, --our_results.")
        sys.exit(1)

    run_evaluation(
        papers_path=args.papers,
        openreviewer_path=args.openreviewer,
        paperreviewer_path=args.paperreviewer,
        output_dir=args.output_dir,
        embed_model_name=args.embed_model,
        paper_ids=args.paper_ids,
        conf_threshold=args.conf_threshold,
        our_results=args.our_results,
        exp_summary_path=args.exp_summary,
        baseline_summary_path=args.baseline_summary,
    )


if __name__ == "__main__":
    main()
