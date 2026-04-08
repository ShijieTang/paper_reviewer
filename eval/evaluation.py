"""
evaluation.py

Evaluate AI paper reviewers against human ground-truth reviews from OpenReview.

Sources evaluated:
    - OpenReviewer  : results in openreviewer.json
    - PaperReviewer : results in paperreviewer.json (our system)
    - Exp Cond A    : single-agent, no rebuttal  (from experiment_summary.json)
    - Exp Cond B    : multi-agent, 3 iterations  (from experiment_summary.json)

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
        [--exp_summary  eval/exp_results/experiment_summary_XXXXXX.json] \\
        [--output_dir   eval/eval_results]        \\
        [--embed_model  all-MiniLM-L6-v2]
"""

from __future__ import annotations

import argparse
import json
import os
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


def _collect_sw_from_masloop(reviewers: list) -> tuple[list, list]:
    """
    Collect strengths / weaknesses from a mas_loop 'reviewers' array.
    Each element has "strengths" and "weaknesses" fields.
    """
    return _collect_sw_from_reviews(reviewers)


# ── Score / decision helpers ──────────────────────────────────────────────────

def _normalise_decision(raw: Optional[str]) -> Optional[str]:
    """Normalise accept/reject string to lowercase 'accept' or 'reject'."""
    if raw is None:
        return None
    return raw.strip().lower()


def _majority_decision(reviewers: list) -> Optional[str]:
    """Derive accept/reject from a list of reviewer dicts (majority vote)."""
    decisions = [_normalise_decision(r.get("decision")) for r in reviewers
                 if r.get("decision")]
    if not decisions:
        return None
    accepts = decisions.count("accept")
    rejects = decisions.count("reject")
    if accepts > rejects:
        return "accept"
    if rejects > accepts:
        return "reject"
    return decisions[-1]  # tie: use last reviewer


def _avg_rating(reviews: list) -> Optional[float]:
    """Average the 'rating' field across reviewer dicts."""
    ratings = [r["rating"] for r in reviews if "rating" in r]
    return sum(ratings) / len(ratings) if ratings else None


def _masloop_avg_score(reviewers: list) -> Optional[float]:
    """
    Average overall score from mas_loop output.
    Uses mean of all sub-dimension scores (novelty/soundness/…) across reviewers.
    """
    all_vals = []
    for rev in reviewers:
        scores = rev.get("scores", {})
        all_vals.extend(v for v in scores.values() if isinstance(v, (int, float)))
    return sum(all_vals) / len(all_vals) if all_vals else None


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
        "conference_check":  _conference_check(gt_decision, sys_score, conf_threshold),
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
    papers_path:       str,
    openreviewer_path: Optional[str],
    paperreviewer_path: Optional[str],
    exp_summary_path:  Optional[str],
    output_dir:        str,
    embed_model_name:  str = "all-MiniLM-L6-v2",
    paper_ids:         Optional[list] = None,
    conf_threshold:    float = 6.0,
) -> dict:

    print("Loading embedding model...")
    model = load_model(embed_model_name)
    print(f"Model '{embed_model_name}' ready.\n")

    gt_index = _papers_index(papers_path)

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
    exp_index = {}
    if exp_summary_path:
        exp_data = _load_json(exp_summary_path)
        exp_index = {p["paper_id"]: p for p in exp_data["papers"]}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp":    timestamp,
        "embed_model":  embed_model_name,
        "sources": {
            "papers":       papers_path,
            "openreviewer": openreviewer_path,
            "paperreviewer": paperreviewer_path,
            "exp_summary":  exp_summary_path,
        },
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

        def _add_system(name, sw_pair, decision, score):
            sys_s, sys_w = sw_pair
            metrics = _evaluate_system(
                name, gt_s, gt_w, gt_decision,
                sys_s, sys_w, decision, score, model,
                conf_threshold=conf_threshold,
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
            _add_system("openreviewer", sw, dec, sc)

        # PaperReviewer
        if paper_id in pr_index:
            p = pr_index[paper_id]
            sw  = _collect_sw_from_reviews(p.get("reviews", []))
            dec = _normalise_decision(p.get("accept_or_not"))
            sc  = p.get("score")
            _add_system("paperreviewer", sw, dec, sc)

        # Experiment conditions A and B
        if paper_id in exp_index:
            ep = exp_index[paper_id]
            for cond_id, cond_data in ep.get("conditions", {}).items():
                reviewers = cond_data.get("result", {}).get("reviewers", [])
                if not reviewers:
                    continue
                sw  = _collect_sw_from_masloop(reviewers)
                dec = _majority_decision(reviewers)
                sc  = _masloop_avg_score(reviewers)
                _add_system(f"exp_cond_{cond_id}", sw, dec, sc)

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
    out_path = os.path.join(output_dir, f"eval_results_{timestamp}.json")
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
    parser.add_argument("--papers",         required=True,
                        help="Path to papers.json (ground truth).")
    parser.add_argument("--openreviewer",   default="eval/papers.json",
                        help="Path to openreviewer.json.")
    parser.add_argument("--paperreviewer",  default=None,
                        help="Path to paperreviewer.json.")
    parser.add_argument("--exp_summary",    default=None,
                        help="Path to experiment_summary JSON from experiment.py.")
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
    args = parser.parse_args()

    if not any([args.openreviewer, args.paperreviewer, args.exp_summary]):
        print("Warning: no system sources provided. "
              "Supply at least one of --openreviewer, --paperreviewer, --exp_summary.")
        sys.exit(1)

    run_evaluation(
        papers_path=args.papers,
        openreviewer_path=args.openreviewer,
        paperreviewer_path=args.paperreviewer,
        exp_summary_path=args.exp_summary,
        output_dir=args.output_dir,
        embed_model_name=args.embed_model,
        paper_ids=args.paper_ids,
        conf_threshold=args.conf_threshold,
    )


if __name__ == "__main__":
    main()
