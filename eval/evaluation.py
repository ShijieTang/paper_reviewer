"""
evaluation.py

Evaluate AI paper reviewers against human ground-truth reviews from OpenReview.

Sources evaluated:
    - OpenReviewer    : results in openreviewer.json
    - PaperReviewer   : results in paperreviewer.json
    - our_single      : Condition A (single-agent)   from experiment_summary_*.json
    - our_multi       : Condition B (multi-agent)    from experiment_summary_*.json
    - our_baseline    : Condition C (no-persona)     from experiment_baseline_summary_*.json
    - our_nopersona   : Condition D (NNN, 3 iter)    from experiment_nopersona_summary_*.json

Metrics per paper per system:
    - src_strengths       : Semantic Review Coverage for strength statements
    - src_weaknesses      : Semantic Review Coverage for weakness statements
    - src_overall         : average of the two SRC scores
    - conference_check    : whether our accept/reject decision matches ground truth
    - norm_score          : our avg score normalised to [0,1] (our scale 1-5)
    - norm_gt_score       : GT avg rating normalised to [0,1] (per-conference scale)

Aggregate metrics:
    - conference_check_accuracy : proportion of papers with correct accept/reject
    - score_spearman_rho        : Spearman ρ between norm_score and norm_gt_score
    - src_strengths_mean        : mean SRC strengths
    - src_weaknesses_mean       : mean SRC weaknesses
    - src_overall_mean          : mean SRC overall

Score normalisation ranges (from official reviewer guidelines):
    ICLR    : 1–10
    ICML    : 1–5
    NeurIPS : 1–6
    Ours    : 1–5

Usage (from project root):
    python eval/evaluation.py \\
        --papers              eval/papers.json \\
        --openreviewer        eval/openreviewer.json \\
        --paperreviewer       eval/paperreviewer.json \\
        [--exp_summary        eval/exp_results/experiment_summary_*.json] \\
        [--baseline_summary   eval/exp_baseline_results/experiment_baseline_summary_*.json] \\
        [--nopersona_summary  eval/exp_results/experiment_nopersona_summary_*.json] \\
        [--output_dir         eval/eval_results]
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

from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.SRC import compute_src_both, load_model


# ── Conference score ranges ───────────────────────────────────────────────────

_CONF_SCORE_RANGE: dict[str, tuple[float, float]] = {
    "ICLR":    (1.0, 10.0),
    "ICML":    (1.0,  5.0),
    "NeurIPS": (1.0,  6.0),
}
_OUR_SCORE_RANGE: tuple[float, float] = (1.0, 5.0)


def _normalise(value: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return (value - lo) / (hi - lo)


def _normalise_gt_score(score: float, conference: str) -> Optional[float]:
    if score is None:
        return None
    lo, hi = _CONF_SCORE_RANGE.get(conference.upper(), (None, None))
    if lo is None:
        return None
    return _normalise(score, lo, hi)


def _normalise_our_score(score: Optional[float]) -> Optional[float]:
    if score is None:
        return None
    lo, hi = _OUR_SCORE_RANGE
    return _normalise(score, lo, hi)


# ── JSON loaders ──────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _papers_index(papers_path: str) -> dict:
    return _load_json(papers_path)


# ── Strength / weakness extraction ───────────────────────────────────────────

def _collect_sw_from_reviews(reviews: list) -> tuple[list, list]:
    strengths, weaknesses = [], []
    for rev in reviews:
        strengths.extend(rev.get("strengths", []))
        weaknesses.extend(rev.get("weaknesses", []))
    return strengths, weaknesses


def _collect_sw_from_gt(paper_meta: dict) -> tuple[list, list]:
    if "reviews" in paper_meta:
        return _collect_sw_from_reviews(paper_meta["reviews"])
    return (
        paper_meta.get("strengths", []),
        paper_meta.get("weaknesses", []),
    )


# ── Experiment summary loaders ────────────────────────────────────────────────

def _load_exp_summary(path: str) -> dict:
    data = _load_json(path)
    return {p["paper_id"]: p for p in data["papers"]}


def _load_baseline_summary(path: str) -> dict:
    data = _load_json(path)
    return {p["paper_id"]: p for p in data["papers"]}


def _load_nopersona_summary(path: str) -> dict:
    data = _load_json(path)
    return {p["paper_id"]: p for p in data["papers"]}


# ── Filename helpers ──────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    return re.sub(r"-{2,}", "-", slug).strip("-") or "unknown"


def _scope_slug(selected_ids: list[str], used_all_papers: bool) -> str:
    n = len(selected_ids)
    if n == 0:
        return "no-papers"
    if used_all_papers:
        return f"all-{n}-papers"
    if n == 1:
        return _slugify(selected_ids[0])
    if n <= 3:
        return "_".join(_slugify(pid) for pid in selected_ids)
    return f"subset-{n}-papers"


def _sources_slug(
    openreviewer_path: Optional[str],
    paperreviewer_path: Optional[str],
    exp_summary_path: Optional[str],
    baseline_summary_path: Optional[str],
    nopersona_summary_path: Optional[str],
) -> str:
    parts = []
    if openreviewer_path:    parts.append("openreviewer")
    if paperreviewer_path:   parts.append("paperreviewer")
    if exp_summary_path:     parts.append("our-experiment")
    if baseline_summary_path: parts.append("our-baseline")
    if nopersona_summary_path: parts.append("our-nopersona")
    return "__".join(parts) if parts else "no-systems"


# ── Score / decision helpers ──────────────────────────────────────────────────

def _normalise_decision(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    return raw.strip().lower()


def _avg_rating(reviews: list) -> Optional[float]:
    ratings = [r["rating"] for r in reviews if "rating" in r]
    return sum(ratings) / len(ratings) if ratings else None


def _decision_from_reviewers(reviewers: list) -> Optional[str]:
    votes = [_normalise_decision(r.get("decision")) for r in reviewers]
    votes = [v for v in votes if v in ("accept", "reject")]
    if not votes:
        return None
    return "accept" if votes.count("accept") > votes.count("reject") else "reject"


def _score_from_reviewers(reviewers: list) -> Optional[float]:
    per_reviewer = []
    for r in reviewers:
        vals = list(r.get("scores", {}).values())
        if vals:
            per_reviewer.append(sum(vals) / len(vals))
    return round(sum(per_reviewer) / len(per_reviewer), 4) if per_reviewer else None


# ── Per-system evaluation ─────────────────────────────────────────────────────

def _evaluate_system(
    system_name: str,
    gt_strengths: list,
    gt_weaknesses: list,
    gt_decision: Optional[str],
    gt_score: Optional[float],
    conference: str,
    sys_strengths: list,
    sys_weaknesses: list,
    sys_decision: Optional[str],
    sys_score: Optional[float],
    model,
) -> dict:
    src = compute_src_both(
        sys_strengths, sys_weaknesses,
        gt_strengths,  gt_weaknesses,
        model=model,
    )

    # conference_check: does our decision match GT?
    conference_check = None
    if gt_decision and sys_decision:
        conference_check = (
            _normalise_decision(gt_decision) == _normalise_decision(sys_decision)
        )

    norm_score    = _normalise_our_score(sys_score)
    norm_gt_score = _normalise_gt_score(gt_score, conference)

    return {
        "system":           system_name,
        "decision":         sys_decision,
        "score":            sys_score,
        "conference_check": conference_check,
        "norm_score":       norm_score,
        "norm_gt_score":    norm_gt_score,
        "src_strengths":    src["strengths"],
        "src_weaknesses":   src["weaknesses"],
        "src_overall":      src["overall"],
        "n_strengths_gen":  len(sys_strengths),
        "n_weaknesses_gen": len(sys_weaknesses),
        "n_strengths_gt":   len(gt_strengths),
        "n_weaknesses_gt":  len(gt_weaknesses),
    }


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_evaluation(
    papers_path:             str,
    openreviewer_path:       Optional[str],
    paperreviewer_path:      Optional[str],
    output_dir:              str,
    embed_model_name:        str = "all-MiniLM-L6-v2",
    paper_ids:               Optional[list] = None,
    our_results:             Optional[str] = None,
    exp_summary_path:        Optional[str] = None,
    baseline_summary_path:   Optional[str] = None,
    nopersona_summary_path:  Optional[str] = None,
) -> dict:

    print("Loading embedding model...")
    model = load_model(embed_model_name)
    print(f"Model '{embed_model_name}' ready.\n")

    gt_index       = _papers_index(papers_path)
    used_all_papers = paper_ids is None

    if paper_ids:
        missing = [pid for pid in paper_ids if pid not in gt_index]
        if missing:
            print(f"Warning: paper_id(s) not found in papers.json: {missing}")
        gt_index = {pid: gt_index[pid] for pid in paper_ids if pid in gt_index}

    def _index(path):
        if not path:
            return {}
        data = _load_json(path)
        return {p["paper_id"]: p for p in data["papers"]}

    or_index          = _index(openreviewer_path)
    pr_index          = _index(paperreviewer_path)
    exp_index         = _load_exp_summary(exp_summary_path)          if exp_summary_path         else {}
    baseline_index    = _load_baseline_summary(baseline_summary_path) if baseline_summary_path    else {}
    nopersona_index   = _load_nopersona_summary(nopersona_summary_path) if nopersona_summary_path else {}

    our_results_stem = Path(our_results).stem if our_results else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp":   timestamp,
        "embed_model": embed_model_name,
        "papers":      [],
    }

    all_metrics: dict[str, list] = {}

    for paper_id, gt_meta in gt_index.items():
        print(f"--- Evaluating paper: {paper_id} ---")

        gt_s, gt_w  = _collect_sw_from_gt(gt_meta)
        gt_decision = _normalise_decision(gt_meta.get("accept_or_not"))
        gt_score    = gt_meta.get("score")
        conference  = gt_meta.get("conference", "")

        paper_entry = {
            "paper_id":   paper_id,
            "title":      gt_meta.get("title", ""),
            "conference": conference,
            "ground_truth": {
                "accept_or_not": gt_decision,
                "score":         gt_score,
                "n_strengths":   len(gt_s),
                "n_weaknesses":  len(gt_w),
            },
            "systems": {},
        }

        def _add(name, sw_pair, decision, score):
            sys_s, sys_w = sw_pair
            m = _evaluate_system(
                name, gt_s, gt_w, gt_decision, gt_score, conference,
                sys_s, sys_w, decision, score, model,
            )
            paper_entry["systems"][name] = m
            all_metrics.setdefault(name, []).append(m)
            print(
                f"  [{name}] SRC_s={m['src_strengths']:.4f}  "
                f"SRC_w={m['src_weaknesses']:.4f}  "
                f"SRC_overall={m['src_overall']:.4f}  "
                f"conf_check={m['conference_check']}  "
                f"norm_score={m['norm_score']}  norm_gt={m['norm_gt_score']}"
            )

        # OpenReviewer
        if paper_id in or_index:
            p   = or_index[paper_id]
            sw  = _collect_sw_from_reviews(p.get("reviews", []))
            dec = _normalise_decision(p.get("accept_or_not"))
            sc  = p.get("score")
            _add("openreviewer", sw, dec, sc)

        # PaperReviewer
        if paper_id in pr_index:
            p   = pr_index[paper_id]
            sw  = _collect_sw_from_reviews(p.get("reviews", []))
            dec = _normalise_decision(p.get("accept_or_not"))
            sc  = p.get("score")
            _add("paperreviewer", sw, dec, sc)

        # Condition A (single-agent) and B (multi-agent)
        if paper_id in exp_index:
            exp_paper = exp_index[paper_id]
            for cond_id, sys_name in [("A", "our_single"), ("B", "our_multi")]:
                cond      = exp_paper.get("conditions", {}).get(cond_id, {})
                reviewers = cond.get("result", {}).get("reviewers", [])
                if reviewers:
                    sw  = _collect_sw_from_reviews(reviewers)
                    dec = _decision_from_reviewers(reviewers)
                    sc  = _score_from_reviewers(reviewers)
                    _add(sys_name, sw, dec, sc)

        # Condition C (no-persona baseline)
        if paper_id in baseline_index:
            reviewers = baseline_index[paper_id].get("result", {}).get("reviewers", [])
            if reviewers:
                sw  = _collect_sw_from_reviews(reviewers)
                dec = _decision_from_reviewers(reviewers)
                sc  = _score_from_reviewers(reviewers)
                _add("our_baseline", sw, dec, sc)

        # Condition D (3×no-persona, 3 iterations)
        if paper_id in nopersona_index:
            reviewers = nopersona_index[paper_id].get("result", {}).get("reviewers", [])
            if reviewers:
                sw  = _collect_sw_from_reviews(reviewers)
                dec = _decision_from_reviewers(reviewers)
                sc  = _score_from_reviewers(reviewers)
                _add("our_nopersona", sw, dec, sc)

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

        # Spearman correlation between normalised our score and normalised GT score
        score_corr = None
        pairs = [
            (m["norm_score"], m["norm_gt_score"])
            for m in metrics_list
            if m["norm_score"] is not None and m["norm_gt_score"] is not None
        ]
        if len(pairs) >= 3:
            ours_norm, gt_norm = zip(*pairs)
            rho, pval = spearmanr(ours_norm, gt_norm)
            score_corr = {"rho": round(float(rho), 4), "pval": round(float(pval), 4), "n": len(pairs)}

        aggregate[sys_name] = {
            "n_papers":                   n,
            "conference_check_accuracy":  _bool_acc("conference_check"),
            "score_spearman":             score_corr,
            "src_strengths_mean":         _mean("src_strengths"),
            "src_weaknesses_mean":        _mean("src_weaknesses"),
            "src_overall_mean":           _mean("src_overall"),
        }

    results["aggregate"] = aggregate

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    scope_slug   = _scope_slug(list(gt_index.keys()), used_all_papers)
    sources_slug = _sources_slug(
        openreviewer_path, paperreviewer_path,
        exp_summary_path, baseline_summary_path, nopersona_summary_path,
    )
    out_path = os.path.join(
        output_dir,
        f"evaluation_{scope_slug}__{sources_slug}__{timestamp}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Pretty summary
    print(f"\n{'='*65}")
    print("AGGREGATE RESULTS")
    print(f"{'='*65}")
    for sys_name, agg in aggregate.items():
        print(f"\n  System: {sys_name}  (n={agg['n_papers']})")
        print(f"    {'conference_check_accuracy':<30}: {agg['conference_check_accuracy']}")
        sc = agg['score_spearman']
        if sc:
            print(f"    {'score_spearman_rho':<30}: {sc['rho']}  (p={sc['pval']}, n={sc['n']})")
        else:
            print(f"    {'score_spearman_rho':<30}: N/A")
        print(f"    {'src_strengths_mean':<30}: {agg['src_strengths_mean']}")
        print(f"    {'src_weaknesses_mean':<30}: {agg['src_weaknesses_mean']}")
        print(f"    {'src_overall_mean':<30}: {agg['src_overall_mean']}")

    print(f"\nFull results saved: {out_path}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AI reviewers against OpenReview ground truth.")
    parser.add_argument("--papers",              default="eval/papers.json")
    parser.add_argument("--openreviewer",        default=None)
    parser.add_argument("--paperreviewer",       default=None)
    parser.add_argument("--output_dir",          default="eval/eval_results")
    parser.add_argument("--embed_model",         default="all-MiniLM-L6-v2")
    parser.add_argument("--paper_ids",           default=None, nargs="+")
    parser.add_argument("--exp_summary",         default=None,
                        help="experiment_summary_*.json (Cond A+B)")
    parser.add_argument("--baseline_summary",    default=None,
                        help="experiment_baseline_summary_*.json (Cond C)")
    parser.add_argument("--nopersona_summary",   default=None,
                        help="experiment_nopersona_summary_*.json (Cond D)")
    args = parser.parse_args()

    if not any([args.openreviewer, args.paperreviewer,
                args.exp_summary, args.baseline_summary, args.nopersona_summary]):
        print("Error: supply at least one system source (--openreviewer, "
              "--paperreviewer, --exp_summary, --baseline_summary, --nopersona_summary).")
        sys.exit(1)

    run_evaluation(
        papers_path=args.papers,
        openreviewer_path=args.openreviewer,
        paperreviewer_path=args.paperreviewer,
        output_dir=args.output_dir,
        embed_model_name=args.embed_model,
        paper_ids=args.paper_ids,
        exp_summary_path=args.exp_summary,
        baseline_summary_path=args.baseline_summary,
        nopersona_summary_path=args.nopersona_summary,
    )


if __name__ == "__main__":
    main()
