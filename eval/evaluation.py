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
    - our_C*          : additional conditions from experiment summaries

Metrics per paper per system:
    - src_strengths       : Semantic Review Coverage for strength statements
    - src_weaknesses      : Semantic Review Coverage for weakness statements
    - src_overall         : average of the two SRC scores
    - decision_match      : whether accept/reject decision matches ground truth
    - conference_check    : optional method-specific conference/policy check
    - norm_score          : our avg score normalised to [0,1] (our scale 1-5)
    - norm_gt_score       : GT avg rating normalised to [0,1] (per-conference scale)

Aggregate metrics:
    - decision_accuracy         : proportion of papers with correct accept/reject
    - conference_check_accuracy : optional conference/policy check accuracy
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
import hashlib
import importlib.metadata
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
from eval.evaluation_protocol import EMBED_MODEL, EMBED_REVISION
from review_schema import REVIEW_SCORE_KEYS, review_is_valid, validate_review_schema, workflow_is_complete


# ── Conference score ranges ───────────────────────────────────────────────────

_CONF_SCORE_RANGE: dict[str, tuple[float, float]] = {
    "ICLR":    (1.0, 10.0),
    "ICML":    (1.0,  5.0),
    "NEURIPS": (1.0,  6.0),
}
_OUR_SCORE_RANGE: tuple[float, float] = (1.0, 5.0)
_OUR_REVIEW_SCORE_FIELDS = REVIEW_SCORE_KEYS


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


def _artifact_info(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    artifact = Path(path)
    payload = artifact.read_bytes()
    info = {
        "path": str(artifact),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    try:
        parsed = json.loads(payload)
        canonical = json.dumps(
            parsed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        info["json_content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        pass
    return info


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


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
    if n == 1:
        return _slugify(selected_ids[0])
    if used_all_papers:
        return f"all-{n}-papers"
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


def _advanced_decision_from_reviewers(
    reviewers: list,
    expected_reviewers: int,
) -> Optional[str]:
    if len(reviewers) != expected_reviewers or not all(review_is_valid(r) for r in reviewers):
        return None
    votes = []
    for reviewer in reviewers:
        decision = _normalise_decision(reviewer.get("decision"))
        if decision in ("accept", "reject"):
            votes.append(decision)

    if not votes:
        return None
    threshold = len(votes) // 2 + 1
    if votes.count("accept") >= threshold:
        return "accept"
    if votes.count("reject") >= threshold:
        return "reject"
    return None


def _score_from_reviewers(reviewers: list) -> Optional[float]:
    if not reviewers or not all(review_is_valid(review) for review in reviewers):
        return None
    per_reviewer = []
    for r in reviewers:
        scores = r.get("scores", {})
        vals = [scores[key] for key in _OUR_REVIEW_SCORE_FIELDS if key in scores]
        if len(vals) == len(_OUR_REVIEW_SCORE_FIELDS):
            per_reviewer.append(sum(vals) / len(vals))
    return round(sum(per_reviewer) / len(per_reviewer), 4) if per_reviewer else None


def _collected_reviewers(reviewers: list) -> list[dict]:
    """Use the same schema as generation, repair, and audit."""
    if not isinstance(reviewers, list):
        return []
    return [
        reviewer for reviewer in reviewers
        if review_is_valid(reviewer)
    ]


def _validate_frozen_metric(summary: dict, model_name: str, revision: Optional[str]) -> None:
    """Reject a changed metric before loading a model or producing scores."""
    if summary.get("experiment") != "trigger_rag_screening":
        return
    actual = {
        "expected_evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "expected_src_sha256": hashlib.sha256(Path(__file__).with_name("SRC.py").read_bytes()).hexdigest(),
        "expected_embed_model": model_name,
        "expected_embed_revision": revision,
    }
    for key, value in actual.items():
        if not value or summary.get(key) != value:
            raise ValueError(f"Frozen evaluation protocol mismatch: {key}")


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

    decision_match = None
    if gt_decision and sys_decision:
        decision_match = (
            _normalise_decision(gt_decision) == _normalise_decision(sys_decision)
        )
    conference_check = decision_match if system_name.startswith("our_") else None

    norm_score    = _normalise_our_score(sys_score)
    norm_gt_score = _normalise_gt_score(gt_score, conference)

    return {
        "system":           system_name,
        "decision":         sys_decision,
        "score":            sys_score,
        "decision_match":   decision_match,
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
    embed_model_name:        Optional[str] = None,
    paper_ids:               Optional[list] = None,
    our_results:             Optional[str] = None,
    exp_summary_path:        Optional[str] = None,
    baseline_summary_path:   Optional[str] = None,
    nopersona_summary_path:  Optional[str] = None,
    output_path:             Optional[str] = None,
    embed_revision:          Optional[str] = None,
) -> dict:

    exp_document = _load_json(exp_summary_path) if exp_summary_path else {}
    is_trigger = exp_document.get("experiment") == "trigger_rag_screening"
    embed_model_name = embed_model_name or (EMBED_MODEL if is_trigger else "all-MiniLM-L6-v2")
    if is_trigger and embed_revision is None:
        embed_revision = EMBED_REVISION
    _validate_frozen_metric(exp_document, embed_model_name, embed_revision)
    if is_trigger and output_path and Path(output_path).exists():
        raise FileExistsError(f"Refusing to overwrite sealed evaluation: {output_path}")
    print("Loading embedding model...")
    model = load_model(embed_model_name, revision=embed_revision) if embed_revision else load_model(embed_model_name)
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
    exp_index         = {p["paper_id"]: p for p in exp_document.get("papers", [])}
    exp_condition_configs = exp_document.get("conditions", {})
    baseline_index    = _load_baseline_summary(baseline_summary_path) if baseline_summary_path    else {}
    nopersona_index   = _load_nopersona_summary(nopersona_summary_path) if nopersona_summary_path else {}

    our_results_stem = Path(our_results).stem if our_results else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp":   timestamp,
        "embed_model": embed_model_name,
        "embed_revision": embed_revision,
        "evaluation_runtime": {
            "evaluation_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "src_code_sha256": hashlib.sha256(Path(__file__).with_name("SRC.py").read_bytes()).hexdigest(),
            "sentence_transformers_version": _distribution_version("sentence-transformers"),
        },
        "input_artifacts": {
            "papers": _artifact_info(papers_path),
            "exp_summary": (
                {
                    **(_artifact_info(exp_summary_path) or {}),
                    "repeat_id": exp_document.get("repeat_id"),
                    "experiment": exp_document.get("experiment"),
                    "code_bundle_sha256": exp_document.get("code_bundle_sha256"),
                    "prompt_bundle_sha256": exp_document.get("prompt_bundle_sha256"),
                }
                if exp_summary_path
                else None
            ),
        },
        "papers":      [],
        "invalid_conditions": [],
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

        # Experiment conditions (legacy A/B and advanced C1-C5)
        if paper_id in exp_index:
            exp_paper = exp_index[paper_id]
            legacy_names = {"A": "our_single", "B": "our_multi"}
            for cond_id, cond in exp_paper.get("conditions", {}).items():
                raw_result = cond.get("result") or {}
                raw_reviews = raw_result.get("reviewers", [])
                reviewers = _collected_reviewers(raw_reviews)
                config = exp_condition_configs.get(cond_id, {})
                configured_agents = config.get("agents") or config.get("reviewers") or []
                expected = len(configured_agents) or (len(raw_reviews) if isinstance(raw_reviews, list) else 0)
                invalid = (
                    not isinstance(raw_reviews, list) or not reviewers
                    or len(reviewers) != len(raw_reviews) or len(reviewers) != expected
                    or bool(raw_result.get("turn_failures"))
                    or raw_result.get("status") in {"invalid", "failed", "interrupted"}
                    or raw_result.get("workflow_status") in {"in_progress", "invalid", "failed", "interrupted"}
                    or (is_trigger and not workflow_is_complete(raw_result, 3, 3))
                )
                if invalid:
                    errors = {
                        "paper_id": paper_id, "condition_id": cond_id,
                        "expected_reviews": expected, "valid_reviews": len(reviewers),
                        "review_errors": [validate_review_schema(r) for r in raw_reviews] if isinstance(raw_reviews, list) else ["reviewers must be a list"],
                        "turn_failures": raw_result.get("turn_failures", []),
                    }
                    results["invalid_conditions"].append(errors)
                    if is_trigger:
                        raise ValueError(f"Sealed trigger condition is invalid: {paper_id}/{cond_id}: {errors}")
                    print(f"  [{cond_id}] INVALID: expected {expected} valid reviews, got {len(reviewers)}; excluded from metrics")
                    continue
                if reviewers:
                    sw  = _collect_sw_from_reviews(reviewers)
                    if cond_id in legacy_names:
                        dec = _decision_from_reviewers(reviewers)
                    else:
                        dec = _advanced_decision_from_reviewers(reviewers, expected)
                    sc  = _score_from_reviewers(reviewers)
                    sys_name = legacy_names.get(cond_id, f"our_{cond_id}")
                    _add(sys_name, sw, dec, sc)

        # Condition C (no-persona baseline)
        if paper_id in baseline_index:
            reviewers = _collected_reviewers(
                baseline_index[paper_id].get("result", {}).get("reviewers", [])
            )
            if reviewers:
                sw  = _collect_sw_from_reviews(reviewers)
                dec = _decision_from_reviewers(reviewers)
                sc  = _score_from_reviewers(reviewers)
                _add("our_baseline", sw, dec, sc)

        # Condition D (3×no-persona, 3 iterations)
        if paper_id in nopersona_index:
            reviewers = _collected_reviewers(
                nopersona_index[paper_id].get("result", {}).get("reviewers", [])
            )
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
            "decision_accuracy":           _bool_acc("decision_match"),
            "conference_check_accuracy":  _bool_acc("conference_check"),
            "src_strengths_mean":         _mean("src_strengths"),
            "src_weaknesses_mean":        _mean("src_weaknesses"),
            "src_overall_mean":           _mean("src_overall"),
        }
        if score_corr:
            aggregate[sys_name]["score_spearman"] = score_corr

    results["aggregate"] = aggregate

    # ── Save ──────────────────────────────────────────────────────────────────
    if output_path:
        out_path = output_path
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    else:
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
        print(f"    {'decision_accuracy':<30}: {agg['decision_accuracy']}")
        print(f"    {'conference_check_accuracy':<30}: {agg['conference_check_accuracy']}")
        sc = agg.get('score_spearman')
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
    parser.add_argument("--output_dir",          default="experiment_artifacts/local/eval/eval_results")
    parser.add_argument("--output_file",         default=None,
                        help="Optional exact output JSON path (useful for sealed experiment pipelines)")
    parser.add_argument("--embed_model", default=None,
                        help="Trigger runs default to the frozen model in evaluation_protocol.py")
    parser.add_argument("--embed_revision", default=None,
                        help="Immutable embedding revision; trigger defaults to its frozen revision")
    parser.add_argument("--paper_ids",           default=None, nargs="+")
    parser.add_argument("--exp_summary",         default=None,
                        help="experiment_summary_*.json (all conditions)")
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
        embed_revision=args.embed_revision,
        paper_ids=args.paper_ids,
        exp_summary_path=args.exp_summary,
        baseline_summary_path=args.baseline_summary,
        nopersona_summary_path=args.nopersona_summary,
        output_path=args.output_file,
    )


if __name__ == "__main__":
    main()
