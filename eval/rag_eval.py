#!/usr/bin/env python3
"""Evaluate the improved related-work RAG conditions (C8, C9) against the no-RAG control (C1).

Reads experiment summaries written by eval/experiment.py and reports, per RAG condition:

1. Usage of the shown works
   * share of novelty_check entries that name a shown work (instead of "none");
   * share of reviews with at least one missing_prior_work entry;
   * which works get used: cited by the manuscript vs not cited;
   * mentions of shown works in the free-text review (same matcher as check_rag_usage.py),
     compared with the control's mentions of the same works.
2. Missing-prior-work detection vs human reviewers
   * a paper is "human-flagged" when a human weakness talks about missing related work,
     missing baselines or limited novelty relative to prior work;
   * the system flags a paper when a reviewer lists missing_prior_work (or, for C9, the
     analyst lists possibly_missing_references);
   * precision / recall / agreement, plus semantic similarity between the system's
     "why relevant" statements and the human related-work weaknesses (MiniLM if
     sentence-transformers is installed, token overlap otherwise);
   * exact hits: the human weakness names a work the system flagged (title or author-year).
3. Citation validity
   * labels must be shown labels (the validator enforces it; any slip is counted);
   * [R#] mentions in free text that are not shown labels;
   * flagged works that are cited by the manuscript or are the submission itself (should be 0).
4. Paired comparison with C1 (paper bootstrap, 95% CI)
   * decision accuracy, Spearman rho with the venue-normalized human score,
     mean novelty score shift, decision agreement with C1.

Optionally restricted to papers first public after the model's training cutoff
(--subset eval/first_public_dates_60.json, from eval/first_public_dates.py).

Usage (from the repository root):
    python eval/rag_eval.py \\
        --dataset eval/openreview_60_module_test.json \\
        --summary eval/exp_results_rag_improve/experiment_summary_<ts>.json \\
        --control_cond 1 --rag_conds 8 9 \\
        [--control_summary <summary with C1 if it is a separate run>] \\
        [--subset eval/first_public_dates_60.json] \\
        [--output eval/rag_eval_report.json]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from review_schema import REVIEW_SCORE_KEYS, review_is_valid  # noqa: E402

CONF_RANGE = {"ICLR": (1.0, 10.0), "ICML": (1.0, 5.0), "NEURIPS": (1.0, 6.0)}
LABEL_RE = re.compile(r"\[R(\d+)\]")
HUMAN_RW_RE = re.compile(
    r"\b(related work|prior work|previous work|existing work|existing methods|literature|"
    r"missing (references?|citations?|baselines?|comparisons?)|not cited|fails? to cite|"
    r"(lack|absence) of (comparison|baselines?)|compare (with|to|against)|comparison (with|to|against)|"
    r"baselines?|limited novelty|incremental|novelty is (limited|unclear|marginal)|"
    r"similar to (the )?(work|approach|method)|already (been )?(proposed|explored|studied))\b", re.I)
TOKEN_RE = re.compile(r"[a-z][a-z0-9]+")
STOP = set("the a an of to in and or for on with is are be this that by as it from at which their its "
           "paper work works method methods approach model models".split())


# ── loading ──────────────────────────────────────────────────────────────────

def load_conditions(summary_path: Path, cond_ids: list[str]) -> dict[str, dict[str, dict]]:
    """{cond_id: {paper_id: result}}, preferring the result file next to the summary."""
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, dict]] = {c: {} for c in cond_ids}
    for paper in data.get("papers", []):
        for cid in cond_ids:
            cond = (paper.get("conditions") or {}).get(cid)
            if not cond:
                continue
            result = cond.get("result") or {}
            path = summary_path.parent / (cond.get("result_file") or "")
            if cond.get("result_file") and path.is_file():
                try:
                    result = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass
            if isinstance(result, dict) and result.get("reviewers"):
                out[cid][paper["paper_id"]] = result
    return out


def valid_reviews(result: dict) -> list[dict]:
    return [r for r in result.get("reviewers", []) if isinstance(r, dict) and not r.get("parse_error")]


# ── scores and decisions ─────────────────────────────────────────────────────

def system_score(result: dict) -> float | None:
    per = []
    for review in valid_reviews(result):
        if not review_is_valid(review):
            continue
        scores = review.get("scores") or {}
        vals = [scores[k] for k in REVIEW_SCORE_KEYS if isinstance(scores.get(k), (int, float))]
        if len(vals) == len(REVIEW_SCORE_KEYS):
            per.append(sum(vals) / len(vals))
    return (sum(per) / len(per) - 1.0) / 4.0 if per else None


def novelty_score(result: dict) -> float | None:
    vals = [r["scores"]["novelty"] for r in valid_reviews(result)
            if isinstance((r.get("scores") or {}).get("novelty"), (int, float))]
    return sum(vals) / len(vals) if vals else None


def system_decision(result: dict) -> str | None:
    votes = [str(r.get("decision", "")).strip().lower() for r in valid_reviews(result)]
    votes = [v for v in votes if v in ("accept", "reject")]
    if not votes:
        return None
    return "accept" if votes.count("accept") > votes.count("reject") else "reject"


def human_score(meta: dict) -> float | None:
    lo, hi = CONF_RANGE.get(str(meta.get("conference", "")).upper(), (None, None))
    try:
        score = float(meta.get("score"))
    except (TypeError, ValueError):
        return None
    return (score - lo) / (hi - lo) if lo is not None else None


def human_decision(meta: dict) -> str | None:
    value = str(meta.get("accept_or_not", "")).strip().lower()
    return value if value in ("accept", "reject") else None


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


# ── text helpers ─────────────────────────────────────────────────────────────

def review_free_text(review: dict) -> str:
    parts = [str(review.get("summary_comment", ""))]
    for key in ("strengths", "weaknesses"):
        parts += [x for x in review.get(key, []) or [] if isinstance(x, str)]
    for detail in review.get("weakness_details", []) or []:
        if isinstance(detail, dict):
            parts += [str(v) for v in detail.values() if isinstance(v, str)]
    for gate in (review.get("acceptance_gates") or {}).values():
        if isinstance(gate, dict):
            parts.append(str(gate.get("justification", "")))
    for item in review.get("novelty_check", []) or []:
        if isinstance(item, dict):
            parts.append(str(item.get("explanation", "")))
    return "\n".join(parts)


def tokens(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall((text or "").lower()) if t not in STOP}


class Similarity:
    """Max cosine similarity between statement sets: MiniLM if available, token Jaccard otherwise."""

    def __init__(self, use_embeddings: bool = True):
        self.model = None
        self.method = "token_jaccard"
        if use_embeddings:
            try:
                print("loading MiniLM for statement similarity (first time downloads ~90 MB)...", flush=True)
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self.method = "all-MiniLM-L6-v2"
            except Exception:
                print("sentence-transformers not available; using token overlap instead", flush=True)
                self.model = None

    def best_match(self, system: list[str], human: list[str]) -> float | None:
        if not system or not human:
            return None
        if self.model is not None:
            import numpy as np
            a = self.model.encode(system, normalize_embeddings=True)
            b = self.model.encode(human, normalize_embeddings=True)
            return float(np.max(a @ b.T, axis=1).mean())
        scores = []
        for s in system:
            ts = tokens(s)
            scores.append(max((len(ts & tokens(h)) / len(ts | tokens(h)) if ts | tokens(h) else 0.0) for h in human))
        return sum(scores) / len(scores)


def work_matchers(work: dict) -> list[re.Pattern]:
    from check_rag_usage import matchers
    return matchers(work)


def text_mentions(text: str, work: dict) -> bool:
    from check_rag_usage import norm
    low, lown = (text or "").lower(), norm(text)
    return any(p.search(low) or p.search(lown) for p in work_matchers(work))


# ── per-paper extraction ─────────────────────────────────────────────────────

def shown_works(pkg: dict) -> list[dict]:
    """Labeled shown works; for packages from runs before refinement existed, the top 8 reranked works
    (what the old prompt showed), labeled in rank order, with cited status unknown."""
    if pkg.get("shown_works"):
        return list(pkg["shown_works"])
    meta = {m.get("paper_id"): m for m in pkg.get("paper_metadata", []) or [] if isinstance(m, dict)}
    ranked = sorted((r for r in pkg.get("reranking_results", []) or [] if isinstance(r, dict)),
                    key=lambda r: r.get("rank", 10**6))
    works = [dict(meta[r["paper_id"]]) for r in ranked if r.get("paper_id") in meta][:8]
    for i, work in enumerate(works, 1):
        work["label"], work["cited_in_manuscript"] = f"R{i}", None
    return works


def paper_rag_record(result: dict, analyst: bool) -> dict:
    shown = {w.get("label"): w for w in shown_works(result.get("rag_package") or {}) if w.get("label")}
    uncited = {lab for lab, w in shown.items() if w.get("cited_in_manuscript") is False}
    cited = {lab for lab, w in shown.items() if w.get("cited_in_manuscript") is True}
    rec = {"num_shown": len(shown), "num_shown_uncited": len(uncited), "novelty_entries": 0,
           "novelty_named": 0, "named_cited": 0, "named_uncited": 0, "reviews": 0,
           "reviews_with_missing": 0, "missing_items": [], "invalid_labels": 0, "free_text_bad_labels": 0,
           "free_text_mentions": 0, "flagged_cited_or_self": 0}
    for review in valid_reviews(result):
        rec["reviews"] += 1
        for item in review.get("novelty_check", []) or []:
            if not isinstance(item, dict):
                continue
            rec["novelty_entries"] += 1
            label = str(item.get("closest_prior_work", "")).strip().strip("[]").upper()
            if label and label != "NONE":
                if label in shown:
                    rec["novelty_named"] += 1
                    rec["named_cited"] += label in cited
                    rec["named_uncited"] += label in uncited
                else:
                    rec["invalid_labels"] += 1
        missing = [m for m in review.get("missing_prior_work", []) or [] if isinstance(m, dict)]
        rec["reviews_with_missing"] += bool(missing)
        for item in missing:
            label = str(item.get("work", "")).strip().strip("[]").upper()
            if label not in shown:
                rec["invalid_labels"] += 1
                continue
            work = shown[label]
            if work.get("cited_in_manuscript") is not False or work.get("is_submission"):
                rec["flagged_cited_or_self"] += 1
            rec["missing_items"].append({"work": work, "why": str(item.get("why_relevant", "")), "source": "reviewer"})
        text = review_free_text(review)
        rec["free_text_bad_labels"] += sum(1 for n in LABEL_RE.findall(text) if f"R{n}" not in shown)
        rec["free_text_mentions"] += sum(1 for w in shown.values() if text_mentions(text, w))
    if analyst:
        for item in (result.get("related_work_report") or {}).get("possibly_missing_references", []) or []:
            label = str(item.get("work", "")).strip().strip("[]").upper()
            if label in shown:
                rec["missing_items"].append({"work": shown[label], "why": str(item.get("why_relevant", "")),
                                             "source": "analyst"})
    return rec


def human_rw_weaknesses(meta: dict) -> list[str]:
    out = []
    for review in meta.get("reviews", []) or []:
        weak = review.get("weaknesses")
        items = weak if isinstance(weak, list) else re.split(r"\n+|(?<=[.;])\s+", str(weak or ""))
        out += [w.strip() for w in items if isinstance(w, str) and HUMAN_RW_RE.search(w)]
    return out


# ── bootstrap ────────────────────────────────────────────────────────────────

def bootstrap(pids: list[str], stat, n: int = 2000, seed: int = 0) -> dict:
    point = stat(pids)
    if point is None or len(pids) < 3:
        return {"value": point, "ci95": None, "n": len(pids)}
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        value = stat([rng.choice(pids) for _ in pids])
        if value is not None:
            draws.append(value)
    draws.sort()
    ci = [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]] if draws else None
    return {"value": point, "ci95": ci, "n": len(pids)}


def accuracy(results: dict, dataset: dict, pids: list[str]) -> float | None:
    pairs = [(system_decision(results[p]), human_decision(dataset[p])) for p in pids]
    pairs = [(a, b) for a, b in pairs if a and b]
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else None


def rho(results: dict, dataset: dict, pids: list[str]) -> float | None:
    pairs = [(system_score(results[p]), human_score(dataset[p])) for p in pids]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    return spearman([a for a, _ in pairs], [b for _, b in pairs]) if pairs else None


# ── report ───────────────────────────────────────────────────────────────────

def evaluate(dataset: dict, control: dict, rag_runs: dict[str, dict], analyst_conds: set[str],
             subset: list[str] | None = None, use_embeddings: bool = True, n_boot: int = 2000) -> dict:
    sim = Similarity(use_embeddings)
    report = {"similarity_method": sim.method, "subset_size": len(subset) if subset is not None else None,
              "conditions": {}}
    for cid, runs in rag_runs.items():
        pids = sorted(p for p in runs if p in control and p in dataset and (subset is None or p in subset))
        print(f"evaluating C{cid}: {len(pids)} paired paper(s), {n_boot} bootstrap draws...", flush=True)
        recs = {p: paper_rag_record(runs[p], cid in analyst_conds) for p in pids}
        tot = lambda key: sum(r[key] for r in recs.values())           # noqa: E731

        # control mentions of the same shown works (training-knowledge baseline)
        ctl_mentions = 0
        for p in pids:
            shown = shown_works(runs[p].get("rag_package") or {})
            text = "\n".join(review_free_text(r) for r in valid_reviews(control[p]))
            ctl_mentions += sum(1 for w in shown if text_mentions(text, w))

        # detection vs humans
        tp = fp = fn = tn = exact_hits = 0
        sims = []
        for p in pids:
            human = human_rw_weaknesses(dataset[p])
            flagged = recs[p]["missing_items"]
            h, s = bool(human), bool(flagged)
            tp += h and s; fp += s and not h; fn += h and not s; tn += not h and not s
            if h and s:
                joined = "\n".join(human)
                exact_hits += any(text_mentions(joined, it["work"]) for it in flagged)
                value = sim.best_match([it["why"] or it["work"].get("title", "") for it in flagged], human)
                if value is not None:
                    sims.append(value)

        nov_entries = tot("novelty_entries")
        report["conditions"][cid] = {
            "papers": len(pids),
            "usage": {
                "mean_shown": st.mean([r["num_shown"] for r in recs.values()]) if recs else None,
                "mean_shown_uncited": st.mean([r["num_shown_uncited"] for r in recs.values()]) if recs else None,
                "novelty_entries": nov_entries,
                "novelty_names_shown_work": tot("novelty_named") / nov_entries if nov_entries else None,
                "named_works_cited_by_manuscript": tot("named_cited"),
                "named_works_not_cited": tot("named_uncited"),
                "reviews_with_missing_prior_work": (tot("reviews_with_missing") / tot("reviews")) if tot("reviews") else None,
                "free_text_mentions_per_paper": tot("free_text_mentions") / len(pids) if pids else None,
                "control_mentions_per_paper": ctl_mentions / len(pids) if pids else None,
            },
            "detection_vs_human": {
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "agreement": (tp + tn) / len(pids) if pids else None,
                "papers_with_exact_hit": exact_hits,
                "mean_best_similarity": st.mean(sims) if sims else None,
            },
            "citation_validity": {
                "invalid_labels": tot("invalid_labels"),
                "free_text_unknown_labels": tot("free_text_bad_labels"),
                "flagged_cited_or_self": tot("flagged_cited_or_self"),
            },
            "paired_vs_control": {
                "accuracy_rag": bootstrap(pids, lambda ps: accuracy(runs, dataset, ps), n_boot),
                "accuracy_control": bootstrap(pids, lambda ps: accuracy(control, dataset, ps), n_boot),
                "accuracy_diff": bootstrap(pids, lambda ps: _diff(accuracy(runs, dataset, ps),
                                                                  accuracy(control, dataset, ps)), n_boot),
                "rho_rag": bootstrap(pids, lambda ps: rho(runs, dataset, ps), n_boot),
                "rho_control": bootstrap(pids, lambda ps: rho(control, dataset, ps), n_boot),
                "rho_diff": bootstrap(pids, lambda ps: _diff(rho(runs, dataset, ps), rho(control, dataset, ps)), n_boot),
                "novelty_shift": bootstrap(pids, lambda ps: _mean_diff(runs, control, ps), n_boot),
                "decision_agreement_with_control": _agreement(runs, control, pids),
            },
        }
    return report


def _diff(a, b):
    return None if a is None or b is None else a - b


def _mean_diff(runs, control, pids):
    vals = [_diff(novelty_score(runs[p]), novelty_score(control[p])) for p in pids]
    vals = [v for v in vals if v is not None]
    return st.mean(vals) if vals else None


def _agreement(runs, control, pids):
    pairs = [(system_decision(runs[p]), system_decision(control[p])) for p in pids]
    pairs = [(a, b) for a, b in pairs if a and b]
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else None


def _fmt(value, pct=False):
    if value is None:
        return "n/a"
    if isinstance(value, dict):
        ci = value.get("ci95")
        base = _fmt(value.get("value"), pct)
        return base + (f" [{_fmt(ci[0], pct)}, {_fmt(ci[1], pct)}]" if ci else "")
    return f"{100 * value:.1f}%" if pct else f"{value:.3f}" if isinstance(value, float) else str(value)


def print_report(report: dict) -> None:
    print(f"similarity: {report['similarity_method']}"
          + (f"; post-cutoff subset: {report['subset_size']} papers" if report["subset_size"] is not None else ""))
    for cid, c in report["conditions"].items():
        u, d, v, p = c["usage"], c["detection_vs_human"], c["citation_validity"], c["paired_vs_control"]
        print(f"\n=== C{cid} vs control ({c['papers']} papers)")
        print(f"usage: shown {_fmt(u['mean_shown'])} works/paper ({_fmt(u['mean_shown_uncited'])} uncited); "
              f"novelty_check names a shown work {_fmt(u['novelty_names_shown_work'], True)} "
              f"(cited {u['named_works_cited_by_manuscript']}, uncited {u['named_works_not_cited']}); "
              f"reviews with missing prior work {_fmt(u['reviews_with_missing_prior_work'], True)}")
        print(f"       free-text mentions of shown works per paper: {_fmt(u['free_text_mentions_per_paper'])} "
              f"(control {_fmt(u['control_mentions_per_paper'])})")
        print(f"detection vs human related-work weaknesses: precision {_fmt(d['precision'], True)}, "
              f"recall {_fmt(d['recall'], True)}, agreement {_fmt(d['agreement'], True)}, "
              f"exact hits {d['papers_with_exact_hit']}, similarity {_fmt(d['mean_best_similarity'])}")
        print(f"citation validity: invalid labels {v['invalid_labels']}, unknown [R#] in text "
              f"{v['free_text_unknown_labels']}, flagged cited/self {v['flagged_cited_or_self']}")
        print(f"accuracy {_fmt(p['accuracy_rag'], True)} vs {_fmt(p['accuracy_control'], True)} "
              f"(diff {_fmt(p['accuracy_diff'], True)})")
        print(f"rho {_fmt(p['rho_rag'])} vs {_fmt(p['rho_control'])} (diff {_fmt(p['rho_diff'])})")
        print(f"novelty score shift {_fmt(p['novelty_shift'])}; decision agreement with control "
              f"{_fmt(p['decision_agreement_with_control'], True)}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="eval/openreview_60_module_test.json")
    parser.add_argument("--summary", required=True, help="Experiment summary holding the RAG conditions.")
    parser.add_argument("--control_summary", default=None, help="Summary holding the control (default: --summary).")
    parser.add_argument("--control_cond", default="1")
    parser.add_argument("--rag_conds", nargs="+", default=["8", "9"])
    parser.add_argument("--analyst_conds", nargs="*", default=["9"])
    parser.add_argument("--subset", default=None, help="Output of eval/first_public_dates.py (uses its post-cutoff list).")
    parser.add_argument("--no_embeddings", action="store_true", help="Use token overlap instead of MiniLM.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", default="eval/rag_eval_report.json")
    args = parser.parse_args(argv)

    dataset = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))
    if isinstance(dataset, list):
        dataset = {item["paper_id"]: item for item in dataset}
    print(f"loading results from {args.summary}...", flush=True)
    rag_runs = load_conditions(ROOT / args.summary, args.rag_conds)
    control = load_conditions(ROOT / (args.control_summary or args.summary), [args.control_cond])[args.control_cond]
    subset = None
    if args.subset:
        subset = json.loads((ROOT / args.subset).read_text(encoding="utf-8"))["subset"]["post_cutoff"]
    report = evaluate(dataset, control, rag_runs, set(args.analyst_conds), subset,
                      use_embeddings=not args.no_embeddings, n_boot=args.bootstrap)
    print_report(report)
    (ROOT / args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
