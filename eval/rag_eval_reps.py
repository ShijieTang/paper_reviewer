#!/usr/bin/env python3
"""Compare conditions over repeated runs on the same papers (mean across runs, paper bootstrap).

Each condition is given one or more runs. A run is an experiment summary (or a folder holding one;
the latest experiment_summary_*.json is used) plus the condition id inside it:

    python eval/rag_eval_reps.py \\
        --run C1=eval/exp_results_rag_improve:1  --run C1=eval/exp_results_reps/rep2:1  --run C1=eval/exp_results_reps/rep3:1 \\
        --run C8=eval/exp_results_rag_improve_v3:8 --run C8=eval/exp_results_reps/rep2:8 --run C8=eval/exp_results_reps/rep3:8 \\
        --run C10=eval/exp_results_rag_improve_v4:10 --run C10=eval/exp_results_reps/rep2:10 --run C10=eval/exp_results_reps/rep3:10 \\
        --control C1 --subscores eval/review_subscores_60.json

Only papers with a completed result in every run of every listed condition are used, so all
conditions are compared on the same papers. For each condition it reports, per run and as the
mean +- sample SD across runs:
  * decision accuracy, Spearman rho with the venue-normalized human score;
  * Spearman rho of the novelty score with the human contribution/originality rating (--subscores);
  * mean novelty score; share of papers flagged with missing prior work;
  * decision consistency: papers with the same decision in every run.
Differences from --control use a paired paper bootstrap: papers are resampled, every run of a
resampled paper stays with it, and the statistic is the mean over runs (as in the manuscript).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import rag_eval as R  # noqa: E402


def resolve_summary(path: str) -> Path:
    p = ROOT / path
    if p.is_dir():
        found = sorted(p.glob("experiment_summary_*.json"))
        if not found:
            raise SystemExit(f"No experiment_summary_*.json in {path}")
        return found[-1]
    return p


def load_runs(specs: list[str]) -> dict[str, list[dict]]:
    """{label: [ {paper_id: result}, ... one dict per run ]}"""
    runs: dict[str, list[dict]] = {}
    for spec in specs:
        label, _, rest = spec.partition("=")
        path, _, cond = rest.rpartition(":")
        if not (label and path and cond):
            raise SystemExit(f"--run must look like LABEL=PATH:COND, got {spec!r}")
        summary = resolve_summary(path)
        results = R.load_conditions(summary, [cond])[cond]
        complete = {p: r for p, r in results.items() if r.get("workflow_status") in (None, "complete")
                    and not r.get("turn_failures")}
        print(f"{label}: {summary} condition {cond}: {len(complete)} completed paper(s)", flush=True)
        runs.setdefault(label, []).append(complete)
    return runs


def flagged(result: dict) -> bool:
    return any(R.valid_reviews(result)[0].get("missing_prior_work") or []) if R.valid_reviews(result) else False


def metric(run: dict, dataset: dict, subscores: dict | None, pids: list[str], key: str):
    if key == "accuracy":
        return R.accuracy(run, dataset, pids)
    if key == "rho":
        return R.rho(run, dataset, pids)
    if key == "novelty_rho_human":
        if not subscores:
            return None
        have = [p for p in pids if R.human_dimension(subscores, p, "novelty")[0] is not None]
        return R.dimension_rho(run, subscores, have, "novelty")
    if key == "mean_novelty":
        nov = [v for v in (R.novelty_score(run[p]) for p in pids) if v is not None]
        return st.mean(nov) if nov else None
    if key == "flag_rate":
        return sum(flagged(run[p]) for p in pids) / len(pids) if pids else None
    raise KeyError(key)


def metrics(run: dict, dataset: dict, subscores: dict | None, pids: list[str]) -> dict:
    return {key: metric(run, dataset, subscores, pids, key) for key in KEYS}


KEYS = ("accuracy", "rho", "novelty_rho_human", "mean_novelty", "flag_rate")


def mean_over_runs(label_runs: list[dict], dataset: dict, subscores, pids: list[str], key: str):
    vals = [metric(run, dataset, subscores, pids, key) for run in label_runs]
    vals = [v for v in vals if v is not None]
    return st.mean(vals) if vals else None


def consistency(label_runs: list[dict], pids: list[str]) -> float | None:
    if len(label_runs) < 2:
        return None
    same = sum(len({R.system_decision(run[p]) for run in label_runs}) == 1 for p in pids)
    return same / len(pids)


def paired_bootstrap(a_runs, b_runs, dataset, subscores, pids, key, n=2000, seed=0) -> dict:
    stat = lambda ps: R._diff(mean_over_runs(a_runs, dataset, subscores, ps, key),   # noqa: E731
                              mean_over_runs(b_runs, dataset, subscores, ps, key))
    point = stat(pids)
    rng = random.Random(seed)
    draws = sorted(v for v in (stat([rng.choice(pids) for _ in pids]) for _ in range(n)) if v is not None)
    ci = [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]] if draws else None
    return {"value": point, "ci95": ci}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="eval/openreview_60_module_test.json")
    parser.add_argument("--run", action="append", required=True, help="LABEL=SUMMARY_OR_FOLDER:COND (repeatable)")
    parser.add_argument("--control", required=True, help="Label of the control condition, e.g. C1")
    parser.add_argument("--subscores", default=None, help="Output of eval/fetch_review_subscores.py")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", default="eval/rag_eval_reps_report.json")
    args = parser.parse_args(argv)

    dataset = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))
    subscores = json.loads((ROOT / args.subscores).read_text(encoding="utf-8")) if args.subscores else None
    runs = load_runs(args.run)
    if args.control not in runs:
        raise SystemExit(f"--control {args.control} is not among the --run labels {sorted(runs)}")
    pids = sorted(set.intersection(*(set(run) for label_runs in runs.values() for run in label_runs)) & set(dataset))
    print(f"\n{len(pids)} paper(s) completed in every run of every condition\n", flush=True)

    keys = KEYS
    report = {"papers": len(pids), "conditions": {}}
    for label, label_runs in runs.items():
        per_run = [metrics(run, dataset, subscores, pids) for run in label_runs]
        summary = {}
        for key in keys:
            vals = [m[key] for m in per_run if m[key] is not None]
            summary[key] = {"per_run": vals, "mean": st.mean(vals) if vals else None,
                            "sd": st.stdev(vals) if len(vals) > 1 else None}
        entry = {"runs": len(label_runs), "metrics": summary, "decision_consistency": consistency(label_runs, pids)}
        if label != args.control:
            print(f"bootstrapping {label} vs {args.control}...", flush=True)
            entry["vs_control"] = {key: paired_bootstrap(label_runs, runs[args.control], dataset, subscores,
                                                         pids, key, args.bootstrap)
                                   for key in ("accuracy", "rho", "novelty_rho_human", "mean_novelty")}
        report["conditions"][label] = entry

    fmt = lambda v, pct=False: "n/a" if v is None else (f"{100 * v:.1f}%" if pct else f"{v:.3f}")   # noqa: E731
    pct_keys = {"accuracy", "flag_rate"}
    for label, entry in report["conditions"].items():
        print(f"\n=== {label} ({entry['runs']} run(s))")
        for key in keys:
            m = entry["metrics"][key]
            runs_txt = ", ".join(fmt(v, key in pct_keys) for v in m["per_run"])
            sd = f" +- {fmt(m['sd'], key in pct_keys)}" if m["sd"] is not None else ""
            print(f"  {key:18s} mean {fmt(m['mean'], key in pct_keys)}{sd}   runs: {runs_txt}")
        if entry["decision_consistency"] is not None:
            print(f"  same decision in all runs: {fmt(entry['decision_consistency'], True)}")
        for key, d in (entry.get("vs_control") or {}).items():
            ci = f" [{fmt(d['ci95'][0], key in pct_keys)}, {fmt(d['ci95'][1], key in pct_keys)}]" if d["ci95"] else ""
            print(f"  vs {args.control}: {key:18s} {fmt(d['value'], key in pct_keys)}{ci}")
    (ROOT / args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
