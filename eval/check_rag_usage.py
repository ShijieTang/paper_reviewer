"""Did reviewers use the related work that RAG put in their prompt?

For each paper, take the ranked related works the reviewer was shown (top 8 of the stored
rag_package, excluding the submission's own preprint) and count how often reviews mention
them, by title or by first-author citation ("Smith et al.", "Smith (2023)").

Control: the no-RAG C1 run on the same papers, checked against the same list of works,
measures how often reviewers mention those papers anyway (from training knowledge).
For RAG runs with several rounds, mentions are also counted per round to see whether use
fades as the conversation grows (the RAG block sits after the paper in the first message).

Usage (from code/):  python eval/check_rag_usage.py
"""
from __future__ import annotations

import difflib
import json
import re
import statistics as st
from pathlib import Path

EVAL = Path(__file__).resolve().parent
RUNS = {
    # model: (RAG run summary, condition id, no-RAG control summary, condition id)
    "DeepSeek V3.1": ("exp_results_cond7_rag60/experiment_summary_2607261657.json", "7",
                      "exp_results_cond1_deepseek_60_v2/experiment_summary_2607262052.json", "1"),
    "Claude Haiku 4.5": ("exp_results_cond7_haiku45_60/experiment_summary_2607261724.json", "7",
                         "exp_results_cond1_haiku45_60/experiment_summary_2607261801.json", "1"),
}
MAX_SHOWN = 8           # the reviewer prompt lists up to eight ranked papers
LIT_WORDS = re.compile(r"\b(prior work|related work|existing work|recent work|previous work|concurrent work|"
                       r"literature|state[- ]of[- ]the[- ]art|baselines? such as|compared to \w+ et al)\b", re.I)
EVIDENCE_WORDS = re.compile(r"\b(rag evidence|related[- ]work (summary|evidence|block|package)|retrieved (papers?|works?|evidence|literature)|"
                            r"provided (related work|evidence|literature)|supplied (related work|evidence))\b", re.I)
COMMON_TITLE_WORDS = {"learning", "language", "models", "model", "neural", "networks", "deep", "large", "towards",
                      "toward", "efficient", "via", "with", "for", "the", "and", "on", "of", "a", "an", "in"}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())).strip()


def review_text(review: dict) -> str:
    parts = [review.get("summary_comment", "")]
    for key in ("strengths", "weaknesses"):
        parts += [x for x in review.get(key, []) if isinstance(x, str)]
    for detail in review.get("weakness_details", []) or []:
        if isinstance(detail, dict):
            parts += [str(v) for v in detail.values() if isinstance(v, str)]
    for gate in (review.get("acceptance_gates") or {}).values():
        if isinstance(gate, dict):
            parts.append(str(gate.get("justification", "")))
    return "\n".join(parts)


def as_review(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(re.sub(r"^```(?:json)?|```$", "", value.strip()).strip())
            return parsed if isinstance(parsed, dict) else {"summary_comment": value}
        except json.JSONDecodeError:
            return {"summary_comment": value}
    return {}


def novelty_text(review: dict) -> str:
    gate = (review.get("acceptance_gates") or {}).get("novelty") or {}
    return str(gate.get("justification", "")) if isinstance(gate, dict) else ""


def shown_works(package: dict) -> list[dict]:
    meta = {m["paper_id"]: m for m in package.get("paper_metadata", [])}
    target = norm((package.get("target_paper_summary") or {}).get("title", ""))
    works = []
    for item in sorted(package.get("reranking_results", []), key=lambda r: r.get("rank", 99)):
        m = meta.get(item.get("paper_id"))
        if not m:
            continue
        works.append(m)
        if len(works) >= MAX_SHOWN:
            break
    # drop the submission's own preprint (self-retrieval)
    return [w for w in works if difflib.SequenceMatcher(None, target, norm(w.get("title", ""))).ratio() < 0.85]


def matchers(work: dict):
    """Patterns that identify a mention of one work."""
    pats = []
    title = norm(work.get("title", ""))
    if len(title.split()) >= 4:
        pats.append(re.compile(r"\b" + re.escape(title) + r"\b"))
    head = (work.get("title") or "").split(":")[0].strip()
    if 4 <= len(head) <= 20 and " " not in head and re.search(r"[A-Z].*[A-Z]|\d", head) \
            and head.lower() not in COMMON_TITLE_WORDS:
        pats.append(re.compile(r"\b" + re.escape(head.lower()) + r"\b"))          # e.g. 'RULER', 'LoRA'
    authors = work.get("authors") or []
    first = authors[0] if authors else ""
    first = first if isinstance(first, str) else first.get("name", "")
    last = first.split()[-1] if first.split() else ""
    year = work.get("year")
    if len(last) >= 3 and isinstance(year, int):
        years = "|".join(str(y) for y in (year - 1, year, year + 1))
        pats.append(re.compile(r"\b" + re.escape(last.lower()) + r"\b(\s+et\s+al\.?)?\s*,?\s*\(?\s*(" + years + r")\b"))
    return pats


def mentions(text: str, works: list[dict]) -> int:
    low = text.lower()
    lown = norm(text)
    count = 0
    for work in works:
        if any(p.search(low) or p.search(lown) for p in matchers(work)):
            count += 1
    return count


def load(summary_path: str, cond: str):
    path = EVAL / summary_path
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for paper in data["papers"]:
        c = paper.get("conditions", {}).get(cond)
        if not c:
            continue
        result = c.get("result") or {}
        rf = path.parent / (c.get("result_file") or "")
        if rf.is_file():
            try:
                result = json.loads(rf.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        if result.get("reviewers"):
            out[paper["paper_id"]] = result
    return out


def pct(values):
    return f"{100 * st.mean(values):5.1f}%" if values else "  n/a"


def main() -> None:
    report = {}
    for model, (rag_path, rag_c, ctl_path, ctl_c) in RUNS.items():
        rag, ctl = load(rag_path, rag_c), load(ctl_path, ctl_c)
        papers = sorted(p for p in rag if p in ctl and rag[p].get("rag_package"))
        rows = {"rag_any": [], "ctl_any": [], "rag_n": [], "ctl_n": [], "rag_nov": [], "ctl_nov": [],
                "rag_lit": [], "ctl_lit": [], "rag_evid": [], "ctl_evid": [], "shown": [], "self_retrieved": 0, "rounds": {}}
        examples = []
        for pid in papers:
            package = rag[pid]["rag_package"]
            all_shown = sorted(package.get("reranking_results", []), key=lambda r: r.get("rank", 99))[:MAX_SHOWN]
            works = shown_works(package)
            rows["self_retrieved"] += int(len(works) < min(len(all_shown), MAX_SHOWN))
            rows["shown"].append(len(works))
            r_text = "\n".join(review_text(r) for r in rag[pid]["reviewers"])
            c_text = "\n".join(review_text(r) for r in ctl[pid]["reviewers"])
            n_r, n_c = mentions(r_text, works), mentions(c_text, works)
            rows["rag_any"].append(n_r > 0); rows["ctl_any"].append(n_c > 0)
            rows["rag_n"].append(n_r); rows["ctl_n"].append(n_c)
            rows["rag_nov"].append(mentions(" ".join(novelty_text(r) for r in rag[pid]["reviewers"]), works) > 0)
            rows["ctl_nov"].append(mentions(" ".join(novelty_text(r) for r in ctl[pid]["reviewers"]), works) > 0)
            rows["rag_lit"].append(bool(LIT_WORDS.search(r_text))); rows["ctl_lit"].append(bool(LIT_WORDS.search(c_text)))
            rows["rag_evid"].append(bool(EVIDENCE_WORDS.search(r_text))); rows["ctl_evid"].append(bool(EVIDENCE_WORDS.search(c_text)))
            for it in rag[pid].get("iterations", []):
                txt = "\n".join(review_text(as_review(r.get("review"))) for r in it.get("reviewers", []) if isinstance(r, dict))
                if txt.strip():
                    rows["rounds"].setdefault(it.get("iteration"), []).append(mentions(txt, works) > 0)
            if n_r:
                for work in works:
                    for sent in re.split(r"(?<=[.;])\s+", r_text):
                        if any(p.search(sent.lower()) or p.search(norm(sent)) for p in matchers(work)):
                            examples.append((pid[:60], work.get("title", "")[:80], sent.strip()[:240]))
                            break
        report[model] = {"papers": len(papers), "rows": rows, "examples": examples}

        print(f"\n=== {model}: {len(papers)} papers with both a RAG run (C{rag_c}) and a no-RAG control (C{ctl_c})")
        print(f"related works shown per paper (excluding self): mean {st.mean(rows['shown']):.1f}; "
              f"papers whose package included the submission itself: {rows['self_retrieved']}")
        print(f"{'':48s}{'with RAG':>10s}{'no RAG':>10s}")
        print(f"{'final review mentions >=1 shown work':48s}{pct(rows['rag_any']):>10s}{pct(rows['ctl_any']):>10s}")
        print(f"{'mean # shown works mentioned':48s}{st.mean(rows['rag_n']):10.2f}{st.mean(rows['ctl_n']):10.2f}")
        print(f"{'novelty justification cites a shown work':48s}{pct(rows['rag_nov']):>10s}{pct(rows['ctl_nov']):>10s}")
        print(f"{'review refers to the supplied related work':48s}{pct(rows['rag_evid']):>10s}{pct(rows['ctl_evid']):>10s}")
        print(f"{'review uses literature language':48s}{pct(rows['rag_lit']):>10s}{pct(rows['ctl_lit']):>10s}")
        if rows["rounds"]:
            print("RAG run, papers whose review mentions >=1 shown work, by round: " +
                  ", ".join(f"round {k}: {pct(v).strip()}" for k, v in sorted(rows["rounds"].items())))
        for pid, title, sent in examples:
            print(f"  e.g. [{pid}] cites '{title}': \"{sent}\"")

    out = EVAL / "rag_usage_check.json"
    out.write_text(json.dumps({m: {"papers": r["papers"],
                                   **{k: (st.mean(v) if isinstance(v, list) and v else v) for k, v in r["rows"].items()
                                      if k != "rounds"},
                                   "by_round": {str(k): st.mean(v) for k, v in r["rows"]["rounds"].items()},
                                   "examples": r["examples"]} for m, r in report.items()}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
