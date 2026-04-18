import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

from agents import Reviewer, Author, AIDetector, ConferenceRecommender
from prompts.reviewer_iter import reviewer_iteration


# ── JSON helpers ────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    """Strip optional ```json fences then parse JSON."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    return json.loads(text)


# ── Citation check ───────────────────────────────────────────────────────────

def _extract_refs_section(text: str) -> str:
    """Pull the references section out of a full-paper markdown string."""
    m = re.search(r'(?im)^#+\s*references\s*$', text)
    if m:
        start = m.end()
        nxt = re.search(r'(?m)^#+\s+\S', text[start:])
        end = start + nxt.start() if nxt else len(text)
        return text[start:end].strip()
    return text.strip()


def _run_citation_check(paper_text: str) -> dict:
    """
    Run the citation checker on a paper's references section.

    Returns a dict with:
        stats   : {total, verified, url_only, not_found, suspicious, skipped}
        failed  : [{index, title, link, fail_reason}, ...]
    """
    try:
        from citation_checker import parse_references, check_references, failed_references
        from citation_checker.models import VerificationStatus

        refs_text = _extract_refs_section(paper_text)
        refs = parse_references(refs_text)
        if not refs:
            return {"stats": {}, "failed": [], "error": "No references found in paper."}

        results = check_references(refs)
        failed  = failed_references(results)

        stats = {
            "total":      len(results),
            "verified":   sum(1 for r in results if r.status == VerificationStatus.VERIFIED),
            "url_only":   sum(1 for r in results if r.status == VerificationStatus.URL_ONLY),
            "not_found":  sum(1 for r in results if r.status == VerificationStatus.NOT_FOUND),
            "suspicious": sum(1 for r in results if r.status == VerificationStatus.SUSPICIOUS),
            "skipped":    sum(1 for r in results if r.status == VerificationStatus.SKIPPED),
        }
        failed_list = [
            {"index": f.index, "title": f.title, "link": f.link, "fail_reason": f.fail_reason}
            for f in failed
        ]
        return {"stats": stats, "failed": failed_list}

    except Exception as exc:
        return {"stats": {}, "failed": [], "error": str(exc)}


# ── Loop helpers ─────────────────────────────────────────────────────────────

def construct_reviewer_prompt(author_resp: str, aicheck_resp: str) -> str:
    return (
        f"###AUTHOR_RESPONSE###\n{author_resp}\n\n"
        f"###AICHECKER_RESPONSE###\n{aicheck_resp}\n\n"
        f"###TASK###\n{reviewer_iteration}"
    )


def construct_conf_rec_prompt(topic: str, reviews: list) -> str:
    review_block = "\n\n".join(
        f"Reviewer {i+1}:\n{r}" for i, r in enumerate(reviews) if r
    )
    return (
        f"Paper topic: {topic}\n\n"
        f"###REVIEWER_SCORES_AND_COMMENTS###\n{review_block}"
    )


def get_review(reviewer: Reviewer, reviewer_prompt: str, iteration: int,
               reviewer_ind: int, reviews: list) -> list:
    review = reviewer.call(reviewer_prompt)
    reviews[iteration][reviewer_ind] = review
    return reviews


# ── Main loop ────────────────────────────────────────────────────────────────

def main(paper: str, topic: str = "", n_iter: int = 10,
         reviewer_types: list = None, api_key: str = "",
         on_event=None, on_agent_status=None, on_message=None,
         run_citation_check: bool = True) -> dict:
    """
    Run the multi-agent review loop.

    Returns a structured dict:
        {
          "reviewers"  : [ {reviewer, decision, scores, strengths,
                            weaknesses, summary_comment}, ... ],
          "conference" : { "ICML": {...}, "NeurIPS": {...}, "ICLR": {...} },
          "citations"  : { "stats": {...}, "failed": [...] },
        }
    """
    if reviewer_types is None:
        reviewer_types = ["reviewer_a", "reviewer_b"]

    def emit(msg: str):
        print(msg)
        if on_event:
            on_event(msg)

    def emit_agent_status(agent_name: str, status: str):
        if on_agent_status:
            on_agent_status(agent_name, status)

    def emit_message(agent_name: str, content: str):
        if on_message:
            on_message(agent_name, content)

    # ── Init agents ──────────────────────────────────────────────────────────
    emit("Initializing agents...")
    _type_label = {
        "reviewer_a":        "Novelty",
        "reviewer_b":        "Rigor",
        "reviewer_c":        "Practical",
        "reviewer_nopersona":"Neutral",
    }
    reviewers = []
    for i, rt in enumerate(reviewer_types, 1):
        r = Reviewer(paper=paper, reviewer_type=rt, topic=topic, api_key=api_key)
        r.name = f"Reviewer {i} ({_type_label.get(rt, rt)})"
        reviewers.append(r)
    author     = Author(paper=paper, topic=topic, api_key=api_key)
    ai_detect  = AIDetector(paper=paper, topic=topic, api_key=api_key)
    conf_rec   = ConferenceRecommender(paper=paper, topic=topic, api_key=api_key)
    emit(f"Initialized {len(reviewers)} reviewer(s), AI Author, AI Detector, "
         f"Conference Recommender.")

    # ── Storage ──────────────────────────────────────────────────────────────
    reviews       = [[None] * len(reviewers) for _ in range(n_iter)]
    author_resps  = [[None] * len(reviewers) for _ in range(n_iter)]
    aicheck_resps = [[None] * len(reviewers) for _ in range(n_iter)]

    # ── Iteration 0: initial reviews (parallel) ──────────────────────────────
    emit(f"--- Iteration 1 / {n_iter}: Initial Reviews ---")
    init_prompt = "Based on the given paper and your persona, provide your initial review."

    for r in reviewers:
        emit_agent_status(r.name, "waiting")

    def _run_initial(args):
        i, reviewer = args
        emit_agent_status(reviewer.name, "running")
        emit(f"{reviewer.name} is writing initial review...")
        review = reviewer.call(init_prompt)
        emit(f"{reviewer.name} completed initial review.")
        emit_agent_status(reviewer.name, "done")
        emit_message(reviewer.name, review)
        return i, review

    with ThreadPoolExecutor(max_workers=len(reviewers)) as ex:
        for i, review in ex.map(_run_initial, enumerate(reviewers)):
            reviews[0][i] = review

    # ── Iterations 1..n_iter-1: rebuttal loop ────────────────────────────────
    for iteration in range(1, n_iter):
        # Phase B: Author + Detector process previous reviews (sequential, shared agents)
        emit(f"--- Iteration {iteration + 1} / {n_iter}: Author & Detector Processing ---")
        for i, reviewer in enumerate(reviewers):
            emit(f"AI Author writing rebuttal to {reviewer.name}...")
            emit_agent_status(author.name, "running")
            author_resps[iteration][i] = author.call(
                f"[Reviewer name — use exactly this in your JSON: {reviewer.name}]\n\n"
                f"{reviews[iteration - 1][i]}"
            )
            emit_agent_status(author.name, "done")
            emit_message(author.name, author_resps[iteration][i])
            aicheck_resps[iteration][i] = ai_detect.call(reviews[iteration - 1][i])

        # Phase A: All reviewers update in parallel
        emit(f"--- Iteration {iteration + 1} / {n_iter}: Reviewer Updates ---")
        for r in reviewers:
            emit_agent_status(r.name, "waiting")

        def _run_update(args, _iter=iteration):
            i, reviewer = args
            emit_agent_status(reviewer.name, "running")
            reviewer_prompt = construct_reviewer_prompt(
                author_resps[_iter][i], aicheck_resps[_iter][i])
            emit(f"{reviewer.name} updating review based on rebuttal...")
            review = reviewer.call(reviewer_prompt)
            emit(f"{reviewer.name} completed iteration {_iter + 1} review.")
            emit_agent_status(reviewer.name, "done")
            emit_message(reviewer.name, review)
            return i, review

        with ThreadPoolExecutor(max_workers=len(reviewers)) as ex:
            for i, review in ex.map(_run_update, enumerate(reviewers)):
                reviews[iteration][i] = review

    # ── Conference recommendation ─────────────────────────────────────────────
    emit("Generating conference recommendation...")
    final_reviews = reviews[n_iter - 1]
    conf_prompt   = construct_conf_rec_prompt(topic, final_reviews)
    emit_agent_status(conf_rec.name, "running")
    conf_rec_resp = conf_rec.call(conf_prompt)
    emit_agent_status(conf_rec.name, "done")
    emit_message(conf_rec.name, conf_rec_resp)
    emit("Conference recommendation complete!")

    # ── Citation check ────────────────────────────────────────────────────────
    if run_citation_check:
        emit("Running citation check...")
        citation_results = _run_citation_check(paper)
        n_fail = len(citation_results.get("failed", []))
        emit(f"Citation check complete. {n_fail} reference(s) flagged.")
    else:
        emit("Skipping citation check.")
        citation_results = {
            "stats": {},
            "failed": [],
            "skipped": True,
            "reason": "disabled_for_experiment",
        }

    # ── Parse structured outputs ──────────────────────────────────────────────
    parsed_reviews = []
    for raw in final_reviews:
        if raw is None:
            continue
        try:
            parsed_reviews.append(_parse_json(raw))
        except Exception:
            parsed_reviews.append({"raw": raw, "parse_error": True})

    try:
        parsed_conf = _parse_json(conf_rec_resp)
    except Exception:
        parsed_conf = {"raw": conf_rec_resp, "parse_error": True}

    return {
        "reviewers":  parsed_reviews,
        "conference": parsed_conf,
        "citations":  citation_results,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the multi-agent paper review loop.")
    parser.add_argument("--paper",  default="data/md/example_paper.md")
    parser.add_argument("--topic",  default="")
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.paper, "r", encoding="utf-8") as f:
        paper = f.read()

    result = main(paper=paper, topic=args.topic, n_iter=args.n_iter)

    lines = []
    for i, rev in enumerate(result["reviewers"]):
        lines.append(f"\n{'='*60}\nREVIEWER {i+1}\n{'='*60}")
        lines.append(json.dumps(rev, indent=2))

    lines.append(f"\n{'='*60}\nCONFERENCE RECOMMENDATION\n{'='*60}")
    lines.append(json.dumps(result["conference"], indent=2))

    lines.append(f"\n{'='*60}\nCITATION CHECK\n{'='*60}")
    lines.append(json.dumps(result["citations"], indent=2))

    output = "\n".join(lines)
    print(output)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\nOutput saved to {args.output}")
