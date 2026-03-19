import argparse
import json
import re

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
         on_event=None) -> dict:
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

    # ── Init agents ──────────────────────────────────────────────────────────
    emit("Initializing agents...")
    reviewers  = [Reviewer(paper=paper, reviewer_type=rt,
                           topic=topic, api_key=api_key)
                  for rt in reviewer_types]
    author     = Author(paper=paper, topic=topic, api_key=api_key)
    ai_detect  = AIDetector(paper=paper, topic=topic, api_key=api_key)
    conf_rec   = ConferenceRecommender(paper=paper, topic=topic, api_key=api_key)
    emit(f"Initialized {len(reviewers)} reviewer(s), AI Author, AI Detector, "
         f"Conference Recommender.")

    # ── Storage ──────────────────────────────────────────────────────────────
    reviews       = [[None] * len(reviewers) for _ in range(n_iter)]
    author_resps  = [[None] * len(reviewers) for _ in range(n_iter)]
    aicheck_resps = [[None] * len(reviewers) for _ in range(n_iter)]

    # ── Iteration 0: initial reviews ─────────────────────────────────────────
    emit(f"--- Iteration 1 / {n_iter}: Initial Reviews ---")
    init_prompt = "Based on the given paper and your persona, provide your initial review."
    for i, reviewer in enumerate(reviewers):
        emit(f"{reviewer.name} is writing initial review...")
        reviews = get_review(reviewer, init_prompt, 0, i, reviews)
        emit(f"{reviewer.name} completed initial review.")

    # ── Iterations 1..n_iter-1: rebuttal loop ────────────────────────────────
    for iteration in range(1, n_iter):
        emit(f"--- Iteration {iteration + 1} / {n_iter} ---")
        for i, reviewer in enumerate(reviewers):
            review = reviews[iteration - 1][i]
            emit(f"AI Author writing rebuttal to {reviewer.name}...")
            author_resp = author.call(review)
            author_resps[iteration][i] = author_resp
            emit(f"AI Detector checking review from {reviewer.name}...")
            aicheck_resp = ai_detect.call(review)
            aicheck_resps[iteration][i] = aicheck_resp
            reviewer_prompt = construct_reviewer_prompt(author_resp, aicheck_resp)
            emit(f"{reviewer.name} updating review based on rebuttal...")
            reviews = get_review(reviewer, reviewer_prompt, iteration, i, reviews)
            emit(f"{reviewer.name} completed iteration {iteration + 1} review.")

    # ── Conference recommendation ─────────────────────────────────────────────
    emit("Generating conference recommendation...")
    final_reviews = reviews[n_iter - 1]
    conf_prompt   = construct_conf_rec_prompt(topic, final_reviews)
    conf_rec_resp = conf_rec.call(conf_prompt)
    emit("Conference recommendation complete!")

    # ── Citation check ────────────────────────────────────────────────────────
    emit("Running citation check...")
    citation_results = _run_citation_check(paper)
    n_fail = len(citation_results.get("failed", []))
    emit(f"Citation check complete. {n_fail} reference(s) flagged.")

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
