import os
import yaml
import argparse

from agents import Reviewer, Author, AIDetector

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(key: str) -> str:
    path = os.path.join(_PROMPTS_DIR, f"{key}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)[key]


def construct_reviewer_prompt(author_resp: str, aicheck_resp: str) -> str:
    task = _load_prompt("reviewer_iteration")
    return (
        f"#AUTHOR_RESPONSE:\n{author_resp}\n\n"
        f"#AICHECKER_RESPONSE:\n{aicheck_resp}\n\n"
        f"#TASK:\n{task}"
    )


def get_review(reviewer: Reviewer, reviewer_prompt: str, iteration: int, reviewer_ind: int, reviews: list) -> list:
    review = reviewer.call(reviewer_prompt)
    reviews[iteration][reviewer_ind] = review
    return reviews


def main(paper: str, n_iter: int = 10):
    reviewer_names = ["reviewer_a", "reviewer_b"]
    reviewers = [Reviewer(paper=paper, reviewer_name=rn) for rn in reviewer_names]
    author = Author(paper=paper)
    ai_detect = AIDetector(paper=paper)

    # n_iter x n_reviewers grids
    reviews = [[None] * len(reviewers) for _ in range(n_iter)]
    author_resps = [[None] * len(reviewers) for _ in range(n_iter)]
    aicheck_resps = [[None] * len(reviewers) for _ in range(n_iter)]

    # iteration 0: initial reviews
    init_prompt = "Based on the given paper and your persona, provide your initial review."
    for i in range(len(reviewers)):
        reviews = get_review(reviewers[i], init_prompt, 0, i, reviews)

    # iterations 1..n_iter-1: author rebuts, ai checks, reviewers update
    for iteration in range(1, n_iter):
        for i in range(len(reviewers)):
            review = reviews[iteration - 1][i]
            author_resp = author.call(review)
            aicheck_resp = ai_detect.call(review)
            author_resps[iteration][i] = author_resp
            aicheck_resps[iteration][i] = aicheck_resp
            reviewer_prompt = construct_reviewer_prompt(author_resp, aicheck_resp)
            reviews = get_review(reviewers[i], reviewer_prompt, iteration, i, reviews)

    return reviews, author_resps, aicheck_resps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the multi-agent paper review loop.")
    parser.add_argument("--paper", default="data/md/example_paper.md", help="Path to the paper markdown file.")
    parser.add_argument("--n_iter", type=int, default=10, help="Number of review iterations.")
    parser.add_argument("--output", default=None, help="Path to save the output (optional).")
    args = parser.parse_args()

    with open(args.paper, "r", encoding="utf-8") as f:
        paper = f.read()

    reviews, author_resps, aicheck_resps = main(paper=paper, n_iter=args.n_iter)

    lines = []
    for iteration in range(len(reviews)):
        lines.append(f"\n{'='*60}")
        lines.append(f"ITERATION {iteration}")
        lines.append(f"{'='*60}")
        for i in range(len(reviews[iteration])):
            lines.append(f"\n--- Reviewer {i+1} ---")
            lines.append(reviews[iteration][i] or "")
            if author_resps[iteration][i]:
                lines.append(f"\n--- Author Rebuttal (to Reviewer {i+1}) ---")
                lines.append(author_resps[iteration][i])
            if aicheck_resps[iteration][i]:
                lines.append(f"\n--- AI Checker (on Reviewer {i+1}) ---")
                lines.append(aicheck_resps[iteration][i])

    output = "\n".join(lines)
    print(output)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\nOutput saved to {args.output}")
