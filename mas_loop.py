import os
import yaml

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
    reviewer_types = ["reviewer_a", "reviewer_b"]
    reviewers = [Reviewer(paper=paper, reviewer_type=rt) for rt in reviewer_types]
    author = Author(paper=paper)
    ai_detect = AIDetector(paper=paper)

    # n_iter x n_reviewers grid to store all reviews
    reviews = [[None] * len(reviewers) for _ in range(n_iter)]

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
            reviewer_prompt = construct_reviewer_prompt(author_resp, aicheck_resp)
            reviews = get_review(reviewers[i], reviewer_prompt, iteration, i, reviews)

    return reviews


if __name__ == "__main__":
    main(paper="")
