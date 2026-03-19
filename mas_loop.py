import argparse

from agents import Reviewer, Author, AIDetector, ConferenceRecommender
from prompts.reviewer_iter import reviewer_iteration


def construct_reviewer_prompt(author_resp: str, aicheck_resp: str) -> str:
    return (
        f"###AUTHOR_RESPONSE###\n{author_resp}\n\n"
        f"###AICHECKER_RESPONSE###\n{aicheck_resp}\n\n"
        f"###TASK###\n{reviewer_iteration}"
    )


def construct_conf_rec_prompt(topic: str, reviews: list) -> str:
    """Build the final prompt for the conference recommender."""
    review_block = "\n\n".join(
        f"Reviewer {i+1}:\n{r}" for i, r in enumerate(reviews) if r
    )
    return (
        f"Paper topic: {topic}\n\n"
        f"###REVIEWER_SCORES_AND_COMMENTS###\n{review_block}"
    )


def get_review(reviewer: Reviewer, reviewer_prompt: str, iteration: int, reviewer_ind: int, reviews: list) -> list:
    review = reviewer.call(reviewer_prompt)
    reviews[iteration][reviewer_ind] = review
    return reviews


def main(paper: str, topic: str = "", n_iter: int = 10):
    print("================= Initialization =================")
    reviewer_types = ["reviewer_a", "reviewer_b"]
    reviewers  = [Reviewer(paper=paper, reviewer_type=rt, topic=topic) for rt in reviewer_types]
    author     = Author(paper=paper, topic=topic)
    ai_detect  = AIDetector(paper=paper, topic=topic)
    conf_rec   = ConferenceRecommender(paper=paper, topic=topic)

    # n_iter x n_reviewers grids
    reviews = [[None] * len(reviewers) for _ in range(n_iter)]
    author_resps = [[None] * len(reviewers) for _ in range(n_iter)]
    aicheck_resps = [[None] * len(reviewers) for _ in range(n_iter)]

    # iteration 0: initial reviews
    print("\n================= Iteration 1 =================")
    init_prompt = "Based on the given paper and your persona, provide your initial review."
    for i in range(len(reviewers)):
        reviews = get_review(reviewers[i], init_prompt, 0, i, reviews)

    # iterations 1..n_iter-1: author rebuts, ai checks, reviewers update
    for iteration in range(1, n_iter):
        print(f"\n================= Iteration {iteration+1} =================")
        for i in range(len(reviewers)):
            review = reviews[iteration - 1][i]
            print("\nAuthor rebutting...")
            author_resp = author.call(review)
            print("\n\AI detector checking...")
            aicheck_resp = ai_detect.call(review)
            author_resps[iteration][i] = author_resp
            aicheck_resps[iteration][i] = aicheck_resp
            reviewer_prompt = construct_reviewer_prompt(author_resp, aicheck_resp)
            print("\nReviewers updating...")
            reviews = get_review(reviewers[i], reviewer_prompt, iteration, i, reviews)

    # Final step: conference recommendation using all last-iteration reviews
    print("\n================= Conference Recommendation =================")
    final_reviews = reviews[n_iter - 1]
    conf_prompt   = construct_conf_rec_prompt(topic, final_reviews)
    conf_rec_resp = conf_rec.call(conf_prompt)

    return reviews, author_resps, aicheck_resps, conf_rec_resp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the multi-agent paper review loop.")
    parser.add_argument("--paper",  default="data/md/example_paper.md", help="Path to the paper markdown file.")
    parser.add_argument("--topic",  default="",  help="Paper topic: 'Machine Learning Algorithm', 'NLP', or 'AI for Science'.")
    parser.add_argument("--n_iter", type=int, default=10, help="Number of review iterations.")
    parser.add_argument("--output", default=None, help="Path to save the output (optional).")
    args = parser.parse_args()

    with open(args.paper, "r", encoding="utf-8") as f:
        paper = f.read()

    reviews, author_resps, aicheck_resps, conf_rec_resp = main(
        paper=paper, topic=args.topic, n_iter=args.n_iter
    )

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

    lines.append(f"\n{'='*60}")
    lines.append("CONFERENCE RECOMMENDATION")
    lines.append(f"{'='*60}")
    lines.append(conf_rec_resp)

    output = "\n".join(lines)
    print(output)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\nOutput saved to {args.output}")
