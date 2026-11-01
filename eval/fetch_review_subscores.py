#!/usr/bin/env python3
"""Download the human reviewers' per-dimension ratings from OpenReview for each dataset paper.

The datasets keep only each review's overall rating. Several venues also ask reviewers for
dimension ratings, which let us test the system's novelty score directly instead of through the
accept/reject decision. As far as we know (the script records whatever fields it finds, so check
the output): ICLR 2025 asks for soundness, presentation and contribution; NeurIPS 2025 for quality,
clarity, significance and originality; ICML 2025 uses a different form.

Every numeric field of each official review is stored, so nothing depends on guessing the form.
eval/rag_eval.py --subscores then maps them onto the system's score dimensions (see DIMENSION_MAP).

If OpenReview refuses anonymous requests (HTTP 401/403), log in with your OpenReview account:
    export OPENREVIEW_USERNAME=you@university.edu OPENREVIEW_PASSWORD=...
The token is used only for these requests and is not stored.

Usage (from the repository root; needs network access):
    python eval/fetch_review_subscores.py \\
        --dataset eval/openreview_60_module_test.json \\
        --output eval/review_subscores_60.json

Responses are cached in eval/.cache/openreview_reviews/, so reruns are free and resume.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
import re
import statistics as st
import sys
import urllib.parse
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from first_public_dates import Fetcher, forum_id  # noqa: E402

API_V2 = "https://api2.openreview.net/notes"
API_V1 = "https://api.openreview.net/notes"
SKIP_FIELDS = {"confidence", "code_of_conduct", "flag_for_ethics_review", "first_time_reviewer"}


def field_value(raw):
    """Number at the start of a rating field: 3, '3', '3: good', {'value': '3: good'} -> 3.0."""
    if isinstance(raw, dict):
        raw = raw.get("value")
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        match = re.match(r"\s*(-?\d+(?:\.\d+)?)\b", raw)
        return float(match.group(1)) if match else None
    return None


def is_official_review(note: dict) -> bool:
    invitations = note.get("invitations") or [note.get("invitation") or ""]
    return any("Official_Review" in str(inv) for inv in invitations)


def parse_reviews(payload: dict) -> list[dict]:
    """Numeric fields of each official review in a forum's notes."""
    reviews = []
    for note in (payload or {}).get("notes") or []:
        if not is_official_review(note):
            continue
        scores = {}
        for key, raw in (note.get("content") or {}).items():
            name = key.strip().lower()
            if name in SKIP_FIELDS:
                continue
            value = field_value(raw)
            if value is not None:
                scores[name] = value
        if scores:
            reviews.append({"id": note.get("id"), "scores": scores})
    return reviews


def mean_scores(reviews: list[dict]) -> dict:
    fields = sorted({k for r in reviews for k in r["scores"]})
    return {f: round(st.mean(r["scores"][f] for r in reviews if f in r["scores"]), 4) for f in fields}


def login_headers() -> dict:
    """Bearer token from OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD, if set."""
    user, password = os.environ.get("OPENREVIEW_USERNAME"), os.environ.get("OPENREVIEW_PASSWORD")
    if not (user and password):
        return {}
    request = urllib.request.Request(
        "https://api2.openreview.net/login", data=json.dumps({"id": user, "password": password}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "paper-reviewer-eval/1.0"}, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        token = json.loads(response.read().decode()).get("token")
    if not token:
        raise RuntimeError("OpenReview login returned no token")
    return {"Authorization": f"Bearer {token}"}


def fetch_forum(fetcher: Fetcher, fid: str, headers: dict | None = None) -> tuple[list[dict], str]:
    """Official reviews of a forum, and the last error if every attempt failed."""
    error = ""
    for api in (API_V2, API_V1):
        try:
            payload = json.loads(fetcher.get(f"{api}?forum={urllib.parse.quote(fid)}&limit=1000", headers=headers))
        except Exception as exc:
            error = f"{api.split('/')[2]}: {type(exc).__name__}: {exc}"
            continue
        reviews = parse_reviews(payload)
        if reviews:
            return reviews, ""
        error = error or f"{api.split('/')[2]}: no official reviews in {len(payload.get('notes') or [])} note(s)"
    return [], error


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="eval/openreview_60_module_test.json")
    parser.add_argument("--output", default="eval/review_subscores_60.json")
    parser.add_argument("--cache_dir", default="eval/.cache/openreview_reviews")
    args = parser.parse_args(argv)

    dataset = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))
    if isinstance(dataset, list):
        dataset = {item["paper_id"]: item for item in dataset}
    fetcher = Fetcher(ROOT / args.cache_dir, delay=1.0)
    try:
        headers = login_headers()
        print("Logged in to OpenReview." if headers else "Requesting reviews without logging in.", flush=True)
    except Exception as exc:
        print(f"OpenReview login failed: {exc}", flush=True)
        sys.exit(1)
    out, fields_by_venue, errors = {}, {}, []
    for i, (pid, meta) in enumerate(sorted(dataset.items()), 1):
        fid = forum_id(meta.get("paper_url", ""))
        reviews, error = fetch_forum(fetcher, fid, headers) if fid else ([], "no OpenReview forum id in paper_url")
        if error and not reviews:
            errors.append(error)
            print(f"   {error[:200]}", flush=True)
            if len(errors) == 3 and i == 3:
                print("The first 3 papers all failed; stopping. If the error is HTTP 401/403, set "
                      "OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD and rerun.", flush=True)
                sys.exit(1)
        venue = str(meta.get("conference", "")).upper()
        out[pid] = {"conference": venue, "forum_id": fid, "num_reviews": len(reviews),
                    "mean": mean_scores(reviews), "reviews": reviews}
        fields_by_venue.setdefault(venue, Counter()).update(out[pid]["mean"].keys())
        print(f"[{i}/{len(dataset)}] {venue:8s} {len(reviews)} review(s) "
              f"{', '.join(sorted(out[pid]['mean'])) or 'none found'}", flush=True)
    (ROOT / args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nRating fields found, by venue (number of papers):")
    for venue, counts in sorted(fields_by_venue.items()):
        print(f"  {venue}: " + ", ".join(f"{k} ({v})" for k, v in counts.most_common()))
    missing = [p for p, r in out.items() if not r["num_reviews"]]
    print(f"{len(missing)} paper(s) with no reviews found. Wrote {args.output}")


if __name__ == "__main__":
    main()
