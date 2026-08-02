"""
run_paperreviewer.py

Two decoupled steps against the paperreview.ai API, so a crash never loses
upload progress:

    submit  Upload PDFs, get a review token back, write ONE token file per
            paper to --token_dir. This is the only step that touches S3.
    poll    Round-robin over token files, check each token's review status,
            and once ready parse it and write ONE output file per paper to
            --output_dir. Safe to stop and rerun any time — already-finished
            papers (output file present) and already-submitted-but-pending
            papers (token file present) are both skipped/resumed correctly.

Run `poll` any time after `submit` has written at least one token — it
doesn't need to wait for submit to finish the whole batch first.

Use eval/merge_paperreviewer.py afterwards to combine --output_dir into the
single {"papers": [...]} file eval/evaluation.py expects.

Flow (reverse-engineered from the site's own network calls):
    1. POST /api/get-upload-url      {filename, venue} -> S3 presigned POST
    2. POST <presigned_url>          multipart upload of the PDF bytes
    3. POST /api/confirm-upload      {s3_key, venue, email} -> {token}
    4. GET  /api/review/{token}      poll until review_date is set

Usage (run from the project root):
    python eval/run_paperreviewer.py submit --pdf_dir data/pdf
    python eval/run_paperreviewer.py poll
    python eval/run_paperreviewer.py submit --paper iclr_accept_001
    python eval/run_paperreviewer.py submit \\
        --papers_json eval/openreview_2025_300_qwen.rebalanced.icml_topup.pdf_fixed.json \\
        --pdf_dir data/openreview_pdf --proxy http://user:pass@host:port
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_URL = "https://paperreview.ai"
DEFAULT_PDF_DIR = "data/pdf"
DEFAULT_PAPERS_JSON = "eval/papers.json"
DEFAULT_TOKEN_DIR = "eval/paperreviewer_tokens"
DEFAULT_OUTPUT_DIR = "eval/paperreviewer_papers"
DEFAULT_EMAIL = "paperreviewer-bot@example.com"

DEFAULT_POLL_INTERVAL = 30      # seconds between rounds
DEFAULT_CHECK_INTERVAL = 10     # seconds between individual token checks within a round
DEFAULT_TOKEN_TIMEOUT = 60 * 60  # give up on a single token after this long
DEFAULT_ACCEPT_THRESHOLD = 6.0  # matches eval/evaluation.py's --conf_threshold default
DEFAULT_SUBMIT_INTERVAL = 15    # seconds to sleep between submissions
MAX_PDF_SIZE = 10 * 1024 * 1024  # 10MB, S3 presigned POST's hard limit on paperreview.ai
PROXY_ENV_VAR = "PAPERREVIEWER_PROXY"

HEADERS = {
    "accept": "*/*",
    "origin": BASE_URL,
    "referer": f"{BASE_URL}/",
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
    ),
}


def _proxies_from_arg(proxy: str | None) -> dict | None:
    return {"http": proxy, "https": proxy} if proxy else None


# ── paperreview.ai client ─────────────────────────────────────────────────────

def submit_paper(pdf_path: Path, venue: str, email: str, proxies: dict | None) -> str:
    """Upload a PDF and return the review token.

    proxies applies only to the paperreview.ai API calls, never to the S3
    presigned-POST upload — that goes to AWS directly.
    """
    filename = pdf_path.name

    resp = requests.post(
        f"{BASE_URL}/api/get-upload-url",
        headers={**HEADERS, "content-type": "application/json"},
        json={"filename": filename, "venue": venue},
        proxies=proxies,
        timeout=30,
    )
    resp.raise_for_status()
    upload_info = resp.json()
    if not upload_info.get("success"):
        raise RuntimeError(f"get-upload-url failed: {upload_info}")

    presigned_url = upload_info["presigned_url"]
    fields = dict(upload_info["presigned_fields"])
    s3_key = upload_info["s3_key"]

    with pdf_path.open("rb") as f:
        s3_resp = requests.post(
            presigned_url,
            data=fields,
            files={"file": (filename, f, "application/pdf")},
            timeout=120,
        )
    if s3_resp.status_code not in (200, 201, 204):
        raise RuntimeError(f"S3 upload failed ({s3_resp.status_code}): {s3_resp.text[:500]}")

    confirm_resp = requests.post(
        f"{BASE_URL}/api/confirm-upload",
        headers=HEADERS,
        data={"s3_key": s3_key, "venue": venue, "email": email},
        proxies=proxies,
        timeout=30,
    )
    confirm_resp.raise_for_status()
    confirm_info = confirm_resp.json()
    if not confirm_info.get("success") or "token" not in confirm_info:
        raise RuntimeError(f"confirm-upload failed: {confirm_info}")

    return confirm_info["token"]


def check_review(token: str, proxies: dict | None) -> dict:
    """Single, non-blocking check of /api/review/{token}."""
    resp = requests.get(f"{BASE_URL}/api/review/{token}", headers=HEADERS, proxies=proxies, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Review parser ─────────────────────────────────────────────────────────────

_BULLET_RE = re.compile(r"^\s*-\s+(.*)\S*$", re.MULTILINE)
_SCORE_LINE_RE = re.compile(r"^-\s*([A-Za-z_]+)\s*:\s*\[?([+-]?\d+)\]?", re.MULTILINE)


def _split_bullets(text: str) -> list[str]:
    return [m.group(1).strip() for m in _BULLET_RE.finditer(text or "") if m.group(1).strip()]


def _parse_binary_scores(text: str) -> dict:
    return {name: int(value) for name, value in _SCORE_LINE_RE.findall(text or "")}


def parse_review(raw: dict) -> dict:
    """Turn the /api/review/{token} JSON payload into our storage schema."""
    sections = raw.get("sections", {}) or {}

    binary_scores = _parse_binary_scores(sections.get("binary_scores", ""))

    return {
        "reviewer_id": "PaperReviewer.ai",
        "summary": sections.get("summary", ""),
        "strengths": _split_bullets(sections.get("strengths", "")),
        "weaknesses": _split_bullets(sections.get("weaknesses", "")),
        "questions": sections.get("questions", ""),
        "assessment": sections.get("assessment", ""),
        "binary_scores": binary_scores,
        "numerical_score": raw.get("numerical_score"),
    }


def build_entry(paper_id: str, title: str, review: dict, accept_threshold: float) -> dict:
    """Build the per-paper output record.

    accept_or_not/score here are paperreview.ai's OWN prediction, derived
    from its numerical_score — never copied from ground truth, since
    eval/evaluation.py compares this against eval/papers.json as the
    system's output.
    """
    entry = {"paper_id": paper_id, "title": title, "accept_or_not": None, "score": None, "reviews": [review]}
    num_score = review.get("numerical_score")
    if num_score is not None:
        entry["score"] = num_score
        entry["accept_or_not"] = "accept" if num_score >= accept_threshold else "reject"
    return entry


def _load_papers(papers_json: str, only_paper: str | None) -> dict:
    papers_path = Path(papers_json)
    if not papers_path.exists():
        print(f"Error: papers json '{papers_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    papers = json.loads(papers_path.read_text(encoding="utf-8"))
    if only_paper:
        if only_paper not in papers:
            print(f"Error: paper_id '{only_paper}' not found in {papers_path}.", file=sys.stderr)
            sys.exit(1)
        return {only_paper: papers[only_paper]}
    return papers


# ── submit ────────────────────────────────────────────────────────────────────

def cmd_submit(args):
    pdf_dir = Path(args.pdf_dir)
    token_dir = Path(args.token_dir)
    output_dir = Path(args.output_dir)
    proxies = _proxies_from_arg(args.proxy)
    papers = _load_papers(args.papers_json, args.paper)

    token_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = n_fail = 0
    for paper_id, meta in papers.items():
        output_path = output_dir / f"{paper_id}.json"
        token_path = token_dir / f"{paper_id}.json"
        if output_path.exists() or token_path.exists():
            n_skip += 1
            continue

        pdf_path = pdf_dir / f"{paper_id}.pdf"
        if not pdf_path.exists():
            print(f"[{paper_id}] Skipping: PDF not found at {pdf_path}", file=sys.stderr)
            n_fail += 1
            continue

        pdf_size = pdf_path.stat().st_size
        if pdf_size > MAX_PDF_SIZE:
            print(f"[{paper_id}] Skipping: PDF too large ({pdf_size} bytes > {MAX_PDF_SIZE}).", file=sys.stderr)
            n_fail += 1
            continue

        venue = meta.get("conference", "NeurIPS")
        title = meta.get("title", paper_id.replace("_", " ").title())
        print(f"[{paper_id}] Uploading (venue={venue})...")
        try:
            token = submit_paper(pdf_path, venue, args.email, proxies)
        except Exception as e:
            print(f"[{paper_id}] Upload failed: {e}", file=sys.stderr)
            n_fail += 1
            if args.submit_interval > 0:
                time.sleep(args.submit_interval)
            continue

        token_path.write_text(json.dumps({
            "paper_id": paper_id,
            "title": title,
            "venue": venue,
            "token": token,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{paper_id}] Token saved -> {token_path}")
        n_ok += 1

        if args.submit_interval > 0:
            time.sleep(args.submit_interval)

    print(f"\nDone. submitted={n_ok} skipped={n_skip} failed={n_fail}")


# ── poll ──────────────────────────────────────────────────────────────────────

def cmd_poll(args):
    token_dir = Path(args.token_dir)
    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)
    proxies = _proxies_from_arg(args.proxy)

    if not token_dir.is_dir():
        print(f"Error: token dir '{token_dir}' does not exist. Run `submit` first.", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    while True:
        token_files = sorted(token_dir.glob("*.json"))
        pending = []
        for token_path in token_files:
            paper_id = token_path.stem
            if (output_dir / f"{paper_id}.json").exists():
                continue
            pending.append(token_path)

        if not pending:
            print("No pending tokens left. Done.")
            return

        print(f"\n{len(pending)} pending, checking...")
        still_pending = 0
        for token_path in pending:
            info = json.loads(token_path.read_text(encoding="utf-8"))
            paper_id = info["paper_id"]
            submitted_at = datetime.fromisoformat(info["submitted_at"])
            elapsed = (datetime.now(timezone.utc) - submitted_at).total_seconds()

            try:
                raw = check_review(info["token"], proxies)
            except Exception as e:
                print(f"[{paper_id}] Check failed: {e}", file=sys.stderr)
                still_pending += 1
                if args.check_interval > 0:
                    time.sleep(args.check_interval)
                continue

            if not raw.get("review_date"):
                if elapsed > args.token_timeout:
                    print(f"[{paper_id}] Gave up after {elapsed:.0f}s (timeout={args.token_timeout}s).",
                          file=sys.stderr)
                else:
                    print(f"[{paper_id}] Not ready yet ({elapsed:.0f}s elapsed).")
                    still_pending += 1
                if args.check_interval > 0:
                    time.sleep(args.check_interval)
                continue

            raw_path = raw_dir / f"{paper_id}.json"
            raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

            review = parse_review(raw)
            entry = build_entry(paper_id, info.get("title", paper_id), review, args.accept_threshold)
            output_path = output_dir / f"{paper_id}.json"
            output_path.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[{paper_id}] Ready -> {output_path} "
                  f"(strengths={len(review['strengths'])} weaknesses={len(review['weaknesses'])} "
                  f"score={review['numerical_score']})")

            if args.check_interval > 0:
                time.sleep(args.check_interval)

        if still_pending == 0:
            continue  # loop immediately; the next pass will find nothing pending and exit
        if not args.loop:
            print(f"\n{still_pending} still pending. Rerun with --loop to keep polling, or run again later.")
            return
        print(f"Sleeping {args.poll_interval}s before next round...")
        time.sleep(args.poll_interval)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _add_common_args(p):
    p.add_argument("--token_dir", default=DEFAULT_TOKEN_DIR,
                    help=f"Directory of per-paper token files (default: {DEFAULT_TOKEN_DIR}).")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                    help="Directory for per-paper output files, one <paper_id>.json each "
                         f"(default: {DEFAULT_OUTPUT_DIR}). Merge with eval/merge_paperreviewer.py.")
    p.add_argument("--proxy", default=os.environ.get(PROXY_ENV_VAR),
                    help="HTTP(S) proxy URL used for the paperreview.ai API calls only — "
                         "never for the S3 upload, which always goes direct. "
                         f"e.g. http://user:pass@host:port. Also read from ${PROXY_ENV_VAR}.")


def main():
    parser = argparse.ArgumentParser(description="Run paperreview.ai on papers from a papers json.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="Upload PDFs and save review tokens (does not wait for reviews).")
    p_submit.add_argument("--papers_json", default=DEFAULT_PAPERS_JSON)
    p_submit.add_argument("--pdf_dir", default=DEFAULT_PDF_DIR)
    p_submit.add_argument("--paper", default=None, help="Single paper_id to run (default: all).")
    p_submit.add_argument("--email", default=DEFAULT_EMAIL,
                           help="Email passed to confirm-upload (not used for retrieval; "
                                "results are fetched by polling the token directly).")
    p_submit.add_argument("--submit_interval", type=float, default=DEFAULT_SUBMIT_INTERVAL,
                           help="Seconds to sleep between submissions, to avoid hammering the API "
                                f"(default: {DEFAULT_SUBMIT_INTERVAL}). Set to 0 to disable.")
    _add_common_args(p_submit)
    p_submit.set_defaults(func=cmd_submit)

    p_poll = sub.add_parser("poll", help="Check pending tokens and write finished reviews.")
    p_poll.add_argument("--raw_dir", default="results/paperreviewer",
                         help="Directory for raw /api/review/{token} responses (default: results/paperreviewer).")
    p_poll.add_argument("--poll_interval", type=int, default=DEFAULT_POLL_INTERVAL,
                         help="Seconds to sleep between rounds when --loop is set.")
    p_poll.add_argument("--check_interval", type=float, default=DEFAULT_CHECK_INTERVAL,
                         help="Seconds to sleep between individual token checks within a round "
                              f"(default: {DEFAULT_CHECK_INTERVAL}). Set to 0 to disable.")
    p_poll.add_argument("--token_timeout", type=int, default=DEFAULT_TOKEN_TIMEOUT,
                         help="Give up on a token after this many seconds since it was submitted "
                              f"(default: {DEFAULT_TOKEN_TIMEOUT}).")
    p_poll.add_argument("--accept_threshold", type=float, default=DEFAULT_ACCEPT_THRESHOLD,
                         help="numerical_score >= this maps to accept_or_not='accept' "
                              f"(default: {DEFAULT_ACCEPT_THRESHOLD}, matches evaluation.py's --conf_threshold).")
    p_poll.add_argument("--loop", action="store_true",
                         help="Keep polling in rounds until every token is done or times out. "
                              "Without this, do one round and exit (good for a cron job).")
    _add_common_args(p_poll)
    p_poll.set_defaults(func=cmd_poll)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
