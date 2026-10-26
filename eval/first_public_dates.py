#!/usr/bin/env python3
"""Find when each dataset paper first became public, and mark the papers newer than a model's training cutoff.

A reviewer model may already know a paper (and the work around it) if the paper was public before
the model's training cutoff. RAG should matter most for papers the model cannot have seen, so the
evaluation reports results on that subset separately.

The first-public date is the earliest of:
  * the first arXiv version (v1 "published" date), found by exact-title search on the arXiv API;
  * the OpenReview public date of the forum (odate, else cdate/tcdate) from the paper's
    OpenReview URL (forum id).

No list is needed from the user: both dates come from the archives.

Usage (from the repository root; needs network access):
    python eval/first_public_dates.py \\
        --dataset eval/openreview_60_module_test.json \\
        --model_cutoff 2025-03-31 \\
        [--output eval/first_public_dates_60.json]

--model_cutoff is the reviewer model's training-data cutoff from the provider's official model
documentation (look it up for the exact model you run). Papers whose first-public date is after it
form the "post-cutoff" subset. Re-running reuses cached lookups (eval/.cache/first_public_dates/).
The subset can be empty: then the comparison is limited to papers the model may have seen, and
the report says so.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_API = "https://export.arxiv.org/api/query"
OPENREVIEW_APIS = ("https://api2.openreview.net/notes", "https://api.openreview.net/notes")
TITLE_MATCH = 0.9


def norm_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (title or "").lower())).strip()


def forum_id(paper_url: str) -> str:
    query = urllib.parse.urlparse(paper_url or "").query
    return (urllib.parse.parse_qs(query).get("id") or [""])[0]


def ms_to_date(value) -> str:
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).date().isoformat()
    except (TypeError, ValueError, OSError):
        return ""


# ── parsing (kept separate from network access so it can be tested offline) ──

def parse_arxiv_feed(xml_text: str, title: str) -> dict:
    """Best title match in an arXiv Atom feed -> {arxiv_id, published, matched_title, similarity}."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {}
    want = norm_title(title)
    best = {}
    for entry in root.findall(f"{ATOM}entry"):
        got = " ".join((entry.findtext(f"{ATOM}title") or "").split())
        sim = difflib.SequenceMatcher(None, want, norm_title(got)).ratio()
        if sim >= TITLE_MATCH and sim > best.get("similarity", 0):
            raw_id = entry.findtext(f"{ATOM}id") or ""
            best = {
                "arxiv_id": re.sub(r"v\d+$", "", raw_id.rsplit("/abs/", 1)[-1]),
                "published": (entry.findtext(f"{ATOM}published") or "")[:10],
                "matched_title": got,
                "similarity": round(sim, 3),
            }
    return best


def parse_openreview_notes(payload: dict) -> dict:
    """Public date of an OpenReview forum note -> {openreview_date, field}."""
    notes = (payload or {}).get("notes") or []
    if not notes:
        return {}
    note = notes[0]
    for field in ("odate", "pdate", "cdate", "tcdate"):
        if note.get(field):
            day = ms_to_date(note[field])
            if day:
                return {"openreview_date": day, "field": field}
    return {}


def first_public(record: dict) -> str:
    days = [d for d in (record.get("arxiv", {}).get("published"),
                        record.get("openreview", {}).get("openreview_date")) if d]
    return min(days) if days else ""


# ── network with a small file cache ──────────────────────────────────────────

class Fetcher:
    def __init__(self, cache_dir: Path, delay: float = 3.0, timeout: float = 30.0):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay            # arXiv asks for one request every 3 seconds
        self.timeout = timeout
        self._last = 0.0

    def get(self, url: str) -> str:
        path = self.cache_dir / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".txt")
        if path.exists():
            return path.read_text(encoding="utf-8")
        wait = self.delay - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        request = urllib.request.Request(url, headers={"User-Agent": "paper-reviewer-eval/1.0"})
        error = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    text = response.read().decode("utf-8", errors="replace")
                self._last = time.time()
                path.write_text(text, encoding="utf-8")
                return text
            except urllib.error.HTTPError as exc:
                error = exc
                if exc.code not in (429, 500, 502, 503, 504):
                    break
            except Exception as exc:      # timeouts, resets
                error = exc
            time.sleep(5 * (attempt + 1))
        self._last = time.time()
        raise error


def lookup_arxiv(fetcher: Fetcher, title: str) -> dict:
    words = norm_title(title)
    if not words:
        return {}
    query = urllib.parse.urlencode({"search_query": f'ti:"{words}"', "max_results": 5})
    return parse_arxiv_feed(fetcher.get(f"{ARXIV_API}?{query}"), title)


def lookup_openreview(fetcher: Fetcher, paper_url: str) -> dict:
    fid = forum_id(paper_url)
    if not fid:
        return {}
    for api in OPENREVIEW_APIS:
        try:
            found = parse_openreview_notes(json.loads(fetcher.get(f"{api}?id={urllib.parse.quote(fid)}")))
        except Exception:
            continue
        if found:
            return {**found, "forum_id": fid}
    return {}


def subset_report(records: dict, model_cutoff: str) -> dict:
    cutoff = date.fromisoformat(model_cutoff)
    post, pre, unknown = [], [], []
    for pid, rec in sorted(records.items()):
        day = rec.get("first_public")
        if not day:
            unknown.append(pid)
        elif date.fromisoformat(day) > cutoff:
            post.append(pid)
        else:
            pre.append(pid)
    return {"model_cutoff": model_cutoff, "num_post_cutoff": len(post), "num_pre_cutoff": len(pre),
            "num_unknown": len(unknown), "post_cutoff": post, "pre_cutoff": pre, "unknown": unknown}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="eval/openreview_60_module_test.json")
    parser.add_argument("--model_cutoff", required=True,
                        help="Reviewer model's training cutoff (YYYY-MM-DD) from the provider's official docs.")
    parser.add_argument("--output", default="eval/first_public_dates_60.json")
    parser.add_argument("--cache_dir", default="eval/.cache/first_public_dates")
    args = parser.parse_args(argv)

    date.fromisoformat(args.model_cutoff)     # fail early on a bad date
    dataset = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))
    if isinstance(dataset, list):
        dataset = {item["paper_id"]: item for item in dataset}
    fetcher = Fetcher(ROOT / args.cache_dir)
    records = {}
    for i, (pid, meta) in enumerate(sorted(dataset.items()), 1):
        rec = {"title": meta.get("title", ""), "conference": meta.get("conference"), "year": meta.get("year")}
        for key, fn, arg in (("arxiv", lookup_arxiv, meta.get("title", "")),
                             ("openreview", lookup_openreview, meta.get("paper_url", ""))):
            try:
                rec[key] = fn(fetcher, arg)
            except Exception as exc:
                rec[key] = {"error": f"{type(exc).__name__}: {exc}"}
        rec["first_public"] = first_public(rec)
        records[pid] = rec
        print(f"[{i}/{len(dataset)}] {rec['first_public'] or 'unknown':10s}  {rec['title'][:80]}", flush=True)

    report = subset_report(records, args.model_cutoff)
    out = ROOT / args.output
    out.write_text(json.dumps({"subset": report, "papers": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{report['num_post_cutoff']} paper(s) first public after {args.model_cutoff}; "
          f"{report['num_pre_cutoff']} before; {report['num_unknown']} unknown. Wrote {args.output}")
    if not report["num_post_cutoff"]:
        print("No post-cutoff papers: the post-cutoff comparison will be empty for this model.")


if __name__ == "__main__":
    sys.exit(main())
