"""
Check whether a paper title exists on OpenReview using the public API.

OpenReview v2 API:
  https://api2.openreview.net/notes/search?term=<query>&limit=5&source=forum
  https://api2.openreview.net/notes?content.title=<title>&limit=3
"""

import logging
import time
import urllib.parse
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_OR_SEARCH_URL = "https://api2.openreview.net/notes/search"
_OR_NOTES_URL = "https://api2.openreview.net/notes"
_REQUEST_TIMEOUT = 15  # seconds
_RATE_LIMIT_DELAY = 3  # seconds between requests


def _title_similarity(a: str, b: str) -> float:
    """Simple token overlap similarity, case-insensitive."""
    # Strip common stop words to reduce noise
    _STOP = {'a', 'an', 'the', 'of', 'in', 'on', 'for', 'to', 'and', 'with',
             'via', 'using', 'toward', 'towards', 'from', 'is', 'are', 'by'}
    ta = {w.lower() for w in a.split() if w.lower() not in _STOP and len(w) > 2}
    tb = {w.lower() for w in b.split() if w.lower() not in _STOP and len(w) > 2}
    if not ta or not tb:
        return 0.0
    intersection = ta & tb
    return len(intersection) / max(len(ta), len(tb))


def _clean_title(title: str) -> str:
    """Remove markdown formatting and trailing punctuation from title."""
    import re
    title = re.sub(r'[*_`#]', '', title)
    title = title.strip(' .,;:')
    return title


def check_on_openreview(title: str) -> Tuple[bool, Optional[str]]:
    """
    Search OpenReview for a paper by title.

    Returns:
        (found: bool, openreview_url: Optional[str])
        - found=True means a strong title match was found
        - openreview_url is the link to the submission if found
    """
    if not title or len(title.strip()) < 5:
        return False, None

    title = _clean_title(title)
    time.sleep(_RATE_LIMIT_DELAY)

    # --- Strategy 1: full-text search ---
    try:
        params = {
            "term": title,
            "limit": 5,
            "source": "forum",
        }
        resp = requests.get(
            _OR_SEARCH_URL,
            params=params,
            timeout=_REQUEST_TIMEOUT,
            headers={"User-Agent": "paper-reviewer-citation-checker/1.0"},
        )
        if resp.status_code == 200:
            data = resp.json()
            notes = data.get("notes", [])
            for note in notes:
                content = note.get("content", {})
                # v2 API: content values may be dicts with "value" key
                note_title = content.get("title", "")
                if isinstance(note_title, dict):
                    note_title = note_title.get("value", "")
                if note_title and _title_similarity(title, note_title) >= 0.75:
                    forum_id = note.get("forum", note.get("id", ""))
                    url = f"https://openreview.net/forum?id={forum_id}" if forum_id else None
                    logger.debug("OpenReview search hit: %s", note_title)
                    return True, url
    except requests.RequestException as e:
        logger.warning("OpenReview search request failed: %s", e)

    # --- Strategy 2: exact title field query ---
    time.sleep(_RATE_LIMIT_DELAY)
    try:
        params = {
            "content.title": title,
            "limit": 3,
        }
        resp = requests.get(
            _OR_NOTES_URL,
            params=params,
            timeout=_REQUEST_TIMEOUT,
            headers={"User-Agent": "paper-reviewer-citation-checker/1.0"},
        )
        if resp.status_code == 200:
            data = resp.json()
            notes = data.get("notes", [])
            for note in notes:
                content = note.get("content", {})
                note_title = content.get("title", "")
                if isinstance(note_title, dict):
                    note_title = note_title.get("value", "")
                if note_title and _title_similarity(title, note_title) >= 0.75:
                    forum_id = note.get("forum", note.get("id", ""))
                    url = f"https://openreview.net/forum?id={forum_id}" if forum_id else None
                    logger.debug("OpenReview notes hit: %s", note_title)
                    return True, url
    except requests.RequestException as e:
        logger.warning("OpenReview notes request failed: %s", e)

    return False, None
