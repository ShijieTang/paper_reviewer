"""
Check whether a paper title exists on OpenReview using the public API.

OpenReview v2 API:
  https://api2.openreview.net/notes/search?term=<query>&limit=5&source=forum
  https://api2.openreview.net/notes?content.title=<title>&limit=3
"""

import logging
import threading
from typing import Optional, Tuple

import requests

from .lookup_utils import RequestThrottle, compact_title_for_lookup, normalize_title_for_lookup

logger = logging.getLogger(__name__)

_OR_SEARCH_URL = "https://api2.openreview.net/notes/search"
_OR_NOTES_URL = "https://api2.openreview.net/notes"
_REQUEST_TIMEOUT = 8  # seconds
_THROTTLE = RequestThrottle(min_interval=0.5, max_concurrency=2)
_CACHE_LOCK = threading.Lock()
_CACHE = {}
_USE_CACHE = True
_DISABLED = False
_DISABLE_LOCK = threading.Lock()


def _disable_backend(reason: str) -> None:
    global _DISABLED
    with _DISABLE_LOCK:
        if not _DISABLED:
            logger.warning("Disabling OpenReview checks for this run: %s", reason)
        _DISABLED = True


def _title_similarity(a: str, b: str) -> float:
    """Simple token overlap similarity, case-insensitive."""
    if compact_title_for_lookup(a) and compact_title_for_lookup(a) == compact_title_for_lookup(b):
        return 1.0

    _STOP = {'a', 'an', 'the', 'of', 'in', 'on', 'for', 'to', 'and', 'with',
             'via', 'using', 'toward', 'towards', 'from', 'is', 'are', 'by'}
    ta = {w for w in normalize_title_for_lookup(a).split() if w not in _STOP and len(w) > 2}
    tb = {w for w in normalize_title_for_lookup(b).split() if w not in _STOP and len(w) > 2}
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


def configure_openreview(concurrency: int = 2, use_cache: bool = True, reset_cache: bool = False) -> None:
    global _USE_CACHE, _DISABLED
    _USE_CACHE = use_cache
    _DISABLED = False
    _THROTTLE.reconfigure(max_concurrency=concurrency)
    if reset_cache:
        with _CACHE_LOCK:
            _CACHE.clear()


def _request_json(url: str, params: dict) -> Optional[dict]:
    if _DISABLED:
        return None

    for attempt in range(2):
        try:
            with _THROTTLE:
                resp = requests.get(
                    url,
                    params=params,
                    timeout=_REQUEST_TIMEOUT,
                    headers={"User-Agent": "paper-reviewer-citation-checker/1.0"},
                )
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 403:
                _disable_backend("received HTTP 403 from OpenReview API")
                return None
            logger.warning("OpenReview request returned %s for %s", resp.status_code, url)
            return None
        except requests.RequestException as e:
            if attempt == 1:
                logger.warning("OpenReview request failed: %s", e)
            continue
    return None


def _query_exact_title(title: str) -> Tuple[bool, Optional[str]]:
    data = _request_json(_OR_NOTES_URL, {"content.title": title, "limit": 3})
    if not data:
        return False, None

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
    return False, None


def _query_search(title: str) -> Tuple[bool, Optional[str]]:
    data = _request_json(_OR_SEARCH_URL, {"term": title, "limit": 5, "source": "forum"})
    if not data:
        return False, None

    notes = data.get("notes", [])
    for note in notes:
        content = note.get("content", {})
        note_title = content.get("title", "")
        if isinstance(note_title, dict):
            note_title = note_title.get("value", "")
        if note_title and _title_similarity(title, note_title) >= 0.75:
            forum_id = note.get("forum", note.get("id", ""))
            url = f"https://openreview.net/forum?id={forum_id}" if forum_id else None
            logger.debug("OpenReview search hit: %s", note_title)
            return True, url
    return False, None


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
    if _DISABLED:
        return False, None

    title = _clean_title(title)
    cache_key = normalize_title_for_lookup(title)
    if _USE_CACHE:
        with _CACHE_LOCK:
            cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached

    found, url = _query_exact_title(title)
    if not found:
        found, url = _query_search(title)

    if _USE_CACHE:
        with _CACHE_LOCK:
            _CACHE[cache_key] = (found, url)

    return found, url
