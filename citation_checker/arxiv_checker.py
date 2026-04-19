"""
Check whether a paper exists on arXiv using the arXiv API.

Uses the Atom feed search endpoint:
  https://export.arxiv.org/api/query?search_query=ti:"<title>"&max_results=5
"""

import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from typing import Optional, Tuple

import requests

from .lookup_utils import RequestThrottle, compact_title_for_lookup, normalize_title_for_lookup

logger = logging.getLogger(__name__)

_ARXIV_API = "https://export.arxiv.org/api/query"
_REQUEST_TIMEOUT = 6

_NS = "http://www.w3.org/2005/Atom"
_ARXIV_ID_RE = re.compile(r"(?:arXiv:|abs/)(\d{4}\.\d{4,5})", re.IGNORECASE)
_THROTTLE = RequestThrottle(min_interval=3.0, max_concurrency=1)
_CACHE_LOCK = threading.Lock()
_CACHE = {}
_USE_CACHE = True
_DISABLED = False
_DISABLE_LOCK = threading.Lock()


def _disable_backend(reason: str) -> None:
    global _DISABLED
    with _DISABLE_LOCK:
        if not _DISABLED:
            logger.warning("Disabling arXiv checks for this run: %s", reason)
        _DISABLED = True


def _title_similarity(a: str, b: str) -> float:
    if compact_title_for_lookup(a) and compact_title_for_lookup(a) == compact_title_for_lookup(b):
        return 1.0

    _STOP = {'a', 'an', 'the', 'of', 'in', 'on', 'for', 'to', 'and', 'with',
             'via', 'using', 'toward', 'towards', 'from', 'is', 'are', 'by'}
    ta = {w for w in normalize_title_for_lookup(a).split() if w not in _STOP and len(w) > 2}
    tb = {w for w in normalize_title_for_lookup(b).split() if w not in _STOP and len(w) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def configure_arxiv(concurrency: int = 1, use_cache: bool = True, reset_cache: bool = False) -> None:
    global _USE_CACHE, _DISABLED
    _USE_CACHE = use_cache
    _DISABLED = False
    _THROTTLE.reconfigure(max_concurrency=concurrency)
    if reset_cache:
        with _CACHE_LOCK:
            _CACHE.clear()


def _extract_arxiv_id(text: str) -> Optional[str]:
    if not text:
        return None
    match = _ARXIV_ID_RE.search(text)
    if match:
        return match.group(1)
    return None


def _query_arxiv(params: dict) -> Tuple[bool, Optional[str]]:
    if _DISABLED:
        return False, None

    for attempt in range(3):
        try:
            with _THROTTLE:
                resp = requests.get(
                    _ARXIV_API,
                    params=params,
                    timeout=_REQUEST_TIMEOUT,
                    headers={"User-Agent": "paper-reviewer-citation-checker/1.0"},
                )
            if resp.status_code == 429:
                _disable_backend("received HTTP 429 from arXiv API")
                return False, None
            if resp.status_code != 200:
                logger.warning("arXiv API returned %s", resp.status_code)
                return False, None

            root = ET.fromstring(resp.text)
            entries = root.findall(f"{{{_NS}}}entry")
            if not entries:
                return False, None

            id_el = entries[0].find(f"{{{_NS}}}id")
            arxiv_url = (id_el.text or "").strip() if id_el is not None else None
            return True, arxiv_url
        except (requests.RequestException, ET.ParseError) as e:
            if attempt == 1:
                logger.warning("arXiv request failed: %s", e)
            continue
    return False, None


def check_on_arxiv(title: str, raw_text: str = "") -> Tuple[bool, Optional[str]]:
    """
    Search arXiv for a paper by title.

    Returns:
        (found: bool, arxiv_url: Optional[str])
    """
    if _DISABLED:
        return False, None

    arxiv_id = _extract_arxiv_id(raw_text)
    if arxiv_id:
        return _query_arxiv({"id_list": arxiv_id})

    if not title or len(title.strip()) < 5:
        return False, None

    cache_key = normalize_title_for_lookup(title)
    if _USE_CACHE:
        with _CACHE_LOCK:
            cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached

    query = f'ti:"{title}"'
    params = {
        "search_query": query,
        "max_results": 5,
        "sortBy": "relevance",
    }

    found, url = False, None
    for attempt in range(3):
        try:
            with _THROTTLE:
                resp = requests.get(
                    _ARXIV_API,
                    params=params,
                    timeout=_REQUEST_TIMEOUT,
                    headers={"User-Agent": "paper-reviewer-citation-checker/1.0"},
                )
            if resp.status_code == 429:
                _disable_backend("received HTTP 429 from arXiv API")
                break
            if resp.status_code != 200:
                logger.warning("arXiv API returned %s", resp.status_code)
                break

            root = ET.fromstring(resp.text)
            entries = root.findall(f"{{{_NS}}}entry")

            for entry in entries:
                t_el = entry.find(f"{{{_NS}}}title")
                if t_el is None:
                    continue
                entry_title = (t_el.text or "").strip().replace("\n", " ")
                if _title_similarity(title, entry_title) >= 0.75:
                    id_el = entry.find(f"{{{_NS}}}id")
                    url = (id_el.text or "").strip() if id_el is not None else None
                    logger.debug("arXiv hit: %s", entry_title)
                    found = True
                    break
            break
        except (requests.RequestException, ET.ParseError) as e:
            if attempt == 1:
                logger.warning("arXiv request failed: %s", e)

    if _USE_CACHE:
        with _CACHE_LOCK:
            _CACHE[cache_key] = (found, url)

    return found, url
