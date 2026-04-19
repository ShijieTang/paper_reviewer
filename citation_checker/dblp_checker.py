"""
Check whether a paper exists on DBLP using the public search API.

DBLP search API:
  https://dblp.org/search/publ/api?q=<title>&format=json&h=5
"""

import logging
import threading
from typing import Optional, Tuple

import requests

from .lookup_utils import RequestThrottle, compact_title_for_lookup, normalize_title_for_lookup

logger = logging.getLogger(__name__)

_DBLP_API = "https://dblp.org/search/publ/api"
_REQUEST_TIMEOUT = 6
_THROTTLE = RequestThrottle(min_interval=0.2, max_concurrency=2)
_CACHE_LOCK = threading.Lock()
_CACHE = {}
_USE_CACHE = True


def _title_similarity(a: str, b: str) -> float:
    if compact_title_for_lookup(a) and compact_title_for_lookup(a) == compact_title_for_lookup(b):
        return 1.0

    _STOP = {
        "a", "an", "the", "of", "in", "on", "for", "to", "and", "with",
        "via", "using", "toward", "towards", "from", "is", "are", "by",
    }
    ta = {w for w in normalize_title_for_lookup(a).split() if w not in _STOP and len(w) > 2}
    tb = {w for w in normalize_title_for_lookup(b).split() if w not in _STOP and len(w) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def configure_dblp(concurrency: int = 2, use_cache: bool = True, reset_cache: bool = False) -> None:
    global _USE_CACHE
    _USE_CACHE = use_cache
    _THROTTLE.reconfigure(max_concurrency=concurrency)
    if reset_cache:
        with _CACHE_LOCK:
            _CACHE.clear()


def check_on_dblp(title: str) -> Tuple[bool, Optional[str]]:
    """
    Search DBLP for a paper by title.

    Returns:
        (found: bool, dblp_url: Optional[str])
    """
    if not title or len(title.strip()) < 5:
        return False, None

    cache_key = normalize_title_for_lookup(title)
    if _USE_CACHE:
        with _CACHE_LOCK:
            cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached

    found, url = False, None
    for attempt in range(2):
        try:
            with _THROTTLE:
                resp = requests.get(
                    _DBLP_API,
                    params={"q": title, "format": "json", "h": 5},
                    timeout=_REQUEST_TIMEOUT,
                    headers={"User-Agent": "paper-reviewer-citation-checker/1.0"},
                )
            if resp.status_code != 200:
                logger.warning("DBLP API returned %s", resp.status_code)
                break

            hits = resp.json().get("result", {}).get("hits", {}).get("hit", [])
            if isinstance(hits, dict):
                hits = [hits]

            for hit in hits:
                info = hit.get("info", {})
                dblp_title = info.get("title", "")
                if dblp_title and _title_similarity(title, dblp_title) >= 0.75:
                    url = info.get("url") or info.get("ee")
                    logger.debug("DBLP hit: %s", dblp_title)
                    found = True
                    break
            break
        except (requests.RequestException, ValueError) as e:
            if attempt == 1:
                logger.warning("DBLP request failed: %s", e)

    if _USE_CACHE:
        with _CACHE_LOCK:
            _CACHE[cache_key] = (found, url)

    return found, url
