"""
Check whether a paper exists via CrossRef (DOI lookup) using the public API.

Two strategies:
  1. If the reference contains a DOI, resolve it directly.
  2. Otherwise, query CrossRef by title and check the top result.

CrossRef API:
  https://api.crossref.org/works/<doi>
  https://api.crossref.org/works?query.bibliographic=<title>&rows=3
"""

import logging
import re
import threading
from typing import Optional, Tuple

import requests

from .lookup_utils import RequestThrottle, compact_title_for_lookup, normalize_title_for_lookup

logger = logging.getLogger(__name__)

_CR_BASE = "https://api.crossref.org/works"
_REQUEST_TIMEOUT = 6

_DOI_RE = re.compile(r'\b(10\.\d{4,9}/[^\s\)\]\},"\'<>]+)', re.IGNORECASE)
_THROTTLE = RequestThrottle(min_interval=0.15, max_concurrency=4)
_CACHE_LOCK = threading.Lock()
_CACHE = {}
_USE_CACHE = True


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


def _extract_doi(text: str) -> Optional[str]:
    m = _DOI_RE.search(text)
    return m.group(1).rstrip('.,;)') if m else None


def configure_crossref(concurrency: int = 4, use_cache: bool = True, reset_cache: bool = False) -> None:
    global _USE_CACHE
    _USE_CACHE = use_cache
    _THROTTLE.reconfigure(max_concurrency=concurrency)
    if reset_cache:
        with _CACHE_LOCK:
            _CACHE.clear()


def _request_json(url: str, params: Optional[dict] = None) -> Optional[dict]:
    for attempt in range(2):
        try:
            with _THROTTLE:
                resp = requests.get(
                    url,
                    params=params,
                    timeout=_REQUEST_TIMEOUT,
                    headers={
                        "User-Agent": "paper-reviewer-citation-checker/1.0 (mailto:checker@example.com)",
                    },
                )
            if resp.status_code == 200:
                return resp.json()
            logger.warning("CrossRef request returned %s for %s", resp.status_code, url)
            return None
        except requests.RequestException as e:
            if attempt == 1:
                logger.warning("CrossRef request failed: %s", e)
            continue
    return None


def _query_by_title(title: str, year: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    query_variants = [title]
    if ":" in title:
        prefix = title.split(":", 1)[0].strip()
        if prefix and prefix not in query_variants:
            query_variants.append(prefix)

    for query in query_variants:
        params = {
            "query.bibliographic": query,
            "rows": 10,
            "select": "title,DOI,URL,published-print,published-online,created",
        }
        data = _request_json(_CR_BASE, params=params)
        if not data:
            continue
        items = data.get("message", {}).get("items", [])
        for item in items:
            cr_titles = item.get("title", [])
            cr_title = cr_titles[0] if cr_titles else ""
            if cr_title and _title_similarity(title, cr_title) >= 0.75:
                doi_url = item.get("URL") or (
                    f"https://doi.org/{item['DOI']}" if item.get("DOI") else None
                )
                logger.debug("CrossRef title hit: %s", cr_title)
                return True, doi_url

        if year:
            params["query.bibliographic"] = f"{query} {year}"
            data = _request_json(_CR_BASE, params=params)
            if not data:
                continue
            items = data.get("message", {}).get("items", [])
            for item in items:
                cr_titles = item.get("title", [])
                cr_title = cr_titles[0] if cr_titles else ""
                if cr_title and _title_similarity(title, cr_title) >= 0.75:
                    doi_url = item.get("URL") or (
                        f"https://doi.org/{item['DOI']}" if item.get("DOI") else None
                    )
                    logger.debug("CrossRef title/year hit: %s", cr_title)
                    return True, doi_url
    return False, None


def check_via_doi(title: str, raw_text: str = "", year: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Check CrossRef by DOI (if present in raw_text) or by title query.

    Returns:
        (found: bool, doi_url: Optional[str])
    """
    if not title or len(title.strip()) < 5:
        return False, None

    cache_key = normalize_title_for_lookup(title)
    if _USE_CACHE and not _extract_doi(raw_text):
        with _CACHE_LOCK:
            cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached

    doi = _extract_doi(raw_text) if raw_text else None
    if doi:
        data = _request_json(f"{_CR_BASE}/{doi}")
        if data:
            message = data.get("message", {})
            doi_url = message.get("URL") or f"https://doi.org/{doi}"
            logger.debug("CrossRef DOI hit: %s", doi)
            return True, doi_url
        return False, None

    found, url = _query_by_title(title, year=year)
    if _USE_CACHE:
        with _CACHE_LOCK:
            _CACHE[cache_key] = (found, url)

    return found, url
