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
import time
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_CR_BASE = "https://api.crossref.org/works"
_REQUEST_TIMEOUT = 10
_RATE_LIMIT_DELAY = 0.3

_DOI_RE = re.compile(r'\b(10\.\d{4,9}/[^\s\)\]\},"\'<>]+)', re.IGNORECASE)


def _title_similarity(a: str, b: str) -> float:
    _STOP = {'a', 'an', 'the', 'of', 'in', 'on', 'for', 'to', 'and', 'with',
             'via', 'using', 'toward', 'towards', 'from', 'is', 'are', 'by'}
    ta = {w.lower() for w in a.split() if w.lower() not in _STOP and len(w) > 2}
    tb = {w.lower() for w in b.split() if w.lower() not in _STOP and len(w) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _extract_doi(text: str) -> Optional[str]:
    m = _DOI_RE.search(text)
    return m.group(1).rstrip('.,;)') if m else None


def check_via_doi(title: str, raw_text: str = "") -> Tuple[bool, Optional[str]]:
    """
    Check CrossRef by DOI (if present in raw_text) or by title query.

    Returns:
        (found: bool, doi_url: Optional[str])
    """
    if not title or len(title.strip()) < 5:
        return False, None

    time.sleep(_RATE_LIMIT_DELAY)

    # --- Strategy 1: direct DOI resolution ---
    doi = _extract_doi(raw_text) if raw_text else None
    if doi:
        try:
            resp = requests.get(
                f"{_CR_BASE}/{doi}",
                timeout=_REQUEST_TIMEOUT,
                headers={
                    "User-Agent": "paper-reviewer-citation-checker/1.0 (mailto:checker@example.com)",
                },
            )
            if resp.status_code == 200:
                data = resp.json().get("message", {})
                doi_url = data.get("URL") or f"https://doi.org/{doi}"
                logger.debug("CrossRef DOI hit: %s", doi)
                return True, doi_url
        except requests.RequestException as e:
            logger.warning("CrossRef DOI request failed: %s", e)

    # --- Strategy 2: title query ---
    time.sleep(_RATE_LIMIT_DELAY)
    try:
        params = {
            "query.bibliographic": title,
            "rows": 3,
            "select": "title,DOI,URL",
        }
        resp = requests.get(
            _CR_BASE,
            params=params,
            timeout=_REQUEST_TIMEOUT,
            headers={
                "User-Agent": "paper-reviewer-citation-checker/1.0 (mailto:checker@example.com)",
            },
        )
        if resp.status_code == 200:
            items = resp.json().get("message", {}).get("items", [])
            for item in items:
                cr_titles = item.get("title", [])
                cr_title = cr_titles[0] if cr_titles else ""
                if cr_title and _title_similarity(title, cr_title) >= 0.75:
                    doi_url = item.get("URL") or (
                        f"https://doi.org/{item['DOI']}" if item.get("DOI") else None
                    )
                    logger.debug("CrossRef title hit: %s", cr_title)
                    return True, doi_url
    except requests.RequestException as e:
        logger.warning("CrossRef title query failed: %s", e)

    return False, None
