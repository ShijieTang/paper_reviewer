"""
Check whether a paper exists on arXiv using the arXiv API.

Uses the Atom feed search endpoint:
  https://export.arxiv.org/api/query?search_query=ti:"<title>"&max_results=5
"""

import logging
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_ARXIV_API = "https://export.arxiv.org/api/query"
_REQUEST_TIMEOUT = 10
_RATE_LIMIT_DELAY = 0.5  # arXiv asks for >=3s between bursts; we space calls out upstream

_NS = "http://www.w3.org/2005/Atom"


def _title_similarity(a: str, b: str) -> float:
    _STOP = {'a', 'an', 'the', 'of', 'in', 'on', 'for', 'to', 'and', 'with',
             'via', 'using', 'toward', 'towards', 'from', 'is', 'are', 'by'}
    ta = {w.lower() for w in a.split() if w.lower() not in _STOP and len(w) > 2}
    tb = {w.lower() for w in b.split() if w.lower() not in _STOP and len(w) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def check_on_arxiv(title: str) -> Tuple[bool, Optional[str]]:
    """
    Search arXiv for a paper by title.

    Returns:
        (found: bool, arxiv_url: Optional[str])
    """
    if not title or len(title.strip()) < 5:
        return False, None

    time.sleep(_RATE_LIMIT_DELAY)

    query = f'ti:"{title}"'
    params = {
        "search_query": query,
        "max_results": 5,
        "sortBy": "relevance",
    }

    try:
        resp = requests.get(
            _ARXIV_API,
            params=params,
            timeout=_REQUEST_TIMEOUT,
            headers={"User-Agent": "paper-reviewer-citation-checker/1.0"},
        )
        if resp.status_code != 200:
            logger.warning("arXiv API returned %s", resp.status_code)
            return False, None

        root = ET.fromstring(resp.text)
        entries = root.findall(f"{{{_NS}}}entry")

        for entry in entries:
            t_el = entry.find(f"{{{_NS}}}title")
            if t_el is None:
                continue
            entry_title = (t_el.text or "").strip().replace("\n", " ")
            if _title_similarity(title, entry_title) >= 0.75:
                id_el = entry.find(f"{{{_NS}}}id")
                arxiv_url = (id_el.text or "").strip() if id_el is not None else None
                logger.debug("arXiv hit: %s", entry_title)
                return True, arxiv_url

    except (requests.RequestException, ET.ParseError) as e:
        logger.warning("arXiv request failed: %s", e)

    return False, None
