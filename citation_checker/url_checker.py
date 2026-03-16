"""
Check whether a URL in a reference is accessible.

Tries a HEAD request first (faster), falls back to GET for servers
that reject HEAD. Handles redirects, timeouts, and common SSL errors.
"""

import logging
from typing import Tuple

import requests

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 10  # seconds
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; paper-reviewer-citation-checker/1.0; "
        "+https://github.com/paper-reviewer)"
    )
}


def check_url_accessible(url: str) -> Tuple[bool, str]:
    """
    Check if a URL is accessible.

    Returns:
        (accessible: bool, reason: str)
    """
    if not url or not url.startswith("http"):
        return False, "Invalid or missing URL"

    # Try HEAD first
    try:
        resp = requests.head(
            url,
            timeout=_REQUEST_TIMEOUT,
            headers=_HEADERS,
            allow_redirects=True,
        )
        if resp.status_code < 400:
            return True, f"HTTP {resp.status_code}"
        if resp.status_code == 405:
            # Server doesn't allow HEAD, fall through to GET
            pass
        else:
            return False, f"HTTP {resp.status_code}"
    except requests.exceptions.SSLError:
        return False, "SSL error"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused / DNS failure"
    except requests.exceptions.Timeout:
        return False, "Request timed out"
    except requests.RequestException as e:
        return False, str(e)

    # Fallback: GET with stream to avoid downloading the full body
    try:
        resp = requests.get(
            url,
            timeout=_REQUEST_TIMEOUT,
            headers=_HEADERS,
            allow_redirects=True,
            stream=True,
        )
        resp.close()
        if resp.status_code < 400:
            return True, f"HTTP {resp.status_code}"
        return False, f"HTTP {resp.status_code}"
    except requests.exceptions.SSLError:
        return False, "SSL error"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused / DNS failure"
    except requests.exceptions.Timeout:
        return False, "Request timed out"
    except requests.RequestException as e:
        return False, str(e)
