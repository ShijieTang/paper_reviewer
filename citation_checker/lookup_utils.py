import re
import threading
import time
from typing import Optional


def normalize_title_for_lookup(title: str) -> str:
    """Normalize a title string for cache keys and coarse matching."""
    if not title:
        return ""
    normalized = title.casefold()
    normalized = re.sub(r"[*_`#\"'“”‘’]", " ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def compact_title_for_lookup(title: str) -> str:
    """Compact alphanumeric-only normalization for hyphen/space variants."""
    if not title:
        return ""
    return re.sub(r"[^a-z0-9]+", "", title.casefold())


class RequestThrottle:
    """Combine bounded concurrency with a minimum spacing between requests."""

    def __init__(self, min_interval: float, max_concurrency: int):
        self._min_interval = min_interval
        self._semaphore = threading.Semaphore(max(1, max_concurrency))
        self._lock = threading.Lock()
        self._last_request_at = 0.0

    def reconfigure(self, min_interval: Optional[float] = None, max_concurrency: Optional[int] = None) -> None:
        if min_interval is not None:
            self._min_interval = min_interval
        if max_concurrency is not None:
            self._semaphore = threading.Semaphore(max(1, max_concurrency))

    def __enter__(self):
        self._semaphore.acquire()
        with self._lock:
            now = time.monotonic()
            remaining = self._min_interval - (now - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
            self._last_request_at = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._semaphore.release()
        return False
