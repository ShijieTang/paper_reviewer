from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from rag.cache import JsonCache
from rag.models import PaperMetadata, RelatedWorkQuery


@dataclass
class ProviderResult:
    provider: str
    papers: list[PaperMetadata] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    status: str = "used"


class ProviderHTTPError(RuntimeError):
    def __init__(self, url: str, status_code: int, reason: str = ""):
        self.url = url
        self.status_code = int(status_code)
        self.reason = str(reason or "").strip()
        message = f"HTTP {self.status_code}"
        if self.reason:
            message = f"{message} {self.reason}"
        super().__init__(message)


def http_status_code(exc: BaseException) -> int | None:
    if isinstance(exc, ProviderHTTPError):
        return exc.status_code
    for attr in ("code", "status", "status_code"):
        value = getattr(exc, attr, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def is_rate_limited_error(exc: BaseException) -> bool:
    status = http_status_code(exc)
    return status == 429 or "429" in str(exc) or "RateLimitError" in str(exc)


def is_forbidden_error(exc: BaseException) -> bool:
    status = http_status_code(exc)
    return status == 403 or "403" in str(exc)


def is_retryable_error(exc: BaseException) -> bool:
    """Transient failure worth retrying: rate limit, server error, or timeout."""
    status = http_status_code(exc)
    if status is not None:
        return status == 429 or 500 <= status < 600
    return isinstance(exc, (urllib.error.URLError, TimeoutError, OSError))


class PaperSearchProvider:
    name = "provider"
    default_headers = {"User-Agent": "paper-reviewer-rag/1.0"}
    inter_query_delay_seconds = 0.0
    # A batch run hammers these public APIs from several workers at once, so a
    # single 429 or read timeout must not silently drop a paper's evidence.
    max_retries = 4
    retry_base_delay_seconds = 3.0

    def __init__(self, cache_dir: str = "data/rag_cache", timeout: int = 20):
        self.cache = JsonCache(cache_dir, self.name)
        self.timeout = timeout
        self._rate_limited: BaseException | None = None

    def _sleep_between_queries(self, query_index: int) -> None:
        if query_index > 0 and self.inter_query_delay_seconds > 0:
            time.sleep(self.inter_query_delay_seconds)

    def _retry_delay(self, exc: BaseException, attempt: int) -> float:
        """Backoff for one retry, honouring Retry-After when the server sends it."""
        retry_after = getattr(exc, "headers", None)
        if retry_after is not None:
            try:
                return max(1.0, float(retry_after.get("Retry-After")))
            except (TypeError, ValueError):
                pass
        return self.retry_base_delay_seconds * (2 ** attempt)

    def _fetch(self, url: str, headers: dict[str, str] | None, decode) -> Any:
        cached = self.cache.get(url)
        if cached is not None:
            return cached
        # A sustained 429 is an IP-level block, not a per-request limit: once one
        # query has exhausted its retries there is nothing to gain from putting
        # every remaining query through the same backoff.
        if getattr(self, "_rate_limited", None) is not None:
            raise self._rate_limited
        request = urllib.request.Request(url, headers={**self.default_headers, **(headers or {})})
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = decode(response.read())
                break
            except urllib.error.HTTPError as exc:
                reason = getattr(exc, "reason", "") or getattr(exc, "msg", "")
                error = ProviderHTTPError(url, exc.code, reason)
                error.headers = getattr(exc, "headers", None)
            except Exception as exc:                       # timeouts, connection resets
                error = exc
            if attempt == self.max_retries - 1 or not is_retryable_error(error):
                if is_rate_limited_error(error):
                    self._rate_limited = error
                raise error
            time.sleep(self._retry_delay(error, attempt))
        self.cache.set(url, data)
        return data

    def _json_get(self, url: str, headers: dict[str, str] | None = None) -> Any:
        return self._fetch(url, headers, lambda raw: json.loads(raw.decode("utf-8")))

    def _text_get(self, url: str, headers: dict[str, str] | None = None) -> str:
        return self._fetch(url, headers, lambda raw: raw.decode("utf-8", errors="replace"))

    def search(self, queries: list[RelatedWorkQuery], limit: int = 10) -> ProviderResult:
        raise NotImplementedError


def clean_search_query(query: str, max_chars: int = 260) -> str:
    text = str(query or "")
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\*\*\s*anonymous author\(s\)\s*\*\*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\banonymous author\(s\)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[*_`#>\[\]{}|\\~^]+", " ", text)
    text = re.sub(r"[^\w\s\-+/:.,()]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip(" .,;:-")
    return text[:max_chars].strip(" .,;:-")


def per_query_limit(total_limit: int, num_queries: int) -> int:
    num_queries = max(1, num_queries)
    total_limit = max(1, total_limit)
    return max(1, min(total_limit, (total_limit + num_queries - 1) // num_queries))


def encoded_query(query: str) -> str:
    return urllib.parse.quote_plus(clean_search_query(query))
