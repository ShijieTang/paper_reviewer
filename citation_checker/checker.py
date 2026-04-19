"""
Main citation checker orchestrator.

Verification pipeline (stops at first pass):
  1. arXiv       — covers preprints and many ML papers
  2. CrossRef    — DOI resolution + title query; covers journals and all major venues
  3. DBLP        — fallback for CS conference papers with weak DOI coverage
  4. URL check   — last resort: any URL in the reference must be reachable
  5. OpenReview  — optional backend for OR-hosted venues when explicitly enabled
"""

import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from .arxiv_checker import check_on_arxiv, configure_arxiv
from .dblp_checker import check_on_dblp, configure_dblp
from .doi_checker import check_via_doi, configure_crossref
from .lookup_utils import normalize_title_for_lookup
from .models import CheckResult, FailedReference, Reference, VerificationStatus
from .openreview_checker import check_on_openreview, configure_openreview
from .url_checker import check_url_accessible

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


_OPENREVIEW_VENUE_HINTS = (
    "iclr",
    "international conference on learning representations",
    "neurips",
    "nips",
    "icml",
    "international conference on machine learning",
    "openreview",
)
_NON_OPENREVIEW_HINTS = (
    "acm transactions",
    "tog",
    "siggraph",
    "mit press",
    "gdc",
    "tutorial",
    "thesis",
    "book",
    "springer",
    "journal",
    "transactions",
    "nature",
    "science",
    "proc. of gdc",
)
_WEB_REFERENCE_HINTS = (
    "gdc",
    "tutorial",
    "vault",
    "blog",
    "website",
    "webpage",
)


def _raw_lower(ref: Reference) -> str:
    return (ref.raw_text or "").casefold()


def _is_openreview_candidate(ref: Reference) -> bool:
    raw = _raw_lower(ref)
    if any(hint in raw for hint in _NON_OPENREVIEW_HINTS):
        return False
    if any(hint in raw for hint in _OPENREVIEW_VENUE_HINTS):
        return True
    if ref.year and ref.year.isdigit() and int(ref.year) < 2018:
        return False
    return False


def _has_doi(ref: Reference) -> bool:
    return bool(re.search(r"\b10\.\d{4,9}/", ref.raw_text or "", re.IGNORECASE))


def _has_arxiv(ref: Reference) -> bool:
    raw = ref.raw_text or ""
    return "arxiv" in raw.casefold() or "arxiv.org/abs/" in raw.casefold()


def _is_web_reference(ref: Reference) -> bool:
    raw = _raw_lower(ref)
    return bool(ref.url) and any(hint in raw for hint in _WEB_REFERENCE_HINTS)


def _is_dblp_candidate(ref: Reference) -> bool:
    raw = _raw_lower(ref)
    if any(hint in raw for hint in ("mit press", "gdc", "tutorial", "website", "webpage", "blog", "thesis", "book")):
        return False
    return True


def _check_single(ref: Reference, enable_openreview: bool = False) -> CheckResult:
    if not ref.title:
        return CheckResult(
            reference=ref,
            status=VerificationStatus.SKIPPED,
            warning="Could not extract title from reference — skipped.",
            details="No title parsed.",
        )

    # --- Early exits: explicit arXiv/DOI metadata and stable web URLs ---
    if _has_arxiv(ref):
        logger.info("[%s] Checking arXiv via explicit metadata: %s", ref.index or "?", ref.title[:80])
        ax_found, ax_url = check_on_arxiv(ref.title, raw_text=ref.raw_text)
        if ax_found:
            return CheckResult(
                reference=ref,
                status=VerificationStatus.VERIFIED,
                verified_by="arXiv",
                details=f"Found on arXiv: {ax_url or 'N/A'}",
            )

    if _has_doi(ref):
        logger.info("[%s] Checking CrossRef DOI: %s", ref.index or "?", ref.title[:80])
        cr_found, cr_url = check_via_doi(ref.title, raw_text=ref.raw_text, year=ref.year)
        if cr_found:
            return CheckResult(
                reference=ref,
                status=VerificationStatus.VERIFIED,
                verified_by="CrossRef/DOI",
                details=f"Found via CrossRef: {cr_url or 'N/A'}",
            )

    if _is_web_reference(ref):
        logger.info("[%s] Checking URL early for web reference: %s", ref.index or "?", ref.url)
        url_accessible, url_reason = check_url_accessible(ref.url)
        if url_accessible:
            return CheckResult(
                reference=ref,
                status=VerificationStatus.URL_ONLY,
                verified_by="URL",
                url_accessible=True,
                details=f"Stable URL accessible ({url_reason}): {ref.url}",
            )

    # --- Optional OpenReview ---
    if enable_openreview and _is_openreview_candidate(ref):
        logger.info("[%s] Checking OpenReview: %s", ref.index or "?", ref.title[:80])
        or_found, or_url = check_on_openreview(ref.title)
        if or_found:
            return CheckResult(
                reference=ref,
                status=VerificationStatus.VERIFIED,
                verified_by="OpenReview",
                openreview_found=True,
                openreview_url=or_url,
                details=f"Found on OpenReview: {or_url or 'N/A'}",
            )

    # --- Step 1: arXiv ---
    logger.info("[%s] Checking arXiv: %s", ref.index or "?", ref.title[:80])
    ax_found, ax_url = check_on_arxiv(ref.title, raw_text=ref.raw_text)
    if ax_found:
        return CheckResult(
            reference=ref,
            status=VerificationStatus.VERIFIED,
            verified_by="arXiv",
            details=f"Found on arXiv: {ax_url or 'N/A'}",
        )

    # --- Step 2: CrossRef / DOI ---
    logger.info("[%s] Checking CrossRef: %s", ref.index or "?", ref.title[:80])
    cr_found, cr_url = check_via_doi(ref.title, raw_text=ref.raw_text, year=ref.year)
    if cr_found:
        return CheckResult(
            reference=ref,
            status=VerificationStatus.VERIFIED,
            verified_by="CrossRef/DOI",
            details=f"Found via CrossRef: {cr_url or 'N/A'}",
        )

    # --- Step 3: DBLP ---
    if _is_dblp_candidate(ref):
        logger.info("[%s] Checking DBLP: %s", ref.index or "?", ref.title[:80])
        dblp_found, dblp_url = check_on_dblp(ref.title)
        if dblp_found:
            return CheckResult(
                reference=ref,
                status=VerificationStatus.VERIFIED,
                verified_by="DBLP",
                details=f"Found on DBLP: {dblp_url or 'N/A'}",
            )

    # --- Step 4: URL check ---
    url = ref.url
    url_accessible = None
    url_reason = ""

    if url:
        logger.info("[%s] Checking URL: %s", ref.index or "?", url)
        url_accessible, url_reason = check_url_accessible(url)

    if url_accessible:
        return CheckResult(
            reference=ref,
            status=VerificationStatus.URL_ONLY,
            verified_by="URL",
            url_accessible=True,
            details=f"Not in arXiv/CrossRef/DBLP, but URL accessible ({url_reason}): {url}",
        )

    # --- Not found anywhere ---
    if url and not url_accessible:
        warning = (
            f"WARNING: Not found on arXiv, CrossRef, or DBLP, and URL is "
            f"inaccessible ({url_reason}). This reference may be AI-hallucinated."
        )
        status = VerificationStatus.SUSPICIOUS
    else:
        warning = (
            "WARNING: Not found on arXiv, CrossRef, or DBLP, and no URL provided. "
            "This reference may be AI-hallucinated."
        )
        status = VerificationStatus.NOT_FOUND

    return CheckResult(
        reference=ref,
        status=status,
        url_accessible=url_accessible,
        warning=warning,
        details=f"URL checked: {url or 'none'} — {url_reason or 'N/A'}",
    )


def _fail_reason(result: CheckResult) -> str:
    ref = result.reference
    if result.status == VerificationStatus.SKIPPED:
        return "title could not be parsed"
    if result.status == VerificationStatus.URL_ONLY:
        return "not in arXiv/CrossRef/DBLP; URL accessible only"
    if result.status == VerificationStatus.SUSPICIOUS:
        url_part = result.details.split("—")[-1].strip() if "—" in result.details else ""
        return f"not found anywhere; URL inaccessible ({url_part})" if url_part else "not found anywhere; URL inaccessible"
    if result.status == VerificationStatus.NOT_FOUND:
        return "not found on arXiv, CrossRef, or DBLP; no URL provided"
    return "unknown"


def failed_references(results: List[CheckResult]) -> List[FailedReference]:
    """
    Return all references that were not fully verified, including:
      - URL_ONLY   : accessible URL but not found in any paper database
      - NOT_FOUND  : not found anywhere, no URL
      - SUSPICIOUS : not found anywhere, URL dead
      - SKIPPED    : title could not be parsed

    Returns:
        List of FailedReference(index, title, link, fail_reason)
    """
    _UNVERIFIED = (
        VerificationStatus.URL_ONLY,
        VerificationStatus.NOT_FOUND,
        VerificationStatus.SUSPICIOUS,
        VerificationStatus.SKIPPED,
    )
    return [
        FailedReference(
            index=r.reference.index,
            title=r.reference.title,
            link=r.reference.url,
            fail_reason=_fail_reason(r),
        )
        for r in results if r.status in _UNVERIFIED
    ]


def check_references(
    references: List[Reference],
    show_progress: Optional[bool] = None,
    progress_desc: str = "Checking citations",
    max_workers: Optional[int] = None,
    backend_concurrency: Optional[dict] = None,
    use_cache: bool = True,
    enable_openreview: bool = False,
) -> List[CheckResult]:
    """
    Check all references and return a list of CheckResult objects.

    Processes references sequentially to respect API rate limits.
    """
    if show_progress is None:
        show_progress = sys.stderr.isatty()

    if backend_concurrency is None:
        backend_concurrency = {}

    if enable_openreview:
        configure_openreview(
            concurrency=backend_concurrency.get("openreview", 2),
            use_cache=use_cache,
            reset_cache=True,
        )
    configure_arxiv(
        concurrency=backend_concurrency.get("arxiv", 2),
        use_cache=use_cache,
        reset_cache=True,
    )
    configure_crossref(
        concurrency=backend_concurrency.get("crossref", 4),
        use_cache=use_cache,
        reset_cache=True,
    )
    configure_dblp(
        concurrency=backend_concurrency.get("dblp", 2),
        use_cache=use_cache,
        reset_cache=True,
    )

    if max_workers is None:
        max_workers = min(8, max(1, len(references)))

    def _run_one(item):
        i, ref = item
        logger.info("Checking reference %d/%d", i + 1, len(references))
        return i, _check_single(ref, enable_openreview=enable_openreview)

    results = [None] * len(references)
    unique_items = []
    duplicate_to_source = {}
    if use_cache:
        seen = {}
        for i, ref in enumerate(references):
            if not ref.title:
                unique_items.append((i, ref))
                continue
            key = (normalize_title_for_lookup(ref.title), ref.year or "")
            if key in seen:
                duplicate_to_source[i] = seen[key]
            else:
                seen[key] = i
                unique_items.append((i, ref))
    else:
        unique_items = list(enumerate(references))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        iterator = executor.map(_run_one, unique_items)
        if show_progress and tqdm is not None:
            iterator = tqdm(iterator, total=len(unique_items), desc=progress_desc, unit="ref")
        for i, result in iterator:
            results[i] = result
    for dup_index, source_index in duplicate_to_source.items():
        results[dup_index] = results[source_index]
    return results
