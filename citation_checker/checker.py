"""
Main citation checker orchestrator.

Verification pipeline (stops at first pass):
  1. OpenReview  — covers ICLR, NeurIPS, ICML and other OR-hosted venues
  2. arXiv       — covers preprints and most ML papers
  3. CrossRef    — DOI resolution + title query; covers journals and all major venues
  4. URL check   — last resort: any URL in the reference must be reachable
"""

import logging
import sys
from typing import List

from .arxiv_checker import check_on_arxiv
from .doi_checker import check_via_doi
from .models import CheckResult, FailedReference, Reference, VerificationStatus
from .openreview_checker import check_on_openreview
from .url_checker import check_url_accessible

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def _check_single(ref: Reference) -> CheckResult:
    if not ref.title:
        return CheckResult(
            reference=ref,
            status=VerificationStatus.SKIPPED,
            warning="Could not extract title from reference — skipped.",
            details="No title parsed.",
        )

    # --- Step 1: OpenReview ---
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

    # --- Step 2: arXiv ---
    logger.info("[%s] Checking arXiv: %s", ref.index or "?", ref.title[:80])
    ax_found, ax_url = check_on_arxiv(ref.title)
    if ax_found:
        return CheckResult(
            reference=ref,
            status=VerificationStatus.VERIFIED,
            verified_by="arXiv",
            details=f"Found on arXiv: {ax_url or 'N/A'}",
        )

    # --- Step 3: CrossRef / DOI ---
    logger.info("[%s] Checking CrossRef: %s", ref.index or "?", ref.title[:80])
    cr_found, cr_url = check_via_doi(ref.title, raw_text=ref.raw_text)
    if cr_found:
        return CheckResult(
            reference=ref,
            status=VerificationStatus.VERIFIED,
            verified_by="CrossRef/DOI",
            details=f"Found via CrossRef: {cr_url or 'N/A'}",
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
            details=f"Not in OpenReview/arXiv/CrossRef, but URL accessible ({url_reason}): {url}",
        )

    # --- Not found anywhere ---
    if url and not url_accessible:
        warning = (
            f"WARNING: Not found on OpenReview, arXiv, or CrossRef, and URL is "
            f"inaccessible ({url_reason}). This reference may be AI-hallucinated."
        )
        status = VerificationStatus.SUSPICIOUS
    else:
        warning = (
            "WARNING: Not found on OpenReview, arXiv, or CrossRef, and no URL provided. "
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
        return "not in OpenReview/arXiv/CrossRef; URL accessible only"
    if result.status == VerificationStatus.SUSPICIOUS:
        url_part = result.details.split("—")[-1].strip() if "—" in result.details else ""
        return f"not found anywhere; URL inaccessible ({url_part})" if url_part else "not found anywhere; URL inaccessible"
    if result.status == VerificationStatus.NOT_FOUND:
        return "not found on OpenReview, arXiv, or CrossRef; no URL provided"
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
    show_progress: bool | None = None,
    progress_desc: str = "Checking citations",
) -> List[CheckResult]:
    """
    Check all references and return a list of CheckResult objects.

    Processes references sequentially to respect API rate limits.
    """
    if show_progress is None:
        show_progress = sys.stderr.isatty()

    iterator = references
    if show_progress and tqdm is not None:
        iterator = tqdm(references, total=len(references), desc=progress_desc, unit="ref")

    results = []
    for i, ref in enumerate(iterator, 1):
        logger.info("Checking reference %d/%d", i, len(references))
        result = _check_single(ref)
        results.append(result)
    return results
