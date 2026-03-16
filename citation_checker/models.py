from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class VerificationStatus(Enum):
    VERIFIED = "verified"           # Found on OpenReview or URL accessible
    NOT_FOUND = "not_found"         # Not found on OpenReview, no URL or URL inaccessible
    URL_ONLY = "url_only"           # Not on OpenReview but URL accessible
    SUSPICIOUS = "suspicious"       # Strong signs of AI hallucination
    SKIPPED = "skipped"             # Could not parse enough info to check


@dataclass
class Reference:
    raw_text: str
    index: Optional[str] = None         # e.g. "1", "2", "LeCun2015"
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[str] = None
    venue: Optional[str] = None
    url: Optional[str] = None


@dataclass
class CheckResult:
    reference: Reference
    status: VerificationStatus
    verified_by: Optional[str] = None      # "OpenReview", "arXiv", "CrossRef/DOI", "URL"
    openreview_found: bool = False
    openreview_url: Optional[str] = None
    url_accessible: Optional[bool] = None  # None = not checked
    warning: Optional[str] = None
    details: str = ""

    @property
    def is_suspicious(self) -> bool:
        return self.status in (VerificationStatus.NOT_FOUND, VerificationStatus.SUSPICIOUS)


@dataclass
class FailedReference:
    """Compact record for a reference that failed verification."""
    index: Optional[str]       # e.g. "1", "LeCun2015"
    title: Optional[str]       # Title as written in the paper
    link: Optional[str]        # URL from the reference (if any)
    fail_reason: str           # Human-readable reason for failure
