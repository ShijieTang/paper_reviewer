"""
Parse references from markdown text extracted from academic PDFs.

Handles common reference formats:
  [1] Author, A. (2020). Title. In Proceedings of ...
  [LeCun2015] Author et al. Title. NeurIPS 2015.
  1. Author, A., & Author, B. (2020). Title. Journal, 10(2), 1–20.
"""

import re
from typing import List, Optional

from .models import Reference

# Patterns to split the references section into individual entries
_ENTRY_SPLIT_PATTERNS = [
    # Numbered: [1], [12], [123]
    re.compile(r'(?=^\[(\d{1,3})\])', re.MULTILINE),
    # Author-key: [LeCun2015], [Vaswani+2017]
    re.compile(r'(?=^\[([A-Z][^[\]]{2,30})\])', re.MULTILINE),
    # Plain numbered list: "1." or "1 "
    re.compile(r'(?=^(\d{1,3})[.\s]\s+[A-Z])', re.MULTILINE),
]

# URL regex
_URL_RE = re.compile(
    r'https?://[^\s\)\]\},"\'<>]+'
)

# Year: four digits between 1900-2030
_YEAR_RE = re.compile(r'\b(19\d{2}|20[0-2]\d)\b')

# Title heuristic: text after authors/year that starts with a capital letter
# and is enclosed in quotes, markdown italics/bold, or straight quotes
_QUOTED_TITLE_RE = re.compile(
    r'(?:'
    r'["\u201c]([A-Z][^"\u201d]{10,200})["\u201d]'   # "Title" or \u201cTitle\u201d
    r'|\*{1,2}([A-Z][^*]{10,200}?)\*{1,2}'            # *Title* or **Title**
    r')'
)

# "Title. In Venue" or "Title. Journal"
_TITLE_BEFORE_IN_RE = re.compile(
    r'[\.\,]\s+([A-Z][^\.\n]{10,200}?)\.\s+(?:In |Proceedings|arXiv|CoRR|Journal|IEEE|ACM|NeurIPS|ICLR|ICML|CVPR|ECCV|AAAI|ACL|EMNLP)'
)

# ArXiv ID
_ARXIV_RE = re.compile(r'arXiv[:\s]*(\d{4}\.\d{4,5})', re.IGNORECASE)


def _extract_index(text: str) -> Optional[str]:
    m = re.match(r'^\[([^\]]+)\]', text.strip())
    if m:
        return m.group(1)
    m = re.match(r'^(\d{1,3})[.\s]', text.strip())
    if m:
        return m.group(1)
    return None


def _extract_url(text: str) -> Optional[str]:
    # Prefer arXiv abstract URL
    m = _ARXIV_RE.search(text)
    if m:
        return f"https://arxiv.org/abs/{m.group(1)}"
    urls = _URL_RE.findall(text)
    # Strip trailing punctuation
    urls = [u.rstrip('.,;)]}') for u in urls]
    return urls[0] if urls else None


def _extract_year(text: str) -> Optional[str]:
    m = _YEAR_RE.search(text)
    return m.group(1) if m else None


def _extract_title(text: str) -> Optional[str]:
    # Try quoted or italic-wrapped title first
    m = _QUOTED_TITLE_RE.search(text)
    if m:
        # group(1) = quoted, group(2) = italic/bold markdown
        return (m.group(1) or m.group(2)).strip()

    # Try "... Title. In Conference/Journal ..."
    m = _TITLE_BEFORE_IN_RE.search(text)
    if m:
        return m.group(1).strip()

    # Fallback: remove the [key] prefix and the first author segment,
    # then take the next sentence-like fragment
    clean = re.sub(r'^\[.*?\]\s*', '', text.strip())
    # Remove author block: "LastName, F., LastName, F. (year)." or similar
    clean = re.sub(r'^[A-Z][^.]{3,80}?\.\s+\(?\d{4}\)?\.?\s*', '', clean)
    clean = clean.strip()
    # Take up to first period or newline
    sentence = re.split(r'\.\s+|\n', clean)[0].strip()
    if len(sentence) > 10:
        return sentence
    return None


def _split_into_entries(text: str) -> List[str]:
    """Split the references section text into individual reference strings."""
    text = text.strip()

    # Strip markdown bullet list prefixes ("- ", "* ", "+ ") so that
    # "- [1] Author..." becomes "[1] Author..." and the numbered patterns match.
    text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)

    for pattern in _ENTRY_SPLIT_PATTERNS:
        parts = pattern.split(text)
        # parts[0] is the text before first match (usually empty or header)
        entries = [p.strip() for p in parts if p.strip()]
        # Keep only parts that look like reference entries (not single tokens)
        entries = [e for e in entries if len(e) > 20]
        if len(entries) > 1:
            return entries

    # Last resort: split by double newlines and filter short lines
    entries = [e.strip() for e in re.split(r'\n\s*\n', text) if len(e.strip()) > 20]
    return entries if entries else [text]


def parse_references(text: str) -> List[Reference]:
    """
    Parse a references section (markdown text) into a list of Reference objects.

    Args:
        text: The full text of the references section.

    Returns:
        List of Reference dataclass instances.
    """
    entries = _split_into_entries(text)
    references = []
    for entry in entries:
        ref = Reference(
            raw_text=entry,
            index=_extract_index(entry),
            title=_extract_title(entry),
            year=_extract_year(entry),
            url=_extract_url(entry),
        )
        references.append(ref)
    return references
