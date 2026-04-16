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

# "Title. In Venue" or "Title. Journal/Conference"
# Also handles italic-prefixed venues like *ACM, *Advances, *Proceedings, etc.
_TITLE_BEFORE_IN_RE = re.compile(
    r'[\.\,]\s+([A-Z][^\.\n]{10,200}?)\.\s+'
    r'(?:In\s+|Proceedings|\*?Advances|\*?Journal|\*?Transactions|'
    r'\*?IEEE|\*?ACM|\*?Nature|\*?Frontiers|\*?International|\*?Annual|'
    r'\*?[Aa]r[Xx]iv|\*?CoRR|NeurIPS|ICLR|ICML|CVPR|ECCV|ICCV|AAAI|ACL|EMNLP|NAACL|'
    r'\*[A-Z])'
)

# "Title. Journal Name, 12(3):..." for plain-text journal references.
_TITLE_BEFORE_JOURNAL_RE = re.compile(
    r'[\.\,]\s+([A-Z][^\.\n]{10,200}?)\.\s+'
    r'[A-Z][A-Za-z0-9&/\-\' ]{2,80},\s+\d'
)

# "Title, 2024. URL ..." or "Title. URL ..." for arXiv / report / web-first refs.
_TITLE_BEFORE_YEAR_RE = re.compile(
    r'[\.\,]\s+((?!(?:Technical report|arXiv preprint)\b)[A-Z][^\.\n]{10,200}?)'
    r'(?:,\s*(?:19\d{2}|20[0-2]\d)\b|\.\s+(?:URL|https?://))'
)

# ArXiv ID
_ARXIV_RE = re.compile(r'arXiv[:\s]*(\d{4}\.\d{4,5})', re.IGNORECASE)


def _normalize_entry_text(text: str) -> str:
    """Collapse common PDF extraction artifacts inside a single reference."""
    text = re.sub(r'(?<=\w)-\s*\n\s*(?=\w)', '', text)
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def _looks_like_author_fragment(text: str) -> bool:
    text = text.strip(' "\'.,;:')
    if ":" in text or len(text) > 80:
        return False
    if " and " not in text and "," not in text and "&" not in text and "et al" not in text.lower():
        return False

    tokens = re.findall(r"[A-Za-z][A-Za-z'.-]*", text)
    if not 2 <= len(tokens) <= 10:
        return False

    capitalized = sum(1 for token in tokens if token[0].isupper())
    return capitalized >= max(2, len(tokens) - 2)


def _clean_extracted_title(title: str) -> str:
    """Drop obvious author fragments accidentally captured as part of the title."""
    parts = [part.strip() for part in re.split(r'\.\s+', title) if part.strip()]
    if len(parts) >= 2 and _looks_like_author_fragment(parts[0]) and len(parts[1]) >= 10:
        title = ". ".join(parts[1:])
    return title.strip(' "\'.,;:')


def _looks_like_metadata_fragment(text: str) -> bool:
    text = text.strip(' "\'.,;:')
    if not text:
        return True

    tokens = re.findall(r"[A-Za-z][A-Za-z'.-]*", text)
    if not tokens:
        return True

    return len(tokens) <= 2 and len(text) <= 24


def _extract_index(text: str) -> Optional[str]:
    text = _normalize_entry_text(text)
    m = re.match(r'^\[([^\]]+)\]', text.strip())
    if m:
        return m.group(1)
    m = re.match(r'^(\d{1,3})[.\s]', text.strip())
    if m:
        return m.group(1)
    return None


def _extract_url(text: str) -> Optional[str]:
    text = _normalize_entry_text(text)
    # Prefer arXiv abstract URL
    m = _ARXIV_RE.search(text)
    if m:
        return f"https://arxiv.org/abs/{m.group(1)}"
    urls = _URL_RE.findall(text)
    # Strip trailing punctuation
    urls = [u.rstrip('.,;)]}') for u in urls]
    return urls[0] if urls else None


def _extract_year(text: str) -> Optional[str]:
    text = _normalize_entry_text(text)
    m = _YEAR_RE.search(text)
    return m.group(1) if m else None


def _extract_title(text: str) -> Optional[str]:
    text = _normalize_entry_text(text)

    # 1. Explicitly double/curly-quoted title — most precise signal
    m = re.search(r'["\u201c]([A-Z][^"\u201d]{10,200})["\u201d]', text)
    if m:
        return _clean_extracted_title(m.group(1))

    # 2. Plain-text title immediately before a venue keyword or italic-prefixed
    #    venue (e.g. NeurIPS/ICML/APA style: "Title. *Journal* ..." or
    #    "Title. In Proceedings ...").  Try this BEFORE italic matching so we
    #    don't accidentally capture the venue name itself as the title.
    m = _TITLE_BEFORE_IN_RE.search(text)
    if m:
        return _clean_extracted_title(m.group(1))

    # 3. Plain-text title before a journal name and volume, e.g.
    #    "Title. Nature, 123(4):..." or "Title. Processes, 8(10):..."
    m = _TITLE_BEFORE_JOURNAL_RE.search(text)
    if m:
        return _clean_extracted_title(m.group(1))

    # 4. Plain-text title followed directly by a year / URL, common in arXiv,
    #    technical report, and software benchmark references.
    m = _TITLE_BEFORE_YEAR_RE.search(text)
    if m:
        return _clean_extracted_title(m.group(1))

    # 5. Italic/bold-wrapped title — last resort only; risk of grabbing venue
    m = re.search(r'\*{1,2}([A-Z][^*]{10,200}?)\*{1,2}', text)
    if m:
        return _clean_extracted_title(m.group(1))

    # Fallback: remove the [key] prefix and the first author segment,
    # then take the next sentence-like fragment
    clean = re.sub(r'^\[.*?\]\s*', '', text.strip())
    clean = re.sub(r'^\d{1,3}[.\s]\s*', '', clean)
    # Remove author block: "LastName, F., LastName, F. (year)." or similar
    clean = re.sub(r'^[A-Z][^.]{3,80}?\.\s+\(?\d{4}\)?\.?\s*', '', clean)
    clean = clean.strip()
    parts = [part.strip() for part in re.split(r'\.\s+|\n', clean) if part.strip()]
    if parts:
        sentence = parts[0]
        if len(parts) >= 2 and (_looks_like_metadata_fragment(sentence) or _looks_like_author_fragment(sentence)):
            sentence = parts[1]
        if len(sentence) > 10:
            return _clean_extracted_title(sentence)
    return None


def _split_into_entries(text: str) -> List[str]:
    """Split the references section text into individual reference strings."""
    text = text.strip()
    text = re.sub(r'(?<=\w)-\s*\n\s*(?=\w)', '', text)

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
