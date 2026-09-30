from __future__ import annotations

import hashlib
import re

from .models import TargetPaperSummary


CLAIM_PATTERNS = (
    r"\bwe (?:propose|present|introduce|show|demonstrate|prove|develop|study)\b[^.?!]*[.?!]",
    r"\bour (?:method|model|approach|framework|algorithm)\b[^.?!]*[.?!]",
)


def _clean_title(text: str) -> str:
    text = re.sub(r"\*\*\s*anonymous author\(s\)\s*\*\*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\banonymous author\(s\)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[*_`]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" #\t-")


def sanitize_markdown(text: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<script\b.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _first_heading(text: str) -> str:
    match = re.search(r"(?m)^#\s+(.+?)\s*$", text)
    if match:
        title = _clean_title(match.group(1))
        return title or "Untitled paper"
    for line in text.splitlines():
        stripped = _clean_title(line)
        if stripped:
            return stripped[:240]
    return "Untitled paper"


def _section(text: str, name: str, max_chars: int = 3000) -> str:
    match = re.search(rf"(?im)^#+\s*{re.escape(name)}\s*$", text)
    if not match:
        return ""
    start = match.end()
    next_header = re.search(r"(?m)^#+\s+\S", text[start:])
    end = start + next_header.start() if next_header else len(text)
    return text[start:end].strip()[:max_chars]


def _abstract(text: str) -> str:
    abstract = _section(text, "abstract", max_chars=2500)
    if abstract:
        return abstract
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs[1][:2500] if len(paragraphs) > 1 else (paragraphs[0][:2500] if paragraphs else "")


def _claims(text: str) -> list[str]:
    claims: list[str] = []
    search_text = " ".join([_abstract(text), _section(text, "introduction", max_chars=5000)])
    for pattern in CLAIM_PATTERNS:
        for match in re.finditer(pattern, search_text, flags=re.IGNORECASE):
            claim = re.sub(r"\s+", " ", match.group(0)).strip()
            if 20 <= len(claim) <= 500 and claim not in claims:
                claims.append(claim)
            if len(claims) >= 12:
                return claims
    return claims


def summarize_target_paper(
    markdown: str,
    topic: str = "",
    title_override: str = "",
) -> TargetPaperSummary:
    clean = sanitize_markdown(markdown)
    title = _clean_title(title_override) if str(title_override or "").strip() else _first_heading(clean)
    paper_hash = hashlib.sha1((title + "\n" + clean[:5000]).encode("utf-8")).hexdigest()[:12]
    return TargetPaperSummary(
        paper_id=f"paper_{paper_hash}",
        title=title,
        abstract=_abstract(clean),
        topic=topic,
        claims=_claims(clean),
    )


def llm_context_excerpt(
    markdown: str,
    max_chars: int = 12000,
    title_override: str = "",
) -> str:
    clean = sanitize_markdown(markdown)
    title = _clean_title(title_override) if str(title_override or "").strip() else _first_heading(clean)
    parts = [
        "# Title\n" + title,
        "## Abstract\n" + _abstract(clean),
        "## Introduction\n" + _section(clean, "introduction", max_chars=5000),
        "## Experiments\n" + _section(clean, "experiments", max_chars=2500),
        "## Limitations\n" + (_section(clean, "limitations", max_chars=1500) or _section(clean, "limitations and conclusions", max_chars=1500)),
    ]
    return "\n\n".join(part for part in parts if part.strip())[:max_chars]
