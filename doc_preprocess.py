import re
from pathlib import Path
from typing import List

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# ---------------------------------------------------------------------------
# Reference-normalisation helpers
# ---------------------------------------------------------------------------

# Detects the end of an author block in author-year format (ICML, NeurIPS,
# Nature, Science, APA journal, …) — either "et al." or a lone capital initial
# not preceded by another word character (e.g. "O." in "Winther, O.").
# The title follows immediately as plain, un-italicised text.
# The venue starts with "*", "In *", "arXiv", or "biorxiv/bioRxiv".
_AUTHOR_YEAR_TITLE_RE = re.compile(
    r'(?:et al\.|(?<!\w)[A-Z]\.)[ \t]+'           # end of author block
    r'(?![A-Z]\.)'                                  # NOT followed by another initial (e.g. "J. J.")
    r'([A-Z][^\n*"]{10,200}?)'                      # plain-text title
    r'\.\s+'                                         # closing period
    r'(?:[Ii]n\s+\*|\*[A-Za-z]|arXiv|[Bb]io[Rr]xiv)',  # start of venue
)


def _clean_document(text: str) -> str:
    """
    Document-wide cleanup of marker PDF-to-markdown conversion artifacts.

    1. Remove all ``<span id="page-X-Y"></span>`` anchor tags wherever they
       appear: inside headings, figure captions, standalone anchor lines before
       equations, and inline within paragraphs.
    2. Demote ``Algorithm N …`` blocks that marker incorrectly promotes to
       top-level headings (the whole algorithm ends up on one line as a title).
    """
    # 1. Remove every <span id="..."></span> tag and any space that follows it.
    #    Handles both ``#### <span …></span>3.1 Title`` and standalone lines.
    text = re.sub(r'<span\s+id="[^"]*"\s*></span>\s*', '', text)

    # After stripping standalone anchor lines the document may gain extra blank
    # lines; collapse runs of 3+ newlines back to two.
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 2. Marker renders algorithm captions as headings, e.g.:
    #      # Algorithm 1 Training … Require: … 1: … 2: …
    #    Strip the leading ``#`` markers so the line becomes plain text.
    text = re.sub(
        r'^#{1,6}\s+(Algorithm\s+\d+\b)',
        r'\1',
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    return text


def _normalize_ref_entries(entries: List[str]) -> List[str]:
    """
    Normalise a list of raw reference strings so the citation checker can
    parse them regardless of the original formatting style:

    1. Add a ``[N]`` numbered prefix if none exists — this lets the checker's
       entry-splitter work for any citation style (ICML, NeurIPS, APA, etc.).
    2. For author-year entries whose title is plain text (not quoted or italic),
       wrap the title in double quotes so ``_QUOTED_TITLE_RE`` in the checker
       finds the correct field instead of the italic venue name.
    """
    result = []
    for i, entry in enumerate(entries, 1):
        # Strip markdown list markers ("- ", "* ", "+ ")
        entry = re.sub(r'^[-*+]\s+', '', entry, flags=re.MULTILINE)

        # 1. Add [N] prefix when no index marker is present
        if not re.match(r'^\[', entry) and not re.match(r'^\d{1,3}[.\s]', entry):
            entry = f'[{i}] {entry}'

        # 2. Wrap plain-text title in quotes for author-year format.
        #    Only act when the title is not already quoted or italic.
        if '"' not in entry:
            m = _AUTHOR_YEAR_TITLE_RE.search(entry)
            if m:
                title = m.group(1).strip()
                entry = entry[:m.start(1)] + f'"{title}"' + entry[m.end(1):]

        result.append(entry)
    return result


def _clean_references(text: str) -> str:
    """
    Post-process the References section in a marker-converted markdown document.

    1. Merge page-break-split italic entries (marker splits ``*Confer-*`` and
       ``*ence on …*`` across two list items when a word straddles a page).
    2. Ensure blank lines between entries so the renderer shows them separately.
    3. Normalise each entry for the citation checker: add ``[N]`` index and
       wrap plain-text titles in quotes (see ``_normalize_ref_entries``).
    """
    ref_match = re.search(r'^(#+\s+References?|#+\s+Bibliography)\s*$',
                          text, re.MULTILINE | re.IGNORECASE)
    if not ref_match:
        return text

    # Locate where the next section heading starts (end of references body)
    next_match = re.search(r'^#{1,6}\s+\S', text[ref_match.end():], re.MULTILINE)
    body_end = ref_match.end() + next_match.start() if next_match else len(text)

    before  = text[:ref_match.start()]
    heading = text[ref_match.start():ref_match.end()]
    body    = text[ref_match.end():body_end]
    after   = text[body_end:]

    # 1. Merge page-break-split italic entries.
    #    ``*The Eleventh International Confer-*``  ← hyphen before closing *
    #    ``- *ence on Learning Representations*``  ← continuation on next item
    #    Fix: remove the hyphen, the surrounding ``*`` pair, the blank line,
    #    and the spurious ``- `` list prefix so both halves join seamlessly.
    body = re.sub(
        r'\*([^*\n]+)-\*[ \t]*\n+[ \t]*-[ \t]+\*',
        r'*\1',
        body,
    )

    # 2. Ensure blank lines between entries for clean rendering.
    body = re.sub(r'\n(- \S)', r'\n\n\1', body)
    body = re.sub(r'\n{3,}', '\n\n', body)

    # 3. Normalise entries for the citation checker.
    entries = [e.strip() for e in re.split(r'\n\s*\n', body) if e.strip()]
    entries = _normalize_ref_entries(entries)
    body = '\n\n'.join(entries) + '\n'

    return before + heading + '\n\n' + body + after


def doc_preprocess(pdf_name: str, pdf_path: str = "data/pdf", md_path: str = "data/md") -> str:
    """
    Convert a PDF file to markdown using the marker library.

    Args:
        pdf_name: Name of the PDF file (with or without .pdf extension).
        pdf_path: Directory containing the PDF file. Defaults to "data/pdf".
        md_path: Directory for the output markdown file. Defaults to "data/md".

    Returns:
        Path to the output markdown file.
    """
    pdf_name = Path(pdf_name)
    if pdf_name.suffix.lower() != ".pdf":
        pdf_name = pdf_name.with_suffix(".pdf")

    full_pdf_path = Path(pdf_path) / pdf_name
    if not full_pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {full_pdf_path}")

    output_dir = Path(md_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / pdf_name.with_suffix(".md")

    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(str(full_pdf_path))
    text, _, _ = text_from_rendered(rendered)
    text = _clean_document(text)
    text = _clean_references(text)

    output_path.write_text(text, encoding="utf-8")
    print(f"Saved {pdf_name}.md at {output_path}")
    return str(output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python doc_preprocess.py <pdf_name> [pdf_path] [md_path]")
        sys.exit(1)

    out = doc_preprocess(*sys.argv[1:])
    print(f"Markdown saved to: {out}")
