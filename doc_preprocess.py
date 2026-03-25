import re
from pathlib import Path

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


def _clean_references(text: str) -> str:
    """
    Post-process the References section in a marker-converted markdown document.

    Fixes three common artifacts produced by PDF-to-markdown conversion:
    1. Inline ``<span id="page-X-Y"></span>`` anchor tags injected by marker.
    2. Page-break-split list items: marker sometimes breaks a single reference
       across two ``-`` items when a hyphenated word straddles a page boundary,
       e.g. ``*International Confer-*`` on one line and ``- *ence on ...*`` on
       the next.  These are merged back into one clean entry.
    3. Missing blank lines between reference entries (tight list → wall of text).
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

    # 1. Strip <span id="..."></span> anchor tags
    body = re.sub(r'<span\s[^>]*></span>', '', body)

    # 2. Merge page-break-split italic entries.
    #    Matches a word ending in a hyphen just before the closing italic marker,
    #    e.g. ``*International Confer-*`` followed (possibly across a blank line)
    #    by ``- *ence on ...``.  Removes the hyphen, both adjacent ``*`` markers,
    #    the blank line, and the spurious ``- `` list prefix.
    body = re.sub(
        r'\*([^*\n]+-)-\*[ \t]*\n+[ \t]*-[ \t]+\*',
        r'*\1',
        body,
    )

    # 3. Ensure every reference entry is preceded by a blank line so that the
    #    markdown renderer displays them as separate paragraphs, not a wall of text.
    body = re.sub(r'\n(- \S)', r'\n\n\1', body)
    body = re.sub(r'\n{3,}', '\n\n', body)   # collapse accidental triple+ blanks

    return before + heading + body + after


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
