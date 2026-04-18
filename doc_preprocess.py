import re
from pathlib import Path
from typing import List

import torch
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# Pick the best available device: CUDA > MPS (Apple Silicon) > CPU.
if torch.cuda.is_available():
    _DEVICE = "cuda"
elif torch.backends.mps.is_available():
    _DEVICE = "mps"
else:
    _DEVICE = "cpu"

# Load models once and reuse across all calls in the same process.
_MODEL_DICT: dict | None = None


def _get_model_dict() -> dict:
    global _MODEL_DICT
    if _MODEL_DICT is None:
        _MODEL_DICT = create_model_dict(device=_DEVICE)
    return _MODEL_DICT

# ---------------------------------------------------------------------------
# Reference-normalisation helpers
# ---------------------------------------------------------------------------

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
    """
    result = []
    for i, entry in enumerate(entries, 1):
        # Strip markdown list markers ("- ", "* ", "+ ")
        entry = re.sub(r'^[-*+]\s+', '', entry, flags=re.MULTILINE)

        # 1. Add [N] prefix when no index marker is present
        if not re.match(r'^\[', entry) and not re.match(r'^\d{1,3}[.\s]', entry):
            entry = f'[{i}] {entry}'

        result.append(entry)
    return result


def _clean_references(text: str) -> str:
    """
    Post-process the References section in a marker-converted markdown document.

    1. Merge page-break-split italic entries (marker splits ``*Confer-*`` and
       ``*ence on …*`` across two list items when a word straddles a page).
    2. Ensure blank lines between entries so the renderer shows them separately.
    3. Normalise each entry for the citation checker by adding ``[N]`` indices.
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

    converter = PdfConverter(artifact_dict=_get_model_dict())
    rendered = converter(str(full_pdf_path))
    text, _, _ = text_from_rendered(rendered)
    text = _clean_document(text)
    text = _clean_references(text)

    output_path.write_text(text, encoding="utf-8")
    print(f"Saved {pdf_name}.md at {output_path}")
    return str(output_path)


def preprocess_md(md_file: str, md_path: str = "data/md") -> str:
    """
    Run cleanup on an existing markdown file (skipping PDF conversion).

    Args:
        md_file: Path to the input .md file.
        md_path: Directory to write the cleaned output. Defaults to "data/md".
                 If md_file already lives there, it is overwritten in place.

    Returns:
        Path to the (cleaned) output markdown file.
    """
    src = Path(md_file)
    if not src.exists():
        raise FileNotFoundError(f"Markdown file not found: {src}")

    output_dir = Path(md_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / src.name

    text = src.read_text(encoding="utf-8")
    text = _clean_document(text)
    text = _clean_references(text)

    output_path.write_text(text, encoding="utf-8")
    print(f"Saved cleaned markdown at {output_path}")
    return str(output_path)


def load_or_create_markdown(pdf_file: str, md_path: str = "data/md") -> str:
    """
    Return the markdown text for a PDF, reusing an existing markdown file when present.

    Args:
        pdf_file: Path to the PDF file, e.g. "data/pdf/example_paper.pdf".
        md_path: Directory containing generated markdown files.

    Returns:
        The markdown contents as a string.
    """
    pdf_path = Path(pdf_file)
    md_dir = Path(md_path)
    md_file = md_dir / pdf_path.with_suffix(".md").name

    if md_file.exists():
        return md_file.read_text(encoding="utf-8")

    md_out = doc_preprocess(
        pdf_name=pdf_path.name,
        pdf_path=str(pdf_path.parent),
        md_path=str(md_dir),
    )
    return Path(md_out).read_text(encoding="utf-8")


if __name__ == "__main__":
    import sys

    pdf_dir = Path("data/pdf")
    md_dir = Path("data/md")

    if len(sys.argv) >= 2 and sys.argv[1] == "--md":
        # Preprocess all .md files in data/unprocessed_md/ → data/md/
        unprocessed_dir = Path("data/unprocessed_md")
        mds = sorted(unprocessed_dir.glob("*.md"))
        if not mds:
            print(f"No markdown files found in {unprocessed_dir}")
            sys.exit(1)
        for md in mds:
            out = preprocess_md(str(md), str(md_dir))
            print(f"Markdown saved to: {out}")
    elif len(sys.argv) >= 2 and Path(sys.argv[1]).suffix.lower() == ".md":
        # Single .md file provided explicitly.
        out = preprocess_md(*sys.argv[1:])
        print(f"Markdown saved to: {out}")
    elif len(sys.argv) >= 2:
        # Single PDF file provided explicitly.
        out = doc_preprocess(*sys.argv[1:])
        print(f"Markdown saved to: {out}")
    else:
        # Auto-discover all PDFs in data/pdf/ and convert each one.
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if not pdfs:
            print(f"No PDF files found in {pdf_dir}")
            sys.exit(1)
        for pdf in pdfs:
            md_file = md_dir / pdf.with_suffix(".md").name
            if md_file.exists():
                print(f"Skipping {pdf.name} — {md_file} already exists")
                continue
            out = doc_preprocess(pdf.name, str(pdf_dir), str(md_dir))
            print(f"Markdown saved to: {out}")
