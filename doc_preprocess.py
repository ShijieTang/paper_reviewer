from pathlib import Path

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


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

    output_path.write_text(text, encoding="utf-8")
    print(f"Saved {pdf_name}.md at {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python doc_preprocess.py <pdf_name> [pdf_path] [md_path]")
        sys.exit(1)

    out = doc_preprocess(*sys.argv[1:])
    print(f"Markdown saved to: {out}")
