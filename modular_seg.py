import re
from pathlib import Path


def segment_md(md_name: str, md_path: str = "data/md") -> dict[str, str]:
    """
    Split a markdown file into all its sections.

    Args:
        md_name: Name of the markdown file (with or without .md extension).
        md_path: Directory containing the markdown file. Defaults to "data/md".

    Returns:
        Dict mapping "## Section Title" (with original # prefix) to section content.
    """
    md_name = Path(md_name)
    if md_name.suffix.lower() != ".md":
        md_name = md_name.with_suffix(".md")

    full_md_path = Path(md_path) / md_name
    if not full_md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {full_md_path}")

    text = full_md_path.read_text(encoding="utf-8")

    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(header_pattern.finditer(text))

    sections = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        key = f"{m.group(1)} {m.group(2).strip()}"
        sections[key] = text[start:end].strip()

    return sections


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python modular_seg.py <md_name> [md_path]")
        sys.exit(1)

    result = segment_md(*sys.argv[1:])
    for sec, content in result.items():
        print(f"[{sec}]\n{content[:400]}\n")
