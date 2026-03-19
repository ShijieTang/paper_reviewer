import json
import re
from pathlib import Path

# Matches headings whose title is "References", "Reference", or "Bibliography"
_REF_HEADING_RE = re.compile(r"(?i)^(references?|bibliography)$")

# Detects the start of a numbered reference entry: [1], [Vaswani17], or "1. "
_ENTRY_START_RE = re.compile(
    r"^\[(\d{1,3}|[A-Z][^\[\]]{2,30})\]"   # [1] or [AuthorKey]
    r"|^(\d{1,3})[.\s]\s+[A-Z]"             # "1. " or "1 A..."
)


def normalize_references(content: str) -> str:
    """
    Normalize a references section so the citation checker can parse every entry.

    Transformations:
      1. Strip markdown bullet list prefixes  ``- [1]`` → ``[1]``
      2. Join soft-wrapped continuation lines back onto their parent entry
         so each reference is a single unbroken line
      3. Collapse consecutive blank lines to a single blank line
    """
    # 1. Strip leading bullet prefix only when immediately before a ref entry
    content = re.sub(r"^[-*+]\s+", "", content, flags=re.MULTILINE)

    # 2. Join continuation lines
    joined: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            joined.append("")                          # preserve blank separator
        elif _ENTRY_START_RE.match(line):
            joined.append(line)                        # new entry → new line
        elif joined and joined[-1] != "":
            joined[-1] += " " + line                   # continuation → append
        else:
            joined.append(line)

    # 3. Collapse multiple consecutive blank lines
    result: list[str] = []
    for line in joined:
        if line == "" and result and result[-1] == "":
            continue
        result.append(line)

    return "\n".join(result).strip()


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
        start   = m.end()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        key     = f"{m.group(1)} {m.group(2).strip()}"
        content = text[start:end].strip()

        # Auto-normalize any references / bibliography section
        if _REF_HEADING_RE.match(m.group(2).strip()):
            content = normalize_references(content)

        sections[key] = content

    return sections


def user_update_sec(sections: dict[str, str], user_responses: dict[str, str]) -> dict[str, str]:
    """
    Update section header levels based on user input.

    Args:
        sections: Dict of {"## Title": content} from segment_md.
        user_responses: Dict of {"## Title": "h1"|"h2"|...|"h6"} from the user.
                        Sections without a response are kept as-is.

    Returns:
        Updated dict with header keys adjusted to the user-specified # level.
    """
    updated = {}
    for header, content in sections.items():
        response = user_responses.get(header, "").strip().lower()
        if response in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(response[1])
            title = re.sub(r"^#{1,6}\s*", "", header)
            new_header = f"{'#' * level} {title}"
        else:
            new_header = header
        updated[new_header] = content
    return updated


def save_sections(md_name: str, sections: dict[str, str], suffix: str = "raw", save_path: str = "data/test") -> str:
    """
    Save a sections dict to test/<md_stem>_<suffix>.json.

    Args:
        md_name: Name of the source markdown file (used to derive the output filename).
        sections: Dict of {header: content} to save.
        suffix: "raw" for pre-user-input, "reviewed" for post-user-input.
        save_path: Output directory. Defaults to "test".

    Returns:
        Path to the saved file.
    """
    stem = Path(md_name).stem
    out_dir = Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_{suffix}.json"
    out_path.write_text(json.dumps(sections, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out_path)


def reconstruct_md(sections: dict[str, str]) -> str:
    """
    Rebuild a markdown string from a sections dict (inverse of segment_md).

    Args:
        sections: Dict of {"## Title": "content..."} as returned by segment_md.

    Returns:
        Full markdown text with each section reassembled as "header\n\ncontent\n\n".
    """
    parts = []
    for header, content in sections.items():
        parts.append(f"{header}\n\n{content}")
    return "\n\n".join(parts) + "\n"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python modular_seg.py <md_name> [md_path]")
        sys.exit(1)

    result = segment_md(*sys.argv[1:])
    for sec, content in result.items():
        print(f"[{sec}]\n{content[:400]}\n")
