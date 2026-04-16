#!/usr/bin/env python3
"""
Citation checker CLI for the paper reviewer.

Usage:
    python check_citations.py references.md
    python check_citations.py paper.md --section references
    python check_citations.py references.md --output report.json
    python check_citations.py references.md --verbose

The input file should contain the extracted references section from a paper
(either the full MD file or just the references section).
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from citation_checker import check_references, failed_references, parse_references
from citation_checker.models import FailedReference, VerificationStatus

# ANSI colors
_RED = "\033[91m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"
_BLUE = "\033[94m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _extract_references_section(text: str, section_name: str = "references") -> str:
    """
    Extract the references/bibliography section from a full paper MD file.
    Falls back to returning the full text if the section is not found.
    """
    # Common section header patterns
    patterns = [
        rf'(?im)^#+\s*{re.escape(section_name)}\s*$',
        rf'(?im)^{re.escape(section_name.title())}\s*\n[-=]+',
        rf'(?im)^\*\*{re.escape(section_name.title())}\*\*\s*$',
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            # Take text from this heading to the next heading or end of file
            start = m.end()
            next_heading = re.search(r'(?m)^#+\s+\S', text[start:])
            end = start + next_heading.start() if next_heading else len(text)
            return text[start:end].strip()

    # No section found — assume the whole file is references
    return text.strip()


def _status_color(status: VerificationStatus) -> str:
    if status == VerificationStatus.VERIFIED:
        return _GREEN
    if status == VerificationStatus.URL_ONLY:
        return _BLUE
    if status == VerificationStatus.SUSPICIOUS:
        return _RED
    if status == VerificationStatus.NOT_FOUND:
        return _YELLOW
    return ""


def _print_failed_table(failed: list[FailedReference]) -> None:
    """Print failed references as a formatted table."""
    if not failed:
        print(f"{_GREEN}{_BOLD}No failed references — all citations verified.{_RESET}\n")
        return

    # Column widths
    W_IDX = 8
    W_TITLE = 55
    W_LINK = 40
    W_REASON = 45

    header = (
        f"{'Index':<{W_IDX}}  {'Title':<{W_TITLE}}  {'Link':<{W_LINK}}  {'Fail Reason':<{W_REASON}}"
    )
    sep = "-" * len(header)

    print(f"\n{_RED}{_BOLD}=== Failed References ==={_RESET}")
    print(f"{_BOLD}{header}{_RESET}")
    print(sep)

    for f in failed:
        idx = (f.index or "?")[:W_IDX]
        title = (f.title or "(unknown)")[:W_TITLE]
        link = (f.link or "—")[:W_LINK]
        reason = f.fail_reason[:W_REASON]
        print(f"{idx:<{W_IDX}}  {title:<{W_TITLE}}  {link:<{W_LINK}}  {reason}")

    print(sep)
    print(f"{_RED}{_BOLD}{len(failed)} reference(s) failed.{_RESET}\n")


def _print_report(results, verbose: bool = False) -> None:
    total = len(results)
    verified = sum(1 for r in results if r.status == VerificationStatus.VERIFIED)
    url_only = sum(1 for r in results if r.status == VerificationStatus.URL_ONLY)
    not_found = sum(1 for r in results if r.status == VerificationStatus.NOT_FOUND)
    suspicious = sum(1 for r in results if r.status == VerificationStatus.SUSPICIOUS)
    skipped = sum(1 for r in results if r.status == VerificationStatus.SKIPPED)

    verified_by = {}
    for r in results:
        if r.status == VerificationStatus.VERIFIED and r.verified_by:
            verified_by[r.verified_by] = verified_by.get(r.verified_by, 0) + 1

    print(f"\n{_BOLD}=== Citation Check Summary ==={_RESET}")
    print(f"Total references checked : {total}")
    if verified:
        by_str = ", ".join(f"{src}: {n}" for src, n in sorted(verified_by.items()))
        print(f"{_GREEN}  Verified               : {verified}  ({by_str}){_RESET}")
    else:
        print(f"{_GREEN}  Verified               : {verified}{_RESET}")
    print(f"{_BLUE}  URL accessible only    : {url_only}{_RESET}")
    print(f"{_YELLOW}  Not found              : {not_found}{_RESET}")
    print(f"{_RED}  Suspicious (AI risk)   : {suspicious}{_RESET}")
    print(f"  Skipped (no title)     : {skipped}")

    if verbose:
        print()
        for result in results:
            ref = result.reference
            color = _status_color(result.status)
            idx = f"[{ref.index}]" if ref.index else "[?]"
            by = f" via {result.verified_by}" if result.verified_by else ""
            print(f"{color}{_BOLD}{idx} {result.status.value.upper()}{by}{_RESET}  {(ref.title or '(unknown)')[:90]}")
            if ref.url:
                print(f"     URL     : {ref.url}")
            if result.openreview_url:
                print(f"     OR URL  : {result.openreview_url}")
            print(f"     Details : {result.details}")
            print()


def _to_json(failed: list[FailedReference]) -> list:
    """Serialize only failed references to the output format."""
    return [
        {
            "index": f.index,
            "title": f.title,
            "link": f.link,
            "fail_reason": f.fail_reason,
        }
        for f in failed
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Check citations in a paper's references section."
    )
    parser.add_argument("input", help="Path to a .md file with references")
    parser.add_argument(
        "--section",
        default="references",
        help="Name of the references section heading to extract (default: 'references')",
    )
    parser.add_argument(
        "--output",
        help="Write JSON report to this file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show details for all references, not just suspicious ones",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    text = input_path.read_text(encoding="utf-8")
    refs_text = _extract_references_section(text, section_name=args.section)

    print(f"Parsing references from: {input_path}")
    references = parse_references(refs_text)
    print(f"Found {len(references)} reference entries.\n")

    if not references:
        print("No references found. Check that the file contains a references section.")
        sys.exit(0)

    print("Checking each reference (this may take a moment due to API rate limits)...")
    results = check_references(references, show_progress=True)
    failed = failed_references(results)

    _print_report(results, verbose=args.verbose)
    _print_failed_table(failed)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps(_to_json(failed), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"JSON report written to: {out_path}")


if __name__ == "__main__":
    main()
