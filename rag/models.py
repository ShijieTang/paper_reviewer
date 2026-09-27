from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TargetPaperSummary:
    paper_id: str
    title: str
    abstract: str
    topic: str = ""
    claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RelatedWorkQuery:
    group: str
    query: str
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PaperMetadata:
    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    publication_date: str | None = None
    venue: str = ""
    abstract: str = ""
    url: str = ""
    doi: str = ""
    arxiv_id: str = ""
    source_ids: dict[str, str] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    matched_query_groups: list[str] = field(default_factory=list)
    citation_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RerankedPaper:
    rank: int
    paper_id: str
    relevance_score: float
    relevance_types: list[str] = field(default_factory=list)
    rationale: str = ""
    evidence_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


