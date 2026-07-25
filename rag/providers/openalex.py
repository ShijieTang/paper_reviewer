from __future__ import annotations

from typing import Any

from rag.models import PaperMetadata, RelatedWorkQuery

from .base import PaperSearchProvider, ProviderResult, encoded_query, per_query_limit


def _abstract_from_index(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions: list[tuple[int, str]] = []
    for token, offsets in index.items():
        for offset in offsets:
            positions.append((offset, token))
    return " ".join(token for _, token in sorted(positions))


class OpenAlexProvider(PaperSearchProvider):
    name = "OpenAlex"
    inter_query_delay_seconds = 1.0

    def search(self, queries: list[RelatedWorkQuery], limit: int = 10) -> ProviderResult:
        warnings: list[str] = []
        papers: list[PaperMetadata] = []
        seen: dict[str, PaperMetadata] = {}
        query_limit = per_query_limit(limit, len(queries))
        for index, query in enumerate(queries):
            self._sleep_between_queries(index)
            url = (
                "https://api.openalex.org/works"
                f"?search={encoded_query(query.query)}&per-page={query_limit}"
                "&sort=relevance_score:desc"
            )
            try:
                data = self._json_get(url)
            except Exception as exc:
                warnings.append(f"OpenAlex query failed for {query.group}: {exc}")
                continue
            for item in data.get("results", []):
                paper = self._parse_item(item, query.group)
                key = paper.doi or paper.source_ids.get("OpenAlex", "") or paper.title.lower()
                if key in seen:
                    seen[key].matched_query_groups = sorted(set(seen[key].matched_query_groups + [query.group]))
                    continue
                seen[key] = paper
                papers.append(paper)
        return ProviderResult(provider=self.name, papers=papers[:limit], warnings=warnings, status="used" if papers else "failed")

    def _parse_item(self, item: dict[str, Any], query_group: str) -> PaperMetadata:
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in item.get("authorships", [])
            if a.get("author", {}).get("display_name")
        ]
        primary = item.get("primary_location") or {}
        source = primary.get("source") or {}
        url = primary.get("landing_page_url") or item.get("doi") or item.get("id") or ""
        return PaperMetadata(
            paper_id="",
            title=item.get("title") or "",
            authors=authors,
            year=item.get("publication_year"),
            publication_date=item.get("publication_date"),
            venue=source.get("display_name") or "",
            abstract=_abstract_from_index(item.get("abstract_inverted_index")),
            url=url,
            doi=(item.get("doi") or "").lower(),
            source_ids={"OpenAlex": item.get("id") or ""},
            sources=["OpenAlex"],
            matched_query_groups=[query_group],
            citation_count=item.get("cited_by_count"),
        )
