"""Reference expansion: recent papers that cite the submission's own references.

The submission's reference list says which prior work the authors consider relevant;
papers that later cite those references are likely to be close, recent related work —
including competitors the authors may not know about.

Source: the Semantic Scholar Graph API when SEMANTIC_SCHOLAR_API_KEY is set (without a key
it is rate-limited almost immediately); otherwise OpenAlex, which needs no key
(set OPENALEX_MAILTO to your email for its faster "polite pool"). If Semantic Scholar
starts refusing requests, expansion switches to OpenAlex for the remaining references.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from .models import PaperMetadata
from .providers.base import PaperSearchProvider, encoded_query, is_forbidden_error, is_rate_limited_error
from .refine import ManuscriptReferences

GROUP = "reference_expansion"
_FIELDS = "title,authors,year,publicationDate,venue,abstract,url,citationCount,externalIds"
_STOP = {"a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "into", "is", "of", "on", "or",
         "the", "to", "via", "with", "we", "our", "this", "that", "using", "towards", "learning", "model", "models"}


@dataclass
class ExpansionResult:
    papers: list[PaperMetadata] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    status: str = "skipped"


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z0-9\-]{2,}", (text or "").lower()) if w not in _STOP}


class ReferenceExpander(PaperSearchProvider):
    name = "Reference expansion"
    inter_query_delay_seconds = 1.0

    def _headers(self) -> dict[str, str]:
        key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        return {"x-api-key": key} if key else {}

    def match_title(self, title: str) -> str:
        url = ("https://api.semanticscholar.org/graph/v1/paper/search/match"
               f"?query={encoded_query(title)}&fields=paperId,title")
        data = self._json_get(url, headers=self._headers())
        items = data.get("data") or []
        return (items[0].get("paperId") or "") if items else ""

    def citing_papers(self, paper_id: str, limit: int) -> list[dict[str, Any]]:
        url = (f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
               f"?fields={_FIELDS}&limit={max(1, int(limit))}")
        data = self._json_get(url, headers=self._headers())
        return [item.get("citingPaper") or {} for item in data.get("data") or []]


class OpenAlexReferenceExpander(PaperSearchProvider):
    """Same interface as ReferenceExpander, backed by OpenAlex (cites: filter). Returns S2-shaped dicts."""
    name = "Reference expansion (OpenAlex)"
    inter_query_delay_seconds = 0.5
    source_label = "OpenAlex (cites a manuscript reference)"

    def match_title(self, title: str) -> str:
        from .providers.openalex import mailto_param
        from .refine import title_similarity
        url = (f"https://api.openalex.org/works?search={encoded_query(title)}&per-page=3"
               f"&select=id,title{mailto_param()}")
        for item in self._json_get(url).get("results") or []:
            if title_similarity(item.get("title") or "", title) >= 0.85:
                return (item.get("id") or "").rsplit("/", 1)[-1]
        return ""

    def citing_papers(self, paper_id: str, limit: int) -> list[dict[str, Any]]:
        from .providers.openalex import _abstract_from_index, mailto_param
        url = (f"https://api.openalex.org/works?filter=cites:{paper_id}&sort=cited_by_count:desc"
               f"&per-page={max(1, min(200, int(limit)))}{mailto_param()}")
        out = []
        for item in self._json_get(url).get("results") or []:
            ids = item.get("ids") or {}
            arxiv = ""
            for loc in item.get("locations") or []:
                landing = (loc or {}).get("landing_page_url") or ""
                if "arxiv.org/abs/" in landing:
                    arxiv = landing.rsplit("/abs/", 1)[-1]
                    break
            out.append({
                "paperId": (item.get("id") or "").rsplit("/", 1)[-1],
                "title": item.get("title") or "",
                "authors": [{"name": a.get("author", {}).get("display_name", "")}
                            for a in item.get("authorships") or [] if a.get("author", {}).get("display_name")],
                "year": item.get("publication_year"),
                "publicationDate": item.get("publication_date"),
                "venue": ((item.get("primary_location") or {}).get("source") or {}).get("display_name") or "",
                "abstract": _abstract_from_index(item.get("abstract_inverted_index")),
                "url": (item.get("primary_location") or {}).get("landing_page_url") or item.get("doi") or "",
                "citationCount": item.get("cited_by_count"),
                "externalIds": {"DOI": (ids.get("doi") or "").replace("https://doi.org/", ""), "ArXiv": arxiv},
            })
        return out


def default_expander(cache_dir: str) -> PaperSearchProvider:
    if os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
        return ReferenceExpander(cache_dir)
    return OpenAlexReferenceExpander(cache_dir)


def _to_metadata(item: dict[str, Any], via_reference: str, source_label: str = "") -> PaperMetadata:
    external = item.get("externalIds") or {}
    return PaperMetadata(
        paper_id="",
        title=item.get("title") or "",
        authors=[a.get("name", "") for a in item.get("authors") or [] if a.get("name")],
        year=item.get("year"),
        publication_date=item.get("publicationDate"),
        venue=item.get("venue") or "",
        abstract=item.get("abstract") or "",
        url=item.get("url") or "",
        doi=(external.get("DOI") or "").lower(),
        arxiv_id=external.get("ArXiv") or "",
        source_ids={"Semantic Scholar": item.get("paperId") or "", "cites_reference": via_reference[:200]},
        sources=[source_label or "Semantic Scholar (cites a manuscript reference)"],
        matched_query_groups=[GROUP],
        citation_count=item.get("citationCount"),
    )


def expand_from_references(paper_text: str, cache_dir: str = "data/rag_cache", max_refs: int = 10,
                           citing_limit: int = 20, provider: ReferenceExpander | None = None) -> ExpansionResult:
    """Return up to `citing_limit` papers that cite the manuscript's most topical references."""
    refs = ManuscriptReferences(paper_text)
    result = ExpansionResult()
    if not refs.titles:
        result.warnings.append("Reference expansion skipped: no reference titles could be parsed from the manuscript.")
        return result
    topic_words = _words(paper_text[:6000])
    titles = [t for t in dict.fromkeys(refs.titles) if len(t.split()) >= 4 and not t.lower().startswith("arxiv")]
    titles.sort(key=lambda t: -len(_words(t) & topic_words) / max(1, len(_words(t))))
    titles = titles[:max(0, int(max_refs))]
    expander = provider or default_expander(cache_dir)
    can_fall_back = provider is None and isinstance(expander, ReferenceExpander)

    found: dict[str, PaperMetadata] = {}
    stop = False
    for index, title in enumerate(titles):
        if stop:
            break
        for _attempt in range(2):          # second attempt only after switching to OpenAlex
            expander._sleep_between_queries(index)
            try:
                paper_id = expander.match_title(title)
                if paper_id:
                    for item in expander.citing_papers(paper_id, citing_limit):
                        meta = _to_metadata(item, title, getattr(expander, "source_label", ""))
                        key = meta.doi or meta.arxiv_id or meta.title.lower()
                        if meta.title and key not in found:
                            found[key] = meta
            except Exception as exc:
                refused = is_rate_limited_error(exc) or is_forbidden_error(exc)
                if refused and can_fall_back:
                    result.warnings.append(f"Semantic Scholar refused requests ({exc}); switching to OpenAlex.")
                    expander, can_fall_back = OpenAlexReferenceExpander(cache_dir), False
                    continue           # retry this reference with OpenAlex
                if refused:
                    result.warnings.append(f"Reference expansion stopped early ({getattr(expander, 'name', '')}: "
                                           f"{exc}); retry later.")
                    stop = True
                else:
                    result.warnings.append(f"Reference expansion failed for one reference: {exc}")
            break
    # Keep the citing papers most topical to the submission.
    ranked = sorted(found.values(), key=lambda m: -len(_words(m.title + " " + m.abstract) & topic_words))
    result.papers = ranked[:max(0, int(citing_limit))]
    result.status = "used" if result.papers else ("failed" if result.warnings else "empty")
    return result


def interleave(search_results: list[PaperMetadata], expansion: list[PaperMetadata]) -> list[PaperMetadata]:
    """Alternate search and expansion candidates so both reach the reranker before the candidate cap."""
    out: list[PaperMetadata] = []
    for i in range(max(len(search_results), len(expansion))):
        if i < len(search_results):
            out.append(search_results[i])
        if i < len(expansion):
            out.append(expansion[i])
    return out
