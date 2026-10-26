from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RAGConfig:
    enable_rag: bool = False
    enable_related_work_rag: bool = True
    cutoff_date: str = "2024-12-31"
    allow_undated_evidence: bool = False
    rag_cache_dir: str = "data/rag_cache"
    provider_top_k: int = 10
    rerank_top_k: int = 12
    rerank_batch_size: int = 10          # candidates per rerank LLM call (larger sets are split; see _rerank_in_batches)

    # ── RAG improvements (all off by default, so existing runs are unchanged) ──
    # Retrieval (option 3)
    cutoff_mode: str = "venue"            # "venue": use cutoff_date; "today": exclude nothing published before today
    strict_self_exclusion: bool = False   # also drop near-duplicate titles / same arXiv ID or DOI as the submission
    self_match_threshold: float = 0.9     # title similarity at or above which a candidate counts as the submission
    claim_queries: int = 0                # extra search queries, one per claimed contribution (0 = off)
    reference_expansion: bool = False     # add recent papers that cite the submission's own references (Semantic Scholar)
    reference_expansion_max_refs: int = 10
    reference_expansion_citing_limit: int = 20
    tag_cited: bool = False               # mark each retrieved work as cited / not cited in the manuscript
    min_relevance: float = 0.0            # drop reranked works below this relevance score
    max_shown: int = 8                    # number of works shown to reviewers
    min_uncited_shown: int = 0            # reserve this many of the shown slots for works the manuscript does not cite
    # Reviewer use of the evidence (option 1)
    prompt_style: str = "neutral"         # "neutral" (original block) or "engaged" ([R#] labels, abstracts, novelty check)
    # Related-work analyst step (option 2)
    related_work_analyst: bool = False
    # Missing-references report for authors (option 4)
    missing_references_report: bool = False
    missing_reference_min_relevance: float = 0.65

    @property
    def improvements_enabled(self) -> bool:
        return (self.strict_self_exclusion or self.tag_cited or self.min_relevance > 0
                or self.min_uncited_shown > 0 or self.prompt_style == "engaged"
                or self.related_work_analyst or self.missing_references_report)


def rag_config_from_dict(values: dict | None = None) -> RAGConfig:
    values = values or {}
    allowed = {field.name for field in RAGConfig.__dataclass_fields__.values()}
    return RAGConfig(**{k: v for k, v in values.items() if k in allowed})
