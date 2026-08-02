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


def rag_config_from_dict(values: dict | None = None) -> RAGConfig:
    values = values or {}
    allowed = {field.name for field in RAGConfig.__dataclass_fields__.values()}
    return RAGConfig(**{k: v for k, v in values.items() if k in allowed})
