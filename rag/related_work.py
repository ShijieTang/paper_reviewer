from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
from typing import Any

from .config import RAGConfig
from .date_filter import filter_by_cutoff
from .llm import RAGLLMAgent
from .models import PaperMetadata, RelatedWorkQuery, RerankedPaper, TargetPaperSummary
from .providers import ArxivProvider, OpenAlexProvider
from .providers.base import clean_search_query
from .security import prompt_injection_warnings
from .target_parser import llm_context_excerpt, summarize_target_paper


QUERY_GROUPS = [
    "same_problem",
    "same_method",
    "same_constraints",
    "benchmark_baseline",
    "novelty_competitor",
    "limitations_counterevidence",
]

QUERY_GROUP_FALLBACK_LABELS = {
    "same_problem": "the same research problem or task",
    "same_method": "a related method or architecture",
    "same_constraints": "similar assumptions, constraints, or deployment conditions",
    "benchmark_baseline": "a benchmark, dataset, metric, or baseline",
    "novelty_competitor": "a prior contribution relevant to the target paper's novelty",
    "limitations_counterevidence": "limitations, negative results, or counterevidence",
}

SUMMARY_ASPECT_TERMS = {
    "contribution": (
        "contribution", "introduce", "propose", "present", "develop", "novel", "new",
        "first", "release", "create",
    ),
    "dataset": (
        "benchmark", "dataset", "corpus", "data", "suite", "sample", "example", "split",
        "domain", "task", "annotation", "training set", "test set",
    ),
    "performance": (
        "performance", "result", "achieve", "outperform", "improve", "accuracy", "precision",
        "recall", "f1", "bleu", "rouge", "score", "state-of-the-art", "sota", "speedup",
        "faster", "latency", "memory", "throughput", "percent", "%",
    ),
    "method": (
        "method", "model", "architecture", "mechanism", "framework", "algorithm", "approach",
        "attention", "convolution", "transformer", "fusion",
    ),
    "problem": (
        "problem", "task", "application", "study", "investigate", "address", "prediction",
        "classification", "generation",
    ),
    "constraints": (
        "constraint", "assumption", "efficient", "efficiency", "quadratic", "subquadratic",
        "compute", "resource", "scalable", "deployment", "hardware", "runtime",
    ),
    "limitations": (
        "limitation", "failure", "fail", "negative", "however", "degrade", "weakness",
        "boundary", "trade-off", "tradeoff", "counterexample",
    ),
}

QUERY_GROUP_ASPECT_PRIORITY = {
    "same_problem": ("problem", "performance", "contribution"),
    "same_method": ("method", "contribution", "performance"),
    "same_constraints": ("constraints", "performance", "method"),
    "benchmark_baseline": ("dataset", "performance", "contribution"),
    "novelty_competitor": ("contribution", "method", "performance"),
    "limitations_counterevidence": ("limitations", "performance", "constraints"),
}


FALLBACK_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "beyond",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "learning",
    "machine",
    "model",
    "models",
    "of",
    "on",
    "or",
    "our",
    "paper",
    "that",
    "the",
    "this",
    "through",
    "to",
    "using",
    "via",
    "we",
    "with",
}


def _fallback_keywords(target: TargetPaperSummary, max_terms: int = 10) -> list[str]:
    title = clean_search_query(target.title)
    text = clean_search_query(" ".join([title, target.abstract, " ".join(target.claims[:4])]), max_chars=3000)
    title_terms = re.findall(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*", title)
    all_terms = re.findall(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*", text)

    ordered: list[str] = []
    seen: set[str] = set()
    for term in [*title_terms, *all_terms]:
        normalized = term.lower().strip("-/")
        if len(normalized) < 3 or normalized in FALLBACK_QUERY_STOPWORDS or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
        if len(ordered) >= max_terms:
            break
    return ordered


def _fallback_query_text(target: TargetPaperSummary, group: str) -> str:
    keywords = _fallback_keywords(target)
    generic_topics = {"machine learning", "deep learning", "others"}
    topic_terms = [] if not target.topic or target.topic.strip().casefold() in generic_topics else _fallback_keywords(
        TargetPaperSummary(paper_id="", title=target.topic, abstract="", claims=[]),
        max_terms=2,
    )
    suffix_terms = {
        "same_problem": [],
        "same_method": ["architecture"],
        "same_constraints": ["scalable", "efficient"],
        "benchmark_baseline": ["benchmark", "baseline", "evaluation", "dataset"],
        "novelty_competitor": ["prior", "competing", "approach"],
        "limitations_counterevidence": ["limitations", "failure", "analysis"],
    }[group]
    query_terms = [*keywords, *topic_terms, *suffix_terms]
    return clean_search_query(" ".join(query_terms), max_chars=500)


def _generate_queries(
    paper: str,
    target: TargetPaperSummary,
    llm_agent: RAGLLMAgent,
    warnings: list[str],
) -> tuple[list[RelatedWorkQuery], str]:
    system_prompt = (
        "You generate scholarly search queries for related-work retrieval. "
        "Return only valid JSON. Do not invent bibliographic metadata."
    )
    user_prompt = f"""
Read the target paper excerpt and generate exactly one concise search query for each group:
{", ".join(QUERY_GROUPS)}.

The groups mean:
- same_problem: papers addressing the same research problem/task
- same_method: papers using similar methods or architectures
- same_constraints: papers sharing constraints, assumptions, or deployment setting
- benchmark_baseline: papers defining datasets, metrics, baselines, or benchmark comparisons
- novelty_competitor: prior papers that could challenge the novelty claim
- limitations_counterevidence: papers that expose limitations, negative results, or counterevidence

Return JSON:
{{
  "queries": [
    {{"group": "same_problem", "query": "...", "rationale": "..."}}
  ]
}}

Target title: {target.title}
Target topic: {target.topic}
Target abstract: {target.abstract}
Target claims: {json.dumps(target.claims, ensure_ascii=False)}

Target excerpt:
{llm_context_excerpt(paper)}
""".strip()
    try:
        data = llm_agent.complete_json(system_prompt, user_prompt)
        raw_queries = data.get("queries", [])
        source = "llm"
    except Exception as exc:
        warnings.append(f"Related-work query LLM failed; using deterministic fallback queries: {exc}")
        raw_queries = []
        source = "fallback"

    by_group: dict[str, RelatedWorkQuery] = {}
    for item in raw_queries:
        group = str(item.get("group", "")).strip()
        query = clean_search_query(item.get("query", ""), max_chars=500)
        if group in QUERY_GROUPS and query:
            by_group[group] = RelatedWorkQuery(
                group=group,
                query=query,
                rationale=str(item.get("rationale", "")).strip()[:500],
            )
    for group in QUERY_GROUPS:
        if group not in by_group:
            if source == "llm":
                source = "mixed"
            by_group[group] = RelatedWorkQuery(
                group=group,
                query=_fallback_query_text(target, group),
                rationale="Fallback query generated from target title, abstract, and extracted claims.",
            )
    return [by_group[group] for group in QUERY_GROUPS], source


def _canonical_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _canonical_doi(doi: str) -> str:
    value = str(doi or "").strip().lower()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    return value.removeprefix("doi:").strip()


def _canonical_arxiv_id(arxiv_id: str) -> str:
    value = str(arxiv_id or "").strip().lower()
    value = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", value)
    value = value.removeprefix("arxiv:").removesuffix(".pdf")
    return re.sub(r"v\d+$", "", value).strip()


def _paper_identity_keys(paper: PaperMetadata) -> list[str]:
    keys = []
    doi = _canonical_doi(paper.doi)
    arxiv_id = _canonical_arxiv_id(paper.arxiv_id)
    title = _canonical_title(paper.title)
    if doi:
        keys.append(f"doi:{doi}")
    if arxiv_id:
        keys.append(f"arxiv:{arxiv_id}")
    if title:
        keys.append(f"title:{title}")
    return keys


def _merge_paper_metadata(existing: PaperMetadata, paper: PaperMetadata) -> None:
    existing.sources = sorted(set(existing.sources + paper.sources))
    existing.matched_query_groups = sorted(
        set(existing.matched_query_groups + paper.matched_query_groups)
    )
    existing.source_ids.update({key: value for key, value in paper.source_ids.items() if value})
    if len(paper.abstract or "") > len(existing.abstract or ""):
        existing.abstract = paper.abstract
    if len(paper.authors) > len(existing.authors):
        existing.authors = paper.authors
    for field_name in ("year", "publication_date", "venue", "url", "doi", "arxiv_id"):
        if not getattr(existing, field_name) and getattr(paper, field_name):
            setattr(existing, field_name, getattr(paper, field_name))
    counts = [count for count in (existing.citation_count, paper.citation_count) if count is not None]
    if counts:
        existing.citation_count = max(counts)


def _dedupe_papers(papers: list[PaperMetadata]) -> list[PaperMetadata]:
    merged: list[PaperMetadata] = []
    identity_index: dict[str, PaperMetadata] = {}
    for paper in papers:
        if not paper.title:
            continue
        identity_keys = _paper_identity_keys(paper)
        existing = next(
            (identity_index[key] for key in identity_keys if key in identity_index),
            None,
        )
        if existing is None:
            existing = paper
            merged.append(existing)
        else:
            _merge_paper_metadata(existing, paper)
        for key in set(identity_keys + _paper_identity_keys(existing)):
            identity_index[key] = existing

    deduped = merged
    for idx, paper in enumerate(deduped, 1):
        seed = paper.doi or paper.arxiv_id or paper.source_ids.get("OpenAlex") or paper.source_ids.get("Semantic Scholar") or paper.title
        suffix = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:9]
        paper.paper_id = f"rw_{idx:03d}_{suffix}"
    return deduped


def _author_last_name(author: str) -> str:
    author = re.sub(r"\s+", " ", str(author or "")).strip()
    if not author:
        return ""
    candidate = author.split(",", 1)[0] if "," in author else author.split()[-1]
    return re.sub(r"[^A-Za-z0-9\-']", "", candidate).strip()


def _citation_label(paper: PaperMetadata) -> str:
    year = str(paper.year or (paper.publication_date or "")[:4] or "n.d.")
    names = [_author_last_name(author) for author in paper.authors if _author_last_name(author)]
    if not names:
        words = re.findall(r"[A-Za-z0-9]+", paper.title)
        title_label = " ".join(words[:4]) if words else "Untitled work"
        return f"{title_label}, {year}"
    if len(names) == 1:
        return f"{names[0]}, {year}"
    if len(names) == 2:
        return f"{names[0]} and {names[1]}, {year}"
    return f"{names[0]} et al., {year}"


def _reference_text(paper: PaperMetadata) -> str:
    return f"{paper.title or 'Untitled work'} ({_citation_label(paper)})"


def _replace_internal_ids_with_references(summary: str, papers: list[PaperMetadata]) -> str:
    by_id = {paper.paper_id: paper for paper in papers}

    def replace_bracket(match: re.Match) -> str:
        ids = re.findall(r"rw_[A-Za-z0-9_]+", match.group(0))
        refs = [_reference_text(by_id[paper_id]) for paper_id in ids if paper_id in by_id]
        if not refs:
            return match.group(0)
        return "(" + "; ".join(refs) + ")"

    summary = re.sub(r"\[(?:\s*rw_[A-Za-z0-9_]+(?:\s*,\s*)?)+\]", replace_bracket, summary)
    for paper_id, paper in by_id.items():
        summary = summary.replace(paper_id, _reference_text(paper))
    return summary


def _looks_like_reviewer_guidance(summary: str) -> bool:
    lowered = summary.lower()
    return any(
        phrase in lowered
        for phrase in (
            "reviewers should",
            "reviewer should",
            "the reviewer should",
            "reviewers can",
            "reviewer can",
            "reviewers must",
            "reviewer must",
            "for reviewers to",
            "should refer to",
            "should compare",
            "must compare",
        )
    )


def _neutralize_summary_voice(summary: str, protected_phrases: tuple[str, ...] = ()) -> str:
    """Convert accidental source-author voice without rejecting useful content."""
    text = str(summary or "")
    protected: dict[str, str] = {}
    for index, phrase in enumerate(sorted(set(protected_phrases), key=len, reverse=True)):
        if not phrase or phrase not in text:
            continue
        placeholder = f"__RAG_PROTECTED_TITLE_{index}__"
        text = text.replace(phrase, placeholder)
        protected[placeholder] = phrase

    first_person = re.compile(
        r"(?P<our>\bour\s+(?:works?|paper|study|method|approach|model|results?|findings?|experiments?)\b)"
        r"|(?P<we>\bwe\s+(?:propose|introduce|present|develop|describe|demonstrate|show|find|report|"
        r"release|create|construct|evaluate|achieve|outperform|use)\b)"
        r"|(?P<our_generic>\bour\b)"
        r"|(?P<we_generic>\bwe\b)",
        flags=re.IGNORECASE,
    )

    def replace(match: re.Match) -> str:
        phrase = match.group(0)
        if match.group("our"):
            noun = re.sub(r"^our\s+", "", phrase, flags=re.IGNORECASE)
            if noun.casefold() in {"work", "works", "paper", "study"}:
                return "the paper" if phrase[:1].islower() else "The paper"
            replacement = f"the {noun.lower()}"
            return replacement if phrase[:1].islower() else replacement.capitalize()
        if match.group("our_generic"):
            return "the authors'" if phrase[:1].islower() else "The authors'"
        if match.group("we_generic"):
            return "the authors" if phrase[:1].islower() else "The authors"
        verb = re.sub(r"^we\s+", "", phrase, flags=re.IGNORECASE).lower()
        replacement = f"the authors {verb}"
        return replacement if phrase[:1].islower() else replacement.capitalize()

    text = first_person.sub(replace, text)
    for placeholder, phrase in protected.items():
        text = text.replace(placeholder, phrase)
    return text


def _author_phrase(paper: PaperMetadata) -> str:
    authors = [
        re.sub(r"\s+", " ", str(author)).strip()
        for author in paper.authors
        if str(author).strip()
    ]
    if not authors:
        return "the authors"
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    return f"{authors[0]} et al."


def _publication_sentence(paper: PaperMetadata) -> str:
    title = paper.title or "an untitled paper"
    authors = _author_phrase(paper)
    year = paper.year or (paper.publication_date or "")[:4]
    venue = re.sub(r"\s+", " ", str(paper.venue or "")).strip()
    venue_clause = f" in {venue}" if venue else ""
    if year and authors == "the authors":
        return f'In {year}, "{title}" was published{venue_clause}.'
    if year:
        return f'In {year}, {authors} published "{title}"{venue_clause}.'
    author_lead = authors[:1].upper() + authors[1:]
    return f'{author_lead} published "{title}"{venue_clause}.'


def _abstract_sentences(abstract: str) -> list[str]:
    text = re.sub(r"<[^>]+>", " ", str(abstract or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        if sentence.strip()
    ] or [text]


def _sentence_aspect_score(sentence: str, aspect: str) -> int:
    lowered = sentence.casefold()
    score = sum(lowered.count(term) for term in SUMMARY_ASPECT_TERMS.get(aspect, ()))
    if aspect == "performance":
        score += len(re.findall(r"\b\d+(?:\.\d+)?\s*%", sentence))
        score += len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:x|times|points?)\b", lowered))
    if aspect == "dataset":
        score += len(re.findall(r"\b\d[\d,]*(?:\.\d+)?\s+(?:examples?|samples?|images?|documents?|tasks?|domains?)\b", lowered))
    return score


def _summary_aspect_priority(paper: PaperMetadata) -> list[str]:
    priority: list[str] = []
    for group in paper.matched_query_groups:
        for aspect in QUERY_GROUP_ASPECT_PRIORITY.get(group, ()):
            if aspect not in priority:
                priority.append(aspect)
    for aspect in ("contribution", "dataset", "performance", "method", "problem", "constraints", "limitations"):
        if aspect not in priority:
            priority.append(aspect)
    return priority


def _clean_evidence_sentence(sentence: str, paper_title: str) -> str:
    text = _neutralize_summary_voice(sentence, (paper_title,) if paper_title else ())
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 500:
        text = text[:497].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."
    elif text and text[-1] not in ".!?":
        text += "."
    return text


def _focused_abstract_evidence(paper: PaperMetadata, max_sentences: int = 3) -> str:
    sentences = _abstract_sentences(paper.abstract)
    if not sentences:
        groups = [
            group
            for group in paper.matched_query_groups
            if group in QUERY_GROUP_FALLBACK_LABELS
        ]
        if not groups:
            return "The available metadata does not include an abstract with contribution, dataset, or result details."
        labels = "; ".join(QUERY_GROUP_FALLBACK_LABELS[group] for group in groups[:3])
        return (
            f"The paper was retrieved as {labels}, but the available metadata does not include "
            "an abstract with concrete contribution, dataset, or result details."
        )

    priority = _summary_aspect_priority(paper)
    selected: list[int] = []
    covered_aspects: set[str] = set()
    for aspect in priority:
        if aspect in covered_aspects:
            continue
        candidates = [
            (_sentence_aspect_score(sentence, aspect), index)
            for index, sentence in enumerate(sentences)
            if index not in selected
        ]
        if not candidates:
            continue
        score, index = max(candidates, key=lambda item: (item[0], -item[1]))
        if score <= 0:
            continue
        selected.append(index)
        covered_aspects.update(
            candidate_aspect
            for candidate_aspect in SUMMARY_ASPECT_TERMS
            if _sentence_aspect_score(sentences[index], candidate_aspect) > 0
        )
        if len(selected) >= max_sentences:
            break

    if len(selected) < max_sentences:
        remaining = []
        for index, sentence in enumerate(sentences):
            if index in selected:
                continue
            detail_score = sum(
                _sentence_aspect_score(sentence, aspect)
                for aspect in SUMMARY_ASPECT_TERMS
            )
            remaining.append((detail_score, index))
        for score, index in sorted(remaining, key=lambda item: (-item[0], item[1])):
            if score <= 0 and selected:
                break
            selected.append(index)
            if len(selected) >= max_sentences:
                break

    if not selected:
        selected = [0]
    return " ".join(_clean_evidence_sentence(sentences[index], paper.title) for index in selected)


def _compose_background_summary(papers: list[PaperMetadata]) -> str:
    papers = [paper for paper in papers if paper.title][:6]
    if not papers:
        return "No cutoff-valid related-work metadata was available to summarize."
    return " ".join(
        f"{_publication_sentence(paper)} {_focused_abstract_evidence(paper)}"
        for paper in papers
    )


def _looks_like_publication_results(summary: str, papers: list[PaperMetadata]) -> bool:
    summary_folded = summary.casefold()
    checked_papers = [paper for paper in papers if paper.title][:6]
    if not checked_papers:
        return False
    for paper in checked_papers:
        if paper.title.casefold() not in summary_folded:
            return False
        year = str(paper.year or (paper.publication_date or "")[:4] or "")
        if year and year not in summary:
            return False
    return len(re.findall(r"\bpublished\b", summary_folded)) >= len(checked_papers)


def _summary_from_ranked(
    summary: str,
    papers: list[PaperMetadata],
    ranked: list[RerankedPaper],
) -> str:
    by_id = {paper.paper_id: paper for paper in papers}
    ranked_papers = [by_id[item.paper_id] for item in ranked if item.paper_id in by_id]
    source_papers = ranked_papers or papers
    normalized = _replace_internal_ids_with_references(str(summary or "").strip(), source_papers)
    normalized = _neutralize_summary_voice(
        normalized,
        tuple(paper.title for paper in source_papers if paper.title),
    )
    if (
        not normalized
        or re.search(r"\brw_[A-Za-z0-9_]+\b", normalized)
        or _looks_like_reviewer_guidance(normalized)
        or not _looks_like_publication_results(normalized, source_papers)
    ):
        return _compose_background_summary(source_papers)
    return normalized


def _candidate_payload(papers: list[PaperMetadata]) -> list[dict[str, Any]]:
    payload = []
    for paper in papers:
        payload.append({
            "paper_id": paper.paper_id,
            "title": paper.title,
            "reference": _reference_text(paper),
            "citation_label": _citation_label(paper),
            "authors": paper.authors[:8],
            "year": paper.year,
            "publication_date": paper.publication_date,
            "venue": paper.venue,
            "sources": paper.sources,
            "url": paper.url,
            "doi": paper.doi,
            "arxiv_id": paper.arxiv_id,
            "matched_query_groups": paper.matched_query_groups,
            "citation_count": paper.citation_count,
            "abstract": paper.abstract[:1600],
        })
    return payload


def _fallback_rerank(papers: list[PaperMetadata], target: TargetPaperSummary, top_k: int) -> tuple[list[RerankedPaper], str, str]:
    scored = []
    for paper in papers:
        scored.append((_lexical_relevance_score(paper, target), paper))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected_papers = [paper for _, paper in scored[:top_k]]
    reranked = [
        RerankedPaper(
            rank=i + 1,
            paper_id=paper.paper_id,
            relevance_score=round(float(score), 3),
            relevance_types=paper.matched_query_groups[:3],
            rationale="Fallback lexical overlap ranking because LLM reranking was unavailable.",
            evidence_summary=_focused_abstract_evidence(paper, max_sentences=2)[:1000],
        )
        for i, (score, paper) in enumerate(scored[:top_k])
    ]
    summary = _compose_background_summary(selected_papers)
    return reranked, summary, "fallback"


def _lexical_relevance_score(paper: PaperMetadata, target: TargetPaperSummary) -> float:
    target_terms = set(re.findall(r"[a-z0-9]{3,}", f"{target.title} {target.abstract}".lower()))
    terms = set(re.findall(r"[a-z0-9]{3,}", f"{paper.title} {paper.abstract}".lower()))
    overlap = len(target_terms & terms)
    denom = max(1, len(target_terms | terms))
    return overlap / denom


def _calibrate_saturated_scores(ranked: list[RerankedPaper]) -> None:
    if len(ranked) <= 1:
        return
    if not all(item.relevance_score >= 0.999 for item in ranked):
        return
    for index, item in enumerate(ranked):
        item.relevance_score = round(max(0.5, 1.0 - (0.05 * index)), 3)


def _fill_missing_reranked(
    ranked: list[RerankedPaper],
    papers: list[PaperMetadata],
    target: TargetPaperSummary,
    top_k: int,
    seen: set[str],
) -> bool:
    target_count = min(top_k, len(papers))
    if len(ranked) >= target_count:
        return False
    filled = False
    for paper in papers:
        if paper.paper_id in seen:
            continue
        score = round(float(_lexical_relevance_score(paper, target)), 3)
        ranked.append(RerankedPaper(
            rank=len(ranked) + 1,
            paper_id=paper.paper_id,
            relevance_score=score,
            relevance_types=paper.matched_query_groups[:3],
            rationale="LLM omitted this candidate; appended in cleaned candidate order to preserve the rerank input set.",
            evidence_summary=_focused_abstract_evidence(paper, max_sentences=2)[:1000],
        ))
        seen.add(paper.paper_id)
        filled = True
        if len(ranked) >= target_count:
            break
    return filled


def _rerank_with_llm(
    target: TargetPaperSummary,
    queries: list[RelatedWorkQuery],
    papers: list[PaperMetadata],
    llm_agent: RAGLLMAgent,
    top_k: int,
    warnings: list[str],
) -> tuple[list[RerankedPaper], str, str]:
    if not papers:
        return [], "No cutoff-valid related-work metadata was available for reranking.", "none"
    summary_limit = min(6, top_k, len(papers))
    system_prompt = (
        "You are a neutral background-information provider for downstream scholarly-review agents "
        "and a related-work reranking agent. "
        "Report what prior papers published and found using third-person attribution. "
        "Use only the provided candidate metadata. "
        "Do not invent titles, authors, venues, years, URLs, DOIs, or paper IDs. "
        "Return only valid JSON."
    )
    user_prompt = f"""
Target paper:
{json.dumps(target.to_dict(), ensure_ascii=False)}

Search queries:
{json.dumps([q.to_dict() for q in queries], ensure_ascii=False)}

Candidate metadata:
{json.dumps(_candidate_payload(papers), ensure_ascii=False)}

Rerank candidates for usefulness as related work. Prefer papers that are directly relevant to novelty, baselines, benchmarks, methods, constraints, and limitations.
Return exactly {min(top_k, len(papers))} unique papers unless fewer valid candidates are provided. Do not return only one paper per query group.

Score on a calibrated 0.0-1.0 scale:
- 1.0: same paper, earlier version, or direct duplicate of the target.
- 0.85-0.95: central method, benchmark, or baseline needed to understand the target.
- 0.65-0.84: close related method, constraint, or limitation evidence.
- 0.40-0.64: useful background but not a direct comparison.
- below 0.40: weakly related.
Avoid giving the same score to every paper.

The summary will be passed to downstream agents as background information. It is evidence, not reviewer guidance.
Write it as neutral, factual reporting of the first {summary_limit} ranked prior papers, in ranked order. Use two or three concise sentences per paper when the metadata contains enough detail.
Begin each paper's result with its supplied year, authors, and title, following this style:
"In 2022, Lee-Thorp et al. published 'FNet: Mixing Tokens with Fourier Transforms,' which introduced ..."
If a venue is supplied, it may be stated. Do not invent a venue or publication status.

Use third-person attribution throughout. Avoid source-author or target-author language such as "our work", "our paper", "our method", "our results", "we propose", or "we show".
Do not praise, recommend, or instruct. Do not say "reviewers should", "must compare", "important", "promising", or similar evaluative guidance.
Do not write a generic introduction about the target paper. State concrete details about each retrieved paper.
Do not cite internal paper IDs or expose internal query-group names in the prose.

For each paper, report every concrete aspect supported by its metadata:
- its main contribution or what was new;
- its method or architecture;
- datasets or benchmarks used, including scale, splits, domains, tasks, metrics, and baselines when available;
- quantitative or qualitative performance results when available;
- constraints, efficiency findings, limitations, or negative results when available.
Never invent a missing detail. If the metadata does not provide an aspect, omit it.

Use each candidate's matched_query_groups and the matching search query to decide which supported details to emphasize first:
- same_problem: the research problem, task, application setting, and main result.
- same_method: the method, architecture, mechanism, and how it operates.
- same_constraints: assumptions, efficiency requirements, resource constraints, or deployment setting.
- benchmark_baseline: the benchmark or dataset, its contents and scale, metrics, and baselines.
- novelty_competitor: the main contribution, claimed innovation, method, and result.
- limitations_counterevidence: the limitation, failure mode, negative result, boundary condition, or counterevidence.
When a paper matches multiple groups, combine all supported aspects into one paper entry instead of repeating the paper.

Return JSON:
{{
  "reranked_papers": [
    {{
      "rank": 1,
      "paper_id": "rw_...",
      "relevance_score": 0.0,
      "relevance_types": ["same_problem"],
      "rationale": "which supplied metadata makes this paper relevant to the target",
      "evidence_summary": "two or three neutral factual sentences covering all supported aspects"
    }}
  ],
  "summary": "Neutral publication-style results for each unique ranked paper, including concrete contribution, dataset, and performance details when supplied."
}}
""".strip()
    paper_by_id = {paper.paper_id: paper for paper in papers}
    valid_ids = set(paper_by_id)
    try:
        data = llm_agent.complete_json(system_prompt, user_prompt)
        raw_ranked = data.get("reranked_papers", [])
        summary = str(data.get("summary", "")).strip()
    except Exception as exc:
        warnings.append(f"Related-work rerank LLM failed; using lexical fallback reranking: {exc}")
        return _fallback_rerank(papers, target, top_k)

    ranked: list[RerankedPaper] = []
    seen: set[str] = set()
    for item in raw_ranked:
        paper_id = str(item.get("paper_id", "")).strip()
        if paper_id not in valid_ids or paper_id in seen:
            continue
        seen.add(paper_id)
        try:
            score = float(item.get("relevance_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        candidate = paper_by_id[paper_id]
        protected_title = (candidate.title,) if candidate.title else ()
        rationale = _neutralize_summary_voice(
            str(item.get("rationale", "")).strip(),
            protected_title,
        )[:1000]
        if _looks_like_reviewer_guidance(rationale):
            rationale = "The supplied metadata matched the related-work retrieval criteria."
        evidence_summary = _neutralize_summary_voice(
            str(item.get("evidence_summary", "")).strip(),
            protected_title,
        )[:1000]
        if not evidence_summary or _looks_like_reviewer_guidance(evidence_summary):
            evidence_summary = _focused_abstract_evidence(candidate, max_sentences=2)[:1000]
        ranked.append(RerankedPaper(
            rank=len(ranked) + 1,
            paper_id=paper_id,
            relevance_score=max(0.0, min(1.0, score)),
            relevance_types=[str(x) for x in item.get("relevance_types", []) if str(x)],
            rationale=rationale,
            evidence_summary=evidence_summary,
        ))
        if len(ranked) >= top_k:
            break
    if not ranked:
        warnings.append("Related-work rerank LLM returned no valid paper IDs; using lexical fallback reranking.")
        return _fallback_rerank(papers, target, top_k)
    _calibrate_saturated_scores(ranked)
    filled_from_fallback = _fill_missing_reranked(ranked, papers, target, top_k, seen)
    if not summary:
        summary = _compose_background_summary(
            [paper_by_id[item.paper_id] for item in ranked if item.paper_id in valid_ids]
        )
    summary = _summary_from_ranked(summary, papers, ranked)
    return ranked, summary, "mixed" if filled_from_fallback else "llm"


def build_related_work_rag(
    paper: str,
    topic: str = "",
    provider: str = "cmu",
    model: str = "",
    api_key: str = "",
    config: RAGConfig | None = None,
    providers: list[Any] | None = None,
    llm_agent: RAGLLMAgent | None = None,
) -> dict[str, Any]:
    config = config or RAGConfig()
    target = summarize_target_paper(paper, topic=topic)
    warnings = prompt_injection_warnings(paper, "target paper")
    llm_agent = llm_agent or RAGLLMAgent(provider=provider, api_key=api_key, model=model)

    queries, query_source = _generate_queries(paper, target, llm_agent, warnings)

    search_providers = providers or [
        OpenAlexProvider(config.rag_cache_dir),
        ArxivProvider(config.rag_cache_dir),
    ]
    provider_status: dict[str, dict[str, Any]] = {}
    retrieved: list[PaperMetadata] = []
    max_workers = max(1, len(search_providers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(search_provider.search, queries, limit=config.provider_top_k)
            for search_provider in search_providers
        ]
    for search_provider, future in zip(search_providers, futures):
        try:
            result = future.result()
        except Exception as exc:
            provider_name = getattr(search_provider, "name", search_provider.__class__.__name__)
            warnings.append(f"{provider_name}: provider failed: {exc}")
            provider_status[provider_name] = {"status": "failed", "retrieved": 0, "warnings": [str(exc)]}
            continue
        retrieved.extend(result.papers)
        provider_status[result.provider] = {
            "status": result.status,
            "retrieved": len(result.papers),
            "warnings": result.warnings,
        }
        warnings.extend(result.warnings)

    deduped = _dedupe_papers(retrieved)
    target_title = _canonical_title(target.title)
    related_only = [
        candidate
        for candidate in deduped
        if not target_title or _canonical_title(candidate.title) != target_title
    ]
    num_removed_as_target = len(deduped) - len(related_only)
    cutoff_valid, cutoff_report = filter_by_cutoff(
        related_only,
        cutoff_date=config.cutoff_date,
        allow_undated=config.allow_undated_evidence,
    )
    cutoff_report["num_removed_as_target"] = num_removed_as_target
    candidate_cap = max(0, int(config.rerank_top_k))
    filtered = cutoff_valid[:candidate_cap] if candidate_cap else []
    cutoff_report["num_cutoff_valid"] = len(cutoff_valid)
    cutoff_report["candidate_cap"] = candidate_cap
    cutoff_report["num_removed_by_candidate_cap"] = max(0, len(cutoff_valid) - len(filtered))
    cutoff_report["num_used"] = len(filtered)
    evidence_warnings = []
    for paper_meta in filtered:
        evidence_warnings.extend(prompt_injection_warnings(paper_meta.abstract, paper_meta.paper_id))
    warnings.extend(evidence_warnings)

    reranked, summary, rerank_source = _rerank_with_llm(
        target=target,
        queries=queries,
        papers=filtered,
        llm_agent=llm_agent,
        top_k=len(filtered),
        warnings=warnings,
    )

    package_id_seed = target.paper_id + json.dumps([q.to_dict() for q in queries], sort_keys=True)
    package_id = "rag_rw_" + hashlib.sha1(package_id_seed.encode("utf-8")).hexdigest()[:12]
    return {
        "rag_package_id": package_id,
        "paper_id": target.paper_id,
        "target_paper_summary": target.to_dict(),
        "query_generation": {
            "groups": QUERY_GROUPS,
            "queries": [q.to_dict() for q in queries],
            "source": query_source,
        },
        "provider_status": provider_status,
        "paper_metadata": [paper.to_dict() for paper in filtered],
        "reranking_results": [item.to_dict() for item in reranked],
        "reranking": {"source": rerank_source},
        "related_work_summary": summary,
        "warnings": warnings,
        "cutoff_report": cutoff_report,
    }
