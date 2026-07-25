import json
import re
import unittest
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

from rag.config import RAGConfig
from rag.llm import RAGLLMAgent
from rag.models import PaperMetadata, ReviewMemoryCase
from rag.orchestrator import build_rag_package
from rag.prompt_views import format_rag_prompt_block
from rag.providers.arxiv import ArxivProvider
from rag.providers.openalex import OpenAlexProvider
from rag.providers.base import ProviderHTTPError, ProviderResult, clean_search_query
from rag.providers.semantic_scholar import SemanticScholarProvider
from rag.related_work import QUERY_GROUPS, build_related_work_rag
from rag.review_memory import (
    OPENREVIEW_NOTES_URL,
    OPENREVIEW_SEARCH_URL,
    OpenReviewMemoryProvider,
    build_review_memory_from_package,
)


class FakeLLM:
    def __init__(self):
        self.calls = []

    def complete_json(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if "generate exactly one concise search query" in user_prompt:
            return {
                "queries": [
                    {"group": group, "query": f"{group} target retrieval", "rationale": f"why {group}"}
                    for group in QUERY_GROUPS
                ]
            }
        if "Candidate metadata:" in user_prompt:
            ids = re.findall(r'"paper_id":\s*"(rw_[^"]+)"', user_prompt)
            if not ids:
                ids = ["rw_001_fake"]
            return {
                "reranked_papers": [
                    {
                        "rank": 1,
                        "paper_id": ids[0],
                        "relevance_score": 0.91,
                        "relevance_types": ["same_problem", "benchmark_baseline"],
                        "rationale": "Directly studies the same task and baseline.",
                        "evidence_summary": "Relevant metadata summary.",
                    }
                ],
                "summary": f"Most relevant prior work is [{ids[0]}].",
            }
        if "OpenReview review-memory case:" in user_prompt:
            ids = re.findall(r'"source_paper_id":\s*"([^"]+)"', user_prompt)
            source_id = ids[0] if ids else "rw_unknown"
            return {
                "summary": f"Reviewers accepted [{source_id}] while noting strong empirical scope and limited baselines.",
                "common_strengths": ["Strong empirical evaluation", "Clear task motivation"],
                "common_weaknesses": ["Missing baseline comparisons"],
                "score_range": {"min": 6, "max": 7, "mean": 6.5, "count": 2},
                "decision_pattern": "accept with moderate scores",
                "calibration_notes": ["Use as calibration context only"],
                "caveats": ["The old paper is related but not identical"],
            }
        return {}


class FakeProvider:
    def __init__(self, name, papers):
        self.name = name
        self._papers = papers
        self.seen_limit = None
        self.seen_queries = None

    def search(self, queries, limit=10):
        self.seen_limit = limit
        self.seen_queries = queries
        return ProviderResult(provider=self.name, papers=self._papers[:limit], warnings=[], status="used")


class FailingLLM:
    def complete_json(self, system_prompt, user_prompt):
        raise RuntimeError("LLM unavailable")


class TestRelatedWorkRAG(unittest.TestCase):
    def test_rag_llm_auth_error_hides_gateway_internals(self):
        class AuthFailingClient:
            def complete(self, system_prompt, messages):
                raise RuntimeError("Error code: 401 - LiteLLM Virtual Key expected.")

        agent = RAGLLMAgent(provider="cmu", api_key="", client=AuthFailingClient())

        with self.assertRaises(RuntimeError) as ctx:
            agent.complete_json("system", "user")

        message = str(ctx.exception)
        self.assertIn("CMU AI Gateway authentication failed", message)
        self.assertNotIn("LiteLLM", message)

    def test_search_query_cleanup_removes_markdown_and_anonymous_author_noise(self):
        query = "Beyond Self-Attention: A Model **Anonymous Author(s)** same problem task objective"

        cleaned = clean_search_query(query)

        self.assertEqual(
            cleaned,
            "Beyond Self-Attention: A Model same problem task objective",
        )
        self.assertNotIn("**", cleaned)
        self.assertNotIn("Anonymous Author", cleaned)

    def test_fallback_queries_are_cleaned_for_preview_and_provider_payload(self):
        provider = FakeProvider("OpenAlex", [])

        package = build_related_work_rag(
            paper="# Beyond Self-Attention **Anonymous Author(s)**\n\n## Abstract\n\nWe propose a transformer.",
            api_key="test",
            config=RAGConfig(),
            providers=[provider],
            llm_agent=FailingLLM(),
        )

        queries = [item["query"] for item in package["query_generation"]["queries"]]
        self.assertEqual(package["query_generation"]["source"], "fallback")
        self.assertEqual(package["reranking"]["source"], "none")
        self.assertTrue(queries)
        self.assertTrue(all("Anonymous Author" not in query for query in queries))
        self.assertTrue(all("**" not in query for query in queries))
        self.assertIn("self-attention", queries[0])
        self.assertIn("transformer", queries[0])
        self.assertNotIn("Beyond", queries[0])

    def test_reranking_source_marks_lexical_fallback(self):
        provider = FakeProvider("OpenAlex", [
            PaperMetadata(
                paper_id="",
                title="Subquadratic Transformer Baseline",
                year=2023,
                publication_date="2023-01-01",
                abstract="Efficient transformer sequence modeling baseline.",
                sources=["OpenAlex"],
                source_ids={"OpenAlex": "oa1"},
                matched_query_groups=["same_problem"],
            )
        ])

        package = build_related_work_rag(
            paper="# Target\n\n## Abstract\n\nWe propose an efficient transformer.",
            api_key="test",
            config=RAGConfig(),
            providers=[provider],
            llm_agent=FailingLLM(),
        )

        self.assertEqual(package["query_generation"]["source"], "fallback")
        self.assertEqual(package["reranking"]["source"], "fallback")
        self.assertEqual(len(package["reranking_results"]), 1)

    def test_provider_queries_all_groups_before_top_k_cap(self):
        class StubOpenAlex(OpenAlexProvider):
            def __init__(self):
                self.urls = []

            def _json_get(self, url, headers=None):
                self.urls.append(url)
                idx = len(self.urls)
                return {
                    "results": [
                        {
                            "id": f"https://openalex.org/W{idx}",
                            "title": f"Paper {idx}",
                            "publication_year": 2024,
                            "publication_date": "2024-01-01",
                            "authorships": [],
                            "primary_location": {},
                        }
                    ]
                }

        provider = StubOpenAlex()
        queries = [
            type("Q", (), {"group": group, "query": f"{group} query"})()
            for group in QUERY_GROUPS
        ]
        result = provider.search(queries, limit=2)

        self.assertEqual(len(provider.urls), len(QUERY_GROUPS))
        self.assertEqual(len(result.papers), 2)
        self.assertTrue(all("per-page=1" in url for url in provider.urls))

    def test_arxiv_uses_all_query_groups_with_inter_query_throttle(self):
        class StubArxiv(ArxivProvider):
            def __init__(self):
                self.urls = []
                self.sleep_indexes = []

            def _text_get(self, url, headers=None):
                self.urls.append(url)
                return """<?xml version="1.0" encoding="UTF-8"?>
                <feed xmlns="http://www.w3.org/2005/Atom">
                  <entry>
                    <id>http://arxiv.org/abs/2301.00001v1</id>
                    <title>Subquadratic Transformer Baseline</title>
                    <summary>Efficient sequence modeling.</summary>
                    <published>2023-01-01T00:00:00Z</published>
                    <author><name>A. Author</name></author>
                  </entry>
                </feed>"""

            def _sleep_between_queries(self, query_index):
                if query_index > 0:
                    self.sleep_indexes.append(query_index)

        provider = StubArxiv()
        queries = [
            type("Q", (), {"group": group, "query": f"{group} subquadratic transformer"})()
            for group in QUERY_GROUPS
        ]
        result = provider.search(queries, limit=10)

        self.assertEqual(len(provider.urls), len(QUERY_GROUPS))
        self.assertEqual(len(result.papers), 1)
        self.assertEqual(set(result.papers[0].matched_query_groups), set(QUERY_GROUPS))
        self.assertTrue(all("max_results=2" in url for url in provider.urls))
        self.assertEqual(provider.sleep_indexes, list(range(1, len(QUERY_GROUPS))))

    def test_semantic_scholar_uses_all_query_groups_with_inter_query_throttle(self):
        class StubSemanticScholar(SemanticScholarProvider):
            def __init__(self):
                self.urls = []
                self.sleep_indexes = []

            def _json_get(self, url, headers=None):
                self.urls.append(url)
                return {
                    "data": [
                        {
                            "paperId": "ss1",
                            "title": "Subquadratic Transformer Baseline",
                            "year": 2023,
                            "publicationDate": "2023-01-01",
                            "authors": [{"name": "A. Author"}],
                            "externalIds": {},
                        }
                    ]
                }

            def _sleep_between_queries(self, query_index):
                if query_index > 0:
                    self.sleep_indexes.append(query_index)

        provider = StubSemanticScholar()
        queries = [
            type("Q", (), {"group": group, "query": f"{group} subquadratic transformer"})()
            for group in QUERY_GROUPS
        ]
        result = provider.search(queries, limit=10)

        self.assertEqual(len(provider.urls), len(QUERY_GROUPS))
        self.assertEqual(len(result.papers), 1)
        self.assertEqual(set(result.papers[0].matched_query_groups), set(QUERY_GROUPS))
        self.assertTrue(all("limit=2" in url for url in provider.urls))
        self.assertEqual(provider.sleep_indexes, list(range(1, len(QUERY_GROUPS))))

    def test_semantic_scholar_stops_after_rate_limit(self):
        class RateLimitedSemanticScholar(SemanticScholarProvider):
            def __init__(self):
                self.urls = []
                self.sleep_indexes = []

            def _json_get(self, url, headers=None):
                self.urls.append(url)
                raise ProviderHTTPError(url, 429, "Too Many Requests")

            def _sleep_between_queries(self, query_index):
                if query_index > 0:
                    self.sleep_indexes.append(query_index)

        provider = RateLimitedSemanticScholar()
        queries = [
            type("Q", (), {"group": group, "query": f"{group} subquadratic transformer"})()
            for group in QUERY_GROUPS
        ]

        result = provider.search(queries, limit=10)

        self.assertEqual(len(provider.urls), 1)
        self.assertEqual(provider.sleep_indexes, [])
        self.assertEqual(result.status, "rate_limited")
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("Semantic Scholar rate-limited", result.warnings[0])

    def test_related_work_searches_providers_concurrently(self):
        barrier = Barrier(2)

        class BlockingProvider:
            def __init__(self, name):
                self.name = name

            def search(self, queries, limit=10):
                barrier.wait(timeout=1)
                return ProviderResult(
                    provider=self.name,
                    papers=[
                        PaperMetadata(
                            paper_id="",
                            title=f"{self.name} Paper",
                            year=2024,
                            publication_date="2024-01-01",
                            abstract="target retrieval",
                            sources=[self.name],
                            source_ids={self.name: self.name},
                            matched_query_groups=["same_problem"],
                        )
                    ],
                    warnings=[],
                    status="used",
                )

        package = build_related_work_rag(
            paper="# Target\n\n## Abstract\n\ntarget retrieval",
            api_key="test",
            config=RAGConfig(provider_top_k=2),
            providers=[BlockingProvider("Provider A"), BlockingProvider("Provider B")],
            llm_agent=FakeLLM(),
        )

        self.assertEqual(package["provider_status"]["Provider A"]["status"], "used")
        self.assertEqual(package["provider_status"]["Provider B"]["status"], "used")

    def test_llm_queries_provider_metadata_dedupe_cutoff_and_rerank(self):
        valid_a = PaperMetadata(
            paper_id="",
            title="Relevant Motion Control Baselines",
            authors=["A. Researcher"],
            year=2024,
            publication_date="2024-05-01",
            abstract="A paper about text-driven motion control baselines.",
            doi="10.1000/relevant",
            sources=["OpenAlex"],
            source_ids={"OpenAlex": "oa1"},
            matched_query_groups=["same_problem"],
        )
        duplicate = PaperMetadata(
            paper_id="",
            title="Relevant Motion Control Baselines",
            authors=["A. Researcher"],
            year=2024,
            publication_date="2024-05-01",
            abstract="Duplicate metadata from another provider.",
            doi="10.1000/relevant",
            sources=["Semantic Scholar"],
            source_ids={"Semantic Scholar": "ss1"},
            matched_query_groups=["benchmark_baseline"],
        )
        future = PaperMetadata(
            paper_id="",
            title="Future Leakage Paper",
            year=2025,
            publication_date="2025-01-10",
            abstract="Should be filtered.",
            sources=["arXiv"],
            source_ids={"arXiv": "2501.00001"},
            matched_query_groups=["novelty_competitor"],
        )
        providers = [
            FakeProvider("OpenAlex", [valid_a, future]),
            FakeProvider("Semantic Scholar", [duplicate]),
        ]

        package = build_related_work_rag(
            paper="# Target\n\n## Abstract\n\nWe propose text-driven motion control with new baselines.",
            topic="Computer Vision",
            api_key="test",
            config=RAGConfig(provider_top_k=10, rerank_top_k=5),
            providers=providers,
            llm_agent=FakeLLM(),
        )

        self.assertEqual(len(package["query_generation"]["queries"]), len(QUERY_GROUPS))
        self.assertEqual(package["query_generation"]["source"], "llm")
        self.assertEqual(package["reranking"]["source"], "llm")
        self.assertEqual(providers[0].seen_limit, 10)
        self.assertEqual(len(providers[0].seen_queries), len(QUERY_GROUPS))
        self.assertEqual(len(package["paper_metadata"]), 1)
        self.assertEqual(package["cutoff_report"]["num_removed_post_cutoff"], 1)
        paper = package["paper_metadata"][0]
        self.assertEqual(paper["sources"], ["OpenAlex", "Semantic Scholar"])
        self.assertEqual(paper["matched_query_groups"], ["benchmark_baseline", "same_problem"])
        self.assertEqual(package["reranking_results"][0]["paper_id"], paper["paper_id"])
        self.assertNotIn(paper["paper_id"], package["related_work_summary"])
        self.assertIn("Relevant Motion Control Baselines", package["related_work_summary"])
        self.assertIn("In 2024, A. Researcher published", package["related_work_summary"])

    def test_related_work_summary_uses_background_style_not_reviewer_guidance(self):
        class GuidanceSummaryLLM(FakeLLM):
            def complete_json(self, system_prompt, user_prompt):
                if "Candidate metadata:" in user_prompt:
                    ids = re.findall(r'"paper_id":\s*"(rw_[^"]+)"', user_prompt)
                    return {
                        "reranked_papers": [
                            {
                                "rank": 1,
                                "paper_id": ids[0],
                                "relevance_score": 0.9,
                                "relevance_types": ["same_method"],
                                "rationale": "Relevant method.",
                                "evidence_summary": "Relevant method.",
                            }
                        ],
                        "summary": f"Reviewers should compare against [{ids[0]}].",
                    }
                return super().complete_json(system_prompt, user_prompt)

        provider = FakeProvider("OpenAlex", [
            PaperMetadata(
                paper_id="",
                title="FNet: Mixing Tokens with Fourier Transforms",
                authors=["James Lee-Thorp", "Joshua Ainslie", "Ilya Eckstein"],
                year=2022,
                publication_date="2022-01-01",
                abstract="Fourier token mixing for transformers.",
                sources=["OpenAlex"],
                source_ids={"OpenAlex": "oa-fnet"},
                matched_query_groups=["same_method"],
            )
        ])

        package = build_related_work_rag(
            paper="# Target\n\n## Abstract\n\nWe propose Fourier-wavelet token mixing.",
            api_key="test",
            config=RAGConfig(),
            providers=[provider],
            llm_agent=GuidanceSummaryLLM(),
        )

        summary = package["related_work_summary"]
        self.assertNotIn("Reviewers should", summary)
        self.assertNotRegex(summary, r"rw_[A-Za-z0-9_]+")
        self.assertIn("FNet: Mixing Tokens with Fourier Transforms", summary)
        self.assertIn("In 2022, James Lee-Thorp et al. published", summary)

    def test_fallback_deduplicates_cross_identifier_records_and_reports_details(self):
        provider = FakeProvider("OpenAlex", [
            PaperMetadata(
                paper_id="",
                title="Unified Vision-Benchmark",
                authors=["Alice Smith", "Bob Jones"],
                year=2023,
                publication_date="2023-05-01",
                abstract="The paper introduces a new multi-modal fusion architecture.",
                doi="10.1000/unified",
                sources=["OpenAlex"],
                source_ids={"OpenAlex": "oa-unified"},
                matched_query_groups=["novelty_competitor"],
            ),
            PaperMetadata(
                paper_id="",
                title="Unified Vision Benchmark",
                authors=["Alice Smith", "Bob Jones"],
                year=2023,
                publication_date="2023-05-01",
                abstract=(
                    "We introduce a new multi-modal fusion architecture for robust classification. "
                    "The Unified Vision Benchmark contains 50,000 images across five domains with fixed "
                    "training and test splits. "
                    "The method improves accuracy by 4.2 percentage points over the strongest baseline."
                ),
                arxiv_id="2305.00001v2",
                sources=["arXiv"],
                source_ids={"arXiv": "2305.00001v2"},
                matched_query_groups=["benchmark_baseline"],
            ),
        ])

        package = build_related_work_rag(
            paper="# Target Fusion Model\n\n## Abstract\n\nA multi-modal classification system.",
            api_key="test",
            config=RAGConfig(rerank_top_k=5),
            providers=[provider],
            llm_agent=FailingLLM(),
        )

        self.assertEqual(package["reranking"]["source"], "fallback")
        self.assertEqual(len(package["paper_metadata"]), 1)
        self.assertEqual(
            package["paper_metadata"][0]["matched_query_groups"],
            ["benchmark_baseline", "novelty_competitor"],
        )
        self.assertEqual(package["paper_metadata"][0]["sources"], ["OpenAlex", "arXiv"])

        summary = package["related_work_summary"]
        self.assertEqual(summary.count('published "Unified Vision-Benchmark"'), 1)
        self.assertIn("new multi-modal fusion architecture", summary)
        self.assertIn("50,000 images across five domains", summary)
        self.assertIn("improves accuracy by 4.2 percentage points", summary)
        self.assertNotIn("Our work", summary)
        self.assertNotIn("provides benchmark or baseline context", summary)

    def test_fallback_excludes_the_target_paper_from_related_work(self):
        provider = FakeProvider("OpenAlex", [
            PaperMetadata(
                paper_id="",
                title="Beyond Self Attention A Subquadratic Fourier Wavelet Transformer",
                authors=["Target Author"],
                year=2024,
                publication_date="2024-01-01",
                abstract="The target paper itself.",
                doi="10.1000/target",
                sources=["OpenAlex"],
                source_ids={"OpenAlex": "oa-target"},
                matched_query_groups=["same_method"],
            ),
            PaperMetadata(
                paper_id="",
                title="Simple Hardware-Efficient Long Convolutions",
                authors=["Daniel Fu", "Elli Triantafillou", "Tatsunori Hashimoto"],
                year=2023,
                publication_date="2023-01-01",
                abstract=(
                    "The authors introduce a hardware-efficient long convolution architecture. "
                    "Experiments show higher throughput and lower memory use on long-sequence benchmarks."
                ),
                sources=["OpenAlex"],
                source_ids={"OpenAlex": "oa-long-conv"},
                matched_query_groups=["same_constraints", "benchmark_baseline"],
            ),
        ])

        package = build_related_work_rag(
            paper=(
                "# Beyond Self-Attention: A Subquadratic Fourier-Wavelet Transformer\n\n"
                "## Abstract\n\nA subquadratic transformer."
            ),
            api_key="test",
            config=RAGConfig(rerank_top_k=5),
            providers=[provider],
            llm_agent=FailingLLM(),
        )

        self.assertEqual(package["cutoff_report"]["num_removed_as_target"], 1)
        self.assertEqual(len(package["paper_metadata"]), 1)
        self.assertNotIn("Beyond Self Attention", package["related_work_summary"])
        self.assertIn("Simple Hardware-Efficient Long Convolutions", package["related_work_summary"])

    def test_reranker_invalid_ids_fall_back_to_valid_candidates(self):
        class BadRerankLLM(FakeLLM):
            def complete_json(self, system_prompt, user_prompt):
                if "Candidate metadata:" in user_prompt:
                    return {
                        "reranked_papers": [{"paper_id": "invented_id", "relevance_score": 1.0}],
                        "summary": "bad",
                    }
                return super().complete_json(system_prompt, user_prompt)

        provider = FakeProvider("OpenAlex", [
            PaperMetadata(
                paper_id="",
                title="Valid Pre Cutoff Paper",
                year=2023,
                publication_date="2023-02-01",
                abstract="valid target overlap paper",
                sources=["OpenAlex"],
                source_ids={"OpenAlex": "oa2"},
                matched_query_groups=["same_problem"],
            )
        ])
        package = build_related_work_rag(
            paper="# Target\n\n## Abstract\n\nvalid target overlap paper",
            api_key="test",
            config=RAGConfig(),
            providers=[provider],
            llm_agent=BadRerankLLM(),
        )

        self.assertEqual(len(package["reranking_results"]), 1)
        self.assertNotEqual(package["reranking_results"][0]["paper_id"], "invented_id")
        self.assertTrue(any("no valid paper IDs" in w for w in package["warnings"]))

    def test_reranker_fills_short_llm_output_and_calibrates_saturated_scores(self):
        class ShortTiedRerankLLM(FakeLLM):
            def complete_json(self, system_prompt, user_prompt):
                if "Candidate metadata:" in user_prompt:
                    ids = re.findall(r'"paper_id":\s*"(rw_[^"]+)"', user_prompt)
                    return {
                        "reranked_papers": [
                            {
                                "rank": 1,
                                "paper_id": ids[0],
                                "relevance_score": 1.0,
                                "relevance_types": ["same_method"],
                                "rationale": "Most direct.",
                                "evidence_summary": "Most direct.",
                            },
                            {
                                "rank": 2,
                                "paper_id": ids[1],
                                "relevance_score": 1.0,
                                "relevance_types": ["benchmark_baseline"],
                                "rationale": "Important benchmark.",
                                "evidence_summary": "Important benchmark.",
                            },
                        ],
                        "summary": "Useful related work includes the ranked papers.",
                    }
                return super().complete_json(system_prompt, user_prompt)

        provider = FakeProvider("OpenAlex", [
            PaperMetadata(
                paper_id="",
                title="Fourier Transformer Baseline",
                year=2021,
                publication_date="2021-01-01",
                abstract="Fourier transformer token mixing baseline.",
                sources=["OpenAlex"],
                source_ids={"OpenAlex": "oa1"},
                matched_query_groups=["same_method"],
            ),
            PaperMetadata(
                paper_id="",
                title="Long Range Arena",
                year=2020,
                publication_date="2020-01-01",
                abstract="Benchmark for efficient long sequence transformers.",
                sources=["OpenAlex"],
                source_ids={"OpenAlex": "oa2"},
                matched_query_groups=["benchmark_baseline"],
            ),
            PaperMetadata(
                paper_id="",
                title="Wavelet Transformer",
                year=2022,
                publication_date="2022-01-01",
                abstract="Wavelet transformer local frequency representation.",
                sources=["OpenAlex"],
                source_ids={"OpenAlex": "oa3"},
                matched_query_groups=["novelty_competitor"],
            ),
            PaperMetadata(
                paper_id="",
                title="Low Rank Multimodal Fusion Transformer",
                year=2020,
                publication_date="2020-01-01",
                abstract="Low rank multimodal fusion for sequence models.",
                sources=["OpenAlex"],
                source_ids={"OpenAlex": "oa4"},
                matched_query_groups=["same_problem"],
            ),
        ])

        package = build_related_work_rag(
            paper="# Target\n\n## Abstract\n\nWe propose Fourier wavelet transformer multimodal fusion.",
            api_key="test",
            config=RAGConfig(rerank_top_k=4),
            providers=[provider],
            llm_agent=ShortTiedRerankLLM(),
        )

        self.assertEqual(package["reranking"]["source"], "mixed")
        self.assertEqual(len(package["reranking_results"]), 4)
        self.assertEqual([item["rank"] for item in package["reranking_results"]], [1, 2, 3, 4])
        self.assertNotEqual(
            len({item["relevance_score"] for item in package["reranking_results"][:2]}),
            1,
        )
        self.assertTrue(any("appended in cleaned candidate order" in item["rationale"] for item in package["reranking_results"][2:]))

    def test_candidate_cap_is_applied_before_llm_rerank(self):
        class CapturingRerankLLM(FakeLLM):
            def __init__(self):
                super().__init__()
                self.seen_candidate_count = None

            def complete_json(self, system_prompt, user_prompt):
                if "Candidate metadata:" in user_prompt:
                    metadata_json = user_prompt.split("Candidate metadata:", 1)[1].split("\n\nRerank candidates", 1)[0].strip()
                    ids = [paper["paper_id"] for paper in json.loads(metadata_json)]
                    self.seen_candidate_count = len(ids)
                    return {
                        "reranked_papers": [
                            {
                                "rank": index + 1,
                                "paper_id": paper_id,
                                "relevance_score": 0.9 - (index * 0.05),
                                "relevance_types": ["same_problem"],
                                "rationale": "Candidate retained for reranking.",
                                "evidence_summary": "Candidate retained for reranking.",
                            }
                            for index, paper_id in enumerate(ids)
                        ],
                        "summary": "Useful related work includes the retained candidates.",
                    }
                return super().complete_json(system_prompt, user_prompt)

        provider = FakeProvider("OpenAlex", [
            PaperMetadata(
                paper_id="",
                title=f"Candidate {index}",
                year=2023,
                publication_date="2023-01-01",
                abstract=f"candidate {index} Fourier transformer multimodal fusion",
                sources=["OpenAlex"],
                source_ids={"OpenAlex": f"oa{index}"},
                matched_query_groups=["same_problem"],
            )
            for index in range(5)
        ])
        llm = CapturingRerankLLM()

        package = build_related_work_rag(
            paper="# Target\n\n## Abstract\n\nWe propose Fourier transformer multimodal fusion.",
            api_key="test",
            config=RAGConfig(rerank_top_k=3),
            providers=[provider],
            llm_agent=llm,
        )

        self.assertEqual(llm.seen_candidate_count, 3)
        self.assertEqual(len(package["paper_metadata"]), 3)
        self.assertEqual(len(package["reranking_results"]), 3)
        self.assertEqual(package["cutoff_report"]["num_cutoff_valid"], 5)
        self.assertEqual(package["cutoff_report"]["candidate_cap"], 3)
        self.assertEqual(package["cutoff_report"]["num_removed_by_candidate_cap"], 2)

    def test_orchestrator_adds_review_memory_to_shared_package(self):
        class FakeMemoryProvider:
            def find_review_memory_case(self, ranked_candidates, max_reviews=4):
                candidate = ranked_candidates[0]
                return (
                    ReviewMemoryCase(
                        source_paper_id=candidate["paper_id"],
                        source_rank=candidate["rank"],
                        title=candidate["title"],
                        year=candidate.get("year"),
                        openreview_forum_id="forum123",
                        openreview_url="https://openreview.net/forum?id=forum123",
                        decision="accept",
                        score_range={"min": 6, "max": 7, "mean": 6.5, "count": 2},
                        reviews=[
                            {"reviewer_id": "Reviewer_1", "rating": 6, "confidence": 4, "text": "Strong empirical evaluation but missing baselines."},
                            {"reviewer_id": "Reviewer_2", "rating": 7, "confidence": 3, "text": "Clear motivation with some baseline concerns."},
                        ],
                    ),
                    [],
                    [candidate["paper_id"]],
                )

        valid = PaperMetadata(
            paper_id="",
            title="Relevant OpenReview Paper",
            authors=["A. Reviewer"],
            year=2024,
            publication_date="2024-05-01",
            abstract="Relevant method and benchmark.",
            url="https://openreview.net/forum?id=forum123",
            sources=["OpenAlex"],
            source_ids={"OpenAlex": "oa3"},
            matched_query_groups=["same_problem"],
        )

        package = build_rag_package(
            paper="# Target\n\n## Abstract\n\nWe propose a relevant method.",
            topic="NLP",
            api_key="test",
            config=RAGConfig(enable_review_memory_rag=True),
            providers=[FakeProvider("OpenAlex", [valid])],
            review_memory_provider=FakeMemoryProvider(),
            llm_agent=FakeLLM(),
        )

        self.assertEqual(package["review_memory"]["status"], "used")
        self.assertEqual(package["review_memory"]["selected_case"]["openreview_forum_id"], "forum123")
        self.assertIn("Reviewers accepted", package["review_memory"]["summary"]["summary"])

    def test_example_markdown_runs_related_work_and_review_memory_rag(self):
        class FakeMemoryProvider:
            def find_review_memory_case(self, ranked_candidates, max_reviews=4):
                candidate = ranked_candidates[0]
                return (
                    ReviewMemoryCase(
                        source_paper_id=candidate["paper_id"],
                        source_rank=candidate["rank"],
                        title=candidate["title"],
                        year=candidate.get("year"),
                        openreview_forum_id="example-forum",
                        openreview_url="https://openreview.net/forum?id=example-forum",
                        decision="accept",
                        score_range={"min": 6, "max": 7, "mean": 6.5, "count": 2},
                        reviews=[
                            {"reviewer_id": "Reviewer_1", "rating": 6, "confidence": 4, "text": "Clear experimental motivation with limited baselines."},
                            {"reviewer_id": "Reviewer_2", "rating": 7, "confidence": 3, "text": "Readable paper with useful empirical setup."},
                        ],
                    ),
                    [],
                    [candidate["paper_id"]],
                )

        paper = Path("data/md/example_paper.md").read_text(encoding="utf-8")
        provider = FakeProvider("OpenAlex", [
            PaperMetadata(
                paper_id="",
                title="URSA: A Unified Resource Allocator for Registers and Functional Units in VLIW Architectures",
                authors=["David A. Berson", "Rajiv Gupta", "Mary Lou Soffa"],
                year=1993,
                publication_date="1993-01-01",
                abstract="A resource allocation method for registers and functional units in VLIW architectures.",
                sources=["OpenAlex"],
                source_ids={"OpenAlex": "oa-ursa"},
                matched_query_groups=["same_problem", "benchmark_baseline"],
            )
        ])

        package = build_rag_package(
            paper=paper,
            api_key="test",
            config=RAGConfig(enable_review_memory_rag=True, cutoff_date="2024-12-31"),
            providers=[provider],
            review_memory_provider=FakeMemoryProvider(),
            llm_agent=FakeLLM(),
        )

        self.assertEqual(package["query_generation"]["source"], "llm")
        self.assertEqual(package["provider_status"]["OpenAlex"]["status"], "used")
        self.assertEqual(package["reranking"]["source"], "llm")
        self.assertEqual(package["review_memory"]["status"], "used")
        self.assertEqual(package["review_memory"]["selected_case"]["openreview_forum_id"], "example-forum")
        self.assertIn("URSA: A Unified Resource Allocator", package["related_work_summary"])
        self.assertNotRegex(package["related_work_summary"], r"rw_[A-Za-z0-9_]+")


class TestMASLoopRAGInjection(unittest.TestCase):
    def test_rag_package_is_injected_only_into_reviewer_paper(self):
        import mas_loop

        seen = {"reviewer_papers": [], "author_papers": []}

        class FakeReviewer:
            name = "Reviewer"

            def __init__(self, paper, reviewer_type, topic, api_key, provider, model):
                seen["reviewer_papers"].append(paper)
                self.name = "Reviewer 1 (Novelty)"

            def call(self, prompt):
                return json.dumps({
                    "reviewer": self.name,
                    "decision": "reject",
                    "scores": {"novelty": 1, "soundness": 1, "significance": 1, "evaluation": 1, "clarity": 1},
                    "strengths": [],
                    "weaknesses": [],
                    "summary_comment": "summary",
                })

        class FakeAuthor:
            name = "Author"

            def __init__(self, paper, topic, api_key, provider, model):
                seen["author_papers"].append(paper)

            def call(self, prompt):
                return "{}"

        class FakeDetector(FakeAuthor):
            name = "AI Detector"

        class FakeConf(FakeAuthor):
            name = "Conference Recommender"

            def call(self, prompt):
                return json.dumps({"ICML": {"fit_score": 1, "why_it_fits": [], "why_it_does_not_fit": []}})

        package = {
            "related_work_summary": "Use [rw_001] for baselines.",
            "paper_metadata": [{"paper_id": "rw_001", "title": "Baseline Paper", "year": 2024, "sources": ["OpenAlex"], "authors": []}],
            "reranking_results": [{"paper_id": "rw_001", "rank": 1, "relevance_score": 0.9, "rationale": "baseline"}],
            "cutoff_report": {"cutoff_date": "2024-12-31", "num_used": 1, "num_removed_post_cutoff": 0, "num_removed_undated": 0},
            "warnings": [],
        }

        with patch.object(mas_loop, "Reviewer", FakeReviewer), \
             patch.object(mas_loop, "Author", FakeAuthor), \
             patch.object(mas_loop, "AIDetector", FakeDetector), \
             patch.object(mas_loop, "ConferenceRecommender", FakeConf):
            result = mas_loop.main(
                paper="# Paper",
                topic="NLP",
                n_iter=1,
                reviewer_types=["reviewer_a"],
                api_key="key",
                enable_rag=True,
                precomputed_rag_package=package,
                run_citation_check=False,
            )

        self.assertIn("###RAG_EVIDENCE###", seen["reviewer_papers"][0])
        self.assertNotIn("###RAG_EVIDENCE###", seen["author_papers"][0])
        self.assertEqual(result["rag_package"], package)
        self.assertEqual(result["reviewers"][0]["reviewer"], "Reviewer 1 (Novelty)")


class TestReviewMemoryRAG(unittest.TestCase):
    def test_selects_highest_ranked_candidate_with_public_reviews(self):
        fetched_forums = []

        class StubOpenReviewMemoryProvider(OpenReviewMemoryProvider):
            def __init__(self):
                pass

            def _lookup_forum_by_title(self, title):
                return ""

            def _fetch_forum_notes(self, forum):
                fetched_forums.append(forum)
                return [
                    {
                        "id": "review1",
                        "forum": forum,
                        "invitation": "ICLR.cc/2024/Conference/Submission1/-/Official_Review",
                        "signatures": ["ICLR.cc/2024/Conference/Submission1/AnonReviewer1"],
                        "content": {
                            "summary": {"value": "The paper studies the same task."},
                            "strengths": {"value": "Strong empirical evaluation."},
                            "weaknesses": {"value": "Missing some baselines."},
                            "rating": {"value": "6: marginally above acceptance threshold"},
                            "confidence": {"value": "4"},
                        },
                    },
                    {
                        "id": "decision1",
                        "forum": forum,
                        "invitation": "ICLR.cc/2024/Conference/Submission1/-/Decision",
                        "content": {"decision": {"value": "Accept: poster"}},
                    },
                ]

        package = {
            "paper_metadata": [
                {"paper_id": "rw_top", "title": "Top Candidate Without Reviews", "year": 2024, "url": ""},
                {"paper_id": "rw_second", "title": "Second Candidate With Reviews", "year": 2024, "url": "https://openreview.net/forum?id=forum-second"},
            ],
            "reranking_results": [
                {"paper_id": "rw_top", "rank": 1, "relevance_score": 0.95},
                {"paper_id": "rw_second", "rank": 2, "relevance_score": 0.9},
            ],
        }

        result = build_review_memory_from_package(
            package,
            llm_agent=FakeLLM(),
            config=RAGConfig(enable_review_memory_rag=True),
            provider=StubOpenReviewMemoryProvider(),
        )

        self.assertEqual(result["status"], "used")
        self.assertEqual(result["selected_case"]["source_paper_id"], "rw_second")
        self.assertEqual(result["selected_case"]["decision"], "accept")
        self.assertEqual(result["selected_case"]["score_range"]["min"], 6.0)
        self.assertEqual(result["attempted_source_paper_ids"], ["rw_top", "rw_second"])
        self.assertEqual(fetched_forums, ["forum-second"])

    def test_review_memory_stops_after_openreview_forbidden(self):
        class ForbiddenOpenReviewMemoryProvider(OpenReviewMemoryProvider):
            def __init__(self):
                pass

            def _lookup_forum_by_title(self, title):
                raise ProviderHTTPError("https://api2.openreview.net/notes", 403, "Forbidden")

        package = {
            "paper_metadata": [
                {"paper_id": "rw_top", "title": "Top Candidate", "year": 2024, "url": ""},
                {"paper_id": "rw_second", "title": "Second Candidate", "year": 2024, "url": ""},
            ],
            "reranking_results": [
                {"paper_id": "rw_top", "rank": 1, "relevance_score": 0.95},
                {"paper_id": "rw_second", "rank": 2, "relevance_score": 0.9},
            ],
        }

        result = build_review_memory_from_package(
            package,
            llm_agent=FakeLLM(),
            config=RAGConfig(enable_review_memory_rag=True),
            provider=ForbiddenOpenReviewMemoryProvider(),
        )

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["attempted_source_paper_ids"], ["rw_top"])
        self.assertEqual(
            len([warning for warning in result["warnings"] if "OpenReview review-memory lookup requires OpenReview authentication" in warning]),
            1,
        )
        self.assertTrue(any("Review-memory lookup was unavailable" in warning for warning in result["warnings"]))

    def test_review_memory_is_disabled_by_default(self):
        class FailingProvider:
            def find_review_memory_case(self, ranked_candidates, max_reviews=4):
                raise AssertionError("review-memory provider should not be called by default")

        package = {
            "paper_metadata": [
                {"paper_id": "rw_top", "title": "Top Candidate", "year": 2024, "url": "https://openreview.net/forum?id=forum-top"},
            ],
            "reranking_results": [
                {"paper_id": "rw_top", "rank": 1, "relevance_score": 0.95},
            ],
        }

        result = build_review_memory_from_package(
            package,
            llm_agent=FakeLLM(),
            config=RAGConfig(),
            provider=FailingProvider(),
        )

        self.assertEqual(result["status"], "disabled")
        self.assertEqual(result["warnings"], [])

    def test_openreview_title_lookup_uses_normalized_forum_search(self):
        class SearchProvider(OpenReviewMemoryProvider):
            def __init__(self):
                super().__init__(cache_dir="data/rag_cache")
                self.calls = []

            def _json_endpoint(self, base_url, params):
                self.calls.append((base_url, params))
                return {
                    "notes": [
                        {
                            "id": "forum-search",
                            "forum": "forum-search",
                            "content": {"title": {"value": "Relevant: OpenReview Paper"}},
                        }
                    ]
                }

        provider = SearchProvider()

        self.assertEqual(provider._lookup_forum_by_title("Relevant: OpenReview Paper"), "forum-search")
        self.assertEqual(
            provider.calls,
            [(OPENREVIEW_SEARCH_URL, {"term": "relevant openreview paper", "limit": 50, "source": "forum"})],
        )

    def test_openreview_forum_fetch_merges_direct_replies_and_deduplicates(self):
        class ForumProvider(OpenReviewMemoryProvider):
            def __init__(self):
                super().__init__(cache_dir="data/rag_cache")

            def _json_endpoint(self, base_url, params):
                review_one = {"id": "review-1", "forum": "forum-1"}
                if "forum" in params:
                    return {"notes": [review_one]}
                return {"notes": [review_one, {"id": "review-2", "forum": "forum-1"}]}

        provider = ForumProvider()

        self.assertEqual(
            [note["id"] for note in provider._fetch_forum_notes("forum-1")],
            ["review-1", "review-2"],
        )

    def test_openreview_forum_fetch_keeps_forum_results_when_replyto_is_unsupported(self):
        class ForumProvider(OpenReviewMemoryProvider):
            def __init__(self):
                super().__init__(cache_dir="data/rag_cache")
                self.calls = []

            def _json_endpoint(self, base_url, params):
                self.calls.append((base_url, params))
                if "forum" in params:
                    return {"notes": [{"id": "review-1", "forum": "forum-1"}]}
                raise ProviderHTTPError(base_url, 400, "Bad Request")

        provider = ForumProvider()

        self.assertEqual(provider._fetch_forum_notes("forum-1"), [{"id": "review-1", "forum": "forum-1"}])
        self.assertEqual(
            provider.calls,
            [
                (OPENREVIEW_NOTES_URL, {"forum": "forum-1", "limit": 1000}),
                (OPENREVIEW_NOTES_URL, {"replyto": "forum-1", "limit": 1000}),
            ],
        )

    def test_openreview_auth_token_is_sent_to_api(self):
        class TokenProvider(OpenReviewMemoryProvider):
            def __init__(self):
                super().__init__(cache_dir="data/rag_cache")
                self.headers = None

            def _json_get(self, url, headers=None):
                self.headers = headers
                return {"notes": []}

        provider = TokenProvider()

        with patch.dict("os.environ", {"OPENREVIEW_ACCESS_TOKEN": "token-123"}, clear=False):
            provider._json_endpoint(OPENREVIEW_NOTES_URL, {"forum": "forum123"})

        self.assertIn("Bearer token-123", provider.headers["Authorization"])
        self.assertIn("openreview.accessToken=token-123", provider.headers["Cookie"])

    def test_review_memory_ignores_non_public_openreview_notes(self):
        class MixedVisibilityProvider(OpenReviewMemoryProvider):
            def __init__(self):
                pass

            def _fetch_forum_notes(self, forum):
                return [
                    {
                        "id": "private-review",
                        "forum": forum,
                        "readers": ["ICLR.cc/2024/Conference"],
                        "invitation": "ICLR.cc/2024/Conference/Submission1/-/Official_Review",
                        "signatures": ["ICLR.cc/2024/Conference/Submission1/AnonReviewer1"],
                        "content": {
                            "summary": {"value": "Private review should not be used."},
                            "rating": {"value": "1"},
                        },
                    },
                    {
                        "id": "public-review",
                        "forum": forum,
                        "readers": ["everyone"],
                        "invitation": "ICLR.cc/2024/Conference/Submission1/-/Official_Review",
                        "signatures": ["ICLR.cc/2024/Conference/Submission1/AnonReviewer2"],
                        "content": {
                            "summary": {"value": "Public review should be used."},
                            "strengths": {"value": "Clear empirical result."},
                            "rating": {"value": "6"},
                        },
                    },
                    {
                        "id": "public-decision",
                        "forum": forum,
                        "readers": ["everyone"],
                        "invitation": "ICLR.cc/2024/Conference/Submission1/-/Decision",
                        "content": {"decision": {"value": "Reject"}},
                    },
                ]

        package = {
            "paper_metadata": [
                {"paper_id": "rw_top", "title": "Top Candidate", "year": 2024, "url": "https://openreview.net/forum?id=forum-top"},
            ],
            "reranking_results": [
                {"paper_id": "rw_top", "rank": 1, "relevance_score": 0.95},
            ],
        }

        llm = FakeLLM()
        result = build_review_memory_from_package(
            package,
            llm_agent=llm,
            config=RAGConfig(enable_review_memory_rag=True),
            provider=MixedVisibilityProvider(),
        )

        self.assertEqual(result["status"], "used")
        self.assertEqual(result["selected_case"]["decision"], "reject")
        self.assertEqual(len(result["selected_case"]["reviews"]), 1)
        self.assertEqual(result["selected_case"]["reviews"][0]["review_id"], "public-review")
        self.assertNotIn("Private review", llm.calls[-1][1])

    def test_review_memory_captures_review_comment_fields_but_not_comment_notes(self):
        class ReviewAndCommentProvider(OpenReviewMemoryProvider):
            def __init__(self):
                pass

            def _fetch_forum_notes(self, forum):
                return [
                    {
                        "id": "official-review",
                        "forum": forum,
                        "readers": ["everyone"],
                        "invitations": ["ICML.cc/2025/Conference/Submission1/-/Review_Rating"],
                        "signatures": ["ICML.cc/2025/Conference/Submission1/Reviewer_abcd"],
                        "content": {
                            "comments": {"value": "The method is clear, but the evaluation needs another baseline."},
                            "strengths": {"value": "Clear motivation and presentation."},
                            "overall_recommendation": {"value": "5: Marginally above threshold"},
                        },
                    },
                    {
                        "id": "public-comment",
                        "forum": forum,
                        "readers": ["everyone"],
                        "invitation": "ICML.cc/2025/Conference/Submission1/-/Comment",
                        "signatures": ["ICML.cc/2025/Conference/Submission1/Reviewer_efgh"],
                        "content": {
                            "review": {"value": "This discussion reply must not become a separate review."},
                            "rating": {"value": "1"},
                        },
                    },
                    {
                        "id": "author-response",
                        "forum": forum,
                        "readers": ["everyone"],
                        "invitation": "ICML.cc/2025/Conference/Submission1/-/Rebuttal",
                        "signatures": ["ICML.cc/2025/Conference/Submission1/Authors"],
                        "content": {
                            "review": {"value": "This author response must not become a review."},
                            "rating": {"value": "10"},
                        },
                    },
                ]

        package = {
            "paper_metadata": [
                {
                    "paper_id": "rw_top",
                    "title": "Top Candidate",
                    "year": 2025,
                    "url": "https://openreview.net/forum?id=forum-top",
                },
            ],
            "reranking_results": [
                {"paper_id": "rw_top", "rank": 1, "relevance_score": 0.95},
            ],
        }

        result = build_review_memory_from_package(
            package,
            llm_agent=FakeLLM(),
            config=RAGConfig(enable_review_memory_rag=True),
            provider=ReviewAndCommentProvider(),
        )

        reviews = result["selected_case"]["reviews"]
        self.assertEqual([review["review_id"] for review in reviews], ["official-review"])
        self.assertIn("comments: The method is clear", reviews[0]["text"])
        self.assertNotIn("discussion reply", reviews[0]["text"])
        self.assertEqual(reviews[0]["rating"], 5.0)

    def test_prompt_marks_review_memory_as_auxiliary_context(self):
        package = {
            "related_work_summary": "Related work summary.",
            "paper_metadata": [
                {"paper_id": "rw_second", "title": "Second Candidate With Reviews", "year": 2024, "sources": ["OpenAlex"], "authors": []}
            ],
            "reranking_results": [
                {"paper_id": "rw_second", "rank": 1, "relevance_score": 0.9, "rationale": "related"}
            ],
            "review_memory": {
                "status": "used",
                "selected_case": {
                    "source_paper_id": "rw_second",
                    "source_rank": 1,
                    "title": "Second Candidate With Reviews",
                    "year": 2024,
                    "openreview_forum_id": "forum-second",
                    "decision": "accept",
                    "score_range": {"min": 6, "max": 7, "mean": 6.5, "count": 2},
                },
                "summary": {
                    "summary": "Reviewers praised evaluation but criticized missing baselines.",
                    "common_strengths": ["Strong empirical evaluation"],
                    "common_weaknesses": ["Missing baseline comparisons"],
                    "score_range": {"min": 6, "max": 7, "mean": 6.5, "count": 2},
                    "decision_pattern": "accept with moderate scores",
                    "calibration_notes": ["Use as calibration context only"],
                },
            },
            "cutoff_report": {"cutoff_date": "2024-12-31", "num_used": 1, "num_removed_post_cutoff": 0, "num_removed_undated": 0},
        }

        block = format_rag_prompt_block(package)

        self.assertIn("Review-memory auxiliary context", block)
        self.assertIn("calibration only", block)
        self.assertIn("not direct evidence about the target paper", block)
        self.assertIn("Missing baseline comparisons", block)


if __name__ == "__main__":
    unittest.main()
