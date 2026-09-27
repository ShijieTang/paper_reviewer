import json
import re
import unittest
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

from rag.config import RAGConfig
from rag.llm import RAGLLMAgent
from rag.models import PaperMetadata
from rag.orchestrator import build_rag_package
from rag.prompt_views import format_rag_prompt_block
from rag.providers.arxiv import ArxivProvider
from rag.providers.openalex import OpenAlexProvider
from rag.providers.base import ProviderHTTPError, ProviderResult, clean_search_query
from rag.providers.semantic_scholar import SemanticScholarProvider
from rag.related_work import QUERY_GROUPS, build_related_work_rag


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


if __name__ == "__main__":
    unittest.main()
