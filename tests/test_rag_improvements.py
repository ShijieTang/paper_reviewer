"""Offline tests for the related-work RAG improvements (options 1-5)."""
import copy
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import mas_loop  # noqa: E402
from rag.config import RAGConfig, rag_config_from_dict  # noqa: E402
from rag.missing_references import build_missing_references  # noqa: E402
from rag.models import PaperMetadata  # noqa: E402
from rag.prompt_views import format_analyst_input, format_engaged_rag_block  # noqa: E402
from rag.providers.base import ProviderResult  # noqa: E402
from rag.reference_expansion import GROUP, expand_from_references, interleave  # noqa: E402
from rag.refine import ManuscriptReferences, is_submission_itself, refine_package, select_shown  # noqa: E402
from rag.related_work import build_related_work_rag  # noqa: E402
from review_schema import REVIEW_SCORE_KEYS, validate_rag_review_fields, validate_rw_analyst_report  # noqa: E402

PAPER = """# Sparse Mixture Routing for Long-Context Retrieval

## Abstract

We propose sparse mixture routing for long-context retrieval augmented generation.

## Method

Our router selects experts for retrieval.

# References

Smith, J. and Lee, K. Dense passage retrieval for open-domain question answering. In EMNLP, 2020.
Gao, L. Retrieval augmented generation for knowledge intensive tasks with sparse experts. arXiv:2101.00001, 2021.

# Appendix

Extra material.
"""

TITLE = "Sparse Mixture Routing for Long-Context Retrieval"


def _work(pid, title, year=2023, authors=("Ann Author",), abstract="An abstract."):
    return {"paper_id": pid, "title": title, "authors": list(authors), "year": year,
            "publication_date": f"{year}-01-01", "venue": "ICLR", "url": f"https://example.org/{pid}",
            "doi": "", "arxiv_id": "", "abstract": abstract, "matched_query_groups": ["core_task"]}


def make_package():
    works = [
        _work("rw_self", "Sparse mixture routing for long context retrieval"),               # the submission
        _work("rw_cited", "Dense Passage Retrieval for Open-Domain Question Answering", 2020, ("Jon Smith",)),
        _work("rw_new1", "Routing Experts for Retrieval Augmented Language Models", 2024, ("Bo Chen",)),
        _work("rw_low", "A Survey of Image Classification", 2019),
        _work("rw_new2", "Mixture of Retrievers for Long Documents", 2024, ("Cy Diaz",)),
        _work("rw_new3", "Long-Context Benchmarks for Retrieval", 2023, ("Di Evans",)),
    ]
    scores = {"rw_self": 0.99, "rw_cited": 0.9, "rw_new1": 0.85, "rw_low": 0.2, "rw_new2": 0.7, "rw_new3": 0.6}
    return {
        "paper_id": "paper_x",
        "target_paper_summary": {"paper_id": "paper_x", "title": TITLE, "abstract": "We propose routing.",
                                 "claims": ["We propose sparse mixture routing."]},
        "paper_metadata": works,
        "reranking_results": [{"paper_id": w["paper_id"], "rank": i + 1, "relevance_score": scores[w["paper_id"]],
                               "rationale": f"why {w['paper_id']}", "evidence_summary": ""}
                              for i, w in enumerate(works)],
        "related_work_summary": "summary",
        "warnings": [],
        "cutoff_report": {"cutoff_date": "2024-12-31"},
    }


IMPROVED = dict(prompt_style="engaged", strict_self_exclusion=True, tag_cited=True, min_relevance=0.5,
                max_shown=3, min_uncited_shown=2, missing_references_report=True)


# ── Option 3: refinement (self-exclusion, cited tagging, selection) ──────────

class TestRefine(unittest.TestCase):
    def test_defaults_leave_improvements_off(self):
        self.assertFalse(RAGConfig().improvements_enabled)
        self.assertTrue(rag_config_from_dict(IMPROVED).improvements_enabled)

    def test_reference_section_and_cited_matching(self):
        refs = ManuscriptReferences(PAPER)
        self.assertTrue(refs)
        self.assertNotIn("Extra material", refs.section)
        self.assertTrue(refs.cites({"title": "Dense Passage Retrieval for Open-Domain Question Answering"}))
        self.assertTrue(refs.cites({"title": "Unrelated", "arxiv_id": "2101.00001"}))
        self.assertFalse(refs.cites({"title": "Routing Experts for Retrieval Augmented Language Models"}))

    def test_self_detection(self):
        self.assertTrue(is_submission_itself({"title": TITLE.lower() + "."}, TITLE, 0.9))
        self.assertFalse(is_submission_itself({"title": "Mixture of Retrievers"}, TITLE, 0.9))
        self.assertTrue(is_submission_itself({"title": "x", "arxiv_id": "2401.12345v2"}, TITLE, 0.9,
                                             target_arxiv_id="2401.12345"))

    def test_refine_removes_self_and_low_relevance_tags_cited_and_reserves_uncited(self):
        pkg = make_package()
        original = copy.deepcopy(pkg)
        out = refine_package(pkg, PAPER, rag_config_from_dict(IMPROVED))
        self.assertEqual(pkg, original)                                   # input untouched
        ref = out["refinement"]
        self.assertEqual(ref["removed_as_submission"], ["Sparse mixture routing for long context retrieval"])
        self.assertEqual(ref["removed_low_relevance"], ["A Survey of Image Classification"])
        cited = {w["paper_id"]: w["cited_in_manuscript"] for w in out["candidate_works"]}
        self.assertTrue(cited["rw_cited"])
        self.assertFalse(cited["rw_new1"])
        shown = out["shown_works"]
        self.assertEqual([w["label"] for w in shown], ["R1", "R2", "R3"])
        self.assertEqual([w["paper_id"] for w in shown], ["rw_cited", "rw_new1", "rw_new2"])   # rank order kept
        self.assertEqual(ref["num_shown_uncited"], 2)
        labels = {w["paper_id"]: w["label"] for w in out["candidate_works"]}
        self.assertEqual((labels["rw_new1"], labels["rw_new3"]), ("R2", None))   # shown works carry their label

    def test_reserved_uncited_slots_displace_cited_works(self):
        cands = [{"cited_in_manuscript": True, "id": i} for i in range(4)] + \
                [{"cited_in_manuscript": False, "id": 9}]
        chosen = select_shown(cands, max_shown=3, min_uncited=1)
        self.assertEqual([c["id"] for c in chosen], [0, 1, 9])
        self.assertEqual([c["id"] for c in select_shown(cands, 3, 0)], [0, 1, 2])

    def test_cited_works_kept_but_tagged_when_not_strict(self):
        cfg = rag_config_from_dict({"tag_cited": True})
        out = refine_package(make_package(), PAPER, cfg)
        ids = [w["paper_id"] for w in out["candidate_works"]]
        self.assertIn("rw_self", ids)                                     # strict exclusion off
        self.assertIn("rw_cited", ids)
        self.assertTrue(next(w for w in out["candidate_works"] if w["paper_id"] == "rw_self")["is_submission"])


# ── Option 2: retrieval (claim queries, reference expansion, cutoff mode) ───

class ClaimLLM:
    def complete_json(self, system_prompt, user_prompt):
        if "claimed contributions" in user_prompt:
            return {"claims": [{"claim": "Sparse routing improves retrieval.", "query": "sparse routing retrieval"},
                               {"claim": "A new benchmark.", "query": "long context retrieval benchmark"}]}
        return {}


class RecordingProvider:
    name = "Fake"

    def __init__(self, papers):
        self.papers = papers
        self.queries = None

    def search(self, queries, limit=10):
        self.queries = queries
        return ProviderResult(provider=self.name, papers=list(self.papers), warnings=[], status="used")


class FakeExpander:
    def __init__(self, fail_with=None):
        self.matched = []
        self.fail_with = fail_with

    def _sleep_between_queries(self, index):
        pass

    def match_title(self, title):
        if self.fail_with:
            raise self.fail_with
        self.matched.append(title)
        return "s2_" + str(len(self.matched))

    def citing_papers(self, paper_id, limit):
        return [{"paperId": paper_id + "_c", "title": f"Citing paper on sparse retrieval routing {paper_id}",
                 "year": 2024, "publicationDate": "2024-03-01", "authors": [{"name": "Eve Fox"}],
                 "abstract": "Sparse routing for retrieval.", "externalIds": {}}]


def _pm(title, year=2023):
    return PaperMetadata(paper_id="", title=title, authors=["A B"], year=year, publication_date=f"{year}-02-01",
                         venue="V", abstract="retrieval routing", url="", sources=["Fake"],
                         matched_query_groups=["core_task"])


class TestRetrieval(unittest.TestCase):
    def test_claim_queries_are_added_as_extra_groups(self):
        provider = RecordingProvider([_pm("Some Prior Work on Routing")])
        pkg = build_related_work_rag(PAPER, api_key="t", config=RAGConfig(claim_queries=2),
                                     providers=[provider], llm_agent=ClaimLLM(), target_title=TITLE)
        groups = [q.group for q in provider.queries]
        self.assertIn("claim_1", groups)
        self.assertIn("claim_2", groups)
        self.assertIn("claim_1", pkg["query_generation"]["groups"])

    def test_claim_queries_off_by_default(self):
        provider = RecordingProvider([])
        build_related_work_rag(PAPER, api_key="t", config=RAGConfig(), providers=[provider],
                               llm_agent=ClaimLLM(), target_title=TITLE)
        self.assertFalse(any(q.group.startswith("claim_") for q in provider.queries))

    def test_reference_expansion_adds_citing_papers(self):
        result = expand_from_references(PAPER, provider=FakeExpander(), max_refs=2, citing_limit=5)
        self.assertEqual(result.status, "used")
        self.assertTrue(result.papers)
        self.assertTrue(all(GROUP in p.matched_query_groups for p in result.papers))

    def test_reference_expansion_stops_on_rate_limit(self):
        from rag.providers.base import ProviderHTTPError
        result = expand_from_references(PAPER, provider=FakeExpander(ProviderHTTPError("u", 429, "Too Many")))
        self.assertEqual(result.papers, [])
        self.assertTrue(any("stopped early" in w for w in result.warnings))

    def test_openalex_expander_and_fallback_from_semantic_scholar(self):
        from rag.providers.base import ProviderHTTPError
        from rag.reference_expansion import OpenAlexReferenceExpander, ReferenceExpander
        oa = OpenAlexReferenceExpander(tempfile.mkdtemp())
        responses = {
            "search": {"results": [{"id": "https://openalex.org/W1", "title": "Dense passage retrieval for "
                                                                       "open-domain question answering"}]},
            "cites": {"results": [{"id": "https://openalex.org/W9", "title": "Sparse Routing Cites Dense Retrieval",
                                   "publication_year": 2024, "publication_date": "2024-05-01",
                                   "authorships": [{"author": {"display_name": "Gia Ho"}}],
                                   "abstract_inverted_index": {"Sparse": [0], "routing": [1]},
                                   "ids": {"doi": "https://doi.org/10.1/x"},
                                   "locations": [{"landing_page_url": "https://arxiv.org/abs/2405.00001"}],
                                   "cited_by_count": 3}]},
        }
        with patch.object(OpenAlexReferenceExpander, "_json_get",
                          side_effect=lambda url, headers=None: responses["cites" if "cites:" in url else "search"]):
            self.assertEqual(oa.match_title("Dense Passage Retrieval for Open-Domain Question Answering"), "W1")
            item = oa.citing_papers("W1", 5)[0]
            self.assertEqual((item["abstract"], item["externalIds"]["ArXiv"], item["externalIds"]["DOI"]),
                             ("Sparse routing", "2405.00001", "10.1/x"))
            with patch.dict("os.environ", {"SEMANTIC_SCHOLAR_API_KEY": "k"}), \
                 patch.object(ReferenceExpander, "match_title", side_effect=ProviderHTTPError("u", 429, "Too Many")), \
                 patch.object(ReferenceExpander, "_sleep_between_queries"), \
                 patch.object(OpenAlexReferenceExpander, "_sleep_between_queries"):
                result = expand_from_references(PAPER, max_refs=2, citing_limit=5)
        self.assertTrue(any("switching to OpenAlex" in w for w in result.warnings))
        self.assertEqual(result.status, "used")
        self.assertEqual(result.papers[0].sources, ["OpenAlex (cites a manuscript reference)"])

    def test_openalex_key_added_at_request_time_only(self):
        from rag.providers.openalex import OpenAlexProvider, openalex_request_url
        with patch.dict("os.environ", {"OPENALEX_API_KEY": "secret123"}):
            self.assertEqual(openalex_request_url("https://api.openalex.org/works?search=x"),
                             "https://api.openalex.org/works?search=x&api_key=secret123")
            provider = OpenAlexProvider(tempfile.mkdtemp())
            seen = []

            class Resp:
                def __enter__(self): return self
                def __exit__(self, *a): return False
                def read(self): return b'{"results": []}'

            class Opener:
                def open(self, request, timeout=None):
                    seen.append(request.full_url)
                    return Resp()

            with patch("rag.providers.base._rag_opener", return_value=Opener()), \
                 patch.object(provider.cache, "set") as cache_set:
                provider._json_get("https://api.openalex.org/works?search=x")
        self.assertIn("api_key=secret123", seen[0])
        self.assertNotIn("secret123", cache_set.call_args.args[0])

    def test_expansion_in_package_and_interleave(self):
        provider = RecordingProvider([_pm("Search Result One About Routing"), _pm("Search Result Two About Routing")])
        pkg = build_related_work_rag(PAPER, api_key="t", config=RAGConfig(reference_expansion=True),
                                     providers=[provider], llm_agent=ClaimLLM(), target_title=TITLE,
                                     reference_expander=FakeExpander())
        self.assertEqual(pkg["provider_status"]["Reference expansion"]["status"], "used")
        a, b = [_pm("a")], [_pm("b"), _pm("c")]
        self.assertEqual([p.title for p in interleave(a, b)], ["a", "b", "c"])
        s3 = [_pm("s1"), _pm("s2"), _pm("s3")]
        self.assertEqual([p.title for p in interleave(s3, b)], ["s1", "s2", "b", "s3", "c"])

    def test_cutoff_mode_today_keeps_recent_work(self):
        recent = _pm("A Very Recent Paper About Routing", year=date.today().year)
        recent.publication_date = date.today().isoformat()
        for mode, expected in (("venue", 0), ("today", 1)):
            pkg = build_related_work_rag(PAPER, api_key="t",
                                         config=RAGConfig(cutoff_date="2024-12-31", cutoff_mode=mode),
                                         providers=[RecordingProvider([recent])], llm_agent=ClaimLLM(),
                                         target_title=TITLE)
            self.assertEqual(pkg["cutoff_report"]["cutoff_mode"], mode)
            self.assertEqual(pkg["cutoff_report"]["num_used"], expected)
        with self.assertRaises(ValueError):
            build_related_work_rag(PAPER, api_key="t", config=RAGConfig(cutoff_mode="bogus"),
                                   providers=[RecordingProvider([])], llm_agent=ClaimLLM())

    def test_rerank_is_batched_and_merged_by_score(self):
        import re as _re

        class BatchLLM:
            def __init__(self):
                self.rerank_calls = 0

            def complete_json(self, system_prompt, user_prompt):
                if "Candidate metadata:" not in user_prompt:
                    return {}
                self.rerank_calls += 1
                ids = _re.findall(r'"paper_id":\s*"(rw_\d+_[^"]+)"', user_prompt.split("Candidate metadata:")[1])
                return {"reranked_papers": [{"rank": i + 1, "paper_id": pid, "relevance_score": 0.3 + 0.02 * int(pid.split("_")[1]),
                                             "rationale": "r", "evidence_summary": "e"} for i, pid in enumerate(ids)],
                        "summary": ""}

        papers = [_pm(f"Distinct Routing Paper Number {i}") for i in range(26)]
        llm = BatchLLM()
        pkg = build_related_work_rag(PAPER, api_key="t", config=RAGConfig(rerank_top_k=30, provider_top_k=50),
                                     providers=[RecordingProvider(papers)], llm_agent=llm, target_title=TITLE)
        self.assertEqual(llm.rerank_calls, 3)                    # 26 candidates -> batches of 10, 10, 6
        self.assertEqual(pkg["reranking"]["source"], "llm", pkg["warnings"])
        scores = [r["relevance_score"] for r in sorted(pkg["reranking_results"], key=lambda r: r["rank"])]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(len(scores), 26)

    def test_strict_self_exclusion_at_build_time(self):
        provider = RecordingProvider([_pm("Sparse mixture routing for long-context retrieval."),
                                      _pm("Other Work About Routing")])
        pkg = build_related_work_rag(PAPER, api_key="t", config=RAGConfig(strict_self_exclusion=True),
                                     providers=[provider], llm_agent=ClaimLLM(), target_title=TITLE)
        titles = [m["title"] for m in pkg["paper_metadata"]]
        self.assertNotIn("Sparse mixture routing for long-context retrieval.", titles)


# ── Option 1: engaged prompt and validation ─────────────────────────────────

class TestEngagedPromptAndValidation(unittest.TestCase):
    def setUp(self):
        self.pkg = refine_package(make_package(), PAPER, rag_config_from_dict(IMPROVED))

    def test_engaged_block_lists_labels_and_cited_status(self):
        block = format_engaged_rag_block(self.pkg)
        self.assertIn("###RAG_EVIDENCE###", block)
        self.assertIn("###END_RAG_EVIDENCE###", block)
        self.assertIn("[R1] Dense Passage Retrieval", block)
        self.assertIn("Cited in the manuscript: yes", block)
        self.assertIn("Cited in the manuscript: no", block)
        self.assertIn("An abstract.", block)
        self.assertNotIn("An abstract.", format_engaged_rag_block(self.pkg, with_abstracts=False))
        self.assertIn("Submission title:", format_analyst_input(self.pkg))

    def test_review_field_validator(self):
        good = {"novelty_check": [{"claimed_contribution": "c", "closest_prior_work": "[r2]",
                                   "overlap": "Partial", "explanation": "e"}],
                "missing_prior_work": [{"work": "R3", "why_relevant": "w"}]}
        self.assertEqual(validate_rag_review_fields(good, {"R1", "R2", "R3"}, {"R2", "R3"}), [])
        self.assertEqual(good["novelty_check"][0]["closest_prior_work"], "R2")
        self.assertEqual(good["novelty_check"][0]["overlap"], "partial")
        bad = {"novelty_check": [], "missing_prior_work": [{"work": "R1", "why_relevant": "w"}]}
        errors = validate_rag_review_fields(bad, {"R1", "R2"}, {"R2"})
        self.assertTrue(any("novelty_check" in e for e in errors))
        self.assertTrue(any("already cited" in e for e in errors))
        self.assertTrue(validate_rag_review_fields({"novelty_check": [{"claimed_contribution": "c",
                        "closest_prior_work": "R9", "overlap": "none", "explanation": "e"}],
                        "missing_prior_work": []}, {"R1"}))

    def test_analyst_validator(self):
        report = {"contribution_overlap": [{"claimed_contribution": "c", "closest_prior_work": ["R2", "none"],
                                            "overlap": "substantial", "explanation": "e"}],
                  "possibly_missing_references": [{"work": "[R3]", "why_relevant": "w"}], "summary": "s"}
        self.assertEqual(validate_rw_analyst_report(report, {"R1", "R2", "R3"}, {"R2", "R3"}), [])
        self.assertEqual(report["contribution_overlap"][0]["closest_prior_work"], ["R2"])
        report["possibly_missing_references"] = [{"work": "R1", "why_relevant": "w"}]
        self.assertTrue(validate_rw_analyst_report(report, {"R1", "R2", "R3"}, {"R2", "R3"}))


# ── Workflow (options 1, 3 and 4 through mas_loop) ──────────────────────────

BASE_REVIEW = {"decision": "reject", "scores": {k: 3 for k in REVIEW_SCORE_KEYS},
               "strengths": ["Clear objective"], "weaknesses": ["Missing baseline"]}
RAG_FIELDS = {"novelty_check": [{"claimed_contribution": "Sparse routing", "closest_prior_work": "R2",
                                 "overlap": "partial", "explanation": "R2 also routes experts."}],
              "missing_prior_work": [{"work": "R2", "why_relevant": "Routes experts for retrieval."}]}
CONFERENCE = {venue: {"fit_score": 3} for venue in ("ICML", "ICLR", "NeurIPS")}
ANALYST = {"contribution_overlap": [{"claimed_contribution": "Sparse routing", "closest_prior_work": ["R2"],
                                     "overlap": "partial", "explanation": "Similar routing."}],
           "possibly_missing_references": [{"work": "R3", "why_relevant": "Mixture of retrievers."}],
           "summary": "Partial overlap with R2."}


class Replies:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def complete(self, persona, messages):
        self.requests.append((persona, copy.deepcopy(messages)))
        value = next(self.replies)
        return value if isinstance(value, str) else json.dumps(value)


def run_mas(clients, rag_config, n_iter=1):
    with patch("agents.create_llm_client", side_effect=clients), patch("agents.time.sleep"):
        return mas_loop.main(PAPER, n_iter=n_iter, reviewer_types=["reviewer_nopersona"],
                             run_citation_check=False, enable_rag=True,
                             precomputed_rag_package=make_package(), rag_config=rag_config)


class TestWorkflow(unittest.TestCase):
    def test_engaged_review_retries_bad_label_and_reports_missing_references(self):
        bad = {**BASE_REVIEW, "novelty_check": [{**RAG_FIELDS["novelty_check"][0], "closest_prior_work": "R9"}],
               "missing_prior_work": []}
        reviewer = Replies([bad, {**BASE_REVIEW, **RAG_FIELDS}])
        result = run_mas([reviewer, Replies([]), Replies([CONFERENCE])], IMPROVED)
        prompt = "\n".join(m["content"] for m in reviewer.requests[0][1])
        self.assertIn("[R1]", prompt)
        self.assertIn("novelty_check", prompt)
        self.assertEqual(len(reviewer.requests), 2)                    # one retry for the invalid label
        self.assertEqual(result["reviewers"][0]["novelty_check"][0]["closest_prior_work"], "R2")
        self.assertTrue(result["rag_mode"]["engaged"])
        self.assertEqual(result["rag_mode"]["uncited_labels"], ["R2", "R3"])
        missing = result["possibly_missing_references"]
        titles = [m["title"] for m in missing]
        self.assertNotIn("Dense Passage Retrieval for Open-Domain Question Answering", titles)   # cited
        self.assertNotIn("Sparse mixture routing for long context retrieval", titles)           # the submission
        top = missing[0]
        self.assertEqual(top["title"], "Routing Experts for Retrieval Augmented Language Models")
        self.assertEqual(set(top["sources"]), {"retrieval", "Reviewer 1 (Neutral)"})

    def test_analyst_runs_first_and_feeds_reviewer(self):
        analyst = Replies([ANALYST])
        reviewer = Replies([{**BASE_REVIEW, **RAG_FIELDS}])
        cfg = {**IMPROVED, "related_work_analyst": True}
        result = run_mas([reviewer, Replies([]), Replies([CONFERENCE]), analyst], cfg)
        self.assertEqual(len(analyst.requests), 1)
        prompt = "\n".join(m["content"] for m in reviewer.requests[0][1])
        self.assertIn("###RELATED_WORK_REPORT###", prompt)
        self.assertEqual(result["related_work_report"]["summary"], "Partial overlap with R2.")
        sources = {m["title"]: m["sources"] for m in result["possibly_missing_references"]}
        self.assertIn("analyst", sources["Mixture of Retrievers for Long Documents"])

    def test_no_work_passes_filter_keeps_engaged_format_and_skips_analyst(self):
        reviewer = Replies([{**BASE_REVIEW, "novelty_check": [{"claimed_contribution": "Sparse routing",
                             "closest_prior_work": "none", "overlap": "none", "explanation": "Nothing close."}],
                             "missing_prior_work": []}])
        cfg = {**IMPROVED, "related_work_analyst": True, "min_relevance": 0.995}
        with patch("agents.create_llm_client", side_effect=[reviewer, Replies([]), Replies([CONFERENCE])]), \
             patch("agents.time.sleep"):
            result = mas_loop.main(PAPER, n_iter=1, reviewer_types=["reviewer_nopersona"],
                                   run_citation_check=False, enable_rag=True,
                                   precomputed_rag_package=make_package(), rag_config=cfg)
        prompt = "\n".join(m["content"] for m in reviewer.requests[0][1])
        self.assertIn("No retrieved prior work passed the relevance filter", prompt)
        self.assertIn("novelty_check", prompt)
        self.assertNotIn("Dense Passage Retrieval", prompt)        # old block not used as a fallback
        self.assertTrue(result["rag_mode"]["engaged"])
        self.assertEqual(result["rag_mode"]["num_shown"], 0)
        self.assertTrue(result["rag_mode"]["analyst_skipped_no_works"])
        self.assertTrue(any("No retrieved work passed" in w for w in result["rag_warnings"]))

    def test_improvements_off_keeps_legacy_prompt(self):
        reviewer = Replies([BASE_REVIEW])
        result = run_mas([reviewer, Replies([]), Replies([CONFERENCE])], {})
        self.assertNotIn("rag_mode", result)
        self.assertNotIn("[R1]", "\n".join(m["content"] for m in reviewer.requests[0][1]))


class TestMissingReferencesMerge(unittest.TestCase):
    def test_only_uncited_non_self_and_sorted_by_sources(self):
        pkg = refine_package(make_package(), PAPER, rag_config_from_dict({**IMPROVED, "strict_self_exclusion": False}))
        reviews = [{"reviewer": "Reviewer 1", "missing_prior_work": [{"work": "r3", "why_relevant": "x"}]}]
        out = build_missing_references(pkg, {"possibly_missing_references": [{"work": "R3", "why_relevant": "y"}]},
                                       reviews, min_relevance=0.65)
        titles = [e["title"] for e in out]
        self.assertNotIn("Sparse mixture routing for long context retrieval", titles)
        self.assertNotIn("Dense Passage Retrieval for Open-Domain Question Answering", titles)
        shown = {w["label"]: w["title"] for w in pkg["shown_works"]}
        self.assertEqual(out[0]["title"], shown["R3"])
        self.assertGreaterEqual(len(out[0]["sources"]), 2)


def import_experiment():
    """eval/experiment.py imports the PDF converter (torch); stub it so the test stays offline."""
    import types
    from unittest.mock import MagicMock
    if "experiment" in sys.modules:
        return sys.modules["experiment"]
    stub = types.ModuleType("doc_preprocess")
    stub.doc_preprocess = MagicMock(return_value="stub.md")
    stub.load_or_create_markdown = MagicMock(return_value="stub markdown")
    with patch.dict(sys.modules, {"doc_preprocess": sys.modules.get("doc_preprocess") or stub}):
        import experiment
    return experiment


# ── Option 5: experiment plumbing and evaluation scripts ────────────────────

class TestExperimentAndEval(unittest.TestCase):
    def test_conditions_and_presets(self):
        experiment = import_experiment()
        conds = {c["id"]: c for c in experiment.CONDITIONS}
        self.assertEqual(conds["8"]["rag_overrides"], "improved")
        self.assertEqual(conds["9"]["rag_overrides"], "improved_analyst")
        merged = experiment.condition_rag_config(conds["9"], {"cutoff_date": "2024-12-31"})
        self.assertTrue(merged["related_work_analyst"])
        self.assertEqual(merged["cutoff_date"], "2024-12-31")
        self.assertNotIn("prompt_style", experiment.condition_rag_config(conds["7"], {}))

    def test_provider_is_passed_through(self):
        experiment = import_experiment()
        cond = {c["id"]: c for c in experiment.CONDITIONS}["8"]
        with patch.object(experiment, "mas_main", return_value={}) as fake:
            experiment.run_condition("paper", "", cond, "key", model="m", provider="claude",
                                     precomputed_rag_package={"x": 1})
        kwargs = fake.call_args.kwargs
        self.assertEqual(kwargs["provider"], "claude")
        self.assertEqual(kwargs["rag_config"]["prompt_style"], "engaged")
        self.assertEqual(kwargs["precomputed_rag_package"], {"x": 1})

    def test_builder_does_not_save_packages_starved_by_openalex_429(self):
        import build_rag_packages as b
        ok = {"warnings": ["arXiv query failed for claim_1: HTTP 429 Unknown Error"],
              "provider_status": {"OpenAlex": {"status": "used", "warnings": []}}}
        starved = {"warnings": ["OpenAlex query failed for same_problem: HTTP 429 Too Many Requests"],
                   "provider_status": {"OpenAlex": {"status": "failed", "warnings": []}}}
        self.assertFalse(b.openalex_refused(ok))
        self.assertTrue(b.openalex_refused(starved))

    def test_first_public_dates_parsing(self):
        import first_public_dates as fpd
        feed = """<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2402.01234v3</id>
        <published>2024-02-02T10:00:00Z</published><title>Sparse Mixture Routing for
        Long-Context Retrieval</title></entry><entry><id>http://arxiv.org/abs/2001.00001v1</id>
        <published>2020-01-01T00:00:00Z</published><title>Something Else</title></entry></feed>"""
        hit = fpd.parse_arxiv_feed(feed, TITLE)
        self.assertEqual(hit["arxiv_id"], "2402.01234")
        self.assertEqual(hit["published"], "2024-02-02")
        self.assertEqual(fpd.parse_arxiv_feed(feed, "Totally different title words"), {})
        notes = {"notes": [{"cdate": 1727740800000, "odate": 1728000000000}]}
        self.assertEqual(fpd.parse_openreview_notes(notes)["field"], "odate")
        self.assertEqual(fpd.forum_id("https://openreview.net/forum?id=abc123"), "abc123")
        rec = {"arxiv": {"published": "2024-02-02"}, "openreview": {"openreview_date": "2024-10-04"}}
        self.assertEqual(fpd.first_public(rec), "2024-02-02")
        report = fpd.subset_report({"a": {"first_public": "2025-06-01"}, "b": {"first_public": "2024-01-01"},
                                    "c": {"first_public": ""}}, "2025-03-31")
        self.assertEqual((report["post_cutoff"], report["pre_cutoff"], report["unknown"]), (["a"], ["b"], ["c"]))

    def test_rag_eval_on_synthetic_runs(self):
        import rag_eval
        pkg = refine_package(make_package(), PAPER, rag_config_from_dict(IMPROVED))
        dataset, control, rag = {}, {}, {}
        for i in range(6):
            pid = f"p{i}"
            accept = i % 2 == 0
            dataset[pid] = {"conference": "ICLR", "score": 8 if accept else 3,
                            "accept_or_not": "accept" if accept else "reject",
                            "reviews": [{"weaknesses": ["Missing comparison with prior work on routing."]
                                         if i < 3 else ["Proofs are unclear."]}]}
            review = {**BASE_REVIEW, "decision": "accept" if accept else "reject",
                      "scores": {k: (4 if accept else 2) for k in REVIEW_SCORE_KEYS}}
            control[pid] = {"reviewers": [dict(review)]}
            rag[pid] = {"reviewers": [{**review, **copy.deepcopy(RAG_FIELDS)}], "rag_package": pkg}
        report = rag_eval.evaluate(dataset, control, {"8": rag}, set(), use_embeddings=False, n_boot=50)
        c = report["conditions"]["8"]
        self.assertEqual(c["papers"], 6)
        self.assertEqual(c["usage"]["novelty_names_shown_work"], 1.0)
        self.assertEqual(c["usage"]["named_works_not_cited"], 6)
        d = c["detection_vs_human"]
        self.assertEqual((d["tp"], d["fp"], d["fn"], d["tn"]), (3, 3, 0, 0))
        self.assertEqual(c["citation_validity"]["flagged_cited_or_self"], 0)
        self.assertEqual(c["paired_vs_control"]["accuracy_rag"]["value"], 1.0)
        self.assertEqual(c["paired_vs_control"]["decision_agreement_with_control"], 1.0)
        sub = rag_eval.evaluate(dataset, control, {"8": rag}, set(), subset=["p0", "p1"],
                                use_embeddings=False, n_boot=10)
        self.assertEqual(sub["conditions"]["8"]["papers"], 2)


class TestWebapp(unittest.TestCase):
    def test_missing_references_panel_present(self):
        html = (ROOT / "webapp" / "templates" / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="sec-missing-refs"', html)
        self.assertIn("renderMissingReferences(data.possibly_missing_references)", html)
        from config import DEFAULT_RAG_CONFIG
        self.assertTrue(DEFAULT_RAG_CONFIG["missing_references_report"])


if __name__ == "__main__":
    unittest.main()
