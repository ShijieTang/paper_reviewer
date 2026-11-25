"""Offline tests for the standalone related-work check (related_work_check.py)."""
import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from related_work_check import STANDALONE_RAG_CONFIG, run_related_work_check  # noqa: E402
from test_rag_improvements import PAPER, Replies, make_package  # noqa: E402

# With the standalone settings, the example package yields (submission removed, relevance >= 0.5):
# R1 Dense Passage Retrieval (cited), R2 Routing Experts (0.85), R3 Mixture of Retrievers (0.70),
# R4 Long-Context Benchmarks (0.60); R2-R4 are not cited.
REPORT = {"contribution_overlap": [{"claimed_contribution": "Sparse routing", "closest_prior_work": ["R2"],
                                    "overlap": "partial", "explanation": "R2 also routes experts."}],
          "possibly_missing_references": [{"work": "R2", "why_relevant": "Routes experts for retrieval."},
                                          {"work": "R3", "why_relevant": "Background only."}],
          "summary": "Partial overlap with R2."}


def run_check(replies, **kwargs):
    events = []
    with patch("agents.create_llm_client", side_effect=replies), patch("agents.time.sleep"):
        result = run_related_work_check(PAPER, provider="openrouter", model="m", api_key="k",
                                        precomputed_rag_package=make_package(), on_event=events.append,
                                        **kwargs)
    return result, events


class TestRelatedWorkCheck(unittest.TestCase):
    def test_settings_match_condition_11_without_review(self):
        self.assertEqual(STANDALONE_RAG_CONFIG["min_uncited_shown"], 0)
        self.assertTrue(STANDALONE_RAG_CONFIG["missing_requires_overlap"])
        self.assertTrue(STANDALONE_RAG_CONFIG["related_work_analyst"])
        self.assertEqual(STANDALONE_RAG_CONFIG["cutoff_mode"], "today")

    def test_report_keeps_only_overlap_backed_missing_references(self):
        analyst = Replies([REPORT])
        result, events = run_check([analyst])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(analyst.requests), 1)
        persona, messages = analyst.requests[0]
        prompt = "\n".join(m["content"] for m in messages)
        self.assertIn("[R2] Routing Experts", prompt)
        self.assertIn("only if you named it in contribution_overlap", prompt)
        self.assertNotIn("Our router selects experts", prompt)     # short context: not the full paper
        self.assertEqual([w["label"] for w in result["shown_works"]], ["R1", "R2", "R3", "R4"])
        missing = [m["title"] for m in result["possibly_missing_references"]]
        self.assertEqual(missing, ["Routing Experts for Retrieval Augmented Language Models"])
        # R3 was listed by the analyst without an overlap, so it is dropped. Every relevant work was
        # shown to the analyst, so none is repeated in the retrieval-only list.
        self.assertEqual(result["related_work_report"]["possibly_missing_references_dropped_unlinked"], 1)
        self.assertEqual(result["other_relevant_uncited"], [])
        self.assertNotIn("Dense Passage Retrieval for Open-Domain Question Answering", missing)
        self.assertTrue(any("analyst" in e.lower() for e in events))
        json.dumps(result)                                          # serializable for the web app

    def test_retrieval_only_list_excludes_works_the_analyst_saw(self):
        # max_shown 2: R1 (cited) and R2 are shown; R3 (0.70, uncited) is not shown and is listed by
        # relevance only; R4 (0.60) is below the 0.65 threshold.
        report = {**REPORT, "possibly_missing_references": [{"work": "R2", "why_relevant": "Routes experts."}]}
        result, _ = run_check([Replies([report])], rag_config={"max_shown": 2})
        self.assertEqual([m["title"] for m in result["possibly_missing_references"]],
                         ["Routing Experts for Retrieval Augmented Language Models"])
        self.assertEqual([m["title"] for m in result["other_relevant_uncited"]],
                         ["Mixture of Retrievers for Long Documents"])

    def test_invalid_label_is_retried(self):
        bad = {**REPORT, "contribution_overlap": [{**REPORT["contribution_overlap"][0], "closest_prior_work": ["R9"]}]}
        analyst = Replies([bad, REPORT])
        result, _ = run_check([analyst])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(analyst.requests), 2)

    def test_no_relevant_work_skips_the_analyst(self):
        result, _ = run_check([], rag_config={"min_relevance": 0.995})
        self.assertEqual(result["status"], "no_relevant_work")
        self.assertEqual(result["shown_works"], [])
        self.assertEqual(result["possibly_missing_references"], [])

    def test_analyst_failure_is_reported(self):
        result, _ = run_check([Replies(["not json"] * 10)])
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["errors"])


class TestWebappRoute(unittest.TestCase):
    def test_route_runs_check_as_job(self):
        import webapp.app as webapp
        client = webapp.app.test_client()
        with patch.object(webapp, "current_md_name", None):
            self.assertEqual(client.post("/api/related-work-check", json={"api_key": "k", "provider": "openrouter"})
                             .status_code, 400)
        fake = {"status": "complete", "shown_works": [], "related_work_report": {"summary": "s"},
                "possibly_missing_references": [], "other_relevant_uncited": [], "warnings": [], "errors": []}
        with patch.object(webapp, "current_md_name", "paper.md"), \
             patch.object(webapp.Path, "read_text", return_value=PAPER), \
             patch.object(webapp.Path, "write_text"), patch.object(webapp.Path, "mkdir"), \
             patch("related_work_check.run_related_work_check", return_value=fake), \
             patch.object(webapp, "validate_api_key_for_provider", return_value=None):
            res = client.post("/api/related-work-check", json={"api_key": "k", "provider": "openrouter"})
            self.assertEqual(res.status_code, 200)
            job_id = res.get_json()["job_id"]
            for _ in range(50):
                if job_id in webapp._results:
                    break
                time.sleep(0.05)
        self.assertEqual(webapp._results[job_id]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
