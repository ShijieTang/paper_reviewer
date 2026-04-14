"""
Unit tests for eval/run_review.py and eval/experiment.py.

Covers:
  - normalize_topic(): exact match, case-insensitive match, unknown → "Others"
  - run_paper(): topic is normalized before being forwarded to mas_main
  - run_experiment(): topic is normalized before being forwarded to mas_main
"""

import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

# Stub out heavy dependencies that are not available in the test environment
# before any project module is imported.
for _mod in ("marker", "marker.converters", "marker.converters.pdf",
             "marker.models", "marker.output"):
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

_doc_preprocess_stub = types.ModuleType("doc_preprocess")
_doc_preprocess_stub.doc_preprocess = MagicMock(return_value="data/md/stub.md")
sys.modules["doc_preprocess"] = _doc_preprocess_stub

_mas_loop_stub = types.ModuleType("mas_loop")
_mas_loop_stub.main = MagicMock(return_value={"reviewers": [], "conference": {}, "citations": {}})
sys.modules["mas_loop"] = _mas_loop_stub

from eval.run_review import normalize_topic as rr_normalize, run_paper  # noqa: E402
from eval.experiment import normalize_topic as ex_normalize, run_experiment  # noqa: E402


# ── normalize_topic (run_review) ──────────────────────────────────────────────

class TestRunReviewNormalizeTopic(unittest.TestCase):
    def test_exact_match_returned_unchanged(self):
        for topic in ["Machine Learning", "Deep Learning", "Generative Models",
                      "Transfer Learning", "Computer Vision", "NLP",
                      "AI for Science", "Others"]:
            with self.subTest(topic=topic):
                self.assertEqual(rr_normalize(topic), topic)

    def test_case_insensitive_match(self):
        self.assertEqual(rr_normalize("deep learning"),      "Deep Learning")
        self.assertEqual(rr_normalize("machine learning"),   "Machine Learning")
        self.assertEqual(rr_normalize("COMPUTER VISION"),    "Computer Vision")
        self.assertEqual(rr_normalize("nlp"),                "NLP")
        self.assertEqual(rr_normalize("generative models"),  "Generative Models")

    def test_unknown_topic_returns_others(self):
        self.assertEqual(rr_normalize("Learning Theory"), "Others")
        self.assertEqual(rr_normalize("Optimization"),    "Others")
        self.assertEqual(rr_normalize(""),                "Others")
        self.assertEqual(rr_normalize("   "),             "Others")
        self.assertEqual(rr_normalize("Robotics"),        "Others")

    def test_whitespace_stripped_before_matching(self):
        self.assertEqual(rr_normalize("  NLP  "),          "NLP")
        self.assertEqual(rr_normalize("  deep learning "), "Deep Learning")


# ── normalize_topic (experiment) ──────────────────────────────────────────────

class TestExperimentNormalizeTopic(unittest.TestCase):
    def test_exact_match_returned_unchanged(self):
        for topic in ["Machine Learning", "Deep Learning", "Generative Models",
                      "Transfer Learning", "Computer Vision", "NLP",
                      "AI for Science", "Others"]:
            with self.subTest(topic=topic):
                self.assertEqual(ex_normalize(topic), topic)

    def test_case_insensitive_match(self):
        self.assertEqual(ex_normalize("Deep learning"),    "Deep Learning")
        self.assertEqual(ex_normalize("machine learning"), "Machine Learning")

    def test_unknown_topic_returns_others(self):
        self.assertEqual(ex_normalize("Learning Theory"), "Others")
        self.assertEqual(ex_normalize("Optimization"),    "Others")
        self.assertEqual(ex_normalize(""),                "Others")


# ── run_paper topic normalization ─────────────────────────────────────────────

class TestRunPaperTopicNormalization(unittest.TestCase):
    def _make_meta(self, topic: str) -> dict:
        return {
            "paper_id":  "test_001",
            "paper_dir": "data/pdf/test_001.pdf",
            "topic":     topic,
        }

    def _run(self, meta: dict, topic_override=None):
        fake_result = {"reviewers": [], "conference": {}, "citations": {}}
        with patch("eval.run_review.pdf_to_markdown", return_value="paper text"), \
             patch("eval.run_review.mas_main", return_value=fake_result) as mock_mas, \
             tempfile.TemporaryDirectory() as tmpdir:
            run_paper(meta, agents=["reviewer_a"], api_key="key",
                      output_dir=tmpdir, n_iter=1,
                      topic_override=topic_override)
            return mock_mas.call_args

    def _topic_used(self, call_args):
        # mas_main is called with keyword args; topic is always keyword
        return call_args[1]["topic"]

    def test_unknown_topic_passed_as_others(self):
        self.assertEqual(self._topic_used(self._run(self._make_meta("Learning Theory"))), "Others")

    def test_case_mismatch_corrected(self):
        self.assertEqual(self._topic_used(self._run(self._make_meta("Deep learning"))), "Deep Learning")

    def test_valid_topic_passed_through(self):
        self.assertEqual(self._topic_used(self._run(self._make_meta("NLP"))), "NLP")

    def test_topic_override_normalized(self):
        self.assertEqual(self._topic_used(self._run(self._make_meta("NLP"), topic_override="Optimization")), "Others")

    def test_topic_override_takes_precedence(self):
        self.assertEqual(self._topic_used(self._run(self._make_meta("Learning Theory"), topic_override="NLP")), "NLP")


# ── run_experiment topic normalization ────────────────────────────────────────

class TestRunExperimentTopicNormalization(unittest.TestCase):
    def _make_papers(self, topic: str) -> list:
        return [{
            "paper_id":      "test_001",
            "paper_dir":     "data/pdf/test_001.pdf",
            "topic":         topic,
            "conference":    "ICLR",
            "accept_or_not": "accept",
            "score":         5.0,
        }]

    def _run(self, papers: list):
        fake_result = {"reviewers": [], "conference": {}, "citations": {}}
        with patch("eval.experiment.pdf_to_markdown", return_value="paper text"), \
             patch("eval.experiment.mas_main", return_value=fake_result) as mock_mas, \
             tempfile.TemporaryDirectory() as tmpdir:
            run_experiment(papers, api_key="key", output_dir=tmpdir)
            return mock_mas

    def test_unknown_topic_becomes_others(self):
        mock_mas = self._run(self._make_papers("Optimization"))
        for c in mock_mas.call_args_list:
            self.assertEqual(c[1]["topic"], "Others")

    def test_case_mismatch_corrected(self):
        mock_mas = self._run(self._make_papers("machine learning"))
        for c in mock_mas.call_args_list:
            self.assertEqual(c[1]["topic"], "Machine Learning")

    def test_valid_topic_unchanged(self):
        mock_mas = self._run(self._make_papers("Computer Vision"))
        for c in mock_mas.call_args_list:
            self.assertEqual(c[1]["topic"], "Computer Vision")

    def test_both_conditions_run_per_paper(self):
        """Each paper should trigger both condition A and B."""
        mock_mas = self._run(self._make_papers("NLP"))
        self.assertEqual(mock_mas.call_count, 2)


if __name__ == "__main__":
    unittest.main()
