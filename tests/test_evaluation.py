import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.evaluation import (
    _collect_sw_from_gt,
    run_evaluation,
)


class EvaluationTests(unittest.TestCase):
    def test_collect_sw_from_gt_supports_nested_and_legacy_formats(self):
        nested = {
            "reviews": [
                {"strengths": ["clear writing"], "weaknesses": ["small dataset"]},
                {"strengths": ["strong ablation"], "weaknesses": ["missing baseline"]},
            ]
        }
        legacy = {
            "strengths": ["clear writing"],
            "weaknesses": ["small dataset"],
        }

        self.assertEqual(
            _collect_sw_from_gt(nested),
            (
                ["clear writing", "strong ablation"],
                ["small dataset", "missing baseline"],
            ),
        )
        self.assertEqual(
            _collect_sw_from_gt(legacy),
            (["clear writing"], ["small dataset"]),
        )

    def test_run_evaluation_aggregates_results_and_writes_output(self):
        papers = {
            "paper-1": {
                "title": "Test Paper",
                "conference": "ICLR",
                "accept_or_not": "Accept",
                "score": 4.0,
                "reviews": [
                    {
                        "strengths": ["clear motivation"],
                        "weaknesses": ["limited experiments"],
                        "rating": 4,
                        "decision": "Accept",
                    }
                ],
            }
        }
        openreviewer = {
            "papers": [
                {
                    "paper_id": "paper-1",
                    "accept_or_not": "Accept",
                    "score": 4.5,
                    "reviews": [
                        {
                            "strengths": ["clear motivation"],
                            "weaknesses": ["limited experiments"],
                        }
                    ],
                }
            ]
        }
        paperreviewer = {
            "papers": [
                {
                    "paper_id": "paper-1",
                    "accept_or_not": "Reject",
                    "score": 2.0,
                    "reviews": [
                        {
                            "strengths": ["strong novelty"],
                            "weaknesses": ["unclear setup"],
                        }
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            papers_path = tmpdir_path / "papers.json"
            openreviewer_path = tmpdir_path / "openreviewer.json"
            paperreviewer_path = tmpdir_path / "paperreviewer.json"
            output_dir = tmpdir_path / "results"

            for path, payload in [
                (papers_path, papers),
                (openreviewer_path, openreviewer),
                (paperreviewer_path, paperreviewer),
            ]:
                path.write_text(json.dumps(payload), encoding="utf-8")

            src_values = [
                {"strengths": 0.9, "weaknesses": 0.8, "overall": 0.85},
                {"strengths": 0.4, "weaknesses": 0.3, "overall": 0.35},
            ]

            with patch("eval.evaluation.load_model", return_value="mock-model") as mock_load_model:
                with patch("eval.evaluation.compute_src_both", side_effect=src_values) as mock_compute_src:
                    results = run_evaluation(
                        papers_path=str(papers_path),
                        openreviewer_path=str(openreviewer_path),
                        paperreviewer_path=str(paperreviewer_path),
                        output_dir=str(output_dir),
                        embed_model_name="mock-embed-model",
                    )

            mock_load_model.assert_called_once_with("mock-embed-model")
            self.assertEqual(mock_compute_src.call_count, 2)

            paper_results = results["papers"][0]["systems"]
            self.assertEqual(results["embed_model"], "mock-embed-model")
            self.assertTrue(output_dir.exists())

            self.assertEqual(paper_results["openreviewer"]["decision_match"], True)
            self.assertEqual(paper_results["paperreviewer"]["decision_match"], False)
            self.assertNotIn("score_mae", paper_results["openreviewer"])
            self.assertNotIn("score_mae", paper_results["paperreviewer"])

            # conference_check is not evaluated for openreviewer / paperreviewer
            self.assertEqual(
                results["aggregate"]["openreviewer"],
                {
                    "n_papers": 1,
                    "decision_accuracy": 1.0,
                    "conference_check_accuracy": None,
                    "src_strengths_mean": 0.9,
                    "src_weaknesses_mean": 0.8,
                    "src_overall_mean": 0.85,
                },
            )
            self.assertEqual(
                results["aggregate"]["paperreviewer"],
                {
                    "n_papers": 1,
                    "decision_accuracy": 0.0,
                    "conference_check_accuracy": None,
                    "src_strengths_mean": 0.4,
                    "src_weaknesses_mean": 0.3,
                    "src_overall_mean": 0.35,
                },
            )
            self.assertNotIn("score_mae_mean", results["aggregate"]["openreviewer"])
            self.assertNotIn("score_mae_mean", results["aggregate"]["paperreviewer"])

            output_files = list(output_dir.glob("eval_results_*.json"))
            self.assertEqual(len(output_files), 1)
            saved_results = json.loads(output_files[0].read_text(encoding="utf-8"))
            self.assertEqual(saved_results["aggregate"], results["aggregate"])


if __name__ == "__main__":
    unittest.main()
