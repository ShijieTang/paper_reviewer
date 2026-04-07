import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.evaluation import (
    _collect_sw_from_gt,
    _majority_decision,
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

    def test_majority_decision_uses_last_reviewer_on_tie(self):
        reviewers = [
            {"decision": "Accept"},
            {"decision": "Reject"},
        ]

        self.assertEqual(_majority_decision(reviewers), "reject")

    def test_run_evaluation_aggregates_results_and_writes_output(self):
        papers = {
            "papers": [
                {
                    "paper_id": "paper-1",
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
            ]
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
        exp_summary = {
            "papers": [
                {
                    "paper_id": "paper-1",
                    "conditions": {
                        "A": {
                            "result": {
                                "reviewers": [
                                    {
                                        "decision": "Accept",
                                        "strengths": ["good method"],
                                        "weaknesses": ["weak comparison"],
                                        "scores": {"novelty": 4, "soundness": 3},
                                    },
                                    {
                                        "decision": "Reject",
                                        "strengths": ["good clarity"],
                                        "weaknesses": ["small-scale eval"],
                                        "scores": {"novelty": 2, "soundness": 1},
                                    },
                                ]
                            }
                        }
                    },
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            papers_path = tmpdir_path / "papers.json"
            openreviewer_path = tmpdir_path / "openreviewer.json"
            paperreviewer_path = tmpdir_path / "paperreviewer.json"
            exp_summary_path = tmpdir_path / "experiment_summary.json"
            output_dir = tmpdir_path / "results"

            for path, payload in [
                (papers_path, papers),
                (openreviewer_path, openreviewer),
                (paperreviewer_path, paperreviewer),
                (exp_summary_path, exp_summary),
            ]:
                path.write_text(json.dumps(payload), encoding="utf-8")

            src_values = [
                {"strengths": 0.9, "weaknesses": 0.8, "overall": 0.85},
                {"strengths": 0.4, "weaknesses": 0.3, "overall": 0.35},
                {"strengths": 0.7, "weaknesses": 0.6, "overall": 0.65},
            ]

            with patch("evaluation.evaluation.load_model", return_value="mock-model") as mock_load_model:
                with patch("evaluation.evaluation.compute_src_both", side_effect=src_values) as mock_compute_src:
                    results = run_evaluation(
                        papers_path=str(papers_path),
                        openreviewer_path=str(openreviewer_path),
                        paperreviewer_path=str(paperreviewer_path),
                        exp_summary_path=str(exp_summary_path),
                        output_dir=str(output_dir),
                        embed_model_name="mock-embed-model",
                    )

            mock_load_model.assert_called_once_with("mock-embed-model")
            self.assertEqual(mock_compute_src.call_count, 3)

            paper_results = results["papers"][0]["systems"]
            self.assertEqual(results["embed_model"], "mock-embed-model")
            self.assertTrue(output_dir.exists())

            self.assertEqual(paper_results["openreviewer"]["decision_match"], True)
            self.assertEqual(paper_results["openreviewer"]["score_mae"], 0.5)
            self.assertEqual(paper_results["paperreviewer"]["decision_match"], False)
            self.assertEqual(paper_results["paperreviewer"]["score_mae"], 2.0)
            self.assertEqual(paper_results["exp_cond_A"]["decision"], "reject")
            self.assertEqual(paper_results["exp_cond_A"]["score"], 2.5)
            self.assertEqual(paper_results["exp_cond_A"]["score_mae"], 1.5)

            self.assertEqual(
                results["aggregate"]["openreviewer"],
                {
                    "n_papers": 1,
                    "decision_accuracy": 1.0,
                    "score_mae_mean": 0.5,
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
                    "score_mae_mean": 2.0,
                    "src_strengths_mean": 0.4,
                    "src_weaknesses_mean": 0.3,
                    "src_overall_mean": 0.35,
                },
            )
            self.assertEqual(
                results["aggregate"]["exp_cond_A"],
                {
                    "n_papers": 1,
                    "decision_accuracy": 0.0,
                    "score_mae_mean": 1.5,
                    "src_strengths_mean": 0.7,
                    "src_weaknesses_mean": 0.6,
                    "src_overall_mean": 0.65,
                },
            )

            output_files = list(output_dir.glob("eval_results_*.json"))
            self.assertEqual(len(output_files), 1)
            saved_results = json.loads(output_files[0].read_text(encoding="utf-8"))
            self.assertEqual(saved_results["aggregate"], results["aggregate"])


if __name__ == "__main__":
    unittest.main()
