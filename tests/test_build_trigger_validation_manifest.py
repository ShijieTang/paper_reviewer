import copy
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_trigger_validation_manifest import (
    build_manifests,
    hard_invalid_reasons,
    merge_historical_exclusions,
    normalize_title,
)


def record(conference, label, index):
    paper_id = f"{conference.lower()}_{label}_{index:03d}"
    return paper_id, {
        "title": f"{conference} {label} Unique Paper {index}",
        "paper_dir": f"data/openreview_pdf/{paper_id}.pdf",
        "paper_url": f"https://example.test/{paper_id}",
        "conference": conference,
        "year": 2025,
        "topic": "Others",
        "accept_or_not": label,
        "collection_decision_category": label,
        "score": 6 if label == "accept" else 3,
        "reviews": [
            {
                "reviewer_id": "r1",
                "strengths": ["clear"],
                "weaknesses": ["limited"],
                "rating": 6,
                "decision": label,
                "rebuttal": "response",
            }
        ],
        "ground_truth_extra": "private",
    }


def synthetic_source(per_cell=8):
    data = {}
    for conference in ("ICLR", "ICML", "NeurIPS"):
        for label in ("accept", "reject"):
            for index in range(per_cell):
                paper_id, value = record(conference, label, index)
                data[paper_id] = value
    return data


def test_title_normalization_handles_punctuation_and_smart_apostrophe():
    assert normalize_title("Why Models Don't Memorize") == normalize_title("WHY MODELS DON’T MEMORIZE")
    assert normalize_title("Graph—Transformer:  A Test") == normalize_title("graph transformer a test")


def test_hard_invalid_rejects_only_double_empty_review_segments():
    _, value = record("ICLR", "accept", 1)
    one_sided = copy.deepcopy(value)
    one_sided["reviews"][0]["strengths"] = []
    assert hard_invalid_reasons(one_sided) == []
    both_empty = copy.deepcopy(value)
    both_empty["reviews"][0]["strengths"] = []
    both_empty["reviews"][0]["weaknesses"] = []
    assert "review_0_both_segments_empty" in hard_invalid_reasons(both_empty)


def test_selection_is_stratified_deterministic_and_model_input_blinded():
    source = synthetic_source()
    dev = {"old": {"title": source["iclr_accept_000"]["title"]}}
    invalid_id = "icml_reject_000"
    source[invalid_id]["reviews"][0]["strengths"] = []
    source[invalid_id]["reviews"][0]["weaknesses"] = []

    first = build_manifests(source, dev, per_stratum=2, seed=11766)
    reversed_source = dict(reversed(list(source.items())))
    second = build_manifests(reversed_source, dev, per_stratum=2, seed=11766)
    run_manifest, evaluation_manifest, labels, metadata = first

    assert first == second
    assert len(run_manifest) == 12
    assert list(run_manifest) == [f"trigger_val_{index:03d}" for index in range(1, 13)]
    assert set(metadata["selected_counts"].values()) == {2}
    assert metadata["hard_invalid_count"] == 1
    assert metadata["development_overlap_count"] == 1
    assert invalid_id not in {item["source_paper_id"] for item in metadata["opaque_mapping"].values()}

    forbidden = {
        "accept_or_not",
        "score",
        "reviews",
        "collection_decision_category",
        "ground_truth_extra",
        "source_paper_id",
    }
    assert all(not (set(item) & forbidden) for item in run_manifest.values())
    assert all("source_paper_id" in item and "reviews" in item for item in evaluation_manifest.values())
    assert all(set(item) == {"accept_or_not", "score", "conference"} for item in labels.values())


def test_real_264_inventory_produces_expected_clean_holdout():
    root = Path(__file__).resolve().parents[1]
    source = json.loads(
        (root / "eval/openreview_2025_264_qwen.balanced_by_conference_under10mb.json").read_text(encoding="utf-8")
    )
    development = json.loads((root / "eval/papers.json").read_text(encoding="utf-8"))
    historical = json.loads(
        subprocess.check_output(
            [
                "git",
                "show",
                "85138df:eval/exp_results_60/experiment_summary_2607222040.json",
            ],
            cwd=root,
        )
    )
    exclusions, historical_present, historical_missing = merge_historical_exclusions(
        source,
        development,
        [historical],
    )
    run_manifest, _, _, metadata = build_manifests(source, exclusions, per_stratum=6, seed=11766)
    assert len(run_manifest) == 36
    assert len(historical_present) == 50
    assert len(historical_missing) == 10
    assert metadata["hard_invalid_count"] == 7
    assert metadata["development_overlap_count"] == 61
    assert metadata["remaining_eligible_after_selection"] == 160
    assert metadata["selected_counts"] == {
        "ICLR:accept": 6,
        "ICLR:reject": 6,
        "ICML:accept": 6,
        "ICML:reject": 6,
        "NeurIPS:accept": 6,
        "NeurIPS:reject": 6,
    }
