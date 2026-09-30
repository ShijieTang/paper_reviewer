import json
from pathlib import Path
import subprocess

import pytest

from scripts.archive_experiment_artifacts import apply_plan, build_plan, verify_archive


def _write(repo, name, content):
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _write(repo, "eval/exp_results/old.json", '{"old": true}')
    _write(repo, "README.md", "# Experiment conclusions")
    _write(repo, ".env", "private")
    subprocess.run(["git", "-C", str(repo), "add", "README.md", "eval/exp_results/old.json"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Archive Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
    _write(repo, "eval/advanced_exp_results/run.json", '{"new": true}')
    _write(repo, "eval/trigger_validation_36.labels.json", '{"label": "accept"}')
    return repo


def test_preserves_tracked_moves_untracked_and_resumes(repository):
    archive = repository / "experiment_artifacts"
    plan = build_plan(repository, "HEAD", [], include_markdown=False)
    catalog = apply_plan(repository, archive, plan)
    assert (repository / "eval/exp_results/old.json").exists()
    assert not (repository / "eval/advanced_exp_results/run.json").exists()
    assert (archive / "local/eval/advanced_exp_results/run.json").read_text() == '{"new": true}'
    assert (archive / "github/main/README.md").read_text() == "# Experiment conclusions"
    assert (archive / "private/local/eval/trigger_validation_36.labels.json").exists()
    assert (repository / "eval/trigger_validation_36.labels.json").exists()
    assert all(".env" not in e["source"] for e in catalog["entries"])
    apply_plan(repository, archive, plan)
    verify_archive(archive, plan)


def test_collision_preflight_prevents_any_moves(repository):
    archive = repository / "experiment_artifacts"
    plan = build_plan(repository, "HEAD", [], include_markdown=False)
    _write(archive, "github/main/README.md", "different")
    with pytest.raises(ValueError, match="overwrite"):
        apply_plan(repository, archive, plan)
    assert (repository / "eval/advanced_exp_results/run.json").exists()


def test_source_change_blocks_archive_before_move(repository):
    plan = build_plan(repository, "HEAD", [], include_markdown=False)
    _write(repository, "README.md", "updated")
    with pytest.raises(ValueError, match="Source changed"):
        apply_plan(repository, repository / "experiment_artifacts", plan)
    assert (repository / "eval/advanced_exp_results/run.json").exists()


def test_newly_tracked_result_cannot_be_moved(repository):
    plan = build_plan(repository, "HEAD", [], include_markdown=False)
    subprocess.run(["git", "-C", str(repository), "add", "eval/advanced_exp_results/run.json"], check=True)
    with pytest.raises(ValueError, match="now tracked"):
        apply_plan(repository, repository / "experiment_artifacts", plan)


def test_explicit_tracked_move_preserves_git_blob_and_input(repository):
    plan = build_plan(repository, "HEAD", [], include_markdown=False, move_tracked_results=True)
    archive = repository / "experiment_artifacts"
    apply_plan(repository, archive, plan)
    assert not (repository / "eval/exp_results/old.json").exists()
    assert (archive / "local/eval/exp_results/old.json").read_bytes() == (archive / "github/main/eval/exp_results/old.json").read_bytes()
    assert (repository / "README.md").exists()
    assert (repository / "eval/trigger_validation_36.labels.json").exists()
    assert b"eval/exp_results/old.json" in subprocess.check_output(["git", "-C", str(repository), "diff", "--name-only"])


def test_secret_and_symlink_rejected(repository, tmp_path):
    _write(repository, "eval/advanced_exp_results/key.json", '"sk-or-v1-' + "a" * 64 + '"')
    with pytest.raises(ValueError, match="Credential-like"):
        build_plan(repository, "HEAD", [], include_markdown=False)
    (repository / "eval/advanced_exp_results/key.json").unlink()
    external = _write(tmp_path, "outside.json", "sensitive")
    (repository / "eval/advanced_exp_results/link.json").symlink_to(external)
    with pytest.raises(ValueError, match="Symlink"):
        build_plan(repository, "HEAD", [], include_markdown=False)


def test_ignored_legacy_outputs_are_still_inventoried(repository):
    _write(repository, ".gitignore", "/eval/advanced_exp_results/\n/experiment_artifacts/\n")
    _write(repository, "eval/advanced_exp_results/trace.log", "diagnostic output")
    plan = build_plan(repository, "HEAD", [], include_markdown=False)
    sources = {entry["source"] for entry in plan["entries"] if entry["operation"] == "move"}
    assert "eval/advanced_exp_results/run.json" in sources
    assert "eval/advanced_exp_results/trace.log" in sources
