#!/usr/bin/env python3
"""Prepare a checked, resumable artifact archive for an operator's shared Drive.

Default: print a read-only plan. --apply freezes that plan, copies tracked/Git
artifacts, and moves only untracked result files after verifying their hashes.
--move-tracked-results also relocates tracked results, leaving visible Git deletions.
This program never uploads, commits, changes the Git index, or checks out branches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from datetime import datetime, timezone


DEFAULT_HISTORY = ("backup-before-rebase", "backup/pre-jul20-rollback-20260724")
RESULT_PREFIXES = (
    "advanced_exp_results", "exp_results", "exp_baseline_results", "eval_results",
    "gpt4o_mini_evaluation", "openreviewer_papers", "paperreviewer_papers",
)
PRIVATE_MANIFESTS = (
    "eval/trigger_validation_36.evaluation.json",
    "eval/trigger_validation_36.labels.json",
    "eval/trigger_validation_36.selection.json",
)
SOURCE_FILES = {
    "agents.py", "mas_loop.py", "review_schema.py", "config.py", "requirements.txt",
    "scripts/archive_experiment_artifacts.py",
    "scripts/audit_experiment_invalid_json.py",
    "scripts/repair_incomplete_advanced_experiment.py",
    "scripts/build_trigger_validation_manifest.py",
}
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bsk-(?:or-v1-)?[A-Za-z0-9_-]{24,}"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}"),
)


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_git_blobs(repo: Path, object_ids: list[str]) -> dict[str, bytes]:
    unique = sorted(set(object_ids))
    if not unique:
        return {}
    result = subprocess.run(["git", "-C", str(repo), "cat-file", "--batch"],
                            input=("\n".join(unique) + "\n").encode(), capture_output=True, check=True)
    output, offset, blobs = result.stdout, 0, {}
    for expected in unique:
        newline = output.index(b"\n", offset)
        header = output[offset:newline].decode().split()
        if len(header) != 3 or header[0] != expected or header[1] != "blob":
            raise ValueError(f"Invalid Git blob response: {expected}")
        size = int(header[2])
        offset = newline + 1
        blobs[expected] = output[offset:offset + size]
        offset += size + 1
    return blobs


def safe_relative(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(p in ("..", ".git") for p in path.parts):
        raise ValueError(f"Unsafe archive path: {value}")
    return Path(*path.parts)


def safe_path(root: Path, relative: str) -> Path:
    path = root / safe_relative(relative)
    if path.is_symlink() or root.resolve() not in path.resolve().parents:
        raise ValueError(f"Symlink/outside-root path refused: {path}")
    return path


def is_private(path: str) -> bool:
    if path in ("eval/openreviewer.json", "eval/paperreviewer.json"):
        return False
    return path in PRIVATE_MANIFESTS or (
        path.startswith("eval/") and "/" not in path[5:]
        and path.endswith((".json", ".jsonl"))
        and not path.endswith(".blinded.json")
        and not path.endswith(".template.json")
    )


def is_result(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return (len(parts) >= 3 and parts[0] == "eval" and parts[1].startswith(RESULT_PREFIXES)) or path in ("eval/openreviewer.json", "eval/paperreviewer.json") or path.startswith("results/")


def is_artifact(path: str) -> bool:
    """Explicit experiment allowlist; never export arbitrary home/config files."""
    parts = PurePosixPath(path).parts
    if any(p.startswith(".") for p in parts) or path.endswith((".pyc", ".env")):
        return False
    if path == "README.md" or path in SOURCE_FILES:
        return True
    if path.startswith(("docs/experiments/", "prompts/", "rag/")) and path.endswith((".md", ".py", ".json")):
        return True
    if path.endswith(".ipynb") and (len(parts) == 1 or parts[0] == "eval"):
        return True
    if is_result(path):
        return path.endswith((".json", ".jsonl", ".txt", ".log", ".md", ".png", ".jpg", ".csv", ".tsv", ".pdf"))
    if len(parts) == 2 and parts[0] == "eval":
        return path.endswith((".json", ".jsonl", ".py", ".png", ".csv"))
    return len(parts) >= 3 and parts[0] == "data" and parts[1].startswith("rag_cache_before_") and path.endswith(".json")


def check_content(data: bytes, label: str) -> None:
    if any(pattern.search(data) for pattern in SECRET_PATTERNS):
        raise ValueError(f"Credential-like content detected; review before sharing: {label}")


def make_entry(*, source: str, destination: str, data: bytes, operation: str, **extra) -> dict:
    check_content(data, source)
    safe_relative(source)
    safe_relative(destination)
    return {"source": source, "destination": destination, "operation": operation,
            "sha256": sha256(data), "size_bytes": len(data), **extra}


def build_plan(repo: Path, remote_ref: str, history_refs: list[str], include_markdown: bool = True,
               move_tracked_results: bool = False, include_scratch: bool = False) -> dict:
    tracked = set(git(repo, "ls-files", "-z").decode().split("\0")) - {""}
    untracked = set(git(repo, "ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")) - {""}
    # Legacy output roots may already be ignored to prevent accidental commits.
    # Inventory this explicit allowlist independently of Git's ignore filtering.
    legacy_roots = [p for p in (repo / "eval").glob("*")
                    if p.is_dir() and p.name.startswith(RESULT_PREFIXES)]
    legacy_roots.extend((repo / "data").glob("rag_cache_before_*"))
    for root in legacy_roots:
        safe_path(repo, root.relative_to(repo).as_posix())
        untracked.update(p.relative_to(repo).as_posix() for p in root.rglob("*")
                         if p.is_file() and p.relative_to(repo).as_posix() not in tracked)
    for name in ("eval/openreviewer.json", "eval/paperreviewer.json"):
        if (repo / name).exists() and name not in tracked:
            untracked.add(name)
    if include_scratch and (repo / "results").exists():
        untracked.update(p.relative_to(repo).as_posix() for p in (repo / "results").rglob("*") if p.is_file() and p.relative_to(repo).as_posix() not in tracked)
    entries = []
    for source in sorted(tracked | untracked):
        if not is_artifact(source):
            continue
        path = safe_path(repo, source)
        if not path.exists():
            continue
        move = ((source not in tracked or move_tracked_results) and is_result(source)) or (source not in tracked and source.startswith("data/rag_cache_before_"))
        prefix = "private/local" if is_private(source) else "local"
        entries.append(make_entry(source=source, destination=f"{prefix}/{source}",
                                  data=path.read_bytes(), operation="move" if move else "copy",
                                  tracked=source in tracked))
    refs = {}
    blob_cache = {}
    for ref, prefix in [(remote_ref, "github/main"), *((ref, "history/" + ref.replace("/", "__")) for ref in history_refs)]:
        commit = git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
        refs[ref] = commit
        selected = []
        for record in git(repo, "ls-tree", "-rz", "--full-tree", commit).split(b"\0"):
            if not record:
                continue
            metadata, raw_path = record.split(b"\t", 1)
            mode, kind, blob = metadata.decode().split()
            source = raw_path.decode()
            if not is_artifact(source):
                continue
            if kind != "blob" or mode not in ("100644", "100755"):
                raise ValueError(f"Non-regular artifact refused: {ref}:{source}")
            selected.append((source, blob))
        blob_cache.update(read_git_blobs(repo, [blob for _, blob in selected if blob not in blob_cache]))
        for source, blob in selected:
            data = blob_cache[blob]
            destination_prefix = f"private/{prefix}" if is_private(source) else prefix
            entries.append(make_entry(source=source, destination=f"{destination_prefix}/{source}",
                                      data=data, operation="git_export", ref=ref, commit=commit, blob=blob))
    manifest = repo / "eval/trigger_validation_36.blinded.json"
    if include_markdown and manifest.exists():
        for meta in json.loads(manifest.read_text()).values():
            filename = Path(meta["paper_dir"]).stem + ".md"
            source = "data/marker_md/" + filename
            path = safe_path(repo, source)
            entries.append(make_entry(source=source, destination=f"private/reproducibility_inputs/{source}",
                                      data=path.read_bytes(), operation="copy", tracked=source in tracked))
    destinations = [entry["destination"] for entry in entries]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Duplicate destination in archive plan")
    return {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "move_tracked_results": move_tracked_results, "include_scratch": include_scratch,
            "repository_head": git(repo, "rev-parse", "HEAD").decode().strip(),
            "remote_refs": refs, "entries": entries}


def read_source(repo: Path, entry: dict, blob_cache: dict | None = None) -> bytes:
    if entry["operation"] == "git_export":
        data = blob_cache[entry["blob"]] if blob_cache is not None else git(repo, "cat-file", "blob", entry["blob"])
    else:
        data = safe_path(repo, entry["source"]).read_bytes()
    if sha256(data) != entry["sha256"] or len(data) != entry["size_bytes"]:
        raise ValueError(f"Source changed after plan: {entry['source']}")
    check_content(data, entry["source"])
    return data


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".archive-", delete=False) as handle:
        temp = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def verify_archive(archive: Path, plan: dict) -> None:
    for entry in plan["entries"]:
        destination = safe_path(archive, entry["destination"])
        data = destination.read_bytes()
        if sha256(data) != entry["sha256"] or len(data) != entry["size_bytes"]:
            raise ValueError(f"Archive verification failed: {entry['destination']}")


def apply_plan(repo: Path, archive: Path, plan: dict) -> dict:
    archive.mkdir(parents=True, exist_ok=True)
    # Preflight ALL collisions and sources before moving a single result.
    tracked = set(git(repo, "ls-files", "-z").decode().split("\0"))
    blob_cache = read_git_blobs(repo, [e["blob"] for e in plan["entries"] if e["operation"] == "git_export"])
    for entry in plan["entries"]:
        destination = safe_path(archive, entry["destination"])
        if destination.exists():
            if sha256(destination.read_bytes()) != entry["sha256"]:
                raise ValueError(f"Refusing to overwrite different artifact: {destination}")
        source = safe_path(repo, entry["source"])
        if entry["operation"] == "move" and entry["source"] in tracked and not (plan.get("move_tracked_results") and entry.get("tracked") and is_result(entry["source"])):
            raise ValueError(f"Source is now tracked; refusing move: {entry['source']}")
        if entry["operation"] == "git_export" or source.exists():
            read_source(repo, entry, blob_cache)
        elif not (entry["operation"] == "move" and destination.exists()):
            raise FileNotFoundError(source)
    for entry in plan["entries"]:
        destination = safe_path(archive, entry["destination"])
        if not destination.exists():
            data = read_source(repo, entry, blob_cache)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".archive-", delete=False) as handle:
                temp = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # An exclusive hard link publishes a complete file atomically.
                os.link(temp, destination)
            finally:
                temp.unlink()
        if sha256(destination.read_bytes()) != entry["sha256"]:
            raise ValueError(f"Copy verification failed: {destination}")
        if entry["operation"] == "move":
            source = safe_path(repo, entry["source"])
            if source.exists():
                # Recheck immediately before removing the verified original copy.
                read_source(repo, entry, blob_cache)
                source.unlink()
        if entry["destination"].startswith("private/"):
            destination.chmod(0o600)
    parents = {safe_path(repo, entry["source"]).parent for entry in plan["entries"] if entry["operation"] == "move"}
    for parent in sorted(parents, key=lambda p: len(p.parts), reverse=True):
        while parent != repo and parent.relative_to(repo).as_posix() not in ("eval", "data"):
            try:
                parent.rmdir()  # Removes only an empty, explicitly archived source directory.
            except OSError:
                break
            parent = parent.parent
    verify_archive(archive, plan)
    catalog = {**plan, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
               "file_count": len(plan["entries"]), "size_bytes": sum(x["size_bytes"] for x in plan["entries"])}
    atomic_json(archive / "CATALOG.json", catalog)
    lines = ["# Archive catalog", "", "Exact sources, Git commits and SHA-256 are recorded in CATALOG.json.", "",
             "| Archive path | Operation | Bytes | SHA-256 |", "|---|---|---:|---|"]
    for entry in plan["entries"]:
        lines.append(f"| `{entry['destination']}` | {entry['operation']} | {entry['size_bytes']} | `{entry['sha256']}` |")
    (archive / "CATALOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive-dir", type=Path, default=Path("experiment_artifacts"))
    parser.add_argument("--remote-ref", default="origin/main")
    parser.add_argument("--history-ref", action="append", dest="history_refs")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--no-markdown", action="store_true")
    parser.add_argument("--move-tracked-results", action="store_true", help="Relocate tracked results too; creates deliberate working-tree deletions, without staging them")
    parser.add_argument("--include-scratch", action="store_true", help="Include ignored results/ smoke/debug outputs, separately identified by source paths")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    archive = args.archive_dir if args.archive_dir.is_absolute() else repo / args.archive_dir
    if archive.resolve() == repo or archive.is_symlink():
        raise ValueError("Archive must be a dedicated directory, not repository root or a symlink")
    plan_path = archive / "ARCHIVE_PLAN.json"
    if args.verify:
        plan = json.loads(plan_path.read_text())
        verify_archive(archive, plan)
        print(f"Verified {len(plan['entries'])} files: {archive}")
        return 0
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        refs = [] if args.no_history else (args.history_refs if args.history_refs is not None else list(DEFAULT_HISTORY))
        plan = build_plan(repo, args.remote_ref, refs, not args.no_markdown, args.move_tracked_results, args.include_scratch)
    stats = {operation: sum(e["operation"] == operation for e in plan["entries"]) for operation in ("move", "copy", "git_export")}
    print(json.dumps({"archive": str(archive), "apply": args.apply, "files": len(plan["entries"]),
                      "size_bytes": sum(e["size_bytes"] for e in plan["entries"]), "operations": stats,
                      "move_tracked_results": plan.get("move_tracked_results", False),
                      "tracked_results_to_move": sum(e["operation"] == "move" and e.get("tracked", False) for e in plan["entries"]),
                      "remote_refs": plan["remote_refs"]}, ensure_ascii=False, indent=2))
    if args.apply:
        if not plan_path.exists():
            atomic_json(plan_path, plan)
        apply_plan(repo, archive, plan)
        print("Archive complete; source and destination SHA-256 verified. No upload performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
