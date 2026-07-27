#!/usr/bin/env bash
# Build (if needed) and run the doc_preprocess parallel converter in Docker.
#
# Usage:
#   scripts/docker_run_preprocess.sh                       # all PDFs, 4 workers
#   scripts/docker_run_preprocess.sh --workers 2 --limit 3 # smoke test
#   scripts/docker_run_preprocess.sh --workers 3 --overwrite  # recon all
#
# Any args are forwarded to run_doc_preprocess_parallel.py after the default
# positional dirs (data/openreview_pdf data/openreview_md). Pass --workers,
# --limit, --overwrite, etc.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="paper-reviewer-preprocess:latest"

# Build the image if missing or if any COPY'd source file is newer than the image.
build() {
  echo "Building image ${IMAGE_TAG} ..."
  docker build \
    -t "${IMAGE_TAG}" \
    -f "${REPO_ROOT}/Dockerfile" \
    "${REPO_ROOT}"
}

# Rebuild if the image doesn't exist. (Add --build after a code change to force.)
if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  build
fi
if [[ "${1:-}" == "--build" ]]; then
  build
  shift
fi

# Bind-mount the repo so the container reads data/openreview_pdf and writes
# data/openreview_md on the host filesystem. Mount the HF cache so marker
# models are downloaded once and reused across runs.
mkdir -p "${REPO_ROOT}/.docker_cache/hf"

echo "Running converter in container (image ${IMAGE_TAG}) ..."
exec docker run --rm -it \
  -v "${REPO_ROOT}:/app" \
  -v "${REPO_ROOT}/.docker_cache/hf:/app/.cache/huggingface" \
  -w /app \
  "${IMAGE_TAG}" \
  data/openreview_pdf data/openreview_md "$@"
