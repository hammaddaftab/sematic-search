#!/usr/bin/env bash
set -euo pipefail

# Configurable defaults
TAG="${1:-model-v1}"
MODEL_NAME="all-MiniLM-L6-v2"
TARGET_DIR="models/${MODEL_NAME}"
REPO_SLUG="hammaddaftab/sematic-search"

# Auto-detect GitHub owner/repo from git remote if available
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    REMOTE_URL=$(git config --get remote.origin.url || true)
    if [[ "$REMOTE_URL" =~ github\.com[:/]([^/]+)/([^/.]+)(\.git)?$ ]]; then
        REPO_SLUG="${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
    fi
fi

# Check if model is already installed
if [ -f "${TARGET_DIR}/model.onnx" ]; then
    echo "✓ Model '${MODEL_NAME}' already exists in ${TARGET_DIR}. Skipping download."
    exit 0
fi

URL="https://github.com/${REPO_SLUG}/releases/download/${TAG}/${MODEL_NAME}.tar.gz"
TMP_FILE="$(mktemp -t "${MODEL_NAME}-XXXXXX.tar.gz")"

echo "==> Downloading model '${MODEL_NAME}' (${TAG}) from:"
echo "    ${URL}"

if ! curl -fL --progress-bar -o "${TMP_FILE}" "${URL}"; then
    echo "" >&2
    echo "Error: Failed to download model asset from ${URL}" >&2
    echo "Verify that release '${TAG}' with asset '${MODEL_NAME}.tar.gz' is published at:" >&2
    echo "https://github.com/${REPO_SLUG}/releases/tag/${TAG}" >&2
    rm -f "${TMP_FILE}"
    exit 1
fi

echo "==> Extracting into ${TARGET_DIR}..."
mkdir -p "${TARGET_DIR}"
tar -xzf "${TMP_FILE}" -C "${TARGET_DIR}"
rm -f "${TMP_FILE}"

echo "✓ Model '${MODEL_NAME}' ready in ${TARGET_DIR}!"
