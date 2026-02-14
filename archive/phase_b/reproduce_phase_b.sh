#!/usr/bin/env bash
set -euo pipefail

ARCH="$(uname -m)"

# Map uname arch -> docker platform
case "${ARCH}" in
  x86_64)  DEFAULT_PLATFORM="linux/amd64" ;;
  aarch64|arm64) DEFAULT_PLATFORM="linux/arm64" ;;
  *)
    echo "Unknown arch: ${ARCH}. Defaulting to linux/amd64 (override with TF_PLATFORM/TOPO_PLATFORM)."
    DEFAULT_PLATFORM="linux/amd64"
    ;;
esac

# Allow overrides
TF_PLATFORM="${TF_PLATFORM:-$DEFAULT_PLATFORM}"
TOPO_PLATFORM="${TOPO_PLATFORM:-$DEFAULT_PLATFORM}"

echo "Host arch: ${ARCH}"
echo "Default platform: ${DEFAULT_PLATFORM}"
echo "TF_PLATFORM: ${TF_PLATFORM}"
echo "TOPO_PLATFORM: ${TOPO_PLATFORM}"

echo "== Phase B: SR generation (TF1.15.5 container) =="
# NOTE: TF 1.15 images are usually
