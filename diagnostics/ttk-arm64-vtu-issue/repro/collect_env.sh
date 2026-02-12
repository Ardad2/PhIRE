#!/usr/bin/env bash
set -euxo pipefail

echo "=== DATE (UTC) ==="
date -u
echo

echo "=== UNAME ==="
uname -a
uname -m
echo

echo "=== OS RELEASE ==="
cat /etc/os-release || true
echo

echo "=== PYTHON ==="
python --version || true
command -v python || true
python -c "import platform; print(platform.platform())" || true
echo

echo "=== PIP FREEZE (TOP) ==="
python -m pip freeze | head -n 200 || true
echo
