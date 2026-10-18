#!/usr/bin/env bash
# Read-only diagnostics; no installation, downloads, inference or training.
set -eu
cd -- "$(dirname -- "$0")/.."
git rev-parse HEAD
git status --short
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi
PY=/mnt/workspace/rsi-v100-env/bin/python
if [ ! -x "$PY" ]; then
  echo "Python 3.11 environment not found at $PY; complete environment setup first."
  exit 2
fi
"$PY" --version
"$PY" -B doctor.py --gpu
