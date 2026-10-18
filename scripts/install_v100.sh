#!/usr/bin/env bash
# Explicit future cloud setup. Never invoked by CPU checks or package creation.
set -euo pipefail
TASK_PYTHON="${1:-/mnt/workspace/rsi-v100-env/bin/python}"
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"$TASK_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11), "Use the independent Python3.11environment"'
"$TASK_PYTHON" -m pip install 'torch==2.6.0+cu124' --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu124
"$TASK_PYTHON" -m pip install -r "$TASK_ROOT/requirements-v100.txt"
"$TASK_PYTHON" -m pip check
"$TASK_PYTHON" -B "$TASK_ROOT/doctor.py" --gpu
