#!/usr/bin/env bash
# Usage: nohup bash scripts/rsi_r1_reuse.sh <gpu> <index>... >logs/r1_reuse_driver.log 2>&1 &
# Freeze --plan once before launching lanes. Reruns skip complete work and replay
# interrupted training from seed 0, preserving the interrupted directory.
set -euo pipefail
if [[ $# -lt 2 || ! $1 =~ ^([0-9]+|GPU-[A-Za-z0-9-]+)$ ]]; then
    echo "Usage: $0 <gpu index or UUID> <run index 0..11>..." >&2
    exit 2
fi
case $(hostname) in
    sn4622131043|rsi*) ;;
    *) echo "Qwen execution requires the rsi host" >&2; exit 2 ;;
esac
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"$ROOT/.venv-gemma4/bin/python"}
export CUDA_VISIBLE_DEVICES="$1" PYTHONUNBUFFERED=1
shift
for index in "$@"; do
    if [[ ! $index =~ ^([0-9]|10|11)$ ]]; then
        echo "Invalid run index: $index (expected 0..11)" >&2
        exit 2
    fi
done
cd -- "$ROOT"
mkdir -p logs
failures=0
for index in "$@"; do
    log="logs/r1_reuse_${index}.log"
    echo "[$(date -Is)] run $index" >> "$log"
    if "$PYTHON_BIN" analysis/r1_reuse_family.py --run --index "$index" >> "$log" 2>&1 &&
       "$PYTHON_BIN" analysis/r1_reuse_family.py --evaluate --index "$index" >> "$log" 2>&1; then
        echo "run $index complete: $log"
    else
        echo "run $index failed: $log" >&2
        failures=1
    fi
done
exit "$failures"
