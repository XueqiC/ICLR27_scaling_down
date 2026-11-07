#!/usr/bin/env bash
# Example: nohup scripts/rsi_r1_accuracy.sh 0 'olmo2-1b@399B' > logs/r1_acc_driver.log 2>&1 &
# All six dense development pilots run first, even when only a test state is requested.
set -euo pipefail
if (( $# < 2 )); then
    echo "Usage: $0 <gpu> <state>... (BATCH_SIZE defaults to 8)" >&2
    exit 2
fi
gpu=$1
shift
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
mkdir -p logs
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
export HF_HUB_CACHE="${HF_HUB_CACHE:-$root/envs/hf_cache_a18/hub}"
python_env="${R1_PYTHON:-$root/.venv-gemma4/bin/python}"
runner=(uv run --no-project --python "$python_env" python -u analysis/r1_accuracy_efficiency.py)
# Validate all requested names before planning or loading a checkpoint.
uv run --no-project --python "$python_env" python - "$@" <<'PY'
import sys
from analysis.a18_second_family import STATES
unknown = set(sys.argv[1:]) - STATES.keys()
if unknown:
    raise SystemExit(f"Unknown states: {sorted(unknown)}; valid states: {list(STATES)}")
PY
"${runner[@]}" --plan
pilots=('olmo2-1b@399B' 'olmo2-1b@1993B' 'olmo2-1b@3608B'
        'olmo2-7b@391B' 'olmo2-7b@1947B' 'olmo2-7b@3507B')
for state in "${pilots[@]}"; do
    "${runner[@]}" --measure --state "$state" --dense-only --batch-size "${BATCH_SIZE:-8}" \
        >> "logs/r1_acc_${state}.log" 2>&1
done
for state in "$@"; do
    "${runner[@]}" --measure --state "$state" --batch-size "${BATCH_SIZE:-8}" \
        >> "logs/r1_acc_${state}.log" 2>&1
done
