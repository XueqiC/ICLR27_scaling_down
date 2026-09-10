#!/usr/bin/env bash
# Forward-only V54 measurements. Test measurements wait for a committed freeze.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=GPU-8b270cf8-6bb4-cee0-7060-88eba83d2fb0
PYTHON_BIN="${PYTHON:-python3}"
OUT_BASE="results/v54-quant-group"
mkdir -p logs "$OUT_BASE"
exec > >(tee -a logs/v54_grid.log) 2>&1

log() {
    printf '[v54 %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

STATES=(
    pythia-160m@step16000
    pythia-160m@step143000
    pythia-410m@step16000
    pythia-410m@step143000
    pythia-1.4b@step16000
    pythia-1.4b@step143000
)

run_state() {
    local phase="$1" tag="$2" configs="$3"
    log "$phase model=$tag configs=$configs"
    # The runner logs each config's synchronized wall_time_s (quantize, forward
    # losses, restore), and loads each state once per phase. Dense is cached.
    "$PYTHON_BIN" -u analysis/v54_quant_group.py \
        --model "$tag" --configs "$configs" --device cuda:0 \
        --model-dtype bf16 --n-probe 128 --reference-device cpu
}

log "DEV start"
for tag in "${STATES[@]}"; do
    run_state DEV "$tag" b3_g64,b3_g256,b5_g64,b5_g256
done
touch "$OUT_BASE/DEV_DONE"
log "DEV_DONE; waiting for $OUT_BASE/FREEZE_COMMITTED"
# This script never creates FREEZE_COMMITTED. The freeze must be committed
# externally before that sentinel is supplied; all test configs are below it.
while [[ ! -f "$OUT_BASE/FREEZE_COMMITTED" ]]; do
    sleep 30
done
log "FREEZE_COMMITTED found; test measurements released"

for tag in "${STATES[@]}"; do
    run_state BIT_TEST "$tag" b4_g64,b4_g256
done
for tag in "${STATES[@]}"; do
    run_state GRANULARITY_TEST "$tag" b3_g128,b4_g128,b5_g128
done
run_state JOINT_TEST pythia-1b@step96000 b3_g128,b4_g128,b5_g128
log "Grid complete"
