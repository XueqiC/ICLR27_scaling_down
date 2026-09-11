#!/usr/bin/env bash
# 21 forward-only confirmation cells; V69 freeze.json gates execution.
# Caller chooses the physical GPU, e.g. CUDA_VISIBLE_DEVICES=GPU-<UUID>.
# --dry-run needs neither a GPU nor a freeze and writes nothing.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# The paper mirror keeps scripts beside analysis/ rather than in scripts/.
if [[ ! -f "$PROJECT_ROOT/analysis/v69_quant_confirm.py" ]]; then
    PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON:-python3}"
exec "$PYTHON_BIN" -B -u analysis/v69_quant_confirm.py measure "$@"
