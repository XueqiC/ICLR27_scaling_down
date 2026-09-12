#!/usr/bin/env bash
# V72: two dense evaluations -> write-once freeze -> six V6 pruning cells.
# No GPU is touched by --dry-run or --selftest. Caller selects one GPU by UUID.
set -euo pipefail
V72_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V72_PROJECT_ROOT="$V72_SCRIPT_DIR"
while [[ "$V72_PROJECT_ROOT" != / ]]; do
  if [[ -f "$V72_PROJECT_ROOT/results/v53-prune-dev/register.json" && -f "$V72_PROJECT_ROOT/analysis/v72_prune_repeat.py" ]]; then
    break
  fi
  V72_PROJECT_ROOT="$(dirname "$V72_PROJECT_ROOT")"
done
if [[ "$V72_PROJECT_ROOT" == / ]]; then
  V72_PROJECT_ROOT="$(dirname "$V72_SCRIPT_DIR")"
fi
cd "$V72_PROJECT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
exec python3 -B analysis/v72_prune_repeat.py measure "$@"
