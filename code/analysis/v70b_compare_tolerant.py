#!/usr/bin/env python3
"""Run the frozen V70 compare with a documented tolerance on the re-measured dense descriptor. The frozen script checks
p["L0"] == frozen L0 exactly; dense losses re-measured on a different GPU drift by ~5e-4 nats. This wrapper leaves the
frozen script, predictions, and run responses untouched: it records the drift per trajectory, requires it below 0.01 nats,
and substitutes the frozen descriptor only for that equality check. Output: results/v70-distill-confirm/dense_drift.json."""
import json, sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import v70_distill_confirm as v70
ROOT = v70.ROOT; drift_log = {}
_orig = v70.read_trajectory
def read_trajectory(root, inputs, st, U, seed, suffix, pool, triggers):
    frozen = json.loads((root / "results/v70-distill-confirm/freeze.json").read_text())["refs"]["L0_by_student"][st]
    out = []
    for p in _orig(root, inputs, st, U, seed, suffix, pool, triggers):
        drift = max(abs(p["L0"][c] - frozen[c]) for c in v70.CAPS)
        if drift >= 0.01: raise ValueError(f"dense descriptor drift {drift:.4f} nats for {st} seed {seed}")
        drift_log[f"{st}|U{U}_s{seed}"] = {"run_dense": dict(p["L0"]), "frozen_dense": dict(frozen), "max_abs_drift": drift}
        p = dict(p); p["L0"] = dict(frozen); out.append(p)
    return out
v70.read_trajectory = read_trajectory
if __name__ == "__main__":
    sys.argv = [sys.argv[0], "compare"]; v70.main()
    (ROOT / "results/v70-distill-confirm/dense_drift.json").write_text(json.dumps(drift_log, indent=1))
    print("max drift over trajectories:", max(v["max_abs_drift"] for v in drift_log.values()))
