#!/usr/bin/env python3
"""P1 frozen prospective: does the controlled-panel predictor beat the frozen baseline on a
NEW, never-fit training step (step96000)? Coefficients are frozen from the 9-state v36 fit
(committed BEFORE measurement). N0/D0 are known a priori; only dense L0 comes from the new
measurement. Prediction is therefore pre-registered as a function of the single measured input.

Modes:
  register  -> write results/v38-prospective/register.json (frozen coeffs sha + N0/D0 + formula). No GPU.
  compare   -> after v6/v10 forward-only runs on the new cells exist, plug measured L0 into the frozen
               predictor, read actual compressed dL, report full-vs-baseline error on the NEW source-state.
"""
import json, hashlib, sys, itertools
from pathlib import Path
import numpy as np
import analysis.v36_pythia_controlled_fit as v36

ROOT = v36.ROOT
OUT = ROOT / "results/v38-prospective"
NEW_STEP = 96000
NEW_SIZES = ("160m", "1.4b")
FIT = ROOT / "results/v36-pythia-controlled/summary.json"

def a_priori_inputs(size, step):
    # N0 = non-embedding param proxy already used by v36 (architecture table); D0 = tokens = step*2097152.
    return {"size": size, "step": step,
            "N0": v36.matrix_n0(size), "D0": step * v36.TOKENS_PER_STEP}

def register():
    fit = json.loads(FIT.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    reg = {"frozen_fit_sha256": hashlib.sha256(FIT.read_bytes()).hexdigest(),
           "new_step": NEW_STEP, "new_sizes": list(NEW_SIZES),
           "predictor": "config-indicator OLS from v36 9-state fit; inputs {N0,D0,L0}; L0 measured, N0/D0 a priori",
           "a_priori_inputs": [a_priori_inputs(s, NEW_STEP) for s in NEW_SIZES],
           "note": "Prediction registered before compressed measurement; only dense L0 is read from the new runs."}
    (OUT / "register.json").write_text(json.dumps(reg, indent=2))
    print("registered", OUT / "register.json")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "register"
    if mode == "register":
        register()
    # compare handled by the dedicated __main__ block below


# ---- compare mode (run after forward-only measurement of the new cells exists) ----
import numpy as _np

def _new_cell_rows(arm):
    """Build v36-format rows for the two new step96000 cells from their measured JSON."""
    directory, filename, _ = v36.FILES[arm]
    rows = []
    for size in NEW_SIZES:
        cell = f"pythia-{size}--step{NEW_STEP}"
        path = ROOT / f"results/{directory}/{cell}/{filename}"
        dense, compressed = v36.parse_losses(json.loads(path.read_text()), arm)
        for cap, config in itertools.product(v36.CAPS, v36.CONFIGS[arm]):
            rows.append({"row_id": f"{arm}|{cell}|{cap}|{config:g}", "arm": arm, "capability": cap,
                         "cell": cell, "size": size, "step": NEW_STEP, "config": config,
                         "N0": v36.matrix_n0(size), "D0": NEW_STEP * v36.TOKENS_PER_STEP,
                         "L0": dense[cap], "loss": compressed[config][cap],
                         "observed": compressed[config][cap] - dense[cap]})
    return rows

def compare():
    train, _hashes = v36.load_grid(sizes=("160m", "410m", "1.4b"))
    out = {"register": json.loads((OUT / "register.json").read_text()), "arms": {}}
    for arm in v36.ARMS:
        test = _new_cell_rows(arm)
        out["arms"][arm] = {}
        for cap in v36.CAPS:
            tr = [r for r in train if r["arm"] == arm and r["capability"] == cap]
            te = [r for r in test if r["capability"] == cap]
            # frozen predictors fit on all 9 states
            fitB = v36.fit_direct(tr, input_fields=("N0", "L0", "D0"))
            fitA = v36.fit_direct(tr, input_fields=("N0", "L0"))
            medians = {f"{q:g}": float(_np.median([r["observed"] for r in tr if r["config"] == q]))
                       for q in v36.CONFIGS[arm]}
            obs = _np.array([r["observed"] for r in te])
            predB = v36.predict(fitB, [v36.basic_input(r, input_fields=("N0", "L0", "D0")) for r in te])
            predA = v36.predict(fitA, [v36.basic_input(r, input_fields=("N0", "L0")) for r in te])
            predM = _np.array([medians[f"{r['config']:g}"] for r in te])
            mae = lambda p: float(_np.mean(_np.abs(obs - p)))
            byc = {}
            for q in v36.CONFIGS[arm]:
                idx = [i for i, r in enumerate(te) if r["config"] == q]
                o = obs[idx]
                byc[f"{q:g}"] = {"full": float(_np.mean(_np.abs(o - predB[idx]))),
                                 "baseline_median": float(_np.mean(_np.abs(o - predM[idx]))),
                                 "n": len(idx)}
            out["arms"][arm][cap] = {
                "n_test": len(te),
                "mae_full_N0_D0_L0": mae(predB), "mae_noD0_N0_L0": mae(predA),
                "mae_baseline_config_median": mae(predM),
                "full_beats_baseline": mae(predB) < mae(predM),
                "full_beats_noD0": mae(predB) < mae(predA),
                "by_config": byc}
    (OUT / "compare.json").write_text(json.dumps(out, indent=2))
    # console summary
    for arm in out["arms"]:
        print(f"\n### {arm} (NEW step{NEW_STEP}, sizes {list(NEW_SIZES)})")
        for cap, b in out["arms"][arm].items():
            print(f" {cap}: full={b['mae_full_N0_D0_L0']:.3f} noD0={b['mae_noD0_N0_L0']:.3f} "
                  f"baseline={b['mae_baseline_config_median']:.3f} "
                  f"{'✓beats-base' if b['full_beats_baseline'] else 'x'} "
                  f"{'✓beats-noD0' if b['full_beats_noD0'] else 'x'}")
            print("    by_config: " + ", ".join(
                f"{q}:full{v['full']:.2f}/base{v['baseline_median']:.2f}" for q, v in b["by_config"].items()))
    print("\nwrote", OUT / "compare.json")

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "compare":
    compare()
