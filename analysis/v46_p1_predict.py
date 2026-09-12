#!/usr/bin/env python3
"""P1 FROZEN predictions for a NEW source (pythia-1b@step96000) -- written and committed BEFORE any
compressed measurement. Inputs: K0 only (N0 from architecture, D0 from step, dense L0 measured by v46_dense_eval).
Pruning at d in {0.65, 0.55}: original shared-power (v40 register, frozen), A2 per-density regression + linear
interp/extrap (dev only), A1 gamma=1 refit (dev only), strength-only, zero. Quantization at b in {4, 3}: frozen
config-indicator {N0,L0,D0}, no-D0 {N0,L0}, per-bit median (dev), zero. Nothing here touches the target's
compressed outcomes. Output: results/v46-p1-newsource/predictions_frozen.json (+ input hashes)."""
import json, hashlib, itertools, datetime
import numpy as np
import analysis.v36_pythia_controlled_fit as v36
import analysis.v40_prune_strength_axis as v40
import analysis.v42_prune_sameinput as v42

OUT = v36.ROOT / "results/v46-p1-newsource"
SIZE, STEP = "1b", 96000
dense = json.loads((OUT / "dense.json").read_text())["dense"]
N0, D0 = v36.matrix_n0(SIZE), STEP * v36.TOKENS_PER_STEP

def phi_raw(cap): return np.array([np.log(N0), dense[cap], np.log(D0)])
# --- pruning (dev = v40.load_dev: 9-state grid at seen densities)
rows = v40.load_dev(); phi = np.array([r["phi"] for r in rows]); center, scale = phi.mean(0), phi.std(0)
reg = json.loads((v40.OUT / "register.json").read_text())
pred = {"target": {"size": SIZE, "step": STEP, "N0": N0, "D0": D0, "dense": dense},
        "pruning": {}, "quantization": {}}
for cap in v40.CAPS:
    z = np.concatenate([[1.0], (phi_raw(cap) - center) / scale])
    c = reg["caps"][cap]; A = float(np.array(c["beta"]) @ z)
    A1 = v42._fit_A1(rows, cap, center, scale); A2 = v42._fit_A2(rows, cap, center, scale)
    so, mc = c["strength_only"], c["median_curve"]
    for d in (0.65, 0.55):
        p7, p6 = float(z @ A2[0.7]), float(z @ A2[0.6]); slope = (p6 - p7) / (0.6 - 0.7)
        pred["pruning"][f"{cap}|{d}"] = {
            "power": A * v40._shape(d, c["gamma"]),
            "A2": p7 + slope * (d - 0.7),
            "A1_gamma1": float(z @ A1) * v40._shape(d, 1.0),
            "strength_only": so["A"] * v40._shape(d, so["gamma"]),
            "median_curve": mc["c"] * v40._shape(d, mc["p"]), "zero": 0.0}
# --- quantization (dev = 3x3 grid, frozen config-indicator fits)
train, hashes = v36.load_grid(sizes=("160m", "410m", "1.4b"))
for cap in v36.CAPS:
    tr = [r for r in train if r["arm"] == "quantization" and r["capability"] == cap]
    fB = v36.fit_direct(tr, input_fields=("N0", "L0", "D0")); fA = v36.fit_direct(tr, input_fields=("N0", "L0"))
    for b in (4, 3):
        row = {"arm": "quantization", "capability": cap, "config": b, "N0": N0, "D0": D0, "L0": dense[cap]}
        med = float(np.median([r["observed"] for r in tr if r["config"] == b]))
        pred["quantization"][f"{cap}|{b}"] = {
            "full_N0_L0_D0": float(v36.predict(fB, [v36.basic_input(row, input_fields=("N0", "L0", "D0"))])[0]),
            "noD0_N0_L0": float(v36.predict(fA, [v36.basic_input(row, input_fields=("N0", "L0"))])[0]),
            "per_bit_median": med, "zero": 0.0}
pred["provenance"] = {"frozen_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
                      "v40_register_sha": hashlib.sha256((v40.OUT / "register.json").read_bytes()).hexdigest(),
                      "dev_file_hashes": hashes, "dense_json_sha": hashlib.sha256((OUT / "dense.json").read_bytes()).hexdigest(),
                      "note": "K0 only; no target compressed outcome used; A2/A1 refit on original dev at freeze time"}
(OUT / "predictions_frozen.json").write_text(json.dumps(pred, indent=2))
for k, v in pred["pruning"].items(): print("prune", k, {a: round(b, 3) for a, b in v.items()})
for k, v in pred["quantization"].items(): print("quant", k, {a: round(b, 3) for a, b in v.items()})
