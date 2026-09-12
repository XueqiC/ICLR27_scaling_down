#!/usr/bin/env python3
"""P2 FREEZE (run after the six dev trajectories, BEFORE any test training): fit the four low-DOF candidates
(constant, T-only, E-only, 2D) per capability on the six random-pool DEV trajectories (two milestone snapshots
each = 12 points; final-epoch evals are NOT used), with E in the registered COMPLETION convention
E = T_completion / D_U_completion. No candidate selection. Write predictions for the three U375 test pools at
the planned milestones (T=148k/294k, E from registered D_U). Also: old v41 predictors vs the new dev pools."""
import json, glob, hashlib, datetime
import numpy as np
from pathlib import Path
import analysis.v41_distill_newpool as v41
ROOT = Path(__file__).resolve().parents[1]; REG = ROOT / "results/v47-p2-register"
reg = json.loads((REG / "register.json").read_text()); CAPS = ("math", "code", "qa")
def run_points(U, S, suffix):
    d = ROOT / f"results/v12-distill/gemma3-1b/gpt-5.6-luna_full_{U}_{suffix}_lora_dseed{S}"
    DU = reg["pools"][f"U{U}_s{S}"]["D_U_completion"]; pts = []
    for f in sorted(glob.glob(str(d / "trajectory/update-*/eval.json"))):
        j = json.loads(Path(f).read_text())
        if j.get("processed_tokens", 0) > 0:
            pts.append({"Tc": j["completion_tokens_seen"], "E": j["completion_tokens_seen"] / DU,
                        "E_proc": j["processed_tokens"] / j["unique_data_pool_tokens"], "delta": j["delta"],
                        "processed": j["processed_tokens"], "steps": j.get("updates")})
    pts.sort(key=lambda p: p["Tc"]); return pts
dev = {f"U{U}_s{S}": run_points(U, S, "p2dev") for U in (75, 450) for S in (11, 12, 13)}
assert all(len(v) >= 2 for v in dev.values()), {k: len(v) for k, v in dev.items()}
pts = [p for v in dev.values() for p in v[:2]]  # two milestones per run
fits = {c: {k: v41._fit(pts, c, k).tolist() for k in ("constant", "T", "E", "2D")} for c in CAPS}
test_pred = {}
for S in (21, 22, 23):
    k = f"U375_s{S}"; DU = reg["pools"][k]["D_U_completion"]; test_pred[k] = {}
    for i, T in enumerate(reg["protocol"]["T_milestones_completion"]):
        pt = {"Tc": T, "E": T / DU}
        test_pred[k][f"milestone{i+1}"] = {"planned_T": T, "planned_E": T / DU,
            "pred": {c: {kind: float(v41._predict([pt], c, kind, np.array(fits[c][kind]))[0]) for kind in fits[c]} for c in CAPS}}
# old v41 predictors vs new dev pools (robustness of historical prefix-pool fits to random pools), at ACTUAL points
ep = v41._endpoint_rows(); old = {c: {k: v41._fit(ep, c, k) for k in ("constant", "T", "E", "2D")} for c in CAPS}
old_vs_dev = {}
for k, v in dev.items():
    old_vs_dev[k] = [{"Tc": p["Tc"], "E_proc": p["E_proc"], "actual": p["delta"],
        "old_pred": {c: {kind: float(v41._predict([{"Tc": p["Tc"], "E": p["E_proc"]}], c, kind, old[c][kind])[0]) for kind in old[c]} for c in CAPS}} for p in v[:2]]
out = {"frozen_at_utc": datetime.datetime.utcnow().isoformat() + "Z", "E_convention": "completion: T_completion/D_U_completion",
       "dev_points": {k: v[:2] for k, v in dev.items()}, "fits": fits, "test_predictions_planned": test_pred,
       "old_v41_predictors_vs_new_dev": old_vs_dev,
       "dev_hash": hashlib.sha256(json.dumps({k: v[:2] for k, v in dev.items()}, sort_keys=True).encode()).hexdigest()}
(REG / "freeze.json").write_text(json.dumps(out, indent=2, default=float))
for k, v in dev.items(): print(k, [(round(p["Tc"]/1e3), round(p["E"], 2), {c: round(p["delta"][c], 3) for c in CAPS}) for p in v[:2]])
print("test predictions written for", list(test_pred)); print("wrote", REG / "freeze.json")
