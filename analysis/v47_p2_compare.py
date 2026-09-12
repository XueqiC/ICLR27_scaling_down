#!/usr/bin/env python3
"""P2-B compare: FROZEN fits (freeze.json, committed before test training) vs the three U375 test trajectories.
Evaluates each frozen candidate at the ACTUAL milestone (T_c, E completion-convention) and reports the planned
predictions alongside. Per capability x milestone x pool: actual, predictions, signed bias, abs error; MAE per
candidate; improvement vs constant and vs delta=0. Old v41 (prefix-pool) predictors at actual points for reference."""
import json, glob
import numpy as np
from pathlib import Path
import analysis.v41_distill_newpool as v41
ROOT = Path(__file__).resolve().parents[1]; REG = ROOT / "results/v47-p2-register"
reg = json.loads((REG / "register.json").read_text()); fz = json.loads((REG / "freeze.json").read_text())
CAPS = ("math", "code", "qa"); KINDS = ("constant", "T", "E", "2D")
ep = v41._endpoint_rows(); old = {c: {k: v41._fit(ep, c, k) for k in KINDS} for c in CAPS}
rows = []
for S in (21, 22, 23):
    k = f"U375_s{S}"; DU = reg["pools"][k]["D_U_completion"]
    d = ROOT / f"results/v12-distill/gemma3-1b/gpt-5.6-luna_full_375_p2test_lora_dseed{S}"
    pts = []
    for f in sorted(glob.glob(str(d / "trajectory/update-*/eval.json"))):
        j = json.loads(Path(f).read_text())
        if j.get("processed_tokens", 0) > 0:
            pts.append({"Tc": j["completion_tokens_seen"], "E": j["completion_tokens_seen"] / DU,
                        "E_proc": j["processed_tokens"] / j["unique_data_pool_tokens"], "delta": j["delta"]})
    pts.sort(key=lambda p: p["Tc"])
    for i, p in enumerate(pts[:2]):
        planned = fz["test_predictions_planned"][k][f"milestone{i+1}"]
        for c in CAPS:
            act = p["delta"][c]
            pred_actual = {kd: float(v41._predict([p], c, kd, np.array(fz["fits"][c][kd]))[0]) for kd in KINDS}
            pred_old = {kd: float(v41._predict([{"Tc": p["Tc"], "E": p["E_proc"]}], c, kd, old[c][kd])[0]) for kd in KINDS}
            rows.append({"pool": k, "milestone": i + 1, "Tc_actual": p["Tc"], "E_actual": p["E"], "Tc_planned": planned["planned_T"],
                         "E_planned": planned["planned_E"], "cap": c, "actual": act,
                         "pred_at_actual": pred_actual, "pred_planned": planned["pred"][c], "old_v41_pred_at_actual": pred_old,
                         "signed": {kd: v - act for kd, v in pred_actual.items()}, "abs": {kd: abs(v - act) for kd, v in pred_actual.items()},
                         "abs_zero": abs(act)})
out = {"rows": rows, "summary": {}}
L = ["# P2-B new-pool frozen prospective: U375 x data-seeds {21,22,23} (P-new)\n",
     "Fits frozen on six random-pool dev trajectories (freeze.json, committed before test training). E = T_c/D_U (completion).",
     "Predictions evaluated at ACTUAL milestone (T_c,E); planned predictions retained. Signed = pred - actual.\n"]
for c in CAPS:
    for m in (1, 2):
        rr = [r for r in rows if r["cap"] == c and r["milestone"] == m]
        if not rr: continue
        mae = {kd: float(np.mean([r["abs"][kd] for r in rr])) for kd in KINDS}; bias = {kd: float(np.mean([r["signed"][kd] for r in rr])) for kd in KINDS}
        z = float(np.mean([r["abs_zero"] for r in rr]))
        out["summary"][f"{c}|m{m}"] = {"mae": mae, "bias": bias, "mae_zero": z, "improvement_vs_constant": {kd: mae["constant"] - mae[kd] for kd in KINDS},
                                       "improvement_vs_zero": {kd: z - mae[kd] for kd in KINDS}, "n_pools": len(rr),
                                       "per_pool_actual": {r["pool"]: r["actual"] for r in rr}}
        L.append(f"## {c} milestone {m} (n=3 pools; actual Tc≈{np.mean([r['Tc_actual'] for r in rr])/1e3:.0f}k, E≈{np.mean([r['E_actual'] for r in rr]):.2f})")
        L.append("| candidate | MAE | signed bias | vs constant | vs zero |"); L.append("|---|---|---|---|---|")
        for kd in KINDS:
            L.append(f"| {kd} | {mae[kd]:.3f} | {bias[kd]:+.3f} | {mae['constant']-mae[kd]:+.3f} | {z-mae[kd]:+.3f} |")
        L.append(f"| zero | {z:.3f} | | | |")
        L.append("per-pool actual: " + ", ".join(f"{r['pool']}={r['actual']:+.3f}" for r in rr) + "\n")
(REG / "compare_test.json").write_text(json.dumps(out, indent=2, default=float))
(ROOT / "paper/docs/P2_NEWPOOL.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
