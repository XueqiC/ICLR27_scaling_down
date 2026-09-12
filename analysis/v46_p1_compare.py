#!/usr/bin/env python3
"""P1 compare: frozen predictions (predictions_frozen.json, committed before measurement) vs measured
pythia-1b@step96000 responses. Reports per cell prediction/actual/signed/abs error; MAE per candidate per
(capability, config); pruning 0.65 and 0.55 separately; quant int4 and int3 separately."""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"
P = json.loads((R / "v46-p1-newsource/predictions_frozen.json").read_text())
pr = json.loads((R / "v6-capability-geometry/pythia-1b--step96000/prune_losses.json").read_text())
qt = json.loads((R / "v10-quantization/pythia-1b--step96000/quant_losses.json").read_text())
out = {"pruning": [], "quantization": [], "dense_frozen": P["target"]["dense"],
       "dense_remeasured": {"prune_run": pr["1.0"], "quant_run": qt["dense"]}}
lines = ["# P1 new-source frozen prospective: pythia-1b@step96000 (P-new)\n",
         "Predictions frozen+committed before measurement (b1bf631). Signed = pred - actual.\n",
         "| arm | cap | config | actual dL | " + " | ".join(["power/full", "A2/noD0", "A1/median", "strength-only", "median", "zero"]) + " |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for k, v in P["pruning"].items():
    cap, d = k.split("|"); d = float(d)
    actual = pr[f"{d:g}"][cap] - pr["1.0"][cap]
    rec = {"cap": cap, "d": d, "actual": actual, **{m: {"pred": p, "signed": p - actual, "abs": abs(p - actual)} for m, p in v.items()}}
    out["pruning"].append(rec)
    lines.append(f"| prune | {cap} | d={d} | {actual:+.3f} | " + " | ".join(f"{v[m]:+.3f} ({abs(v[m]-actual):.3f})" for m in ("power", "A2", "A1_gamma1", "strength_only", "median_curve", "zero")) + " |")
for k, v in P["quantization"].items():
    cap, b = k.split("|"); b = int(b)
    actual = qt[f"{b}"][cap] - qt["dense"][cap]
    rec = {"cap": cap, "bit": b, "actual": actual, **{m: {"pred": p, "signed": p - actual, "abs": abs(p - actual)} for m, p in v.items()}}
    out["quantization"].append(rec)
    lines.append(f"| quant | {cap} | int{b} | {actual:+.3f} | " + " | ".join(f"{v[m]:+.3f} ({abs(v[m]-actual):.3f})" for m in ("full_N0_L0_D0", "noD0_N0_L0", "per_bit_median", "zero")) + " | | |")
def mae(arm, key, filt):
    xs = [r[key]["abs"] for r in out[arm] if filt(r)]; return sum(xs) / len(xs) if xs else float("nan")
lines.append("\n## MAE by config (over 3 capabilities)")
for d in (0.65, 0.55):
    lines.append(f"- prune d={d}: " + ", ".join(f"{m}={mae('pruning', m, lambda r: r['d']==d):.3f}" for m in ("power", "A2", "A1_gamma1", "strength_only", "median_curve", "zero")))
for b in (4, 3):
    lines.append(f"- quant int{b}: " + ", ".join(f"{m}={mae('quantization', m, lambda r: r['bit']==b):.3f}" for m in ("full_N0_L0_D0", "noD0_N0_L0", "per_bit_median", "zero")))
(R / "v46-p1-newsource/compare.json").write_text(json.dumps(out, indent=2))
(Path(__file__).resolve().parents[1] / "paper/docs/P1_NEWSOURCE.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
