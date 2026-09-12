#!/usr/bin/env python3
"""P1-v2 compare for one new source tag: frozen predictions (protocols A and B) vs measured pruning (7 densities)
and RTN (5 bits). Per cap/config: actual, prediction per candidate, signed, |err|; MAE per candidate split by
density regime (interp: 0.9-0.6 incl. 0.75/0.65 vs deep extrap: 0.55) and per bit (5-bit = interpolation-rule
baseline). Writes results/v49-p1v2/compare_<tag>.json and paper/docs/P1V2_<tag>.md."""
import sys, json
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results/v49-p1v2"
tag = sys.argv[1]; tagf = tag.replace("@", "--")
P = json.loads((OUT / f"predictions_{tag}.json").read_text())
pr = json.loads((ROOT / f"results/v6-capability-geometry/{tagf}/prune_losses.json").read_text())
qt = json.loads((ROOT / f"results/v10-quantization/{tagf}/quant_losses.json").read_text())
CAPS = ("math", "code", "qa"); PC = ("power", "A2", "A1", "cont", "strength_only", "median_curve", "zero"); QC = ("full", "noD0", "per_bit_mean", "per_bit_median", "zero")
out = {"tag": tag, "dense_frozen": P["target"]["dense"], "dense_prune_run": pr["1.0"], "dense_quant_run": qt["dense"], "protocols": {}}
L = [f"# P1-v2 new-source frozen prospective: {tag} (P-new)\n", "Predictions committed before measurement (see git log). Signed = pred − actual. MAE over 3 capabilities.\n"]
for proto, res in P["protocols"].items():
    rows_p, rows_q = [], []
    for k, v in res["prune"].items():
        cap, d = k.split("|"); d = float(d); act = pr[f"{d:g}"][cap] - pr["1.0"][cap]
        rows_p.append({"cap": cap, "d": d, "actual": act, **{c: {"pred": v[c], "signed": v[c] - act, "abs": abs(v[c] - act)} for c in PC}})
    for k, v in res["quant"].items():
        cap, b = k.split("|"); b = int(b); act = qt[str(b)][cap] - qt["dense"][cap]
        rows_q.append({"cap": cap, "bit": b, "actual": act, "rule": v.get("rule"), **{c: {"pred": v[c], "signed": v[c] - act, "abs": abs(v[c] - act)} for c in QC}})
    def mae(rows, c, f): xs = [r[c]["abs"] for r in rows if f(r)]; return float(np.mean(xs)) if xs else float("nan")
    summ = {"prune": {"interp_0.9-0.6": {c: mae(rows_p, c, lambda r: r["d"] >= 0.6) for c in PC}, "extrap_0.55": {c: mae(rows_p, c, lambda r: r["d"] == 0.55) for c in PC},
                      "per_density": {f"{d:g}": {c: mae(rows_p, c, lambda r, d=d: r["d"] == d) for c in PC} for d in (0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55)}},
            "quant": {f"int{b}": {c: mae(rows_q, c, lambda r, b=b: r["bit"] == b) for c in QC} for b in (8, 6, 5, 4, 3)},
            "quant_ge4": {c: mae(rows_q, c, lambda r: r["bit"] >= 4) for c in QC}}
    out["protocols"][proto] = {"rows_prune": rows_p, "rows_quant": rows_q, "summary": summ}
    L.append(f"## Protocol {proto} ({'full dev' if proto=='A' else 'dev step<=64k only'})")
    L.append("| regime | " + " | ".join(PC) + " |"); L.append("|---" * (len(PC) + 1) + "|")
    for reg in ("interp_0.9-0.6", "extrap_0.55"): L.append(f"| prune {reg} | " + " | ".join(f"{summ['prune'][reg][c]:.3f}" for c in PC) + " |")
    L.append("\n| bits | " + " | ".join(QC) + " |"); L.append("|---" * (len(QC) + 1) + "|")
    for b in (8, 6, 5, 4, 3): L.append(f"| int{b}{' (interp-rule)' if b==5 else ''} | " + " | ".join(f"{summ['quant'][f'int{b}'][c]:.3f}" for c in QC) + " |")
    L.append(f"| ≥4-bit | " + " | ".join(f"{summ['quant_ge4'][c]:.3f}" for c in QC) + " |")
    L.append("\nper-point prune (cap, d, actual, power, A2, cont):"); 
    for r in rows_p: L.append(f"- {r['cap']} d={r['d']:g}: act {r['actual']:+.3f} | power {r['power']['pred']:+.3f} | A2 {r['A2']['pred']:+.3f} | cont {r['cont']['pred']:+.3f}")
    L.append("")
(OUT / f"compare_{tag}.json").write_text(json.dumps(out, indent=2, default=float)); (ROOT / "paper/docs" / f"P1V2_{tagf}.md").write_text("\n".join(L) + "\n")
print("\n".join(L[:40])); print("wrote", OUT / f"compare_{tag}.json")
