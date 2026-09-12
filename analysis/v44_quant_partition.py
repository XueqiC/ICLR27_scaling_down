#!/usr/bin/env python3
"""Closeout Package C (RETROSPECTIVE DIAGNOSTIC): quantization partition error table.

Organizes the EXISTING frozen {N0,L0,D0} config-indicator predictor and existing simple baselines (zero,
train per-bit mean/median, no-D0) on the new @step96000 source-states, per bit, with signed bias and paired
improvement vs the strongest compatible baseline. No new predictor, no risk classifier, no new threshold model.
Distinguishes v38-measured (160M/1.4B@96k) from v40-first-measured (410M@96k); does not merge versions into one
prospective. Wording per closeout: state the discrete-config validation RANGE, keep int3 as real data.
"""
import json, itertools
from pathlib import Path
import numpy as np
import analysis.v36_pythia_controlled_fit as v36

OUT = v36.ROOT / "results/v44-quant-partition"
REPORT = v36.ROOT / "paper/docs/QUANT_PARTITION.md"
ARM = "quantization"
BITS = (8, 6, 4, 3)
NEW = [("160m", 96000, "v38"), ("1.4b", 96000, "v38"), ("410m", 96000, "v40")]

def _q(size, step):
    return json.loads((v36.ROOT / f"results/v10-quantization/pythia-{size}--step{step}/quant_losses.json").read_text())

def main():
    train, _ = v36.load_grid(sizes=("160m", "410m", "1.4b"))
    out = {"note": "RETROSPECTIVE DIAGNOSTIC; frozen {N0,L0,D0} predictor + existing baselines; per bit/source/version",
           "rows": []}
    lines = ["# Quantization partition error table (Closeout Package C, retrospective)\n",
             "Frozen config-indicator {N0,L0,D0} predictor vs simple baselines on new @step96000 sources, per bit.",
             "int3 is real measured data. v38-measured (160M/1.4B@96k) and v40-first-measured (410M@96k) are NOT",
             "merged into one prospective. Signed bias = pred - actual.\n",
             "| source (ver) | bit | actual dL | pred | signed bias | |err| cand | |err| median-base | improve |",
             "|---|---|---|---|---|---|---|---|"]
    agg = {}  # (cap, region) -> lists
    for size, step, ver in NEW:
        q = _q(size, step); dense = q["dense"]
        for cap in v36.CAPS:
            tr = [r for r in train if r["arm"] == ARM and r["capability"] == cap]
            fitB = v36.fit_direct(tr, input_fields=("N0", "L0", "D0"))
            meds = {b: float(np.median([r["observed"] for r in tr if r["config"] == b])) for b in BITS}
            for b in BITS:
                obs = q[f"{b:g}"][cap] - dense[cap]
                row = {"arm": ARM, "capability": cap, "config": b, "N0": v36.matrix_n0(size),
                       "D0": step * v36.TOKENS_PER_STEP, "L0": dense[cap]}
                pred = float(v36.predict(fitB, [v36.basic_input(row, input_fields=("N0", "L0", "D0"))])[0])
                ec, em = abs(pred - obs), abs(meds[b] - obs)
                region = "int3" if b == 3 else "ge4"
                agg.setdefault((cap, region), {"cand": [], "med": []})
                agg[(cap, region)]["cand"].append(ec); agg[(cap, region)]["med"].append(em)
                out["rows"].append({"source": f"{size}@{step}", "version": ver, "capability": cap, "bit": b,
                                    "actual": obs, "pred": pred, "signed_bias": pred - obs,
                                    "abs_err_cand": ec, "abs_err_median": em, "improvement": em - ec})
                lines.append(f"| {size}@{step} ({ver}) | {b} | {obs:+.3f} | {pred:+.3f} | {pred-obs:+.3f} | "
                             f"{ec:.3f} | {em:.3f} | {em-ec:+.3f} |")
    lines.append("\n## Partition summary (MAE, candidate vs per-bit median baseline)")
    lines.append("| cap | region | cand MAE | median MAE | improvement |")
    lines.append("|---|---|---|---|---|")
    for cap in v36.CAPS:
        for region in ("ge4", "int3"):
            a = agg.get((cap, region))
            if not a:
                continue
            cm, mm = float(np.mean(a["cand"])), float(np.mean(a["med"]))
            out.setdefault("partition", {})[f"{cap}_{region}"] = {"cand_mae": cm, "median_mae": mm, "improvement": mm - cm}
            lines.append(f"| {cap} | {region} | {cm:.3f} | {mm:.3f} | {mm-cm:+.3f} |")
    lines += ["\n## Verdict (Package C)",
        "- Validation range is the DISCRETE integer bit-widths {8,6,4,3}; there is no integer point between 3 and 4,",
        "  so this is a per-configuration validation, not a claim that quantization admits no law.",
        "- In the >=4-bit region the responses are near-zero and candidate/median errors are both tiny: at this error",
        "  scale no incremental predictive value over a simple per-bit baseline is demonstrated (not a proof of none).",
        "- int3 responses are REAL measured collapse, not an artifact; the candidate's advantage there is unstable",
        "  across sources (helps on some, misses on others) -> magnitude transfer at int3 is not established.",
        "- The fixed 4^{-b} shape is inconsistent with the measured decay under this RTN quantizer definition; a",
        "  learned exponential fits better (prior C8), reported within this tested range only.",
        "- No collapse-RISK prediction law is claimed (no independent risk model validated here).",
        "- v38 (160M/1.4B@96k) and v40 (410M@96k) are separate source measurements, not one merged prospective."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", OUT / "summary.json")

if __name__ == "__main__":
    main()
