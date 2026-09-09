#!/usr/bin/env python3
"""P2: distillation source-transfer test on the controlled Pythia LoRA panel.

Parallel to v36 (pruning/quant) but for distillation delta_c = L_c(S_KD) - L_c(S0). There is NO
compression config here (one fixed distillation setting: teacher gpt-5.6-luna, recipe full, pool
n=600, epochs=2, strict LoRA, seed 0), so the predictor is a single linear model per capability,
NOT config-indicators:

    delta_hat_c = beta0 + beta1 z(log N0) + beta2 z(L0) [ + beta3 z(log D0) ]

Question (advisor P2): can distillation establish a training-state prediction relationship parallel
to the other two arms? Test = does D0 lower held-out error beyond {N0,L0}, and does the model beat a
constant-delta baseline, under leave-one-size-out and leave-one-step-out over the 9 (size,step) cells.
"""
import json, itertools
from pathlib import Path
import numpy as np
import analysis.v36_pythia_controlled_fit as v36

ROOT = v36.ROOT
OUT = ROOT / "results/v39-distill-controlled"
REPORT = ROOT / "paper/docs/DISTILL_CONTROLLED.md"
SIZES = ("160m", "410m", "1.4b")
STEPS = (16000, 64000, 143000)
CAPS = ("math", "code", "qa")
RUN = "gpt-5.6-luna_full_600_lora"

def load_cells():
    rows = []
    for size, step in itertools.product(SIZES, STEPS):
        tag = f"pythia-{size}--step{step}"
        p = ROOT / f"results/v12-distill/{tag}/{RUN}/eval.json"
        if not p.exists():
            raise FileNotFoundError(f"missing distill cell: {p}")
        e = json.loads(p.read_text())
        assert e["training_mode"] == "lora", f"{tag} not lora"
        for cap in CAPS:
            rows.append({"size": size, "step": step, "cap": cap,
                         "N0": v36.matrix_n0(size), "D0": step * v36.TOKENS_PER_STEP,
                         "L0": e["dense"][cap], "delta": e["delta"][cap]})
    return rows

def _design(rows, with_d0, center, scale):
    cols = [np.log([r["N0"] for r in rows]), [r["L0"] for r in rows]]
    if with_d0:
        cols.append(np.log([r["D0"] for r in rows]))
    raw = np.array(cols, float).T
    z = (raw - center) / scale
    return np.column_stack([np.ones(len(rows)), z])

def _fit(rows, with_d0):
    cols = [np.log([r["N0"] for r in rows]), [r["L0"] for r in rows]]
    if with_d0:
        cols.append(np.log([r["D0"] for r in rows]))
    raw = np.array(cols, float).T
    center, scale = raw.mean(0), raw.std(0)
    if np.any(scale <= 0):
        raise ValueError("no covariate variation")
    x = _design(rows, with_d0, center, scale)
    y = np.array([r["delta"] for r in rows])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return {"with_d0": with_d0, "center": center, "scale": scale, "coef": coef}

def _predict(fit, rows):
    return _design(rows, fit["with_d0"], fit["center"], fit["scale"]) @ fit["coef"]

def _cv(rows, group):
    groups = sorted({r[group] for r in rows})
    recs = []
    for g in groups:
        tr = [r for r in rows if r[group] != g]
        te = [r for r in rows if r[group] == g]
        fitA, fitB = _fit(tr, False), _fit(tr, True)
        mean_d = float(np.mean([r["delta"] for r in tr]))
        med_d = float(np.median([r["delta"] for r in tr]))
        for r in te:
            recs.append({**r, "held": g, "pA": None, "pB": None})
        pA, pB = _predict(fitA, te), _predict(fitB, te)
        for r, a, b in zip([x for x in recs if x["held"] == g], pA, pB):
            r["pA"], r["pB"], r["mean_base"], r["med_base"] = a, b, mean_d, med_d
    obs = np.array([r["delta"] for r in recs])
    def mae(key): return float(np.mean(np.abs(obs - np.array([r[key] for r in recs]))))
    strongest = "mean_base" if mae("mean_base") <= mae("med_base") else "med_base"
    return {"scheme": f"leave_one_{group}_out", "n": len(recs),
            "mae_noD0": mae("pA"), "mae_D0": mae("pB"),
            "mae_baseline_mean": mae("mean_base"), "mae_baseline_median": mae("med_base"),
            "strongest_baseline": strongest, "mae_baseline": mae(strongest),
            "D0_beats_noD0": mae("pB") < mae("pA"),
            "D0_beats_baseline": mae("pB") < mae(strongest),
            "per_fold": [{"held": g,
                          "mae_D0": float(np.mean([abs(r["delta"]-r["pB"]) for r in recs if r["held"]==g])),
                          "mae_noD0": float(np.mean([abs(r["delta"]-r["pA"]) for r in recs if r["held"]==g])),
                          "mae_baseline": float(np.mean([abs(r["delta"]-r[strongest]) for r in recs if r["held"]==g]))}
                         for g in groups]}

def main():
    rows = load_cells()
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"panel": {"sizes": SIZES, "steps": STEPS, "run": RUN, "n_cells": 9},
           "note": "distillation delta_c = L_c(S_KD)-L_c(S0); single-setting linear predictor, no config indicators",
           "by_cap": {}}
    lines = ["# Distillation controlled-panel source-transfer test (P2)\n",
             f"Panel: pythia-{{{','.join(SIZES)}}} x step{{{','.join(map(str,STEPS))}}} = 9 cells, run `{RUN}`.",
             "Endpoint delta_c = L_c(S_KD) - L_c(S0). Predictor: linear in {N0,L0,[D0]}, no config indicators.",
             "Question: does D0 add held-out predictive value for delta beyond {N0,L0}, and does the model",
             "beat a constant-delta baseline? (parallel to pruning/quant v36).\n"]
    for cap in CAPS:
        cr = [r for r in rows if r["cap"] == cap]
        res = {g: _cv(cr, g) for g in ("size", "step")}
        out["by_cap"][cap] = res
        lines.append(f"## {cap}")
        for g in ("size", "step"):
            r = res[g]
            lines.append(f"- **leave-one-{g}-out**: noD0 MAE={r['mae_noD0']:.3f}, +D0 MAE={r['mae_D0']:.3f}, "
                         f"baseline({r['strongest_baseline']}) MAE={r['mae_baseline']:.3f} "
                         f"-> D0{'✓' if r['D0_beats_noD0'] else '✗'}beats-noD0, "
                         f"{'✓' if r['D0_beats_baseline'] else '✗'}beats-baseline")
        lines.append("")
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", OUT / "summary.json", "and", REPORT)

if __name__ == "__main__":
    main()
