#!/usr/bin/env python3
"""Next-round main line: connect the SOURCE axis to the COMPRESSION-STRENGTH axis for pruning.

Frozen shared-shape candidate (advisor round 6):
    dL_hat_c(x, d) = A_c(x) * ((1-d)/0.3)^gamma_c ,   A_c(x) = beta_c . [1, z(logN0), z(L_c0), z(logD0)]
- gamma_c shared across densities per capability; amplitude A_c(x) is signed and source-dependent.
- Standardization and all coefficients are fit on DEV ONLY (the 9-state grid at seen densities
  {0.9,0.8,0.7,0.6}). The key improvement over v36 (per-config indicators) is that a SINGLE shape is
  shared across densities, so the candidate makes an explicit prediction for UNSEEN densities.

Panel (predict mode), source x compression-strength:
    seen-source/unseen-density : 1.4b@64k, 160m@143k at d in {0.65 (interp), 0.55 (deeper extrap)}
    new-source /seen-density   : 410m@96k at d in {0.8, 0.6}
    new-source /unseen-density : 410m@96k at d in {0.65, 0.55}
Domain PRE-DECLARED: candidate is declared for the smooth pre-cliff region; d=0.55 is flagged in ADVANCE
as a deeper extrapolation that may reach the cliff. Per advisor, held-out points are NOT dropped after
seeing collapse; a collapse at 0.55 is reported as a pre-declared domain-boundary result.

Baselines (all compression-strength-only, no source dependence): zero-change; strength-only (one global
amplitude A + gamma per cap); median-strength-curve (fit c,p to per-density DEV medians, extrapolate).

Modes:
  fit      -> fit on dev, write results/v40-prune-strength/register.json (frozen coeffs + gamma + z-stats). No GPU.
  predict  -> after v6 forward-only pruning at the new (source,density) points exists, score candidate vs
              baselines on each panel cell; per-cell errors; write compare.json + PRUNE_STRENGTH_AXIS.md.
"""
import json, sys, itertools
from pathlib import Path
import numpy as np
import analysis.v36_pythia_controlled_fit as v36

ROOT = v36.ROOT
OUT = ROOT / "results/v40-prune-strength"
REPORT = ROOT / "paper/docs/PRUNE_STRENGTH_AXIS.md"
CAPS = ("math", "code", "qa")
SEEN_D = (0.9, 0.8, 0.7, 0.6)
NEW_D = (0.65, 0.55)                 # 0.65 interpolation, 0.55 deeper extrapolation (pre-declared)
DEV_SIZES = ("160m", "410m", "1.4b")
DEV_STEPS = (16000, 64000, 143000)
GAMMA_GRID = np.round(np.arange(0.4, 4.01, 0.05), 3)
PANEL = [  # (size, step, densities_to_predict, is_new_source)
    ("1.4b", 64000, NEW_D, False),
    ("160m", 143000, NEW_D, False),
    ("410m", 96000, (0.8, 0.6) + NEW_D, True),
]

def _prune_path(size, step):
    return ROOT / f"results/v6-capability-geometry/pythia-{size}--step{step}/prune_losses.json"

def _delta(size, step, d, cap):
    t = json.loads(_prune_path(size, step).read_text())
    return t[f"{d:g}"][cap] - t["1.0"][cap]

def _phi_raw(size, step, cap):
    return np.array([np.log(v36.matrix_n0(size)),
                     json.loads(_prune_path(size, step).read_text())["1.0"][cap],
                     np.log(step * v36.TOKENS_PER_STEP)])

def load_dev():
    rows = []
    for size, step in itertools.product(DEV_SIZES, DEV_STEPS):
        for d, cap in itertools.product(SEEN_D, CAPS):
            rows.append({"size": size, "step": step, "d": d, "cap": cap,
                         "phi": _phi_raw(size, step, cap), "y": _delta(size, step, d, cap)})
    return rows

def _shape(d, gamma):
    return ((1 - d) / 0.3) ** gamma

def _fit_cap(rows, center, scale):
    """Return best gamma, beta (4,), and dev sse for one capability's shared-shape candidate."""
    Z = np.array([np.concatenate([[1.0], (r["phi"] - center) / scale]) for r in rows])  # (n,4)
    y = np.array([r["y"] for r in rows])
    best = None
    for g in GAMMA_GRID:
        s = np.array([_shape(r["d"], g) for r in rows])
        X = Z * s[:, None]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(np.sum((X @ beta - y) ** 2))
        if best is None or sse < best[2]:
            best = (float(g), beta, sse)
    return best

def _fit_strength_only(rows, cap):
    """Global amplitude A and gamma (no source dependence)."""
    y = np.array([r["y"] for r in rows if r["cap"] == cap])
    ds = np.array([r["d"] for r in rows if r["cap"] == cap])
    best = None
    for g in GAMMA_GRID:
        s = _shape(ds, g)
        A = float(np.sum(s * y) / np.sum(s * s))
        sse = float(np.sum((A * s - y) ** 2))
        if best is None or sse < best[2]:
            best = (float(g), A, sse)
    return {"gamma": best[0], "A": best[1]}

def _fit_median_curve(rows, cap):
    """Per-density DEV median, then log-log fit median ~ c*((1-d)/0.3)^p for interp/extrapolation."""
    meds = {d: float(np.median([r["y"] for r in rows if r["cap"] == cap and r["d"] == d])) for d in SEEN_D}
    x = np.log(np.array([(1 - d) / 0.3 for d in SEEN_D]))
    # signed medians: fit in linear space on the shape basis to keep sign
    best = None
    for p in GAMMA_GRID:
        s = np.array([_shape(d, p) for d in SEEN_D])
        yv = np.array([meds[d] for d in SEEN_D])
        c = float(np.sum(s * yv) / np.sum(s * s))
        sse = float(np.sum((c * s - yv) ** 2))
        if best is None or sse < best[2]:
            best = (float(p), c, sse)
    return {"p": best[0], "c": best[1], "seen_medians": meds}

def fit():
    rows = load_dev()
    phi = np.array([r["phi"] for r in rows])
    center, scale = phi.mean(0), phi.std(0)
    reg = {"model": "dL_c = (beta_c . [1,z(logN0),z(L0),z(logD0)]) * ((1-d)/0.3)^gamma_c",
           "dev": {"sizes": DEV_SIZES, "steps": DEV_STEPS, "seen_densities": SEEN_D, "n_rows": len(rows)},
           "phi_center": center.tolist(), "phi_scale": scale.tolist(),
           "domain_declaration": "smooth pre-cliff; d=0.65 interpolation, d=0.55 deeper extrapolation (may hit cliff); held-out points are NOT dropped after observing collapse",
           "caps": {}}
    for cap in CAPS:
        cr = [r for r in rows if r["cap"] == cap]
        g, beta, sse = _fit_cap(cr, center, scale)
        reg["caps"][cap] = {"gamma": g, "beta": beta.tolist(), "dev_sse": sse,
                            "strength_only": _fit_strength_only(rows, cap),
                            "median_curve": _fit_median_curve(rows, cap)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "register.json").write_text(json.dumps(reg, indent=2))
    for cap in CAPS:
        c = reg["caps"][cap]
        print(f"{cap}: gamma={c['gamma']}, beta={[round(b,3) for b in c['beta']]}, dev_sse={c['dev_sse']:.4f}")
    print("wrote", OUT / "register.json")

def _predict_cap(reg, cap, size, step, d):
    c = reg["caps"][cap]
    center = np.array(reg["phi_center"]); scale = np.array(reg["phi_scale"])
    z = np.concatenate([[1.0], (_phi_raw(size, step, cap) - center) / scale])
    A = float(np.array(c["beta"]) @ z)
    return A * _shape(d, c["gamma"])

def predict():
    reg = json.loads((OUT / "register.json").read_text())
    out = {"register_sha": None, "panel": []}
    import hashlib
    out["register_sha"] = hashlib.sha256((OUT / "register.json").read_bytes()).hexdigest()
    lines = ["# Pruning source x compression-strength prediction (round-6 main line)\n",
             "Frozen shared-shape candidate dL_c(x,d)=A_c(x)*((1-d)/0.3)^gamma_c fit on DEV (9-state grid, "
             "seen d={0.9,0.8,0.7,0.6}); predicts UNSEEN densities. Domain pre-declared: 0.65 interp, 0.55 "
             "deeper extrap (not dropped on collapse). Baselines: zero, strength-only, median-strength-curve.\n"]
    agg = {cap: {"cand": [], "zero": [], "so": [], "med": []} for cap in CAPS}
    for size, step, densities, is_new in PANEL:
        cell = {"size": size, "step": step, "new_source": is_new, "cells": {}}
        lines.append(f"## pythia-{size}@step{step} ({'NEW source' if is_new else 'seen source'})")
        for cap in CAPS:
            for d in densities:
                try:
                    obs = _delta(size, step, d, cap)
                except (FileNotFoundError, KeyError):
                    lines.append(f"- {cap} d={d}: MEASUREMENT MISSING")
                    continue
                cand = _predict_cap(reg, cap, size, step, d)
                so = reg["caps"][cap]["strength_only"]; sop = so["A"] * _shape(d, so["gamma"])
                mc = reg["caps"][cap]["median_curve"]; mcp = mc["c"] * _shape(d, mc["p"])
                cell["cells"][f"{cap}|{d:g}"] = {"observed": obs, "candidate": cand,
                                                 "strength_only": sop, "median_curve": mcp}
                for k, v in (("cand", cand), ("so", sop), ("med", mcp), ("zero", 0.0)):
                    agg[cap][k].append(abs(obs - v))
                kind = "interp" if d == 0.65 else ("extrap" if d == 0.55 else "seen-d")
                lines.append(f"- {cap} d={d} ({kind}): obs={obs:+.3f} cand={cand:+.3f} "
                             f"strength-only={sop:+.3f} median={mcp:+.3f} | |err| cand={abs(obs-cand):.3f}")
        out["panel"].append(cell)
        lines.append("")
    lines.append("## Aggregate MAE (all predicted cells)")
    lines.append("| cap | candidate | strength-only | median-curve | zero |")
    lines.append("|---|---|---|---|---|")
    for cap in CAPS:
        a = agg[cap]
        m = lambda k: (np.mean(a[k]) if a[k] else float("nan"))
        lines.append(f"| {cap} | **{m('cand'):.3f}** | {m('so'):.3f} | {m('med'):.3f} | {m('zero'):.3f} |")
    out["aggregate_mae"] = {cap: {k: (float(np.mean(agg[cap][k])) if agg[cap][k] else None)
                                  for k in ("cand", "so", "med", "zero")} for cap in CAPS}
    (OUT / "compare.json").write_text(json.dumps(out, indent=2, default=float))
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", OUT / "compare.json", "and", REPORT)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fit"
    (fit if mode == "fit" else predict)()
