#!/usr/bin/env python3
"""Closeout Package A (RETROSPECTIVE DIAGNOSTIC): pruning same-input baselines.

Question: how much of v40's win comes from SOURCE information vs the shared power shape? Is the shared
power necessary once the same source inputs are available?

v40's frozen candidate stays immutable (results/v40-prune-strength/register.json). Here we add EXACTLY the
two same-input baselines the closeout specifies, both fit on the SAME dev / phi / units as v40:
  A1: source-conditioned model with gamma_c FIXED to 1, beta REFIT on dev (not v40's beta with gamma swapped).
  A2: per-density source regression + predefined linear interp/extrap.
      For each old density d0 in {0.9,0.8,0.7,0.6}, fit beta_{c,d0}.phi(x) to Delta L at d0 (K0, dev only).
      Predict the target source at each old density, then predict unseen densities by LINEAR-in-d interp/extrap
      through the PREDICTED d=0.7 and d=0.6 values (never using the target source's measured compression points).
      d=0.65 = interpolation, d=0.55 = extrapolation.
Baselines strength-only / median-curve / zero reproduced from v40 for reference. Output = per-source-state
signed per-point errors, per-cap MAE, and paired improvement = MAE(baseline) - MAE(candidate).
"""
import json, itertools
from pathlib import Path
import numpy as np
import analysis.v40_prune_strength_axis as v40
import analysis.v36_pythia_controlled_fit as v36

ROOT = v40.ROOT
OUT = ROOT / "results/v42-prune-sameinput"
REPORT = ROOT / "paper/docs/PRUNE_SAMEINPUT.md"
CAPS = v40.CAPS
SEEN_D = v40.SEEN_D
OLD_D_FOR_INTERP = (0.7, 0.6)  # anchor densities for A2 linear interp/extrap

def _phi_z(size, step, cap, center, scale):
    return np.concatenate([[1.0], (v40._phi_raw(size, step, cap) - center) / scale])

def _fit_A1(rows, cap, center, scale):
    """gamma fixed to 1; refit beta on dev via linear LS."""
    X = np.array([_phi_z(r["size"], r["step"], cap, center, scale) * v40._shape(r["d"], 1.0)
                  for r in rows if r["cap"] == cap])
    y = np.array([r["y"] for r in rows if r["cap"] == cap])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

def _predict_A1(size, step, cap, d, beta, center, scale):
    return float(_phi_z(size, step, cap, center, scale) @ beta) * v40._shape(d, 1.0)

def _fit_A2(rows, cap, center, scale):
    """One source regression per old density."""
    betas = {}
    for d0 in SEEN_D:
        X = np.array([_phi_z(r["size"], r["step"], cap, center, scale)
                      for r in rows if r["cap"] == cap and r["d"] == d0])
        y = np.array([r["y"] for r in rows if r["cap"] == cap and r["d"] == d0])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        betas[d0] = b
    return betas

def _predict_A2(size, step, cap, d, betas, center, scale):
    z = _phi_z(size, step, cap, center, scale)
    p7 = float(z @ betas[0.7]); p6 = float(z @ betas[0.6])
    if d in betas:  # seen density: direct source-regression prediction
        return float(z @ betas[d])
    slope = (p6 - p7) / (0.6 - 0.7)      # linear in d through predicted (0.7,0.6)
    return p7 + slope * (d - 0.7)

def main():
    rows = v40.load_dev()
    phi = np.array([r["phi"] for r in rows]); center, scale = phi.mean(0), phi.std(0)
    reg = json.loads((v40.OUT / "register.json").read_text())  # v40 frozen candidate
    A1 = {c: _fit_A1(rows, c, center, scale) for c in CAPS}
    A2 = {c: _fit_A2(rows, c, center, scale) for c in CAPS}
    per_point, agg = [], {c: {k: [] for k in ("cand", "A1", "A2", "so", "med", "zero")} for c in CAPS}
    for size, step, densities, is_new in v40.PANEL:
        for cap in CAPS:
            for d in densities:
                try:
                    obs = v40._delta(size, step, d, cap)
                except (FileNotFoundError, KeyError):
                    continue
                cand = v40._predict_cap(reg, cap, size, step, d)
                a1 = _predict_A1(size, step, cap, d, A1[cap], center, scale)
                a2 = _predict_A2(size, step, cap, d, A2[cap], center, scale)
                so = reg["caps"][cap]["strength_only"]; sop = so["A"] * v40._shape(d, so["gamma"])
                mc = reg["caps"][cap]["median_curve"]; mcp = mc["c"] * v40._shape(d, mc["p"])
                kind = "interp(0.65)" if d == 0.65 else ("extrap(0.55)" if d == 0.55 else f"seen({d:g})")
                per_point.append({"source": f"{size}@{step}", "new_source": is_new, "cap": cap, "d": d,
                                  "kind": kind, "obs": obs, "cand": cand, "A1": a1, "A2": a2,
                                  "signed_err_cand": cand - obs, "abs_err_cand": abs(cand - obs),
                                  "abs_err_A1": abs(a1 - obs), "abs_err_A2": abs(a2 - obs)})
                for k, v in (("cand", cand), ("A1", a1), ("A2", a2), ("so", sop), ("med", mcp), ("zero", 0.0)):
                    agg[cap][k].append((abs(v - obs), size, step, d, is_new))
    OUT.mkdir(parents=True, exist_ok=True)
    def mae(cap, k, filt=None):
        xs = [e[0] for e in agg[cap][k] if (filt is None or filt(e))]
        return float(np.mean(xs)) if xs else float("nan")
    summary = {"note": "RETROSPECTIVE DIAGNOSTIC; v40 candidate frozen; A1(gamma=1 refit)/A2(per-density+interp) are same-input baselines",
               "n_points": len(per_point), "by_cap": {}, "per_point": per_point}
    lines = ["# Pruning same-input baselines (Closeout Package A, retrospective diagnostic)\n",
             "v40 candidate is frozen; A1 = source-conditioned with gamma fixed to 1 (beta refit on dev);",
             "A2 = per-density source regression + linear-in-d interp(0.65)/extrap(0.55). Same phi/dev/units.\n",
             "## Per-capability MAE (full test set) and paired improvement vs same-input baselines",
             "| cap | candidate | A1(g=1) | A2(per-d) | strength-only | median | zero | cand vs A1 | cand vs A2 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cap in CAPS:
        m = {k: mae(cap, k) for k in ("cand", "A1", "A2", "so", "med", "zero")}
        imp_a1 = m["A1"] - m["cand"]; imp_a2 = m["A2"] - m["cand"]
        summary["by_cap"][cap] = {"mae": m, "improvement_vs_A1": imp_a1, "improvement_vs_A2": imp_a2,
            "mae_no_maxdamage": {k: mae(cap, k, lambda e: not (e[1] == "160m" and e[2] == 143000)) for k in ("cand", "A1", "A2")}}
        lines.append(f"| {cap} | **{m['cand']:.3f}** | {m['A1']:.3f} | {m['A2']:.3f} | {m['so']:.3f} | "
                     f"{m['med']:.3f} | {m['zero']:.3f} | {imp_a1:+.3f} | {imp_a2:+.3f} |")
    lines.append("\n(improvement = MAE(baseline) - MAE(candidate); positive = candidate better)\n")
    lines.append("## Signed per-point errors (candidate) — for correct over/under-prediction wording")
    lines.append("| source | cap | d | kind | obs | cand | signed(cand-obs) | A1 |A2 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for p in per_point:
        lines.append(f"| {p['source']} | {p['cap']} | {p['d']:g} | {p['kind']} | {p['obs']:+.3f} | "
                     f"{p['cand']:+.3f} | {p['signed_err_cand']:+.3f} | {p['A1']:+.3f} | {p['A2']:+.3f} |")
    lines.append("\n## Sensitivity: MAE excluding the max-damage source (160m@143k)")
    lines.append("| cap | candidate | A1 | A2 |")
    lines.append("|---|---|---|---|")
    for cap in CAPS:
        n = summary["by_cap"][cap]["mae_no_maxdamage"]
        lines.append(f"| {cap} | {n['cand']:.3f} | {n['A1']:.3f} | {n['A2']:.3f} |")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:20])); print("...\nwrote", OUT / "summary.json", "and", REPORT)

if __name__ == "__main__":
    main()
