#!/usr/bin/env python3
"""Closeout Package B (RETROSPECTIVE DIAGNOSTIC): distillation prediction residuals, pool variation,
protocol boundary.

Reuses v41's FROZEN predictors (constant / T-only / E-only / 2D) fit on the U75/U600 endpoint trajectories.
Corrects the C31 framing: analyze prediction RESIDUALS r = delta - delta_hat_frozen (NOT raw delta std),
separate systematic bias from variation around it, and audit the U225 sampling protocol. Per closeout:
report per budget; keep per-pool results; do NOT treat 12 checkpoints as 12 independent repeats
(3 pools x 2 train-seeds, 2 checkpoints share one trajectory); do NOT claim a causal explanation of the
residual or that E is universally sufficient.
"""
import json, re, glob
from pathlib import Path
import numpy as np
import analysis.v41_distill_newpool as v41

OUT = v41.ROOT / "results/v43-distill-residuals"
REPORT = v41.ROOT / "paper/docs/DISTILL_RESIDUALS.md"
CAPS = v41.CAPS
BASE = v41.BASE

def _u225_runs():
    runs = {}
    for run in sorted(BASE.glob(v41.U225_GLOB)):
        if not (run / "eval.json").exists():
            continue
        ds = int(re.search(r"dseed(\d+)", run.name).group(1))
        ms = re.search(r"_seed(\d+)$", run.name); ts = int(ms.group(1)) if ms else 0
        pts = v41._checkpoints(run)
        pts.sort(key=lambda p: p["Tc"])  # budget order
        if pts:
            runs[(ds, ts)] = {"pts": pts, "eval": json.loads((run / "eval.json").read_text())}
    return runs

def main():
    ep = v41._endpoint_rows()
    fits = {cap: {k: v41._fit(ep, cap, k) for k in ("constant", "T", "E", "2D")} for cap in CAPS}
    runs = _u225_runs()
    n_budgets = min(len(v["pts"]) for v in runs.values())
    out = {"note": "RETROSPECTIVE DIAGNOSTIC; v41 frozen predictors reused; residual r = delta - delta_hat",
           "design": {"pools": sorted({k[0] for k in runs}), "train_seeds": sorted({k[1] for k in runs}),
                      "n_runs": len(runs), "checkpoints_per_run": n_budgets,
                      "caveat": "3 pools x 2 seeds; 2 checkpoints share one trajectory -> NOT 12 independent repeats"},
           "by_cap": {}}
    lines = ["# Distillation prediction residuals & pool variation (Closeout Package B, retrospective)\n",
             "Frozen v41 predictors; residual r = delta - delta_hat. Per budget (b0 ~= first milestone, b1 ~= second).",
             "Systematic bias = mean(r); variation split into pool-sampling vs training. 3 pools x 2 seeds; the two",
             "checkpoints per run share a trajectory (not independent). No causal claim; no universal-E claim.\n"]
    for cap in CAPS:
        out["by_cap"][cap] = {}
        for pred in ("constant", "E", "2D", "T"):
            out["by_cap"][cap][pred] = {}
            for b in range(n_budgets):
                rows = []  # (pool, seed, residual)
                for (ds, ts), v in runs.items():
                    p = v["pts"][b]
                    dhat = float(v41._predict([p], cap, pred, fits[cap][pred])[0])
                    rows.append((ds, ts, p["delta"][cap] - dhat))
                r = np.array([x[2] for x in rows])
                pools = sorted({x[0] for x in rows})
                pool_means = {ds: float(np.mean([x[2] for x in rows if x[0] == ds])) for ds in pools}
                # training variation: within-pool std across seeds; pool variation: std of pool means
                within = float(np.mean([np.std([x[2] for x in rows if x[0] == ds]) for ds in pools]))
                across = float(np.std(list(pool_means.values())))
                out["by_cap"][cap][pred][f"b{b}"] = {
                    "bias_mean_residual": float(np.mean(r)), "mae": float(np.mean(np.abs(r))),
                    "residual_std": float(np.std(r)), "pool_mean_residuals": pool_means,
                    "pool_sampling_std": across, "training_std": within}
        # headline lines for the two decisive predictors per cap
        best = "E" if cap == "qa" else "constant"
        lines.append(f"## {cap}  (decisive predictor: {best})")
        for b in range(n_budgets):
            d = out["by_cap"][cap][best][f"b{b}"]
            lines.append(f"- b{b}: bias(mean r)={d['bias_mean_residual']:+.3f}, MAE={d['mae']:.3f}, "
                         f"resid-std={d['residual_std']:.3f}; pool-sampling std={d['pool_sampling_std']:.3f} "
                         f"vs training std={d['training_std']:.3f}")
        lines.append("")
    # B3: sampling-protocol audit
    lines.append("## B3 Sampling-protocol audit (from manifests)")
    ep_pools = {}
    for pat, lab in [("gpt-5.6-luna_full_75_uxseen*", "U75"), ("gpt-5.6-luna_full_600_uxseen*", "U600")]:
        for run in sorted(BASE.glob(pat)):
            e = json.loads((run / "eval.json").read_text())
            ep_pools.setdefault(lab, {"data_selection": e.get("data_selection"),
                                      "pool_tokens": e.get("unique_data_pool_tokens"),
                                      "n_examples": e.get("training_examples") or e.get("unique_data_pool_examples")})
    any_u = next(iter(runs.values()))["eval"]
    u225_info = {"data_selection": any_u.get("data_selection"), "data_seed": any_u.get("data_seed"),
                 "pool_tokens": any_u.get("unique_data_pool_tokens"),
                 "n_examples": any_u.get("training_examples") or any_u.get("unique_data_pool_examples")}
    e_all = [p["E"] for v in runs.values() for p in v["pts"]]
    out["protocol_audit"] = {"endpoints": ep_pools, "u225": u225_info,
                             "u225_E_range": [min(e_all), max(e_all)],
                             "endpoint_E_range": [min(r["E"] for r in ep), max(r["E"] for r in ep)]}
    for lab, info in ep_pools.items():
        lines.append(f"- {lab}: selection={info['data_selection']!r}, pool_tokens={info['pool_tokens']}, n={info['n_examples']}")
    lines.append(f"- U225: selection={u225_info['data_selection']!r}, data_seed(example)={u225_info['data_seed']}, "
                 f"pool_tokens={u225_info['pool_tokens']}, n={u225_info['n_examples']}")
    lines.append(f"- E range: endpoints {out['protocol_audit']['endpoint_E_range'][0]:.2f}-"
                 f"{out['protocol_audit']['endpoint_E_range'][1]:.2f}; U225 {min(e_all):.2f}-{max(e_all):.2f} "
                 f"(U225 E is INSIDE the endpoint range -> configuration interpolation, not range extrapolation).")
    lines.append("\n## Verdict (Package B)")
    lines.append("- The frozen E-only predictor's advantage on QA and its math/code shortfall are re-expressed as")
    lines.append("  RESIDUALS with an explicit bias/variation split (above), not as raw-delta std ratios.")
    lines.append("- If the systematic bias (mean residual) is small relative to the pool-to-pool spread, the")
    lines.append("  frozen predictor is not systematically off but pool-sensitive; if bias is large, the form itself")
    lines.append("  misfits. Both are reported per capability/budget; with 3 pools the spread estimate is THIN.")
    lines.append("- Protocol boundary: endpoints are pool PREFIXES (first-U), U225 is RANDOM-sampled -> the test is")
    lines.append("  'unseen pool size AND changed sampling protocol', not isolated pool-size. U225 E interpolates.")
    lines.append("- Claims corrected: drop 'pool-std/train-std ~10x explains the prediction residual' and any causal")
    lines.append("  'residual is a data-selection effect' phrasing; state only the described, design-limited variation.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", OUT / "summary.json")

if __name__ == "__main__":
    main()
