#!/usr/bin/env python3
"""P3: distillation new-pool prediction. Fit low-DOF predictors on the ENDPOINT pools (U75, U600) and
predict a never-fit MIDDLE pool (U=225), attributing error to pool-sampling seed vs training seed.

Predictors (frozen on endpoint data, per capability):
  constant : delta_hat_c = mean(delta over endpoint checkpoints)
  T-only   : delta_hat_c = a + b*log(T_completion)                 (supervised-token budget)
  E-only   : delta_hat_c = a + b*log(E),  E = processed / pool_tokens (reuse count)
  2D       : delta_hat_c = a + b*log(E) + c*log(T_completion)      (low-DOF reuse + budget)

Endpoint fit data = U75/U600 uxseen trajectories (+ seeds): each checkpoint gives (T_completion, E, delta).
Test = the U=225 runs (data-seed {1,2,3} x train-seed {0,1}) trajectory checkpoints. U=225's E range
(~2.4-4.7) interpolates between U600 (~0.9-1.9) and U75 (~7.5-15). Per advisor: budgets aligned to actual
~140k/280k completion tokens (we use measured completion tokens, not nominal). Variance decomposition:
pool-sampling (across data-seed) vs training (across train-seed).
"""
import json, glob, itertools, re
from pathlib import Path
import numpy as np
import analysis.v36_pythia_controlled_fit as v36

ROOT = v36.ROOT
OUT = ROOT / "results/v41-distill-newpool"
REPORT = ROOT / "paper/docs/DISTILL_NEWPOOL.md"
CAPS = ("math", "code", "qa")
BASE = ROOT / "results/v12-distill/gemma3-1b"
ENDPOINT_GLOBS = ["gpt-5.6-luna_full_75_uxseen*", "gpt-5.6-luna_full_600_uxseen*"]
U225_GLOB = "gpt-5.6-luna_full_225_lora_dseed*"

def _checkpoints(run_dir):
    """(T_completion, E, delta) from a run's trajectory + final eval; skip the zero checkpoint."""
    pts = []
    for ev in sorted(glob.glob(str(run_dir / "trajectory/update-*/eval.json"))):
        d = json.loads(Path(ev).read_text())
        T, pool = d.get("completion_tokens_seen", 0), d.get("unique_data_pool_tokens", 0)
        proc = d.get("processed_tokens", 0)
        if T <= 0 or pool <= 0 or "delta" not in d:
            continue
        pts.append({"Tc": T, "E": proc / pool, "delta": d["delta"]})
    return pts

def _endpoint_rows():
    rows = []
    for pat in ENDPOINT_GLOBS:
        for run in sorted(BASE.glob(pat)):
            if not (run / "eval.json").exists():
                continue
            rows.extend(_checkpoints(run))
    return rows

def _design(pts, kind):
    lE = np.log([p["E"] for p in pts]); lT = np.log([p["Tc"] for p in pts])
    one = np.ones(len(pts))
    if kind == "constant": return np.column_stack([one])
    if kind == "T": return np.column_stack([one, lT])
    if kind == "E": return np.column_stack([one, lE])
    if kind == "2D": return np.column_stack([one, lE, lT])
    raise ValueError(kind)

def _fit(pts, cap, kind):
    X = _design(pts, kind); y = np.array([p["delta"][cap] for p in pts])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

def _predict(pts, cap, kind, beta):
    return _design(pts, kind) @ beta

def main():
    endpoint = _endpoint_rows()
    assert endpoint, "no endpoint checkpoints found"
    # U225 runs, tagged by (data_seed, train_seed)
    u225 = {}
    for run in sorted(BASE.glob(U225_GLOB)):
        if not (run / "eval.json").exists():
            continue
        m = re.search(r"dseed(\d+)", run.name); ds = int(m.group(1))
        ms = re.search(r"_seed(\d+)$", run.name); ts = int(ms.group(1)) if ms else 0
        pts = _checkpoints(run)
        if pts:
            u225[(ds, ts)] = pts
    out = {"n_endpoint_checkpoints": len(endpoint), "n_u225_runs": len(u225),
           "endpoint_E_range": [min(p["E"] for p in endpoint), max(p["E"] for p in endpoint)],
           "u225_E_range": [min(p["E"] for r in u225.values() for p in r),
                            max(p["E"] for r in u225.values() for p in r)] if u225 else None,
           "by_cap": {}}
    lines = ["# Distillation new-pool prediction (P3): endpoint pools -> unseen U=225\n",
             f"Endpoint fit checkpoints: {len(endpoint)} (U75+U600 uxseen +seeds). U=225 runs: {len(u225)} "
             f"(data-seed x train-seed). Predictors frozen on endpoints, predict U=225.\n",
             f"E range — endpoints {out['endpoint_E_range'][0]:.2f}-{out['endpoint_E_range'][1]:.2f}, "
             f"U=225 {out['u225_E_range'][0]:.2f}-{out['u225_E_range'][1]:.2f} (interpolation).\n" if u225 else "",
             "## Predictor MAE on U=225 (per capability)\n",
             "| cap | constant | T-only | E-only | 2D |", "|---|---|---|---|---|"]
    all_u225 = [p for r in u225.values() for p in r]
    for cap in CAPS:
        fits = {k: _fit(endpoint, cap, k) for k in ("constant", "T", "E", "2D")}
        mae = {}
        for k in fits:
            errs = [abs(p["delta"][cap] - float(_predict([p], cap, k, fits[k])[0])) for p in all_u225]
            mae[k] = float(np.mean(errs)) if errs else float("nan")
        # variance decomposition on U225 final-checkpoint delta: pool-seed vs train-seed
        finals = {(ds, ts): r[-1]["delta"][cap] for (ds, ts), r in u225.items()}
        pool_means = {}
        for ds in sorted({k[0] for k in finals}):
            vals = [v for (d, t), v in finals.items() if d == ds]
            pool_means[ds] = float(np.mean(vals))
        train_spread = float(np.mean([np.std([v for (d, t), v in finals.items() if d == ds]) for ds in pool_means])) if finals else float("nan")
        pool_spread = float(np.std(list(pool_means.values()))) if pool_means else float("nan")
        out["by_cap"][cap] = {"mae": mae, "pool_sampling_std": pool_spread, "training_std": train_spread,
                              "fits": {k: v.tolist() for k, v in fits.items()}}
        best = min(mae, key=mae.get)
        lines.append(f"| {cap} | {mae['constant']:.3f} | {mae['T']:.3f} | {mae['E']:.3f} | {mae['2D']:.3f} | "
                     .rstrip() + f" (best: {best})")
    lines.append("\n## Error source: pool-sampling vs training (std of final-checkpoint delta)")
    lines.append("| cap | pool-sampling std | training std |")
    lines.append("|---|---|---|")
    for cap in CAPS:
        b = out["by_cap"][cap]
        lines.append(f"| {cap} | {b['pool_sampling_std']:.3f} | {b['training_std']:.3f} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print("\nwrote", OUT / "summary.json", "and", REPORT)

if __name__ == "__main__":
    main()
