#!/usr/bin/env python3
"""V1: recipe-stratified re-test of the prior transfer-law falsification.

Prior claim (wrap-up 2026-08-16): transfer law θ_S = θ_base + η(θ_T − θ_base)
is falsified by the pre-registered r=1 kill shot (same-size self-distillation
must give Δθ=0; measured math +1.03θ, QA +0.42θ) and by D1 held-out RMSE
4–10× the noise floor.

Audit hypothesis (stats_audit.md §e): the anchor failure is driven entirely
by the M1 (answer-only) recipe, a known data-composition effect; pooled
anchor conflates it with capacity transfer. Here we re-run:
  1. anchor check stratified by recipe, with proper SE units (SD/√n);
  2. η fits per (domain, recipe), M1 separated, with cell bootstrap CI;
  3. leave-one-size-out extrapolation RMSE per recipe vs pooled,
     against the per-domain seed-SD noise floor.

Inputs: locked grid-fill IRT JSONs (item difficulties frozen 2026-07-28).
Outputs: results/v1-recipe-strat/{anchor_by_recipe.json, eta_by_recipe.json,
summary.md}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCKED = (ROOT / "prior/xueqi-intern-backup/docs/scaling-down/results/runs/"
          "irt-grid-fill-locked-2026-07-28")
OUT = ROOT / "results/v1-recipe-strat"

# wrap-up §4.2: median seed-to-seed SD per domain, the prior yardstick.
NOISE_FLOOR = {"math": 0.193, "code": 0.137, "qa": 0.117}
SIZE_B = {"0p6b": 0.6, "1p7b": 1.7, "4b": 4.0, "8b": 8.0,
          "14b": 14.0, "32b": 32.0}
RECIPE_LABEL = {"distill": "M0", "answer_only": "M1",
                "soft_kd": "M2", "onpolicy": "M3"}


def load(domain: str) -> list[dict]:
    with open(LOCKED / f"{domain}.json") as fh:
        return json.load(fh)["respondents"]


def teacher_size_b(teacher: str) -> float | None:
    t = teacher.lower()
    for tag, n in SIZE_B.items():
        if tag in t:
            return n
    return None  # minimax: size unknown


def anchor_by_recipe(rows: list[dict]) -> dict:
    """r=1 cells: student size equals teacher size; law demands Δθ=0."""
    base = {r["size"]: r["theta"] for r in rows if r["kind"] == "base"}
    out: dict[str, dict] = {}
    for r in rows:
        rec = r.get("recipe")
        if not rec or not r.get("teacher"):
            continue
        tsz = teacher_size_b(r["teacher"])
        if tsz is None or SIZE_B.get(r["size"]) != tsz:
            continue
        d = r["theta"] - base[r["size"]]
        out.setdefault(RECIPE_LABEL.get(rec, rec), {"deltas": []})[
            "deltas"].append(d)
    for rec, e in out.items():
        v = np.asarray(e["deltas"])
        e["n"] = int(v.size)
        e["mean"] = float(v.mean())
        e["sd"] = float(v.std(ddof=1)) if v.size > 1 else None
        e["se"] = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else None
        e["holds_2se"] = (abs(e["mean"]) <= 2 * e["se"]) if e["se"] else None
        e["deltas"] = [round(float(x), 4) for x in v]
    return out


def transfer_rows(rows: list[dict]) -> list[dict]:
    """gap/gain pairs against each student's own dense base (prior §5.2)."""
    base = {r["size"]: r["theta"] for r in rows if r["kind"] == "base"}
    ability = dict(base)
    for r in rows:
        if r["kind"] == "teacher" and r.get("teacher"):
            ability[r["teacher"]] = r["theta"]
    pairs = []
    for r in rows:
        rec = r.get("recipe")
        if not rec or not r.get("teacher"):
            continue
        if r["size"] not in base or r["teacher"] not in ability:
            continue
        pairs.append({"gap": ability[r["teacher"]] - base[r["size"]],
                      "gain": r["theta"] - base[r["size"]],
                      "recipe": RECIPE_LABEL.get(rec, rec),
                      "size": r["size"]})
    return pairs


def eta_fit(pairs: list[dict], reps: int = 2000, seed: int = 0) -> dict:
    x = np.asarray([p["gap"] for p in pairs])
    y = np.asarray([p["gain"] for p in pairs])
    eta = float(x @ y / (x @ x))
    resid = y - eta * x
    r2 = 1.0 - float(resid @ resid) / float(y @ y) if y.any() else 0.0
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        i = rng.integers(0, x.size, x.size)
        d = float(x[i] @ x[i])
        if d > 0:
            draws.append(float(x[i] @ y[i] / d))
    lo, hi = np.quantile(draws, [0.05, 0.95])
    return {"eta": eta, "ci90": [float(lo), float(hi)], "r2": r2,
            "n": int(x.size)}


def loo_size_rmse(pairs: list[dict]) -> dict:
    """Leave-one-student-size-out: fit η on the rest, predict the left-out
    size's gains, report RMSE over left-out cells (extrapolation test)."""
    sizes = sorted({p["size"] for p in pairs}, key=lambda s: SIZE_B[s])
    errs = []
    for s in sizes:
        train = [p for p in pairs if p["size"] != s]
        test = [p for p in pairs if p["size"] == s]
        if not train or not test:
            continue
        x = np.asarray([p["gap"] for p in train])
        y = np.asarray([p["gain"] for p in train])
        eta = float(x @ y / (x @ x))
        for p in test:
            errs.append(p["gain"] - eta * p["gap"])
    e = np.asarray(errs)
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "n": int(e.size)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    anchor_out, eta_out = {}, {}
    lines = ["# V1: recipe-stratified anchor + transfer law (2026-08-25)",
             "", "Data: locked grid-fill IRT (frozen difficulties "
             "2026-07-28). Noise floor = prior per-domain seed SD.", ""]
    for dom in ("math", "code", "qa"):
        rows = load(dom)
        floor = NOISE_FLOOR[dom]
        anchor = anchor_by_recipe(rows)
        anchor_out[dom] = anchor
        pairs = transfer_rows(rows)
        lines.append(f"## {dom} (noise floor {floor})")
        lines.append("")
        lines.append("### r=1 anchor by recipe (law demands mean Δθ = 0)")
        lines.append("| recipe | n | mean Δθ | SE | |mean|≤2·SE |")
        lines.append("|---|---|---|---|---|")
        pooled = [d for e in anchor.values() for d in e["deltas"]]
        for rec in sorted(anchor):
            e = anchor[rec]
            lines.append(f"| {rec} | {e['n']} | {e['mean']:+.3f} | "
                         f"{e['se']:.3f} | {'PASS' if e['holds_2se'] else 'FAIL'} |")
        pv = np.asarray(pooled)
        no_m1 = np.asarray([d for r, e in anchor.items() if r != "M1"
                            for d in e["deltas"]])
        lines.append(f"| pooled (prior) | {pv.size} | {pv.mean():+.3f} | "
                     f"{pv.std(ddof=1)/np.sqrt(pv.size):.3f} | "
                     f"{'PASS' if abs(pv.mean()) <= 2*pv.std(ddof=1)/np.sqrt(pv.size) else 'FAIL'} |")
        lines.append(f"| pooled minus M1 | {no_m1.size} | {no_m1.mean():+.3f} | "
                     f"{no_m1.std(ddof=1)/np.sqrt(no_m1.size):.3f} | "
                     f"{'PASS' if abs(no_m1.mean()) <= 2*no_m1.std(ddof=1)/np.sqrt(no_m1.size) else 'FAIL'} |")
        lines.append("")
        lines.append("### η per recipe + leave-one-size-out extrapolation")
        lines.append("| recipe | n | η [90% CI] | R² | LOO-RMSE | ×floor |")
        lines.append("|---|---|---|---|---|---|")
        eta_out[dom] = {}
        for rec in ("M0", "M1", "M2", "M3", "pooled", "pooled-noM1"):
            if rec == "pooled":
                sub = pairs
            elif rec == "pooled-noM1":
                sub = [p for p in pairs if p["recipe"] != "M1"]
            else:
                sub = [p for p in pairs if p["recipe"] == rec]
            if len(sub) < 6:
                continue
            fit = eta_fit(sub)
            loo = loo_size_rmse(sub)
            fit["loo_rmse"] = loo["rmse"]
            fit["loo_over_floor"] = loo["rmse"] / floor
            eta_out[dom][rec] = fit
            lines.append(
                f"| {rec} | {fit['n']} | {fit['eta']:+.3f} "
                f"[{fit['ci90'][0]:+.3f},{fit['ci90'][1]:+.3f}] | "
                f"{fit['r2']:.3f} | {loo['rmse']:.3f} | "
                f"{fit['loo_over_floor']:.1f}× |")
        lines.append("")

    with open(OUT / "anchor_by_recipe.json", "w") as fh:
        json.dump(anchor_out, fh, indent=1)
    with open(OUT / "eta_by_recipe.json", "w") as fh:
        json.dump(eta_out, fh, indent=1)
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
