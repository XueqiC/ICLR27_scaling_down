#!/usr/bin/env python3
"""V2: item-level damage spectrum under pruning — quanta vs uniform shift.

Question (method_design §3.3): when pruning damages a capability, do items
die discretely ("single-gene collapse": a subset drops to zero while the
rest stay intact — the quanta/skill-deletion prediction) or does the whole
panel shift uniformly (smooth capacity loss — the Rasch-shift prediction)?

Test: for each (source, task) pruning ladder with item-level responses,
compare two models of the response vector at each sparsity s:
  H-shift: p_i(s) = sigmoid(theta_s - b_i)  — one free theta_s, frozen
           locked item difficulties b_i (uniform logit shift).
  H-mix:   with prob (1-pi_s) item i keeps its base success p_i(0)
           (empirical, shrunk), with prob pi_s it is dead (p = eps).
           One free pi_s: "a fraction of skills deleted, rest intact".
Both models have one free parameter per sparsity level; compare summed
log-likelihood over items. Report per-ladder AIC-equivalent LL difference
and the damage-concentration profile (Gini of per-item drop).

Data: prune-mild-matched-2026-07-27/eval (src32b hotpotqa/humaneval,
src8b gsm8k/hotpotqa/humaneval; per-item scores, n=50/31 items).
Item difficulties: locked grid-fill IRT (matching canonical slices).
Outputs: results/v2-damage-spectrum/{summary.md, ladders.json}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "prior/anon-intern-backup/docs/scaling-down/results/runs"
EVAL = RUNS / "prune-mild-matched-2026-07-27/eval"
LOCKED = RUNS / "irt-grid-fill-locked-2026-07-28"
OUT = ROOT / "results/v2-damage-spectrum"

TASK_DOMAIN = {"gsm8k": "math", "humaneval": "code", "hotpotqa": "qa"}
EPS = 1e-3  # dead-item success prob (parser noise floor)


def load_difficulties() -> dict[str, float]:
    b: dict[str, float] = {}
    for dom in ("math", "code", "qa"):
        with open(LOCKED / f"{dom}.json") as fh:
            d = json.load(fh)
        for iid, diff in zip(d["item_ids"], d["difficulty"]):
            b[iid] = float(diff)
    return b


def parse_eval(path: Path) -> tuple[np.ndarray, list[str], float]:
    with open(path) as fh:
        d = json.load(fh)
    task = d.get("task") or path.name.split("-")[1]
    scores, iids = [], []
    for it in d["items"]:
        s = it.get("score")
        scores.append(float(s) if s not in (None, "") else 0.0)
        if it.get("task_id"):
            iids.append(f"{task}:{it['task_id']}")
        else:
            iids.append(f"{task}:{int(it['i'])}")
    return np.asarray(scores) >= 0.5, iids, float(d["sparsity"])


def ll_shift(y: np.ndarray, b: np.ndarray) -> float:
    """Max log-likelihood of the uniform-shift Rasch model (free theta)."""
    def nll(theta: float) -> float:
        p = 1.0 / (1.0 + np.exp(-(theta - b)))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    res = minimize_scalar(nll, bounds=(-12, 12), method="bounded")
    return -float(res.fun)


def ll_mix(y: np.ndarray, p0: np.ndarray) -> float:
    """Max log-likelihood of the deletion-mixture model (free dead frac).

    Marginal per item: P(y=1) = (1-pi)*p0_i + pi*EPS.
    """
    def nll(pi: float) -> float:
        p = (1 - pi) * p0 + pi * EPS
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    res = minimize_scalar(nll, bounds=(0.0, 1.0), method="bounded")
    return -float(res.fun)


def gini(drops: np.ndarray) -> float | None:
    v = np.sort(np.clip(drops, 0, None)).astype(float)
    if v.sum() <= 0:
        return None
    n = v.size
    return float((2 * np.arange(1, n + 1) - n - 1) @ v / (n * v.sum()))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bmap = load_difficulties()

    ladders: dict[str, dict[float, Path]] = {}
    for f in sorted(EVAL.glob("*.json")):
        m = re.match(r"(src\d+b)-([a-z0-9]+)-sp(.+)\.json", f.name)
        if not m:
            continue
        src, task, sp = m.group(1), m.group(2), m.group(3)
        sparsity = float(sp.replace("p", "."))
        ladders.setdefault(f"{src}-{task}", {})[sparsity] = f

    results = {}
    lines = ["# V2: item-level damage spectrum, prune ladders "
             "(2026-08-25)", "",
             "Models per sparsity (1 free param each): uniform Rasch shift "
             "vs deletion mixture. ΔLL = LL_mix − LL_shift; positive favors "
             "quanta-style discrete deletion.", ""]
    for key, byspar in sorted(ladders.items()):
        if 0.0 not in byspar:
            continue
        y0, iids, _ = parse_eval(byspar[0.0])
        # shrunk base success prob per item (single binary obs -> shrink)
        p0 = np.where(y0, 0.9, 0.1)
        b = np.asarray([bmap.get(i, np.nan) for i in iids])
        ok = ~np.isnan(b)
        lines.append(f"## {key} (items with locked difficulty: "
                     f"{int(ok.sum())}/{len(iids)})")
        lines.append("| sparsity | acc | ΔLL (mix−shift) | verdict | "
                     "dead frac π̂ | Gini(drop) |")
        lines.append("|---|---|---|---|---|---|")
        entry = []
        for sp in sorted(s for s in byspar if s > 0):
            y, iids_s, _ = parse_eval(byspar[sp])
            assert iids_s == iids, f"item mismatch in {key} sp{sp}"
            ys, bs, p0s = y[ok], b[ok], p0[ok]
            l_shift = ll_shift(ys, bs)
            l_mix = ll_mix(ys, p0s)
            dll = l_mix - l_shift
            # dead fraction estimate at optimum
            from scipy.optimize import minimize_scalar as ms
            pi_hat = ms(lambda pi: -np.sum(
                ys * np.log(np.clip((1 - pi) * p0s + pi * EPS, 1e-9, 1)) +
                (1 - ys) * np.log(np.clip(1 - ((1 - pi) * p0s + pi * EPS),
                                          1e-9, 1))),
                bounds=(0, 1), method="bounded").x
            drop = (y0.astype(float) - y.astype(float))[ok]
            g = gini(drop)
            verdict = ("mix" if dll > 2 else
                       "shift" if dll < -2 else "tie")
            lines.append(f"| {sp:.3f} | {y.mean():.2f} | {dll:+.1f} | "
                         f"{verdict} | {pi_hat:.2f} | "
                         f"{'-' if g is None else f'{g:.2f}'} |")
            entry.append({"sparsity": sp, "acc": float(y.mean()),
                          "dll": float(dll), "pi": float(pi_hat),
                          "gini": g})
        results[key] = entry
        lines.append("")

    with open(OUT / "ladders.json", "w") as fh:
        json.dump(results, fh, indent=1)
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
