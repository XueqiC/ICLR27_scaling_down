#!/usr/bin/env python3
"""A14: same-complexity smooth baselines for the measurement-efficiency confirmation. CPU only.

    python -B analysis/a14_same_complexity_baselines.py

Refits, on exactly the A11 development rows at the 18- and 36-measurement budgets,
every form that shares the compact power form's inputs and a similar parameter
count, plus a structural baseline adapted from the pruning-law literature, and
scores all of them on the A11 confirmation cells that were measured after the
predictions were fixed. Nothing here changes A11: its stored predictions and
errors are reproduced first as a check, then the added forms are scored on the
same cells. Retrospective; the added forms did not exist when A11 was fixed.

Forms (per capability; inputs z = (1, z log N0, z L0c, z log D0) as in the power form):
  power        (beta . z) s^gamma, gamma on the fixed grid                 5 parameters
  linear       (beta . z) s, gamma fixed at one (A1)                       4
  quadratic    (beta . z)(s + kappa s^2), kappa on a grid                  5
  strength     A s^gamma, no source inputs                                 2
  median       per-density development median, linear interpolation       one per density
  per-density  (beta . z) per retained density, linear interpolation (A2)  4 per density, 36 only
  pruning-law  L0 (P0 d^alpha - 1): the literature form, fitted on loss    2
with s = (1 - d)/0.3, d the retained density. K1 variants calibrate one number
on the target's mildest pruned measurement (d = 0.9), as in the K1 audit: the
power form's amplitude and the pruning-law form's P0.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import v53_prune_dev as p53  # noqa: E402


def lstsq(x, y):
    """A11's estimator: unregularised least squares at relative SVD tolerance 1e-10."""
    return np.linalg.lstsq(x, y, rcond=1e-10)[0]

A11 = ROOT / "results/a11-efficiency-confirmation"
OUT = ROOT / "results/a14-same-complexity"
TABLE = ROOT / "paper/paper/tables/a14_baselines.tex"
CAPS = ("math", "code", "qa")
KAPPA_GRID = [round(k, 2) for k in np.arange(-0.9, 2.01, 0.05)]
FORMS = ("power", "linear", "quadratic", "strength", "median", "per_density", "pruning_law")
PARAMETERS = {"power": 5, "linear": 4, "quadratic": 5, "strength": 2, "median": "one per density",
              "per_density": "four per density", "pruning_law": 2}


def read(path):
    return json.loads(Path(path).read_text())


def fit_forms(rows, cap, stats, budget):
    cr = [r for r in rows if r["cap"] == cap]
    z = p53.standardize([r["phi_raw"] for r in cr], stats)
    y = np.asarray([r["y"] for r in cr]); d = np.asarray([r["d"] for r in cr])
    L0 = np.asarray([r["L0"] for r in cr])
    fits = {}
    best = None
    for gamma in p53.GAMMA_GRID:
        x = z * p53.shape(d, gamma)[:, None]; beta = lstsq(x, y)
        sse = float(np.sum((x @ beta - y) ** 2))
        if best is None or sse < best["sse"]:
            best = {"beta": beta.tolist(), "gamma": float(gamma), "sse": sse}
    fits["power"] = best
    fits["linear"] = {"beta": lstsq(z * p53.shape(d, 1.0)[:, None], y).tolist(), "gamma": 1.0}
    best = None
    s = p53.shape(d, 1.0)
    for kappa in KAPPA_GRID:
        x = z * (s + kappa * s ** 2)[:, None]; beta = lstsq(x, y)
        sse = float(np.sum((x @ beta - y) ** 2))
        if best is None or sse < best["sse"]:
            best = {"beta": beta.tolist(), "kappa": float(kappa), "sse": sse}
    fits["quadratic"] = best
    best = None
    for gamma in p53.GAMMA_GRID:
        sh = p53.shape(d, gamma); A = float(sh @ y / (sh @ sh))
        sse = float(np.sum((A * sh - y) ** 2))
        if best is None or sse < best["sse"]:
            best = {"A": A, "gamma": float(gamma), "sse": sse}
    fits["strength"] = best
    fits["median"] = {"anchors": {str(v): float(np.median(y[d == v])) for v in sorted(set(d.tolist()))}}
    # One regression per development density (nine states each at either budget), linearly interpolated
    # between densities and extended from the boundary pair, as for the median curve.
    fits["per_density"] = {"anchors": {str(v): lstsq(z[d == v], y[d == v]).tolist() for v in sorted(set(d.tolist()))}}
    # Pruning-law form on loss: L = L0 P0 d^alpha, so log((L0+y)/L0) = log P0 + alpha log d.
    X = np.c_[np.ones(len(d)), np.log(d)]; target = np.log((L0 + y) / L0)
    coef, *_ = np.linalg.lstsq(X, target, rcond=None)
    fits["pruning_law"] = {"log_P0": float(coef[0]), "alpha": float(coef[1]), "P0": float(np.exp(coef[0]))}
    return fits


def predict(form, fit, z, d, L0):
    s = p53.shape(d, 1.0)
    if form == "power":
        return float(np.dot(fit["beta"], z) * p53.shape(d, fit["gamma"]))
    if form == "linear":
        return float(np.dot(fit["beta"], z) * s)
    if form == "quadratic":
        return float(np.dot(fit["beta"], z) * (s + fit["kappa"] * s ** 2))
    if form == "strength":
        return float(fit["A"] * p53.shape(d, fit["gamma"]))
    if form == "median":
        return p53.linear_curve(fit["anchors"], d)
    if form == "per_density":
        return p53.linear_curve({k: float(np.dot(b, z)) for k, b in fit["anchors"].items()}, d)
    if form == "pruning_law":
        return float(L0 * (fit["P0"] * d ** fit["alpha"] - 1.0))
    raise ValueError(form)


LABELS = {"power": "Compact power form", "linear": "Linear strength (fixed exponent one)",
          "quadratic": "Quadratic strength, same inputs", "strength": "Strength only, no source inputs",
          "median": "Median density curve", "per_density": "Per-density regression",
          "pruning_law": "Pruning-law form on loss"}


def render(results):
    """Appendix table: mean absolute error of every form at both budgets on the 24 confirmation cells."""
    from analysis.paper_table_layout import house_style, table_layout
    fmt = lambda x: f"{x:.3f}"
    rows = []
    for form in FORMS:
        cells = [LABELS[form], str(PARAMETERS[form])]
        for budget in ("18", "36"):
            mae = results["budgets"][budget]["mae"].get(form)
            cells += [fmt(mae[c]) for c in CAPS] if mae else ["/"] * 3
        rows.append(" & ".join(cells) + r" \\")
    k1 = results["k1"]
    rows.append(r"\midrule")
    for key, label in (("power_K1", "Compact power form, amplitude calibrated"),
                       ("pruning_law_K1", "Pruning-law form, $P_0$ calibrated")):
        cells = [label, "one target measurement"]
        for budget in ("18", "36"):
            cells += [fmt(k1[budget]["mae"][key][c]) for c in CAPS]
        rows.append(" & ".join(cells) + r" \\")
    caption = (
        "Same-complexity baselines for the measurement-efficiency confirmation. Every form is fitted on the "
        "same development rows as the compact power form, at 18 and at 36 measurements per capability, with "
        "the same inputs and estimator, and scored on the same 24 confirmation cells per capability; entries are "
        "mean absolute errors in nats per token. The pruning-law form is the literature relation "
        "$L=L_0P_0d^{\\alpha}$ \\citep{sengupta2025compression} fitted on capability loss, with one exponent "
        "and one scale per capability and no calibration on the target. The last two rows calibrate one number "
        "on each target's mildest pruned measurement and are scored on the remaining five densities, twenty cells "
        "per capability, so they are not comparable with the rows above. All forms other than the compact power "
        "form, the median curve and the per-density regression at 36 measurements were fitted after the confirmation, "
        "on its stored development rows.")
    header = ("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llrrrrrr@{}}\n\\toprule\n"
              " & & \\multicolumn{3}{c}{18 measurements} & \\multicolumn{3}{c}{36 measurements} \\\\\n"
              "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
              "Form & Parameters & Math & Code & QA & Math & Code & QA \\\\\n\\midrule\n")
    text = ("% Generated by analysis/a14_same_complexity_baselines.py; retrospective, do not edit.\n"
            "\\begin{table}[tb]\n\\centering\\footnotesize\n"
            f"\\caption{{{caption}}}\n\\label{{tab:a14-baselines}}\n" + header + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    return house_style(table_layout(text))


def main():
    plan, frozen, summary = read(A11 / "plan.json"), read(A11 / "predictions.json"), read(A11 / "summary.json")
    rows = frozen["development_rows"]
    dev = plan["development"]
    subsets = {18: [r for r in rows if r["d"] in dev["reduced_densities"]], 36: rows}
    if len(subsets[18]) != 54 or len(subsets[36]) != 108:
        raise ValueError("Unexpected development subset sizes")
    targets = {t["state"]: t for t in plan["targets"]}
    cells = summary["cells"]
    results = {"status": "RETROSPECTIVE", "forms": FORMS, "parameters": PARAMETERS,
               "note": "Forms other than power, median and the per-density regression at 36 measurements were fitted "
                       "after the A11 confirmation on its stored development rows; A11's own predictions are reproduced "
                       "first as a check.",
               "budgets": {}, "k1": {}}
    for budget, sub in subsets.items():
        stats = p53.zstats(sub)
        fits = {cap: fit_forms(sub, cap, stats, budget) for cap in CAPS}
        # Reproduce A11's stored predictions for the forms it fixed.
        stored = frozen["coefficients"]
        for cap in CAPS:
            key = f"power_{budget}"
            if abs(fits[cap]["power"]["gamma"] - stored[key]["fits"][cap]["gamma"]) > 1e-12 or \
               not np.allclose(fits[cap]["power"]["beta"], stored[key]["fits"][cap]["beta"], atol=1e-9):
                raise ValueError(f"Refit of {key}/{cap} differs from the stored A11 coefficients")
        errors = {form: {cap: [] for cap in CAPS} for form in FORMS}
        for cell in cells:
            cap, state, d = cell["capability"], cell["state"], cell["density"]
            t = targets[state]
            z = p53.standardize(p53.raw_features(t["N0"], t["D0"], cell["dense_loss"]), stats)
            for form in errors:
                pred = predict(form, fits[cap][form], z, d, cell["dense_loss"])
                if form == "power" and abs(pred - cell["predicted_delta_L"][f"power_{budget}"]) > 1e-9:
                    raise ValueError("Power prediction differs from the stored A11 prediction")
                if form == "median" and abs(pred - cell["predicted_delta_L"][f"median_curve_{budget}"]) > 1e-9:
                    raise ValueError("Median prediction differs from the stored A11 prediction")
                if form == "per_density" and budget == 36 and abs(pred - cell["predicted_delta_L"]["A2_36"]) > 1e-9:
                    raise ValueError("Per-density prediction differs from the stored A11 prediction")
                errors[form][cap].append(abs(pred - cell["observed_delta_L"]))
        results["budgets"][str(budget)] = {
            "n_cells_per_capability": len(cells) // len(CAPS),
            "mae": {form: {cap: statistics.mean(v) for cap, v in by.items()} for form, by in errors.items()},
            "fits": fits, "standardizer": stats}
        # K1: one calibration measurement at the mildest density per target; score the other five.
        k1 = {"power_K1": {c: [] for c in CAPS}, "pruning_law_K1": {c: [] for c in CAPS},
              "power_K0_same_cells": {c: [] for c in CAPS}, "median_same_cells": {c: [] for c in CAPS},
              "pruning_law_K0_same_cells": {c: [] for c in CAPS}}
        mildest = max(plan["densities"])
        for state in targets:
            for cap in CAPS:
                anchor = next(c for c in cells if c["state"] == state and c["capability"] == cap and c["density"] == mildest)
                t = targets[state]
                z = p53.standardize(p53.raw_features(t["N0"], t["D0"], anchor["dense_loss"]), stats)
                amplitude = anchor["observed_delta_L"] / p53.shape(mildest, fits[cap]["power"]["gamma"])
                L0 = anchor["dense_loss"]
                P0 = (L0 + anchor["observed_delta_L"]) / (L0 * mildest ** fits[cap]["pruning_law"]["alpha"])
                for cell in cells:
                    if cell["state"] != state or cell["capability"] != cap or cell["density"] == mildest:
                        continue
                    d, y = cell["density"], cell["observed_delta_L"]
                    k1["power_K1"][cap].append(abs(amplitude * p53.shape(d, fits[cap]["power"]["gamma"]) - y))
                    k1["pruning_law_K1"][cap].append(abs(L0 * (P0 * d ** fits[cap]["pruning_law"]["alpha"] - 1) - y))
                    k1["power_K0_same_cells"][cap].append(abs(predict("power", fits[cap]["power"], z, d, L0) - y))
                    k1["median_same_cells"][cap].append(abs(predict("median", fits[cap]["median"], z, d, L0) - y))
                    k1["pruning_law_K0_same_cells"][cap].append(abs(predict("pruning_law", fits[cap]["pruning_law"], z, d, L0) - y))
        results["k1"][str(budget)] = {"calibration_density": mildest, "n_cells_per_capability": len(k1["power_K1"]["math"]),
                                      "mae": {k: {c: statistics.mean(v) for c, v in by.items()} for k, by in k1.items()}}
    OUT.mkdir(exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(results, indent=1))
    lines = ["# A14 same-complexity baselines on the A11 confirmation (retrospective)", ""]
    for budget in ("18", "36"):
        lines += [f"## {budget} development measurements per capability, 24 confirmation cells", "",
                  "| Form | Parameters | Math | Code | QA |", "|---|---|---:|---:|---:|"]
        for form, by in results["budgets"][budget]["mae"].items():
            lines.append(f"| {form} | {PARAMETERS[form]} | " + " | ".join(f"{by[c]:.3f}" for c in CAPS) + " |")
        lines += ["", f"K1 at d={results['k1'][budget]['calibration_density']} (20 cells per capability):", "",
                  "| Predictor | Math | Code | QA |", "|---|---:|---:|---:|"]
        for k, by in results["k1"][budget]["mae"].items():
            lines.append(f"| {k} | " + " | ".join(f"{by[c]:.3f}" for c in CAPS) + " |")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    TABLE.write_text(render(results))
    print("\n".join(lines))
    print(f"wrote {TABLE}")


if __name__ == "__main__":
    main()
