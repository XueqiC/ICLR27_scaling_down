#!/usr/bin/env python3
"""A17: the comparison with Pruning Laws on its own metric, task accuracy (retrospective).

Pruning Laws (Sengupta et al., 2025) predict a task's post-pruning accuracy from its unpruned
accuracy and the retained density, acc(d) = acc0 * P0 * d^alpha, with one exponent per task
transferred across models and P0 either 1 (no calibration) or recalibrated on one pruned
measurement. This script fits that form on measured accuracies and scores it against the paper's
loss route on the same cells, at the same information budgets:

  Pruning-law K0   alpha per capability fitted on the other three models, P0 = 1.
  Pruning-law K1   the same alpha; P0 from the model's own measurement at density 0.9.
  Loss route K0    the development median density curve of the final rule (a new-state prediction
                   that uses no measurement of the target) gives the loss; a per-capability link
                   fitted on the other three models' (loss, accuracy) pairs maps it to accuracy.
  Loss route K1    the median curve rescaled by the model's own loss change at density 0.9.
  Link on measured loss   the link applied to the measured loss, which bounds the link's own error.

Measurements: V15 protocol, `--accuracy-benchmark easy` (GSM8K exact match, MBPP execution, TriviaQA
exact match; 64 probes per capability, 4-shot greedy), on gemma3-1b, gemma3-4b, Qwen3-1.7B and
Qwen3-4B, dense and magnitude-pruned to densities 0.9, 0.8, 0.7 and 0.6 with the V6 sampled global
threshold. Every fit is leave-one-model-out over the four models; scored cells are the densities
0.8, 0.7 and 0.6 (0.9 is the calibration point of the K1 budgets and is scored for K0 only in the
sidecar). Errors are mean absolute errors in accuracy points (0 to 1).

    python3 analysis/a17_accuracy_comparison.py            # summary.json / summary.md
    python3 analysis/a17_accuracy_comparison.py --table    # + paper/paper/tables/accuracy_comparison.tex
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/a17-accuracy-comparison"
ACC = ROOT / "results/v15-accuracy"
LOCKED = ROOT / "results/s3-selection-validation/inputs/locked_models.json"
TABLE = ROOT / "paper/paper/tables/accuracy_comparison.tex"
MODELS = ("gemma3-1b", "gemma3-4b", "Qwen3-1.7B", "Qwen3-4B")
NAMES = {"gemma3-1b": "Gemma 3, 1B", "gemma3-4b": "Gemma 3, 4B", "Qwen3-1.7B": "Qwen3, 1.7B", "Qwen3-4B": "Qwen3, 4B"}
DENSITIES = (0.9, 0.8, 0.7, 0.6)
SCORED = (0.8, 0.7, 0.6)
CAPS = ("math", "code", "qa")
FLOOR = 1 / 128  # half an item out of 64, so a zero accuracy has a finite logarithm


def read(p):
    return json.loads(Path(p).read_text())


def cell(model, density):
    tag = "dense__easy" if density == 1.0 else f"prune-d{density}__easy"
    p = ACC / model / tag / "accuracy.json"
    if not p.exists():
        return None
    d = read(p)

    def accuracy(agg):
        for key in ("accuracy", "exact_match", "pass_at_1"):
            if key in agg:
                return float(agg[key])
        for key in ("correct", "passed", "exact_matches"):
            if key in agg and "n" in agg:
                return float(agg[key]) / float(agg["n"])
        raise KeyError(f"no accuracy field in {sorted(agg)}")
    return {c: {"accuracy": accuracy(d["aggregates"][c]), "loss": float(d["losses"][c])} for c in CAPS}


def load_panel():
    panel = {}
    for m in MODELS:
        rows = {str(d): cell(m, d) for d in (1.0, *DENSITIES)}
        missing = [d for d, r in rows.items() if r is None]
        if missing:
            raise SystemExit(f"{m}: missing V15 cells at densities {missing}")
        panel[m] = rows
    return panel


# ---------------------------------------------------------------- pruning-law form on accuracy
def fit_alpha(panel, models, c):
    """One exponent per capability: least squares of log(acc(d)/acc0) on log d over the given models (P0 free per model)."""
    import numpy as np
    xs, ys, groups = [], [], []
    for m in models:
        a0 = max(panel[m]["1.0"][c]["accuracy"], FLOOR)
        for d in DENSITIES:
            a = max(panel[m][str(d)][c]["accuracy"], FLOOR)
            xs.append(math.log(d)); ys.append(math.log(a / a0)); groups.append(m)
    # per-model intercept (log P0) absorbs the recalibration freedom the law allows; alpha is shared
    X = np.zeros((len(xs), 1 + len(models)))
    for i, (x, g) in enumerate(zip(xs, groups)):
        X[i, 0] = x; X[i, 1 + models.index(g)] = 1.0
    coef = np.linalg.lstsq(X, np.asarray(ys), rcond=None)[0]
    return float(coef[0])


def law_predict(panel, m, c, d, alpha, calibrate):
    a0 = panel[m]["1.0"][c]["accuracy"]
    P0 = 1.0
    if calibrate:
        a9 = max(panel[m]["0.9"][c]["accuracy"], FLOOR)
        P0 = a9 / (max(a0, FLOOR) * 0.9 ** alpha)
    return a0 * P0 * d ** alpha


# ---------------------------------------------------------------- loss route
def median_delta(models, c, d):
    from analysis import final_rule
    dummy_L0 = 0.0
    return float(final_rule.predict("pruning", c, "new_stage", {"models": models, "L0": dummy_L0, "d": d, "N0": 1, "D0": 1}))


def fit_link(panel, models, c):
    """Logistic link acc = 1/(1+exp(-(a+b L))) fitted on the given models' measured (loss, accuracy) pairs at every density."""
    import numpy as np
    from scipy.optimize import least_squares
    L = np.array([panel[m][k][c]["loss"] for m in models for k in ("1.0", *map(str, DENSITIES))])
    A = np.array([panel[m][k][c]["accuracy"] for m in models for k in ("1.0", *map(str, DENSITIES))])
    def resid(p):
        return 1 / (1 + np.exp(-(p[0] + p[1] * L))) - A
    b0 = -1.0 if A.std() > 0 else 0.0
    sol = least_squares(resid, x0=[float(np.log(max(A.mean(), 1e-3) / max(1 - A.mean(), 1e-3)) - b0 * L.mean()), b0])
    return [float(v) for v in sol.x]


def link_apply(link, loss):
    return 1 / (1 + math.exp(-(link[0] + link[1] * loss)))


def analyse():
    import numpy as np
    panel = load_panel()
    locked = read(LOCKED)
    summary = {"models": list(MODELS), "densities": list(DENSITIES), "scored_densities": list(SCORED), "floor": FLOOR,
               "per_model": {}, "cells": []}
    for held in MODELS:
        dev = [m for m in MODELS if m != held]
        entry = {"alpha": {}, "link": {}}
        for c in CAPS:
            alpha = fit_alpha(panel, dev, c); link = fit_link(panel, dev, c)
            entry["alpha"][c] = alpha; entry["link"][c] = link
            L0 = panel[held]["1.0"][c]["loss"]
            obs9 = panel[held]["0.9"][c]["loss"] - L0
            med9 = median_delta(locked, c, 0.9)
            scale = obs9 / med9 if med9 != 0 else 1.0
            for d in DENSITIES:
                truth = panel[held][str(d)][c]["accuracy"]
                Lhat0 = L0 + median_delta(locked, c, d)
                Lhat1 = L0 + scale * median_delta(locked, c, d)
                preds = {"law_K0": law_predict(panel, held, c, d, alpha, False),
                         "law_K1": law_predict(panel, held, c, d, alpha, True),
                         "loss_K0": link_apply(link, Lhat0), "loss_K1": link_apply(link, Lhat1),
                         "link_measured_loss": link_apply(link, panel[held][str(d)][c]["loss"])}
                summary["cells"].append({"model": held, "capability": c, "density": d, "accuracy": truth,
                                         "loss": panel[held][str(d)][c]["loss"], "loss_predicted_K0": Lhat0,
                                         "loss_predicted_K1": Lhat1, "scored": d in SCORED,
                                         **{k: v for k, v in preds.items()},
                                         **{f"abs_error_{k}": abs(v - truth) for k, v in preds.items()}})
        summary["per_model"][held] = entry
    preds = ("law_K0", "law_K1", "loss_K0", "loss_K1", "link_measured_loss")
    mae = {}
    for scope, cells in (("scored", [x for x in summary["cells"] if x["scored"]]), ("all", summary["cells"])):
        mae[scope] = {}
        for k in preds:
            mae[scope][k] = {c: float(np.mean([x[f"abs_error_{k}"] for x in cells if x["capability"] == c])) for c in CAPS}
            mae[scope][k]["pooled"] = float(np.mean([x[f"abs_error_{k}"] for x in cells]))
    summary["mae"] = mae
    # paired comparison on the scored cells: loss route minus law, per budget
    for budget in ("K0", "K1"):
        diffs = [x[f"abs_error_loss_{budget}"] - x[f"abs_error_law_{budget}"] for x in summary["cells"] if x["scored"]]
        rng = np.random.default_rng(2026); idx = rng.integers(0, len(diffs), size=(5000, len(diffs)))
        means = np.asarray(diffs)[idx].mean(axis=1)
        summary[f"paired_loss_minus_law_{budget}"] = {"mean": float(np.mean(diffs)), "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
                                                       "cells": len(diffs), "wins_loss_route": int(sum(1 for v in diffs if v < 0))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = ["# A17 accuracy comparison with Pruning Laws (retrospective, leave-one-model-out)", "",
             "MAE in accuracy points on the scored cells (densities 0.8/0.7/0.6, four models, 64 probes per capability)", "",
             "| Predictor | Math | Code | QA | Pooled |", "|---|---:|---:|---:|---:|"]
    for k in preds:
        r = mae["scored"][k]; lines.append(f"| {k} | {r['math']:.3f} | {r['code']:.3f} | {r['qa']:.3f} | {r['pooled']:.3f} |")
    for budget in ("K0", "K1"):
        p = summary[f"paired_loss_minus_law_{budget}"]
        lines.append(f"\n{budget}: loss route minus law, mean {p['mean']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] over {p['cells']} cells; loss route lower in {p['wins_loss_route']}")
    lines += ["", "| Model | Capability | Density | Accuracy | Law K0 | Law K1 | Loss K0 | Loss K1 | Link on measured loss |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for x in summary["cells"]:
        lines.append(f"| {x['model']} | {x['capability']} | {x['density']} | {x['accuracy']:.3f} | {x['law_K0']:.3f} | {x['law_K1']:.3f} | {x['loss_K0']:.3f} | {x['loss_K1']:.3f} | {x['link_measured_loss']:.3f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    return summary


WORDS = {"law_K0": ("Pruning-law form", "none"), "law_K1": ("Pruning-law form", "one accuracy at density 0.9"),
         "loss_K0": ("Loss route: median curve and link", "none"), "loss_K1": ("Loss route: median curve and link", "one loss at density 0.9"),
         "link_measured_loss": ("Link on the measured loss", "the target's own loss")}


def render(summary):
    from analysis.paper_table_layout import house_style, table_layout
    rows = []
    for k, (name, budget) in WORDS.items():
        r = summary["mae"]["scored"][k]
        rows.append(" & ".join([name, budget] + [f"{r[c]:.3f}" for c in (*CAPS, "pooled")]) + r" \\")
    p0, p1 = summary["paired_loss_minus_law_K0"], summary["paired_loss_minus_law_K1"]
    caption = ("Prediction of task accuracy after magnitude pruning, the endpoint of the pruning law, on Gemma 3 1B and 4B and "
               "Qwen3 1.7B and 4B at densities 0.8, 0.7 and 0.6, leave-one-model-out: mean absolute error in accuracy points over "
               "36 cells per column, 64 probes per capability. The pruning-law form fits one exponent per capability on the other "
               "three models; the loss route predicts the loss with the final rule's development median curve and maps it to "
               "accuracy with a logistic link fitted on the other three models; the last row applies that link to the measured "
               "loss. Calibration uses one measurement of the target at density 0.9. Paired over cells, the loss route minus the "
               f"law is {p0['mean']:+.3f} [{p0['ci95'][0]:+.3f}, {p0['ci95'][1]:+.3f}] without calibration and "
               f"{p1['mean']:+.3f} [{p1['ci95'][0]:+.3f}, {p1['ci95'][1]:+.3f}] with it.")
    text = ("% Generated by analysis/a17_accuracy_comparison.py --table; retrospective, do not edit.\n"
            "\\begin{table}[tb]\n\\centering\\footnotesize\n"
            f"\\caption{{{caption}}}\n\\label{{tab:accuracy-comparison}}\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llrrrr@{}}\n\\toprule\n"
            "Predictor & Target measurement used & Math & Code & QA & Pooled \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    return house_style(table_layout(text))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    s = analyse()
    if a.table:
        TABLE.write_text(render(s)); print("wrote", TABLE)
