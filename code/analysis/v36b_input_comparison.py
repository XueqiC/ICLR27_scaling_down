#!/usr/bin/env python3
"""CPU-only four-input comparison on the available clean Pythia loss panel.

SDL_V36_SIZES=410m,1.4b python3 analysis/v36b_input_comparison.py [--dry-run]
Only existing grid JSON is read. No weights, inference, network or model runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile

try:
    from . import v36_pythia_controlled_fit as v36
except ImportError:
    import v36_pythia_controlled_fit as v36

import numpy as np  # V36's audit import sets CPU BLAS thread limits first.

ROOT = v36.ROOT
OUT = ROOT / "results/v36b-input-comparison"
REPORT = ROOT / "paper/docs/PYTHIA_INPUT_COMPARISON.md"
SIZES = tuple(s for s in v36.SIZES if s != "2.8b")
ARMS, CAPS, STEPS, CONFIGS = v36.ARMS, v36.CAPS, v36.STEPS, v36.CONFIGS
# Preserve V36's feature order so its A/B fits remain numerically identical.
# Config is an always-known coordinate, separate from the advertised inputs.
INPUT_SETS = {"simple": (), "N0_D0": ("N0", "D0"),
              "N0_L0": ("N0", "L0"), "N0_D0_L0": ("N0", "L0", "D0")}
LABELS = {"simple": "Strongest simple", "N0_D0": "{N0, D0}",
          "N0_L0": "{N0, L0}", "N0_D0_L0": "{N0, D0, L0}"}
BASELINES = ("zero", "config_mean", "config_median")
MODELS = tuple(k for k in INPUT_SETS if k != "simple")


def load_grid(root=ROOT):
    """Respect V36's size environment, exclude 2.8B, require the clean 2x3 panel."""
    if len(SIZES) != 2 or set(SIZES) != {"410m", "1.4b"}:
        raise ValueError("V36b requires SDL_V36_SIZES=410m,1.4b (2.8b is excluded)")
    return v36.load_grid(root, sizes=SIZES)


def basic_input(row, model):
    return v36.basic_input(row, input_fields=INPUT_SETS[model])


def fit_direct(rows, model):
    """Reuse V36's one OLS solve on signed config-level Delta L, never a_c labels."""
    return v36.fit_direct(rows, input_fields=INPUT_SETS[model])


def paired_comparison(records, reference, model):
    """V36's exact 27 whole-step paired resamples, with both models on every row."""
    paired = [{"step": r["step"], "observed": r["observed"],
               "predictions": {"A": r["predictions"][reference],
                               "B": r["predictions"][model]}} for r in records]
    m = v36.paired_metrics(paired, "step")
    return {"reference": reference, "model": model, "n": m["n"],
            "n_groups": m["n_groups"], "reference_mae": m["A_mae"],
            "mae": m["B_mae"], "mae_ci95": m["B_mae_ci95"],
            "improvement": m["improvement"], "improvement_ci95": m["improvement_ci95"],
            "interval_assessment": m["interval_assessment"],
            "leave_one_group_out": m["leave_one_group_out"],
            "leave_one_group_out_range": m["leave_one_group_out_range"]}


def validate_panel(rows):
    if not rows or len({(r["arm"], r["capability"]) for r in rows}) != 1:
        raise ValueError("Need exactly one arm/capability panel")
    arm = rows[0]["arm"]
    expected = set(itertools.product(SIZES, STEPS, CONFIGS[arm]))
    coordinates = [(r["size"], r["step"], r["config"]) for r in rows]
    if len(coordinates) != len(expected) or set(coordinates) != expected:
        raise ValueError("Need the complete clean size/step/config panel, without duplicates")
    if len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("Row IDs must be unique")


def correlations(rows):
    """Pearson correlations on unique dense cells, never replicated per config."""
    cells = {r["cell"]: r for r in rows}

    def measure(sequence):
        d0 = np.array([r["D0"] for r in sequence], float)
        l0 = np.array([r["L0"] for r in sequence], float)

        def pearson(x):
            return float(np.corrcoef(x, l0)[0, 1]) if np.std(x) > 0 and np.std(l0) > 0 else None

        return {"n_cells": len(sequence), "pearson_D0_L0": pearson(d0),
                "pearson_log_D0_L0": pearson(np.log(d0)),
                "undefined_policy": "null if either variable is constant"}

    return {"all_cells": measure(list(cells.values())),
            "by_size": {s: measure([r for r in cells.values() if r["size"] == s]) for s in SIZES}}


def cross_validate(rows):
    validate_panel(rows)
    records, fitted_folds = [], []
    for fold in v36.folds(rows, "step"):
        train = [rows[i] for i in fold["train_indices"]]
        test = [rows[i] for i in fold["test_indices"]]
        fits = {m: fit_direct(train, m) for m in MODELS}
        fits["config_mean"] = fit_direct(train, "simple")
        medians = {f"{q:g}": float(np.median([r["observed"] for r in train if r["config"] == q]))
                   for q in CONFIGS[rows[0]["arm"]]}
        fits["config_median"] = {"objective": "training-only config-level L1 constant",
                                 "train_row_ids": [r["row_id"] for r in train], "values": medians}
        predictions = {m: v36.predict(fits[m], [basic_input(r, m) for r in test]) for m in MODELS}
        predictions.update(zero=np.zeros(len(test)),
                           config_mean=v36.predict(fits["config_mean"], [basic_input(r, "simple") for r in test]),
                           config_median=np.array([medians[f"{r['config']:g}"] for r in test]))
        fold_rows = [{**r, "held_out": fold["held_out"],
                      "predictions": {m: float(p[i]) for m, p in predictions.items()}}
                     for i, r in enumerate(test)]
        fitted_folds.append({**fold, "n_train": len(train), "n_test": len(test),
                             "train_row_ids": [r["row_id"] for r in train],
                             "test_row_ids": [r["row_id"] for r in test], "fits": fits,
                             "training_correlations": correlations(train),
                             "mae": {m: float(np.mean([abs(r["observed"]-r["predictions"][m])
                                                         for r in fold_rows])) for m in predictions}})
        records.extend(fold_rows)
    baseline_metrics = {m: paired_comparison(records, "zero", m) for m in BASELINES}
    # One empirical reference per entire arm/capability, NOT a per-row/fold oracle.
    # Predictions above never use outer-test outcomes to choose or fit a baseline.
    strongest = min(BASELINES, key=lambda m: baseline_metrics[m]["mae"])
    for r in records:
        r["predictions"]["simple"] = r["predictions"][strongest]
    for fold in fitted_folds:
        fold["mae"]["simple"] = fold["mae"][strongest]
    return {"scheme": "leave_one_step_out", "strongest_simple_baseline": strongest,
            "baseline_candidates": baseline_metrics,
            "core_table": {m: paired_comparison(records, strongest, m) for m in INPUT_SETS},
            "input_pairs": {"L0_vs_D0": paired_comparison(records, "N0_D0", "N0_L0"),
                            "combined_vs_D0": paired_comparison(records, "N0_D0", "N0_D0_L0"),
                            "combined_vs_L0": paired_comparison(records, "N0_L0", "N0_D0_L0")},
            "by_config": {f"{q:g}": {m: paired_comparison([r for r in records if r["config"] == q], strongest, m)
                                       for m in INPUT_SETS} for q in CONFIGS[rows[0]["arm"]]},
            "folds": fitted_folds, "records": records}


def trajectory(values):
    changes = np.diff(values)
    tol = v36.MONOTONIC_ATOL
    if np.all(np.abs(changes) <= tol):
        direction = "flat"
    elif np.all(changes >= -tol):
        direction = "increasing"
    elif np.all(changes <= tol):
        direction = "decreasing"
    else:
        direction = "nonmonotonic"
    return {"values": values, "adjacent_changes": changes.tolist(),
            "late_minus_early": values[-1]-values[0], "monotonicity": direction}


def curves(rows):
    """Three observed loss trajectories; absolute compressed loss is NOT abs(Delta L)."""
    dense, compressed, trajectories = [], [], []
    for size in SIZES:
        size_rows = [r for r in rows if r["size"] == size]
        for step in STEPS:
            cell = [r for r in size_rows if r["step"] == step]
            if len({r["L0"] for r in cell}) != 1:
                raise ValueError("Every config must share its arm/cell dense reference")
            r = cell[0]
            dense.append({k: r[k] for k in ("size", "step", "N0", "D0", "L0")})
        for q in CONFIGS[rows[0]["arm"]]:
            sequence = sorted([r for r in size_rows if r["config"] == q], key=lambda r: r["step"])
            for r in sequence:
                if not np.isclose(r["L0"]+r["observed"], r["loss"], rtol=0, atol=1e-12):
                    raise ValueError("Absolute loss must equal L0 + Delta L")
                compressed.append({"size": size, "step": r["step"], "N0": r["N0"],
                                   "D0": r["D0"], "config": q, "L0": r["L0"],
                                   "delta_loss": r["observed"], "compressed_loss": r["loss"]})
            t = {"size": size, "config": q, "steps": list(STEPS), "D0": [r["D0"] for r in sequence],
                 "dense_loss": trajectory([r["L0"] for r in sequence]),
                 "delta_loss": trajectory([r["observed"] for r in sequence]),
                 "compressed_loss": trajectory([r["loss"] for r in sequence])}
            t["increment_rises_but_absolute_loss_falls"] = (
                t["delta_loss"]["late_minus_early"] > v36.MONOTONIC_ATOL and
                t["compressed_loss"]["late_minus_early"] < -v36.MONOTONIC_ATOL)
            trajectories.append(t)
    return {"dense": dense, "compressed": compressed, "trajectories": trajectories,
            "n_increment_rises_but_absolute_loss_falls":
                sum(t["increment_rises_but_absolute_loss_falls"] for t in trajectories)}


def build_summary(root=ROOT):
    rows, hashes = load_grid(root)
    integrity = v36.checkpoint_integrity(rows, hashes, sizes=SIZES)
    if not integrity["headline_ready"]:
        raise ValueError("Clean panel contains duplicate checkpoint payloads; inspect provenance")
    # V36's explanatory text refers to its historical nine-cell diagnostic.
    integrity["interpretation"] = "No exact full-panel duplicate step payloads in the available six-cell panel per arm; 2.8b excluded."
    results = {}
    for arm in ARMS:
        results[arm] = {}
        for cap in CAPS:
            panel = [r for r in rows if (r["arm"], r["capability"]) == (arm, cap)]
            results[arm][cap] = {"n_rows": len(panel), "comparison": cross_validate(panel),
                                 "correlations": correlations(panel), "curves": curves(panel),
                                 "flag_counts": {k: sum(r[k] for r in panel)
                                                 for k in ("near_zero", "negative", "int3", "large_damage")}}
    return {"version": "36b", "analysis": "Pythia four-input direct loss comparison",
            "cpu_only": True, "new_model_runs": 0, "n_rows": len(rows), "n_cells_per_arm": 6,
            "input_sha256": hashes, "integrity": integrity,
            "inputs": {"sizes": list(SIZES), "excluded_sizes": {"2.8b": "Upstream step revisions collapse to identical weights; excluded per V36 clean-panel scope"},
                       "steps": list(STEPS), "D0_by_step": {str(s): v36.training_tokens(s) for s in STEPS},
                       "tokens_per_step": v36.TOKENS_PER_STEP,
                       "architectures": {s: {**v36.ARCHITECTURES[s], "N0": v36.matrix_n0(s)} for s in SIZES},
                       "input_sets": {m: list(fields) for m, fields in INPUT_SETS.items()},
                       "baseline_candidates": list(BASELINES)},
            "protocol": {
                "endpoint": "Delta L_c = L_c(config) - L_c(dense), signed nats/native token, own arm/capability/cell dense reference",
                "fit": "V36 direct unweighted OLS with config indicators interacted with an intercept and selected inputs; log(N0/1e9), log(D0/1e9), raw L0, standardized on training rows only. No a_c-label regression, clipping, regularization or tuning.",
                "folds": "V36 leave-one-step-out: 3 folds, each 4 train / 2 test checkpoints (16 train / 8 test observations) per arm/capability; 24 distinct OOF rows. All sizes/configs at a step travel together. No leave-one-size-out on the two-size panel.",
                "parameters": {"config_mean": 4, "N0_D0": 12, "N0_L0": 12, "N0_D0_L0": 16},
                "saturation": "Combined model has four coefficients per config fitted to four train cells; full-rank but saturated. Standalone models have three coefficients per config. Condition numbers and all held-out steps are reported.",
                "baselines": "Zero; train-only mean by config (V36 intercept-only OLS); train-only median by config. All share the same OOF rows. Strongest = minimum overall OOF MAE per arm/capability, ties follow candidate order. This is an empirical reference, not a learned deployable selector; never choose per row/fold. All candidates are disclosed.",
                "improvement": "Reference MAE minus model MAE; positive means lower prediction error. Core reference is the strongest declared simple baseline, fixed for every model and config slice within an arm/capability.",
                "uncertainty": "Reuse V36 paired_metrics: exact 27 ordered resamples of 3 whole held-out steps, 95% inverted-CDF percentiles. Models are paired on identical rows; no row-wise bootstrap. Fits, predictions and selected reference remain fixed. Reference selection is not repeated or adjusted for in the intervals.",
                "limits": "Only three step clusters with overlapping training sets and shared trajectories/probes: descriptive panel-conditional stability intervals, not population, seed, probe, retraining or causal uncertainty. No multiplicity correction or equivalence margin; crossing zero cannot establish redundancy. Two sizes do not support a universal scaling law.",
                "units_and_scope": "Pruning and quantization, and math/code/qa, are fitted, scored and plotted separately. All losses are nats per native token; D0 is processed pretraining tokens, not unique tokens. N0 follows V36's transformer-matrix count, excluding embeddings/head/vectors; compression itself includes language matrices.",
                "handling": "Retain all measured configurations, signed negative/near-zero responses and int3/large damage; dense anchors are not scored as artificial zero-response observations. Target dense L0 is an allowed predictor only for the L0 input models, although it defines the evaluation endpoint for all models.",
                "curves": "Observed L0(D0), signed Delta L(D0), and absolute compressed loss L0 + Delta L at each fixed size/config. Raw linear native-token loss axes, no normalization or pooling. Endpoint changes and both adjacent changes are reported; larger increments need not imply worse absolute loss.",
                "correlation": "Pearson D0 vs L0 and log(D0) vs L0 on unique dense cells, both pooled across the six cells within an arm/capability and separately along each three-checkpoint size trajectory. Fold training correlations also retained.",
                "timing": "Retrospective CPU analysis of existing aggregate loss JSON; no model runs."},
            "results": results}


def with_ci(metric, key="mae"):
    lo, hi = metric[f"{key}_ci95"]
    return f"{metric[key]:.5f} [{lo:.5f}, {hi:.5f}]"


def assessment(comparison):
    core = comparison["core_table"]
    a, b = core["N0_D0"], core["N0_L0"]
    if a["improvement"] <= 0 and b["improvement"] <= 0:
        return "Both standalone predictors fail to beat the simple reference in point MAE; their comparison cannot demonstrate information redundancy."
    if a["improvement"] > 0 and b["improvement"] > 0:
        return "Both standalone predictors beat the simple reference in point MAE; this provides a predictive signal against which to assess their overlap."
    winner = "{N0, D0}" if a["improvement"] > 0 else "{N0, L0}"
    return f"Only {winner} beats the simple reference in point MAE; the inputs are not interchangeable in this fitted model class."


def render(summary):
    lines = ["# Pythia four-input comparison (V36b)", "",
             "The KEY comparison asks whether either training scale or dense measurement predicts compression loss increments well enough to beat a simple baseline, before asking whether combining them helps. "
             "Failure to gain from adding D0 alone cannot distinguish information redundancy from both predictors failing.", "",
             "Endpoint: **ΔL_c = L_c(config) − L_c(dense)**, in **nats per native token**. "
             "This is signed capability loss damage, not an a_c label or a task-accuracy endpoint. "
             "The available clean panel is **410m + 1.4b × step16k/64k/143k**; **2.8b is excluded**. "
             "The 12 existing V6/V10 JSON files supply six cells per arm, 24 observations per arm/capability, and 144 total. "
             "Pruning (density 0.9/0.8/0.7/0.6, dense key `1.0`) and quantization (8/6/4/3 bits, dense key `dense`) use their own dense references and are never pooled. No model runs.", "",
             "The [V36 analysis](PYTHIA_CONTROLLED.md) supplies the loader, architecture/token accounting, direct OLS machinery, whole-step folds and paired interval computation. "
             "V36b computes all counts from the clean panel, with no dependency on the earlier report's historical three-size prose.", "",
             "| Step | D0 processed tokens |", "|---|---:|"]
    for step, tokens in summary["inputs"]["D0_by_step"].items():
        lines.append(f"| {step} | {tokens:,} |")
    lines += ["", "N0 transformer-matrix counts: " + "; ".join(f"{s}: {a['N0']:,}" for s, a in summary["inputs"]["architectures"].items()) + ".", "",
              "**Same held-out set:** three leave-one-step-out folds, each with 16 training and 8 test observations per arm/capability. "
              "All configurations and both sizes at the held-out step travel together; every one of the 24 observations is scored once by every model. "
              "The middle step is interpolation and endpoints are extrapolation. There is no leave-one-size-out analysis on this two-size panel.", "",
              "For known compression setting q, the input models are:", "", "```text",
              "simple:        0, or training-only mean/median Delta L at q",
              "{N0,D0}:       alpha_q + beta_q log(N0/1e9) + delta_q log(D0/1e9)",
              "{N0,L0}:       alpha_q + beta_q log(N0/1e9) + gamma_q L0",
              "{N0,D0,L0}:    alpha_q + beta_q log(N0/1e9) + gamma_q L0 + delta_q log(D0/1e9)",
              "```", "",
              "All predictor models directly fit the raw signed configuration-level ΔL by the same unweighted V36 OLS solve. "
              "Continuous features are standardized on training rows only. No a_c-label regression, response transform, clipping, tuning or regularization is used. "
              "The compression setting is always known; this is prediction at the four measured settings. "
              "The standalone models have 12 coefficients and the combined model has 16. "
              "The combined fit is **saturated: four coefficients per config, four training cells**. "
              "Full rank does not ensure stable extrapolation; fold condition numbers are shown below. "
              "All negative, near-zero, int3 and large-damage observations remain in the primary scores.", "",
              "**Reference and intervals:** strongest simple means the lowest overall OOF MAE among zero change, training-only config mean and training-only config median, selected once per arm/capability. "
              "The selection is an empirical comparison reference, not a deployable selector; there is no per-row or per-fold oracle. "
              "All candidates are disclosed below. Positive improvement = reference MAE − model MAE. "
              "Brackets on improvements are **paired-model 95% intervals** from V36's exact 27 resamples of the three whole held-out steps. "
              "Fits, OOF predictions and reference selection are held fixed; selection uncertainty is not covered. "
              "With three groups and overlapping training sets/shared trajectories, these are coarse descriptive, panel-conditional intervals, not seed/probe/population/causal inference. "
              "There is no multiplicity correction or equivalence test; an interval containing zero does not prove redundancy.", ""]
    for arm in ARMS:
        lines += [f"## KEY four-input table: {arm}", "",
                  "| Capability | Inputs | MAE [95%] | Improvement over strongest simple [paired 95%] |",
                  "|---|---|---:|---:|"]
        for cap in CAPS:
            comparison = summary["results"][arm][cap]["comparison"]
            for model, metric in comparison["core_table"].items():
                label = LABELS[model]
                if model == "simple":
                    label += f" ({comparison['strongest_simple_baseline']})"
                lines.append(f"| {cap} | {label} | {with_ci(metric)} | {with_ci(metric, 'improvement')} |")
        lines += ["", "Zero-change MAE is the mean |ΔL|: it shows the response magnitude available to model. "
                  "The reference row has zero improvement by definition; its gain over zero is shown here.", "",
                  "| Capability | Zero MAE | Config mean MAE | Config median MAE | Selected reference | Reference gain over zero [paired 95%] |",
                  "|---|---:|---:|---:|---|---:|"]
        for cap in CAPS:
            c = summary["results"][arm][cap]["comparison"]
            b = c["baseline_candidates"]
            lines.append(f"| {cap} | {b['zero']['mae']:.5f} | {b['config_mean']['mae']:.5f} | {b['config_median']['mae']:.5f} | "
                         f"{c['strongest_simple_baseline']} | {with_ci(b[c['strongest_simple_baseline']], 'improvement')} |")
        lines += ["", "| Capability | L0 model gain over D0 model [paired 95%] | Combined gain over D0 model [paired 95%] | Combined gain over L0 model [paired 95%] |",
                  "|---|---:|---:|---:|"]
        for cap in CAPS:
            pairs = summary["results"][arm][cap]["comparison"]["input_pairs"]
            lines.append(f"| {cap} | " + " | ".join(with_ci(pairs[k], "improvement") for k in ("L0_vs_D0", "combined_vs_D0", "combined_vs_L0")) + " |")
        lines.append("")
        for cap in CAPS:
            c = summary["results"][arm][cap]["comparison"]
            lines.append(f"**{cap}:** {assessment(c)} "
                         "Read the paired intervals alongside these point comparisons; a non-significant combined gain alone establishes neither redundancy nor absence of signal.")
            lines.append("")
    lines += ["## Fold stability", "",
              "All values are native-token MAE. The simple reference is fixed across the three folds. "
              "These are the same held-out predictions as the KEY table.", "",
              "| Arm | Capability | Held-out step | Simple | N0,D0 | N0,L0 | Combined | Combined condition number |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm, cap in itertools.product(ARMS, CAPS):
        for fold in summary["results"][arm][cap]["comparison"]["folds"]:
            values = " | ".join(f"{fold['mae'][m]:.5f}" for m in INPUT_SETS)
            lines.append(f"| {arm} | {cap} | {fold['held_out']} | {values} | {fold['fits']['N0_D0_L0']['condition_number']:.2f} |")
    lines += ["", "## Training trajectory correlation", "",
              "Pearson correlations use unique dense cells (not four repetitions per compression setting). "
              "Both raw D0 and the fitted log(D0) scale are shown. Within-size correlations describe the training trajectory; the six-cell correlation also includes size variation. "
              "Three checkpoints per size make these descriptive. Correlated inputs and unstable saturated fits preclude a causal coefficient interpretation.", "",
              "| Arm | Capability | Cells | n | corr(D0,L0) | corr(log D0,L0) |",
              "|---|---|---|---:|---:|---:|"]
    for arm, cap in itertools.product(ARMS, CAPS):
        c = summary["results"][arm][cap]["correlations"]
        for label, m in [("all sizes", c["all_cells"]), *c["by_size"].items()]:
            fmt = lambda x: "undefined (constant)" if x is None else f"{x:.5f}"
            lines.append(f"| {arm} | {cap} | {label} | {m['n_cells']} | {fmt(m['pearson_D0_L0'])} | {fmt(m['pearson_log_D0_L0'])} |")
    lines += ["", "## The three loss curves", "",
              "**L_c,0(D0)** is dense capability loss; **ΔL_m,c(D0)** is its compression increment; "
              "**L_m,c(D0) = L_c,0(D0) + ΔL_m,c(D0)** is absolute compressed loss. "
              "Absolute here means the actual compressed loss, not |ΔL|. Lower loss is better. "
              "A larger compression penalty with more training need not erase the improvement in dense loss. "
              "Each figure has one row per size and these three columns; no averaging across sizes/configs/arms. "
              "Lines join measured checkpoints, with linear token and native-token-loss axes. "
              "Every table sequence follows **step16000 → step64000 → step143000**. "
              "Endpoint changes describe late versus early only; the trajectory shapes use both adjacent transitions with V36's 1e-12 numerical tolerance.", ""]
    for arm, cap in itertools.product(ARMS, CAPS):
        c = summary["results"][arm][cap]["curves"]
        stem = f"curves_{arm}_{cap}"
        relative = f"../../results/v36b-input-comparison/{stem}"
        seq = lambda values: " → ".join(f"{v:.5f}" for v in values)
        lines += [f"### {arm}: {cap}", "", f"![{arm} {cap}: dense, increment and absolute compressed loss]({relative}.png)", "",
                  f"[Full precision curve table (CSV)]({relative}.csv). "
                  f"The increment rises while absolute compressed loss falls from early to late in **{c['n_increment_rises_but_absolute_loss_falls']}/8** fixed size/config trajectories.", "",
                  "| Size | Dense L0 at the three steps | Dense late−early | Dense shape |", "|---|---|---:|---|"]
        for size in SIZES:
            t = next(t for t in c["trajectories"] if t["size"] == size)["dense_loss"]
            lines.append(f"| {size} | {seq(t['values'])} | {t['late_minus_early']:+.5f} | {t['monotonicity']} |")
        lines += ["", "| Size | Config | ΔL at the three steps | Absolute compressed L at the three steps | ΔL late−early | L late−early | ΔL / L shape |",
                  "|---|---:|---|---|---:|---:|---|"]
        for t in c["trajectories"]:
            d, a = t["delta_loss"], t["compressed_loss"]
            lines.append(f"| {t['size']} | {t['config']:g} | {seq(d['values'])} | {seq(a['values'])} | "
                         f"{d['late_minus_early']:+.5f} | {a['late_minus_early']:+.5f} | {d['monotonicity']} / {a['monotonicity']} |")
        examples = [t for t in c["trajectories"] if t["increment_rises_but_absolute_loss_falls"]]
        if examples:
            t = examples[0]
            lines += ["", f"For example, {t['size']} at config {t['config']:g}: the increment rises by {t['delta_loss']['late_minus_early']:.5f}, "
                      f"while dense loss changes by {t['dense_loss']['late_minus_early']:+.5f} and absolute compressed loss changes by "
                      f"{t['compressed_loss']['late_minus_early']:+.5f} nats/native token. This is increased fragility alongside improved absolute compressed loss."]
        lines.append("")
    lines += ["## Reproduction and audit trail", "", "```bash",
              "SDL_V36_SIZES=410m,1.4b python3 analysis/v36b_input_comparison.py --dry-run",
              "SDL_V36_SIZES=410m,1.4b python3 analysis/v36b_input_comparison.py",
              "python3 -m pytest -q tests/test_v36b_inputs.py tests/test_v36.py", "```", "",
              "With V36's default size environment, V36b also excludes 2.8b automatically and requires both clean sizes. "
              "Explicit loader subsets avoid changing V36's global SIZES. Analysis uses NumPy and the standard library; PNG rendering uses matplotlib's Agg CPU backend. "
              "The dry run prints the report without writing or importing matplotlib. "
              "`results/v36b-input-comparison/summary.json` retains source SHA256 hashes, architecture metadata, all train/test memberships, coefficients, training standardizers, ranks/condition numbers, "
              "every held-out prediction, baseline selection, paired metrics (including per-config slices), correlations and all raw curves. "
              "Inputs are rehashed before writing. This report, six full-precision CSV curve tables and six PNG figures are generated from that summary. "
              "The existing aggregate losses cannot establish independent probe/seed uncertainty. These are fixed-recipe-series predictive comparisons, not isolated causal effects of training tokens.", ""]
    return "\n".join(lines)


def plot_curves(summary, output):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sdl-v36b-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#0072B2", "#009E73", "#E69F00", "#D55E00")
    for arm, cap in itertools.product(ARMS, CAPS):
        c = summary["results"][arm][cap]["curves"]
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), sharex=True, sharey="col", layout="constrained")
        for i, size in enumerate(SIZES):
            ts = [t for t in c["trajectories"] if t["size"] == size]
            x = np.array(ts[0]["D0"])/1e9
            axes[i, 0].plot(x, ts[0]["dense_loss"]["values"], "o-", color="#333333", label="dense")
            for color, t in zip(colors, ts):
                label = f"density {t['config']:g}" if arm == "pruning" else f"{t['config']:g}-bit"
                axes[i, 1].plot(x, t["delta_loss"]["values"], "o-", color=color, label=label)
                axes[i, 2].plot(x, t["compressed_loss"]["values"], "o-", color=color, label=label)
            axes[i, 1].axhline(0, color="0.6", lw=.8, linestyle="--")
            axes[i, 0].set_ylabel(f"Pythia-{size}\nLoss (nats/native token)")
            for j, title in enumerate((r"Dense loss $L_{c,0}$", r"Compression increment $\Delta L_{m,c}$",
                                       r"Absolute compressed loss $L_{m,c}$")):
                ax = axes[i, j]
                if i == 0:
                    ax.set_title(title)
                ax.grid(alpha=.2)
                ax.set_xticks(x, [f"{v:.1f}" for v in x])
                if i == 1:
                    ax.set_xlabel(r"Training tokens $D_0$ (billions)")
            axes[i, 2].legend(fontsize=8, loc="best")
        fig.suptitle(f"{arm.capitalize()} / {cap} — measured clean Pythia trajectories\n"
                     r"$L_{m,c}=L_{c,0}+\Delta L_{m,c}$; larger increments can coexist with lower absolute loss", fontsize=12)
        fig.savefig(output / f"curves_{arm}_{cap}.png", dpi=160)
        plt.close(fig)


def write_outputs(summary, report, root=ROOT):
    root = Path(root)
    for relative, expected in summary["input_sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Input changed during analysis: {relative}")
    serialized = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    output = root / OUT.relative_to(ROOT)
    destination = root / REPORT.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plot_curves(summary, output)
    for arm, cap in itertools.product(ARMS, CAPS):
        data = summary["results"][arm][cap]["curves"]["compressed"]
        with (output / f"curves_{arm}_{cap}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (output / "summary.json").write_text(serialized)
    destination.write_text(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print report without writing artifacts")
    args = parser.parse_args(argv)
    summary = build_summary()
    report = render(summary)
    if args.dry_run:
        print(report, end="")
    else:
        write_outputs(summary, report)
        print(f"Wrote {OUT / 'summary.json'}, six CSV/PNG curve pairs, and {REPORT}")


if __name__ == "__main__":
    main()
