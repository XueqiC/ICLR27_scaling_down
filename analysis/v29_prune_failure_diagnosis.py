#!/usr/bin/env python3
"""CPU-only pruning diagnosis from saved JSON; no model/NPZ loading.

Run: python analysis/v29_prune_failure_diagnosis.py [--n-boot 10000]
All fits and preprocessing are repeated within model holdouts. Outcome-based
pre-cliff slices and full-curve amplitude fits are retrospective diagnostics,
never inputs to the unfiltered Mode A predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np
from scipy.optimize import minimize_scalar

try:
    from . import prediction_audit as audit
    from . import v18_law_fit as old
    from . import v28_new_source_prediction as v28
except ImportError:
    import prediction_audit as audit
    import v18_law_fit as old
    import v28_new_source_prediction as v28

ROOT = audit.ROOT
OUT = ROOT / "results/v29-prune-diagnosis"
REPORT = ROOT / "paper/docs/PRUNE_FAILURE_DIAGNOSIS.md"
FROZEN = ROOT / "results/v28-new-source-pred/frozen_predictions.json"
CAPS = audit.CAPABILITIES
GRID = (.9, .8, .7, .6, .55, .5, .45, .4, .35, .3)
# Fixed before evaluation. Lambda has units of squared normalized shape.
LAMBDAS = (0., 1e-8, 1e-6, 1e-4, .01, 1., 100.)


def parse_losses(payload):
    """The actual schema is density-string -> capability -> absolute CE."""
    parsed = {}
    for key, block in payload.items():
        if key.startswith("_"):
            continue
        density = audit.finite(key)
        if not 0 < density <= 1 or density in parsed:
            raise ValueError("Invalid or duplicate pruning density")
        parsed[density] = {cap: audit.finite(block[cap]) for cap in CAPS}
    if 1. not in parsed:
        raise ValueError("Missing source-dense reference at density 1.0")
    return parsed


def restrict_range(rows, minimum=.6, precliff=False):
    return [r for r in rows if r["density"] >= minimum
            and (not precliff or r["precliff"])]


def load_rows():
    metadata = audit.read_json(v28.METADATA)
    frozen = audit.read_json(FROZEN)
    if frozen["freeze_sha256"] != v28.seal(frozen):
        raise ValueError("Frozen predictor integrity check failed")
    paths = [v28.METADATA, FROZEN]
    rows, inventory = [], []
    specs = {**metadata["models"], "Qwen3-8B": frozen["source"]}
    for model, info in sorted(specs.items()):
        tags = ["Qwen--Qwen3-8B"] if model == "Qwen3-8B" else [model, "Qwen--" + model]
        candidates = [ROOT / "results" / base / tag / "prune_losses.json"
                      for base in ("v6-capability-geometry", "v9-capability-regions") for tag in tags]
        existing = list(dict.fromkeys(p for p in candidates if p.is_file()))
        if not existing:
            raise ValueError(f"No pruning losses for {model}")
        merged, used, anchors = {}, {}, []
        for path in existing:
            data = parse_losses(audit.read_json(path)); paths.append(path)
            anchors.append(data[1.])
            if any(not np.isclose(data[1.][c], anchors[0][c], atol=1e-8, rtol=0) for c in CAPS):
                raise ValueError(f"Cannot mix pruning runs with different dense anchors: {model}")
            for density, block in data.items():
                if density in merged and block != merged[density]:
                    raise ValueError(f"Conflicting duplicate pruning observation: {model}/{density}")
                merged.setdefault(density, block); used.setdefault(density, str(path.relative_to(ROOT)))
        required = (.9, .8, .7, .6) if model == "Qwen3-8B" else GRID
        if not set(required).issubset(merged):
            raise ValueError(f"Incomplete common density grid: {model}")
        for cap in CAPS:
            crossed = False
            for density in sorted((d for d in merged if d < 1), reverse=True):
                y = merged[density][cap] - merged[1.][cap]
                crossed = crossed or y > old.PRECLIFF_CAP
                rows.append({"row_id": f"{model}|{cap}|{density:g}", "model": model,
                             "family": info["family"], "N0": info["N0"], "capability": cap,
                             "density": density, "dense_loss": merged[1.][cap],
                             "loss": merged[density][cap], "observed": y,
                             "precliff": not crossed, "source_path": used[density]})
        inventory.append({"model": model, "densities": sorted(merged, reverse=True),
                          "dense": merged[1.], "files": [str(p.relative_to(ROOT)) for p in existing],
                          "metadata": {k: v for k, v in audit.read_json(existing[0]).items() if k.startswith("_")}})
    # Verify saved development outcomes, metadata and row selection against v28.
    for cap in CAPS:
        for saved in frozen["development_rows"]["pruning"][cap]:
            rs = [r for r in rows if r["model"] == saved["model"] and r["capability"] == cap]
            for r in rs:
                if r["density"] in v28.PRUNE_DEV:
                    if any(r[k] != saved[k] for k in ("family", "N0", "dense_loss")) or not np.isclose(
                            r["observed"], saved["deltas"][str(r["density"])], atol=1e-12, rtol=0):
                        raise ValueError("Saved v28 development inputs differ from current JSON")
    v9 = sorted((ROOT / "results/v9-capability-regions").glob("*/prune*loss*.json"))
    return rows, frozen, paths, {"schema": "density string -> {math, code, qa}; 1.0 is own dense; underscore keys are metadata",
                               "models": inventory, "v9_pruning_files_found": [str(p.relative_to(ROOT)) for p in v9],
                               "frozen_development_inputs_verified": True}


def panel_rows(rows):
    result = []
    for model in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == model]
        result.append({**v28.basic_input(rs[0]), "capability": rs[0]["capability"],
                       "deltas": {str(r["density"]): r["observed"] for r in rs}})
    return result


def fit_v28(rows):
    """Exact v28 fit on its rectangular panel; same form for range refits."""
    panels = panel_rows(rows)
    if all(set(r["deltas"]) == {str(d) for d in v28.PRUNE_DEV} for r in panels):
        return v28.fit_arm(panels, "pruning")
    if len(panels) < 2:
        raise ValueError("Need two development models")
    def profile(gamma):
        labels, sse = [], 0.
        for r in panels:
            x = np.array([v28.shape("pruning", float(d), gamma) for d in r["deltas"]])
            y = np.array(list(r["deltas"].values()))
            a = float(x @ y / (x @ x)); labels.append(a)
            sse += float(np.sum((y-a*x)**2))
        return labels, sse
    grid = np.linspace(.05, 8., 161)
    i = int(np.argmin([profile(g)[1] for g in grid]))
    opt = minimize_scalar(lambda g: profile(g)[1], bounds=(grid[max(i-1, 0)], grid[min(i+1, 160)]), method="bounded")
    gamma = float(min([.05, 8., opt.x], key=lambda g: profile(g)[1]))
    labels, sse = profile(gamma)
    return {"arm": "pruning", "gamma": gamma, "shape_fit_sse": sse,
            "mapping": v28.fit_mapping([v28.basic_input(r) for r in panels], labels),
            "development_coefficient_labels": dict(zip([r["model"] for r in panels], labels)),
            "mean_coefficient": float(np.mean(labels))}


def calibrated_amplitude(response, shape_at_calibration, population_mean, regularization=0.):
    """argmin_a (y0-a*x0)^2 + lambda*(a-mu_dev)^2, signed and unclipped."""
    if regularization < 0 or shape_at_calibration <= 0:
        raise ValueError("Need nonnegative regularization and positive calibration shape")
    x = shape_at_calibration
    return float((x*response + regularization*population_mean)/(x*x+regularization))


def select_regularization(train):
    """Inner LOMO, model-equal MAE at .8/.7/.6; outer target never enters."""
    errors = np.zeros(len(LAMBDAS)); folds = []
    for split in audit.splits(train, "model"):
        inner = [train[i] for i in split["train_indices"]]
        test = [train[i] for i in split["test_indices"]]
        fit = fit_v28(inner)
        point, = [r for r in test if r["density"] == .9]
        test = [r for r in test if r["density"] in v28.PRUNE_TEST]
        x0 = v28.shape("pruning", .9, fit["gamma"])
        scores = []
        for lam in LAMBDAS:
            a = calibrated_amplitude(point["observed"], x0, fit["mean_coefficient"], lam)
            scores.append(float(np.mean([abs(a*v28.shape("pruning", r["density"], fit["gamma"])-r["observed"]) for r in test])))
        errors += scores
        folds.append({"held_out": split["held_out"], "train_models": fit["mapping"]["train_models"], "mae_by_lambda": scores})
    errors /= len(folds)
    # Larger regularization wins exact ties; no outer-outcome selection.
    index = min(range(len(LAMBDAS)), key=lambda i: (errors[i], -LAMBDAS[i]))
    return {"lambda": LAMBDAS[index], "grid": list(LAMBDAS), "inner_folds": folds,
            "inner_model_mean_mae_by_lambda": errors.tolist()}


def evaluate_lomo(rows, training="v28", shrinkage=False):
    """Every predictor sees identical training rows and identical test cells."""
    common = [r for r in rows if r["density"] in GRID]
    records, folds = [], []
    for split in audit.splits(common, "model"):
        available = [common[i] for i in split["train_indices"]]
        target = [common[i] for i in split["test_indices"]]
        train = available if training == "all" else restrict_range(available, .6, training == "smooth")
        old_fit = old.fit_pruning_candidate(train, "density_only")
        old_gamma = old_fit["parameters"]["gamma"]
        for cap in CAPS:
            tr = [r for r in train if r["capability"] == cap]
            te = [r for r in target if r["capability"] == cap]
            fit = fit_v28(tr)
            cap_fit = old.fit_pruning_candidate(tr, "density_only")
            point, = [r for r in te if r["density"] == .9]
            point80, = [r for r in te if r["density"] == .8]
            x0 = v28.shape("pruning", .9, fit["gamma"])
            direct = calibrated_amplitude(point["observed"], x0, fit["mean_coefficient"])
            mapped = v28.predict_mapping(fit["mapping"], v28.basic_input(point))
            selection = select_regularization(tr) if shrinkage else None
            shrunk = calibrated_amplitude(point["observed"], x0, fit["mean_coefficient"], selection["lambda"]) if selection else None
            folds.append({"held_out": split["held_out"], "capability": cap, "training": training,
                          "train_row_ids": [r["row_id"] for r in train], "old_fit": old_fit,
                          "old_cap_fit": cap_fit, "v28_fit": fit,
                          "mode_A_inputs": v28.basic_input(point),
                          "mode_B_calibration": {"density": .9, "response": point["observed"], "row_id": point["row_id"]},
                          "regularization_selection": selection,
                          "mapped_amplitude": mapped, "direct_amplitude": direct, "shrunk_amplitude": shrunk})
            for r in te:
                if r["density"] == .9:
                    continue
                x = v28.shape("pruning", r["density"], fit["gamma"])
                pred = {"zero": 0., "old_A": old.predict_pruning_candidate(old_fit, r),
                        "old_cap_A": old.predict_pruning_candidate(cap_fit, r),
                        "v28_mean_A": fit["mean_coefficient"]*x, "v28_A": mapped*x,
                        "old_B": point["observed"]*((1-r["density"])/.1)**old_gamma,
                        "v28_B": direct*x}
                if selection:
                    pred["v28_B_shrink"] = shrunk*x
                    # A distinct sensitivity mode, scored only at densities <= .7.
                    if r["density"] != .8:
                        pred["v28_B80"] = point80["observed"]*x/v28.shape("pruning", .8, fit["gamma"])
                records.append({**r, "predictions": pred})
    return {"folds": folds, "records": records}


def estimate(values, groups, n_boot):
    arr = np.asarray(values, dtype=float)
    draws = audit.bootstrap_means(arr, groups, n_boot=n_boot)
    return {"value": float(arr.mean()), "ci95": audit.interval(draws[:, 0]),
            "n_models": len(set(groups))}


def summarize(records, n_boot):
    if not records:
        return {"n_rows": 0, "n_models": 0, "metrics": {}}
    names = [n for n in records[0]["predictions"] if all(n in r["predictions"] for r in records)]
    predictions = {n: [r["predictions"][n] for r in records] for n in names}
    metrics, errors, draws = audit.compare_predictions(records, predictions, "zero", n_boot=n_boot)
    for name, metric in metrics.items():
        metric["improvement_over_zero"] = -metric["mae_minus_reference"]
        metric["improvement_ci95"] = [-metric["difference_ci95"][1], -metric["difference_ci95"][0]]
    contrasts = {}
    for a, b in (("v28_A", "old_A"), ("old_A", "old_B"), ("v28_A", "v28_B"),
                 ("v28_A", "v28_mean_A"), ("old_cap_A", "old_A"),
                 ("v28_mean_A", "old_cap_A"), ("v28_B", "v28_B_shrink"), ("v28_B", "v28_B80")):
        if a in names and b in names:
            i, j = names.index(a), names.index(b)
            contrasts[a+"_minus_"+b] = {"value": float(np.mean(errors[:, i]-errors[:, j])),
                                       "ci95": audit.interval(draws[:, i]-draws[:, j])}
    groups = [r["model"] for r in records]
    return {"n_rows": len(records), "n_models": len(set(groups)), "models": sorted(set(groups)),
            "densities": sorted({r["density"] for r in records}, reverse=True),
            "metrics": metrics, "contrasts": contrasts,
            "mean_signed_response": estimate([r["observed"] for r in records], groups, n_boot),
            "mean_absolute_response": estimate([abs(r["observed"]) for r in records], groups, n_boot)}


def summaries(records, n_boot):
    return {cap: summarize([r for r in records if cap == "pooled" or r["capability"] == cap], n_boot)
            for cap in (*CAPS, "pooled")}


def paired_estimands(terms, contrasts, n_boot):
    """Joint model draws for differently sized slices; keep cell weighting.

    Each term is (records, candidate). Bootstrap sums AND denominators with
    the same sampled models, then form differences. Never resample densities.
    """
    models = sorted({r["model"] for rs, _ in terms.values() for r in rs})
    labels = list(terms)
    cells = []
    for model in models:
        row = []
        for rs, name in terms.values():
            subset = [r for r in rs if r["model"] == model]
            row.extend([sum(abs(r["predictions"][name]-r["observed"]) for r in subset), len(subset)])
        cells.append(row)
    cells = np.array(cells, dtype=float)
    boots = audit.bootstrap_means(cells, models, n_boot=n_boot)
    valid = np.all(boots[:, 1::2] > 0, axis=1)
    boots = boots[valid, 0::2]/boots[valid, 1::2]
    values = cells[:, 0::2].sum(axis=0)/cells[:, 1::2].sum(axis=0)
    out = {}
    for name, weights in contrasts.items():
        w = np.array([weights.get(label, 0) for label in labels])
        out[name] = {"value": float(values @ w), "ci95": audit.interval(boots @ w),
                     "n_models": len(models), "undefined_bootstrap_draws": int((~valid).sum())}
    return out


def range_attribution(records, n_boot, smooth_records=None):
    out = {}
    for cap in (*CAPS, "pooled"):
        full = [r for r in records if r["density"] >= .6 and (cap == "pooled" or r["capability"] == cap)]
        smooth = restrict_range(full, .6, True)
        terms = {"old_sA": (smooth, "old_A"), "old_sB": (smooth, "old_B"),
                 "old_fA": (full, "old_A"), "new_sA": (smooth, "v28_A"), "new_fA": (full, "v28_A")}
        out[cap] = paired_estimands(terms, {
            "calibration_old_A_minus_B_smooth": {"old_sA": 1, "old_sB": -1},
            "range_old_full_minus_smooth": {"old_fA": 1, "old_sA": -1},
            "residual_form_full_v28_minus_old_A": {"new_fA": 1, "old_fA": -1},
            "controlled_total_v28_full_A_minus_old_smooth_B": {"new_fA": 1, "old_sB": -1},
            "range_v28_full_minus_smooth": {"new_fA": 1, "new_sA": -1},
            "residual_form_smooth_v28_minus_old_A": {"new_sA": 1, "old_sA": -1},
            "range_by_form_interaction": {"new_fA": 1, "new_sA": -1, "old_fA": -1, "old_sA": 1},
        }, n_boot)
        if smooth_records is not None:
            smooth_refit = [r for r in restrict_range(smooth_records, .6, True)
                            if cap == "pooled" or r["capability"] == cap]
            out[cap].update(paired_estimands({**terms, "old_refit": (smooth_refit, "old_A"),
                                            "new_refit": (smooth_refit, "v28_A")}, {
                "training_range_old_original_minus_smooth_refit": {"old_sA": 1, "old_refit": -1},
                "residual_form_both_smooth_refit": {"new_refit": 1, "old_refit": -1},
                "range_training_plus_scoring_old": {"old_fA": 1, "old_refit": -1},
                "controlled_A_gap_v28_full_minus_old_smooth_refit": {"new_fA": 1, "old_refit": -1},
            }, n_boot))
    return out


def historical_audit(rows, n_boot):
    pre = [r for r in rows if r["precliff"]]
    train = [r for r in pre if r["density"] >= .55]
    test = [r for r in pre if r["density"] < .55]
    fit = old.fit_pruning_candidate(train, "density_only")
    records, folds = [], []
    for model in sorted({r["model"] for r in test}):
        independent = [r for r in train if r["model"] != model]
        transfer = old.fit_pruning_candidate(independent, "density_only")
        own = [r for r in train if r["model"] == model]
        folds.append({"held_out": model, "historical_target_training_row_ids": [r["row_id"] for r in own],
                      "historical_target_compressed_configurations": sorted({r["density"] for r in own}, reverse=True),
                      "transfer_train_row_ids": [r["row_id"] for r in independent], "transfer_fit": transfer})
        for r in test:
            if r["model"] == model:
                records.append({**r, "predictions": {"zero": 0., "historical_own_points": old.predict_pruning_candidate(fit, r),
                                                      "historical_transfer_A": old.predict_pruning_candidate(transfer, r)}})
    family_records = []
    # The original report pooled gemma3/gemma4 into one paper family. v28's
    # metadata families remain unchanged everywhere outside this reproduction.
    paper_pre = [{**r, "family": old.paper_family(r["family"])} for r in pre]
    for split in audit.splits(paper_pre, "family"):
        fit_family = old.fit_pruning_candidate([paper_pre[i] for i in split["train_indices"]], "density_only")
        for i in split["test_indices"]:
            r = paper_pre[i]
            family_records.append({**r, "predictions": {"zero": 0., "old_family_A": old.predict_pruning_candidate(fit_family, r)}})
    comparison = paired_estimands({"A": (records, "historical_transfer_A"), "own": (records, "historical_own_points")},
                                  {"target_point_benefit_A_minus_own": {"A": 1, "own": -1}}, n_boot)
    return {"n_precliff": len(pre), "n_train": len(train), "n_test": len(test), "fit": fit,
            "folds": folds, "records": records, "summaries": summaries(records, n_boot),
            "own_point_effect": comparison, "family_records": family_records,
            "family_summaries": summaries(family_records, n_boot)}


def calibration_diagnostics(result, n_boot):
    diagnostics = []
    for fold in result["folds"]:
        rs = [r for r in result["records"] if r["model"] == fold["held_out"]
              and r["capability"] == fold["capability"] and r["density"] in v28.PRUNE_TEST]
        fit = fold["v28_fit"]; gamma = fit["gamma"]
        x = np.array([v28.shape("pruning", r["density"], gamma) for r in rs])
        y = np.array([r["observed"] for r in rs])
        oracle = float(x @ y/(x @ x))  # Diagnostic only; never feeds a prediction.
        x0 = v28.shape("pruning", .9, gamma)
        lam = fold["regularization_selection"]["lambda"]
        weight = x0*x0/(x0*x0+lam)
        y0 = fold["mode_B_calibration"]["response"]
        direct, shrunk = fold["direct_amplitude"], fold["shrunk_amplitude"]
        diagnostics.append({"model": fold["held_out"], "capability": fold["capability"],
                            "calibration_signed_response": y0, "calibration_absolute_response": abs(y0),
                            "mean_absolute_test_response": float(np.abs(y).mean()),
                            "calibration_to_test_response_ratio": float(abs(y0)/max(np.abs(y).mean(), 1e-12)),
                            "gamma": gamma, "lambda": lam, "weight_on_direct": weight,
                            "population_amplitude": fit["mean_coefficient"], "mapped_amplitude": fold["mapped_amplitude"],
                            "direct_amplitude": direct, "shrunk_amplitude": shrunk, "diagnostic_best_test_amplitude": oracle,
                            "direct_amplitude_error": abs(direct-oracle), "shrink_amplitude_error": abs(shrunk-oracle),
                            "amplitude_error_reduction": abs(direct-oracle)-abs(shrunk-oracle),
                            "calibration_shape_residual": y0-oracle*x0,
                            "abs_calibration_shape_residual": abs(y0-oracle*x0),
                            "direct_gain_at_0.6": v28.shape("pruning", .6, gamma)/x0,
                            "shrink_gain_at_0.6": weight*v28.shape("pruning", .6, gamma)/x0,
                            "direct_gain_at_0.3": v28.shape("pruning", .3, gamma)/x0,
                            "shrink_gain_at_0.3": weight*v28.shape("pruning", .3, gamma)/x0,
                            "direct_mae": float(np.mean([abs(r["predictions"]["v28_B"]-r["observed"]) for r in rs])),
                            "shrink_mae": float(np.mean([abs(r["predictions"]["v28_B_shrink"]-r["observed"]) for r in rs]))})
    statistics = {}
    for cap in (*CAPS, "pooled"):
        ds = [r for r in diagnostics if cap == "pooled" or r["capability"] == cap]
        groups = [r["model"] for r in ds]
        statistics[cap] = {k: estimate([r[k] for r in ds], groups, n_boot)
                           for k in ds[0] if k not in ("model", "capability", "lambda")}
        statistics[cap]["fraction_shrink_improves_model_capability"] = estimate(
            [float(r["shrink_mae"] < r["direct_mae"]) for r in ds], groups, n_boot)
    return {"model_diagnostics": diagnostics, "statistics": statistics,
            "common_test_0.7_0.6": summaries([r for r in result["records"] if r["density"] in (.7, .6)], n_boot),
            "noise_identifiability": "Only aggregate losses are saved. Gains are exact sensitivity to perturbations of delta at 0.9; residuals also contain shape error. No measurement variance or SNR can be estimated."}


def amplitude_decomposition(predicted, observed):
    p, y = np.asarray(predicted, float), np.asarray(observed, float)
    denom = float(p @ p)
    if denom <= 0:
        raise ValueError("Cannot scale a zero predicted curve")
    scale = float(p @ y/denom)
    residual = scale*p-y
    mse = float(np.mean((p-y)**2))
    amplitude_mse = float(np.mean(((1-scale)*p)**2))
    shape_mse = float(np.mean(residual**2))
    positive = max(0., scale)
    return {"signed_scale": scale, "nonnegative_scale": positive,
            "raw_mae": float(np.mean(np.abs(p-y))), "scaled_mae": float(np.mean(np.abs(residual))),
            "mae_reduction_after_scale": float(np.mean(np.abs(p-y))-np.mean(np.abs(residual))),
            "nonnegative_scaled_mae": float(np.mean(np.abs(positive*p-y))),
            "raw_mse": mse, "amplitude_or_sign_mse": amplitude_mse, "shape_mse": shape_mse,
            "amplitude_or_sign_mse_fraction": amplitude_mse/mse if mse else 0.,
            "shape_mse_fraction": shape_mse/mse if mse else 0.,
            "mean_observed": float(y.mean()), "mean_absolute_observed": float(np.abs(y).mean()),
            "mean_predicted": float(p.mean()), "sign_reversal_required": scale < 0}


def threshold_crossing(densities, responses, threshold=1., absolute=False):
    """Bracket first crossing on measured grid; do not invent an actual cliff."""
    previous = 1.
    for d, response in sorted(zip(densities, responses), reverse=True):
        value = abs(response) if absolute else response
        if value > threshold:
            return {"status": "crossed", "first_grid_density": float(d), "density_bracket": [float(d), float(previous)]}
        previous = d
    return {"status": "not_observed", "first_grid_density": None, "density_bracket": None,
            "if_crossing_exists_density_below": float(min(densities))}


def qwen_diagnosis(rows, dev, frozen, n_boot):
    records, diagnostics, crossings = [], {}, {}
    for cap in CAPS:
        rs = [r for r in rows if r["capability"] == cap]
        fit = frozen["methods"]["pruning"]["by_capability"][cap]["fit"]
        point, = [r for r in rs if r["density"] == .9]
        training = [r for r in dev if r["density"] in v28.PRUNE_DEV]
        old_fit = old.fit_pruning_candidate(training, "density_only")
        cap_fit = old.fit_pruning_candidate([r for r in training if r["capability"] == cap], "density_only")
        selection = select_regularization([r for r in training if r["capability"] == cap])
        x0 = v28.shape("pruning", .9, fit["gamma"])
        shrunk = calibrated_amplitude(point["observed"], x0, fit["mean_coefficient"], selection["lambda"])
        for r in rs:
            if r["density"] not in v28.PRUNE_TEST:
                continue
            pred = {"zero": 0., "old_A": old.predict_pruning_candidate(old_fit, r),
                    "old_cap_A": old.predict_pruning_candidate(cap_fit, r),
                    "v28_mean_A": fit["mean_coefficient"]*v28.shape("pruning", r["density"], fit["gamma"]),
                    "old_B": point["observed"]*((1-r["density"])/.1)**old_fit["parameters"]["gamma"],
                    "v28_B_shrink": shrunk*v28.shape("pruning", r["density"], fit["gamma"])}
            for mode in ("A", "B"):
                entry, = [p for p in frozen["predictions"] if p["arm"] == "pruning" and p["mode"] == mode
                          and p["capability"] == cap and p["coordinate"] == r["density"]]
                formula = entry["delta_formula"]
                pred["v28_"+mode] = formula["intercept"] + formula["dense_slope"]*r["dense_loss"] + formula["calibration_slope"]*point["loss"]
                recomputed = v28.predict_arm(fit, v28.basic_input(r), [r["density"]], mode,
                                            {"coordinate": .9, "loss": point["loss"]} if mode == "B" else None)[0]
                if not np.isclose(pred["v28_"+mode], recomputed, atol=1e-10, rtol=0):
                    raise ValueError("Frozen formula and stored coefficient fit disagree")
            records.append({**r, "predictions": pred})
        scored = [r for r in records if r["capability"] == cap]
        stats = amplitude_decomposition([r["predictions"]["v28_A"] for r in scored], [r["observed"] for r in scored])
        diagnostics[cap] = {k: v if isinstance(v, bool) else estimate([v], ["Qwen3-8B"], n_boot) for k, v in stats.items()}
        diagnostics[cap]["calibration_response"] = estimate([point["observed"]], ["Qwen3-8B"], n_boot)
        diagnostics[cap]["shrinkage_selection"] = selection
        p = [v28.predict_arm(fit, v28.basic_input(r), [r["density"]])[0] for r in rs]
        a = v28.predict_mapping(fit["mapping"], v28.basic_input(point))
        crossings[cap] = {}
        for threshold in (.5, 1., 2., 8.):
            by_definition = {}
            for absolute in (False, True):
                aa = abs(a) if absolute else a
                theoretical = 1-.3*(threshold/aa)**(1/fit["gamma"]) if aa > 0 else None
                by_definition["absolute_response" if absolute else "positive_damage"] = {
                    "actual": threshold_crossing([r["density"] for r in rs], [r["observed"] for r in rs], threshold, absolute),
                    "predicted": threshold_crossing([r["density"] for r in rs], p, threshold, absolute),
                    "predicted_after_signed_scale": threshold_crossing([r["density"] for r in rs], [stats["signed_scale"]*v for v in p], threshold, absolute),
                    "model_implied_continuous_crossing": estimate([theoretical], ["Qwen3-8B"], n_boot) if theoretical is not None and theoretical >= 0 else None}
            crossings[cap][str(threshold)] = by_definition
    pooled = amplitude_decomposition([r["predictions"]["v28_A"] for r in records], [r["observed"] for r in records])
    diagnostics["pooled_single_scale"] = {k: v if isinstance(v, bool) else estimate([v], ["Qwen3-8B"], n_boot) for k, v in pooled.items()}
    return {"records": records, "summaries": summaries(records, n_boot), "amplitude": diagnostics, "crossings": crossings,
            "ci_limitation": "One source model: every model-bootstrap interval is degenerate [value,value], descriptive only. No population or measurement uncertainty is identified.",
            "identity_limitation": "Freeze names Qwen/Qwen3-8B-Base; actual pruning JSON contains no checkpoint or revision identity. Directory Qwen--Qwen3-8B alone cannot verify Base versus post-trained identity."}


def mfmt(metric, key="value", ci="ci95"):
    return audit.with_ci(metric[key], metric[ci])


def metric_table(summary, names):
    lines = ["| Capability | Models / rows | Mean absolute response | " + " | ".join(names) + " |",
             "|---|---:|---:|" + "---:|"*len(names)]
    for cap in (*CAPS, "pooled"):
        s = summary[cap]
        cells = [mfmt(s["metrics"][name], "mae", "mae_ci95") if name in s["metrics"] else "—" for name in names]
        lines.append(f"| {cap} | {s['n_models']} / {s['n_rows']} | {mfmt(s['mean_absolute_response'])} | " + " | ".join(cells) + " |")
    return lines


def render_report(s):
    hist, panels, attr, cal, qwen = (s[k] for k in ("historical", "matched", "attribution", "calibration", "qwen3_8b"))
    full = panels["v28_grid"]; smooth = panels["v28_grid_precliff"]
    h = hist["summaries"]["pooled"]["metrics"]["historical_own_points"]
    qs = qwen["summaries"]["pooled"]["metrics"]
    lines = ["# Pruning prediction failure diagnosis", "",
             f"The old density-only headline reproduces as {mfmt(h, 'mae', 'mae_ci95')} nats on 28 pre-cliff rows from four models. "
             "The reported 1.3–1.9 values are v28 development LOMO errors, not Qwen3-8B prospective errors. "
             f"Frozen Qwen3-8B Mode A MAE is {mfmt(qs['v28_A'], 'mae', 'mae_ci95')}, versus zero-change {mfmt(qs['zero'], 'mae', 'mae_ci95')}. "
             "Range selection, coefficient transfer, and calibration conditioning are distinguishable failure sources; these data do not support a blanket unpredictability claim.", "",
             "All intervals below are 95% paired MODEL-bootstrap intervals, using " + str(s["n_boot"]) + " resamples. "
             "Whole models retain their capability/density cells; means weight cells equally. Fits are held fixed, so intervals describe this small panel, not retraining or measurement uncertainty. "
             "Cross-range contrasts resample the same models and recompute each slice's denominator. Counts, chosen hyperparameters, and coordinate brackets are design facts, not estimated metrics. "
             "Qwen3-8B has one model: its [x,x] intervals are explicitly degenerate and cannot establish population uncertainty.", "",
             "![Qwen curve decomposition and development calibration errors](../../results/v29-prune-diagnosis/diagnosis.png)", "",
             "Figure: top, Qwen3-8B observed curves, frozen Mode A, and an oracle scalar correction on the scored grid (one model; no nondegenerate model CI). "
             "Bottom, development LOMO MAEs with model-bootstrap intervals, using a logarithmic vertical scale. Oracle correction is never used in the LOMO results.", "",
             "## Inputs and matched protocol", "",
             "Only JSON observations were loaded. Twelve development models come from the v28 metadata whitelist; Qwen3-8B is held out completely. "
             "All v6 grids are complete, and v9 contains no alternate pruning-loss JSON. The schema is `{density: {math, code, qa}}`; `1.0` supplies each source's own dense CE. "
             "Underscore metadata and archive copies are excluded. Saved v28 development inputs and its frozen formula seal were verified. Historical reproduction alone includes the two models' infill densities.", "",
             "Mode A permits N0, family, and dense capability loss, with zero target compression points. Mode B adds exactly density 0.9 per capability; 0.9 is excluded from every matched test. "
             "Main training uses 0.9/0.8/0.7/0.6 on other models; testing uses the same target rows at 0.8/0.7/0.6. All gamma, coefficient, family, and feature standardization fits exclude the target model. "
             "Mode B shrinkage is separate and never replaces Mode A.", "",
             "The exact v18 `density_only` form is a single signed `A(1-d)^gamma`, pooled across capabilities, fitted by original-scale least squares with gamma in [0.05,8]. "
             "We import its original fitter and predictor. This differs from v14's mechanism-corrected, positive-residual log fits, which are not the C5 headline estimator. "
             "`old_cap_A` keeps the same form but fits each capability separately. v28 uses `a_c(model)((1-d)/0.3)^gamma_c`, profiles a shared shape with per-model labels, "
             "then maps labels from log(N0/1e9), family, and dense L_c using its unchanged ridge helper. `v28_mean_A` replaces that mapping with the training-population coefficient mean. "
             "Thus the density exponent family is shared by both approaches; a remaining difference can arise from pooling, shape estimation, or amplitude mapping rather than a new density function.", "",
             "## Historical score and calibration access", "",
             f"Reconstruction yields {hist['n_precliff']} pre-cliff rows, {hist['n_train']} training rows at d≥0.55, and {hist['n_test']} test rows below 0.55. "
             "The test models are Qwen3-1.7B, Qwen3-4B, olmo3-32b, and olmo3-7b. The historical fit includes their shallow compressed observations in its pooled coefficients; "
             "it does not fit a separate amplitude to each target. Removing each target's contribution while retaining exactly its test cells isolates that access.", ""]
    lines += metric_table(hist["summaries"], ["historical_own_points", "historical_transfer_A", "zero"])
    lines += ["", "Target-compression access benefit (transfer-only MAE minus original MAE): " +
              mfmt(hist["own_point_effect"]["target_point_benefit_A_minus_own"]) + ". "
              "This measures the actual historical use of target points; the one-point B experiment below is a different calibration protocol.", "",
              "| Historical target | Compressed configurations contributing to pooled training |", "|---|---|"]
    for f in hist["folds"]:
        lines.append(f"| {f['held_out']} | {', '.join(map(str, f['historical_target_compressed_configurations']))} |")
    lines += ["", "The historical leave-one-family-out density-only score is " +
              mfmt(hist["family_summaries"]["pooled"]["metrics"]["old_family_A"], "mae", "mae_ci95") +
              "; it has no held-out-family compression access. This excludes target calibration as a universal explanation for the old ~0.33 headline.", "",
              "## Matched-condition LOMO", "", "Entries are MAEs; response amplitudes and zero-change are reported on identical rows.", ""]
    lines += metric_table(full, ["old_A", "old_cap_A", "v28_mean_A", "v28_A", "zero"])
    lines += ["", "| Capability | old A gain over zero | v28 A gain over zero | v28 A minus old A | Mapping effect: v28 A minus mean |",
              "|---|---:|---:|---:|---:|"]
    for cap in (*CAPS, "pooled"):
        x = full[cap]
        lines.append(f"| {cap} | {mfmt(x['metrics']['old_A'], 'improvement_over_zero', 'improvement_ci95')} | "
                     f"{mfmt(x['metrics']['v28_A'], 'improvement_over_zero', 'improvement_ci95')} | "
                     f"{mfmt(x['contrasts']['v28_A_minus_old_A'])} | {mfmt(x['contrasts']['v28_A_minus_v28_mean_A'])} |")
    lines += ["", "Positive gain over zero is better; negative candidate-minus-candidate MAE is better for the first candidate. "
              "Zero-change is the required simple comparator. Training-population mean is also shown; zero is not assumed to win every capability. "
              "Matched Mode A excludes unequal target calibration and unequal test-row difficulty as explanations for its remaining predictor gap.", "",
              "On this matched grid v28 is lower-error than the old form in all three capabilities, although the paired intervals include zero. "
              "The apparent jump from the historical score is therefore not a demonstrated deterioration in v28's functional form. "
              "Neither Mode A method establishes an improvement over zero-change across this small model panel.", "",
              "## Damage range and three-way attribution", "",
              "The historical pre-cliff rule removes the first positive ΔL>1 nat and every deeper point, separately per model/capability. "
              "It is an outcome-selected diagnostic, not an available Mode A deployment filter. Negative responses are retained, including large QA decreases. "
              "Density≥0.6 alone is not smooth: several models cross the historical damage threshold by 0.8 or 0.7. "
              "The next table holds both fitted predictors fixed and restricts scoring to d≥0.6 AND pre-cliff.", ""]
    lines += metric_table(smooth, ["old_A", "v28_A", "old_B", "v28_B", "zero"])
    lines += ["", "The following controlled bridge is additive: old B on smooth rows → old A on smooth rows (calibration access) → old A on all v28 rows (damage range) → v28 A on those same rows (residual form/estimation). "
              "It diagnoses matched conditions; it is not a causal decomposition of two historical scores with different test sets. Changing the order changes components when range and form interact.", "",
              "| Capability | Calibration: old A−B, smooth | Range: old A full−smooth | Residual form: v28 A−old A, full | Sum / controlled gap |",
              "|---|---:|---:|---:|---:|"]
    for cap, a in attr.items():
        keys = ("calibration_old_A_minus_B_smooth", "range_old_full_minus_smooth", "residual_form_full_v28_minus_old_A", "controlled_total_v28_full_A_minus_old_smooth_B")
        lines.append(f"| {cap} | " + " | ".join(mfmt(a[k]) for k in keys) + " |")
    lines += ["", "| Capability | v28 range penalty full−smooth | Residual v28−old on smooth | Range × form interaction |", "|---|---:|---:|---:|"]
    for cap, a in attr.items():
        lines.append(f"| {cap} | " + " | ".join(mfmt(a[k]) for k in ("range_v28_full_minus_smooth", "residual_form_smooth_v28_minus_old_A", "range_by_form_interaction")) + " |")
    lines += ["", "Matched one-point calibration on the full original v28 test grid (exactly d=0.9; no target shape fitting):", ""]
    lines += metric_table(full, ["old_A", "old_B", "v28_A", "v28_B", "zero"])
    lines += ["", "Deep/collapse extension: train both on the original grid, then score identical common-grid rows from d=0.8 through 0.3. "
              "This is extrapolation beyond v28's frozen domain and is labelled as such; it cannot explain the original v28 errors at d≥0.6 by itself.", ""]
    lines += metric_table(panels["extended_grid"], ["old_A", "v28_A", "v28_B", "zero"])
    lines += ["", "To distinguish training-range contamination from scoring-range selection, both forms are also refitted on other models' d≥0.6 pre-cliff rows and evaluated on the same pre-cliff target rows:", ""]
    lines += metric_table(panels["smooth_refit"], ["old_A", "v28_A", "zero"])
    lines += ["", "Matching the training range matters as well as restricting test rows. "
              "The next bridge keeps Mode A throughout: old smooth-fit/smooth-test → old original-fit/smooth-test → old original-fit/full-test → v28 original-fit/full-test. "
              "The first two columns isolate inclusion of rapid-damage training observations and test observations. Add the residual full-grid form difference above to obtain the total controlled A gap.", "",
              "| Capability | Training-range effect, old | Scoring-range effect, old | Combined range effect, old | Residual form with BOTH smooth refits | Total controlled A gap |",
              "|---|---:|---:|---:|---:|---:|"]
    for cap, a in attr.items():
        keys = ("training_range_old_original_minus_smooth_refit", "range_old_full_minus_smooth",
                "range_training_plus_scoring_old", "residual_form_both_smooth_refit",
                "controlled_A_gap_v28_full_minus_old_smooth_refit")
        lines.append(f"| {cap} | " + " | ".join(mfmt(a[k]) for k in keys) + " |")
    a = attr["pooled"]
    lines += ["", "On this controlled Mode A bridge, combined training/scoring range contributes " +
              mfmt(a["range_training_plus_scoring_old"]) + " nats, whereas switching to v28 on matched full rows changes error by " +
              mfmt(a["residual_form_full_v28_minus_old_A"]) + ". "
              "The actual historical calibration-access effect is negligible within its interval. Range selection is therefore the main supported explanation for the headline discrepancy; "
              "it is not evidence that the new form universally predicts less accurately."]
    lines += ["", "With smooth training and smooth scoring, old density-only returns close to the historical error scale. "
              "Here a residual v28 disadvantage remains for math/code after matching range and calibration. "
              "This is a distinct prediction-recipe error: outcome censoring leaves unequal per-model support, and v28's profiled model coefficients differ from a pooled fit. "
              "The following ablation separates capability pooling, profiled shape/coefficient estimation, and metadata mapping on those exact rows; it does not attribute all error to metadata.", "",
              "| Capability | Capability-specific old minus pooled old | v28 population mean minus capability-specific old | v28 mapping minus population mean |",
              "|---|---:|---:|---:|"]
    for cap in (*CAPS, "pooled"):
        c = panels["smooth_refit"][cap]["contrasts"]
        lines.append(f"| {cap} | " + " | ".join(mfmt(c[k]) for k in
                     ("old_cap_A_minus_old_A", "v28_mean_A_minus_old_cap_A", "v28_A_minus_v28_mean_A")) + " |")
    lines += ["", "Both forms refitted on other models' full 0.9–0.3 grid, scored on 0.8–0.3 (includes saturation and nonmonotonic collapse):", ""]
    lines += metric_table(panels["deep_refit"], ["old_A", "v28_A", "zero"])
    lines += ["", "The response amplitudes reveal how much the historical ≤1-nat selection changes task difficulty. "
              "The matched residual difference isolates the prediction recipe, while the per-capability and population-mean ablations distinguish pooling from coefficient mapping. "
              "The two laws share a power-density family; a residual gap is not evidence that an entirely different density function was tested.", "",
              "## Qwen3-8B: amplitude, sign, shape, and damage onset", "",
              "Frozen formulas were bound directly to saved source-dense losses and, for Mode B only, its d=0.9 loss. No Qwen3-8B curve entered fitting or shrinkage selection. "
              + qwen["identity_limitation"] + " This limits checkpoint-level attribution; no identity difference is assumed.", ""]
    lines += metric_table(qwen["summaries"], ["old_A", "v28_A", "v28_B", "v28_B_shrink", "zero"])
    lines += ["", "| Capability | Actual calibration ΔL at 0.9 | Frozen A improvement over zero | Shrink B improvement over zero |",
              "|---|---:|---:|---:|"]
    for cap in CAPS:
        x = qwen["summaries"][cap]["metrics"]
        lines.append(f"| {cap} | {mfmt(qwen['amplitude'][cap]['calibration_response'])} | "
                     f"{mfmt(x['v28_A'], 'improvement_over_zero', 'improvement_ci95')} | "
                     f"{mfmt(x['v28_B_shrink'], 'improvement_over_zero', 'improvement_ci95')} |")
    lines += ["", "| Capability / density | Actual signed ΔL | Frozen A ΔL | Frozen B ΔL |", "|---|---:|---:|---:|"]
    for r in qwen["records"]:
        vals = [r["observed"], r["predictions"]["v28_A"], r["predictions"]["v28_B"]]
        lines.append(f"| {r['capability']} / {r['density']} | " + " | ".join(audit.with_ci(v, [v, v]) for v in vals) + " |")
    lines += ["", "For each capability, fit a single scalar s by least squares on the THREE scored points (an oracle diagnostic). "
              "The exact orthogonal decomposition is MSE(p−y) = MSE((1−s)p) + MSE(sp−y). "
              "The first term is amplitude **or sign** mismatch; negative s means a sign reversal, which positive amplitude correction cannot repair. "
              "MAE before/after is also reported, but only the squared errors have this orthogonal decomposition. A pooled row fits ONE scalar across all capabilities.", "",
              "| Capability | Signed scale | Raw MAE | Scaled MAE | Nonnegative-scale MAE | Amplitude/sign MSE fraction | Shape MSE fraction |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for cap, a in qwen["amplitude"].items():
        lines.append(f"| {cap} | " + " | ".join(mfmt(a[k]) for k in ("signed_scale", "raw_mae", "scaled_mae", "nonnegative_scaled_mae", "amplitude_or_sign_mse_fraction", "shape_mse_fraction")) + " |")
    lines += ["", "**Amplitude transfer dominates math/code; QA is primarily a sign error.** "
              "Their best positive scales are far below one and the residual curve MAEs are small. "
              "QA requires a negative scale: shrinking a positive curve alone cannot predict its observed loss reduction. "
              "The single pooled scale mostly suppresses all predictions toward zero; capability-specific scales explain much more. "
              "These are diagnostic oracle corrections, not validated predictors."]
    lines += ["", "Positive-damage onset uses the historical ΔL>1 nat threshold; it denotes departure from the perturbative region, not proven catastrophic collapse. "
              "Actual locations are grid brackets or censored below d=0.6. A power law has no explicit collapse parameter. Its continuous threshold crossing is reported with that limitation. "
              "Thresholds 0.5, 2, and 8 nats plus absolute-response crossings are retained in summary.json to expose threshold and QA-sign sensitivity.", "",
              "| Capability | Predicted continuous crossing | Predicted measured-grid crossing | Actual crossing | After signed amplitude correction |", "|---|---:|---|---|---|"]
    for cap in CAPS:
        c = qwen["crossings"][cap]["1.0"]["positive_damage"]
        def crossing_text(x):
            return str(x["density_bracket"]) if x["status"] == "crossed" else "not observed through 0.6; if present, below 0.6"
        lines.append(f"| {cap} | {mfmt(c['model_implied_continuous_crossing']) if c['model_implied_continuous_crossing'] else 'no crossing'} | "
                     f"{crossing_text(c['predicted'])} | {crossing_text(c['actual'])} | {crossing_text(c['predicted_after_signed_scale'])} |")
    lines += ["", "Premature threshold entry overlaps with amplitude error: changing amplitude alone can move or remove the predicted crossing. "
              "It must not be added as a third independent squared-error component. Math/code overestimation and QA sign mismatch are assessed separately from the density-shape residual; "
              "the measured source grid cannot identify its eventual collapse density.", "",
              "## One-point calibration instability", "",
              "Direct calibration uses a=y(0.9)/x(0.9). Shrinkage minimizes `(y0−a*x0)^2 + lambda*(a−mean_dev)^2`. "
              "Lambda is selected separately for each capability by inner LOMO on the eleven outer-training models, minimizing model-equal MAE at 0.8/0.7/0.6. "
              f"Fixed grid: {list(LAMBDAS)}. Gamma, population mean, and mapping are refitted in every inner fold. "
              "Every mode still receives exactly one target compressed point. Shrinkage targets the dev-population mean, not the target curve or an oracle amplitude.", ""]
    lines += metric_table(full, ["v28_A", "v28_B", "v28_B_shrink", "zero"])
    lines += ["", "| Capability | Direct−shrink MAE (positive favors shrink) | Fraction of model/capability curves improved | Mean absolute ΔL(0.9) | Mean absolute ΔL(test) |",
              "|---|---:|---:|---:|---:|"]
    for cap in (*CAPS, "pooled"):
        x = cal["statistics"][cap]
        lines.append(f"| {cap} | {mfmt(full[cap]['contrasts']['v28_B_minus_v28_B_shrink'])} | "
                     f"{mfmt(x['fraction_shrink_improves_model_capability'])} | {mfmt(x['calibration_absolute_response'])} | {mfmt(x['mean_absolute_test_response'])} |")
    lines += ["", "Nested shrinkage is more stable on development models: the paired MAE reduction excludes zero for math, QA, and pooled errors; code remains uncertain. "
              "It does not establish a gain over zero-change, and it does not improve on Mode A for Qwen3-8B. "
              "The near-zero QA weight on direct calibration shows how strongly the shallow observation must be discounted under this power shape.", "",
              "| Capability | Shrink B improvement over zero |", "|---|---:|"]
    for cap in (*CAPS, "pooled"):
        lines.append(f"| {cap} | {mfmt(full[cap]['metrics']['v28_B_shrink'], 'improvement_over_zero', 'improvement_ci95')} |")
    lines += ["", "Sensitivity is the exact multiplier from an additive error in the calibration RESPONSE to the predicted response at d=0.6. "
              "For example, multiplying the gain by 0.001 gives the prediction change caused by a 0.001-nat perturbation; this is a sensitivity experiment, not an estimated noise level.", "",
              "| Capability | Direct gain at 0.6 | Shrink gain at 0.6 | Weight on direct amplitude | Direct amplitude error | Shrink amplitude error |",
              "|---|---:|---:|---:|---:|---:|"]
    for cap, x in cal["statistics"].items():
        lines.append(f"| {cap} | " + " | ".join(mfmt(x[k]) for k in ("direct_gain_at_0.6", "shrink_gain_at_0.6", "weight_on_direct", "direct_amplitude_error", "shrink_amplitude_error")) + " |")
    lines += ["", "Amplitude errors here compare with the diagnostic best scalar fitted to target test points; these oracle labels never enter prediction. "
              "The exact per-model responses, amplitudes, selected lambda, and 0.3 extrapolation sensitivities are in `calibration.model_diagnostics`. "
              "As a separate one-point-location sensitivity, d=0.8 calibration is compared with d=0.9 on common test points 0.7/0.6:", ""]
    lines += metric_table(cal["common_test_0.7_0.6"], ["v28_B", "v28_B80", "v28_B_shrink", "zero"])
    lines += ["", cal["noise_identifiability"] + " Small response and large extrapolation gain establish poor conditioning. "
              "They do not establish that stochastic measurement error, rather than local-versus-deep shape mismatch, caused the observed failures. "
              "One-point calibration locks in shallow sign and shape deviations; shrinkage limits this amplification but may bias robust or negative-response sources toward a damaging population mean.", "",
              "## Distinguishable explanations", "",
              "| Candidate explanation | Isolating evidence / exclusions |", "|---|---|",
              "| Historical task was easier | Exact score reproduction, pre-cliff versus full fixed-fit scoring, and response amplitudes isolate range selection. Deep-grid extrapolation is a separate stress test. |",
              "| Historical predictor used target compression | Same historical test rows with and without the target's shallow contributions isolate the actual benefit. Historical family holdout already forbids that access. Matched A/B isolates the separate one-point protocol. |",
              "| Basic-input amplitude transfer is misspecified | Matched A comparison plus per-capability pooling and dev-mean ablations isolate estimation/mapping effects. Qwen oracle scaling tests whether a correct density shape could survive an amplitude correction. |",
              "| Qwen enters rapid damage later | Predicted versus censored actual threshold crossings test this in the observed range. Timing and amplitude overlap; no unmeasured collapse density is invented. |",
              "| Calibration measurement noise is amplified | Small saved shallow responses and exact sensitivity gains support poor conditioning; nested shrinkage tests stabilization. Aggregate JSON cannot distinguish measurement noise from deterministic shape error. |", "",
              "Artifacts: `results/v29-prune-diagnosis/summary.json` contains every fold, held-out row, paired contrast, amplitude, and CI; "
              "`predictions.csv` provides matched and Qwen predictions. This report is also copied to that directory. "
              "No existing predictor, source result, paper ledger, or frozen prediction file was modified.", ""]
    return "\n".join(lines)


def write_figure(summary):
    """Standalone scientific figure; only already-computed JSON statistics."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/v29-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), constrained_layout=True)
    qwen = summary["qwen3_8b"]
    names = ("old_A", "v28_A", "v28_B", "v28_B_shrink", "zero")
    colors = ("#7b6d8d", "#c14b30", "#d29922", "#237b70", "#666666")
    for j, cap in enumerate(CAPS):
        rs = sorted([r for r in qwen["records"] if r["capability"] == cap], key=lambda r: -r["density"])
        d = [r["density"] for r in rs]
        p = np.array([r["predictions"]["v28_A"] for r in rs])
        scale = qwen["amplitude"][cap]["signed_scale"]["value"]
        ax = axes[0, j]
        ax.plot(d, [r["observed"] for r in rs], "o-", color="#222222", label="Observed")
        ax.plot(d, p, "s--", color=colors[1], label="Frozen A")
        ax.plot(d, scale*p, "^:", color=colors[3], label="Oracle signed scale")
        ax.axhline(0, color="#aaaaaa", lw=.7)
        ax.axhline(1, color="#bbbbbb", lw=.7, ls=":")
        ax.set(xlim=(.815, .585), xticks=d, xlabel="Retained density", ylabel="Signed response (nats)",
               title=f"Qwen3-8B: {cap}")
        if j == 0:
            ax.legend(fontsize=8)
        ax = axes[1, j]
        metrics = summary["matched"]["v28_grid"][cap]["metrics"]
        for i, (name, color) in enumerate(zip(names, colors)):
            metric = metrics[name]
            value, (lo, hi) = metric["mae"], metric["mae_ci95"]
            ax.errorbar(i, value, yerr=[[value-lo], [hi-value]], fmt="o", color=color, capsize=4)
        ax.set(xticks=range(len(names)), xticklabels=["Old A", "v28 A", "Direct B", "Shrink B", "Zero"],
               ylabel="LOMO MAE (nats; log scale)", title=f"12 development models: {cap}")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=.2)
        ax.tick_params(axis="x", labelsize=9, rotation=20)
    fig.savefig(OUT / "diagnosis.png", dpi=180)
    fig.savefig(OUT / "diagnosis.pdf")
    plt.close(fig)


def run(n_boot=10000):
    rows, frozen, paths, inventory = load_rows()
    dev = [r for r in rows if r["model"] != "Qwen3-8B"]
    target = [r for r in rows if r["model"] == "Qwen3-8B"]
    standard = evaluate_lomo(dev, shrinkage=True)
    # Fail closed if the reimplementation stops matching the saved audit.
    for r in standard["records"]:
        if r["density"] in v28.PRUNE_TEST:
            saved, = [x for x in frozen["methods"]["pruning"]["by_capability"][r["capability"]]["lomo"]["records"]
                      if x["model"] == r["model"] and x["coordinate"] == r["density"]]
            for mode in ("A", "B"):
                if not np.isclose(r["predictions"]["v28_"+mode], saved[mode], rtol=0, atol=1e-9):
                    raise ValueError("LOMO reconstruction differs from frozen v28")
    smooth_refit = evaluate_lomo(dev, "smooth")
    deep_refit = evaluate_lomo(dev, "all")
    main_rows = restrict_range(standard["records"], .6)
    matched = {"v28_grid": summaries(main_rows, n_boot),
               "v28_grid_precliff": summaries(restrict_range(main_rows, .6, True), n_boot),
               "v28_grid_postcliff": summaries([r for r in main_rows if not r["precliff"]], n_boot),
               "extended_grid": summaries(standard["records"], n_boot),
               "deep_only": summaries([r for r in standard["records"] if r["density"] < .6], n_boot),
               "smooth_refit": summaries(restrict_range(smooth_refit["records"], .6, True), n_boot),
               "deep_refit": summaries(deep_refit["records"], n_boot)}
    source_files = [Path(__file__), Path(audit.__file__), Path(old.__file__), Path(v28.__file__),
                    ROOT / "analysis/v14_fitting.py", ROOT / "analysis/v17_unification.py", ROOT / "analysis/v25_distill_delta.py"]
    summary = {"version": 29, "n_boot": n_boot, "cpu_only": True, "input_sha256": audit.provenance(paths+source_files),
               "inventory": inventory, "rows": rows, "matched": matched,
               "lomo": {"original_training": standard, "smooth_training": smooth_refit, "deep_training": deep_refit},
               "attribution": range_attribution(standard["records"], n_boot, smooth_refit["records"]),
               "historical": historical_audit(dev, n_boot), "calibration": calibration_diagnostics(standard, n_boot),
               "qwen3_8b": qwen_diagnosis(target, dev, frozen, n_boot)}
    report = render_report(summary)
    audit.write_outputs(summary, report, OUT)
    write_figure(summary)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report)
    with (OUT / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "model", "capability", "density", "precliff", "observed", "predictor", "predicted", "target_calibration_density"])
        writer.writeheader()
        for experiment, rs in [("original_training", standard["records"]), ("smooth_training", smooth_refit["records"]),
                               ("deep_training", deep_refit["records"]), ("qwen3_8b", summary["qwen3_8b"]["records"])]:
            for r in rs:
                for name, value in r["predictions"].items():
                    writer.writerow({"experiment": experiment, **{k: r[k] for k in ("model", "capability", "density", "precliff", "observed")},
                                     "predictor": name, "predicted": value,
                                     "target_calibration_density": .8 if name == "v28_B80" else .9 if "_B" in name else ""})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()
    if args.n_boot < 1:
        parser.error("--n-boot must be positive")
    s = run(args.n_boot)
    print(f"Wrote {OUT} and {REPORT}")
    for cap in CAPS:
        x = s["matched"]["v28_grid"][cap]["metrics"]
        print(cap, {k: round(x[k]["mae"], 6) for k in ("old_A", "v28_A", "v28_B", "v28_B_shrink", "zero")})


if __name__ == "__main__":
    main()
