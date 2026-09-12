#!/usr/bin/env python3
"""CPU-only candidate-shape reanalysis of saved V10 capability-loss JSON.

Run: python analysis/v30_quant_shape_candidates.py [--dry-run] [--bootstrap 10000]
No model loading, GPU libraries, network calls, or writes to earlier analyses.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

try:
    from . import prediction_audit as audit
except ImportError:
    import prediction_audit as audit

# prediction_audit sets BLAS thread counts before importing NumPy.
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = audit.ROOT
OUT = ROOT / "results/v30-quant-candidates"
REPORT = ROOT / "paper/docs/QUANT_SHAPE_CANDIDATES.md"
BITS = (3, 4, 5, 6, 8)
CANDIDATES = ("fixed_4_power", "actual_step", "shared_eta")
LABELS = {"zero_change": "Zero-change", "fixed_4_power": "4^-b",
          "actual_step": "Actual step²", "shared_eta": "Shared η"}
ETA_BOUNDS = (0.05, 8.0)
SEED = 300906
# A fixed roster prevents a newly arrived prospective run from entering dev.
DEV_MODELS = frozenset(("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "gemma3-270m",
                        "gemma3-1b", "gemma3-4b", "gemma3-12b", "gemma3-27b",
                        "gemma4-31b", "muse-30b", "olmo3-7b", "olmo3-32b"))
RESERVED_MODELS = frozenset(("qwen3-8b", "qwen3-14b"))


def canonical(model):
    return model.lower().replace("--", "/").split("/")[-1]


def shape(bit, candidate, b_ref=16, eta=None):
    """Unscaled g(b), including the finite reference subtraction."""
    bit = np.asarray(bit, dtype=float)
    ref = np.asarray(b_ref, dtype=float)
    if np.any(bit < 2) or np.any(ref < 2):
        raise ValueError("Quantization bits must be at least two")
    if candidate == "fixed_4_power":
        return np.power(4., -bit) - np.power(4., -ref)
    if candidate == "actual_step":
        return np.square(1 / (np.exp2(bit - 1) - 1)) - np.square(1 / (np.exp2(ref - 1) - 1))
    if candidate != "shared_eta" or eta is None or not np.isfinite(eta) or eta <= 0:
        raise ValueError("Shared shape requires a positive finite eta")
    # expm1 keeps the difference accurate for small eta and b near b_ref.
    return np.exp2(-eta * (bit - 4)) * (-np.expm1(-eta * (ref - bit) * np.log(2)))


def normalized_shape(bit, candidate, b_ref=16, eta=None):
    """g/g(4): numerically comparable amplitudes, exactly the same predictions."""
    return shape(bit, candidate, b_ref, eta) / shape(4, candidate, b_ref, eta)


def reference(losses):
    if "16" in losses:
        return "16", 16
    if "dense" in losses:
        return "dense", 16
    if "8" in losses:
        return "8", 8
    raise ValueError("Need a dense/16-bit reference or an 8-bit fallback")


def model_features(model):
    """Nominal size from the recorded identifier, NOT a measured matrix count."""
    match = re.search(r"-(\d+(?:\.\d+)?)([bm])$", model)
    if match is None:
        raise ValueError(f"Cannot read nominal model size: {model}")
    size = float(match[1]) * (0.001 if match[2] == "m" else 1.)
    return {"nominal_size_b": size, "family": model.split("-")[0]}


def load_panel(directory):
    rows, paths, inventory, excluded = [], [], [], []
    seen = set()
    for path in sorted(Path(directory).glob("*/quant_losses.json")):
        model = canonical(path.parent.name)
        # Do not even read reserved models' loss values.
        if model not in DEV_MODELS:
            excluded.append({"model": model, "reason": "reserved prospective source" if
                             model in RESERVED_MODELS else "outside fixed development roster"})
            continue
        if model in seen:
            raise ValueError(f"Duplicate model aliases in panel: {model}")
        seen.add(model)
        losses = audit.read_json(path)
        ref_key, b_ref = reference(losses)
        bits = [b for b in BITS if str(b) in losses and b != b_ref]
        if not bits:
            raise ValueError(f"No target bits: {path}")
        paths.append(path)
        source = str(path.relative_to(ROOT))
        inventory.append({"model": model, "bits": bits, "b_ref": b_ref,
                          "reference_key": ref_key, "source_path": source,
                          "measurement_caveat": losses.get("_5bit_meta")})
        for cap in audit.CAPABILITIES:
            anchor = audit.finite(losses[ref_key][cap])
            for bit in bits:
                loss = audit.finite(losses[str(bit)][cap])
                rows.append({"row_id": f"{model}|{cap}|{bit}", "model": model,
                             "capability": cap, "bit": bit, "b_ref": b_ref,
                             "reference_key": ref_key, "reference_loss": anchor,
                             "loss": loss, "observed": loss - anchor,
                             "source_path": source, **model_features(model)})
    if len(seen) < 3:
        raise ValueError("Need at least three development models per panel")
    return rows, paths, inventory, excluded


class ShapeProfile:
    """Profile signed nuisance amplitudes per dev model/capability, one eta total.

    Original loss-space SSE gives every observed cell equal weight. Missing bits
    are masked, never imputed. Single-bit curves can label amplitudes, but provide
    no information about eta. Whole-model multiplicities implement refit bootstrap.
    """

    def __init__(self, rows):
        self.models = sorted({r["model"] for r in rows})
        self.keys = sorted({(r["model"], r["capability"]) for r in rows})
        self.bits = np.array(sorted({r["bit"] for r in rows}))
        self.y = np.zeros((len(self.keys), len(self.bits)))
        self.mask = np.zeros_like(self.y, dtype=bool)
        self.refs = np.zeros(len(self.keys))
        self.examples = []
        for i, key in enumerate(self.keys):
            curve = [r for r in rows if (r["model"], r["capability"]) == key]
            self.examples.append(curve[0])
            if len({r["b_ref"] for r in curve}) != 1:
                raise ValueError("Reference changes within a curve")
            self.refs[i] = curve[0]["b_ref"]
            for r in curve:
                j = int(np.flatnonzero(self.bits == r["bit"])[0])
                if self.mask[i, j]:
                    raise ValueError("Duplicate model/capability/bit")
                self.mask[i, j], self.y[i, j] = True, r["observed"]
        self.model_index = np.array([self.models.index(key[0]) for key in self.keys])
        self.informative = (self.mask.sum(axis=1) >= 2) & np.any(self.y != 0, axis=1)

    def evaluate(self, candidate, eta=None):
        x = normalized_shape(self.bits[None, :], candidate, self.refs[:, None], eta)
        x = np.where(self.mask, x, 0.)
        amplitudes = np.sum(x * self.y, axis=1) / np.sum(x * x, axis=1)
        residual = np.where(self.mask, self.y - amplitudes[:, None] * x, 0.)
        curve_sse = np.sum(residual * residual, axis=1)
        curve_sse[~self.informative] = 0.
        by_model = np.bincount(self.model_index, weights=curve_sse, minlength=len(self.models))
        return amplitudes, by_model

    def eta_grid(self):
        grid = np.linspace(*ETA_BOUNDS, 160)
        sse = np.array([self.evaluate("shared_eta", eta)[1] for eta in grid])
        return grid, sse

    def fit_eta(self, weights=None, grid_data=None):
        if not self.informative.any():
            raise ValueError("Eta requires a nonzero response measured at at least two bits")
        weights = np.ones(len(self.models)) if weights is None else np.asarray(weights)
        info_by_model = np.bincount(self.model_index, weights=self.informative,
                                    minlength=len(self.models))
        if not np.any((weights > 0) & (info_by_model > 0)):
            return None  # Explicitly record unidentified bootstrap draws.
        grid, values = self.eta_grid() if grid_data is None else grid_data
        objective = lambda eta: float(self.evaluate("shared_eta", eta)[1] @ weights)
        scores = values @ weights
        # Refine every coarse local minimum, retaining endpoints as candidates.
        local = [i for i in range(1, len(grid)-1)
                 if scores[i] <= scores[i-1] and scores[i] <= scores[i+1]]
        local = sorted(set(local + [int(np.argmin(scores))]))
        choices = [float(grid[0]), float(grid[-1])]
        for i in local:
            opt = minimize_scalar(objective, bounds=(grid[max(0, i-1)], grid[min(len(grid)-1, i+1)]),
                                  method="bounded", options={"xatol": 1e-8})
            choices.append(float(opt.x))
        eta = min(choices, key=objective)
        return {"eta": eta, "sse": objective(eta),
                "at_boundary": min(abs(eta-b) for b in ETA_BOUNDS) < 1e-5}


def eta_analysis(rows, n_boot):
    profile = ShapeProfile(rows)
    grid_data = profile.eta_grid()
    fit = profile.fit_eta(grid_data=grid_data)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(profile.models), size=(n_boot, len(profile.models)))
    estimates, cache, undefined, boundary = [], {}, 0, 0
    for draw in draws:
        counts = tuple(np.bincount(draw, minlength=len(profile.models)))
        if counts not in cache:
            cache[counts] = profile.fit_eta(counts, grid_data)
        result = cache[counts]
        if result is None:
            undefined += 1
            continue
        estimates.append(result["eta"])
        boundary += result["at_boundary"]
    if len(estimates) < 2:
        raise ValueError("Too few identified bootstrap eta draws")
    ci = audit.interval(estimates)
    fixed = {"4^-b (η=2)": 2., "actual-step local approximation (η=2.20)": 2.20}
    return {**fit, "eta_ci95": ci, "bounds": list(ETA_BOUNDS), "n_boot": n_boot,
            "identified_bootstrap_draws": len(estimates), "unidentified_bootstrap_draws": undefined,
            "boundary_bootstrap_draws": int(boundary),
            "n_models": len(profile.models), "n_cells": len(rows),
            "development_points_per_curve": sorted(set(profile.mask.sum(axis=1).tolist())),
            "n_informative_models": len({k[0] for k, info in zip(profile.keys, profile.informative) if info}),
            "observed_response_amplitude": float(np.mean([abs(r["observed"]) for r in rows])),
            "improvement_over_zero_change": None,
            "calibration_points_used": "development curves only; descriptive fit, not a target prediction",
            "excluded_eta_values": [name for name, eta in fixed.items() if eta < ci[0] or eta > ci[1]],
            "implied_ratio_5_over_4_at_ref16": float(normalized_shape(5, "shared_eta", eta=fit["eta"])),
            "implied_ratio_ci95_at_ref16": audit.interval([
                float(normalized_shape(5, "shared_eta", eta=eta)) for eta in estimates]),
            "profile_grid": [{"eta": float(e), "sse": float(s.sum())}
                             for e, s in zip(*grid_data)]}


def basic_input(row):
    """Allowlist excludes all held-out quantized losses and fitted labels."""
    return {key: row[key] for key in ("model", "family", "nominal_size_b", "reference_loss")}


def fit_mapping(rows, labels):
    """Signed ridge, fixed lambda=1, unpenalized intercept, dev-only transforms."""
    if len(rows) < 2 or len({r["model"] for r in rows}) != len(rows):
        raise ValueError("Mapping requires at least two distinct development models")
    x = np.array([[np.log(r["nominal_size_b"]), r["reference_loss"]] for r in rows])
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-12)
    families = sorted({r["family"] for r in rows})
    effects = np.array([[float(r["family"] == f) for f in families] for r in rows])
    frequencies = effects.mean(axis=0)
    design = np.column_stack([np.ones(len(rows)), (x-center)/scale, effects-frequencies])
    penalty = np.eye(design.shape[1]); penalty[0, 0] = 0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ np.asarray(labels))
    return {"train_models": [r["model"] for r in rows], "center": center.tolist(),
            "scale": scale.tolist(), "families": families, "family_frequencies": frequencies.tolist(),
            "coefficients": coefficients.tolist(), "ridge_lambda": 1.,
            "features": ["intercept", "standardized log(nominal size in B)",
                         "standardized reference L_c", *families]}


def predict_mapping(fit, target):
    if target["model"] in fit["train_models"]:
        raise ValueError("Held-out model leaked into amplitude mapping")
    x = np.array([np.log(target["nominal_size_b"]), target["reference_loss"]])
    effects = np.zeros(len(fit["families"]))
    if target["family"] in fit["families"]:
        effects = np.array([float(target["family"] == f) for f in fit["families"]])
        effects -= fit["family_frequencies"]
    design = np.r_[1., (x-fit["center"])/fit["scale"], effects]
    return float(design @ fit["coefficients"])


def fit_candidate(train, candidate, profile=None):
    profile = ShapeProfile(train) if profile is None else profile
    eta_fit = profile.fit_eta() if candidate == "shared_eta" else None
    eta = eta_fit["eta"] if eta_fit else None
    amplitudes, sse = profile.evaluate(candidate, eta)
    mappings = {}
    for cap in audit.CAPABILITIES:
        indices = [i for i, key in enumerate(profile.keys) if key[1] == cap]
        mappings[cap] = fit_mapping([basic_input(profile.examples[i]) for i in indices], amplitudes[indices])
    return {"candidate": candidate, "eta": eta, "eta_fit": eta_fit, "profile_sse": float(sse.sum()),
            "amplitude_definition": "predicted Delta L(4); raw a = this amplitude / g(4)",
            "development_amplitudes": {f"{m}|{c}": float(a) for (m, c), a in zip(profile.keys, amplitudes)},
            "mappings": mappings}


def predict_candidate(fit, target, capability, bit, b_ref):
    amplitude = predict_mapping(fit["mappings"][capability], target)
    return float(amplitude * normalized_shape(bit, fit["candidate"], b_ref, fit["eta"]))


def lomo(rows):
    records = [{**r, "calibration_points_used": int(r["b_ref"] == 8),
                "predictions": {"zero_change": 0.}} for r in rows]
    folds = []
    for split in audit.splits(rows, "model"):
        train = [rows[i] for i in split["train_indices"]]
        profile = ShapeProfile(train)
        fits = {c: fit_candidate(train, c, profile) for c in CANDIDATES}
        folds.append({"held_out": split["held_out"], "train_models": profile.models,
                      "train_row_ids": [r["row_id"] for r in train],
                      "test_row_ids": [rows[i]["row_id"] for i in split["test_indices"]], "fits": fits})
        for i in split["test_indices"]:
            r = rows[i]
            for candidate, fit in fits.items():
                records[i]["predictions"][candidate] = predict_candidate(
                    fit, basic_input(r), r["capability"], r["bit"], r["b_ref"])
    return records, folds


def summarize_predictions(rows, n_boot):
    predictions = {c: [r["predictions"][c] for r in rows] for c in ("zero_change", *CANDIDATES)}
    metrics, _, draws = audit.compare_predictions(rows, predictions, "zero_change", n_boot=n_boot)
    names = list(predictions)
    comparisons = []
    for i, first in enumerate(CANDIDATES):
        for second in CANDIDATES[i+1:]:
            difference = draws[:, names.index(first)] - draws[:, names.index(second)]
            comparisons.append({"first": first, "second": second,
                                "first_minus_second_mae": metrics[first]["mae"] - metrics[second]["mae"],
                                "ci95": audit.interval(difference)})
    for name, metric in metrics.items():
        metric["improvement_over_zero_change"] = -metric["mae_minus_reference"]
        metric["improvement_ci95"] = [-metric["difference_ci95"][1], -metric["difference_ci95"][0]]
        worse = []
        if name != "zero_change":
            for other in CANDIDATES:
                if other != name and audit.interval(draws[:, names.index(other)] - draws[:, names.index(name)])[0] > 0:
                    worse.append(other)
        metric["excluded_candidate_predictors"] = worse
    winner = min(CANDIDATES, key=lambda c: metrics[c]["mae"])
    best_each_draw = draws[:, 1:].min(axis=1)
    return {"n_models": len({r["model"] for r in rows}), "n_cells": len(rows),
            "observed_response_amplitude": metrics["zero_change"]["mae"],
            "observed_response_amplitude_ci95": metrics["zero_change"]["mae_ci95"],
            "observed_signed_mean": float(np.mean([r["observed"] for r in rows])),
            "calibration_points_used": sorted({r["calibration_points_used"] for r in rows}),
            "metrics": metrics, "winner": winner, "pairwise_comparisons": comparisons,
            "best_candidate_gain_ci95_reselected": audit.interval(draws[:, 0] - best_each_draw)}


def paired_analysis(rows, n_boot):
    indexed = {(r["model"], r["capability"], r["bit"]): r for r in rows}
    paired = []
    for r in rows:
        if r["bit"] != 5 or (r["model"], r["capability"], 4) not in indexed:
            continue
        four = indexed[r["model"], r["capability"], 4]
        if (four["b_ref"], four["reference_loss"]) != (r["b_ref"], r["reference_loss"]):
            raise ValueError("Unpaired reference in 4/5-bit test")
        exact = float(normalized_shape(5, "actual_step", r["b_ref"]))
        ks = {"ratio_0.250": .25, "ratio_0.218": .218, "actual_step_exact": exact}
        paired.append({**r, "delta4": four["observed"], "ratios": ks,
                       "paired_differences": {name: r["observed"] - k*four["observed"] for name, k in ks.items()},
                       "predictions": {"zero_change": 0., **{name: k*four["observed"] for name, k in ks.items()}}})
    if not paired:
        raise ValueError("Need paired 4- and 5-bit observations")
    results = {}
    for cap in (*audit.CAPABILITIES, "pooled"):
        rs = [r for r in paired if cap == "pooled" or r["capability"] == cap]
        names = list(rs[0]["ratios"])
        values = [[r["paired_differences"][name] for name in names] for r in rs]
        draws = audit.bootstrap_means(values, [r["model"] for r in rs], n_boot=n_boot, seed=SEED)
        preds = {name: [r["predictions"][name] for r in rs] for name in ("zero_change", *names)}
        metrics, _, _ = audit.compare_predictions(rs, preds, "zero_change", n_boot=n_boot)
        tests = {}
        for i, name in enumerate(names):
            ci = audit.interval(draws[:, i])
            metric = metrics[name]
            tests[name] = {"k_values": sorted({r["ratios"][name] for r in rs}),
                           "mean_paired_difference": float(np.mean(values, axis=0)[i]), "ci95": ci,
                           "reject_fixed_ratio": ci[0] > 0 or ci[1] < 0,
                           "calibrated_prediction_mae": metric["mae"],
                           "improvement_over_zero_change": -metric["mae_minus_reference"],
                           "improvement_ci95": [-metric["difference_ci95"][1], -metric["difference_ci95"][0]],
                           "calibration_points_used": 1 + int(any(r["b_ref"] == 8 for r in rs)),
                           "excluded_shape": name if ci[0] > 0 or ci[1] < 0 else None}
        results[cap] = {"n_models": len({r["model"] for r in rs}), "n_cells": len(rs),
                        "observed_response_amplitude": float(np.mean([abs(r["observed"]) for r in rs])),
                        "mean_abs_delta4": float(np.mean([abs(r["delta4"]) for r in rs])),
                        "mean_delta4": float(np.mean([r["delta4"] for r in rs])),
                        "mean_delta5": float(np.mean([r["observed"] for r in rs])), "tests": tests}
    return {"rows": paired, "by_capability": results}


def build_summary(n_boot=10000):
    if n_boot < 2:
        raise ValueError("At least two bootstrap draws are required")
    panels = {}
    for name, directory in (("broad", "v10-quantization"), ("shape512", "v10-quant-shape512")):
        rows, source_paths, inventory, excluded = load_panel(ROOT / "results" / directory)
        # Snapshot before any fit; write_outputs verifies these hashes again.
        hashes = audit.provenance(source_paths)
        print(f"{name}: {len(inventory)} dev models, {len(rows)} cells; fitting shared eta and LOMO", flush=True)
        records, folds = lomo(rows)
        scoring = {}
        for scope, selected in (("all_bits", records), ("5bit", [r for r in records if r["bit"] == 5])):
            scoring[scope] = {cap: summarize_predictions(
                [r for r in selected if cap == "pooled" or r["capability"] == cap], n_boot)
                for cap in (*audit.CAPABILITIES, "pooled")}
        panels[name] = {"inventory": inventory, "excluded_models": excluded, "input_sha256": hashes,
                        "paired": paired_analysis(rows, n_boot), "eta_all_available_bits": eta_analysis(rows, n_boot),
                        "lomo": scoring, "folds": folds, "rows": records}
        if name == "broad":
            # Explicit local sensitivity, never used to tune the all-bit LOMO fits.
            panels[name]["eta_4_5_only"] = eta_analysis([r for r in rows if r["bit"] in (4, 5)], n_boot)
    return {"version": 30, "endpoint": "signed capability CE loss delta, native-token nats",
            "n_boot": n_boot, "eta_and_paired_seed": SEED, "mae_seed": 240526,
            "input_sha256": {p: h for panel in panels.values() for p, h in panel["input_sha256"].items()},
            "reference_policy": "explicit 16 first; dense treated as b_ref=16; otherwise b_ref=8",
            "development_roster": sorted(DEV_MODELS), "reserved_models": sorted(RESERVED_MODELS),
            "calibration_policy": "LOMO zero target compressed points with dense/16 reference; an 8-bit fallback costs one",
            "panels": panels}


def ci(value, limits):
    return f"{value:.5f} [{limits[0]:.5f}, {limits[1]:.5f}]"


def render(summary):
    broad = summary["panels"]["broad"]
    lead = broad["lomo"]["all_bits"]["pooled"]
    winner = lead["winner"]
    gain = lead["metrics"][winner]
    local = summary["panels"]["shape512"]
    e512 = local["eta_all_available_bits"]
    b5 = broad["lomo"]["5bit"]["pooled"]["metrics"]["shared_eta"]
    s5 = local["lomo"]["5bit"]["pooled"]["metrics"]["shared_eta"]
    rejected, unresolved = [], []
    for cap, result in local["paired"]["by_capability"].items():
        target = rejected if all(result["tests"][k]["reject_fixed_ratio"]
                                 for k in ("ratio_0.250", "ratio_0.218")) else unresolved
        target.append(cap)
    lines = ["# Quantization bit-response candidates (V30)", "",
             f"The broad-panel pooled all-bit LOMO winner by point MAE is **{LABELS[winner]}**: "
             f"MAE {gain['mae']:.5f} versus zero-change {lead['observed_response_amplitude']:.5f} nats; "
             f"improvement {ci(gain['improvement_over_zero_change'], gain['improvement_ci95'])}. "
             "This is exploratory development-panel model holdout, not a new-source prospective validation.", "",
             f"At 5 bits, shared η improves pooled LOMO MAE over zero-change by "
             f"{ci(b5['improvement_over_zero_change'], b5['improvement_ci95'])} nats in the broad panel "
             f"and {ci(s5['improvement_over_zero_change'], s5['improvement_ci95'])} in the 512-probe panel. "
             f"The 512-probe shared exponent is {ci(e512['eta'], e512['eta_ci95'])}. Its paired tests "
             f"reject both fixed ratios for {', '.join(rejected) or 'no capability'}; "
             f"joint rejection remains unresolved for {', '.join(unresolved) or 'none'}. "
             "The 512-probe pairwise MAE comparison does not resolve shared η versus actual-step, "
             "despite the point ranking and rejection of the fixed mean ratio restriction.", "",
             "The endpoint is capability **LOSS**, ΔL_c(b)=L_c(b)−L_c(b_ref), in native-token CE nats. "
             "Negative changes are retained, including QA improvements. No ratios with observed near-zero "
             "denominators, logarithms of observed damage, cliff filters, or sign censoring enter any fit.", "",
             "## Inputs, reference, and scope", "",
             "The two saved probe panels are analyzed separately. Bits and dense anchors are never spliced "
             "between panels. Reference priority is explicit `16`, then `dense` as the uncompressed **b_ref=16** "
             "anchor, then `8` if neither is present. Every current input uses `dense`/16; an 8-bit fallback "
             "would count as one compressed reference measurement. Dense is an operational 16-bit anchor, "
             "not a claim that a separate 16-bit fake-quantization run was measured.", "",
             "| Panel | Development models | Bits by model | Reference |",
             "|---|---|---|---|"]
    for name, panel in summary["panels"].items():
        for item in panel["inventory"]:
            lines.append(f"| {name} | {item['model']} | {', '.join(map(str, item['bits']))} | "
                         f"{item['reference_key']} → {item['b_ref']} |")
    lines += ["", "`shape512` is the supplied 512-probe high-precision panel. Only four of its seven models "
              "currently have a 5-bit record; the other three contribute their 4-bit cells to all-bit LOMO, "
              "but cannot inform an exponent or the paired 4/5 test. The available aggregate JSON cannot "
              "independently establish item identities, numeric evaluation dtype, or item/seed uncertainty. "
              "The broader panel retains Qwen3-0.6B's recorded 5-bit protocol discrepancy "
              "(dense-gap QA +0.0133 nats); no correction is invented.", "",
              "## Shapes and estimation", "",
              "For ΔL̂_c(b)=a_c(x0)g(b), the candidates are:", "",
              "1. **4^-b:** g(b)=4^-b−4^-b_ref.",
              "2. **Actual step²:** g(b)=(2^(b−1)−1)^−2−(2^(b_ref−1)−1)^−2.",
              "3. **Shared η:** g(b)=2^(−η(b−4))−2^(−η(b_ref−4)); one η across models, capabilities, "
              "and families within the quantization panel/fold.", "",
              "The verified quantizer uses symmetric per-output-channel round-to-nearest, fixed clip "
              "at max_abs, and scale=max_abs/(2^(b−1)−1). Thus its derived 5/4 step² ratio is "
              f"(7/15)²={((7/15)**2):.9f}, not 0.25. With the finite 16-bit subtraction, the ratios are "
              f"{float(normalized_shape(5, 'actual_step')):.9f} and "
              f"{float(normalized_shape(5, 'fixed_4_power')):.9f}. η=2 gives exactly the 4^-b shape "
              "up to an amplitude scale; η≈2.20 matches the actual-step **local 4→5 slope**, not its "
              "entire bit curve. Error-power scaling alone does not prove an identical capability-loss law.", "",
              "For each shape, development curves yield signed least-squares amplitudes; g is normalized "
              "by g(4) for conditioning, so the label equals fitted ΔL(4). A capability-specific ridge "
              "mapping (fixed λ=1, unpenalized intercept) predicts that label from log nominal model size "
              "in billions parsed from the model identifier, measured reference L_c, and centered family "
              "indicators. Nominal size is not a measured non-embedding parameter count. Features are "
              "standardized on training models only; unseen families receive zero family correction. "
              "The same mapping and penalty apply to all candidates and are not selected using holdout errors.", "",
              "Shared η minimizes original loss-space SSE after profiling the development model/capability "
              "amplitudes, over [0.05, 8], using a grid and bounded refinement. This is one fitted η, "
              "never a separate exponent per capability/model/family. Single-bit curves have zero profile "
              "information. Every LOMO fold excludes all capabilities and bits of the held-out model "
              "before refitting η, amplitude labels, preprocessing, and ridge coefficients. Its target "
              "inputs are only basic metadata and the reference loss: **0 compressed calibration points** "
              "for these dense-reference inputs. The full-development η estimates below are descriptive; "
              "they are never substituted into LOMO folds.", "",
              "## Paired 4/5-bit differences", "",
              "D_k=ΔL(5)−kΔL(4). Each 95% interval resamples whole models, preserving the 4/5 pair, "
              "reference, candidates, and all capabilities together. A CI excluding zero rejects the "
              "fixed ratio's population mean paired-difference restriction on this panel. A CI containing "
              "zero does not establish individual-model fit. Requested k=0.250 and 0.218 are shown, "
              "with the exact finite-reference step² ratio as a rounding sensitivity.", "",
              "The improvement column additionally scores kΔL(4) as a prediction of ΔL(5). It spends "
              "**one own-model 4-bit calibration point**; it is distinct from the zero-calibration LOMO "
              "comparison. Observed response amplitude means mean |ΔL|, not the unstable median ratio.", "",
              "| Panel / capability (models) | k | Mean D_k [95% CI] | Observed amplitude: mean abs(ΔL5) / abs(ΔL4) | "
              "Improvement over zero-change [95% CI] | Calibration points used | Candidate shape excluded |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for name, panel in summary["panels"].items():
        for cap, result in panel["paired"]["by_capability"].items():
            for test_name, test in result["tests"].items():
                label = "exact step²" if test_name == "actual_step_exact" else f"{test['k_values'][0]:.3f}"
                excluded = ("4^-b ratio" if test_name == "ratio_0.250" else "step² ratio") if test["reject_fixed_ratio"] else "none resolved"
                lines.append(f"| {name} / {cap} ({result['n_models']}) | {label} | "
                             f"{ci(test['mean_paired_difference'], test['ci95'])} | "
                             f"{result['observed_response_amplitude']:.5f} / {result['mean_abs_delta4']:.5f} | "
                             f"{ci(test['improvement_over_zero_change'], test['improvement_ci95'])} | "
                             f"{test['calibration_points_used']} | {excluded} |")
    lines += ["", "## Shared effective exponent", "",
              "η uncertainty uses whole-model bootstrap **refits**, including all capabilities and available "
              "bits per sampled model. Duplicate draws weight the complete model profile SSE. Bounds are "
              "fixed before fitting; unidentified draws (only one-bit models sampled) are counted and omitted "
              "from the conditional percentile interval. Loss-space SSE weights large loss cliffs strongly.", "",
              "| Fit | η [95% CI] | Implied 5/4 ratio [95% CI] | Observed response amplitude | "
              "Improvement over zero-change | Calibration points used | Candidate shape excluded |",
              "|---|---:|---:|---:|---|---|---|"]
    exponent_results = [(f"{name}, all available bits", panel["eta_all_available_bits"])
                        for name, panel in summary["panels"].items()]
    exponent_results.append(("broad, 4/5 only (local sensitivity)", broad["eta_4_5_only"]))
    for name, result in exponent_results:
        excluded = "; ".join(result["excluded_eta_values"]) or "none resolved"
        point_counts = ",".join(map(str, result["development_points_per_curve"]))
        lines.append(f"| {name} | {ci(result['eta'], result['eta_ci95'])} | "
                     f"{ci(result['implied_ratio_5_over_4_at_ref16'], result['implied_ratio_ci95_at_ref16'])} | "
                     f"{result['observed_response_amplitude']:.5f} | N/A: descriptive fit | "
                     f"dev {point_counts} bits/curve; target N/A | {excluded} |")
    for name, result in exponent_results:
        lo, hi = result["eta_ci95"]
        interpretation = ("the interval lies above 2.20, supporting faster effective decay than both local fixed-law slopes"
                          if lo > 2.2 else "the interval lies below 2.20" if hi < 2.2 else
                          "the interval includes 2.20, so faster decay than actual-step is unresolved")
        lines += ["", f"- {name}: {interpretation}; {result['n_informative_models']} informative models, "
                  f"{result['unidentified_bootstrap_draws']}/{result['n_boot']} unidentified bootstrap draws and "
                  f"{result['boundary_bootstrap_draws']} boundary draws (point fit at boundary: {result['at_boundary']})."]
    lines += ["", "An exponent above 2.2 means damage magnitude decays faster with increasing bits than "
              "either fixed law's local 4→5 prediction. A positive ratio of about 0.13 corresponds to "
              "η=−log2(0.13)≈2.94; that is an illustrative conversion, not a fabricated panel estimate. "
              "The all-bit fit includes the 3-bit cliffs, so its effective exponent need not equal the "
              "4/5-only value. The latter is a separately labeled sensitivity and does not replace the "
              "specified all-bit candidate in LOMO. Excluding η=2.20 in a constant-exponent model is "
              "not an exact global rejection of the non-exponential actual-step curve; use the direct "
              "paired test and held-out actual-step comparison as well.", "",
              "## Held-out model prediction", "",
              "Every candidate and zero-change score exactly the same held-out cells. All-bit MAE includes "
              "3/4/5/6/8 where available; the 5-bit view isolates the small-response endpoint using those "
              "same all-bit-trained folds. Each cell has equal weight. The 512 panel has unequal bit "
              "coverage, so its all-bit estimate is a cell-weighted available-data estimate.", "",
              f"Intervals use {summary['n_boot']:,} paired model-bootstrap draws. MAE/gain intervals hold "
              "OOF predictions fixed, per prediction_audit.py; they describe panel variation, not retraining, "
              "seed, or shared-item uncertainty. Pooled resampling keeps all capabilities of each model "
              "together. Gains are MAE(zero-change)−MAE(candidate), positive favoring the candidate. "
              "Zero-change is the user-specified strongest simple baseline; this analysis does not reselect "
              "among additional simple baselines.", "",
              "The last column identifies **candidate predictors** with significantly worse paired MAE "
              "than the row predictor (95% CI for other−row above zero). It does not exclude a mathematical "
              "shape independent of this amplitude mapping. All intervals are exploratory, marginal, "
              "and unadjusted for multiple comparisons. A point winner alone is not a resolved win."]
    for scope in ("all_bits", "5bit"):
        lines += ["", f"### {scope}", "",
                  "| Panel / capability (models; cells) | Candidate | LOMO MAE [95% CI] | "
                  "Observed response amplitude [95% CI] | Improvement over zero-change [95% CI] | "
                  "Calibration points used | Candidate predictors excluded |",
                  "|---|---|---:|---:|---:|---:|---|"]
        for name, panel in summary["panels"].items():
            for cap, result in panel["lomo"][scope].items():
                for candidate in CANDIDATES:
                    m = result["metrics"][candidate]
                    excluded = ", ".join(LABELS[c] for c in m["excluded_candidate_predictors"])
                    lines.append(f"| {name} / {cap} ({result['n_models']}; {result['n_cells']}) | "
                                 f"{LABELS[candidate]} | {ci(m['mae'], m['mae_ci95'])} | "
                                 f"{ci(result['observed_response_amplitude'], result['observed_response_amplitude_ci95'])} | "
                                 f"{ci(m['improvement_over_zero_change'], m['improvement_ci95'])} | "
                                 f"{','.join(map(str, result['calibration_points_used']))} | {excluded or 'none resolved'} |")
        lines += ["", "Point winners and gains over zero-change (a negative gain means the baseline wins):", ""]
        for name, panel in summary["panels"].items():
            for cap, result in panel["lomo"][scope].items():
                candidate = result["winner"]
                m = result["metrics"][candidate]
                resolved = ("baseline advantage resolved" if m["improvement_ci95"][1] < 0 else
                            "gain resolved" if m["improvement_ci95"][0] > 0 else "gain unresolved")
                lines.append(f"- {name}/{cap}: {LABELS[candidate]}, gain "
                             f"{ci(m['improvement_over_zero_change'], m['improvement_ci95'])}; {resolved}.")
    lines += ["", "Observed response amplitude equals zero-change MAE exactly. The JSON also records signed "
              "means, every pairwise candidate-MAE interval, and best-candidate gain intervals with the "
              "winner reselected within each bootstrap draw; named-winner intervals in the report are "
              "conditional on that candidate and are not selection-adjusted. Cross-model native-token "
              "nats and nominal size/family features limit cross-tokenizer and mechanistic interpretation. "
              "The four paired 512 models are a small available subset, not a random missingness guarantee.", "",
              "## Prospective boundary and reproduction", "",
              "**The frozen Qwen3-8B prospective result in `results/v28-new-source-pred` stays as-is.** "
              "This script never reads its predictions, never reads Qwen3-8B quantization loss JSON, and "
              "never writes to that directory. It does not revise the V28 method or retrospective result. "
              "Qwen3-8B is not presented as independent validation of these newly explored V30 candidates. "
              "**Qwen3-14B is reserved for the next frozen new-source test**; no Qwen3-14B numbers are "
              "loaded, fitted, or fabricated. Every model used here is development data for these candidates "
              "and cannot later validate them as an independent new source. Separate-panel fits and LOMO "
              "do not restore prospective independence to already-seen models.", "",
              "Run `python analysis/v30_quant_shape_candidates.py --bootstrap 10000`; "
              "add `--dry-run` to compute without output writes. Tests: `python -m pytest -q tests/test_v30.py`. "
              "Only V30 outputs are written: `results/v30-quant-candidates/summary.json`, `predictions.csv`, "
              "`report.md`, and this report. Summary JSON contains input SHA-256 hashes, cell predictions, "
              "fold training membership/coefficients/η, model coverage, all comparisons, and η objective "
              "profiles. Source hashes are checked again immediately before output writes.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    summary = build_summary(args.bootstrap)
    report = render(summary)
    if args.dry_run:
        print("DRY RUN: CPU-only; no output writes.\n" + report)
        return
    audit.write_outputs(summary, report, OUT)
    with (OUT / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["panel", "model", "capability", "bit", "b_ref",
                                                   "observed", "calibration_points_used", "zero_change", *CANDIDATES])
        writer.writeheader()
        for panel, result in summary["panels"].items():
            for row in result["rows"]:
                writer.writerow({"panel": panel, **{k: row[k] for k in
                                 ("model", "capability", "bit", "b_ref", "observed", "calibration_points_used")},
                                 **row["predictions"]})
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report)
    print(f"Wrote {OUT} and {REPORT}")


if __name__ == "__main__":
    main()
