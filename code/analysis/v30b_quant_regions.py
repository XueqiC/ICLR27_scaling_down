#!/usr/bin/env python3
"""CPU-only regional audit of the saved V30/V10 quantization predictions.

Run: python analysis/v30b_quant_regions.py --bootstrap 10000 [--dry-run]
No quantization jobs, model imports, network access, or edits to V28/V30.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

try:
    from . import prediction_audit as audit
    from . import v30_quant_shape_candidates as v30
except ImportError:
    import prediction_audit as audit
    import v30_quant_shape_candidates as v30

import numpy as np

ROOT = audit.ROOT
OUT = ROOT / "results/v30b-quant-regions"
REPORT = ROOT / "paper/docs/QUANT_REGIONS.md"
SOURCE = ROOT / "results/v30-quant-candidates/summary.json"
CANDIDATES = v30.CANDIDATES
REGIONS = ("high_bits", "measurable", "collapse", "full_domain")
REGION_LABELS = {"high_bits": "High bits (b≥6)", "measurable": "4–5 bits",
                 "collapse": "Low bits (b≤3)", "full_domain": "Full domain"}
SEED = 300907


def in_region(bit, region):
    if region == "high_bits":
        return bit >= 6
    if region == "measurable":
        return bit in (4, 5)
    if region == "collapse":
        return bit <= 3
    if region == "full_domain":
        return True
    raise ValueError(f"Unknown region: {region}")


def shape(bit, candidate, b_ref=16, eta=None):
    """Full-bitwidth actual step, NOT the eta≈2.2 local surrogate."""
    if candidate != "actual_step":
        return v30.shape(bit, candidate, b_ref, eta)
    bit, ref = np.asarray(bit, dtype=float), np.asarray(b_ref, dtype=float)
    if np.any(bit < 2) or np.any(ref < 2):
        raise ValueError("Quantization bits must be at least two")
    q_max = np.exp2(bit - 1) - 1
    q_max_ref = np.exp2(ref - 1) - 1
    return q_max ** -2 - q_max_ref ** -2


def predict(fit, row):
    amplitude = v30.predict_mapping(fit["mappings"][row["capability"]], v30.basic_input(row))
    candidate, eta, ref = fit["candidate"], fit["eta"], row["b_ref"]
    return float(amplitude * shape(row["bit"], candidate, ref, eta) /
                 shape(4, candidate, ref, eta))


def verify_snapshot(saved, current):
    """Reject changed coverage, anchors, outcomes, or unreproducible predictions."""
    old = {r["row_id"]: r for r in saved["rows"]}
    new = {r["row_id"]: r for r in current}
    if len(old) != len(saved["rows"]) or len(new) != len(current) or old.keys() != new.keys():
        raise ValueError("V10/V30 cell coverage changed")
    folds = {f["held_out"]: f for f in saved["folds"]}
    for key, row in new.items():
        if any(old[key][k] != value for k, value in row.items()):
            raise ValueError(f"V10/V30 observation changed: {key}")
        fold = folds[row["model"]]
        if row["model"] in fold["train_models"] or key in fold["train_row_ids"]:
            raise ValueError("Held-out source leaked into V30 fold")
        if old[key]["calibration_points_used"] != int(row["b_ref"] == 8):
            raise ValueError("V30 target calibration policy changed")
        if old[key]["predictions"]["zero_change"] != 0:
            raise ValueError("Zero-change baseline changed")
        for candidate in CANDIDATES:
            expected = predict(fold["fits"][candidate], row)
            if not np.isclose(expected, old[key]["predictions"][candidate], rtol=1e-11, atol=1e-12):
                raise ValueError(f"V30 prediction changed: {key}/{candidate}")


def score_regions(records, n_boot):
    result = {}
    for region in REGIONS:
        selected = [r for r in records if in_region(r["bit"], region)]
        result[region] = {
            "status": "available" if selected else "no measured cells",
            "models": sorted({r["model"] for r in selected}),
            "bits": sorted({r["bit"] for r in selected}),
            "row_ids": [r["row_id"] for r in selected],
            "by_capability": {cap: v30.summarize_predictions(rs, n_boot) for cap in
                              (*audit.CAPABILITIES, "pooled")
                              if (rs := [r for r in selected if cap == "pooled" or r["capability"] == cap])}}
    return result


def gain_decomposition(records, n_boot):
    """Contribution to full-domain gain, with the full-domain denominator."""
    columns, names = [], []
    for region in REGIONS[:-1]:
        for candidate in CANDIDATES:
            names.append((region, candidate))
            columns.append([(abs(r["observed"]) - abs(r["observed"] - r["predictions"][candidate]))
                            * in_region(r["bit"], region) for r in records])
    values = np.array(columns).T
    draws = audit.bootstrap_means(values, [r["model"] for r in records], n_boot=n_boot)
    result = {region: {} for region in REGIONS[:-1]}
    for j, (region, candidate) in enumerate(names):
        total = np.mean([abs(r["observed"]) - abs(r["observed"] - r["predictions"][candidate])
                         for r in records])
        contribution = float(values[:, j].mean())
        result[region][candidate] = {"full_domain_gain_contribution": contribution,
                                    "ci95": audit.interval(draws[:, j]),
                                    "fraction_of_net_gain": contribution / total if total else None}
    return result


def lomo_fit_range(rows, train_bits=(4, 5)):
    """Refit all candidates on the SAME dev bits; evaluate ALL held-out cells."""
    records = [{**r, "calibration_points_used": int(r["b_ref"] == 8),
                "predictions": {"zero_change": 0.}} for r in rows]
    folds = []
    for split in audit.splits(rows, "model"):
        train = [rows[i] for i in split["train_indices"] if rows[i]["bit"] in train_bits]
        profile = v30.ShapeProfile(train)
        fits = {c: v30.fit_candidate(train, c, profile) for c in CANDIDATES}
        folds.append({"held_out": split["held_out"], "train_models": profile.models,
                      "train_row_ids": [r["row_id"] for r in train],
                      "test_row_ids": [rows[i]["row_id"] for i in split["test_indices"]], "fits": fits})
        for i in split["test_indices"]:
            records[i]["predictions"].update({c: predict(fit, rows[i]) for c, fit in fits.items()})
    return records, folds


def compare_protocols(first, second, n_boot):
    """Positive gain means second improves on first, on exactly common cells."""
    other = {r["row_id"]: r for r in second}
    if {r["row_id"] for r in first} != set(other):
        raise ValueError("Protocol comparison requires identical cells")
    result = {}
    for region in REGIONS:
        rs = [r for r in first if in_region(r["bit"], region)]
        if not rs:
            result[region] = None
            continue
        values = np.array([[abs(r["observed"] - r["predictions"][c]) -
                            abs(r["observed"] - other[r["row_id"]]["predictions"][c])
                            for c in CANDIDATES] for r in rs])
        if any(r["observed"] != other[r["row_id"]]["observed"] for r in rs):
            raise ValueError("Protocol observations differ")
        draws = audit.bootstrap_means(values, [r["model"] for r in rs], n_boot=n_boot)
        result[region] = {c: {"first_mae_minus_second_mae": float(values[:, j].mean()),
                              "ci95": audit.interval(draws[:, j])} for j, c in enumerate(CANDIDATES)}
    return result


def eta_range_comparison(rows, n_boot):
    """Paired whole-model refits; unlike MAE CIs, these include eta refitting."""
    profiles = [v30.ShapeProfile(rows), v30.ShapeProfile([r for r in rows if r["bit"] in (4, 5)])]
    if profiles[0].models != profiles[1].models:
        raise ValueError("Eta range comparison requires identical model sets")
    grids = [p.eta_grid() for p in profiles]
    fits = [p.fit_eta(grid_data=g) for p, g in zip(profiles, grids)]
    models = profiles[0].models
    draws = np.random.default_rng(SEED).integers(len(models), size=(n_boot, len(models)))
    estimates, cache, undefined = [], {}, 0
    for draw in draws:
        counts = tuple(np.bincount(draw, minlength=len(models)))
        if counts not in cache:
            cache[counts] = [p.fit_eta(counts, g) for p, g in zip(profiles, grids)]
        refits = cache[counts]
        if any(f is None for f in refits):
            undefined += 1
        else:
            estimates.append([f["eta"] for f in refits])
    if len(estimates) < 2:
        raise ValueError("Too few paired identified eta draws")
    estimates = np.array(estimates)
    return {"models": models, "n_boot": n_boot, "unidentified_draws": undefined,
            "all_bits": {**fits[0], "eta_ci95": audit.interval(estimates[:, 0])},
            "bits_4_5": {**fits[1], "eta_ci95": audit.interval(estimates[:, 1])},
            "all_minus_4_5": {"eta_difference": fits[0]["eta"] - fits[1]["eta"],
                               "ci95": audit.interval(estimates[:, 0] - estimates[:, 1])}}


def missing_five(inventory):
    missing = sorted(item["model"] for item in inventory if 4 in item["bits"] and 5 not in item["bits"])
    complete = sorted(item["model"] for item in inventory if {4, 5}.issubset(item["bits"]))
    return {"missing_models": missing, "complete_models": complete,
            "commands_executed": False,
            "commands": [f"python analysis/v10_quantization.py --model {model} --device cuda:0 "
                         "--model-dtype bf16 --n-probe 512 --bits 5 "
                         "--output-base results/v10-quant-shape512-fill5" for model in missing],
            "protocol": "bf16 follows job_hpg_v10shape.slurm; saved aggregate JSON lacks dtype/item IDs. "
                        "Confirm the original runtime/probes before collecting matched fills.",
            "merge_policy": "V10 always measures dense and overwrites its dense key on merge. Stage fills "
                            "separately; check dense/probe/dtype agreement, then add only the 5-bit record "
                            "to shape512 without replacing its dense/4-bit anchor. If mismatched, rerun "
                            "dense/4/5 together as a separate panel.",
            "benefit": "Completes the same seven-model 4/5 comparison and reduces comparison-set variation."}


def dev_error_scale(broad, shape512, n_boot):
    """Cross-panel repeatability proxy, never a fabricated within-panel SE."""
    indexed = {r["row_id"]: r for r in broad}
    pairs = []
    for row in shape512:
        if row["bit"] not in (4, 5) or row["row_id"] not in indexed:
            continue
        old = indexed[row["row_id"]]
        if row["b_ref"] != old["b_ref"]:
            raise ValueError("Cross-panel reference bit differs")
        pairs.append({"row_id": row["row_id"], "model": row["model"], "capability": row["capability"],
                      "bit": row["bit"], "broad_delta": old["observed"], "shape512_delta": row["observed"],
                      "difference": row["observed"] - old["observed"]})
    scales = {}
    for cap in audit.CAPABILITIES:
        scales[cap] = {}
        for bit in (4, 5):
            rs = [r for r in pairs if r["capability"] == cap and r["bit"] == bit]
            if not rs:
                raise ValueError(f"Missing dev discrepancy proxy: {cap}/{bit}")
            values = np.array([abs(r["difference"]) for r in rs])
            draws = audit.bootstrap_means(values, [r["model"] for r in rs], n_boot=n_boot)
            scales[cap][str(bit)] = {"n_models": len(rs), "mean_absolute_discrepancy": float(values.mean()),
                                     "ci95": audit.interval(draws[:, 0]),
                                     "max_absolute_discrepancy": float(values.max())}
    return {"measurement_se": None, "status": "proxy only; item/seed losses unavailable",
            "definition": "abs(DeltaL_shape512 - DeltaL_broad), matched model/capability/bit; each panel "
                          "uses its own dense anchor. Includes probe-composition/protocol effects; not a "
                          "measurement standard error or formal power calculation.",
            "by_capability": scales, "rows": pairs}


def affine_prediction(fit, capability, bit):
    """Exact frozen Qwen3-14B prediction A + B * unknown reference L_c."""
    row = {"model": "qwen3-14b", "family": "qwen3", "nominal_size_b": 14.,
           "reference_loss": 0., "capability": capability, "bit": bit, "b_ref": 16}
    intercept = predict(fit, row)
    slope = predict(fit, {**row, "reference_loss": 1.}) - intercept
    return {"intercept": intercept, "reference_loss_coefficient": slope}


def absolute_affine_range(intercept, slope, lower, upper):
    endpoints = [intercept + slope * lower, intercept + slope * upper]
    minimum = 0. if min(endpoints) <= 0 <= max(endpoints) else min(map(abs, endpoints))
    return [minimum, max(map(abs, endpoints))]


def screen_predictions(predictions, threshold):
    gaps = {f"{a}__{b}": abs(predictions[a] - predictions[b])
            for a, b in itertools.combinations(CANDIDATES, 2)}
    near_zero = max(map(abs, predictions.values())) <= threshold
    verdict = ("INCONCLUSIVE_NEAR_ZERO" if near_zero else
               "ALL_PAIRS_ABOVE_PROXY" if min(gaps.values()) > threshold else
               "SOME_PAIRS_ABOVE_PROXY" if max(gaps.values()) > threshold else "INCONCLUSIVE_SMALL_GAPS")
    return {"pairwise_gaps": gaps, "max_gap": max(gaps.values()), "min_gap": min(gaps.values()),
            "pairwise_screening_verdicts": {pair: "INCONCLUSIVE_NEAR_ZERO" if near_zero else
                                           "ABOVE_PROXY" if gap > threshold else "INCONCLUSIVE_SMALL_GAP"
                                           for pair, gap in gaps.items()},
            "all_predictions_near_zero": near_zero, "screening_verdict": verdict}


def discrimination_precheck(broad, error_scale, target_reference_losses=None):
    if target_reference_losses is not None:
        if not isinstance(target_reference_losses, dict) or set(target_reference_losses) != set(audit.CAPABILITIES):
            raise ValueError("Target reference JSON must contain only math, code, qa dense losses")
        target_reference_losses = {c: audit.finite(v) for c, v in target_reference_losses.items()}
        if min(target_reference_losses.values()) < 0:
            raise ValueError("Dense CE loss cannot be negative")
    protocols = {}
    for name, train in (("all_bits", broad), ("bits_4_5", [r for r in broad if r["bit"] in (4, 5)])):
        profile = v30.ShapeProfile(train)
        fits = {c: v30.fit_candidate(train, c, profile) for c in CANDIDATES}
        cells = []
        for cap in audit.CAPABILITIES:
            anchors = {r["model"]: r["reference_loss"] for r in broad
                       if r["family"] == "qwen3" and r["capability"] == cap}
            lower, upper = min(anchors.values()), max(anchors.values())
            reference = (target_reference_losses[cap] if target_reference_losses is not None
                         else float(np.median(list(anchors.values()))))
            for bit in (4, 5):
                equations = {c: affine_prediction(fit, cap, bit) for c, fit in fits.items()}
                predictions = {c: f["intercept"] + f["reference_loss_coefficient"] * reference
                               for c, f in equations.items()}
                scale = error_scale["by_capability"][cap][str(bit)]
                # Upper CI of mean absolute dev discrepancy is a conservative heuristic,
                # not a hypothesis-test threshold or guaranteed error bound.
                threshold = scale["ci95"][1]
                ranges = {}
                for a, b in itertools.combinations(CANDIDATES, 2):
                    ranges[f"{a}__{b}"] = absolute_affine_range(
                        equations[a]["intercept"] - equations[b]["intercept"],
                        equations[a]["reference_loss_coefficient"] - equations[b]["reference_loss_coefficient"],
                        lower, upper)
                cells.append({"capability": cap, "bit": bit, "b_ref": 16,
                              "reference_loss_scenario": reference, "dev_qwen_reference_losses": anchors,
                              "dev_reference_range": [lower, upper], "prediction_affine_in_reference_loss": equations,
                              "predicted_delta_loss": predictions, "dev_discrepancy_scale": scale,
                              "screening_threshold": threshold, "pairwise_gap_ranges_over_dev_reference_range": ranges,
                              **screen_predictions(predictions, threshold)})
        protocols[name] = {"train_models": profile.models, "train_row_ids": [r["row_id"] for r in train],
                           "frozen_fits": fits, "cells": cells}
    return {"target_model": "qwen3-14b", "target_quantized_outcomes_used": 0,
            "target_compressed_calibration_points": 0,
            "target_reference_losses": target_reference_losses,
            "reference_status": "user-supplied matching dense losses" if target_reference_losses is not None
                                else "MISSING: numeric table is a dev-Qwen median-reference scenario, not measured Qwen3-14B",
            "amplitude_policy": "Freeze eta AND dev-fitted amplitude mapping; no target quantized calibration. "
                                "Amplitude is an affine function of the target dense loss; no measured target a_c is fitted.",
            "threshold_policy": "Gap > upper 95% model-bootstrap CI of mean absolute cross-panel dev discrepancy. "
                                "Heuristic screening only, not significance/power.",
            "verdict": "INCONCLUSIVE_FOR_ACTUAL_TARGET" if target_reference_losses is None
                       else "CONDITIONAL_SCREEN_ONLY_MEASUREMENT_SE_UNAVAILABLE",
            "run_decision": ("Obtain matching dense-only losses and a measurement-precision estimate" if
                             target_reference_losses is None else "Obtain a measurement-precision estimate") +
                            " before committing to a shape-discrimination run. The all-bit predictor gaps are not uniformly near zero "
                            "in dev-reference scenarios; local 4/5 shape discrimination can still be weak. "
                            "Do not treat all-bit amplitude-map disagreement as clean shape confirmation.",
            "protocols": protocols}


def build_summary(n_boot=10000, target_reference_losses=None):
    if n_boot < 2:
        raise ValueError("At least two bootstrap draws are required")
    snapshot_hash = audit.provenance([SOURCE])
    saved = audit.read_json(SOURCE)
    paths = [ROOT / p for p in saved["input_sha256"]]
    if audit.provenance(paths) != saved["input_sha256"]:
        raise ValueError("V10 inputs changed since V30; refusing to relabel its predictions")
    hashes = {**saved["input_sha256"], **snapshot_hash, **audit.provenance([
        Path(__file__), Path(v30.__file__), Path(audit.__file__), ROOT / "analysis/v10_quantization.py",
        ROOT / "scripts/job_hpg_v10shape.slurm"])}
    panels, raw = {}, {}
    for name, directory in (("broad", "v10-quantization"), ("shape512", "v10-quant-shape512")):
        rows, _, inventory, excluded = v30.load_panel(ROOT / "results" / directory)
        verify_snapshot(saved["panels"][name], rows)
        raw[name] = rows
        records = saved["panels"][name]["rows"]
        print(f"{name}: rescoring {len(records)} unchanged V30 cells by region", flush=True)
        panels[name] = {"inventory": inventory, "excluded_models": excluded, "rows": records,
                        "regions": score_regions(records, n_boot),
                        "gain_decomposition": gain_decomposition(records, n_boot)}
    print("Refitting 4/5-only LOMO and paired eta range bootstrap", flush=True)
    local, folds = lomo_fit_range(raw["broad"])
    range_sensitivity = {"eta": eta_range_comparison(raw["broad"], n_boot),
                         "rows": local, "folds": folds, "regions": score_regions(local, n_boot),
                         "all_bit_mae_minus_local_fit_mae": compare_protocols(panels["broad"]["rows"], local, n_boot)}
    fills = missing_five(panels["shape512"]["inventory"])
    complete = [r for r in raw["shape512"] if r["model"] in fills["complete_models"]]
    retained = [r for r in panels["shape512"]["rows"] if r["model"] in fills["complete_models"]]
    refitted, complete_folds = v30.lomo(complete)
    sensitivity = {"with_missing_models": panels["shape512"]["regions"],
                   "without_missing_models_fixed_v30_predictions": score_regions(retained, n_boot),
                   "without_missing_models_refitted_lomo": score_regions(refitted, n_boot),
                   "same_complete_cells_refit_gain": compare_protocols(retained, refitted, n_boot),
                   "refitted_rows": refitted, "refitted_folds": complete_folds,
                   "five_bit": {name: v30.summarize_predictions([r for r in rs if r["bit"] == 5], n_boot)
                                for name, rs in (("all_seven_training_models", panels["shape512"]["rows"]),
                                                 ("four_complete_training_models", refitted))}}
    error_scale = dev_error_scale(raw["broad"], raw["shape512"], n_boot)
    precheck = discrimination_precheck(raw["broad"], error_scale, target_reference_losses)
    olmo = [r for r in raw["broad"] if r["model"] == "olmo3-32b" and r["bit"] == 3]
    return {"version": "30b", "endpoint": "capability LOSS: Delta L_c(b) = L_c(b) - L_c(b_ref), native-token CE nats",
            "n_boot": n_boot, "mae_bootstrap_seed": 240526, "eta_bootstrap_seed": SEED,
            "input_sha256": hashes, "source_summary": str(SOURCE.relative_to(ROOT)),
            "region_policy": "Bit-defined regions only: high b>=6, measurable b=4/5, collapse b<=3. "
                             "Descriptive regime names are not outcome filters; full domain retains every V30 dev cell.",
            "bootstrap_policy": "Paired whole-model bootstrap; all capabilities/bits/candidates stay together, "
                                "equal observed-cell weights. MAE fits held fixed; eta comparison refits each draw. "
                                "95% percentile intervals are exploratory, not multiplicity-adjusted or item/seed CIs.",
            "freeze_clarification": {
                "v30_lomo_and_v30b": "Eta and amplitude mapping are frozen before scoring each held-out model. "
                                     "Amplitude uses only target size, family, dense reference L_c: zero compressed "
                                     "calibration points for current dense anchors. Basic-input prediction conditional "
                                     "on measured dense loss; development LOMO is not prospective new-source validation.",
                "v30_paired_4_to_5": "CONDITIONAL SHAPE TRANSFER: observed target Delta L(4) calibrates amplitude "
                                    "(one compressed point). Only shape is frozen; not full basic-parameter prediction.",
                "v30_descriptive_eta": "Per-development-model/capability amplitudes profiled on development curves; "
                                      "this is an in-sample shape estimate, not held-out amplitude prediction."},
            "actual_step_formula": "q_max(b)=2^(b-1)-1; g(b)=q_max(b)^(-2)-q_max(b_ref)^(-2); "
                                   "normalize by g(4) only for conditioning. Full bit curve, never eta=2.2.",
            "olmo3_32b_int3_exception": {r["capability"]: r["observed"] for r in olmo},
            "panels": panels, "fit_range_sensitivity": range_sensitivity,
            "fill_missing_5bit": fills, "missing_model_sensitivity": sensitivity,
            "dev_measurement_error_proxy": error_scale, "qwen3_14b_precheck": precheck}


def metric_table(regions, pooled_only=False):
    lines = ["| Region / capability (models; cells) | Candidate | MAE [95% CI] | Gain over zero-change [95% CI] |",
             "|---|---|---:|---:|"]
    for region, result in regions.items():
        if result["status"] != "available":
            lines.append(f"| {REGION_LABELS[region]} | No measured cells | N/A | N/A |")
        for cap, scores in result["by_capability"].items():
            if pooled_only and cap != "pooled":
                continue
            for candidate in ("zero_change", *CANDIDATES):
                m = scores["metrics"][candidate]
                lines.append(f"| {REGION_LABELS[region]} / {cap} ({scores['n_models']}; {scores['n_cells']}) | "
                             f"{v30.LABELS[candidate]} | {v30.ci(m['mae'], m['mae_ci95'])} | "
                             f"{v30.ci(m['improvement_over_zero_change'], m['improvement_ci95'])} |")
    return lines


def render(summary):
    broad = summary["panels"]["broad"]
    full = broad["regions"]["full_domain"]["by_capability"]["pooled"]["metrics"]
    share = broad["gain_decomposition"]["collapse"]["shared_eta"]["fraction_of_net_gain"]
    mid = broad["regions"]["measurable"]["by_capability"]["pooled"]["metrics"]["shared_eta"]
    lines = ["# Quantization regions (V30b)", "",
             f"The unchanged V30 full-domain shared-η MAE is **{full['shared_eta']['mae']:.5f}**, versus "
             f"zero-change **{full['zero_change']['mae']:.5f}** nats. The low-bit region supplies "
             f"**{share:.1%} of its net full-domain gain**. Region-specific results below qualify the all-bit win.", "",
             f"In the 4/5-bit region, shared η improves over zero-change by only "
             f"{v30.ci(mid['improvement_over_zero_change'], mid['improvement_ci95'])} nats; "
             "the gain is unresolved when its interval includes zero. The strongest simple baseline here is "
             "zero-change, which is included on the identical cells in every comparison.", "",
             summary["endpoint"] + ". Negative changes, QA improvements, near-zero values, and collapse cells are retained. "
             "No observed-damage ratios, log-damage fits, or outcome-based censoring are used.", "",
             "## Common comparison and freeze boundary", "",
             summary["region_policy"], "",
             "The primary tables reuse and verify every saved V30 cell prediction against its frozen fold coefficients "
             "and the original V10 JSON. Candidates use the same models, probes, reference anchors, ridge mapping "
             "(λ=1), and target information. Broad and shape512 remain separate; dense/16 is b_ref=16. "
             "An 8-bit fallback would cost one compressed reference point; no current input uses it. "
             "The terms near-zero, measurable non-collapse, and collapse describe typical behavior, not all cells. "
             "In particular 5-bit signals can be tiny and QA can improve.", "",
             summary["bootstrap_policy"], ""]
    for key, text in summary["freeze_clarification"].items():
        lines += [f"- **{key}**: {text}"]
    lines += ["", "Here a_c means the raw multiplier of g(b). V30 fits signed amplitudes normalized to ΔL(4), "
              "then predicts them from log nominal model size, reference capability loss, and family indicators. "
              "The held-out a_c is predicted by that frozen mapping; it is not fitted to the target quantization curve. "
              "An unknown target dense loss still prevents a unique numerical basic-input prediction.", "",
              "## Full actual-step function", "",
              "The candidates are fixed g(b)=4^(-b)−4^(-b_ref), actual-step below, and shared "
              "g(b)=2^(-η(b−4))−2^(-η(b_ref−4)). One η is shared across development models and capabilities.", "",
              "`q_max(b) = 2^(b-1) - 1`\n\n`g(b) = q_max(b)^(-2) - q_max(b_ref)^(-2)`", "",
              "V30b implements this **full-bitwidth function directly**, including reference subtraction. "
              "η≈2.20 is only its local 4→5 equivalent slope and is never substituted for the actual-step candidate. "
              "Dividing each g by g(4) changes the amplitude units, not predictions.", "",
              "## Regional MAE: unchanged V30 predictions", ""]
    for panel, result in summary["panels"].items():
        lines += [f"### {panel}", "", *metric_table(result["regions"]), ""]
    exception = summary["olmo3_32b_int3_exception"]
    lines += ["OLMo3-32B is the int3 exception and stays in the low-bit bucket and full-domain score: " +
              ", ".join(f"ΔL_{c}={v:.5f}" for c, v in exception.items()) + " nats. It does not show the "
              "catastrophic damage typical of the other int3 curves.", "",
              "### Where the all-bit gain comes from", "",
              "Contributions use all 180 broad cells as the denominator and sum to each candidate's full-domain "
              "gain. They do not confuse a region's per-cell MAE with its weight in the headline result.", "",
              "| Region | Candidate | Contribution to full-domain gain [95% CI] | Share of net gain |",
              "|---|---|---:|---:|"]
    for region, cs in broad["gain_decomposition"].items():
        for candidate, m in cs.items():
            lines.append(f"| {REGION_LABELS[region]} | {v30.LABELS[candidate]} | "
                         f"{v30.ci(m['full_domain_gain_contribution'], m['ci95'])} | {m['fraction_of_net_gain']:.1%} |")
    lines += ["", "The baseline comparison and candidate ranking answer different questions. Almost all shared-η "
              "gain over zero-change comes from int3, where all candidates predict large damage. Shared η actually "
              "has slightly worse point MAE than both fixed candidates in that bucket. Its all-bit advantage over "
              "those candidates comes from avoiding their overprediction in the high-bit and 4/5-bit regions. "
              "Paired candidate comparisons below quantify that distinction; negative differences favor the first candidate.", "",
              "| Region | First / second candidate | First MAE − second MAE [95% CI] |",
              "|---|---|---:|"]
    for region, scored in broad["regions"].items():
        for pair in scored["by_capability"]["pooled"]["pairwise_comparisons"]:
            lines.append(f"| {REGION_LABELS[region]} | {v30.LABELS[pair['first']]} / {v30.LABELS[pair['second']]} | "
                         f"{v30.ci(pair['first_minus_second_mae'], pair['ci95'])} |")
    sensitivity = summary["fit_range_sensitivity"]
    eta = sensitivity["eta"]
    lines += ["", "## Validity range: development fits on 4/5 versus all bits", "",
              f"On the same 12 broad models, η(all bits)={v30.ci(eta['all_bits']['eta'], eta['all_bits']['eta_ci95'])}; "
              f"η(4/5 only)={v30.ci(eta['bits_4_5']['eta'], eta['bits_4_5']['eta_ci95'])}. "
              f"The paired-refit difference is {v30.ci(eta['all_minus_4_5']['eta_difference'], eta['all_minus_4_5']['ci95'])}.", "",
              "The following sensitivity refits **every candidate's amplitude labels and mapping**, plus shared η, "
              "using only training-model 4/5-bit cells. Every held-out bit is still scored, with zero compressed target "
              "calibration. Thus comparisons within each table have identical training information. Differences "
              "between training ranges are not an η-only intervention, and extrapolation to int3 is explicitly retained.", "",
              *metric_table(sensitivity["regions"], pooled_only=True), "",
              "| Evaluation region | Candidate | All-bit-fit MAE − 4/5-fit MAE [paired 95% CI] |",
              "|---|---|---:|"]
    for region, cs in sensitivity["all_bit_mae_minus_local_fit_mae"].items():
        for candidate, m in cs.items():
            lines.append(f"| {REGION_LABELS[region]} | {v30.LABELS[candidate]} | "
                         f"{v30.ci(m['first_mae_minus_second_mae'], m['ci95'])} |")
    lines += ["", "The range-dependent η and these cross-range errors are a validity-range problem for a single "
              "exponent. A good all-bit score does not establish a universal non-collapse loss law.", "",
              "## Fill the missing 5-bit cells (commands only)", ""]
    fills = summary["fill_missing_5bit"]
    lines += [f"Missing 5-bit records: **{', '.join(fills['missing_models'])}** (3 of 7). "
              f"Complete 4/5 models: {', '.join(fills['complete_models'])}. " + fills["benefit"], "",
              "These exact V10 commands are listed, **not executed**. `--n-probe 512` builds 512 probes per "
              "capability and V10 evaluates the odd-indexed half (256); it does not evaluate 512 held-out items. "
              + fills["protocol"], "", "```bash", *fills["commands"], "```", "", fills["merge_policy"], "",
              "### Sensitivity with and without the three incomplete models", "",
              "The seven-model full-domain result retains all 33 observed cells. Removing the incomplete models "
              "leaves four models and 24 cells. First we hold V30 predictions fixed to isolate scoring-set changes; "
              "then we rerun LOMO on those four models to expose training-set changes. Cross-set MAE changes are "
              "descriptive; paired refit comparisons use the same 24 cells. Missing 5-bit outcomes are never imputed.", "",
              "| Scoring / training set | Candidate | Full-domain MAE [95% CI] | Gain over zero [95% CI] |",
              "|---|---|---:|---:|"]
    missing = summary["missing_model_sensitivity"]
    for key in ("with_missing_models", "without_missing_models_fixed_v30_predictions", "without_missing_models_refitted_lomo"):
        for candidate, m in missing[key]["full_domain"]["by_capability"]["pooled"]["metrics"].items():
            lines.append(f"| {key} | {v30.LABELS[candidate]} | {v30.ci(m['mae'], m['mae_ci95'])} | "
                         f"{v30.ci(m['improvement_over_zero_change'], m['improvement_ci95'])} |")
    lines += ["", "| Same 24 cells | Candidate | Original-fit MAE − complete-model-refit MAE [95% CI] |",
              "|---|---|---:|"]
    for candidate, m in missing["same_complete_cells_refit_gain"]["full_domain"].items():
        lines.append(f"| Complete models | {v30.LABELS[candidate]} | {v30.ci(m['first_mae_minus_second_mae'], m['ci95'])} |")
    lines += ["", "| 5-bit scoring set: same four models / 12 cells | Candidate | MAE [95% CI] | Gain over zero [95% CI] |",
              "|---|---|---:|---:|"]
    for key, scores in missing["five_bit"].items():
        for candidate, m in scores["metrics"].items():
            lines.append(f"| {key} | {v30.LABELS[candidate]} | {v30.ci(m['mae'], m['mae_ci95'])} | "
                         f"{v30.ci(m['improvement_over_zero_change'], m['improvement_ci95'])} |")
    pre = summary["qwen3_14b_precheck"]
    lines += ["", "## Qwen3-14B discrimination pre-check", "", f"**Verdict: {pre['verdict']}.** " + pre["run_decision"], "",
              pre["reference_status"] + ". No Qwen3-14B quantized losses or V28 results were read. "
              "Both dev-only candidate coefficient sets below are saved in the V30b JSON; neither was calibrated "
              "on Qwen3-14B compressed outcomes.", "",
              "The exact frozen predictions are supplied in JSON as `intercept + reference_loss_coefficient * L_c(dense)` "
              "for each candidate, capability and bit. The numeric default table substitutes the median dense loss "
              "of development Qwen3-0.6/1.7/4B, and also records pair-gap ranges over those anchors. These scenarios "
              "are not Qwen3-14B observations, estimates of its dense loss, or guaranteed bounds on its reference loss. "
              "No prior ability/accuracy data are converted to capability CE.", "",
              summary["dev_measurement_error_proxy"]["definition"], "", pre["threshold_policy"], "",
              "Broad Qwen3-0.6B also records a dense protocol gap of −0.0003/−0.0008/+0.0133 nats "
              "(math/code/QA). This isolated discrepancy is not a variance estimate. High-bit damage and LOMO "
              "prediction errors likewise are not treated as measurement noise.", ""]
    for protocol, data in pre["protocols"].items():
        lines += [f"### Frozen development fit: {protocol}", "",
                  "| Capability / bit | Reference scenario | Pred ΔL: 4^-b | Pred ΔL: step² | Pred ΔL: η | Gap 4^-b/step² | Gap 4^-b/η | Gap step²/η | Dev discrepancy [95% CI] | Screen |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for cell in data["cells"]:
            preds = " | ".join(f"{cell['predicted_delta_loss'][c]:.5f}" for c in CANDIDATES)
            gaps = " | ".join(f"{v:.5f}" for v in cell["pairwise_gaps"].values())
            scale = cell["dev_discrepancy_scale"]
            lines.append(f"| {cell['capability']} / {cell['bit']} | {cell['reference_loss_scenario']:.5f} | "
                         f"{preds} | {gaps} | {v30.ci(scale['mean_absolute_discrepancy'], scale['ci95'])} | "
                         f"{cell['screening_verdict']} |")
    lines += ["", "The all-bit candidates do not all predict near-zero damage in these scenarios: the fixed laws "
              "carry substantial all-bit-fitted amplitudes into b4/b5. That makes them potentially separable as "
              "complete predictors, but is entangled with collapse-region amplitude fitting. The local 4/5 fit is "
              "a more relevant shape screen: its b4 amplitudes nearly coincide; small code/QA b5 gaps can make "
              "those tests inconclusive. A dev-proxy screen cannot settle the actual target without its basic inputs "
              "and measurement precision. The JSON keeps each pair's verdict and range, not only the maximum gap.", "",
              "## Reproduction and artifact boundary", "",
              f"Run `python analysis/v30b_quant_regions.py --bootstrap {summary['n_boot']}`; add `--dry-run` for no writes. "
              "If matching target dense CE is available, pass `--target-reference-losses '{\"math\": ..., \"code\": ..., \"qa\": ...}'` "
              "with actual numeric values. This option accepts only three dense losses, never quantized outcomes. "
              "Tests: `python -m pytest -q tests/test_v30b.py tests/test_v30.py tests/test_prediction_audits.py`.", "",
              "Only `results/v30b-quant-regions/summary.json`, its `report.md`, and `paper/docs/QUANT_REGIONS.md` are written. "
              "V28 is untouched and V30's summary is read-only. Source hashes are verified before writing; changed "
              "V10 data fail closed rather than silently changing the V30 comparison. All fits use NumPy/SciPy on CPU. "
              "No fill or Qwen3-14B quantization command is run.", ""]
    return "\n".join(lines)


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target-reference-losses", type=json.loads,
                        help='matching Qwen3-14B dense losses as JSON: {"math":1.0,"code":1.0,"qa":5.0}')
    args = parser.parse_args()
    summary = build_summary(args.bootstrap, args.target_reference_losses)
    report = render(summary)
    if args.dry_run:
        print("DRY RUN: no output writes.\n" + report)
        return
    audit.write_outputs(summary, report, OUT)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report)
    print(f"Wrote {OUT} and {REPORT}")


if __name__ == "__main__":
    main()
