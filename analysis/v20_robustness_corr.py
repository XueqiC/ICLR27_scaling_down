#!/usr/bin/env python3
"""V20: pruning/quantization robustness correlation and low-bit size effects.

The script is CPU-only and reads the existing V6 pruning and V10 quantization
loss ladders.  It writes a standardized cross-mechanism robustness audit plus
the E1 int4/int3 family-by-size analysis.

Usage::

    python analysis/v20_robustness_corr.py --dry-run
    python analysis/v20_robustness_corr.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
from scipy.stats import rankdata

try:
    from . import v17_unification as v17
except ImportError:  # direct execution
    import v17_unification as v17


ROOT = Path(__file__).resolve().parents[1]
PRUNE_BASE = ROOT / "results/v6-capability-geometry"
QUANT_BASE = ROOT / "results/v10-quantization"
OUTPUT_DIR = ROOT / "results/v20-robustness-corr"
CAPABILITIES = ("math", "code", "qa")
BOOTSTRAP_RESAMPLES = 1_000
CLIFF_THRESHOLD = 1.0
LOW_BIT_MODELS = {
    "gemma3-270m": ("gemma3", 0.27),
    "gemma3-1b": ("gemma3", 1.0),
    "gemma3-4b": ("gemma3", 4.0),
    "gemma3-12b": ("gemma3", 12.0),
    "gemma3-27b": ("gemma3", 27.0),
    "olmo3-7b": ("olmo3", 7.0),
    "olmo3-32b": ("olmo3", 32.0),
    "Qwen3-0.6B": ("qwen3", 0.6),
    "Qwen3-1.7B": ("qwen3", 1.7),
    "Qwen3-4B": ("qwen3", 4.0),
}


def _stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")


def load_method_rows(
    prune_base: Path = PRUNE_BASE, quant_base: Path = QUANT_BASE
) -> tuple[list[dict], list[dict], dict[str, object]]:
    pruning, prune_notes = v17._load_pruning_rows(prune_base)
    quantization, quant_notes = v17._load_quantization_rows(quant_base)
    prune_models = {str(row["model"]) for row in pruning}
    quant_models = {str(row["model"]) for row in quantization}
    if prune_models != quant_models:
        raise ValueError(
            f"unmatched model panels: prune-only={sorted(prune_models-quant_models)}, "
            f"quant-only={sorted(quant_models-prune_models)}"
        )
    audit = {
        "pruning_files": len(list(prune_base.glob("*/prune_losses.json"))),
        "quantization_files": len(list(quant_base.glob("*/quant_losses.json"))),
        "matched_models": sorted(prune_models),
        "pruning_rows": len(pruning),
        "quantization_rows": len(quantization),
        "notes": prune_notes + quant_notes,
    }
    return pruning, quantization, audit


def _severity(method: str, coordinate: float) -> float:
    if method == "pruning":
        # The common V6 tested interval is density 1.0 -> 0.3.
        return (1.0 - coordinate) / 0.7
    if method == "quantization":
        # The V10 tested interval is dense/int16 -> int3.
        return (16.0 - coordinate) / 13.0
    raise ValueError(method)


def curve_robustness(
    rows: Sequence[Mapping[str, object]], method: str
) -> dict[str, object]:
    """Return capped-damage AUC and observed/clipped cliff survival.

    Once the first >1-nat cell is reached, deeper values are set to one.  This
    prevents a short pre-cliff interval from looking artificially robust and
    prevents unbounded post-cliff magnitudes from dominating.  If no crossing
    occurs, survival is right-censored at the deepest tested point.
    """
    coordinate_key = "density" if method == "pruning" else "bits"
    points = sorted(
        (
            _severity(method, float(row.get(coordinate_key, row["raw_coordinate"]))),
            float(row["delta_L_c"]),
        )
        for row in rows
    )
    if len(points) < 2:
        raise ValueError(f"too few {method} curve points")
    severities = np.asarray([point[0] for point in points])
    damage = np.asarray([point[1] for point in points])
    if not math.isclose(float(severities[0]), 0.0, abs_tol=1e-12):
        raise ValueError(f"{method} curve lacks a dense anchor")
    crossings = np.flatnonzero(damage > CLIFF_THRESHOLD)
    if len(crossings):
        crossing_index = int(crossings[0])
        cliff_survival = float(severities[crossing_index])
        cliff_censored = False
        capped = damage.copy()
        capped[crossing_index:] = CLIFF_THRESHOLD
    else:
        cliff_survival = float(severities[-1])
        cliff_censored = True
        capped = damage.copy()
    capped = np.minimum(capped, CLIFF_THRESHOLD)
    auc = float(np.trapz(capped, severities))
    precliff = damage <= CLIFF_THRESHOLD
    return {
        "auc_capped_damage": auc,
        "cliff_survival": cliff_survival,
        "cliff_censored": cliff_censored,
        "n_points": len(points),
        "n_precliff_points": int(np.sum(precliff)),
        "max_tested_severity": float(severities[-1]),
    }


def build_scores(
    pruning: Sequence[Mapping[str, object]],
    quantization: Sequence[Mapping[str, object]],
) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for method, rows in (("pruning", pruning), ("quantization", quantization)):
        for row in rows:
            grouped[(method, str(row["model"]), str(row["capability"]))].append(row)
    raw = []
    for (method, model, capability), trajectory in sorted(grouped.items()):
        score = curve_robustness(trajectory, method)
        first = trajectory[0]
        raw.append({
            "method": method,
            "model": model,
            "family": str(first["family"]),
            "capability": capability,
            **score,
        })

    # Standardize within capability and method so capability baselines do not
    # create a pooled cross-mechanism correlation by themselves.
    for method in ("pruning", "quantization"):
        for capability in CAPABILITIES:
            subset = [row for row in raw if row["method"] == method and row["capability"] == capability]
            auc_robust = np.asarray([-float(row["auc_capped_damage"]) for row in subset])
            survival = np.asarray([float(row["cliff_survival"]) for row in subset])
            auc_scale = max(float(np.std(auc_robust)), 1e-12)
            survival_scale = max(float(np.std(survival)), 1e-12)
            for row, auc_value, survival_value in zip(subset, auc_robust, survival):
                row["auc_robustness_z"] = float((auc_value - np.mean(auc_robust)) / auc_scale)
                row["cliff_survival_z"] = float((survival_value - np.mean(survival)) / survival_scale)
                row["robustness_z"] = float(
                    (row["auc_robustness_z"] + row["cliff_survival_z"]) / 2.0
                )

    by_cell: dict[tuple[str, str], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in raw:
        by_cell[(str(row["model"]), str(row["capability"]))][str(row["method"])] = row
    paired = []
    for (model, capability), methods in sorted(by_cell.items()):
        if set(methods) != {"pruning", "quantization"}:
            raise ValueError(f"unpaired robustness cell: {model}/{capability}")
        prune, quant = methods["pruning"], methods["quantization"]
        paired.append({
            "model": model,
            "family": prune["family"],
            "capability": capability,
            "pruning_score": prune["robustness_z"],
            "quantization_score": quant["robustness_z"],
            "pruning_auc": prune["auc_capped_damage"],
            "quantization_auc": quant["auc_capped_damage"],
            "pruning_cliff_survival": prune["cliff_survival"],
            "quantization_cliff_survival": quant["cliff_survival"],
            "pruning_cliff_censored": prune["cliff_censored"],
            "quantization_cliff_censored": quant["cliff_censored"],
        })
    return paired


def _correlations(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or float(np.std(x)) <= 1e-14 or float(np.std(y)) <= 1e-14:
        return math.nan, math.nan
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return pearson, spearman


def correlation_summary(
    rows: Sequence[Mapping[str, object]],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    *,
    label: str,
    cluster_key: str = "model",
) -> dict[str, object]:
    x = np.asarray([float(row["pruning_score"]) for row in rows])
    y = np.asarray([float(row["quantization_score"]) for row in rows])
    pearson, spearman = _correlations(x, y)
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[cluster_key])].append(row)
    names = sorted(groups)
    rng = np.random.default_rng(_stable_seed(label))
    estimates = []
    if len(names) >= 2:
        for _ in range(n_resamples):
            sampled = rng.choice(names, len(names), replace=True)
            sample = [row for name in sampled for row in groups[str(name)]]
            values = _correlations(
                np.asarray([float(row["pruning_score"]) for row in sample]),
                np.asarray([float(row["quantization_score"]) for row in sample]),
            )
            if all(math.isfinite(value) for value in values):
                estimates.append(values)
    array = np.asarray(estimates)
    return {
        "n": len(rows),
        "n_clusters": len(names),
        "cluster_key": cluster_key,
        "pearson": pearson,
        "pearson_ci95": (
            [float(value) for value in np.quantile(array[:, 0], [0.025, 0.975])]
            if len(array) else None
        ),
        "spearman": spearman,
        "spearman_ci95": (
            [float(value) for value in np.quantile(array[:, 1], [0.025, 0.975])]
            if len(array) else None
        ),
        "bootstrap_success": len(estimates),
        "bootstrap_requested": n_resamples,
    }


def _aggregate_scores(rows: Sequence[Mapping[str, object]], key: str) -> list[dict]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    output = []
    for name, cells in sorted(groups.items()):
        output.append({
            key: name,
            "model": name,
            "family": name if key == "family" else str(cells[0]["family"]),
            "pruning_score": float(np.mean([float(row["pruning_score"]) for row in cells])),
            "quantization_score": float(np.mean([float(row["quantization_score"]) for row in cells])),
            "n_cells": len(cells),
        })
    return output


def analyze_correlations(
    scores: Sequence[Mapping[str, object]], n_resamples: int
) -> dict[str, object]:
    pooled = correlation_summary(
        scores, n_resamples, label="v20:pooled-cells", cluster_key="model"
    )
    by_capability = {
        capability: correlation_summary(
            [row for row in scores if row["capability"] == capability],
            n_resamples,
            label=f"v20:capability:{capability}",
            cluster_key="model",
        )
        for capability in CAPABILITIES
    }
    model_means = _aggregate_scores(scores, "model")
    model_level = correlation_summary(
        model_means, n_resamples, label="v20:model-means", cluster_key="model"
    )
    family_means = _aggregate_scores(scores, "family")
    family_level = correlation_summary(
        family_means, n_resamples, label="v20:family-means", cluster_key="family"
    )
    within_family = {}
    family_model_counts = {
        family: len({str(row["model"]) for row in scores if row["family"] == family})
        for family in sorted({str(row["family"]) for row in scores})
    }
    for family, n_models in family_model_counts.items():
        if n_models < 2:
            within_family[family] = {
                "status": "not_estimable_single_model", "n_models": n_models
            }
            continue
        result = correlation_summary(
            [row for row in scores if row["family"] == family],
            n_resamples,
            label=f"v20:within:{family}",
            cluster_key="model",
        )
        result["status"] = "thin_two_models" if n_models == 2 else "estimable"
        result["n_models"] = n_models
        within_family[family] = result
    lofo = {}
    for family in sorted(family_model_counts):
        subset = [row for row in scores if row["family"] != family]
        pearson, spearman = _correlations(
            np.asarray([float(row["pruning_score"]) for row in subset]),
            np.asarray([float(row["quantization_score"]) for row in subset]),
        )
        lofo[family] = {"n": len(subset), "pearson": pearson, "spearman": spearman}
    without_olmo = lofo.get("olmo3", {})
    all_r = float(pooled["pearson"])
    no_olmo_r = float(without_olmo.get("pearson", math.nan))
    if not math.isfinite(no_olmo_r) or all_r * no_olmo_r <= 0 or abs(no_olmo_r) < 0.5 * abs(all_r):
        olmo_verdict = "The pooled association is materially OLMo-3-driven; it is not stable evidence of a general family-level factor."
    else:
        olmo_verdict = "The association remains same-signed and at least half as large without OLMo-3, so it is not driven only by OLMo-3."
    return {
        "pooled_model_capability_cells": pooled,
        "by_capability": by_capability,
        "model_means": {"rows": model_means, "correlation": model_level},
        "across_family_means": {"rows": family_means, "correlation": family_level},
        "within_family": within_family,
        "leave_one_family_out": lofo,
        "olmo_verdict": olmo_verdict,
    }


def load_low_bit_rows(quantization: Sequence[Mapping[str, object]]) -> list[dict]:
    rows = []
    for row in quantization:
        model = str(row["model"])
        bit = int(round(float(row["raw_coordinate"])))
        if model not in LOW_BIT_MODELS or bit not in {3, 4}:
            continue
        family, size_b = LOW_BIT_MODELS[model]
        if str(row["family"]) != family:
            raise ValueError(f"family mismatch for {model}")
        rows.append({
            "model": model,
            "family": family,
            "size_b": size_b,
            "log2_size_over_4b": math.log2(size_b / 4.0),
            "capability": str(row["capability"]),
            "bits": bit,
            "degradation": float(row["delta_L_c"]),
            "collapse": float(row["delta_L_c"]) > CLIFF_THRESHOLD,
        })
    expected = len(LOW_BIT_MODELS) * len(CAPABILITIES) * 2
    if len(rows) != expected:
        raise ValueError(f"expected {expected} low-bit cells, found {len(rows)}")
    return rows


def low_bit_design(row: Mapping[str, object]) -> np.ndarray:
    """Capability FE + family-specific intercepts and log-size slopes."""
    family = str(row["family"])
    capability = str(row["capability"])
    x = float(row["log2_size_over_4b"])
    olmo = float(family == "olmo3")
    qwen = float(family == "qwen3")
    return np.asarray([
        1.0,
        float(capability == "code"),
        float(capability == "qa"),
        olmo,
        qwen,
        x,
        olmo * x,
        qwen * x,
    ])


def _ols_fit(rows: Sequence[Mapping[str, object]], columns: Sequence[int]) -> np.ndarray:
    design = np.vstack([low_bit_design(row)[list(columns)] for row in rows])
    observed = np.asarray([float(row["degradation"]) for row in rows])
    return np.linalg.lstsq(design, observed, rcond=None)[0]


def _ols_predict(
    coefficients: np.ndarray,
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[int],
) -> np.ndarray:
    design = np.vstack([low_bit_design(row)[list(columns)] for row in rows])
    return design @ coefficients


def _prediction_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(observed - predicted)))
    denominator = float(np.sum(np.square(observed - np.mean(observed))))
    r2 = 1.0 - float(np.sum(np.square(observed - predicted))) / denominator if denominator > 0 else math.nan
    return {"mae": mae, "r2": r2}


def _leave_one_model_out(
    rows: Sequence[Mapping[str, object]], columns: Sequence[int]
) -> tuple[dict[str, float], list[dict]]:
    records = []
    skipped_models = []
    total = len(rows)
    for model in sorted({str(row["model"]) for row in rows}):
        train = [row for row in rows if row["model"] != model]
        test = [row for row in rows if row["model"] == model]
        train_design = np.vstack([low_bit_design(row)[list(columns)] for row in train])
        if np.linalg.matrix_rank(train_design) < len(columns):
            skipped_models.append(model)
            continue
        coefficients = _ols_fit(train, columns)
        predictions = _ols_predict(coefficients, test, columns)
        for row, prediction in zip(test, predictions):
            records.append({
                "model": model,
                "family": row["family"],
                "capability": row["capability"],
                "observed": float(row["degradation"]),
                "predicted": float(prediction),
            })
    observed = np.asarray([row["observed"] for row in records])
    predicted = np.asarray([row["predicted"] for row in records])
    metrics = _prediction_metrics(observed, predicted) if records else {"mae": math.nan, "r2": math.nan}
    metrics.update({
        "n_predictions": len(records),
        "n_total": total,
        "coverage": len(records) / total if total else 0.0,
        "skipped_models": skipped_models,
    })
    return metrics, records


def _cluster_bootstrap_family_parameters(
    rows: Sequence[Mapping[str, object]], bit: int, n_resamples: int
) -> dict[str, object]:
    grouped_family: dict[str, dict[str, list[Mapping[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped_family[str(row["family"])][str(row["model"])].append(row)
    rng = np.random.default_rng(_stable_seed(f"v20:low-bit:{bit}:coefficients"))
    estimates: dict[str, list[float]] = defaultdict(list)
    full_columns = tuple(range(8))
    for _ in range(n_resamples):
        sample = []
        for family in sorted(grouped_family):
            models = sorted(grouped_family[family])
            sampled = rng.choice(models, len(models), replace=True)
            sample.extend(row for model in sampled for row in grouped_family[family][str(model)])
        design = np.vstack([low_bit_design(row) for row in sample])
        if np.linalg.matrix_rank(design) < len(full_columns):
            continue
        beta = _ols_fit(sample, full_columns)
        mean_capability = (beta[1] + beta[2]) / 3.0
        estimates["gemma3_intercept_4b"].append(float(beta[0] + mean_capability))
        estimates["olmo3_intercept_4b"].append(float(beta[0] + beta[3] + mean_capability))
        estimates["qwen3_intercept_4b"].append(float(beta[0] + beta[4] + mean_capability))
        estimates["gemma3_slope_per_doubling"].append(float(beta[5]))
        estimates["olmo3_slope_per_doubling"].append(float(beta[5] + beta[6]))
        estimates["qwen3_slope_per_doubling"].append(float(beta[5] + beta[7]))
    return {
        key: {
            "estimate_samples": len(values),
            "ci95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
        }
        for key, values in estimates.items()
    }


def _binary_metrics(observed: np.ndarray, predicted: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    accuracy = float(np.mean(observed == predicted))
    positives = observed == 1
    negatives = observed == 0
    tpr = float(np.mean(predicted[positives] == 1)) if np.any(positives) else math.nan
    tnr = float(np.mean(predicted[negatives] == 0)) if np.any(negatives) else math.nan
    balanced = (tpr + tnr) / 2.0 if math.isfinite(tpr) and math.isfinite(tnr) else math.nan
    if np.any(positives) and np.any(negatives):
        ranks = rankdata(scores)
        n_pos, n_neg = int(np.sum(positives)), int(np.sum(negatives))
        auc = float((np.sum(ranks[positives]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    else:
        auc = math.nan
    return {"accuracy": accuracy, "balanced_accuracy": balanced, "roc_auc": auc}


def analyze_low_bits(rows: Sequence[Mapping[str, object]], n_resamples: int) -> dict[str, object]:
    specifications = {
        "capability_only": (0, 1, 2),
        "family_plus_size": (0, 1, 2, 3, 4, 5),
        "family_by_size": tuple(range(8)),
    }
    by_bit = {}
    for bit in (4, 3):
        subset = [row for row in rows if int(row["bits"]) == bit]
        full_beta = _ols_fit(subset, specifications["family_by_size"])
        mean_capability = (full_beta[1] + full_beta[2]) / 3.0
        family_parameters = {
            "gemma3": {
                "mean_capability_degradation_at_4b": float(full_beta[0] + mean_capability),
                "slope_per_size_doubling": float(full_beta[5]),
            },
            "olmo3": {
                "mean_capability_degradation_at_4b": float(full_beta[0] + full_beta[3] + mean_capability),
                "slope_per_size_doubling": float(full_beta[5] + full_beta[6]),
            },
            "qwen3": {
                "mean_capability_degradation_at_4b": float(full_beta[0] + full_beta[4] + mean_capability),
                "slope_per_size_doubling": float(full_beta[5] + full_beta[7]),
            },
        }
        heldout = {}
        heldout_records: dict[str, list[dict]] = {}
        for name, columns in specifications.items():
            metrics, records = _leave_one_model_out(subset, columns)
            heldout[name] = metrics
            heldout_records[name] = records
        by_bit[str(bit)] = {
            "n": len(subset),
            "family_parameters": family_parameters,
            "family_parameter_bootstrap": _cluster_bootstrap_family_parameters(
                subset, bit, n_resamples
            ),
            "leave_one_model_out": heldout,
            "full_coefficients": full_beta.tolist(),
        }
        if bit == 3:
            cliff_results = {}
            for specification in ("family_plus_size", "family_by_size"):
                records = heldout_records[specification]
                observed = np.asarray([float(row["observed"]) > CLIFF_THRESHOLD for row in records], dtype=int)
                scores = np.asarray([float(row["predicted"]) for row in records])
                predicted = np.asarray(scores > CLIFF_THRESHOLD, dtype=int)
                cliff_results[specification] = {
                    **_binary_metrics(observed, predicted, scores),
                    "n_predictions": len(records),
                    "n_total": len(subset),
                    "n_collapsed": int(np.sum(observed)),
                    "n_not_collapsed": int(len(observed) - np.sum(observed)),
                    "majority_accuracy": (
                        float(max(np.mean(observed), 1.0 - np.mean(observed)))
                        if len(observed) else math.nan
                    ),
                    "records": [
                        {**row, "observed_collapse": bool(obs), "predicted_collapse": bool(pred)}
                        for row, obs, pred in zip(records, observed, predicted)
                    ],
                }
            by_bit[str(bit)]["cliff_prediction"] = cliff_results

    model_table = []
    for model in sorted(LOW_BIT_MODELS, key=lambda value: (LOW_BIT_MODELS[value][0], LOW_BIT_MODELS[value][1])):
        model_rows = [row for row in rows if row["model"] == model]
        values = {bit: [float(row["degradation"]) for row in model_rows if row["bits"] == bit] for bit in (4, 3)}
        collapsed = sum(value > CLIFF_THRESHOLD for value in values[3])
        model_table.append({
            "model": model,
            "family": LOW_BIT_MODELS[model][0],
            "size_b": LOW_BIT_MODELS[model][1],
            "int4_mean_degradation": float(np.mean(values[4])),
            "int3_mean_degradation": float(np.mean(values[3])),
            "int3_mean_excess_over_1nat": float(np.mean(np.maximum(np.asarray(values[3]) - 1.0, 0.0))),
            "int3_collapsed_capabilities": collapsed,
            "int3_cliff_indicator": "all" if collapsed == 3 else "none" if collapsed == 0 else "partial",
        })
    return {
        "definition": "collapse iff int3 delta loss > 1 nat; size is nominal checkpoint label in billions",
        "by_bit": by_bit,
        "model_table": model_table,
    }


def analyze(
    pruning: Sequence[Mapping[str, object]],
    quantization: Sequence[Mapping[str, object]],
    audit: Mapping[str, object],
    n_resamples: int,
) -> dict[str, object]:
    scores = build_scores(pruning, quantization)
    low_bit_rows = load_low_bit_rows(quantization)
    return {
        "generated": date.today().isoformat(),
        "audit": dict(audit),
        "score_definition": {
            "damage": "delta capability loss from each method's own dense anchor",
            "auc": "trapezoidal tested-range AUC; first >1-nat crossing and deeper cells capped at 1 nat",
            "cliff": "severity of first >1-nat crossing; right-censored at deepest tested point if absent",
            "standardization": "within method and capability across 12 models; mean of z(-AUC) and z(cliff survival)",
        },
        "scores": scores,
        "correlations": analyze_correlations(scores, n_resamples),
        "low_bit_family_size": analyze_low_bits(low_bit_rows, n_resamples),
    }


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def _fmt_ci(value: object) -> str:
    if not isinstance(value, Sequence) or len(value) != 2:
        return "n/a"
    return f"[{_fmt(value[0])}, {_fmt(value[1])}]"


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def build_markdown(summary: Mapping[str, object], n_resamples: int) -> str:
    correlations = summary["correlations"]
    low_bits = summary["low_bit_family_size"]
    assert isinstance(correlations, Mapping) and isinstance(low_bits, Mapping)
    pooled = correlations["pooled_model_capability_cells"]
    model_level = correlations["model_means"]["correlation"]
    family_level = correlations["across_family_means"]["correlation"]
    lines = [
        "# Pruning–quantization robustness correlation and low-bit family × size",
        "",
        f"Generated {summary['generated']} by `analysis/v20_robustness_corr.py` from existing V6/V10 loss artifacts. This was CPU-only and performed no inference, training, or input mutation.",
        "",
        "## E2: cross-mechanism robustness",
        "",
        "For each model × capability and mechanism, damage is relative to that mechanism's own dense anchor. The damage AUC spans the tested severity interval (pruning density 1.0→0.3; quantization int16→int3). At the first `ΔL>1` nat crossing and thereafter it is capped at one, so enormous post-cliff losses do not dominate and early cliffs do not benefit from a shorter integration range. Cliff survival is the first crossing severity; absent crossings are explicitly right-censored at the deepest tested setting. `robustness_z` is the mean of within-capability `z(-AUC)` and `z(cliff survival)`.",
        "",
        "Bootstrap intervals resample models as clusters and retain all three capability cells. They measure model-panel sensitivity, not seeds; each loss cell has one artifact and no repeated seed.",
        "",
        "| Scope | n cells/units | Pearson r (95% CI) | Spearman ρ (95% CI) |",
        "|---|---:|---:|---:|",
        f"| pooled model × capability | {pooled['n']} cells / {pooled['n_clusters']} models | {_fmt(pooled['pearson'])} {_fmt_ci(pooled['pearson_ci95'])} | {_fmt(pooled['spearman'])} {_fmt_ci(pooled['spearman_ci95'])} |",
        f"| model means across capabilities | {model_level['n']} models | {_fmt(model_level['pearson'])} {_fmt_ci(model_level['pearson_ci95'])} | {_fmt(model_level['spearman'])} {_fmt_ci(model_level['spearman_ci95'])} |",
        f"| across family means | {family_level['n']} families | {_fmt(family_level['pearson'])} {_fmt_ci(family_level['pearson_ci95'])} | {_fmt(family_level['spearman'])} {_fmt_ci(family_level['spearman_ci95'])} |",
    ]
    for capability, result in correlations["by_capability"].items():
        lines.append(
            f"| {capability}, across models | {result['n']} models | {_fmt(result['pearson'])} {_fmt_ci(result['pearson_ci95'])} | {_fmt(result['spearman'])} {_fmt_ci(result['spearman_ci95'])} |"
        )
    lines.extend([
        "",
        "### Within-family and leave-one-family-out stability",
        "",
        "| Family | models | Pearson r (95% CI) | Spearman ρ (95% CI) | status |",
        "|---|---:|---:|---:|---|",
    ])
    for family, result in correlations["within_family"].items():
        lines.append(
            f"| {family} | {result.get('n_models', 0)} | {_fmt(result.get('pearson'))} {_fmt_ci(result.get('pearson_ci95'))} | {_fmt(result.get('spearman'))} {_fmt_ci(result.get('spearman_ci95'))} | {result['status']} |"
        )
    lines.extend([
        "",
        "| Family omitted | remaining cells | Pearson r | Spearman ρ |",
        "|---|---:|---:|---:|",
    ])
    for family, result in correlations["leave_one_family_out"].items():
        lines.append(
            f"| {family} | {result['n']} | {_fmt(result['pearson'])} | {_fmt(result['spearman'])} |"
        )
    lines.extend([
        "",
        f"**OLMo-3 audit:** {correlations['olmo_verdict']}",
        "",
        "**E2 verdict:** the positive pooled association is suggestive, and it does not vanish when OLMo-3 is omitted, but it is not stable enough to establish a shared family-level robustness factor. Spearman's pooled interval includes zero, the five-family interval is extremely broad, Qwen-3 has near-zero within-family correlation, and two families have only one model.",
        "",
        "Singleton Gemma-4 and Muse panels cannot supply within-family correlations. OLMo-3 has only two sizes, so its within-family cluster interval is especially unstable. The five-point family-mean correlation is descriptive and its bootstrap is necessarily broad.",
        "",
        "## E1: int4/int3 family × size",
        "",
        "The continuous model is `ΔL = capability FE + family + log2(size/4B) + family×log2(size/4B)`, fitted separately at int4 and int3. The table reports family-average intercepts at 4B and slopes per size doubling. CIs use a stratified model-cluster bootstrap (models resampled within each family). Nominal checkpoint sizes are used because underlying parameter-count conventions differ between multimodal and text-only artifacts.",
        "",
        f"Only rank-complete interaction resamples enter the coefficient intervals: {low_bits['by_bit']['4']['family_parameter_bootstrap']['gemma3_intercept_4b']['estimate_samples']}/{n_resamples} at int4 and {low_bits['by_bit']['3']['family_parameter_bootstrap']['gemma3_intercept_4b']['estimate_samples']}/{n_resamples} at int3. The rest repeat too few distinct sizes in at least one family and are discarded rather than assigned minimum-norm interaction slopes.",
        "",
        "| bit | family | predicted mean ΔL at 4B (95% CI) | slope per size doubling (95% CI) |",
        "|---:|---|---:|---:|",
    ])
    for bit in (4, 3):
        result = low_bits["by_bit"][str(bit)]
        bootstrap = result["family_parameter_bootstrap"]
        for family, values in result["family_parameters"].items():
            intercept_ci = bootstrap[f"{family}_intercept_4b"]["ci95"]
            slope_ci = bootstrap[f"{family}_slope_per_doubling"]["ci95"]
            lines.append(
                f"| int{bit} | {family} | {_fmt(values['mean_capability_degradation_at_4b'])} {_fmt_ci(intercept_ci)} | {_fmt(values['slope_per_size_doubling'])} {_fmt_ci(slope_ci)} |"
            )
    lines.extend([
        "",
        "### Held-out predictability",
        "",
        "| bit | predictors | held-out coverage | leave-one-model-out MAE | leave-one-model-out R² |",
        "|---:|---|---:|---:|---:|",
    ])
    for bit in (4, 3):
        heldout = low_bits["by_bit"][str(bit)]["leave_one_model_out"]
        for name, metrics in heldout.items():
            lines.append(
                f"| int{bit} | {name} | {metrics['n_predictions']}/{metrics['n_total']} | {_fmt(metrics['mae'])} | {_fmt(metrics['r2'])} |"
            )
    cliff_predictions = low_bits["by_bit"]["3"]["cliff_prediction"]
    additive_cliff = cliff_predictions["family_plus_size"]
    interaction_cliff = cliff_predictions["family_by_size"]
    lines.extend([
        "",
        f"For the binary int3 cliff (`ΔL>1` nat), the identifiable additive family+size model covers {additive_cliff['n_predictions']}/{additive_cliff['n_total']} held-out cells and achieves accuracy {_fmt(additive_cliff['accuracy'])}, balanced accuracy {_fmt(additive_cliff['balanced_accuracy'])}, and ROC AUC {_fmt(additive_cliff['roc_auc'])}. The majority-class accuracy is {_fmt(additive_cliff['majority_accuracy'])} because {additive_cliff['n_collapsed']}/{additive_cliff['n_collapsed'] + additive_cliff['n_not_collapsed']} capability cells collapse.",
        "",
        f"The family×size interaction covers only {interaction_cliff['n_predictions']}/{interaction_cliff['n_total']} cells: holding out either OLMo model leaves one OLMo size, so its family-specific slope is rank-deficient. Both non-collapse cells belong to held-out OLMo-3 32B and are therefore absent from this interaction check; its balanced accuracy and ROC AUC are {_fmt(interaction_cliff['balanced_accuracy'])} and {_fmt(interaction_cliff['roc_auc'])}, not evidence of cliff prediction. This is a data-limit result, not a fitted success.",
        "",
        "**E1 verdict:** family plus size is moderately predictive of degradation (and ranks the rare int3 non-collapses reasonably), but the family×size interaction is not validated. It loses OLMo holdout coverage and does not improve the comparable magnitude fit. More OLMo sizes—and more non-collapse examples—are required before claiming a predictable family-specific low-bit cliff.",
        "",
        "### Observed low-bit cliff table",
        "",
        "| Model | family | size (B) | mean int4 ΔL | mean int3 ΔL (cliff magnitude) | mean int3 excess over 1 nat | int3 collapsed capabilities | indicator |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in low_bits["model_table"]:
        lines.append(
            f"| {row['model']} | {row['family']} | {row['size_b']:g} | {_fmt(row['int4_mean_degradation'])} | {_fmt(row['int3_mean_degradation'])} | {_fmt(row['int3_mean_excess_over_1nat'])} | {row['int3_collapsed_capabilities']}/3 | {row['int3_cliff_indicator']} |"
        )
    lines.extend([
        "",
        "The int3 response is highly imbalanced and family-specific: almost all cells collapse, while OLMo-3 32B is censored in math/QA but crosses by a small margin in code. Accordingly, a high binary accuracy can be trivial; magnitude MAE/R² and the two non-collapse cells carry most of the discriminating information. With only 2 OLMo and 3 Qwen sizes, the interaction slopes should be treated as descriptive unless their leave-one-model-out performance improves materially over the additive form.",
        "",
        "## Provenance and limitations",
        "",
        f"- Inputs: {summary['audit']['pruning_files']} `results/v6-capability-geometry/*/prune_losses.json` files and {summary['audit']['quantization_files']} `results/v10-quantization/*/quant_losses.json` files; 12 models match exactly.",
        f"- Bootstrap count: {n_resamples:,}. No seed-level uncertainty is estimable.",
        "- Cliff survival is censored at the tested boundary, not extrapolated. Treating censored and exactly-at-boundary survival equally is conservative but compresses the most robust models together.",
        "- The composite score is a declared diagnostic, not a fitted latent variable. Separate AUC and cliff fields for every model × capability are retained in `summary.json`.",
    ])
    return "\n".join(lines) + "\n"


def dry_run_text(
    pruning: Sequence[Mapping[str, object]],
    quantization: Sequence[Mapping[str, object]],
    audit: Mapping[str, object],
) -> str:
    matched = audit["matched_models"]
    low_bit = load_low_bit_rows(quantization)
    return "\n".join([
        "V20 dry run — no files written",
        f"pruning inputs: {audit['pruning_files']} files / {len(pruning)} model-capability-coordinate rows",
        f"quantization inputs: {audit['quantization_files']} files / {len(quantization)} model-capability-coordinate rows",
        f"matched robustness panel: {len(matched)} models / {len(matched) * len(CAPABILITIES)} model-capability pairs",
        f"families: {len({row['family'] for row in pruning})}; capabilities: {len(CAPABILITIES)}",
        f"E1 low-bit panel: {len(LOW_BIT_MODELS)} models / {len(low_bit)} int4+int3 model-capability cells",
        "E1 families: gemma3=5 models, olmo3=2, qwen3=3",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args()
    if args.bootstrap_resamples < 100:
        parser.error("--bootstrap-resamples must be at least 100")
    pruning, quantization, audit = load_method_rows()
    if args.dry_run:
        print(dry_run_text(pruning, quantization, audit))
        return
    summary = analyze(pruning, quantization, audit, args.bootstrap_resamples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.md").write_text(
        build_markdown(summary, args.bootstrap_resamples), encoding="utf-8"
    )
    print(f"wrote {args.output_dir / 'summary.md'}")
    print(f"wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
