#!/usr/bin/env python3
"""Stage-A held-out refitting battery over existing compression artifacts.

This module is deliberately CPU-only: it reads V6/V10/V13 artifacts, fits
declared functional forms, and writes the V14 validation report.  It never
loads a model or creates a compressed checkpoint.

Usage:
    python analysis/v14_fitting.py
    python analysis/v14_fitting.py --parts 1,2,3,4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
PRUNE_BASE = ROOT / "results/v6-capability-geometry"
QUANT_BASE = ROOT / "results/v10-quantization"
RECOVERY_BASE = ROOT / "results/v13-recovery"
OUT_BASE = ROOT / "results/v14-fitting"

CAPABILITIES = ("math", "code", "qa")
PRECLIFF_MIN = 0.02
PRECLIFF_MAX = 8.0
SHALLOW_DENSITY = 0.55
MILD_DENSITY = 0.7
BOOTSTRAP_RESAMPLES = 1_000


def parse_parts(value: str | Sequence[int]) -> list[int]:
    """Parse a unique subset of Stage-A parts 1--4."""
    if isinstance(value, str):
        try:
            parts = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
        except ValueError as exc:
            raise argparse.ArgumentTypeError("--parts must be comma-separated integers") from exc
    else:
        parts = [int(part) for part in value]
    if not parts or any(part not in {1, 2, 3, 4} for part in parts):
        raise argparse.ArgumentTypeError("--parts must contain values from 1,2,3,4")
    if len(parts) != len(set(parts)):
        raise argparse.ArgumentTypeError("--parts values must be unique")
    return sorted(parts)


def canonical_model_tag(tag: str) -> str:
    """Normalize directory tags used by different experiment stages."""
    if tag.startswith("Qwen--"):
        return tag.removeprefix("Qwen--")
    return tag


def model_family(tag: str, model_name: str = "") -> str:
    """Infer the stable family label from artifact names only."""
    text = f"{tag} {model_name}".lower()
    if "qwen3" in text:
        return "qwen3"
    if "gemma4" in text or "gemma-4" in text:
        return "gemma4"
    if "gemma3" in text or "gemma-3" in text:
        return "gemma3"
    if "olmo3" in text or "olmo-3" in text:
        return "olmo3"
    if "muse" in text or "glimmer" in text:
        return "muse_glimmer"
    return canonical_model_tag(tag).split("-")[0].lower()


def deleted_mass_fraction(
    counts: Sequence[int | float],
    mass: Sequence[int | float],
    density: float,
) -> float:
    """Interpolate the fraction of capability mass deleted at ``density``.

    The V6 spectrum histogram is ordered by increasing absolute weight.  A
    pruning density ``d`` deletes ``(1-d) * sum(counts)`` parameters.  If that
    target falls inside a histogram bin, the bin's capability mass is included
    in the same fractional proportion as its parameter count.
    """
    count_array = np.asarray(counts, dtype=np.float64)
    mass_array = np.asarray(mass, dtype=np.float64)
    if count_array.ndim != 1 or mass_array.ndim != 1 or count_array.shape != mass_array.shape:
        raise ValueError("counts and mass must be equal-length one-dimensional arrays")
    if not np.isfinite(density) or density < 0.0 or density > 1.0:
        raise ValueError("density must lie in [0, 1]")
    if np.any(~np.isfinite(count_array)) or np.any(count_array < 0):
        raise ValueError("counts must be finite and non-negative")
    if np.any(~np.isfinite(mass_array)) or np.any(mass_array < 0):
        raise ValueError("mass must be finite and non-negative")
    total_count = float(np.sum(count_array))
    total_mass = float(np.sum(mass_array))
    if total_count <= 0.0 or total_mass <= 0.0:
        return 0.0
    target = float(np.clip((1.0 - density) * total_count, 0.0, total_count))
    if target <= 0.0:
        return 0.0
    if target >= total_count:
        return 1.0
    cumulative_count = np.cumsum(count_array)
    index = int(np.searchsorted(cumulative_count, target, side="left"))
    count_before = float(cumulative_count[index - 1]) if index else 0.0
    mass_before = float(np.sum(mass_array[:index]))
    boundary_count = float(count_array[index])
    fraction = 0.0 if boundary_count == 0.0 else (target - count_before) / boundary_count
    deleted_mass = mass_before + float(np.clip(fraction, 0.0, 1.0)) * float(mass_array[index])
    return float(np.clip(deleted_mass / total_mass, 0.0, 1.0))


# Descriptive alias used by downstream notebooks and tests.
interpolate_deleted_mass = deleted_mass_fraction


def sign_accuracy_counter(
    predicted: Sequence[int | float], measured: Sequence[int | float]
) -> dict[str, float | int | list[float]]:
    """Count sign matches and return a two-sided 95% Wilson binomial CI."""
    prediction = np.asarray(predicted, dtype=np.float64)
    observation = np.asarray(measured, dtype=np.float64)
    if prediction.ndim != 1 or observation.ndim != 1 or prediction.shape != observation.shape:
        raise ValueError("predicted and measured must be equal-length one-dimensional arrays")
    valid = np.isfinite(prediction) & np.isfinite(observation)
    prediction, observation = prediction[valid], observation[valid]
    n = int(prediction.size)
    correct = int(np.sum(np.sign(prediction) == np.sign(observation)))
    if n == 0:
        return {"n_cells": 0, "n_correct": 0, "accuracy": 0.0, "ci95": [0.0, 1.0]}
    proportion = correct / n
    z = 1.959963984540054
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)) / denominator
    return {
        "n_cells": n,
        "n_correct": correct,
        "accuracy": float(proportion),
        "ci95": [float(max(0.0, center - radius)), float(min(1.0, center + radius))],
    }


def leave_one_bit_out_splits(
    bits: Sequence[int] = (8, 6, 4, 3),
) -> list[tuple[list[int], int]]:
    """Return the two pre-registered quantization validation splits."""
    available = list(dict.fromkeys(int(bit) for bit in bits))
    required = {8, 6, 4, 3}
    if not required.issubset(available):
        raise ValueError("leave-one-bit-out validation requires bits 8, 6, 4, and 3")
    return [([8, 6, 3], 4), ([8, 6, 4], 3)]


def saturating_recovery_form(
    recovery_tokens: Sequence[int | float] | np.ndarray | float,
    initial_damage: float,
    residual_fraction: float,
    d0: float,
    beta: float,
) -> np.ndarray:
    """Evaluate the anchored, monotone saturating recovery damage form."""
    tokens = np.asarray(recovery_tokens, dtype=np.float64)
    if np.any(~np.isfinite(tokens)) or np.any(tokens < 0):
        raise ValueError("recovery tokens must be finite and non-negative")
    if not 0.0 <= residual_fraction <= 1.0:
        raise ValueError("residual_fraction must lie in [0, 1]")
    if not np.isfinite(d0) or d0 <= 0.0 or not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("d0 and beta must be finite and positive")
    decay = np.power(1.0 + tokens / d0, -beta)
    return float(initial_damage) * (residual_fraction + (1.0 - residual_fraction) * decay)


# Short alias matching the equation's terminology.
saturating_recovery_damage = saturating_recovery_form


def _json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "little")


def _fit_log_power(x: Sequence[float], y: Sequence[float]) -> dict:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_array) & np.isfinite(y_array) & (x_array > 0.0) & (y_array > 0.0)
    x_array, y_array = x_array[valid], y_array[valid]
    if x_array.size < 2 or np.unique(x_array).size < 2:
        return {"status": "too_few_points", "n_points": int(x_array.size), "amplitude": None, "exponent": None, "r2_log": None}
    log_x, log_y = np.log(x_array), np.log(y_array)
    design = np.column_stack([np.ones_like(log_x), log_x])
    intercept, exponent = np.linalg.lstsq(design, log_y, rcond=None)[0]
    fitted = design @ np.array([intercept, exponent])
    residual_sum = float(np.sum(np.square(log_y - fitted)))
    total_sum = float(np.sum(np.square(log_y - np.mean(log_y))))
    r2 = 1.0 if total_sum <= np.finfo(float).eps and residual_sum <= np.finfo(float).eps else (0.0 if total_sum <= np.finfo(float).eps else 1.0 - residual_sum / total_sum)
    return {
        "status": "ok",
        "n_points": int(x_array.size),
        "amplitude": float(math.exp(float(intercept))),
        "exponent": float(exponent),
        "r2_log": float(r2),
    }


def _fit_shared_log_power(rows: Sequence[dict], x_key: str, y_key: str, group_keys: Sequence[str]) -> dict:
    valid_rows = [row for row in rows if np.isfinite(row[x_key]) and np.isfinite(row[y_key]) and row[x_key] > 0.0 and row[y_key] > 0.0]
    groups = sorted({tuple(row[key] for key in group_keys) for row in valid_rows})
    if len(valid_rows) < len(groups) + 1 or len(groups) == 0:
        return {"status": "too_few_points", "n_points": len(valid_rows), "n_groups": len(groups), "exponent": None, "amplitudes": {}, "r2_log": None}
    group_index = {group: index for index, group in enumerate(groups)}
    design = np.zeros((len(valid_rows), len(groups) + 1), dtype=np.float64)
    target = np.empty(len(valid_rows), dtype=np.float64)
    for index, row in enumerate(valid_rows):
        group = tuple(row[key] for key in group_keys)
        design[index, group_index[group]] = 1.0
        design[index, -1] = math.log(float(row[x_key]))
        target[index] = math.log(float(row[y_key]))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return {"status": "rank_deficient", "n_points": len(valid_rows), "n_groups": len(groups), "exponent": None, "amplitudes": {}, "r2_log": None}
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    fitted = design @ coefficients
    residual_sum = float(np.sum(np.square(target - fitted)))
    total_sum = float(np.sum(np.square(target - np.mean(target))))
    r2 = 1.0 if total_sum <= np.finfo(float).eps and residual_sum <= np.finfo(float).eps else (0.0 if total_sum <= np.finfo(float).eps else 1.0 - residual_sum / total_sum)
    amplitudes = {"|".join(map(str, group)): float(math.exp(float(coefficients[index]))) for group, index in group_index.items()}
    return {
        "status": "ok",
        "n_points": len(valid_rows),
        "n_groups": len(groups),
        "exponent": float(coefficients[-1]),
        "amplitudes": amplitudes,
        "r2_log": float(r2),
    }


def _bootstrap_error_metrics(records: Sequence[dict], label: str, n_resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    if not records:
        return {"n_cells": 0, "mae_nats": None, "mae_ci95": None, "relative_error": None, "relative_error_ci95": None}
    observed = np.asarray([record["observed"] for record in records], dtype=np.float64)
    predicted = np.asarray([record["predicted"] for record in records], dtype=np.float64)
    absolute_error = np.abs(predicted - observed)

    def metrics(indices: np.ndarray) -> tuple[float, float]:
        mae = float(np.mean(absolute_error[indices]))
        scale = float(np.mean(np.abs(observed[indices])))
        relative = mae / max(scale, np.finfo(float).eps)
        return mae, relative

    full_index = np.arange(observed.size)
    mae, relative = metrics(full_index)
    generator = np.random.default_rng(_stable_seed(label))
    samples = generator.integers(0, observed.size, size=(n_resamples, observed.size))
    boot_mae = np.mean(absolute_error[samples], axis=1)
    boot_scale = np.mean(np.abs(observed[samples]), axis=1)
    boot_relative = boot_mae / np.maximum(boot_scale, np.finfo(float).eps)
    return {
        "n_cells": int(observed.size),
        "mae_nats": mae,
        "mae_ci95": [float(value) for value in np.percentile(boot_mae, [2.5, 97.5])],
        "relative_error": relative,
        "relative_error_ci95": [float(value) for value in np.percentile(boot_relative, [2.5, 97.5])],
        "bootstrap_resamples": n_resamples,
    }


def _summarize_prediction_records(records: Sequence[dict], label: str) -> dict:
    aggregate = _bootstrap_error_metrics(records, f"{label}:pooled")
    by_family = {}
    for family in sorted({record["family"] for record in records}):
        subset = [record for record in records if record["family"] == family]
        by_family[family] = _bootstrap_error_metrics(subset, f"{label}:{family}")
    return {"aggregate": aggregate, "by_family": by_family, "predictions": list(records)}


def _paired_prediction_comparison(
    candidate_records: Sequence[dict],
    baseline_records: Sequence[dict],
    label: str,
) -> dict:
    """Compare two forms on exactly matching held-out cells."""
    def key(record: Mapping) -> tuple:
        return (
            record.get("model"),
            record.get("capability"),
            record.get("density", record.get("heldout_bit")),
        )

    candidate = {key(record): record for record in candidate_records}
    baseline = {key(record): record for record in baseline_records}
    common = sorted(set(candidate) & set(baseline), key=str)
    if not common:
        return {
            "n_paired_cells": 0,
            "candidate_mae_nats": None,
            "baseline_mae_nats": None,
            "mae_difference_candidate_minus_baseline": None,
            "mae_difference_ci95": None,
        }
    candidate_error = np.asarray(
        [abs(candidate[cell]["predicted"] - candidate[cell]["observed"]) for cell in common],
        dtype=np.float64,
    )
    baseline_error = np.asarray(
        [abs(baseline[cell]["predicted"] - baseline[cell]["observed"]) for cell in common],
        dtype=np.float64,
    )
    difference = candidate_error - baseline_error
    generator = np.random.default_rng(_stable_seed(f"paired:{label}"))
    samples = generator.integers(0, len(common), size=(BOOTSTRAP_RESAMPLES, len(common)))
    bootstrap_difference = np.mean(difference[samples], axis=1)
    return {
        "n_paired_cells": len(common),
        "candidate_mae_nats": float(np.mean(candidate_error)),
        "baseline_mae_nats": float(np.mean(baseline_error)),
        "mae_difference_candidate_minus_baseline": float(np.mean(difference)),
        "mae_difference_ci95": [float(value) for value in np.percentile(bootstrap_difference, [2.5, 97.5])],
        "candidate_wins": bool(np.mean(candidate_error) < np.mean(baseline_error)),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }


def _pruning_comparisons(by_form: Mapping[str, Sequence[dict]], protocol: str) -> dict:
    pairs = {
        "P2_vs_direct_P2": ("P2_deleted_mass", "baseline_P2_direct"),
        "P3_vs_direct_P2": ("P3_shared_gamma", "baseline_P2_direct"),
        "P2_vs_P1": ("P2_deleted_mass", "P1_raw_sparsity"),
        "P3_vs_P1": ("P3_shared_gamma", "P1_raw_sparsity"),
    }
    return {
        name: {
            "candidate": candidate,
            "baseline": baseline,
            **_paired_prediction_comparison(
                by_form[candidate], by_form[baseline], f"{protocol}:{name}"
            ),
        }
        for name, (candidate, baseline) in pairs.items()
    }


def load_pruning_rows(base: Path = PRUNE_BASE) -> tuple[list[dict], dict[str, dict], list[dict]]:
    """Load complete V6 model directories and construct capability cells."""
    rows: list[dict] = []
    models: dict[str, dict] = {}
    skipped: list[dict] = []
    required = ("prune_losses.json", "alignment.json", "spectrum_bins.npz", "fisher_meta.json")
    if not base.exists():
        return rows, models, [{"path": str(base), "reason": "input_directory_missing"}]
    for directory in sorted(path for path in base.iterdir() if path.is_dir()):
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            skipped.append({"model": directory.name, "reason": "incomplete_artifacts", "missing": missing})
            continue
        try:
            losses = _json_load(directory / "prune_losses.json")
            alignment = _json_load(directory / "alignment.json")
            metadata = _json_load(directory / "fisher_meta.json")
            spectrum = np.load(directory / "spectrum_bins.npz")
            counts = spectrum["counts"]
            dense = losses["1.0"]
            model = canonical_model_tag(directory.name)
            family = model_family(model, str(metadata.get("model", "")))
            model_rows = 0
            for capability in CAPABILITIES:
                mass_key = f"mass_{capability}"
                if capability not in dense or capability not in alignment or mass_key not in spectrum.files:
                    continue
                loss_at_density = {float(key): value for key, value in losses.items() if key != "1.0" and capability in value}
                alignment_at_density = {float(key): value for key, value in alignment[capability].items()}
                for density in sorted(set(loss_at_density) & set(alignment_at_density), reverse=True):
                    damage = float(loss_at_density[density][capability]) - float(dense[capability])
                    first_order = float(alignment_at_density[density]["first_order"])
                    residual = damage - first_order
                    deleted_mass = deleted_mass_fraction(counts, spectrum[mass_key], density)
                    rows.append({
                        "model": model,
                        "family": family,
                        "capability": capability,
                        "density": float(density),
                        "sparsity": float(1.0 - density),
                        "damage": damage,
                        "first_order": first_order,
                        "residual": residual,
                        "deleted_mass": deleted_mass,
                        "precliff": bool(PRECLIFF_MIN < residual < PRECLIFF_MAX),
                    })
                    model_rows += 1
            models[model] = {
                "model": model,
                "resolved_model": metadata.get("model"),
                "family": family,
                "n_params": int(metadata.get("n_params", int(np.sum(counts)))),
                "n_cells": model_rows,
                "artifact_dir": str(directory.relative_to(ROOT)),
            }
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
            skipped.append({"model": directory.name, "reason": "artifact_read_error", "detail": str(exc)})
    return rows, models, skipped


_PRUNE_FORMS = {
    "P1_raw_sparsity": {"x": "sparsity", "target": "residual", "corrected": True, "shared": False},
    "P2_deleted_mass": {"x": "deleted_mass", "target": "residual", "corrected": True, "shared": False},
    "P3_shared_gamma": {"x": "deleted_mass", "target": "residual", "corrected": True, "shared": True},
    "baseline_P1_direct": {"x": "sparsity", "target": "damage", "corrected": False, "shared": False},
    "baseline_P2_direct": {"x": "deleted_mass", "target": "damage", "corrected": False, "shared": False},
}


def _prediction_record(row: dict, predicted: float, protocol: str, form: str, extra: Mapping | None = None) -> dict:
    record = {
        "protocol": protocol,
        "form": form,
        "model": row["model"],
        "family": row["family"],
        "capability": row["capability"],
        "density": row["density"],
        "observed": row["damage"],
        "predicted": float(predicted),
        "absolute_error": float(abs(predicted - row["damage"])),
    }
    if extra:
        record.update(extra)
    return record


def _predict_power(row: dict, spec: dict, amplitude: float, exponent: float) -> float:
    curve = float(amplitude) * float(row[spec["x"]]) ** float(exponent)
    return float(row["first_order"] + curve if spec["corrected"] else curve)


def _all_pruning_fits(branch: Sequence[dict], models: Mapping[str, dict]) -> list[dict]:
    fits: list[dict] = []
    for model in sorted(models):
        model_rows = [row for row in branch if row["model"] == model]
        for form, spec in _PRUNE_FORMS.items():
            if spec["shared"]:
                fit = _fit_shared_log_power(model_rows, spec["x"], spec["target"], ("capability",))
                fits.append({"model": model, "family": models[model]["family"], "capability": "shared", "form": form, **fit})
                continue
            for capability in CAPABILITIES:
                capability_rows = [row for row in model_rows if row["capability"] == capability]
                fit = _fit_log_power([row[spec["x"]] for row in capability_rows], [row[spec["target"]] for row in capability_rows])
                fits.append({"model": model, "family": models[model]["family"], "capability": capability, "form": form, **fit})
    return fits


def _validation_a(branch: Sequence[dict], models: Mapping[str, dict]) -> dict:
    by_form: dict[str, list[dict]] = {form: [] for form in _PRUNE_FORMS}
    fit_details: list[dict] = []
    for model in sorted(models):
        model_rows = [row for row in branch if row["model"] == model]
        for form, spec in _PRUNE_FORMS.items():
            train = [row for row in model_rows if row["density"] >= SHALLOW_DENSITY]
            test = [row for row in model_rows if row["density"] < SHALLOW_DENSITY]
            if spec["shared"]:
                fit = _fit_shared_log_power(train, spec["x"], spec["target"], ("capability",))
                fit_details.append({"model": model, "form": form, "train": "density>=0.55", **fit})
                if fit["status"] != "ok":
                    continue
                for row in test:
                    key = row["capability"]
                    amplitude = fit["amplitudes"].get(key)
                    if amplitude is not None:
                        by_form[form].append(_prediction_record(row, _predict_power(row, spec, amplitude, fit["exponent"]), "V-a", form))
                continue
            for capability in CAPABILITIES:
                cap_train = [row for row in train if row["capability"] == capability]
                cap_test = [row for row in test if row["capability"] == capability]
                fit = _fit_log_power([row[spec["x"]] for row in cap_train], [row[spec["target"]] for row in cap_train])
                fit_details.append({"model": model, "capability": capability, "form": form, "train": "density>=0.55", **fit})
                if fit["status"] != "ok":
                    continue
                for row in cap_test:
                    by_form[form].append(_prediction_record(row, _predict_power(row, spec, fit["amplitude"], fit["exponent"]), "V-a", form))
    return {
        "description": "fit density >= 0.55; predict deeper pre-cliff cells",
        "fits": fit_details,
        "forms": {form: _summarize_prediction_records(records, f"V-a:{form}") for form, records in by_form.items()},
        "paired_comparisons": _pruning_comparisons(by_form, "V-a"),
    }


def _individual_slope_median(rows: Sequence[dict], spec: dict) -> tuple[float | None, list[dict]]:
    slopes = []
    details = []
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["capability"])].append(row)
    for (model, capability), group_rows in sorted(groups.items()):
        fit = _fit_log_power([row[spec["x"]] for row in group_rows], [row[spec["target"]] for row in group_rows])
        details.append({"model": model, "capability": capability, **fit})
        if fit["status"] == "ok":
            slopes.append(float(fit["exponent"]))
    return (float(np.median(slopes)) if slopes else None), details


def _transfer_slope(rows: Sequence[dict], form: str, spec: dict) -> tuple[float | None, dict]:
    if spec["shared"]:
        fit = _fit_shared_log_power(rows, spec["x"], spec["target"], ("model", "capability"))
        return (float(fit["exponent"]) if fit["status"] == "ok" else None), {"strategy": "pooled_fixed_effects", "fit": fit}
    slope, fits = _individual_slope_median(rows, spec)
    return slope, {"strategy": "median_source_group_exponent", "source_fits": fits}


def _calibrate_and_predict(target_rows: Sequence[dict], form: str, spec: dict, exponent: float, protocol: str) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    calibrations: list[dict] = []
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in target_rows:
        groups[(row["model"], row["capability"])].append(row)
    for (model, capability), group_rows in sorted(groups.items()):
        candidates = sorted(group_rows, key=lambda row: row["density"], reverse=True)
        if not spec["corrected"]:
            candidates = [row for row in candidates if row["damage"] > 0.0]
        if not candidates:
            continue
        calibration = candidates[0]
        x_value = float(calibration[spec["x"]])
        target_value = float(calibration[spec["target"]])
        if x_value <= 0.0 or target_value <= 0.0:
            continue
        amplitude = target_value / x_value ** exponent
        calibrations.append({"model": model, "capability": capability, "density": calibration["density"], "amplitude": float(amplitude), "exponent": float(exponent)})
        for row in group_rows:
            if row is calibration:
                continue
            prediction = _predict_power(row, spec, amplitude, exponent)
            records.append(_prediction_record(row, prediction, protocol, form, {"calibration_density": calibration["density"]}))
    return records, calibrations


def _validation_b(branch: Sequence[dict], models: Mapping[str, dict]) -> dict:
    families: dict[str, list[str]] = defaultdict(list)
    for model, metadata in models.items():
        families[metadata["family"]].append(model)
    by_form: dict[str, list[dict]] = {form: [] for form in _PRUNE_FORMS}
    transfers: list[dict] = []
    skipped = []
    for family, family_models in sorted(families.items()):
        if len(family_models) < 2:
            skipped.append({"family": family, "reason": "only_one_model"})
            continue
        target_model = max(family_models, key=lambda model: models[model]["n_params"])
        source_models = sorted(model for model in family_models if model != target_model)
        source_rows = [row for row in branch if row["model"] in source_models]
        target_rows = [row for row in branch if row["model"] == target_model]
        for form, spec in _PRUNE_FORMS.items():
            exponent, detail = _transfer_slope(source_rows, form, spec)
            transfer = {"family": family, "target_model": target_model, "source_models": source_models, "form": form, "exponent": exponent, **detail}
            if exponent is not None:
                predictions, calibrations = _calibrate_and_predict(target_rows, form, spec, exponent, "V-b")
                by_form[form].extend(predictions)
                transfer["calibrations"] = calibrations
                transfer["n_test_cells"] = len(predictions)
            transfers.append(transfer)
    return {
        "description": "leave largest model out; transfer exponent and calibrate B on one shallow target cell",
        "transfers": transfers,
        "skipped": skipped,
        "forms": {form: _summarize_prediction_records(records, f"V-b:{form}") for form, records in by_form.items()},
        "paired_comparisons": _pruning_comparisons(by_form, "V-b"),
    }


def _validation_c(branch: Sequence[dict], models: Mapping[str, dict]) -> dict:
    all_families = sorted({metadata["family"] for metadata in models.values()})
    by_form: dict[str, list[dict]] = {form: [] for form in _PRUNE_FORMS}
    transfers: list[dict] = []
    for target_family in all_families:
        source_rows = [row for row in branch if row["family"] != target_family]
        target_rows = [row for row in branch if row["family"] == target_family]
        source_families = sorted({row["family"] for row in source_rows})
        for form, spec in _PRUNE_FORMS.items():
            exponent, detail = _transfer_slope(source_rows, form, spec)
            transfer = {"target_family": target_family, "source_families": source_families, "form": form, "exponent": exponent, **detail}
            if exponent is not None:
                predictions, calibrations = _calibrate_and_predict(target_rows, form, spec, exponent, "V-c")
                by_form[form].extend(predictions)
                transfer["calibrations"] = calibrations
                transfer["n_test_cells"] = len(predictions)
            transfers.append(transfer)
    return {
        "description": "leave one family out; transfer exponent and calibrate B on one shallow target cell",
        "transfers": transfers,
        "forms": {form: _summarize_prediction_records(records, f"V-c:{form}") for form, records in by_form.items()},
        "paired_comparisons": _pruning_comparisons(by_form, "V-c"),
    }


def analyze_pruning(base: Path = PRUNE_BASE) -> dict:
    rows, models, skipped = load_pruning_rows(base)
    branch = [row for row in rows if row["precliff"]]
    mild = [row for row in rows if row["density"] >= MILD_DENSITY]
    sign = sign_accuracy_counter([row["first_order"] for row in mild], [row["damage"] for row in mild])
    sign_by_family = {}
    for family in sorted({row["family"] for row in mild}):
        subset = [row for row in mild if row["family"] == family]
        sign_by_family[family] = sign_accuracy_counter([row["first_order"] for row in subset], [row["damage"] for row in subset])
    return {
        "input_audit": {"n_complete_models": len(models), "models": list(models.values()), "skipped": skipped},
        "dataset": {"n_cells": len(rows), "n_precliff_cells": len(branch), "precliff_rule": f"{PRECLIFF_MIN} < residual < {PRECLIFF_MAX}", "rows": rows},
        "in_sample_fits": _all_pruning_fits(branch, models),
        "validation": {
            "V-a": _validation_a(branch, models),
            "V-b": _validation_b(branch, models),
            "V-c": _validation_c(branch, models),
            "V-d": {"description": "sign(first_order) versus sign(measured damage), density >= 0.7", "aggregate": sign, "by_family": sign_by_family},
        },
    }


def _fit_quant_form(form: str, bits: Sequence[int], damages: Sequence[float]) -> dict:
    x = np.asarray(bits, dtype=np.float64)
    y = np.asarray(damages, dtype=np.float64)
    if x.size < 2 or x.shape != y.shape:
        return {"status": "too_few_points", "n_points": int(x.size), "parameters": None}
    if form == "fixed_4^-b":
        basis = np.power(4.0, -x)
        q = max(0.0, float(np.dot(basis, y) / np.dot(basis, basis)))
        parameters = {"q": q}
        prediction = q * basis
    elif form == "learned_exponential":
        positive = y > 0.0
        if np.sum(positive) >= 2:
            slope, intercept = np.polyfit(x[positive], np.log(y[positive]), 1)
            initial_k = max(1e-4, -float(slope))
            initial_q = max(1e-12, math.exp(float(intercept)))
        else:
            initial_k = math.log(4.0)
            initial_q = max(float(np.max(y)), 1e-6) * math.exp(initial_k * float(np.min(x)))
        result = least_squares(
            lambda theta: np.exp(theta[0]) * np.exp(-np.exp(theta[1]) * x) - y,
            x0=np.array([math.log(initial_q), math.log(initial_k)]),
            bounds=(np.array([-30.0, -9.0]), np.array([30.0, 3.0])),
            max_nfev=20_000,
        )
        q, k = math.exp(float(result.x[0])), math.exp(float(result.x[1]))
        parameters = {"q": q, "k": k}
        prediction = q * np.exp(-k * x)
    elif form == "quadratic_b":
        degree = min(2, x.size - 1)
        coefficients = np.polyfit(x, y, degree)
        if degree < 2:
            coefficients = np.pad(coefficients, (3 - coefficients.size, 0))
        parameters = {"b2": float(coefficients[0]), "b1": float(coefficients[1]), "intercept": float(coefficients[2])}
        prediction = np.polyval(coefficients, x)
    else:
        raise ValueError(f"unknown quantization form {form!r}")
    residual_sum = float(np.sum(np.square(y - prediction)))
    total_sum = float(np.sum(np.square(y - np.mean(y))))
    r2 = 1.0 if total_sum <= np.finfo(float).eps and residual_sum <= np.finfo(float).eps else (0.0 if total_sum <= np.finfo(float).eps else 1.0 - residual_sum / total_sum)
    return {"status": "ok", "n_points": int(x.size), "parameters": parameters, "r2": float(r2)}


def _predict_quant_form(form: str, parameters: Mapping[str, float], bit: int) -> float:
    if form == "fixed_4^-b":
        return float(parameters["q"] * 4.0 ** (-bit))
    if form == "learned_exponential":
        return float(parameters["q"] * math.exp(-parameters["k"] * bit))
    if form == "quadratic_b":
        return float(parameters["b2"] * bit * bit + parameters["b1"] * bit + parameters["intercept"])
    raise ValueError(f"unknown quantization form {form!r}")


def analyze_quantization(base: Path = QUANT_BASE) -> dict:
    forms = ("fixed_4^-b", "learned_exponential", "quadratic_b")
    all_fits = []
    predictions: dict[str, list[dict]] = {form: [] for form in forms}
    eligible_cells = []
    skipped = []
    models_seen = set()
    if not base.exists():
        return {"input_audit": {"n_models": 0, "skipped": [{"path": str(base), "reason": "input_directory_missing"}]}, "fits": [], "validation": {}}
    for path in sorted(base.glob("*/quant_losses.json")):
        model = canonical_model_tag(path.parent.name)
        family = model_family(model)
        try:
            payload = _json_load(path)
            dense = payload["dense"]
        except (KeyError, OSError, json.JSONDecodeError) as exc:
            skipped.append({"model": model, "reason": "artifact_read_error", "detail": str(exc)})
            continue
        models_seen.add(model)
        for capability in sorted(set(CAPABILITIES) & set(dense)):
            if not all(str(bit) in payload and capability in payload[str(bit)] for bit in (8, 6, 4, 3)):
                skipped.append({"model": model, "capability": capability, "reason": "missing_required_bit"})
                continue
            damage = {bit: float(payload[str(bit)][capability]) - float(dense[capability]) for bit in (8, 6, 4, 3)}
            if not all(np.isfinite(damage[bit]) and damage[bit] > 0.0 for bit in (6, 4, 3)):
                skipped.append({"model": model, "capability": capability, "reason": "nonpositive_damage_at_6_4_or_3", "damages": damage})
                continue
            eligible_cells.append({"model": model, "family": family, "capability": capability, "damages": damage})
            for form in forms:
                fit = _fit_quant_form(form, (8, 6, 4, 3), [damage[bit] for bit in (8, 6, 4, 3)])
                all_fits.append({"model": model, "family": family, "capability": capability, "form": form, **fit})
                for train_bits, heldout_bit in leave_one_bit_out_splits():
                    heldout_fit = _fit_quant_form(form, train_bits, [damage[bit] for bit in train_bits])
                    if heldout_fit["status"] != "ok":
                        continue
                    predicted = _predict_quant_form(form, heldout_fit["parameters"], heldout_bit)
                    predictions[form].append({
                        "form": form,
                        "model": model,
                        "family": family,
                        "capability": capability,
                        "train_bits": train_bits,
                        "heldout_bit": heldout_bit,
                        "observed": damage[heldout_bit],
                        "predicted": predicted,
                        "absolute_error": abs(predicted - damage[heldout_bit]),
                        "fit_parameters": heldout_fit["parameters"],
                    })
    summaries = {form: _summarize_prediction_records(records, f"quant:{form}") for form, records in predictions.items()}
    families = sorted({cell["family"] for cell in eligible_cells})
    winners = {}
    for family in families:
        errors = {form: summaries[form]["by_family"].get(family, {}).get("mae_nats") for form in forms}
        eligible_errors = {form: error for form, error in errors.items() if error is not None}
        winners[family] = {"winner": min(eligible_errors, key=eligible_errors.get) if eligible_errors else None, "mae_nats": errors}
    pooled_errors = {form: summaries[form]["aggregate"]["mae_nats"] for form in forms}
    winners["pooled"] = {"winner": min(pooled_errors, key=pooled_errors.get) if eligible_cells else None, "mae_nats": pooled_errors}
    return {
        "input_audit": {"n_models": len(models_seen), "models": sorted(models_seen), "n_eligible_model_capability_cells": len(eligible_cells), "eligibility": "positive damage at b=6,4,3; b=8 retained as observed near-zero anchor", "skipped": skipped},
        "eligible_cells": eligible_cells,
        "fits": all_fits,
        "validation": {"protocol": "leave b=4 out from {8,6,3}; leave b=3 out from {8,6,4}", "forms": summaries, "winners": winners},
    }


def fit_saturating_recovery(
    tokens: Sequence[int | float],
    damages: Sequence[int | float],
    initial_damage: float,
    *,
    weak_identifiability_tiebreak: bool = False,
) -> dict:
    """Fit the saturating recovery form with bounded deterministic restarts."""
    x = np.asarray(tokens, dtype=np.float64)
    y = np.asarray(damages, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0)
    x, y = x[valid], y[valid]
    if x.size < 2 or not np.isfinite(initial_damage) or initial_damage <= 0.0:
        return {"status": "too_few_points" if x.size < 2 else "nonpositive_initial_damage", "n_points": int(x.size), "residual_fraction": None, "D0": None, "beta": None, "r2": None, "predictions": []}
    order = np.argsort(x)
    x, y = x[order], y[order]
    minimum, maximum = float(x[0]), float(x[-1])
    damage_scale = max(abs(float(initial_damage)), float(np.max(np.abs(y))), 1e-6)
    # These are floating-point safety rails, not scientific parameter bounds:
    # they span twenty-four orders of magnitude in D0 and fourteen in beta.
    log_d0_bounds = (
        math.log(max(minimum * 1e-12, np.finfo(float).tiny)),
        math.log(maximum * 1e12),
    )
    log_beta_bounds = (math.log(1e-8), math.log(1e6))
    regularization = 1e-6 if weak_identifiability_tiebreak else 0.0
    geometric_tokens = math.sqrt(minimum * maximum)

    def residual(theta: np.ndarray) -> np.ndarray:
        r, log_d0, log_beta = map(float, theta)
        prediction = saturating_recovery_form(x, initial_damage, r, math.exp(log_d0), math.exp(log_beta))
        base = (prediction - y) / damage_scale
        if regularization:
            tie = math.sqrt(regularization) * np.array([r - 0.25, log_d0 - math.log(geometric_tokens), log_beta])
            return np.concatenate([base, tie])
        return base

    best = None
    lower = np.array([0.0, log_d0_bounds[0], log_beta_bounds[0]])
    upper = np.array([1.0, log_d0_bounds[1], log_beta_bounds[1]])
    for r0 in (0.0, 0.1, 0.3, 0.6, 0.9):
        for d0_factor in (0.1, 1.0, math.sqrt(maximum / minimum), maximum / minimum, 10.0 * maximum / minimum):
            d00 = float(np.clip(minimum * d0_factor, math.exp(lower[1]), math.exp(upper[1])))
            for beta0 in (0.25, 0.5, 1.0, 2.0, 4.0):
                result = least_squares(residual, np.array([r0, math.log(d00), math.log(beta0)]), bounds=(lower, upper), max_nfev=20_000)
                r, log_d0, log_beta = map(float, result.x)
                prediction = saturating_recovery_form(x, initial_damage, r, math.exp(log_d0), math.exp(log_beta))
                sse = float(np.sum(np.square(y - prediction)))
                objective = float(np.sum(np.square(residual(result.x))))
                candidate = (objective, sse, result.x, prediction)
                if best is None or candidate[0] < best[0]:
                    best = candidate
    if best is None:
        return {"status": "fit_failed", "n_points": int(x.size), "residual_fraction": None, "D0": None, "beta": None, "r2": None, "predictions": []}
    _, residual_sum, parameters, prediction = best
    r, log_d0, log_beta = map(float, parameters)
    total_sum = float(np.sum(np.square(y - np.mean(y))))
    r2 = 1.0 if total_sum <= np.finfo(float).eps and residual_sum <= np.finfo(float).eps else (0.0 if total_sum <= np.finfo(float).eps else 1.0 - residual_sum / total_sum)
    return {
        "status": "ok",
        "n_points": int(x.size),
        "residual_fraction": r,
        "D0": float(math.exp(log_d0)),
        "beta": float(math.exp(log_beta)),
        "r2": float(r2),
        "predictions": [{"tokens": float(token), "damage": float(value)} for token, value in zip(x, prediction)],
        "identifiability": "weak deterministic tiebreak: ridge 1e-6 toward r=0.25, D0=geometric mean budget, beta=1" if weak_identifiability_tiebreak else "three-parameter fit",
        "numerical_safety_bounds": {
            "residual_fraction": [0.0, 1.0],
            "D0": [float(math.exp(log_d0_bounds[0])), float(math.exp(log_d0_bounds[1]))],
            "beta": [float(math.exp(log_beta_bounds[0])), float(math.exp(log_beta_bounds[1]))],
        },
        "at_numerical_bound": bool(
            r <= 1e-7
            or r >= 1.0 - 1e-7
            or log_d0 <= log_d0_bounds[0] + 1e-6
            or log_d0 >= log_d0_bounds[1] - 1e-6
            or log_beta <= log_beta_bounds[0] + 1e-6
            or log_beta >= log_beta_bounds[1] - 1e-6
        ),
    }


def analyze_recovery(base: Path = RECOVERY_BASE) -> dict:
    fits = []
    skipped = []
    models = set()
    paths = sorted(base.glob("*/*/recovery.json")) if base.exists() else []
    for path in paths:
        try:
            payload = _json_load(path)
            dense = payload["anchors"]["dense"]
            damaged = payload["anchors"]["damaged"]
            ladder = payload["ladder"]
        except (KeyError, OSError, json.JSONDecodeError) as exc:
            skipped.append({"path": str(path), "reason": "artifact_read_error", "detail": str(exc)})
            continue
        model = canonical_model_tag(str(payload.get("model", path.parents[1].name)))
        family = model_family(model, str(payload.get("resolved_model", "")))
        run = str(payload.get("run_name", path.parent.name))
        models.add(model)
        for capability in sorted(set(CAPABILITIES) & set(dense) & set(damaged)):
            tokens = [float(row.get("tokens_seen", row["budget_requested"])) for row in ladder if capability in row.get("losses", {})]
            damages = [float(row["losses"][capability]) - float(dense[capability]) for row in ladder if capability in row.get("losses", {})]
            initial_damage = float(damaged[capability]) - float(dense[capability])
            fit = fit_saturating_recovery(tokens, damages, initial_damage)
            extrapolation = {"status": "too_few_smaller_points", "predicted_damage": None, "observed_damage": None, "absolute_error": None}
            if len(tokens) >= 3:
                small_fit = fit_saturating_recovery(tokens[:-1], damages[:-1], initial_damage, weak_identifiability_tiebreak=True)
                if small_fit["status"] == "ok":
                    predicted = float(saturating_recovery_form(tokens[-1], initial_damage, small_fit["residual_fraction"], small_fit["D0"], small_fit["beta"]))
                    carry_forward = float(damages[-2])
                    extrapolation = {
                        "status": "ok_underidentified",
                        "fit": small_fit,
                        "tokens": float(tokens[-1]),
                        "predicted_damage": predicted,
                        "observed_damage": float(damages[-1]),
                        "absolute_error": float(abs(predicted - damages[-1])),
                        "carry_forward_prediction": carry_forward,
                        "carry_forward_absolute_error": float(abs(carry_forward - damages[-1])),
                        "beats_carry_forward": bool(abs(predicted - damages[-1]) < abs(carry_forward - damages[-1])),
                        "warning": "Two nonzero budgets cannot identify r, D0, and beta; the numeric extrapolation uses the documented weak tiebreak and is diagnostic only.",
                    }
            fits.append({
                "model": model,
                "family": family,
                "run": run,
                "capability": capability,
                "initial_damage": initial_damage,
                "tokens": tokens,
                "observed_damages": damages,
                **fit,
                "largest_budget_extrapolation": extrapolation,
                "artifact": str(path.relative_to(ROOT)),
            })
    return {
        "input_audit": {"n_models": len(models), "models": sorted(models), "n_ladders": len(paths), "n_capability_fits": len(fits), "skipped": skipped},
        "form": "DeltaL(D_R) = DeltaL(0) * [r + (1-r) * (1 + D_R/D0)^(-beta)]",
        "r2_scope": "nonzero-budget ladder points only; the exact D_R=0 anchor is not counted in R^2",
        "fits": fits,
    }


def analyze_noise_screen(prune_base: Path = PRUNE_BASE, quant_base: Path = QUANT_BASE) -> dict:
    rows, _, skipped = load_pruning_rows(prune_base)
    mild = [abs(row["damage"]) for row in rows if math.isclose(row["density"], 0.9, abs_tol=1e-9)]
    median_scale = float(np.median(mild)) if mild else None
    threshold = 2.0 * median_scale if median_scale is not None else None
    improvements = []
    for row in rows:
        if row["damage"] < 0.0:
            improvements.append({
                "method": "pruning",
                "model": row["model"],
                "family": row["family"],
                "capability": row["capability"],
                "setting": {"density": row["density"]},
                "damage": row["damage"],
                "absolute_improvement": abs(row["damage"]),
                "exceeds_noise_scale": bool(threshold is not None and abs(row["damage"]) > threshold),
            })
    quant_skipped = []
    quant_models = set()
    if quant_base.exists():
        for path in sorted(quant_base.glob("*/quant_losses.json")):
            model = canonical_model_tag(path.parent.name)
            family = model_family(model)
            try:
                payload = _json_load(path)
                dense = payload["dense"]
            except (KeyError, OSError, json.JSONDecodeError) as exc:
                quant_skipped.append({"model": model, "reason": "artifact_read_error", "detail": str(exc)})
                continue
            quant_models.add(model)
            for key, losses in payload.items():
                if key == "dense":
                    continue
                for capability in sorted(set(dense) & set(losses)):
                    damage = float(losses[capability]) - float(dense[capability])
                    if damage < 0.0:
                        improvements.append({
                            "method": "quantization",
                            "model": model,
                            "family": family,
                            "capability": capability,
                            "setting": {"bits": int(key)},
                            "damage": damage,
                            "absolute_improvement": abs(damage),
                            "exceeds_noise_scale": bool(threshold is not None and abs(damage) > threshold),
                        })
    return {
        "label": "noise-scale screen",
        "is_significance_test": False,
        "method": "The yardstick is 2x the median absolute pruning damage at d=0.9 across all complete V6 model x capability cells. It does not use per-sample uncertainty and is not a significance test.",
        "n_mild_pruning_cells": len(mild),
        "median_abs_damage_at_d_0.9": median_scale,
        "threshold_2x_median": threshold,
        "n_improvement_cells": len(improvements),
        "n_exceeding": sum(item["exceeds_noise_scale"] for item in improvements),
        "improvements": improvements,
        "input_audit": {"pruning_skipped": skipped, "quantization_models": sorted(quant_models), "quantization_skipped": quant_skipped},
    }


def _format_number(value: float | int | None, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _format_metric(metric: Mapping, key: str, ci_key: str) -> str:
    value = metric.get(key)
    interval = metric.get(ci_key)
    if value is None or interval is None:
        return "n/a"
    return f"{value:.4f} [{interval[0]:.4f}, {interval[1]:.4f}]"


def _best_form(forms: Mapping[str, dict]) -> str | None:
    values = {name: result["aggregate"]["mae_nats"] for name, result in forms.items() if result["aggregate"]["mae_nats"] is not None}
    return min(values, key=values.get) if values else None


def build_verdict(summary: Mapping[str, dict]) -> list[dict]:
    verdicts = []
    pruning = summary.get("part1_pruning")
    if pruning:
        for protocol in ("V-a", "V-b", "V-c"):
            comparisons = pruning["validation"][protocol]["paired_comparisons"]
            for comparison_name, claim in (
                (
                    "P3_vs_direct_P2",
                    "shared-gamma first-order + deleted-mass law beats the direct-damage deleted-mass baseline",
                ),
                (
                    "P3_vs_P1",
                    "shared-gamma deleted mass is more portable than per-capability raw sparsity",
                ),
            ):
                comparison = comparisons[comparison_name]
                interval = comparison.get("mae_difference_ci95")
                pass_with_ci = bool(interval is not None and interval[1] < 0.0)
                verdicts.append(
                    {
                        "claim": f"{protocol}: {claim}",
                        "status": "PASS" if pass_with_ci else "FAIL",
                        "numbers": comparison,
                        "decision_rule": "PASS only when the paired cell-bootstrap 95% CI for candidate-minus-baseline MAE is entirely below zero.",
                    }
                )
        sign = pruning["validation"]["V-d"]["aggregate"]
        verdicts.append({"claim": "V-d: signed first-order term predicts the sign of mild-density measured damage above chance", "status": "PASS" if sign["ci95"][0] > 0.5 else "FAIL", "numbers": sign})
    quant = summary.get("part2_quantization")
    if quant:
        pooled = quant["validation"]["winners"].get("pooled", {})
        verdicts.append({"claim": "The fixed q*4^(-b) quantization form wins pooled leave-one-bit-out validation", "status": "PASS" if pooled.get("winner") == "fixed_4^-b" else "FAIL", "numbers": pooled})
    recovery = summary.get("part3_recovery")
    if recovery:
        usable = [fit for fit in recovery["fits"] if fit["largest_budget_extrapolation"].get("status") == "ok_underidentified"]
        wins = sum(fit["largest_budget_extrapolation"]["beats_carry_forward"] for fit in usable)
        verdicts.append({"claim": "Saturating recovery extrapolation beats carrying forward the second budget", "status": "PASS" if usable and wins > len(usable) / 2 else "FAIL", "numbers": {"wins": wins, "n_fits": len(usable)}, "caveat": "The two-point three-parameter extrapolation is underidentified and diagnostic."})
        extrapolation_runs = defaultdict(list)
        for fit in recovery["fits"]:
            extrapolation = fit["largest_budget_extrapolation"]
            if extrapolation.get("status") == "ok_underidentified":
                extrapolation_runs[fit["run"]].append(extrapolation["beats_carry_forward"])
        for run, run_results in sorted(extrapolation_runs.items()):
            run_wins = sum(run_results)
            verdicts.append(
                {
                    "claim": f"Recovery ladder {run}: saturating-form largest-budget prediction beats carry-forward in every capability",
                    "status": "PASS" if run_wins == len(run_results) else "FAIL",
                    "numbers": {"wins": run_wins, "n_capabilities": len(run_results)},
                    "caveat": "The two-point three-parameter extrapolation is underidentified and diagnostic.",
                }
            )
    noise = summary.get("part4_noise_scale_screen")
    if noise:
        verdicts.append({"claim": "Reported negative-damage cells clear the descriptive noise-scale screen", "status": "SCREEN_ONLY", "numbers": {"n_exceeding": noise["n_exceeding"], "n_improvement_cells": noise["n_improvement_cells"], "threshold": noise["threshold_2x_median"]}, "caveat": "This is not a significance test."})
    return verdicts


def build_paper_numbers(summary: Mapping[str, dict], verdicts: Sequence[dict]) -> dict:
    paper: dict[str, object] = {"generated_at_utc": summary["generated_at_utc"], "parts": summary["parts"], "verdicts": list(verdicts)}
    pruning = summary.get("part1_pruning")
    if pruning:
        fits = pruning["in_sample_fits"]
        paper["pruning"] = {
            "n_models": pruning["input_audit"]["n_complete_models"],
            "n_families": len({model["family"] for model in pruning["input_audit"]["models"]}),
            "n_measured_cells": pruning["dataset"]["n_cells"],
            "n_precliff_cells": pruning["dataset"]["n_precliff_cells"],
            "n_fits": len(fits),
            "n_successful_fits": sum(fit["status"] == "ok" for fit in fits),
            "validation": {
                protocol: {
                    "best_form": _best_form(pruning["validation"][protocol]["forms"]),
                    "forms": {
                        form: result["aggregate"]
                        for form, result in pruning["validation"][protocol]["forms"].items()
                    },
                    "paired_comparisons": pruning["validation"][protocol]["paired_comparisons"],
                }
                for protocol in ("V-a", "V-b", "V-c")
            },
            "sign_accuracy": pruning["validation"]["V-d"]["aggregate"],
        }
    quant = summary.get("part2_quantization")
    if quant:
        paper["quantization"] = {
            "n_models": quant["input_audit"]["n_models"],
            "n_eligible_model_capability_cells": quant["input_audit"]["n_eligible_model_capability_cells"],
            "n_fits": len(quant["fits"]),
            "n_heldout_predictions": sum(result["aggregate"]["n_cells"] for result in quant["validation"]["forms"].values()),
            "winners": quant["validation"]["winners"],
        }
    recovery = summary.get("part3_recovery")
    if recovery:
        paper["recovery"] = {
            **{key: recovery["input_audit"][key] for key in ("n_models", "n_ladders", "n_capability_fits")},
            "fits": [{key: fit.get(key) for key in ("model", "run", "capability", "residual_fraction", "beta", "D0", "r2", "largest_budget_extrapolation")} for fit in recovery["fits"]],
        }
    noise = summary.get("part4_noise_scale_screen")
    if noise:
        paper["noise_scale_screen"] = {key: noise[key] for key in ("n_mild_pruning_cells", "median_abs_damage_at_d_0.9", "threshold_2x_median", "n_improvement_cells", "n_exceeding")}
    return paper


def render_report(summary: Mapping[str, dict], verdicts: Sequence[dict]) -> str:
    lines = [
        "# V14 Stage-A refitting report",
        "",
        "This is an artifact-only, CPU refit. No model inference or new measurement was run.",
        "",
        "Held-out pruning errors use cell-bootstrap 95% percentile intervals from 1,000 resamples. MAE is in CE nats; relative error is MAE divided by the mean absolute held-out damage and is dimensionless.",
    ]
    pruning = summary.get("part1_pruning")
    if pruning:
        lines.extend(["", "## Part 1 — pruning law battery", "", f"Complete models: {pruning['input_audit']['n_complete_models']}; measured cells: {pruning['dataset']['n_cells']}; pre-cliff cells: {pruning['dataset']['n_precliff_cells']}.", "", "Deleted mass is normalized by each capability spectrum's total mass and linearly interpolated within the boundary count bin."])
        lines.extend(
            [
                "",
                "### Shared-gamma in-sample parameter audit",
                "",
                "These parameter fits are descriptive; the validation tables below determine claim status.",
                "",
                "| model | family | cells | gamma | log-space R² | B by capability |",
                "|---|---|---:|---:|---:|---|",
            ]
        )
        for fit in pruning["in_sample_fits"]:
            if fit["form"] != "P3_shared_gamma":
                continue
            amplitudes = ", ".join(
                f"{capability}={value:.4g}"
                for capability, value in sorted(fit.get("amplitudes", {}).items())
            ) or "n/a"
            lines.append(
                f"| {fit['model']} | {fit['family']} | {fit['n_points']} | "
                f"{_format_number(fit.get('exponent'))} | "
                f"{_format_number(fit.get('r2_log'))} | {amplitudes} |"
            )
        for protocol in ("V-a", "V-b", "V-c"):
            section = pruning["validation"][protocol]
            lines.extend(["", f"### {protocol}", "", section["description"], "", "| form | cells | MAE nats [95% CI] | relative error [95% CI] |", "|---|---:|---:|---:|"])
            for form, result in section["forms"].items():
                metric = result["aggregate"]
                lines.append(f"| {form} | {metric['n_cells']} | {_format_metric(metric, 'mae_nats', 'mae_ci95')} | {_format_metric(metric, 'relative_error', 'relative_error_ci95')} |")
            lines.extend(
                [
                    "",
                    "Paired comparisons use only cells predicted by both forms.",
                    "",
                    "| comparison | paired cells | candidate MAE | baseline MAE | candidate - baseline MAE [95% CI] |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for name, comparison in section["paired_comparisons"].items():
                interval = comparison.get("mae_difference_ci95")
                difference = comparison.get("mae_difference_candidate_minus_baseline")
                difference_text = (
                    "n/a"
                    if difference is None or interval is None
                    else f"{difference:.4f} [{interval[0]:.4f}, {interval[1]:.4f}]"
                )
                lines.append(
                    f"| {name} | {comparison['n_paired_cells']} | "
                    f"{_format_number(comparison.get('candidate_mae_nats'))} | "
                    f"{_format_number(comparison.get('baseline_mae_nats'))} | "
                    f"{difference_text} |"
                )
        sign = pruning["validation"]["V-d"]
        agg = sign["aggregate"]
        lines.extend(["", "### V-d sign prediction", "", "| group | correct / cells | accuracy [95% Wilson CI] |", "|---|---:|---:|", f"| pooled | {agg['n_correct']} / {agg['n_cells']} | {agg['accuracy']:.4f} [{agg['ci95'][0]:.4f}, {agg['ci95'][1]:.4f}] |"])
        for family, metric in sign["by_family"].items():
            lines.append(f"| {family} | {metric['n_correct']} / {metric['n_cells']} | {metric['accuracy']:.4f} [{metric['ci95'][0]:.4f}, {metric['ci95'][1]:.4f}] |")
    quant = summary.get("part2_quantization")
    if quant:
        lines.extend(["", "## Part 2 — quantization forms", "", f"Eligible positive-damage model/capability cells: {quant['input_audit']['n_eligible_model_capability_cells']} across {quant['input_audit']['n_models']} artifact models. Bit 8 is retained as its observed near-zero anchor, including small negative values.", "", "| family | fixed 4^-b MAE | learned exponential MAE | quadratic MAE | winner |", "|---|---:|---:|---:|---|"])
        for family, result in quant["validation"]["winners"].items():
            errors = result["mae_nats"]
            lines.append(f"| {family} | {_format_number(errors['fixed_4^-b'])} | {_format_number(errors['learned_exponential'])} | {_format_number(errors['quadratic_b'])} | {result['winner'] or 'n/a'} |")
    recovery = summary.get("part3_recovery")
    if recovery:
        lines.extend(["", "## Part 3 — recovery refit", "", "R² is computed only over nonzero-budget points; the D_R=0 damaged anchor is exact by construction.", "", "| model / run / capability | r | beta | D0 tokens | R² | bound? | largest observed | largest predicted from smaller two | abs. error |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for fit in recovery["fits"]:
            extrap = fit["largest_budget_extrapolation"]
            lines.append(f"| {fit['model']} / {fit['run']} / {fit['capability']} | {_format_number(fit.get('residual_fraction'))} | {_format_number(fit.get('beta'))} | {_format_number(fit.get('D0'), 0)} | {_format_number(fit.get('r2'))} | {'yes' if fit.get('at_numerical_bound') else 'no'} | {_format_number(extrap.get('observed_damage'))} | {_format_number(extrap.get('predicted_damage'))} | {_format_number(extrap.get('absolute_error'))} |")
        lines.extend(["", "Caveat: fitting r, D0, and beta to only two nonzero budgets is underidentified even with the exact zero-budget anchor. The numeric extrapolation uses a weak, documented deterministic tiebreak and is a diagnostic check, not an identified parameter estimate. Boundary fits and extremely large beta values indicate that the monotone form is not identifying a stable saturation timescale."])
    noise = summary.get("part4_noise_scale_screen")
    if noise:
        lines.extend(["", "## Part 4 — noise-scale screen", "", "This is **not a significance test**. Per-sample losses were not stored, so the requested honest yardstick is used instead of a fabricated bootstrap.", "", f"Median |Delta L| at pruning d=0.9: {_format_number(noise['median_abs_damage_at_d_0.9'], 6)} nats; screen threshold (2x median): {_format_number(noise['threshold_2x_median'], 6)} nats. {noise['n_exceeding']} of {noise['n_improvement_cells']} negative-damage cells exceed it.", "", "| method | model | capability | setting | Delta L | clears screen |", "|---|---|---|---|---:|---:|"])
        for item in noise["improvements"]:
            setting = ", ".join(f"{key}={value}" for key, value in item["setting"].items())
            lines.append(f"| {item['method']} | {item['model']} | {item['capability']} | {setting} | {item['damage']:+.6f} | {'yes' if item['exceeds_noise_scale'] else 'no'} |")
    lines.extend(["", "## VERDICT", "", "For pruning comparisons, PASS requires the paired 95% cell-bootstrap interval for candidate-minus-baseline MAE to lie fully below zero. Sign prediction requires a Wilson lower bound above 0.5. Quantization uses the pooled held-out winner. A recovery run passes only if its diagnostic extrapolation beats carry-forward in every capability; the underidentification caveat still applies.", "", "| status | claim | exact comparison |", "|---|---|---|"])
    for verdict in verdicts:
        lines.append(f"| {verdict['status']} | {verdict['claim']} | `{json.dumps(verdict['numbers'], sort_keys=True)}` |")
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(parts: Sequence[int], out: Path = OUT_BASE) -> tuple[dict, dict, str]:
    selected = parse_parts(parts)
    summary: dict[str, object] = {
        "version": 14,
        "stage": "A",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parts": selected,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }
    if 1 in selected:
        summary["part1_pruning"] = analyze_pruning()
    if 2 in selected:
        summary["part2_quantization"] = analyze_quantization()
    if 3 in selected:
        summary["part3_recovery"] = analyze_recovery()
    if 4 in selected:
        summary["part4_noise_scale_screen"] = analyze_noise_screen()
    verdicts = build_verdict(summary)
    summary["verdicts"] = verdicts
    paper_numbers = build_paper_numbers(summary, verdicts)
    report = render_report(summary, verdicts)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "summary.json", summary)
    _write_json(out / "paper_numbers.json", paper_numbers)
    temporary_report = out / "report.md.tmp"
    temporary_report.write_text(report, encoding="utf-8")
    temporary_report.replace(out / "report.md")
    return summary, paper_numbers, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=parse_parts, default=[1, 2, 3, 4], help="comma-separated subset of 1,2,3,4 (default: all)")
    args = parser.parse_args()
    summary, _, _ = run(args.parts)
    print(f"wrote {OUT_BASE / 'summary.json'}")
    print(f"parts: {','.join(map(str, summary['parts']))}")


if __name__ == "__main__":
    main()
