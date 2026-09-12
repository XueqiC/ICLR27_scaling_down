#!/usr/bin/env python3
"""Unified held-out fitting for the four scaling-down-law arms.

The report produced by this module is deliberately prediction-first.  Every
headline number is calculated from a row that was absent from its fit.  Full
data fits are retained only as ``NON-DECISIONAL`` diagnostics.  The module is
CPU-only and reads existing JSON/NPZ artifacts; it never loads a model or
modifies ``results/``.

All damage-law fits use the V17 perturbative rule: within a compression
trajectory, the first point with ``Delta L > 1 nat`` and every deeper point is
post-cliff.  Recovery observations have no compression-coordinate trajectory,
so the same numerical cap is applied directly to each observed damage.

Usage::

    python analysis/v18_law_fit.py --dry-run
    python analysis/v18_law_fit.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Callable, Mapping, Sequence

# Tiny regressions are dramatically slower when BLAS starts one worker per
# host core.  Keep this explicitly CPU-only and single-threaded.
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

try:
    from . import v17_unification as v17
    from .v14_fitting import deleted_mass_fraction
except ImportError:  # direct execution
    import v17_unification as v17
    from v14_fitting import deleted_mass_fraction


ROOT = Path(__file__).resolve().parents[1]
PRUNE_BASE = ROOT / "results/v6-capability-geometry"
QUANT_BASE = ROOT / "results/v10-quantization"
DISTILL_BASE = ROOT / "results/v16-style-residual"
RECOVERY_BASE = ROOT / "results/v13-recovery/gemma3-1b"
REPORT_PATH = ROOT / "paper/docs/LAW_FIT_REPORT.md"

CAPABILITIES = ("math", "code", "qa")
PRECLIFF_CAP = 1.0
SHALLOW_DENSITY = 0.55
BOOTSTRAP_RESAMPLES = 1_000
EPS = np.finfo(np.float64).eps

PRUNING_CANDIDATES = (
    "density_only",
    "source_size_density",
    "dense_anchor_density",
    "hierarchical_capability_family",
    "mechanism_upper_bound",
    "smooth_cliff_extension",
    "baseline_raw_sparsity",
    "baseline_remaining_params",
    "baseline_removed_weight_norm",
    "baseline_anchor_only",
)
QUANTIZATION_CANDIDATES = (
    "fixed_4^-b",
    "learned_exponential",
    "effective_parameter_multiplier",
    "family_conditioned_smooth_cliff",
    "bit_width_categorical_baseline",
)
RECOVERY_CANDIDATES = (
    "monotonic_saturation",
    "change_point",
    "early_recovery_late_penalty",
)

PRUNING_FORMS = {
    "density_only": "A(1-d)^gamma",
    "source_size_density": "[A+B log(N0)](1-d)^gamma",
    "dense_anchor_density": "[A+B L_c0](1-d)^gamma",
    "hierarchical_capability_family":
        "[A+u_c+v_f](1-d)^gamma (unseen effects=0)",
    "mechanism_upper_bound":
        "g·delta_w + B m_c(d)^gamma (measured-mechanism upper bound)",
    "smooth_cliff_extension":
        "A(1-d)^gamma+C{sigmoid[(d*-d)/w]-sigmoid[(d*-1)/w]}",
    "baseline_raw_sparsity": "A(1-d)",
    "baseline_remaining_params":
        "A{(N0 d)^(-alpha)-N0^(-alpha)}",
    "baseline_removed_weight_norm": "A times binned removed-weight L2 fraction",
    "baseline_anchor_only": "Delta L=0 (dense anchor)",
}
QUANTIZATION_FORMS = {
    "fixed_4^-b": "q[4^(4-b)-4^(4-16)]",
    "learned_exponential": "q{exp[k(4-b)]-exp[k(4-16)]}",
    "effective_parameter_multiplier":
        "A{(N0 b/16)^(-alpha)-N0^(-alpha)}",
    "family_conditioned_smooth_cliff":
        "smooth exponential + family-conditioned sigmoid bit cliff",
    "bit_width_categorical_baseline": "mean Delta L by (capability, bit)",
}
RECOVERY_FORMS = {
    "monotonic_saturation": "r_c+(1-r_c)(1+D_R/D0)^(-beta)",
    "change_point": "1+a log(1+D_R/1M)+c[log(1+D_R/1M)-tau]_+",
    "early_recovery_late_penalty":
        "(1+D_R/D0)^(-beta)+kappa(D_R/1M)^p",
}


def _stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def paper_family(family: str) -> str:
    """Return the directive's four-family grouping."""
    value = str(family)
    if value in {"gemma3", "gemma4"}:
        return "gemma"
    if value in {"muse", "muse_glimmer"}:
        return "muse_glimmer"
    return value


def _effect_design(levels: Sequence[str], values: Sequence[str]) -> np.ndarray:
    """Zero-sum coding; a level absent from training gets zero correction."""
    levels = list(levels)
    if len(levels) <= 1:
        return np.zeros((len(values), 0), dtype=np.float64)
    columns = {level: index for index, level in enumerate(levels[:-1])}
    out = np.zeros((len(values), len(levels) - 1), dtype=np.float64)
    for i, value in enumerate(values):
        if value in columns:
            out[i, columns[value]] = 1.0
        elif value == levels[-1]:
            out[i, :] = -1.0
    return out


def _standardizer(values: Sequence[float]) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    center = float(np.mean(x))
    scale = float(np.std(x))
    return center, max(scale, 1e-12)


def split_shallow_to_deep(
    rows: Sequence[Mapping[str, object]], cutoff: float = SHALLOW_DENSITY
) -> list[dict[str, object]]:
    """Fit at densities >= cutoff and test deeper pre-cliff densities."""
    train = [i for i, row in enumerate(rows) if float(row["density"]) >= cutoff]
    test = [i for i, row in enumerate(rows) if float(row["density"]) < cutoff]
    if not train or not test:
        return []
    return [{
        "held_out": f"density<{cutoff:g}",
        "train_indices": train,
        "test_indices": test,
    }]


def leave_one_family_out_splits(
    rows: Sequence[Mapping[str, object]], family_key: str = "family"
) -> list[dict[str, object]]:
    """Hold out every row from one whole family."""
    if any(family_key not in row for row in rows):
        raise KeyError(family_key)
    output = []
    for family in sorted({str(row[family_key]) for row in rows}):
        train = [i for i, row in enumerate(rows) if str(row[family_key]) != family]
        test = [i for i, row in enumerate(rows) if str(row[family_key]) == family]
        if train and test:
            output.append({
                "held_out": family,
                "train_indices": train,
                "test_indices": test,
            })
    return output


def leave_largest_model_out_splits(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Within every family, hold out all rows of its largest model."""
    sizes: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        family, model = str(row["family"]), str(row["model"])
        size = float(row["N0"])
        old = sizes[family].get(model)
        if old is not None and not math.isclose(old, size):
            raise ValueError(f"inconsistent size for {model}")
        sizes[family][model] = size
    output = []
    for family, model_sizes in sorted(sizes.items()):
        if len(model_sizes) < 2:
            continue
        largest = max(model_sizes, key=lambda model: (model_sizes[model], model))
        train = [i for i, row in enumerate(rows) if str(row["model"]) != largest]
        test = [i for i, row in enumerate(rows) if str(row["model"]) == largest]
        if train and test:
            output.append({
                "held_out": largest,
                "family": family,
                "train_indices": train,
                "test_indices": test,
            })
    return output


def leave_one_bit_out_splits(
    rows_or_bits: Sequence[Mapping[str, object]] | Sequence[int] = (8, 6, 4, 3),
) -> list[dict[str, object]] | list[tuple[list[int], int]]:
    """Hold out one bit globally; numeric input keeps the V14 public API."""
    if not rows_or_bits:
        return []
    first = rows_or_bits[0]  # type: ignore[index]
    if not isinstance(first, Mapping):
        bits = list(dict.fromkeys(int(bit) for bit in rows_or_bits))  # type: ignore[arg-type]
        return [([other for other in bits if other != bit], bit) for bit in bits]
    rows = rows_or_bits  # type: ignore[assignment]
    output = []
    for bit in sorted({int(round(float(row["bits"]))) for row in rows}, reverse=True):
        train = [i for i, row in enumerate(rows) if int(round(float(row["bits"]))) != bit]
        test = [i for i, row in enumerate(rows) if int(round(float(row["bits"]))) == bit]
        if train and test:
            output.append({
                "held_out": f"int{bit}",
                "train_indices": train,
                "test_indices": test,
            })
    return output


def leave_one_student_size_out_splits(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output = []
    for model in sorted({str(row["model"]) for row in rows}):
        train = [i for i, row in enumerate(rows) if str(row["model"]) != model]
        test = [i for i, row in enumerate(rows) if str(row["model"]) == model]
        if train and test:
            output.append({
                "held_out": model,
                "train_indices": train,
                "test_indices": test,
            })
    return output


def leave_one_budget_out_splits(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output = []
    for budget in sorted({float(row["tokens"]) for row in rows}):
        train = [i for i, row in enumerate(rows) if float(row["tokens"]) != budget]
        test = [i for i, row in enumerate(rows) if float(row["tokens"]) == budget]
        if len(train) >= 2 and test:
            output.append({
                "held_out": f"{budget / 1e6:g}M",
                "train_indices": train,
                "test_indices": test,
            })
    return output


def _fraction_from_histogram(
    counts: np.ndarray, values: np.ndarray, density: float
) -> float:
    """Fraction of a binned nonnegative total removed by magnitude pruning."""
    counts = np.asarray(counts, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    total_count, total = float(np.sum(counts)), float(np.sum(values))
    if total_count <= 0 or total <= 0:
        return 0.0
    target = np.clip((1.0 - density) * total_count, 0.0, total_count)
    cumulative = np.cumsum(counts)
    index = min(int(np.searchsorted(cumulative, target, side="left")), len(counts) - 1)
    before_count = float(cumulative[index - 1]) if index else 0.0
    before_value = float(np.sum(values[:index]))
    fraction = 0.0 if counts[index] == 0 else (target - before_count) / counts[index]
    return float(np.clip((before_value + fraction * values[index]) / total, 0.0, 1.0))


def _removed_weight_l2_fraction(spectrum: Mapping[str, np.ndarray], density: float) -> float:
    """Approximate removed L2 fraction from V6's committed |w| bins."""
    edges = np.asarray(spectrum["edges"], dtype=np.float64)
    counts = np.asarray(spectrum["counts"], dtype=np.float64)
    if not len(counts):
        return 0.0
    if not len(edges):
        representatives = np.ones_like(counts)
    else:
        interior = (edges[:-1] + edges[1:]) / 2.0
        first = max(0.0, edges[0] / 2.0)
        last = edges[-1] + max(edges[-1] - edges[-2], edges[-1] * 0.05) if len(edges) > 1 else edges[-1] * 1.5
        representatives = np.concatenate([[first], interior, [last]])
    squared_mass = counts * np.square(representatives)
    removed_squared = _fraction_from_histogram(counts, squared_mass, density)
    return float(math.sqrt(max(removed_squared, 0.0)))


def _load_method_rows() -> tuple[list[dict], list[dict], list[dict], dict]:
    assembled, audit = v17.assemble_table(PRUNE_BASE, QUANT_BASE, DISTILL_BASE)
    annotated, filter_audit = v17.apply_precliff_filter(assembled, PRECLIFF_CAP)
    prune: list[dict] = []
    quant: list[dict] = []
    distill: list[dict] = []
    mechanism_cache: dict[str, tuple[dict, dict[str, np.ndarray]]] = {}

    for row in annotated:
        if bool(row.get("is_baseline")):
            continue
        method = str(row["method"])
        common = {
            **dict(row),
            "family": paper_family(str(row["family"])),
            "L_c0": float(row["reference_L_c"]),
            "N0": float(row["n0_params"]),
            "observed": float(row["delta_L_c"]),
        }
        if method == "pruning":
            model = str(row["model"])
            if model not in mechanism_cache:
                model_dir = PRUNE_BASE / model
                alignment = _json(model_dir / "alignment.json")
                with np.load(model_dir / "spectrum_bins.npz") as loaded:
                    spectrum = {key: loaded[key].copy() for key in loaded.files}
                mechanism_cache[model] = alignment, spectrum
            alignment, spectrum = mechanism_cache[model]
            density = float(row["raw_coordinate"])
            capability = str(row["capability"])
            cell = alignment.get(capability, {}).get(str(density), {})
            if not cell:
                # JSON may spell an exact float differently.
                candidates = alignment.get(capability, {})
                match = next((value for key, value in candidates.items()
                              if math.isclose(float(key), density, abs_tol=1e-12)), None)
                cell = match or {}
            mass_key = f"mass_{capability}"
            common.update({
                "density": density,
                "sparsity": 1.0 - density,
                "first_order": float(cell["first_order"]) if "first_order" in cell else math.nan,
                "deleted_mass": deleted_mass_fraction(
                    spectrum["counts"], spectrum[mass_key], density
                ),
                "removed_weight_norm": _removed_weight_l2_fraction(spectrum, density),
            })
            prune.append(common)
        elif method == "quantization":
            common["bits"] = float(row["raw_coordinate"])
            quant.append(common)
        elif method == v17.DISTILL_SOURCE:
            common["r_storage"] = float(row["r_storage"])
            distill.append(common)

    audit = {**audit, "precliff": filter_audit}
    return prune, quant, distill, audit


def _profile_power_fit(
    x: np.ndarray,
    y: np.ndarray,
    covariates: np.ndarray,
    *,
    offset: np.ndarray | None = None,
    gamma_bounds: tuple[float, float] = (0.05, 8.0),
) -> tuple[float, np.ndarray]:
    offset = np.zeros_like(y) if offset is None else offset

    def solve(gamma: float) -> tuple[np.ndarray, float]:
        design = np.power(np.clip(x, 0.0, None), gamma)[:, None] * covariates
        ridge = 1e-8 * np.eye(design.shape[1])
        ridge[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + ridge, design.T @ (y - offset))
        residual = offset + design @ coefficients - y
        return coefficients, float(np.dot(residual, residual))

    optimum = minimize_scalar(lambda gamma: solve(float(gamma))[1],
                              bounds=gamma_bounds, method="bounded")
    gamma = float(optimum.x)
    coefficients, _ = solve(gamma)
    return gamma, coefficients


def fit_pruning_candidate(
    rows: Sequence[Mapping[str, object]], candidate: str
) -> dict[str, object]:
    """Fit one declared pruning candidate on supplied training rows."""
    if candidate not in PRUNING_CANDIDATES:
        raise ValueError(f"unknown pruning candidate {candidate!r}")
    if not rows:
        return {"status": "too_few_points", "candidate": candidate, "n_params": 0}
    y = np.asarray([row.get("observed", row.get("delta_L_c")) for row in rows], dtype=float)
    d = np.asarray([row.get("density", row.get("raw_coordinate")) for row in rows], dtype=float)
    s = 1.0 - d
    fit: dict[str, object] = {"candidate": candidate}
    if candidate == "baseline_anchor_only":
        return {**fit, "status": "ok", "n_params": 0, "parameters": {}}

    capabilities = sorted({str(row.get("capability", "all")) for row in rows})
    families = sorted({str(row.get("family", "all")) for row in rows})
    cap_design = _effect_design(capabilities, [str(row.get("capability", "all")) for row in rows])
    family_design = _effect_design(families, [str(row.get("family", "all")) for row in rows])
    metadata: dict[str, object] = {"capabilities": capabilities, "families": families}

    if candidate in {"density_only", "baseline_raw_sparsity", "baseline_removed_weight_norm"}:
        if candidate == "baseline_removed_weight_norm":
            x = np.asarray([row.get("removed_weight_norm", math.nan) for row in rows], dtype=float)
            if not np.all(np.isfinite(x)):
                return {**fit, "status": "missing_mechanism_input", "n_params": 1}
        else:
            x = s
        gamma_bounds = (1.0, 1.0 + 1e-9) if candidate != "density_only" else (0.05, 8.0)
        gamma, coefficients = _profile_power_fit(x, y, np.ones((len(rows), 1)), gamma_bounds=gamma_bounds)
        return {**fit, "status": "ok", "n_params": 1 if candidate != "density_only" else 2,
                "parameters": {"gamma": gamma, "coefficients": coefficients.tolist()}, **metadata}

    if candidate == "source_size_density":
        log_n = np.log(np.asarray([float(row["N0"]) for row in rows]) / 1e9)
        center, scale = _standardizer(log_n)
        covariates = np.column_stack([np.ones(len(rows)), (log_n - center) / scale])
        gamma, coefficients = _profile_power_fit(s, y, covariates)
        return {**fit, "status": "ok", "n_params": 3,
                "parameters": {"gamma": gamma, "coefficients": coefficients.tolist(),
                               "center": center, "scale": scale}, **metadata}

    if candidate == "dense_anchor_density":
        anchors = np.asarray([float(row["L_c0"]) for row in rows])
        center, scale = _standardizer(anchors)
        covariates = np.column_stack([np.ones(len(rows)), (anchors - center) / scale])
        gamma, coefficients = _profile_power_fit(s, y, covariates)
        return {**fit, "status": "ok", "n_params": 3,
                "parameters": {"gamma": gamma, "coefficients": coefficients.tolist(),
                               "center": center, "scale": scale}, **metadata}

    if candidate == "hierarchical_capability_family":
        covariates = np.column_stack([np.ones(len(rows)), cap_design, family_design])
        gamma, coefficients = _profile_power_fit(s, y, covariates)
        return {**fit, "status": "ok", "n_params": int(1 + len(coefficients)),
                "parameters": {"gamma": gamma, "coefficients": coefficients.tolist()}, **metadata}

    if candidate == "mechanism_upper_bound":
        mass = np.asarray([row.get("deleted_mass", math.nan) for row in rows], dtype=float)
        first = np.asarray([row.get("first_order", math.nan) for row in rows], dtype=float)
        if not np.all(np.isfinite(mass)) or not np.all(np.isfinite(first)):
            return {**fit, "status": "missing_mechanism_input", "n_params": 0}
        # This is exactly the V14 mechanism form.  Do not quietly promote its
        # measured mechanism inputs into a more flexible reduced form.
        covariates = np.ones((len(rows), 1))
        gamma, coefficients = _profile_power_fit(mass, y, covariates, offset=first)
        return {**fit, "status": "ok", "n_params": 2,
                "parameters": {"gamma": gamma, "coefficients": coefficients.tolist()}, **metadata}

    if candidate == "baseline_remaining_params":
        n = np.asarray([float(row["N0"]) for row in rows]) / 1e9

        def prediction(theta: np.ndarray) -> np.ndarray:
            amplitude, alpha = theta
            return amplitude * (np.power(np.clip(n * d, 1e-12, None), -alpha)
                                - np.power(np.clip(n, 1e-12, None), -alpha))

        scale_y = max(float(np.std(y)), float(np.mean(np.abs(y))), 1e-6)
        result = least_squares(lambda theta: (prediction(theta) - y) / scale_y,
                               np.array([scale_y, 0.5]),
                               bounds=(np.array([-1e5 * scale_y, 0.01]),
                                       np.array([1e5 * scale_y, 5.0])))
        return {**fit, "status": "ok" if result.success else "fit_failed", "n_params": 2,
                "parameters": {"amplitude": float(result.x[0]), "alpha": float(result.x[1])}, **metadata}

    # Smooth-cliff fit uses only the passed rows (which callers pre-filter).
    def sigmoid(value: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(value, -60, 60)))

    scale_y = max(float(np.std(y)), float(np.mean(np.abs(y))), 1e-6)

    def prediction(theta: np.ndarray) -> np.ndarray:
        amplitude, gamma, cliff_amplitude, d_star, width = theta
        cliff = sigmoid((d_star - d) / width) - sigmoid((d_star - 1.0) / width)
        return amplitude * np.power(np.clip(s, 0, None), gamma) + cliff_amplitude * cliff

    result = least_squares(
        lambda theta: (prediction(theta) - y) / scale_y,
        np.array([scale_y, 1.5, scale_y, 0.5, 0.08]),
        bounds=(np.array([-1e4 * scale_y, 0.05, -1e4 * scale_y, 0.05, 0.01]),
                np.array([1e4 * scale_y, 8.0, 1e4 * scale_y, 0.95, 0.5])),
        max_nfev=30_000,
    )
    return {**fit, "status": "ok" if result.success else "fit_failed", "n_params": 5,
            "parameters": {name: float(value) for name, value in zip(
                ("amplitude", "gamma", "cliff_amplitude", "d_star", "width"), result.x)}, **metadata}


def predict_pruning_candidate(
    fit: Mapping[str, object], row: Mapping[str, object]
) -> float | None:
    if fit.get("status") != "ok":
        return None
    candidate = str(fit["candidate"])
    p = fit.get("parameters", {})
    assert isinstance(p, Mapping)
    d = float(row.get("density", row.get("raw_coordinate")))
    s = 1.0 - d
    if candidate == "baseline_anchor_only":
        return 0.0
    if candidate in {"density_only", "baseline_raw_sparsity", "baseline_removed_weight_norm",
                     "source_size_density", "dense_anchor_density",
                     "hierarchical_capability_family", "mechanism_upper_bound"}:
        coefficients = np.asarray(p["coefficients"], dtype=float)
        if candidate == "source_size_density":
            value = (math.log(float(row["N0"]) / 1e9) - float(p["center"])) / float(p["scale"])
            covariates = np.array([1.0, value])
            x, offset = s, 0.0
        elif candidate == "dense_anchor_density":
            value = (float(row["L_c0"]) - float(p["center"])) / float(p["scale"])
            covariates = np.array([1.0, value])
            x, offset = s, 0.0
        elif candidate == "hierarchical_capability_family":
            capabilities = list(fit["capabilities"])
            families = list(fit["families"])
            cap = _effect_design(capabilities, [str(row.get("capability", "all"))])[0]
            fam = _effect_design(families, [str(row.get("family", "all"))])[0]
            covariates = np.concatenate([[1.0], cap, fam])
            x, offset = s, 0.0
        elif candidate == "mechanism_upper_bound":
            covariates = np.array([1.0])
            x = float(row.get("deleted_mass", math.nan))
            offset = float(row.get("first_order", math.nan))
            if not math.isfinite(x) or not math.isfinite(offset):
                return None
        else:
            covariates = np.array([1.0])
            x = float(row.get("removed_weight_norm")) if candidate == "baseline_removed_weight_norm" else s
            offset = 0.0
        return float(offset + x ** float(p["gamma"]) * np.dot(covariates, coefficients))
    if candidate == "baseline_remaining_params":
        n = float(row["N0"]) / 1e9
        alpha = float(p["alpha"])
        return float(p["amplitude"]) * ((n * d) ** (-alpha) - n ** (-alpha))
    if candidate == "smooth_cliff_extension":
        sigmoid = lambda value: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))
        cliff = sigmoid((float(p["d_star"]) - d) / float(p["width"])) - sigmoid(
            (float(p["d_star"]) - 1.0) / float(p["width"]))
        return float(p["amplitude"]) * s ** float(p["gamma"]) + float(p["cliff_amplitude"]) * cliff
    return None


def fit_quantization_candidate(
    rows: Sequence[Mapping[str, object]], candidate: str
) -> dict[str, object]:
    """Fit one declared quantization candidate on pre-cliff training rows."""
    if candidate not in QUANTIZATION_CANDIDATES:
        raise ValueError(f"unknown quantization candidate {candidate!r}")
    if not rows:
        return {"status": "too_few_points", "candidate": candidate, "n_params": 0}
    b = np.asarray([float(row.get("bits", row.get("raw_coordinate"))) for row in rows])
    y = np.asarray([row.get("observed", row.get("delta_L_c")) for row in rows], dtype=float)
    fit: dict[str, object] = {"candidate": candidate}
    if candidate == "bit_width_categorical_baseline":
        groups: dict[str, list[float]] = defaultdict(list)
        for row, value in zip(rows, y):
            key = f"{row.get('capability', 'all')}|{int(round(float(row.get('bits', row.get('raw_coordinate')))))}"
            groups[key].append(float(value))
        means = {key: float(np.mean(values)) for key, values in groups.items()}
        return {**fit, "status": "ok", "n_params": len(means), "parameters": {"means": means}}

    scale_y = max(float(np.std(y)), float(np.mean(np.abs(y))), 1e-6)
    if candidate == "fixed_4^-b":
        basis = np.power(4.0, 4.0 - b) - np.power(4.0, 4.0 - 16.0)
        q = float(np.dot(basis, y) / max(float(np.dot(basis, basis)), EPS))
        return {**fit, "status": "ok", "n_params": 1, "parameters": {"q": q}}

    if candidate == "learned_exponential":
        def pred(theta: np.ndarray) -> np.ndarray:
            q, k = theta
            return q * (np.exp(np.clip(k * (4.0 - b), -50, 50))
                        - math.exp(k * (4.0 - 16.0)))
        result = least_squares(lambda theta: (pred(theta) - y) / scale_y,
                               np.array([scale_y, math.log(4.0)]),
                               bounds=(np.array([-1e5 * scale_y, 0.01]),
                                       np.array([1e5 * scale_y, 5.0])))
        return {**fit, "status": "ok" if result.success else "fit_failed", "n_params": 2,
                "parameters": {"q": float(result.x[0]), "k": float(result.x[1])}}

    if candidate == "effective_parameter_multiplier":
        n = np.asarray([float(row["N0"]) for row in rows]) / 1e9
        def pred(theta: np.ndarray) -> np.ndarray:
            amplitude, alpha = theta
            return amplitude * (np.power(np.clip(n * b / 16.0, 1e-12, None), -alpha)
                                - np.power(np.clip(n, 1e-12, None), -alpha))
        result = least_squares(lambda theta: (pred(theta) - y) / scale_y,
                               np.array([scale_y, 0.5]),
                               bounds=(np.array([-1e5 * scale_y, 0.01]),
                                       np.array([1e5 * scale_y, 5.0])))
        return {**fit, "status": "ok" if result.success else "fit_failed", "n_params": 2,
                "parameters": {"amplitude": float(result.x[0]), "alpha": float(result.x[1])}}

    families = sorted({str(row.get("family", "all")) for row in rows})
    family_design = _effect_design(families, [str(row.get("family", "all")) for row in rows])
    n_effects = family_design.shape[1]
    def sigmoid(value: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(value, -60, 60)))
    def pred(theta: np.ndarray) -> np.ndarray:
        q, k, cliff_amplitude, b_star, width = theta[:5]
        family_b_star = b_star + (family_design @ theta[5:] if n_effects else 0.0)
        smooth = q * (np.exp(np.clip(k * (4.0 - b), -50, 50)) - math.exp(k * (4.0 - 16.0)))
        cliff = sigmoid((family_b_star - b) / width) - sigmoid((family_b_star - 16.0) / width)
        return smooth + cliff_amplitude * cliff
    x0 = np.concatenate([[scale_y, 1.0, scale_y, 3.5, 0.3], np.zeros(n_effects)])
    lower = np.concatenate([[-1e5 * scale_y, 0.01, -1e5 * scale_y, 2.0, 0.05],
                            np.full(n_effects, -1e5 * scale_y)])
    upper = np.concatenate([[1e5 * scale_y, 5.0, 1e5 * scale_y, 8.0, 2.0],
                            np.full(n_effects, 1e5 * scale_y)])
    result = least_squares(lambda theta: (pred(theta) - y) / scale_y, x0,
                           bounds=(lower, upper), max_nfev=30_000)
    return {**fit, "status": "ok" if result.success else "fit_failed", "n_params": int(5 + n_effects),
            "parameters": {"core": result.x[:5].tolist(), "family_effects": result.x[5:].tolist()},
            "families": families}


def predict_quantization_candidate(
    fit: Mapping[str, object], row: Mapping[str, object]
) -> float | None:
    if fit.get("status") != "ok":
        return None
    candidate = str(fit["candidate"])
    p = fit["parameters"]
    assert isinstance(p, Mapping)
    b = float(row.get("bits", row.get("raw_coordinate")))
    if candidate == "bit_width_categorical_baseline":
        key = f"{row.get('capability', 'all')}|{int(round(b))}"
        return p["means"].get(key)  # type: ignore[index,union-attr]
    if candidate == "fixed_4^-b":
        return float(p["q"]) * (4.0 ** (4.0 - b) - 4.0 ** (4.0 - 16.0))
    if candidate == "learned_exponential":
        return float(p["q"]) * (math.exp(float(p["k"]) * (4.0 - b))
                                - math.exp(float(p["k"]) * (4.0 - 16.0)))
    if candidate == "effective_parameter_multiplier":
        n = float(row["N0"]) / 1e9
        alpha = float(p["alpha"])
        return float(p["amplitude"]) * ((n * b / 16.0) ** (-alpha) - n ** (-alpha))
    core = np.asarray(p["core"], dtype=float)
    q, k, cliff_amplitude, b_star, width = core
    families = list(fit["families"])
    effect = _effect_design(families, [str(row.get("family", "all"))])[0]
    family_effects = np.asarray(p["family_effects"], dtype=float)
    family_b_star = b_star + float(effect @ family_effects)
    sigmoid = lambda value: 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))
    smooth = q * (math.exp(k * (4.0 - b)) - math.exp(k * (4.0 - 16.0)))
    cliff = sigmoid((family_b_star - b) / width) - sigmoid((family_b_star - 16.0) / width)
    return float(smooth + cliff_amplitude * cliff)


def fit_distillation_law(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Fit floor_c - alpha_c log(r_storage), separately by capability."""
    fits: dict[str, dict[str, float]] = {}
    for capability in sorted({str(row["capability"]) for row in rows}):
        subset = [row for row in rows if str(row["capability"]) == capability]
        x = np.asarray([math.log(float(row["r_storage"])) for row in subset])
        y = np.asarray([float(row.get("observed", row.get("delta_L_c"))) for row in subset])
        if len(np.unique(np.round(x, 14))) < 2:
            continue
        design = np.column_stack([np.ones(len(x)), x])
        floor, slope = np.linalg.lstsq(design, y, rcond=None)[0]
        fits[capability] = {"floor": float(floor), "alpha": float(-slope)}
    return {
        "status": "ok" if len(fits) == len({str(row["capability"]) for row in rows}) else "partial",
        "candidate": "capacity_floor_log_storage",
        "n_params": 2 * len(fits),
        "parameters": fits,
    }


def predict_distillation_law(
    fit: Mapping[str, object], row: Mapping[str, object]
) -> float | None:
    parameters = fit.get("parameters", {})
    assert isinstance(parameters, Mapping)
    capability = str(row["capability"])
    if capability not in parameters:
        return None
    p = parameters[capability]
    assert isinstance(p, Mapping)
    return float(p["floor"]) - float(p["alpha"]) * math.log(float(row["r_storage"]))


def recovery_curve(
    candidate: str, tokens: Sequence[float] | np.ndarray, parameters: Mapping[str, float]
) -> np.ndarray:
    """Evaluate a normalized recovery candidate (value is 1 at D_R=0)."""
    d = np.asarray(tokens, dtype=float)
    if candidate == "monotonic_saturation":
        r, d0, beta = parameters["r"], parameters["D0"], parameters["beta"]
        return r + (1.0 - r) * np.power(1.0 + d / d0, -beta)
    x = np.log1p(d / 1e6)
    if candidate == "change_point":
        return 1.0 + parameters["a"] * x + parameters["c"] * np.maximum(x - parameters["tau"], 0.0)
    if candidate == "early_recovery_late_penalty":
        return (np.power(1.0 + d / parameters["D0"], -parameters["beta"])
                + parameters["kappa"] * np.power(d / 1e6, parameters["p"]))
    raise ValueError(f"unknown recovery candidate {candidate!r}")


def fit_recovery_candidate(
    rows: Sequence[Mapping[str, object]], candidate: str
) -> dict[str, object]:
    """Fit one recovery form; return numeric fits but flag n<=k explicitly."""
    if candidate not in RECOVERY_CANDIDATES:
        raise ValueError(f"unknown recovery candidate {candidate!r}")
    if not rows:
        return {"status": "too_few_points", "candidate": candidate, "n_params": 0}
    d = np.asarray([float(row["tokens"]) for row in rows])
    y = np.asarray([float(row["normalized_damage"]) for row in rows])
    scale = max(float(np.std(y)), float(np.mean(np.abs(y))), 1e-6)
    max_x = max(float(np.max(np.log1p(d / 1e6))), 1e-6)
    if candidate == "monotonic_saturation":
        names = ("r", "D0", "beta")
        x0 = np.array([0.2, math.log(max(float(np.median(d)), 1.0)), 0.0])
        lower = np.array([0.0, math.log(max(float(np.min(d)) * 1e-4, 1.0)), math.log(0.02)])
        upper = np.array([1.0, math.log(max(float(np.max(d)) * 1e4, 2.0)), math.log(20.0)])
        def unpack(theta: np.ndarray) -> dict[str, float]:
            return {"r": float(theta[0]), "D0": math.exp(float(theta[1])), "beta": math.exp(float(theta[2]))}
    elif candidate == "change_point":
        names = ("a", "c", "tau")
        x0 = np.array([-0.2, 0.2, max_x / 2.0])
        lower = np.array([-20.0, -20.0, 0.0])
        upper = np.array([20.0, 20.0, max_x])
        def unpack(theta: np.ndarray) -> dict[str, float]:
            return {"a": float(theta[0]), "c": float(theta[1]), "tau": float(theta[2])}
    else:
        names = ("D0", "beta", "kappa", "p")
        x0 = np.array([math.log(max(float(np.min(d)), 1.0)), 0.0, 0.01, 1.0])
        lower = np.array([math.log(max(float(np.min(d)) * 1e-4, 1.0)), math.log(0.02), 0.0, 0.1])
        upper = np.array([math.log(max(float(np.max(d)) * 1e4, 2.0)), math.log(20.0), 20.0, 4.0])
        def unpack(theta: np.ndarray) -> dict[str, float]:
            return {"D0": math.exp(float(theta[0])), "beta": math.exp(float(theta[1])),
                    "kappa": float(theta[2]), "p": float(theta[3])}

    def residual(theta: np.ndarray) -> np.ndarray:
        base = (recovery_curve(candidate, d, unpack(theta)) - y) / scale
        # A tiny deterministic tie-break makes diagnostics reproducible when
        # two held-in points cannot identify three/four shape parameters.
        ridge = math.sqrt(1e-6) * (theta - x0)
        return np.concatenate([base, ridge])

    result = least_squares(residual, x0, bounds=(lower, upper), max_nfev=30_000)
    n_params = len(names)
    status = "ok" if result.success and len(rows) > n_params else "underidentified"
    return {
        "status": status,
        "candidate": candidate,
        "n_params": n_params,
        "n_points": len(rows),
        "parameters": unpack(result.x),
        "warning": "n<=k; deterministic weak tie-break used" if len(rows) <= n_params else "",
    }


def predict_recovery_candidate(
    fit: Mapping[str, object], row: Mapping[str, object]
) -> float | None:
    if fit.get("status") not in {"ok", "underidentified"}:
        return None
    parameters = fit["parameters"]
    assert isinstance(parameters, Mapping)
    return float(recovery_curve(str(fit["candidate"]), [float(row["tokens"])], parameters)[0])


def _wilson(correct: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    p = correct / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _bootstrap_interval(values: np.ndarray, statistic: Callable[[np.ndarray], float], label: str,
                        n_resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float]:
    if len(values) == 0:
        return math.nan, math.nan
    if len(values) == 1:
        value = statistic(values)
        return value, value
    rng = np.random.default_rng(_stable_seed(label))
    estimates = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = values[rng.integers(0, len(values), len(values))]
        estimates[i] = statistic(sample)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _conformal_radius(observed: np.ndarray, predicted: np.ndarray, coverage: float = 0.95) -> float:
    residuals = np.sort(np.abs(observed - predicted))
    if not len(residuals):
        return math.nan
    rank = min(len(residuals), int(math.ceil((len(residuals) + 1) * coverage)))
    return float(residuals[rank - 1])


def _prediction_metrics(records: Sequence[Mapping[str, object]], label: str,
                        total_test: int | None = None) -> dict[str, object]:
    total = len(records) if total_test is None else total_test
    if not records:
        return {"n": 0, "total": total, "coverage": 0.0, "mae": None,
                "relative_error": None, "sign_accuracy": None,
                "calibration_coverage": None}
    observed = np.asarray([float(record["observed"]) for record in records])
    predicted = np.asarray([float(record["predicted"]) for record in records])
    errors = np.abs(observed - predicted)
    mae = float(np.mean(errors))
    denominator = float(np.mean(np.abs(observed)))
    relative = mae / denominator if denominator > EPS else math.inf
    mae_ci = _bootstrap_interval(errors, np.mean, label + ":mae")
    paired = np.column_stack([errors, np.abs(observed)])
    rel_ci = _bootstrap_interval(
        paired,
        lambda sample: float(np.mean(sample[:, 0]) / max(np.mean(sample[:, 1]), EPS)),
        label + ":relative",
    )
    sign_correct = int(np.sum(np.sign(observed) == np.sign(predicted)))
    sign_ci = _wilson(sign_correct, len(records))
    interval_records = [record for record in records if math.isfinite(float(record.get("pi_radius", math.nan)))]
    if interval_records:
        in_interval = sum(
            abs(float(record["observed"]) - float(record["predicted"])) <= float(record["pi_radius"])
            for record in interval_records
        )
        calibration = in_interval / len(interval_records)
        calibration_ci = _wilson(in_interval, len(interval_records))
        mean_width = 2 * float(np.mean([float(record["pi_radius"]) for record in interval_records]))
    else:
        calibration = None
        calibration_ci = None
        mean_width = None
    return {
        "n": len(records), "total": total, "coverage": len(records) / total if total else 0.0,
        "mae": mae, "mae_ci95": mae_ci,
        "relative_error": relative, "relative_error_ci95": rel_ci,
        "sign_accuracy": sign_correct / len(records), "sign_ci95": sign_ci,
        "n_sign_correct": sign_correct,
        "calibration_coverage": calibration, "calibration_ci95": calibration_ci,
        "mean_pi_width": mean_width,
    }


def _heldout_cell_key(record: Mapping[str, object]) -> tuple[object, ...]:
    """Semantic key shared by predictions from two candidates on one cell."""
    coordinate = next(
        (
            (name, round(float(record[name]), 12))
            for name in ("density", "bits", "tokens", "r_storage")
            if record.get(name) is not None and record.get(name) != ""
        ),
        ("coordinate", ""),
    )
    return (
        str(record.get("held_out", "")),
        str(record.get("model", "")),
        str(record.get("capability", "")),
        coordinate,
    )


def paired_bootstrap_mae_difference(
    candidate: Mapping[str, object] | Sequence[Mapping[str, object]],
    baseline: Mapping[str, object] | Sequence[Mapping[str, object]],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    label: str = "paired-mae",
) -> dict[str, object]:
    """Bootstrap paired held-out-cell ``candidate MAE - baseline MAE``.

    Predictions are never refit inside this bootstrap: the resampling unit is
    the held-out cell requested by the directive.  Partial methods are compared
    only on their exact common cells, and zero overlap is reported rather than
    imputed.
    """
    candidate_records = (
        candidate.get("records", []) if isinstance(candidate, Mapping) else candidate
    )
    baseline_records = (
        baseline.get("records", []) if isinstance(baseline, Mapping) else baseline
    )
    candidate_by_key = {_heldout_cell_key(row): row for row in candidate_records}
    baseline_by_key = {_heldout_cell_key(row): row for row in baseline_records}
    keys = sorted(set(candidate_by_key) & set(baseline_by_key), key=str)
    if not keys:
        return {"n_paired": 0, "difference": None, "ci95": None,
                "includes_zero": None}
    differences = np.asarray([
        abs(float(candidate_by_key[key]["observed"])
            - float(candidate_by_key[key]["predicted"]))
        - abs(float(baseline_by_key[key]["observed"])
              - float(baseline_by_key[key]["predicted"]))
        for key in keys
    ])
    rng = np.random.default_rng(_stable_seed(label))
    estimates = np.empty(n_resamples)
    for index in range(n_resamples):
        estimates[index] = float(np.mean(
            differences[rng.integers(0, len(differences), len(differences))]
        ))
    low, high = (float(value) for value in np.quantile(estimates, [0.025, 0.975]))
    return {
        "n_paired": len(keys),
        "difference": float(np.mean(differences)),
        "ci95": (low, high),
        "includes_zero": low <= 0.0 <= high,
    }


def evaluate_splits(
    rows: Sequence[Mapping[str, object]],
    splits: Sequence[Mapping[str, object]],
    fit_fn: Callable[[Sequence[Mapping[str, object]]], Mapping[str, object]],
    predict_fn: Callable[[Mapping[str, object], Mapping[str, object]], float | None],
    label: str,
) -> dict[str, object]:
    """Pool genuinely held-out predictions and train-only calibration radii."""
    records: list[dict[str, object]] = []
    folds = []
    total_test = 0
    for split in splits:
        train = [rows[int(i)] for i in split["train_indices"]]  # type: ignore[index]
        test_indices = [int(i) for i in split["test_indices"]]  # type: ignore[index]
        test = [rows[i] for i in test_indices]
        total_test += len(test)
        fit = fit_fn(train)
        train_pairs = []
        for row in train:
            value = predict_fn(fit, row)
            if value is not None and math.isfinite(value):
                train_pairs.append((float(row.get("observed", row.get("delta_L_c"))), value))
        radius = (_conformal_radius(np.asarray([x[0] for x in train_pairs]),
                                    np.asarray([x[1] for x in train_pairs]))
                  if train_pairs else math.nan)
        fold_records = []
        for row, row_index in zip(test, test_indices):
            value = predict_fn(fit, row)
            if value is None or not math.isfinite(value):
                continue
            record = {
                "held_out": split["held_out"],
                "model": row.get("model", ""), "family": row.get("family", ""),
                "capability": row.get("capability", ""),
                "observed": float(row.get("observed", row.get("delta_L_c"))),
                "predicted": float(value), "pi_radius": radius,
                "row_index": row_index,
            }
            for coordinate in ("density", "bits", "tokens", "r_storage"):
                if row.get(coordinate) is not None and row.get(coordinate) != "":
                    record[coordinate] = float(row[coordinate])
            records.append(record)
            fold_records.append(record)
        folds.append({"held_out": split["held_out"], "n_train": len(train), "n_test": len(test),
                      "fit_status": fit.get("status"), "n_predictions": len(fold_records)})
    metrics = _prediction_metrics(records, label, total_test)
    statuses = {fold["fit_status"] for fold in folds}
    status = "OK" if metrics["coverage"] == 1.0 and statuses <= {"ok"} else "PARTIAL"
    return {"status": status, "metrics": metrics, "folds": folds, "records": records}


def evaluate_in_sample(
    rows: Sequence[Mapping[str, object]],
    fit_fn: Callable[[Sequence[Mapping[str, object]]], Mapping[str, object]],
    predict_fn: Callable[[Mapping[str, object], Mapping[str, object]], float | None],
    label: str,
) -> dict[str, object]:
    fit = fit_fn(rows)
    records = []
    for row in rows:
        value = predict_fn(fit, row)
        if value is not None and math.isfinite(value):
            records.append({"observed": float(row.get("observed", row.get("delta_L_c"))),
                            "predicted": value})
    return {"status": "NON-DECISIONAL", "fit": fit,
            "metrics": _prediction_metrics(records, label, len(rows)), "records": records}


def _cliff_records(rows: Sequence[Mapping[str, object]], coordinate: str) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["capability"]))].append(row)
    observed, censored = [], []
    for (model, capability), trajectory in sorted(grouped.items()):
        post = [row for row in trajectory if row.get("precliff_status") == "post_cliff"]
        common = {
            "model": model, "capability": capability,
            "family": trajectory[0]["family"], "N0": trajectory[0]["N0"],
            "L_c0": trajectory[0]["L_c0"],
        }
        if post:
            # The first crossing is the least-compressed post-cliff point.
            common["observed"] = max(float(row[coordinate]) for row in post)
            observed.append(common)
        else:
            common["censoring"] = f"below {min(float(row[coordinate]) for row in trajectory):g}"
            censored.append(common)
    return observed, censored


def _fit_cliff_location(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(rows) < 2:
        return {"status": "too_few_points", "n_params": 0}
    y = np.asarray([float(row["observed"]) for row in rows])
    log_n = np.log(np.asarray([float(row["N0"]) for row in rows]) / 1e9)
    anchors = np.asarray([float(row["L_c0"]) for row in rows])
    n_center, n_scale = _standardizer(log_n)
    l_center, l_scale = _standardizer(anchors)
    capabilities = sorted({str(row["capability"]) for row in rows})
    families = sorted({str(row["family"]) for row in rows})
    design = np.column_stack([
        np.ones(len(rows)), (log_n - n_center) / n_scale, (anchors - l_center) / l_scale,
        _effect_design(capabilities, [str(row["capability"]) for row in rows]),
        _effect_design(families, [str(row["family"]) for row in rows]),
    ])
    ridge = 1e-5 * np.eye(design.shape[1]); ridge[0, 0] = 0
    coefficients = np.linalg.solve(design.T @ design + ridge, design.T @ y)
    return {"status": "ok", "n_params": len(coefficients),
            "parameters": {"coefficients": coefficients.tolist(), "n_center": n_center,
                           "n_scale": n_scale, "l_center": l_center, "l_scale": l_scale},
            "capabilities": capabilities, "families": families,
            "range": [float(np.min(y)), float(np.max(y))]}


def _predict_cliff_location(fit: Mapping[str, object], row: Mapping[str, object]) -> float | None:
    if fit.get("status") != "ok":
        return None
    p = fit["parameters"]; assert isinstance(p, Mapping)
    design = np.concatenate([
        [1.0,
         (math.log(float(row["N0"]) / 1e9) - float(p["n_center"])) / float(p["n_scale"]),
         (float(row["L_c0"]) - float(p["l_center"])) / float(p["l_scale"])],
        _effect_design(list(fit["capabilities"]), [str(row["capability"])])[0],
        _effect_design(list(fit["families"]), [str(row["family"])])[0],
    ])
    value = float(design @ np.asarray(p["coefficients"], dtype=float))
    low, high = fit["range"]
    return float(np.clip(value, low, high))


def _evaluate_cliff(rows: Sequence[Mapping[str, object]], splits: Sequence[Mapping[str, object]],
                    label: str) -> dict[str, object]:
    result = evaluate_splits(rows, splits, _fit_cliff_location, _predict_cliff_location, label)
    metrics = result["metrics"]
    metrics["cliff_error"] = metrics["mae"]
    metrics["cliff_error_ci95"] = metrics.get("mae_ci95")
    return result


def _bootstrap_distillation_parameters(rows: Sequence[Mapping[str, object]], n_resamples: int) -> dict[str, dict[str, tuple[float, float]]]:
    models = sorted({str(row["model"]) for row in rows})
    estimates: dict[str, dict[str, list[float]]] = {
        capability: {"floor": [], "alpha": []} for capability in CAPABILITIES
    }
    rng = np.random.default_rng(_stable_seed("distillation-parameter-bootstrap"))
    for _ in range(n_resamples):
        selected = [models[i] for i in rng.integers(0, len(models), len(models))]
        if len(set(selected)) < 2:
            continue
        sample = []
        for draw, model in enumerate(selected):
            for row in rows:
                if str(row["model"]) == model:
                    sample.append({**row, "model": f"{model}#{draw}"})
        fit = fit_distillation_law(sample)
        for capability, values in fit["parameters"].items():
            estimates[capability]["floor"].append(float(values["floor"]))
            estimates[capability]["alpha"].append(float(values["alpha"]))
    output = {}
    for capability, values in estimates.items():
        output[capability] = {}
        for name, samples in values.items():
            output[capability][name] = tuple(float(x) for x in np.quantile(samples, [0.025, 0.975])) if samples else (math.nan, math.nan)
    return output


def _load_recovery_rows() -> tuple[dict[str, list[dict]], list[str]]:
    runs: dict[str, list[dict]] = {}
    notes = []
    for path in sorted(RECOVERY_BASE.glob("*/recovery.json")):
        payload = _json(path)
        source_value = payload.get("recovery_source", path.parent.name)
        source = str(source_value.get("name", source_value) if isinstance(source_value, Mapping) else source_value)
        dense = payload.get("anchors", {}).get("dense", {})
        damaged = payload.get("anchors", {}).get("damaged", {})
        rows = []
        for capability in CAPABILITIES:
            if capability not in dense or capability not in damaged:
                continue
            initial = float(damaged[capability]) - float(dense[capability])
            for cell in payload.get("ladder", []):
                if capability not in cell.get("losses", {}):
                    continue
                tokens = cell.get("tokens_seen", cell.get("budget_requested", cell.get("budget")))
                if tokens is None:
                    continue
                damage = float(cell["losses"][capability]) - float(dense[capability])
                rows.append({
                    "source": source, "model": "gemma3-1b", "family": "gemma",
                    "capability": capability, "tokens": float(tokens), "damage": damage,
                    "observed": damage, "initial_damage": initial,
                    "normalized_damage": damage / initial if abs(initial) > EPS else math.nan,
                    "precliff_status": "pre_cliff" if damage <= PRECLIFF_CAP else "post_cliff",
                    "path": str(path.relative_to(ROOT)),
                })
        runs[source] = rows
    if not runs:
        notes.append("No recovery artifacts found")
    return runs, notes


def _fit_points(result: Mapping[str, object]) -> str:
    folds = result.get("folds", [])
    if not folds:
        return "0"
    train = [int(fold["n_train"]) for fold in folds]
    test = sum(int(fold["n_test"]) for fold in folds)
    train_text = str(train[0]) if len(set(train)) == 1 else f"{min(train)}-{max(train)}"
    return f"{train_text}/fold; {test} held-out"


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def _fmt_metric(metrics: Mapping[str, object], key: str, ci_key: str) -> str:
    if metrics.get(key) is None:
        return "n/a"
    interval = metrics.get(ci_key)
    if interval is None:
        return _fmt(metrics[key])
    return f"{_fmt(metrics[key])} [{_fmt(interval[0])}, {_fmt(interval[1])}]"


def _table_row(method: str, inputs: str, law: str, fit_points: str, split: str,
               result: Mapping[str, object], *, cliff: bool = False,
               status: str | None = None) -> dict[str, str]:
    metrics = result.get("metrics", {})
    assert isinstance(metrics, Mapping)
    row_status = status or str(result.get("status", "OK"))
    calibration = "n/a"
    if metrics.get("calibration_coverage") is not None:
        ci = metrics.get("calibration_ci95")
        calibration = (f"95% PI cover {_fmt(metrics['calibration_coverage'])} "
                       f"[{_fmt(ci[0])}, {_fmt(ci[1])}], width {_fmt(metrics.get('mean_pi_width'))}")
    return {
        "Method": f"{method} [{row_status}]",
        "Inputs": inputs,
        "Candidate law": law,
        "Fit points": fit_points,
        "Held-out split": split,
        "MAE": "n/a" if cliff else _fmt_metric(metrics, "mae", "mae_ci95"),
        "Relative error": "n/a" if cliff else _fmt_metric(metrics, "relative_error", "relative_error_ci95"),
        "Sign accuracy": "n/a" if cliff else _fmt_metric(metrics, "sign_accuracy", "sign_ci95"),
        "Cliff error": _fmt_metric(metrics, "cliff_error", "cliff_error_ci95") if cliff else "n/a",
        "Calibration": calibration,
    }


def _planned_row(method: str, inputs: str, law: str, split: str, missing: str,
                 status: str = "PLANNED") -> dict[str, str]:
    return {
        "Method": f"{method} [{status}]", "Inputs": inputs, "Candidate law": law,
        "Fit points": f"0; missing {missing}", "Held-out split": split,
        "MAE": "n/a", "Relative error": "n/a", "Sign accuracy": "n/a",
        "Cliff error": "n/a", "Calibration": "n/a",
    }


def _markdown_table(rows: Sequence[Mapping[str, str]]) -> list[str]:
    columns = ("Method", "Inputs", "Candidate law", "Fit points", "Held-out split",
               "MAE", "Relative error", "Sign accuracy", "Cliff error", "Calibration")
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]).replace("|", "\\|") for column in columns) + " |")
    return lines


def build_report(bootstrap_resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[str, dict[str, object]]:
    prune_all, quant_all, distill_all, audit = _load_method_rows()
    prune = [row for row in prune_all if row["precliff_status"] == "pre_cliff"]
    quant = [row for row in quant_all if row["precliff_status"] == "pre_cliff"]
    distill = [row for row in distill_all if row["precliff_status"] == "pre_cliff"]

    pruning_rows = []
    pruning_results: dict[str, dict[str, dict]] = defaultdict(dict)
    pruning_protocols = {
        "fit d>=0.55 -> deeper pre-cliff": split_shallow_to_deep(prune),
        "leave-largest-model-out": leave_largest_model_out_splits(prune),
        "leave-one-family-out": leave_one_family_out_splits(prune),
    }
    for candidate in PRUNING_CANDIDATES:
        candidate_rows = prune
        if candidate == "mechanism_upper_bound":
            # The two infilled density grids have loss measurements but no
            # matching V6 alignment measurement.  Keep them for observable
            # reduced forms and omit them only from this measured-mechanism
            # comparator.
            candidate_rows = [
                row for row in prune
                if math.isfinite(float(row["first_order"]))
                and math.isfinite(float(row["deleted_mass"]))
            ]
        fit_fn = lambda rows, candidate=candidate: fit_pruning_candidate(rows, candidate)
        inputs = "L_c0, N0, f, d"
        if candidate == "mechanism_upper_bound":
            inputs += ", measured g·delta_w, m_c"
        if candidate == "baseline_removed_weight_norm":
            inputs += ", binned removed-weight L2"
        for split_name, splits in pruning_protocols.items():
            candidate_splits = splits
            if candidate_rows is not prune:
                if split_name.startswith("fit d"):
                    candidate_splits = split_shallow_to_deep(candidate_rows)
                elif split_name == "leave-largest-model-out":
                    candidate_splits = leave_largest_model_out_splits(candidate_rows)
                else:
                    candidate_splits = leave_one_family_out_splits(candidate_rows)
            result = evaluate_splits(candidate_rows, candidate_splits, fit_fn, predict_pruning_candidate,
                                     f"pruning:{candidate}:{split_name}")
            pruning_results[candidate][split_name] = result
            pruning_rows.append(_table_row("Pruning", inputs, PRUNING_FORMS[candidate],
                                           _fit_points(result), split_name, result))
        in_sample = evaluate_in_sample(candidate_rows, fit_fn, predict_pruning_candidate,
                                       f"pruning:{candidate}:in-sample")
        pruning_rows.append(_table_row("Pruning", inputs, PRUNING_FORMS[candidate],
                                       str(len(candidate_rows)), "NON-DECISIONAL in-sample", in_sample,
                                       status="NON-DECISIONAL"))

    prune_cliffs, prune_censored = _cliff_records(prune_all, "density")
    prune_cliff_results = {}
    for name, splits in {
        "held-out cliff density: largest model": leave_largest_model_out_splits(prune_cliffs),
        "held-out cliff density: family": leave_one_family_out_splits(prune_cliffs),
    }.items():
        result = _evaluate_cliff(prune_cliffs, splits, "pruning-cliff:" + name)
        prune_cliff_results[name] = result
        pruning_rows.append(_table_row(
            "Pruning", "L_c0, N0, f, d", PRUNING_FORMS["smooth_cliff_extension"] + "; location submodel",
            _fit_points(result), name, result, cliff=True,
        ))

    quantization_rows = []
    quantization_results: dict[str, dict[str, dict]] = defaultdict(dict)
    quant_protocols = {
        "leave-one-bit-out (pre-cliff only)": leave_one_bit_out_splits(quant),
        "leave-largest-model-out": leave_largest_model_out_splits(quant),
        "leave-one-family-out": leave_one_family_out_splits(quant),
    }
    for candidate in QUANTIZATION_CANDIDATES:
        fit_fn = lambda rows, candidate=candidate: fit_quantization_candidate(rows, candidate)
        for split_name, splits in quant_protocols.items():
            result = evaluate_splits(quant, splits, fit_fn, predict_quantization_candidate,
                                     f"quant:{candidate}:{split_name}")
            quantization_results[candidate][split_name] = result
            quantization_rows.append(_table_row(
                "Quantization", "L_c0, N0, f, b; r_storage=b/16 nominal",
                QUANTIZATION_FORMS[candidate], _fit_points(result), split_name, result,
            ))
        in_sample = evaluate_in_sample(quant, fit_fn, predict_quantization_candidate,
                                       f"quant:{candidate}:in-sample")
        quantization_rows.append(_table_row(
            "Quantization", "L_c0, N0, f, b; r_storage=b/16 nominal",
            QUANTIZATION_FORMS[candidate], str(len(quant)), "NON-DECISIONAL in-sample",
            in_sample, status="NON-DECISIONAL",
        ))

    quant_cliffs, quant_censored = _cliff_records(quant_all, "bits")
    quant_cliff_results = {}
    for name, splits in {
        "int3-collapse prediction: leave-model-out": v17.leave_one_out_splits(quant_cliffs, "model"),
        "int3-collapse prediction: leave-family-out": leave_one_family_out_splits(quant_cliffs),
    }.items():
        result = _evaluate_cliff(quant_cliffs, splits, "quant-cliff:" + name)
        quant_cliff_results[name] = result
        quantization_rows.append(_table_row(
            "Quantization", "L_c0, N0, f, b; r_storage=b/16 nominal",
            QUANTIZATION_FORMS["family_conditioned_smooth_cliff"] + "; location submodel",
            _fit_points(result), name, result, cliff=True,
        ))

    distill_rows = []
    distill_splits = leave_one_student_size_out_splits(distill)
    distill_heldout = evaluate_splits(distill, distill_splits, fit_distillation_law,
                                      predict_distillation_law,
                                      "distillation:leave-one-size")
    distill_rows.append(_table_row(
        "Distillation", "source L_c0 (Gemma-3 27B), N0, f, r_storage",
        "Delta L_c=floor_c-alpha_c log(r_storage)", _fit_points(distill_heldout),
        "fit 3 student sizes -> predict 4th (all four rotations)", distill_heldout,
        status="PARTIAL",
    ))
    distill_in = evaluate_in_sample(distill, fit_distillation_law,
                                    predict_distillation_law, "distillation:in-sample")
    distill_rows.append(_table_row(
        "Distillation", "source L_c0 (Gemma-3 27B), N0, f, r_storage",
        "Delta L_c=floor_c-alpha_c log(r_storage)", str(len(distill)),
        "NON-DECISIONAL in-sample", distill_in, status="NON-DECISIONAL",
    ))
    distill_rows.append(_planned_row(
        "Distillation", "L_c0, N0, f, r_storage", "same source-referenced log-size law",
        "Qwen/Gemma cross-family", "a clean Qwen source-referenced student ladder",
    ))
    distill_rows.append(_planned_row(
        "Distillation", "L_c0, N0, f, r_storage, D", "capacity floor plus D-ladder term",
        "held-out D-ladder", "completed multi-D runs for >=3 training budgets plus a held-out budget",
    ))
    distill_fit = fit_distillation_law(distill)
    distill_ci = _bootstrap_distillation_parameters(distill, bootstrap_resamples)

    recovery_runs, recovery_notes = _load_recovery_rows()
    recovery_rows = []
    recovery_results = {}
    for source in ("c4", "traces"):
        all_rows = recovery_runs.get(source, [])
        eligible = [row for row in all_rows if row["precliff_status"] == "pre_cliff"]
        candidates = ("monotonic_saturation",) if source == "c4" else RECOVERY_CANDIDATES
        for candidate in candidates:
            records = []
            fold_summaries = []
            total_test = 0
            for capability in CAPABILITIES:
                cap_rows = [row for row in eligible if row["capability"] == capability]
                splits = leave_one_budget_out_splits(cap_rows)
                if not splits:
                    continue
                result = evaluate_splits(
                    cap_rows, splits,
                    lambda rows, candidate=candidate: fit_recovery_candidate(rows, candidate),
                    lambda fit, row: (
                        None if (value := predict_recovery_candidate(fit, row)) is None
                        else value * float(row["initial_damage"])
                    ),
                    f"recovery:{source}:{candidate}:{capability}",
                )
                records.extend(result["records"])
                fold_summaries.extend(result["folds"])
                total_test += int(result["metrics"]["total"])
            metrics = _prediction_metrics(records, f"recovery:{source}:{candidate}:pooled", total_test)
            result = {"status": "PARTIAL", "metrics": metrics, "folds": fold_summaries,
                      "records": records}
            recovery_results[f"{source}:{candidate}"] = result
            missing = (
                "only QA retains 3 pre-cliff budgets; math/code retain 2/1"
                if source == "c4" else
                "only the 1M point is pre-cliff in each capability; 4M/16M exceed 1 nat"
            )
            recovery_rows.append(_table_row(
                "Recovery", "Delta L_c(0), D_R, source, capability",
                RECOVERY_FORMS[candidate], _fit_points(result),
                f"leave-one-budget-out ({source}); PARTIAL: {missing}", result,
                status="PARTIAL",
            ))
            # Full-data fits are diagnostic and per-capability; summarize their
            # predictions without allowing them into a decision.
            in_records = []
            for capability in CAPABILITIES:
                cap_rows = [row for row in eligible if row["capability"] == capability]
                if not cap_rows:
                    continue
                fit = fit_recovery_candidate(cap_rows, candidate)
                for row in cap_rows:
                    normalized = predict_recovery_candidate(fit, row)
                    if normalized is not None:
                        in_records.append({"observed": row["damage"],
                                           "predicted": normalized * row["initial_damage"]})
            in_result = {"status": "NON-DECISIONAL",
                         "metrics": _prediction_metrics(in_records, f"recovery:{source}:{candidate}:in", len(eligible))}
            recovery_rows.append(_table_row(
                "Recovery", "Delta L_c(0), D_R, source, capability",
                RECOVERY_FORMS[candidate], str(len(eligible)),
                f"NON-DECISIONAL in-sample ({source}; underidentified)", in_result,
                status="NON-DECISIONAL",
            ))

    def sign_breakdown(result: Mapping[str, object]) -> str:
        by_cap = []
        records = result.get("records", [])
        for capability in CAPABILITIES:
            subset = [row for row in records if row.get("capability") == capability]
            if subset:
                correct = sum(np.sign(row["observed"]) == np.sign(row["predicted"]) for row in subset)
                by_cap.append(f"{capability} {correct}/{len(subset)}")
        return ", ".join(by_cap) or "n/a"

    def best_result(results: Mapping[str, Mapping[str, Mapping]], candidates: Sequence[str],
                    protocol: str) -> tuple[str, float]:
        scored = {
            candidate: float(results[candidate][protocol]["metrics"]["mae"])
            for candidate in candidates
            if results.get(candidate, {}).get(protocol, {}).get("metrics", {}).get("mae") is not None
        }
        return min(scored.items(), key=lambda item: item[1]) if scored else ("none", math.nan)

    pruning_findings = []
    paired_selection_rows = []
    paired_selection: dict[str, dict[str, object]] = {"pruning": {}, "quantization": {}}
    for protocol in pruning_protocols:
        candidate, candidate_mae = best_result(
            pruning_results, PRUNING_CANDIDATES[:6], protocol
        )
        baseline, baseline_mae = best_result(
            pruning_results, PRUNING_CANDIDATES[6:], protocol
        )
        pruning_findings.append(
            f"`{protocol}`: candidate `{candidate}` {_fmt(candidate_mae)} vs baseline `{baseline}` {_fmt(baseline_mae)}"
        )
        paired = paired_bootstrap_mae_difference(
            pruning_results[candidate][protocol], pruning_results[baseline][protocol],
            n_resamples=bootstrap_resamples,
            label=f"pruning:selected:{candidate}:{baseline}:{protocol}",
        )
        paired_selection["pruning"][protocol] = {
            "candidate": candidate, "baseline": baseline, **paired,
        }
        if paired["ci95"] is None:
            decision = "not estimable"
        elif paired["includes_zero"]:
            decision = "unresolved; prefer density_only"
        elif float(paired["difference"]) < 0.0:
            decision = "candidate lower error"
        else:
            decision = "baseline lower error; prefer density_only"
        paired_selection_rows.append({
            "Method": "Pruning", "Protocol": protocol,
            "Candidate": f"{candidate} ({_fmt(candidate_mae)})",
            "Baseline": f"{baseline} ({_fmt(baseline_mae)})",
            "Paired cells": str(paired["n_paired"]),
            "Paired bootstrap ΔMAE candidate−baseline (95% CI)": (
                "n/a" if paired["ci95"] is None else
                f"{_fmt(paired['difference'])} [{_fmt(paired['ci95'][0])}, {_fmt(paired['ci95'][1])}]"
            ),
            "Simplicity decision": decision,
        })
    quant_findings = []
    for protocol in quant_protocols:
        candidate, candidate_mae = best_result(
            quantization_results, QUANTIZATION_CANDIDATES[:4], protocol
        )
        baseline, baseline_mae = best_result(
            quantization_results, QUANTIZATION_CANDIDATES[4:], protocol
        )
        baseline_text = f"baseline `{baseline}` {_fmt(baseline_mae)}" if math.isfinite(baseline_mae) else "categorical baseline not estimable"
        quant_findings.append(
            f"`{protocol}`: candidate `{candidate}` {_fmt(candidate_mae)} vs {baseline_text}"
        )
        paired = (
            paired_bootstrap_mae_difference(
                quantization_results[candidate][protocol],
                quantization_results[baseline][protocol],
                n_resamples=bootstrap_resamples,
                label=f"quant:selected:{candidate}:{baseline}:{protocol}",
            )
            if baseline != "none" else
            {"n_paired": 0, "difference": None, "ci95": None,
             "includes_zero": None}
        )
        paired_selection["quantization"][protocol] = {
            "candidate": candidate, "baseline": baseline, **paired,
        }
        if paired["ci95"] is None:
            decision = "categorical baseline cannot predict unseen bit" if protocol.startswith("leave-one-bit") else "not estimable"
        elif paired["includes_zero"]:
            decision = "unresolved; prefer fixed_4^-b"
        elif float(paired["difference"]) < 0.0:
            decision = "candidate lower error"
        else:
            decision = "baseline lower error; prefer fixed_4^-b"
        paired_selection_rows.append({
            "Method": "Quantization", "Protocol": protocol,
            "Candidate": f"{candidate} ({_fmt(candidate_mae)})",
            "Baseline": baseline_text,
            "Paired cells": str(paired["n_paired"]),
            "Paired bootstrap ΔMAE candidate−baseline (95% CI)": (
                "n/a" if paired["ci95"] is None else
                f"{_fmt(paired['difference'])} [{_fmt(paired['ci95'][0])}, {_fmt(paired['ci95'][1])}]"
            ),
            "Simplicity decision": decision,
        })

    def selection_table(rows: Sequence[Mapping[str, str]]) -> list[str]:
        columns = (
            "Method", "Protocol", "Candidate", "Baseline", "Paired cells",
            "Paired bootstrap ΔMAE candidate−baseline (95% CI)",
            "Simplicity decision",
        )
        output = ["| " + " | ".join(columns) + " |",
                  "|" + "|".join("---" for _ in columns) + "|"]
        for row in rows:
            output.append("| " + " | ".join(
                str(row[column]).replace("|", "\\|") for column in columns
            ) + " |")
        return output

    pruning_sign_pass = any(
        all(
            pruning_results[candidate][protocol]["metrics"].get("sign_accuracy") == 1.0
            for protocol in pruning_protocols
        )
        for candidate in PRUNING_CANDIDATES[:6]
    )
    quant_sign_pass = any(
        all(
            quantization_results[candidate][protocol]["metrics"].get("sign_accuracy") == 1.0
            for protocol in quant_protocols
        )
        for candidate in QUANTIZATION_CANDIDATES[:4]
    )

    distill_parameters = distill_fit["parameters"]
    lines = [
        "# Unified held-out law-fit report",
        "",
        f"Generated {date.today().isoformat()} by `analysis/v18_law_fit.py` from committed/local result artifacts only. This was a CPU-only refit: no inference, training, GPU use, or result-input mutation occurred.",
        "",
        "## Decision rule and scope",
        "",
        f"Headline metrics below are pooled predictions for observations excluded from their fold's fit. MAE is in CE nats; relative error is MAE divided by mean absolute held-out damage. Brackets on MAE/relative error are deterministic {bootstrap_resamples:,}-resample held-out-cell bootstrap 95% intervals; sign brackets and calibration brackets are Wilson 95% intervals. Calibration is empirical coverage of a nominal 95% absolute-residual interval calibrated on that fold's training residuals. These intervals quantify variation across available cells, not benchmark-item or seed uncertainty.",
        "",
        "Every compression-damage fit uses the V17 cliff filter: scan each `(model, method, capability)` trajectory toward stronger compression; the first `Delta L_c > 1.0` nat cell and all deeper cells are excluded. Dense anchors are never scored. Recovery has no varying compression coordinate, so each recovery observation is admitted only when its own `Delta L_c <= 1.0` nat. Negative deltas remain eligible; censoring them would invalidate sign evaluation.",
        "",
        "`r_storage=b/16` for quantization is a nominal bit budget, not serialized size or runtime memory. Unstructured pruning density is likewise nominal and does not imply sparse-kernel speedup.",
        "",
        "Status `OK` means the declared fold produced predictions for every held-out eligible cell. `PARTIAL` means a requested axis is underidentified or some held-out cells are not estimable. `PLANNED` names the missing data. `NON-DECISIONAL` is an in-sample diagnostic and never a headline.",
        "",
        "## Held-out findings",
        "",
        "- **Pruning remains on probation.** Best candidate-versus-baseline held-out MAEs are " + "; ".join(pruning_findings) + ". " + ("At least one candidate clears the no-sign-error rule." if pruning_sign_pass else "No candidate clears the directive's no-held-out-sign-error rule."),
        "- **Quantization remains on probation.** Best held-out comparisons are " + "; ".join(quant_findings) + ". " + ("At least one candidate clears the no-sign-error rule." if quant_sign_pass else "No candidate clears the no-held-out-sign-error rule."),
        f"- **Distillation is PARTIAL:** the real four-rotation size-held-out MAE is {_fmt(distill_heldout['metrics']['mae'])} nats, but only one family/teacher/data budget is available; cross-family and D-ladder rows remain PLANNED.",
        "- **Recovery is PARTIAL:** C4 has a diagnostic QA-only held-out-budget score after filtering; aligned traces have no estimable held-out law because only one budget per capability remains below the 1-nat cap.",
        "- **Cliffs are separately held out:** pruning cliff-density MAE is " + _fmt(prune_cliff_results["held-out cliff density: largest model"]["metrics"]["cliff_error"]) + " for largest-model holdout and " + _fmt(prune_cliff_results["held-out cliff density: family"]["metrics"]["cliff_error"]) + " for family holdout; quantization bit-cliff MAE is " + _fmt(quant_cliff_results["int3-collapse prediction: leave-model-out"]["metrics"]["cliff_error"]) + " bits for model holdout and " + _fmt(quant_cliff_results["int3-collapse prediction: leave-family-out"]["metrics"]["cliff_error"]) + " bits for family holdout.",
        "",
        "## Paired law-versus-baseline bootstrap",
        "",
        "The comparison below resamples common held-out cells and recomputes `candidate MAE - baseline MAE` in each resample. Negative favors the candidate. It is conditional on the displayed candidates having been selected by pooled MAE and therefore does not correct selection bias. A CI containing zero supplies no evidence that added form complexity improves prediction.",
        "",
        *selection_table(paired_selection_rows),
        "",
        "**Simplicity verdict.** Where the paired interval includes zero, use the continuous `density_only` pruning law and `fixed_4^-b` quantization law as the default reduced forms rather than a more complex family/size/cliff extension. The bit-width categorical baseline cannot extrapolate to a globally unseen bit at all; thus a continuous bit law has scaling-law value even when its MAE ties a categorical baseline on seen-bit model/family holdouts.",
        "",
        "## 1. Pruning",
        "",
        f"Available: {len({row['model'] for row in prune_all})} models in {len({row['family'] for row in prune_all})} directive families, {len(prune)} pre-cliff cells from {len(prune_all)} non-anchor cells. There are {len(prune_cliffs)} observed cliff trajectories and {len(prune_censored)} right-censored trajectories.",
        "",
        "The mechanism row is intentionally an upper-bound comparator: it consumes measured `g·delta_w` and capability Fisher mass `m_c`, so it is not the observable-only reduced law. The removed-weight-norm baseline is derived from the committed V6 magnitude histogram and is explicitly binned/approximate.",
        "The mechanism comparator has 153 rather than 174 eligible cells because the later Qwen-1.7B/Gemma-1B density infill measured loss but did not re-measure `g·delta_w`; those 21 cells remain in every observable-input candidate.",
        "",
        *_markdown_table(pruning_rows),
        "",
        "Held-out sign audit (leave-one-family-out): " + sign_breakdown(
            pruning_results["hierarchical_capability_family"]["leave-one-family-out"]
        ) + ". The two cliff-censored trajectories are " + ", ".join(
            f"{row['model']}/{row['capability']} ({row['censoring']})" for row in prune_censored
        ) + "; they are not converted to invented cliff locations.",
        "",
        "## 2. Quantization",
        "",
        f"Available now: {len({row['model'] for row in quant_all})} model artifacts and {len(quant)} pre-cliff cells from {len(quant_all)} non-anchor cells. The earlier v14 positive-damage screen retained 9 models; v18 uses all 12 current artifacts because negative damage is required for the requested sign test. There are {len(quant_cliffs)} observed bit-cliff trajectories and {len(quant_censored)} right-censored trajectories.",
        "",
        *_markdown_table(quantization_rows),
        "",
        "A pure bit-category baseline is deliberately non-estimable when that bit is globally held out; those folds are `PARTIAL`, not silently imputed. Int3 damage magnitudes beyond the 1-nat cap are not used. The separate int3-collapse rows evaluate held-out cliff onset in bits. Capability-specific held-out sign audit for the family-conditioned candidate (leave-one-family-out): " + sign_breakdown(
            quantization_results["family_conditioned_smooth_cliff"]["leave-one-family-out"]
        ) + ". Censored: " + ", ".join(
            f"{row['model']}/{row['capability']} ({row['censoring']})" for row in quant_censored
        ) + ".",
        "",
        "## 3. Distillation",
        "",
        "Only the registered clean source-referenced ladder is fitted: Gemma-3 students 270M/1B/4B/12B distilled with the same `gpt-5.6-luna`, `full`, 600-trace configuration, referenced to the tested Gemma-3 27B dense source. This is one family, one teacher, one data budget and one recorded seed, so the law remains `PARTIAL` even though every size rotation is genuinely held out.",
        "",
        *_markdown_table(distill_rows),
        "",
        "Full-ladder parameter diagnostic (NON-DECISIONAL; student-size cluster bootstrap):",
        "",
        "| Capability | floor_c | bootstrap 95% CI | alpha_c | bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for capability in CAPABILITIES:
        p = distill_parameters.get(capability, {})
        ci = distill_ci.get(capability, {})
        lines.append(
            f"| {capability} | {_fmt(p.get('floor'))} | [{_fmt(ci.get('floor', (None, None))[0])}, {_fmt(ci.get('floor', (None, None))[1])}] | "
            f"{_fmt(p.get('alpha'))} | [{_fmt(ci.get('alpha', (None, None))[0])}, {_fmt(ci.get('alpha', (None, None))[1])}] |"
        )
    lines.extend([
        "",
        "For QA: **distilled students achieve lower QA probe loss than the tested 27B source under this evaluation**.",
        "",
        "## 4. Recovery",
        "",
        "The same Gemma-3 1B prune-0.6 anchor feeds both sources. Recovery is fitted to normalized damage `z_c(D_R)=Delta L_c(D_R)/Delta L_c(0)`, making the displayed saturation law dimensionless and anchored at one. C4 is assessed only with the monotonic saturation form. Traces compare monotonic saturation, change-point, and early-recovery-plus-late-penalty, but the comparison is underdetermined. In particular, enforcing the common pre-cliff rule excludes the 4M/16M aligned points that exhibit late re-damage; fitting a penalty to them would violate the report's own scope.",
        "",
        *_markdown_table(recovery_rows),
        "",
        "Aligned availability audit (not fitted): all three requested budgets exist for math/code/QA; at 1M all three are pre-cliff, while every 4M/16M capability cell exceeds the 1-nat cap. Thus each aligned candidate row is emitted as `PARTIAL` with no held-out estimate, rather than fitting forbidden points or presenting a three-point interpolation as prediction.",
        "",
        "Eligible recovery counts after the 1-nat cap: " + "; ".join(
            f"{source}: " + ", ".join(
                f"{capability}={sum(row['precliff_status']=='pre_cliff' and row['capability']==capability for row in rows)}"
                for capability in CAPABILITIES
            ) for source, rows in sorted(recovery_runs.items())
        ) + ". A defensible aligned held-out comparison needs at least three pre-cliff training budgets plus one held-out budget per capability, denser sampling around the early optimum, and another seed/model. The existing late points remain descriptive evidence of non-monotonicity, not fit points.",
        "",
        "## Provenance and limitations",
        "",
        "- Pruning: `results/v6-capability-geometry/*/{prune_losses.json,alignment.json,spectrum_bins.npz,fisher_meta.json}`; V6 artifacts do not record a run seed in these files.",
        "- Quantization: `results/v10-quantization/*/quant_losses.json`; nominal `b/16`; these compact loss files do not record a seed.",
        "- Distillation: selected V16 `gpt-5.6-luna_full_600/residual.json` files plus the V6 Gemma-3 27B dense reference; the selected V16 records use seed 0.",
        "- Recovery: `results/v13-recovery/gemma3-1b/prune_0.6_{c4,traces}/recovery.json`; traces records seed 0, while the salvaged C4 compact artifact does not record a seed.",
        "- There are no repeated measurement seeds for these law cells. Consequently the calibration/MAE intervals are cell-sampling diagnostics and cannot estimate seed or benchmark-item uncertainty.",
        "- No post-cliff loss magnitude contributes to any law coefficient or headline damage metric. Cliff-location models use only observed threshold-crossing coordinates; censored trajectories stay censored.",
    ])
    summary = {
        "audit": audit,
        "counts": {"pruning_precliff": len(prune), "quantization_precliff": len(quant),
                   "distillation_precliff": len(distill)},
        "pruning": pruning_results, "quantization": quantization_results,
        "paired_selection": paired_selection,
        "pruning_cliff": prune_cliff_results, "quantization_cliff": quant_cliff_results,
        "distillation": {"heldout": distill_heldout, "fit": distill_fit, "ci": distill_ci},
        "recovery": recovery_results, "recovery_notes": recovery_notes,
    }
    return "\n".join(lines) + "\n", summary


def dry_run_text() -> str:
    prune_all, quant_all, distill_all, audit = _load_method_rows()
    recovery_runs, _ = _load_recovery_rows()
    prune = [row for row in prune_all if row["precliff_status"] == "pre_cliff"]
    quant = [row for row in quant_all if row["precliff_status"] == "pre_cliff"]
    distill = [row for row in distill_all if row["precliff_status"] == "pre_cliff"]
    lines = ["V18 dry run — no files written", f"pre-cliff cap: Delta L <= {PRECLIFF_CAP} nat",
             f"pruning: {len(prune)} fit cells / {len(prune_all)} non-anchor; "
             f"{len({row['model'] for row in prune_all})} models / {len({row['family'] for row in prune_all})} families",
             "  by density: " + ", ".join(f"{key:g}={value}" for key, value in sorted(Counter(float(row['density']) for row in prune).items(), reverse=True)),
             f"quantization: {len(quant)} fit cells / {len(quant_all)} non-anchor; "
             f"{len({row['model'] for row in quant_all})} models",
             "  by bit: " + ", ".join(f"int{int(key)}={value}" for key, value in sorted(Counter(float(row['bits']) for row in quant).items(), reverse=True)),
             f"distillation: {len(distill)} fit cells; "
             f"{len({row['model'] for row in distill})} student sizes",
             "  students: " + ", ".join(sorted({str(row['model']) for row in distill})),
             "recovery (fit-eligible points after cap):"]
    for source, rows in sorted(recovery_runs.items()):
        lines.append("  " + source + ": " + ", ".join(
            f"{capability}={sum(row['precliff_status']=='pre_cliff' and row['capability']==capability for row in rows)}"
            for capability in CAPABILITIES))
    precliff_audit = audit["precliff"]
    lines.append("excluded post-cliff cells: " + ", ".join(
        f"{method}={count}" for method, count in precliff_audit["postcliff_cells"].items()
        if method in {"pruning", "quantization", v17.DISTILL_SOURCE}))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="list available fit points per arm without writing")
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args()
    if args.bootstrap_resamples < 100:
        parser.error("--bootstrap-resamples must be at least 100")
    if args.dry_run:
        print(dry_run_text())
        return
    report, _ = build_report(args.bootstrap_resamples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
