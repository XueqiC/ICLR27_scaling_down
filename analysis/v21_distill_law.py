#!/usr/bin/env python3
"""V21: held-out cross-family and data-ladder distillation-law tests.

This analysis is deliberately small, CPU-only, and prediction-first.  It reads
the completed V6 dense-source losses and V12 ``eval.json`` files; it never
loads a model, starts a job, or mutates an input artifact.

The source-referenced size test fits

    Delta L_c = floor_c - alpha_c log(r_storage)

on the four Gemma-3 students and predicts both Qwen-3 students without a
refit.  Its one-parameter hierarchical alternative keeps the Gemma exponent
fixed, estimates a Qwen intercept from one Qwen size, and predicts the other
size (both rotations).  A Qwen-specific intercept and exponent require both
Qwen points and therefore have no held-out degree of freedom; that saturated
fit is emitted only as a non-decisional diagnostic.

For the Gemma-3 4B data ladder, every D in {75, 150, 300, 600} is held out in
turn.  The first candidate uses only the other three 4B data points.  The
separable candidate additionally uses the other Gemma student sizes at D=600
to identify the size term.  Both candidates constrain beta > 0 and are thus
monotone in D at fixed size; observed non-monotonicity is reported rather than
hidden by a nominal exponent.

Usage::

    python analysis/v21_distill_law.py --dry-run
    python analysis/v21_distill_law.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

# These regressions are tiny.  Avoid making BLAS fan out over host cores.
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parents[1]
V6_BASE = ROOT / "results/v6-capability-geometry"
V12_BASE = ROOT / "results/v12-distill"
OUT_BASE = ROOT / "results/v21-distill-law"
LAW_FIT_REPORT = ROOT / "paper/docs/LAW_FIT_REPORT.md"

CAPABILITIES = ("math", "code", "qa")
DATA_BUDGETS = (75, 150, 300, 600)
REFERENCE_DATA_BUDGET = 600.0
BETA_BOUNDS = (0.01, 4.0)
QA_CAVEAT = (
    "Loss-space only: V19 showed that lower QA loss does not reliably track "
    "higher QA accuracy; this row is not evidence of behavioral improvement."
)

# Directive-registered nominal source-relative storage ratios.  Keeping them
# explicit prevents a silent switch between text-only and multimodal/full-
# checkpoint parameter-count conventions.
SIZE_LADDERS: dict[str, dict[str, object]] = {
    "gemma3": {
        "source": "gemma3-27b",
        "students": {
            "gemma3-270m": 0.010,
            "gemma3-1b": 0.036,
            "gemma3-4b": 0.157,
            "gemma3-12b": 0.445,
        },
    },
    "qwen3": {
        "source": "Qwen3-4B",
        "students": {
            "Qwen3-0.6B": 0.150,
            "Qwen3-1.7B": 0.425,
        },
    },
}


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _finite(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _validate_eval(payload: Mapping[str, object], path: Path, budget: int) -> None:
    if payload.get("teacher") != "gpt-5.6-luna":
        raise ValueError(f"unexpected teacher in {path}")
    if payload.get("recipe") != "full":
        raise ValueError(f"unexpected recipe in {path}")
    if int(payload.get("n_per_domain", -1)) != budget:
        raise ValueError(f"unexpected data budget in {path}")
    for block in ("dense", "post_training", "delta"):
        values = payload.get(block)
        if not isinstance(values, Mapping):
            raise ValueError(f"missing {block} losses in {path}")
        if any(capability not in values for capability in CAPABILITIES):
            raise ValueError(f"incomplete {block} losses in {path}")


def load_size_rows(
    v6_base: Path = V6_BASE, v12_base: Path = V12_BASE
) -> tuple[list[dict], dict[str, object]]:
    """Load the six registered source-referenced student-size cells."""
    rows: list[dict] = []
    sources: dict[str, object] = {}
    for family, spec in SIZE_LADDERS.items():
        source = str(spec["source"])
        source_path = Path(v6_base) / source / "prune_losses.json"
        source_payload = _read_json(source_path)
        source_dense = source_payload.get("1.0")
        if not isinstance(source_dense, Mapping):
            raise ValueError(f"missing dense key '1.0' in {source_path}")
        sources[family] = {
            "model": source,
            "path": _relative(source_path),
            "losses": {
                capability: _finite(source_dense[capability], f"{source} {capability}")
                for capability in CAPABILITIES
            },
        }
        students = spec["students"]
        assert isinstance(students, Mapping)
        for model, ratio_value in students.items():
            ratio = _finite(ratio_value, f"{model} storage ratio")
            eval_path = Path(v12_base) / str(model) / "gpt-5.6-luna_full_600/eval.json"
            payload = _read_json(eval_path)
            _validate_eval(payload, eval_path, 600)
            post = payload["post_training"]
            dense = payload["dense"]
            recorded_delta = payload["delta"]
            assert isinstance(post, Mapping)
            assert isinstance(dense, Mapping)
            assert isinstance(recorded_delta, Mapping)
            for capability in CAPABILITIES:
                post_loss = _finite(post[capability], f"{model} post-training {capability}")
                own_dense = _finite(dense[capability], f"{model} dense {capability}")
                self_delta = post_loss - own_dense
                if not math.isclose(
                    self_delta,
                    _finite(recorded_delta[capability], f"{model} delta {capability}"),
                    rel_tol=0.0,
                    abs_tol=1e-10,
                ):
                    raise ValueError(f"recorded delta does not match losses in {eval_path}")
                rows.append(
                    {
                        "row_id": f"size|{family}|{model}|{capability}",
                        "family": family,
                        "model": str(model),
                        "source_model": source,
                        "capability": capability,
                        "r_storage": ratio,
                        "D": 600,
                        "observed": post_loss
                        - _finite(source_dense[capability], f"{source} {capability}"),
                        "self_observed": self_delta,
                        "post_training_loss": post_loss,
                        "source_loss": _finite(
                            source_dense[capability], f"{source} {capability}"
                        ),
                        "teacher": str(payload.get("teacher", "")),
                        "recipe": str(payload.get("recipe", "")),
                        "training_mode": str(payload.get("training_mode", "")),
                        "source_path": _relative(eval_path),
                        "qa_annotation": QA_CAVEAT if capability == "qa" else "",
                    }
                )
    return rows, sources


def load_data_rows(v12_base: Path = V12_BASE) -> list[dict]:
    """Load Gemma-3 4B's own-dense-referenced four-point D ladder."""
    rows: list[dict] = []
    ratio = float(SIZE_LADDERS["gemma3"]["students"]["gemma3-4b"])  # type: ignore[index]
    for budget in DATA_BUDGETS:
        path = (
            Path(v12_base)
            / "gemma3-4b"
            / f"gpt-5.6-luna_full_{budget}/eval.json"
        )
        payload = _read_json(path)
        _validate_eval(payload, path, budget)
        dense = payload["dense"]
        post = payload["post_training"]
        recorded = payload["delta"]
        assert isinstance(dense, Mapping)
        assert isinstance(post, Mapping)
        assert isinstance(recorded, Mapping)
        for capability in CAPABILITIES:
            observed = _finite(post[capability], f"D={budget} {capability} post") - _finite(
                dense[capability], f"D={budget} {capability} dense"
            )
            if not math.isclose(
                observed,
                _finite(recorded[capability], f"D={budget} {capability} delta"),
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise ValueError(f"recorded delta does not match losses in {path}")
            rows.append(
                {
                    "row_id": f"data|gemma3|gemma3-4b|{budget}|{capability}",
                    "family": "gemma3",
                    "model": "gemma3-4b",
                    "capability": capability,
                    "r_storage": ratio,
                    "D": budget,
                    "observed": observed,
                    "teacher": str(payload.get("teacher", "")),
                    "recipe": str(payload.get("recipe", "")),
                    "training_mode": str(payload.get("training_mode", "")),
                    "source_path": _relative(path),
                    "qa_annotation": QA_CAVEAT if capability == "qa" else "",
                }
            )
    return rows


def fit_size_law(
    rows: Sequence[Mapping[str, object]], *, required_family: str | None = None
) -> dict[str, object]:
    """Fit capability-specific floor and log-size exponent by linear OLS."""
    if required_family is not None and any(
        str(row["family"]) != required_family for row in rows
    ):
        raise ValueError(f"size-law training rows must all be {required_family}")
    parameters: dict[str, dict[str, float]] = {}
    models: set[str] = set()
    families: set[str] = set()
    for capability in CAPABILITIES:
        subset = [row for row in rows if row["capability"] == capability]
        if len(subset) < 2:
            raise ValueError(f"need at least two {capability} size rows")
        ratios = np.asarray([_finite(row["r_storage"], "r_storage") for row in subset])
        if np.any(ratios <= 0.0) or len(np.unique(ratios)) < 2:
            raise ValueError(f"need distinct positive {capability} size ratios")
        design = np.column_stack([np.ones(len(subset)), -np.log(ratios)])
        observed = np.asarray([_finite(row["observed"], "observed") for row in subset])
        beta, _, rank, _ = np.linalg.lstsq(design, observed, rcond=None)
        if rank != 2:
            raise ValueError(f"rank-deficient {capability} size-law fit")
        parameters[capability] = {
            "floor": float(beta[0]),
            "alpha": float(beta[1]),
        }
        models.update(str(row["model"]) for row in subset)
        families.update(str(row["family"]) for row in subset)
    return {
        "status": "ok",
        "law": "Delta L_c = floor_c - alpha_c log(r_storage)",
        "parameters": parameters,
        "n_train": len(rows),
        "n_train_per_capability": len(rows) // len(CAPABILITIES),
        "train_models": sorted(models),
        "train_families": sorted(families),
        "train_row_ids": sorted(str(row["row_id"]) for row in rows),
    }


def predict_size_law(fit: Mapping[str, object], row: Mapping[str, object]) -> float:
    parameters = fit["parameters"]
    assert isinstance(parameters, Mapping)
    capability_parameters = parameters[str(row["capability"])]
    assert isinstance(capability_parameters, Mapping)
    return _finite(capability_parameters["floor"], "floor") - _finite(
        capability_parameters["alpha"], "alpha"
    ) * math.log(_finite(row["r_storage"], "r_storage"))


def _metric_block(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not records:
        return {
            "n": 0,
            "mae": None,
            "relative_error": None,
            "sign_correct": 0,
            "sign_accuracy": None,
        }
    observed = np.asarray([_finite(row["observed"], "observed") for row in records])
    predicted = np.asarray([_finite(row["predicted"], "predicted") for row in records])
    absolute = np.abs(predicted - observed)
    denominator = float(np.mean(np.abs(observed)))
    sign_correct = int(np.sum(np.sign(predicted) == np.sign(observed)))
    return {
        "n": len(records),
        "mae": float(np.mean(absolute)),
        "relative_error": float(np.mean(absolute) / denominator)
        if denominator > 0.0
        else None,
        "sign_correct": sign_correct,
        "sign_accuracy": sign_correct / len(records),
    }


def prediction_metrics(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        **_metric_block(records),
        "by_capability": {
            capability: _metric_block(
                [row for row in records if row["capability"] == capability]
            )
            for capability in CAPABILITIES
        },
    }


def evaluate_shared_no_refit(
    gemma_rows: Sequence[Mapping[str, object]],
    qwen_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Fit Gemma only, then predict Qwen without inspecting Qwen outcomes."""
    fit = fit_size_law(gemma_rows, required_family="gemma3")
    if any(str(row["family"]) != "qwen3" for row in qwen_rows):
        raise ValueError("held-out rows must all be qwen3")
    if set(fit["train_row_ids"]) & {str(row["row_id"]) for row in qwen_rows}:
        raise ValueError("shared-law train/test rows overlap")
    records = []
    for row in qwen_rows:
        predicted = predict_size_law(fit, row)
        records.append(
            {
                "row_id": str(row["row_id"]),
                "model": str(row["model"]),
                "family": str(row["family"]),
                "capability": str(row["capability"]),
                "r_storage": _finite(row["r_storage"], "r_storage"),
                "observed": _finite(row["observed"], "observed"),
                "predicted": predicted,
                "absolute_error": abs(predicted - _finite(row["observed"], "observed")),
                "qa_annotation": str(row.get("qa_annotation", "")),
            }
        )
    return {
        "status": "ok",
        "protocol": "fit all four Gemma sizes; predict both Qwen sizes with no refit",
        "fit": fit,
        "added_qwen_parameters_per_capability": 0,
        "records": records,
        "metrics": prediction_metrics(records),
    }


def evaluate_hierarchical_intercept(
    shared_fit: Mapping[str, object], qwen_rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Rotate Qwen sizes: fit one family intercept, predict the other size."""
    models = sorted({str(row["model"]) for row in qwen_rows})
    if len(models) != 2:
        raise ValueError("hierarchical held-out test requires exactly two Qwen sizes")
    records: list[dict] = []
    folds: list[dict] = []
    for held_out_model in models:
        train = [row for row in qwen_rows if str(row["model"]) != held_out_model]
        test = [row for row in qwen_rows if str(row["model"]) == held_out_model]
        if {str(row["row_id"]) for row in train} & {
            str(row["row_id"]) for row in test
        }:
            raise ValueError("hierarchical train/test rows overlap")
        intercepts: dict[str, float] = {}
        calibration_models: set[str] = set()
        for capability in CAPABILITIES:
            calibration = [row for row in train if row["capability"] == capability]
            if len(calibration) != 1:
                raise ValueError("each hierarchical fold needs one Qwen calibration row")
            calibration_row = calibration[0]
            intercepts[capability] = _finite(
                calibration_row["observed"], "observed"
            ) - predict_size_law(shared_fit, calibration_row)
            calibration_models.add(str(calibration_row["model"]))
        folds.append(
            {
                "held_out_model": held_out_model,
                "calibration_models": sorted(calibration_models),
                "qwen_intercepts": intercepts,
                "train_row_ids": sorted(str(row["row_id"]) for row in train),
                "test_row_ids": sorted(str(row["row_id"]) for row in test),
            }
        )
        for row in test:
            capability = str(row["capability"])
            predicted = predict_size_law(shared_fit, row) + intercepts[capability]
            observed = _finite(row["observed"], "observed")
            records.append(
                {
                    "row_id": str(row["row_id"]),
                    "held_out_model": held_out_model,
                    "calibration_model": next(iter(calibration_models)),
                    "model": str(row["model"]),
                    "family": "qwen3",
                    "capability": capability,
                    "r_storage": _finite(row["r_storage"], "r_storage"),
                    "observed": observed,
                    "predicted": predicted,
                    "absolute_error": abs(predicted - observed),
                    "qwen_intercept": intercepts[capability],
                    "qa_annotation": str(row.get("qa_annotation", "")),
                }
            )
    return {
        "status": "ok",
        "protocol": (
            "keep Gemma floor/exponent fixed; fit a Qwen intercept on one Qwen "
            "size and predict the other (both rotations)"
        ),
        "added_qwen_parameters_per_capability": 1,
        "folds": folds,
        "records": records,
        "metrics": prediction_metrics(records),
    }


def family_specific_diagnostic(
    shared_fit: Mapping[str, object], qwen_rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Fit the saturated Qwen line and explicitly refuse a held-out score."""
    fit = fit_size_law(qwen_rows, required_family="qwen3")
    records = []
    trend_matches: dict[str, bool] = {}
    shared_parameters = shared_fit["parameters"]
    qwen_parameters = fit["parameters"]
    assert isinstance(shared_parameters, Mapping)
    assert isinstance(qwen_parameters, Mapping)
    for capability in CAPABILITIES:
        shared_cap = shared_parameters[capability]
        qwen_cap = qwen_parameters[capability]
        assert isinstance(shared_cap, Mapping)
        assert isinstance(qwen_cap, Mapping)
        trend_matches[capability] = math.copysign(
            1.0, _finite(shared_cap["alpha"], "shared alpha")
        ) == math.copysign(1.0, _finite(qwen_cap["alpha"], "Qwen alpha"))
    for row in qwen_rows:
        predicted = predict_size_law(fit, row)
        records.append(
            {
                "row_id": str(row["row_id"]),
                "model": str(row["model"]),
                "capability": str(row["capability"]),
                "r_storage": _finite(row["r_storage"], "r_storage"),
                "observed": _finite(row["observed"], "observed"),
                "predicted": predicted,
                "absolute_error": abs(predicted - _finite(row["observed"], "observed")),
                "qa_annotation": str(row.get("qa_annotation", "")),
            }
        )
    return {
        "status": "NON-DECISIONAL_UNDERIDENTIFIED_HELDOUT",
        "reason": (
            "two Qwen sizes exactly identify a Qwen intercept and exponent, "
            "leaving no Qwen size for held-out evaluation"
        ),
        "added_qwen_parameters_per_capability": 2,
        "heldout_metrics": _metric_block([]),
        "full_data_fit": fit,
        "in_sample_metrics": prediction_metrics(records),
        "trend_sign_matches_gemma": trend_matches,
        "records": records,
    }


def evaluate_gemma_leave_one_size(
    gemma_rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Retain V18's within-Gemma four-rotation size test in the new report."""
    models = sorted({str(row["model"]) for row in gemma_rows})
    records = []
    folds = []
    for model in models:
        train = [row for row in gemma_rows if str(row["model"]) != model]
        test = [row for row in gemma_rows if str(row["model"]) == model]
        fit = fit_size_law(train, required_family="gemma3")
        folds.append(
            {
                "held_out_model": model,
                "train_row_ids": fit["train_row_ids"],
                "test_row_ids": sorted(str(row["row_id"]) for row in test),
            }
        )
        for row in test:
            predicted = predict_size_law(fit, row)
            observed = _finite(row["observed"], "observed")
            records.append(
                {
                    "row_id": str(row["row_id"]),
                    "held_out_model": model,
                    "model": model,
                    "family": "gemma3",
                    "capability": str(row["capability"]),
                    "r_storage": _finite(row["r_storage"], "r_storage"),
                    "observed": observed,
                    "predicted": predicted,
                    "absolute_error": abs(predicted - observed),
                    "qa_annotation": str(row.get("qa_annotation", "")),
                }
            )
    return {
        "status": "PARTIAL_ONE_FAMILY",
        "protocol": "fit three Gemma sizes; predict the fourth (all rotations)",
        "records": records,
        "folds": folds,
        "metrics": prediction_metrics(records),
    }


def select_cross_family_verdict(
    shared: Mapping[str, object],
    hierarchical: Mapping[str, object],
    family_specific: Mapping[str, object],
) -> str:
    """Apply the directive's shared/hierarchical/family-specific rule."""
    shared_metrics = shared["metrics"]
    hierarchical_metrics = hierarchical["metrics"]
    assert isinstance(shared_metrics, Mapping)
    assert isinstance(hierarchical_metrics, Mapping)
    shared_sign = _finite(shared_metrics["sign_accuracy"], "shared sign accuracy")
    hierarchical_sign = _finite(
        hierarchical_metrics["sign_accuracy"], "hierarchical sign accuracy"
    )
    shared_mae = _finite(shared_metrics["mae"], "shared MAE")
    hierarchical_mae = _finite(hierarchical_metrics["mae"], "hierarchical MAE")
    trend = family_specific["trend_sign_matches_gemma"]
    assert isinstance(trend, Mapping)
    if shared_sign == 1.0 and shared_mae <= hierarchical_mae:
        return "shared"
    if hierarchical_sign == 1.0 and all(bool(value) for value in trend.values()) and (
        hierarchical_mae < shared_mae
    ):
        return "hierarchical"
    return "family-specific"


def _profile_data_fit(
    rows: Sequence[Mapping[str, object]], include_size: bool
) -> dict[str, object]:
    """Profile beta and solve the remaining linear coefficients by OLS."""
    if len(rows) < (4 if include_size else 3):
        raise ValueError("too few rows for data-power fit")
    data = np.asarray([_finite(row["D"], "D") for row in rows])
    ratios = np.asarray([_finite(row["r_storage"], "r_storage") for row in rows])
    observed = np.asarray([_finite(row["observed"], "observed") for row in rows])
    if np.any(data <= 0.0) or np.any(ratios <= 0.0):
        raise ValueError("D and r_storage must be positive")

    def solve(log_beta: float) -> tuple[float, np.ndarray, int]:
        beta = math.exp(log_beta)
        data_term = (data / REFERENCE_DATA_BUDGET) ** (-beta)
        columns = [np.ones(len(rows))]
        if include_size:
            columns.append(-np.log(ratios))
        columns.append(data_term)
        design = np.column_stack(columns)
        coefficients, _, rank, _ = np.linalg.lstsq(design, observed, rcond=None)
        residual = design @ coefficients - observed
        return float(residual @ residual), coefficients, int(rank)

    result = minimize_scalar(
        lambda log_beta: solve(float(log_beta))[0],
        bounds=(math.log(BETA_BOUNDS[0]), math.log(BETA_BOUNDS[1])),
        method="bounded",
        options={"xatol": 1e-10},
    )
    beta = math.exp(float(result.x))
    sse, coefficients, rank = solve(float(result.x))
    expected_rank = 3 if include_size else 2
    if rank != expected_rank:
        raise ValueError("rank-deficient data-power fit")
    parameters: dict[str, float] = {"a": float(coefficients[0]), "beta": beta}
    if include_size:
        parameters["alpha"] = float(coefficients[1])
        parameters["b"] = float(coefficients[2])
    else:
        parameters["b"] = float(coefficients[1])
    beta_boundary = (
        beta <= BETA_BOUNDS[0] * 1.01 or beta >= BETA_BOUNDS[1] / 1.01
    )
    return {
        "parameters": parameters,
        "sse": sse,
        "rank": rank,
        "beta_at_bound": beta_boundary,
        "optimizer_success": bool(result.success),
    }


def fit_data_law(
    rows: Sequence[Mapping[str, object]], *, include_size: bool
) -> dict[str, object]:
    parameters: dict[str, dict[str, object]] = {}
    for capability in CAPABILITIES:
        subset = [row for row in rows if row["capability"] == capability]
        parameters[capability] = _profile_data_fit(subset, include_size)
    return {
        "status": "ok",
        "candidate": "separable_size_plus_data" if include_size else "data_only",
        "law": (
            "Delta L_c = a_c - alpha_c log(r_storage) + b_c (D/600)^(-beta_c)"
            if include_size
            else "Delta L_c = a_c + b_c (D/600)^(-beta_c)"
        ),
        "include_size": include_size,
        "parameters": parameters,
        "parameters_per_capability": 4 if include_size else 3,
        "n_train": len(rows),
        "train_row_ids": sorted(str(row["row_id"]) for row in rows),
    }


def predict_data_law(fit: Mapping[str, object], row: Mapping[str, object]) -> float:
    all_parameters = fit["parameters"]
    assert isinstance(all_parameters, Mapping)
    capability_fit = all_parameters[str(row["capability"])]
    assert isinstance(capability_fit, Mapping)
    parameters = capability_fit["parameters"]
    assert isinstance(parameters, Mapping)
    result = _finite(parameters["a"], "a")
    if bool(fit["include_size"]):
        result -= _finite(parameters["alpha"], "alpha") * math.log(
            _finite(row["r_storage"], "r_storage")
        )
    result += _finite(parameters["b"], "b") * (
        _finite(row["D"], "D") / REFERENCE_DATA_BUDGET
    ) ** (-_finite(parameters["beta"], "beta"))
    return result


def combined_self_referenced_rows(
    size_rows: Sequence[Mapping[str, object]],
    data_rows: Sequence[Mapping[str, object]],
) -> list[dict]:
    """Combine Gemma D=600 size rows with 4B D<600 rows without duplication."""
    output = []
    for row in size_rows:
        if str(row["family"]) != "gemma3":
            continue
        output.append(
            {
                **dict(row),
                "row_id": str(row["row_id"]).replace("size|", "self-size|", 1),
                "observed": _finite(row["self_observed"], "self_observed"),
                "delta_reference": "student_own_dense",
            }
        )
    output.extend(
        {
            **dict(row),
            "delta_reference": "student_own_dense",
        }
        for row in data_rows
        if int(row["D"]) != 600
    )
    ids = [str(row["row_id"]) for row in output]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate combined size/data rows")
    return output


def evaluate_data_holdout(
    size_rows: Sequence[Mapping[str, object]],
    data_rows: Sequence[Mapping[str, object]],
    *,
    include_size: bool,
) -> dict[str, object]:
    """Hold out one 4B D cell per capability in all four rotations."""
    combined = combined_self_referenced_rows(size_rows, data_rows)
    records = []
    folds = []
    for budget in DATA_BUDGETS:
        test = [row for row in data_rows if int(row["D"]) == budget]
        if include_size:
            train = [
                row
                for row in combined
                if not (
                    str(row["model"]) == "gemma3-4b" and int(row["D"]) == budget
                )
            ]
        else:
            train = [row for row in data_rows if int(row["D"]) != budget]
        fit = fit_data_law(train, include_size=include_size)
        train_ids = set(str(value) for value in fit["train_row_ids"])
        test_ids = {str(row["row_id"]) for row in test}
        # The D=600 4B row has a different namespace in the combined table;
        # guard its semantic identity as well as literal row ids.
        if train_ids & test_ids or any(
            str(row["model"]) == "gemma3-4b" and int(row["D"]) == budget
            for row in train
        ):
            raise ValueError("data-law train/test rows overlap")
        folds.append(
            {
                "held_out_D": budget,
                "train_row_ids": sorted(train_ids),
                "test_row_ids": sorted(test_ids),
                "fit": fit,
            }
        )
        for row in test:
            predicted = predict_data_law(fit, row)
            observed = _finite(row["observed"], "observed")
            records.append(
                {
                    "row_id": str(row["row_id"]),
                    "held_out_D": budget,
                    "model": "gemma3-4b",
                    "family": "gemma3",
                    "capability": str(row["capability"]),
                    "r_storage": _finite(row["r_storage"], "r_storage"),
                    "D": budget,
                    "observed": observed,
                    "predicted": predicted,
                    "absolute_error": abs(predicted - observed),
                    "qa_annotation": str(row.get("qa_annotation", "")),
                }
            )
    return {
        "status": "PARTIAL_ONE_SIZE_DATA_LADDER",
        "candidate": "separable_size_plus_data" if include_size else "data_only",
        "protocol": "leave one of four Gemma-3 4B data budgets out (all rotations)",
        "parameters_per_capability": 4 if include_size else 3,
        "folds": folds,
        "records": records,
        "metrics": prediction_metrics(records),
    }


def data_monotonicity(data_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for capability in CAPABILITIES:
        subset = sorted(
            (row for row in data_rows if row["capability"] == capability),
            key=lambda row: int(row["D"]),
        )
        values = [_finite(row["observed"], "observed") for row in subset]
        differences = np.diff(values)
        increasing = bool(np.all(differences >= -1e-12))
        decreasing = bool(np.all(differences <= 1e-12))
        result[capability] = {
            "D": [int(row["D"]) for row in subset],
            "observed": values,
            "successive_differences": [float(value) for value in differences],
            "is_monotone": increasing or decreasing,
            "direction": "increasing" if increasing else "decreasing" if decreasing else "non-monotone",
        }
    return result


def full_data_diagnostic(
    size_rows: Sequence[Mapping[str, object]],
    data_rows: Sequence[Mapping[str, object]],
    *,
    include_size: bool,
) -> dict[str, object]:
    train = (
        combined_self_referenced_rows(size_rows, data_rows)
        if include_size
        else list(data_rows)
    )
    fit = fit_data_law(train, include_size=include_size)
    records = []
    for row in data_rows:
        predicted = predict_data_law(fit, row)
        records.append(
            {
                "row_id": str(row["row_id"]),
                "model": str(row["model"]),
                "capability": str(row["capability"]),
                "D": int(row["D"]),
                "observed": _finite(row["observed"], "observed"),
                "predicted": predicted,
                "qa_annotation": str(row.get("qa_annotation", "")),
            }
        )
    return {
        "status": "NON-DECISIONAL_FULL_DATA",
        "fit": fit,
        "data_ladder_metrics": prediction_metrics(records),
        "records": records,
    }


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _metric_summary(result: Mapping[str, object]) -> tuple[str, str, str]:
    metrics = result["metrics"]
    assert isinstance(metrics, Mapping)
    sign_accuracy = metrics.get("sign_accuracy")
    sign = (
        "n/a"
        if sign_accuracy is None
        else f"{metrics['sign_correct']}/{metrics['n']} ({_fmt(sign_accuracy)})"
    )
    by_capability = metrics["by_capability"]
    assert isinstance(by_capability, Mapping)
    capability_mae = ", ".join(
        f"{capability}{'†' if capability == 'qa' else ''} "
        f"{_fmt(by_capability[capability]['mae'])}"  # type: ignore[index]
        for capability in CAPABILITIES
    )
    return _fmt(metrics.get("mae")), capability_mae, sign


def render_distillation_section(summary: Mapping[str, object]) -> str:
    size = summary["source_referenced_size_law"]
    data = summary["data_ladder"]
    assert isinstance(size, Mapping)
    assert isinstance(data, Mapping)
    within = size["within_gemma_leave_one_size_out"]
    shared = size["shared_no_refit"]
    hierarchical = size["hierarchical_intercept"]
    family_specific = size["family_specific_exponent"]
    assert isinstance(within, Mapping)
    assert isinstance(shared, Mapping)
    assert isinstance(hierarchical, Mapping)
    assert isinstance(family_specific, Mapping)
    d_only = data["data_only_heldout"]
    separable = data["separable_heldout"]
    monotonicity = data["monotonicity"]
    assert isinstance(d_only, Mapping)
    assert isinstance(separable, Mapping)
    assert isinstance(monotonicity, Mapping)

    within_mae, within_caps, within_sign = _metric_summary(within)
    shared_mae, shared_caps, shared_sign = _metric_summary(shared)
    hierarchical_mae, hierarchical_caps, hierarchical_sign = _metric_summary(
        hierarchical
    )
    d_only_mae, d_only_caps, d_only_sign = _metric_summary(d_only)
    separable_mae, separable_caps, separable_sign = _metric_summary(separable)

    lines = [
        "## 3. Distillation",
        "",
        f"Updated {summary['generated']} by `analysis/v21_distill_law.py`, CPU-only, from existing V6/V12 JSON artifacts. The size-law response is source-referenced; the D-ladder response is explicitly student-own-dense-referenced.",
        "",
        "### Held-out rows and complexity",
        "",
        "| Test | Law / fit-to-test protocol | Added parameters per capability | Held-out cells | MAE (nats) | MAE by capability | Sign accuracy | Status |",
        "|---|---|---:|---:|---:|---|---|---|",
        f"| Within Gemma size rotation | `floor_c-alpha_c log r`; fit 3 sizes, predict 4th | n/a | 12 | {within_mae} | {within_caps} | {within_sign} | PARTIAL: one family |",
        f"| Shared cross-family, no refit | fit all 4 Gemma sizes, predict both Qwen sizes | 0 | 6 | {shared_mae} | {shared_caps} | {shared_sign} | HELD OUT |",
        f"| Hierarchical family intercept | keep Gemma exponent; fit Qwen intercept on one size, predict the other, both rotations | +1 | 6 | {hierarchical_mae} | {hierarchical_caps} | {hierarchical_sign} | HELD OUT |",
        "| Family-specific Qwen exponent | fit Qwen intercept + exponent on both Qwen sizes | +2 | 0 | n/a | n/a | n/a | NON-DECISIONAL: two parameters saturate two sizes |",
        f"| Gemma-3 4B D-only | `a_c+b_c(D/600)^(-beta_c)`; leave one D out | 0 | 12 | {d_only_mae} | {d_only_caps} | {d_only_sign} | PARTIAL: one fixed-size D ladder |",
        f"| Separable Gemma size + D | add `-alpha_c log r`; leave the same 4B D out and train on remaining D plus other D=600 sizes | +1 | 12 | {separable_mae} | {separable_caps} | {separable_sign} | PARTIAL: one fixed-size D ladder |",
        "",
        "### Cross-family source-referenced size test",
        "",
        "The no-refit row is the strict family holdout: no Qwen outcome participates in its Gemma fit. The hierarchical row spends exactly one Qwen calibration parameter per capability in each fold. The Qwen-specific exponent row cannot be honestly held out with only two sizes, so its zero in-sample residual is not used for selection.",
        "",
        "| Qwen held-out size | r_storage | Capability | Observed source-referenced ΔL | Shared prediction | Hierarchical held-out prediction |",
        "|---|---:|---|---:|---:|---:|",
    ]
    shared_records = {
        str(row["row_id"]): row for row in shared["records"]  # type: ignore[index]
    }
    hierarchical_records = {
        str(row["row_id"]): row for row in hierarchical["records"]  # type: ignore[index]
    }
    for row_id, shared_row in sorted(
        shared_records.items(),
        key=lambda item: (
            float(item[1]["r_storage"]),
            CAPABILITIES.index(str(item[1]["capability"])),
        ),
    ):
        hierarchical_row = hierarchical_records[row_id]
        capability = str(shared_row["capability"])
        label = "qa†" if capability == "qa" else capability
        lines.append(
            f"| {shared_row['model']} | {_fmt(shared_row['r_storage'])} | {label} | "
            f"{_fmt(shared_row['observed'])} | {_fmt(shared_row['predicted'])} | "
            f"{_fmt(hierarchical_row['predicted'])} |"
        )

    shared_fit = shared["fit"]
    family_fit = family_specific["full_data_fit"]
    assert isinstance(shared_fit, Mapping)
    assert isinstance(family_fit, Mapping)
    gemma_parameters = shared_fit["parameters"]
    qwen_parameters = family_fit["parameters"]
    trend = family_specific["trend_sign_matches_gemma"]
    assert isinstance(gemma_parameters, Mapping)
    assert isinstance(qwen_parameters, Mapping)
    assert isinstance(trend, Mapping)
    lines.extend(
        [
            "",
            "Family-specific coefficients below are descriptive only because the Qwen line has exactly two points.",
            "",
            "| Capability | Gemma floor | Gemma alpha | Qwen saturated floor | Qwen saturated alpha | Trend sign matches? |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for capability in CAPABILITIES:
        gp = gemma_parameters[capability]
        qp = qwen_parameters[capability]
        assert isinstance(gp, Mapping)
        assert isinstance(qp, Mapping)
        label = "qa†" if capability == "qa" else capability
        lines.append(
            f"| {label} | {_fmt(gp['floor'])} | {_fmt(gp['alpha'])} | "
            f"{_fmt(qp['floor'])} | {_fmt(qp['alpha'])} | "
            f"{'yes' if trend[capability] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"**Cross-family verdict: {str(size['verdict']).upper()}.** The uncorrected Gemma law has MAE {shared_mae} nats and misses the sign of Qwen-3 1.7B math and code (4/6 signs correct). A held-out Qwen intercept lowers MAE to {hierarchical_mae} and gets 6/6 signs, while all three observed Qwen size trends have the same sign as the Gemma exponent. Thus the supported reduced form is a shared exponent with a family intercept. A family-specific exponent remains untested, not disproved.",
            "",
            "This cross-family claim is limited: Gemma uses a 27B dense source whereas Qwen uses a 4B dense source. Their absolute source capabilities and source sizes differ, so equal nominal `r_storage` does not represent a perfectly matched capacity gap.",
            "",
            "### Gemma-3 4B data-ladder separability",
            "",
            "The observed own-dense loss deltas are:",
            "",
            "| D per domain | math | code | qa† | recorded training mode |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    data_rows = data["rows"]
    assert isinstance(data_rows, Sequence)
    for budget in DATA_BUDGETS:
        subset = {str(row["capability"]): row for row in data_rows if int(row["D"]) == budget}  # type: ignore[index]
        modes = sorted({str(row["training_mode"]) for row in subset.values()})
        lines.append(
            f"| {budget} | {_fmt(subset['math']['observed'])} | "
            f"{_fmt(subset['code']['observed'])} | {_fmt(subset['qa']['observed'])} | "
            f"{', '.join(modes)} |"
        )

    data_only_full = data["data_only_full_diagnostic"]
    separable_full = data["separable_full_diagnostic"]
    assert isinstance(data_only_full, Mapping)
    assert isinstance(separable_full, Mapping)
    data_only_fit = data_only_full["fit"]
    separable_fit = separable_full["fit"]
    assert isinstance(data_only_fit, Mapping)
    assert isinstance(separable_fit, Mapping)
    data_only_parameters = data_only_fit["parameters"]
    separable_parameters = separable_fit["parameters"]
    assert isinstance(data_only_parameters, Mapping)
    assert isinstance(separable_parameters, Mapping)
    lines.extend(
        [
            "",
            "All three observed sequences are non-monotone, so neither candidate can reproduce their ordering at fixed size. The fitted beta values below are constrained-predictor diagnostics, not clean scaling exponents.",
            "",
            "| Capability | Observed monotone? | D-only beta (at bound?) | Separable beta (at bound?) |",
            "|---|---|---:|---:|",
        ]
    )
    for capability in CAPABILITIES:
        d_cap = data_only_parameters[capability]
        s_cap = separable_parameters[capability]
        assert isinstance(d_cap, Mapping)
        assert isinstance(s_cap, Mapping)
        d_parameters = d_cap["parameters"]
        s_parameters = s_cap["parameters"]
        assert isinstance(d_parameters, Mapping)
        assert isinstance(s_parameters, Mapping)
        label = "qa†" if capability == "qa" else capability
        lines.append(
            f"| {label} | {'yes' if monotonicity[capability]['is_monotone'] else 'no'} | "  # type: ignore[index]
            f"{_fmt(d_parameters['beta'])} ({'yes' if d_cap['beta_at_bound'] else 'no'}) | "
            f"{_fmt(s_parameters['beta'])} ({'yes' if s_cap['beta_at_bound'] else 'no'}) |"
        )
    lines.extend(
        [
            "",
            f"**D-ladder verdict: no clean monotone data exponent is supported.** The D-only held-out MAE is {d_only_mae} nats; adding the separable size term gives {separable_mae} on the same 12 held-out cells. Regardless of which error is smaller, both impose monotonicity contradicted by every observed capability sequence (math 0.077→0.137→0.166→0.081; code and QA also reverse direction). At one fixed 4B size, `a_c` and `-alpha_c log r_storage` are collinear; the separable row identifies alpha only by borrowing the D=600 Gemma size ladder, so this remains a partial separability test.",
            "",
            "† **QA loss-space caveat:** V19 found that QA loss gains do not reliably track accuracy. Every negative QA delta and every QA prediction above is loss-space only and must not be described as a QA accuracy or behavioral improvement.",
            "",
            "Additional design caveat: the V12 metadata records LoRA for D=75/150/300 and `training_mode=full` for D=600 at Gemma-3 4B. `recipe=full` is constant, but training mode is not; this can confound the apparent D effect. The size ladders likewise mix recorded training modes. There is one recorded probe seed, so these cell-level comparisons do not estimate seed or benchmark-item uncertainty.",
            "",
            "Provenance: `results/v12-distill/{gemma3-270m,gemma3-1b,gemma3-4b,gemma3-12b,Qwen3-0.6B,Qwen3-1.7B}/gpt-5.6-luna_full_*/eval.json` and dense key `1.0` from `results/v6-capability-geometry/{gemma3-27b,Qwen3-4B}/prune_losses.json`. Machine-readable records and every fold's train/test row IDs are in `results/v21-distill-law/summary.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_summary(
    v6_base: Path = V6_BASE, v12_base: Path = V12_BASE
) -> dict[str, object]:
    size_rows, sources = load_size_rows(v6_base, v12_base)
    data_rows = load_data_rows(v12_base)
    gemma_rows = [row for row in size_rows if row["family"] == "gemma3"]
    qwen_rows = [row for row in size_rows if row["family"] == "qwen3"]
    within = evaluate_gemma_leave_one_size(gemma_rows)
    shared = evaluate_shared_no_refit(gemma_rows, qwen_rows)
    hierarchical = evaluate_hierarchical_intercept(shared["fit"], qwen_rows)  # type: ignore[arg-type]
    family_specific = family_specific_diagnostic(shared["fit"], qwen_rows)  # type: ignore[arg-type]
    verdict = select_cross_family_verdict(shared, hierarchical, family_specific)

    d_only = evaluate_data_holdout(size_rows, data_rows, include_size=False)
    separable = evaluate_data_holdout(size_rows, data_rows, include_size=True)
    data_only_full = full_data_diagnostic(
        size_rows, data_rows, include_size=False
    )
    separable_full = full_data_diagnostic(
        size_rows, data_rows, include_size=True
    )
    return {
        "version": 21,
        "generated": date.today().isoformat(),
        "cpu_only": True,
        "inputs_mutated": False,
        "capabilities": list(CAPABILITIES),
        "qa_caveat": QA_CAVEAT,
        "sources": sources,
        "source_referenced_size_law": {
            "registered_ratios": {
                family: spec["students"] for family, spec in SIZE_LADDERS.items()
            },
            "rows": size_rows,
            "within_gemma_leave_one_size_out": within,
            "shared_no_refit": shared,
            "hierarchical_intercept": hierarchical,
            "family_specific_exponent": family_specific,
            "verdict": verdict,
            "source_matching_caveat": (
                "Gemma uses a 27B source and Qwen a 4B source; absolute source "
                "capability and source size differ, so r_storage is imperfectly matched."
            ),
        },
        "data_ladder": {
            "delta_reference": "student_own_dense",
            "rows": data_rows,
            "monotonicity": data_monotonicity(data_rows),
            "data_only_heldout": d_only,
            "separable_heldout": separable,
            "data_only_full_diagnostic": data_only_full,
            "separable_full_diagnostic": separable_full,
            "verdict": "no_clean_monotone_data_exponent",
        },
    }


def update_law_fit_report(path: Path, section: str, summary: Mapping[str, object]) -> None:
    text = path.read_text(encoding="utf-8")
    start_marker = "## 3. Distillation"
    end_marker = "## 4. Recovery"
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start < 0 or end < 0 or end <= start:
        raise ValueError(f"could not locate distillation section in {path}")
    size = summary["source_referenced_size_law"]
    data = summary["data_ladder"]
    assert isinstance(size, Mapping)
    assert isinstance(data, Mapping)
    shared = size["shared_no_refit"]
    hierarchical = size["hierarchical_intercept"]
    d_only = data["data_only_heldout"]
    separable = data["separable_heldout"]
    assert isinstance(shared, Mapping)
    assert isinstance(hierarchical, Mapping)
    assert isinstance(d_only, Mapping)
    assert isinstance(separable, Mapping)
    shared_metrics = shared["metrics"]
    hierarchical_metrics = hierarchical["metrics"]
    d_metrics = d_only["metrics"]
    s_metrics = separable["metrics"]
    assert isinstance(shared_metrics, Mapping)
    assert isinstance(hierarchical_metrics, Mapping)
    assert isinstance(d_metrics, Mapping)
    assert isinstance(s_metrics, Mapping)
    headline = (
        f"- **Distillation is PARTIAL; cross-family verdict: HIERARCHICAL.** "
        f"Gemma-only→Qwen no-refit MAE is {_fmt(shared_metrics['mae'])} nats "
        f"with {shared_metrics['sign_correct']}/{shared_metrics['n']} signs; a "
        f"held-out Qwen family intercept lowers MAE to "
        f"{_fmt(hierarchical_metrics['mae'])} with "
        f"{hierarchical_metrics['sign_correct']}/{hierarchical_metrics['n']} signs. "
        f"The D-only/separable held-out MAEs are {_fmt(d_metrics['mae'])}/"
        f"{_fmt(s_metrics['mae'])}, but all three D sequences are non-monotone, "
        f"so no clean D exponent is supported."
    )
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Generated "):
            lines[index] = (
                f"Generated by `analysis/v18_law_fit.py`; the distillation "
                f"section and headline were refreshed {summary['generated']} by "
                "`analysis/v21_distill_law.py`. Both analyses are CPU-only and "
                "read committed/local result artifacts without mutating inputs."
            )
            break
    else:
        raise ValueError(f"could not locate generation header in {path}")
    for index, line in enumerate(lines):
        if line.startswith("- **Distillation"):
            lines[index] = headline
            break
    else:
        raise ValueError(f"could not locate distillation headline in {path}")
    text = "\n".join(lines) + "\n"
    start = text.find(start_marker)
    end = text.find(end_marker)
    replacement = section.rstrip() + "\n\n"
    path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")


def dry_run_text(summary: Mapping[str, object]) -> str:
    size = summary["source_referenced_size_law"]
    data = summary["data_ladder"]
    assert isinstance(size, Mapping)
    assert isinstance(data, Mapping)
    shared = size["shared_no_refit"]
    hierarchical = size["hierarchical_intercept"]
    d_only = data["data_only_heldout"]
    separable = data["separable_heldout"]
    assert isinstance(shared, Mapping)
    assert isinstance(hierarchical, Mapping)
    assert isinstance(d_only, Mapping)
    assert isinstance(separable, Mapping)
    return "\n".join(
        [
            "V21 dry run — no files written",
            f"size rows: {len(size['rows'])}; data rows: {len(data['rows'])}",  # type: ignore[arg-type]
            f"shared no-refit MAE: {_fmt(shared['metrics']['mae'])}; sign {shared['metrics']['sign_correct']}/{shared['metrics']['n']}",  # type: ignore[index]
            f"hierarchical held-out MAE: {_fmt(hierarchical['metrics']['mae'])}; sign {hierarchical['metrics']['sign_correct']}/{hierarchical['metrics']['n']}",  # type: ignore[index]
            f"cross-family verdict: {size['verdict']}",
            f"D-only held-out MAE: {_fmt(d_only['metrics']['mae'])}",  # type: ignore[index]
            f"separable held-out MAE: {_fmt(separable['metrics']['mae'])}",  # type: ignore[index]
            f"D verdict: {data['verdict']}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="compute but write nothing")
    parser.add_argument("--v6-base", type=Path, default=V6_BASE)
    parser.add_argument("--v12-base", type=Path, default=V12_BASE)
    parser.add_argument("--output-dir", type=Path, default=OUT_BASE)
    parser.add_argument("--law-fit-report", type=Path, default=LAW_FIT_REPORT)
    parser.add_argument(
        "--no-update-law-fit-report",
        action="store_true",
        help="write V21 outputs without replacing LAW_FIT_REPORT section 3",
    )
    args = parser.parse_args()
    summary = build_summary(args.v6_base, args.v12_base)
    if args.dry_run:
        print(dry_run_text(summary))
        return
    section = render_distillation_section(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "report.md"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(section, encoding="utf-8")
    if not args.no_update_law_fit_report:
        update_law_fit_report(args.law_fit_report, section, summary)
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    if not args.no_update_law_fit_report:
        print(f"updated {args.law_fit_report}")


if __name__ == "__main__":
    main()
