#!/usr/bin/env python3
"""V19: held-out behavioral links from the corrected V15 easy benchmark.

This is a CPU-only analysis of existing ``accuracy.json`` artifacts.  It
refuses archived zero-shot-floor paths and any file that does not carry the
fixed easy-evaluation provenance (domain stopping plus first-block
truncation).  The four outcomes are GSM8K exact accuracy, MBPP pass@1, and
TriviaQA exact match / token F1.

Usage::

    python analysis/v19_links.py --dry-run
    python analysis/v19_links.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
INPUT_BASE = ROOT / "results/v15-accuracy"
OUTPUT_DIR = ROOT / "results/v19-links"
DOC_PATH = ROOT / "paper/docs/BEHAVIORAL_LINKS.md"
BOOTSTRAP_RESAMPLES = 500
MODELS = (
    "gemma3-4b",
    "gemma3-12b",
    "gemma3-27b",
    "gemma4-31b",
    "muse-30b",
    "olmo3-7b",
    "olmo3-32b",
)
MODEL_FAMILY = {
    "gemma3-4b": "gemma3",
    "gemma3-12b": "gemma3",
    "gemma3-27b": "gemma3",
    "gemma4-31b": "gemma4",
    "muse-30b": "muse",
    "olmo3-7b": "olmo3",
    "olmo3-32b": "olmo3",
}
CHECKPOINTS = (
    ("dense", 1.0),
    ("prune-d0.8", 0.8),
    ("prune-d0.7", 0.7),
    ("prune-d0.6", 0.6),
)
OUTCOMES = {
    "math": ("math", "accuracy", "GSM8K accuracy"),
    "code": ("code", "pass_at_1", "MBPP pass@1"),
    "qa_em": ("qa", "exact_match", "TriviaQA exact match"),
    "qa_f1": ("qa", "token_f1", "TriviaQA token F1"),
}
EXPECTED_BENCHMARKS = {"math": "GSM8K", "code": "MBPP", "qa": "TriviaQA"}
CORRECTED_DECODING_MARKERS = (
    "domain-delimiter stopping",
    "post-hoc first-block truncation",
)


def _stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def validate_fixed_easy_artifact(
    path: Path,
    payload: Mapping[str, object],
    *,
    model: str | None = None,
    checkpoint: str | None = None,
) -> None:
    """Reject archived or pre-fix evaluations before reading their scores."""
    if "_zeroshot_floor" in path.parts:
        raise ValueError(f"archived zero-shot-floor artifact is forbidden: {path}")
    if path.parent.name.endswith("__easy") is False:
        raise ValueError(f"not an easy-benchmark checkpoint directory: {path}")
    if int(payload.get("version", -1)) != 15:
        raise ValueError(f"not a V15 artifact: {path}")
    if payload.get("accuracy_benchmark_mode") != "easy":
        raise ValueError(f"artifact lacks easy benchmark mode: {path}")
    if payload.get("probe_source") != "analysis.v15_accuracy_link.build_easy_probes":
        raise ValueError(f"unexpected easy-probe source: {path}")
    benchmarks = payload.get("accuracy_benchmark")
    if not isinstance(benchmarks, Mapping) or dict(benchmarks) != EXPECTED_BENCHMARKS:
        raise ValueError(f"unexpected easy benchmark registry: {path}")
    decoding = str(payload.get("decoding", ""))
    missing = [marker for marker in CORRECTED_DECODING_MARKERS if marker not in decoding]
    if missing:
        raise ValueError(
            f"artifact predates the stop-sequence fix ({', '.join(missing)} absent): {path}"
        )
    samples = payload.get("measurement_samples")
    if not isinstance(samples, Mapping) or any(int(samples.get(cap, -1)) != 64 for cap in EXPECTED_BENCHMARKS):
        raise ValueError(f"unexpected measurement sample counts: {path}")
    if model is not None and payload.get("model") != model:
        raise ValueError(f"model mismatch for {path}: {payload.get('model')!r} != {model!r}")
    if checkpoint is not None:
        expected = f"{model}/{checkpoint}__easy" if model else None
        if expected is not None and payload.get("checkpoint") != expected:
            raise ValueError(
                f"checkpoint mismatch for {path}: {payload.get('checkpoint')!r} != {expected!r}"
            )
        expected_density = dict(CHECKPOINTS)[checkpoint]
        observed_density = payload.get("prune_density")
        if checkpoint == "dense":
            if observed_density is not None:
                raise ValueError(f"dense checkpoint records pruning: {path}")
        elif observed_density is None or not math.isclose(
            float(observed_density), expected_density, abs_tol=1e-12
        ):
            raise ValueError(f"pruning density mismatch: {path}")


def load_observations(base: Path = INPUT_BASE) -> tuple[list[dict], list[dict]]:
    """Load exactly the requested 7 x 4 corrected checkpoint panel."""
    rows: list[dict] = []
    provenance: list[dict] = []
    resolved_base = base.resolve()
    if "_zeroshot_floor" in resolved_base.parts:
        raise ValueError("input base may not point into _zeroshot_floor")
    for model in MODELS:
        for checkpoint, density in CHECKPOINTS:
            path = base / model / f"{checkpoint}__easy" / "accuracy.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing requested fixed-eval artifact: {path}")
            resolved = path.resolve()
            if resolved_base not in resolved.parents:
                raise ValueError(f"artifact escapes the requested input root: {path}")
            payload = _json(path)
            validate_fixed_easy_artifact(
                path, payload, model=model, checkpoint=checkpoint
            )
            losses = payload.get("losses")
            aggregates = payload.get("aggregates")
            if not isinstance(losses, Mapping) or not isinstance(aggregates, Mapping):
                raise ValueError(f"missing losses/aggregates: {path}")
            for outcome, (capability, score_key, label) in OUTCOMES.items():
                aggregate = aggregates.get(capability)
                if not isinstance(aggregate, Mapping):
                    raise ValueError(f"missing {capability} aggregate: {path}")
                loss = float(losses[capability])
                accuracy = float(aggregate[score_key])
                if not math.isfinite(loss) or not math.isfinite(accuracy):
                    raise ValueError(f"non-finite observation: {path} / {outcome}")
                if not 0.0 <= accuracy <= 1.0:
                    raise ValueError(f"accuracy outside [0,1]: {path} / {outcome}")
                rows.append(
                    {
                        "model": model,
                        "family": MODEL_FAMILY[model],
                        "checkpoint": checkpoint,
                        "density": density,
                        "outcome": outcome,
                        "label": label,
                        "capability": capability,
                        "loss": loss,
                        "accuracy": accuracy,
                        "n": int(aggregate["n"]),
                        "source_path": _relative(path),
                    }
                )
            provenance.append(
                {
                    "model": model,
                    "checkpoint": checkpoint,
                    "density": density,
                    "source_path": _relative(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "fixed_eval": True,
                    "decoding": str(payload["decoding"]),
                }
            )
    return rows, provenance


def sigmoid_accuracy(loss: np.ndarray | float, a_max: float, k: float, l_half: float):
    value = np.asarray(loss, dtype=np.float64)
    prediction = a_max / (1.0 + np.exp(np.clip(k * (value - l_half), -60.0, 60.0)))
    return float(prediction) if prediction.ndim == 0 else prediction


def fit_sigmoid(
    rows: Sequence[Mapping[str, object]], *, multistart: bool = True
) -> dict[str, object]:
    """Fit A_max, positive k, and L_half by bounded least squares."""
    if len(rows) < 3:
        return {"status": "too_few_points", "n": len(rows)}
    loss = np.asarray([float(row["loss"]) for row in rows])
    accuracy = np.asarray([float(row["accuracy"]) for row in rows])
    if not np.all(np.isfinite(loss)) or not np.all(np.isfinite(accuracy)):
        return {"status": "nonfinite", "n": len(rows)}
    l_bounds = (
        min(-5.0, float(np.min(loss)) - 10.0),
        max(30.0, float(np.max(loss)) + 10.0),
    )
    if multistart:
        l_starts = np.quantile(loss, [0.2, 0.5, 0.8])
        k_starts = (0.5, 2.0, 8.0)
    else:
        l_starts = (float(np.median(loss)),)
        k_starts = (2.0,)
    best = None
    for l_start in l_starts:
        for k_start in k_starts:
            # For fixed (k, L_half), least-squares A_max has a closed form.
            # Profiling it out makes thousands of cluster-bootstrap fits
            # fast while preserving the exact bounded objective.
            def objective(theta: np.ndarray) -> float:
                k_value = math.exp(float(theta[0]))
                basis = np.asarray(sigmoid_accuracy(loss, 1.0, k_value, float(theta[1])))
                a_value = float(np.clip(
                    np.dot(basis, accuracy) / max(float(np.dot(basis, basis)), 1e-15),
                    1e-6,
                    1.0,
                ))
                return float(np.sum(np.square(a_value * basis - accuracy)))

            result = minimize(
                objective,
                np.array([math.log(k_start), float(l_start)]),
                method="L-BFGS-B",
                bounds=((math.log(1e-4), math.log(100.0)), l_bounds),
                options={"maxiter": 400, "ftol": 1e-13},
            )
            k_value = math.exp(float(result.x[0]))
            l_value = float(result.x[1])
            basis = np.asarray(sigmoid_accuracy(loss, 1.0, k_value, l_value))
            a_value = float(np.clip(
                np.dot(basis, accuracy) / max(float(np.dot(basis, basis)), 1e-15),
                1e-6,
                1.0,
            ))
            fitted = np.array([a_value, k_value, l_value])
            sse = float(np.sum(np.square(a_value * basis - accuracy)))
            if best is None or sse < best[0]:
                best = (sse, result, fitted)
    assert best is not None
    result, fitted = best[1], best[2]
    parameters = {
        "A_max": float(fitted[0]),
        "k": float(fitted[1]),
        "L_half": float(fitted[2]),
    }
    return {
        "status": "ok" if result.success else "fit_failed",
        "n": len(rows),
        "n_models": len({str(row["model"]) for row in rows}),
        "parameters": parameters,
        "mae": float(np.mean(np.abs(
            sigmoid_accuracy(loss, *fitted) - accuracy
        ))),
        "sse": best[0],
    }


def predict_sigmoid(fit: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    if fit.get("status") != "ok":
        return np.full(len(rows), np.nan)
    parameters = fit["parameters"]
    assert isinstance(parameters, Mapping)
    return np.asarray(
        sigmoid_accuracy(
            [float(row["loss"]) for row in rows],
            float(parameters["A_max"]),
            float(parameters["k"]),
            float(parameters["L_half"]),
        )
    )


def bootstrap_sigmoid_parameters(
    rows: Sequence[Mapping[str, object]],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    *,
    label: str = "sigmoid",
) -> dict[str, object]:
    """Model-cluster bootstrap; checkpoint rows stay paired within a model."""
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    models = sorted(grouped)
    if len(models) < 2:
        return {"n_resamples": 0, "n_success": 0, "ci95": {}}
    rng = np.random.default_rng(_stable_seed(label))
    estimates: list[tuple[float, float, float]] = []
    for _ in range(n_resamples):
        sampled = rng.choice(models, size=len(models), replace=True)
        bootstrap_rows = [row for model in sampled for row in grouped[str(model)]]
        fit = fit_sigmoid(bootstrap_rows, multistart=False)
        if fit.get("status") != "ok":
            continue
        p = fit["parameters"]
        assert isinstance(p, Mapping)
        values = (float(p["A_max"]), float(p["k"]), float(p["L_half"]))
        if all(math.isfinite(value) for value in values):
            estimates.append(values)
    names = ("A_max", "k", "L_half")
    array = np.asarray(estimates, dtype=float)
    ci = {
        name: [float(value) for value in np.quantile(array[:, index], [0.025, 0.975])]
        for index, name in enumerate(names)
    } if len(array) else {}
    return {"n_resamples": n_resamples, "n_success": len(estimates), "ci95": ci}


def leave_one_family_out(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Compare a pooled link with leakage-free source-family link averaging.

    A genuinely family-specific curve has no parameters for a never-seen
    family.  The only defined no-leakage transport used here is an equal-weight
    average of the separately fitted training-family curves.  We expose that
    limitation instead of fitting the held-out family to its own test rows.
    """
    records: list[dict] = []
    folds = []
    families = sorted({str(row["family"]) for row in rows})
    for held_out in families:
        train = [row for row in rows if str(row["family"]) != held_out]
        test = [row for row in rows if str(row["family"]) == held_out]
        shared_fit = fit_sigmoid(train)
        family_fits = {
            family: fit_sigmoid([row for row in train if str(row["family"]) == family])
            for family in sorted({str(row["family"]) for row in train})
        }
        usable = [fit for fit in family_fits.values() if fit.get("status") == "ok"]
        shared_prediction = predict_sigmoid(shared_fit, test)
        family_matrix = np.vstack([predict_sigmoid(fit, test) for fit in usable]) if usable else np.empty((0, len(test)))
        family_prediction = np.mean(family_matrix, axis=0) if len(family_matrix) else np.full(len(test), np.nan)
        fold_records = []
        for row, shared_value, family_value in zip(test, shared_prediction, family_prediction):
            record = {
                "held_out_family": held_out,
                "model": row["model"],
                "checkpoint": row["checkpoint"],
                "observed": float(row["accuracy"]),
                "shared_predicted": float(shared_value),
                "family_specific_transport_predicted": float(family_value),
            }
            if math.isfinite(shared_value) and math.isfinite(family_value):
                records.append(record)
                fold_records.append(record)
        folds.append(
            {
                "held_out_family": held_out,
                "n_train": len(train),
                "n_test": len(test),
                "n_predictions": len(fold_records),
                "training_family_fits": {
                    family: fit.get("status") for family, fit in family_fits.items()
                },
                "shared_mae": (
                    float(np.mean([abs(r["observed"] - r["shared_predicted"]) for r in fold_records]))
                    if fold_records else None
                ),
                "family_specific_transport_mae": (
                    float(np.mean([abs(r["observed"] - r["family_specific_transport_predicted"]) for r in fold_records]))
                    if fold_records else None
                ),
            }
        )
    shared_errors = np.asarray(
        [abs(row["observed"] - row["shared_predicted"]) for row in records]
    )
    family_errors = np.asarray(
        [abs(row["observed"] - row["family_specific_transport_predicted"]) for row in records]
    )
    return {
        "n": len(records),
        "shared_mae": float(np.mean(shared_errors)) if len(records) else None,
        "family_specific_transport_mae": float(np.mean(family_errors)) if len(records) else None,
        "paired_error_difference": float(np.mean(shared_errors - family_errors)) if len(records) else None,
        "folds": folds,
        "records": records,
    }


def _paired_error_ci(
    records: Sequence[Mapping[str, object]], n_resamples: int, label: str
) -> list[float] | None:
    if not records:
        return None
    differences = np.asarray([
        abs(float(row["observed"]) - float(row["shared_predicted"]))
        - abs(float(row["observed"]) - float(row["family_specific_transport_predicted"]))
        for row in records
    ])
    rng = np.random.default_rng(_stable_seed(label))
    estimates = np.empty(n_resamples)
    for index in range(n_resamples):
        estimates[index] = float(np.mean(
            differences[rng.integers(0, len(differences), len(differences))]
        ))
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def cliff_coincidence(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Audit whether the first >1-nat loss-damage cell is already at floor."""
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["outcome"]))].append(row)
    records = []
    for (model, outcome), trajectory in sorted(grouped.items()):
        ordered = sorted(trajectory, key=lambda row: float(row["density"]), reverse=True)
        dense = next(row for row in ordered if math.isclose(float(row["density"]), 1.0))
        loss_cliff = next(
            (
                row for row in ordered[1:]
                if float(row["loss"]) - float(dense["loss"]) > 1.0
            ),
            None,
        )
        floor_threshold = 0.05 if outcome == "qa_f1" else 1.0 / float(dense["n"])
        accuracy_cliff = next(
            (row for row in ordered[1:] if float(row["accuracy"]) <= floor_threshold),
            None,
        )
        records.append(
            {
                "model": model,
                "family": MODEL_FAMILY[model],
                "outcome": outcome,
                "loss_cliff_density": None if loss_cliff is None else float(loss_cliff["density"]),
                "loss_cliff_delta": None if loss_cliff is None else float(loss_cliff["loss"]) - float(dense["loss"]),
                "accuracy_at_loss_cliff": None if loss_cliff is None else float(loss_cliff["accuracy"]),
                "floor_threshold": floor_threshold,
                "at_floor_at_loss_cliff": None if loss_cliff is None else bool(float(loss_cliff["accuracy"]) <= floor_threshold),
                "accuracy_floor_density": None if accuracy_cliff is None else float(accuracy_cliff["density"]),
                "density_coincides": bool(
                    loss_cliff is not None
                    and accuracy_cliff is not None
                    and math.isclose(float(loss_cliff["density"]), float(accuracy_cliff["density"]))
                ),
            }
        )
    summary = {}
    for outcome in OUTCOMES:
        subset = [row for row in records if row["outcome"] == outcome]
        observed = [row for row in subset if row["loss_cliff_density"] is not None]
        summary[outcome] = {
            "n_trajectories": len(subset),
            "n_observed_loss_cliffs": len(observed),
            "n_loss_cliffs_at_accuracy_floor": sum(row["at_floor_at_loss_cliff"] is True for row in observed),
            "n_density_coincidences": sum(bool(row["density_coincides"]) for row in observed),
            "n_loss_cliff_censored": len(subset) - len(observed),
        }
    return {"definition": "first pruning cell with L-L_dense > 1 nat; exact-like floor <=1/64, F1 floor <=0.05", "summary": summary, "records": records}


def qa_free_lunch_audit(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    qa_rows = [row for row in rows if row["outcome"] in {"qa_em", "qa_f1"}]
    by_model_outcome: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in qa_rows:
        by_model_outcome[(str(row["model"]), str(row["outcome"]))].append(row)
    records = []
    for (model, outcome), trajectory in sorted(by_model_outcome.items()):
        dense = next(row for row in trajectory if math.isclose(float(row["density"]), 1.0))
        for row in trajectory:
            if math.isclose(float(row["density"]), 1.0):
                continue
            loss_delta = float(row["loss"]) - float(dense["loss"])
            accuracy_delta = float(row["accuracy"]) - float(dense["accuracy"])
            if loss_delta < 0.0:
                records.append(
                    {
                        "model": model,
                        "outcome": outcome,
                        "density": float(row["density"]),
                        "loss_delta": loss_delta,
                        "accuracy_delta": accuracy_delta,
                        "accuracy_improved": accuracy_delta > 0.0,
                    }
                )
    return {
        "n_loss_improving_metric_cells": len(records),
        "n_accuracy_improving_metric_cells": sum(row["accuracy_improved"] for row in records),
        "flag": bool(records and not all(row["accuracy_improved"] for row in records)),
        "records": records,
    }


def analyze(
    rows: Sequence[Mapping[str, object]],
    provenance: Sequence[Mapping[str, object]],
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    results = {}
    for outcome in OUTCOMES:
        subset = [row for row in rows if row["outcome"] == outcome]
        fit = fit_sigmoid(subset)
        bootstrap = bootstrap_sigmoid_parameters(
            subset, bootstrap_resamples, label=f"v19:{outcome}:parameters"
        )
        heldout = leave_one_family_out(subset)
        heldout["paired_shared_minus_family_ci95"] = _paired_error_ci(
            heldout["records"], bootstrap_resamples, f"v19:{outcome}:lofo-paired"
        )
        if heldout["shared_mae"] is None or heldout["family_specific_transport_mae"] is None:
            verdict = "underidentified"
        elif float(heldout["shared_mae"]) <= float(heldout["family_specific_transport_mae"]):
            verdict = "supports the shared link under leakage-free family holdout"
        else:
            ci = heldout["paired_shared_minus_family_ci95"]
            verdict = (
                "family-specific transport is better on held-out families"
                if ci is not None and ci[0] > 0.0
                else "compatible with a shared form; no resolved family-specific gain"
            )
        results[outcome] = {
            "label": OUTCOMES[outcome][2],
            "shared_fit": fit,
            "parameter_bootstrap": bootstrap,
            "leave_one_family_out": heldout,
            "verdict": verdict,
        }
    return {
        "generated": date.today().isoformat(),
        "counts": {
            "files": len(provenance),
            "models": len({row["model"] for row in rows}),
            "families": len({row["family"] for row in rows}),
            "checkpoints": len({(row["model"], row["checkpoint"]) for row in rows}),
            "outcome_rows": len(rows),
        },
        "provenance": list(provenance),
        "outcomes": results,
        "cliff_coincidence": cliff_coincidence(rows),
        "qa_free_lunch": qa_free_lunch_audit(rows),
    }


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def _fmt_ci(value: object, digits: int = 3) -> str:
    if not isinstance(value, Sequence) or len(value) != 2:
        return "n/a"
    return f"[{_fmt(value[0], digits)}, {_fmt(value[1], digits)}]"


def build_markdown(summary: Mapping[str, object], bootstrap_resamples: int) -> str:
    outcomes = summary["outcomes"]
    assert isinstance(outcomes, Mapping)
    cliff = summary["cliff_coincidence"]
    assert isinstance(cliff, Mapping)
    cliff_summary = cliff["summary"]
    assert isinstance(cliff_summary, Mapping)
    qa_audit = summary["qa_free_lunch"]
    assert isinstance(qa_audit, Mapping)
    lines = [
        "# Behavioral loss-to-accuracy links",
        "",
        f"Generated {summary['generated']} by `analysis/v19_links.py` from existing V15 artifacts only. This was a CPU-only refit; it performed no inference or training and did not mutate its inputs.",
        "",
        "## Result",
        "",
        "The fitted link is `A=A_max/(1+exp(k(L-L_half)))`. Rows are equally weighted checkpoint observations. Parameter intervals are deterministic model-cluster bootstrap 95% intervals, so all four checkpoints from a sampled model move together. They do not measure benchmark-item or seed uncertainty; every checkpoint has one recorded seed and 64 evaluation items.",
        "",
        "| Outcome | n | A_max | L_half (95% CI) | k (95% CI) | shared LOFO MAE | family-specific transport LOFO MAE | paired shared−family error (95% CI) | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for outcome, result in outcomes.items():
        assert isinstance(result, Mapping)
        fit = result["shared_fit"]
        boot = result["parameter_bootstrap"]
        heldout = result["leave_one_family_out"]
        assert isinstance(fit, Mapping) and isinstance(boot, Mapping) and isinstance(heldout, Mapping)
        parameters = fit.get("parameters", {})
        cis = boot.get("ci95", {})
        assert isinstance(parameters, Mapping) and isinstance(cis, Mapping)
        lines.append(
            f"| {result['label']} | {fit.get('n', 0)} | {_fmt(parameters.get('A_max'))} | "
            f"{_fmt(parameters.get('L_half'))} {_fmt_ci(cis.get('L_half'))} | "
            f"{_fmt(parameters.get('k'))} {_fmt_ci(cis.get('k'))} | "
            f"{_fmt(heldout.get('shared_mae'))} | "
            f"{_fmt(heldout.get('family_specific_transport_mae'))} | "
            f"{_fmt(heldout.get('paired_error_difference'))} {_fmt_ci(heldout.get('paired_shared_minus_family_ci95'))} | "
            f"{result['verdict']} |"
        )
    lines.extend([
        "",
        "Here, “family-specific transport” is deliberately leakage-free: each training family gets its own sigmoid, and their predictions are averaged equally for the never-seen family. A literal `g_{c,f}` has no learned parameters for a held-out family; fitting it on that family's test rows would make the comparison in-sample. The singleton Gemma-4 and Muse families each supply only four curve points, so family-specific conclusions are thin.",
        "",
        "**Interpretation.** Math, code, and both QA metrics are compatible with a shared cross-family sigmoid within each outcome because no family-specific transport improvement is resolved by the paired held-out intervals. This is strongest for the existence of a monotone form, not equality across capabilities: code has a weakly identified steepness whose bootstrap reaches the upper fit bound, TriviaQA EM has broad parameter intervals, and token-F1 has a much shallower fitted slope. Thus code/QA support shared-within-capability behavior provisionally, while the link parameters and the literal zero-accuracy cliff remain capability-specific.",
        "",
        "## Cliff coincidence",
        "",
        "A loss cliff is the first measured pruning cell with `L-L_dense > 1` nat. “At floor” means at most one correct item (`<=1/64`) for accuracy/pass@1/EM and `F1<=0.05` for token-F1. These thresholds are declared diagnostics, not fitted change points.",
        "",
        "| Outcome | observed loss cliffs / 7 | already at accuracy floor | same first measured density | loss-cliff censored |",
        "|---|---:|---:|---:|---:|",
    ])
    for outcome, result in outcomes.items():
        values = cliff_summary[outcome]
        assert isinstance(values, Mapping)
        lines.append(
            f"| {result['label']} | {values['n_observed_loss_cliffs']}/7 | "
            f"{values['n_loss_cliffs_at_accuracy_floor']}/{values['n_observed_loss_cliffs']} | "
            f"{values['n_density_coincidences']}/{values['n_observed_loss_cliffs']} | "
            f"{values['n_loss_cliff_censored']} |"
        )
    lines.extend([
        "",
        "The exact-zero coincidence test is stricter than the sigmoid half-accuracy transition. In particular, TriviaQA often retains partial credit at the first >1-nat loss crossing, so the data support a monotone behavioral transition more strongly than literal loss-cliff = zero-accuracy coincidence.",
        "",
        "## QA loss-improvement audit",
        "",
    ])
    if qa_audit.get("flag"):
        lines.append(
            "**Flag: lower QA loss does not reliably imply higher QA accuracy in this panel.** "
            f"There are {qa_audit['n_loss_improving_metric_cells']} QA metric-cells below their model's dense loss and only {qa_audit['n_accuracy_improving_metric_cells']} have strictly higher corresponding accuracy."
        )
    else:
        lines.append("Every observed lower-QA-loss cell also has higher QA accuracy in this panel.")
    lines.extend([
        "",
        "| Model | density | QA metric | Δ loss vs dense | Δ accuracy vs dense |",
        "|---|---:|---|---:|---:|",
    ])
    for row in qa_audit.get("records", []):
        lines.append(
            f"| {row['model']} | {row['density']:.1f} | {row['outcome']} | "
            f"{row['loss_delta']:+.4f} | {row['accuracy_delta']:+.4f} |"
        )
    lines.extend([
        "",
        "This directly limits the stronger “QA free lunch” interpretation: a negative teacher-forced QA loss delta is not, by itself, evidence of improved TriviaQA behavior. The present audit is pruning-only and therefore does not validate distillation's negative QA loss delta behaviorally.",
        "",
        "## Fixed-evaluation provenance",
        "",
        f"All {summary['counts']['files']} requested files passed all gates: V15, `accuracy_benchmark_mode=easy`, GSM8K/MBPP/TriviaQA registry, 64 measurement examples per capability, `analysis.v15_accuracy_link.build_easy_probes`, and a decoding record containing both domain-delimiter stopping and post-hoc first-block truncation. Paths containing `_zeroshot_floor` are rejected before scores are read.",
        "",
        "Inputs: `results/v15-accuracy/<model>/{dense,prune-d0.8,prune-d0.7,prune-d0.6}__easy/accuracy.json`. Exact paths and SHA-256 hashes are recorded in `results/v19-links/summary.json`; the fitted observations are in `results/v19-links/observations.csv`.",
        "",
        "## Limitations",
        "",
        f"- Bootstrap count: {bootstrap_resamples:,}; model clusters: 7; raw families: Gemma-3, Gemma-4, Muse, and OLMo-3.",
        "- Each accuracy is based on only 64 items, one decoding run, and seed 0. Apparent non-monotonic cells can be item-sampling noise.",
        "- The four loss ranges differ substantially. OLMo-3 never enters the steep accuracy-collapse regime at the tested densities, while singleton-family links are nearly saturated by four points.",
        "- These are absolute loss links from each benchmark's own V15 measurement prompts. They should not be substituted with V6 losses, especially for QA, whose probe construction differs.",
    ])
    return "\n".join(lines) + "\n"


def write_outputs(
    rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    output_dir: Path,
    doc_path: Path,
    bootstrap_resamples: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = (
        "model", "family", "checkpoint", "density", "outcome", "label",
        "capability", "loss", "accuracy", "n", "source_path",
    )
    with (output_dir / "observations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    markdown = build_markdown(summary, bootstrap_resamples)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    doc_path.write_text(markdown, encoding="utf-8")


def dry_run_text(rows: Sequence[Mapping[str, object]], provenance: Sequence[Mapping[str, object]]) -> str:
    checkpoint_counts = Counter(str(row["checkpoint"]) for row in rows if row["outcome"] == "math")
    lines = [
        "V19 dry run — no files written",
        f"fixed easy accuracy files: {len(provenance)} / {len(MODELS) * len(CHECKPOINTS)} requested",
        f"models: {len(set(row['model'] for row in rows))}; families: {len(set(row['family'] for row in rows))}",
        f"checkpoint observations: {len(provenance)}; outcome rows: {len(rows)}",
        "by checkpoint: " + ", ".join(f"{name}={checkpoint_counts[name]}" for name, _ in CHECKPOINTS),
        "by outcome: " + ", ".join(
            f"{outcome}={sum(row['outcome'] == outcome for row in rows)}" for outcome in OUTCOMES
        ),
        "provenance: all files passed fixed stop-sequence evaluation gates; _zeroshot_floor excluded",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--input-base", type=Path, default=INPUT_BASE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--doc", type=Path, default=DOC_PATH)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args()
    if args.bootstrap_resamples < 100:
        parser.error("--bootstrap-resamples must be at least 100")
    rows, provenance = load_observations(args.input_base)
    if args.dry_run:
        print(dry_run_text(rows, provenance))
        return
    summary = analyze(rows, provenance, args.bootstrap_resamples)
    write_outputs(rows, summary, args.output_dir, args.doc, args.bootstrap_resamples)
    print(f"wrote {args.doc}")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
