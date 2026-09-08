#!/usr/bin/env python3
"""CPU-only direct-response analysis of the existing 3-size x 3-step Pythia grid.

python3 analysis/v36_pythia_controlled_fit.py [--dry-run]

NOTE: pythia-2.8b step-revisions collapse to identical weights upstream (verified
after clean re-download); run the available panel with SDL_V36_SIZES=410m,1.4b
(leave_one_size_out auto-drops when <3 sizes). Full 3-size path stays valid for when
2.8b is fixed upstream.

Only NumPy and the standard library are needed. No model loading, GPU work,
network access, source-amplitude fitting, or hyperparameter search occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import json
from pathlib import Path

try:
    from . import prediction_audit as audit
except ImportError:
    import prediction_audit as audit

import numpy as np  # prediction_audit sets CPU BLAS thread limits first.

ROOT = audit.ROOT
OUT = ROOT / "results/v36-pythia-controlled"
REPORT = ROOT / "paper/docs/PYTHIA_CONTROLLED.md"
CAPS = ("math", "code", "qa")
SIZES = tuple(os.environ.get("SDL_V36_SIZES", "410m,1.4b,2.8b").split(","))  # env-overridable to drop upstream-broken sizes
STEPS = (16000, 64000, 143000)
TOKENS_PER_STEP = 1024 * 2048
ARMS = ("pruning", "quantization")
CONFIGS = {"pruning": (.9, .8, .7, .6), "quantization": (8, 6, 4, 3)}
FILES = {"pruning": ("v6-capability-geometry", "prune_losses.json", "1.0"),
         "quantization": ("v10-quantization", "quant_losses.json", "dense")}
_ALL_HOLDOUTS = {"leave_one_size_out": "size", "leave_one_step_out": "step"}
# leave-one-size-out needs >=3 sizes (training on 1 size has no N0 variation); drop it otherwise.
HOLDOUTS = {k: v for k, v in _ALL_HOLDOUTS.items()
            if not (v == "size" and len(SIZES) < 3)}
NEAR_ZERO = .01  # Native-token nats, annotation only; not a fitting threshold.
LARGE_DAMAGE = 1.  # Annotation only; int3 is separately flagged regardless of loss.
MONOTONIC_ATOL = 1e-12  # Floating-point tolerance, not a statistical noise floor.
PYTHIA_PAPER = "https://proceedings.mlr.press/v202/biderman23a/biderman23a.pdf"
PYTHIA_REPO = "https://github.com/EleutherAI/pythia"

# Transcribed from locally cached official step143000 config.json files; all
# three cached step configs were checked to have these same dimensions per size.
# Embed the transcription so reproduction needs only the 18 loss JSON files.
ARCHITECTURES = {
    "410m": {"hidden_size": 1024, "intermediate_size": 4096, "num_hidden_layers": 24,
             "revision": "bba6a464f54bbf08fc174cfb351d9794d58af21d",
             "config_sha256": "d4c11e84a59c8af4d88446bba53b718f7aef740daa070ded08fd6a9a3aca4fc6"},
    "1.4b": {"hidden_size": 2048, "intermediate_size": 8192, "num_hidden_layers": 24,
             "revision": "9cc5c8c8148a4e0115d9e29c6b4f21124cfe748a",
             "config_sha256": "6ea552aa42b7437f019ccdd30b7c9b83a32dccb5170ce31b9ddbe94d00f4671c"},
    "2.8b": {"hidden_size": 2560, "intermediate_size": 10240, "num_hidden_layers": 32,
             "revision": "dbe7ae300a54abcdc475a33907b3dff81d25709f",
             "config_sha256": "fe1a9b9550935517588c98f89df24db7d7eadf4c4a00dd32e127a2d4585e9827"},
}


def training_tokens(step):
    """Processed pretraining tokens, not unique tokens or distillation data."""
    if step not in STEPS:
        raise ValueError(f"Unknown Pythia grid step: {step}")
    return step * TOKENS_PER_STEP


def matrix_n0(size):
    """GPT-NeoX QKV + attention output + two MLP matrices; no vectors/embeddings/head."""
    a = ARCHITECTURES[size]
    h, m = a["hidden_size"], a["intermediate_size"]
    return a["num_hidden_layers"] * (4*h*h + 2*h*m)


def response_flags(arm, config, observed):
    return {"near_zero": abs(observed) <= NEAR_ZERO,
            "negative": observed < 0,
            "int3": arm == "quantization" and config == 3,
            "large_damage": observed > LARGE_DAMAGE}


def parse_losses(table, arm):
    """Require the real arm-local dense reference and every planned coordinate."""
    anchor = FILES[arm][2]
    if anchor not in table:
        raise ValueError(f"Missing true dense key {anchor!r} for {arm}")

    def losses(values):
        result = {c: audit.finite(values[c]) for c in CAPS}
        if min(result.values()) < 0:
            raise ValueError("Absolute losses must be nonnegative")
        return result

    dense, compressed = losses(table[anchor]), {}
    for key, values in table.items():
        if key == anchor or key.startswith("_"):
            continue
        q = audit.finite(key)
        if q not in CONFIGS[arm] or q in compressed:
            raise ValueError(f"Unexpected or duplicate {arm} coordinate: {key}")
        compressed[q] = losses(values)
    if set(compressed) != set(CONFIGS[arm]):
        raise ValueError(f"Incomplete {arm} configuration grid")
    return dense, compressed


def load_grid(root=ROOT):
    """Load exactly the requested 18 files, hash bytes before parsing, retain all deltas."""
    root = Path(root)
    rows, hashes = [], {}
    for arm in ARMS:
        directory, filename, _ = FILES[arm]
        for size, step in itertools.product(SIZES, STEPS):
            cell = f"pythia-{size}--step{step}"
            relative = f"results/{directory}/{cell}/{filename}"
            raw = (root / relative).read_bytes()
            hashes[relative] = hashlib.sha256(raw).hexdigest()
            dense, compressed = parse_losses(json.loads(raw), arm)
            for cap, config in itertools.product(CAPS, CONFIGS[arm]):
                observed = compressed[config][cap] - dense[cap]
                rows.append({"row_id": f"{arm}|{cell}|{cap}|{config:g}",
                             "arm": arm, "capability": cap, "cell": cell,
                             "size": size, "step": step, "N0": matrix_n0(size),
                             "D0": training_tokens(step), "L0": dense[cap],
                             "config": config, "loss": compressed[config][cap],
                             "observed": observed, "source": relative,
                             **response_flags(arm, config, observed)})
    return rows, hashes


def checkpoint_integrity(rows, hashes):
    """Flag identical full dense/compressed panels across differently named steps.

    Numerical payload comparison catches duplicates even if JSON formatting
    differs. This is an integrity annotation, never an exclusion rule.
    """
    duplicates, unique = [], {}
    for arm in ARMS:
        unique[arm] = 0
        for size in SIZES:
            signatures = {}
            for step in STEPS:
                cell_rows = sorted([r for r in rows if (r['arm'], r['size'], r['step']) ==
                                    (arm, size, step)], key=lambda r: r['row_id'])
                signature = tuple((r['capability'], r['config'], r['L0'], r['loss']) for r in cell_rows)
                signatures.setdefault(signature, []).append(cell_rows[0])
            unique[arm] += len(signatures)
            for identical in signatures.values():
                if len(identical) > 1:
                    sources = [r['source'] for r in identical]
                    duplicates.append({'arm': arm, 'size': size,
                                       'steps': [r['step'] for r in identical],
                                       'cells': [r['cell'] for r in identical],
                                       'sources': sources,
                                       'byte_identical': len({hashes[p] for p in sources}) == 1})
    return {'status': 'PROVISIONAL_DUPLICATE_CHECKPOINT_PAYLOADS' if duplicates else 'NO_EXACT_DUPLICATE_PAYLOADS',
            'headline_ready': not bool(duplicates),
            'duplicate_step_payloads': duplicates, 'unique_size_step_payloads_per_arm': unique,
            'interpretation': 'Exact full-panel repeats across named checkpoints require provenance verification. Directory names alone do not establish which weights were measured. Retain all nine named cells for the requested analysis, but do not treat duplicate flat trajectories or CV scores as validated training-history evidence. Duplicate payloads cross train/test in step holdouts. Cause is not established by these loss JSON files.'}


def folds(rows, group_key):
    """One whole size or step held out, across every config and the other axis."""
    if group_key not in ("size", "step"):
        raise ValueError("Holdout key must be size or step")
    expected = SIZES if group_key == "size" else STEPS
    if {r[group_key] for r in rows} != set(expected):
        raise ValueError(f"Need all three {group_key} groups")
    return [{"held_out": value,
             "train_indices": [i for i, r in enumerate(rows) if r[group_key] != value],
             "test_indices": [i for i, r in enumerate(rows) if r[group_key] == value]}
            for value in expected]


def basic_input(row, with_d0):
    return {k: row[k] for k in ("N0", "L0", "config", *(('D0',) if with_d0 else ()))}


def covariates(inputs, with_d0):
    expected = {"N0", "L0", "config"} | ({"D0"} if with_d0 else set())
    values = []
    for row in inputs:
        if set(row) != expected:
            raise ValueError("Predictors require exactly N0/L0/config and, for B only, D0; no outcomes")
        n0, l0 = audit.finite(row["N0"]), audit.finite(row["L0"])
        if n0 <= 0 or l0 < 0:
            raise ValueError("N0 must be positive and L0 nonnegative")
        value = [np.log(n0/1e9), l0]
        if with_d0:
            d0 = audit.finite(row["D0"])
            if d0 <= 0:
                raise ValueError("D0 must be positive")
            value.append(np.log(d0/1e9))
        values.append(value)
    return np.asarray(values, dtype=float)


def design_matrix(inputs, arm, with_d0, center, scale):
    """Config indicators interacted with [1, z(log N0), z(L0), optional z(log D0)]."""
    continuous = (covariates(inputs, with_d0) - np.asarray(center)) / np.asarray(scale)
    base = np.column_stack([np.ones(len(inputs)), continuous])
    configs = CONFIGS[arm]
    if any(r["config"] not in configs for r in inputs):
        raise ValueError("Prediction requires one of the four measured configurations")
    indicator = np.array([[r["config"] == q for q in configs] for r in inputs], float)
    # Feature-major order: B's first 12 columns equal A's entire design.
    return np.concatenate([base[:, j, None] * indicator for j in range(base.shape[1])], axis=1)


def fit_direct(rows, with_d0):
    """One OLS solve on all config-level signed Delta L observations, never a_c labels."""
    if not rows or len({(r["arm"], r["capability"]) for r in rows}) != 1:
        raise ValueError("Fit exactly one arm and capability at a time")
    arm, cap = rows[0]["arm"], rows[0]["capability"]
    inputs = [basic_input(r, with_d0) for r in rows]
    raw = covariates(inputs, with_d0)
    center, scale = raw.mean(axis=0), raw.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError("Training covariate has no variation")
    x = design_matrix(inputs, arm, with_d0, center, scale)
    y = np.array([audit.finite(r["observed"]) for r in rows])
    coefficients, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
    if rank != x.shape[1]:
        raise ValueError("Rank-deficient direct-response fit")
    terms = ["intercept", "z_log_N0", "z_L0", *(["z_log_D0"] if with_d0 else [])]
    return {"model": "B" if with_d0 else "A", "with_d0": with_d0,
            "arm": arm, "capability": cap,
            "target": "observed signed config-level Delta L", "objective": "unweighted OLS",
            "n_observations": len(y), "n_cells": len({r["cell"] for r in rows}),
            "n_parameters": x.shape[1], "rank": int(rank),
            "condition_number": float(singular[0]/singular[-1]),
            "train_row_ids": [r["row_id"] for r in rows],
            "center": center.tolist(), "scale": scale.tolist(),
            "feature_names": [f"{t}:config={q:g}" for t in terms for q in CONFIGS[arm]],
            "coefficients": coefficients.tolist(),
            "training_mse": float(np.mean((x @ coefficients - y)**2)),
            "log_D0_L0_correlation": float(np.corrcoef(
                np.log([r["D0"] for r in rows]), [r["L0"] for r in rows])[0, 1])}


def predict(fit, inputs):
    """Target interface receives only basic inputs, with no compressed outcomes."""
    return design_matrix(inputs, fit["arm"], fit["with_d0"], fit["center"], fit["scale"]) @ np.array(fit["coefficients"])


def paired_metrics(records, group_key):
    """Exact paired group bootstrap, conditional on the fitted OOF predictions.

    Enumerate all 3**3 ordered cluster resamples; use the discrete inverse CDF
    for percentile endpoints. Configurations and the other grid axis travel
    together within a held-out group. No row-wise pseudo-replication.
    """
    groups = sorted({r[group_key] for r in records})
    if len(groups) != 3:
        raise ValueError("Intervals require three whole held-out groups")
    errors = np.array([[abs(r["observed"]-r["predictions"][m]) for m in ("A", "B")]
                      for r in records])
    indices = [[i for i, r in enumerate(records) if r[group_key] == g] for g in groups]
    sums = np.array([errors[i].sum(axis=0) for i in indices])
    counts = np.array([len(i) for i in indices])
    draws = np.array(list(itertools.product(range(3), repeat=3)))
    samples = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)[:, None]
    ci = lambda x: np.quantile(x, [.025, .975], method="inverted_cdf").tolist()
    a, b = errors.mean(axis=0)
    difference = float(a-b)
    interval = ci(samples[:, 0]-samples[:, 1])
    leave_out = []
    for g, idx in zip(groups, indices):
        keep = np.ones(len(records), bool)
        keep[idx] = False
        leave_out.append({"omitted": g, "improvement": float(np.mean(errors[keep, 0]-errors[keep, 1]))})
    return {"n": len(records), "n_groups": 3,
            "A_mae": float(a), "B_mae": float(b),
            "A_mae_ci95": ci(samples[:, 0]), "B_mae_ci95": ci(samples[:, 1]),
            "improvement": difference, "improvement_ci95": interval,
            "point_estimate_reduces_error": difference > 0,
            "interval_assessment": ("lower_error_interval_above_zero" if interval[0] > 0 else
                                    "higher_error_interval_below_zero" if interval[1] < 0 else
                                    "inconclusive_interval_includes_zero"),
            "leave_one_group_out": leave_out,
            "leave_one_group_out_range": [min(v["improvement"] for v in leave_out),
                                           max(v["improvement"] for v in leave_out)]}


def cross_validate(rows, group_key):
    records, fitted_folds = [], []
    for fold in folds(rows, group_key):
        train = [rows[i] for i in fold["train_indices"]]
        test = [rows[i] for i in fold["test_indices"]]
        fits = {m: fit_direct(train, m == "B") for m in ("A", "B")}
        predictions = {m: predict(fits[m], [basic_input(r, m == "B") for r in test]) for m in fits}
        fold_rows = [{**r, "held_out": fold["held_out"],
                      "predictions": {m: float(predictions[m][i]) for m in fits}}
                     for i, r in enumerate(test)]
        errors = {m: float(np.mean([abs(r["observed"]-r["predictions"][m]) for r in fold_rows])) for m in fits}
        fitted_folds.append({**fold, "fits": fits, "n_train": len(train), "n_test": len(test),
                             "A_mae": errors["A"], "B_mae": errors["B"],
                             "improvement": errors["A"]-errors["B"]})
        records.extend(fold_rows)
    return {"metrics": paired_metrics(records, group_key), "folds": fitted_folds,
            "by_config": {f"{q:g}": paired_metrics([r for r in records if r["config"] == q], group_key)
                          for q in CONFIGS[rows[0]["arm"]]},
            "records": records}


def raw_trends(rows):
    """Matched size/config step trajectories. Signed damage is primary; |damage| secondary."""
    trajectories = []
    for size, config in itertools.product(SIZES, CONFIGS[rows[0]["arm"]]):
        sequence = sorted([r for r in rows if r["size"] == size and r["config"] == config],
                          key=lambda r: r["step"])
        if [r["step"] for r in sequence] != list(STEPS):
            raise ValueError("Raw trends require exactly three ordered checkpoints")
        y = np.array([r["observed"] for r in sequence])
        changes = np.diff(y)
        if np.all(np.abs(changes) <= MONOTONIC_ATOL):
            direction = "flat"
        elif np.all(changes >= -MONOTONIC_ATOL):
            direction = "increasing"
        elif np.all(changes <= MONOTONIC_ATOL):
            direction = "decreasing"
        else:
            direction = "nonmonotonic"
        # Do not describe negative damage (improvement) as positive fragility.
        # Ratios alone are undefined/unstable around zero; every raw row stays.
        ratio_valid = bool(y[0] > NEAR_ZERO and y[-1] > NEAR_ZERO)
        trajectories.append({"size": size, "config": config, "steps": list(STEPS),
                             "D0": [r["D0"] for r in sequence], "observed": y.tolist(),
                             "L0": [r["L0"] for r in sequence],
                             "adjacent_changes": changes.tolist(),
                             "late_minus_early": float(y[-1]-y[0]),
                             "late_abs_minus_early_abs": float(abs(y[-1])-abs(y[0])),
                             "monotonicity": direction,
                             "strictly_increasing": bool(np.all(changes > MONOTONIC_ATOL)),
                             "any_near_zero": any(r["near_zero"] for r in sequence),
                             "any_negative": any(r["negative"] for r in sequence),
                             "int3": sequence[0]["int3"],
                             "any_large_damage": any(r["large_damage"] for r in sequence),
                             "duplicate_checkpoint_payload": any(r.get("duplicate_checkpoint_payload", False) for r in sequence),
                             "late_over_early_positive_damage": float(y[-1]/y[0]) if ratio_valid else None,
                             "ratio_status": "defined" if ratio_valid else "nonpositive_or_near_zero_endpoint"})
    counts = {k: sum(t["monotonicity"] == k for t in trajectories)
              for k in ("increasing", "decreasing", "flat", "nonmonotonic")}
    ratios = [t["late_over_early_positive_damage"] for t in trajectories
              if t["late_over_early_positive_damage"] is not None]
    by_size = {}
    for size in SIZES:
        by_size[size] = {
            "mean_signed_damage_by_step": [float(np.mean([r["observed"] for r in rows
                                                          if r["size"] == size and r["step"] == s])) for s in STEPS],
            "mean_absolute_damage_by_step": [float(np.mean([abs(r["observed"]) for r in rows
                                                            if r["size"] == size and r["step"] == s])) for s in STEPS]}
    return {"n_trajectories": len(trajectories), "monotonicity_counts": counts,
            "late_greater_than_early": sum(t["late_minus_early"] > MONOTONIC_ATOL for t in trajectories),
            "late_absolute_greater_than_early_absolute": sum(t["late_abs_minus_early_abs"] > MONOTONIC_ATOL for t in trajectories),
            "positive_damage_ratio_count": len(ratios),
            "positive_damage_ratio_median": float(np.median(ratios)) if ratios else None,
            "positive_damage_ratio_range": [min(ratios), max(ratios)] if ratios else None,
            "by_size": by_size, "trajectories": trajectories}


def build_summary(root=ROOT):
    rows, hashes = load_grid(root)
    integrity = checkpoint_integrity(rows, hashes)
    suspect = {(g['arm'], cell) for g in integrity['duplicate_step_payloads'] for cell in g['cells']}
    for row in rows:
        row['duplicate_checkpoint_payload'] = (row['arm'], row['cell']) in suspect
    results = {}
    for arm in ARMS:
        results[arm] = {}
        for cap in CAPS:
            panel = [r for r in rows if (r["arm"], r["capability"]) == (arm, cap)]
            results[arm][cap] = {
                "n_rows": len(panel),
                "flag_counts": {k: sum(r[k] for r in panel) for k in ("near_zero", "negative", "int3", "large_damage", "duplicate_checkpoint_payload")},
                "holdouts": {name: cross_validate(panel, key) for name, key in HOLDOUTS.items()},
                "raw_trend": raw_trends(panel)}
    architectures = {s: {**a, "N0": matrix_n0(s),
                         "source_url": f"https://huggingface.co/EleutherAI/pythia-{s}/blob/{a['revision']}/config.json"}
                     for s, a in ARCHITECTURES.items()}
    dense_differences = [abs(p["L0"]-q["L0"]) for p, q in zip(
        [r for r in rows if r["arm"] == "pruning"], [r for r in rows if r["arm"] == "quantization"])]
    return {"version": 36, "analysis": "Pythia controlled training-history direct-response fit",
            "cpu_only": True, "new_model_runs": 0, "n_rows": len(rows),
            "n_cells_per_arm": 9, "input_sha256": hashes, "integrity": integrity,
            "inputs": {"architectures": architectures,
                       "N0_convention": "layers * (4*hidden_size**2 + 2*hidden_size*intermediate_size); transformer matrices only; excludes embeddings, LM head, biases and norms",
                       "D0_by_step": {str(s): training_tokens(s) for s in STEPS},
                       "tokens_per_step": TOKENS_PER_STEP, "D0_source": PYTHIA_PAPER,
                       "recipe_source": PYTHIA_REPO,
                       "max_arm_dense_difference": max(dense_differences)},
            "protocol": {
                "endpoint": "Delta L_c = L_c(config) - own arm/cell dense L0_c, signed nats/native token",
                "scope": "fixed-training-recipe SERIES; standard non-deduped Pythia; architectures/hyperparameters still vary across sizes",
                "A": "sum_q 1[config=q] * (alpha_q + beta_q*log(N0/1e9) + gamma_q*L0_c)",
                "B": "A + sum_q 1[config=q] * delta_q*log(D0/1e9)",
                "fitting": "Direct unweighted OLS on all signed configuration-level responses; no per-source a_c labels, response transform, clipping, ridge or tuning. Continuous inputs standardized on training rows only.",
                "parameters": {"A": 12, "B": 16},
                "folds": "3 leave-one-size-out and separately 3 leave-one-step-out; 24 training / 12 test observations, 6 / 3 source cells per fold, per arm/capability",
                "configuration_scope": "Four categorical measured configurations; all four represented in every train/test split. No unseen-config interpolation/extrapolation claim; dense is a reference, not a scored zero-response row.",
                "weighting": "Equal weight per observation within each arm/capability; balanced grid also gives equal weight per config, cell and held-out group. Never pool arms or capabilities.",
                "uncertainty": "Paired exact cluster bootstrap of fixed out-of-fold errors: all 27 ordered draws of 3 held-out sizes or steps; 95% discrete inverse-CDF percentile intervals. Leave-one-held-out-group-out score range also reported; it is a sensitivity range, not a second CI. No refitting in either score resampling.",
                "uncertainty_limits": "Only 3 clusters; training sets overlap, steps share trajectories and probes. Intervals are descriptive panel-conditional stability, not independent-seed, probe, retraining, population or causal uncertainty. No multiplicity correction for 12 comparisons.",
                "flags": {"near_zero_abs_delta_le": NEAR_ZERO, "large_damage_delta_gt": LARGE_DAMAGE,
                          "int3": "All 3-bit rows flagged, including those without >1 nat damage",
                          "handling": "Retain every finite near-zero, negative, int3 and large-damage row in fitting and primary MAE; no censoring, winsorizing or log-response fit. Missing/nonfinite inputs fail loudly.",
                          "timing": "Rules and model form fixed before running V36 fits; retrospective analysis of existing outcomes, not a prospective preregistration."},
                "raw_trend": "Matched fixed-size/config signed Delta L across all three steps; monotonicity tolerance 1e-12. Absolute-damage changes separately descriptive. Late/early ratio only when both endpoint damages exceed .01; otherwise null with reason, without dropping the trajectory.",
                "compression_scope": "Existing V6/V10 compress all language weight matrices, including embeddings and LM head; N0 remains a transformer-matrix size covariate, not the total compressed parameter count.",
                "distillation": "PENDING; not included. Training-blocked on flaky GPUs; no outcomes imputed."},
            "results": results}


def render(summary):
    f = audit.fmt
    ci = audit.with_ci
    metrics = [summary['results'][a][c]['holdouts'][h]['metrics']
               for a, c, h in itertools.product(ARMS, CAPS, HOLDOUTS)]
    lines = ["# Pythia controlled training-history analysis (V36)", ""]
    if not summary['integrity']['headline_ready']:
        lines += ["**PROVISIONAL — checkpoint provenance issue.** "
                  "The named grid contains exact duplicate full loss panels across different training-step labels:", ""]
        for g in summary['integrity']['duplicate_step_payloads']:
            lines.append(f"- {g['arm']}, Pythia-{g['size']}: steps {', '.join(str(s) for s in g['steps'])}; "
                         f"all dense and compressed capability losses identical; byte-identical JSON: {g['byte_identical']}.")
        lines += ["", "All nine named cells remain in the requested fits and scores. "
                  "These duplicates are flagged in summary.json; their cause cannot be established from aggregate losses. "
                  "In step holdouts they place identical outcomes under different D₀ labels in training and test. "
                  "**The results below are an as-supplied diagnostic, not yet a validated headline controlled-training-history result.** "
                  "Verify the weight revisions behind the repeated cells and replace any incorrect measurements before drawing that conclusion. "
                  "Exact flat trajectories for the duplicated size must not be interpreted as training invariance.", ""]
    lines += [f"On the supplied grid, adding D₀ lowers point-estimate MAE in {sum(m['improvement'] > 0 for m in metrics)} of 12 arm/capability/holdout comparisons; "
              f"{sum(m['improvement_ci95'][0] > 0 for m in metrics)} improvement intervals lie wholly above zero. "
              "The tables keep the arms and capabilities separate; this count is descriptive and is not a pooled score.", "",
             "The headline question is whether training history D₀ reduces held-out prediction error for capability **loss** damage beyond model size N₀ and measured dense loss L₀. "
             "The target is the observed signed response ΔL_c = L_c(config) − L₀,c. This is a loss endpoint; it does not by itself establish changes in task accuracy.", "",
             "This is a **fixed-training-recipe SERIES**, using standard (non-deduped) Pythia checkpoints. "
             "The suite shares training data/order and token accounting; architectures and hyperparameters still vary across sizes (including learning rate). "
             f"See the [Pythia paper]({PYTHIA_PAPER}) and [official training documentation]({PYTHIA_REPO}). "
             "Within a size, D₀ also tracks checkpoint age, cumulative optimization and the learning-rate schedule. "
             "A conditional prediction gain is evidence about this series, not an isolated causal token effect or a universal compression law.", "",
             "**Distillation is PENDING and not included:** its training runs are blocked on flaky GPUs. This analysis launches no training or inference.", "",
             "## Inputs and fixed analysis specification", "",
             "The 18 existing V6/V10 loss JSON files yield 9 size×step cells per arm and 36 compressed observations per capability/arm (216 total). "
             "Pruning uses densities 0.9/0.8/0.7/0.6 with dense key `1.0`; quantization uses bits 8/6/4/3 with dense key `dense`. "
             "Each response and predictor L₀ use that file's own measured capability-specific dense reference. "
             f"The maximum dense-reference discrepancy between arms is {f(summary['inputs']['max_arm_dense_difference'])}. "
             "Arms and capabilities are fitted and scored separately in nats per native token, with no pooled headline score.", "",
             "N₀ is computed from official GPT-NeoX architecture configs: `layers × (4h² + 2hm)` for attention QKV/output and the two MLP matrices. "
             "It excludes embeddings, the LM head, biases and normalization vectors. These are exact matrix counts, not nominal model names or the paper's slightly larger non-embedding counts including vectors. "
             "The dimensions below were transcribed from local cached official configs, verified equal across the three revisions per size; pinned config URLs and file SHA256s are in summary.json. "
             "Reproduction needs no weights or config download.", "",
             "| Size label | Layers | h | m | N₀ matrix parameters |",
             "|---|---:|---:|---:|---:|"]
    for s, a in summary["inputs"]["architectures"].items():
        lines.append(f"| [{s}]({a['source_url']}) | {a['num_hidden_layers']} | {a['hidden_size']} | {a['intermediate_size']} | {a['N0']:,} |")
    lines += ["", "The existing measurement code compresses **all language matrices, including embeddings and the LM head** "
              "(`language_weight_parameters` in V6, reused by V10). Thus N₀ is a size covariate rather than the total compressed parameter count. "
              "V6 uses global magnitude pruning; V10 uses symmetric per-output-channel fake quantization. "
              "The grid runner specifies bf16 and 128 probes, with the measurement routines using the odd-indexed held-out half. "
              "Only aggregate capability losses are available here, so no probe-level or seed uncertainty is estimated.", "",
              f"D₀ is **processed pretraining tokens**: `step × 1024 × 2048`, from the [Pythia training batch specification]({PYTHIA_PAPER}). "
              "It is not unique-token count or distillation data volume.", "",
              "| Step | D₀ tokens | Billions (rounded) |", "|---|---:|---:|"]
    for s, d in summary["inputs"]["D0_by_step"].items():
        lines.append(f"| {s} | {d:,} | {d/1e9:.1f} |")
    lines += ["", "For each arm and capability, let q index its four measured configurations:", "",
              "```text", "A: F(N₀,L₀,q)    = α_q + β_q log(N₀/10⁹) + γ_q L₀",
              "B: F(N₀,L₀,D₀,q) = α_q + β_q log(N₀/10⁹) + γ_q L₀ + δ_q log(D₀/10⁹)", "```", "",
              "All coefficients are fitted jointly by ordinary least squares against the **raw configuration-level ΔL responses**. "
              "There is no per-source a_c estimation followed by label regression. Configuration indicators and their covariate interactions let int3 have a different response from high bits without forcing a common amplitude or shape. "
              "A has 12 coefficients and B has 16; setting B's four D₀ coefficients to zero gives A. "
              "Each configuration effectively has six training cells and three (A) or four (B) coefficients. "
              "No model-form selection, regularization, hyperparameter tuning, log-response transform or prediction clipping is used. "
              "Input standardization uses only the training rows of each fold. Full-rank designs are required. "
              "The domain is the four measured configurations; there is no unseen-configuration prediction claim. Dense anchors supply references and are not fitted/scored as artificial zero-damage rows.", "",
              "The same A/B specification is used for **leave-one-SIZE-out** (all three steps/configurations of that size withheld) and, separately, "
              "**leave-one-STEP-out** (all sizes/configurations at that step withheld). Each of the three folds has 24 training and 12 test observations per arm/capability, "
              "covering six training and three test source cells. The middle size/step is interpolation and the endpoints require extrapolation. "
              "Every observation is predicted once per holdout scheme. Target compressed outcomes never enter training or preprocessing; target dense L₀ is an explicitly permitted input.", "",
              "**Pre-specified handling for this run:** retain and flag all int3 rows, all `|ΔL| ≤ 0.01` near-zero rows, all negative responses, and all `ΔL > 1` large-damage rows. "
              "The latter is a descriptive large-damage threshold, not a validated definition of collapse. "
              "No censoring, clipping, winsorization or response-dependent weights enter either fit or primary MAE. "
              "Missing/nonfinite values fail the run instead of disappearing. These rules and forms were fixed before running V36 fits; the existing smoke results were known, so this is **retrospective**, not a new prospective preregistration.", "",
              "## Held-out prediction: does D₀ help beyond L₀?", "",
              "MAE weights all 36 held-out observations equally within a capability/arm. Positive **A−B** means adding D₀ reduces error. "
              "Brackets are paired 95% cluster-bootstrap intervals. The bootstrap enumerates all 27 ordered resamples of the three whole held-out sizes or steps, "
              "carrying every configuration and the other axis together, and uses discrete inverse-CDF percentiles. Fits remain fixed. "
              "With only three groups, overlapping training folds and shared trajectories/probes, these are coarse **panel-conditional descriptive intervals**, "
              "not independent-seed, retraining, population, probe or causal uncertainty. The leave-one-group-out score range in summary.json is a sensitivity range, not another CI. "
              "There is no multiplicity correction across the 12 comparisons.", "",
              "| Arm | Capability | Holdout | A MAE [95%] | B MAE [95%] | A−B [95%] | Assessment |",
              "|---|---|---|---:|---:|---:|---|"]
    for arm in ARMS:
        for cap in CAPS:
            for scheme, result in summary["results"][arm][cap]["holdouts"].items():
                m = result["metrics"]
                assessment = ("lower error throughout interval" if m["improvement_ci95"][0] > 0 else
                              "higher error throughout interval" if m["improvement_ci95"][1] < 0 else "inconclusive")
                lines.append(f"| {arm} | {cap} | {HOLDOUTS[scheme]} | {ci(m['A_mae'], m['A_mae_ci95'])} | {ci(m['B_mae'], m['B_mae_ci95'])} | {ci(m['improvement'], m['improvement_ci95'])} | {assessment} |")
    lines.append("")
    for arm in ARMS:
        for scheme, key in HOLDOUTS.items():
            metrics = {c: summary["results"][arm][c]["holdouts"][scheme]["metrics"] for c in CAPS}
            improved = [c for c, m in metrics.items() if m["improvement"] > 0]
            supported = [c for c, m in metrics.items() if m["improvement_ci95"][0] > 0]
            lines.append(f"For **{arm}, held-out {key}**, D₀ lowers point-estimate MAE for {', '.join(improved) or 'no capability'}; "
                         f"the descriptive improvement interval lies wholly above zero for {', '.join(supported) or 'no capability'}. "
                         "This answers the conditional question for the stated model class and panel.")
            lines.append("")
    lines += ["A strong raw checkpoint trend alone does not establish a D₀ gain beyond dense L₀. "
              "Conversely, failure of this simple B model does not rule out a nonlinear history effect. "
              "D₀ and dense L₀ are strongly related during training; coefficient signs should not be interpreted causally. "
              "Training-design condition numbers and D₀/L₀ correlations are retained for every fold.", "",
              "### Every fold (positive improvement favors B)", "",
              "| Arm | Capability | Axis | Held out | A MAE | B MAE | A−B | B design condition number |",
              "|---|---|---|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        for cap in CAPS:
            for scheme, result in summary["results"][arm][cap]["holdouts"].items():
                for fold in result["folds"]:
                    lines.append(f"| {arm} | {cap} | {HOLDOUTS[scheme]} | {fold['held_out']} | {f(fold['A_mae'])} | {f(fold['B_mae'])} | {f(fold['improvement'])} | {fold['fits']['B']['condition_number']:.2f} |")
    lines += ["", "### Configuration contributions", "",
              "These are slices of the same primary out-of-fold predictions, with no refitting or removals. "
              "They expose int3's contribution and the near-zero high-bit errors; primary comparisons above retain all configurations.", "",
              "| Arm | Capability | Config | Size A / B / A−B | Step A / B / A−B |", "|---|---|---:|---:|---:|"]
    for arm in ARMS:
        for cap in CAPS:
            for q in CONFIGS[arm]:
                entries = []
                for scheme in HOLDOUTS:
                    m = summary["results"][arm][cap]["holdouts"][scheme]["by_config"][f"{q:g}"]
                    entries.append(" / ".join(f(m[k]) for k in ("A_mae", "B_mae", "improvement")))
                lines.append(f"| {arm} | {cap} | {q:g} | {' | '.join(entries)} |")
    lines += ["", "## Raw D₀ trend at fixed size and configuration", "",
              "Fragility here means **signed loss damage**: higher ΔL is worse. Negative ΔL means the measured loss improved under compression. "
              "Monotonicity checks both adjacent differences across all three steps (numerical tolerance 10⁻¹², not a measurement-noise test). "
              "The smoke's roughly 3–5× claim is checked against all matched trajectories, rather than assumed to hold across capabilities/configurations. "
              "Ratios are descriptive and reported only when both early and late damage exceed 0.01; unstable or nonpositive endpoint ratios are marked N/A with raw observations retained. "
              "Absolute-damage trajectories and endpoint changes are also recorded separately in summary.json.", "",
              "| Arm | Capability | Late > early / 12 | Increasing / decreasing / flat / nonmonotonic | Positive-damage ratio median [range]; eligible n |",
              "|---|---|---:|---|---|"]
    for arm in ARMS:
        for cap in CAPS:
            t = summary["results"][arm][cap]["raw_trend"]
            counts = t["monotonicity_counts"]
            ratio = (f"{t['positive_damage_ratio_median']:.2f} [{t['positive_damage_ratio_range'][0]:.2f}, {t['positive_damage_ratio_range'][1]:.2f}]"
                     if t["positive_damage_ratio_count"] else "N/A")
            lines.append(f"| {arm} | {cap} | {t['late_greater_than_early']} / 12 | " +
                         " / ".join(str(counts[k]) for k in ("increasing", "decreasing", "flat", "nonmonotonic")) +
                         f" | {ratio}; n={t['positive_damage_ratio_count']} |")
    lines.append("")
    for arm in ARMS:
        q = CONFIGS[arm][-1]
        examples = []
        for cap in ("math", "code"):
            ts = summary["results"][arm][cap]["raw_trend"]["trajectories"]
            for size in ("410m", "1.4b"):
                t, = [t for t in ts if t["size"] == size and t["config"] == q]
                ratio = t['late_over_early_positive_damage']
                examples.append(f"{size} {cap} {ratio:.2f}× ({t['monotonicity']})" if ratio is not None else
                                f"{size} {cap}: ratio undefined ({t['monotonicity']})")
        lines += [f"At {arm} config {q:g}, the late/early damage ratios are " + "; ".join(examples) + ".", ""]
    lines += ["The 3–5× smoke description is therefore not a universal magnitude across configurations and capabilities. "
              "QA's signed responses and the full three-step trajectories are shown below; a loss improvement is not relabeled as positive damage. "
              "Any exact-repeat checkpoint flags qualify the corresponding flat trajectories and aggregate ratios.", ""]
    lines += ["", "### All matched step trajectories", "",
              "Steps are 16000 → 64000 → 143000. Flags apply if any step is near zero (`Z`), negative (`−`), "
              "above 1 nat damage (`H`), has a duplicated checkpoint payload (`C`), or if the configuration is int3 (`3`). All flagged observations enter the primary analysis.", "",
              "| Arm | Capability | Size | Config | ΔL at three steps | Late−early | Monotonicity | Late/early | Flags |",
              "|---|---|---|---:|---|---:|---|---:|---|"]
    for arm in ARMS:
        for cap in CAPS:
            for t in summary["results"][arm][cap]["raw_trend"]["trajectories"]:
                flags = "".join(label for key, label in (("any_near_zero", "Z"), ("any_negative", "−"),
                                                       ("any_large_damage", "H"), ("int3", "3"),
                                                       ("duplicate_checkpoint_payload", "C")) if t[key]) or "none"
                values = " → ".join(f(y) for y in t["observed"])
                lines.append(f"| {arm} | {cap} | {t['size']} | {t['config']:g} | {values} | {f(t['late_minus_early'])} | {t['monotonicity']} | {f(t['late_over_early_positive_damage'])} | {flags} |")
    lines += ["", "## Reproduction and audit trail", "", "```bash",
              "python3 analysis/v36_pythia_controlled_fit.py --dry-run",
              "python3 analysis/v36_pythia_controlled_fit.py",
              "python3 -m pytest -q tests/test_v36.py", "```", "",
              "Only NumPy and the Python standard library are used for analysis. The dry run prints this report without writing. "
              "`results/v36-pythia-controlled/summary.json` contains the SHA256 of each of the 18 loss inputs, "
              "architecture provenance, all fold memberships, training-only standardizers, direct-fit coefficients/ranks/condition numbers, "
              "every observed and held-out predicted response, paired uncertainty, flags and raw trajectories. "
              "Inputs are rehashed before outputs are written. This report is generated from that same summary; no results are manually substituted.", ""]
    return "\n".join(lines)


def write_outputs(summary, report, root=ROOT):
    root = Path(root)
    for relative, expected in summary["input_sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Input changed during analysis: {relative}")
    # Serialize before creating directories, and forbid non-standard NaN/Infinity.
    serialized = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    output = root / OUT.relative_to(ROOT)
    destination = root / REPORT.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(serialized)
    destination.write_text(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print report without writing")
    args = parser.parse_args(argv)
    summary = build_summary()
    report = render(summary)
    if args.dry_run:
        print(report, end="")
    else:
        write_outputs(summary, report)
        print(f"Wrote {OUT / 'summary.json'} and {REPORT}")


if __name__ == "__main__":
    main()
