#!/usr/bin/env python3
"""CPU-only, fixed retrospective interaction and training-exposure audit.

Reuses the strict V37 roster and signed V12 endpoints; never trains or evaluates
models. Protocol below is fixed before fitting, not a prospective preregistration.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

try:  # Both `python analysis/...py` and namespace-package imports.
    from . import v37_reuse_sufficiency as v37
except ImportError:
    import v37_reuse_sufficiency as v37

ROOT = v37.ROOT
STUDENTS, CAPS, SEEDS = v37.STUDENTS, v37.CAPS, v37.SEEDS
TOLERANCE_NAT = 0.05
D_REF = 100000.0
LARGELY_EXPLAINED_FRACTION = 0.5
LEVELS = ("matched_low", "matched_high", "reuse_8", "reuse_16")
MODELS = ("E_only", "interaction", "additive_diagnostic")
EXPOSURES = ("log_supervised_ratio", "optimizer_step_difference_per_100")
reuse_count = v37.reuse_count
paired_interval = v37.paired_interval

PROTOCOL = {
    "status": "Fixed retrospective protocol before V37b fits; V37 residuals already known; not prospective preregistration",
    "endpoint": "signed delta_c = post_training[c] - dense[c], legacy V12 nats/target token; distillation only",
    "practical_tolerance_nat": TOLERANCE_NAT,
    "models": {
        "E_only": "h_c(E) = b1*log1p(E) + b2*log1p(E)^2 (2 parameters)",
        "interaction": "h_c(E) + gamma_c*E*log(D_U/100000) (3 parameters); k_c(E)=gamma_c*E; h_c(0)=k_c(0)=0",
        "additive_diagnostic": "h_c(E) + a_c*log(D_U/100000) (3 parameters); unanchored comparator only, fails delta(0)=0",
    },
    "fit": "Separate student/capability weighted least squares; equal pools, then seeds, then checkpoints within pool/seed; no latent variables, sign constraints, fitted offsets, tuning, or student transfer claim",
    "folds": "Four leave-one-nominal-E-level-out folds within each student; all seeds/pools/run variants of that level held out together. Actual E enters basis. Levels determined only by requested milestone and pool, never outcomes or rounded actual E",
    "level_mapping": "matched_low/high: U75 low milestones paired with U600 500k/1M; reuse_8/16: U75 500k/1M. Names nominal; actual reuse is about 0.94/1.9/7.6/15",
    "scoring": "Primary MAE on held-out matched_low/high raw snapshots; all-four-level panel secondary with extrapolation flags. Equal pool/seed/checkpoint weighting. No interpolated outcome enters interaction fitting or scoring",
    "interaction_decision": "Held-out gain supported iff paired 95% CI for MAE(E_only)-MAE(interaction) is entirely above zero; practically material iff mean gain >=0.05 nat. Report additive comparison on identical folds separately",
    "matched_E": "At each actual U600 E, set U75 T*=E*D_U; interpolate U75 delta within dense/low/high checkpoints only. No extrapolation. Raw nominal gaps retained as sensitivity",
    "exposure_accounting": "Verify snapshots against train_log trajectory milestones and every-step loss_curve; interpolate cumulative steps/supervised tokens in processed-token space between logged optimizer updates, including zero origin. Fractional steps are estimates, not executed partial updates",
    "residual_regression": "Per student/capability paired gap r=delta75(E)-delta600(E); fit separately r=beta*log(S75/S600) and r=beta*(steps75-steps600)/100, one slope and no intercept. This is paired differencing of residual-versus-log(S) or steps regression: common E-only h cancels at fixed E",
    "exposure_validation": "Leave matched-low/high level out (2 folds), all seeds together. Also fit a one-parameter constant-gap comparator on identical train folds; no jointly fitted collinear exposures. Full-data slopes and uncentered R2 descriptive only",
    "largely_explained_rule": "Cross-level SSE reduction versus zero gap >=50% AND corrected signed-gap paired 95% intervals at BOTH E levels entirely within +/-0.05 nat. Gain over constant-gap comparator reported separately; this rule is predictive, never causal",
    "intervals": "Paired two-sided 95% Student-t intervals over n=3 seed blocks, df=2, averaging levels/pools within seed before computing intervals. Fixed out-of-fold predictions, not refitting uncertainty; checkpoints never counted as independent seeds; pointwise intervals",
    "schedule": "Record optimizer, base learning rate, warmup steps/ratio, total planned updates and checkpoint schedule fraction. Logged lr is AFTER scheduler.step(), the next-update rate, not an integrated dose; schedule variables are diagnostics only",
}


def e_level(row):
    student, pool, milestone = row["student"], row["pool"], row["milestone"]
    if student not in STUDENTS or pool not in (75, 600):
        raise ValueError("Unknown student/pool")
    milestones = v37.LOW_MILESTONES[student] if pool == 75 else v37.HIGH_MILESTONES
    if milestone in milestones:
        return LEVELS[milestones.index(milestone)]
    if pool == 75 and milestone in v37.HIGH_MILESTONES:
        return LEVELS[2 + v37.HIGH_MILESTONES.index(milestone)]
    raise ValueError("Unknown nominal E-level milestone")


def validate_train_log(log):
    """Reject inconsistent cumulative accounting instead of estimating from epochs."""
    reuse_count(log["processed_tokens"], log["unique_data_pool_tokens"])
    if not log["loss_curve"]:
        raise ValueError("Empty optimizer loss curve")
    previous_t = previous_s = 0
    for step, item in enumerate(log["loss_curve"], 1):
        t, s = item["processed_tokens"], item["completion_tokens_seen"]
        if (item["step"] != step or not math.isfinite(t) or not math.isfinite(s)
                or t <= previous_t or s <= previous_s or s > t
                or t - previous_t != item["tokens"]
                or s - previous_s != item["completion_tokens"]
                or item["unique_data_pool_tokens"] != log["unique_data_pool_tokens"]):
            raise ValueError("Inconsistent per-step cumulative accounting")
        if not math.isfinite(item["lr"]) or item["lr"] < 0:
            raise ValueError("Invalid logged learning rate")
        previous_t, previous_s = t, s
    if (len(log["loss_curve"]) != log["optimizer_steps"] or log["updates"] != log["optimizer_steps"]
            or previous_t != log["processed_tokens"] or previous_s != log["completion_tokens_seen"]):
        raise ValueError("Full-run totals disagree with optimizer curve")
    if (log["total_updates_planned"] < log["optimizer_steps"]
            or log["warmup_steps"] != int(log["warmup_ratio"] * log["total_updates_planned"])
            or not math.isfinite(log["learning_rate"]) or log["learning_rate"] <= 0):
        raise ValueError("Inconsistent training schedule")
    for milestone in log["trajectory"]:
        item = log["loss_curve"][milestone["updates"] - 1]
        if any(milestone[k] != item[k] for k in ("processed_tokens", "completion_tokens_seen", "lr")):
            raise ValueError("Trajectory milestone disagrees with optimizer curve")


def interpolate_exposure(log, processed_tokens):
    """Linear interpolation between adjacent logged updates; never extrapolate."""
    t = float(processed_tokens)
    curve = log["loss_curve"]
    grid = np.array([0.] + [p["processed_tokens"] for p in curve])
    if not math.isfinite(t) or t < 0 or t > grid[-1]:
        raise ValueError("Exposure interpolation would require extrapolation")
    right = int(np.searchsorted(grid, t, side="left"))
    left = max(0, right - 1) if grid[right] != t else right
    fraction = float((t-grid[left])/(grid[right]-grid[left])) if right != left else 0.
    steps = float(np.interp(t, grid, [0.] + [p["step"] for p in curve]))
    supervised = float(np.interp(t, grid, [0.] + [p["completion_tokens_seen"] for p in curve]))
    return {"processed_tokens": t, "D_U": log["unique_data_pool_tokens"],
            "E": reuse_count(t, log["unique_data_pool_tokens"]),
            "optimizer_steps_so_far": steps, "completion_tokens_seen": supervised,
            "schedule_fraction": steps / log["total_updates_planned"],
            "interpolation": {"bracket_steps": [left, right],
                              "bracket_tokens": [float(grid[left]), float(grid[right])],
                              "fraction": fraction, "exact_logged_update": left == right}}


def load_data(root=ROOT):
    root = Path(root)
    rows, runs, hashes = v37.load_trajectories(root)
    logs = {}
    for run in runs:
        log = json.loads((root / run["train_log"]).read_text())
        validate_train_log(log)
        if any(log[k] != run[k] for k in ("student", "seed", "learning_rate", "scheduler", "total_updates_planned")):
            raise ValueError("Run metadata disagrees with train_log")
        logs[run["trajectory_id"]] = log
        run.update({k: log[k] for k in ("optimizer_steps", "completion_tokens_seen", "processed_tokens",
                                        "optimizer", "warmup_steps", "warmup_ratio", "effective_batch_size_sequences")})
        run["final_E"] = reuse_count(log["processed_tokens"], log["unique_data_pool_tokens"])
    for row in rows:
        log = logs[row["trajectory_id"]]
        matches = [m for m in log["trajectory"] if m["requested_token_milestones"] == [row["milestone"]]]
        if len(matches) != 1:
            raise ValueError("Missing or duplicate train_log trajectory milestone")
        milestone = matches[0]
        snap = json.loads((root / row["row_id"]).read_text())
        if any(snap[k] != milestone[k] for k in ("processed_tokens", "updates", "completion_tokens_seen", "lr")):
            raise ValueError("Snapshot exposure disagrees with train_log milestone")
        row["E_level"] = e_level(row)
        row["exposure"] = interpolate_exposure(log, row["processed_tokens"])
        row["logged_next_update_lr"] = milestone["lr"]
    return rows, runs, logs, hashes


def matched_pairs(rows, logs):
    pairs = []
    for student in STUDENTS:
        for seed in SEEDS:
            small = sorted([r for r in rows if r["student"] == student and r["seed"] == seed
                            and r["pool"] == 75 and r["E_level"] in LEVELS[:2]], key=lambda r: r["E"])
            grid = [0.] + [r["E"] for r in small]
            for level in LEVELS[:2]:
                a, = [r for r in small if r["E_level"] == level]
                b, = [r for r in rows if (r["student"], r["seed"], r["pool"], r["E_level"]) ==
                      (student, seed, 600, level)]
                target_e = b["E"]
                if not grid[0] <= target_e <= grid[-1]:
                    raise ValueError("Matched outcome interpolation would require extrapolation")
                aligned = {c: float(np.interp(target_e, grid, [0.] + [r["delta"][c] for r in small])) for c in CAPS}
                ea = interpolate_exposure(logs[a["trajectory_id"]], target_e*a["D_U"])
                eb = b["exposure"]
                ratios = {k: eb[k]/ea[k] for k in ("D_U", "processed_tokens", "optimizer_steps_so_far", "completion_tokens_seen")}
                pairs.append({"pair_id": f"{student}/seed{seed}/{level}", "student": student,
                              "seed": seed, "E_level": level, "E": target_e,
                              "U75_source": a["row_id"], "U600_source": b["row_id"],
                              "outcome_interpolation_sources": [r["row_id"] for r in small],
                              "outcome_interpolation_E_grid": grid,
                              "nominal_U75": a["exposure"], "nominal_U600": eb,
                              "relative_nominal_E_mismatch": a["E"]/target_e-1,
                              "aligned_U75": ea, "aligned_U600": eb,
                              "U600_over_U75": ratios,
                              "aligned_delta_U75": aligned, "delta_U600": b["delta"],
                              "residual": {c: aligned[c]-b["delta"][c] for c in CAPS},
                              "nominal_residual": {c: a["delta"][c]-b["delta"][c] for c in CAPS},
                              "exposure_predictors": {
                                  "log_supervised_ratio": math.log(ea["completion_tokens_seen"]/eb["completion_tokens_seen"]),
                                  "optimizer_step_difference_per_100": (ea["optimizer_steps_so_far"]-eb["optimizer_steps_so_far"])/100}})
    return pairs


def confound_summary(pairs, runs):
    columns = ("log_D_U", "log_T", "log_steps", "log_supervised")
    strata, identity_errors = [], []
    for p in pairs:
        a, b = p["aligned_U75"], p["aligned_U600"]
        values = np.log([[q[k] for k in ("D_U", "processed_tokens", "optimizer_steps_so_far", "completion_tokens_seen")]
                         for q in (a, b)])
        strata.extend(values-values.mean(axis=0))
        identity_errors.extend(abs(math.log(q["processed_tokens"])-math.log(p["E"])-math.log(q["D_U"])) for q in (a, b))
    matrix = np.array(strata)
    standardized = matrix / matrix.std(axis=0)
    singular = np.linalg.svd(standardized, compute_uv=False)
    ratios = {k: {"min": min(p["U600_over_U75"][k] for p in pairs),
                  "max": max(p["U600_over_U75"][k] for p in pairs)} for k in pairs[0]["U600_over_U75"]}
    example = {r["pool"]: r for r in runs if r["student"] == "gemma3-1b" and r["seed"] == 0
               and ((r["pool"] == 75 and r["variant"] == "uxseenE") or r["pool"] == 600)}
    return {"columns": list(columns), "within_pair_centered_log_correlation": np.corrcoef(matrix.T).tolist(),
            "centered_standardized_singular_values": singular.tolist(),
            "centered_standardized_rank": int(np.linalg.matrix_rank(standardized)),
            "condition_number": None, "condition_note": "Singular: within exact-E pair log T and log D_U are identical after centering; cannot identify separate coefficients",
            "max_abs_logT_minus_logE_minus_logD": max(identity_errors), "U600_over_U75_ratio_ranges": ratios,
            "gemma_full_run_diagnostic": {str(pool): {k: r[k] for k in
                ("trajectory_id", "train_log", "D_U", "final_E", "optimizer_steps", "completion_tokens_seen", "processed_tokens")}
                for pool, r in example.items()}}


def make_folds(rows):
    """Split nominal levels, not seed jitter; no held-level target is fitted."""
    folds = []
    for student in sorted({r["student"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["student"] == student]
        levels = {e_level(rows[i]) for i in indices}
        if levels != set(LEVELS):
            raise ValueError("Need all four nominal E levels for each student")
        for level in LEVELS:
            train = [i for i in indices if e_level(rows[i]) != level]
            test = [i for i in indices if e_level(rows[i]) == level]
            folds.append({"student": student, "held_out_E_level": level,
                          "train_indices": train, "test_indices": test})
    return folds


def design_matrix(inputs, model):
    if model not in MODELS:
        raise ValueError("Unknown response model")
    if not inputs or any(set(r) != {"E", "D_U"} for r in inputs):
        raise ValueError("Prediction accepts only E and D_U")
    e = np.array([r["E"] for r in inputs], dtype=float)
    d = np.array([r["D_U"] for r in inputs], dtype=float)
    if np.any(~np.isfinite(e)) or np.any(e < 0) or np.any(~np.isfinite(d)) or np.any(d <= 0):
        raise ValueError("Invalid E or D_U")
    x, z = np.log1p(e), np.log(d/D_REF)
    base = np.column_stack([x, x*x])
    if model == "E_only":
        return base
    return np.column_stack([base, e*z if model == "interaction" else z])


def fit_response(rows, capability, model):
    if len({r["student"] for r in rows}) != 1:
        raise ValueError("Fit each student separately")
    matrix = design_matrix(v37.prediction_inputs(rows), model)
    y = np.array([r["delta"][capability] for r in rows])
    if np.any(~np.isfinite(y)):
        raise ValueError("Nonfinite signed response")
    w = np.sqrt(v37.balanced_weights(rows))
    coef, _, rank, singular = np.linalg.lstsq(matrix*w[:, None], y*w, rcond=None)
    if rank != matrix.shape[1]:
        raise ValueError("Rank deficient response design; no minimum-norm fallback")
    return {"model": model, "coefficients": coef.tolist(), "n_parameters": matrix.shape[1],
            "rank": int(rank), "condition_number": float(singular[0]/singular[-1])}


def predict_response(fit, inputs):
    return design_matrix(inputs, fit["model"]) @ np.array(fit["coefficients"])


def prediction_metrics(records):
    """Same observations and equal pool weights for every paired MAE contrast."""
    blocks = defaultdict(lambda: defaultdict(list))
    for r in records:
        err = [abs(r["observed"]-r["predictions"][m]) for m in MODELS]
        blocks[r["seed"]][r["pool"]].append(err + [err[0]-err[1], err[2]-err[1]])
    names = ("mae_E_only", "mae_interaction", "mae_additive_diagnostic",
             "gain_over_E_only", "gain_over_additive")
    values = {s: np.mean([np.mean(v, axis=0) for v in pools.values()], axis=0) for s, pools in blocks.items()}
    metrics = {name: paired_interval({s: v[i] for s, v in values.items()}) for i, name in enumerate(names)}
    metrics["n_snapshots"] = len(records)
    metrics["supported_interaction_gain"] = metrics["gain_over_E_only"]["ci95"][0] > 0
    metrics["practically_material_gain"] = metrics["gain_over_E_only"]["mean"] >= TOLERANCE_NAT
    return metrics


def cross_validate(rows):
    folds, records = [], []
    for fold in make_folds(rows):
        train, test = [[rows[i] for i in fold[k]] for k in ("train_indices", "test_indices")]
        lo, hi = min(r["E"] for r in train), max(r["E"] for r in train)
        detail = {"student": fold["student"], "held_out_E_level": fold["held_out_E_level"],
                  "train_row_ids": [r["row_id"] for r in train], "test_row_ids": [r["row_id"] for r in test],
                  "train_E_range": [lo, hi], "fits": {}}
        for cap in CAPS:
            fits = {m: fit_response(train, cap, m) for m in MODELS}
            pred = {m: predict_response(fits[m], v37.prediction_inputs(test)) for m in MODELS}
            detail["fits"][cap] = fits
            records.extend({**{k: r[k] for k in ("row_id", "student", "seed", "pool", "E_level", "E", "D_U")},
                            "capability": cap, "observed": r["delta"][cap],
                            "predictions": {m: float(pred[m][i]) for m in MODELS},
                            "E_outside_training_range": not lo <= r["E"] <= hi} for i, r in enumerate(test))
        folds.append(detail)
    metrics = {}
    for student in STUDENTS:
        metrics[student] = {}
        for cap in CAPS:
            subset = [r for r in records if r["student"] == student and r["capability"] == cap]
            metrics[student][cap] = {
                "matched_levels_primary": prediction_metrics([r for r in subset if r["E_level"] in LEVELS[:2]]),
                "all_levels_secondary": prediction_metrics(subset),
                "per_level": {lev: prediction_metrics([r for r in subset if r["E_level"] == lev]) for lev in LEVELS}}
    return {"folds": folds, "predictions": records, "metrics": metrics}


def exposure_regression(pairs, student, cap, predictor, residual_key="residual"):
    if predictor not in EXPOSURES:
        raise ValueError("Unknown exposure predictor")
    subset = [p for p in pairs if p["student"] == student]
    x = np.array([p["exposure_predictors"][predictor] for p in subset])
    y = np.array([p[residual_key][cap] for p in subset])
    if not len(x) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or float(x@x) <= 0:
        raise ValueError("Need finite, nonzero exposure differences")
    slope = float(x@y/(x@x))
    pred, constant, folds = np.zeros(len(y)), np.zeros(len(y)), []
    for level in LEVELS[:2]:
        test = np.array([p["E_level"] == level for p in subset])
        train = ~test
        if not train.any() or not test.any() or float(x[train]@x[train]) <= 0:
            raise ValueError("Need both E levels and nonzero training exposure differences")
        beta = float(x[train]@y[train]/(x[train]@x[train]))
        pred[test], constant[test] = beta*x[test], y[train].mean()
        folds.append({"held_out_E_level": level, "slope": beta, "constant_gap": float(y[train].mean()),
                      "train_pair_ids": [p["pair_id"] for p, keep in zip(subset, train) if keep],
                      "test_pair_ids": [p["pair_id"] for p, keep in zip(subset, test) if keep]})
    corrected = y-pred
    by_seed = {s: np.array([p["seed"] == s for p in subset]) for s in SEEDS}
    quantities = {"mae_zero_gap": np.abs(y), "mae_exposure": np.abs(corrected),
                  "mae_constant_gap": np.abs(y-constant),
                  "mae_gain_over_zero": np.abs(y)-np.abs(corrected),
                  "mae_gain_over_constant": np.abs(y-constant)-np.abs(corrected)}
    intervals = {name: paired_interval({s: float(v[keep].mean()) for s, keep in by_seed.items()})
                 for name, v in quantities.items()}
    level_intervals = {level: paired_interval({p["seed"]: float(corrected[i]) for i, p in enumerate(subset)
                                               if p["E_level"] == level}) for level in LEVELS[:2]}
    total = float(y@y)
    fraction = float(1-corrected@corrected/total) if total > 0 else None
    within = all(v37.interval_within_tolerance(ci, TOLERANCE_NAT) for ci in level_intervals.values())
    return {"predictor": predictor, "residual_key": residual_key, "n_parameters": 1, "n_pairs": len(subset),
            "full_data_slope_descriptive": slope,
            "full_data_uncentered_R2_descriptive": float(1-np.sum((y-slope*x)**2)/total) if total > 0 else None,
            "cross_level_SSE_fraction_explained": fraction,
            "metrics": intervals, "corrected_residual_by_level": level_intervals,
            "largely_explained": fraction is not None and fraction >= LARGELY_EXPLAINED_FRACTION and within,
            "supported_gain_over_constant": intervals["mae_gain_over_constant"]["ci95"][0] > 0,
            "folds": folds,
            "predictions": [{"pair_id": p["pair_id"], "seed": p["seed"], "E_level": p["E_level"],
                             "x": float(x[i]), "residual": float(y[i]), "prediction": float(pred[i]),
                             "constant_prediction": float(constant[i]), "corrected_residual": float(corrected[i])}
                            for i, p in enumerate(subset)]}


def prior_v37_provenance(root):
    path = Path(root) / "results/v37-reuse-sufficiency/summary.json"
    if not path.exists():
        return {"status": "prior summary unavailable", "note": "V37 loader/design implementation is the local dependency; do not label its volume basis additive"}
    raw = path.read_bytes()
    prior = json.loads(raw)
    return {"status": "verified local artifact", "source": str(path.relative_to(root)),
            "sha256": hashlib.sha256(raw).hexdigest(), "actual_model": prior["protocol"]["primary_model"],
            "note": "V37 already used x*z and x^2*z interactions, not an additive log D term. V37b uses one extra E*z coefficient and different within-student E-level folds; its new additive comparator is evaluated on those same folds",
            "reported_MAE_reductions": {scheme: {cap: prior["held_out"][scheme]["capabilities"][cap]["metrics"]["mae_reduction_with_log_D"]
                                               for cap in CAPS} for scheme in v37.SCHEMES}}


def build_summary(root=ROOT):
    rows, runs, logs, hashes = load_data(root)
    pairs = matched_pairs(rows, logs)
    cv = cross_validate(rows)
    exposure = {s: {c: {x: exposure_regression(pairs, s, c, x) for x in EXPOSURES} for c in CAPS} for s in STUDENTS}
    # Raw gaps with their RAW exposures: a nominal-E sensitivity, not exact E.
    nominal = []
    for p in pairs:
        a, b = p["nominal_U75"], p["nominal_U600"]
        nominal.append({**p, "exposure_predictors": {
            "log_supervised_ratio": math.log(a["completion_tokens_seen"]/b["completion_tokens_seen"]),
            "optimizer_step_difference_per_100": (a["optimizer_steps_so_far"]-b["optimizer_steps_so_far"])/100}})
    nominal_exposure = {s: {c: {x: exposure_regression(nominal, s, c, x, "nominal_residual") for x in EXPOSURES}
                            for c in CAPS} for s in STUDENTS}
    decisions = {}
    for s in STUDENTS:
        decisions[s] = {}
        for c in CAPS:
            m = cv["metrics"][s][c]["matched_levels_primary"]
            explained = [x for x in EXPOSURES if exposure[s][c][x]["largely_explained"]]
            decisions[s][c] = {"interaction_gain_supported": m["supported_interaction_gain"],
                               "interaction_gain_material": m["practically_material_gain"],
                               "exposure_models_meeting_rule": explained,
                               "exposure_explanation": "predictively compatible with measured exposure" if explained else
                                   "simple measured-exposure models do not explain residual to fixed tolerance",
                               "latent_variable_required": False,
                               "interpretation": "This panel cannot identify a causal exposure effect or justify a new latent variable; pool, exposure, and schedule remain confounded"}
    return {"version": "37b", "priority": 3, "cpu_only": True, "new_training_runs": 0,
            "protocol": PROTOCOL, "v37_provenance": prior_v37_provenance(root),
            "data": {"n_trajectories": len(runs), "n_nonbaseline_snapshots": len(rows),
                     "runs": runs, "snapshots": rows, "source_sha256": hashes},
            "confound": confound_summary(pairs, runs), "matched_E_pairs": pairs,
            "held_out_interaction": cv, "residual_vs_exposure": exposure,
            "nominal_E_exposure_sensitivity": nominal_exposure, "decisions": decisions,
            "limitations": [
                "Only two students, two fixed nested pools, one teacher/recipe, and three shuffle/training seeds. No data-subset or evaluation-sample resampling; QA has only 251/271 measured target tokens.",
                "At fixed E, log T = log E + log D_U exactly. Steps and supervised tokens track T closely; regression cannot separate unique-data diversity from cumulative exposure or schedule effects.",
                "All E-level CV predictions condition on other saved checkpoints of the same runs. These are held-level curve checks, not independent-run, held-pool, or new-student transfer estimates.",
                "Matched-level interaction fits use high-reuse U75 checkpoints (E about 7.5 and 15). In Gemma these come from a different schedule horizon than uxseenE; no schedule covariate is fitted. All-level secondary scores include substantial extrapolation.",
                "Only two matched E levels identify the exposure diagnostic. Log supervised ratios are nearly constant; comparison with a constant gap is necessary. No ordinary six-independent-observation slope p-values are reported.",
                "Interpolated signed outcomes use observed target checkpoints, sometimes including the other matched level or the dense origin. Exposure cross-level validation is therefore descriptive interpolation sensitivity, not independent held-out-outcome prediction; raw nominal sensitivity is also reported.",
                "A low-DOF model failing does not rule out nonlinear optimization/exposure effects. A model succeeding does not establish causation; neither result demonstrates a new latent capability variable.",
            ]}


def _ci(value):
    lo, hi = value["ci95"]
    return f"{value['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def render_report(summary):
    conf = summary["confound"]
    example = conf["gemma_full_run_diagnostic"]
    small, large = example["75"], example["600"]
    supported = [f"{s}/{c}" for s, caps in summary["decisions"].items() for c, d in caps.items()
                 if d["interaction_gain_supported"]]
    lines = ["# V37b: reuse interaction and training-exposure confound", "",
             "**Matched reuse does not match training exposure.** The train logs verify Gemma U75 uxseenE: "
             f"{small['optimizer_steps']} optimizer steps / {small['completion_tokens_seen']:,} supervised completion tokens, versus U600 uxseen: "
             f"{large['optimizer_steps']} / {large['completion_tokens_seen']:,}, full-run E={small['final_E']:g} / {large['final_E']:g}. "
             "These full-run totals are not substituted for snapshot exposure. "
             "The exact matched-E tables below quantify the same confound at each evaluated comparison.", "",
             ("The primary matched-level interaction gain is supported for " + ", ".join(supported) + "." if supported else
              "**The prespecified one-coefficient interaction has no supported primary matched-level MAE gain for any student/capability.**"), "",
             "This Priority-3 CPU-only audit reuses 15 runs and 36 nonbaseline snapshots, fits signed capability "
             "loss changes separately for each student and math/code/QA, and introduces no latent variable. "
             "Positive delta means worse loss; negative means improvement. Pruning and quantization are outside this distillation panel.", "",
             "## Fixed protocol and provenance", "",
             "The retrospective specification was fixed before V37b fitting, with previously observed V37 residuals already known. "
             "It is not a prospective preregistration. Practical tolerance is **0.05 nat per target token**, unchanged across capabilities. "
             "All logarithms are natural; D_ref=100,000 input tokens.", "",
             "**Correction to the premise:** the checked-in V37 model already used `x*z` and `x^2*z`, "
             "where x=log1p(E), z=log(D_U/100000). It was an interaction, not an additive log-D model. "
             "V37b tests a stricter one-coefficient interaction; its additive comparator is newly evaluated on identical V37b folds. "
             "Different student conditioning and folds prevent treating a change from V37 scores as an interaction-only improvement.", "",
             "1. E-only: delta_hat = b1*x + b2*x² (2 parameters).",
             "2. Interaction: delta_hat = b1*x + b2*x² + gamma*E*z (3 parameters), so k(E)=gamma*E and h(0)=k(0)=0.",
             "3. Additive diagnostic: delta_hat = b1*x + b2*x² + a*z (3 parameters). This comparator violates the zero-reuse anchor and is not an eligible anchored response law.", "",
             "Separate student/capability weighted least squares gives equal pools, then seeds, then checkpoints per pool/seed. "
             "There is no tuning, fitted offset, sign restriction, or target calibration. Four nominal E-level folds hold out all seeds "
             "and pools at matched-low (~1), matched-high (~2), U75 ~8, or U75 ~16 reuse. Actual T/D_U enters the basis. "
             "The primary score uses the 24 raw matched-level snapshots; all 36 snapshots form a secondary score. "
             "High-reuse U75 observations support the three-parameter fit when one matched level is held out. "
             "The same run can occur at another training level; this estimates held-level prediction, not new-run or new-student transfer.", "",
             "Gain means MAE(E-only) minus MAE(interaction). A gain is supported when its paired 95% interval is entirely positive; "
             "a mean gain of at least 0.05 nat is practically material. Intervals are Student-t over three seed blocks (df=2), "
             "averaging levels/pools within each seed first. They condition on fixed OOF predictions, not model-refitting uncertainty. "
             "They are pointwise and do not treat checkpoints as independent replicates.", "",
             "## Training-exposure confound", "",
             "Input T includes prompt and completion tokens after truncation, with repetitions. D_U is one-pass input-token pool volume. "
             "Supervised S is `completion_tokens_seen`, not T. Every snapshot is checked against the matching train-log milestone "
             "and the per-update cumulative curve. No final-run total is scaled down to invent snapshot counts.", "",
             "At each U600 snapshot E, U75 is evaluated at T*=E*D_U. Steps and S are linearly interpolated between adjacent "
             "logged updates, while delta is interpolated between dense/low/high saved checkpoints without extrapolation. "
             "Fractional steps describe interpolation only. The JSON retains original E/T/steps/S, interpolation brackets, "
             "signed nominal gaps, exact-E gaps and source paths for every pair.", "",
             "| Student | Level | Seed | Matched E | Nominal U75 E | D_U 75 / 600 | Aligned T 75 / 600 | Steps 75 / 600 | Supervised S 75 / 600 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in sorted(summary["matched_E_pairs"], key=lambda p: (p["student"], LEVELS.index(p["E_level"]), p["seed"])):
        a, b = p["aligned_U75"], p["aligned_U600"]
        lines.append(f"| {p['student']} | {p['E_level']} | {p['seed']} | {p['E']:.5f} | {p['nominal_U75']['E']:.5f} | "
                     f"{a['D_U']:,} / {b['D_U']:,} | {a['processed_tokens']:,.1f} / {b['processed_tokens']:,.0f} | "
                     f"{a['optimizer_steps_so_far']:.2f} / {b['optimizer_steps_so_far']:.0f} | "
                     f"{a['completion_tokens_seen']:,.1f} / {b['completion_tokens_seen']:,.0f} |")
    lines += ["", "| U600 / U75 at matched E | Minimum | Maximum |", "|---|---:|---:|"]
    for k, v in conf["U600_over_U75_ratio_ranges"].items():
        lines.append(f"| {k} | {v['min']:.4f} | {v['max']:.4f} |")
    lines += ["", f"The maximum numerical error in **log T = log E + log D_U** is {conf['max_abs_logT_minus_logE_minus_logD']:.2e}. "
              "After centering within each student/seed/E pair, log T and log D_U are identical. "
              "The table gives correlations of these centered log exposures across all pairs; correlations near one "
              "measure a shared pool contrast, not independent identifying variation.", "",
              "| | log D_U | log T | log steps | log S |", "|---|---:|---:|---:|---:|"]
    for name, values in zip(conf["columns"], conf["within_pair_centered_log_correlation"]):
        lines.append(f"| {name} | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += ["", f"The four-column centered design has rank {conf['centered_standardized_rank']}; it is singular. "
              "A joint regression cannot disentangle D_U from T at fixed E, and steps/S add little independent variation.", "",
              "**Schedule check (full-run settings, identical across the three seeds in each row).** "
              "All runs use AdamW, learning rate 1e-4, cosine scheduling, warmup ratio 0.03 and effective batch size 16 sequences. "
              "Warmup counts are integer-truncated. Snapshot schedule fractions and logged next-update rates are retained in JSON.", "",
              "| Student | Pool / variant | Planned steps | Warmup steps | Final E | Final supervised S |",
              "|---|---|---:|---:|---:|---:|"]
    for r in summary["data"]["runs"]:
        if r["seed"] == 0:
            lines.append(f"| {r['student']} | {r['pool']} / {r['variant']} | {r['total_updates_planned']} | "
                         f"{r['warmup_steps']} | {r['final_E']:.1f} | {r['completion_tokens_seen']:,} |")
    lines += ["", "Gemma's matched-E U75 run has a 28-step horizon and zero warmup versus U600's 224 and six. "
              "Qwen shares the 224-step horizon across pools but reaches matched E at very different schedule fractions. "
              "Thus equal base learning rates do not match optimization histories. The logged rate follows scheduler.step(); "
              "it is not the rate applied to the preceding update or an integrated training dose.", "",
              "## Held-out interaction results", "",
              "Primary score: raw matched-level checkpoints, nats per target token. All model comparisons use the same folds and weights.", "",
              "| Student | Capability | E-only MAE | Interaction MAE | Additive MAE | Gain over E-only [95% CI] | Gain over additive [95% CI] |",
              "|---|---|---:|---:|---:|---|---|"]
    for s in STUDENTS:
        for c in CAPS:
            m = summary["held_out_interaction"]["metrics"][s][c]["matched_levels_primary"]
            lines.append(f"| {s} | {c} | {m['mae_E_only']['mean']:.4f} | {m['mae_interaction']['mean']:.4f} | "
                         f"{m['mae_additive_diagnostic']['mean']:.4f} | {_ci(m['gain_over_E_only'])} | {_ci(m['gain_over_additive'])} |")
    lines += ["", "Secondary, all-level panel (includes high-E extrapolation; not substituted for the primary endpoint):", "",
              "| Student | Capability | E-only MAE | Interaction MAE | Additive MAE | Gain over E-only [95% CI] |",
              "|---|---|---:|---:|---:|---|"]
    for s in STUDENTS:
        for c in CAPS:
            m = summary["held_out_interaction"]["metrics"][s][c]["all_levels_secondary"]
            lines.append(f"| {s} | {c} | {m['mae_E_only']['mean']:.4f} | {m['mae_interaction']['mean']:.4f} | "
                         f"{m['mae_additive_diagnostic']['mean']:.4f} | {_ci(m['gain_over_E_only'])} |")
    lines += ["", "Fold coefficients, ranks, condition numbers, per-level errors and each raw prediction are in the summary. "
              "A supported gain does not establish 0.05-nat predictive sufficiency or isolate optimizer exposure from diversity.", "",
              "## Matched-E residual versus exposure", "",
              "Define r_c = delta75(E) − delta600(E). The common E-only prediction cancels in this signed paired residual. "
              "Fit separately **r=beta*log(S75/S600)** and **r=beta*(steps75−steps600)/100**, each one slope with no intercept. "
              "These are differences of regressions on log cumulative supervised tokens and on optimizer steps. "
              "They predict zero gap at equal exposure. No simultaneous collinear exposure coefficients are fitted.", "",
              "For each student/capability, train on one matched E level (all three seeds) and predict the other; reverse for the second fold. "
              "A one-parameter constant gap is trained on the same folds because log(S75/S600) is almost constant. "
              "Full-data uncentered R² is descriptive; held-level SSE reduction can be negative. "
              "Call a residual **largely explained** only if held-level SSE falls by at least 50% versus zero gap and "
              "both corrected-gap paired intervals lie entirely within ±0.05 nat. This is a predictive compatibility rule, not causal attribution.", "",
              "Exact-E outcome interpolation itself uses observed target checkpoints, sometimes the other E level. "
              "Therefore this regression validation is a descriptive diagnostic; the raw nominal-gap sensitivity below "
              "avoids cross-level outcome interpolation but retains E mismatch. Neither is the primary raw-snapshot interaction CV.", "",
              "| Student | Cap | Exposure | Full-data slope | Descriptive R² (uncentered) | Held-level SSE reduction | MAE zero → exposure | MAE gain over constant [95% CI] | Largely explained? |",
              "|---|---|---|---:|---:|---:|---|---|---|"]
    for s in STUDENTS:
        for c in CAPS:
            for x in EXPOSURES:
                r = summary["residual_vs_exposure"][s][c][x]
                m = r["metrics"]
                lines.append(f"| {s} | {c} | {x} | {r['full_data_slope_descriptive']:+.4f} | "
                             f"{r['full_data_uncentered_R2_descriptive']:.3f} | {r['cross_level_SSE_fraction_explained']:+.3f} | "
                             f"{m['mae_zero_gap']['mean']:.4f} → {m['mae_exposure']['mean']:.4f} | "
                             f"{_ci(m['mae_gain_over_constant'])} | {'yes' if r['largely_explained'] else 'no'} |")
    lines += ["", "Signed gaps before and after held-level exposure correction (paired 95% intervals):", "",
              "| Student | Cap | Level | Raw nominal gap | Exact-E gap | Corrected: log S | Corrected: steps |",
              "|---|---|---|---|---|---|---|"]
    for s in STUDENTS:
        for c in CAPS:
            for level in LEVELS[:2]:
                p = [p for p in summary["matched_E_pairs"] if p["student"] == s and p["E_level"] == level]
                raw = paired_interval({q["seed"]: q["nominal_residual"][c] for q in p})
                aligned = paired_interval({q["seed"]: q["residual"][c] for q in p})
                corrected = [summary["residual_vs_exposure"][s][c][x]["corrected_residual_by_level"][level] for x in EXPOSURES]
                lines.append(f"| {s} | {c} | {level} | {_ci(raw)} | {_ci(aligned)} | {_ci(corrected[0])} | {_ci(corrected[1])} |")
    lines += ["", "Raw nominal-E sensitivity (uses nominal gaps and nominal exposures consistently):", "",
              "| Student | Cap | Exposure | Held-level SSE reduction | Largely explained? |",
              "|---|---|---|---:|---|"]
    for s in STUDENTS:
        for c in CAPS:
            for x in EXPOSURES:
                r = summary["nominal_E_exposure_sensitivity"][s][c][x]
                lines.append(f"| {s} | {c} | {x} | {r['cross_level_SSE_fraction_explained']:+.3f} | {'yes' if r['largely_explained'] else 'no'} |")
    lines += ["", "## Interpretation at the fixed tolerance", "",
              "| Student | Capability | Interaction gain supported / material? | Exposure models meeting rule |",
              "|---|---|---|---|"]
    for s, caps in summary["decisions"].items():
        for c, d in caps.items():
            lines.append(f"| {s} | {c} | {'yes' if d['interaction_gain_supported'] else 'no'} / "
                         f"{'yes' if d['interaction_gain_material'] else 'no'} | {', '.join(d['exposure_models_meeting_rule']) or 'none'} |")
    passed = [f"{s}/{c}" for s, caps in summary["decisions"].items() for c, d in caps.items() if d["exposure_models_meeting_rule"]]
    lines += ["", ("The measured-exposure rule is met for " + ", ".join(passed) + "." if passed else
                   "Neither one-slope exposure model explains any student/capability residual to the fixed 0.05-nat rule."),
              "The data establish the advisor's confound: matched E varies pool volume, processed tokens, optimizer steps and supervised "
              "tokens together by about eightfold. They do not identify which of those changes causes the residual. "
              "Predictive improvement is capability-specific, and a high descriptive R² alone is insufficient. "
              "There is **no evidence here that requires a new latent capability variable**; equally, exposure-only sufficiency "
              "must not be claimed where the fixed rule fails. Measured optimization history remains an unresolved explanation.", ""]
    for s in STUDENTS:
        for c in CAPS:
            r = summary["residual_vs_exposure"][s][c]["log_supervised_ratio"]
            if r["largely_explained"] and not r["supported_gain_over_constant"]:
                lines += [f"For {s}/{c}, log supervised exposure removes "
                          f"{100*r['cross_level_SSE_fraction_explained']:.1f}% of held-level squared gap, but its MAE gain "
                          f"over a constant gap is {_ci(r['metrics']['mae_gain_over_constant'])} nat. "
                          "This supports compatibility with an observed exposure difference, without evidence that exposure "
                          "explains more than a persistent pool gap. The almost constant exposure ratio cannot distinguish those accounts.", ""]
    lines += [f"- {item}" for item in summary["limitations"]]
    lines += ["", "A discriminating follow-up would cross pool size with controlled optimizer/supervised-token budgets and schedule "
              "horizons. Because T=E*D_U, E, T and D_U cannot all be independently held fixed; the experiment must explicitly vary "
              "the chosen exposure axis, for example through batch size or supervision density. No new training is launched here.", "",
              "Regenerate with NumPy and SciPy on CPU:", "", "```bash",
              "python analysis/v37b_reuse_interaction.py", "python -m pytest -q tests/test_v37b.py tests/test_v37.py", "```", "",
              "Artifacts: [summary.json](../../results/v37b-reuse-interaction/summary.json), "
              "[implementation](../../analysis/v37b_reuse_interaction.py), "
              "[tests](../../tests/test_v37b.py). Prior: [V37](REUSE_SUFFICIENCY.md).", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    summary = build_summary(args.root)
    output = args.output or args.root / "results/v37b-reuse-interaction/summary.json"
    report = args.report or args.root / "paper/docs/REUSE_INTERACTION.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report.write_text(render_report(summary))
    for student, caps in summary["decisions"].items():
        for cap, decision in caps.items():
            print(f"{student}/{cap}: {decision['exposure_explanation']}")
    print(f"Wrote {output}\nWrote {report}")


if __name__ == "__main__":
    main()
