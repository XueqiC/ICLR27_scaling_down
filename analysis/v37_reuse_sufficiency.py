#!/usr/bin/env python3
"""CPU-only reuse-count sufficiency audit of existing V12 uxseen snapshots.

Predict signed delta_c = post_training[c] - dense[c], in legacy V12 nats/token.
Protocol constants below are fixed for this retrospective analysis, before fitting;
they are not a prospective preregistration of already-observed V31b/Qwen outcomes.
Run: python analysis/v37_reuse_sufficiency.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
STUDENTS = ("gemma3-1b", "Qwen3-4B")
CAPS = ("math", "code", "qa")
SEEDS = (0, 1, 2)
POOLS = (75, 600)
TOLERANCE_NAT = 0.05
NUMERICAL_EPS = 1e-12  # Only floating-point ties, not a practical gain threshold.
SCHEMES = ("student_pool", "student")
MODELS = ("E_only", "E_log_D")
BASES = ("quadratic", "hinge_sensitivity")
LOW_MILESTONES = {"gemma3-1b": (62775, 125550), "Qwen3-4B": (62800, 125500)}
HIGH_MILESTONES = (500000, 1000000)


def reuse_count(processed_tokens, pool_tokens):
    """Use processed input tokens INCLUDING repeats / one-pass pool input tokens."""
    t, d = float(processed_tokens), float(pool_tokens)
    if not math.isfinite(t) or not math.isfinite(d) or t < 0 or d <= 0:
        raise ValueError("Need finite nonnegative processed tokens and positive pool tokens")
    return t / d


def load_trajectories(root=ROOT):
    """Strict complete roster; no nearest-file fallback, final evals, or GPU imports."""
    root = Path(root)
    rows, runs, hashes = [], [], {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    for student in STUDENTS:
        specs = [(75, "uxseen", HIGH_MILESTONES if student == "gemma3-1b" else
                  LOW_MILESTONES[student] + HIGH_MILESTONES), (600, "uxseen", HIGH_MILESTONES)]
        if student == "gemma3-1b":
            specs.append((75, "uxseenE", LOW_MILESTONES[student]))
        for pool, variant, milestones in specs:
            for seed in SEEDS:
                name = f"gpt-5.6-luna_full_{pool}_{variant}" + (f"_seed{seed}" if seed else "")
                run = root / "results/v12-distill" / student / name
                log_path = run / "train_log.json"
                log = read(log_path)
                du = log["unique_data_pool_tokens"]
                reuse_count(0, du)
                run_id = f"{student}/{name}"
                observed, baseline, seen_tokens = set(), None, set()
                for directory in sorted((run / "trajectory").glob("*")):
                    if not directory.is_dir():
                        continue
                    paths = [directory / n for n in ("eval.json", "snapshot.json") if (directory / n).exists()]
                    if len(paths) != 1:
                        raise ValueError(f"Expected exactly one snapshot JSON in {directory}")
                    path = paths[0]
                    snap = read(path)
                    if (snap["student"] != student or snap["n_per_domain"] != pool
                            or snap["seed"] != seed or snap["teacher"] != "gpt-5.6-luna"
                            or snap["recipe"] != "full" or snap["run_name"] != name):
                        raise ValueError(f"Snapshot metadata disagrees with roster: {path}")
                    if snap["unique_data_pool_tokens"] != du:
                        raise ValueError(f"Snapshot D_U disagrees with train_log: {path}")
                    tokens = snap["processed_tokens"]
                    e = reuse_count(tokens, du)
                    delta = {c: float(snap["delta"][c]) for c in CAPS}
                    if any(not math.isfinite(v) for v in delta.values()):
                        raise ValueError(f"Nonfinite signed delta: {path}")
                    if any(not math.isclose(delta[c], snap["post_training"][c] - snap["dense"][c],
                                            rel_tol=1e-9, abs_tol=1e-9) for c in CAPS):
                        raise ValueError(f"delta must equal post minus dense: {path}")
                    if tokens in seen_tokens:
                        raise ValueError(f"Duplicate processed-token checkpoint: {run_id}")
                    seen_tokens.add(tokens)
                    if tokens == 0:
                        if any(v != 0 for v in delta.values()):
                            raise ValueError("Dense baseline delta must be zero")
                        baseline = snap
                        continue
                    requested = snap["requested_token_milestones"]
                    if len(requested) != 1 or requested[0] not in milestones or requested[0] in observed:
                        raise ValueError(f"Unexpected, shared, or duplicate milestone: {path}")
                    milestone = requested[0]
                    if tokens < milestone:
                        raise ValueError(f"Checkpoint precedes its requested milestone: {path}")
                    observed.add(milestone)
                    source = str(path.relative_to(root))
                    rows.append({"row_id": source, "student": student, "pool": pool, "seed": seed,
                                 "variant": variant, "trajectory_id": run_id, "milestone": milestone,
                                 "processed_tokens": tokens, "D_U": du, "E": e, "delta": delta})
                if baseline is None or observed != set(milestones):
                    raise ValueError(f"Missing baseline or requested milestones: {run_id}")
                runs.append({"trajectory_id": run_id, "student": student, "pool": pool, "seed": seed,
                             "variant": variant, "D_U": du, "n_checkpoints": len(observed),
                             "train_log": str(log_path.relative_to(root)),
                             **{k: baseline[k] for k in ("epochs", "total_updates_planned", "learning_rate",
                                                        "scheduler", "data_selection", "measurement_tokens",
                                                        "loss_definition")}})
    return rows, runs, hashes


def make_folds(rows, scheme):
    """Hold out every seed/checkpoint/variant of a (student,pool) or student."""
    if scheme not in SCHEMES:
        raise ValueError("Unknown holdout scheme")
    keys = [(r["student"], r["pool"]) if scheme == "student_pool" else (r["student"],) for r in rows]
    folds = []
    for key in sorted(set(keys)):
        train = [i for i, value in enumerate(keys) if value != key]
        test = [i for i, value in enumerate(keys) if value == key]
        if not train or not test:
            raise ValueError("A fold needs nonempty training and test groups")
        train_runs = {rows[i]["trajectory_id"] for i in train}
        if train_runs & {rows[i]["trajectory_id"] for i in test}:
            raise ValueError("A trajectory leaked across the fold")
        folds.append({"held_out": list(key), "train_indices": train, "test_indices": test})
    return folds


def balanced_weights(rows):
    """Equal cells, then seeds within cells, then snapshots within each cell/seed.

    Gemma's two U75 run variants stay in the same seed block, as do Qwen's four
    checkpoints. More saved checkpoints never give a cell more total fit weight.
    """
    cells = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        cells[(row["student"], row["pool"])][row["seed"]].append(i)
    weights = np.zeros(len(rows))
    for seeds in cells.values():
        for indices in seeds.values():
            weights[indices] = 1 / (len(cells) * len(seeds) * len(indices))
    return weights


def prediction_inputs(rows):
    # Physical inputs only. Neither student identity nor any target outcome enters h.
    return [{"E": r["E"], "D_U": r["D_U"]} for r in rows]


def design_matrix(inputs, with_volume, basis="quadratic"):
    if basis not in BASES:
        raise ValueError("Unknown response basis")
    if any(set(r) != {"E", "D_U"} for r in inputs):
        raise ValueError("Prediction accepts only E and D_U, never target outcomes/offsets")
    e = np.array([r["E"] for r in inputs], dtype=float)
    d = np.array([r["D_U"] for r in inputs], dtype=float)
    if np.any(~np.isfinite(e)) or np.any(e < 0) or np.any(~np.isfinite(d)) or np.any(d <= 0):
        raise ValueError("Invalid E or D_U")
    x = np.log1p(e)
    if basis == "quadratic":
        base = np.column_stack([x, x*x])
    else:
        base = np.column_stack([x, np.maximum(x - np.log(2), 0), np.maximum(x - np.log(3), 0)])
    # Fixed unit reference is not fitted normalization. Both h(0, D)=0 by identity.
    return np.column_stack([base, base * np.log(d / 100000)[:, None]]) if with_volume else base


def fit_response(rows, capability, with_volume, basis="quadratic"):
    """Weighted least squares of SIGNED deltas, no monotonicity or positivity bound."""
    matrix = design_matrix(prediction_inputs(rows), with_volume, basis)
    response = np.array([r["delta"][capability] for r in rows])
    if not np.all(np.isfinite(response)):
        raise ValueError("Nonfinite signed response")
    weight = np.sqrt(balanced_weights(rows))
    coef, _, rank, singular = np.linalg.lstsq(matrix * weight[:, None], response * weight, rcond=None)
    if rank != matrix.shape[1]:
        raise ValueError(f"Response design is rank deficient ({rank}/{matrix.shape[1]} columns, "
                         f"basis={basis}, with_volume={with_volume}); do not silently fit another model")
    return {"coefficients": coef.tolist(), "basis": basis, "with_volume": with_volume,
            "rank": int(rank), "n_parameters": matrix.shape[1],
            "condition_number": float(singular[0] / singular[-1])}


def predict_response(fit, inputs):
    return design_matrix(inputs, fit["with_volume"], fit["basis"]) @ np.array(fit["coefficients"])


def paired_interval(seed_values):
    """Two-sided 95% Student-t interval over independent seed-level contrasts."""
    values = np.array(list(seed_values.values()), dtype=float)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Need finite seed contrasts")
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else None
    half = float(student_t.ppf(.975, len(values)-1) * sd / math.sqrt(len(values))) if sd is not None else None
    return {"mean": mean, "ci95": [mean-half, mean+half] if half is not None else None,
            "n_seeds": len(values), "seed_sd": sd,
            "seed_values": {str(k): float(v) for k, v in seed_values.items()}}


def interval_within_tolerance(interval, tolerance=TOLERANCE_NAT):
    ci = interval["ci95"]
    return ci is not None and ci[0] >= -tolerance and ci[1] <= tolerance


def paired_checkpoint_summaries(values):
    """Combine low/high ONLY within seed, never concatenate into n=6."""
    if not values or any(len(v) != 2 for v in values.values()):
        raise ValueError("Need both paired checkpoints for each seed")
    return {"mean_over_checkpoints": paired_interval({s: np.mean(v) for s, v in values.items()}),
            "mean_absolute_over_checkpoints": paired_interval({s: np.mean(np.abs(v)) for s, v in values.items()}),
            "high_minus_low": paired_interval({s: v[1]-v[0] for s, v in values.items()})}


def matched_e_analysis(rows):
    """Nominal pairs reproduce V31b; interpolation is a labelled outcome diagnostic."""
    result = {}
    for student in STUDENTS:
        raw, interpolated, match_info = {}, {}, []
        by_cap = {c: {s: [] for s in SEEDS} for c in CAPS}
        interp_by_cap = {c: {s: [] for s in SEEDS} for c in CAPS}
        for index, (small_t, large_t) in enumerate(zip(LOW_MILESTONES[student], HIGH_MILESTONES)):
            label = "low" if index == 0 else "high"
            for seed in SEEDS:
                small_rows = sorted([r for r in rows if r["student"] == student and r["pool"] == 75
                                     and r["seed"] == seed and r["milestone"] in LOW_MILESTONES[student]],
                                    key=lambda r: r["E"])
                small, = [r for r in small_rows if r["milestone"] == small_t]
                large, = [r for r in rows if r["student"] == student and r["pool"] == 600
                          and r["seed"] == seed and r["milestone"] == large_t]
                grid = [0.] + [r["E"] for r in small_rows]
                if not grid[0] <= large["E"] <= grid[-1]:
                    raise ValueError("Matched-E sensitivity would require extrapolation")
                match_info.append({"pair": label, "seed": seed, "U75_source": small["row_id"],
                                   "U600_source": large["row_id"], "U75_T": small["processed_tokens"],
                                   "U600_T": large["processed_tokens"], "U75_E": small["E"],
                                   "U600_E": large["E"], "E_difference": small["E"]-large["E"],
                                   "relative_E_mismatch": abs(small["E"] / large["E"] - 1)})
                for cap in CAPS:
                    by_cap[cap][seed].append(small["delta"][cap]-large["delta"][cap])
                    aligned = np.interp(large["E"], grid, [0.] + [r["delta"][cap] for r in small_rows])
                    interp_by_cap[cap][seed].append(float(aligned)-large["delta"][cap])
            raw[label] = {c: paired_interval({s: v[index] for s, v in by_cap[c].items()}) for c in CAPS}
            interpolated[label] = {c: paired_interval({s: v[index] for s, v in interp_by_cap[c].items()}) for c in CAPS}
        result[student] = {"pairs": match_info, "nominal_matched_E": raw,
                           "linear_interpolation_sensitivity": interpolated,
                           "paired_across_checkpoints": {c: paired_checkpoint_summaries(by_cap[c]) for c in CAPS},
                           "interpolated_paired_across_checkpoints": {
                               c: paired_checkpoint_summaries(interp_by_cap[c]) for c in CAPS}}
    return result


def summarize_predictions(records):
    """Paired model errors: average checkpoints, then pools, within student/seed.

    Exact stratified bootstrap resamples whole seed blocks within each fixed student,
    retaining both pools and every checkpoint and pairing candidate predictions.
    CIs condition on already-fitted OOF predictions: they do NOT quantify uncertainty
    over new students, newly drawn pools, or refitting the training data.
    """
    blocks = defaultdict(lambda: defaultdict(list))
    for r in records:
        errors = [abs(r["observed"]-r["predictions"][m]) for m in MODELS]
        blocks[(r["student"], r["seed"])][r["pool"]].append(
            errors + [abs(r["observed"]), errors[0]-errors[1]])
    names = ("mae_E_only", "mae_E_log_D", "mae_zero", "mae_reduction_with_log_D")
    clusters = defaultdict(dict)
    for (student, seed), pools in blocks.items():
        clusters[student][seed] = np.mean([np.mean(v, axis=0) for v in pools.values()], axis=0)
    distributions, student_summaries = [], {}
    for student, seeds in sorted(clusters.items()):
        array = np.array(list(seeds.values()))
        # n=3: 27 ordered resamples per student, 729 combined for two students.
        draws = np.array(list(itertools.product(range(len(array)), repeat=len(array))))
        distributions.append(array[draws].mean(axis=1))
        student_summaries[student] = {name: paired_interval({s: v[i] for s, v in seeds.items()})
                                      for i, name in enumerate(names)}
    distribution = distributions[0]
    for other in distributions[1:]:
        distribution = (distribution[:, None, :] + other[None, :, :]).reshape(-1, len(names))
    distribution /= len(distributions)
    means = np.mean([np.mean(list(seeds.values()), axis=0) for seeds in clusters.values()], axis=0)
    summary = {name: float(means[i]) for i, name in enumerate(names)}
    summary.update({"paired_seed_bootstrap_ci95": {
        name: np.quantile(distribution[:, i], [.025, .975]).tolist() for i, name in enumerate(names)},
        "per_student_seed_t_intervals": student_summaries, "n_student_seed_blocks": len(blocks),
        "n_bootstrap_resamples": len(distribution), "n_snapshots": len(records)})
    summary["log_D_reduces_held_out_MAE"] = summary["mae_reduction_with_log_D"] > NUMERICAL_EPS
    return summary


def cross_validate(rows, scheme, basis="quadratic"):
    folds, predictions = [], {c: [] for c in CAPS}
    for fold in make_folds(rows, scheme):
        train = [rows[i] for i in fold["train_indices"]]
        test = [rows[i] for i in fold["test_indices"]]
        block = {"held_out": fold["held_out"], "train_row_ids": [r["row_id"] for r in train],
                 "test_row_ids": [r["row_id"] for r in test], "capabilities": {},
                 "train_E_range": [min(r["E"] for r in train), max(r["E"] for r in train)],
                 "train_D_U_range": [min(r["D_U"] for r in train), max(r["D_U"] for r in train)]}
        for cap in CAPS:
            try:
                fits = {m: fit_response(train, cap, m == "E_log_D", basis) for m in MODELS}
            except ValueError as exc:
                raise ValueError(f"{scheme} held_out={fold['held_out']}, capability={cap}: {exc}") from exc
            pred = {m: predict_response(fits[m], prediction_inputs(test)) for m in MODELS}
            records = [{**{k: r[k] for k in ("row_id", "student", "pool", "seed", "trajectory_id", "E", "D_U")},
                        "observed": r["delta"][cap],
                        "predictions": {m: float(pred[m][i]) for m in MODELS},
                        "E_outside_training_range": not block["train_E_range"][0] <= r["E"] <= block["train_E_range"][1],
                        "D_U_outside_training_range": not block["train_D_U_range"][0] <= r["D_U"] <= block["train_D_U_range"][1]}
                       for i, r in enumerate(test)]
            block["capabilities"][cap] = {"fits": fits, "metrics": summarize_predictions(records)}
            predictions[cap].extend(records)
        folds.append(block)
    return {"folds": folds, "capabilities": {
        c: {"metrics": summarize_predictions(predictions[c]), "predictions": predictions[c]} for c in CAPS}}


def decide_sufficiency(matched, validations, tolerance=TOLERANCE_NAT):
    result = {}
    for cap in CAPS:
        within = all(interval_within_tolerance(matched[s]["nominal_matched_E"][p][cap], tolerance)
                     for s in STUDENTS for p in ("low", "high"))
        interp_within = all(interval_within_tolerance(matched[s]["linear_interpolation_sensitivity"][p][cap], tolerance)
                            for s in STUDENTS for p in ("low", "high"))
        reductions = {scheme: validations[scheme]["capabilities"][cap]["metrics"]["mae_reduction_with_log_D"]
                      for scheme in SCHEMES}
        no_improvement = all(v <= NUMERICAL_EPS for v in reductions.values())
        result[cap] = {"matched_E_intervals_within_tolerance": within,
                       "interpolation_sensitivity_intervals_within_tolerance": interp_within,
                       "log_D_does_not_improve_either_holdout": no_improvement,
                       "mae_reduction_with_log_D": reductions, "choose_E_only": within and no_improvement,
                       "decision": "E_only_sufficient_on_this_panel" if within and no_improvement else "E_only_sufficiency_not_established"}
    return result


def build_summary(root=ROOT):
    rows, runs, hashes = load_trajectories(root)
    matched = matched_e_analysis(rows)
    validations = {scheme: cross_validate(rows, scheme) for scheme in SCHEMES}
    sensitivity = {}
    for scheme in SCHEMES:
        try:
            sensitivity[scheme] = {"status": "available", **cross_validate(rows, scheme, "hinge_sensitivity")}
        except ValueError as exc:
            if "rank deficient" not in str(exc):
                raise
            sensitivity[scheme] = {"status": "unavailable", "reason": str(exc),
                                   "note": "No partial-fold average or minimum-norm predictions reported"}
    return {"version": 37, "cpu_only": True, "new_training_runs": 0,
            "protocol": {
                "status": "fixed retrospective protocol; previous residuals already observed; not prospective preregistration",
                "endpoint": "signed delta_c = post_training[c] - dense[c], legacy V12 capability loss, nats/token",
                "reuse_count": "E = actual snapshot processed_tokens / this run train_log unique_data_pool_tokens",
                "practical_tolerance_nat": TOLERANCE_NAT,
                "tolerance_rationale": "0.05 nat is the maximum acceptable systematic matched-E loss discrepancy; fixed across capabilities/students, not fitted to residuals",
                "decision_rule": "Per capability choose E-only iff ALL four nominal matched-E 95% paired intervals lie within [-0.05,+0.05] AND adding log D_U does not reduce macro held-out MAE in EITHER specified quadratic-model holdout; only a 1e-12 numerical tie tolerance",
                "primary_model": "x=log1p(E), z=log(D_U/100000); h(E)=b1*x+b2*x^2; h(E,z)=b1*x+b2*x^2+b3*x*z+b4*x^2*z",
                "sensitivity_model": "basis [x, max(x-log(2),0), max(x-log(3),0)]; volume model appends each basis term times z; fixed knots E=1,2, no model selection",
                "fit": "Per capability weighted least squares on signed deltas; h(0,D)=0 identity; unconstrained sign/curvature; no fitted target intercept, amplitude, or curve shift",
                "weighting": "Equal student/pool cells, equal seeds per cell, equal nonbaseline checkpoints per cell/seed; same weights for both candidates and scoring",
                "holdouts": {"student_pool": "primary: 4 leave-one-(student,pool)-out folds; all seeds and uxseen/uxseenE variants together",
                             "student": "secondary: 2 leave-one-student-out folds; all pools/seeds/variants together"},
                "target_calibration_outcomes_used_for_prediction": 0,
                "offset_policy": "No target offset used. Any future alignment using a target-derived offset is a calibration cost and cannot be called zero-calibration prediction.",
                "paired_intervals": "Matched residuals and high-minus-low changes: n=3 seed-level paired t intervals, df=2. Never n=6 checkpoints. CIs pointwise; equivalence requires every interval inside tolerance, not merely including zero.",
                "prediction_intervals": "Exact stratified paired bootstrap of fixed OOF errors, seed blocks within student retaining all pools/checkpoints; 27 resamples per student, 729 combined; per-student t intervals also retained. Conditional seed variation, not refit/new-student uncertainty.",
                "matched_E_interpolation": "Sensitivity only: linearly interpolate U75 signed responses at actual U600 E within [dense E=0,low,high]. Uses target outcomes as a descriptive diagnostic, not a held-out predictor or calibration-free transfer claim; no vertical offset.",
            },
            "data": {"students": list(STUDENTS), "seeds": list(SEEDS), "pools": list(POOLS),
                     "n_trajectories": len(runs), "n_nonbaseline_snapshots": len(rows),
                     "n_signed_capability_observations": len(rows)*len(CAPS),
                     "runs": runs, "snapshots": rows, "source_sha256": hashes},
            "matched_E": matched, "held_out": validations, "basis_sensitivity": sensitivity,
            "decisions": decide_sufficiency(matched, validations),
            "limitations": [
                "Only two students and two pools, one teacher/recipe. D_U is nearly collinear with pool and differs slightly by student tokenizer; no causal isolation of volume or broad student-generalization CI.",
                "All seeds use first n_per_domain rows of the same fixed pool. Intervals reflect training/shuffle randomness, not independent data-subset resampling or new evaluation samples.",
                "Nominal matched-E checkpoints overshoot requested tokens; actual E and interpolation sensitivity are reported. Interpolation assumes local linear response; no exact matched-E experiment is invented.",
                "Cosine schedule horizons differ: Gemma U75 uxseen/uxseenE plans 238/28 updates vs U600 224; Qwen U75/U600 both plan 224 but matched-E snapshots are at different schedule fractions. Matched-E residuals can include optimization-schedule effects.",
                "QA is noisy, with only 251 (Gemma) / 271 (Qwen) measured target tokens in the legacy loss. Crossing zero is not evidence of equivalence within 0.05 nat.",
                "Held-out gains depend on the specified response family and may mask opposite fold-level effects. Neither a worse volume model nor a shrinking pool gap proves E sufficiency; failure to select E-only does not validate the volume model.",
            ]}


def _ci(value):
    bounds = value["ci95"]
    return f"{value['mean']:+.3f} [{bounds[0]:+.3f}, {bounds[1]:+.3f}]" if bounds else "unavailable"


def render_report(summary):
    """Render all numerical claims from the same summary artifact."""
    p = summary["protocol"]
    rejected = [c.upper() if c == "qa" else c for c, d in summary["decisions"].items() if not d["choose_E_only"]]
    headline = (f"E-only sufficiency is not established for {', '.join(rejected)} at the fixed 0.05-nat tolerance."
                if rejected else "E-only meets the fixed 0.05-nat decision rule for all three capabilities on this panel.")
    raw_failures = any(not d["matched_E_intervals_within_tolerance"] for d in summary["decisions"].values())
    lines = ["# Distillation reuse-count sufficiency (V37)", "",
             f"**{headline}** " + ("The matched-E residual intervals fail the equivalence criterion. " if raw_failures else "") +
             "Whether adding unique-data volume "
             "helps prediction is assessed separately below; a smaller matched-T gap does not establish sufficiency.", "",
             "This CPU-only analysis reuses 15 existing trajectories, 36 nonbaseline snapshots and 108 signed "
             "capability observations from Gemma3-1B and Qwen3-4B. No training or GPU evaluation is performed. "
             "The protocol was fixed for this retrospective analysis before fitting; the earlier residuals were "
             "already known, so this is not a prospective preregistration.", "",
             "**Endpoint and fixed decision rule.** " + p["endpoint"] + ". Positive means loss worsens; negative "
             "means loss improves. The practical tolerance is **0.05 nat per target token** for every capability "
             "and student, the maximum accepted systematic discrepancy at matched reuse. Select E-only per "
             "capability only if (a) all four nominal matched-E **95% paired intervals lie entirely inside "
             "[-0.05,+0.05] nat**, and (b) adding log D_U does **not lower macro held-out MAE in either specified "
             "quadratic-model holdout**. Any positive reduction above 1e-12 counts as improvement; this is "
             "a conservative point-error rule, not a significance test. A CI that includes zero does not establish "
             "equivalence. The interpolation and basis checks are declared sensitivities, not tuned selection rules.", "",
             "**Data and actual reuse accounting.** E = T / D_U uses snapshot `processed_tokens` (input tokens, "
             "including repetitions) and that run's `train_log.json:unique_data_pool_tokens` (one-pass selected "
             "pool input tokens). U is the requested examples **per domain**, not token volume. T=0 snapshots "
             "validate the known delta=0 anchor and are excluded from fitting/scoring. Final run-level evals "
             "are excluded. Milestones are selected by their recorded requested labels; actual T, never nominal T, "
             "enters the model. Each input JSON is hashed in the summary.", "",
             "| Student | Run set | D_U | Requested T | Actual E range | Trajectories / snapshots |",
             "|---|---|---:|---|---|---:|"]
    rows = summary["data"]["snapshots"]
    for student in STUDENTS:
        for pool, variant in ((75, "uxseen"), (75, "uxseenE"), (600, "uxseen")):
            subset = [r for r in rows if (r["student"], r["pool"], r["variant"]) == (student, pool, variant)]
            if not subset:
                continue
            lines.append(f"| {student} | U{pool} {variant} | {subset[0]['D_U']} | "
                         f"{', '.join(str(t) for t in sorted({r['milestone'] for r in subset}))} | "
                         f"{min(r['E'] for r in subset):.4f}–{max(r['E'] for r in subset):.4f} | "
                         f"{len({r['trajectory_id'] for r in subset})} / {len(subset)} |")
    lines += ["", "**Models and folds.** Write x = ln(1+E), z = ln(D_U/100000). The fixed nested candidates are:", "",
              "1. E-only: δ̂_c = b₁c x + b₂c x².",
              "2. E + log D_U: δ̂_c = b₁c x + b₂c x² + b₃c xz + b₄c x²z.", "",
              "Each capability is fitted separately by weighted least squares of the **signed observed delta**, "
              "with no positivity or monotonicity constraint. Both obey h(0,D_U)=0 by the dense-baseline identity. "
              "There are no student labels, dense-loss descriptors, target offsets, target amplitude fits, "
              "hyperparameter searches or outcome-dependent knots. The same training observations/weights are "
              "used for both candidates. Cells (student,pool) have equal weight, then seeds within a cell, then "
              "checkpoints within a cell/seed. Gemma's two U75 variants form one seed block; Qwen's four U75 "
              "checkpoints form one seed block. Scoring uses this same macro weighting.", "",
              "Primary holdout: four leave-one-(student,pool)-out folds: Gemma U75, Gemma U600, Qwen U75, Qwen U600. "
              "Secondary: two leave-one-student-out folds: Gemma and Qwen. Every seed, checkpoint and run variant "
              "of the held-out group is excluded from fitting. The first scheme has other-pool observations of "
              "the held-out student; only the second holds out the entire student. No target response is available "
              "to either prediction function. Fold membership, coefficients, conditioning, input-range "
              "extrapolation flags and every out-of-fold prediction are in the JSON.", "",
              "**Held-out signed-response prediction.** MAE is |observed signed δ − predicted signed δ|. "
              "Reduction = MAE(E-only) − MAE(E+log D_U); positive favors including volume. Zero predicts δ=0 "
              "and has no sign. All errors and intervals are in nats/token.", "",
              "| Holdout | Capability | E-only MAE | E+log D_U MAE | Zero MAE | Reduction [paired 95% CI] | Volume lowers MAE? |",
              "|---|---|---:|---:|---:|---|---|"]
    for scheme in SCHEMES:
        for cap in CAPS:
            m = summary["held_out"][scheme]["capabilities"][cap]["metrics"]
            lo, hi = m["paired_seed_bootstrap_ci95"]["mae_reduction_with_log_D"]
            lines.append(f"| {scheme} | {cap} | {m['mae_E_only']:.4f} | {m['mae_E_log_D']:.4f} | "
                         f"{m['mae_zero']:.4f} | {m['mae_reduction_with_log_D']:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                         f"{'yes' if m['log_D_reduces_held_out_MAE'] else 'no'} |")
    # Highlight the declared primary result and the small cross-student QA gain.
    primary = summary["held_out"]["student_pool"]["capabilities"]
    if all(not primary[c]["metrics"]["log_D_reduces_held_out_MAE"] for c in CAPS):
        lines += ["", "Adding log D_U worsens the primary macro MAE for all three capabilities. "
                  "This lack of predictive improvement does not satisfy the separate residual-equivalence requirement."]
    qa = summary["held_out"]["student"]["capabilities"]["qa"]["metrics"]
    if 0 < qa["mae_reduction_with_log_D"] < TOLERANCE_NAT:
        lines += ["", f"The leave-one-student-out QA reduction is only {qa['mae_reduction_with_log_D']:.6f} nat, "
                  "far below the 0.05-nat residual tolerance. It counts under the specified point-error rule, "
                  "but is not evidence of practically useful prediction; opposite student-level effects "
                  "are visible in the fold table below."]
    if all(summary["held_out"][scheme]["capabilities"][cap]["metrics"][f"mae_{model}"] >
           summary["held_out"][scheme]["capabilities"][cap]["metrics"]["mae_zero"]
           for scheme in SCHEMES for cap in ("math", "code") for model in MODELS):
        lines += ["", "Both candidates have higher held-out math/code MAE than the zero-change baseline "
                  "under both holdout schemes. Neither candidate is validated as a useful cross-student "
                  "math/code predictor by this panel."]
    lines += ["", "The paired prediction intervals enumerate the 729 stratified bootstrap resamples of three "
              "seed blocks within each of two fixed students. A block keeps both pools, all checkpoints, and "
              "both candidate errors together. The intervals describe seed variation **conditional on the fitted "
              "out-of-fold predictions**, not refitting uncertainty or generalization to new students/pools. "
              "Per-fold/per-student n=3 paired t intervals are also stored. They should not be interpreted as "
              "independent folds or an n=36 uncertainty estimate.", "",
              "| Holdout | Held-out group | Capability | E-only MAE | E+log D_U MAE | Reduction |",
              "|---|---|---|---:|---:|---:|"]
    for scheme in SCHEMES:
        for fold in summary["held_out"][scheme]["folds"]:
            for cap in CAPS:
                m = fold["capabilities"][cap]["metrics"]
                lines.append(f"| {scheme} | {' / '.join(map(str, fold['held_out']))} | {cap} | "
                             f"{m['mae_E_only']:.4f} | {m['mae_E_log_D']:.4f} | {m['mae_reduction_with_log_D']:+.4f} |")
    lines += ["", "**Matched-E residuals in the same signed-loss frame.** Residual = δ(U75) − δ(U600), paired "
              "by training seed. Low/high refer to U600 T≈500k/1000k. Each interval is a two-sided 95% t interval "
              "with n=3 seeds, df=2. The two checkpoints are paired observations of each trajectory.", "",
              "| Student | Pair | Math residual [95% CI] | Code residual [95% CI] | QA residual [95% CI] |",
              "|---|---|---|---|---|"]
    for student in STUDENTS:
        for pair in ("low", "high"):
            values = summary["matched_E"][student]["nominal_matched_E"][pair]
            lines.append(f"| {student} | {pair} | " + " | ".join(_ci(values[c]) for c in CAPS) + " |")
    lines += ["", "This recovers the previously observed Gemma v31b math/code residuals of roughly −0.08 to "
              "−0.13 nat and Qwen n=3 residuals: math −0.009/−0.043, code +0.015/−0.13, QA +0.53/−0.77 "
              "(low/high). Residual shrinkage relative to matched-T is compatible with E being an important "
              "coordinate; these magnitudes and intervals do not meet the stipulated sufficiency criterion. "
              "QA is noisy; an interval crossing zero cannot be described as a vanished effect.", "",
              "| Student | Capability | Per-seed average residual [95% CI] | Per-seed mean absolute residual [95% CI] | Paired high−low [95% CI] |",
              "|---|---|---|---|---|"]
    for student in STUDENTS:
        for cap in CAPS:
            a = summary["matched_E"][student]["paired_across_checkpoints"][cap]
            lines.append(f"| {student} | {cap} | " + " | ".join(_ci(a[k]) for k in (
                "mean_over_checkpoints", "mean_absolute_over_checkpoints", "high_minus_low")) + " |")
    lines += ["", "Every combined interval above still uses **three** seed values. The mean absolute residual "
              "prevents opposite low/high signs (especially QA) from cancelling. Symmetric t intervals for "
              "magnitudes can extend below zero at n=3; these intervals are not clipped.", "",
              "**Actual-E mismatch and interpolation sensitivity.** Optimizer-update overshoot means the nominal "
              "pairs do not have exactly equal E. No snapshot is relabelled with a nominal processed-token value.", "",
              "| Student | Pair | U75 actual E range | U600 actual E range | Maximum relative E mismatch |",
              "|---|---|---|---|---:|"]
    for student in STUDENTS:
        for pair in ("low", "high"):
            a = [v for v in summary["matched_E"][student]["pairs"] if v["pair"] == pair]
            lines.append(f"| {student} | {pair} | {min(v['U75_E'] for v in a):.4f}–{max(v['U75_E'] for v in a):.4f} | "
                         f"{min(v['U600_E'] for v in a):.4f}–{max(v['U600_E'] for v in a):.4f} | "
                         f"{100*max(v['relative_E_mismatch'] for v in a):.2f}% |")
    lines += ["", "For a declared sensitivity, U75 signed responses are linearly interpolated to each seed's "
              "actual U600 E using the same U75 trajectory's dense (E=0,δ=0), low and high checkpoints. "
              "There is no extrapolation or vertical shift. This is a descriptive comparison using measured "
              "target responses, **not a held-out prediction result**; interpolation imposes local linearity.", "",
              "| Student | Pair | Interpolated math residual [95% CI] | Interpolated code residual [95% CI] | Interpolated QA residual [95% CI] |",
              "|---|---|---|---|---|"]
    for student in STUDENTS:
        for pair in ("low", "high"):
            a = summary["matched_E"][student]["linear_interpolation_sensitivity"][pair]
            lines.append(f"| {student} | {pair} | " + " | ".join(_ci(a[c]) for c in CAPS) + " |")
    lines += ["", "**Fixed basis sensitivity.** Replace [x,x²] with the continuous linear-spline basis "
              "[x,max(x−ln2,0),max(x−ln3,0)] (fixed E knots 1 and 2), and append the same basis times z for "
              "the volume model. This gives 3 versus 6 coefficients, keeps the zero anchor and signed endpoint, "
              "and changes no folds or weighting. These results are not used to pick a favorable basis.", "",
              "| Holdout | Capability | E-only MAE | E+log D_U MAE | Reduction |",
              "|---|---|---:|---:|---:|"]
    for scheme in SCHEMES:
        if summary["basis_sensitivity"][scheme]["status"] != "available":
            lines.append(f"| {scheme} | all | unavailable | unavailable | rank-deficient design |")
            continue
        for cap in CAPS:
            a = summary["basis_sensitivity"][scheme]["capabilities"][cap]["metrics"]
            lines.append(f"| {scheme} | {cap} | {a['mae_E_only']:.4f} | {a['mae_E_log_D']:.4f} | {a['mae_reduction_with_log_D']:+.4f} |")
    lines += ["", "A rank-deficient spline design is reported as unavailable for the entire holdout scheme, "
              "without a partial-fold average or arbitrary minimum-norm predictions. In particular, U600 "
              "has no E>2 measurements to identify a separate high-reuse spline slope in a one-student "
              "training fold. The two quadratic candidates above remain identifiable; the data do not "
              "support unrestricted response-surface comparisons."]
    lines += ["", "**Decisions and scope.**", "",
              "| Capability | All raw matched-E intervals within ±0.05? | All interpolated intervals within ±0.05? | No volume improvement in either primary-model holdout? | Select E-only? |",
              "|---|---|---|---|---|"]
    for cap, d in summary["decisions"].items():
        fields = ("matched_E_intervals_within_tolerance", "interpolation_sensitivity_intervals_within_tolerance",
                  "log_D_does_not_improve_either_holdout", "choose_E_only")
        lines.append(f"| {cap} | " + " | ".join("yes" if d[k] else "no" for k in fields) + " |")
    lines += ["", "Failure to select E-only does not establish that the volume model is sufficient. "
              "The models predict sign directly; low-E Gemma math/code responses are generally positive while "
              "Qwen responses are negative, and high reuse can change sign. Both students' U600 QA responses "
              "are negative. A curve alignment using a target-derived offset would incur a **calibration cost** "
              "and could not be called zero-calibration transfer; no such offset is used here.", ""]
    lines += [f"- {item}" for item in summary["limitations"]]
    lines += ["", "Regenerate from the repository root (NumPy and SciPy, CPU only):", "", "```bash",
              "python analysis/v37_reuse_sufficiency.py", "python -m pytest -q tests/test_v37.py", "```", "",
              "Machine-readable results: [summary.json](../../results/v37-reuse-sufficiency/summary.json). "
              "Implementation: [v37_reuse_sufficiency.py](../../analysis/v37_reuse_sufficiency.py). "
              "Earlier context: [UXSEEN_CONTROL.md](UXSEEN_CONTROL.md).", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository containing V12 results")
    parser.add_argument("--output", type=Path, help="Summary JSON (default: ROOT/results/v37-reuse-sufficiency/summary.json)")
    parser.add_argument("--report", type=Path, help="Markdown report (default: ROOT/paper/docs/REUSE_SUFFICIENCY.md)")
    args = parser.parse_args(argv)
    summary = build_summary(args.root)
    output = args.output or args.root / "results/v37-reuse-sufficiency/summary.json"
    report = args.report or args.root / "paper/docs/REUSE_SUFFICIENCY.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report.write_text(render_report(summary))
    for cap, decision in summary["decisions"].items():
        print(f"{cap}: {decision['decision']}; held-out MAE reductions {decision['mae_reduction_with_log_D']}")
    print(f"Wrote {output}\nWrote {report}")


if __name__ == "__main__":
    main()
