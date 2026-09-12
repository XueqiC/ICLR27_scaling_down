#!/usr/bin/env python3
"""V89: retrospective development-set analysis of already unblinded trajectories.

CPU NumPy only; no model imports, training, inference, or new measurements.
Run: python -B analysis/v89_intervention_retro.py
Only results/v89-intervention-retro/{summary.md,summary.json} are written.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_REL = Path("results/v89-intervention-retro")
REGISTER = Path("results/v47-p2-register/register.json")
STUDENTS = ("gemma3-270m", "gemma3-1b", "gemma3-4b")
CAPABILITIES = ("math", "code", "qa")
TARGETS = ("I1", "I2")
SPLITS = ("leave_one_student_size_out", "leave_one_pool_seed_out_within_student")
CANDIDATES = ("zero", "constant", "T_only", "reuse_only", "two_dimensional",
              "saturation_p1", "saturation_p2")
T_REF = D_REF = 100_000.0
RESAMPLES, SEED, PROFILE_GRID_SIZE = 5000, 0, 161
LABEL = "Retrospective development-set analysis on already unblinded data"
RULE = (
    "A capability has signal only if the SAME fixed candidate beats BOTH zero and "
    "the per-capability constant for BOTH I1 and I2 on leave-one-student-size-out. "
    "For each baseline and target, improvement = baseline MAE - candidate MAE must "
    "be strictly greater than the full width (upper minus lower) of its paired "
    "95% trajectory-bootstrap improvement interval. Otherwise signal is absent. "
    "Target-specific verdicts are also reported. No candidate is selected or refit "
    "after comparison."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Inputs:
    def __init__(self, root):
        self.root = Path(root)
        self.digests = {}

    def read(self, path):
        raw = path.read_bytes()
        self.digests[str(path.relative_to(self.root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def verify(self):
        for name, expected in self.digests.items():
            require(sha256(self.root / name) == expected, f"Input changed during analysis: {name}")


@dataclass(frozen=True)
class Point:
    id: str
    trajectory: str
    student: str
    seed: int
    capability: str
    distribution: str
    update: int
    T: float
    D_U: float
    delta: float
    loss: float
    initial_loss: float
    source: str
    baseline_source: str

    @property
    def E(self):
        return self.T / self.D_U


@dataclass(frozen=True)
class Pair:
    target: str
    first: Point
    second: Point

    @property
    def actual(self):
        return self.second.delta - self.first.delta

    @property
    def clusters(self):
        return tuple(sorted({self.first.trajectory, self.second.trajectory}))

    @property
    def id(self):
        return f"{self.target}:{self.first.id}->{self.second.id}"


def registered_pools(register):
    """Read all register generations; repeated definitions must agree."""
    pools = {}

    def visit(value, path=""):
        if isinstance(value, dict):
            if {"U", "data_seed", "D_U_completion"} <= value.keys():
                key = (int(value["U"]), int(value["data_seed"]))
                require(value["D_U_completion"] > 0, f"Nonpositive registered D_U: {path}")
                if key in pools:
                    require(pools[key]["D_U_completion"] == value["D_U_completion"],
                            f"Conflicting registered pool: {key}")
                    pools[key]["register_locations"].append(path)
                else:
                    pools[key] = {"U": key[0], "data_seed": key[1],
                                  "D_U_completion": value["D_U_completion"],
                                  "register_locations": [path]}
            else:
                for key, item in value.items():
                    visit(item, f"{path}/{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}/{index}")

    visit(register)
    return pools


def completion_by_update(log):
    """Actual supervised tokens only. Validate every available accounting source."""
    result = {0: 0}

    def add(update, tokens):
        require(isinstance(update, int) and update >= 0, "Invalid logged update")
        require(isinstance(tokens, (int, float)) and math.isfinite(tokens)
                and tokens >= 0 and int(tokens) == tokens, "Invalid logged completion count")
        require(update not in result or result[update] == tokens,
                f"Conflicting completion token counts at update {update}")
        result[update] = int(tokens)

    cumulative, previous_step, complete_curve = 0, 0, True
    for row in log.get("loss_curve", []):
        step = row["step"]
        require(step > previous_step, "Non-increasing loss_curve steps")
        complete_curve = complete_curve and step == previous_step + 1
        previous_step = step
        if "completion_tokens" in row and complete_curve:
            cumulative += row["completion_tokens"]
            add(step, cumulative)
        else:
            complete_curve = False
        if "completion_tokens_seen" in row:
            add(step, row["completion_tokens_seen"])
    for row in log.get("trajectory", []):
        if "completion_tokens_seen" in row:
            add(row["updates"], row["completion_tokens_seen"])
    if "completion_tokens_seen" in log:
        add(log.get("updates", log.get("optimizer_steps")), log["completion_tokens_seen"])
    ordered = sorted(result.items())
    require(all(b[1] >= a[1] for a, b in zip(ordered, ordered[1:])),
            "Logged cumulative completion tokens decrease")
    return result


def evaluation_distribution(payload, capability):
    fields = ("probe_source", "probe_seed", "probe_half", "n_probe_requested", "loss_definition")
    require(all(k in payload for k in fields), "Missing evaluation distribution metadata")
    metadata = {key: payload[key] for key in fields}
    for key in ("measurement_benchmarks", "measurement_samples", "measurement_tokens"):
        require(capability in payload.get(key, {}), f"Missing {key}/{capability}")
        metadata[key] = payload[key][capability]
    require(metadata["measurement_tokens"] > 0, "Zero evaluation tokens")
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:12]
    return f"{metadata['measurement_benchmarks']}:{digest}", metadata


def load_data(root):
    inputs = Inputs(root)
    pools = registered_pools(inputs.read(inputs.root / REGISTER))
    points, audits, exclusions, distributions = [], [], [], {}
    for student in STUDENTS:
        directory = inputs.root / "results/v12-distill" / student
        require(directory.is_dir(), f"Missing student input directory: {directory}")
        for run in sorted(p for p in directory.iterdir() if p.is_dir()):
            name = str(run.relative_to(inputs.root / "results/v12-distill"))
            paths = sorted(set(run.glob("trajectory/update-*/eval.json"))
                           | {p for p in (run / "eval.json", run / "train_log.json") if p.exists()})
            documents, reasons = {}, []
            for path in paths:
                try:
                    documents[path] = inputs.read(path)
                except (ValueError, OSError) as error:
                    reasons.append(f"Unreadable {path.relative_to(run)}: {error}")
            final, log = documents.get(run / "eval.json"), documents.get(run / "train_log.json")
            if final is None:
                reasons.append("Missing or unreadable final eval.json")
            if log is None:
                reasons.append("Missing or unreadable train_log.json")
            audit = {"trajectory": name, "student": student,
                     "input_paths": [str(p.relative_to(inputs.root)) for p in paths]}
            run_points, run_exclusions = [], []
            if not reasons:
                try:
                    require(final.get("student") == student and log.get("student") == student,
                            "Student metadata disagree with directory")
                    u, seed = final.get("n_per_domain"), final.get("data_seed")
                    require((u, seed) in pools,
                            f"No pool entry in V47 register for U={u}, data_seed={seed}; "
                            "D_U cannot be assigned from the required source")
                    pool = pools[u, seed]
                    require(final.get("teacher") == "gpt-5.6-luna" and final.get("recipe") == "full",
                            "Teacher/recipe do not match the registered full-trace pool")
                    require(log.get("n_per_domain") == u, "Pool size disagrees with train log")
                    require(log.get("data_sampling_seed") == seed, "Data seed disagrees with train log")
                    times = completion_by_update(log)
                    zero = run / "trajectory/update-00000000/eval.json"
                    baseline = documents[zero] if zero in documents else final
                    baseline_losses = baseline.get("post_training") if zero in documents else final.get("dense")
                    require(isinstance(baseline_losses, dict), "Missing own initial or dense losses")
                    evals = sorted((p, d) for p, d in documents.items() if p.name == "eval.json")
                    # Snapshot at a duplicated final update takes precedence; record the alias.
                    evals.sort(key=lambda item: (item[1].get("updates", -1), item[0] == run / "eval.json"))
                    seen = {}
                    for path, payload in evals:
                        try:
                            update = payload.get("updates", log.get("updates") if path == run / "eval.json" else None)
                            require(update in times, f"No actual supervised token count in train_log for update {update}")
                            if path.parent.name.startswith("update-"):
                                require(update == int(path.parent.name.removeprefix("update-")),
                                        "Checkpoint directory and update metadata disagree")
                            require(payload.get("data_seed") == seed and payload.get("n_per_domain") == u
                                    and payload.get("student") == student, "Checkpoint identity mismatch")
                            if "completion_tokens_seen" in payload:
                                require(payload["completion_tokens_seen"] == times[update],
                                        "Evaluation completion count disagrees with actual train_log count")
                            losses = payload.get("post_training", payload.get("capability_losses", {}))
                            if update in seen:
                                require(losses == seen[update][1], "Conflicting evaluations of the same update")
                                run_exclusions.append({"trajectory": name, "path": str(path.relative_to(inputs.root)),
                                                       "reason": f"Duplicate update {update}; retained {seen[update][0]}"})
                                continue
                            seen[update] = (str(path.relative_to(inputs.root)), losses)
                            for cap in CAPABILITIES:
                                try:
                                    distribution, metadata = evaluation_distribution(payload, cap)
                                    base_distribution, _ = evaluation_distribution(baseline, cap)
                                    require(distribution == base_distribution, "Initial/checkpoint distribution mismatch")
                                    require(all(isinstance(v, (int, float)) and math.isfinite(v)
                                                for v in (losses.get(cap), baseline_losses.get(cap))),
                                            "Missing or nonfinite capability loss")
                                    distributions[distribution] = metadata
                                    point = Point(f"{name}@{update}:{cap}:{distribution}", name, student, seed, cap,
                                                  distribution, update, times[update], pool["D_U_completion"],
                                                  losses[cap] - baseline_losses[cap], losses[cap], baseline_losses[cap],
                                                  str(path.relative_to(inputs.root)),
                                                  str((zero if zero in documents else run / "eval.json").relative_to(inputs.root)))
                                    run_points.append(point)
                                except (ValueError, KeyError, TypeError) as error:
                                    run_exclusions.append({"trajectory": name, "path": str(path.relative_to(inputs.root)),
                                                           "capability": cap, "reason": str(error)})
                        except (ValueError, KeyError, TypeError) as error:
                            run_exclusions.append({"trajectory": name, "path": str(path.relative_to(inputs.root)),
                                                   "reason": str(error)})
                    require(any(p.T > 0 for p in run_points), "No usable positive-budget checkpoints")
                    audit.update({"pool": pool, "training_seed": final.get("training_seed", final.get("seed")),
                                  "schedule_tokens": final.get("schedule_tokens"),
                                  "baseline": "own update 0" if zero in documents else "own final eval dense",
                                  "positive_checkpoints": len({p.update for p in run_points if p.T > 0}),
                                  "T_range": [min(p.T for p in run_points), max(p.T for p in run_points)],
                                  "checkpoint_capability_rows": len(run_points)})
                except (ValueError, KeyError, TypeError) as error:
                    reasons.append(str(error))
            audit.update({"status": "dropped" if reasons else "included", "reasons": reasons})
            audits.append(audit)
            exclusions.extend(run_exclusions)
            if not reasons:
                points.extend(run_points)
    return points, audits, exclusions, distributions, inputs


def construct_pairs(points):
    """All positive-T matches; I1 ordered by T, I2 oriented increasing D_U.

    Shared budget means max(T1,T2)/min(T1,T2) <= 1.10. All matches are
    retained, with actual endpoint budgets (no nearest-only matching/interpolation).
    T=0 cannot have a defined relative budget match or doubling ratio.
    """
    groups = defaultdict(list)
    for point in points:
        if point.T > 0:
            groups[point.student, point.capability, point.distribution].append(point)
    pairs = []
    for group in groups.values():
        for first, second in combinations(sorted(group, key=lambda p: p.id), 2):
            if first.trajectory == second.trajectory:
                first, second = sorted((first, second), key=lambda p: p.T)
                if 1.7 <= second.T / first.T <= 2.3:
                    pairs.append(Pair("I1", first, second))
            elif first.D_U != second.D_U:
                if max(first.T, second.T) / min(first.T, second.T) <= 1.10:
                    first, second = sorted((first, second), key=lambda p: p.D_U)
                    pairs.append(Pair("I2", first, second))
    return sorted(pairs, key=lambda p: p.id)


def split_folds(points, split):
    if split == SPLITS[0]:
        for student in sorted({p.student for p in points}):
            yield student, [p for p in points if p.student != student], [p for p in points if p.student == student]
    elif split == SPLITS[1]:
        for student, seed in sorted({(p.student, p.seed) for p in points}):
            yield f"{student}/seed={seed}", [p for p in points if p.student == student and p.seed != seed], [
                p for p in points if p.student == student and p.seed == seed]
    else:
        raise ValueError(f"Unknown split {split}")


def features(points, candidate, tau=None):
    t = np.array([p.T for p in points], dtype=float)
    du = np.array([p.D_U for p in points], dtype=float)
    x, reuse = np.log1p(t / T_REF), np.log1p(t / du)
    if candidate == "zero":
        return np.zeros((len(points), 1))
    if candidate == "constant":
        return np.ones((len(points), 1))
    if candidate == "T_only":
        return x[:, None]
    if candidate == "reuse_only":
        return reuse[:, None]
    if candidate == "two_dimensional":
        return np.column_stack((np.ones(len(points)), x, np.log(du / D_REF)))
    if candidate in ("saturation_p1", "saturation_p2"):
        require(tau is not None and tau > 0, "Positive tau required")
        power = 1 if candidate == "saturation_p1" else 2
        return np.column_stack((np.expm1(-t / tau), reuse ** power))
    raise ValueError(f"Unregistered candidate: {candidate}")


@dataclass
class Fit:
    candidate: str
    coefficients: list
    tau: float | None = None
    profile: dict | None = None

    def predict(self, points):
        return features(points, self.candidate, self.tau) @ np.asarray(self.coefficients)


def predicted_intervention(fit, pairs):
    """ONE fitted delta function at both endpoints, never regress differences."""
    return fit.predict([p.second for p in pairs]) - fit.predict([p.first for p in pairs])


def cluster_draws(clusters, resamples=RESAMPLES, seed=SEED):
    """Multinomial multiplicities: sample n whole trajectories n times."""
    require(clusters > 0, "No trajectory clusters")
    return np.random.default_rng(seed).multinomial(clusters, np.full(clusters, 1 / clusters), size=resamples)


def profile_fit(points, candidate, resamples=RESAMPLES, seed=SEED):
    """Grid profile of tau, cluster-bootstrap calibrated profile-loss region.

    At each tau, refit unconstrained a,b to DELTA by least squares. On each
    trajectory resample, refit nuisance coefficients over the same tau grid.
    The 95th percentile of resampled profile excess loss at the original tau
    estimate calibrates the original profile confidence set. Its envelope is
    reported, along with disconnected components and boundary contact. No
    checkpoint-iid likelihood or residual degrees of freedom are assumed.
    """
    positive = [p.T for p in points if p.T > 0]
    require(positive, "No positive training budgets for tau")
    grid = np.geomspace(min(positive), max(positive), PROFILE_GRID_SIZE)
    y = np.array([p.delta for p in points])
    design = np.stack([features(points, candidate, tau) for tau in grid])
    coefficients = np.stack([np.linalg.lstsq(x, y, rcond=None)[0] for x in design])
    sse = np.square(np.einsum("gni,gi->gn", design, coefficients) - y).sum(axis=1)
    best = int(np.argmin(sse))
    clusters = sorted({p.trajectory for p in points})
    # Sufficient statistics aggregate entire trajectories before any resampling.
    gram, rhs, yy, sizes = [], [], [], []
    for trajectory in clusters:
        mask = np.array([p.trajectory == trajectory for p in points])
        x, target = design[:, mask], y[mask]
        gram.append(np.einsum("gni,gnj->gij", x, x))
        rhs.append(np.einsum("gni,n->gi", x, target))
        yy.append(float(target @ target))
        sizes.append(int(mask.sum()))
    gram, rhs, yy, sizes = map(np.asarray, (gram, rhs, yy, sizes))
    draws = cluster_draws(len(clusters), resamples, seed)
    excess = []
    for start in range(0, resamples, 250):
        weights = draws[start:start + 250]
        g = np.einsum("bk,kgij->bgij", weights, gram)
        r = np.einsum("bk,kgi->bgi", weights, rhs)
        beta = np.einsum("bgij,bgj->bgi", np.linalg.pinv(g, rcond=1e-12), r)
        loss = np.maximum(0, (weights @ yy)[:, None] - np.einsum("bgi,bgi->bg", r, beta))
        loss /= (weights @ sizes)[:, None]
        excess.extend(np.maximum(0, loss[:, best] - loss.min(axis=1)))
    cutoff = float(np.quantile(excess, 0.95))
    accepted = (sse - sse[best]) / len(points) <= cutoff + 1e-14
    indices = np.flatnonzero(accepted)
    lower, upper = float(grid[indices[0]]), float(grid[indices[-1]])
    boundary = bool(accepted[0] or accepted[-1])
    components = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    profile = {
        "tau": float(grid[best]), "ci95": [lower, upper], "clusters": len(clusters),
        "below_six_clusters": len(clusters) < 6,
        "observed_positive_T_range": [float(grid[0]), float(grid[-1])],
        "reaches_lower_boundary": bool(accepted[0]), "reaches_upper_boundary": bool(accepted[-1]),
        "reaches_range_boundary": boundary, "identified_inside_observed_T_range": not boundary,
        "status": "not identified" if boundary else "identified inside the observed T range (grid profile)",
        "profile_components": [[float(grid[c[0]]), float(grid[c[-1]])] for c in components],
        "profile_grid_size": PROFILE_GRID_SIZE, "bootstrap_resamples": resamples, "bootstrap_seed": seed,
        "bootstrap_profile_cutoff": cutoff, "tau_grid": grid.tolist(),
        "profile_excess_delta_squared_loss": ((sse - sse[best]) / len(points)).tolist(),
        "method": "95% trajectory-bootstrap calibrated profile-loss confidence set; grid envelope",
    }
    return Fit(candidate, coefficients[best].tolist(), float(grid[best]), profile)


def fit_candidate(points, candidate, resamples=RESAMPLES, seed=SEED):
    require(points, "Empty training split")
    require(len({(p.capability, p.distribution) for p in points}) == 1,
            "Fit must be per capability and evaluation distribution")
    if candidate.startswith("saturation_"):
        return profile_fit(points, candidate, resamples, seed)
    y = np.array([p.delta for p in points])
    coefficients = np.linalg.lstsq(features(points, candidate), y, rcond=None)[0]
    return Fit(candidate, coefficients.tolist())


def pair_weights(pairs, counts, clusters):
    """Reconstruct pairs from a sampled trajectory multiset.

    I1 weight is the trajectory multiplicity; I2 weight is the product of
    BOTH trajectories' multiplicities. Shared checkpoints are never drawn.
    """
    index = {name: i for i, name in enumerate(clusters)}
    weights = np.ones((len(counts), len(pairs)), dtype=np.int64)
    for column, pair in enumerate(pairs):
        for name in pair.clusters:
            weights[:, column] *= counts[:, index[name]]
    return weights


def bootstrap_statistics(pairs, errors, resamples=RESAMPLES, seed=SEED):
    """Paired MAE/bias bootstrap of fixed cross-fitted intervention predictions."""
    errors = np.asarray(errors, dtype=float)
    require(errors.ndim == 2 and errors.shape[0] == len(pairs) and len(pairs) > 0,
            "Errors must have one row per intervention pair")
    clusters = sorted({name for pair in pairs for name in pair.clusters})
    rng = np.random.default_rng(seed)
    absolute, signed, discarded = [], [], 0
    remaining = resamples
    while remaining:
        # Bound memory for the dense I2 graph; always retain exactly 5000 draws.
        batch = min(250, remaining)
        counts = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=batch)
        weights = pair_weights(pairs, counts, clusters)
        denominator = weights.sum(axis=1)
        keep = denominator > 0
        discarded += int((~keep).sum())
        weights, denominator = weights[keep], denominator[keep]
        absolute.append((weights @ np.abs(errors)) / denominator[:, None])
        signed.append((weights @ errors) / denominator[:, None])
        remaining -= int(keep.sum())
    return {"mae": np.mean(np.abs(errors), axis=0), "bias": np.mean(errors, axis=0),
            "mae_draws": np.concatenate(absolute), "bias_draws": np.concatenate(signed),
            "clusters": len(clusters), "cluster_ids": clusters,
            "empty_pair_resamples_redrawn": discarded}


def interval(estimate, samples, clusters):
    ci = np.quantile(samples, [0.025, 0.975]).tolist()
    return {"estimate": float(estimate), "ci95": ci, "width": float(ci[1] - ci[0]),
            "clusters": clusters, "below_six_clusters": clusters < 6}


def clears_reading_rule(comparisons):
    """Strict > full paired interval width, for BOTH baselines."""
    return all(name in comparisons and comparisons[name]["estimate"] > comparisons[name]["width"]
               for name in ("zero", "constant"))


def capability_verdict(rows):
    """Same candidate, both targets; seed split never rescues a size-split failure."""
    eligible = [r for r in rows if r["split"] == SPLITS[0] and r["student"] == "all"]
    by_candidate = defaultdict(dict)
    for row in eligible:
        by_candidate[row["candidate"]][row["target"]] = clears_reading_rule(row["improvement_over_baselines"])
    passing = [name for name in CANDIDATES if all(by_candidate[name].get(t, False) for t in TARGETS)]
    return {"signal": "present" if passing else "absent", "split": SPLITS[0],
            "passing_candidates": passing, "candidate_target_passes": dict(by_candidate), "rule": RULE}


def evaluate(points, pairs, resamples=RESAMPLES, seed=SEED, progress=print):
    predictions, fits, fold_audit = [], [], []
    for split in SPLITS:
        for fold, training, held_out in split_folds(points, split):
            progress(f"{LABEL}: fitting {split} / {fold}", flush=True)
            held_ids = {p.id for p in held_out}
            fold_pairs = [p for p in pairs if p.first.id in held_ids and p.second.id in held_ids]
            fold_audit.append({"split": split, "fold": fold,
                               "training_trajectories": sorted({p.trajectory for p in training}),
                               "held_out_trajectories": sorted({p.trajectory for p in held_out}),
                               "pairs_by_target": dict(Counter(p.target for p in fold_pairs))})
            for cap, distribution in sorted({(p.capability, p.distribution) for p in held_out}):
                train = [p for p in training if (p.capability, p.distribution) == (cap, distribution)]
                selected = [p for p in fold_pairs if (p.first.capability, p.first.distribution) == (cap, distribution)]
                # Fail rather than silently skip any candidate or fold.
                require(train, f"No matching training observations for {split}/{fold}/{cap}/{distribution}")
                models = {}
                for candidate in CANDIDATES:
                    fit = fit_candidate(train, candidate, resamples, seed)
                    models[candidate] = fit
                    fits.append({"split": split, "fold": fold, "capability": cap,
                                 "distribution": distribution, "training_points": len(train),
                                 "training_clusters": len({p.trajectory for p in train}), **asdict(fit)})
                endpoint1 = {name: model.predict([p.first for p in selected]) for name, model in models.items()}
                endpoint2 = {name: model.predict([p.second for p in selected]) for name, model in models.items()}
                for index, pair in enumerate(selected):
                    values = {name: float(endpoint2[name][index] - endpoint1[name][index]) for name in CANDIDATES}
                    require(all(math.isfinite(value) for value in values.values()), "Nonfinite intervention prediction")
                    predictions.append({"split": split, "fold": fold, "pair_id": pair.id,
                                        "student": pair.first.student, "capability": cap, "distribution": distribution,
                                        "target": pair.target, "actual": pair.actual,
                                        "predicted_delta_first": {k: float(v[index]) for k, v in endpoint1.items()},
                                        "predicted_delta_second": {k: float(v[index]) for k, v in endpoint2.items()},
                                        "predicted_intervention": values})
    # Descriptive all-data fits only for identifiability, never scored as held-out predictions.
    progress(f"{LABEL}: fitting descriptive full-data tau profiles", flush=True)
    for cap, distribution in sorted({(p.capability, p.distribution) for p in points}):
        train = [p for p in points if (p.capability, p.distribution) == (cap, distribution)]
        for candidate in ("saturation_p1", "saturation_p2"):
            fit = fit_candidate(train, candidate, resamples, seed)
            fits.append({"split": "descriptive_all_data", "fold": "all", "capability": cap,
                         "distribution": distribution, "training_points": len(train),
                         "training_clusters": len({p.trajectory for p in train}), **asdict(fit)})
    return predictions, fits, fold_audit


def score_predictions(predictions, pairs, resamples=RESAMPLES, seed=SEED, progress=print):
    by_id = {p.id: p for p in pairs}
    groups = defaultdict(list)
    for row in predictions:
        for student in ("all", row["student"]):
            groups[row["split"], student, row["capability"], row["distribution"], row["target"]].append(row)
    metrics, per_trajectory = [], []
    for (split, student, cap, distribution, target), rows in sorted(groups.items()):
        selected = [by_id[row["pair_id"]] for row in rows]
        errors = np.array([[row["predicted_intervention"][name] - row["actual"]
                            for name in CANDIDATES] for row in rows])
        boot = bootstrap_statistics(selected, errors, resamples, seed)
        context = {"split": split, "student": student, "capability": cap,
                   "distribution": distribution, "target": target, "pairs": len(rows)}
        for index, candidate in enumerate(CANDIDATES):
            comparisons = {}
            for baseline in ("zero", "constant"):
                b = CANDIDATES.index(baseline)
                comparisons[baseline] = interval(boot["mae"][b] - boot["mae"][index],
                                                  boot["mae_draws"][:, b] - boot["mae_draws"][:, index],
                                                  boot["clusters"])
            metrics.append({**context, "candidate": candidate, "clusters": boot["clusters"],
                            "mae": interval(boot["mae"][index], boot["mae_draws"][:, index], boot["clusters"]),
                            "signed_bias": interval(boot["bias"][index], boot["bias_draws"][:, index], boot["clusters"]),
                            "improvement_over_baselines": comparisons,
                            "clears_target_reading_rule": clears_reading_rule(comparisons),
                            "empty_pair_resamples_redrawn": boot["empty_pair_resamples_redrawn"]})
        if student != "all":
            for trajectory in boot["cluster_ids"]:
                mask = np.array([trajectory in pair.clusters for pair in selected])
                per_trajectory.append({**context, "trajectory": trajectory, "pairs": int(mask.sum()),
                                       "focal_clusters": 1,
                                       "partner_trajectories": sorted({c for p in np.array(selected, dtype=object)[mask]
                                                                       for c in p.clusters if c != trajectory}),
                                       "candidate_metrics": {
                                           name: {"mae": float(np.abs(errors[mask, i]).mean()),
                                                  "signed_bias": float(errors[mask, i].mean())}
                                           for i, name in enumerate(CANDIDATES)},
                                       "interval_note": "Descriptive incident-pair breakdown; one focal trajectory, no within-trajectory CI."})
    progress(f"{LABEL}: scored {len(metrics)} metric rows with {resamples} trajectory resamples", flush=True)
    return metrics, per_trajectory


def pair_counts(points, pairs, predictions):
    covered = {split: {r["pair_id"] for r in predictions if r["split"] == split} for split in SPLITS}
    rows = []
    for student in STUDENTS:
        for cap, distribution in sorted({(p.capability, p.distribution) for p in points if p.student == student}):
            for target in TARGETS:
                selected = [p for p in pairs if (p.first.student, p.first.capability, p.first.distribution, p.target)
                            == (student, cap, distribution, target)]
                for split in ("all_usable_pairs", *SPLITS):
                    subset = selected if split == "all_usable_pairs" else [p for p in selected if p.id in covered[split]]
                    rows.append({"student": student, "capability": cap, "distribution": distribution,
                                 "target": target, "split": split, "pairs": len(subset),
                                 "clusters": len({c for p in subset for c in p.clusters}),
                                 "not_evaluated_pairs": len(selected) - len(subset),
                                 "not_evaluated_reason": "I2 endpoints have different pool seeds; both cannot belong to one held-out seed fold"
                                 if len(selected) > len(subset) else None})
    return rows


def number(value):
    return f"{value:.6f}"


def ci_text(value):
    label = f"n={value['clusters']} trajectories"
    if value["clusters"] < 6:
        label += "; BELOW SIX"
    return f"{number(value['estimate'])} [{number(value['ci95'][0])}, {number(value['ci95'][1])}] ({label})"


def verdict_block(summary):
    lines = [f"{LABEL}. This is not an independent test.", ""]
    for verdict in summary["verdicts"]:
        cap, distribution = verdict["capability"], verdict["distribution"]
        rows = [r for r in summary["metrics"] if r["split"] == SPLITS[0] and r["student"] == "all"
                and r["capability"] == cap and r["distribution"] == distribution]
        lines.append(f"**{cap} ({distribution}): signal {verdict['signal']} on leave-one-student-size-out.**")
        for target in TARGETS:
            rr = [r for r in rows if r["target"] == target]
            baseline = next((r for r in rr if r["candidate"] == "zero"), None)
            if baseline is None:
                lines.append(f"{target}: no usable evaluation pairs; signal absent.")
                continue
            details = []
            for row in rr:
                if row["candidate"] not in ("zero", "constant"):
                    gain = row["improvement_over_baselines"]["zero"]
                    details.append(f"{row['candidate']}: MAE {number(row['mae']['estimate'])}, "
                                   f"gain {number(gain['estimate'])}, width {number(gain['width'])} "
                                   f"({'clears' if row['clears_target_reading_rule'] else 'fails'})")
            lines.append(f"{target}: zero = constant MAE {ci_text(baseline['mae'])}; "
                         f"{baseline['pairs']} pairs. " + "; ".join(details) + ".")
        lines.append("Same-candidate, both-target passes: " + (", ".join(verdict["passing_candidates"]) or "none") + ".")
        lines.append("")
    return "\n".join(lines).strip()


def markdown(summary):
    lines = [verdict_block(summary), "", "## Retrospective development-set tables", "", RULE, ""]
    lines.extend(summary["method_notes"])
    lines += ["", f"Inventory: {len(summary['runs'])} run directories; "
              f"{sum(r['status'] == 'included' for r in summary['runs'])} included; "
              f"{sum(r['status'] == 'dropped' for r in summary['runs'])} dropped. "
              "All previously named development, test, confirmation, repeat, and throughput runs are already unblinded here.",
              "", "### Fixed candidates fitted to delta", "",
              "| Candidate | Formula |", "|---|---|"]
    for name, formula in summary["candidate_formulas"].items():
        lines.append(f"| {name} | {formula} |")
    lines += ["", "### Usable intervention pairs", "", "Counts are per evaluation distribution and capability. "
              "Clusters are distinct trajectory directories contributing at least one pair.", "",
              "| Split | Student | Capability / distribution | Target | Pairs | Clusters | Unevaluable pairs |",
              "|---|---|---|---|---:|---:|---:|"]
    for row in summary["pair_counts"]:
        lines.append(f"| {row['split']} | {row['student']} | {row['capability']} / {row['distribution']} | "
                     f"{row['target']} | {row['pairs']} | {row['clusters']} | {row['not_evaluated_pairs']} |")
    lines += ["", "### Intervention MAE and signed bias", "",
              "All intervals below are 95% percentile intervals with 5,000 trajectory resamples, seed 0. "
              "Bias = predicted intervention minus measured intervention. All losses and errors are nats/token. "
              "A positive paired gain means improvement. Every interval states its trajectory count. "
              "Zero and constant comparisons are both included, even though they agree algebraically."]
    for split in SPLITS:
        for cap, distribution in sorted({(r["capability"], r["distribution"]) for r in summary["metrics"]}):
            lines += ["", f"#### {split}: {cap} / {distribution} — retrospective development-set", "",
                      "| Student | Target | Candidate | Pairs | MAE [95% CI] | Bias [95% CI] | "
                      "Gain over zero [95% CI] | Gain over constant [95% CI] | Clears target rule |",
                      "|---|---|---|---:|---|---|---|---|---|"]
            for row in summary["metrics"]:
                if (row["split"], row["capability"], row["distribution"]) != (split, cap, distribution):
                    continue
                comparisons = row["improvement_over_baselines"]
                lines.append(f"| {row['student']} | {row['target']} | {row['candidate']} | {row['pairs']} | "
                             f"{ci_text(row['mae'])} | {ci_text(row['signed_bias'])} | "
                             f"{ci_text(comparisons['zero'])} | {ci_text(comparisons['constant'])} | "
                             f"{'yes' if row['clears_target_reading_rule'] else 'no'} |")
    lines += ["", "### Per-trajectory breakdown", "",
              "Each cell is MAE / signed bias. For I2, a pair appears in both endpoint trajectories' descriptive "
              "breakdowns, but once in the overall metric. No checkpoint bootstrap or within-trajectory interval "
              "is reported: each row conditions on one focal trajectory (below six)."]
    for split in SPLITS:
        for cap, distribution, target in sorted({(r["capability"], r["distribution"], r["target"])
                                                for r in summary["per_trajectory"]}):
            lines += ["", f"#### {split}: {cap} / {distribution}, {target} — retrospective development-set", "",
                      "| Trajectory | Incident pairs | " + " | ".join(CANDIDATES) + " |",
                      "|---|---:|" + "---|" * len(CANDIDATES)]
            for row in summary["per_trajectory"]:
                if (row["split"], row["capability"], row["distribution"], row["target"]) == (split, cap, distribution, target):
                    cells = [f"{number(row['candidate_metrics'][name]['mae'])} / "
                             f"{number(row['candidate_metrics'][name]['signed_bias'])}" for name in CANDIDATES]
                    lines.append(f"| {row['trajectory']} | {row['pairs']} | " + " | ".join(cells) + " |")
    lines += ["", "### Included trajectory accounting", "",
              "| Trajectory | Registered U | Registered D_U | Data seed | Training seed | Checkpoints with T > 0 | Actual T range | Baseline |",
              "|---|---:|---:|---:|---:|---:|---|---|"]
    for row in summary["runs"]:
        if row["status"] == "included":
            pool = row["pool"]
            lines.append(f"| {row['trajectory']} | {pool['U']} | {pool['D_U_completion']} | {pool['data_seed']} | "
                         f"{row['training_seed']} | {row['positive_checkpoints']} | {row['T_range']} | {row['baseline']} |")
    lines += ["", "## Retrospective development-set identifiability", "", summary["identifiability_method"], "",
              "The observed range means the positive supervised-token range in that fit's training data; "
              "tau cannot be zero. A profile set touching either range boundary is reported as not identified. "
              "No statement about an asymptotic floor or an optimal budget is made. All-data fits below are descriptive; "
              "fold fits are the actual functions used for intervention predictions.", "",
              "| Split / fold | Capability / distribution | Candidate | Fitted tau | Profile 95% interval, clusters | "
              "Observed positive T range | Lower / upper boundary | Identifiability |", "|---|---|---|---:|---|---|---|---|"]
    profiles = sorted((r for r in summary["fits"] if r["profile"] is not None),
                      key=lambda r: (r["split"] != "descriptive_all_data", r["split"], r["fold"], r["capability"], r["candidate"]))
    for row in profiles:
        profile = row["profile"]
        label = f"n={profile['clusters']} trajectories" + ("; BELOW SIX" if profile["clusters"] < 6 else "")
        bounds = ", ".join(f"{v:.1f}" for v in profile["ci95"])
        observed = ", ".join(f"{v:.0f}" for v in profile["observed_positive_T_range"])
        lines.append(f"| {row['split']} / {row['fold']} | {row['capability']} / {row['distribution']} | {row['candidate']} | "
                     f"{row['tau']:.1f} | [{bounds}] ({label}) | [{observed}] | "
                     f"{profile['reaches_lower_boundary']} / {profile['reaches_upper_boundary']} | {profile['status']} |")
    lines += ["", "## Every dropped run and its reason", "", f"{LABEL}; exclusions use input availability and identity, never fit performance."]
    for row in summary["runs"]:
        if row["status"] == "dropped":
            lines.append(f"\n- `{row['trajectory']}`: " + "; ".join(row["reasons"]) + ".")
    lines += ["", "### Checkpoint exclusions and duplicate aliases", ""]
    if not summary["checkpoint_exclusions"]:
        lines.append("None.")
    for row in summary["checkpoint_exclusions"]:
        lines.append(f"- `{row['path']}` {row.get('capability', '')}: {row['reason']}.")
    lines += ["", "Full endpoint records, fold membership, fitted coefficients, grid profiles, distribution definitions, "
              "input paths and SHA-256 digests are in `summary.json`. " + LABEL + ".", ""]
    return "\n".join(lines)


def analyze(root=ROOT, resamples=RESAMPLES, seed=SEED, progress=print):
    points, audits, exclusions, distributions, inputs = load_data(root)
    pairs = construct_pairs(points)
    predictions, fits, folds = evaluate(points, pairs, resamples, seed, progress)
    metrics, per_trajectory = score_predictions(predictions, pairs, resamples, seed, progress)
    verdicts = []
    for cap, distribution in sorted({(p.capability, p.distribution) for p in points}):
        verdict = capability_verdict([r for r in metrics if (r["capability"], r["distribution"]) == (cap, distribution)])
        verdicts.append({"capability": cap, "distribution": distribution, **verdict})
    summary = {
        "schema_version": 1, "analysis": "v89-intervention-retro", "status": LABEL,
        "independent_test": False, "device": "cpu", "new_measurements": False,
        "reading_rule": RULE, "verdicts": verdicts,
        "candidate_formulas": {"zero": "0", "constant": "a (per capability and distribution)",
                               "T_only": "a*log(1+T/T_ref)", "reuse_only": "a*log(1+E)",
                               "two_dimensional": "a+b*log(1+T/T_ref)+q*log(D_U/D_ref)",
                               "saturation_p1": "-a*(1-exp(-T/tau))+b*log(1+E)",
                               "saturation_p2": "-a*(1-exp(-T/tau))+b*log(1+E)**2"},
        "references": {"T_ref": T_REF, "D_ref": D_REF, "units": "supervised completion tokens",
                       "choice": "fixed at 100000 before fitting; no outcome-based scaling"},
        "bootstrap": {"unit": "trajectory", "resamples": resamples, "seed": seed,
                      "interval": "paired 95% percentile", "refit_predictors_for_metric_bootstrap": False,
                      "I1_weight": "trajectory multiplicity", "I2_weight": "product of both endpoint trajectory multiplicities",
                      "empty_pair_draws": "redrawn to retain exactly 5000 nonempty trajectory resamples; counts recorded per metric",
                      "scope": "conditional on fitted cross-validation functions and the observed trajectories; not new-student inference"},
        "method_notes": [
            "This is a retrospective development-set analysis on already unblinded data, not an independent test. "
            "There are only three observed student sizes; trajectory intervals do not establish uncertainty over a population of sizes.",
            "Each fit is per capability and evaluation distribution. Ordinary least squares fits delta, giving each observed "
            "checkpoint equal weight, including update 0 when observed. No intervention difference is used for fitting, "
            "and no average delta-fit score is evaluated. All seven candidates are retained. Coefficients are unconstrained "
            "real numbers; tau alone is positive and profiled on the observed training T range. T_ref = D_ref = 100000.",
            "Delta = checkpoint loss minus that trajectory's own update-0 loss, or its own dense evaluation if update 0 is absent. "
            "T comes from actual cumulative supervised completion counts in train_log.json, checked against per-update counts "
            "where available. D_U is read from the matching U and data_seed in the V47 register; E = T/D_U. Planned counts are never substituted.",
            "I1 retains every ordered positive-budget checkpoint pair with T2/T1 in [1.7,2.3]. "
            "I2 retains every unordered trajectory/checkpoint pair of the same student and distribution with different D_U and "
            "max(T1,T2)/min(T1,T2) <= 1.10, oriented from smaller to larger D_U. Both predictions use the actual endpoint T "
            "(no interpolation). Zero-budget pairs have no defined relative T match and are excluded from both targets. "
            "All matches count; nearby final and milestone checkpoints can both participate.",
            "I2 includes changes in completion-token pool size caused by changing seeds even at the same nominal U. "
            "Different schedules, training seeds, and registered run generations are retained as requested; these retrospective "
            "contrasts can include their effects and the allowed budget mismatch, so they do not isolate a randomized causal effect of pool size.",
            "Leave-one-student-size-out fits to the other students and predicts both endpoints with the same fitted function. "
            "Leave-one-pool-seed-out fits only the same student's other data seeds; all runs and pool sizes with the held seed "
            "are withheld together. I2 is scored there only when both trajectories share the held-out data seed. Cross-seed "
            "I2 pairs are explicitly counted as unevaluable on that split. A separately cross-fitted endpoint from another "
            "seed fold is never substituted. The zero and per-capability constant therefore have identical intervention predictions.",
            "Reported overall MAE and bias weight each usable intervention pair equally, with per-student results alongside. "
            "Metric intervals resample whole contributing trajectory directories with replacement and retain all their checkpoints. "
            "I2 pairs are reconstructed using both endpoint multiplicities; they are never assigned to just one endpoint cluster. "
            "The same draws are shared across candidates and baseline differences. Any graph-empty resample is redrawn and counted.",
            "Each per-trajectory row describes all incident pairs; I2 appears in both endpoint rows but only once in overall metrics. "
            "Trajectories with no pair on a split/target have zero counts in trajectory_pair_coverage in JSON. "
            "Repeated data/training seeds and shared pools can leave dependence between trajectories beyond the requested bootstrap unit; "
            "95% intervals are conditional descriptive uncertainty, especially limited below six trajectories.",
        ],
        "identifiability_method": "For each fixed p=1 and p=2, profile tau on a fixed 161-point logarithmic grid "
            "from min positive T to max T in the fit's training set. At each tau refit a,b by least squares to delta. "
            "Calibrate a profile-loss confidence set with 5000 whole-trajectory bootstrap resamples, seed 0: within each "
            "resample refit a,b at every grid tau, and calculate excess mean squared loss at the original fitted tau over "
            "the resample optimum. The 95th percentile of this excess sets the cutoff for the observed profile. "
            "Report the envelope of all accepted grid points as the approximate 95% profile interval; disconnected components "
            "and the full grid are recorded in JSON. This diagnostic uses delta squared loss only for parameter profiling, "
            "not for candidate performance ranking; no checkpoint independence assumption is used. Boundary contact means tau is not identified.",
        "runs": audits, "checkpoint_exclusions": exclusions, "distributions": distributions,
        "pair_counts": pair_counts(points, pairs, predictions), "metrics": metrics,
        "per_trajectory": per_trajectory, "folds": folds, "fits": fits,
        "points": [{**asdict(p), "E": p.E} for p in points],
        "pairs": [{"id": p.id, "target": p.target, "first": p.first.id, "second": p.second.id,
                   "actual": p.actual, "trajectory_clusters": list(p.clusters),
                   "T_ratio": p.second.T / p.first.T,
                   "relative_T_gap": max(p.first.T, p.second.T) / min(p.first.T, p.second.T) - 1}
                  for p in pairs],
        "predictions": predictions,
        "input_sha256": inputs.digests,
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                        (Path(__file__), ROOT / "tests/test_v89_intervention_retro.py") if path.exists()},
        "software": {"numpy": np.__version__},
    }
    incident = {(r["split"], r["trajectory"], r["capability"], r["distribution"], r["target"]): r["pairs"]
                for r in per_trajectory}
    summary["trajectory_pair_coverage"] = [
        {"split": split, "trajectory": run, "capability": cap, "distribution": distribution,
         "target": target, "pairs": incident.get((split, run, cap, distribution, target), 0)}
        for run, cap, distribution in sorted({(p.trajectory, p.capability, p.distribution) for p in points})
        for split in SPLITS for target in TARGETS]
    inputs.verify()
    return summary


def write_outputs(summary, root=ROOT):
    """Fixed output boundary; reject directory or file symlinks before writing."""
    root = Path(root).resolve()
    destination = root / OUT_REL
    require(not any(p.is_symlink() for p in (root / "results", destination)), "Symlink output directory forbidden")
    paths = [destination / name for name in ("summary.json", "summary.md")]
    require(not any(p.is_symlink() for p in paths), "Symlink output file forbidden")
    destination.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    paths[0].write_text(payload)
    paths[1].write_text(markdown(summary))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)  # No measurement, device, candidate-selection, or destination options.
    summary = analyze()
    write_outputs(summary)
    print("\n" + verdict_block(summary))


if __name__ == "__main__":
    main()
