#!/usr/bin/env python3
"""V95 controlled matrix: CPU/NumPy analysis of saved losses, never model execution.

Run with python3 -B analysis/v95_matrix_intervention.py. The only output files are
results/v95-matrix-intervention/{summary.md,summary.json}. All seven registered
forms fit checkpoint delta by OLS, including each trajectory's update-0 anchor.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations, product
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_REL = Path("results/v95-matrix-intervention")
STUDENTS = ("gemma3-270m", "gemma3-1b", "gemma3-4b")
POOLS, SEEDS = (66, 198, 594), (41, 42)
CAPABILITIES = ("math", "code", "qa")
TARGETS = ("I1", "I2")
SPLITS = ("leave_one_student_out", "leave_one_pool_seed_out")
T_REF = D_REF = 100_000.0
HALVING_WINDOW, MATCHED_BUDGET_RATIO = (1.7, 2.3), 1.10
BOOTSTRAPS, BOOTSTRAP_SEED = 5000, 0
FORMULAS = {
    "zero": "0",
    "constant": "a (per capability)",
    "T_only": "a*log(1+T/T_ref)",
    "reuse_only": "a*log(1+T/D_U)",
    "two_dimensional": "a+b*log(1+T/T_ref)+q*log(D_U/D_ref)",
    "student_log_parameters": "a+(a0+a1*z_S)*log(1+T/T_ref)+q*log(D_U/D_ref); z_S=standardized log N",
    "student_initial_loss": "a+(a0+a1*z_S)*log(1+T/T_ref)+q*log(D_U/D_ref); z_S=standardized own initial capability loss",
}
CANDIDATES = tuple(FORMULAS)
BASELINE_NOTE = (
    "A constant fitted to delta and then differenced is identically zero, so zero "
    "and the constant are ONE baseline for the intervention targets."
)
RULE = (
    "Predictable if and only if the SAME registered candidate has baseline MAE minus "
    "candidate MAE strictly greater than the FULL width (upper minus lower) of its "
    "paired 95% trajectory-bootstrap MAE-improvement interval on BOTH I1 and I2 "
    "under pooled leave-one-student-out. Equality fails. Pool-seed results cannot "
    "rescue a failure. No candidate search or refit after comparison."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Inputs:
    def __init__(self, root):
        self.root, self.digests = Path(root), {}

    def read(self, path):
        raw = path.read_bytes()
        self.digests[str(path.relative_to(self.root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def verify(self):
        for name, digest in self.digests.items():
            require(hashlib.sha256((self.root / name).read_bytes()).hexdigest() == digest,
                    f"Input changed during analysis: {name}")


@dataclass(frozen=True)
class Point:
    trajectory: str
    student: str
    U: int
    seed: int
    capability: str
    distribution: str
    update: int
    processed_tokens: int
    T: int
    D_U: int
    loss: float
    initial_loss: float
    non_embedding_parameters: int
    source: str = ""
    baseline_source: str = ""

    @property
    def delta(self):
        return self.loss - self.initial_loss

    @property
    def E(self):
        return self.T / self.D_U

    @property
    def id(self):
        return f"{self.trajectory}@{self.update}:{self.capability}:{self.distribution}"


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


def token_accounting(log):
    """Reconcile all updates; recover D_U from one verified complete pool pass.

    unique_data_pool_tokens counts prompt+target and MUST NOT be used as D_U.
    A complete first epoch with no duplicate pool rows gives D_U exactly from
    its logged supervised completion increments, without tokenization/inference.
    """
    curve = log["loss_curve"]
    require(curve and log["status"] == "trained" and log["stopped_after_trajectory"],
            "Incomplete matrix trajectory")
    require(not log["trajectory_tokens_unreached"], "Unreached checkpoint")
    require(log["training_examples"] == log["unique_data_pool_examples"],
            "Duplicate pool examples prevent first-epoch D_U reconstruction")
    counts, processed, completion, epoch = {0: (0, 0)}, 0, 0, 1
    for step, row in enumerate(curve, 1):
        require(row["step"] == step, "Missing/noncontiguous loss_curve update")
        require(row["epoch"] in (epoch, epoch + 1), "Noncontiguous epoch")
        epoch = row["epoch"]
        for key in ("tokens", "completion_tokens"):
            require(type(row[key]) is int and row[key] > 0, f"Invalid {key}")
        processed += row["tokens"]
        completion += row["completion_tokens"]
        require(completion <= processed, "Completion count exceeds processed count")
        require(all(row[k] == processed for k in ("processed_tokens", "seen_tokens", "tokens_seen")),
                f"Processed accounting mismatch at update {step}")
        require(row["completion_tokens_seen"] == completion,
                f"Completion accounting mismatch at update {step}")
        require(row["unique_data_pool_tokens"] == log["unique_data_pool_tokens"], "Pool changed")
        counts[step] = (processed, completion)
    require(log["updates"] == log["optimizer_steps"] == len(curve), "Final update mismatch")
    require(all(log[k] == processed for k in ("processed_tokens", "seen_tokens", "tokens_seen")),
            "Final processed accounting mismatch")
    require(log["completion_tokens_seen"] == completion, "Final completion accounting mismatch")
    first = [r for r in curve if r["epoch"] == 1]
    require(first and first[-1]["unique_examples_seen"] == log["unique_data_pool_examples"],
            "First epoch does not cover the entire pool")
    require(sum(r["tokens"] for r in first) == first[-1]["unique_data_tokens"]
            == log["unique_data_pool_tokens"], "First epoch is not one complete distinct pool pass")
    du = sum(r["completion_tokens"] for r in first)
    require(du > 0, "Nonpositive D_U")
    for e in range(1, epoch):
        rows = [r for r in curve if r["epoch"] == e]
        require(sum(r["completion_tokens"] for r in rows) == du
                and sum(r["tokens"] for r in rows) == log["unique_data_pool_tokens"],
                "Completed epochs disagree on pool token counts")
    snapshots = {r["updates"]: r for r in log["trajectory"]}
    require(len(snapshots) == len(log["trajectory"]) == 4, "Expected four distinct logged checkpoints")
    for step, row in snapshots.items():
        require((row["processed_tokens"], row["completion_tokens_seen"]) == counts[step],
                "Trajectory log disagrees with loss_curve")
    return counts, du, first[-1]["step"]


def load_data(root=ROOT):
    inputs = Inputs(root)
    metadata = inputs.read(inputs.root / "configs/v28_source_metadata.json")
    points, audits, distributions, protocols, parameter_counts = [], [], {}, {}, {}
    pool_signatures = {}
    for student, u, seed in product(STUDENTS, POOLS, SEEDS):
        run = inputs.root / "results/v12-distill" / student / f"gpt-5.6-luna_full_{u}_matrix2_lora_dseed{seed}"
        name = str(run.relative_to(inputs.root / "results/v12-distill"))
        log_path = run / "train_log.json"
        log = inputs.read(log_path)
        counts, du, first_epoch_end = token_accounting(log)
        require((log["student"], log["n_per_domain"], log["data_sampling_seed"])
                == (student, u, seed), f"Run identity mismatch: {name}")
        expected = {"teacher": "gpt-5.6-luna", "recipe": "full", "training_mode": "lora",
                    "learning_rate": 1e-4, "optimizer": "AdamW", "scheduler": "cosine",
                    "warmup_ratio": 0.03, "training_seed": 0}
        require(all(log[k] == v for k, v in expected.items()), f"Protocol mismatch: {name}")
        protocol = {k: log[k] for k in (*expected, "effective_batch_size_sequences", "max_len", "dtype", "lora")}
        require(not protocols or protocol == next(iter(protocols.values())), "Different training protocols")
        protocols[name] = protocol
        # Logged totals include LoRA. Remove adapters and the tied token embedding;
        # retain 1-D language weights, unlike the older metadata's matrix-only N0.
        md = metadata["models"][student]
        require(md["hf_id"] == log["resolved_student"]
                and md["excluded_matrices"] == ["model.embed_tokens.weight"], "Unexpected model/embedding convention")
        embedding = md["hidden_size"] * md["vocab_size"]
        n = log["total_parameters"] - log["trainable_parameters"] - embedding
        require(md["N0"] <= n < 1.01 * md["N0"], "Non-embedding count inconsistent with language matrices")
        require(student not in parameter_counts or parameter_counts[student] == n, "Student count changed")
        parameter_counts[student] = n
        paths = sorted(run.glob("trajectory/update-*/eval.json"))
        require(len(paths) == 5, f"Expected update-0 plus four checkpoints: {name}")
        documents = [(p, inputs.read(p)) for p in paths]
        baseline_path, baseline = documents[0]
        require(baseline["updates"] == 0 and baseline_path.parent.name == "update-00000000", "Missing own update-0")
        require({d["updates"] for _, d in documents} == {0, *(r["updates"] for r in log["trajectory"])},
                "Snapshot updates disagree with trajectory log")
        signature = (du, log["unique_data_pool_tokens"], baseline["data_pool_sha256"])
        require((u, seed) not in pool_signatures or pool_signatures[u, seed] == signature,
                "Same pool differs across students")
        pool_signatures[u, seed] = signature
        previous = (-1, -1)
        for path, payload in documents:
            step = payload["updates"]
            require(path.parent.name == f"update-{step:08d}", "Snapshot path/update mismatch")
            require(all(payload[k] == v for k, v in expected.items()), "Snapshot protocol mismatch")
            require((payload["student"], payload["n_per_domain"], payload["data_seed"])
                    == (student, u, seed), "Snapshot identity mismatch")
            require(payload["schedule_tokens"] == 678000 and payload["stop_after_trajectory"], "Schedule mismatch")
            require(payload["data_pool_sha256"] == baseline["data_pool_sha256"]
                    and payload["unique_data_pool_tokens"] == log["unique_data_pool_tokens"], "Snapshot pool changed")
            actual = (payload["processed_tokens"], payload["completion_tokens_seen"])
            require(actual == counts[step] and payload["seen_tokens"] == actual[0], "Eval/log token mismatch")
            require(actual[0] > previous[0] and actual[1] > previous[1], "Nonincreasing checkpoint counts")
            previous = actual
            for cap in CAPABILITIES:
                distribution = {k: payload[k] for k in ("probe_source", "probe_seed", "probe_half", "n_probe_requested", "loss_definition")}
                distribution.update({k: payload[k][cap] for k in ("measurement_benchmarks", "measurement_samples", "measurement_tokens")})
                require(distribution["measurement_tokens"] > 0, "Empty evaluation")
                require(cap not in distributions or distributions[cap] == distribution, "Evaluation distribution changed")
                distributions[cap] = distribution
                digest = hashlib.sha256(json.dumps(distribution, sort_keys=True).encode()).hexdigest()[:12]
                loss, initial = payload["post_training"][cap], baseline["post_training"][cap]
                require(math.isfinite(loss) and math.isfinite(initial), "Nonfinite loss")
                point = Point(name, student, u, seed, cap, digest, step, *actual, du, loss, initial, n,
                              str(path.relative_to(inputs.root)), str(baseline_path.relative_to(inputs.root)))
                require(math.isclose(payload["delta"][cap], point.delta, abs_tol=1e-10), "Saved delta disagrees with own update-0")
                points.append(point)
        require(documents[-1][1]["updates"] == log["updates"], "Final logged checkpoint missing")
        audits.append({"trajectory": name, "student": student, "U": u, "seed": seed,
                       "D_U": du, "pool_processed_tokens": log["unique_data_pool_tokens"],
                       "pool_examples": log["unique_data_pool_examples"], "data_pool_sha256": baseline["data_pool_sha256"],
                       "D_U_source": str(log_path.relative_to(inputs.root)), "first_complete_epoch_end_update": first_epoch_end,
                       "final_processed_tokens": previous[0], "final_completion_tokens": previous[1],
                       "actual_total_updates": log["total_updates_planned"], "actual_warmup_updates": log["warmup_steps"],
                       "non_embedding_parameters": n, "embedding_parameters": embedding,
                       "total_parameters_including_adapters": log["total_parameters"], "adapter_parameters": log["trainable_parameters"],
                       "baseline_source": str(baseline_path.relative_to(inputs.root))})
    return points, audits, distributions, next(iter(protocols.values())), parameter_counts, inputs


def construct_pairs(points):
    """All eligible observed endpoints; no nominal milestones or interpolation.

    I1: inclusive 1.7 <= T_high/T_low <= 2.3, same trajectory.
    I2: inclusive max(T)/min(T) <= 1.10, same student, seed, capability and
    distribution; orient from smaller to larger pool. T=0 is never paired.
    """
    groups = defaultdict(list)
    for point in points:
        if point.T > 0:
            groups[point.student, point.seed, point.capability, point.distribution].append(point)
    pairs = []
    for group in groups.values():
        for first, second in combinations(group, 2):
            if first.trajectory == second.trajectory:
                first, second = sorted((first, second), key=lambda p: p.T)
                if HALVING_WINDOW[0] <= second.T / first.T <= HALVING_WINDOW[1]:
                    pairs.append(Pair("I1", first, second))
            elif first.U != second.U:
                first, second = sorted((first, second), key=lambda p: p.U)
                require(second.D_U > first.D_U, "Larger trace pool must contain more completion tokens")
                if max(first.T, second.T) / min(first.T, second.T) <= MATCHED_BUDGET_RATIO:
                    pairs.append(Pair("I2", first, second))
    return sorted(pairs, key=lambda p: p.id)


def descriptor_values(points, candidate):
    require(candidate in CANDIDATES[-2:], "Unknown student descriptor")
    return np.array([math.log(p.non_embedding_parameters) if candidate == "student_log_parameters"
                     else p.initial_loss for p in points])


@dataclass(frozen=True)
class Standardizer:
    mean: float
    scale: float
    training_std: float
    training_point_ids: tuple

    @classmethod
    def fit(cls, training, candidate):
        require(training, "Empty descriptor training fold")
        values = descriptor_values(training, candidate)
        constant = bool(np.all(values == values[0]))
        mean = float(values[0] if constant else values.mean())
        std = 0.0 if constant else float(values.std(ddof=0))
        return cls(mean, std if std > 0 else 1.0, std, tuple(p.id for p in training))

    def transform(self, points, candidate):
        return (descriptor_values(points, candidate) - self.mean) / self.scale


def features(points, candidate, standardizer=None):
    require(candidate in CANDIDATES, f"Unregistered candidate: {candidate}")
    t, du = np.array([p.T for p in points]), np.array([p.D_U for p in points])
    one, x = np.ones(len(points)), np.log1p(t / T_REF)
    if candidate == "zero":
        return np.zeros((len(points), 0))
    if candidate == "constant":
        return one[:, None]
    if candidate == "T_only":
        return x[:, None]
    if candidate == "reuse_only":
        return np.log1p(t / du)[:, None]
    base = np.column_stack((one, x, np.log(du / D_REF)))
    if candidate == "two_dimensional":
        return base
    require(standardizer is not None, "Student descriptor must be fit on training data")
    return np.column_stack((base, x * standardizer.transform(points, candidate)))


@dataclass
class Fit:
    candidate: str
    coefficients: list
    standardizer: Standardizer | None = None
    rank: int = 0

    def predict(self, points):
        return features(points, self.candidate, self.standardizer) @ np.asarray(self.coefficients)


def fit_candidate(training, candidate):
    require(training and len({(p.capability, p.distribution) for p in training}) == 1,
            "Fit must use checkpoint delta from one capability/distribution")
    standardizer = Standardizer.fit(training, candidate) if candidate in CANDIDATES[-2:] else None
    design = features(training, candidate, standardizer)
    coef, _, rank, _ = np.linalg.lstsq(design, [p.delta for p in training], rcond=None)
    return Fit(candidate, coef.tolist(), standardizer, int(rank))


def predicted_intervention(model, pairs):
    return model.predict([p.second for p in pairs]) - model.predict([p.first for p in pairs])


def split_folds(points, split):
    require(split in SPLITS, "Unknown split")
    key = (lambda p: p.student) if split == SPLITS[0] else (lambda p: p.seed)
    for value in sorted({key(p) for p in points}):
        yield str(value), [p for p in points if key(p) != value], [p for p in points if key(p) == value]


def evaluate(points, pairs):
    predictions, fits, folds = [], [], []
    for split in SPLITS:
        for fold, training, held in split_folds(points, split):
            held_ids = {p.id for p in held}
            selected = [p for p in pairs if p.first.id in held_ids and p.second.id in held_ids]
            folds.append({"split": split, "fold": fold,
                          "training_trajectories": sorted({p.trajectory for p in training}),
                          "held_out_trajectories": sorted({p.trajectory for p in held}),
                          "pairs_by_target": dict(Counter(p.target for p in selected))})
            for cap in sorted({p.capability for p in held}):
                train = [p for p in training if p.capability == cap]
                cap_pairs = [p for p in selected if p.first.capability == cap]
                require(cap_pairs, f"No evaluable pairs: {split}/{fold}/{cap}")
                endpoint1, endpoint2 = {}, {}
                for candidate in CANDIDATES:
                    model = fit_candidate(train, candidate)
                    fits.append({"split": split, "fold": fold, "capability": cap,
                                 "training_points": len(train), **asdict(model)})
                    endpoint1[candidate] = model.predict([p.first for p in cap_pairs])
                    endpoint2[candidate] = model.predict([p.second for p in cap_pairs])
                for i, pair in enumerate(cap_pairs):
                    first = {k: float(v[i]) for k, v in endpoint1.items()}
                    second = {k: float(v[i]) for k, v in endpoint2.items()}
                    effects = {k: second[k] - first[k] for k in CANDIDATES}
                    require(all(math.isfinite(v) for v in effects.values()), "Nonfinite prediction")
                    require(effects["zero"] == effects["constant"] == 0, "Baseline intercept failed to cancel")
                    predictions.append({"split": split, "fold": fold, "capability": cap, "target": pair.target,
                                        "pair_id": pair.id, "actual": pair.actual,
                                        "predicted_delta_first": first, "predicted_delta_second": second,
                                        "predicted_intervention": effects})
    require(len(predictions) == len(SPLITS) * len(pairs), "Not every pair has exactly one held-out prediction per split")
    return predictions, fits, folds


def pair_weights(pairs, counts, clusters):
    """Whole-trajectory multiset: I1 multiplicity once, I2 both endpoints' product."""
    index = {name: i for i, name in enumerate(clusters)}
    weights = np.ones((len(counts), len(pairs)), dtype=np.int64)
    for j, pair in enumerate(pairs):
        for cluster in pair.clusters:
            weights[:, j] *= counts[:, index[cluster]]
    return weights


def bootstrap_statistics(pairs, errors, resamples=BOOTSTRAPS, seed=BOOTSTRAP_SEED):
    """Paired bootstrap of fixed cross-fitted errors, never checkpoint sampling."""
    errors = np.asarray(errors, dtype=float)
    require(pairs and errors.ndim == 2 and len(errors) == len(pairs) and np.isfinite(errors).all(),
            "Expected finite errors with one row per pair")
    require(type(resamples) is int and resamples > 0, "Invalid bootstrap count")
    clusters = sorted({c for p in pairs for c in p.clusters})
    rng = np.random.default_rng(seed)
    maes, biases, rejected, remaining = [], [], 0, resamples
    while remaining:
        counts = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=min(250, remaining))
        weights = pair_weights(pairs, counts, clusters)
        denominator = weights.sum(axis=1)
        keep = denominator > 0
        rejected += int((~keep).sum())
        weights, denominator = weights[keep], denominator[keep]
        maes.append(weights @ np.abs(errors) / denominator[:, None])
        biases.append(weights @ errors / denominator[:, None])
        remaining -= int(keep.sum())
    return {"mae": np.abs(errors).mean(axis=0), "bias": errors.mean(axis=0),
            "mae_draws": np.concatenate(maes), "bias_draws": np.concatenate(biases),
            "cluster_ids": clusters, "clusters": len(clusters), "empty_pair_resamples_redrawn": rejected}


def interval(estimate, samples, clusters):
    ci = np.quantile(samples, [.025, .975]).tolist()
    return {"estimate": float(estimate), "ci95": ci, "width": ci[1] - ci[0],
            "clusters": clusters, "below_six_clusters": clusters < 6,
            "cluster_note": "Below six trajectory clusters." if clusters < 6 else "At least six trajectory clusters."}


def clears_reading_rule(improvement):
    return improvement["estimate"] > improvement["width"]


def capability_verdict(rows):
    passes = defaultdict(dict)
    for row in rows:
        if row["split"] == SPLITS[0] and row["scope"] == "pooled":
            passes[row["candidate"]][row["target"]] = clears_reading_rule(row["mae_improvement_over_baseline"])
    passing = [c for c in CANDIDATES[2:] if all(passes[c].get(t, False) for t in TARGETS)]
    return {"verdict": "PREDICTABLE" if passing else "NOT PREDICTABLE",
            "passing_candidates": passing, "candidate_target_passes": dict(passes)}


def score_predictions(predictions, pairs, resamples=BOOTSTRAPS):
    by_id, groups, metrics = {p.id: p for p in pairs}, defaultdict(list), []
    for row in predictions:
        for scope in ("pooled", f"held_out={row['fold']}"):
            groups[row["split"], scope, row["capability"], row["target"]].append(row)
    for (split, scope, cap, target), rows in sorted(groups.items()):
        selected = [by_id[r["pair_id"]] for r in rows]
        errors = np.array([[r["predicted_intervention"][c] - r["actual"] for c in CANDIDATES] for r in rows])
        boot = bootstrap_statistics(selected, errors, resamples)
        for j, candidate in enumerate(CANDIDATES):
            # The constant was fitted and differenced above. Its effects are
            # exactly zero; reuse the one baseline's draws to avoid BLAS
            # roundoff creating a fictitious second baseline interval.
            if candidate == "constant":
                require(np.array_equal(errors[:, j], errors[:, 0]), "Baseline effects differ")
                j = 0
            gain = interval(boot["mae"][0] - boot["mae"][j], boot["mae_draws"][:, 0] - boot["mae_draws"][:, j], boot["clusters"])
            metrics.append({"split": split, "scope": scope, "capability": cap, "target": target, "candidate": candidate,
                            "pairs": len(rows), "cluster_ids": boot["cluster_ids"],
                            "mae": interval(boot["mae"][j], boot["mae_draws"][:, j], boot["clusters"]),
                            "signed_bias": interval(boot["bias"][j], boot["bias_draws"][:, j], boot["clusters"]),
                            "mae_improvement_over_baseline": gain, "clears_target_reading_rule": clears_reading_rule(gain),
                            "empty_pair_resamples_redrawn": boot["empty_pair_resamples_redrawn"]})
    return metrics


def response_curves(points):
    groups = defaultdict(list)
    for point in points:
        groups[point.capability, point.student, point.U, point.seed].append(point)
    return [{"capability": cap, "student": student, "U": u, "seed": seed,
             "D_U": values[0].D_U, "initial_loss": values[0].initial_loss,
             "trajectory": values[0].trajectory,
             "points": [{"T": p.T, "processed_tokens": p.processed_tokens, "delta": p.delta,
                         "loss": p.loss, "E": p.E, "update": p.update, "source": p.source}
                        for p in sorted(values, key=lambda p: p.T)]}
            for (cap, student, u, seed), values in sorted(groups.items(),
                key=lambda item: (CAPABILITIES.index(item[0][0]), STUDENTS.index(item[0][1]), *item[0][2:]))]


def unmatched_budget_audit(points, pairs):
    """Report same-ordinal checkpoint mismatches; ordinal never defines pairing."""
    groups, result = defaultdict(dict), []
    existing = {p.id for p in pairs}
    for curve in response_curves(points):
        groups[curve["capability"], curve["student"], curve["seed"]][curve["U"]] = sorted(
            [p for p in points if p.trajectory == curve["trajectory"] and p.capability == curve["capability"] and p.T > 0], key=lambda p: p.T)
    for (cap, student, seed), pools in groups.items():
        for u1, u2 in combinations(sorted(pools), 2):
            for first, second in zip(pools[u1], pools[u2]):
                if Pair("I2", first, second).id not in existing:
                    result.append({"capability": cap, "student": student, "seed": seed, "U_first": u1, "U_second": u2,
                                   "T_first": first.T, "T_second": second.T,
                                   "ratio": max(first.T, second.T) / min(first.T, second.T),
                                   "reason": "Actual completion budgets exceed the fixed 1.10 ratio tolerance"})
    return result


def analyze(root=ROOT):
    points, audits, distributions, protocol, counts, inputs = load_data(root)
    pairs = construct_pairs(points)
    predictions, fits, folds = evaluate(points, pairs)
    metrics = score_predictions(predictions, pairs)
    verdicts = {cap: capability_verdict([r for r in metrics if r["capability"] == cap]) for cap in CAPABILITIES}
    summary = {
        "version": 95, "dataset": "18 controlled matrix2 trajectories; 3 students x 3 pools x 2 pool seeds",
        "cpu_only": True, "baseline_count": 1, "baseline_note": BASELINE_NOTE, "reading_rule": RULE,
        "definitions": {"delta": "checkpoint loss minus that trajectory's own update-0 loss (nats/completion token)",
                        "T": "actual cumulative supervised completion tokens, all training domains",
                        "D_U": "pool completion tokens across all training domains: sum loss_curve.completion_tokens over verified complete first epoch",
                        "E": "T/D_U", "T_ref": T_REF, "D_ref": D_REF,
                        "N": "logged total parameters minus trainable LoRA adapters minus hidden_size*vocab_size tied token embedding; includes 1-D language parameters",
                        "N_dimension_source": "configs/v28_source_metadata.json (dimensions only; its matrix-only N0 is a cross-check)"},
        "registered_targets": {
            "I1": "Fixed pool: delta at larger actual T minus delta at about half its T within one trajectory",
            "I2": "Fixed budget: delta in larger independent pool minus delta in smaller pool, matched actual T within student and seed",
            "I3": "Unseen-student transfer of I1 and I2, reported separately by held-out student and pooled LOSO; never pooled across targets"},
        "pairing": {"I1_inclusive_T_ratio_window": list(HALVING_WINDOW), "I2_max_T_over_min_T": MATCHED_BUDGET_RATIO,
                    "convention_source": "V89 numeric windows fixed before fitting V95; V95 additionally requires same seed and distinct pool sizes",
                    "all_eligible_observed_pairs": True, "include_66_to_594": True, "interpolation": False,
                    "use_actual_T_at_each_prediction_endpoint": True,
                    "counts_per_capability": {cap: dict(Counter(p.target for p in pairs if p.first.capability == cap)) for cap in CAPABILITIES},
                    "unmatched_same_ordinal_checkpoints": unmatched_budget_audit(points, pairs)},
        "candidates": FORMULAS, "fitting": {
            "method": "Unweighted per-capability OLS of delta at all five checkpoints including update-0; no tuning, no direct effect regression",
            "conditioned_coefficient_order": ["a", "a0", "q", "a1"],
            "student_standardization": "Training-fold checkpoint descriptors only; population std; scale=1 when training std=0; held-out own initial loss is an allowed input",
            "primary_split": SPLITS[0], "secondary_split": "Hold each pool seed out across all students and all pool sizes"},
        "bootstrap": {"draws": BOOTSTRAPS, "seed": BOOTSTRAP_SEED, "unit": "whole trajectory", "interval": "95% percentile",
                      "pair_weights": "I1: trajectory multiplicity once; I2: product of both endpoint trajectory multiplicities",
                      "empty_pair_draws": "redraw until exactly 5000 nonempty draws", "refit_in_bootstrap": False,
                      "conditioning": "Intervals condition on fitted cross-validation predictions; shared pool seeds may induce dependence beyond the requested trajectory unit",
                      "metric_weighting": "Each observed intervention pair has equal point-estimate weight; identical draws across candidates yield paired improvement intervals"},
        "protocol": protocol, "parameter_counts": counts, "evaluation_distributions": distributions,
        "trajectory_audit": audits, "verdicts": verdicts, "response_curves": response_curves(points),
        "points": [{**asdict(p), "id": p.id, "delta": p.delta, "E": p.E} for p in points],
        "pairs": [{"id": p.id, "target": p.target, "first_id": p.first.id, "second_id": p.second.id,
                   "actual": p.actual, "trajectory_clusters": p.clusters} for p in pairs],
        "folds": folds, "fits": fits, "predictions": predictions, "metrics": metrics, "input_sha256": inputs.digests,
        "method_notes": [
            "All input loss deltas use their own update-0; final root eval aliases are not duplicated.",
            "Actual processed and completion counts are reconciled across every loss-curve update, trajectory log, and saved eval.",
            "Nominal processed milestones and schedule_tokens validate protocol only; they never set T, D_U, E, or pairs.",
            "Snapshot total_updates_planned describes the original epoch ceiling; actual schedule update counts come from the completed train_log.",
            "I2 uses near-matched observed budgets, so residual T mismatch remains part of both the actual effect and each candidate's differenced prediction.",
            "Negative response delta means lower loss; signed bias is predicted intervention minus observed intervention.",
            "Response curves retain each seed separately; no smoothing, seed averaging, extra candidates, or additional fitting.",
        ],
    }
    inputs.verify()
    return summary


def verdict_block(summary):
    lines = ["V95 VERDICT", "", BASELINE_NOTE, "", RULE, ""]
    for cap in CAPABILITIES:
        row = summary["verdicts"][cap]
        lines.append(f"- {cap}: **{row['verdict']}**. Candidates passing both targets: "
                     + (", ".join(row["passing_candidates"]) or "none") + ".")
    return "\n".join(lines)


def response_curve_table(summary):
    lines = ["Raw response curves (T : delta); T is actual supervised completion tokens.",
             "Each row starts at its own update-0. Loss deltas are nats/token; seeds are separate.", "",
             "| Capability | Student | Pool U | Seed | D_U | Initial loss | T : delta (all five checkpoints) |",
             "|---|---|---:|---:|---:|---:|---|"]
    for curve in summary["response_curves"]:
        values = "; ".join(f"{p['T']:,} : {p['delta']:+.6f}" for p in curve["points"])
        lines.append(f"| {curve['capability']} | {curve['student']} | {curve['U']} | {curve['seed']} | "
                     f"{curve['D_U']:,} | {curve['initial_loss']:.6f} | {values} |")
    return "\n".join(lines)


def format_interval(value):
    low, high = value["ci95"]
    warning = "; below six trajectory clusters" if value["below_six_clusters"] else ""
    return f"{value['estimate']:.6f} [{low:.6f}, {high:.6f}] (clusters={value['clusters']}{warning})"


def render_markdown(summary):
    lines = [verdict_block(summary), "", summary["dataset"], "", "Protocol and accounting", "",
             "CPU/NumPy only. No GPU access, model loading, training, inference, or new measurements.", "",
             *summary["method_notes"], "", "D_U is reconstructed from a verified complete first epoch, not from prompt-plus-target pool tokens.",
             "T_ref = D_ref = 100,000 completion tokens. I1 uses inclusive T ratios [1.7, 2.3]. "
             "I2 uses max(T)/min(T) <= 1.10 within the same student and pool seed, for all three pool contrasts.", "",
             "Fits use all five delta checkpoints, including update-0, with equal OLS weight per checkpoint. "
             "The student-conditioned form has four coefficients: a + (a0 + a1*z)*log(1+T/T_ref) + q*log(D_U/D_ref). "
             "Descriptor means and scales come only from training folds.", "",
             "Non-embedding counts remove LoRA adapters and the tied token embedding from logged parameter totals, "
             "retaining 1-D language weights: " + ", ".join(f"{s}={n:,}" for s, n in summary["parameter_counts"].items()) + ".", "",
             "| Student | U | Seed | D_U completion | Pool processed | Final processed | Final completion |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for a in summary["trajectory_audit"]:
        lines.append(f"| {a['student']} | {a['U']} | {a['seed']} | {a['D_U']:,} | {a['pool_processed_tokens']:,} | "
                     f"{a['final_processed_tokens']:,} | {a['final_completion_tokens']:,} |")
    lines += ["", "| Capability | I1 budget pairs | I2 independent-data pairs |", "|---|---:|---:|"]
    for cap, counts in summary["pairing"]["counts_per_capability"].items():
        lines.append(f"| {cap} | {counts.get('I1', 0)} | {counts.get('I2', 0)} |")
    lines += ["", "Unmatched same-ordinal I2 endpoints (not interpolated or scored):", "",
              "| Capability | Student | Seed | Pool contrast | Actual T endpoints | max(T)/min(T) |", "|---|---|---:|---|---|---:|"]
    for row in summary["pairing"]["unmatched_same_ordinal_checkpoints"]:
        lines.append(f"| {row['capability']} | {row['student']} | {row['seed']} | {row['U_first']}→{row['U_second']} | "
                     f"{row['T_first']:,}, {row['T_second']:,} | {row['ratio']:.6f} |")
    lines += ["", response_curve_table(summary), "", "Registered intervention prediction metrics", "",
              "I1 = budget doubling at fixed pool. I2 = more independent data at matched budget. "
              "I3 = the separate held-out-student I1 and I2 results below; pooled LOSO is primary.", "",
              "Secondary leave-one-pool-seed-out holds seed 41 or 42 out across every student and pool. "
              "Every displayed interval uses 5,000 whole-trajectory bootstrap draws (seed 0), conditional on the fitted predictions. "
              "I2 resamples both endpoint trajectories, multiplying their multiplicities; empty pair draws are redrawn. "
              "All candidates share draws for paired MAE improvements. Bias = prediction minus actual; positive gain means lower MAE. "
              "The rule uses the full width of the paired gain interval.", "",
              "Shared pools may create dependence beyond the requested trajectory bootstrap unit. "
              "Cluster counts are trajectory counts, not checkpoint, student, or pool-seed counts.", ""]
    small = [m for m in summary["metrics"] if m["mae"]["clusters"] < 6]
    lines.append("Some reported intervals have below six trajectory clusters; they are explicitly marked."
                 if small else "No reported interval has below six trajectory clusters: pooled=18, held-out student=6, held-out pool seed=9.")
    for split in SPLITS:
        scopes = ["pooled"] + sorted({m["scope"] for m in summary["metrics"] if m["split"] == split and m["scope"] != "pooled"})
        for scope in scopes:
            lines += ["", f"{split} / {scope}", "",
                      "| Capability | Target | Candidate | Pairs | MAE [95% CI] | Signed bias [95% CI] | Baseline MAE − candidate MAE [95% CI] | Full gain CI width | Gain > width |",
                      "|---|---|---|---:|---|---|---|---:|---|"]
            for m in summary["metrics"]:
                if m["split"] != split or m["scope"] != scope:
                    continue
                lines.append(f"| {m['capability']} | {m['target']} | {m['candidate']} | {m['pairs']} | "
                             f"{format_interval(m['mae'])} | {format_interval(m['signed_bias'])} | "
                             f"{format_interval(m['mae_improvement_over_baseline'])} | "
                             f"{m['mae_improvement_over_baseline']['width']:.6f} | {'yes' if m['clears_target_reading_rule'] else 'no'} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Input repository root; output location remains fixed within it")
    args = parser.parse_args()
    summary = analyze(args.root)
    output = args.root / OUT_REL
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "summary.md").write_text(render_markdown(summary))
    print(verdict_block(summary))
    print()
    print(response_curve_table(summary))


if __name__ == "__main__":
    main()
