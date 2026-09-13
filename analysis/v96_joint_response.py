#!/usr/bin/env python3
"""CPU-only analysis of the 18 saved matrix2 trajectories; no model execution.

python3 -B analysis/v96_joint_response.py --print-tables
Only writes results/v96-joint-response/{summary.json,summary.md}.
Missing domain accounting and missing exact-T observations are reported, never
replaced by example shares, nominal budgets, or interpolated measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from itertools import combinations, product
import json
import math
from pathlib import Path

import numpy as np

try:
    from analysis import v95_matrix_intervention as old
except ModuleNotFoundError:
    import v95_matrix_intervention as old

ROOT = Path(__file__).resolve().parents[1]
OUT_REL = Path("results/v96-joint-response")
T_REF = 100_000.0  # Fixed before fitting, identical to V95.
BOOTSTRAPS = 5000
DESCRIPTORS = ("log_parameters", "initial_loss")
CANDIDATES = ("zero", "T_only", "joint_log_parameters", "surface_log_parameters",
              "joint_initial_loss", "surface_initial_loss")
BASELINE = "mean_effect"
SIZE_NAMES = {66: "small", 198: "middle", 594: "large"}
FAIL = "did not meet the pre-registered threshold"
MISSING_SHARES = (
    "Not recorded: logs contain total completion tokens and per-domain example "
    "counts, but no per-domain supervised-token totals or per-example token ledger. "
    "Example proportions and evaluation measurement_tokens are not training token shares."
)


def assert_controlled_directories(selected, discarded):
    """Fail closed, including if a discarded launch is explicitly supplied."""
    selected, discarded = list(map(Path, selected)), list(map(Path, discarded))
    assert all(p.name.endswith(("_matrix2_lora_dseed41", "_matrix2_lora_dseed42"))
               for p in selected), "Excluded _matrix_ launch entered the selected trajectories"
    assert len(selected) == len(set(selected)) == 18, "Expected exactly 18 controlled trajectories"
    assert len(discarded) == 5, "Expected five discarded-launch directories"
    assert set(selected).isdisjoint(discarded), "Discarded directories must be excluded"
    assert all(p.name.endswith(("_matrix_lora_dseed41", "_matrix_lora_dseed42"))
               for p in discarded), "Unexpected discarded-directory suffix"


def directory_audit(root):
    base = Path(root) / "results/v12-distill"
    directories = [p for p in base.glob("*/*") if p.is_dir()]
    selected = sorted(p for p in directories if p.name.endswith(
        ("_matrix2_lora_dseed41", "_matrix2_lora_dseed42")))
    discarded = sorted(p for p in directories if p.name.endswith(
        ("_matrix_lora_dseed41", "_matrix_lora_dseed42")))
    assert_controlled_directories(selected, discarded)
    expected = {base / s / f"gpt-5.6-luna_full_{u}_matrix2_lora_dseed{seed}"
                for s, u, seed in product(old.STUDENTS, old.POOLS, old.SEEDS)}
    assert set(selected) == expected, "Controlled matrix identities changed"
    return {"passed": True, "selected_count": 18, "excluded_count": 5,
            "selected": [str(p.relative_to(root)) for p in selected],
            "excluded": [str(p.relative_to(root)) for p in discarded],
            "excluded_contents_read": False}


def recorded_domain_shares(payload):
    """Only an explicit supervised-token ledger can supply these shares."""
    totals = payload.get("completion_tokens_seen_by_domain")
    if totals is None:
        return {cap: None for cap in old.CAPABILITIES}
    old.require(set(totals) == set(old.CAPABILITIES), "Incomplete domain token ledger")
    old.require(all(type(v) is int and v >= 0 for v in totals.values()), "Invalid domain counts")
    total = payload["completion_tokens_seen"]
    old.require(sum(totals.values()) == total, "Domain completion accounting disagrees")
    return {cap: totals[cap] / total if total else None for cap in old.CAPABILITIES}


def accounting_rows(points, inputs):
    result = {}
    for p in points:
        if p.id in result:
            continue
        payload = inputs.read(inputs.root / p.source)
        log = inputs.read(inputs.root / "results/v12-distill" / p.trajectory / "train_log.json")
        step = next((r for r in log["loss_curve"] if r["step"] == p.update), {})
        result[p.id] = {
            "T": p.T, "D_U": p.D_U, "processed_tokens": p.processed_tokens,
            "supervised_token_shares": recorded_domain_shares(payload),
            "shares_status": "recorded" if "completion_tokens_seen_by_domain" in payload else MISSING_SHARES,
            "source": p.source, "update": p.update, "epoch": step.get("epoch"),
            "lr": step.get("lr", 0.0), "schedule_updates": log["total_updates_planned"],
            "warmup_updates": log["warmup_steps"],
            "training_example_counts": {c: log["trace_counts"][c]["training_rows"] for c in old.CAPABILITIES},
        }
    return result


def pool_contrasts(points, accounting):
    """Same-ordinal diagnostics include all endpoints, even ineligible V95 pairs."""
    curves = defaultdict(list)
    for p in points:
        if p.T > 0:
            curves[p.capability, p.student, p.U, p.seed].append(p)
    curves = {k: sorted(v, key=lambda p: p.T) for k, v in curves.items()}
    rows, pairs = [], []
    for cap, student in product(old.CAPABILITIES, old.STUDENTS):
        specs = [("size", u, seed, v, seed) for seed in old.SEEDS for u, v in combinations(old.POOLS, 2)]
        specs += [("seed", u, 41, u, 42) for u in old.POOLS]
        for kind, u1, s1, u2, s2 in specs:
            contrast = f"{SIZE_NAMES[u1]}->{SIZE_NAMES[u2]}" if kind == "size" else f"{SIZE_NAMES[u1]}:seed41->42"
            for checkpoint, (first, second) in enumerate(zip(curves[cap, student, u1, s1],
                                                           curves[cap, student, u2, s2]), 1):
                pair = old.Pair(f"I2_{kind}", first, second)
                pairs.append(pair)
                rows.append({"pair_id": pair.id, "kind": kind, "capability": cap, "student": student,
                             "checkpoint": checkpoint, "contrast": contrast,
                             "seed_first": s1, "seed_second": s2, "U_first": u1, "U_second": u2,
                             "first": accounting[first.id], "second": accounting[second.id],
                             "T2_over_T1": second.T / first.T,
                             "max_T_over_min_T": max(first.T, second.T) / min(first.T, second.T),
                             "exact_T": first.T == second.T,
                             "eligible_near_match": max(first.T, second.T) / min(first.T, second.T) <= old.MATCHED_BUDGET_RATIO,
                             "measured_effect": pair.actual, "trajectory_clusters": list(pair.clusters)})
    assert len(rows) == 324 and Counter(r["kind"] for r in rows) == {"size": 216, "seed": 108}
    return rows, pairs


def mismatch_audit(prior, points, pairs):
    """Reuse V95 numbers; verify every I2 prediction is only a log-budget difference."""
    pair_by_id = {p.id: p for p in old.construct_pairs(points)}
    fits = {(r["split"], r["fold"], r["capability"]): r for r in prior["fits"] if r["candidate"] == "T_only"}
    maximum_error, effects = 0.0, []
    for row in prior["predictions"]:
        if row["target"] != "I2":
            continue
        pair = pair_by_id[row["pair_id"]]
        a = fits[row["split"], row["fold"], row["capability"]]["coefficients"][0]
        expected = a * (np.log1p(pair.second.T / T_REF) - np.log1p(pair.first.T / T_REF))
        actual = row["predicted_intervention"]["T_only"]
        maximum_error = max(maximum_error, abs(expected - actual))
        assert math.isclose(expected, actual, rel_tol=1e-11, abs_tol=1e-13)
        effects.append(abs(actual))
    reused = [{"split": r["split"], "scope": r["scope"], "capability": r["capability"],
               "gain_over_zero": r["mae_improvement_over_baseline"],
               "fraction_explained_by_T_mismatch": 1.0, "fixed_T_prediction": 0.0,
               "fixed_T_gain_over_zero": 0.0}
              for r in prior["metrics"] if r["candidate"] == "T_only" and r["target"] == "I2"]
    return {"resolution": "100% of V95 T-only nonzero predictions and MAE gains/losses on I2 arise from residual budget mismatch; its T-only F has no other moving input.",
            "V95_matching_tolerance": old.MATCHED_BUDGET_RATIO,
            "max_prediction_reconstruction_error": maximum_error,
            "max_absolute_V95_T_only_pool_prediction": max(effects),
            "nonzero_budget_exact_pool_pairs": sum(p.first.T == p.second.T for p in pairs),
            "matched_T_prediction_exactly_zero": True, "reused_V95_metrics": reused,
            "qualification": "The counterfactual fixed-T prediction is algebraic, not a measured fixed-T effect. No exact nonzero-T pool observations exist. The decomposition explains model gains, not how much of the observed loss difference is causally due to budget."}


def descriptor_candidate(descriptor):
    old.require(descriptor in DESCRIPTORS, "Unknown descriptor")
    return "student_log_parameters" if descriptor == "log_parameters" else "student_initial_loss"


def design(points, candidate, scaler=None):
    old.require(candidate in CANDIDATES or candidate.startswith("interaction_"), "Unknown candidate")
    t = np.array([p.T / T_REF for p in points])
    e = np.array([p.E for p in points])
    x, y = np.log1p(t), np.log1p(e)
    if candidate == "zero":
        return np.zeros((len(points), 0))
    if candidate == "T_only":
        return x[:, None]
    descriptor = candidate.split("_", 1)[1]
    old.require(scaler is not None, "Descriptor must be standardized on training fold")
    z = scaler.transform(points, descriptor_candidate(descriptor))
    if candidate.startswith("surface_"):
        # Degree <=2 in exactly the same (T/T_ref, E, z) inputs, zero at T=0.
        return np.column_stack((t, e, z*t, z*e, t*t, t*e, e*e))
    columns = [x, z*x, y, z*y]
    if candidate.startswith("interaction_"):
        columns.append(x*y)
    return np.column_stack(columns)


@dataclass
class ResponseFit:
    candidate: str
    coefficients: list
    standardizer: old.Standardizer | None
    rank: int
    training_point_ids: tuple

    def response(self, points):
        return design(points, self.candidate, self.standardizer) @ np.asarray(self.coefficients)

    def intervention(self, pairs):
        return self.response([p.second for p in pairs]) - self.response([p.first for p in pairs])

    def budget_doubling(self, points):
        return self.response([replace(p, T=2*p.T) for p in points]) - self.response(points)

    def fixed_T_pool_change(self, pairs):
        assert all(p.first.student == p.second.student and p.first.capability == p.second.capability
                   and p.first.initial_loss == p.second.initial_loss for p in pairs)
        endpoints = [replace(p.second, T=p.first.T) for p in pairs]
        return self.response(endpoints) - self.response([p.first for p in pairs])


def fit_response(training, candidate, *, stable_residual=False):
    old.require(training and len({(p.capability, p.distribution) for p in training}) == 1,
                "Response fit requires one capability/distribution")
    if candidate.startswith("interaction_"):
        old.require(stable_residual, "Extra interaction requires stable budget residual")
    scaler = None
    if candidate not in ("zero", "T_only"):
        scaler = old.Standardizer.fit(training, descriptor_candidate(candidate.split("_", 1)[1]))
    matrix = design(training, candidate, scaler)
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, [p.delta for p in training], rcond=None)
    model = ResponseFit(candidate, coefficients.tolist(), scaler, int(rank), tuple(p.id for p in training))
    assert np.array_equal(model.response([replace(p, T=0) for p in training]), np.zeros(len(training)))
    return model


def same_column_space(first, second):
    # Normalize columns before rank tests; comparison is independent of units.
    def norm(x):
        scales = np.linalg.norm(x, axis=0)
        return x / np.where(scales > 0, scales, 1)
    a, b = norm(first), norm(second)
    ranks = [int(np.linalg.matrix_rank(x)) if x.shape[1] else 0
             for x in (a, b, np.column_stack((a, b)))]
    return ranks[0] == ranks[1] == ranks[2], ranks


def equivalence_check(prior):
    assert prior["candidates"] == old.FORMULAS, "V95 registry changed; re-audit before fitting"
    # A finite witness with unequal spans disproves algebraic equivalence.
    # Same-span numerical results alone would not prove global equivalence.
    witness = [old.Point(f"w{s}", str(s), 66, 41, "math", "w", 0, 0, t, du,
                         float(s+1), float(s+1), int(10**(s+6)))
               for s, t, du in product(range(3), (0, 1000, 100000, 900000), (13000, 50000, 170000))]
    comparisons = []
    for descriptor in DESCRIPTORS:
        scaler = old.Standardizer.fit(witness, descriptor_candidate(descriptor))
        joint = design(witness, f"joint_{descriptor}", scaler)
        for candidate in old.CANDIDATES:
            old_scaler = old.Standardizer.fit(witness, candidate) if candidate in old.CANDIDATES[-2:] else None
            equal, ranks = same_column_space(joint, old.features(witness, candidate, old_scaler))
            assert not equal, "Potential equivalence: reuse earlier numbers after symbolic verification"
            comparisons.append({"descriptor": descriptor, "V95_candidate": candidate,
                                "equivalent": False, "ranks_joint_old_union": ranks})
    return {"performed_before_fitting": True, "equivalent_candidates": [], "comparisons": comparisons,
            "algebra": "Joint basis is [x,z*x,y,z*y], x=log(1+T/T_ref), y=log(1+T/D_U). V95 conditioned basis was [1,x,log(D_U/D_ref),z*x]. At T=0 the joint is zero for every D_U; the V95 family is a+q*log(D_U/D_ref), forcing a=q=0 to obey that constraint. This leaves only [x,z*x], which cannot span the joint y and z*y terms. log(1+T/D_U)=log(T+D_U)-log(D_U), not log(T)-log(D_U). The one-term T/reuse forms are proper submodels, not equivalent four-parameter families.",
            "numbers_reused": "V95 T-only diagnostic effects, gains, and intervals; no joint equivalent exists."}


@dataclass(frozen=True)
class MeanEffect:
    capability: str
    target: str
    value: float
    training_pair_ids: tuple
    training_trajectories: tuple


def fit_mean_effect(training, pairs, capability, target):
    """Select both endpoints from training before inspecting outcomes."""
    ids = {p.id for p in training}
    selected = [p for p in pairs if p.target == target and p.first.capability == capability
                and p.first.id in ids and p.second.id in ids]
    if not selected:
        return None
    return MeanEffect(capability, target, float(np.mean([p.actual for p in selected])),
                      tuple(p.id for p in selected), tuple(sorted({c for p in selected for c in p.clusters})))


def evaluation_pairs(points, pool_pairs):
    pairs = [p for p in old.construct_pairs(points) if p.target == "I1"]
    pairs.extend(p for p in pool_pairs if max(p.first.T, p.second.T) / min(p.first.T, p.second.T) <= old.MATCHED_BUDGET_RATIO)
    anchors = {(p.trajectory, p.capability): p for p in points if p.T == 0}
    pairs.extend(old.Pair("response", anchors[p.trajectory, p.capability], p) for p in points if p.T > 0)
    # The unchanged V95 I2 set is exactly the size stratum, with an explicit name.
    assert {replace(p, target="I2").id for p in pairs if p.target == "I2_size"} == {
        p.id for p in old.construct_pairs(points) if p.target == "I2"}
    return pairs


def evaluate(points, pairs):
    predictions, fits, means, folds = [], [], [], []
    for split in old.SPLITS:
        for fold, training, held in old.split_folds(points, split):
            held_ids = {p.id for p in held}
            selected = [p for p in pairs if p.first.id in held_ids and p.second.id in held_ids]
            folds.append({"split": split, "fold": fold,
                          "training_trajectories": sorted({p.trajectory for p in training}),
                          "held_out_trajectories": sorted({p.trajectory for p in held}),
                          "target_counts": dict(Counter(p.target for p in selected))})
            for cap in old.CAPABILITIES:
                train = [p for p in training if p.capability == cap]
                cap_pairs = [p for p in selected if p.first.capability == cap]
                models = {c: fit_response(train, c) for c in CANDIDATES}
                effects = {c: m.intervention(cap_pairs) for c, m in models.items()}
                endpoints = {c: (m.response([p.first for p in cap_pairs]), m.response([p.second for p in cap_pairs]))
                             for c, m in models.items()}
                baseline = {target: fit_mean_effect(training, pairs, cap, target)
                            for target in {p.target for p in cap_pairs}}
                assert all(v is not None for v in baseline.values())
                for target, m in baseline.items():
                    means.append({"split": split, "fold": fold, **asdict(m)})
                for candidate, m in models.items():
                    fits.append({"split": split, "fold": fold, "capability": cap, **asdict(m)})
                for i, pair in enumerate(cap_pairs):
                    predicted = {c: float(effects[c][i]) for c in CANDIDATES}
                    predicted[BASELINE] = baseline[pair.target].value
                    fixed = {c: float(m.fixed_T_pool_change([pair])[0]) for c, m in models.items()} if pair.target.startswith("I2") else None
                    if fixed is not None:
                        assert fixed["T_only"] == fixed["zero"] == 0.0
                    predictions.append({"split": split, "fold": fold, "capability": cap, "target": pair.target,
                                        "pair_id": pair.id, "actual": pair.actual, "predictions": predicted,
                                        "endpoint_predictions": {c: [float(a[i]), float(b[i])] for c, (a, b) in endpoints.items()},
                                        "fixed_T_counterfactual_predictions_unscored": fixed})
    return predictions, fits, means, folds


def score(predictions, pairs, names=(*CANDIDATES, BASELINE)):
    index = {p.id: p for p in pairs}
    groups = defaultdict(list)
    for row in predictions:
        for scope in ("pooled", f"held_out={row['fold']}"):
            groups[row["split"], scope, row["capability"], row["target"]].append(row)
    results = []
    for (split, scope, cap, target), rows in sorted(groups.items()):
        selected = [index[r["pair_id"]] for r in rows]
        errors = np.array([[r["predictions"][c]-r["actual"] for c in names] for r in rows])
        boot = old.bootstrap_statistics(selected, errors, BOOTSTRAPS)
        for j, candidate in enumerate(names):
            gains = {}
            references = ["zero", BASELINE]
            if candidate.startswith(("joint_", "interaction_")):
                descriptor = candidate.split("_", 1)[1]
                references.append(f"surface_{descriptor}")
                if candidate.startswith("interaction_"):
                    references.append(f"joint_{descriptor}")
            for baseline in references:
                k = names.index(baseline)
                gains[baseline] = old.interval(boot["mae"][k]-boot["mae"][j],
                                               boot["mae_draws"][:, k]-boot["mae_draws"][:, j], boot["clusters"])
            results.append({"split": split, "scope": scope, "capability": cap, "target": target,
                            "candidate": candidate, "pairs": len(rows), "cluster_ids": boot["cluster_ids"],
                            "mae": old.interval(boot["mae"][j], boot["mae_draws"][:, j], boot["clusters"]),
                            "bias": old.interval(boot["bias"][j], boot["bias_draws"][:, j], boot["clusters"]),
                            "gains": gains, "clears_zero_rule": old.clears_reading_rule(gains["zero"]),
                            "empty_draws_redrawn": boot["empty_pair_resamples_redrawn"]})
    return results


def verdicts(metrics):
    primary = {(r["capability"], r["candidate"], r["target"]): r for r in metrics
               if r["split"] == old.SPLITS[0] and r["scope"] == "pooled"}
    result = []
    for cap, candidate in product(old.CAPABILITIES, (*CANDIDATES[1:], BASELINE)):
        rows = [primary[cap, candidate, target] for target in ("I1", "I2_size")]
        passed = all(r["clears_zero_rule"] for r in rows)
        result.append({"capability": cap, "candidate": candidate,
                       "status": "met the pre-registered threshold" if passed else FAIL,
                       "both_targets_over_training_mean": all(old.clears_reading_rule(r["gains"][BASELINE]) for r in rows),
                       "both_targets_over_same_input_surface": all(old.clears_reading_rule(r["gains"][candidate.replace("joint_", "surface_", 1)]) for r in rows) if candidate.startswith("joint_") else None})
    return result


def cluster_weights(pairs, seed=0):
    clusters = sorted({c for p in pairs for c in p.clusters})
    rng = np.random.default_rng(seed)
    blocks, remaining = [], BOOTSTRAPS
    while remaining:
        counts = rng.multinomial(len(clusters), np.full(len(clusters), 1/len(clusters)), size=remaining)
        weights = old.pair_weights(pairs, counts, clusters)
        weights = weights[weights.sum(axis=1) > 0]
        blocks.append(weights)
        remaining -= len(weights)
    return np.concatenate(blocks), len(clusters)


def slope_statistic(rows, pairs, values):
    """Within-dyad slope: student/pool-pair intercepts cancel before fitting."""
    x = np.array([math.log1p((p.first.T+p.second.T)/(2*T_REF)) for p in pairs])
    y = np.asarray(values)
    groups = defaultdict(list)
    for i, p in enumerate(pairs):
        groups[p.clusters].append(i)
    numerator, denominator = np.zeros(len(y)), np.zeros(len(y))
    for ids in groups.values():
        dx, dy = x[ids]-x[ids].mean(), y[ids]-y[ids].mean()
        numerator[ids], denominator[ids] = dx*dy, dx*dx
    weights, n = cluster_weights(pairs)
    den = weights @ denominator
    assert np.all(den > 0), "Slope requires multiple checkpoints per trajectory dyad"
    samples = (weights @ numerator) / den
    estimate = numerator.sum()/denominator.sum()
    interval = old.interval(estimate, samples, n)
    interval["bootstrap_sign_tail_probability"] = min(1.0, 2*min((1+sum(samples <= 0))/(BOOTSTRAPS+1),
                                                                 (1+sum(samples >= 0))/(BOOTSTRAPS+1)))
    interval["excludes_zero"] = interval["ci95"][0] > 0 or interval["ci95"][1] < 0
    return interval


def ratio_diagnostics(rows, pairs, fits):
    pair_index = {p.id: p for p in pairs}
    fit_index = {(f["capability"], f["fold"], f["candidate"]): ResponseFit(
        f["candidate"], f["coefficients"], old.Standardizer(**f["standardizer"]), f["rank"], tuple(f["training_point_ids"]))
        for f in fits if f["split"] == old.SPLITS[0] and f["candidate"].startswith("joint_")}
    detailed = []
    for row in rows:
        pair = pair_index[row["pair_id"]]
        bracket = np.log1p(pair.second.E)-np.log1p(pair.first.E)
        assert bracket != 0, "Zero reuse bracket: ratio undefined"
        common_bracket = np.log1p(pair.first.T/pair.second.D_U)-np.log1p(pair.first.E)
        assert common_bracket != 0
        adjusted, residual = {}, {}
        for descriptor in DESCRIPTORS:
            model = fit_index[row["capability"], row["student"], f"joint_{descriptor}"]
            common = replace(pair.second, T=pair.first.T)
            mismatch = float(model.response([pair.second])[0]-model.response([common])[0])
            adjusted[descriptor] = (pair.actual-mismatch)/common_bracket
            predicted_ratio = float(model.fixed_T_pool_change([pair])[0])/common_bracket
            residual[descriptor] = adjusted[descriptor]-predicted_ratio
        detailed.append({**{k: row[k] for k in ("pair_id", "kind", "capability", "student", "checkpoint", "contrast", "seed_first", "eligible_near_match")},
                         "actual_T_first": pair.first.T, "actual_T_second": pair.second.T,
                         "measured_effect": pair.actual, "observed_endpoint_bracket": float(bracket),
                         "raw_ratio": float(pair.actual/bracket), "common_T_bracket": float(common_bracket),
                         "model_adjusted_ratio_sensitivity": adjusted,
                         "model_adjusted_ratio_residual": residual,
                         "exact_fixed_T_measurement": pair.first.T == pair.second.T})
    groups = defaultdict(list)
    for r in detailed:
        groups[r["kind"], r["capability"], r["student"], r["contrast"], r["checkpoint"]].append(r)
    table = []
    for (kind, cap, student, contrast, checkpoint), group in sorted(groups.items()):
        selected = [pair_index[r["pair_id"]] for r in group]
        values = np.array([r["raw_ratio"] for r in group])
        weights, n = cluster_weights(selected)
        table.append({"kind": kind, "capability": cap, "student": student, "contrast": contrast,
                      "checkpoint": checkpoint, "ratio": old.interval(values.mean(), weights@values/weights.sum(axis=1), n),
                      "pairs": len(group), "near_match_pairs": sum(r["eligible_near_match"] for r in group),
                      "interval_note": "Conditional whole-trajectory bootstrap; two clusters for one seed contrast produce a degenerate interval, not measurement certainty."})
    slope_groups = defaultdict(list)
    for r in detailed:
        # Pool across checkpoints only after retaining a separate intercept for
        # each trajectory dyad. Size/seed contrasts and capabilities never mix.
        scopes = [r["student"], "all_students"]
        if r["eligible_near_match"]:
            scopes.append("all_students_near_match")
        if r["kind"] == "size":
            scopes.append(f"all_students_seed{r['seed_first']}")
        for student in scopes:
            slope_groups[r["kind"], r["capability"], student, r["contrast"]].append(r)
    slopes = []
    for (kind, cap, student, contrast), group in sorted(slope_groups.items()):
        selected = [pair_index[r["pair_id"]] for r in group]
        raw = slope_statistic(group, selected, [r["raw_ratio"] for r in group])
        sensitivities = {d: slope_statistic(group, selected, [r["model_adjusted_ratio_residual"][d] for r in group]) for d in DESCRIPTORS}
        slopes.append({"kind": kind, "capability": cap, "student": student, "contrast": contrast,
                       "raw_ratio_slope": raw, "adjusted_residual_slopes": sensitivities})
    # This is an explicit descriptive gate, not a new fitted threshold: drift
    # must survive mismatch adjustment, the registered eligibility filter, all
    # three size contrasts and each seed. It does not establish exact control.
    gates, gate_evidence = {}, {}
    for cap in old.CAPABILITIES:
        selected = [r for r in slopes if r["kind"] == "size" and r["capability"] == cap]
        pooled = [r for r in selected if r["student"] in ("all_students", "all_students_near_match")]
        seedwise = [r for r in selected if r["student"].startswith("all_students_seed")]
        statistics = [r["raw_ratio_slope"] for r in pooled]
        statistics += [s for r in pooled for s in r["adjusted_residual_slopes"].values()]
        direction = np.sign(statistics[0]["estimate"])
        supported = all(s["excludes_zero"] and np.sign(s["estimate"]) == direction for s in statistics)
        replicated = all(np.sign(s["estimate"]) == direction for r in seedwise
                         for s in (r["raw_ratio_slope"], *r["adjusted_residual_slopes"].values()))
        gates[cap] = bool(supported and replicated)
        gate_evidence[cap] = {"pooled_and_eligible_interval_support": bool(supported),
                              "each_seed_same_slope_direction": bool(replicated)}
    return {"exact_fixed_T_ratio_table": [], "observed_near_budget_ratio_table": table,
            "endpoint_ratios": detailed, "slopes": slopes,
            "ratio_definition": "Measured delta2-delta1 divided by log(1+T2/D_U2)-log(1+T1/D_U1). T2 differs from T1: these are observed-endpoint diagnostics, not the exact fixed-T falsification statistic.",
            "slope_definition": "OLS within trajectory-dyad slope versus log(1+(T1+T2)/(2*T_ref)); dyad centering controls student, seed and pool-contrast intercepts. Exactly 5000 whole-trajectory draws; bootstrap sign-tail probability is descriptive, not an exact null p-value. No multiple-comparison claim.",
            "sensitivity": "Separately for each joint descriptor, subtract F(T2,D_U2,z)-F(T1,D_U2,z) from the measured effect, divide by the common-T1 bracket, then subtract b+b'*z. This is model-dependent mismatch adjustment, not an independent fixed-T measurement. All F parameters come from the leave-one-student-out training fold.",
            "conclusion": "Exact fixed-T support/falsification cannot be identified from these saved checkpoints. A nonzero observed-endpoint slope diagnoses drift plus possible budget/composition/schedule confounding; a CI containing zero does not establish flatness.",
            "interaction_gate": {"stable_controlled_budget_residual_established": False,
                                 "stable_observed_residual_by_capability": gates,
                                 "evidence_by_capability": gate_evidence,
                                 "refitted": False, "permitted_extra_term": "k*log(1+T/T_ref)*log(1+E)",
                                 "rule": "Before extension fitting: raw and both descriptor-adjusted residual slope intervals exclude zero in one common direction for all three size contrasts, both with all ordinal checkpoints and with only registered near-matches; slope signs must replicate separately in each pool seed. This tests stability of an observed residual, not exact fixed-T causality.",
                                 "reason": "No nonzero-budget exact-T pool pairs and no recorded domain shares. A stable observed residual permits the requested one-term descriptive refit, with the exact-T limitation retained. Its post-diagnostic selection cannot revise the frozen four-coefficient registered decisions."}}


def conditional_extension(points, pairs, predictions, diagnostic):
    """Only the requested fifth coefficient; still fit responses, never effects."""
    gate = diagnostic["interaction_gate"]["stable_observed_residual_by_capability"]
    eligible = {cap for cap, passed in gate.items() if passed}
    if not eligible:
        return {"refitted": False, "reason": "No capability passed the observed-residual stability gate."}
    index = {p.id: p for p in pairs}
    models, fits = {}, []
    for split in old.SPLITS:
        for fold, training, _ in old.split_folds(points, split):
            for cap, descriptor in product(sorted(eligible), DESCRIPTORS):
                candidate = f"interaction_{descriptor}"
                model = fit_response([p for p in training if p.capability == cap], candidate, stable_residual=True)
                models[split, fold, cap, descriptor] = model
                fits.append({"split": split, "fold": fold, "capability": cap, **asdict(model)})
    augmented = []
    for row in predictions:
        if row["capability"] not in eligible:
            continue
        effects = dict(row["predictions"])
        endpoints = dict(row["endpoint_predictions"])
        for descriptor in DESCRIPTORS:
            model = models[row["split"], row["fold"], row["capability"], descriptor]
            pair = index[row["pair_id"]]
            effects[model.candidate] = float(model.intervention([pair])[0])
            endpoints[model.candidate] = model.response([pair.first, pair.second]).tolist()
        augmented.append({**row, "predictions": effects, "endpoint_predictions": endpoints})
    metrics = score(augmented, pairs, (*CANDIDATES, BASELINE, *(f"interaction_{d}" for d in DESCRIPTORS)))
    metrics = [r for r in metrics if r["candidate"].startswith("interaction_")]
    # Diagnose the fifth term on held-out residuals, controlling its OWN
    # mismatch response as well as its budget-dependent predicted pool ratio.
    pool_index = {p.id: p for p in pairs if p.target.startswith("I2")}
    residuals, groups = {}, defaultdict(list)
    for row in diagnostic["endpoint_ratios"]:
        if row["capability"] not in eligible or row["pair_id"] not in pool_index:
            continue
        pair = pool_index[row["pair_id"]]
        residuals[row["pair_id"]] = {}
        for descriptor in DESCRIPTORS:
            model = models[old.SPLITS[0], row["student"], row["capability"], descriptor]
            common = replace(pair.second, T=pair.first.T)
            mismatch = float(model.response([pair.second])[0]-model.response([common])[0])
            predicted_fixed = float(model.fixed_T_pool_change([pair])[0])
            residuals[row["pair_id"]][descriptor] = (pair.actual-mismatch-predicted_fixed)/row["common_T_bracket"]
        groups[row["kind"], row["capability"], row["contrast"]].append(row)
    slopes = []
    for (kind, cap, contrast), rows in sorted(groups.items()):
        selected = [pool_index[r["pair_id"]] for r in rows]
        for descriptor in DESCRIPTORS:
            before = slope_statistic(rows, selected, [r["model_adjusted_ratio_residual"][descriptor] for r in rows])
            after = slope_statistic(rows, selected, [residuals[r["pair_id"]][descriptor] for r in rows])
            slopes.append({"kind": kind, "capability": cap, "contrast": contrast, "descriptor": descriptor,
                           "before": before, "after": after,
                           "remaining_detected_drift": after["excludes_zero"],
                           "absolute_slope_reduced": abs(after["estimate"]) < abs(before["estimate"]),
                           "interpretation": "Drift remains detectable" if after["excludes_zero"] else "Residual interval includes zero; this does not prove flatness"})
    return {"refitted": True, "eligible_capabilities": sorted(eligible),
            "formula": "F_joint + k*log(1+T/T_ref)*log(1+E)", "coefficients_per_capability": 5,
            "coefficient_order": ["a", "a_prime", "b", "b_prime", "k"], "fits": fits,
            "metrics": metrics, "predictions": augmented, "registered_near_match_residual_slopes": slopes,
            "selection_note": "Conditional exploratory refit after full-data residual diagnostics. Each F is fit on its training fold only, but gate selection used the diagnostic data; these comparisons cannot change the original registered decisions or establish exact fixed-T falsification.",
            "scope": "Residual before/after comparison uses identical registered near-matched endpoints; both descriptors and size/seed targets remain separate."}


def analyze(root=ROOT):
    root = Path(root)
    directories = directory_audit(root)
    points, audits, distributions, protocol, counts, inputs = old.load_data(root)
    assert {"results/v12-distill/"+p.trajectory for p in points} == set(directories["selected"])
    accounting = accounting_rows(points, inputs)
    table_a, pool_pairs = pool_contrasts(points, accounting)
    prior = inputs.read(root / "results/v95-matrix-intervention/summary.json")
    # Part A is completed before any V96 response fits.
    diagnostic = mismatch_audit(prior, points, pool_pairs)
    # Verify prior numbers correspond to these exact source bytes.
    for name, digest in inputs.digests.items():
        if name in prior["input_sha256"]:
            assert digest == prior["input_sha256"][name], f"V95 source changed: {name}"
    equivalence = equivalence_check(prior)
    pairs = evaluation_pairs(points, pool_pairs)
    predictions, fits, means, folds = evaluate(points, pairs)
    metrics = score(predictions, pairs)
    ratios = ratio_diagnostics(table_a, pool_pairs, fits)
    extension = conditional_extension(points, pairs, predictions, ratios)
    ratios["conditional_extension"] = extension
    ratios["interaction_gate"]["refitted"] = extension["refitted"]
    summary = {
        "version": 96, "cpu_only": True, "no_training": True,
        "part_a": {"directory_assertion": directories, "mismatch_resolution": diagnostic,
                   "table": table_a, "domain_share_limitation": MISSING_SHARES,
                   "other_changes": "Pool contents change with size as well as seed; pools are independently sampled, not nested. Actual schedule lengths vary 143–153 optimizer updates (warmup=4); checkpoint update, epoch and LR also vary. These are in every endpoint accounting record. T-only has none of these as inputs, so they do not explain its nonzero predictions, but they limit causal interpretation of measured effects."},
        "part_b": {"constant_note": "A per-capability constant fitted to the response and then differenced is identically zero; it cannot serve as a second independent baseline.",
                   "mean_effect": "Per-capability, per-target arithmetic mean of intervention effects whose BOTH endpoints are on the training fold only. It is not forced to zero. For response scoring the mean is the training checkpoint response. This benchmark directly estimates an effect and is not presented as one fitted response law.",
                   "surface": "Separate same-input second-order surfaces for each descriptor: [t,E,z*t,z*E,t^2,t*E,E^2], t=T/T_ref. Seven OLS coefficients, no intercept; zero at zero budget. Fit only checkpoint responses, then difference the same fitted F.",
                   "mean_fits": means},
        "part_c": {"equivalence": equivalence, "formula": "(a+a_prime*z)*log(1+T/T_ref)+(b+b_prime*z)*log(1+T/D_U)",
                   "coefficient_order": ["a", "a_prime", "b", "b_prime"], "T_ref": T_REF,
                   "descriptors": list(DESCRIPTORS), "descriptor_standardization": "Training-fold checkpoint mean/population std, separately per capability; own initial capability loss is a permitted held-out input, never post-training loss.",
                   "response_definition": "delta = saved capability loss minus the same trajectory's own update-0 loss; negative means loss reduction.",
                   "same_F": "One unweighted per-capability OLS fit on all training checkpoints including zero. F predicts response; F(T_high)-F(T_low) predicts registered about-doubling; F(T2,D_U2,z)-F(T1,D_U1,z) predicts observed pool contrast. Exact doubling and exact-T counterfactual methods use the same coefficients, without intervention fits.",
                   "fits": fits, "verdicts": verdicts(metrics)},
        "part_d": ratios,
        "evaluation": {"primary": old.SPLITS[0], "secondary": old.SPLITS[1],
                       "registered_rule": "The same candidate must have zero-baseline MAE improvement strictly greater than the FULL paired 95% improvement-CI width on BOTH I1 and I2_size under pooled leave-one-student-out. Equality fails; seed holdout cannot rescue failure. Response, seed-only and additional-baseline comparisons are separate diagnostics.",
                       "pairing": {"I1_ratio_window": list(old.HALVING_WINDOW), "I2_max_T_over_min_T": old.MATCHED_BUDGET_RATIO,
                                   "unchanged_V95_I2_size_pairs_asserted": True, "interpolation": False,
                                   "response_scoring": "Positive-budget checkpoints only; zero anchors still used in fitting.",
                                   "size_seed_never_pooled": True},
                       "seed_only_secondary_unavailable": "Each seed41->42 contrast crosses the two leave-one-pool-seed-out folds. Neither fold contains both held-out endpoints, and neither training fold contains a seed contrast. Scoring it with an in-training endpoint would leak; report unavailable, with zero eligible secondary seed-only pairs.",
                       "bootstrap": {"draws": BOOTSTRAPS, "seed": 0, "unit": "whole trajectory", "interval": "95% percentile",
                                     "weights": "One multiplicity for response/I1, product of both trajectory multiplicities for pool pairs; redraw empty draws. Repeated checkpoints stay together. Paired draws across candidates.",
                                     "refit_in_bootstrap": False, "conditioning": "Fixed cross-fitted predictions, as V95; training-fit uncertainty and shared-seed dependence beyond whole trajectories are not included. Two seeds cannot support precise seed-population inference."},
                       "folds": folds, "metrics": metrics, "predictions": predictions,
                       "counts_per_capability": dict(Counter(p.target for p in pairs if p.first.capability == "math"))},
        "provenance": {"trajectory_audit": audits, "protocol": protocol, "parameter_counts": counts,
                       "parameter_dimension_source": "configs/v28_source_metadata.json: dimensions only, no extra trajectory responses",
                       "distributions": distributions, "input_sha256": inputs.digests},
    }
    inputs.verify()
    return summary


def fmt(value):
    a, b = value["ci95"]
    return f"{value['estimate']:.6g} [{a:.6g}, {b:.6g}] (clusters={value['clusters']})"


def shares(endpoint):
    values = endpoint["supervised_token_shares"]
    return "/".join("NA" if values[c] is None else f"{values[c]:.6f}" for c in old.CAPABILITIES)


def part_a_table(summary):
    lines = ["Part A: recorded pool-contrast accounting", "",
             "Checkpoint is ordinal, not a nominal supervised-token target. Shares are math/code/qa; NA means not recorded. Ratio is T2/T1. Every seed remains explicit.", ""]
    for kind in ("size", "seed"):
        lines += [f"Pool {kind} contrasts", "",
                  "| Capability | Student | Checkpoint | Contrast | Seeds | T1 | T2 | T2/T1 | D_U1 | D_U2 | Shares1 m/c/q | Shares2 m/c/q | <=1.10 max/min |",
                  "|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|---|"]
        for r in summary["part_a"]["table"]:
            if r["kind"] != kind:
                continue
            a, b = r["first"], r["second"]
            lines.append(f"| {r['capability']} | {r['student']} | {r['checkpoint']} | {r['contrast']} | {r['seed_first']}->{r['seed_second']} | {a['T']} | {b['T']} | {r['T2_over_T1']:.8f} | {a['D_U']} | {b['D_U']} | {shares(a)} | {shares(b)} | {r['eligible_near_match']} |")
        lines.append("")
    return "\n".join(lines)


def part_d_table(summary):
    lines = ["Part D: measured observed-endpoint ratios (T1 != T2; approximate diagnostic)", "",
             "Exact fixed-T ratio table: unavailable (zero nonzero-budget matched-T pairs). These intervals resample trajectories; clusters<6 are weak, and a single seed dyad gives a degenerate interval. Size rows average the two seed-specific ratios; seed rows never enter that average.", ""]
    for kind in ("size", "seed"):
        lines += [f"Pool {kind} contrasts", "",
                  "| Capability | Student | Contrast | Checkpoint | Measured effect / observed reuse bracket [95% CI] | Eligible near-match pairs / all pairs |",
                  "|---|---|---|---:|---|---|"]
        for r in summary["part_d"]["observed_near_budget_ratio_table"]:
            if r["kind"] == kind:
                lines.append(f"| {r['capability']} | {r['student']} | {r['contrast']} | {r['checkpoint']} | {fmt(r['ratio'])} | {r['near_match_pairs']}/{r['pairs']} |")
        lines.append("")
    return "\n".join(lines)


def render(summary):
    a, b, c, d, ev = (summary[k] for k in ("part_a", "part_b", "part_c", "part_d", "evaluation"))
    lines = ["V96 joint response audit — CPU only, saved losses only", "",
             "Part A comes first: the directory assertion passed (18 matrix2 trajectories selected; all five discarded _matrix_lora_dseed41/42 directories excluded, with no contents read).",
             "", a["mismatch_resolution"]["resolution"], "",
             "V95 used max(T1,T2)/min(T1,T2)<=1.10, not exact equality. Every T-only effect reconstructs as a*[log(1+T2/100000)-log(1+T1/100000)]. At identical T the prediction and its MAE gain over zero are EXACTLY ZERO. No nonzero-budget exactly matched pool pair exists in these 18 trajectories.",
             "", a["domain_share_limitation"], "", a["other_changes"], "",
             "Reused V95 T-only pool-change gains (zero MAE minus T-only MAE; 100% attributable to residual T mismatch):", "",
             "| Split | Capability | Gain [95% CI] | Explained | Fixed-T prediction/gain |", "|---|---|---|---|---|"]
    for r in a["mismatch_resolution"]["reused_V95_metrics"]:
        if r["scope"] == "pooled":
            lines.append(f"| {r['split']} | {r['capability']} | {fmt(r['gain_over_zero'])} | 100% | 0 / 0 |")
    lines += ["", a["mismatch_resolution"]["qualification"], "", part_a_table(summary), "",
              "Part B: independent baselines", "", b["constant_note"], "", b["mean_effect"], "", b["surface"], "",
              "Training-fold mean-effect estimates (each capability and target fitted separately):", "",
              "| Split | Held out | Capability | Target | Mean effect | Training pairs | Training trajectories |", "|---|---|---|---|---:|---:|---:|"]
    for r in b["mean_fits"]:
        lines.append(f"| {r['split']} | {r['fold']} | {r['capability']} | {r['target']} | {r['value']:.8g} | {len(r['training_pair_ids'])} | {len(r['training_trajectories'])} |")
    lines += ["", "Part C: equivalence check before fits", "", c["equivalence"]["algebra"], "",
              "All 14 comparisons (two descriptors x seven V95 candidates) have unequal function-space witness ranks; no earlier candidate is a renamed joint form.", "",
              f"F = {c['formula']}; T_ref=100000 fixed; four coefficients per capability in order a,a_prime,b,b_prime. No intercept, fitted scale, floor, threshold, or extra time constant.",
              "", c["response_definition"], "", c["descriptor_standardization"], "", c["same_F"], "", ev["registered_rule"], "",
              "Joint candidate decisions:", "", "| Capability | Descriptor | Registered I1 + I2_size decision | Also clears both vs mean | Also clears both vs surface |", "|---|---|---|---|---|"]
    for r in c["verdicts"]:
        if r["candidate"].startswith("joint_"):
            lines.append(f"| {r['capability']} | {r['candidate'][6:]} | {r['status']} | {r['both_targets_over_training_mean']} | {r['both_targets_over_same_input_surface']} |")
    lines += ["", "Evaluation: 5000 paired percentile bootstrap draws clustered on whole trajectories. Cluster counts accompany every interval. Refit-in-bootstrap=false, matching V95; intervals condition on cross-fitted predictions. All fold metrics, coefficients, endpoint predictions and training IDs are in summary.json. Response scoring excludes the trivial zero-budget anchors; fitting includes them.",
              "", ev["seed_only_secondary_unavailable"], ""]
    for split in old.SPLITS:
        for descriptor in DESCRIPTORS:
            lines += [f"{split}, descriptor={descriptor}; pooled held-out predictions; targets remain separate", "",
                      "| Capability | Target | Candidate | Pairs | MAE [95% CI] | Bias [95% CI] | Gain vs zero [95% CI] | Gain vs mean [95% CI] | Gain vs surface [95% CI] |",
                      "|---|---|---|---:|---|---|---|---|---|"]
            names = {"zero", "T_only", BASELINE, f"joint_{descriptor}", f"surface_{descriptor}"}
            for r in ev["metrics"]:
                if r["split"] != split or r["scope"] != "pooled" or r["candidate"] not in names:
                    continue
                surface = r["gains"].get(f"surface_{descriptor}")
                lines.append(f"| {r['capability']} | {r['target']} | {r['candidate']} | {r['pairs']} | {fmt(r['mae'])} | {fmt(r['bias'])} | {fmt(r['gains']['zero'])} | {fmt(r['gains'][BASELINE])} | {fmt(surface) if surface else '—'} |")
            lines.append("")
    lines += ["Part D: falsifiable implication and observation limit", "",
              "At fixed student and T, F(pool2)-F(pool1) = (b+b_prime*z)*[log(1+T/D_U2)-log(1+T/D_U1)]. Thus the measured effect divided by that bracket should be flat in budget. A controlled systematic drift would falsify this implication.",
              "", d["ratio_definition"], "", part_d_table(summary), "", d["slope_definition"], "", d["sensitivity"], "",
              "Slopes pooled across students with separate trajectory-dyad intercepts (individual-student slopes also saved in JSON):", "",
              "| Kind | Capability | Contrast | Raw ratio slope [95% CI] | Sign-tail probability | Adjusted residual slope: log parameters [95% CI] | Adjusted residual slope: initial loss [95% CI] |",
              "|---|---|---|---|---:|---|---|"]
    for r in d["slopes"]:
        if r["student"] == "all_students":
            lines.append(f"| {r['kind']} | {r['capability']} | {r['contrast']} | {fmt(r['raw_ratio_slope'])} | {r['raw_ratio_slope']['bootstrap_sign_tail_probability']:.6g} | {fmt(r['adjusted_residual_slopes']['log_parameters'])} | {fmt(r['adjusted_residual_slopes']['initial_loss'])} |")
    lines += ["", d["conclusion"], "", "Conditional interaction gate: " + d["interaction_gate"]["rule"], "",
              d["interaction_gate"]["reason"], ""]
    extension = d["conditional_extension"]
    for cap, evidence in d["interaction_gate"]["evidence_by_capability"].items():
        lines.append(f"Observed-residual gate, {cap}: pooled and eligible-only intervals agree={evidence['pooled_and_eligible_interval_support']}; direction replicates in each seed={evidence['each_seed_same_slope_direction']}.")
    lines.append("")
    if extension["refitted"]:
        lines += [f"The observed-residual gate passed for {', '.join(extension['eligible_capabilities'])}. Refit only {extension['formula']} (five coefficients per capability).",
                  "", extension["selection_note"], "", extension["scope"], "",
                  "| Kind | Capability | Contrast | Descriptor | Before residual slope [95% CI] | After residual slope [95% CI] | Interpretation |",
                  "|---|---|---|---|---|---|---|"]
        for r in extension["registered_near_match_residual_slopes"]:
            lines.append(f"| {r['kind']} | {r['capability']} | {r['contrast']} | {r['descriptor']} | {fmt(r['before'])} | {fmt(r['after'])} | {r['interpretation']} |")
        for descriptor in DESCRIPTORS:
            lines += ["", f"Conditional extension evaluation, descriptor={descriptor}; pooled held-out predictions", "",
                      "| Split | Capability | Target | MAE [95% CI] | Gain vs four-term joint [95% CI] | Gain vs zero [95% CI] | Gain vs training mean [95% CI] |",
                      "|---|---|---|---|---|---|---|"]
            for r in extension["metrics"]:
                if r["scope"] == "pooled" and r["candidate"] == f"interaction_{descriptor}":
                    lines.append(f"| {r['split']} | {r['capability']} | {r['target']} | {fmt(r['mae'])} | {fmt(r['gains'][f'joint_{descriptor}'])} | {fmt(r['gains']['zero'])} | {fmt(r['gains'][BASELINE])} |")
    else:
        lines += ["Conditional interaction refit: not performed. " + extension["reason"]]
    lines += ["",
              "Data limitations are explicit: requested per-domain token shares and the exact fixed-T measured falsification statistic cannot be recovered from the allowed saved accounting. No new training, inference, tokenization, external trajectory, or interpolated measurement was used.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-tables", action="store_true")
    args = parser.parse_args()
    out = ROOT / OUT_REL
    old.require(not out.is_symlink(), "Refusing symlink output directory")
    out.mkdir(exist_ok=True)
    for name in ("summary.json", "summary.md"):
        old.require(not (out/name).is_symlink(), "Refusing symlink output file")
    summary = analyze()
    (out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    (out/"summary.md").write_text(render(summary))
    print(f"Wrote {OUT_REL}/summary.md and summary.json (CPU only).")
    if args.print_tables:
        print(part_a_table(summary))
        print(part_d_table(summary))


if __name__ == "__main__":
    main()
