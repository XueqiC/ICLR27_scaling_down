#!/usr/bin/env python3
"""CPU/NumPy analysis of saved V100 checkpoints; no training or model loading.

Run: OPENBLAS_NUM_THREADS=1 python3 -B analysis/v100_critical_region.py --print-tables
Only writes results/v100-critical-region/{summary.json,summary.md}.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import hashlib
from itertools import combinations, product
import json
import math
from pathlib import Path

import numpy as np

try:
    from analysis import v96_joint_response as joint
except ModuleNotFoundError:
    import v96_joint_response as joint

old = joint.old
ROOT = Path(__file__).resolve().parents[1]
OUT_REL = Path("results/v100-critical-region")
BOOTSTRAPS = 5000
TOLERANCES = (.01, .003)
CRITICAL_STUDENTS = ("gemma3-1b", "gemma3-4b")
CRITICAL_SEEDS = (51, 52)
POOLS = (66, 132, 198, 594)
BANDS = ("25k", "50k", "100k", "200k")
BAND_EDGES = np.sqrt(np.array([25000*50000, 50000*100000, 100000*200000]))
CANDIDATES = (*joint.CANDIDATES, *(f"interaction_{d}" for d in joint.DESCRIPTORS))
LAW_CANDIDATES = tuple(c for c in CANDIDATES if c.startswith(("joint_", "interaction_")))
SCORE_CANDIDATES = (*CANDIDATES, "mean_effect")
SPLIT_STUDENT = "leave_one_student_out"
SPLIT_POOL = "held_out_pool_132"
SPLIT_TWO_POOLS = "held_out_132_and_reference_pool"
RULE = (
    "The SAME candidate must have zero-baseline MAE improvement strictly greater "
    "than the FULL paired 95% improvement-interval width on BOTH I1 and I2_size "
    "under pooled leave-one-student-out. Equality fails. Apply separately at each "
    "requested matching tolerance; other splits cannot rescue failure. Response "
    "and other-baseline scores are diagnostics."
)


def assert_controlled_directories(selected, discarded):
    selected, discarded = list(map(Path, selected)), list(map(Path, discarded))
    assert len(selected) == len(set(selected)) == 22, "Expected 22 distinct trajectories"
    assert not any(p.name.endswith(("_matrix_lora_dseed41", "_matrix_lora_dseed42"))
                   for p in selected), "Excluded _matrix_ launch selected"
    matrix = [p for p in selected if "_matrix2_lora_" in p.name]
    critical = [p for p in selected if "_critical_lora_" in p.name]
    joint.assert_controlled_directories(matrix, discarded)
    expected = {(s, f"gpt-5.6-luna_full_132_critical_lora_dseed{seed}")
                for s, seed in product(CRITICAL_STUDENTS, CRITICAL_SEEDS)}
    assert {(p.parent.name, p.name) for p in critical} == expected and len(critical) == 4, \
        "Expected exactly four critical trajectories"
    expected_matrix = {(s, f"gpt-5.6-luna_full_{u}_matrix2_lora_dseed{seed}")
                       for s, u, seed in product(old.STUDENTS, old.POOLS, old.SEEDS)}
    assert {(p.parent.name, p.name) for p in matrix} == expected_matrix, "Matrix identities changed"
    assert set(selected).isdisjoint(discarded), "Discarded trajectories selected"


def directory_audit(root):
    base = Path(root) / "results/v12-distill"
    directories = [p for p in base.glob("*/*") if p.is_dir()]
    selected = sorted(p for p in directories if p.name.endswith((
        "_matrix2_lora_dseed41", "_matrix2_lora_dseed42",
        "_critical_lora_dseed51", "_critical_lora_dseed52")))
    discarded = sorted(p for p in directories if p.name.endswith((
        "_matrix_lora_dseed41", "_matrix_lora_dseed42")))
    assert_controlled_directories(selected, discarded)
    return {"passed": True, "selected_count": 22, "matrix_count": 18, "critical_count": 4,
            "excluded_count": 5, "excluded_contents_read": False,
            "selected": [str(p.relative_to(root)) for p in selected],
            "excluded": [str(p.relative_to(root)) for p in discarded]}


def critical_token_accounting(log):
    """Reuse the full V95 update/epoch audit; additionally validate all 14 snapshots.

    V95's only grid-dependent assertion expects four trajectory entries. Its
    update/epoch/final-total checks examine the entire unchanged loss_curve.
    Validate the full dense snapshot list against those audited counts here.
    """
    snapshots = log["trajectory"]
    old.require(len(snapshots) == len({r["updates"] for r in snapshots}) == 14,
                "Expected fourteen distinct critical checkpoints")
    counts, du, epoch_end = old.token_accounting({**log, "trajectory": snapshots[:4]})
    previous = 0
    for row in snapshots:
        old.require(row["updates"] > previous, "Nonincreasing dense snapshot updates")
        old.require((row["processed_tokens"], row["completion_tokens_seen"])
                    == counts[row["updates"]], "Dense trajectory accounting mismatch")
        previous = row["updates"]
    old.require(previous == log["updates"], "Final dense snapshot missing")
    return counts, du, epoch_end


def load_data(root=ROOT):
    root = Path(root)
    directories = directory_audit(root)  # Fail before reading any trajectory contents.
    points, audits, distributions, protocol, parameter_counts, inputs = old.load_data(root)
    metadata = inputs.read(root / "configs/v28_source_metadata.json")
    signatures = {}
    for student, seed in product(CRITICAL_STUDENTS, CRITICAL_SEEDS):
        run = root / "results/v12-distill" / student / f"gpt-5.6-luna_full_132_critical_lora_dseed{seed}"
        name = str(run.relative_to(root / "results/v12-distill"))
        log_path = run / "train_log.json"
        log = inputs.read(log_path)
        counts, du, epoch_end = critical_token_accounting(log)
        old.require((log["student"], log["n_per_domain"], log["data_sampling_seed"])
                    == (student, 132, seed), "Critical run identity mismatch")
        old.require({k: log[k] for k in protocol} == protocol, "Critical training protocol changed")
        md = metadata["models"][student]
        old.require(md["hf_id"] == log["resolved_student"], "Critical model changed")
        embedding = md["hidden_size"] * md["vocab_size"]
        n = log["total_parameters"] - log["trainable_parameters"] - embedding
        old.require(n == parameter_counts[student], "Critical parameter count changed")
        documents = [(p, inputs.read(p)) for p in sorted(run.glob("trajectory/update-*/eval.json"))]
        old.require(len(documents) == 15, "Expected update-0 plus fourteen critical evaluations")
        baseline_path, baseline = documents[0]
        old.require(baseline["updates"] == 0, "Critical baseline missing")
        old.require({d["updates"] for _, d in documents} == {0, *(r["updates"] for r in log["trajectory"])},
                    "Dense eval updates disagree with log")
        signature = (du, log["unique_data_pool_tokens"], baseline["data_pool_sha256"])
        old.require(seed not in signatures or signatures[seed] == signature, "Pool differs across students")
        signatures[seed] = signature
        previous = (-1, -1)
        for path, payload in documents:
            step = payload["updates"]
            old.require(path.parent.name == f"update-{step:08d}", "Dense path/update mismatch")
            old.require((payload["student"], payload["n_per_domain"], payload["data_seed"])
                        == (student, 132, seed), "Dense evaluation identity changed")
            expected_keys = ("teacher", "recipe", "training_mode", "learning_rate", "optimizer",
                             "scheduler", "warmup_ratio", "training_seed")
            old.require(all(payload[k] == protocol[k] for k in expected_keys), "Dense eval protocol changed")
            old.require(payload["schedule_tokens"] == 678000 and payload["stop_after_trajectory"],
                        "Dense schedule changed")
            old.require(payload["data_pool_sha256"] == baseline["data_pool_sha256"]
                        and payload["unique_data_pool_tokens"] == log["unique_data_pool_tokens"], "Dense pool changed")
            actual = (payload["processed_tokens"], payload["completion_tokens_seen"])
            old.require(actual == counts[step] and payload["seen_tokens"] == actual[0], "Dense eval/log mismatch")
            old.require(all(a > b for a, b in zip(actual, previous)), "Nonincreasing dense counts")
            previous = actual
            for cap in old.CAPABILITIES:
                distribution = {k: payload[k] for k in (
                    "probe_source", "probe_seed", "probe_half", "n_probe_requested", "loss_definition")}
                distribution.update({k: payload[k][cap] for k in (
                    "measurement_benchmarks", "measurement_samples", "measurement_tokens")})
                old.require(distribution == distributions[cap], "Dense evaluation distribution changed")
                digest = hashlib.sha256(json.dumps(distribution, sort_keys=True).encode()).hexdigest()[:12]
                loss, initial = payload["post_training"][cap], baseline["post_training"][cap]
                old.require(math.isfinite(loss) and math.isfinite(initial), "Nonfinite dense loss")
                old.require(all(p.initial_loss == initial for p in points
                                if p.student == student and p.capability == cap), "Own baseline differs across pools")
                point = old.Point(name, student, 132, seed, cap, digest, step, *actual, du, loss, initial, n,
                                  str(path.relative_to(root)), str(baseline_path.relative_to(root)))
                old.require(math.isclose(payload["delta"][cap], point.delta, abs_tol=1e-10), "Dense delta mismatch")
                points.append(point)
        audits.append({"trajectory": name, "student": student, "U": 132, "seed": seed,
                       "D_U": du, "D_U_source": str(log_path.relative_to(root)),
                       "first_complete_epoch_end_update": epoch_end,
                       "data_pool_sha256": baseline["data_pool_sha256"], "positive_checkpoints": 14,
                       "pool_processed_tokens": log["unique_data_pool_tokens"],
                       "final_processed_tokens": previous[0], "final_completion_tokens": previous[1],
                       "actual_total_updates": log["total_updates_planned"], "actual_warmup_updates": log["warmup_steps"]})
    assert len(points) == 450 and len({p.trajectory for p in points}) == 22
    assert {"results/v12-distill/" + p.trajectory for p in points} == set(directories["selected"])
    assert not any(any(name.startswith(d + "/") for d in directories["excluded"]) for name in inputs.digests)
    return points, {"directory_assertion": directories, "trajectory_audit": audits,
                    "distributions": distributions, "protocol": protocol,
                    "parameter_counts": parameter_counts}, inputs


def mismatch(pair):
    return abs(pair.second.T - pair.first.T) / min(pair.first.T, pair.second.T)


def validate_matched_pair(pair, tolerance):
    a, b = pair.first, pair.second
    assert pair.target == "I2_size", "Slope/ratio requires matched pool pairs"
    assert a.T > 0 and b.T > 0 and a.U < b.U and a.trajectory != b.trajectory, "Invalid pool endpoints"
    assert (a.student, a.capability, a.distribution) == (b.student, b.capability, b.distribution), \
        "Pair crosses student/capability/distribution"
    assert mismatch(pair) <= tolerance, "Unmatched pair exceeds tolerance"
    assert b.D_U > a.D_U, "Larger pool must have larger supervised D_U"


def matched_pairs(points, tolerance):
    """All cross-pool checkpoint combinations, including cross-seed pools.

    Relative mismatch is |T2-T1|/min(T1,T2); endpoints are inclusive. No
    ordinal alignment, nearest-neighbour pruning, nominal tokens or interpolation.
    """
    old.require(0 <= tolerance <= .01, "Matching tolerance must be between zero and 1%")
    assert len({p.id for p in points}) == len(points), "Duplicate checkpoint IDs"
    groups = defaultdict(list)
    for point in points:
        if point.T > 0:
            groups[point.student, point.capability, point.distribution].append(point)
    result = []
    for group in groups.values():
        for a, b in combinations(group, 2):
            if a.U == b.U:
                continue
            a, b = sorted((a, b), key=lambda p: p.U)
            pair = old.Pair("I2_size", a, b)
            if mismatch(pair) <= tolerance:
                validate_matched_pair(pair, tolerance)
                result.append(pair)
    return sorted(result, key=lambda p: p.id)


def budget_band(pair):
    return BANDS[int(np.searchsorted(BAND_EDGES, (pair.first.T + pair.second.T) / 2, side="right"))]


def pool_contrast(pair):
    return f"{pair.first.U}->{pair.second.U}"


def ratio(pair):
    bracket = float(np.log1p(pair.second.E) - np.log1p(pair.first.E))
    assert bracket != 0, "Undefined reuse ratio"
    return pair.actual / bracket


def mismatch_summary(pairs):
    if not pairs:
        return {"pairs": 0, "min_percent": None, "mean_percent": None, "max_percent": None,
                "max_tokens": None, "exact_pairs": 0}
    fractions = np.array([mismatch(p) for p in pairs]) * 100
    return {"pairs": len(pairs), "min_percent": float(fractions.min()), "mean_percent": float(fractions.mean()),
            "max_percent": float(fractions.max()),
            "max_tokens": max(abs(p.second.T-p.first.T) for p in pairs),
            "exact_pairs": sum(p.first.T == p.second.T for p in pairs)}


def bootstrap_weights(pairs, denominator=None):
    """5000 valid whole-trajectory draws; dyadic multiplicities are products."""
    assert pairs, "Cannot bootstrap an empty panel"
    clusters = sorted({c for p in pairs for c in p.clusters})
    denominator = np.ones(len(pairs)) if denominator is None else np.asarray(denominator)
    assert denominator.sum() > 0, "Unidentified statistic"
    rng = np.random.default_rng(0)
    blocks, remaining, rejected = [], BOOTSTRAPS, 0
    while remaining:
        counts = rng.multinomial(len(clusters), np.full(len(clusters), 1/len(clusters)), size=min(500, remaining))
        weights = old.pair_weights(pairs, counts, clusters)
        keep = weights @ denominator > 0
        rejected += int((~keep).sum())
        blocks.append(weights[keep])
        remaining -= int(keep.sum())
    return np.concatenate(blocks), clusters, rejected


def slope_test(pairs, tolerance, values=None):
    """Within-trajectory-dyad slope across budget bands, only matched endpoints.

    One (mean log-budget, mean ratio) observation per dyad/band. Only dyads
    occupying >=2 bands identify a slope; a new dyad in a later band cannot
    create drift. Bootstrap weights are constant within each whole dyad.
    """
    for pair in pairs:
        validate_matched_pair(pair, tolerance)
    assert len({p.id for p in pairs}) == len(pairs), "Duplicate slope pairs"
    assert len({(p.first.capability, p.first.distribution) for p in pairs}) <= 1, "Mixed slope capabilities"
    y = [ratio(p) for p in pairs] if values is None else values
    assert len(y) == len(pairs) and np.isfinite(y).all(), "Invalid slope values"
    groups = defaultdict(lambda: defaultdict(list))
    for i, p in enumerate(pairs):
        groups[p.clusters][budget_band(p)].append(i)
    representatives, numerator, denominator, used = [], [], [], []
    for bands in groups.values():
        if len(bands) < 2:
            continue
        indices = list(bands.values())
        x = np.array([np.mean([np.log1p((pairs[i].first.T+pairs[i].second.T)/(2*joint.T_REF))
                                     for i in ids]) for ids in indices])
        values_band = np.array([np.mean([y[i] for i in ids]) for ids in indices])
        dx, dy = x-x.mean(), values_band-values_band.mean()
        numerator.extend(dx*dy)
        denominator.extend(dx*dx)
        representatives.extend(pairs[ids[0]] for ids in indices)
        used.extend(pairs[i] for ids in indices for i in ids)
    result = {"input_pairs": len(pairs), "used_pairs": len(used), "used_pair_ids": sorted(p.id for p in used),
              "input_pair_ids": sorted(p.id for p in pairs), "bands": sorted({budget_band(p) for p in used}, key=BANDS.index),
              "informative_dyads": len({p.clusters for p in used}), "mismatch": mismatch_summary(used),
              "single_band_pairs_excluded": len(pairs)-len(used)}
    if not used:
        return {**result, "status": "unavailable: no trajectory dyad spans two budget bands", "slope": None}
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    weights, clusters, rejected = bootstrap_weights(representatives, denominator)
    samples = (weights @ numerator) / (weights @ denominator)
    interval = old.interval(numerator.sum()/denominator.sum(), samples, len(clusters))
    interval["bootstrap_sign_tail_probability"] = float(min(1., 2*min(
        (1+np.sum(samples <= 0))/(BOOTSTRAPS+1), (1+np.sum(samples >= 0))/(BOOTSTRAPS+1))))
    interval["excludes_zero"] = interval["ci95"][0] > 0 or interval["ci95"][1] < 0
    return {**result, "status": "computed", "slope": interval, "cluster_ids": clusters,
            "invalid_draws_redrawn": rejected, "valid_draws": BOOTSTRAPS,
            "conditional_degenerate_interval": bool(np.ptp(samples) < 1e-12)}


def counts_table(points, pairs, tolerance):
    rows = []
    for cap, student, (u, v), band in product(old.CAPABILITIES, old.STUDENTS, combinations(POOLS, 2), BANDS):
        group = [p for p in pairs if (p.first.capability, p.first.student, p.first.U, p.second.U, budget_band(p))
                 == (cap, student, u, v, band)]
        available = {p.U for p in points if p.student == student}
        rows.append({"capability": cap, "student": student, "contrast": f"{u}->{v}", "band": band,
                     "tolerance": tolerance, "pools_observed": u in available and v in available,
                     "pairs": len(group), "same_seed_pairs": sum(p.first.seed == p.second.seed for p in group),
                     "critical_pairs": sum(132 in (p.first.U, p.second.U) for p in group),
                     "clusters": len({c for p in group for c in p.clusters}), "mismatch": mismatch_summary(group)})
    return rows


def response_folds(points):
    for student in old.STUDENTS:
        yield SPLIT_STUDENT, student, [p for p in points if p.student != student], [p for p in points if p.student == student]
    yield SPLIT_POOL, "132", [p for p in points if p.U != 132], [p for p in points if p.U == 132]
    # A single held-out pool cannot contain a cross-pool intervention. Hold its
    # reference pool out too for an additional evaluation with BOTH endpoints unseen.
    for reference in old.POOLS:
        yield (SPLIT_TWO_POOLS, f"132+{reference}", [p for p in points if p.U not in (132, reference)],
               [p for p in points if p.U in (132, reference)])


def fit_response(training, candidate):
    # The user explicitly requested the already specified one-term extension;
    # no data-dependent gate or new candidate selection is performed in V100.
    return joint.fit_response(training, candidate, stable_residual=candidate.startswith("interaction_"))


def evaluate(points, panels):
    fits, folds, predictions, mean_fits = [], [], [], []
    i1 = [p for p in old.construct_pairs(points) if p.target == "I1"]
    responses = [old.Pair("response", p, p) for p in points if p.T > 0]
    for split, fold, training, held in response_folds(points):
        train_ids, held_ids = {p.id for p in training}, {p.id for p in held}
        assert train_ids.isdisjoint(held_ids)
        folds.append({"split": split, "fold": fold, "training_trajectories": sorted({p.trajectory for p in training}),
                      "held_out_trajectories": sorted({p.trajectory for p in held}),
                      "I2_note": "Unavailable: only one held-out pool" if split == SPLIT_POOL else "Both endpoints held out"})
        for cap in old.CAPABILITIES:
            models = {c: fit_response([p for p in training if p.capability == cap], c) for c in CANDIDATES}
            fit_ids = {c: f"{split}:{fold}:{cap}:{c}" for c in models}
            for c, model in models.items():
                fits.append({"fit_id": fit_ids[c], "split": split, "fold": fold, "capability": cap, **asdict(model)})
            for tolerance, pool_pairs in panels.items():
                selected = [p for p in (*responses, *i1, *pool_pairs) if p.first.capability == cap
                            and p.first.id in held_ids and p.second.id in held_ids]
                if split == SPLIT_TWO_POOLS:
                    selected = [p for p in selected if p.target == "I2_size" or p.first.U == 132]
                means = {}
                for target in sorted({p.target for p in selected}):
                    if target == "response":
                        train_response = [p for p in training if p.capability == cap and p.T > 0]
                        baseline = {"capability": cap, "target": target,
                                    "value": float(np.mean([p.delta for p in train_response])),
                                    "training_point_ids": [p.id for p in train_response],
                                    "training_trajectories": sorted({p.trajectory for p in train_response})}
                    else:
                        model = joint.fit_mean_effect(training, [*i1, *pool_pairs], cap, target)
                        assert model is not None, "No training-only mean-effect baseline for observed target"
                        baseline = asdict(model)
                    means[target] = baseline["value"]
                    mean_fits.append({"split": split, "fold": fold, "tolerance": tolerance, **baseline})
                for pair in selected:
                    actual = pair.first.delta if pair.target == "response" else pair.actual
                    endpoint = {c: model.response([pair.first, pair.second]).tolist() for c, model in models.items()}
                    predicted = {c: v[0] if pair.target == "response" else v[1]-v[0] for c, v in endpoint.items()}
                    predicted["mean_effect"] = means[pair.target]
                    predictions.append({"split": split, "fold": fold, "tolerance": tolerance,
                                        "capability": cap, "target": pair.target, "pair_id": pair.id,
                                        "actual": actual, "predictions": predicted, "fit_ids": fit_ids,
                                        "endpoint_predictions": endpoint})
    return predictions, fits, folds, {p.id: p for p in (*responses, *i1, *(p for ps in panels.values() for p in ps))}, mean_fits


def score_predictions(predictions, pair_index):
    groups = defaultdict(list)
    for r in predictions:
        for scope in ("pooled", f"held_out={r['fold']}"):
            groups[r["split"], scope, r["tolerance"], r["capability"], r["target"]].append(r)
    result = []
    for (split, scope, tolerance, cap, target), rows in sorted(groups.items()):
        pairs = [pair_index[r["pair_id"]] for r in rows]
        errors = np.array([[r["predictions"][c]-r["actual"] for c in SCORE_CANDIDATES] for r in rows])
        boot = old.bootstrap_statistics(pairs, errors, BOOTSTRAPS)
        for j, c in enumerate(SCORE_CANDIDATES):
            gains = {}
            baselines = ["zero", "mean_effect"]
            if c in LAW_CANDIDATES:
                baselines.append("surface_"+c.split("_", 1)[1])
                if c.startswith("interaction_"):
                    baselines.append(c.replace("interaction_", "joint_", 1))
            for baseline in baselines:
                k = SCORE_CANDIDATES.index(baseline)
                gains[baseline] = old.interval(boot["mae"][k]-boot["mae"][j],
                    boot["mae_draws"][:, k]-boot["mae_draws"][:, j], boot["clusters"])
            result.append({"split": split, "scope": scope, "tolerance": tolerance, "capability": cap,
                           "target": target, "candidate": c, "pairs": len(pairs), "cluster_ids": boot["cluster_ids"],
                           "mae": old.interval(boot["mae"][j], boot["mae_draws"][:, j], boot["clusters"]),
                           "bias": old.interval(boot["bias"][j], boot["bias_draws"][:, j], boot["clusters"]),
                           "gains": gains, "clears_zero_rule": old.clears_reading_rule(gains["zero"]),
                           "empty_draws_redrawn": boot["empty_pair_resamples_redrawn"]})
    return result


def decisions(metrics):
    index = {(r["split"], r["tolerance"], r["capability"], r["candidate"], r["target"]): r
             for r in metrics if r["scope"] == "pooled"}
    rows = []
    for split, tolerance, cap, candidate in product(
            (SPLIT_STUDENT, SPLIT_POOL, SPLIT_TWO_POOLS), TOLERANCES, old.CAPABILITIES, LAW_CANDIDATES):
        targets = {t: index.get((split, tolerance, cap, candidate, t)) for t in ("I1", "I2_size")}
        passed = all(r and r["clears_zero_rule"] for r in targets.values())
        rows.append({"split": split, "tolerance": tolerance, "capability": cap, "candidate": candidate,
                     "status": "met" if passed else joint.FAIL,
                     "target_passes": {t: r["clears_zero_rule"] if r else None for t, r in targets.items()},
                     "unavailable_targets": [t for t, r in targets.items() if r is None],
                     "registered_primary": split == SPLIT_STUDENT,
                     "note": "Missing I2 is lack of identification, not an observed predictive failure."
                     if split == SPLIT_POOL else "Same numerical reading rule; only LOSO is the primary decision."})
    return rows


def fixed_budget_analysis(points, panels, fits):
    fit_index = {}
    for f in fits:
        if f["split"] == SPLIT_STUDENT and f["candidate"].startswith("joint_"):
            fit_index[f["capability"], f["fold"], f["candidate"]] = joint.ResponseFit(
                f["candidate"], f["coefficients"], old.Standardizer(**f["standardizer"]),
                f["rank"], tuple(f["training_point_ids"]))
    all_counts, details, table, slopes = [], [], [], []
    for tolerance, pairs in panels.items():
        all_counts.extend(counts_table(points, pairs, tolerance))
        values = {}
        for p in pairs:
            validate_matched_pair(p, tolerance)
            common_t = (p.first.T+p.second.T)/2
            common = old.Pair("I2_size", replace(p.first, T=common_t), replace(p.second, T=common_t))
            bracket = float(np.log1p(common.second.E)-np.log1p(common.first.E))
            adjusted = {}
            for d in joint.DESCRIPTORS:
                model = fit_index[p.first.capability, p.first.student, f"joint_{d}"]
                correction = float(model.intervention([p])[0]-model.intervention([common])[0])
                adjusted[d] = (p.actual-correction)/bracket
            values[p.id] = {"raw": ratio(p), **adjusted}
            details.append({"tolerance": tolerance, "pair_id": p.id, "capability": p.first.capability,
                            "student": p.first.student, "contrast": pool_contrast(p), "band": budget_band(p),
                            "seed_first": p.first.seed, "seed_second": p.second.seed,
                            "T_first": p.first.T, "T_second": p.second.T, "D_U_first": p.first.D_U,
                            "D_U_second": p.second.D_U, "measured_effect": p.actual,
                            "reuse_bracket": float(np.log1p(p.second.E)-np.log1p(p.first.E)),
                            "common_T": common_t, "common_T_bracket": bracket,
                            "raw_ratio": ratio(p), "model_adjusted_ratio_sensitivity": adjusted,
                            "mismatch_percent": 100*mismatch(p), "trajectory_clusters": list(p.clusters),
                            "first_source": p.first.source, "second_source": p.second.source})
        groups = defaultdict(list)
        for p in pairs:
            groups[p.first.capability, p.first.student, pool_contrast(p), budget_band(p)].append(p)
        order = lambda item: (old.CAPABILITIES.index(item[0][0]), old.STUDENTS.index(item[0][1]),
                              tuple(map(int, item[0][2].split("->"))), BANDS.index(item[0][3]))
        for (cap, student, contrast, band), selected in sorted(groups.items(), key=order):
            weights, clusters, rejected = bootstrap_weights(selected)
            ys = np.array([values[p.id]["raw"] for p in selected])
            table.append({"tolerance": tolerance, "capability": cap, "student": student,
                          "contrast": contrast, "band": band, "pairs": len(selected),
                          "ratio": old.interval(ys.mean(), weights@ys/weights.sum(axis=1), len(clusters)),
                          "cluster_ids": clusters, "mismatch": mismatch_summary(selected),
                          "T_range": [min(min(p.first.T, p.second.T) for p in selected),
                                      max(max(p.first.T, p.second.T) for p in selected)],
                          "empty_draws_redrawn": rejected})
        for cap, student, contrast in product(old.CAPABILITIES, (*old.STUDENTS, "all_students"),
                (*[f"{u}->{v}" for u, v in combinations(POOLS, 2)], "all_contrasts", "critical_contrasts")):
            selected = [p for p in pairs if p.first.capability == cap
                        and (student == "all_students" or p.first.student == student)
                        and (contrast == pool_contrast(p) or contrast == "all_contrasts"
                             or (contrast == "critical_contrasts" and 132 in (p.first.U, p.second.U)))]
            raw = slope_test(selected, tolerance)
            sensitivities = {d: slope_test(selected, tolerance, [values[p.id][d] for p in selected])
                             for d in joint.DESCRIPTORS}
            slopes.append({"tolerance": tolerance, "capability": cap, "student": student, "contrast": contrast,
                           "raw": raw, "model_adjusted_sensitivities": sensitivities})
    return {"counts": all_counts, "endpoint_ratios": details, "ratio_table": table, "slopes": slopes,
            "mismatch_by_tolerance": {str(t): mismatch_summary(ps) for t, ps in panels.items()},
            "exact_nonzero_budget_pairs": len(matched_pairs(points, 0)),
            "ratio_definition": "(delta2-delta1)/(log(1+T2/D_U2)-log(1+T1/D_U1)); orient smaller to larger U. Arithmetic mean of endpoint ratios within each band.",
            "slope_definition": "Within-trajectory-dyad OLS of band-mean ratio against band-mean log(1+mean(T1,T2)/100000); equal weight per occupied dyad/band, dyad intercepts. Drop dyads occupying only one band; they do not identify drift.",
            "sensitivity_definition": "Subtract the LOSO joint model's observed-budget minus common-midpoint-budget intervention from the measured effect, then divide by the common-T bracket. Each descriptor separately; model-dependent sensitivity, never a measured exact-T effect.",
            "interval_note": "5000 whole-trajectory percentile bootstrap draws. Product multiplicities for both endpoints; redraw empty/unidentified draws. Two clusters in a single dyad yield a degenerate conditional interval, not population precision. Sign-tail probability is a bootstrap diagnostic, not a calibrated null-test p-value."}


def conclusions(fixed, verdicts):
    rows = []
    for cap in old.CAPABILITIES:
        selected = [r for r in fixed["slopes"] if r["capability"] == cap and r["student"] == "all_students"
                    and r["contrast"] in ("all_contrasts", "critical_contrasts")]
        for r in selected:
            s = r["raw"]["slope"]
            state = "unavailable" if s is None else ("drift interval excludes zero" if s["excludes_zero"]
                                                      else "drift interval includes zero; flatness not established")
            rows.append({"capability": cap, "tolerance": r["tolerance"], "scope": r["contrast"],
                         "finding": state, "slope": s, "used_pairs": r["raw"]["used_pairs"],
                         "mismatch": r["raw"]["mismatch"]})
    critical_slopes = [r for r in fixed["slopes"] if r["tolerance"] == .01
                       and r["student"] == "all_students" and r["contrast"] == "critical_contrasts"]
    used_ids = {pair_id for r in critical_slopes for pair_id in r["raw"]["used_pair_ids"]}
    informative_seeds = sorted({r["seed_first"] if r["contrast"].startswith("132->") else r["seed_second"]
                               for r in fixed["endpoint_ratios"] if r["pair_id"] in used_ids})
    pooled = [r for r in rows if r["scope"] == "all_contrasts"]
    all_positive = all(r["slope"] and r["slope"]["ci95"][0] > 0 for r in pooled)
    primary = [r for r in verdicts if r["registered_primary"]]
    passing = [r for r in primary if r["status"] == "met"]
    return {"drift": rows,
            "headline": "Pooled ratio drift survives at both requested tolerances for math, code and qa (positive slope intervals)."
            if all_positive else "Drift support varies across capabilities or tolerances; inspect the reported slope intervals.",
            "decision_headline": "Both the four-coefficient joint form and the five-coefficient extension did not meet the pre-registered threshold for every capability and descriptor at both tolerances."
            if not passing else "Some primary candidate/capability panels met the pre-registered threshold; see primary_met.",
            "exact_T_caveat": "No exactly equal nonzero supervised budgets exist across pools. The requested tests remove the former broad tolerance, not all budget mismatch; exact fixed-budget causal falsification remains unobserved.",
            "critical_coverage": "At 0.3%, all U=132 matches occupy only the 200k band. They estimate a high-budget ratio but cannot test its drift. Any tighter-tolerance slope is identified by matrix pools, not the new pool.",
            "critical_informative_seeds_at_1pct": informative_seeds,
            "seed_replication_limit": f"The 1% U=132 slope is identified only by critical seed(s) {informative_seeds}; the other critical seed has no repeated matched trajectory dyad spanning bands. This does not provide two-seed replication of the critical-region slope.",
            "primary_met": passing}


def analyze(root=ROOT):
    points, provenance, inputs = load_data(root)
    panels = {t: matched_pairs(points, t) for t in TOLERANCES}
    assert {p.id for p in panels[.003]}.issubset({p.id for p in panels[.01]})
    predictions, fits, folds, pair_index, mean_fits = evaluate(points, panels)
    metrics = score_predictions(predictions, pair_index)
    verdicts = decisions(metrics)
    fixed = fixed_budget_analysis(points, panels, fits)
    full_fits = [{"capability": cap, **asdict(fit_response([p for p in points if p.capability == cap], c))}
                 for cap, c in product(old.CAPABILITIES, LAW_CANDIDATES)]
    provenance["accounting"] = list(joint.accounting_rows(points, inputs).values())
    provenance["analysis_sha256"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in (Path(__file__), Path(joint.__file__), Path(old.__file__))}
    provenance["input_sha256"] = inputs.digests
    inputs.verify()
    return {"version": 100, "cpu_only": True, "no_gpu": True, "no_training": True,
            "conclusions": conclusions(fixed, verdicts), "provenance": provenance,
            "protocol": {"reading_rule": RULE, "source_registration": "analysis/v95_matrix_intervention.py and analysis/v96_joint_response.py",
                         "T_ref": joint.T_REF, "tolerances": list(TOLERANCES),
                         "bands": {"labels": list(BANDS), "internal_edges": BAND_EDGES.tolist(),
                                   "definition": "Nearest registered nominal 25k/50k/100k/200k budget in log space, using arithmetic mean of actual endpoint T. Half-open bands; outer edges 0 and infinity. Defined without losses."},
                         "matching": "All different-U endpoints within the same student/capability/evaluation distribution; |T2-T1|/min(T1,T2)<=tolerance; all seed combinations. Seed labels 41/51 and 42/52 are not paired or equated. No interpolation or ordinal matching.",
                         "I1": "Unchanged observed budget ratio window [1.7,2.3], same whole trajectory, all eligible positive checkpoint pairs.",
                         "bootstrap": {"draws": BOOTSTRAPS, "seed": 0, "unit": "whole trajectory", "ci": "95% percentile",
                                       "weights": "response/I1: one trajectory multiplicity; cross-pool: product of both. Paired draws across candidates.",
                                       "refit_in_bootstrap": False,
                                       "conditioning": "Fixed cross-fitted predictions, as V95/V96. Does not include training-fit uncertainty or shared pool-seed dependence across students."},
                         "joint": "(a+a_prime*z)*x+(b+b_prime*z)*y; x=log(1+T/100000), y=log(1+T/D_U)",
                         "interaction": "joint + k*x*y; exactly one previously specified extra coefficient; explicitly requested refit, no V100 gate selection",
                         "descriptors": list(joint.DESCRIPTORS),
                         "fitting": "Unweighted OLS of checkpoint delta including own update-0 anchors. Training-fold descriptor mean/population SD. Four or five coefficients per capability/descriptor; the same F generates response, budget difference and pool difference. No intervention fits. Dense trajectories intentionally supply more checkpoint rows, as registered.",
                         "mean_effect_baseline": "Separate per-capability/per-target arithmetic mean using only training-fold endpoints. Response mean uses positive training checkpoints; intervention means require BOTH endpoints in training. This existing diagnostic baseline fits effects directly and is not a response law or a second zero baseline.",
                         "holdout_132": "Train on all 18 matrix trajectories, score held-out U=132 responses and I1. I2 unavailable because a single held-out pool has no cross-pool pair. Additional 132+reference-pool folds exclude BOTH pools from fitting and score I2 with both endpoints held out; I1/response in those folds score U=132 only. Repeated U=132 predictions across reference folds retain their original trajectory clusters.",
                         "domain_share_limitation": joint.MISSING_SHARES,
                         "other_limitations": "Pools are independently sampled rather than nested. Seed and pool contents change with U; actual optimizer update/epoch/LR at similar T differ despite the common protocol. Only two pool seeds per rung. No capability-specific training-token shares are inferred."},
            "fixed_budget": fixed,
            "evaluation": {"folds": folds, "fits": fits, "full_combined_fits": full_fits,
                           "mean_effect_fits": mean_fits, "metrics": metrics, "decisions": verdicts, "predictions": predictions},
            "raw_response_curves": old.response_curves(points)}


def fmt(interval):
    return joint.fmt(interval) if interval else "unavailable"


def count_lines(summary):
    lines = ["Matched-pair counts (identical for math, code and qa; each cell is per capability)", "",
             "| Tolerance | Student | Pool contrast | 25k | 50k | 100k | 200k | Total |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    groups = defaultdict(dict)
    for r in summary["fixed_budget"]["counts"]:
        if r["capability"] == "math":
            groups[r["tolerance"], r["student"], r["contrast"]][r["band"]] = r["pairs"]
    for (tol, student, contrast), bands in sorted(groups.items(), key=lambda kv: (
            -kv[0][0], old.STUDENTS.index(kv[0][1]), tuple(map(int, kv[0][2].split("->"))))):
        ns = [bands[b] for b in BANDS]
        lines.append(f"| {100*tol:g}% | {student} | {contrast} | " + " | ".join(map(str, [*ns, sum(ns)])) + " |")
    lines += ["", "U=132 is unobserved for gemma3-270m; its zero counts are structural. All zero band cells are retained in JSON, by capability, with cluster counts and mismatch ranges."]
    return lines


def ratio_lines(summary):
    lines = ["Fixed-budget ratio table (actual budgets within tolerance)", "",
             "| Tol. | Capability | Student | U contrast | Band | Pairs | Ratio [95% CI], clusters | Residual mismatch % min–max |",
             "|---|---|---|---|---|---:|---|---|"]
    for r in summary["fixed_budget"]["ratio_table"]:
        m = r["mismatch"]
        lines.append(f"| {100*r['tolerance']:g}% | {r['capability']} | {r['student']} | {r['contrast']} | {r['band']} | "
                     f"{r['pairs']} | {fmt(r['ratio'])} | {m['min_percent']:.6g}–{m['max_percent']:.6g} |")
    return lines


def render(summary):
    fixed, ev, protocol = summary["fixed_budget"], summary["evaluation"], summary["protocol"]
    lines = ["# V100 critical region", "", summary["conclusions"]["headline"], "",
             summary["conclusions"]["decision_headline"], "",
             "CPU only; saved measurements only; 22 trajectories (18 matrix2 + four U=132), 150 checkpoints including anchors, 450 capability responses. The five discarded matrix launches are excluded by assertion before loading.", "",
             summary["conclusions"]["exact_T_caveat"], "", summary["conclusions"]["critical_coverage"], "",
             "At 0.3%, the pooled slope uses six trajectories in the 66→198 contrast, covering only the 25k and 50k bands. Its residual differences are 22–64 supervised tokens (0.043529%–0.252008%). At 1%, the pooled slope uses 19 trajectories. The new-pool-only slope at 1% excludes zero for QA; math and code intervals include zero.", "",
             summary["conclusions"]["seed_replication_limit"], "",
             protocol["reading_rule"], "", "## Matched budgets", "", protocol["matching"], "",
             "Bands use log-space midpoints of the registered 25k, 50k, 100k, 200k supervised budgets: "
             + ", ".join(f"{x:.3f}" for x in BAND_EDGES) + " tokens. Pair band uses mean actual T; no outcome-based binning.", ""]
    lines += count_lines(summary)
    lines += ["", "The matrix alone supplies 15 pairs per student/capability at 1% and six at 0.3% when all seed combinations are allowed. U=132 adds 12 and three, respectively, for each of the 1B and 4B students. These are additional close matches; neither cohort has exactly equal nonzero cross-pool budgets."]
    lines += ["", "| Tolerance | Pairs across capabilities | Minimum mismatch % | Mean mismatch % | Maximum mismatch % | Maximum token difference | Exact pairs |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for tol, m in fixed["mismatch_by_tolerance"].items():
        lines.append(f"| {100*float(tol):g}% | {m['pairs']} | {m['min_percent']:.6g} | {m['mean_percent']:.6g} | {m['max_percent']:.6g} | {m['max_tokens']} | {m['exact_pairs']} |")
    lines += ["", "## Fixed-budget falsification", "", fixed["ratio_definition"], "", fixed["interval_note"], ""]
    lines += ratio_lines(summary)
    lines += ["", fixed["slope_definition"], "", fixed["sensitivity_definition"], "",
              "Slope support counts below refer only to dyads spanning at least two bands. A single-band ratio is never converted into slope evidence. Positive slope means the ratio rises with budget.", "",
              "| Tol. | Capability | Student | Contrast | Used/input pairs | Dyads | Raw slope [95% CI], clusters | Sign-tail probability | Adjusted: log N | Adjusted: initial loss | Max residual mismatch % |",
              "|---|---|---|---|---:|---:|---|---|---|---|---:|"]
    for r in fixed["slopes"]:
        raw = r["raw"]
        if not raw["input_pairs"] and r["student"] != "all_students":
            continue
        s = raw["slope"]
        probability = f"{s['bootstrap_sign_tail_probability']:.6g}" if s else "NA"
        sensitivities = [fmt(r["model_adjusted_sensitivities"][d]["slope"]) for d in joint.DESCRIPTORS]
        max_mismatch = raw["mismatch"]["max_percent"]
        lines.append(f"| {100*r['tolerance']:g}% | {r['capability']} | {r['student']} | {r['contrast']} | "
                     f"{raw['used_pairs']}/{raw['input_pairs']} | {raw['informative_dyads']} | {fmt(s)} | {probability} | "
                     + " | ".join(sensitivities) + f" | {max_mismatch if max_mismatch is not None else 'NA'} |")
    lines += ["", "Pooled drift findings (capabilities kept separate):", ""]
    for r in summary["conclusions"]["drift"]:
        lines.append(f"- {r['capability']}, {100*r['tolerance']:g}%, {r['scope']}: {r['finding']}; {fmt(r['slope'])}.")
    lines += ["", "## Joint and interaction refits", "", protocol["joint"], "", protocol["interaction"], "",
              protocol["fitting"], "", protocol["holdout_132"], "",
              "The fixed comparison set includes zero, T-only, the training-only mean-effect baseline, and the two existing same-input seven-coefficient surfaces. The registered decision uses zero. Mean-effect, surface and interaction-over-joint gains, per-student scores, exact fit coefficients, and every law prediction with its fit ID are in summary.json.", "",
              "| Split | Tol. | Capability | Candidate | I1 clears | I2 clears | Decision |",
              "|---|---|---|---|---|---|---|"]
    for r in ev["decisions"]:
        lines.append(f"| {r['split']} | {100*r['tolerance']:g}% | {r['capability']} | {r['candidate']} | "
                     f"{r['target_passes']['I1']} | {r['target_passes']['I2_size']} | {r['status']} |")
    lines += ["", "Single-pool I2 is unavailable; its decision cannot be met because the second target is unobserved. This is not a measured negative result. Pool holdouts cannot rescue a failed primary LOSO decision.", "",
              "Pooled MAE and paired zero-baseline improvement (all intervals include trajectory cluster counts):", "",
              "| Split | Tol. | Capability | Target | Candidate | N | MAE [95% CI] | Zero MAE minus candidate MAE [95% CI] | Full gain CI width |",
              "|---|---|---|---|---|---:|---|---|---:|"]
    for r in ev["metrics"]:
        if r["scope"] == "pooled" and r["candidate"] in LAW_CANDIDATES:
            gain = r["gains"]["zero"]
            lines.append(f"| {r['split']} | {100*r['tolerance']:g}% | {r['capability']} | {r['target']} | {r['candidate']} | "
                         f"{r['pairs']} | {fmt(r['mae'])} | {fmt(gain)} | {gain['width']:.6g} |")
    lines += ["", "## Raw response curves", "", "Every cell is **(actual supervised tokens, delta)**; delta is loss minus own update-0 loss (negative is improvement). Pools appear side by side. Rows are checkpoint ordinals for display only, never budget matches. Matrix seeds 41/42 and critical seeds 51/52 are independent and are shown together solely to keep all raw curves visible.", ""]
    curves = {(r["capability"], r["student"], r["U"], r["seed"]): r for r in summary["raw_response_curves"]}
    for cap, student, slot in product(old.CAPABILITIES, old.STUDENTS, (0, 1)):
        seeds = [old.SEEDS[slot], CRITICAL_SEEDS[slot], old.SEEDS[slot], old.SEEDS[slot]]
        cols = [curves.get((cap, student, u, seed)) for u, seed in zip(POOLS, seeds)]
        lines += [f"### {cap} / {student} / matrix seed {old.SEEDS[slot]}, critical seed {CRITICAL_SEEDS[slot]}", "",
                  "| Checkpoint | " + " | ".join(f"U={u}, seed={seed}" for u, seed in zip(POOLS, seeds)) + " |",
                  "|---:|---|---|---|---|"]
        for i in range(max(len(c["points"]) for c in cols if c)):
            cells = [f"({c['points'][i]['T']}, {c['points'][i]['delta']:+.7f})" if c and i < len(c["points"]) else "—" for c in cols]
            lines.append(f"| {i} | " + " | ".join(cells) + " |")
    lines += ["", "## Accounting and interpretation", "", protocol["domain_share_limitation"], "",
              protocol["other_limitations"], "", "Bootstrap: " + json.dumps(protocol["bootstrap"], sort_keys=True), "",
              "All logged supervised increments and complete pool passes were reconciled. D_U uses completion tokens in a verified complete distinct epoch; processed tokens and per-domain example proportions are never substituted. Source hashes, per-endpoint accounting (update, epoch, LR, schedule), pool hashes and raw values are in summary.json.", "",
              "Reproduce: `OPENBLAS_NUM_THREADS=1 python3 -B analysis/v100_critical_region.py --print-tables`. Tests: `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python3 -B -m pytest -q -p no:cacheprovider tests/test_v100_critical_region.py`.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-tables", action="store_true")
    args = parser.parse_args()
    summary = analyze()
    out = ROOT / OUT_REL
    assert not out.is_symlink(), "Refuse symlink output directory"
    out.mkdir(parents=True, exist_ok=True)
    for name, contents in (("summary.json", json.dumps(summary, indent=2, allow_nan=False)+"\n"),
                           ("summary.md", render(summary))):
        path = out / name
        assert not path.is_symlink(), "Refuse symlink output file"
        path.write_text(contents)
    if args.print_tables:
        print("\n".join([*count_lines(summary), "", *ratio_lines(summary)]))
    print(f"Wrote {out}/summary.{{json,md}}")


if __name__ == "__main__":
    main()
