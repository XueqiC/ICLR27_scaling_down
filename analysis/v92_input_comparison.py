#!/usr/bin/env python3
"""V92: CPU-only, development-only comparison of pre-compression input budgets.

Run: OPENBLAS_NUM_THREADS=1 python3 -B analysis/v92_input_comparison.py
Only the two summaries in results/v92-input-comparison/ are written. No model,
measurement, accelerator, network, or frozen prediction artifact is loaded.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import json
import os
from pathlib import Path

for _thread_key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_key] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("results/v92-input-comparison")
CAPS = ("math", "code", "qa")
ARMS = ("pruning", "grouped_quantization", "per_channel_quantization")
SIZES = ("160m", "410m", "1.4b")
STEPS = (16000, 64000, 143000)
BUDGETS = ("K0", "dense_anchor", "dense_statistics")
FORMS = ("zero", "constant", "median_curve", "ols", "ridge", "delivered")
LINEAR_FORMS = ("ols", "ridge")
ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
BOOTSTRAPS = 20000
SEED = 9201
CONFIG_FIELDS = {"pruning": ("density",), "grouped_quantization": ("bits", "group_size"),
                 "per_channel_quantization": ("bits",)}
ORIGINS = {"N0": "source_metadata", "D0": "source_metadata",
           "density": "compression_configuration", "bits": "compression_configuration",
           "group_size": "compression_configuration", "L0": "dense_anchor",
           "B": "dense_descriptor", "V": "dense_descriptor", "W": "dense_descriptor"}
FILES = {"pruning": ("v6-capability-geometry", "prune_losses.json", "1.0"),
         "grouped_quantization": ("v54-quant-group", "quant_group_losses.json", "dense"),
         "per_channel_quantization": ("v10-quantization", "quant_losses.json", "dense")}
COMMON = {"pruning": ("0.6", "0.7", "0.8", "0.9"),
          "grouped_quantization": tuple(f"b{b}_g{g}" for b in (3, 4, 5) for g in (64, 128, 256)),
          "per_channel_quantization": ("3", "4", "6", "8")}
DISTRIBUTIONS = dict(zip(CAPS, ("MATH-500", "MBPP", "2WikiMultihopQA")))


@dataclass(frozen=True)
class Feature:
    value: float
    origin: str


@dataclass(frozen=True)
class Query:
    row_id: str
    state: str
    size: str
    capability: str
    arm: str
    config: str
    inputs: dict[str, Feature]

    @property
    def strength(self):
        return self.inputs["density" if self.arm == "pruning" else "bits"].value


@dataclass(frozen=True)
class Observation:
    query: Query
    target: float


def fields(arm, budget):
    if budget not in BUDGETS or arm not in ARMS:
        raise ValueError("Unknown budget or arm")
    return ("N0", "D0", *CONFIG_FIELDS[arm],
            *(("L0",) if budget != "K0" else ()),
            *(("B", "V", "W") if budget == "dense_statistics" else ()))


def validate_inputs(values, expected):
    """Closed allowlist plus provenance: a compressed value cannot be renamed L0/B.

    This verifies supplied provenance, not the truth of arbitrary external data.
    The production loader establishes provenance from fixed dense-only paths/keys.
    """
    if set(values) != set(expected):
        raise ValueError(f"Budget separation: expected {tuple(expected)}, got {tuple(values)}")
    for key, feature in values.items():
        if not isinstance(feature, Feature) or feature.origin != ORIGINS[key]:
            raise ValueError(f"Pre-compression provenance required for {key}")
        if not np.isfinite(feature.value):
            raise ValueError(f"Nonfinite input {key}")


def budget_inputs(query, budget):
    validate_inputs(query.inputs, fields(query.arm, "dense_statistics"))
    result = {key: query.inputs[key] for key in fields(query.arm, budget)}
    validate_inputs(result, fields(query.arm, budget))
    return result


def design(queries, budget):
    # Raw N0, D0 and configuration coordinates; no logs, powers or interactions.
    return np.asarray([[f.value for f in budget_inputs(q, budget).values()] for q in queries])


def n0(size):
    h, m, layers = {"160m": (768, 3072, 12), "410m": (1024, 4096, 24),
                    "1.4b": (2048, 8192, 24)}[size]
    return layers * (4 * h * h + 2 * h * m)


def load_data(root=ROOT):
    """Exact state/config allowlists; descriptors required, never imputed.

    L0 is the arm-local true dense reference. The descriptor L is only an audit;
    using it as another predictor would silently enlarge the declared budget.
    """
    root = Path(root)
    hashes, descriptors, dense_audit, missing, excluded = {}, {}, [], [], []

    def read(relative):
        raw = (root / relative).read_bytes()
        hashes[str(relative)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    for size, step, cap in itertools.product(SIZES, STEPS, CAPS):
        state = f"pythia-{size}--step{step}"
        path = Path("results/v91-dense-stats") / state / cap / "descriptor_bv.json"
        data = read(path)
        if (data["model_id"], data["model_revision"]) != (f"EleutherAI/pythia-{size}", f"step{step}"):
            raise ValueError(f"Descriptor source mismatch: {path}")
        if set(data["aggregates"]) != {cap} or set(data["aggregates"][cap]) != {DISTRIBUTIONS[cap]}:
            raise ValueError(f"Descriptor capability/distribution mismatch: {path}")
        agg = data["aggregates"][cap][DISTRIBUTIONS[cap]]
        if agg["n_scored_samples"] <= 0 or agg["scored_token_count"] <= 0:
            raise ValueError(f"Empty dense descriptor: {path}")
        descriptors[state, cap] = (agg, data)

    rows, extra = [], []
    for arm, size, step in itertools.product(ARMS, SIZES, STEPS):
        state = f"pythia-{size}--step{step}"
        directory, filename, dense_key = FILES[arm]
        path = Path("results") / directory / state / filename
        if not (root / path).is_file():
            if arm == "pruning":
                raise ValueError(f"Required pruning panel missing: {path}")
            missing.append({"arm": arm, "state": state, "path": str(path)})
            continue
        table = read(path)
        if any(key not in table for key in (dense_key, *COMMON[arm])):
            raise ValueError(f"Incomplete common configuration panel: {path}")
        keys = list(COMMON[arm])
        if arm == "pruning":
            keys += [k for k in table if not k.startswith("_") and k not in (dense_key, *keys)]
        else:
            excluded += [{"path": str(path), "config": k,
                          "reason": "Outside common development grid; includes prior confirmation cells"}
                         for k in table if not k.startswith("_") and k not in (dense_key, *keys)]
        for cap in CAPS:
            anchor = float(table[dense_key][cap])
            agg, desc = descriptors[state, cap]
            dense_audit.append({"arm": arm, "state": state, "capability": cap,
                                "arm_L0": anchor, "descriptor_L": agg["L"],
                                "difference": agg["L"] - anchor,
                                "descriptor_samples": agg["n_scored_samples"],
                                "descriptor_tokens": agg["scored_token_count"],
                                "probe_sha256": desc["input_hashes"]["probes_sha256"],
                                "resolved_revision": desc["resolved_model_revision"]})
            for config in keys:
                if arm == "pruning":
                    compression = {"density": float(config)}
                    if not 0 < compression["density"] < 1:
                        raise ValueError("Invalid pruning density")
                elif arm == "grouped_quantization":
                    b, g = config.split("_g")
                    compression = {"bits": float(b[1:]), "group_size": float(g)}
                else:
                    compression = {"bits": float(config)}
                values = {"N0": n0(size), "D0": step * 2097152, **compression,
                          "L0": anchor, **{key: float(agg[key]) for key in ("B", "V", "W")}}
                query = Query(f"{arm}|{state}|{cap}|{config}", state, size, cap, arm, config,
                              {key: Feature(float(value), ORIGINS[key]) for key, value in values.items()})
                budget_inputs(query, "dense_statistics")
                target = float(table[config][cap]) - anchor
                if not np.isfinite(target) or anchor < 0:
                    raise ValueError("Invalid response or dense anchor")
                (rows if config in COMMON[arm] else extra).append(Observation(query, target))
    return rows, extra, {"input_sha256": hashes, "missing_states": missing,
                         "excluded_configurations": excluded, "dense_anchor_audit": dense_audit}


@dataclass
class Standardizer:
    center: np.ndarray
    scale: np.ndarray
    training_ids: tuple

    @classmethod
    def fit(cls, queries, budget):
        raw = design(queries, budget)
        center, scale = raw.mean(axis=0), raw.std(axis=0)
        scale = np.where(scale > 1e-14 * np.maximum(1, abs(center)), scale, 1.0)
        return cls(center, scale, tuple(q.row_id for q in queries))

    def transform(self, queries, budget):
        raw = design(queries, budget)
        return np.column_stack((np.ones(len(queries)), (raw - self.center) / self.scale))

    def audit(self):
        return {"training_ids": self.training_ids, "center": self.center.tolist(),
                "scale": self.scale.tolist()}


def linear_fit(x, y, alpha):
    if alpha:
        penalty = np.eye(x.shape[1])[1:] * np.sqrt(len(y) * alpha)
        x, y = np.vstack((x, penalty)), np.r_[y, np.zeros(len(penalty))]
    return np.linalg.lstsq(x, y, rcond=None)[0]


def state_mean(values, queries):
    buckets = defaultdict(list)
    for value, query in zip(values, queries):
        buckets[query.state].append(float(value))
    return float(np.mean([np.mean(v) for v in buckets.values()]))


def choose_penalty(training, budget, group="state"):
    """Inner grouped CV refits its own scaler; outer test queries are not accepted."""
    queries = [r.query for r in training]
    groups = sorted({getattr(q, group) for q in queries})
    if len(groups) < 2:
        raise ValueError("Penalty selection needs at least two inner groups")
    errors, validation_queries, audit = [[] for _ in ALPHAS], [], []
    for held in groups:
        tr = [r for r in training if getattr(r.query, group) != held]
        va = [r for r in training if getattr(r.query, group) == held]
        tq, vq = [r.query for r in tr], [r.query for r in va]
        scaler = Standardizer.fit(tq, budget)
        x, xv = scaler.transform(tq, budget), scaler.transform(vq, budget)
        y, yv = np.array([r.target for r in tr]), np.array([r.target for r in va])
        for i, alpha in enumerate(ALPHAS):
            errors[i].extend(abs(xv @ linear_fit(x, y, alpha) - yv))
        validation_queries.extend(vq)
        audit.append({"held_out": held, "validation_ids": [q.row_id for q in vq],
                      "standardizer": scaler.audit()})
    scores = [state_mean(error, validation_queries) for error in errors]
    # Exact ties prefer stronger shrinkage; no outer-fold result selects anything.
    best = min(range(len(ALPHAS)), key=lambda i: (scores[i], -ALPHAS[i]))
    return ALPHAS[best], {"group": group, "alpha_grid": ALPHAS, "mae": scores, "folds": audit}


def fit(training, budget, form, inner_group="state"):
    if not training or form not in FORMS:
        raise ValueError("Empty training data or unknown candidate")
    queries, y = [r.query for r in training], np.array([r.target for r in training])
    if len({(q.arm, q.capability) for q in queries}) != 1:
        raise ValueError("Fit exactly one arm and capability")
    for query in queries:
        budget_inputs(query, budget)
    model = {"budget": budget, "form": form, "arm": queries[0].arm,
             "training_ids": [q.row_id for q in queries]}
    if form in ("zero", "constant"):
        model["value"] = 0.0 if form == "zero" else float(y.mean())
    elif form in ("median_curve", "delivered"):
        # Equal source weight at each configuration; no per-source amplitude.
        cells = defaultdict(lambda: defaultdict(list))
        for r in training:
            cells[r.query.config][r.query.state].append(r.target)
        model["anchors"] = {c: float(np.median([np.mean(v) for v in states.values()]))
                            for c, states in cells.items()}
    else:
        alpha, selection = choose_penalty(training, budget, inner_group) if form == "ridge" else (0.0, None)
        scaler = Standardizer.fit(queries, budget)
        x = scaler.transform(queries, budget)
        model.update(alpha=alpha, selection=selection, standardizer=scaler.audit(),
                     coefficients=linear_fit(x, y, alpha).tolist(),
                     design_rank=int(np.linalg.matrix_rank(x)), design_columns=x.shape[1])
    return model


def curve_value(anchors, strength, extrapolate=True):
    xs = sorted(anchors)
    if strength in anchors:
        return anchors[strength]
    if len(xs) < 2 or (not extrapolate and not xs[0] <= strength <= xs[-1]):
        return None
    i = int(np.clip(np.searchsorted(xs, strength) - 1, 0, len(xs) - 2))
    lo, hi = xs[i:i + 2]
    return anchors[lo] + (anchors[hi] - anchors[lo]) * (strength - lo) / (hi - lo)


def predict(model, queries):
    """Prediction takes Query objects with no compressed outcome or diagnostic."""
    form, budget = model["form"], model["budget"]
    for q in queries:
        budget_inputs(q, budget)
    if form in LINEAR_FORMS:
        audit = model["standardizer"]
        scaler = Standardizer(np.array(audit["center"]), np.array(audit["scale"]), tuple(audit["training_ids"]))
        return (scaler.transform(queries, budget) @ model["coefficients"]).tolist()
    if form in ("zero", "constant"):
        return [model["value"]] * len(queries)
    values = []
    for q in queries:
        anchors = model["anchors"]
        # The locked *new-source* delivered branch (final_rule.py) is a median.
        # Preserve its domain: prune d in [.6,.9], channel seen bit, grouped no
        # bit extrapolation. An undefined delivered prediction stays unavailable.
        if form == "delivered" and q.arm == "pruning" and not .6 <= q.strength <= .9:
            values.append(None)
        elif q.config in anchors:
            values.append(anchors[q.config])
        elif q.arm == "per_channel_quantization" and form == "delivered":
            values.append(None)
        elif q.arm == "grouped_quantization":
            group = int(q.inputs["group_size"].value)
            selected = {float(np.log2(2 ** (int(c.split("_")[0][1:]) - 1) - 1)): value
                        for c, value in anchors.items() if c.endswith(f"_g{group}")}
            x = float(np.log2(2 ** (int(q.strength) - 1) - 1))
            values.append(curve_value(selected, x, extrapolate=form != "delivered"))
        else:
            values.append(curve_value({float(c): v for c, v in anchors.items()}, q.strength))
    return values


def make_folds(rows, scheme, extra=()):
    """All test strengths are removed globally in both strength experiments."""
    if scheme in ("leave_one_source_state_out", "leave_one_size_out"):
        group = "state" if scheme == "leave_one_source_state_out" else "size"
        for held in sorted({getattr(r.query, group) for r in rows}):
            yield str(held), [r for r in rows if getattr(r.query, group) != held], [
                r for r in rows if getattr(r.query, group) == held]
    elif scheme in ("unseen_strength_same_sources", "unseen_strength_new_source"):
        for strength in sorted({r.query.strength for r in rows}):
            held_states = sorted({r.query.state for r in rows}) if scheme.endswith("new_source") else [None]
            for state in held_states:
                train = [r for r in rows if r.query.strength != strength and r.query.state != state]
                test = [r for r in rows if r.query.strength == strength and (state is None or r.query.state == state)]
                yield f"strength={strength:g};state={state}", train, test
    elif scheme == "extra_pruning_strength_new_source":
        for state in sorted({r.query.state for r in extra}):
            yield state, [r for r in rows if r.query.state != state], [r for r in extra if r.query.state == state]
    else:
        raise ValueError("Unknown evaluation scheme")


@lru_cache(maxsize=None)
def bootstrap_weights(n):
    return np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n), size=BOOTSTRAPS) / n


def clustered(values, queries):
    by_state = defaultdict(list)
    for value, query in zip(values, queries):
        by_state[query.state].append(float(value))
    means = {state: float(np.mean(v)) for state, v in sorted(by_state.items())}
    n = len(means)
    if not n:
        return {"estimate": None, "ci95": None, "interval_width": None, "n_clusters": 0,
                "n_rows": 0, "source_means": {}}
    ci = np.quantile(bootstrap_weights(n) @ list(means.values()), [.025, .975]).tolist() if n > 1 else None
    return {"estimate": float(np.mean(list(means.values()))), "ci95": ci,
            "interval_width": ci[1] - ci[0] if ci else None, "n_clusters": n,
            "n_rows": len(values), "source_means": means}


def paired(rows, lower, upper):
    valid = [i for i in range(len(rows)) if lower[i] is not None and upper[i] is not None]
    values = [abs(rows[i].target - lower[i]) - abs(rows[i].target - upper[i]) for i in valid]
    return clustered(values, [rows[i].query for i in valid])


def reading_rule(effect, *, scheme, lower_form, upper_form, lower_budget, upper_budget):
    """Strictly greater than FULL upper-minus-lower CI width, not half-width."""
    comparable = (scheme == "leave_one_source_state_out" and lower_form == upper_form
                  and lower_budget == "dense_anchor" and upper_budget == "dense_statistics")
    width, gain = effect["interval_width"], effect["estimate"]
    return bool(comparable and effect["n_clusters"] >= 2 and width is not None
                and gain is not None and gain > width)


def fragility(rows, predictions):
    """Largest signed Delta L at the same state/config, with fair prediction ties."""
    cells = defaultdict(dict)
    for row in rows:
        cells[row.query.state, row.query.config][row.query.capability] = row
    accuracy, regret, pairs, queries, records = [], [], [], [], []
    for (state, config), cell in sorted(cells.items()):
        if set(cell) != set(CAPS):
            continue
        actual = np.array([cell[c].target for c in CAPS])
        pred = [predictions.get(cell[c].query.row_id) for c in CAPS]
        if any(p is None for p in pred):
            continue
        pred = np.array(pred)
        true_top = np.isclose(actual, actual.max(), atol=1e-12, rtol=0)
        guessed = np.isclose(pred, pred.max(), atol=1e-12, rtol=0)
        accuracy.append(float(np.mean(true_top[guessed])))
        regret.append(float(actual.max() - actual[guessed].mean()))
        pair_scores = []
        for i, j in itertools.combinations(range(3), 2):
            if abs(actual[i] - actual[j]) <= 1e-12:
                continue
            pair_scores.append(.5 if abs(pred[i] - pred[j]) <= 1e-12
                               else float(np.sign(pred[i] - pred[j]) == np.sign(actual[i] - actual[j])))
        pairs.append(float(np.mean(pair_scores)) if pair_scores else 1.0)
        queries.append(cell[CAPS[0]].query)
        records.append({"state": state, "config": config,
                        "actual_most_fragile": [c for c, yes in zip(CAPS, true_top) if yes],
                        "predicted_most_fragile": [c for c, yes in zip(CAPS, guessed) if yes]})
    return {"top1_accuracy": clustered(accuracy, queries), "regret_nats": clustered(regret, queries),
            "pairwise_accuracy": clustered(pairs, queries), "cells": records}


def evaluate(rows, scheme, extra=()):
    result = {"scheme": scheme, "capabilities": {}, "fragility": {}}
    all_predictions = {f"{budget}/{form}": {} for budget in BUDGETS for form in FORMS}
    scored_rows = list(extra) if scheme == "extra_pruning_strength_new_source" else list(rows)
    for cap in CAPS:
        cr = [r for r in rows if r.query.capability == cap]
        ce = [r for r in extra if r.query.capability == cap]
        folds = list(make_folds(cr, scheme, ce))
        tests = [r for _, _, test in folds for r in test]
        if len({r.query.row_id for r in tests}) != len(tests):
            raise ValueError("Each observation must be scored exactly once per evaluation")
        fits, vectors = {}, {}
        for budget, form in itertools.product(BUDGETS, FORMS):
            key, predictions, audit = f"{budget}/{form}", [], []
            for fold_id, train, test in folds:
                inner_group = "size" if scheme == "leave_one_size_out" else "state"
                model = fit(train, budget, form, inner_group)
                predictions.extend(predict(model, [r.query for r in test]))
                audit.append({"fold": fold_id, "test_ids": [r.query.row_id for r in test], "model": model})
            vectors[key] = predictions
            valid = [i for i, p in enumerate(predictions) if p is not None]
            mae = clustered([abs(tests[i].target - predictions[i]) for i in valid], [tests[i].query for i in valid])
            fits[key] = {"mae_nats": mae, "coverage": len(valid) / len(tests),
                         "predictions": predictions, "folds": audit}
            if "strength" in scheme:
                fits[key]["by_strength"] = {}
                for strength in sorted({r.query.strength for r in tests}):
                    indices = [i for i in valid if tests[i].query.strength == strength]
                    fits[key]["by_strength"][str(strength)] = clustered(
                        [abs(tests[i].target - predictions[i]) for i in indices], [tests[i].query for i in indices])
            all_predictions[key].update({r.query.row_id: p for r, p in zip(tests, predictions)})
        input_gains = []
        for lower, upper in zip(BUDGETS, BUDGETS[1:]):
            for form in FORMS:
                effect = paired(tests, vectors[f"{lower}/{form}"], vectors[f"{upper}/{form}"])
                input_gains.append({"lower_budget": lower, "upper_budget": upper, "form": form, **effect,
                                    "statistics_useful": reading_rule(effect, scheme=scheme, lower_form=form,
                                        upper_form=form, lower_budget=lower, upper_budget=upper)})
        form_gains = []
        for budget in BUDGETS:
            for lower, upper in itertools.combinations(FORMS, 2):
                form_gains.append({"budget": budget, "lower_form": lower, "upper_form": upper,
                                   **paired(tests, vectors[f"{budget}/{lower}"], vectors[f"{budget}/{upper}"])})
        decomposition = []
        # Exact additive path, on identical rows: lower-budget median ->
        # upper-budget median (input) -> upper-budget linear (change of form).
        for lower, upper in zip(BUDGETS, BUDGETS[1:]):
            for form in LINEAR_FORMS:
                a, b, c = vectors[f"{lower}/median_curve"], vectors[f"{upper}/median_curve"], vectors[f"{upper}/{form}"]
                decomposition.append({"lower_budget": lower, "upper_budget": upper,
                                      "from_form": "median_curve", "to_form": form,
                                      "input_at_fixed_median": paired(tests, a, b),
                                      "form_at_upper_budget": paired(tests, b, c),
                                      "total": paired(tests, a, c),
                                      "input_at_fixed_linear": paired(tests, vectors[f"{lower}/{form}"], c),
                                      "form_at_lower_budget": paired(tests, a, vectors[f"{lower}/{form}"])})
        result["capabilities"][cap] = {"row_order": [r.query.row_id for r in tests], "fits": fits,
                                      "input_gains": input_gains, "within_budget_form_gains": form_gains,
                                      "gain_decomposition": decomposition}
    result["fragility"] = {key: fragility(scored_rows, predictions) for key, predictions in all_predictions.items()}
    return result


def verdicts(evaluations):
    output = []
    for arm in ARMS:
        primary = evaluations[arm]["leave_one_source_state_out"]["capabilities"]
        within = evaluations[arm]["unseen_strength_same_sources"]["capabilities"]
        for cap in CAPS:
            comparisons = [g for g in primary[cap]["input_gains"]
                           if g["lower_budget"] == "dense_anchor" and g["form"] in LINEAR_FORMS]
            passing = [g["form"] for g in comparisons if g["statistics_useful"]]
            within_only = [g["form"] for g in within[cap]["input_gains"]
                           if g["lower_budget"] == "dense_anchor" and g["form"] in LINEAR_FORMS
                           and g["estimate"] is not None and g["estimate"] > 0
                           and not next(c for c in comparisons if c["form"] == g["form"])["statistics_useful"]]
            fits = primary[cap]["fits"]
            switched = fits["dense_anchor/ols"]["mae_nats"]["estimate"] - fits["dense_statistics/ridge"]["mae_nats"]["estimate"]
            output.append({"arm": arm, "capability": cap, "helped": bool(passing), "qualifying_forms": passing,
                           "against_budget": "K0 + dense anchor", "same_form_comparisons": comparisons,
                           "within_source_strength_gain_without_primary_clearance": within_only,
                           "ols_anchor_to_ridge_statistics_gain_not_input_evidence": switched,
                           "interpretation": "Clears the fixed development reading rule" if passing else
                           "Dense statistics do not clear the fixed across-source reading rule"})
    return output


def fmt(value):
    return "NA" if value is None else f"{value:.5f}"


def interval(effect):
    return "NA" if effect["ci95"] is None else f"[{fmt(effect['ci95'][0])}, {fmt(effect['ci95'][1])}]"


def verdict_block(summary):
    lines = ["V92 VERDICT — development only; lower MAE is better.",
             "Statistics budget vs K0 + dense anchor, SAME form; gain must exceed the FULL 95% interval width.",
             "Positive gains are MAE reductions in nats. OLS and ridge are separate fixed comparisons.", "",
             "| Arm | Capability | Statistics helped? | OLS gain [95% CI]; width | Ridge gain [95% CI]; width | Source clusters |",
             "|---|---|---|---|---|---|"]
    for v in summary["verdicts"]:
        a, b = v["same_form_comparisons"]
        helped = "YES: " + ", ".join(v["qualifying_forms"]) if v["helped"] else "NO under fixed rule"
        lines.append(f"| {v['arm']} | {v['capability']} | {helped} | {fmt(a['estimate'])} {interval(a)}; {fmt(a['interval_width'])} | "
                     f"{fmt(b['estimate'])} {interval(b)}; {fmt(b['interval_width'])} | {a['n_clusters']} |")
    worse_than_median = []
    for arm in ARMS:
        primary = summary["evaluations"][arm]["leave_one_source_state_out"]
        for cap, cr in primary["capabilities"].items():
            fits = cr["fits"]
            median = fits["K0/median_curve"]["mae_nats"]["estimate"]
            if all(fits[f"dense_statistics/{form}"]["mae_nats"]["estimate"] > median for form in LINEAR_FORMS):
                worse_than_median.append((arm, cap))
    lines += ["", f"In {len(worse_than_median)}/9 arm/capability pairs, both statistics-augmented linear candidates still have "
              "higher MAE than the K0 source-free median curve. Passing the incremental input rule is not a win over the delivered predictor."]
    lines += ["", "Most-fragile capability accuracy (ridge, dense anchor → dense statistics): " + "; ".join(
        f"{arm} {summary['evaluations'][arm]['leave_one_source_state_out']['fragility']['dense_anchor/ridge']['top1_accuracy']['estimate']:.1%} → "
        f"{summary['evaluations'][arm]['leave_one_source_state_out']['fragility']['dense_statistics/ridge']['top1_accuracy']['estimate']:.1%}"
        for arm in ARMS) + ". These are descriptive ranking results, separate from the MAE reading rule."]
    return "\n".join(lines)


def markdown(summary):
    lines = [verdict_block(summary), "", "Development protocol and coverage", "",
             *summary["protocol"]["notes"], "", "Input definitions", "",
             "| Budget | Inputs |", "|---|---|", "| K0 | N0, D0, density or bits (and group size for grouped RTN) |",
             "| dense_anchor | K0 + arm-local dense capability loss L0 |",
             "| dense_statistics | dense_anchor + capability-specific B, V, W from V91 dense descriptors |", "",
             "N0 counts transformer matrices excluding embeddings/head, following V36; D0 = step × 2,097,152 processed tokens. "
             "OLS/ridge use raw inputs standardized on training response rows, separately per capability. "
             "There is one coefficient per input plus intercept; no nonlinear expansion or interaction search. "
             "B = z[y] − E_p[z], V = 1 − Σp², W = Var_p(z), pooled over reference tokens. Descriptor L is audited but never added as a predictor.",
             "", "Candidate and uncertainty definitions", "",
             "Zero predicts signed ΔL=0. Constant is the training capability mean. Median is the equal-source median at each training configuration. "
             "OLS is unregularized least squares. Ridge minimizes mean squared error + α‖β_nonintercept‖²; "
             f"α ∈ {list(ALPHAS)} is selected by inner grouped CV MAE. Size holdouts use inner size folds; all other evaluations use inner source folds. "
             "Every inner fold refits its scaler. The delivered new-source branch is the median for all three arms/capabilities "
             "(analysis/final_rule.py); it does not consume L0/B/V/W when predicting ΔL. Its absolute-loss presentation would add L0, "
             "which is unavailable to K0, so all candidates are evaluated directly on ΔL.",
             "", f"MAE and paired improvements weight source states equally, then configurations equally within a state. "
             f"95% percentile intervals resample whole source states {BOOTSTRAPS:,} times (seed {SEED}), pairing predictions on identical rows. "
             "Intervals condition on the fitted cross-validation predictions; they do not rerun fitting on bootstrap samples. "
             "Leave-one-size-out has three size folds but intervals still cluster on the 9 or 6 source states as requested; "
             "these small-cluster intervals are development evidence, not independent confirmation. No candidate is selected by outer-fold MAE. "
             "The rule is applied separately to the two fixed linear estimators; YES names the one that clears, without selecting a new predictor or adjusting for multiple comparisons.",
             "", "In grouped quantization, the statistics-budget OLS design is rank deficient in every primary fold: only five training "
             "source states support six source inputs plus an intercept, alongside configuration inputs. OLS uses the minimum-norm least-squares solution; "
             "ridge stabilizes the same design. Design ranks, coefficients, fold-local scalers and penalty scores are recorded in JSON. "
             "Audit row sets are interned as ordered indices into observations to avoid repeating the same training rows thousands of times.", ""]
    lines += ["MAE in nats: primary and size holdout", "",
              "| Split | Arm | Capability | Budget | Zero | Constant | Median | OLS | Ridge | Delivered | Clusters |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for split in ("leave_one_source_state_out", "leave_one_size_out"):
            for cap in CAPS:
                fits = summary["evaluations"][arm][split]["capabilities"][cap]["fits"]
                for budget in BUDGETS:
                    vals = [fmt(fits[f"{budget}/{f}"]["mae_nats"]["estimate"]) for f in FORMS]
                    lines.append(f"| {split} | {arm} | {cap} | {budget} | " + " | ".join(vals)
                                 + f" | {fits[f'{budget}/ols']['mae_nats']['n_clusters']} |")
    lines += ["", "What the extra inputs buy with the same form", "",
              "Zero/constant/median/delivered ignore the added inputs, so both adjacent-budget gains are exactly zero. "
              "The table gives OLS/ridge paired gains; positive means the upper budget improves. The complete comparisons and fold audits are in summary.json.", "",
              "| Split | Arm | Capability | Budget transition | Form | Gain | 95% CI | Width | Clusters |",
              "|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for split in ("leave_one_source_state_out", "leave_one_size_out"):
            for cap, cr in summary["evaluations"][arm][split]["capabilities"].items():
                for gain in cr["input_gains"]:
                    if gain["form"] in LINEAR_FORMS:
                        lines.append(f"| {split} | {arm} | {cap} | {gain['lower_budget']} → {gain['upper_budget']} | {gain['form']} | "
                                     f"{fmt(gain['estimate'])} | {interval(gain)} | {fmt(gain['interval_width'])} | {gain['n_clusters']} |")
    lines += ["", "What changing the form buys at a fixed budget", "",
              "The source-free median already uses a flexible configuration curve; a source-dependent linear form is a different form, "
              "not necessarily a richer configuration curve. Ridge and OLS have the same linear functional class; their contrast is regularization. "
              "Every within-budget pair and two exact additive decompositions of each adjacent-budget contrast are stored in JSON. "
              "For median(lower budget) → linear(upper budget): total gain = same-form input gain + fixed-budget form gain. "
              "Following the median path makes input gain zero, whereas following the linear path isolates the actual input increment. "
              "A gain after changing both budget and form cannot be attributed entirely to inputs.", "",
              "| Arm | Capability | Budget | Constant → median | Median → OLS | Median → ridge | OLS → ridge (regularization) |",
              "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for cap, cr in summary["evaluations"][arm]["leave_one_source_state_out"]["capabilities"].items():
            for budget in BUDGETS:
                gains = {(g["lower_form"], g["upper_form"]): g for g in cr["within_budget_form_gains"] if g["budget"] == budget}
                cells = [f"{fmt(gains[p]['estimate'])} {interval(gains[p])}" for p in
                         (("constant", "median_curve"), ("median_curve", "ols"), ("median_curve", "ridge"), ("ols", "ridge"))]
                lines.append(f"| {arm} | {cap} | {budget} | " + " | ".join(cells) + " |")
    lines += ["", "Which capability is most fragile at a fixed configuration", "",
              "Fragility is the largest signed ΔL among math/code/QA, not the largest absolute loss. "
              "Top-1 accuracy awards the expected credit under uniform prediction ties (zero gets 1/3); "
              "regret is the actual maximum ΔL minus the actual ΔL of the predicted most-fragile capability. "
              "JSON includes pairwise ranking accuracy, source-cluster intervals, and each state's actual/predicted capability.", "",
              "| Split | Arm | Budget/form | Top-1 accuracy | Regret (nats) | Clusters |",
              "|---|---|---|---|---|---|"]
    for arm in ARMS:
        for split in ("leave_one_source_state_out", "leave_one_size_out"):
            for key, f in summary["evaluations"][arm][split]["fragility"].items():
                if key.startswith("K0/") or key.split("/")[1] in LINEAR_FORMS:
                    lines.append(f"| {split} | {arm} | {key} | {fmt(f['top1_accuracy']['estimate'])} | "
                                 f"{fmt(f['regret_nats']['estimate'])} | {f['top1_accuracy']['n_clusters']} |")
    lines += ["", "Unseen strength: separate from the primary result", "",
              "Each common density/bit-width is held out globally in turn. The same-sources diagnostic retains other strengths of the scored source; "
              "the new-source analysis removes that source entirely as well. Endpoint holdouts extrapolate; interior holdouts interpolate. "
              "For example grouped b4 trains on b3/b5 at all three group sizes. Extra pruning d=.65 (interior) and d=.55 (exterior) train only "
              "on the common .6/.7/.8/.9 grid and exclude the scored source. They have only two source clusters. "
              "Median baselines use adjacent linear interpolation/extension (density/raw bits; log2(qmax) for grouped bits). "
              "The delivered new-source rule is scored only where defined: pruning [.6,.9], grouped no bit extrapolation, channel seen bits. "
              "In the same-sources diagnostic 'delivered' remains this source-free new-source reference, not the paper's seen-state power/interpolation branch. "
              "Delivered coverage is explicit in JSON; its partial-coverage MAE must not be compared with full-coverage MAEs.", "",
              "| Split | Arm | Capability | Held strength | Form | K0 MAE | Anchor MAE | Statistics MAE | Clusters |",
              "|---|---|---|---|---|---|---|---|---|"]
    for arm, evaluations in summary["evaluations"].items():
        for split, evaluation in evaluations.items():
            if "strength" not in split:
                continue
            for cap, cr in evaluation["capabilities"].items():
                for form in ("median_curve", "ols", "ridge"):
                    by = [cr["fits"][f"{b}/{form}"]["by_strength"] for b in BUDGETS]
                    for strength in by[0]:
                        vals = [fmt(v[strength]["estimate"]) for v in by]
                        lines.append(f"| {split} | {arm} | {cap} | {strength} | {form} | " + " | ".join(vals)
                                     + f" | {by[0][strength]['n_clusters']} |")
    lines += ["", "Reading-rule qualifications", ""]
    for v in summary["verdicts"]:
        if v["within_source_strength_gain_without_primary_clearance"]:
            lines.append(f"- {v['arm']}/{v['capability']}: {', '.join(v['within_source_strength_gain_without_primary_clearance'])} "
                         "has a positive same-source strength-transfer input gain but does not clear the across-source reading rule; it does not count.")
        if not v["helped"] and v["ols_anchor_to_ridge_statistics_gain_not_input_evidence"] > 0:
            lines.append(f"- {v['arm']}/{v['capability']}: switching anchor OLS to statistics ridge improves MAE by "
                         f"{fmt(v['ols_anchor_to_ridge_statistics_gain_not_input_evidence'])}, but this mixes inputs and regularization and does not count.")
    lines += ["", "All requested comparisons use existing scalar responses. No compressed-model diagnostic enters a predictor. "
              "The small Pythia panel tests transfer between source states/sizes within one family; it does not establish architecture transfer. "
              "Signed improvements in reference CE can be negative and do not by themselves establish downstream task accuracy.", ""]
    return "\n".join(lines)


def build_summary(root=ROOT):
    rows, extra, provenance = load_data(root)
    evaluations = {}
    for arm in ARMS:
        ar = [r for r in rows if r.query.arm == arm]
        if len({r.query.state for r in ar}) < 3:
            raise ValueError(f"Too few available sources for {arm}")
        evaluations[arm] = {}
        for scheme in ("leave_one_source_state_out", "leave_one_size_out", "unseen_strength_same_sources", "unseen_strength_new_source"):
            evaluations[arm][scheme] = evaluate(ar, scheme)
        if arm == "pruning" and extra:
            evaluations[arm]["extra_pruning_strength_new_source"] = evaluate(ar, "extra_pruning_strength_new_source", extra)
    source_hashes = {}
    for name in ("analysis/v92_input_comparison.py", "tests/test_v92_input_comparison.py", "analysis/final_rule.py"):
        source_hashes[name] = hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
    summary = {"version": "V92", "development_only": True, "cpu_only": True,
               "target": "signed capability CE change: measured compressed loss minus arm-local dense loss, nats/reference token",
               "protocol": {"budgets": {b: {arm: fields(arm, b) for arm in ARMS} for b in BUDGETS},
                            "candidates": FORMS, "ridge_alphas": ALPHAS, "bootstrap_replicates": BOOTSTRAPS,
                            "seed": SEED, "reading_rule": "same-form LOSO MAE(anchor)-MAE(statistics) > CI_upper-CI_lower",
                            "notes": [
                                "Pruning and per-channel RTN: nine source states, three sizes × steps 16k/64k/143k. Grouped RTN: six available states; all three step64k files are absent. No missing responses or descriptors are imputed.",
                                "Primary/size evaluation uses the common measured configuration grid: pruning densities .6/.7/.8/.9; grouped bits 3/4/5 × group sizes 64/128/256; per-channel bits 3/4/6/8. Budgets use exactly the same rows within each arm.",
                                "Four additional pruning state-density cells (.55/.65 for 160M-step143k and 1.4B-step64k) are scored separately. Grouped g32/g512 confirmation additions in the V54 files are excluded. Only the listed development response files and V91 dense descriptors are read; no frozen prediction/test artifacts are accessed.",
                                "The V91 dense measurements are reused, not rerun: 64 probe references per source/capability. L0 is the true dense anchor recorded with each response arm. Descriptor L differences, sample counts, revisions and file hashes are retained in the audit. "
                                f"Historical arm-local L0 and V91 L differ by at most {max(abs(a['difference']) for a in provenance['dense_anchor_audit']):.5f} nats; the cause is not remeasured, and descriptor L is never substituted."]},
               "provenance": {**provenance, "code_sha256": source_hashes},
               "observations": [{"id": r.query.row_id, "state": r.query.state, "size": r.query.size,
                                  "arm": r.query.arm, "capability": r.query.capability, "config": r.query.config,
                                  "target": r.target, "precompression_inputs": {k: v.value for k, v in r.query.inputs.items()},
                                  "primary": r in rows} for r in rows + extra],
               "evaluations": evaluations, "verdicts": verdicts(evaluations)}
    compact_audit_rows(summary)
    return summary


def compact_audit_rows(summary):
    """Lossless audit storage: shared ordered row sets index observations.

    Fitting APIs retain full row IDs for tests/reuse. Only report storage uses
    references, preserving every inner/outer training and validation membership.
    """
    indices = {r["id"]: i for i, r in enumerate(summary["observations"])}
    catalogue, reverse = {}, {}
    keys = {"training_ids": "training_rows_ref", "validation_ids": "validation_rows_ref", "test_ids": "test_rows_ref"}

    def visit(node):
        if isinstance(node, dict):
            for key in list(node):
                if key in keys:
                    rows = tuple(indices[row_id] for row_id in node.pop(key))
                    if rows not in reverse:
                        ref = f"r{len(catalogue):04d}"
                        reverse[rows], catalogue[ref] = ref, list(rows)
                    node[keys[key]] = reverse[rows]
                else:
                    visit(node[key])
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(summary["evaluations"])
    summary["audit_row_sets"] = catalogue
    summary["protocol"]["audit_format"] = (
        "training_rows_ref/validation_rows_ref/test_rows_ref resolve through audit_row_sets; "
        "each ordered integer list indexes observations. row_order contains full IDs and aligns prediction vectors."
    )


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    summary = build_summary()
    output = ROOT / OUT
    if output.is_symlink():
        raise ValueError("Output directory cannot be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in (("summary.json", json.dumps(summary, indent=2, allow_nan=False) + "\n"),
                          ("summary.md", markdown(summary))):
        path = output / name
        if path.is_symlink():
            raise ValueError("Output cannot be a symlink")
        path.write_text(content)
    print(verdict_block(summary))


if __name__ == "__main__":
    main()
