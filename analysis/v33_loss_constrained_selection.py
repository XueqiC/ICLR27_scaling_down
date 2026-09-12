#!/usr/bin/env python3
"""CPU-only OFFLINE REPLAY of loss-constrained nominal storage selection.

Run: python analysis/v33_loss_constrained_selection.py [--dry-run]
Only evaluate saved V28/V30 coefficients; never fit on replay outcomes.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

try:
    from . import prediction_audit as audit
    from . import v28_new_source_prediction as v28
    from . import v30_quant_shape_candidates as v30
except ImportError:
    import prediction_audit as audit
    import v28_new_source_prediction as v28
    import v30_quant_shape_candidates as v30

import numpy as np  # audit sets CPU BLAS thread limits before NumPy is imported.

ROOT = audit.ROOT
OUT = ROOT / "results/v33-selection"
REPORT = ROOT / "paper/docs/LOSS_CONSTRAINED_SELECTION.md"
V28 = ROOT / "results/v28-new-source-pred/frozen_predictions.json"
V30 = ROOT / "results/v30-quant-candidates/summary.json"
CAPS = audit.CAPABILITIES
METHODS = ("pruning", "quantization")
POLICIES = ("selector", "always_cheapest", "always_dense")
QUANTILES = (0., .1, .25, .5, .75, 1.)
QUANT_SHAPE = "shared_eta"  # Fixed analysis choice; never select by replay scores.
SEED = 330907
TOL = 1e-12


def losses_at(table, key):
    if key not in table:
        raise ValueError(f"Missing source dense/configuration: {key}")
    return {cap: audit.finite(table[key][cap]) for cap in CAPS}


def validate_freezes(frozen28, frozen30):
    if frozen28.get("freeze_sha256") != v28.seal(frozen28):
        raise ValueError("V28 freeze integrity check failed")
    if frozen28["version"] != 28 or frozen30["version"] != 30:
        raise ValueError("Expected saved V28 and V30 artifacts")


def held_out_fit(folds, model, key="fit"):
    matches = [f for f in folds if v28.canonical(f["held_out"]) == v28.canonical(model)]
    if len(matches) != 1:
        raise ValueError(f"Need exactly one saved dev-LOMO fold for {model}")
    return matches[0][key]


def assert_held_out(mapping, model):
    if v28.canonical(model) in {v28.canonical(m) for m in mapping["train_models"]}:
        raise ValueError(f"Target model leaked into frozen mapping: {model}")


def frozen_prediction(frozen28, frozen30, *, model, info, method, coordinate,
                      method_dense, source_dense, cohort):
    """Dense/basic inputs only. No scored compressed outcomes are accepted.

    Dev targets use stored LOMO fits, never the in-sample full-dev mapping.
    The separate Qwen3-8B replay uses the available full-development V28 fits.
    """
    predictions, fit_sources = {}, {}
    quant_fit = None
    if method == "quantization" and cohort == "development_lomo":
        folds = frozen30["panels"]["broad"]["folds"]
        if any(v28.canonical(f["held_out"]) == v28.canonical(model) for f in folds):
            quant_fit = held_out_fit(folds, model, "fits").get(QUANT_SHAPE)
    for cap in CAPS:
        if quant_fit is not None:
            fit = quant_fit
            assert_held_out(fit["mappings"][cap], model)
            target = v30.basic_input({"model": v30.canonical(model),
                                      "reference_loss": method_dense[cap],
                                      **v30.model_features(v30.canonical(model))})
            delta = v30.predict_candidate(fit, target, cap, coordinate, 16)
            origin = f"V30 broad saved dev-LOMO {QUANT_SHAPE}"
        else:
            saved = frozen28["methods"][method]["by_capability"][cap]
            fit = (held_out_fit(saved["lomo"]["folds"], model)
                   if cohort == "development_lomo" else saved["fit"])
            assert_held_out(fit["mapping"], model)
            target = v28.basic_input({"model": model, "family": info["family"],
                                      "N0": info["N0"], "dense_loss": method_dense[cap]})
            delta, = v28.predict_arm(fit, target, [coordinate], mode="A")
            origin = ("V28 saved dev-LOMO Mode A" if cohort == "development_lomo"
                      else "V28 frozen full-development Mode A; separate offline transfer")
            if method == "quantization" and cohort == "development_lomo":
                origin += "; fallback: V30 saved shape/fold unavailable"
        # Rebase both the prediction and its measured endpoint to ONE source dense.
        predictions[cap] = audit.finite(delta + method_dense[cap] - source_dense[cap])
        fit_sources[cap] = origin
    return predictions, fit_sources


def make_panel(model, info, tables, paths, frozen28, frozen30, cohort):
    source_dense = losses_at(tables["pruning"], "1.0")
    anchors = {"pruning": source_dense,
               "quantization": losses_at(tables["quantization"], "dense")}
    candidates = [{"id": "dense", "method": "dense", "coordinate": 1.,
                   "nominal_storage": 1., "nominal_active_parameter_ratio": 1.,
                   "delta_hat": dict.fromkeys(CAPS, 0.), "actual_delta": dict.fromkeys(CAPS, 0.),
                   "actual_loss": source_dense, "calibration_points_used": 0,
                   "prediction_source": dict.fromkeys(CAPS, "exact source dense reference"),
                   "extrapolated_shape": False}]
    caveats = []
    for method in METHODS:
        table = tables[method]
        caveats.extend({"method": method, "key": k, "detail": v}
                       for k, v in table.items() if k.startswith("_"))
        seen = set()
        for key in sorted(table):
            if key.startswith("_") or key in ("dense", "1.0"):
                continue
            coordinate = audit.finite(key)
            if method == "pruning":
                if not 0 < coordinate < 1:
                    raise ValueError(f"Invalid pruning density: {key}")
                nominal_storage, active = coordinate, coordinate
                extrapolated = not min(v28.PRUNE_DEV) <= coordinate <= max(v28.PRUNE_DEV)
            else:
                if coordinate != int(coordinate) or not 2 <= coordinate < 16:
                    raise ValueError(f"Invalid quantization bit: {key}")
                coordinate = int(coordinate)
                nominal_storage, active = coordinate / 16., 1.
                bounds = v30.BITS if cohort == "development_lomo" else v28.QUANT_DEV
                extrapolated = not min(bounds) <= coordinate <= max(bounds)
            if coordinate in seen:
                raise ValueError(f"Duplicate configuration aliases: {model}/{method}/{key}")
            seen.add(coordinate)
            predicted, origin = frozen_prediction(
                frozen28, frozen30, model=model, info=info, method=method, coordinate=coordinate,
                method_dense=anchors[method], source_dense=source_dense, cohort=cohort)
            if method == "quantization" and all(s.startswith("V28") for s in origin.values()):
                extrapolated = not min(v28.QUANT_DEV) <= coordinate <= max(v28.QUANT_DEV)
            actual = losses_at(table, key)
            candidates.append({"id": f"{method}:{coordinate:g}", "method": method,
                               "coordinate": coordinate, "nominal_storage": nominal_storage,
                               "nominal_active_parameter_ratio": active, "delta_hat": predicted,
                               "actual_delta": {c: actual[c] - source_dense[c] for c in CAPS},
                               "actual_loss": actual, "calibration_points_used": 0,
                               "prediction_source": origin, "extrapolated_shape": extrapolated})
        if not seen:
            raise ValueError(f"No measured compressed configurations: {model}/{method}")
    return {"model": model, "cohort": cohort, "source_dense": source_dense,
            "method_dense": anchors,
            "quant_dense_minus_source_dense": {c: anchors["quantization"][c]-source_dense[c] for c in CAPS},
            "source_paths": {m: str(p.relative_to(ROOT)) for m, p in paths.items()},
            "measurement_caveats": caveats, "candidates": sorted(candidates, key=nominal_order)}


def load_panels():
    frozen28, frozen30 = audit.read_json(V28), audit.read_json(V30)
    validate_freezes(frozen28, frozen30)
    paths = [V28, V30, Path(__file__), Path(audit.__file__), Path(v28.__file__), Path(v30.__file__)]
    # Hash before reading outcomes; audit.write_outputs rechecks before publishing.
    hashes = audit.provenance(paths)
    roster = [(m, i, "development_lomo") for m, i in sorted(frozen28["metadata"]["models"].items())]
    # Available paired legacy target; do not describe it as the named Base freeze target.
    extra_paths = {"pruning": ROOT / "results/v6-capability-geometry/Qwen--Qwen3-8B/prune_losses.json",
                   "quantization": ROOT / "results/v10-quantization/Qwen3-8B/quant_losses.json"}
    if all(p.exists() for p in extra_paths.values()):
        roster.append(("Qwen3-8B", {"family": "qwen3", "N0": frozen28["metadata"]["qwen3_8b"]["N0"]},
                       "separate_offline_transfer"))
    panels = []
    for model, info, cohort in roster:
        source_paths = (extra_paths if cohort == "separate_offline_transfer" else {
            "pruning": ROOT / "results/v6-capability-geometry" / model / "prune_losses.json",
            "quantization": ROOT / "results/v10-quantization" /
            ("Qwen--"+model if model.startswith("Qwen3") else model) / "quant_losses.json"})
        hashes.update(audit.provenance(source_paths.values()))
        tables = {m: audit.read_json(p) for m, p in source_paths.items()}
        panels.append(make_panel(model, info, tables, source_paths, frozen28, frozen30, cohort))
    included = {v28.canonical(p["model"]) for p in panels}
    excluded = []
    for directory, filename in (("v6-capability-geometry", "prune_losses.json"),
                                ("v10-quantization", "quant_losses.json")):
        for path in sorted((ROOT / "results" / directory).glob("*/"+filename)):
            if v28.canonical(path.parent.name) not in included:
                excluded.append({"path": str(path.relative_to(ROOT)),
                                 "model": v28.canonical(path.parent.name),
                                 "reason": "No complete paired panel with supported frozen basic-input prediction"})
    return panels, hashes, excluded, frozen28["freeze_sha256"]


def nominal_order(candidate):
    """Outcome-independent tie breaking, shared by selector and oracle."""
    return candidate["nominal_storage"], candidate["method"], candidate["id"]


def validate_budget(tau):
    if set(tau) != set(CAPS) or any(audit.finite(v) < 0 for v in tau.values()):
        raise ValueError("Need a finite nonnegative loss budget for every capability")


def feasible(delta, tau):
    return all(delta[c] <= tau[c] + TOL for c in CAPS)


def prediction_view(candidates):
    return [{k: row[k] for k in ("id", "method", "nominal_storage", "delta_hat")}
            for row in candidates]


def select(predictions, tau):
    """Minimum nominal storage subject to ALL frozen predicted loss budgets."""
    validate_budget(tau)
    allowed = [p for p in predictions if feasible(p["delta_hat"], tau)]
    if not allowed:
        raise ValueError("No predicted-feasible candidate; source dense must be included")
    return min(allowed, key=nominal_order)["id"]


def budget_grid(candidates, quantiles=QUANTILES):
    """Cartesian capability quantiles plus zero; retrospective test scenarios only."""
    qs = np.asarray(quantiles, dtype=float)
    if qs.ndim != 1 or len(qs) == 0 or not np.isfinite(qs).all() or np.any((qs < 0) | (qs > 1)):
        raise ValueError("Budget quantiles must be a nonempty sequence in [0, 1]")
    compressed = [r for r in candidates if r["method"] != "dense"]
    if not compressed:
        raise ValueError("Budget grid needs measured compressed candidates")
    axes = {c: sorted({0., *np.maximum(0., np.quantile(
        [r["actual_delta"][c] for r in compressed], qs)).tolist()}) for c in CAPS}
    return axes, [dict(zip(CAPS, point)) for point in itertools.product(*(axes[c] for c in CAPS))]


def score_budget(candidates, predictions, tau):
    validate_budget(tau)
    index = {r["id"]: r for r in candidates}
    actually_feasible = [r for r in candidates if feasible(r["actual_delta"], tau)]
    oracle = min(actually_feasible, key=nominal_order)
    choices = {"selector": select(predictions, tau),
               "always_cheapest": min(predictions, key=nominal_order)["id"], "always_dense": "dense"}
    scores = {}
    for policy, chosen in choices.items():
        row = index[chosen]
        failed = [c for c in CAPS if row["actual_delta"][c] > tau[c] + TOL]
        scores[policy] = {"selected": chosen, "method": row["method"],
                          "constraint_satisfied": not failed, "violated_capabilities": failed,
                          "nominal_storage": row["nominal_storage"],
                          "nominal_cost_gap_vs_oracle": row["nominal_storage"]-oracle["nominal_storage"]}
    excluded = {p["id"]: [c for c in CAPS if p["delta_hat"][c] > tau[c] + TOL]
                for p in predictions if not feasible(p["delta_hat"], tau)}
    feasible_ids = [r["id"] for r in actually_feasible]
    wrongly_excluded = [key for key in feasible_ids if key in excluded]
    counts = {}
    for method in ("all_compressed", *METHODS):
        eligible = {r["id"] for r in candidates if r["method"] != "dense" and
                    (method == "all_compressed" or r["method"] == method)}
        denominator = len(eligible.intersection(feasible_ids))
        numerator = len(eligible.intersection(wrongly_excluded))
        counts[method] = {"actually_feasible": denominator, "wrongly_excluded": numerator,
                          "fraction": numerator/denominator if denominator else None}
    return {"tau": tau, "oracle": oracle["id"], "oracle_nominal_storage": oracle["nominal_storage"],
            "oracle_tied_methods": sorted({r["method"] for r in actually_feasible
                                            if r["nominal_storage"] == oracle["nominal_storage"]}),
            "policies": scores, "predicted_infeasible": excluded,
            "actually_feasible": feasible_ids, "wrongly_excluded": wrongly_excluded,
            "exclusion_counts": counts}


def replay_panel(panel, quantiles=QUANTILES):
    candidates = panel["candidates"]
    axes, budgets = budget_grid(candidates, quantiles)
    predictions = prediction_view(candidates)
    rows = [score_budget(candidates, predictions, tau) for tau in budgets]
    totals = {}
    for policy in POLICIES:
        scored = [r["policies"][policy] for r in rows]
        success = np.array([r["constraint_satisfied"] for r in scored])
        gaps = np.array([r["nominal_cost_gap_vs_oracle"] for r in scored])
        totals[f"{policy}/constraint_satisfaction"] = [float(success.mean()), 1.]
        totals[f"{policy}/nominal_storage"] = [float(np.mean([r["nominal_storage"] for r in scored])), 1.]
        totals[f"{policy}/nominal_cost_gap_vs_oracle"] = [float(gaps.mean()), 1.]
        totals[f"{policy}/nominal_cost_gap_when_satisfied"] = [float((gaps*success).mean()), float(success.mean())]
    for method in ("all_compressed", *METHODS):
        counts = [r["exclusion_counts"][method] for r in rows]
        totals[f"exclusions/{method}"] = [float(np.mean([r["wrongly_excluded"] for r in counts])),
                                          float(np.mean([r["actually_feasible"] for r in counts]))]
        compressed = [r for r in candidates if r["method"] != "dense" and
                      (method == "all_compressed" or r["method"] == method)]
        actual = np.array([r["actual_delta"][c] for r in compressed for c in CAPS])
        predicted = np.array([r["delta_hat"][c] for r in compressed for c in CAPS])
        for name, values in (("mean_abs_actual_delta", abs(actual)), ("mean_abs_predicted_delta", abs(predicted)),
                             ("mean_prediction_minus_actual", predicted-actual),
                             ("overprediction_fraction", predicted > actual)):
            totals[f"response/{method}/{name}"] = [float(values.mean()), 1.]
    domination = {}
    for method in ("dense", *METHODS):
        selected = sum(r["policies"]["selector"]["method"] == method for r in rows)
        successful = sum(r["policies"]["selector"]["method"] == method and
                         r["policies"]["selector"]["constraint_satisfied"] for r in rows)
        oracle = sum(candidates_by_id(candidates, r["oracle"])["method"] == method for r in rows)
        tied = sum(method in r["oracle_tied_methods"] for r in rows)
        domination[method] = {"selected_budgets": selected, "actually_satisfying_selected_budgets": successful,
                              "oracle_budgets": oracle,
                              "oracle_optimal_including_ties_budgets": tied,
                              "never_selected": selected == 0, "oracle_never_selected": oracle == 0,
                              "never_actually_nominal_cost_optimal_including_ties": tied == 0}
        totals[f"selection_share/{method}"] = [selected/len(rows), 1.]
        totals[f"oracle_share/{method}"] = [oracle/len(rows), 1.]
    return {**panel, "budget_axes": axes, "n_budgets": len(rows), "rows": rows,
            "metric_numerator_denominator": totals, "domination": domination,
            "metrics": {k: v[0]/v[1] if v[1] else None for k, v in totals.items()}}


def candidates_by_id(candidates, key):
    return next(r for r in candidates if r["id"] == key)


def estimate(value, samples, n_models):
    finite_samples = np.asarray(samples)[np.isfinite(samples)]
    return {"estimate": value,
            "ci95": audit.interval(finite_samples) if len(finite_samples) else None,
            "n_models": n_models, "undefined_bootstrap_draws": len(samples)-len(finite_samples)}


def aggregate(models, n_boot):
    """Paired whole-model bootstrap, including ratio denominators in every draw.

    Every model contributes unit budget mass despite unequal deduplicated grids.
    Feasible exclusion fractions use feasible compressed configuration-budget
    pairs; conditional nominal gaps use successful selection-budget pairs.
    """
    keys = list(models[0]["metric_numerator_denominator"])
    values = np.array([[m["metric_numerator_denominator"][k] for k in keys] for m in models])
    draws = audit.bootstrap_means(values.reshape(len(models), -1), [m["model"] for m in models],
                                  n_boot=n_boot, seed=SEED).reshape(n_boot, len(keys), 2)
    pooled = values.mean(axis=0)
    point = np.divide(pooled[:, 0], pooled[:, 1], out=np.full(len(keys), np.nan), where=pooled[:, 1] > 0)
    samples = np.divide(draws[:, :, 0], draws[:, :, 1], out=np.full((n_boot, len(keys)), np.nan),
                        where=draws[:, :, 1] > 0)
    metrics = {k: estimate(float(point[i]) if np.isfinite(point[i]) else None, samples[:, i], len(models))
               for i, k in enumerate(keys)}
    comparisons = {}
    for baseline in POLICIES[1:]:
        comparisons[baseline] = {}
        for name, metric, direction in (("constraint_satisfaction_gain", "constraint_satisfaction", 1),
                                         ("nominal_storage_saving", "nominal_storage", -1),
                                         ("nominal_cost_gap_reduction", "nominal_cost_gap_vs_oracle", -1)):
            a, b = keys.index(f"selector/{metric}"), keys.index(f"{baseline}/{metric}")
            comparisons[baseline][name] = estimate(float(direction*(point[a]-point[b])),
                                                    direction*(samples[:, a]-samples[:, b]), len(models))
    domination = {}
    for method in ("dense", *METHODS):
        domination[method] = {
            "selected_budgets": sum(m["domination"][method]["selected_budgets"] for m in models),
            "actually_satisfying_selected_budgets": sum(
                m["domination"][method]["actually_satisfying_selected_budgets"] for m in models),
            "oracle_budgets": sum(m["domination"][method]["oracle_budgets"] for m in models),
            "never_selected": all(m["domination"][method]["never_selected"] for m in models),
            "oracle_never_selected": all(m["domination"][method]["oracle_never_selected"] for m in models),
            "never_actually_nominal_cost_optimal_including_ties": all(
                m["domination"][method]["never_actually_nominal_cost_optimal_including_ties"] for m in models),
            "models_never_selected": [m["model"] for m in models if m["domination"][method]["never_selected"]]}
    return {"n_models": len(models), "n_budgets": sum(m["n_budgets"] for m in models),
            "metrics": metrics, "improvement_vs_baselines": comparisons, "domination": domination,
            "calibration_points_used": 0}


def build_summary(n_boot=10000, quantiles=QUANTILES):
    if n_boot < 2:
        raise ValueError("At least two bootstrap draws are required")
    panels, hashes, excluded, seal = load_panels()
    models = [replay_panel(panel, quantiles) for panel in panels]
    cohorts = {name: aggregate([m for m in models if m["cohort"] == name], n_boot)
               for name in sorted({m["cohort"] for m in models})}
    return {"version": 33, "evaluation": "OFFLINE REPLAY; not prospective", "refit": False,
            "calibration_points_used": 0, "input_sha256": hashes, "v28_freeze_sha256": seal,
            "endpoint": "signed capability CE delta from source dense, native-token nats",
            "reference_policy": "V6 source dense (1.0) for both methods; V10 predictions rebased by V10 dense minus V6 dense",
            "nominal_cost": {"objective": "nominal storage relative to dense 16-bit source",
                             "pruning": "density", "quantization": "bits/16", "dense": 1.,
                             "nominal_active_parameter_ratio": "pruning: density; quantization/dense: 1",
                             "limitations": "Nominal only: ignores sparse indices, packing, scales, untouched tensors and runtime representation. No latency claims."},
            "prediction_policy": {"development_pruning": "V28 saved dev-LOMO Mode A",
                                  "development_quantization": f"V30 broad saved dev-LOMO {QUANT_SHAPE}",
                                  "missing_v30_fold_or_shape": "V28 saved dev-LOMO Mode A fallback; labelled per candidate; never refit",
                                  "separate_transfer": "V28 frozen full-development Mode A for both methods",
                                  "shape_selection": "Fixed in this replay; no predictor, coefficient, margin or hyperparameter selection on scored outcomes",
                                  "extrapolation": "Every measured numeric configuration retained, including pruning below the V28 0.6 training boundary"},
            "budget_policy": {"quantiles": list(quantiles), "construction": "per-model, per-capability observed compressed deltas; nonnegative quantiles plus zero; deduplicated Cartesian product",
                              "scope": "Outcome-informed retrospective scenarios only; budgets do not fit or calibrate predictions",
                              "feasibility_tolerance_nats": TOL,
                              "tie_break": "nominal storage, then method name, then configuration id; no measured outcomes"},
            "bootstrap": {"n_boot": n_boot, "seed": SEED, "unit": "whole model, paired across all budgets, policies, capabilities and methods",
                          "weighting": "equal model budget mass; ratio numerators and denominators resampled together",
                          "limitations": "Conditional on saved fits, observed grids and measurements; no retraining, item, seed or dense-anchor uncertainty. One-model intervals are degenerate."},
            "metric_definitions": {"constraint_satisfaction": "selected actual delta <= tau for every capability",
                                   "nominal_cost_gap_vs_oracle": "selected nominal storage minus minimum actually-feasible nominal storage, signed for ALL decisions; negative gaps may be constraint violations",
                                   "nominal_cost_gap_when_satisfied": "same nominal gap conditional on actual constraint satisfaction; undefined if no successes",
                                   "exclusions": "predicted-infeasible AND actually-feasible / actually-feasible compressed configuration-budget pairs; dense excluded from denominator; per-method and all-compressed",
                                   "domination": "never selected on tested budget/nominal-cost grid; oracle and oracle ties reported separately, not a global Pareto claim"},
            "distillation": {"included": False, "reason": "V12 clean student-own-dense adaptation deltas do not supply a paired source-to-student loss/cost table relative to these source dense models; commercial teachers and student sizes are not interchangeable source checkpoints."},
            "excluded_models": excluded,
            "transfer_identity_caveat": "Legacy Qwen3-8B aggregate tables do not establish Base versus post-trained identity. This separate offline transfer is not validation of the named Qwen/Qwen3-8B-Base V28 target.",
            "cohorts": cohorts, "models": models}


def metric_text(metric):
    return "undefined" if metric["estimate"] is None else audit.with_ci(metric["estimate"], metric["ci95"])


def render(summary):
    main = summary["cohorts"]["development_lomo"]
    metrics = main["metrics"]
    lines = ["# Loss-constrained nominal compression selection (V33)", "",
             "**OFFLINE REPLAY — not prospective.** Saved predictions select a method/configuration; "
             "actual measured capability losses score that selection. No fitting, target compressed calibration, "
             "threshold tuning, or safety-margin tuning occurs in this replay.", "",
             f"The main panel has {main['n_models']} development models and {main['n_budgets']} budget vectors. "
             f"Actual constraint satisfaction is **{metric_text(metrics['selector/constraint_satisfaction'])}**; "
             f"the signed nominal storage gap against the actual oracle is "
             f"**{metric_text(metrics['selector/nominal_cost_gap_vs_oracle'])}**. "
             "Brackets are paired-model bootstrap 95% intervals.", "",
             "## Frozen prediction and measured endpoint", "",
             "Pruning imports V28 `predict_arm` and its saved Mode A development-LOMO coefficients. "
             "Quantization imports V30 `predict_candidate` and saved broad-panel development-LOMO shared-eta "
             "coefficients. The full-development fits contain these 12 targets, so the main panel must use "
             "their already-saved held-out folds. Every mapping excludes the complete target model. "
             "If a saved V30 fold/shape is unavailable, the saved V28 quantization dev-LOMO Mode A fit "
             "is the explicitly labelled fallback; no new fold is fitted. "
             "The shared-eta quantization predictor is a fixed replay choice, not chosen by selection scores; "
             "this replay does not establish that it is the best predictor. There are **0 target compressed "
             "calibration points**, although dense measurements are inputs and other models supplied historical "
             "development outcomes. Shapes and fits remain frozen even when poor or extrapolated.", "",
             "V28's basic inputs are log non-embedding N0, family and measured dense capability loss; "
             "V30 uses log nominal model size parsed from the identifier, family and measured dense "
             "capability loss. No target pruning/quantization loss or target fitted amplitude is an input. "
             "V28 pruning evaluates a_c(x0)((1-density)/0.3)^gamma_c; V30 quantization evaluates "
             "a_c(x0) g(bits)/g(4), with g(b)=2^(-eta(b-4))-2^(-eta(16-4)). "
             "All a mappings, gamma and eta values come from the saved appropriate fold.", "",
             "The common reference is each source's V6 `prune_losses.json[\"1.0\"]`. For both methods, "
             "actual delta_c = measured L_c(config) - L_c(source dense). V10 has its own measurement of "
             "that source dense; its prediction is rebased as delta_hat_common = delta_hat_V10 + "
             "L_dense,V10 - L_dense,V6. This offset uses dense measurements only. The small differences "
             "are exposed below, not silently identified as equal or corrected using compressed outcomes. "
             "Native-token CE nats are compared within each model; cross-model averages are descriptive "
             "despite different tokenizers.", "",
             "All measured numeric density/bit settings enter the candidate set, plus the source dense "
             "candidate with exact zero delta. Missing configurations are not imputed. Pruning below 0.6 "
             "extrapolates the V28 shape fitted on 0.6–0.9; signed predictions and losses, cliffs and infills "
             "are retained. Archived copies and the separate 512-probe panel are not mixed in. "
             "Each candidate, prediction source, extrapolation flag, measured loss, and exclusion is in `summary.json`.", "",
             "## Budgets, nominal costs, and scoring", "",
             f"For each model/capability use quantiles {summary['budget_policy']['quantiles']} of its observed "
             "compressed deltas, clip only these budget thresholds at zero, add zero, deduplicate, and take "
             "the Cartesian product across math/code/QA. Thus capabilities vary independently and the dense "
             "candidate is always feasible. These outcome-informed scenarios make this retrospective, not "
             "a prospective budget distribution. Actual outcomes only construct scenarios and score decisions; "
             "fixed predictions plus a supplied budget determine the selector.", "",
             "Minimize **nominal storage**: pruning = density, quantization = bits/16, dense = 1. "
             "The auxiliary **nominal active-parameter ratio** is density for pruning and 1 for quantization/dense; "
             "it is not the selection objective. These nominal ratios omit sparse indices, packing, scales, "
             "untouched tensors and runtime representation. Saved fake-quantized or masked tensors need not "
             "occupy these nominal sizes. **No latency claims.** Equal nominal storage breaks ties by method "
             "name then configuration id, without measured losses.", "",
             "For each budget the selector chooses the minimum nominal storage satisfying all predicted "
             "delta_hat_c <= tau_c (tolerance 1e-12 nats). The oracle applies the same rule to actual deltas. "
             "Always-cheapest ignores budgets; always-dense selects the exact reference. Constraint satisfaction "
             "requires every actual capability constraint. The signed nominal gap is C_selected - C_oracle "
             "over all decisions. A negative nominal gap can accompany a violation and is not successful "
             "compression. The conditional nominal gap reports only actually satisfying decisions.", "",
             "Wrong exclusion = predicted-infeasible but actually-feasible compressed configuration-budget "
             "pairs / actually-feasible compressed configuration-budget pairs. Dense is omitted from this "
             "denominator. A budget with no feasible compressed candidates has an undefined fraction, not zero; "
             "its zero numerator/denominator still participates correctly in the pooled ratio. This measures "
             "predictor exclusion, so it is not assigned to the two trivial policies.", "",
             f"All intervals use {summary['bootstrap']['n_boot']} paired whole-model bootstrap draws "
             f"(seed {SEED}) through `prediction_audit.bootstrap_means`. Normalize each model to unit budget "
             "mass before pooling, retaining all policies/capabilities/configurations together; resample ratio "
             "numerators and denominators together. Baseline differences use the very same draws. Intervals "
             "condition on frozen fits and fixed observed grids and omit retraining, item/seed and dense-anchor "
             "uncertainty. Within-model rows below are descriptive points, not independent-model confidence intervals.", "",
             "## Main development-LOMO replay", "",
             "| Policy | Actual constraint satisfaction [95% CI] | Mean nominal storage [95% CI] | Signed nominal gap vs actual oracle [95% CI] | Nominal gap when actually satisfying [95% CI] |",
             "|---|---|---|---|---|"]
    for policy in POLICIES:
        lines.append("| " + policy + " | " + " | ".join(metric_text(metrics[f"{policy}/{k}"]) for k in
                     ("constraint_satisfaction", "nominal_storage", "nominal_cost_gap_vs_oracle",
                      "nominal_cost_gap_when_satisfied")) + " |")
    lines += ["", "The advisor's four evidence columns are below. Baseline improvements distinguish constraint "
              "satisfaction from nominal savings; neither is a scalar utility that rewards violating a budget.", "",
              "| Response amplitude context | Improvement vs always-cheapest and always-dense | Calibration points used | Which candidate excluded |",
              "|---|---|---|---|"]
    improvements = main["improvement_vs_baselines"]
    gain = "; ".join(f"vs {b}: satisfaction gain {metric_text(v['constraint_satisfaction_gain'])}, "
                     f"nominal storage saving {metric_text(v['nominal_storage_saving'])}"
                     for b, v in improvements.items())
    exclusions = "; ".join(f"{m} wrongly excluded {metric_text(metrics['exclusions/'+m])}" for m in METHODS)
    never = [m for m in METHODS if main["domination"][m]["never_selected"]]
    lines.append(f"| Mean absolute actual compressed delta "
                 f"{metric_text(metrics['response/all_compressed/mean_abs_actual_delta'])} nats | "
                 f"{gain} | **0** | {exclusions}; method never selected: {', '.join(never) or 'none'}; "
                 "exact configuration IDs and failed capabilities recorded for every budget |")
    lines += ["", "| Method | Mean absolute actual delta, nats [95% CI] | Mean predicted minus actual delta, nats [95% CI] | Actually-feasible configurations wrongly excluded [95% CI] |",
              "|---|---|---|---|"]
    for method in ("all_compressed", *METHODS):
        lines.append(f"| {method} | {metric_text(metrics[f'response/{method}/mean_abs_actual_delta'])} | "
                     f"{metric_text(metrics[f'response/{method}/mean_prediction_minus_actual'])} | "
                     f"{metric_text(metrics[f'exclusions/{method}'])} |")
    lines += ["", "Positive mean prediction-minus-actual indicates overprediction of damage on this panel. "
              "The wrongly-excluded fraction measures its decision consequence directly; it is not inferred "
              "from prediction error alone.", "", "## Method domination on the tested nominal range", "",
              "Here 'dominated' means never selected on the tested budget/nominal-cost grid, as requested. "
              "This is a finite-grid operational finding, not a proof of global Pareto domination. "
              "Oracle counts and optimal ties distinguish predictor rejection from actual nominal-cost "
              "inferiority. No optimal region is assumed for either compression method.", "",
              "| Method | Selector budget count (actually satisfying) | Actual oracle budget count | Never selected by selector? | Never actually nominal-cost optimal, including ties? | Selector share [95% CI] |",
              "|---|---|---|---|---|---|"]
    for method, d in main["domination"].items():
        lines.append(f"| {method} | {d['selected_budgets']} ({d['actually_satisfying_selected_budgets']}) | "
                     f"{d['oracle_budgets']} | {d['never_selected']} | "
                     f"{d['never_actually_nominal_cost_optimal_including_ties']} | "
                     f"{metric_text(metrics['selection_share/'+method])} |")
    prune = main["domination"]["pruning"]
    lines += ["", f"Pruning is never selected for {len(prune['models_never_selected'])} of the "
              f"{main['n_models']} development models. Of its {prune['selected_budgets']} selected budgets, "
              f"{prune['actually_satisfying_selected_budgets']} actually satisfy all constraints. "
              "Selection alone therefore does not establish a successful optimal region."]
    lines += ["", "| Source model | Budgets | Actual satisfaction | Signed nominal gap | Feasible pruning wrongly excluded | Feasible quantization wrongly excluded | Methods never selected |",
              "|---|---|---|---|---|---|---|"]
    for model in summary["models"]:
        if model["cohort"] != "development_lomo":
            continue
        m = model["metrics"]
        never = [k for k, d in model["domination"].items() if d["never_selected"]]
        lines.append(f"| {model['model']} | {model['n_budgets']} | "
                     f"{audit.fmt(m['selector/constraint_satisfaction'])} | "
                     f"{audit.fmt(m['selector/nominal_cost_gap_vs_oracle'])} | "
                     f"{audit.fmt(m['exclusions/pruning'])} | {audit.fmt(m['exclusions/quantization'])} | "
                     f"{', '.join(never) or 'none'} |")
    lines += ["", "## Reference and measurement caveats", "",
              "| Model | V10 dense minus common source dense (math / code / QA), nats | Pruning / quantization configuration counts | Extrapolated configurations |",
              "|---|---|---|---|"]
    for model in summary["models"]:
        gaps = " / ".join(f"{model['quant_dense_minus_source_dense'][c]:.7f}" for c in CAPS)
        counts = " / ".join(str(sum(r["method"] == m for r in model["candidates"])) for m in METHODS)
        lines.append(f"| {model['model']} | {gaps} | {counts} | "
                     f"{sum(r['extrapolated_shape'] for r in model['candidates'])} |")
    lines += ["", "Qwen3-0.6B's saved 5-bit metadata records an odd-half protocol discrepancy "
              "(reported dense-gap QA +0.0133 nats). It remains included and flagged; no invented correction "
              "is applied. Infill metadata is preserved in the JSON. Aggregate loss files cannot verify "
              "item identities or estimate measurement uncertainty, so tight-budget outcomes inherit these limits.", "",
              "## Separate offline transfer and excluded candidates", "",
              summary["transfer_identity_caveat"], ""]
    for model in summary["models"]:
        if model["cohort"] == "development_lomo":
            continue
        lines.append(f"{model['model']}: {model['n_budgets']} budgets, V28 frozen full-development Mode A "
                     "coefficients for both methods, with zero target compressed calibration. V30 contains no "
                     "full-development basic-input mapping for this target, so the available V28 frozen "
                     "quantization mapping is used. This result is kept separate from the 12-model LOMO aggregate. "
                     "Its one-model bootstrap intervals are degenerate and carry no across-model uncertainty.")
        lines += ["", "| Response amplitude context | Improvement vs always-cheapest and always-dense | Calibration points used | Which candidate excluded |", "|---|---|---|---|"]
        extra = summary["cohorts"][model["cohort"]]
        gain = "; ".join(f"vs {b}: satisfaction gain {metric_text(v['constraint_satisfaction_gain'])}, "
                         f"nominal storage saving {metric_text(v['nominal_storage_saving'])}"
                         for b, v in extra["improvement_vs_baselines"].items())
        lines.append(f"| {metric_text(extra['metrics']['response/all_compressed/mean_abs_actual_delta'])} nats | "
                     f"{gain} | **0** | " + "; ".join(
                         f"{m}: wrongly excluded {metric_text(extra['metrics']['exclusions/'+m])}"
                         for m in METHODS) + " |")
        lines += ["", f"Actual satisfaction: {metric_text(extra['metrics']['selector/constraint_satisfaction'])}; "
                  f"signed nominal gap: {metric_text(extra['metrics']['selector/nominal_cost_gap_vs_oracle'])}. "
                  "Methods never selected: " + (", ".join(m for m in METHODS if extra["domination"][m]["never_selected"]) or "none") + ".", ""]
    lines += ["**Distillation excluded:** " + summary["distillation"]["reason"], ""]
    for entry in summary["excluded_models"]:
        lines.append(f"- `{entry['path']}`: {entry['reason']}.")
    lines += ["", "Qwen3-14B currently has pruning results but no paired quantization loss table and no "
              "supported saved basic-input prediction panel; it cannot support the same cross-method "
              "comparison. Duplicate aliases are not additional models.", "", "## Reproduction and artifacts", "",
              "```bash", "python analysis/v33_loss_constrained_selection.py --bootstrap 10000",
              "python analysis/v33_loss_constrained_selection.py --dry-run --bootstrap 10000",
              "python -m pytest -q tests/test_v33.py", "```", "",
              "`results/v33-selection/summary.json` contains candidates, budget axes, every decision, actual "
              "oracle, excluded configuration IDs with failing predicted capabilities, feasible/wrongly-excluded "
              "counts, per-model sufficient statistics, paired intervals, and input/code SHA-256 hashes. "
              "The V28 payload seal is validated; all read inputs are hashed and rechecked before writing. "
              "The dry run writes nothing. Execution is CPU-only and loads no model or GPU framework.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    summary = build_summary(args.bootstrap)
    report = render(summary)
    if args.dry_run:
        print("DRY RUN: no files written\n" + report)
    else:
        audit.write_outputs(summary, report, args.output_dir)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
        print(f"OFFLINE REPLAY written to {args.output_dir / 'summary.json'} and {args.report}")
    return summary


if __name__ == "__main__":
    main()
