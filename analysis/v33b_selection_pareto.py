#!/usr/bin/env python3
"""Measured Pareto and paired-model uncertainty audit; CPU-only, no new runs.

python analysis/v33b_selection_pareto.py [--dry-run] [--bootstrap 10000]
Uses saved V33 decisions/predictions, checks their measured endpoints, and never
fits a predictor or loads a model. Matched reliability is retrospective.
"""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from . import prediction_audit as audit
    from . import v33_loss_constrained_selection as v33
except ImportError:
    import prediction_audit as audit
    import v33_loss_constrained_selection as v33

import numpy as np  # prediction_audit sets CPU thread limits first.

ROOT = audit.ROOT
SOURCE = ROOT / "results/v33-selection/summary.json"
OUT = ROOT / "results/v33b-selection-pareto"
REPORT = ROOT / "paper/docs/SELECTION_PARETO.md"
CAPS = audit.CAPABILITIES
TOL = v33.TOL
SEED = v33.SEED
POLICIES = ("selector", "oracle", "quant_only", "always_cheapest", "always_dense")
ARMS = ("dense", "pruning", "quantization")
TARGETS = (.75, .8, .9, .95, .99, 1.)
DISTILL_RUN = "gpt-5.6-luna_full_600"
DISTILL_LIMIT = (
    "No paired smaller source-to-student compression panel for all 12 sources. "
    "The covered-subset distill_only diagnostic selects a fixed V12 GPT/full/600 "
    "adaptation of the source itself, at nominal storage 1; it is not size reduction "
    "or a commercial-teacher storage ratio. Missing models are not dense-imputed."
)


def dominates(a, b):
    """Strict Pareto dominance: no worse in any measured objective, better in one.

    Keep capabilities separate and signed; predictions never enter this test.
    Numerically equal objective vectors do not dominate one another.
    """
    av = [audit.finite(a["nominal_storage"]), *(audit.finite(a["actual_loss"][c]) for c in CAPS)]
    bv = [audit.finite(b["nominal_storage"]), *(audit.finite(b["actual_loss"][c]) for c in CAPS)]
    return all(x <= y + TOL for x, y in zip(av, bv)) and any(x < y - TOL for x, y in zip(av, bv))


def pareto_frontier(candidates):
    """Exact finite-candidate test, independent of the sampled budget grid.

    At tau=max(0, delta_i-TOL), candidate i first becomes feasible in the nonnegative
    budget domain. A cheaper feasible rival there rules out cost optimality at
    EVERY budget where i is feasible. This is separate from Pareto membership:
    an equal-cost loss-dominated configuration can still tie the minimum cost.
    """
    if not candidates or len({c["id"] for c in candidates}) != len(candidates):
        raise ValueError("Need nonempty candidates with unique IDs")
    rows = []
    for row in candidates:
        dominators = [r["id"] for r in candidates if r["id"] != row["id"] and dominates(r, row)]
        tau = {c: max(0., audit.finite(row["actual_delta"][c]) - TOL) for c in CAPS}
        cheaper = [r["id"] for r in candidates
                   if r["nominal_storage"] < row["nominal_storage"] - TOL
                   and v33.feasible(r["actual_delta"], tau)]
        rows.append({"id": row["id"], "method": row["method"],
                     "nominal_storage": row["nominal_storage"], "actual_loss": row["actual_loss"],
                     "on_frontier": not dominators, "dominated_by": sorted(dominators),
                     "cross_arm_dominators": sorted(r["id"] for r in candidates
                                                   if r["id"] in dominators and r["method"] != row["method"]),
                     "minimal_nonnegative_budget": tau,
                     "cost_optimal_at_any_nonnegative_budget_including_ties": not cheaper,
                     "cheaper_at_minimal_budget": sorted(cheaper)})
    arms = {}
    for arm in sorted({r["method"] for r in rows}):
        subset = [r for r in rows if r["method"] == arm]
        frontier = [r["id"] for r in subset if r["on_frontier"]]
        optimal = [r["id"] for r in subset if r["cost_optimal_at_any_nonnegative_budget_including_ties"]]
        arms[arm] = {"n_configurations": len(subset), "frontier_ids": frontier,
                     "appears_on_actual_frontier": bool(frontier),
                     "all_configurations_dominated": not frontier,
                     "any_budget_cost_optimal_ids": optimal,
                     "cost_optimal_at_any_nonnegative_budget_including_ties": bool(optimal)}
    return {"frontier_ids": [r["id"] for r in rows if r["on_frontier"]],
            "candidates": rows, "arms": arms}


def checked_source():
    """Require the saved V33 lineage and raw measured loss/cost endpoints to agree."""
    hashes = audit.provenance([SOURCE, Path(__file__), Path(audit.__file__), Path(v33.__file__)])
    saved = audit.read_json(SOURCE)
    if saved["version"] != 33:
        raise ValueError("Expected V33 selection artifact")
    current = audit.provenance([ROOT / p for p in saved["input_sha256"]])
    if current != saved["input_sha256"]:
        raise ValueError("V33 input provenance mismatch; do not silently reuse stale decisions")
    hashes.update(current)
    for panel in saved["models"]:
        tables = {m: audit.read_json(ROOT / p) for m, p in panel["source_paths"].items()}
        dense = v33.losses_at(tables["pruning"], "1.0")
        if panel["source_dense"] != dense:
            raise ValueError("Saved source dense disagrees with measured table")
        for row in panel["candidates"]:
            arm = row["method"]
            if arm == "dense":
                actual, cost = dense, 1.
            else:
                keys = [k for k in tables[arm] if not k.startswith("_") and k not in ("dense", "1.0")
                        and float(k) == row["coordinate"]]
                if len(keys) != 1:
                    raise ValueError(f"Missing/duplicate measured endpoint: {panel['model']}/{row['id']}")
                actual = v33.losses_at(tables[arm], keys[0])
                cost = row["coordinate"] if arm == "pruning" else row["coordinate"] / 16.
            if (actual != row["actual_loss"] or cost != row["nominal_storage"]
                    or any(abs(actual[c] - dense[c] - row["actual_delta"][c]) > TOL for c in CAPS)):
                raise ValueError(f"Saved measured loss/cost mismatch: {panel['model']}/{row['id']}")
        axes, budgets = v33.budget_grid(panel["candidates"], saved["budget_policy"]["quantiles"])
        if axes != panel["budget_axes"] or budgets != [r["tau"] for r in panel["rows"]]:
            raise ValueError("Saved V33 budget grid mismatch")
    return saved, hashes


def load_distillation(panel, hashes):
    """Same-model adaptation only; never subtract losses across tokenizers/sizes."""
    path = ROOT / "results/v12-distill" / panel["model"] / DISTILL_RUN / "eval.json"
    if not path.exists():
        return None
    hashes.update(audit.provenance([path]))
    data = audit.read_json(path)
    expected = {"version": 12, "student_tag": panel["model"], "teacher": "gpt-5.6-luna",
                "recipe": "full", "n_per_domain": 600,
                "measurement_benchmarks": {"math": "MATH-500", "code": "MBPP", "qa": "2WikiMultihopQA"},
                "probe_source": "analysis.v6_capability_geometry.build_probes", "probe_seed": 0,
                "n_probe_requested": 128, "probe_half": "measurement (odd indices, v[1::2])",
                "measurement_samples": dict.fromkeys(CAPS, 64)}
    if any(data.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Incompatible fixed distillation identity/protocol: {path}")
    actual = {c: audit.finite(data["post_training"][c]) for c in CAPS}
    for c in CAPS:
        if abs(actual[c] - audit.finite(data["dense"][c]) - audit.finite(data["delta"][c])) > 1e-10:
            raise ValueError(f"Inconsistent paired distillation delta: {path}/{c}")
    return {"id": "distillation:" + DISTILL_RUN, "method": "distillation", "nominal_storage": 1.,
            "actual_loss": actual, "actual_delta": {c: actual[c] - panel["source_dense"][c] for c in CAPS},
            "source_path": str(path.relative_to(ROOT)), "training_mode": data["training_mode"],
            "own_dense_minus_source_dense": {c: data["dense"][c] - panel["source_dense"][c] for c in CAPS},
            "interpretation": "Same-source-size adaptation; nominal merged model storage 1, no compression"}


def score_choice(row, tau, oracle):
    failed = [c for c in CAPS if row["actual_delta"][c] > tau[c] + TOL]
    return {"selected": row["id"], "method": row["method"], "constraint_satisfied": not failed,
            "violated_capabilities": failed, "nominal_storage": row["nominal_storage"],
            "nominal_cost_gap_vs_oracle": row["nominal_storage"] - oracle["nominal_storage"]}


def replay_panel(panel, distill=None):
    base = panel["candidates"]
    candidates = base + ([distill] if distill is not None else [])
    policies = (*POLICIES, "distill_only") if distill is not None else POLICIES
    predictions = v33.prediction_view(base)
    quant = [r for r in predictions if r["method"] in ("dense", "quantization")]
    index = {r["id"]: r for r in candidates}
    rows = []
    for saved_row in panel["rows"]:
        tau = saved_row["tau"]
        legacy = v33.score_budget(base, predictions, tau)
        if legacy != saved_row:
            raise ValueError(f"Saved V33 decision/scoring mismatch: {panel['model']}")
        feasible = [r for r in candidates if v33.feasible(r["actual_delta"], tau)]
        oracle = min(feasible, key=v33.nominal_order)
        choices = {"selector": legacy["policies"]["selector"]["selected"], "oracle": oracle["id"],
                   "quant_only": v33.select(quant, tau),
                   "always_cheapest": min(candidates, key=v33.nominal_order)["id"], "always_dense": "dense"}
        if distill is not None:
            choices["distill_only"] = distill["id"]
        rows.append({"tau": tau, "policies": {p: score_choice(index[k], tau, oracle) for p, k in choices.items()},
                     "exclusion_counts": legacy["exclusion_counts"], "wrongly_excluded": legacy["wrongly_excluded"],
                     "oracle_tied_methods": sorted({r["method"] for r in feasible
                                                    if abs(r["nominal_storage"] - oracle["nominal_storage"]) <= TOL})})
    totals = {}
    for policy in policies:
        scores = [r["policies"][policy] for r in rows]
        success = np.array([r["constraint_satisfied"] for r in scores], dtype=float)
        costs = np.array([r["nominal_storage"] for r in scores])
        gaps = np.array([r["nominal_cost_gap_vs_oracle"] for r in scores])
        for name, values in (("constraint_satisfaction", success), ("nominal_storage", costs),
                             ("nominal_cost_gap_vs_oracle", gaps)):
            totals[f"{policy}/{name}"] = [float(values.mean()), 1.]
        totals[f"{policy}/nominal_cost_gap_when_satisfied"] = [float((gaps * success).mean()), float(success.mean())]
    for arm in ("all_compressed", "pruning", "quantization"):
        counts = [r["exclusion_counts"][arm] for r in rows]
        totals[f"exclusions/{arm}"] = [float(np.mean([r["wrongly_excluded"] for r in counts])),
                                      float(np.mean([r["actually_feasible"] for r in counts]))]
    selection = {}
    for arm in ARMS:
        selected = [r["policies"]["selector"] for r in rows if r["policies"]["selector"]["method"] == arm]
        successful = sum(r["constraint_satisfied"] for r in selected)
        selection[arm] = {"selected_budgets": len(selected), "satisfying_selected_budgets": successful,
                          "oracle_budgets": sum(r["policies"]["oracle"]["method"] == arm for r in rows)}
        totals[f"selector_arm/{arm}/selection_share"] = [len(selected) / len(rows), 1.]
        totals[f"selector_arm/{arm}/constraint_satisfaction"] = [successful / len(rows), len(selected) / len(rows)]
    return {"model": panel["model"], "cohort": panel["cohort"], "n_budgets": len(rows),
            "candidates": candidates, "source_dense": panel["source_dense"],
            "measurement_caveats": panel.get("measurement_caveats", []), "policies": list(policies),
            "pareto": pareto_frontier(candidates), "selector_behavior": selection,
            "metric_numerator_denominator": totals,
            "metrics": {k: n / d if d else None for k, (n, d) in totals.items()}, "rows": rows}


def match_reliability(satisfaction, cost, cheapest_satisfaction, cheapest_cost, target):
    """Uniform randomized mixture; same scalar coin for every model/budget.

    Raise reliability with dense; lower it with cheapest. Never rescue only the
    actually failing decisions. Returns NaNs for unattainable targets, no extrapolation.
    Arrays allow re-estimating the mixture in every paired model-bootstrap draw.
    """
    if not np.isfinite(target) or not 0 <= target <= 1:
        raise ValueError("Target satisfaction must be in [0, 1]")
    s, c, low_s, low_c = np.broadcast_arrays(satisfaction, cost, cheapest_satisfaction, cheapest_cost)
    raise_rate = target > s
    anchor_s = np.where(raise_rate, 1., low_s)
    anchor_c = np.where(raise_rate, 1., low_c)
    same = np.abs(s - target) <= TOL
    denominator = s - anchor_s
    weight = np.divide(target - anchor_s, denominator, out=np.full(s.shape, np.nan),
                       where=np.abs(denominator) > TOL)
    weight = np.where(same, 1., weight)
    valid = np.isfinite(weight) & (weight >= -TOL) & (weight <= 1 + TOL)
    weight = np.where(valid, np.clip(weight, 0., 1.), np.nan)
    return {"nominal_storage": weight * c + (1 - weight) * anchor_c,
            "constraint_satisfaction": weight * s + (1 - weight) * anchor_s,
            "base_policy_probability": weight,
            "anchor": np.where(same, "none", np.where(raise_rate, "always_dense", "always_cheapest"))}


def aggregate(models, n_boot=10000, targets=TARGETS):
    if not models or n_boot < 2 or len({m["model"] for m in models}) != len(models):
        raise ValueError("Need unique nonempty models and at least two bootstrap draws")
    keys = list(models[0]["metric_numerator_denominator"])
    policies = models[0]["policies"]
    if any(list(m["metric_numerator_denominator"]) != keys or m["policies"] != policies for m in models):
        raise ValueError("Policies/metrics must share identical model coverage")
    values = np.array([[m["metric_numerator_denominator"][k] for k in keys] for m in models])
    # Exactly ONE row per model. All statistics and policy contrasts share draws.
    boot = audit.bootstrap_means(values.reshape(len(models), -1), [m["model"] for m in models],
                                 n_boot=n_boot, seed=SEED).reshape(n_boot, len(keys), 2)
    means = values.mean(axis=0)
    point = np.divide(means[:, 0], means[:, 1], out=np.full(len(keys), np.nan), where=means[:, 1] > 0)
    draws = np.divide(boot[:, :, 0], boot[:, :, 1], out=np.full((n_boot, len(keys)), np.nan),
                      where=boot[:, :, 1] > 0)

    def estimate(value, samples):
        return v33.estimate(float(value) if np.isfinite(value) else None, samples, len(models))

    metrics = {k: estimate(point[i], draws[:, i]) for i, k in enumerate(keys)}
    contrasts = {}
    for policy in policies:
        if policy == "selector":
            continue
        contrasts[policy] = {}
        for metric in ("constraint_satisfaction", "nominal_storage", "nominal_cost_gap_vs_oracle"):
            a, b = (keys.index(f"{p}/{metric}") for p in ("selector", policy))
            contrasts[policy][f"selector_minus_baseline/{metric}"] = estimate(point[a] - point[b], draws[:, a] - draws[:, b])
    matched = []
    low_s, low_c = (keys.index("always_cheapest/" + k) for k in ("constraint_satisfaction", "nominal_storage"))
    for target in targets:
        matched_policies, policy_samples = {}, {}
        for policy in policies:
            si, ci = (keys.index(f"{policy}/{k}") for k in ("constraint_satisfaction", "nominal_storage"))
            p = match_reliability(point[si], point[ci], point[low_s], point[low_c], target)
            b = match_reliability(draws[:, si], draws[:, ci], draws[:, low_s], draws[:, low_c], target)
            matched_policies[policy] = {k: estimate(p[k], b[k]) for k in
                                        ("nominal_storage", "constraint_satisfaction", "base_policy_probability")}
            matched_policies[policy]["anchor"] = str(p["anchor"])
            policy_samples[policy] = b["nominal_storage"]
        gaps = {}
        for policy in policies:
            if policy == "selector":
                continue
            a, b = (matched_policies[p]["nominal_storage"]["estimate"] for p in ("selector", policy))
            gaps[policy] = estimate(a - b if a is not None and b is not None else np.nan,
                                    policy_samples["selector"] - policy_samples[policy])
        matched.append({"target_constraint_satisfaction": target, "policies": matched_policies,
                        "selector_minus_baseline_nominal_storage": gaps})
    return {"n_models": len(models), "model_ids": [m["model"] for m in models],
            "n_budgets": sum(m["n_budgets"] for m in models), "policies": policies,
            "metrics": metrics, "paired_contrasts": contrasts, "matched_reliability": matched,
            "actual_frontier": {arm: {
                "models_on_frontier": [m["model"] for m in models if m["pareto"]["arms"].get(arm, {}).get("appears_on_actual_frontier")],
                "models_all_configurations_dominated": [m["model"] for m in models if m["pareto"]["arms"].get(arm, {}).get("all_configurations_dominated")],
                "models_cost_optimal_at_any_nonnegative_budget_including_ties": [
                    m["model"] for m in models if m["pareto"]["arms"].get(arm, {}).get(
                        "cost_optimal_at_any_nonnegative_budget_including_ties")],
                "frontier_configurations": sum(len(m["pareto"]["arms"].get(arm, {}).get("frontier_ids", [])) for m in models)}
                for arm in sorted({c["method"] for m in models for c in m["candidates"]})},
            "selector_behavior": {arm: {key: sum(m["selector_behavior"][arm][key] for m in models)
                                         for key in models[0]["selector_behavior"][arm]} for arm in ARMS}}


def build_summary(n_boot=10000):
    if n_boot < 2:
        raise ValueError("At least two bootstrap draws are required")
    saved, hashes = checked_source()
    models = [replay_panel(p) for p in saved["models"]]
    main = [m for m in models if m["cohort"] == "development_lomo"]
    covered, missing = [], []
    for panel in saved["models"]:
        if panel["cohort"] != "development_lomo":
            continue
        distill = load_distillation(panel, hashes)
        if distill is None:
            missing.append(panel["model"])
        else:
            covered.append(replay_panel(panel, distill))
    cohorts = {"development_lomo": aggregate(main, n_boot)}
    transfer = [m for m in models if m["cohort"] != "development_lomo"]
    # One transfer model: descriptive metrics only, no spurious one-model CIs.
    if covered:
        cohorts["distillation_covered_subset"] = aggregate(covered, n_boot)
    return {"version": "33b", "cpu_only": True, "new_runs": False, "refit": False,
            "evaluation": "OFFLINE measured-outcome audit; not prospective",
            "input_sha256": hashes, "v28_freeze_sha256": saved["v28_freeze_sha256"],
            "nominal_cost": saved["nominal_cost"], "reference_policy": saved["reference_policy"],
            "budget_policy": saved["budget_policy"], "prediction_policy": saved["prediction_policy"],
            "bootstrap": {"n_boot": n_boot, "seed": SEED, "unit": "whole model",
                          "main_n_models": len(main), "weighting": "equal model budget mass",
                          "paired": "all policies, ratio numerators/denominators, and matched mixtures share model draws",
                          "limitations": "Conditional on fixed predictions, grids and measured outcomes; no item/seed, retraining, dense-anchor or frontier-measurement uncertainty. Twelve models, not 2210 independent observations."},
            "definitions": {
                "pareto": "Minimize (nominal storage, measured math CE, measured code CE, measured QA CE) per model; no worse in all, strictly better in at least one; tolerance 1e-12. Identical vectors retain all ties.",
                "any_budget": "Independent of quantile grid: test cheaper feasible rivals at tau_c=max(0, actual_delta_c-1e-12), the minimal budget under the feasibility tolerance. Equal-cost optima are retained, even if loss-dominated; signed-loss Pareto membership is reported separately.",
                "quant_only": "Minimum predicted-feasible nominal cost among quantization and dense using the same frozen V33 predictions and tie rule.",
                "oracle": "Minimum actually-feasible nominal cost over the evaluated candidate set; privileged descriptive reference, not deployable.",
                "distill_only": "Fixed same-model V12 GPT/full/600 adaptation, independent of budget/outcomes; no oracle switching or dense fallback in the raw policy.",
                "wrong_exclusion": "V33 predicted-infeasible AND actually-feasible compressed configuration-budget pairs / actually-feasible pairs; normalize counts by each model's budget count before pooling; zero denominators undefined.",
                "cost_gap": "Signed selected nominal storage minus actual oracle over all decisions; a negative gap may be a failure. Conditional-on-success gap is separate.",
                "matched_reliability": "Uniform mixtures: if target r exceeds policy satisfaction s, mix with dense; if r<s, mix with cheapest; alpha=(r-s_anchor)/(s-s_anchor) is base-policy probability. Expected cost and actual satisfaction are mixed on the same model distribution. Recompute alpha inside every paired bootstrap draw; no extrapolation. These are retrospective randomized policy families, not the original deterministic policies or prospectively calibrated guarantees. Exact target refers to pooled equal-model expected satisfaction, not each model."},
            "distillation": {"main_12_model_compression_baseline_available": False, "reason": DISTILL_LIMIT,
                             "fixed_run": DISTILL_RUN, "covered_models": [m["model"] for m in covered],
                             "missing_models": missing, "nominal_storage": 1.},
            "cohorts": cohorts, "models": main, "distillation_subset_models": covered,
            "separate_offline_transfer": transfer, "transfer_identity_caveat": saved["transfer_identity_caveat"],
            "excluded_models": saved["excluded_models"]}


def metric_text(metric):
    if metric["estimate"] is None:
        return "undefined"
    if metric["ci95"] is None:
        return f"{audit.fmt(metric['estimate'])} [CI undefined]"
    return v33.metric_text(metric)


def policy_table(cohort):
    lines = ["| Policy | Actual satisfaction [95% CI] | Nominal storage [95% CI] | Signed nominal gap vs oracle [95% CI] |",
             "|---|---|---|---|"]
    for policy in cohort["policies"]:
        lines.append("| " + policy + " | " + " | ".join(metric_text(cohort["metrics"][f"{policy}/{k}"])
                     for k in ("constraint_satisfaction", "nominal_storage", "nominal_cost_gap_vs_oracle")) + " |")
    return lines


def matched_table(cohort):
    policies = cohort["policies"]
    lines = ["| Matched satisfaction | " + " | ".join(policies) + " |",
             "|---|" + "---|" * len(policies)]
    for row in cohort["matched_reliability"]:
        lines.append(f"| {row['target_constraint_satisfaction']:.0%} | " + " | ".join(
            metric_text(row["policies"][p]["nominal_storage"]) for p in policies) + " |")
    return lines


def render(summary):
    main = summary["cohorts"]["development_lomo"]
    metrics = main["metrics"]
    pruning = main["actual_frontier"]["pruning"]
    selected = main["selector_behavior"]["pruning"]
    lines = ["# Selection, measured Pareto domination, and model uncertainty (V33b)", "",
             "**Offline existing-data analysis; no new runs, no refitting, no prospective calibration.** "
             "The frozen selector is miscalibrated due to prediction error on this replay. Its pruning failures "
             "do not establish pruning domination.", "",
             f"Across {main['n_models']} development models and {main['n_budgets']} total budget vectors, "
             f"pruning has **{pruning['frontier_configurations']} measured Pareto-frontier configurations across "
             f"{len(pruning['models_on_frontier'])}/{main['n_models']} models**. "
             f"The selector's {selected['selected_budgets']} pruning picks have "
             f"{selected['satisfying_selected_budgets']} actual successes, while feasible pruning configurations "
             f"are wrongly excluded at rate **{metric_text(metrics['exclusions/pruning'])}**. "
             "These are distinct statements about measured candidates and predictor decisions.", "",
             "## Measured frontier and true domination", "",
             summary["definitions"]["pareto"], "",
             "This is a four-objective frontier: loss capabilities are never averaged, and predictions never "
             "enter dominance. Every measured configuration is checked against every other configuration of "
             "the same model, including within-arm rivals. An arm is absent only when all its measured "
             "configurations are dominated; different rivals may dominate different configurations. "
             "This establishes domination only within the available measured candidate set, not every possible "
             "density, bit width, compression algorithm, or noisy population loss.", "",
             summary["definitions"]["any_budget"], "",
             "The minimal budget is a witness: if no strictly cheaper rival is feasible there, the candidate "
             "is cost optimal including ties at that budget; otherwise that rival remains feasible wherever "
             "the candidate is feasible. Negative measured deltas are retained. A frontier point with stronger "
             "negative deltas can still be unnecessary when budgets must be nonnegative. A loss-dominated "
             "point can tie a rival's cost. Thus neither sampled-grid oracle counts nor arbitrary tie breaking "
             "define Pareto membership.", "",
             "| Model | Dense frontier IDs | Pruning frontier IDs | Quantization frontier IDs | Pruning optimal at any nonnegative budget, including ties? |",
             "|---|---|---|---|---|"]
    for model in summary["models"]:
        arms = model["pareto"]["arms"]
        ids = [", ".join(arms[a]["frontier_ids"]) or "none (all dominated)" for a in ARMS]
        lines.append(f"| {model['model']} | " + " | ".join(ids) + " | " +
                     str(arms["pruning"]["cost_optimal_at_any_nonnegative_budget_including_ties"]) + " |")
    lines += ["", "`summary.json` supplies every configuration's measured objective vector, complete dominator "
              "IDs, cross-arm dominator IDs, frontier flag, minimal budget witness and cheaper rivals. "
              "Frontier flags are deterministic conditional on aggregate measurements; bootstrap intervals "
              "do not estimate noise in those measurements.", "",
              "All measured pruning configurations are dominated in: " +
              ", ".join(pruning["models_all_configurations_dominated"]) + ". "
              "Pruning can minimize nominal cost at some nonnegative budget (including ties) in: " +
              ", ".join(pruning["models_cost_optimal_at_any_nonnegative_budget_including_ties"]) + ". "
              "The other frontier appearances involve loss improvements beyond what nonnegative budgets require.",
              "", "## Policies on the unchanged V33 grids", "",
              "The 2210 vectors are spread across the 12 models (unequal grids), not 2210 independent "
              "replicates per model. The saved outcome-informed quantile grids, signed loss deltas from each "
              "V6 source dense, and V33 V28/V30 frozen held-out predictions are reused exactly. Raw V6/V10 "
              "measurements and nominal costs are checked against the saved candidates; saved decisions and "
              "input SHA-256 hashes are checked before reuse. V10 predictions keep V33's dense-anchor rebasing.", "",
              "Selector: cheapest candidate meeting every predicted capability constraint. Quant-only: same "
              "rule restricted to quantization plus dense fallback. Oracle: cheapest actually feasible candidate. "
              "Always-cheapest ignores the budget; always-dense chooses the exact source reference. "
              "Oracle is a privileged reference. Ties use nominal storage, method name, configuration ID. "
              "Every reported satisfaction score uses all three actual constraints.", "",
              "**Nominal storage** is pruning density, quantization bits/16, and dense 1, relative to the "
              "source at 16 bits. These ratios omit sparse indices, quantization metadata, untouched tensors, "
              "packing and runtime representation. **No latency claims.**", "",
              *policy_table(main), "",
              "Distill-only on the full 12-model compression panel: **N/A**. " + summary["distillation"]["reason"], "",
              "Signed gaps above use all decisions; lower cost obtained by violating a constraint is not a "
              "successful saving. Conditional-on-success gaps and paired policy differences are in JSON. "
              f"For the selector the conditional gap is {metric_text(metrics['selector/nominal_cost_gap_when_satisfied'])}.", "",
              "| Frozen-predictor exclusion | Rate [95% model CI] |", "|---|---|"]
    for arm in ("all_compressed", "pruning", "quantization"):
        lines.append(f"| {arm} | {metric_text(metrics['exclusions/'+arm])} |")
    lines += ["", summary["definitions"]["wrong_exclusion"], "",
              f"Selected pruning satisfaction is {metric_text(metrics['selector_arm/pruning/constraint_satisfaction'])}; "
              f"{metrics['selector_arm/pruning/constraint_satisfaction']['undefined_bootstrap_draws']} of "
              f"{summary['bootstrap']['n_boot']} bootstrap draws contain no pruning picks and are undefined "
              "for this conditional statistic. The zero observed successes are not a population guarantee.", "",
              "## Nominal cost at matched actual satisfaction", "",
              summary["definitions"]["matched_reliability"], "",
              "Each cell below is expected nominal storage [95% model CI] at the row's exact expected "
              "satisfaction, using all decisions. The mixture coin is independent of the model, budget and "
              "whether a decision actually fails. No outcome-dependent rescue of individual failures is used. "
              "Mixture weights use replay outcomes and must not be described as deployable calibration. "
              "Matched-rate intervals are degenerate by construction after re-estimating weights; they "
              "are not uncertainty intervals for the reliability of a fixed deployed mixture. "
              "Always-dense and always-cheapest trace the same two-anchor mixture; their equality is intentional. "
              "Below 100%, even the oracle mixture deliberately permits failures via the cheapest anchor.", "",
              *matched_table(main), "",
              "| Matched satisfaction | Selector minus oracle nominal storage [paired 95% CI] | Selector minus quant-only [paired 95% CI] |",
              "|---|---|---|"]
    for row in main["matched_reliability"]:
        gaps = row["selector_minus_baseline_nominal_storage"]
        lines.append(f"| {row['target_constraint_satisfaction']:.0%} | {metric_text(gaps['oracle'])} | {metric_text(gaps['quant_only'])} |")
    lines += ["", "At 100% the chosen mixing construction can collapse an imperfect policy to dense. "
              "That is a property of this uniform-fallback family, not a lower bound on what a future "
              "calibrated selector could achieve. Mixture probabilities, anchors, achieved rates and all "
              "paired cost differences are stored in JSON; unreachable targets/draws are undefined, never extrapolated.", "",
              "## Distill-only: common-coverage adaptation diagnostic", "",
              summary["distillation"]["reason"], "",
              "Use the fixed teacher gpt-5.6-luna, full recipe, 600 traces per domain; no choice of run by "
              "outcome. The post-training candidate is the same source checkpoint size, evaluated against "
              "that source's V6 dense loss. It is selected on every budget for distill-only. The subset oracle "
              "also sees this additional candidate; the frozen selector/quant-only candidate sets stay as in V33. "
              "All policies in the following tables use exactly the same covered models and budgets. "
              "Nominal merged model storage is 1; adapter overhead and training cost are omitted.", "",
              "Missing fixed-run models: " + ", ".join(summary["distillation"]["missing_models"]) + ".", ""]
    if "distillation_covered_subset" in summary["cohorts"]:
        subset = summary["cohorts"]["distillation_covered_subset"]
        frontier = subset["actual_frontier"]["distillation"]
        lines += [f"Covered sources: {subset['n_models']} ({', '.join(subset['model_ids'])}); "
                  f"{subset['n_budgets']} budgets, bootstrap over these {subset['n_models']} models.", "",
                  *policy_table(subset), "", "Matched expected nominal storage [95% model CI]:", "",
                  *matched_table(subset), "",
                  f"The same-size adapted candidate is on the expanded measured frontier in "
                  f"{len(frontier['models_on_frontier'])}/{subset['n_models']} covered models, and can "
                  f"tie minimum nominal cost at some nonnegative budget in "
                  f"{len(frontier['models_cost_optimal_at_any_nonnegative_budget_including_ties'])} models. "
                  "Loss improvements can preserve its frontier membership even though always-dense has "
                  "higher aggregate constraint satisfaction at the same nominal cost. Expanded frontiers "
                  "and witnesses for every covered model are in JSON.", ""]
    lines += ["V12 identity, benchmarks, probe source/seed, odd-half convention and counts are validated. "
              "The JSON retains V12 own-dense minus V6 source-dense offsets; raw post-training losses are "
              "used without an invented compressed-loss correction. Aggregate artifacts cannot verify item "
              "identities, so this remains a measurement-qualified adaptation diagnostic, not proof of "
              "source-to-smaller-student compression. No losses from different model tokenizers are subtracted.", "",
              "## Model-level uncertainty and limits", "",
              f"All headline satisfaction, nominal cost/gap, wrong-exclusion and policy-difference intervals "
              f"use {summary['bootstrap']['n_boot']} paired whole-model bootstrap draws, seed {SEED}, "
              "through `prediction_audit.bootstrap_means`. One sufficient-statistics row per model gives "
              "each model unit budget mass, regardless of its grid size. Resample all policies and ratio "
              "numerators/denominators together. Each draw recomputes matched-reliability mixture weights "
              "and cost differences. Undefined ratio or unattainable matching draws are counted explicitly. "
              "V33 already used whole-model resampling; V33b retains and extends it, rather than claiming "
              "that its existing CIs were based on independent budgets.", "",
              "The 12 development models are a small, heterogeneous convenience panel. Shared frozen LOMO "
              "training folds are conditioned on, not independently retrained in the bootstrap. Intervals "
              "describe variation across these model clusters; they omit item, seed, dense-anchor, "
              "measurement, fit and scenario-distribution uncertainty. Within-model budget duplication "
              "does not create additional model evidence. No correction for multiple comparisons is applied.", "",
              "The original measurement caveats remain: Qwen3-0.6B 5-bit metadata flags an odd-half "
              "protocol discrepancy; dense offsets, pruning infills and extrapolated frozen shapes are "
              "retained. Tight-budget decisions and measured frontier flags inherit those limitations. "
              "Missing configurations are not imputed, and archived/512-probe panels are not pooled.", "",
              "Separate Qwen3-8B offline transfer is retained in JSON as a descriptive one-model result, "
              "excluded from the 12-model CIs. " + summary["transfer_identity_caveat"], "",
              "## Conclusion", "",
              "**Established on this replay:** the frozen selector is miscalibrated due to predictor error, "
              "both accepting infeasible candidates and wrongly excluding feasible ones. "
              f"**Measured domination test:** pruning survives on the actual loss/nominal-cost Pareto "
              f"frontier in {len(pruning['models_on_frontier'])} of {main['n_models']} models. "
              "Consequently a blanket claim that pruning is dominated is unsupported. Per-model and "
              "per-configuration domination findings are explicit above and in JSON. Matched-rate nominal "
              "costs describe retrospective mixtures, not a calibrated future selector or measured latency.", "",
              "## Reproduce", "", "```bash",
              "python analysis/v33b_selection_pareto.py --bootstrap 10000",
              "python analysis/v33b_selection_pareto.py --dry-run --bootstrap 10000",
              "python -m pytest -q tests/test_v33b.py tests/test_v33.py", "```", "",
              "Outputs: `results/v33b-selection-pareto/summary.json`, its `report.md`, and "
              "`paper/docs/SELECTION_PARETO.md`. Input/code hashes are recorded and rechecked before "
              "writing. Dry-run writes nothing. No model inference, fitting, training or GPU imports occur.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args(argv)
    summary = build_summary(args.bootstrap)
    report = render(summary)
    if args.dry_run:
        print("DRY RUN: no files written")
        print(report)
    else:
        audit.write_outputs(summary, report, OUT)
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(report)
        print(f"Wrote {OUT / 'summary.json'} and {REPORT}")


if __name__ == "__main__":
    main()
