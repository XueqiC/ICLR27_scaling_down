#!/usr/bin/env python3
"""A4: one CPU-only residual correction of the frozen V92 median curve.

Run: python3 -B analysis/a4_shrunk_source_correction.py
Reads existing JSON scalars only. No model loading, training, measurement, or
new dense statistics. The only writes are the three requested report artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

# Process-local guards; never query an accelerator or change other processes.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from analysis import v92_input_comparison as v92  # noqa: E402

PUBLISHED = Path("results/v92-input-comparison/summary.json")
PUBLISHED_SHA256 = "29383f1b0a12dc2bd7c28479e8dd0257b72333f56b029812d9c3a0bec1d79b39"
OUT = Path("results/a4-shrunk-source")
REPORT = Path("paper/docs/PQ_SHRUNK_CORRECTION_REPORT.md")
BUDGET = "dense_statistics"
SCHEMES = ("leave_one_source_state_out", "leave_one_size_out")
# Fixed before evaluating A4. No outer-score grid, budget, or form search.
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0, 1000000.0, float("inf"))
MAJORITY = 5
CORRECTIONS = (
    "The orthogonal-residual variance finding is not a demonstration that the "
    "residual is random noise. It is not a pre-compression predictor because "
    "computing it needs the compressed model. The scalar response residual "
    "fitted here is a different quantity; predictors consume only V92's "
    "pre-compression inputs."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def alpha_value(alpha):
    value = float("inf") if alpha == "infinity" else float(alpha)
    require(not np.isnan(value) and value >= 0, "Invalid shrinkage strength")
    return value


def alpha_json(alpha):
    return "infinity" if np.isinf(alpha_value(alpha)) else float(alpha)


def correction_coefficients(x, residual, alpha):
    """Minimize mean squared residual error + alpha * ||beta||^2.

    Unlike V92.linear_fit, ALL coefficients are penalized, including intercept.
    This is necessary for the exact zero-correction infinite-shrinkage limit.
    No V92 function is copied.
    """
    alpha = alpha_value(alpha)
    require(len(residual) > 0 and np.isfinite(x).all()
            and np.isfinite(residual).all(), "Invalid residual regression input")
    if np.isinf(alpha):
        return np.zeros(x.shape[1])
    augmented = np.vstack((x, np.sqrt(len(residual) * alpha) * np.eye(x.shape[1])))
    target = np.r_[residual, np.zeros(x.shape[1])]
    result = np.linalg.lstsq(augmented, target, rcond=None)[0]
    require(np.isfinite(result).all(), "Nonfinite correction coefficients")
    return result


def prepare(training, median):
    """Freeze the baseline first; regress only its original-response residual."""
    queries = [r.query for r in training]
    require(median["form"] == "median_curve" and median["budget"] == "K0",
            "Correction requires the unchanged K0 median curve")
    require(median["training_ids"] == [q.row_id for q in queries],
            "Median training membership mismatch")
    scaler = v92.Standardizer.fit(queries, BUDGET)
    residual = np.array([r.target for r in training]) - v92.predict(median, queries)
    return scaler, scaler.transform(queries, BUDGET), residual


def choose_shrinkage(training, group="state"):
    """Nested grouped MAE selection accepts only outer-training observations.

    Each inner median sees ORIGINAL inner-training targets, once. It stays fixed
    for every alpha. Validation targets enter scoring only, never the median,
    residual fit, or scaler. The outer median is not reused inside CV because
    that would expose inner-validation outcomes to the baseline.
    """
    require(group in ("state", "size"), "Unknown inner grouping")
    require(len({getattr(r.query, group) for r in training}) >= 2,
            "Shrinkage selection needs at least two inner groups")
    scheme = SCHEMES[0 if group == "state" else 1]
    errors, validation_queries, audits = [[] for _ in ALPHAS], [], []
    for held, tr, va in v92.make_folds(training, scheme):
        median = v92.fit(tr, "K0", "median_curve")
        scaler, x, residual = prepare(tr, median)
        queries = [r.query for r in va]
        baseline = np.array(v92.predict(median, queries))
        xv = scaler.transform(queries, BUDGET)
        y = np.array([r.target for r in va])
        for i, alpha in enumerate(ALPHAS):
            # Return baseline directly at infinity, avoiding even rounding drift.
            prediction = baseline if np.isinf(alpha) else (
                baseline + xv @ correction_coefficients(x, residual, alpha))
            errors[i].extend(abs(y - prediction))
        validation_queries.extend(queries)
        audits.append({"held_out": held, "validation_ids": [q.row_id for q in queries],
                       "median": median, "standardizer": scaler.audit()})
    scores = [v92.state_mean(error, validation_queries) for error in errors]
    best = min(range(len(ALPHAS)), key=lambda i: (scores[i], -ALPHAS[i]))
    return ALPHAS[best], {"group": group, "alpha_grid": [alpha_json(a) for a in ALPHAS],
                         "mae_nats": scores, "folds": audits,
                         "tie_break": "stronger shrinkage on exact MAE ties"}


def fit(training, inner_group="state", *, median=None, alpha=None):
    require(bool(training), "Empty correction training set")
    require(len({(r.query.arm, r.query.capability) for r in training}) == 1,
            "Fit exactly one arm and capability")
    median = deepcopy(median) if median is not None else v92.fit(training, "K0", "median_curve")
    frozen = deepcopy(median)
    scaler, x, residual = prepare(training, median)
    selection = None
    if alpha is None:
        alpha, selection = choose_shrinkage(training, inner_group)
    coefficients = correction_coefficients(x, residual, alpha)
    require(median == frozen, "Median changed during residual fitting")
    return {"budget": BUDGET, "form": "frozen_median_plus_shrunk_residual",
            "median": median, "training_ids": [r.query.row_id for r in training],
            "standardizer": scaler.audit(), "alpha": alpha_json(alpha),
            "coefficients": coefficients.tolist(), "selection": selection,
            "residual_targets": residual.tolist()}


def predict(model, queries):
    # V92 Query contains no response; its allowlist enforces the input budget.
    baseline = v92.predict(model["median"], queries)
    if np.isinf(alpha_value(model["alpha"])):
        return baseline
    audit = model["standardizer"]
    scaler = v92.Standardizer(np.array(audit["center"]), np.array(audit["scale"]),
                              tuple(audit["training_ids"]))
    result = np.asarray(baseline) + scaler.transform(queries, BUDGET) @ model["coefficients"]
    require(np.isfinite(result).all(), "Nonfinite correction prediction")
    return result.tolist()


def read_published(root=ROOT):
    path = Path(root) / PUBLISHED
    raw = path.read_bytes()
    published = json.loads(raw)
    require(published["version"] == "V92", "Expected published V92 summary")
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == PUBLISHED_SHA256, "Published V92 snapshot changed since A4 was fixed")
    return published, {"path": str(PUBLISHED), "sha256": digest}


def published_ids(published, node, kind):
    ref = node[f"{kind}_rows_ref"]
    return [published["observations"][i]["id"] for i in published["audit_row_sets"][ref]]


def validate_panel(rows, provenance, published, root=ROOT):
    require(provenance["input_sha256"] == published["provenance"]["input_sha256"],
            "V92 input hashes changed; comparison must use its published panel")
    require(sha256(Path(root) / "analysis/v92_input_comparison.py") ==
            published["provenance"]["code_sha256"]["analysis/v92_input_comparison.py"],
            "V92 implementation changed since publication")
    saved = [r for r in published["observations"] if r["primary"]]
    require([r.query.row_id for r in rows] == [r["id"] for r in saved],
            "V92 primary panel/order mismatch")
    for row, old in zip(rows, saved):
        q = row.query
        require(all(getattr(q, key) == old[key] for key in
                    ("state", "size", "arm", "capability", "config"))
                and row.target == old["target"]
                and {k: f.value for k, f in q.inputs.items()} == old["precompression_inputs"],
                f"V92 observation changed: {q.row_id}")
    require(published["protocol"]["bootstrap_replicates"] == v92.BOOTSTRAPS
            and published["protocol"]["seed"] == v92.SEED,
            "V92 bootstrap protocol changed")
    for arm in v92.ARMS:
        require(list(v92.fields(arm, BUDGET)) == published["protocol"]["budgets"][BUDGET][arm],
                "V92 information budget changed")


def evaluate(rows, arm, capability, scheme, published):
    rows = [r for r in rows if (r.query.arm, r.query.capability) == (arm, capability)]
    old = published["evaluations"][arm][scheme]["capabilities"][capability]
    # Both comparison scores and OOF baseline predictions come from publication.
    saved = old["fits"]["K0/median_curve"]
    folds = list(v92.make_folds(rows, scheme))
    tests = [r for _, _, test in folds for r in test]
    require([r.query.row_id for r in tests] == old["row_order"], "V92 scoring order mismatch")
    require(len({r.query.row_id for r in tests}) == len(rows) == len(tests),
            "Every primary observation must be scored exactly once")
    require(len(folds) == len(saved["folds"]) and saved["coverage"] == 1,
            "V92 fold count or baseline coverage mismatch")
    predictions, base_check, audit = [], [], []
    for (held, train, test), old_fold in zip(folds, saved["folds"]):
        require(held == old_fold["fold"], "V92 held-out group mismatch")
        require([r.query.row_id for r in test] == published_ids(published, old_fold, "test"),
                "V92 test membership mismatch")
        baseline = deepcopy(old_fold["model"])
        baseline["training_ids"] = published_ids(published, baseline, "training")
        baseline.pop("training_rows_ref")
        # Verification uses the original fitter on ORIGINAL responses, never
        # residuals. This does not replace the saved prediction/MAE comparator.
        require(baseline == v92.fit(train, "K0", "median_curve"), "Published median mismatch")
        model = fit(train, "size" if scheme == SCHEMES[1] else "state", median=baseline)
        queries = [r.query for r in test]
        predictions.extend(predict(model, queries))
        base_check.extend(v92.predict(model["median"], queries))
        require(model["median"] == baseline, "Published median was modified")
        audit.append({"fold": held, "test_ids": [q.row_id for q in queries], "model": model})
    require(base_check == saved["predictions"], "Median predictions differ from publication")
    queries = [r.query for r in tests]
    mae = v92.clustered([abs(r.target - p) for r, p in zip(tests, predictions)], queries)
    gain = v92.paired(tests, saved["predictions"], predictions)
    limit = v92.paired(tests, saved["predictions"], saved["predictions"])
    require(limit["estimate"] == 0 and limit["ci95"] == [0.0, 0.0],
            "Infinite-shrinkage gain must be exactly zero")
    require(np.isclose(saved["mae_nats"]["estimate"] - mae["estimate"], gain["estimate"],
                       rtol=0, atol=1e-12), "Published baseline and paired MAE disagree")
    counts = Counter(str(f["model"]["alpha"]) for f in audit)
    return {"arm": arm, "capability": capability, "scheme": scheme,
            "row_order": old["row_order"], "median_mae_nats": deepcopy(saved["mae_nats"]),
            "median_predictions": saved["predictions"], "shrunk_mae_nats": mae,
            "predictions": predictions, "paired_gain_nats": gain,
            "positive_gain_ci_excludes_zero": bool(gain["ci95"][0] > 0),
            "selected_shrinkage_counts": dict(counts),
            "maximum_shrinkage_folds": counts.get("infinity", 0), "n_folds": len(folds),
            "maximum_shrinkage_in_most_folds": counts.get("infinity", 0) > len(folds) / 2,
            "mean_absolute_correction_nats": v92.state_mean(
                abs(np.array(predictions) - saved["predictions"]), queries),
            "infinite_shrinkage": {"median_predictions_exact": True,
                                   "mae_nats": deepcopy(saved["mae_nats"]), "paired_gain_nats": limit},
            "published_whole_response_mae_nats": {
                form: deepcopy(old["fits"][f"{BUDGET}/{form}"]["mae_nats"])
                for form in v92.LINEAR_FORMS}, "folds": audit}


def decision(evaluations):
    by_split = {scheme: sum(e["positive_gain_ci_excludes_zero"] for e in evaluations
                            if e["scheme"] == scheme) for scheme in SCHEMES}
    both = sum(all(next(e for e in evaluations if
                       (e["arm"], e["capability"], e["scheme"]) == (arm, cap, scheme))
                   ["positive_gain_ci_excludes_zero"] for scheme in SCHEMES)
               for arm in v92.ARMS for cap in v92.CAPS)
    maximum = sum(e["maximum_shrinkage_folds"] for e in evaluations)
    total = sum(e["n_folds"] for e in evaluations)
    prefers_median = maximum > total / 2
    closed = by_split[SCHEMES[0]] < MAJORITY or prefers_median
    return {"status": "CLOSED" if closed else "MAJORITY_CRITERION_MET",
            "statement": "This branch is CLOSED." if closed else "The fixed majority criterion is met.",
            "primary_split": SCHEMES[0], "required_positive_pairs": MAJORITY,
            "n_arm_capability_pairs": 9, "positive_pairs_by_split": by_split,
            "positive_pairs_on_both_splits": both,
            "maximum_shrinkage_folds": maximum, "total_folds": total,
            "maximum_shrinkage_in_most_folds": prefers_median,
            "selection_reading": "The data prefers the median curve." if prefers_median else
            "The data selects finite corrections in most folds; this alone does not establish held-out gain."}


def build_summary(root=ROOT):
    root = Path(root)
    published, citation = read_published(root)
    rows, extra, provenance = v92.load_data(root)
    validate_panel(rows, provenance, published, root)
    coverage = {arm: {"rows": sum(r.query.arm == arm for r in rows),
                      "source_states": len({r.query.state for r in rows if r.query.arm == arm})}
                for arm in v92.ARMS}
    published_helped = sum(v["helped"] for v in published["verdicts"])
    published_lost = sum(all(c["fits"][f"{BUDGET}/{form}"]["mae_nats"]["estimate"] >
                             c["fits"]["K0/median_curve"]["mae_nats"]["estimate"]
                             for form in v92.LINEAR_FORMS)
                         for arm in v92.ARMS for c in
                         published["evaluations"][arm][SCHEMES[0]]["capabilities"].values())
    evaluations = [evaluate(rows, arm, cap, scheme, published)
                   for scheme in SCHEMES for arm in v92.ARMS for cap in v92.CAPS]
    return {"stage": "A4", "development_only": True, "cpu_only": True,
            "target": published["target"], "published_v92": citation,
            "publication_audit": {
                "numeric_panel_by_arm": coverage,
                "numeric_same_form_statistics_helped_pairs": published_helped,
                "numeric_both_whole_response_candidates_lose_to_median_pairs": published_lost,
                "requested_historical_statistics_helped_pairs": 3,
                "stale_grouped_coverage_note": published["protocol"]["notes"][0],
                "resolution": "Use the exact pinned published numeric artifact: 459 rows, nine states per arm, 4/9 same-form qualifying pairs, 9/9 losses to the median. Its prose and legacy loader test still say six grouped states/378 rows. Do not reconstruct a different six-state baseline or silently change published scores.",
                "legacy_v92_test_failure": "tests/test_v92_input_comparison.py::test_real_loader_uses_only_requested_sources_and_common_development_grid assumes 162 grouped rows and 51 input files; the pinned publication and unchanged loader both have 243 grouped rows and 54 files."},
            "protocol": {
                "budget": BUDGET, "inputs": published["protocol"]["budgets"][BUDGET],
                "schemes": SCHEMES, "primary_split": SCHEMES[0],
                "alpha_grid": [alpha_json(a) for a in ALPHAS],
                "objective": "mean((target - frozen_median(config) - X beta)^2) + alpha * ||beta||^2",
                "design": "V92 raw inputs and training-only standardizer, including intercept; all coefficients penalized; no interactions or nonlinear expansion",
                "selection": "inner grouped CV source-weighted MAE; state within source split, size within size split; exact ties prefer stronger shrinkage",
                "median": "published outer fold median kept unchanged; inner medians fitted once to original inner-training responses, frozen across the entire grid",
                "scoring": "V92 source-state-weighted MAE and paired source-state bootstrap, including size holdouts",
                "bootstrap_replicates": v92.BOOTSTRAPS, "seed": v92.SEED,
                "decision_rule": "At least 5/9 primary (V92 LOSO) pairs must have positive paired gains with 95% intervals excluding zero; otherwise this branch is CLOSED. A majority of maximum-shrinkage selections also ends the branch at the median.",
                "grid_and_rule_fixed_before_a4_evaluation": True,
                "limits": {"infinite_shrinkage": "beta=0 exactly; prediction=published median; gain=0 and CI=[0,0] by construction",
                           "zero_shrinkage": "unpenalized least-squares fit to the frozen median residual, not searched in this strongly shrunk check"},
                "scope": "One correction family, one existing dense_statistics budget, two existing splits; no tuning using outer outcomes",
                "reused_v92_functions": ["load_data", "fields", "budget_inputs (via design/predict)",
                    "Standardizer", "design", "fit (median_curve only)", "predict", "make_folds",
                    "state_mean", "clustered", "paired", "bootstrap_weights"],
                "copied_v92_functions": [],
                "new_solver_reason": "V92.linear_fit leaves the intercept unpenalized, so it cannot give the required zero-correction limit; the A4 solver penalizes the whole residual coefficient vector.",
                "panel_rows": len(rows), "excluded_extra_pruning_rows": len(extra),
                "coverage_notes": [published["protocol"]["notes"][1]],
                "uncertainty_limits": "20,000 paired percentile resamples of nine source states per arm with fixed out-of-fold fits; no refitting uncertainty or architecture-transfer claim; size holdouts still cluster by source state, exactly as V92"},
            "provenance": {**provenance, "code_sha256": {name: sha256(root / name) for name in (
                "analysis/v92_input_comparison.py", "analysis/a4_shrunk_source_correction.py",
                "tests/test_a4_shrunk_source_correction.py")}},
            "observations": [r for r in published["observations"] if r["primary"]],
            "evaluations": evaluations, "decision": decision(evaluations),
            "earlier_corrections": CORRECTIONS}


def strength_counts(e):
    return ", ".join(f"{'∞' if a == 'infinity' else f'{float(a):g}'}×{n}"
                     for a, n in sorted(e["selected_shrinkage_counts"].items(),
                                        key=lambda item: alpha_value(item[0])))


def summary_table(summary):
    lines = ["| Split | Arm | Capability | V92 median MAE | Shrunk MAE | Paired gain | 95% CI | Selected α × folds |",
             "|---|---|---|---:|---:|---:|---|---|"]
    for e in summary["evaluations"]:
        gain = e["paired_gain_nats"]
        # Significant digits keep tiny finite effects distinct from exact zero.
        interval = f"[{gain['ci95'][0]:.6g}, {gain['ci95'][1]:.6g}]"
        lines.append(f"| {'source' if e['scheme'] == SCHEMES[0] else 'size'} | {e['arm']} | {e['capability']} | "
                     f"{v92.fmt(e['median_mae_nats']['estimate'])} | {v92.fmt(e['shrunk_mae_nats']['estimate'])} | "
                     f"{gain['estimate']:.6g} | {interval} | {strength_counts(e)} |")
    return "\n".join(lines)


def markdown(summary):
    d, p = summary["decision"], summary["protocol"]
    median_preferred = [f"{e['arm']}/{e['capability']} ({'source' if e['scheme'] == SCHEMES[0] else 'size'})"
                        for e in summary["evaluations"] if e["maximum_shrinkage_in_most_folds"]]
    whole_response_wins = sum(all(e["shrunk_mae_nats"]["estimate"] < old["estimate"]
                                  for old in e["published_whole_response_mae_nats"].values())
                              for e in summary["evaluations"])
    lines = ["# A4 — shrunk source correction for pruning and quantization", "",
             f"**{d['statement']}**", "",
             f"Positive paired gains with 95% intervals excluding zero: "
             f"**{d['positive_pairs_by_split'][SCHEMES[0]]}/9** leave-one-source-state-out pairs "
             f"(primary; requires at least 5/9), **{d['positive_pairs_by_split'][SCHEMES[1]]}/9** "
             f"leave-one-size-out pairs, and **{d['positive_pairs_on_both_splits']}/9** on both splits.", "",
             f"Maximum shrinkage (α = ∞) was selected in **{d['maximum_shrinkage_folds']}/{d['total_folds']}** "
             f"outer folds. **{d['selection_reading']}**", "",
             ("Within individual comparisons, the data prefers the median curve (∞ in most folds) "
              "for: " + "; ".join(median_preferred) + ".") if median_preferred else
             "No individual comparison selected ∞ in most folds.", "",
             "## Fixed predictor and selection", "",
             "`prediction = unchanged V92 median(configuration) + X beta`", "",
             "The correction minimizes `mean((target − median − X beta)^2) + α ||beta||²`. "
             "It is fitted only to the frozen median's residual. X is V92's unchanged standardized "
             "raw linear design: N0, D0, compression configuration, arm-local L0, and the existing B, V, W "
             "dense descriptors, plus intercept. This is one fixed dense_statistics-budget candidate. "
             "There are no additional input statistics, transformations, or interactions. Configuration "
             "and intercept coefficients belong only to the penalized additive correction; the saved "
             "median anchors never change. All coefficients, including the intercept, are penalized.", "",
             "The grid was fixed before A4 evaluation: **1, 10, 100, 1,000, 10,000, 1,000,000, ∞**. "
             "Only grouped inner-training CV MAE selects α, separately per outer fold; exact ties favor "
             "stronger shrinkage. Source holdouts use inner source folds; size holdouts use inner size folds. "
             "Each inner fold computes its own median once from original inner-training responses, "
             "then freezes it across the grid. No median is ever fitted to residuals. Inner validation "
             "outcomes cannot enter the median, scaler, or residual fit. The outer median is obtained "
             "before selecting/fitting the correction and remains unchanged.", "",
             "The decision rule was fixed before A4 outcomes: fewer than five of nine primary "
             "arm/capability pairs with positive gains and intervals excluding zero closes this branch. "
             "The primary split remains V92's leave-one-source-state-out split; size results are reported "
             "separately and never substituted to choose a favorable split. Maximum shrinkage in most "
             "folds also ends the branch with the median.", "",
             "## Identical panel, published comparator, and scoring", "",
             "**Publication discrepancy:** the current saved V92 numeric artifact has **459 primary "
             "rows, nine states in every arm, and 4/9 qualifying same-form statistics pairs**. "
             "Its coverage prose and legacy loader test still describe six grouped states and "
             "378 rows; the task's 3/9 historical count differs from the current saved verdicts. "
             "Both whole-response candidates still lose to the median in **9/9** pairs. "
             "A4 compares directly with the current saved numeric panel and scores, pinned by SHA256 "
             f"`{summary['published_v92']['sha256']}`. It does not reconstruct a different historical "
             "six-state baseline. This discrepancy changes the coverage description, not A4's "
             "fixed predictor, grid, scoring, or decision rule.", "",
             *p["coverage_notes"], "",
             "459 existing primary response rows are reused (108 pruning, 243 grouped quantization, "
             "108 per-channel quantization), across math, code, and QA. No extra-strength rows are scored. "
             "The V91 descriptors and historical responses are read from existing JSON only. "
             "This analysis uses CPU NumPy with one BLAS thread; it loads no model, trains no model, "
             "accesses no GPU, and performs no new measurement or dense-statistic computation.", "",
             "The comparator MAEs, predictions, and outer median anchors are **read directly** from "
             "`results/v92-input-comparison/summary.json`. The run checks all 54 input hashes, the V92 "
             "implementation hash, primary observations, ordered folds, and exact median predictions. "
             "It verifies the saved anchors using V92's original fitter on original responses; "
             "it does not replace the published comparison scores with a different recomputation.", "",
             "MAE is in nats/reference token on signed compressed-minus-dense CE response. "
             "Positive paired gain means median MAE minus correction MAE. Configurations receive equal "
             "weight within a state, then states receive equal weight. V92's unchanged paired bootstrap "
             "resamples whole source states 20,000 times (seed 9201): nine clusters in every arm. "
             "**Size-split intervals also cluster "
             "by source state**, exactly as V92. These percentile intervals condition on fitted "
             "out-of-fold predictions; they do not include refitting uncertainty. This remains a small "
             "development panel within Pythia, not architecture-transfer evidence.", "",
             "## Summary table", "",
             "`source` = leave one source state out; `size` = leave one size out. "
             "Selected α distributions show every outer fold; exact fold assignments, inner scores, "
             "residual targets, coefficients, and prediction vectors are in `summary.json`.", "",
             summary_table(summary), "",
             "## Degenerate limits", "",
             "At **α = ∞**, every correction coefficient is exactly zero. The predictor **is the "
             "published median curve**, its MAE is identical, and paired gain is **exactly 0 with "
             "interval [0, 0] by construction**. This is an identity, not evidence of improvement. "
             "The implementation returns the median prediction directly at this endpoint. At α = 0, "
             "the correction would be an unpenalized least-squares fit to the frozen median residual; "
             "that limit is documented but excluded from this strongly shrunk grid.", "",
             "For finite α, the amount of departure also depends on the design and residual scale. "
             "The JSON records each coefficient vector and the source-weighted mean absolute "
             "departure from the median, alongside the selected α and its inner CV scores.", "",
             "## Published whole-response candidates", "",
             "These V92 dense_statistics MAEs are also read, without refitting, to contextualize "
             "the original whole-response OLS/ridge candidates.", "",
             "| Split | Arm | Capability | V92 OLS MAE | V92 ridge MAE |",
             "|---|---|---|---:|---:|"]
    for e in summary["evaluations"]:
        old = e["published_whole_response_mae_nats"]
        lines.append(f"| {'source' if e['scheme'] == SCHEMES[0] else 'size'} | {e['arm']} | "
                     f"{e['capability']} | {v92.fmt(old['ols']['estimate'])} | {v92.fmt(old['ridge']['estimate'])} |")
    lines += ["", "## Reuse and validation", "",
              "Imported V92 code paths: " + ", ".join(f"`{name}`" for name in p["reused_v92_functions"]) + ". "
              "**No V92 functions were copied.** " + p["new_solver_reason"], "",
              "Tests cover exact infinite-shrinkage predictions, all-coefficient shrinkage, "
              "training-only nested selection, frozen medians fitted only to original responses, "
              "published baseline read-through, matching panel/folds/scoring, and a non-zero CLI exit "
              "on failed validation. A negative scientific result is a successful run; execution or "
              "integrity failures raise errors and exit non-zero.", "",
              "A4 test command (20 passed):", "",
              "```bash\nCUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\\n"
              "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -B -m pytest -q tests/test_a4_shrunk_source_correction.py\n```", "",
              "The additional legacy V92 test run exposed one pre-existing stale coverage assertion: "
              "`test_real_loader_uses_only_requested_sources_and_common_development_grid` expects "
              "six grouped states. The pinned published numeric artifact and unchanged loader both "
              "contain nine. That legacy test is left unchanged; A4 tests verify exact equality to "
              "the actual published observations, hashes, folds, scores, and cluster counts.", "",
              "## Interpretation", "", CORRECTIONS, "",
              f"The shrunk correction has lower point MAE than both published whole-response "
              f"OLS and ridge in {whole_response_wins}/18 split/arm/capability comparisons. "
              "Keeping the median largely removes their excess error, but the incremental "
              "source correction fails the fixed majority criterion against that median.", "",
              "This check tests the specified residual correction on the existing information budget. "
              "Its result does not establish that source information is worthless or that the "
              "unexplained response is random noise.", "", f"**{d['statement']}**", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root (also useful for integrity tests)")
    args = parser.parse_args(argv)
    summary = build_summary(args.root)
    body = markdown(summary)
    artifacts = ((OUT / "summary.json", json.dumps(summary, indent=2, allow_nan=False) + "\n"),
                 (OUT / "summary.md", body), (REPORT, body))
    for relative, content in artifacts:
        path = args.root / relative
        require(not path.is_symlink() and not path.parent.is_symlink(), "Output cannot be a symlink")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    print(summary["decision"]["statement"])
    print(summary_table(summary))


if __name__ == "__main__":
    main()
