#!/usr/bin/env python3
"""CPU-only audit of immutable V70 forms, dense drift, and item-loss storage.

    python -B analysis/v75_distill_audit.py
    python -B analysis/v75_distill_audit.py --check

Default writes stay in results/v75-distill-audit/. The requested paper paths
are staged under that directory's paper/ tree, including an identical script.
Use --paper-dir paper only if the top-level paper destinations are authorized.
No training, measurement, confirmation fitting, or V70 output writes.
"""
from __future__ import annotations

try:
    from .paper_table_text import proofread_table
except ImportError:  # Direct scripts and file-based imports.
    try:
        from analysis.paper_table_text import proofread_table
    except ImportError:
        from paper_table_text import proofread_table


import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "results/v70-distill-confirm/freeze.json").is_file())
sys.path.insert(0, str(ROOT))
from analysis import v70_distill_confirm as v70

SOURCE = Path("results/v70-distill-confirm")
OUT = Path("results/v75-distill-audit")
SCRIPT = Path("analysis/v75_distill_audit.py")
GLOB = "*/gpt-5.6-luna_full_200_p2v3conf_lora_dseed3*/trajectory/update-*/eval.json"
CAPS, STUDENTS, SEEDS, TARGETS = v70.CAPS, v70.STUDENTS, v70.SEEDS, v70.TARGETS

# Coefficient order is exactly freeze.json's raw-basis `coef` order.
SPECS = [
    ("constant", "c", ["c"], ["1"], True, None),
    ("constant+src", "c+d n", ["c", "d"], ["1", "n"], True, "n"),
    ("T", "a u", ["a"], ["u"], False, None),
    ("T+src", "a u+d n u", ["a", "d"], ["u", "n*u"], False, "n"),
    ("E", "a w", ["a"], ["w"], False, None),
    ("E+src", "a w+d n w", ["a", "d"], ["w", "n*w"], False, "n"),
    ("joint", "a u+b u^2+c u v", ["a", "b", "c"], ["u", "u*u", "u*v"], False, None),
    ("joint+src", "a u+b u^2+c u v+d n u", ["a", "b", "c", "d"],
     ["u", "u*u", "u*v", "n*u"], False, "n"),
    ("F1:L0", "(a+b z_L)w", ["a", "b"], ["w", "z*w"], False, "L0"),
    ("F1:logN", "(a+b z_N)w", ["a", "b"], ["w", "z*w"], False, "logN"),
    ("F2:L0", "(a+b z_L)s+(c+d z_L)w", ["a", "b", "c", "d"],
     ["s", "z*s", "w", "z*w"], False, "L0"),
    ("F2:logN", "(a+b z_N)s+(c+d z_N)w", ["a", "b", "c", "d"],
     ["s", "z*s", "w", "z*w"], False, "logN"),
    ("T-only", "c+b u", ["c", "b"], ["1", "u"], True, None),
    ("E-only", "c+b w", ["c", "b"], ["1", "w"], True, None),
    ("surface:L0", "c+a u+b w+d z_L", ["c", "a", "b", "d"],
     ["1", "u", "w", "z"], True, "L0"),
    ("surface:logN", "c+a u+b w+d z_N", ["c", "a", "b", "d"],
     ["1", "u", "w", "z"], True, "logN"),
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(actual, expected, context, atol=1e-12):
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=atol, err_msg=context)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def snapshot_v70():
    return {str(p.relative_to(ROOT)): digest(p.read_bytes())
            for p in sorted((ROOT / SOURCE).rglob("*")) if p.is_file()}


class Inputs:
    def __init__(self):
        self.hashes = {}
        self.cache = {}

    def read(self, rel):
        rel = str(rel)
        raw = (ROOT / rel).read_bytes()
        require(self.hashes.setdefault(rel, digest(raw)) == digest(raw), f"Input changed: {rel}")
        return raw

    def json(self, rel):
        rel = str(rel)
        if rel not in self.cache:
            self.cache[rel] = json.loads(self.read(rel))
        return self.cache[rel]

    def verify(self):
        for rel, expected in self.hashes.items():
            require(digest((ROOT / rel).read_bytes()) == expected, f"Input changed: {rel}")


def independent_basis(spec, point, cap, refs, t_star):
    _, _, _, terms, _, descriptor = spec
    u, w = math.log1p(point["Tc"] / refs["T_ref"]), math.log1p(point["E"])
    v = math.log(point["DU"] / refs["D_ref"])
    logn = math.log(refs["N"][point["student"]])
    n = math.log(refs["N"][point["student"]] / refs["N_ref"])
    z = 0.
    if descriptor in ("L0", "logN"):
        stats = refs["L0"][cap] if descriptor == "L0" else refs["logN"]
        raw = point["L0"][cap] if descriptor == "L0" else logn
        z = (raw - stats["mean"]) / (stats["std"] or 1.)
    s = -math.expm1(-point["Tc"] / t_star) if t_star else 0.
    values = {"1": 1., "u": u, "w": w, "u*u": u*u, "u*v": u*v,
              "n": n, "n*u": n*u, "n*w": n*w, "z": z, "z*w": z*w,
              "s": s, "z*s": z*s}
    return [values[t] for t in terms]


def audit_forms(dev, frozen):
    for field in ("models", "refs", "selected", "strongest_baseline"):
        require(dev[field] == frozen[field], f"Development/freeze differ: {field}")
    points, refs = dev["points"], frozen["refs"]
    clusters = Counter(p["cluster"] for p in points)
    require(len(points) == 100 and len(clusters) == 25 and set(clusters.values()) == {4},
            "Expected 25 development trajectories with four checkpoints each")
    require([s[0] for s in SPECS] == list(v70.METHODS), "Form inventory changed")
    require(refs == v70.references(points, refs["N"]), "Reference calculation differs")
    require(len(dev["loco"]["folds"]) == 25, "Expected 25 LOCO folds")
    for fold in dev["loco"]["folds"]:
        require(fold["n_train_points"] == 96 and len(fold["train_clusters"]) == 24
                and fold["held_out"] not in fold["train_clusters"], "Invalid LOCO fold")
    forms = []
    max_coef_error = 0.
    for spec in SPECS:
        method, formula, names, terms, intercept, descriptor = spec
        ols = method in v70.CANDIDATES[:8]
        fits = {}
        for cap in CAPS:
            model = frozen["models"][cap][method]
            require(model["n_params"] == len(names), f"Coefficient count: {method}")
            require(model["selection_n_params"] == len(names) + int(method.startswith("F2:")),
                    f"Selection parameter count: {method}")
            require((model["estimator"] == "V50 OLS") == ols, f"Estimator: {method}")
            if not ols:
                require(model["intercept_unpenalized"] == intercept, f"Intercept: {method}")
            for point in points:
                close(independent_basis(spec, point, cap, refs, model["T_star"]),
                      v70.basis(method, point, cap, refs, model["T_star"]), method)
            zero = {**points[0], "Tc": 0., "E": 0.}
            if not intercept:
                close(independent_basis(spec, zero, cap, refs, model["T_star"]),
                      np.zeros(len(names)), f"Zero-budget response: {method}")
            # Validation only: reproduce full-development fits; never replace frozen coefficients.
            refit = v70.fit(points, cap, method, refs)
            close(refit["coef"], model["coef"], f"Frozen fit: {cap}/{method}", atol=1e-10)
            require(refit["T_star"] == model["T_star"], f"Frozen T_star: {cap}/{method}")
            max_coef_error = max(max_coef_error, float(np.max(np.abs(
                np.asarray(refit["coef"]) - model["coef"]))))
            fits[cap] = {"coefficients": dict(zip(names, model["coef"])), "frozen_model": model}
        forms.append({"method": method, "formula": formula, "coefficient_order": names,
            "basis_order": terms, "n_coefficients": len(names),
            "n_free_parameters": len(names) + int(method.startswith("F2:")),
            "n_discrete_parameters": int(method.startswith("F2:")),
            "intercept": intercept, "intercept_penalized": False if intercept else None,
            "student_descriptor_enters": descriptor is not None, "student_descriptor": descriptor,
            "vanishes_at_zero_budget": not intercept, "fit_data": "D",
            "estimator": "O" if ols else "R", "fits": fits})
    return {"response": "delta = post_training loss - run dense loss (nats per scored target token)",
        "definitions": {"T": "T_c: cumulative supervised completion tokens, including repetitions",
            "D_U": "unique training-pool completion tokens", "E": "T_c / D_U",
            "u": "log(1 + T_c / 35000)", "w": "log(1 + E)",
            "v": "log(D_U / D_ref)", "n": "log(N / N_ref), a student-size descriptor",
            "N": "Frozen V50 total meta-device parameter count, not nominal size or trainable adapter count",
            "z_L": "(L0_student,cap - mean_L0_cap) / population_std_L0_cap",
            "z_N": "(log(N_student) - mean_logN) / population_std_logN",
            "s": "1 - exp(-T_c / T_star)", "logs": "natural logarithms"},
        "fit_data": {"D": {**dev["development_structure"],
            "trajectory_counts_by_student": dict(Counter(k.split("|")[0] for k in clusters)),
            "response_weight_per_fit_row": 1., "normalized_weight_per_trajectory": 1/25,
            "normalized_weight_within_trajectory": 1/4, "normalized_weight_per_row": 1/100,
            "note": "All capabilities/methods fit all 100 rows jointly across students; no token-count or equal-student weighting. Baseline choice varies by student, but fitted coefficients do not.",
            "loco": "25 folds: 24 trajectories x 4 checkpoints for training; equal trajectory/checkpoint MAE. Refit coefficients, references, scaling and T_star in each fold."}},
        "estimators": {"O": "Unpenalized OLS: minimize sum_i (x_i beta - y_i)^2.",
            "R": "Minimize sum_i (Z_i beta_std - y_i)^2 + 0.001 sum_j penalized_j beta_std_j^2. Population-SD column scaling; center nonconstant columns only when an intercept exists. Intercept exempt; every other coefficient penalized. F1/F2 are not centered, preserving zero response at zero budget. coef is mapped back to the raw basis.",
            "ridge_normalized_equivalent": "With mean squared loss over 100 rows, lambda is 0.00001, not 0.001.",
            "F2_T_star": "One discrete fitted parameter chosen by lowest unpenalized training SSE on [17500,35000,70000,140000,280000]; four ridge coefficients plus T_star = five parameters."},
        "refs": refs, "forms": forms, "selected": frozen["selected"],
        "strongest_baseline": frozen["strongest_baseline"],
        "selection_rule": frozen["selection_rule"], "baseline_rule": frozen["baseline_rule"],
        "selected_E_distinction": "Selected E (math and code): a*log(1+E), one parameter, no intercept, zero at E=0, OLS. Baseline E-only: c+b*log(1+E), two parameters, unpenalized intercept c and standardized-ridge slope b; response at E=0 is c.",
        "validation": {"development_fits_reproduced": 48,
            "independent_basis_checks": 4800, "max_absolute_coefficient_error": max_coef_error}}


def inspect_items(inputs):
    paths = sorted((ROOT / "results/v12-distill").glob(GLOB))
    require(len(paths) == 48, f"Expected 48 snapshots, found {len(paths)}")
    files, schemas, list_fields = [], Counter(), {}

    def inspect(value, path=""):
        if isinstance(value, dict):
            for key, child in value.items():
                inspect(child, f"{path}.{key}" if path else key)
        elif isinstance(value, list):
            list_fields.setdefault(path, set()).add(len(value))
            suspicious = any(k in path.lower() for k in
                             ("loss", "post_training", "measurement_samples", "nll", "entropy", "per_item", "per_example"))
            require(not suspicious, f"Possible item losses now present at {path}; item bootstrap needs schema review")
            for child in value:
                if isinstance(child, (dict, list)):
                    inspect(child, path + "[]")

    for path in paths:
        rel = str(path.relative_to(ROOT))
        doc = inputs.json(rel)
        require(doc["student"] in STUDENTS and doc["data_seed"] in SEEDS, f"Unexpected run: {rel}")
        inspect(doc)
        for field in ("measurement_samples", "measurement_tokens", "post_training", "capability_losses", "dense", "delta"):
            require(set(doc[field]) == set(CAPS), f"Capability coverage: {rel}/{field}")
            for cap, value in doc[field].items():
                expected = int if field in ("measurement_samples", "measurement_tokens") else float
                require(type(value) is expected and math.isfinite(value), f"Nonaggregate field: {rel}/{field}/{cap}")
                schemas[(field, cap, type(value).__name__)] += 1
        files.append({"file": rel, "student": doc["student"], "data_seed": doc["data_seed"],
            "updates": doc["updates"], "dense_checkpoint": doc["processed_tokens"] == 0,
            "measurement_samples": doc["measurement_samples"], "measurement_tokens": doc["measurement_tokens"],
            "loss_definition": doc["loss_definition"]})
    require(sum(f["dense_checkpoint"] for f in files) == 12, "Expected 12 dense and 36 post-training snapshots")
    counts = []
    for st in STUDENTS:
        for cap in CAPS:
            rows = [r for r in files if r["student"] == st]
            ns = sorted({r["measurement_samples"][cap] for r in rows})
            nt = sorted({r["measurement_tokens"][cap] for r in rows})
            require(len(rows) == 24 and len(ns) == len(nt) == 1, f"Scoring counts changed: {st}/{cap}")
            counts.append({"student": st, "capability": cap, "items_per_evaluation": ns[0],
                "scored_tokens_per_evaluation": nt[0], "n_snapshot_files": len(rows)})
    return {"status": "unavailable_per_example_losses_not_stored",
        "per_example_losses_stored": False, "n_files_inspected": len(files),
        "n_dense_files": 12, "n_post_training_files": 36, "glob": "results/v12-distill/" + GLOB,
        "bootstrap_requested": {"n_resamples": 2000, "seed": 0, "unit": "evaluation items"},
        "bootstrap_resamples_performed": 0, "item_bootstrap_ci95": None,
        "reason": "measurement_samples contains integer counts, not samples/losses; post_training and capability_losses contain one aggregate float per capability. No per-example loss sums or token denominators are stored. An item bootstrap cannot be recovered from these aggregates.",
        "scope": "Scored-token counts are per evaluation, shared across snapshots/students here; do not sum repeated evaluations as independent items. Pool-cluster CIs condition on these fixed evaluation items and do not quantify item-sampling or training-seed uncertainty.",
        "counts": counts, "files": files,
        "field_schema": [{"field": f, "capability": c, "type": t, "n_files": n}
                         for (f, c, t), n in sorted(schemas.items())],
        "list_fields_inspected": {k: sorted(v) for k, v in sorted(list_fields.items())}}


def validate_confirmation(inputs, frozen, compare, drift):
    require(compare["complete"] and compare["freeze_sha256"] == inputs.hashes[str(SOURCE / "freeze.json")],
            "Comparison is incomplete or uses a different freeze")
    require(compare["bootstrap"] == frozen["bootstrap"], "Bootstrap protocol changed")
    for rel, expected in compare["measurement_sha256"].items():
        require(digest(inputs.read(rel)) == expected, f"V70 measurement changed: {rel}")
    index = lambda r: (r["student"], r["data_seed"], r["T_planned"], r["capability"])
    frozen_rows = {index(r): r for r in frozen["predictions"]}
    rows = compare["rows"]
    require(len(rows) == len(frozen_rows) == 108 and len({index(r) for r in rows}) == 108,
            "Incomplete/duplicate confirmation rows")
    require(set(drift) == {f"{st}|U200_s{s}" for st in STUDENTS for s in SEEDS}, "Drift coverage differs")
    specs = {s[0]: s for s in SPECS}
    for r in rows:
        fr = frozen_rows[index(r)]
        require(all(r[k] == v for k, v in fr.items()), "Comparison differs from frozen prediction row")
        st, cap = r["student"], r["capability"]
        require(r["selected"] == frozen["selected"][cap]["method"] and
                r["strongest_baseline"] == frozen["strongest_baseline"][st][cap]["method"], "Selections changed")
        j = inputs.json(r["file"])
        require(j["student"] == st and j["data_seed"] == r["data_seed"], "Snapshot pairing differs")
        close(r["actual"], j["delta"][cap], "Recorded response")
        close(r["actual"], j["post_training"][cap] - j["dense"][cap], "Response definition")
        close(j["post_training"][cap], j["capability_losses"][cap], "Duplicate aggregate loss")
        require(r["T_actual"] == j["completion_tokens_seen"] and
                j["requested_token_milestones"] == [r["trigger_processed"]], "Snapshot budget differs")
        dd = drift[f"{st}|{r['pool']}"]
        require(dd["run_dense"] == j["dense"] and
                dd["frozen_dense"] == frozen["refs"]["L0_by_student"][st], "Drift inputs differ")
        close(dd["max_abs_drift"], max(abs(dd["run_dense"][c] - dd["frozen_dense"][c]) for c in CAPS), "Drift maximum")
        p = {"student": st, "Tc": r["T_planned"], "DU": r["DU"],
             "E": r["E_planned"], "L0": frozen["refs"]["L0_by_student"][st]}
        close(p["E"], p["Tc"] / p["DU"], "Planned exposure")
        for m, prediction in r["predictions"].items():
            model = frozen["models"][cap][m]
            close(prediction, np.dot(independent_basis(specs[m], p, cap, frozen["refs"], model["T_star"]),
                                     model["coef"]), f"Frozen prediction: {m}")
            close(r["absolute_errors"][m], abs(prediction - r["actual"]), "Recorded absolute error")
            close(r["signed_errors"][m], prediction - r["actual"], "Recorded signed error")
    return rows


def ci_sign(interval):
    lo, hi = interval
    return "positive" if lo > 0 else "negative" if hi < 0 else "includes_zero"


def summarize(rows, draws):
    """Same pool draws/operation order as V70; supports a single-budget slice."""
    selected, baseline = rows[0]["selected"], rows[0]["strongest_baseline"]
    clusters = []
    budgets = sorted({r["T_planned"] for r in rows})
    for seed in SEEDS:
        rr = sorted((r for r in rows if r["data_seed"] == seed), key=lambda r: r["T_planned"])
        require([r["T_planned"] for r in rr] == budgets, "Unbalanced paired budget panel")
        clusters.append({"pool": f"U200_s{seed}", "n_checkpoints": len(rr),
            "mae": {m: float(np.mean([r["absolute_errors"][m] for r in rr])) for m in v70.METHODS}})
    errors = np.asarray([[p["mae"][selected], p["mae"][baseline]] for p in clusters])
    mean, boot = errors.mean(0), errors[draws].mean(1)
    interval = np.quantile(boot[:, 1] - boot[:, 0], [.025, .975]).tolist()
    return {"student": rows[0]["student"], "capability": rows[0]["capability"],
        "selected": selected, "strongest_baseline": baseline, "n_pools": 6,
        "n_checkpoints": len(rows), "T_planned": budgets, "clusters": clusters,
        "candidate_mae": float(mean[0]), "baseline_mae": float(mean[1]),
        "candidate_mae_ci95": np.quantile(boot[:, 0], [.025, .975]).tolist(),
        "baseline_mae_ci95": np.quantile(boot[:, 1], [.025, .975]).tolist(),
        "paired_difference": {"estimate": float(mean[1] - mean[0]), "ci95": interval, "ci_sign": ci_sign(interval)},
        "all_method_mae": {m: float(np.mean([p["mae"][m] for p in clusters])) for m in v70.METHODS}}


def dense_sensitivity(rows, compare, drift):
    draws = np.random.default_rng(0).integers(0, 6, size=(5000, 6))
    scenarios = {}
    for name, sign in (("original", 0), ("plus_drift", 1), ("minus_drift", -1)):
        shifted = []
        for r in rows:
            shift = sign * drift[f"{r['student']}|{r['pool']}"]["max_abs_drift"]
            actual = r["actual"] + shift
            shifted.append({**r, "original_actual": r["actual"], "response_shift": shift,
                "actual": actual, "signed_errors": {m: p - actual for m, p in r["predictions"].items()},
                "absolute_errors": {m: abs(p - actual) for m, p in r["predictions"].items()}})
        groups, by_budget = [], []
        reference = v70.paired_summary(shifted)
        for st in STUDENTS:
            for cap in CAPS:
                rr = [r for r in shifted if r["student"] == st and r["capability"] == cap]
                group = summarize(rr, draws)
                ref = next(g for g in reference if g["student"] == st and g["capability"] == cap)
                for k in ("candidate_mae", "baseline_mae"):
                    close(group[k], ref[k], f"V70 reproduction: {name}/{st}/{cap}/{k}")
                for k in ("estimate", "ci95"):
                    close(group["paired_difference"][k], ref["paired_difference"][k], "V70 paired reproduction")
                for m in v70.METHODS:
                    close(group["all_method_mae"][m], ref["all_method_mae"][m], "V70 all-method MAE")
                if name == "original":
                    saved = next(g for g in compare["groups"] if g["student"] == st and g["capability"] == cap)
                    for k in ("candidate_mae", "baseline_mae"):
                        close(group[k], saved[k], f"Saved V70 {k}")
                    for k in ("estimate", "ci95"):
                        close(group["paired_difference"][k], saved["paired_difference"][k], "Saved V70 CI")
                groups.append(group)
                by_budget.extend(summarize([r for r in rr if r["T_planned"] == t], draws) for t in TARGETS)
        scenarios[name] = {"response_shift_multiplier": sign, "groups": groups,
                           "by_budget": by_budget, "rows": shifted}
    changes = {"groups": [], "by_budget": []}
    for name in ("plus_drift", "minus_drift"):
        for section in changes:
            for original, shifted in zip(scenarios["original"][section], scenarios[name][section]):
                old, new = original["paired_difference"], shifted["paired_difference"]
                changed = old["ci_sign"] != new["ci_sign"]
                shifted["paired_difference"].update(ci_sign_changed=changed,
                    estimate_sign_changed=bool(np.sign(old["estimate"]) != np.sign(new["estimate"])))
                if changed:
                    changes[section].append({"scenario": name, "student": shifted["student"],
                        "capability": shifted["capability"], "T_planned": shifted["T_planned"],
                        "original_ci95": old["ci95"], "shifted_ci95": new["ci95"]})
    return {"response_shift": "delta'_trajectory,cap,budget = delta_trajectory,cap,budget +/- recorded max_abs_drift_trajectory. The same nonnegative trajectory maximum shifts every capability and budget; predictions and descriptors stay frozen.",
        "interpretation": "Two deterministic sensitivity scenarios, not a probabilistic drift CI or a bound over all possible mixed-sign perturbations.",
        "bootstrap": compare["bootstrap"], "drift_by_trajectory": drift,
        "scenarios": scenarios, "ci_sign_changes": changes,
        "any_primary_ci_sign_change": bool(changes["groups"]),
        "any_budget_ci_sign_change": bool(changes["by_budget"]),
        "ci_sign_definition": "positive if lower endpoint > 0; negative if upper endpoint < 0; otherwise includes_zero. Compare each shifted interval with its unshifted counterpart."}


@proofread_table
def latex_table(audit):
    lines = [r"% Generated by analysis/v75_distill_audit.py; coefficients come from V70 freeze.json.",
        r"\begin{table}[!htbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\caption{V75 audit of all 16 V70 distillation forms. Response: $\Delta L=L_{\rm post}-L_0$. "
        r"Panel A specifies each form and its fit; panel B gives frozen raw-basis coefficients by capability. "
        r"Selected form E is $a\log(1+E)$ (math/code): one parameter, no intercept, and zero at zero budget. "
        r"Baseline E-only is $c+b\log(1+E)$, with two parameters and an unpenalized intercept.}",
        r"\label{tab:distill_forms_audit}", r"\textbf{A. Exact forms and fitting protocol}\par\smallskip",
        r"\begin{tabular}{@{}llrllll@{}}", r"\toprule",
        r"Form & $\widehat{\Delta L}$ & Free $k$ & Intercept & Int. penalized? & Student input & Fit \\", r"\midrule"]
    for form in audit["forms"]:
        k = r"$4+1=5$" if form["n_discrete_parameters"] else str(form["n_free_parameters"])
        descriptor = {None: "none", "n": "$n$", "L0": "$z_L$", "logN": "$z_N$"}[form["student_descriptor"]]
        lines.append(" & ".join([form["method"], "$" + form["formula"] + "$", k,
            "yes" if form["intercept"] else "no", "no" if form["intercept"] else "n/a",
            descriptor, "D/" + form["estimator"]]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
        r"\begin{minipage}{\textwidth}\scriptsize",
        r"$T=T_c$ is supervised completion-token exposure; $D_U$ is unique-pool completion tokens; "
        r"$E=T/D_U$, $u=\log(1+T/35000)$, $w=\log(1+E)$, $v=\log(D_U/99504)$, "
        r"$n=\log(N/N_{\rm ref})$, $N_{\rm ref}=1.4129708903255556\times10^9$. "
        r"$z_L=(L_{0,s,c}-\mu_{L,c})/\sigma_{L,c}$, $z_N=(\log N_s-\mu_N)/\sigma_N$; "
        r"$N$ is the frozen V50 total meta-device parameter count; "
        r"descriptor means/SDs equally weight the three development students. "
        r"$s=1-\exp(-T/T_\star)$. All logarithms are natural. The suffix +src introduces student size $n$.",
        r"\par D: all 25 trajectories $\times$ 4 checkpoints (100 responses per capability): "
        r"12 former dev + 9 former test + 4 former held-out-4B trajectories. "
        r"Raw fitting weights are 1 per row; normalized weights are $1/25$ per trajectory and $1/4$ within "
        r"trajectory ($1/100$ per row), with no token-count or equal-student weighting. "
        r"Every listed form uses this same pooled fit; only baseline selection varies by student.",
        r"\par O: unpenalized OLS. R: $\mathrm{SSE}+0.001\|\beta_{\rm std,nonint}\|_2^2$; "
        r"columns use training population SDs. Center nonconstant columns only when an intercept exists; "
        r"F1/F2 have no intercept and no centering. All nonintercept ridge coefficients are penalized. "
        r"F2 adds one discrete fitted $T_\star\in\{17500,35000,70000,140000,280000\}$, "
        r"chosen by training SSE. Selection uses 25 LOCO folds, refitting on $24\times4$ points per fold.",
        r"\end{minipage}", r"\par\smallskip", r"\textbf{B. Frozen coefficients (order shown; rounded to 8 decimals)}\par\smallskip",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
        r"Form & Coefficient order & Math & Code & QA \\", r"\midrule"]
    for form in audit["forms"]:
        cells = []
        for cap in CAPS:
            model = form["fits"][cap]["frozen_model"]
            vals = [f"{v:.8f}" for v in model["coef"]]
            parts = [", ".join(vals[i:i+2]) for i in range(0, len(vals), 2)]
            if model["T_star"] is not None:
                parts.append(r"$T_\star=" + str(model["T_star"]) + "$ ")
            cells.append(r"\shortstack[r]{" + r" \\ ".join(parts) + "}")
        lines.append(" & ".join([form["method"], "$(" + ",".join(form["coefficient_order"]) + ")$", *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\par\smallskip",
        r"\begin{minipage}{\textwidth}\scriptsize "
        r"Coefficients multiply the raw bases in panel A, including the uncentered zero-budget forms. "
        r"Full-precision coefficients, all normalization references, fitting weights, and freeze hashes "
        r"are in \texttt{results/v75-distill-audit/summary.json}. QA selects joint; math/code select E. "
        r"Intercepts count toward $k$; estimated descriptor references and column scalings are preprocessing, "
        r"not additional regression coefficients in V70's selection count.\end{minipage}",
        r"\end{table}", ""]
    return "\n".join(lines)


def diff_text(group):
    d = group["paired_difference"]
    return f"{d['estimate']:.9f} [{d['ci95'][0]:.9f}, {d['ci95'][1]:.9f}]"


def markdown(report):
    audit, drift, items = (report[k] for k in ("forms_audit", "dense_drift_sensitivity", "item_sampling_uncertainty"))
    lines = ["# V75 distillation form audit and CPU checks", "", audit["selected_E_distinction"], "",
        f"**Dense drift:** primary CI sign changes: {len(drift['ci_sign_changes']['groups'])}/12 shifted comparisons; "
        f"budget-specific CI sign changes: {len(drift['ci_sign_changes']['by_budget'])}/36. "
        "Positive paired differences favor the selected form.", "",
        "**Item uncertainty:** per-example losses are not stored in any of the 48 inspected snapshots. "
        "The requested 2,000-resample, seed-0 item bootstrap cannot be computed; no item interval is reported. "
        "Each evaluation contains 64 items per capability and scores 13,198 math, 4,516 code, and 251 QA target tokens.", "",
        "## Forms, fit data, and coefficients", "",
        audit["fit_data"]["D"]["note"], "",
        "D = 25 trajectories × 4 checkpoints = 100 responses per capability (12 former dev + 9 former test + 4 former held-out-4B). "
        "There are 9 trajectories each for 270M and 1B, and 7 for 4B. Fitting uses raw weight 1 per row; "
        "normalized weights are 1/25 per trajectory × 1/4 per checkpoint = 1/100 per row. "
        "LOCO uses 25 folds with 96 training points each, refitting references and scaling within folds.", ""]
    lines.extend(f"- `{k}`: {v}" for k, v in audit["definitions"].items())
    lines += ["", "O: " + audit["estimators"]["O"], "", "R: " + audit["estimators"]["R"], "",
        audit["estimators"]["ridge_normalized_equivalent"], "", audit["estimators"]["F2_T_star"], "",
        "| Form | Exact formula | Free parameters | Intercept | Intercept penalized | Student descriptor | Fit |",
        "|---|---|---:|---|---|---|---|"]
    for f in audit["forms"]:
        lines.append(f"| {f['method']} | `{f['formula']}` | {f['n_free_parameters']} | "
            f"{'yes' if f['intercept'] else 'no'} | {'no' if f['intercept'] else 'n/a'} | "
            f"{f['student_descriptor'] or 'none'} | D/{f['estimator']} |")
    lines += ["", "Frozen coefficients below follow each formula's displayed order. "
        "These are raw-basis coefficients, not standardized ridge coefficients. Full precision and normalization constants are retained in summary.json.", "",
        "| Form | Coefficient order | Math | Code | QA |", "|---|---|---|---|---|"]
    for f in audit["forms"]:
        cells = []
        for cap in CAPS:
            model = f["fits"][cap]["frozen_model"]
            cell = ", ".join(f"{x:.9g}" for x in model["coef"])
            if model["T_star"] is not None:
                cell += f"; T_star={model['T_star']}"
            cells.append(cell)
        lines.append("| " + " | ".join([f["method"], ", ".join(f["coefficient_order"]), *cells]) + " |")
    lines += ["", audit["selection_rule"], "", audit["baseline_rule"], "",
        "## Dense-drift sensitivity", "", drift["response_shift"], "", drift["interpretation"], "",
        "The recorded trajectory maxima are 0.0009337649402398895 for every 270M pool and "
        "0.0007196634189547968 for every 1B pool. Primary results use V70's 5,000 PCG64 pool resamples, "
        "seed 0, with the same six pool indices across methods, students, capabilities, and sensitivity scenarios. "
        "Each pool retains all three budgets. Predictions remain the stored planned-budget predictions.", "",
        "| Student | Cap. | Selected / frozen baseline | Shift | Selected MAE | Baseline MAE | Paired difference [95% pool CI] | CI sign |",
        "|---|---|---|---|---:|---:|---|---|"]
    for index in range(6):
        for name in ("original", "plus_drift", "minus_drift"):
            g = drift["scenarios"][name]["groups"][index]
            lines.append(f"| {g['student']} | {g['capability']} | {g['selected']} / {g['strongest_baseline']} | "
                f"{name} | {g['candidate_mae']:.9f} | {g['baseline_mae']:.9f} | {diff_text(g)} | {g['paired_difference']['ci_sign']} |")
    lines += ["", "All 16 methods' shifted MAEs, individual shifted responses/errors, and selected/baseline MAE intervals "
        "are in summary.json. No form or baseline is reselected using these errors.", "",
        "## Evaluation-item uncertainty and separate budget-specific pool intervals", "", items["reason"], "", items["scope"], "",
        "| Student | Capability | Items per evaluation | Scored target tokens per evaluation | Snapshots inspected |",
        "|---|---|---:|---:|---:|"]
    for row in items["counts"]:
        lines.append(f"| {row['student']} | {row['capability']} | {row['items_per_evaluation']} | "
                     f"{row['scored_tokens_per_evaluation']} | {row['n_snapshot_files']} |")
    lines += ["", "The following intervals resample **six training pools at one budget** (5,000 resamples, seed 0). "
        "They are supplementary pool-cluster intervals, not evaluation-item intervals; item intervals are unavailable at every budget. "
        "The primary V70 intervals above average over all three budgets within each pool.", "",
        "| Student | Cap. | Planned T | Original paired difference [pool CI] | +drift [pool CI] | -drift [pool CI] | Any CI sign change | Item CI |",
        "|---|---|---:|---|---|---|---|---|"]
    for groups in zip(*(drift["scenarios"][n]["by_budget"] for n in ("original", "plus_drift", "minus_drift"))):
        g = groups[0]
        changed = any(r["paired_difference"].get("ci_sign_changed", False) for r in groups)
        lines.append("| " + " | ".join([g["student"], g["capability"], str(g["T_planned"][0]),
            *(diff_text(r) for r in groups), "yes" if changed else "no", "unavailable"]) + " |")
    lines += ["", "## Reproduction and write scope", "",
        "Run `python -B analysis/v75_distill_audit.py`; verify existing artifacts without writes using "
        "`python -B analysis/v75_distill_audit.py --check`. Only NumPy/CPU operations are used.", "",
        "Validation reproduced 48 full-development fits, checked 4,800 independently specified basis vectors, "
        "reconstructed frozen planned-budget predictions from coefficients, verified recorded responses against eval.json, "
        "and matched all six original V70 MAEs and paired intervals. Shifted results also match V70's paired-summary implementation. "
        "The SHA256 of every file under results/v70-distill-confirm/ is checked before and after execution.", "",
        report["artifacts"]["write_scope_note"], ""]
    lines.extend(f"- {k}: `{v}`" for k, v in report["artifacts"].items() if k != "write_scope_note")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Recompute and verify existing artifacts without writing")
    parser.add_argument("--paper-dir", type=Path, help="Default: results/v75-distill-audit/paper (staged); top-level paper requires authorization")
    args = parser.parse_args()
    paper = args.paper_dir or OUT / "paper"
    require(paper in (OUT / "paper", Path("paper")), "Only staged or explicitly requested paper destinations supported")
    before = snapshot_v70()
    inputs = Inputs()
    dev, frozen, compare, drift = [inputs.json(SOURCE / f"{name}.json")
                                   for name in ("develop", "freeze", "compare", "dense_drift")]
    require(inputs.read(SOURCE / "FREEZE_V70").decode().strip() == inputs.hashes[str(SOURCE / "freeze.json")],
            "Freeze sentinel mismatch")
    for rel in ("analysis/v70_distill_confirm.py", "analysis/v56_distill_forms.py", str(SOURCE / "develop.json")):
        require(digest(inputs.read(rel)) == frozen["inputs_sha256"][rel], f"Frozen audit dependency changed: {rel}")
    inputs.read(SCRIPT)
    forms = audit_forms(dev, frozen)
    items = inspect_items(inputs)
    rows = validate_confirmation(inputs, frozen, compare, drift)
    sensitivity = dense_sensitivity(rows, compare, drift)
    items["by_budget"] = [{"student": r["student"], "capability": r["capability"],
        "T_planned": r["T_planned"][0], "paired_difference_estimate": r["paired_difference"]["estimate"],
        "item_bootstrap_ci95": None, "item_bootstrap_status": items["status"],
        "pool_cluster_ci95": r["paired_difference"]["ci95"]}
        for r in sensitivity["scenarios"]["original"]["by_budget"]]
    table_rel, mirror_rel = paper / "paper/tables/distill_forms_audit.tex", paper / "code/analysis/v75_distill_audit.py"
    report = {"schema_version": 1, "analysis": "V75 distillation forms and two CPU checks",
        "forms_audit": forms, "dense_drift_sensitivity": sensitivity, "item_sampling_uncertainty": items,
        "provenance": {"inputs_sha256": inputs.hashes, "v70_outputs_sha256": before,
            "v70_outputs_unchanged": True, "confirmation_refit": False, "gpu_used": False},
        "artifacts": {"summary_json": str(OUT / "summary.json"), "summary_markdown": str(OUT / "summary.md"),
            "table": str(table_rel), "code_mirror": str(mirror_rel),
            "write_scope_note": ("Paper artifacts are staged under results/v75-distill-audit/paper/ to respect the write-only restriction, preserving paper/paper/tables and paper/analysis relative paths. Top-level paper files are untouched. No commit was made."
                if paper != Path("paper") else "The explicitly authorized paper table and code mirror are written to the top-level paper destinations. V70 outputs are untouched. No commit was made.")}}
    table = latex_table(forms)
    outputs = {OUT / "summary.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
        OUT / "summary.md": markdown(report), OUT / "distill_forms_audit.tex": table,
        table_rel: table, mirror_rel: inputs.read(SCRIPT).decode()}
    require(table.count(r"\begin{table}[!htbp]") == 1 and table.count(r"\end{table}") == 1,
            "Expected one [!htbp] audit table")
    inputs.verify()
    require(snapshot_v70() == before, "V70 outputs changed during audit")
    for rel, text in outputs.items():
        if args.check:
            require((ROOT / rel).read_text() == text, f"Stale output: {rel}")
        else:
            (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
            (ROOT / rel).write_text(text)
    inputs.verify()
    require(snapshot_v70() == before, "V70 outputs changed during output verification")
    print(json.dumps({"mode": "check" if args.check else "write", "forms": len(forms["forms"]),
        "primary_ci_sign_changes": len(sensitivity["ci_sign_changes"]["groups"]),
        "budget_ci_sign_changes": len(sensitivity["ci_sign_changes"]["by_budget"]),
        "item_bootstrap": items["status"], "snapshots_inspected": items["n_files_inspected"],
        "v70_outputs_unchanged": True, "outputs": [str(p) for p in outputs]}, indent=2))


if __name__ == "__main__":
    main()
