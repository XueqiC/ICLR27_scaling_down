#!/usr/bin/env python3
"""CPU-only retrospective (R) audit of V55 quantization identifiability.

    python -B analysis/v63_quant_identifiability.py

Default writes are confined to results/v63-quant-identifiability/. With explicit
authorization for the paper paths, --write-paper also writes the LaTeX table to
paper/tables/quant_ident.tex and mirrors this script to paper/code/analysis/.
V54/V55 artifacts are read-only. No fitting or choice uses test responses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
try:
    from . import v55_quant_group_fit as v55
except ImportError:
    import v55_quant_group_fit as v55

import numpy as np

CAPS = v55.CAPS
TERMS = {
    "low_order_2d": (0, 1, 2, 3, 4),
    "without_u2": (0, 1, 2, 3),
    "bit_only": (0, 1),
    "bit_group_additive": (0, 1, 2),
}
METHODS = (*TERMS, "median", "zero")
LABELS = {
    "low_order_2d": "Delivered (20)", "without_u2": "Without u^2 (16)",
    "bit_only": "Bit-only (8)", "bit_group_additive": "Bit+group additive (12)",
    "median": "V55 median", "zero": "Zero",
}
TEST_LABELS = {"bit_test": "Bit", "granularity_test": "Granularity", "joint_test": "Joint"}
ATOL = 1e-6
OUT_REL = Path("results/v63-quant-identifiability")
PROTECTED = (Path("results/v54-quant-group"), Path("results/v55-quant-group"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, context, atol=ATOL):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol, err_msg=context)
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_hashes(root):
    return {p.relative_to(root).as_posix(): digest(p) for directory in PROTECTED
            for p in sorted((root / directory).rglob("*")) if p.is_file()}


def key(row, test=False):
    names = (("test_set",) if test else ()) + ("state", "config", "capability")
    return tuple(row[name] for name in names)


def index_rows(rows, test=False):
    indexed = {key(row, test): row for row in rows}
    require(len(indexed) == len(rows), "Duplicate state/config/capability row")
    return indexed


def load_inputs(root):
    paths = {name: root / "results/v55-quant-group" / f"{name}.json"
             for name in ("register", "predictions", "compare")}
    register, frozen, compare = (json.loads(paths[name].read_text()) for name in paths)
    require(digest(paths["register"]) == frozen["provenance"]["register_sha256"],
            "Frozen predictions do not reference this register")
    require(digest(paths["predictions"]) == compare["provenance"]["predictions_sha256"],
            "Compare does not reference these frozen predictions")
    require(register["ridge"]["lambda"] == v55.RIDGE == 1e-3, "Ridge changed")
    require(register["ridge"]["penalize_intercept"], "V55 must penalize intercepts")
    require(register["test_sets"] == frozen["test_sets"] == v55.TEST_SETS, "Test panel changed")
    require(register["dev_configs"] == list(v55.DEV_CONFIGS), "Development configs changed")
    require([s["tag"] for s in register["dev_states"]] == list(v55.DEV_TAGS), "Dev states changed")
    require(not frozen["missing"] and not compare["missing"], "Incomplete V55 test panel")
    rows = register["dev_rows"]
    expected_dev = {(tag, config, cap) for tag in v55.DEV_TAGS
                    for config in v55.DEV_CONFIGS for cap in CAPS}
    require(set(index_rows(rows)) == expected_dev and len(rows) == 72, "Dev row panel changed")
    expected_test = {(test, tag, config, cap) for test, spec in v55.TEST_SETS.items()
                     for tag in spec["states"] for config in spec["configs"] for cap in CAPS}
    require(set(index_rows(frozen["predictions"], True)) == expected_test, "Frozen row panel changed")
    require(set(index_rows(compare["rows"], True)) == expected_test, "Compare row panel changed")
    # Read the actual measured responses from the exact files recorded by compare.
    recorded = compare["provenance"]["measurement_sha256"]
    expected_paths = {v55.measurement_path(PROTECTED[0], tag).as_posix()
                      for tag in (*v55.DEV_TAGS, v55.JOINT_TAG)}
    require(set(recorded) == expected_paths, "Unexpected compare measurement paths")
    measurements = {}
    for tag in (*v55.DEV_TAGS, v55.JOINT_TAG):
        rel = v55.measurement_path(PROTECTED[0], tag).as_posix()
        require(digest(root / rel) == recorded[rel], f"Measurement changed since V55 compare: {rel}")
        measurements[tag] = json.loads((root / rel).read_text())
    states = {s["tag"]: s for s in frozen["states"]}
    for state in register["dev_states"]:
        require(states[state["tag"]] == state, "Frozen development inputs changed")
    # Registration-time hashes can differ: V54 files later gained test cells.
    # Verify the actual whitelisted dev features and signed responses exactly.
    for row in rows:
        table, cap = measurements[row["state"]], row["capability"]
        require(row["dL"] == table[row["config"]][cap] - table["dense"][cap], "Dev response changed")
        require(row["phi_raw"] == v55.raw_features(states[row["state"]], cap), "Dev features changed")
        require(states[row["state"]]["L0"][cap] == table["dense"][cap], "Dev dense changed")
    stats = v55.standardization(rows)
    require(stats == register["standardization"], "V55 pooled standardization not reproduced")
    hashes = {p.relative_to(root).as_posix(): digest(p) for p in paths.values()}
    hashes.update(recorded)
    return register, frozen, compare, measurements, states, hashes


def design(rows, stats, method):
    z = v55.phi([r["phi_raw"] for r in rows], stats)
    x, v = np.asarray([v55.coordinates(r["config"]) for r in rows]).T
    terms = v55.low_order_terms(x, v, stats["u_center"])[:, TERMS[method]]
    return (terms[:, :, None] * z[:, None, :]).reshape(len(rows), -1)


def diagnostics(x):
    singular = np.linalg.svd(x, compute_uv=False)
    tolerance = float(singular[0] * max(x.shape) * np.finfo(x.dtype).eps)
    smoother = x @ np.linalg.solve(x.T @ x + v55.RIDGE * np.eye(x.shape[1]), x.T)
    dof = float(np.trace(smoother))
    svd_dof = float(np.sum(singular**2 / (singular**2 + v55.RIDGE)))
    close(dof, svd_dof, "Trace and SVD effective degrees of freedom", atol=1e-9)
    return {"n_rows": x.shape[0], "n_coefficients": x.shape[1],
            "design_rank": int(np.sum(singular > tolerance)),
            "nullity": x.shape[1] - int(np.sum(singular > tolerance)),
            "singular_values": singular.tolist(), "rank_tolerance": tolerance,
            "ridge_lambda": v55.RIDGE, "effective_dof": dof,
            "effective_dof_svd": svd_dof, "trace_svd_abs_difference": abs(dof - svd_dof)}


def identifiability(rows, stats):
    u3 = v55.coordinates("b3_g64")[0] - stats["u_center"]
    u5 = v55.coordinates("b5_g64")[0] - stats["u_center"]
    s, p = u3 + u5, u3 * u5
    u = np.asarray([v55.coordinates(r["config"])[0] - stats["u_center"] for r in rows])
    residual = u**2 - (s * u - p)
    close(residual, np.zeros_like(residual), "Two-level quadratic identity", atol=1e-12)
    # Each phi coordinate supplies one independent null direction:
    # p*phi - s*u*phi + u^2*phi = 0 on development rows.
    null = np.zeros((20, 4))
    null[:4], null[4:8], null[16:20] = p * np.eye(4), -s * np.eye(4), np.eye(4)
    by_cap = {}
    for cap in CAPS:
        cr = [r for r in rows if r["capability"] == cap]
        x = design(cr, stats, "low_order_2d")
        max_error = float(np.max(np.abs(x @ null)))
        close(x @ null, np.zeros((24, 4)), f"{cap}: null directions", atol=1e-12)
        by_cap[cap] = {"max_abs_design_null_residual": max_error,
                       "without_u2_column_space_rank": int(np.linalg.matrix_rank(x[:, :16])),
                       "rank_of_four_null_directions": int(np.linalg.matrix_rank(null))}
    u4 = v55.coordinates("b4_g64")[0] - stats["u_center"]
    return {"identity": "u^2 = (u3+u5)*u - u3*u5", "u3": u3, "u5": u5,
            "u3_plus_u5": s, "u3_times_u5": p,
            "max_abs_scalar_identity_residual": float(np.max(np.abs(residual))),
            "null_directions_columns_term_then_phi": null.tolist(), "by_capability": by_cap,
            "u4": u4, "identity_residual_at_b4": u4**2 - s * u4 + p,
            "effective_reduced_intercept_penalty_when_symmetric": v55.RIDGE / (1 + p**2),
            "interpretation": (
                "Four coefficient directions are unidentifiable from the dev responses. "
                "Ridge uniquely chooses coefficients but does not identify bit curvature. "
                "The delivered and no-u^2 designs span the same dev column space; applying "
                "isotropic ridge separately to their coefficients changes the penalty. "
                "For u3+u5=0 and c=-u3*u5, the delivered penalty on the combined constant "
                "phi block is lambda/(1+c^2), versus lambda without u^2. "
                "The null polynomial is nonzero at b=4, so its prediction is representation-dependent.")}


def fit_controls(rows, stats):
    models = {}
    for cap in CAPS:
        cr = [r for r in rows if r["capability"] == cap]
        y = np.asarray([r["dL"] for r in cr])
        models[cap] = {}
        for method in TERMS:
            x = design(cr, stats, method)
            beta = v55.ridge_fit(x, y)
            # A separate augmented least-squares solve checks the ridge result.
            independent = np.linalg.lstsq(
                np.vstack((x, np.sqrt(v55.RIDGE) * np.eye(x.shape[1]))),
                np.r_[y, np.zeros(x.shape[1])], rcond=None)[0]
            diff = close(beta, independent, f"{cap}/{method}: independent ridge solve")
            models[cap][method] = {
                **diagnostics(x), "terms": [v55.TERM_NAMES[i] for i in TERMS[method]],
                "standardization": stats, "coefficients": beta.reshape(-1, 4).tolist(),
                "dev_sse": float(np.sum((x @ beta - y)**2)),
                "independent_solver_max_abs_coefficient_difference": diff,
            }
        models[cap]["median"] = {"anchors": {
            config: float(np.median([r["dL"] for r in cr if r["config"] == config]))
            for config in v55.DEV_CONFIGS}}
        models[cap]["zero"] = {"value": 0.0}
    return models


def predict(models, row, stats):
    model = models[row["capability"]]
    z = v55.phi(row["phi_raw"], stats)
    x, v = v55.coordinates(row["config"])
    terms = v55.low_order_terms(x, v, stats["u_center"])
    predictions = {name: float(terms[list(indices)] @ np.asarray(model[name]["coefficients"]) @ z)
                   for name, indices in TERMS.items()}
    predictions.update(median=v55.interpolate(model["median"]["anchors"], row["config"]), zero=0.0)
    return predictions


def scored(row, predictions):
    return {**row, "predictions": predictions,
            "absolute_errors": {name: abs(value - row["dL"]) for name, value in predictions.items()}}


def mae_table(rows):
    result = []
    for method in METHODS:
        errors = {cap: [r["absolute_errors"][method] for r in rows if r["capability"] == cap]
                  for cap in CAPS}
        require(all(errors.values()), "Incomplete capability scores")
        result.append({"candidate": method, "mae": {c: float(np.mean(e)) for c, e in errors.items()},
                       "n": {c: len(e) for c, e in errors.items()}})
    return result


def reproduce_table(actual, reference, context):
    actual, reference = ({r["candidate"]: r for r in table} for table in (actual, reference))
    checks = {}
    for method in ("low_order_2d", "median", "zero"):
        require(actual[method]["n"] == reference[method]["n"], f"{context}: MAE counts changed")
        checks[method] = {cap: {"v55_mae": reference[method]["mae"][cap],
                               "v63_mae": actual[method]["mae"][cap],
                               "abs_difference": close(actual[method]["mae"][cap],
                                                       reference[method]["mae"][cap], context)}
                          for cap in CAPS}
    return checks


def loso(register):
    rows, folds, predictions, checks = register["dev_rows"], [], [], {}
    old_folds = {f["held_out"]: f for f in register["loso_folds"]}
    old_predictions = index_rows(register["loso_predictions"])
    for tag in v55.DEV_TAGS:
        train = [r for r in rows if r["state"] != tag]
        held = [r for r in rows if r["state"] == tag]
        require(len(train) == 60 and len(held) == 12, "LOSO state leakage or wrong counts")
        stats = v55.standardization(train)
        require(stats == old_folds[tag]["standardization"], "V55 LOSO scaler changed")
        models = fit_controls(train, stats)
        fold_rows = [scored(r, predict(models, r, stats)) for r in held]
        predictions.extend(fold_rows)
        for row in fold_rows:
            for method in ("low_order_2d", "median", "zero"):
                close(row["predictions"][method], old_predictions[key(row)]["predictions"][method],
                      f"LOSO {tag}/{method}: V55 prediction")
        table = mae_table(fold_rows)
        checks[tag] = reproduce_table(table, old_folds[tag]["mae"], f"LOSO {tag}")
        folds.append({"held_out": tag, "train_states": [s for s in v55.DEV_TAGS if s != tag],
                      "n_train_cells": 20, "n_held_out_cells": 4, "standardization": stats,
                      "models": models, "mae_table": table})
    table = mae_table(predictions)
    return {"rule": "Hold out each whole state; refit pooled scaler and all models on five states.",
            "n_folds": 6, "n_scored_cells_per_capability": 24, "mae_table": table,
            "folds": folds, "rows": predictions,
            "v55_reproduction": reproduce_table(table, register["loso_table"], "Pooled LOSO"),
            "v55_fold_reproduction": checks}


def test_scores(models, stats, frozen, compare, measurements, states):
    old = index_rows(compare["rows"], True)
    rows, prediction_differences = [], {name: [] for name in ("low_order_2d", "median", "zero")}
    for frozen_row in frozen["predictions"]:
        tag, cap, config = (frozen_row[name] for name in ("state", "capability", "config"))
        table = measurements[tag]
        loss = v55.checked_losses(table[config], f"{tag}/{config}")[cap]
        dense = v55.checked_losses(table["dense"], f"{tag}/dense")[cap]
        require(states[tag]["L0"][cap] == frozen_row["L0"], "Frozen prediction input changed")
        row = {name: frozen_row[name] for name in ("test_set", "state", "config", "capability")}
        row.update(phi_raw=v55.raw_features(states[tag], cap), loss=loss, dense=dense,
                   dL=loss - dense, frozen_L0=frozen_row["L0"],
                   measurement_path=v55.measurement_path(PROTECTED[0], tag).as_posix())
        for field in ("loss", "dense", "dL"):
            require(row[field] == old[key(row, True)][field], f"V55 compare {field} changed")
        estimates = predict(models, row, stats)
        for method in prediction_differences:
            prediction_differences[method].append(close(
                estimates[method], frozen_row["predictions"][method], f"Frozen {key(row, True)}/{method}"))
            require(frozen_row["predictions"][method] == old[key(row, True)]["predictions"][method],
                    "Compare prediction differs from frozen prediction")
        rows.append(scored(row, estimates))
    summaries, checks = {}, {}
    for test, spec in v55.TEST_SETS.items():
        test_rows = [r for r in rows if r["test_set"] == test]
        expected = len(spec["states"]) * len(spec["configs"])
        require(len(test_rows) == expected * 3, "Wrong number of test rows")
        table = mae_table(test_rows)
        summaries[test] = {**spec, "n_cells_per_capability": expected, "mae_table": table}
        checks[test] = reproduce_table(table, compare["test_sets"][test]["mae_table"], test)
    return summaries, rows, {"absolute_tolerance": ATOL, "passed": True, "mae_by_test": checks,
                             "max_abs_frozen_prediction_difference": {
                                 name: max(values) for name, values in prediction_differences.items()}}


def control_agreement(test_rows):
    comparisons = {}
    for test in v55.TEST_SETS:
        comparisons[test] = {}
        for cap in CAPS:
            rows = [r for r in test_rows if r["test_set"] == test and r["capability"] == cap]
            comparisons[test][cap] = {}
            for method in ("without_u2", "bit_group_additive"):
                differences = [r["predictions"][method] - r["predictions"]["bit_only"] for r in rows]
                error_differences = [r["absolute_errors"][method] - r["absolute_errors"]["bit_only"]
                                     for r in rows]
                comparisons[test][cap][method + "_vs_bit_only"] = {
                    "max_abs_prediction_difference": float(np.max(np.abs(differences))),
                    "abs_mae_difference": abs(float(np.mean(error_differences)))}
    return {"comparisons": comparisons, "interpretation": (
        "The three controls without u^2 have equal test MAEs to numerical precision. "
        "Their group blocks are orthogonal to the shared [phi,u*phi] blocks on the balanced "
        "development grid. At g=128, v=0, so their granularity/joint predictions coincide. "
        "On the bit test their predictions differ; symmetric group contributions cancel "
        "in paired absolute errors for these measured responses. Equal MAEs do not mean "
        "identical predictors or establish that group effects are unnecessary.")}


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def render_markdown(summary):
    ident, stats = summary["identifiability"], summary["standardization"]
    parts = ["# V63 quantization identifiability — retrospective (R)",
             "CPU-only audit. The delivered V55 predictions are unchanged. All controls are "
             "retrospective (R); none is selected or substituted using test outcomes. Signed response "
             "and MAE are in nats. Fits use exactly 24 development cells per capability; frozen tests "
             "contain 12 bit, 18 granularity, and 3 joint cells per capability.",
             "## Identifiability and rank / effective degrees of freedom",
             f"With u3={ident['u3']:.16g} and u5={ident['u5']:.16g}, "
             f"u²=(u3+u5)u−u3u5={-ident['u3_times_u5']:.16g} on development rows. "
             f"Maximum scalar residual: {ident['max_abs_scalar_identity_residual']:.3e}. "
             "The delivered design has 20 coefficients, rank 16, and four null directions per capability. "
             "Those directions produce a nonzero polynomial at b=4 "
             f"(residual {ident['identity_residual_at_b4']:.9f}).",
             ident["interpretation"],
             "Effective df = tr[X (XᵀX + λI)⁻¹ Xᵀ], checked independently against "
             "Σ sᵢ²/(sᵢ²+λ). λ=0.001 penalizes every coefficient, including intercepts. "
             "Median is nonlinear in responses, so this linear-smoother df is not assigned to it; "
             "the constant zero predictor has df=0."]
    rank_rows = []
    for cap in CAPS:
        for method in TERMS:
            d = summary["models"][cap][method]
            rank_rows.append([cap, LABELS[method], d["n_rows"], d["n_coefficients"], d["design_rank"],
                              d["nullity"], d["ridge_lambda"], f"{d['effective_dof']:.9f}"])
    parts.append(markdown_table(["Capability", "Control", "Rows", "Coefficients", "Rank", "Nullity", "λ", "Effective df"], rank_rows))
    parts += ["## Standardization", summary["standardization_rule"],
              markdown_table(["Raw feature", "Center", "Population scale"],
                             [[name, f"{mean:.17g}", f"{scale:.17g}"] for name, mean, scale in
                              zip(("ln N0", "L0c", "ln D0"), stats["center"], stats["scale"])]),
              f"Shared across capabilities and controls. u_center={stats['u_center']:.17g}; "
              "qmax=2^(b−1)−1; u=log2(qmax)−u_center; v=log2(g/128). "
              "Only the three raw phi features are z-scored. No constant features. "
              "Each full-fit diagnostic includes these constants in summary.json; each LOSO fold "
              "exports its own training-only constants.", "## Frozen test MAE"]
    for test, test_summary in summary["test_sets"].items():
        spec = v55.TEST_SETS[test]
        parts.append(f"### {TEST_LABELS[test]} test ({test_summary['n_cells_per_capability']} cells/capability)")
        parts.append(f"States: {', '.join(spec['states'])}. Configurations: {', '.join(spec['configs'])}.")
        parts.append(markdown_table(["Control", "Math", "Code", "QA"],
                                   [[LABELS[r['candidate']], *(f"{r['mae'][c]:.9f}" for c in CAPS)]
                                    for r in test_summary["mae_table"]]))
    parts.append(summary["control_agreement"]["interpretation"])
    parts += ["## Development LOSO by state",
              "Six folds, five training states (20 cells/capability) and one held state "
              "(4 cells/capability). Standardization and median anchors are refit using each "
              "fold's training rows. Pooled MAE covers 24 held-out cells/capability; equal fold "
              "sizes make this also the mean of the six fold MAEs.",
              markdown_table(["Control", "Math", "Code", "QA"],
                             [[LABELS[r['candidate']], *(f"{r['mae'][c]:.9f}" for c in CAPS)]
                              for r in summary["loso"]["mae_table"]])]
    fold_rows = [[f["held_out"], LABELS[r["candidate"]], *(f"{r['mae'][c]:.9f}" for c in CAPS)]
                 for f in summary["loso"]["folds"] for r in f["mae_table"]]
    parts.append(markdown_table(["Held-out state", "Control", "Math", "Code", "QA"], fold_rows))
    parts += ["## Delivered-design singular values",
              "Descending, including all four near-zero singular values; rank tolerance is "
              "max(X.shape) × machine epsilon × largest singular value. Full spectra for all "
              "controls and LOSO folds are in summary.json."]
    for cap in CAPS:
        d = summary["models"][cap]["low_order_2d"]
        parts.append(f"{cap} (rank tolerance {d['rank_tolerance']:.9e}):\n\n```text\n" +
                     "\n".join(f"{i + 1:2d}  {s:.17e}" for i, s in enumerate(d["singular_values"])) + "\n```")
    checks = summary["reproduction"]
    parts += ["## Reproduction and provenance",
              f"All nine delivered test MAEs reproduce V55 compare to absolute tolerance {ATOL:g}. "
              f"Maximum delivered MAE difference: {checks['max_abs_delivered_mae_difference']:.3e}; "
              "maximum frozen prediction difference: "
              f"{checks['max_abs_frozen_prediction_difference']['low_order_2d']:.3e}. "
              "Median/zero test MAEs and all six delivered/median/zero LOSO folds also reproduce. "
              "Every ridge solve agrees with independent augmented least squares; trace df agrees "
              "with SVD df. Full input hashes, coefficients, per-cell predictions/errors, and "
              "fold diagnostics are exported in summary.json.",
              "Test features use the frozen V55 state inputs, including the recorded joint dense "
              "fallback. Test targets are read from the exact measurement files named by V55 compare, "
              "subtracting each file's own dense reference. All seven measurement file hashes "
              "match compare; all 72 development rows and 99 test responses match their V55 records. "
              "Protected V54/V55 file hashes are unchanged.",
              "Reproduce: `python -B analysis/v63_quant_identifiability.py`. "
              "The default output includes `quant_ident.tex`; `--write-paper` additionally writes "
              "the two paper destinations when authorized."]
    return "\n\n".join(parts) + "\n"


def render_latex(summary):
    names = {**LABELS, "without_u2": r"Without $u^2$ (16)"}
    lines = [r"\begin{table}[H]", r"\centering\small",
             r"\caption{Quantization identifiability audit, retrospective (R). The delivered V55 "
             r"predictions are unchanged. The four fitted forms use the same 24 development cells per capability, "
             r"pooled V55 standardization, and ridge $\lambda=10^{-3}$ (including intercepts). "
             r"The delivered 20-coefficient design has rank 16: two development bit levels make "
             r"$u^2=(u_3+u_5)u-u_3u_5$, leaving four unidentifiable directions. "
             r"Ridge chooses a unique representation but does not identify curvature; deleting $u^2$ "
             r"also changes the induced penalty. Top: rank and effective degrees of freedom (df). "
             r"Bottom: MAE in nats on the frozen bit (12 cells/capability), granularity (18), and joint "
             r"1B at 96k (3) tests, plus development leave-one-state-out (LOSO, 24), "
             r"with training-only standardization refit in each fold. Median uses V55's per-config "
             r"anchors and bilinear interpolation. No control replaces a delivered prediction.}",
             r"\label{tab:quant_ident}", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Control & Rank & df (Math) & df (Code) & df (QA) \\", r"\midrule"]
    for method in TERMS:
        ranks = {summary["models"][c][method]["design_rank"] for c in CAPS}
        require(len(ranks) == 1, "LaTeX rank column needs capability-specific rows")
        cells = [names[method], str(ranks.pop()),
                 *(f"{summary['models'][c][method]['effective_dof']:.4f}" for c in CAPS)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\medskip",
              r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
              r"Evaluation & Control & Math & Code & QA \\", r"\midrule"]
    panels = [(TEST_LABELS[test], data["mae_table"]) for test, data in summary["test_sets"].items()]
    panels.append(("Dev LOSO", summary["loso"]["mae_table"]))
    for i, (label, table) in enumerate(panels):
        if i:
            lines.append(r"\addlinespace")
        for j, row in enumerate(table):
            cells = [label if j == 0 else "", names[row["candidate"]],
                     *(f"{row['mae'][c]:.4f}" for c in CAPS)]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def audit(root):
    before = protected_hashes(root)
    register, frozen, compare, measurements, states, hashes = load_inputs(root)
    stats, rows = register["standardization"], register["dev_rows"]
    identity = identifiability(rows, stats)
    models = fit_controls(rows, stats)
    coefficient_checks = {}
    for cap in CAPS:
        delivered = models[cap]["low_order_2d"]
        require(delivered["design_rank"] == register["models"][cap]["low_order_2d"]["design_rank"] == 16,
                "Delivered design rank differs from V55")
        require(models[cap]["without_u2"]["design_rank"] == 16, "Reduced design must span delivered design")
        coefficient_checks[cap] = close(delivered["coefficients"],
                                       register["models"][cap]["low_order_2d"]["coefficients"],
                                       f"{cap}: delivered coefficients")
    # Every fit (including LOSO) completes before measured test responses are scored.
    loso_result = loso(register)
    tests, test_rows, reproduction = test_scores(models, stats, frozen, compare, measurements, states)
    reproduction["max_abs_delivered_coefficient_difference"] = coefficient_checks
    reproduction["max_abs_delivered_mae_difference"] = max(
        item["abs_difference"] for test in reproduction["mae_by_test"].values()
        for item in test["low_order_2d"].values())
    source = Path(__file__).resolve()
    hashes["analysis/v63_quant_identifiability.py"] = digest(source)
    hashes["analysis/v55_quant_group_fit.py"] = digest(Path(v55.__file__))
    require(protected_hashes(root) == before, "Protected inputs changed during audit")
    return {"schema_version": 1, "audit": "V63 quantization identifiability", "origin": "R",
            "retrospective": True, "delivered_predictions_unchanged": True,
            "response": register["response"], "device": "CPU", "numpy_version": np.__version__,
            "methods": list(METHODS), "n_params_per_capability": {
                **{name: len(indices) * 4 for name, indices in TERMS.items()}, "median": 4, "zero": 0},
            "coefficient_order": "configuration term, then [1, z(log N0), z(L0c), z(log D0)]",
            "ridge": {"lambda": v55.RIDGE, "penalize_intercept": True,
                      "objective": "SSE + lambda * sum(coefficients**2)"},
            "baseline_definitions": {"median": register["candidate_definitions"]["median"],
                                     "zero": register["candidate_definitions"]["zero"]},
            "standardization": stats, "standardization_rule": register["standardization_rule"],
            "dev_states": register["dev_states"], "dev_configs": register["dev_configs"],
            "n_dev_cells": 24, "n_dev_capability_rows": 72, "dev_rows": rows,
            "frozen_test_states": frozen["states"], "identifiability": identity, "models": models,
            "test_sets": tests, "test_rows": test_rows, "control_agreement": control_agreement(test_rows),
            "loso": loso_result,
            "reproduction": reproduction, "provenance": {"input_and_code_sha256": hashes,
                "protected_files_sha256": before, "protected_files_unchanged": True,
                "measurement_hashes_match_v55_compare": True,
                "dev_rows_match_v55_register": True, "test_responses_match_v55_compare": True}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, help="Repository containing the original V54/V55 results")
    parser.add_argument("--write-paper", action="store_true", help="Also write the two explicitly authorized paper files")
    args = parser.parse_args(argv)
    root = args.root.resolve() if args.root else next(
        (p for p in Path(__file__).resolve().parents if (p / "results/v55-quant-group/register.json").is_file()), None)
    if root is None:
        parser.error("Cannot locate V55 results; pass --root")
    try:
        summary = audit(root)
        out = root / OUT_REL
        tex = render_latex(summary)
        outputs = {out / "summary.json": json.dumps(summary, indent=2, allow_nan=False) + "\n",
                   out / "summary.md": render_markdown(summary), out / "quant_ident.tex": tex}
        if args.write_paper:
            outputs[root / "paper/tables/quant_ident.tex"] = tex
            outputs[root / "paper/code/analysis/v63_quant_identifiability.py"] = Path(__file__).read_text()
        # Resolve every output before writing to reject symlinks into protected trees.
        for path in outputs:
            require(path.resolve() == path, f"Refusing symlink output: {path}")
        for path, content in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(f"WROTE {path.relative_to(root)}")
        require(protected_hashes(root) == summary["provenance"]["protected_files_sha256"],
                "Protected inputs changed while writing outputs")
        print("PASS: rank/null identities, independent ridge/df, exact panels, V55 MAE/LOSO reproduction, protected hashes")
        for test, data in summary["test_sets"].items():
            for row in data["mae_table"]:
                print(test, row["candidate"], " ".join(f"{c}={row['mae'][c]:.6f}" for c in CAPS))
    except (OSError, ValueError, KeyError, TypeError, AssertionError, np.linalg.LinAlgError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
