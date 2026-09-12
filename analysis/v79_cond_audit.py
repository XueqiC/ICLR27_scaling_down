#!/usr/bin/env python3
"""CPU audit of frozen V76; never rerun its writer or alter historical outputs.

Run with python -B analysis/v79_cond_audit.py [--selftest | --check].
Paper table and byte-identical code mirror are staged under the output directory
to respect the explicit write-only boundary, as in V76 and V77.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "results/v76-cap-conditioning/summary.json").is_file())
OUT = ROOT / "results/v79-cond-audit"
CAPS = ("math", "code", "qa")
TAGS = ("pythia-2.8b@step143000", "pythia-2.8b@step16000")
SPEC = importlib.util.spec_from_file_location("v76_audit_input", ROOT / "analysis/v76_cap_conditioning.py")
v76 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v76)
require, close = v76.require, v76.close


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def tree_hashes(path):
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob("*")) if p.is_file()}


def curve_projection(h, y):
    """Signed through-origin projection; centered R^2 measures variance explained."""
    h, y = np.asarray(h, dtype=float), np.asarray(y, dtype=float)
    require(h.shape == y.shape and h @ h > 0, "Invalid curve projection")
    scale = float(h @ y / (h @ h))
    residual = y - scale * h
    sse, sst = float(residual @ residual), float((y - y.mean()) @ (y - y.mean()))
    return {"scale": scale, "r2": 1 - sse / sst if sst > 0 else None,
            "uncentered_r2": 1 - sse / float(y @ y) if y @ y > 0 else None,
            "sse": sse, "centered_sst": sst, "residuals": list(map(float, residual)),
            "normal_equation_residual": float(h @ residual)}


def direction(values, sources):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    opposite = values * mean < 0
    return {"n_sources": len(values), "mean": mean, "median": float(np.median(values)),
            "n_positive": int(sum(values > 0)), "n_negative": int(sum(values < 0)),
            "n_zero": int(sum(values == 0)), "n_opposite_mean": int(sum(opposite)),
            "opposite_mean_share": float(opposite.mean()) if mean != 0 else None,
            "opposite_mean_sources": [s for s, flag in zip(sources, opposite) if flag],
            "responses": dict(zip(sources, map(float, values)))}


def merge_rows(rows, collapse):
    """Pair both aliases in one cluster; optionally give duplicate cells one vote."""
    vectors = [{(r["x"], r["capability"]): (r["actual"], r["predictions"])
                for r in rows if r["state"] == tag} for tag in TAGS]
    require(vectors[0] and vectors[0] == vectors[1], "Duplicate outcomes/predictions differ")
    result = []
    for row in rows:
        if collapse and row["state"] == TAGS[1]:
            continue
        result.append({**row, "cluster": TAGS[0] if row["state"] in TAGS else row["cluster"]})
    return result


def scale_audit(inputs, stored, fits):
    source = inputs.read("analysis/v76_cap_conditioning.py",
                         stored["inputs_sha256"]["analysis/v76_cap_conditioning.py"]).decode()
    nodes = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    evidence = {}
    for name in ("corrections", "fit_anchors", "fit_reuse", "predict_anchors", "quant_curve"):
        n = nodes[name]
        evidence[name] = {"file": "analysis/v76_cap_conditioning.py", "line": n.lineno,
                          "end_line": n.end_lineno, "source": ast.get_source_segment(source, n)}
    ols = next(n for n in ast.walk(nodes["corrections"]) if isinstance(n, ast.If)
               and ast.unparse(n.test) == "objective == 'ols'")
    require(ast.unparse(ols.body[0]) == "scale = x @ y / (x @ x)", "V76 OLS implementation changed")
    x = np.array([1., 2., 3.])
    negative_test, _ = v76.corrections(x, x[:, None] * np.array([2., -3., -4.]))
    close(list(negative_test.values()), [2., -3., -4.], "V76 accepts negative scales")
    arms = {}
    for arm, data in stored["arms"].items():
        scales = data["fit"]["scales"]
        close(list(fits[arm]["scales"].values()), list(scales.values()), f"{arm} stored scale reproduction")
        sensitivities = {name: item["fit"]["scales"] for name, item in data.get("sensitivity", {}).items()}
        arms[arm] = {"scales": scales, "negative_scales_allowed": True,
                     "negative_scales_occurred": any(s < 0 for s in scales.values()),
                     "negative_capabilities": [c for c, s in scales.items() if s < 0],
                     "sensitivity_scales": sensitivities,
                     "negative_sensitivity_scales": [{"fit": name, "capability": c, "scale": s}
                         for name, ss in sensitivities.items() for c, s in ss.items() if s < 0]}
    return {"formula": "s_c = sum_k(h_k*m_kc) / sum_k(h_k^2), unrestricted signed OLS, no intercept",
            "v76_source_matches_hash_recorded_in_v76_summary": True,
            "stored_spec": stored["protocol"]["B"], "arms": arms, "code_evidence": evidence,
            "negative_scale_execution_test": negative_test,
            "conclusion": "Negative scales are allowed by the stored specification and actual code, "
                          "but none occurred in any primary arm or stored sensitivity fit. "
                          "Pruning has no output clipping. Quantization alone keeps V69's boundary "
                          "zero floor after extrapolation; that is not a scale sign constraint."}


def shape_audit(cells, fit):
    keys = sorted(fit["shared_anchors"], key=float)
    h = np.array([fit["shared_anchors"][k] for k in keys])
    capabilities = {}
    for ci, cap in enumerate(CAPS):
        y = np.array([fit["models"]["A"][cap][k] for k in keys])
        projection = curve_projection(h, y)
        close(projection["scale"], fit["scales"][cap], f"{cap} best signed projection")
        by_density = {}
        for key in keys:
            rr = sorted([r for r in cells if r["x"] == key], key=lambda r: r["state"])
            by_density[key] = direction([r["y"][ci] for r in rr], [r["state"] for r in rr])
        crossings = [float(keys[i]) - y[i] * (float(keys[i + 1]) - float(keys[i])) / (y[i + 1] - y[i])
                     for i in range(len(keys) - 1) if y[i] * y[i + 1] < 0]
        capabilities[cap] = {**projection, "median_anchors": dict(zip(keys, map(float, y))),
                             "signs_in_density_order": [int(np.sign(v)) for v in y],
                             "changes_sign": bool(np.any(y > 0) and np.any(y < 0)),
                             "linear_interpolated_zero_crossings": crossings,
                             "source_directions_by_density": by_density}
    return {"density_order": list(map(float, keys)), "shared_anchors": fit["shared_anchors"],
            "shared_signs_in_density_order": list(map(int, np.sign(h))),
            "n_development_states": len({r["state"] for r in cells}), "n_cells": len(cells),
            "r2_definition": "1 - sum((m_c-s_c*h)^2)/sum((m_c-mean(m_c))^2). "
                             "Fit s_c without intercept, equal weight per observed median anchor. "
                             "Uncentered R^2 is supplemental, not the fraction of centered variance.",
            "direction_definition": "At each density, arithmetic mean over available development "
                                    "source states; opposite means response*mean<0. Exact zero responses "
                                    "are not opposite; an exactly zero mean has no direction (share=null).",
            "density_coverage_note": "The historical grid is unbalanced (4-16 sources per density); "
                                     "do not impute missing source-density pairs or add a dense anchor. "
                                     "Median-curve shape is an aggregate over the available sources.",
            "capabilities": capabilities}


def identity_audit(inputs, stored):
    path = "results/v77-model-arch/pythia_2_8b_tensor_identity.json"
    identity = inputs.json(path)
    freeze = inputs.json("results/v72-prune-repeat/freeze.json")
    require(identity["status"] == "ok" and identity["equal_modulo_signed_zero"], "V77 identity evidence failed")
    require(identity["learned_parameter_tensors"] == 388 and identity["byte_identical_parameter_tensors"] == 387,
            "Unexpected learned-tensor identity inventory")
    for rev in identity["revisions"]:
        frozen_hash = freeze["cache"]["revisions"][rev["revision"]]["weights"][0]["blob_sha256"]
        require(Path(rev["blob_path"]).name == frozen_hash, "V77/V72 weight provenance mismatch")
    left, right = [r["parameters"] for r in identity["revisions"]]
    require(left.keys() == right.keys() and len(left) == 388, "Tensor inventories differ")
    differences = []
    for name in left:
        require(all(left[name][k] == right[name][k] for k in ("shape", "dtype", "bytes")), "Tensor shape/dtype mismatch")
        if left[name]["sha256"] != right[name]["sha256"]:
            differences.append(name)
    require(differences == [d["name"] for d in identity["differing_parameter_tensors"]], "Tensor hash evidence mismatch")
    require(all(d["equal_modulo_signed_zero"] and d["other_differing_elements"] == 0
                for d in identity["differing_parameter_tensors"]), "Nonzero tensor difference")
    pairs = []
    for d in (0.65, 0.75, 0.85):
        paths = [f"results/v72-prune-repeat/measurements/{t.replace('@', '--')}/d{d}/prune_losses.json" for t in TAGS]
        raw = [inputs.read(p, stored["inputs_sha256"][p]) for p in paths]
        require(raw[0] == raw[1], "V72 measurement files are not identical")
        pairs.append({"density": d, "files": paths, "sha256": v76.digest(raw[0]), "byte_identical": True})
    return {"aliases": list(TAGS), "representative": TAGS[0], "tensor_evidence_file": path,
            "verification_scope": "Checked stored V77 full learned-tensor audit and its V72 frozen "
                                  "blob identities; no model loading or new multi-GB weight audit.",
            "learned_parameter_tensors": 388, "byte_identical_parameter_tensors": 387,
            "equal_modulo_signed_zero": True, "differences": identity["differing_parameter_tensors"],
            "identical_measurement_pairs": pairs,
            "conclusion": "Distinct serialized blob hashes do not imply distinct learned weights. "
                          "The one differing learned entry is +0.0/-0.0; extra checkpoint buffers "
                          "also differ. These two labels represent one measured state cluster."}


def inventory(inputs):
    args = ["rg", "--files", "--hidden", "--no-ignore", "results", "-g", "*[Ff][Rr][Ee][Ee][Zz][Ee]*",
            "-g", "*[Cc][Oo][Mm][Pp][Aa][Rr][Ee]*"]
    run = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True)
    files = sorted(p for p in run.stdout.splitlines() if not p.startswith("results/v79-cond-audit/"))
    hits = []
    for path in files:
        raw = inputs.read(path).decode(errors="replace")
        lines = raw.splitlines()
        matches = {tag: [i for i, line in enumerate(lines, 1)
                         if tag.lower() in line.lower() or tag.replace("@", "--").lower() in line.lower()]
                   for tag in TAGS}
        if all(matches.values()):
            hits.append({"file": path, "matching_line_numbers": matches})
    # Also catch separated model/revision fields, rather than requiring one tag string.
    broad = [p for p in files if all(t in inputs.read(p).decode(errors="replace").lower()
                                    for t in ("2.8b", "step16000", "step143000"))]
    expected = {"results/v72-prune-repeat/freeze.json", "results/v72-prune-repeat/compare.json"}
    require({h["file"] for h in hits} == set(broad) == expected,
            "New freeze/compare match requires manual independence classification")
    frozen = inputs.json("results/v72-prune-repeat/freeze.json")
    comp = inputs.json("results/v72-prune-repeat/compare.json")
    require({r["source"] for r in frozen["predictions"]} == set(TAGS), "V72 target labels changed")
    require(comp["n_configurations"] == 6 and comp["n_rows"] == 18
            and set(comp["by_state"]) == set(TAGS), "V72 state counting changed")
    code = inputs.read("analysis/v72_prune_repeat.py").decode().splitlines()
    return {"discovery_command": args, "matching_rule": "Case-insensitive full @/-- tags for both revisions; "
                         "also check separate 2.8b, step16000 and step143000 fields.",
            "n_files_scanned": len(files), "files_scanned": files, "files_with_both_tags": hits,
            "broad_separated_field_matches": broad,
            "code_evidence_lines": {str(i): line for i, line in enumerate(code, 1)
                                    if '"n_configurations": len(cells)' in line or '"by_state": {tag:' in line},
            "other_analyses_counting_aliases_as_separate_states": [{"analysis": "V72 pruning repeatability",
                "files": sorted(expected), "evidence": "Freeze has two target labels; compare reports "
                "six state-density configurations, 18 capability rows, and two by_state entries. "
                "V72 treats them as two states; its compare has no state-bootstrap interval. "
                "Covariate-dependent power/A2 predictions can differ across labels even though "
                "measurements agree; the V76 A/B/C/D predictions being audited agree exactly."}],
            "audited_analysis": "V76 pooled confirmation counts five state labels and bootstraps them "
                                "separately; its unique_outcome_sensitivity already counts these aliases once.",
            "scope": "No other freeze/compare file under results/ matched both labels. V77's identity "
                     "files discuss the aliases as audit evidence, not independent performance states. "
                     "This is a literal-tag/separated-field inventory, not proof against unnamed reuse."}


def build(inputs):
    stored = inputs.json("results/v76-cap-conditioning/summary.json")
    cells, prune_fits, raw = v76.load_pruning(inputs)
    qcells, quant_fits, qraw = v76.load_quantization(inputs)
    points, distill_fit, drows = v76.load_distillation(inputs)
    reproduced = {"pruning": v76.predict_anchors(raw, prune_fits["primary"], "pruning"),
                  "quantization": v76.predict_anchors(qraw, quant_fits["primary"], "quantization"),
                  "distillation": drows}
    for arm, rows in reproduced.items():
        require(rows == stored["arms"][arm]["rows"], f"{arm}: frozen rows/predictions changed")
    fits = {"pruning": prune_fits["primary"], "quantization": quant_fits["primary"], "distillation": distill_fit}
    scales = scale_audit(inputs, stored, fits)
    shape = shape_audit(cells, fits["pruning"])
    identity = identity_audit(inputs, stored)
    rows = stored["arms"]["pruning"]["rows"]
    frozen_digest = v76.digest(canonical(rows))
    original = v76.score(rows)
    require(original == stored["arms"]["pruning"]["panels"]["all"], "Original five-state score not reproduced")
    unique_rows, paired_rows = merge_rows(rows, True), merge_rows(rows, False)
    unique, paired = v76.score(unique_rows), v76.score(paired_rows)
    require(unique == stored["arms"]["pruning"]["unique_outcome_sensitivity"]["panels"]["all"],
            "Four-state result disagrees with V76 unique-outcome sensitivity")
    require(unique["n_clusters"] == paired["n_clusters"] == 4, "Incorrect merged cluster count")
    require(v76.digest(canonical(rows)) == frozen_digest, "Stored predictions were mutated")
    changes = {}
    for cap in (*CAPS, "macro"):
        changes[cap] = {}
        for method in ("B", "C", "D"):
            old, new = [p["scores"][cap]["gains_of_A"][method] for p in (original, unique)]
            changes[cap][method] = {"gain_change_nats": new["gain_nats"] - old["gain_nats"],
                "ci_endpoint_changes_nats": [b-a for a, b in zip(old["ci95_nats"], new["ci95_nats"])],
                "interval_direction_changed": new["interval_direction"] != old["interval_direction"]}
        if cap in CAPS:
            shape["capabilities"][cap]["confirmation_gain_attribution"] = {
                name: {"gain_B_minus_A_nats": panel["scores"][cap]["gains_of_A"]["B"]["gain_nats"],
                       "share_of_sum_of_capability_gains": panel["scores"][cap]["gains_of_A"]["B"]["gain_nats"] /
                            sum(panel["scores"][c]["gains_of_A"]["B"]["gain_nats"] for c in CAPS)}
                for name, panel in (("original_five_labels", original), ("merged_four_states", unique))}
    interpretation = {
        "pruning": "The measured A-over-B gain is primarily a non-proportional aggregate QA curve "
                   "shape (including a density sign reversal), not a prohibited negative scale. "
                   "B already minimizes signed OLS over its entire one-scale function family. "
                   "Math/code curves are much closer to proportional; their gain intervals include zero.",
        "source_direction": "QA directions vary across development sources, and its arithmetic mean "
                            "is positive at several densities with a negative median/majority. "
                            "A and B are both source-free: neither adapts its sign to a source at fixed "
                            "density. These diagnostics establish heterogeneity, but do not causally "
                            "decompose its contribution to the aggregate curve or confirmation gain. "
                            "The observed A-B contrast measures extra curve flexibility; it does not "
                            "demonstrate learned source-specific direction.",
        "constraints": "No nonnegativity, regularization, clipping, or optimizer-bound artifact in "
                       "pruning. B is structurally restricted to proportional curves; OLS versus "
                       "confirmation MAE is a criterion choice, not a sign constraint. V76's other "
                       "fitting conventions are sensitivity analyses, not refits on confirmation labels.",
        "distillation": "Distillation A==B equivalence is algebraic for the single-coefficient forms "
                        "and supports no conclusion about conditioning. With w=log(1+E), "
                        "a_c=(w^T y_c)/(w^T w), a=(w^T mean_c(y_c))/(w^T w), and a!=0, "
                        "s_c=((a*w)^T y_c)/((a*w)^T(a*w))=a_c/a. Thus A=a_c*w=a*s_c*w=B. "
                        "The equality is guaranteed by parameterization; it is not empirical evidence "
                        "that conditioning helps, is unnecessary, or generalizes."}
    return {"schema_version": 1, "experiment": "V79 audit of V76 capability conditioning",
        "units": "Signed post-intervention minus dense cross entropy and MAE gains in nats",
        "gain_convention": "Gain of A over B = MAE(B)-MAE(A); positive favors A. "
                           "The literal MAE(A)-MAE(B) contrast has the opposite sign.",
        "scale_audit": scales, "pruning_shape_audit": shape, "duplicate_identity": identity,
        "pruning_confirmation": {
            "bootstrap": stored["bootstrap"], "original_five_labels": original,
            "merged_four_states": unique, "merged_cluster_original_row_weighting": paired,
            "primary_weighting": "Count each physical state-density-capability once: merge identical "
                                 "cells within the 2.8B cluster, retaining step143000 as representative. "
                                 "36 rows, 12 state-density cells, four equally weighted states. "
                                 "This is also the mean error within each cluster followed by equal-state averaging.",
            "row_weighting_sensitivity": "Keep all 45 rows and only pair the two aliases in one cluster. "
                                         "Four bootstrap units, but the duplicate state retains double point-estimate "
                                         "weight. This preserves original gains and changes only their intervals.",
            "changes_from_original": changes,
            "matches_existing_v76_unique_outcome_sensitivity": True,
            "unchanged_original_rows_sha256": v76.digest(canonical(rows)),
            "frozen_rows": rows, "representative_rows": unique_rows,
            "unique_v72_panel": v76.score([r for r in unique_rows if r["panel"] == "v72_repeat"])},
        "reuse_inventory": inventory(inputs), "interpretation": interpretation,
        "distillation_equivalence": {"shared_coefficient": distill_fit["shared_coefficient"],
            "coefficients": distill_fit["coefficients"],
            "max_prediction_difference_A_B": stored["arms"]["distillation"]["max_prediction_difference_A_B"]},
        "validation": {"cpu_only": True, "all_216_frozen_capability_rows_reproduced": True,
            "negative_scales_executed_successfully": True, "original_five_state_scores_reproduced_exactly": True,
            "no_confirmation_refitting": True, "historical_inputs_hash_verified": True,
            "original_v76_files_unchanged": True},
        "outputs": {"summary_json": "results/v79-cond-audit/summary.json",
            "summary_md": "results/v79-cond-audit/summary.md",
            "staged_table": "results/v79-cond-audit/paper/paper/tables/cond_audit.tex",
            "staged_code_mirror": "results/v79-cond-audit/paper/code/analysis/v79_cond_audit.py",
            "requested_table_destination": "paper/paper/tables/cond_audit.tex",
            "requested_code_destination": "paper/code/analysis/v79_cond_audit.py",
            "paper_destination_status": "Staged under results to obey explicit write-only scope, as in V76/V77."}}


def gain_text(gain, digits=6):
    value, ci = gain["gain_nats"], gain["ci95_nats"]
    return f"{value:.{digits}f}" + (f" [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]" if ci else " [CI unavailable]")


def markdown(result):
    p = result["pruning_confirmation"]
    old, new, paired = [p[k] for k in ("original_five_labels", "merged_four_states",
                                      "merged_cluster_original_row_weighting")]
    shape = result["pruning_shape_audit"]
    lines = ["# V79 audit of V76 capability conditioning", "",
        "CPU audit using frozen V76 predictions and historical development inputs. No V76 files were modified. "
        "All gains below use **MAE(B) minus MAE(A)** (positive favors A); literal MAE(A)-MAE(B) has the opposite sign.", "",
        "## 1. Variant B scales and fitting constraints", "",
        "From `results/v76-cap-conditioning/summary.json`, `arms.<arm>.fit.scales`, reproduced from development inputs:", "",
        "| Arm | Math scale | Code scale | QA scale | Negative allowed? | Negative occurred? |",
        "|---|---:|---:|---:|:---:|:---:|"]
    for arm, audit in result["scale_audit"]["arms"].items():
        lines.append("| " + arm.capitalize() + " | " + " | ".join(f"{audit['scales'][c]:.12f}" for c in CAPS)
                     + " | Yes | No |")
    lines += ["", result["scale_audit"]["conclusion"], "",
        "The actual `corrections` implementation at `analysis/v76_cap_conditioning.py:116` uses "
        "`scale = x @ y / (x @ x)` in its OLS branch. `fit_anchors` passes the shared anchors and "
        "capability median anchors with equal anchor weight. There is no intercept, clipping of the scale, "
        "ridge penalty, or sign bound. The source hash matches the one recorded in V76's summary. "
        "Executing that same function on known negative targets returned "
        "scales `(2, -3, -4)`. All stored sensitivity scales are also positive (full values in JSON).", "",
        "## 2. Pruning gain attribution", "",
        "Fit the signed multiple `s_c h(d)` through the origin on the seven development median anchors. "
        "The reported variance fraction is centered `R² = 1 - Σ(m_c-s_c h)² / Σ(m_c-mean(m_c))²`; "
        "no extra intercept is fitted. The uncentered quantity uses `Σm_c²` in the denominator and is shown "
        "only to remove ambiguity about through-origin regression conventions.", "",
        "| Capability | Centered R² | Uncentered R² | Median sign pattern, increasing density | Five-label gain | Four-state gain |",
        "|---|---:|---:|---|---:|---:|"]
    for cap, data in shape["capabilities"].items():
        signs = " ".join("+" if s > 0 else "−" if s < 0 else "0" for s in data["signs_in_density_order"])
        lines.append(f"| {cap} | {data['r2']:.6f} | {data['uncentered_r2']:.6f} | {signs} | "
                     f"{old['scores'][cap]['gains_of_A']['B']['gain_nats']:.6f} | "
                     f"{new['scores'][cap]['gains_of_A']['B']['gain_nats']:.6f} |")
    lines += ["", "Density order: " + ", ".join(map(str, shape["density_order"])) + ".", "",
        "| Density | Shared h | Math median | Code median | QA median |",
        "|---:|---:|---:|---:|---:|"]
    for density in shape["density_order"]:
        key = str(density)
        values = [shape["shared_anchors"][key], *[shape["capabilities"][c]["median_anchors"][key] for c in CAPS]]
        lines.append(f"| {density} | " + " | ".join(f"{v:.9f}" for v in values) + " |")
    qa = shape["capabilities"]["qa"]
    lines += ["", "**QA changes sign across density:** its median is positive at 0.55, then negative "
        "at every observed density from 0.60 through 0.90. Linear interpolation crosses zero at "
        f"d={qa['linear_interpolated_zero_crossings'][0]:.6f}. The shared curve has a different sign pattern; "
        "a single positive or negative scale cannot reproduce the QA curve. Low-density positive anchors "
        "dominate its OLS projection, giving a positive QA scale despite predominantly negative QA medians.", "",
        "Per-source QA direction variability uses the **arithmetic mean**, not the median, at each density. "
        "Each available development source state receives one vote; opposite means `response * mean < 0`.", "",
        "| Density | Sources | QA mean | QA median | Positive / negative / zero | Opposite mean | Share |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for density in shape["density_order"]:
        d = qa["source_directions_by_density"][str(density)]
        lines.append(f"| {density} | {d['n_sources']} | {d['mean']:.9f} | {d['median']:.9f} | "
                     f"{d['n_positive']} / {d['n_negative']} / {d['n_zero']} | "
                     f"{d['n_opposite_mean']}/{d['n_sources']} | {100*d['opposite_mean_share']:.1f}% |")
    lines += ["", shape["density_coverage_note"], "",
        "A majority can oppose the arithmetic mean when a few large positive responses dominate it. "
        "This occurs here at 0.60, 0.65, 0.70, 0.80 and 0.90. The small positive mean at 0.90 is reported "
        "with its actual sign; no significance threshold or zero tolerance is imposed. Individual response "
        "values and source IDs are in JSON, for all three capabilities.", "",
        result["interpretation"]["pruning"], "", result["interpretation"]["source_direction"], "",
        result["interpretation"]["constraints"], "",
        f"QA accounts for {100*qa['confirmation_gain_attribution']['original_five_labels']['share_of_sum_of_capability_gains']:.2f}% "
        f"of the original summed capability gains and {100*qa['confirmation_gain_attribution']['merged_four_states']['share_of_sum_of_capability_gains']:.2f}% "
        "after deduplication. These are descriptive shares of the observed gain, not a causal variance decomposition.", "",
        "## 3. Four-cluster pruning confirmation", "",
        result["duplicate_identity"]["conclusion"], "",
        "The stored V77 full-tensor audit (`results/v77-model-arch/pythia_2_8b_tensor_identity.json`) "
        "checks all 388 learned tensors: 387 are byte-identical, and the remaining layer-normalization bias "
        "has one +0.0/-0.0 difference. Its blob identities match the V72 freeze. V79 checks this stored evidence "
        "and independently verifies that the three paired measurement files and all V76 A/B/C/D predictions "
        "are identical across labels. It does not reload model weights.", "",
        p["primary_weighting"], "",
        "All frozen predictions are retained unchanged in JSON; the representative rows are a subset of them. "
        "Use V76's 5,000 paired state-bootstrap resamples, NumPy PCG64 seed 0, and 95% percentile intervals. "
        "A sampled state brings every density and capability, using the same draws across methods/capabilities. "
        "The intervals condition on the fixed development fit and probes and are descriptive with only four states.", "",
        "| Capability | A MAE, 5 labels | B MAE, 5 labels | Gain, 5 labels [95% CI] | A MAE, 4 states | B MAE, 4 states | Gain, 4 states [95% CI] |",
        "|---|---:|---:|---|---:|---:|---|"]
    for cap in (*CAPS, "macro"):
        a, b = old["scores"][cap], new["scores"][cap]
        lines.append(f"| {cap} | {a['mae']['A']:.6f} | {a['mae']['B']:.6f} | {gain_text(a['gains_of_A']['B'])} | "
                     f"{b['mae']['A']:.6f} | {b['mae']['B']:.6f} | {gain_text(b['gains_of_A']['B'])} |")
    old_gain, new_gain = [x["scores"]["macro"]["gains_of_A"]["B"] for x in (old, new)]
    lines += ["", f"Macro gain decreases by {old_gain['gain_nats']-new_gain['gain_nats']:.6f} nats "
        f"({100*(1-new_gain['gain_nats']/old_gain['gain_nats']):.2f}%); relative MAE reduction changes from "
        f"{old_gain['relative_gain_percent']:.2f}% to {new_gain['relative_gain_percent']:.2f}%. "
        "The qualitative A-versus-B reading is unchanged: QA and macro intervals remain positive; "
        "math and code intervals include zero. The macro interval shifts downward and becomes slightly narrower; "
        "deduplication need not widen every percentile interval because it also changes the empirical state distribution. "
        "These numbers exactly reproduce V76's already-stored `unique_outcome_sensitivity`. "
        "V79 establishes the weight-identity basis for treating it as the four-state comparison. "
        "V72 alone has one unique cluster, so no between-state bootstrap interval can be estimated.", "",
        "For completeness, all V76 contrasts after the same merge (positive baseline-minus-A):", "",
        "| Capability | B-A [95% CI] | C-A [95% CI] | D-A [95% CI] |",
        "|---|---|---|---|"]
    for cap in (*CAPS, "macro"):
        lines.append("| " + cap + " | " + " | ".join(gain_text(new["scores"][cap]["gains_of_A"][m])
                                                      for m in ("B", "C", "D")) + " |")
    lines += ["", "A separate **cluster-only sensitivity** avoids conflating dependence correction with reweighting: "
        + p["row_weighting_sensitivity"], "",
        "| Capability | Gain with original 45-row weighting, 4 clusters [95% CI] |",
        "|---|---|"]
    for cap in (*CAPS, "macro"):
        lines.append(f"| {cap} | {gain_text(paired['scores'][cap]['gains_of_A']['B'])} |")
    inv = result["reuse_inventory"]
    lines += ["", f"Scanned all {inv['n_files_scanned']} freeze/compare-named files under `results/` using "
        "`rg --files --hidden --no-ignore`, then matched both full tags (including `--` path variants); "
        "a broader check for separated `2.8b`, `step16000`, and `step143000` fields found the same files:", ""]
    for hit in inv["files_with_both_tags"]:
        lines.append(f"- `{hit['file']}` (first tag matches at lines "
                     + ", ".join(str(hit["matching_line_numbers"][t][0]) for t in TAGS) + ").")
    lines += ["", inv["other_analyses_counting_aliases_as_separate_states"][0]["evidence"], "",
        inv["audited_analysis"], "", inv["scope"], "",
        "## 4. Distillation interpretation", "", "**" + result["interpretation"]["distillation"] + "**", "",
        f"The fitted shared coefficient is {result['distillation_equivalence']['shared_coefficient']:.12f}, "
        "so the nonzero-shared-coefficient condition holds. Maximum frozen prediction difference A-B is "
        f"{result['distillation_equivalence']['max_prediction_difference_A_B']:.3g} nats (floating-point rounding).", "",
        "## Reproduction and outputs", "",
        "Run `python -B analysis/v79_cond_audit.py --selftest`, then `python -B analysis/v79_cond_audit.py`. "
        "Use `--check` to recompute and verify all output bytes without writing. The audit reproduces all "
        "216 V76 frozen capability rows, checks historical input hashes, and verifies that every V76 file "
        "is unchanged before and after the run. Input SHA-256 values, raw signed responses, source identities, "
        "scale code evidence, and complete scores are recorded in `summary.json`.", "",
        "Paper artifacts are staged under this results directory to obey the explicit write-only boundary:", "",
        "- [Table, with `[H]`](paper/paper/tables/cond_audit.tex), for `paper/paper/tables/cond_audit.tex`.",
        "- [Byte-identical code mirror](paper/code/analysis/v79_cond_audit.py), for `paper/code/analysis/v79_cond_audit.py`.",
        "- No files in the top-level paper tree are changed; no commit is made.", ""]
    return "\n".join(lines)


def latex(result):
    p = result["pruning_confirmation"]
    old, new = p["original_five_labels"], p["merged_four_states"]
    lines = [r"% Generated by analysis/v79_cond_audit.py; frozen predictions unchanged.",
        r"\begin{table}[H]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
        r"\caption{V79 audit of capability conditioning. Variant B uses unrestricted signed scales. "
        r"Pruning confirmation counts the two identical Pythia-2.8B revisions as one state. "
        r"Positive $\Delta=B-A$ is the MAE reduction from A, in nats.}", r"\label{tab:cond_audit}",
        r"\begin{tabular}{lrrr}", r"\toprule", r"B scale & Math & Code & QA \\", r"\midrule"]
    for arm, audit in result["scale_audit"]["arms"].items():
        lines.append(arm.capitalize() + " & " + " & ".join(f"{audit['scales'][c]:.6f}" for c in CAPS) + r" \\")
    lines += [r"\midrule", r"Pruning anchor $R^2$ & " + " & ".join(
        f"{result['pruning_shape_audit']['capabilities'][c]['r2']:.4f}" for c in CAPS) + r" \\",
        r"\bottomrule", r"\end{tabular}", r"\par\medskip", r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrll}",
        r"\toprule", r"Capability & A, 4 states & B, 4 states & $\Delta$, 5 labels [95\% CI] & $\Delta$, 4 states [95\% CI] \\",
        r"\midrule"]
    for cap in (*CAPS, "macro"):
        a, b = old["scores"][cap], new["scores"][cap]
        label = "QA" if cap == "qa" else cap.capitalize()
        lines.append(f"{label} & {b['mae']['A']:.4f} & {b['mae']['B']:.4f} & "
                     + gain_text(a["gains_of_A"]["B"], 4) + " & " + gain_text(b["gains_of_A"]["B"], 4) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\par\smallskip",
        r"\begin{minipage}{0.99\linewidth}\footnotesize "
        r"All fitted scales, including stored sensitivities, are positive, although negatives are permitted. "
        r"$R^2=1-\sum_d(m_c-s_ch)^2/\sum_d(m_c-\bar m_c)^2$ uses seven development median anchors "
        r"and no fitted intercept. QA is positive at $d=0.55$ and negative at $d=0.60,0.65,0.70,0.75,0.80,0.90$; "
        r"math/code medians remain positive. The shares of sources opposite the arithmetic-mean QA direction "
        r"at those seven densities are $1/8,10/16,7/8,12/16,0/4,14/16,10/16$, respectively. "
        r"The gain primarily reflects non-proportional aggregate QA shape; neither source-free variant "
        r"learns source-specific direction. It is not caused by a nonnegative-scale constraint. "
        r"The four-state comparison gives each duplicate cell one vote (12 cells, 36 capability rows); "
        r"5,000 paired state-bootstrap draws, seed 0, fixed fits. Intervals are descriptive with few states. "
        r"QA/macro intervals remain positive and math/code intervals include zero; these results reproduce "
        r"V76's unique-outcome sensitivity. Keeping all 45 rows but merging bootstrap labels preserves "
        r"the original point estimates (see accompanying summary). V72 freeze/compare also count the "
        r"two aliases separately; no other such files matched both tags. "
        r"\textbf{Distillation A$=$B is algebraic for the single-coefficient forms "
        r"($a_c=a s_c$, $a\ne0$) and supports no conclusion about conditioning.} "
        r"\end{minipage}", r"\end{table}", ""]
    return "\n".join(lines)


def selftest():
    v76.selftest()
    projection = curve_projection([1, 2, 3], [-2, -4, -6])
    close(projection["scale"], -2, "Negative projection")
    close(projection["r2"], 1, "Exactly proportional negative curve")
    require(curve_projection([1, 2, 3], [-1, 0, 1])["r2"] < 1, "Non-proportional shape lost")
    d = direction([-1, -1, 10], ["a", "b", "c"])
    close(d["opposite_mean_share"], 2/3, "Majority may oppose mean")
    require(direction([-1, 1], ["a", "b"])["opposite_mean_share"] is None, "Zero mean has no sign")
    require(direction([0, 1], ["a", "b"])["n_opposite_mean"] == 0, "Zero response is not opposite")
    rows = [{"state": state, "cluster": state, "x": "0.7", "capability": c, "actual": 0.,
             "predictions": {"A": 0., "B": error, "C": error, "D": error}}
            for state, error in [(TAGS[0], 3.), (TAGS[1], 3.), ("other", 1.)] for c in CAPS]
    unique, paired = merge_rows(rows, True), merge_rows(rows, False)
    close(v76.score(unique)["scores"]["macro"]["mae"]["B"], 2, "Unique-state weighting")
    close(v76.score(paired)["scores"]["macro"]["mae"]["B"], 7/3, "Retained-row weighting")
    require(v76.score(unique)["n_clusters"] == v76.score(paired)["n_clusters"] == 2, "Cluster pairing failed")
    require(v76.score(merge_rows(rows[:6], True))["scores"]["macro"]["gains_of_A"]["B"]["ci95_nats"] is None,
            "One unique state cannot give a state interval")
    altered = [dict(r, actual=1.) if r["state"] == TAGS[1] else r for r in rows]
    try:
        merge_rows(altered, True)
    except ValueError:
        pass
    else:
        raise AssertionError("Cannot collapse unequal measurements")
    print("PASS: signed projection, direction/zero handling, alias equality, cluster pairing and weighting")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--check", action="store_true", help="Verify exact output bytes without writes")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    before = tree_hashes(ROOT / "results/v76-cap-conditioning")
    inputs = v76.Inputs()
    script = inputs.read("analysis/v79_cond_audit.py")
    result = build(inputs)
    result["v76_tree_sha256"] = before
    result["inputs_sha256"] = dict(sorted(inputs.hashes.items()))
    outputs = {OUT / "summary.json": (json.dumps(result, indent=2, allow_nan=False) + "\n").encode(),
               OUT / "summary.md": markdown(result).encode(),
               OUT / "paper/paper/tables/cond_audit.tex": latex(result).encode(),
               OUT / "paper/code/analysis/v79_cond_audit.py": script}
    inputs.verify()
    require(before == tree_hashes(ROOT / "results/v76-cap-conditioning"), "V76 tree changed during audit")
    for path, raw in outputs.items():
        require(path.resolve().is_relative_to(OUT.resolve()) and not path.is_symlink(), f"Output scope violation: {path}")
        if args.check:
            require(path.read_bytes() == raw, f"Output differs from recomputation: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    inputs.verify()
    require(before == tree_hashes(ROOT / "results/v76-cap-conditioning"), "V76 tree changed after audit")
    for cap, data in result["pruning_confirmation"]["merged_four_states"]["scores"].items():
        print(f"{cap}: four-state B-A = {gain_text(data['gains_of_A']['B'])}")
    print(f"{'CHECKED' if args.check else 'WROTE'} four artifacts; 216 frozen rows reproduced; V76 unchanged; CPU only")


if __name__ == "__main__":
    main()
