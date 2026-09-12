#!/usr/bin/env python3
"""CPU-only information-budget audit of existing losses and frozen V28/V30 fits.

python analysis/v35_info_budget.py [--n-boot 10000] [--dry-run]
No population, mapping, exponent, or model fitting. Only the explicitly
diagnostic Oracle scalar is optimized against target outcomes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from . import prediction_audit as audit
    from . import v25_distill_delta as v25
    from . import v28_new_source_prediction as v28
    from . import v30_quant_shape_candidates as v30
except ImportError:
    import prediction_audit as audit
    import v25_distill_delta as v25
    import v28_new_source_prediction as v28
    import v30_quant_shape_candidates as v30

import numpy as np  # prediction_audit sets CPU BLAS limits first.

ROOT = audit.ROOT
OUT = ROOT / "results/v35-info-budget"
REPORT = ROOT / "paper/docs/INFO_BUDGET_K0K1.md"
V28 = ROOT / "results/v28-new-source-pred/frozen_predictions.json"
V30 = ROOT / "results/v30-quant-candidates/summary.json"
V30B = ROOT / "results/v30b-quant-regions/summary.json"
CAPS = audit.CAPABILITIES
ARMS = ("pruning", "quantization", "distillation")
BUDGETS = ("zero", "K0", "K1", "Oracle")
QUANT_SHAPE = "shared_eta"
NEAR_ZERO = .01  # Annotation only, in nats; never divides/censors a response.
ADEQUACY = .1  # Descriptive absolute-error tolerance, not a significance test.
SHAPE_EPS = 1e-12
CALIBRATION = {"pruning": .9, "quantization": 4, "distillation": 75}
CALIBRATION_RULES = {
    "pruning": "V28 pre-specified dev density 0.9, unchanged; no target-based tuning.",
    "quantization": "V30 dev paired 4-to-5 design's 4-bit anchor, unchanged; shared-eta shape and mapping stay frozen.",
    "distillation": "Smallest positive budget in fixed V25 dev grid: min(75,150,300,600)=75. Cost rule uses coordinates only, no outcomes. V35 replaces V28's D150 anchor for ALL students, without changing any frozen coefficient.",
}
# Architecture-only transcription from locally cached Qwen/Qwen3-14B config,
# snapshot 40c069824f4251a91eefaf281ebe4c544efd3e18/config.json. No weights read.
# The matrix convention matches V28; no nominal-14B substitution into its map.
QWEN14_ARCH = {"hidden_size": 5120, "intermediate_size": 17408,
               "num_hidden_layers": 40, "num_attention_heads": 40,
               "num_key_value_heads": 8, "head_dim": 128}


def architecture_n0(a):
    h, d = a["hidden_size"], a["head_dim"]
    return a["num_hidden_layers"] * (2*h*a["num_attention_heads"]*d +
           2*h*a["num_key_value_heads"]*d + 3*h*a["intermediate_size"])


def held_out_fit(folds, model, key="fit"):
    matches = [f for f in folds if v28.canonical(f["held_out"]) == v28.canonical(model)]
    if len(matches) != 1:
        raise ValueError(f"Need exactly one frozen LOMO fold: {model}")
    return matches[0][key]


def assert_held_out(models, target):
    if v28.canonical(target) in {v28.canonical(m) for m in models}:
        raise ValueError(f"Target leaked into frozen training: {target}")


def predict_k0(arm, fit, basic, cap, coordinates):
    """The target interface accepts ONLY basic/dense inputs and coordinates."""
    if set(basic) != {"model", "N0", "family", "dense_loss"}:
        raise ValueError("K0 requires exactly model/N0/family/dense_loss; no compressed inputs")
    audit.finite(basic["dense_loss"])
    if arm == "pruning":
        assert_held_out(fit["mapping"]["train_models"], basic["model"])
        return v28.predict_arm(fit, basic, coordinates, mode="A")
    if arm == "quantization":
        assert_held_out(fit["mappings"][cap]["train_models"], basic["model"])
        # Preserve the actual frozen feature convention: V30 uses nominal size.
        target = v30.basic_input({"model": v30.canonical(basic["model"]),
                                 "reference_loss": basic["dense_loss"],
                                 **v30.model_features(v30.canonical(basic["model"]))})
        return [v30.predict_candidate(fit, target, cap, c, 16) for c in coordinates]
    if arm == "distillation":
        assert_held_out([r.split("|")[0] for r in fit["train_row_ids"]], basic["model"])
        return [fit["coefficient"] * v28.transfer_shape(cap, c) for c in coordinates]
    raise ValueError(f"Unknown arm: {arm}")


def shape_values(arm, fit, cap, coordinates):
    if arm == "pruning":
        return np.array([v28.shape(arm, c, fit["gamma"]) for c in coordinates])
    if arm == "quantization":
        return np.array([v30.normalized_shape(c, fit["candidate"], 16, fit["eta"])
                         for c in coordinates])
    return np.array([v28.transfer_shape(cap, c) for c in coordinates])


def predict_k1(arm, fit, basic, cap, coordinates, calibration):
    """Exactly one fixed configuration, with one capability response here.

    A near-zero *observed response* stays signed, including exact zero. If the
    frozen shape denominator is numerically zero, explicitly fall back to K0.
    This is never a second point, a clipped amplitude, or a target-selected anchor.
    """
    if set(basic) != {"model", "N0", "family", "dense_loss"}:
        raise ValueError("K1 basic inputs must exclude compressed outcomes")
    if set(calibration) != {"coordinate", "loss"} or calibration["coordinate"] != CALIBRATION[arm]:
        raise ValueError("K1 requires exactly the fixed calibration configuration")
    if CALIBRATION[arm] in coordinates:
        raise ValueError("K1 predictive scores must exclude calibration")
    response = audit.finite(calibration["loss"]) - audit.finite(basic["dense_loss"])
    x0, = shape_values(arm, fit, cap, [CALIBRATION[arm]])
    status = {"coordinate": CALIBRATION[arm], "response": response,
              "near_zero_response": abs(response) <= NEAR_ZERO,
              "shape_at_calibration": float(x0), "target_configurations_used": 1,
              "fallback": None, "amplitude": None}
    if abs(x0) <= SHAPE_EPS:
        status["fallback"] = "numerically_zero_shape_use_K0"
        return predict_k0(arm, fit, basic, cap, coordinates), status
    amplitude = response / x0
    status["amplitude"] = audit.finite(amplitude)
    return (amplitude * shape_values(arm, fit, cap, coordinates)).tolist(), status


def oracle_scalar(shape, observed):
    """Exact signed L1 scalar: weighted median(y/x, weight=abs(x)).

    Minimize mean |a*x-y| over ALL supplied target points, including calibration.
    Midpoint of the minimizing interval resolves ties without favoring any cell.
    Zero-shape observations remain in the error objective; they cannot identify a.
    """
    x, y = np.asarray(shape, float), np.asarray(observed, float)
    if x.ndim != 1 or x.shape != y.shape or not len(x) or not np.isfinite([x, y]).all():
        raise ValueError("Oracle requires matched finite nonempty vectors")
    nz = x != 0
    if not nz.any():
        return 0.
    ratios, weights = y[nz]/x[nz], np.abs(x[nz])
    order = np.argsort(ratios, kind="stable")
    ratios, weights = ratios[order], weights[order]
    cumulative = np.cumsum(weights)
    half = cumulative[-1]/2
    lo = min(int(np.searchsorted(cumulative, half, side="left")), len(ratios)-1)
    hi = min(int(np.searchsorted(cumulative, half, side="right")), len(ratios)-1)
    return audit.finite((ratios[lo]+ratios[hi])/2)


def flags(arm, coordinate, response):
    response = float(response)
    return {"near_zero_response": abs(response) <= NEAR_ZERO,
            "negative_response": response < 0,
            "collapse": (coordinate <= 3 if arm == "quantization" else response > 1.),
            "region": ("collapse_bits" if coordinate <= 3 else "measurable_bits" if coordinate <= 5
                       else "high_bits") if arm == "quantization" else "all"}


def evaluate_target(panel, cap):
    arm, model, fit = panel["arm"], panel["model"], panel["fits"][cap]
    coords = sorted(panel["losses"])
    cal = CALIBRATION[arm]
    if cal not in coords or len(coords) < 2:
        raise ValueError(f"Need the fixed calibration and at least one test point: {model}/{arm}")
    basic = {"model": model, "family": panel["family"], "N0": panel["N0"],
             "dense_loss": panel["dense"][cap]}
    k0 = predict_k0(arm, fit, basic, cap, coords)
    rest = [c for c in coords if c != cal]
    point = {"coordinate": cal, "loss": panel["losses"][cal][cap]}
    k1, calibration = predict_k1(arm, fit, basic, cap, rest, point)
    k1 = dict(zip(rest, k1))
    # Calibration self-fit is present only in explicitly diagnostic all-target scores.
    k1[cal] = calibration["response"] if calibration["fallback"] is None else k0[coords.index(cal)]
    dense_by_coordinate = panel.get("dense_by_coordinate", dict.fromkeys(coords, panel["dense"]))
    y = np.array([panel["losses"][c][cap] - dense_by_coordinate[c][cap] for c in coords])
    x = shape_values(arm, fit, cap, coords)
    amplitude = oracle_scalar(x, y)
    rest_mask = np.array([c != cal for c in coords])
    score_bound = oracle_scalar(x[rest_mask], y[rest_mask])
    records = [{"model": model, "capability": cap, "coordinate": c,
                "row_id": f"{arm}|{model}|{cap}|{c:g}", "observed": float(y[i]),
                "dense_loss": dense_by_coordinate[c][cap], "loss": panel["losses"][c][cap],
                "is_calibration": c == cal, "shape": float(x[i]),
                "predictions": {"zero": 0., "K0": float(k0[i]), "K1": k1[c],
                                "Oracle": float(amplitude*x[i])},
                **flags(arm, c, y[i])} for i, c in enumerate(coords)]
    train_models = (fit["mapping"]["train_models"] if arm == "pruning" else
                    fit["mappings"][cap]["train_models"] if arm == "quantization" else
                    sorted({r.split("|")[0] for r in fit["train_row_ids"]}))
    return {"model": model, "capability": cap, "basic_inputs": basic,
            "frozen_train_models": train_models,
            "frozen_shape_parameters": {k: fit[k] for k in ("gamma", "eta", "candidate", "form") if k in fit},
            "calibration": calibration, "fit_source": panel["fit_source"],
            "oracle": {"amplitude": amplitude, "n_target_configurations_used": len(coords),
                       "fit_coordinates": coords, "objective": "all-target MAE, including calibration",
                       "all_target_mae": float(np.mean(np.abs(amplitude*x-y))),
                       "remainder_only_mae_lower_bound": float(np.mean(np.abs(score_bound*x[rest_mask]-y[rest_mask]))),
                       "remainder_bound_caveat": "Additional diagnostic only; trivial with one remaining point; not the all-target Oracle."},
            "records": records}


class Inputs:
    """Hash each dependency before reading; write_outputs rechecks every hash."""
    def __init__(self):
        self.hashes = {}

    def read(self, path):
        self.hashes.update(audit.provenance([path]))
        return audit.read_json(path)


def parse_table(table, arm):
    anchor = "1.0" if arm == "pruning" else ("16" if "16" in table else "dense")
    if anchor not in table:
        raise ValueError("K0 requires a true dense/16 anchor; compressed fallback is forbidden")
    dense = {c: audit.finite(table[anchor][c]) for c in CAPS}
    losses = {}
    for key, values in table.items():
        if key.startswith("_") or key in ("dense", "16", "1.0"):
            continue
        coordinate = audit.finite(key)
        valid = 0 < coordinate < 1 if arm == "pruning" else coordinate == int(coordinate) and 2 <= coordinate < 16
        if not valid or coordinate in losses:
            raise ValueError(f"Invalid/duplicate configuration: {key}")
        losses[coordinate] = {c: audit.finite(values[c]) for c in CAPS}
    if min([*dense.values(), *(v for b in losses.values() for v in b.values())]) < 0:
        raise ValueError("Absolute CE losses must be nonnegative")
    return dense, losses


def load_panels():
    inputs = Inputs()
    f28, f30, f30b = (inputs.read(p) for p in (V28, V30, V30B))
    if f28.get("freeze_sha256") != v28.seal(f28) or f30["version"] != 30 or f30b["version"] != "30b":
        raise ValueError("Frozen artifact integrity/version mismatch")
    if f28["modes"]["B"]["calibration_coordinates"]["pruning"] != CALIBRATION["pruning"]:
        raise ValueError("Pruning calibration differs from V28 freeze")
    assert CALIBRATION["distillation"] == min(v25.BUDGETS)
    metadata = f28["metadata"]["models"]
    # Resolve fits and coordinate rules before any prospective outcomes are read.
    quant_full = f30b["qwen3_14b_precheck"]["protocols"]["all_bits"]["frozen_fits"][QUANT_SHAPE]
    roster = [(m, i, "development_lomo") for m, i in sorted(metadata.items())]
    roster += [("Qwen3-8B", f28["metadata"]["qwen3_8b"], "prospective"),
               ("Qwen3-14B", {"family": "qwen3", "N0": architecture_n0(QWEN14_ARCH)}, "prospective")]
    panels, excluded = [], []
    for arm in ("pruning", "quantization"):
        for model, info, cohort in roster:
            if arm == "pruning":
                tag = "Qwen--Qwen3-8B" if model == "Qwen3-8B" else model
                path = ROOT / "results/v6-capability-geometry" / tag / "prune_losses.json"
                fits = {c: (held_out_fit(f28["methods"][arm]["by_capability"][c]["lomo"]["folds"], model)
                            if cohort == "development_lomo" else f28["methods"][arm]["by_capability"][c]["fit"])
                        for c in CAPS}
                origin = "V28 saved dev-LOMO" if cohort == "development_lomo" else "V28 saved full-dev map (V29 diagnosis form)"
            else:
                tag = "Qwen--"+model if model.startswith("Qwen3") and cohort == "development_lomo" else model
                path = ROOT / "results/v10-quantization" / tag / "quant_losses.json"
                fit = (held_out_fit(f30["panels"]["broad"]["folds"], model, "fits")[QUANT_SHAPE]
                       if cohort == "development_lomo" else quant_full)
                fits = dict.fromkeys(CAPS, fit)
                origin = ("V30 broad saved dev-LOMO shared_eta" if cohort == "development_lomo"
                          else "V30b saved all_bits full-dev shared_eta; outcome-free precheck fit")
            table = inputs.read(path)
            dense, losses = parse_table(table, arm)
            # Verify duplicates rather than counting Qwen14 twice.
            aliases = []
            if arm == "pruning" and model == "Qwen3-14B":
                alias = path.parent.parent / "Qwen--Qwen3-14B/prune_losses.json"
                if alias.exists():
                    if parse_table(inputs.read(alias), arm) != (dense, losses):
                        raise ValueError("Conflicting Qwen14 pruning aliases")
                    aliases.append(str(alias.relative_to(ROOT)))
            primary = {c: b for c, b in losses.items() if arm != "pruning" or c in v28.PRUNE_DEV}
            if cohort == "development_lomo":
                if arm == "pruning":
                    for cap in CAPS:
                        saved, = [r for r in f28["development_rows"][arm][cap] if r["model"] == model]
                        for c in v28.PRUNE_DEV:
                            if dense[cap] != saved["dense_loss"] or not np.isclose(
                                    losses[c][cap]-dense[cap], saved["deltas"][str(c)], rtol=0, atol=1e-12):
                                raise ValueError("Pruning dev outcomes differ from freeze")
                else:
                    saved = [r for r in f30["panels"]["broad"]["rows"] if r["model"] == v30.canonical(model)]
                    if set(losses) != {r["bit"] for r in saved}:
                        raise ValueError("Quantization dev grid differs from freeze")
                    for r in saved:
                        cap, bit = r["capability"], r["bit"]
                        basic = {"model": model, "N0": info["N0"], "family": info["family"], "dense_loss": dense[cap]}
                        predicted, = predict_k0(arm, fits[cap], basic, cap, [bit])
                        if dense[cap] != r["reference_loss"] or losses[bit][cap] != r["loss"] or not np.isclose(
                                predicted, r["predictions"][QUANT_SHAPE], rtol=0, atol=1e-10):
                            raise ValueError("Quantization dev observations/predictions differ from V30")
            panel = {"arm": arm, "model": model, "family": info.get("family", "qwen3"),
                     "N0": info["N0"], "cohort": cohort, "scope": "primary", "dense": dense,
                     "losses": primary, "fits": fits, "fit_source": origin,
                     "source_paths": [str(path.relative_to(ROOT))], "verified_duplicate_aliases": aliases,
                     "measurement_caveats": {k: v for k, v in table.items() if k.startswith("_")}}
            panels.append(panel)
            if arm == "pruning" and set(primary) != set(losses):
                panels.append({**panel, "scope": "extended_pruning", "losses": losses})
    # V25's fixed dev roster/recipe is preserved; no extra seed or control runs.
    dist_specs = [(m, b, "development_lomo") for m in v25.SIZES for b in v25.BUDGETS]
    dist_specs += [("Qwen3-4B", b, "prospective") for b in (75, 600)]
    dist = {}
    protocol = None
    for model, budget, cohort in dist_specs:
        path = ROOT / f"results/v12-distill/{model}/gpt-5.6-luna_full_{budget}/eval.json"
        p = inputs.read(path)
        if (p["student_tag"], p["teacher"], p["recipe"], p["n_per_domain"]) != (model, "gpt-5.6-luna", "full", budget):
            raise ValueError("Unexpected distillation identity")
        if p["training_mode"] != "lora":
            excluded.append({"path": str(path.relative_to(ROOT)), "model": model, "coordinate": budget,
                             "reason": "V28 recipe exclusion: full training, not LoRA", "capabilities": list(CAPS)})
            continue
        signature = [p[k] for k in ("measurement_benchmarks", "probe_source", "probe_seed",
                                    "n_probe_requested", "probe_half", "measurement_samples")]
        if protocol is not None and signature != protocol:
            raise ValueError("Distillation measurement protocol mismatch")
        protocol = signature
        recipe = f28["methods"]["distillation"]
        log, adapter = (inputs.read(path.parent / name) for name in ("train_log.json", "adapter/adapter_config.json"))
        if any(log[k] != v for k, v in recipe["frozen_training_hyperparameters"].items()) or any(
                adapter[k] != v for k, v in recipe["frozen_lora_hyperparameters"].items()):
            raise ValueError("Distillation training recipe mismatch")
        dense = {c: audit.finite(p["dense"][c]) for c in CAPS}
        post = {c: audit.finite(p["post_training"][c]) for c in CAPS}
        for c in CAPS:
            if not np.isclose(post[c]-dense[c], audit.finite(p["delta"][c]), rtol=0, atol=1e-10):
                raise ValueError("Inconsistent distillation delta")
        if model not in dist:
            fits = {c: (held_out_fit(recipe["by_capability"][c]["lomo"]["folds"], model)
                        if cohort == "development_lomo" else recipe["by_capability"][c]["fit"]) for c in CAPS}
            dist[model] = {"arm": "distillation", "model": model, "cohort": cohort, "scope": "primary",
                           "family": metadata[model]["family"], "N0": metadata[model]["N0"],
                           "dense": dense, "dense_by_coordinate": {}, "losses": {}, "fits": fits, "source_paths": [],
                           "fit_source": "V28 saved dev-LOMO transfer" if cohort == "development_lomo" else "V28 saved full-dev transfer",
                           "measurement_caveats": {"seed": "seed0 only; seeds1/2 and matched-E/uxseen controls excluded by fixed recipe/run identity"}}
        # Preserve V25's run-specific dense anchors (small historical drift).
        # K0 transfer coefficients do not use dense loss; K1 uses D75's anchor.
        dist[model]["dense_by_coordinate"][budget] = dense
        dist[model]["losses"][budget] = post
        dist[model]["source_paths"].append(str(path.relative_to(ROOT)))
    # Verify the current dev ladder against stored V28 outcomes, including D150
    # by reconstructing its saved one-point predictions.
    for cap in CAPS:
        saved = f28["methods"]["distillation"]["by_capability"][cap]["lomo"]
        for r in saved["records"]:
            panel = dist[r["model"]]
            actual = panel["losses"][r["coordinate"]][cap] - panel["dense_by_coordinate"][r["coordinate"]][cap]
            predicted, = predict_k0("distillation", panel["fits"][cap],
                                   {"model": panel["model"], "family": panel["family"], "N0": panel["N0"],
                                    "dense_loss": panel["dense"][cap]}, cap, [r["coordinate"]])
            if not np.isclose(actual, r["observed_delta"], rtol=0, atol=1e-12) or not np.isclose(predicted, r["A"], rtol=0, atol=1e-12):
                raise ValueError("Distillation dev outcomes/predictions differ from freeze")
            old_cal = panel["losses"][150][cap]-panel["dense_by_coordinate"][150][cap]
            old_prediction = old_cal*v28.transfer_shape(cap, r["coordinate"])/v28.transfer_shape(cap, 150)
            if not np.isclose(old_prediction, r["B"], rtol=0, atol=1e-12):
                raise ValueError("Distillation D150 differs from frozen calibration")
    panels.extend(dist.values())
    inputs.hashes.update(audit.provenance([Path(__file__), Path(audit.__file__), Path(v25.__file__),
                                          Path(v28.__file__), Path(v30.__file__)]))
    return panels, excluded, inputs.hashes


def summarize(records, n_boot):
    if not records:
        return {"n_models": 0, "n_cells": 0, "status": "no_measured_score_cells"}
    predictions = {b: [r["predictions"][b] for r in records] for b in BUDGETS}
    metrics, errors, draws = audit.compare_predictions(records, predictions, "zero", n_boot=n_boot)
    models = sorted({r["model"] for r in records})
    for metric in metrics.values():
        metric["improvement_over_zero"] = -metric["mae_minus_reference"]
        metric["improvement_ci95"] = [-metric["difference_ci95"][1], -metric["difference_ci95"][0]]
    contrasts = {}
    for a, b in (("K1", "K0"), ("K0", "Oracle"), ("K1", "Oracle")):
        i, j = BUDGETS.index(a), BUDGETS.index(b)
        contrasts[a+"_minus_"+b] = {"value": float((errors[:, i]-errors[:, j]).mean()),
                                   "ci95": audit.interval(draws[:, i]-draws[:, j])}
    return {"n_models": len(models), "n_cells": len(records), "models": models,
            "coordinates": sorted({r["coordinate"] for r in records}),
            "ci_note": "degenerate model-bootstrap CI (n=1); no generalization/seed uncertainty" if len(models) == 1
                       else "paired whole-model bootstrap; fixed fits, observed-cell weighting",
            "metrics": metrics, "contrasts": contrasts,
            "retained_counts": {k: sum(bool(r[k]) for r in records)
                                for k in ("near_zero_response", "negative_response", "collapse")}}


def diagnosis(remainder, full):
    """Nonexclusive descriptive flags; disclose tolerance and uncertainty."""
    oracle = full["metrics"]["Oracle"]
    k0 = full["metrics"]["K0"]
    unstable = remainder["contrasts"]["K1_minus_K0"]
    mapping_gap = full["contrasts"]["K0_minus_Oracle"]
    multiple_models = full["n_models"] > 1
    return {
        "a_functional_form_insufficient": oracle["mae"] > ADEQUACY,
        "a_ci_support": multiple_models and oracle["mae_ci95"][0] > ADEQUACY,
        "b_calibration_unstable": unstable["value"] > 1e-12,
        "b_ci_support": multiple_models and unstable["ci95"][0] > 0,
        "c_source_mapping_insufficient": k0["mae"] > ADEQUACY and oracle["mae"] <= ADEQUACY,
        "c_ci_support": multiple_models and k0["mae_ci95"][0] > ADEQUACY and oracle["mae_ci95"][1] <= ADEQUACY,
        "mapping_gap_even_if_form_bad": mapping_gap,
        "oracle_all_target_mae": oracle["mae"],
        "oracle_fraction_of_zero_mae": (oracle["mae"]/full["metrics"]["zero"]["mae"]
                                        if full["metrics"]["zero"]["mae"] > 0 else None),
        "interpretation": "Descriptive failure signatures, not uniquely identified causal mechanisms; n=1 intervals cannot support population inference.",
    }


def grouped_results(targets, n_boot):
    def group(selected):
        full = [r for t in selected for r in t["records"]]
        rest = [r for r in full if not r["is_calibration"]]
        remainder, all_target = summarize(rest, n_boot), summarize(full, n_boot)
        return {"remainder": remainder, "all_target_diagnostic": all_target,
                "diagnosis": diagnosis(remainder, all_target),
                "regions": {region: summarize([r for r in rest if r["region"] == region], n_boot)
                            for region in ("high_bits", "measurable_bits", "collapse_bits")}}
    results = {}
    for arm in ARMS:
        arm_targets = [t for t in targets if t["arm"] == arm and t["scope"] == "primary"]
        results[arm] = {
            "development_lomo": {c: group([t for t in arm_targets if t["cohort"] == "development_lomo" and t["capability"] == c]) for c in CAPS},
            "per_target": {m: {c: group([t for t in arm_targets if t["model"] == m and t["capability"] == c]) for c in CAPS}
                           for m in sorted({t["model"] for t in arm_targets})},
            "prospective_models": sorted({t["model"] for t in arm_targets if t["cohort"] == "prospective"}),
        }
    ext = [t for t in targets if t["scope"] == "extended_pruning"]
    results["pruning"]["extended_all_available"] = {c: group([t for t in ext if t["capability"] == c]) for c in CAPS}
    return results


def build_summary(n_boot=10000):
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    panels, excluded, hashes = load_panels()
    targets = []
    inventory = []
    for panel in panels:
        inventory.append({k: v for k, v in panel.items() if k not in ("fits", "losses")})
        inventory[-1]["coordinates"] = sorted(panel["losses"])
        for cap in CAPS:
            targets.append({"arm": panel["arm"], "cohort": panel["cohort"], "scope": panel["scope"],
                            **evaluate_target(panel, cap)})
    return {"version": 35, "cpu_only": True, "new_model_runs": 0, "refit_development": False,
            "endpoint": "capability LOSS: Delta L_c = compressed/post-training L_c minus own dense L_c, native-token CE nats",
            "n_boot": n_boot, "bootstrap_seed": 240526, "input_sha256": hashes,
            "calibration": {a: {"coordinate": CALIBRATION[a], "selection_rule": CALIBRATION_RULES[a],
                                "target_outcomes_used_for_selection": 0} for a in ARMS},
            "protocol": {
                "K0": "MAIN predictive goal: zero target compressed configurations; basic metadata plus own dense loss only.",
                "K1": "Predictive on remaining configurations; exactly one fixed target configuration shared across all capabilities.",
                "Oracle": "DIAGNOSTIC only: independent signed scalar per target/capability, all-target MAE optimum with frozen shape; includes calibration in its fit. Best possible performance (lower error bound) on the all-target objective, not a predictor.",
                "score_cells": "K0/K1/Oracle/zero share identical remainder cells. The calibration point is excluded from every primary score. Full-curve diagnostics include it and therefore K1 full-curve scores are not predictive.",
                "oracle_subset_caveat": "All-target optimum need not bound remainder MAE. Both all-target diagnostic MAEs and each target's remainder-only scalar lower bound are saved; the latter is trivial with one test coordinate.",
                "primary_grids": "Pruning V28 .9/.8/.7/.6 (score .8/.7/.6); quant all available V30 broad bits 3/4/5/6/8 (score excludes 4); distill V25 LoRA budgets (score excludes 75), Qwen4B seed0 D75/600. No imputed Qwen14 5-bit or Qwen4 D150/300.",
                "pruning_extended": "Every extra observed pruning density, including infills/deep collapse, is retained in a separate all-available diagnostic; same frozen gamma/map, new diagnostic scalar. These densities never change primary fits or scores.",
                "near_zero_policy": f"abs(delta)<= {NEAR_ZERO} nat is flagged, signed and retained; exact zero is retained. No division by observed response; no clipping/sign filtering. abs(shape(cal))<={SHAPE_EPS} explicitly falls back to K0 after consuming the one point.",
                "collapse_policy": "No collapse censoring. Pruning/distill delta>1 nat flagged; quant b<=3 is a coordinate-defined region (including OLMo exceptions). Flags never affect scores, calibration or fitting.",
                "diagnosis_rule": f"Descriptive adequacy threshold {ADEQUACY} nat fixed across arms/capabilities, not a scientifically established success cutoff. (a) all-target Oracle MAE>threshold; (b) remainder K1>K0; (c) all-target K0>threshold and Oracle<=threshold. Nonexclusive; raw errors/gaps and CI support provided.",
                "bootstrap": "Paired whole-model percentile bootstrap via prediction_audit.compare_predictions; all candidates and all configurations for a sampled model stay together. Equal observed-cell weights, including unequal distillation grids. Fixed fits/scalars; no refitting, item/seed/measurement uncertainty or multiplicity adjustment. Separate prospective model rows have degenerate CIs, n=1.",
                "quant_shape_status": "INCONCLUSIVE on prospective Qwen3-14B; no shape-transfer claim. All-bit gains can be driven by int3 collapse; report bit regions separately.",
                "quant_size_convention": "The frozen V30 map uses log nominal model size, family and dense L_c, not V28 non-embedding N0. This remains pre-compression basic information; converting its feature to N0 would require a forbidden refit.",
                "distill_mapping": "Frozen V28/V25 math and QA constant response, code beta*log(1+D/150). No size/family effect is fitted; K0 uses a subset of allowed basic inputs. Preserve each run's own dense anchor, including small historical drift across Gemma budgets; K1 uses only D75's dense anchor. Cross-family math/code signs differ; QA improves in both Gemma and Qwen.",
                "prospective_scope": "Rows name held-out new sources; V35 is an existing-data replay, not a newly timestamped preregistration. Qwen8 quant uses V30b shared_eta, not the original V28 fixed-4^-b forecast. Qwen8 pruning actuals cannot establish the V28 named Base checkpoint identity. Qwen14 pruning extends the frozen mapping with architecture-only N0; its old JSON lacks checkpoint revision identity. Qwen4 seed0 only; seed replicas/control runs are not extra models or extra K1 measurements.",
            },
            "qwen14_basic_metadata": {"N0": architecture_n0(QWEN14_ARCH), "architecture": QWEN14_ARCH,
                                      "N0_formula": "layers*(2*hidden*heads*head_dim + 2*hidden*kv_heads*head_dim + 3*hidden*intermediate)",
                                      "source": "Locally cached Qwen/Qwen3-14B config.json, snapshot 40c069824f4251a91eefaf281ebe4c544efd3e18; embedded architecture only, embeddings/head and 1-D weights excluded."},
            "inventory": inventory, "excluded_recipe_cells": excluded,
            "targets": targets, "results": grouped_results(targets, n_boot)}


def ci(metric, field="mae"):
    return audit.with_ci(metric[field], metric["mae_ci95" if field == "mae" else "improvement_ci95"])


def diagnosis_text(d):
    names = [("a", "functional_form_insufficient"), ("b", "calibration_unstable"), ("c", "source_mapping_insufficient")]
    labels = [f"({letter}) {name.replace('_', ' ')}" + (" [CI supports]" if d[letter+"_ci_support"] else " [point estimate]")
              for letter, name in names if d[letter+"_"+name]]
    return "; ".join(labels) or "No flag at the declared tolerance"


def render(summary):
    lines = ["# Unified information budgets: K0, K1 and Oracle (V35)", "",
             "**K0 is the main basic-input prediction goal. K1 predicts the remaining configurations after one calibration. Oracle is a post-hoc diagnostic.** "
             "This CPU-only analysis imports saved coefficients and existing losses; it performs no model runs or development refits.", "",
             summary["endpoint"] + ". Each arm keeps its own dense reference; distillation uses the student's dense loss. "
             "Native-token losses are not tokenizer-invariant, and this report does not pool arms into one MAE.", "",
             "At the declared descriptive 0.1-nat adequacy tolerance, the arms show different failure signatures:", "",
             "- **Pruning:** all three development capabilities retain appreciable Oracle form error. "
             "The Qwen math/code curves have much smaller post-hoc Oracle errors than K0, implicating source mapping; "
             "QA also has residual form error, especially Qwen14. Density-0.9 calibration worsens math and QA. "
             "A good Oracle fit is compatibility after seeing the curve, not independent shape prediction.",
             "- **Quantization:** K0 has useful collapse-region predictions; 4-bit K1 worsens all three development capabilities and all Qwen14 capabilities. "
             "Oracle's near-zero int3 error is a consequence of its all-target fit: the largest shape weight makes the L1 scalar fit that point exactly. "
             "Full-curve errors expose the remaining 4-bit/QA mismatch. The prospective shape test remains inconclusive.",
             "- **Distillation:** development math/code K0 is accurate and D75 calibration worsens it; QA retains Oracle form error. "
             "On Qwen4, calibration repairs much of the math/QA source shift but makes code worse. "
             "The two-budget all-target Oracle still misses code/QA at the stated tolerance; its code remainder error is zero because it fits D600 exactly. "
             "Neither that zero nor the trivial one-test-point lower bound establishes shape transfer.", "",
             "## Information and scoring contract", ""]
    for arm, rule in summary["calibration"].items():
        lines.append(f"- **{arm}: K1 coordinate {rule['coordinate']}**. {rule['selection_rule']}")
    lines += ["", "The same coordinate is used for every held-out model and capability in an arm. "
              "These rules use development coordinates/designs only. They are specified for this retrospective audit, not claimed to predate the existing target runs.", ""]
    for key in ("score_cells", "Oracle", "oracle_subset_caveat", "primary_grids", "near_zero_policy", "collapse_policy", "bootstrap"):
        lines += [summary["protocol"][key], ""]
    lines += ["Oracle optimizes MAE, not squared error: its scalar is the weighted median of ΔL/x with weights |x|; "
              "a midpoint resolves median ties. It replaces the signed amplitude of the frozen shape (equivalently rescales a nonzero K0 curve). "
              "A zero K0 amplitude does not disable the shape's diagnostic amplitude. No target gamma, eta, offset or curvature is fitted.", "",
              "## Primary remainder scores", "",
              "All values are nats with paired-model 95% CIs. Gains are zero-change MAE minus method MAE; positive helps. "
              "Each prospective row is separate (n=1, degenerate CI). Development rows use saved model-held-out fits.", ""]
    def entries():
        for arm in ARMS:
            result = summary["results"][arm]
            for cap in CAPS:
                yield arm, "dev LOMO", cap, result["development_lomo"][cap]
            for model in result["prospective_models"]:
                for cap in CAPS:
                    yield arm, "prospective "+model, cap, result["per_target"][model][cap]
    lines += ["| Arm / cohort | Capability | Models / cells | Zero MAE | K0 MAE [95% CI] | K1 MAE [95% CI] | Oracle MAE [95% CI] |",
              "|---|---|---:|---:|---|---|---|"]
    for arm, cohort, cap, result in entries():
        r = result["remainder"]; m = r["metrics"]
        lines.append(f"| {arm} / {cohort} | {cap} | {r['n_models']} / {r['n_cells']} | {ci(m['zero'])} | {ci(m['K0'])} | {ci(m['K1'])} | {ci(m['Oracle'])} |")
    lines += ["", "| Arm / cohort | Capability | K0 gain [95% CI] | K1 gain [95% CI] | Oracle gain [95% CI], diagnostic |",
              "|---|---|---|---|---|"]
    for arm, cohort, cap, result in entries():
        m = result["remainder"]["metrics"]
        lines.append(f"| {arm} / {cohort} | {cap} | {ci(m['K0'], 'improvement_over_zero')} | {ci(m['K1'], 'improvement_over_zero')} | {ci(m['Oracle'], 'improvement_over_zero')} |")
    lines += ["", "## Failure signatures and full-curve diagnostic bound", "", summary["protocol"]["diagnosis_rule"], "",
              "The bound below fits AND scores all declared target configurations, including calibration; it is the true minimum error for this scalar form. "
              "CI support is descriptive for this panel. In a one-model row it cannot establish population evidence. "
              "Mapping gaps may remain even when the form also fails; the three flags are not an exhaustive causal decomposition.", "",
              "| Arm / cohort | Capability | Full-curve K0 MAE | Full-curve Oracle MAE [95% CI] | Remainder K1−K0 [95% CI] | Failure signatures |",
              "|---|---|---:|---|---|---|"]
    for arm, cohort, cap, result in entries():
        m = result["all_target_diagnostic"]["metrics"]
        contrast = result["remainder"]["contrasts"]["K1_minus_K0"]
        lines.append(f"| {arm} / {cohort} | {cap} | {m['K0']['mae']:.5f} | {ci(m['Oracle'])} | {audit.with_ci(contrast['value'], contrast['ci95'])} | {diagnosis_text(result['diagnosis'])} |")
    lines += ["", "## Quantization regions and scope", "", summary["protocol"]["quant_shape_status"], "",
              "4-bit is the K1 calibration point, so the remaining measurable region is 5-bit only. "
              "Qwen3-14B has no measured 5-bit point: its measurable remainder is unavailable, not zero error. "
              "Regional scores use the SAME all-bit Oracle scalar; they are not independently refitted region oracles.", "",
              "| Cohort / region | Capability | Models / cells | Zero MAE | K0 MAE [95% CI] | K1 MAE [95% CI] | Oracle MAE [95% CI] |",
              "|---|---|---:|---:|---|---|---|"]
    for arm, cohort, cap, result in entries():
        if arm != "quantization":
            continue
        for region, r in result["regions"].items():
            if not r["n_cells"]:
                lines.append(f"| {cohort} / {region} | {cap} | 0 / 0 | unavailable | unavailable | unavailable | unavailable |")
            else:
                m = r["metrics"]
                lines.append(f"| {cohort} / {region} | {cap} | {r['n_models']} / {r['n_cells']} | {m['zero']['mae']:.5f} | {ci(m['K0'])} | {ci(m['K1'])} | {ci(m['Oracle'])} |")
    lines += ["", "## Pruning range sensitivity", "", summary["protocol"]["pruning_extended"], "",
              "| Capability | Models / remainder cells | K0 MAE [95% CI] | K1 MAE [95% CI] | Oracle MAE [95% CI] | All-target failure signatures |",
              "|---|---:|---|---|---|---|"]
    for cap, result in summary["results"]["pruning"]["extended_all_available"].items():
        r = result["remainder"]; m = r["metrics"]
        lines.append(f"| {cap} | {r['n_models']} / {r['n_cells']} | {ci(m['K0'])} | {ci(m['K1'])} | {ci(m['Oracle'])} | {diagnosis_text(result['diagnosis'])} |")
    lines += ["", "## Every held-out development target", "",
              "Per-target n=1 model-bootstrap CIs are degenerate: [MAE, MAE]. The compact values below therefore also specify those CIs. "
              "Prospective targets appear separately above. Gains and complete paired CIs for each target are saved in summary.json.", "",
              "| Arm / target | Capability | Remainder cells | Zero MAE | K0 MAE | K1 MAE | Oracle MAE |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        result = summary["results"][arm]
        for model, caps in result["per_target"].items():
            if model in result["prospective_models"]:
                continue
            for cap, values in caps.items():
                r = values["remainder"]; m = r["metrics"]
                lines.append(f"| {arm} / {model} | {cap} | {r['n_cells']} | " + " | ".join(f"{m[b]['mae']:.5f}" for b in BUDGETS) + " |")
    lines += ["", "## Retention, provenance and limitations", ""]
    primary = [t for t in summary["targets"] if t["scope"] == "primary"]
    for arm in ARMS:
        targets = [t for t in primary if t["arm"] == arm]
        records = [r for t in targets for r in t["records"] if not r["is_calibration"]]
        lines += [f"- {arm}: retained {sum(r['near_zero_response'] for r in records)} near-zero, "
                  f"{sum(r['negative_response'] for r in records)} negative and {sum(r['collapse'] for r in records)} collapse-labelled score cells; "
                  f"{sum(t['calibration']['near_zero_response'] for t in targets)} near-zero capability calibration responses; "
                  f"{sum(t['calibration']['fallback'] is not None for t in targets)} numerical fallbacks."]
    lines += ["", "One declared recipe exclusion: Gemma3-4B/D600 used full training, outside V28's LoRA ladder (all three capabilities excluded). "
              "Only seed0 final distillation runs enter: no seed averaging, trajectory snapshots, or matched exposure controls. "
              "Qwen14 pruning aliases are verified identical and counted once. Raw metadata caveats, including the Qwen0.6B 5-bit dense protocol discrepancy, are retained in the inventory.", ""]
    for key in ("quant_size_convention", "distill_mapping", "prospective_scope"):
        lines += [summary["protocol"][key], ""]
    lines += [f"Qwen14 pruning N0={summary['qwen14_basic_metadata']['N0']:,} non-embedding 2-D parameters is computed from the embedded architecture transcription; "
              "see `qwen14_basic_metadata` for the exact formula and cached config revision. This is basic architecture information, not a fit to compression outcomes.", "",
              "Reproduce: `python analysis/v35_info_budget.py --n-boot 10000`. Validate without writing: add `--dry-run`. "
              "Tests: `python -m pytest tests/test_v35.py`. "
              "[Machine-readable summary](../../results/v35-info-budget/summary.json) contains per-cell predictions, calibration audit, "
              "all-target scalar fits, lower-bound caveats, per-target/region gains and CIs, exclusions and SHA256 input/code provenance. "
              "[Implementation](../../analysis/v35_info_budget.py) imports `prediction_audit.py`, V28 prediction functions and V30 prediction functions. "
              "Development fits come from V28/V30 JSON; prospective quantization uses V30b's already-saved full-development map.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.n_boot < 1:
        parser.error("--n-boot must be positive")
    summary = build_summary(args.n_boot)
    report = render(summary)
    if args.dry_run:
        print(report)
    else:
        audit.write_outputs(summary, report, OUT)
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(report)
        print(f"Wrote {OUT / 'summary.json'} and {REPORT}")
    return summary


if __name__ == "__main__":
    main()
