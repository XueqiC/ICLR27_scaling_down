#!/usr/bin/env python3
"""Freeze new-source coefficient predictions using saved development JSON, CPU only.

Default: freeze conditional functions, with unknown dense losses left null.
--dry-run computes the same fits and prints the tables without writing.
--apply-frozen FILE binds later dense/one-point inputs without reading dev data
or refitting anything. The original freeze is never overwritten.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np
from scipy.optimize import minimize_scalar

try:
    from . import prediction_audit as audit
    from . import v25_distill_delta as v25
except ImportError:
    import prediction_audit as audit
    import v25_distill_delta as v25

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v28-new-source-pred/frozen_predictions.json"
REPORT = ROOT / "paper/docs/NEW_SOURCE_PREDICTION.md"
METADATA = ROOT / "configs/v28_source_metadata.json"
CAPABILITIES = ("math", "code", "qa")
PRUNE_DEV = (0.9, 0.8, 0.7, 0.6)
PRUNE_TEST = (0.8, 0.7, 0.6)
QUANT_DEV = (8, 6, 4)
QUANT_TEST = (5, 4)
CALIBRATION = {"pruning": 0.9, "quantization": 6, "distillation": 150}
PROTOCOL_ID = "v6-v10-v12-reference-completion-native-token-odd64-seed0"
MODES = {
    "A": {
        "name": "NO-compression-calibration",
        "definition": "Target supplies only basic metadata and a pre-declared dense capability measurement. No target compressed results enter prediction. Tests basic-parameter prediction.",
        "target_compressed_points_per_arm": 0,
        "inputs": ["N0 non-embedding language matrix parameters", "family", "dense L_c", "declared method coordinate/recipe"],
    },
    "B": {
        "name": "FEW-compression-calibration",
        "definition": "Mode A information plus exactly ONE pre-specified compressed configuration of that target per arm (three capability losses from that configuration). Tests the value of one calibration point; it is not zero-compression prediction.",
        "target_compressed_points_per_arm": 1,
        "inputs": ["Mode A inputs", "one target compressed L_c at the frozen calibration coordinate"],
        "calibration_coordinates": CALIBRATION,
    },
}


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Expected a finite number")
    return value


def canonical(value):
    return value.lower().replace("--", "/").split("/")[-1]


def q_shape(bit):
    return 4.0 ** -bit - 4.0 ** -16


def shape(arm, coordinate, gamma=None):
    if arm == "pruning":
        return ((1.0 - coordinate) / 0.3) ** gamma
    if arm == "quantization":
        return 1024.0 * q_shape(coordinate)
    raise ValueError(f"Unknown arm: {arm}")


def basic_input(row):
    """An explicit allowlist: outcomes and fitted own-model labels cannot pass."""
    return {key: row[key] for key in ("model", "family", "N0", "dense_loss")}


def fit_mapping(rows, labels):
    """Signed coefficient ridge; one row per model, train-only preprocessing.

    Fixed lambda=1, unpenalized intercept; unseen families have zero centered
    family correction. No hyperparameter selection on LOMO errors.
    """
    if len(rows) < 2 or len({r["model"] for r in rows}) != len(rows):
        raise ValueError("Mapping needs distinct development models")
    x = np.array([[math.log(finite(r["N0"]) / 1e9), finite(r["dense_loss"])] for r in rows])
    y = np.asarray(labels, dtype=float)
    if y.shape != (len(rows),) or not np.isfinite(y).all():
        raise ValueError("One finite coefficient label is required per model")
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-12)
    families = sorted({r["family"] for r in rows})
    effects = np.array([[float(r["family"] == f) for f in families] for r in rows])
    frequencies = effects.mean(axis=0)
    design = np.column_stack([np.ones(len(rows)), (x-center)/scale, effects-frequencies])
    penalty = np.eye(design.shape[1]); penalty[0, 0] = 0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {"train_models": [r["model"] for r in rows], "center": center.tolist(),
            "scale": scale.tolist(), "families": families, "family_frequencies": frequencies.tolist(),
            "coefficients": beta.tolist(), "ridge_lambda": 1.0,
            "feature_order": ["intercept", "standardized log(N0/1e9)", "standardized dense L_c", *families]}


def mapping_formula(fit, target):
    """Return exact affine coefficients in the as-yet-unknown dense L_c."""
    if target["model"] in fit["train_models"]:
        raise ValueError("Held-out model leaked into mapping training")
    beta, center, scale = (np.asarray(fit[k]) for k in ("coefficients", "center", "scale"))
    effects = np.zeros(len(fit["families"]))
    if target["family"] in fit["families"]:
        effects = np.array([float(target["family"] == f) for f in fit["families"]])
        effects -= fit["family_frequencies"]
    offset = beta[0] + beta[1]*(math.log(finite(target["N0"])/1e9)-center[0])/scale[0]
    offset += float(beta[3:] @ effects) - beta[2]*center[1]/scale[1]
    return {"intercept": float(offset), "dense_slope": float(beta[2]/scale[1]), "calibration_slope": 0.0}


def predict_mapping(fit, target):
    form = mapping_formula(fit, target)
    return form["intercept"] + form["dense_slope"] * finite(target["dense_loss"])


def fit_arm(rows, arm):
    """Fit shape + per-dev-model labels, then map labels from basic inputs.

    Pruning shares gamma within a capability and maps a=A*0.3**gamma.
    Shared gamma is re-estimated in EACH outer fold, never on target curves.
    q is signed and rescaled by 1024 solely for numerical conditioning.
    """
    if len({r["capability"] for r in rows}) != 1:
        raise ValueError("Fit capabilities separately")
    coordinates = PRUNE_DEV if arm == "pruning" else QUANT_DEV
    y = np.array([[r["deltas"][str(c)] for c in coordinates] for r in rows])
    def profile(gamma):
        x = np.array([shape(arm, c, gamma) for c in coordinates])
        labels = y @ x / (x @ x)
        return labels, float(np.sum((y - labels[:, None]*x)**2))
    gamma = None
    if arm == "pruning":
        # Coarse scan plus local bounded refinement, with endpoints considered.
        grid = np.linspace(.05, 8.0, 161)
        i = int(np.argmin([profile(g)[1] for g in grid]))
        opt = minimize_scalar(lambda g: profile(g)[1], bounds=(grid[max(i-1, 0)], grid[min(i+1, 160)]), method="bounded")
        gamma = float(min([.05, 8.0, opt.x], key=lambda g: profile(g)[1]))
    labels, sse = profile(gamma)
    return {"arm": arm, "gamma": gamma, "shape_fit_sse": sse,
            "gamma_bounds": [.05, 8.0] if arm == "pruning" else None,
            "mapping": fit_mapping([basic_input(r) for r in rows], labels),
            "development_coefficient_labels": dict(zip([r["model"] for r in rows], labels.tolist())),
            "mean_coefficient": float(labels.mean()),
            "coefficient_definition": "a=A*0.3^gamma" if arm == "pruning" else "q/1024"}


def predict_arm(fit, target, coordinates, mode="A", calibration=None):
    """Prediction receives dense-only target plus a separate one-point object."""
    if mode not in MODES:
        raise ValueError("Unknown evaluation mode")
    if mode == "A" and calibration is not None:
        raise ValueError("Mode A forbids compressed calibration")
    coefficient = predict_mapping(fit["mapping"], target)
    if mode == "B":
        if calibration is None or set(calibration) != {"coordinate", "loss"}:
            raise ValueError("Mode B requires exactly one calibration point")
        c = CALIBRATION[fit["arm"]]
        if calibration["coordinate"] != c or c in coordinates:
            raise ValueError("Calibration must use the pre-specified disjoint coordinate")
        coefficient = (finite(calibration["loss"]) - target["dense_loss"]) / shape(fit["arm"], c, fit["gamma"])
    return [float(coefficient * shape(fit["arm"], c, fit["gamma"])) for c in coordinates]


def lomo_arm(rows, arm):
    coordinates = PRUNE_TEST if arm == "pruning" else QUANT_TEST
    records, folds = [], []
    for held_out in sorted({r["model"] for r in rows}):
        train = [r for r in rows if r["model"] != held_out]
        test, = [r for r in rows if r["model"] == held_out]
        fit = fit_arm(train, arm)
        target = basic_input(test)
        point = {"coordinate": CALIBRATION[arm], "loss": test["dense_loss"] + test["deltas"][str(CALIBRATION[arm])]}
        predictions = {"A": predict_arm(fit, target, coordinates),
                       "B": predict_arm(fit, target, coordinates, "B", point)}
        folds.append({"held_out": held_out, "fit": fit,
                      "mode_A_target_inputs": target, "mode_B_extra_input": point})
        for i, c in enumerate(coordinates):
            records.append({"model": held_out, "coordinate": c, "observed_delta": test["deltas"][str(c)],
                            "A": predictions["A"][i], "B": predictions["B"][i], "zero": 0.0,
                            "dev_mean_coefficient": fit["mean_coefficient"]*shape(arm, c, fit["gamma"])})
    return {"folds": folds, "records": records, **summarize_oof(records)}


def summarize_oof(records):
    metrics = {mode: {"mae": float(np.mean([abs(r[mode]-r["observed_delta"]) for r in records])),
                      "n_cells": len(records), "n_models": len({r["model"] for r in records})}
               for mode in ("A", "B", "zero", "dev_mean_coefficient")}
    paired = [abs(r["A"]-r["observed_delta"])-abs(r["B"]-r["observed_delta"]) for r in records]
    draws = audit.bootstrap_means(paired, [r["model"] for r in records], n_boot=2000, seed=280906)
    residuals = {mode: {} for mode in ("A", "B")}
    for mode in residuals:
        for c in sorted({r["coordinate"] for r in records}):
            errors = [r["observed_delta"]-r[mode] for r in records if r["coordinate"] == c]
            residuals[mode][str(c)] = {"offsets": np.quantile(errors, [.025, .975]).tolist(),
                                      "n_models": len(errors)}
    return {"metrics": metrics, "A_mae_minus_B_mae": float(np.mean(paired)),
            "A_mae_minus_B_mae_ci95": audit.interval(draws[:, 0]),
            "interval_residuals": residuals}


def load_development(metadata):
    """Fixed whitelist: ignore added runs, archive copies, infills and cliffs."""
    panels = {arm: {c: [] for c in CAPABILITIES} for arm in ("pruning", "quantization")}
    paths = [METADATA]
    for model, info in sorted(metadata["models"].items()):
        meta_path = ROOT / "results/v6-capability-geometry" / model / "fisher_meta.json"
        meta = audit.read_json(meta_path); paths.append(meta_path)
        excluded = [n for n in meta["param_names"] if n.endswith("embed_tokens.weight") or n == "lm_head.weight"]
        n0 = meta["n_params"] - len(excluded)*info["hidden_size"]*info["vocab_size"]
        if sorted(excluded) != sorted(info["excluded_matrices"]) or n0 != info["N0"] or n0 <= 0:
            raise ValueError(f"Non-embedding parameter provenance mismatch: {model}")
        if meta["model"] != info["hf_id"]:
            raise ValueError(f"Checkpoint identity mismatch: {model}")
        for arm, subdir, tag, name, anchor, coords in (
            ("pruning", "v6-capability-geometry", model, "prune_losses.json", "1.0", PRUNE_DEV),
            ("quantization", "v10-quantization", "Qwen--"+model if model.startswith("Qwen3") else model,
             "quant_losses.json", "dense", (*QUANT_DEV, 5)),
        ):
            path = ROOT / "results" / subdir / tag / name
            losses = audit.read_json(path); paths.append(path)
            for cap in CAPABILITIES:
                dense = finite(losses[anchor][cap])
                panels[arm][cap].append({"model": model, "family": info["family"], "N0": n0,
                                        "capability": cap, "dense_loss": dense,
                                        "deltas": {str(c): finite(losses[str(c)][cap])-dense for c in coords},
                                        "source_path": str(path.relative_to(ROOT)),
                                        "measurement_caveat": losses.get("_5bit_meta")})
    return panels, paths


def fit_transfer(rows, cap):
    rs = [r for r in rows if r["capability"] == cap]
    y = np.array([r["observed"] for r in rs])
    if cap == "code":
        u = np.log1p(np.array([r["D"] for r in rs])/150.0)
        coefficient = float(u @ y / (u @ u))
        form = "beta_c*log(1+D/150)"
    else:
        coefficient = float(y.mean()); form = "mu_c for D>0; 0 at D=0"
    return {"coefficient": coefficient, "form": form, "capability": cap,
            "train_row_ids": [r["row_id"] for r in rs]}


def transfer_shape(cap, budget):
    if budget < 0:
        raise ValueError("Distillation budget must be nonnegative")
    return math.log1p(budget/150.0) if cap == "code" else float(budget > 0)


def distillation_development():
    rows, paths = v25.load_rows()
    excluded = [r["row_id"] for r in rows if r["training_mode"] != "lora"]
    rows = [r for r in rows if r["training_mode"] == "lora"]
    training = {"epochs": 2, "seed": 0, "learning_rate": 1e-4, "optimizer": "AdamW",
                "scheduler": "cosine", "warmup_ratio": .03, "effective_batch_size_sequences": 16,
                "max_len": 1024, "dtype": "bfloat16"}
    lora = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.0, "bias": "none"}
    for source in sorted({r["source_path"] for r in rows}):
        directory = (ROOT/source).parent
        log_path, adapter_path = directory/"train_log.json", directory/"adapter/adapter_config.json"
        log, adapter = audit.read_json(log_path), audit.read_json(adapter_path)
        if any(log[k] != value for k, value in training.items()) or any(adapter[k] != value for k, value in lora.items()):
            raise ValueError(f"Distillation training recipe differs: {source}")
        paths.extend([log_path, adapter_path])
    results = {}
    for cap in CAPABILITIES:
        rs = [r for r in rows if r["capability"] == cap]
        records, folds = [], []
        for model in sorted({r["model"] for r in rs}):
            train = [r for r in rs if r["model"] != model]
            test = [r for r in rs if r["model"] == model]
            fit = fit_transfer(train, cap)
            cal, = [r for r in test if r["D"] == CALIBRATION["distillation"]]
            b = cal["observed"]/transfer_shape(cap, cal["D"])
            folds.append({"held_out": model, "fit": fit, "mode_B_calibration_row_id": cal["row_id"]})
            for r in test:
                if r["D"] == CALIBRATION["distillation"]:
                    continue
                records.append({"model": model, "coordinate": r["D"], "observed_delta": r["observed"],
                                "A": fit["coefficient"]*transfer_shape(cap, r["D"]),
                                "B": b*transfer_shape(cap, r["D"]), "zero": 0.0,
                                "dev_mean_coefficient": float(np.mean([t["observed"] for t in train]))})
        results[cap] = {"fit": fit_transfer(rs, cap), "lomo": {"folds": folds, "records": records, **summarize_oof(records)}}
    return {"by_capability": results, "excluded_non_lora_row_ids": excluded,
            "recipe": {"teacher": "gpt-5.6-luna", "recipe": "full", "training_mode": "lora"},
            "frozen_training_hyperparameters": training, "frozen_lora_hyperparameters": lora,
            "scope": "Three Gemma3 student sizes only; full-training 4B/D600 excluded by recipe, not outcome. Math/QA constants and code log-data form chosen from existing V25 evidence; LOMO is retrospective development evidence, not fresh model selection validation."}, paths


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(payload):
    return hashlib.sha256(json.dumps({k: v for k, v in payload.items() if k != "freeze_sha256"},
                                     sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_new_source(source, metadata):
    identities = {canonical(v) for tag, m in metadata["models"].items() for v in (tag, m["hf_id"])}
    if any(canonical(source[k]) in identities for k in ("model", "hf_id")):
        raise ValueError("New source overlaps a development checkpoint (including aliases)")
    if finite(source["N0"]) <= 0:
        raise ValueError("Source N0 must be positive")
    if source.get("pretraining_tokens") is not None and finite(source["pretraining_tokens"]) <= 0:
        raise ValueError("Disclosed pretraining tokens must be positive")


def freeze(source, student):
    metadata = audit.read_json(METADATA)
    validate_new_source(source, metadata)
    if student["model"] in metadata["models"] and student["hf_id"] != metadata["models"][student["model"]]["hf_id"]:
        raise ValueError("Student tag/HF identity mismatch")
    if student["budget"] in (0, CALIBRATION["distillation"]) or student["budget"] < 0:
        raise ValueError("Untested student budget must be positive and disjoint from calibration D150")
    existing = ROOT / "results/v12-distill" / student["model"] / f"gpt-5.6-luna_full_{student['budget']}" / "eval.json"
    if existing.exists():
        raise ValueError(f"Student configuration is already tested: {existing.relative_to(ROOT)}")
    panels, paths = load_development(metadata)
    distill, distill_paths = distillation_development(); paths += distill_paths
    methods = {}
    for arm in ("pruning", "quantization"):
        methods[arm] = {"by_capability": {cap: {"fit": fit_arm(panels[arm][cap], arm),
                                              "lomo": lomo_arm(panels[arm][cap], arm)} for cap in CAPABILITIES}}
    methods["distillation"] = distill
    code = [Path(__file__), ROOT/"analysis/prediction_audit.py", ROOT/"analysis/v25_distill_delta.py",
            ROOT/"analysis/v6_capability_geometry.py", ROOT/"analysis/v10_quantization.py", ROOT/"analysis/v12_distill.py"]
    payload = {
        "version": 28, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "code_sha256": audit.provenance(code), "input_sha256": audit.provenance(paths),
        "provenance_note": "Commit identifies the base checkout; code and data SHA-256 identify the actual dirty/untracked working files. Timestamp is a local freeze record, not external preregistration certification.",
        "status": "coefficients_and_conditional_functions_frozen; target_measurements_pending",
        "target_form": "Lhat_c^(m) = L_c,ref + F_m,c(x_dense, u_shared, u_m)",
        "reference": {"pruning": "SOURCE own dense", "quantization": "SOURCE own dense", "distillation": "STUDENT own dense S0"},
        "modes": copy.deepcopy(MODES), "source": source, "student": student,
        "transport_scope": {"source": "New checkpoint; Qwen3 8B extends the development Qwen3 size ladder (0.6/1.7/4B). Family and native-token loss may confound transfer.",
                            "student": "Default Gemma3 12B/D300 is an untested configuration and extrapolates beyond the three development student sizes; D300 is within the dev budget grid. Teacher, recipe and LoRA mode held fixed. No independent student-size coefficient is claimed."},
        "metadata": metadata,
        "measurement": {"protocol_id": PROTOCOL_ID, "benchmark": {"math": "MATH-500", "code": "MBPP", "qa": "2WikiMultihopQA"},
                        "probe_seed": 0, "n_probe_requested": 128, "half": "odd indices", "items_per_capability": 64,
                        "unit": "completion-token CE nats, sum NLL / sum target tokens; native tokenizer; v6 completion_loss",
                        "cost": "One dense forward-only measurement panel, 64 items per capability; no gradients/Fisher/geometry inputs.",
                        "limitations": "Historical native-token endpoints are not tokenizer-invariant per-byte losses. V6/V10 own dense anchors differ slightly; they are preserved arm by arm. Raw archived aggregates do not certify exact checkpoint revisions or text hashes; later dense/calibration inputs must declare their revision and this protocol."},
        "protocol": {"pruning_dev_densities": list(PRUNE_DEV), "pruning_test_densities": list(PRUNE_TEST),
                     "quantization_dev_bits": list(QUANT_DEV), "quantization_test_bits": list(QUANT_TEST),
                     "pruning_recipe": "V6 global sampled-threshold magnitude pruning, language matrices including embeddings/head",
                     "quantization_recipe": "V10 symmetric per-output-channel fake weight quantization, language matrices including embeddings/head",
                     "outcome_filter": "none; fixed coordinates, signed deltas; no cliff/sign filtering",
                     "dev_split": "leave ONE entire model out, separately per capability; same score cells for A and B; B calibration cells excluded from scores",
                     "interval": "central 95% empirical signed LOMO residual offsets, separately per arm/capability/mode/coordinate. Descriptive prediction intervals; 12 source models or 3 student sizes, no finite-sample coverage guarantee, no item/seed/dense-measurement uncertainty. Distillation unseen budgets use all LOMO budget residuals with explicit extrapolation flag.",
                     "gain_interval": "2000 paired model-bootstrap draws of fixed OOF errors, seed 280906; positive A MAE minus B MAE means one point helped"},
        "coefficient_cost": {
            "learnable_on_dev": ["per-capability shared pruning gamma and signed dev amplitudes", "ridge mappings from log N0, family and dense L_c to pruning a and quantization q/1024", "mapping standardizers and family effects", "distillation math/QA mean response and code log-data coefficient", "mode/coordinate-specific LOMO residual offsets"],
            "target_own_compressed_results_required": {"A": [], "B": ["one signed pruning amplitude (gamma stays dev-frozen)", "one signed quantization q", "one distillation response scale"]},
            "not_identifiable_from_one_point": ["target pruning gamma independently of amplitude", "new quantization exponent/cliff", "target distillation size/data curvature"],
            "candidate_only": {"D0_pretraining_tokens": "NOT added: sparse commercial size ladders cannot separate N from D0 effects. Disclosed tokens/stage are recorded as metadata, not fitted covariates; no identifiable independent token/stage effect is established.",
                               "training_stage": "metadata only; unknown development stages are not imputed"}},
        "development_rows": panels, "methods": methods,
    }
    payload["predictions"] = prediction_rows(payload)
    payload["freeze_sha256"] = seal(payload)
    return payload


def read_measurements(path, frozen, kind):
    if path is None:
        return {}
    data = audit.read_json(path)
    allowed = {"source", "student"} if kind == "dense" else set(CALIBRATION)
    if not isinstance(data, dict) or set(data)-allowed-{"protocol_id"}:
        raise ValueError(f"Unexpected {kind} input keys; compressed outcomes cannot enter dense inputs")
    if data.get("protocol_id") != frozen["measurement"]["protocol_id"]:
        raise ValueError("Measurement protocol does not match the freeze")
    for key, value in data.items():
        if key == "protocol_id":
            continue
        expected_keys = {"model", "hf_id", "revision", "losses"} | ({"coordinate"} if kind != "dense" else set())
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise ValueError(f"{kind}/{key} requires exactly {sorted(expected_keys)}")
        who = key if kind == "dense" else ("student" if key == "distillation" else "source")
        target = frozen[who]
        if any(value[k] != target[k] for k in ("model", "hf_id")):
            raise ValueError(f"Measurement identity differs from frozen {who}")
        if not isinstance(value["revision"], str) or not value["revision"].strip():
            raise ValueError("An explicit checkpoint revision is required")
        if target.get("revision") and value["revision"] != target["revision"]:
            raise ValueError("Checkpoint revision differs from frozen metadata")
        if set(value["losses"]) != set(CAPABILITIES) or any(finite(v) < 0 for v in value["losses"].values()):
            raise ValueError("Need finite nonnegative losses for all three capabilities")
        if kind != "dense" and value["coordinate"] != frozen["modes"]["B"]["calibration_coordinates"][key]:
            raise ValueError("Exactly one pre-specified calibration coordinate is permitted")
    return {k: v for k, v in data.items() if k != "protocol_id"}


def residual_interval(result, mode, coordinate):
    intervals = result["lomo"]["interval_residuals"][mode]
    if str(coordinate) in intervals:
        return {**intervals[str(coordinate)], "budget_extrapolation": False}
    errors = [r["observed_delta"]-r[mode] for r in result["lomo"]["records"]]
    return {"offsets": np.quantile(errors, [.025, .975]).tolist(),
            "n_models": len({r["model"] for r in result["lomo"]["records"]}), "budget_extrapolation": True}


def prediction_rows(frozen, dense=None, calibration=None):
    rows = []
    for arm, method in frozen["methods"].items():
        who = "student" if arm == "distillation" else "source"
        target = frozen[who]
        coordinates = (target["budget"],) if arm == "distillation" else frozen["protocol"][arm+"_test_"+("densities" if arm == "pruning" else "bits")]
        for cap, result in method["by_capability"].items():
            fit = result["fit"]
            for mode in ("A", "B"):
                for c in coordinates:
                    if arm == "distillation":
                        factor = transfer_shape(cap, c)
                        form = {"intercept": fit["coefficient"]*factor, "dense_slope": 0.0, "calibration_slope": 0.0}
                        cal_factor = transfer_shape(cap, frozen["modes"]["B"]["calibration_coordinates"][arm])
                    else:
                        factor = shape(arm, c, fit["gamma"])
                        form = {k: v*factor for k, v in mapping_formula(fit["mapping"], target).items()}
                        cal_factor = shape(arm, frozen["modes"]["B"]["calibration_coordinates"][arm], fit["gamma"])
                    if mode == "B":
                        form = {"intercept": 0.0, "dense_slope": -factor/cal_factor, "calibration_slope": factor/cal_factor}
                    rows.append({"arm": arm, "mode": mode, "capability": cap, "coordinate": c,
                                 "reference": frozen["reference"][arm], "delta_formula": form,
                                 "formula_definition": "delta = intercept + dense_slope*L_dense + calibration_slope*L_calibration",
                                 "interval_recipe": residual_interval(result, mode, c)})
    return evaluate_frozen_rows(rows, dense or {}, calibration or {})


def evaluate_frozen_rows(frozen_rows, dense, calibration):
    """Bind ONLY the serialized affine functions and interval offsets.

    Later changes to fitting code, shapes or constants cannot change the
    predictions in a previously frozen table.
    """
    rows = copy.deepcopy(frozen_rows)
    for row in rows:
        arm, mode, cap = (row[k] for k in ("arm", "mode", "capability"))
        who = "student" if arm == "distillation" else "source"
        form = row["delta_formula"]
        dense_loss = dense.get(who, {}).get("losses", {}).get(cap)
        point = calibration.get(arm) if mode == "B" else None
        cal_loss = point["losses"][cap] if point else None
        if point and who in dense and point["revision"] != dense[who]["revision"]:
            raise ValueError("Dense and compressed checkpoint revisions differ")
        missing = []
        if dense_loss is None and (form["dense_slope"] != 0 or mode == "B"):
            missing.append(f"{who}.dense.{cap}")
        if mode == "B" and cal_loss is None:
            missing.append(f"{arm}.pre_specified_calibration.{cap}")
        delta = None if missing else form["intercept"] + form["dense_slope"]*(dense_loss or 0.0) + form["calibration_slope"]*(cal_loss or 0.0)
        delta_pi = None if delta is None else [delta+x for x in row["interval_recipe"]["offsets"]]
        total = None if delta is None or dense_loss is None else dense_loss+delta
        row.update(delta_L=delta, delta_interval95=delta_pi, L_predicted=total,
                   L_interval95=None if total is None else [dense_loss+x for x in delta_pi],
                   missing_delta_inputs=missing,
                   status="conditional_on_pending_inputs" if missing else ("delta_frozen_dense_anchor_pending" if total is None else "numeric_prediction"),
                   target_compressed_points_used=int(mode == "B" and point is not None))
    return rows


def bind_inputs(frozen, dense_path=None, calibration_path=None):
    if frozen.get("freeze_sha256") != seal(frozen):
        raise ValueError("Frozen artifact integrity check failed")
    dense = copy.deepcopy(frozen.get("measurement_inputs", {}).get("dense", {}))
    new_dense = read_measurements(dense_path, frozen, "dense")
    for who, measurement in new_dense.items():
        if who in dense and dense[who] != measurement:
            raise ValueError("Cannot change a dense measurement already recorded in the freeze")
        dense[who] = measurement
    calibration = read_measurements(calibration_path, frozen, "calibration")
    return {"version": 28, "parent_freeze_sha256": frozen["freeze_sha256"],
            "parent_created_at_utc": frozen["created_at_utc"], "bound_at_utc": datetime.now(timezone.utc).isoformat(),
            "modes": frozen["modes"], "source": frozen["source"], "student": frozen["student"],
            "measurement_inputs": {"dense": dense, "calibration": calibration},
            "measurement_input_sha256": {str(p): file_hash(p) for p in (dense_path, calibration_path) if p is not None},
            "predictions": evaluate_frozen_rows(frozen["predictions"], dense, calibration),
            "refit": False, "development_data_read": False}


def formula_text(row):
    f = row["delta_formula"]
    parts = [f"{f['intercept']:.6g}"] if f["intercept"] else []
    if f["dense_slope"]:
        parts.append(f"{f['dense_slope']:+.6g} L_dense")
    if f["calibration_slope"]:
        parts.append(f"{f['calibration_slope']:+.6g} L_cal")
    return " ".join(parts) or "0"


def render(payload):
    lines = ["# V28: frozen new-source prediction", "", "CPU-only; no model loading, GPU work or target compression evaluation.", ""]
    for name, mode in payload["modes"].items():
        lines += [f"**Mode {name}: {mode['name']}.** {mode['definition']}", "Inputs: " + "; ".join(mode["inputs"]) + ".", ""]
    lines += [f"Source: `{payload['source']['model']}`, HF `{payload['source']['hf_id']}`, declared stage `{payload['source']['training_stage']}`.",
              "Pruning/quantization reference = source dense. Distillation reference = student dense.",
              f"Student: `{payload['student']['model']}`, D={payload['student']['budget']} traces/domain, gpt-5.6-luna/full/LoRA.", ""]
    if payload["source"].get("identity_note"):
        lines += [payload["source"]["identity_note"], ""]
    lines += ["Unknown inputs stay symbolic/null. These are frozen conditional predictions, not observed target results.", "",
              "| Arm | Mode | Capability | Coordinate | Predicted delta (nats) / frozen function | 95% interval |",
              "|---|---|---|---:|---|---|"]
    for row in payload["predictions"]:
        delta = formula_text(row) if row["delta_L"] is None else f"{row['delta_L']:.6g}"
        interval = row["delta_interval95"]
        pi = f"[{interval[0]:.6g}, {interval[1]:.6g}]" if interval else "function + [" + ", ".join(f"{v:.6g}" for v in row["interval_recipe"]["offsets"]) + "]"
        lines.append(f"| {row['arm']} | {row['mode']} | {row['capability']} | {row['coordinate']} | {delta} | {pi} |")
    if "methods" in payload:
        lines += ["", "LOMO development evaluation; errors in nats. Each A/B pair scores identical cells. Positive gain means one calibration point helped.", "",
                  "| Arm | Capability | Models | A MAE | B MAE | Zero MAE | Dev mean MAE | A−B MAE (95% paired model bootstrap) |",
                  "|---|---|---:|---:|---:|---:|---:|---|"]
        for arm, method in payload["methods"].items():
            for cap, result in method["by_capability"].items():
                r = result["lomo"]; m = r["metrics"]
                ci = r["A_mae_minus_B_mae_ci95"]
                lines.append(f"| {arm} | {cap} | {m['A']['n_models']} | {m['A']['mae']:.5f} | {m['B']['mae']:.5f} | {m['zero']['mae']:.5f} | {m['dev_mean_coefficient']['mae']:.5f} | {r['A_mae_minus_B_mae']:.5f} [{ci[0]:.5f}, {ci[1]:.5f}] |")
        lines += ["", payload["protocol"]["interval"], "", payload["coefficient_cost"]["candidate_only"]["D0_pretraining_tokens"], ""]
    return "\n".join(lines)+"\n"


def write_new(path, payload):
    """Exclusive creation prevents accidental replacement of a frozen artifact."""
    text = json.dumps(payload, indent=2, allow_nan=False)+"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source", default="Qwen3-8B")
    parser.add_argument("--source-hf", help="default Qwen/Qwen3-8B for the default source")
    parser.add_argument("--source-stage", default="Base")
    parser.add_argument("--source-family", help="default qwen3 for the default source")
    parser.add_argument("--source-n0", type=float, help="non-embedding matrix parameter count, not nominal 8B")
    parser.add_argument("--source-revision")
    parser.add_argument("--pretraining-tokens", type=float, help="candidate metadata only; never enters regression")
    parser.add_argument("--student", default="gemma3-12b")
    parser.add_argument("--student-hf", help="defaults to the documented HF ID for known students")
    parser.add_argument("--student-budget", type=int, default=300)
    parser.add_argument("--student-revision")
    parser.add_argument("--dense-json", type=Path)
    parser.add_argument("--calibration-json", type=Path, help="allowed only with --apply-frozen")
    parser.add_argument("--apply-frozen", type=Path, help="bind measurements to frozen functions; no dev data or refitting")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.apply_frozen:
            frozen = audit.read_json(args.apply_frozen)
            payload = bind_inputs(frozen, args.dense_json, args.calibration_json)
            output = args.output or args.apply_frozen.with_name("bound_predictions.json")
        else:
            if args.calibration_json:
                raise ValueError("Target compression calibration is allowed only when applying an existing freeze")
            metadata = audit.read_json(METADATA)
            if args.source != "Qwen3-8B" and (not args.source_hf or not args.source_family):
                raise ValueError("Custom sources require --source-hf and --source-family")
            args.source_hf = args.source_hf or "Qwen/Qwen3-8B"
            args.source_family = args.source_family or "qwen3"
            student_meta = metadata["models"].get(args.student, {})
            args.student_hf = args.student_hf or student_meta.get("hf_id")
            if args.student_hf is None:
                raise ValueError("Unknown students require --student-hf")
            known = args.source_hf in ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B-Base")
            n0 = args.source_n0 if args.source_n0 is not None else (metadata["qwen3_8b"]["N0"] if known else None)
            if n0 is None:
                raise ValueError("Supply --source-n0 for an unknown architecture; no nominal-size imputation")
            source = {"model": args.source, "hf_id": args.source_hf, "training_stage": args.source_stage,
                      "family": args.source_family, "N0": finite(n0), "revision": args.source_revision,
                      "pretraining_tokens": None if args.pretraining_tokens is None else finite(args.pretraining_tokens),
                      "N0_provenance": "explicit CLI non-embedding matrix count" if args.source_n0 is not None else "configs/v28_source_metadata.json:qwen3_8b"}
            if args.source_hf == "Qwen/Qwen3-8B" and args.source_stage.lower() == "base":
                source["identity_note"] = ("Requested HF ID/stage conflict: [Qwen/Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B) declares Pretraining & Post-training and names Qwen/Qwen3-8B-Base as its base. Defaults are retained as requested, but this artifact does not establish a Base checkpoint identity. Use --source-hf Qwen/Qwen3-8B-Base for the Base experiment before binding/evaluating measurements.")
            student = {"model": args.student, "hf_id": args.student_hf, "budget": args.student_budget,
                       "revision": args.student_revision, "teacher": "gpt-5.6-luna", "recipe": "full", "training_mode": "lora",
                       "family": student_meta.get("family", "undisclosed"), "N0": student_meta.get("N0"),
                       "size_effect_used": False}
            payload = freeze(source, student)
            if args.dense_json:
                dense = read_measurements(args.dense_json, payload, "dense")
                payload["measurement_inputs"] = {"dense": dense}
                payload["input_sha256"][str(args.dense_json.resolve())] = file_hash(args.dense_json)
                payload["predictions"] = prediction_rows(payload, dense)
                payload["freeze_sha256"] = seal(payload)
            output = args.output or OUT
        report = render(payload)
        if args.dry_run:
            print("DRY RUN: no writes.\n"+report)
            return payload
        if not args.apply_frozen:
            for key in ("input_sha256", "code_sha256"):
                for path, digest in payload[key].items():
                    if file_hash(ROOT/path) != digest:
                        raise ValueError(f"Input changed during freeze: {path}")
        write_new(output, payload)
        print(report)
        print(f"Wrote {output}")
        return payload
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
