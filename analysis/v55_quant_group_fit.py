#!/usr/bin/env python3
"""V55: signed quantization response fits using only CPU, NumPy and stdlib.

    python -B analysis/v55_quant_group_fit.py develop
    python -B analysis/v55_quant_group_fit.py freeze
    python -B analysis/v55_quant_group_fit.py compare

Outputs are created exclusively; existing files are never overwritten. No test
response enters a fit, grid search, standardizer or frozen prediction. As in
v53, standardization pools training rows across capabilities, and the shared
ridge requirement applies to every candidate (including per-anchor regression
and intercepts). Thus the interpolation candidate's anchor OLS is regularized
by the requested common 1e-3 penalty. No response log, clipping or selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "results/v54-quant-group"
OUT = ROOT / "results/v55-quant-group"
JOINT_DENSE = ROOT / "results/v46-p1-newsource/dense.json"
CAPS = ("math", "code", "qa")
DEV_TAGS = tuple(f"pythia-{size}@step{step}"
                 for size in ("160m", "410m", "1.4b") for step in (16000, 143000))
DEV_CONFIGS = ("b3_g64", "b3_g256", "b5_g64", "b5_g256")
JOINT_TAG = "pythia-1b@step96000"
TEST_SETS = {
    "bit_test": {"states": list(DEV_TAGS), "configs": ["b4_g64", "b4_g256"]},
    "granularity_test": {"states": list(DEV_TAGS),
                         "configs": ["b3_g128", "b4_g128", "b5_g128"]},
    "joint_test": {"states": [JOINT_TAG], "configs": ["b3_g128", "b4_g128", "b5_g128"]},
}
CANDIDATES = ("separable", "low_order_2d", "same_input_interpolation")
BASELINES = ("mean", "median", "zero")
METHODS = CANDIDATES + BASELINES
N_PARAMS = dict(zip(METHODS, (6, 20, 16, 4, 4, 0)))
RIDGE = 1e-3
P_GRID = np.arange(5, 61, dtype=float) / 10
Q_GRID = np.arange(-20, 21, dtype=float) / 20
TOKENS_PER_STEP = 2097152
FEATURE_NAMES = ["1", "z(log N0)", "z(L0c)", "z(log D0)"]
TERM_NAMES = ("1", "u", "v", "u*v", "u^2")
# v36/v53 matrix counting: QKV, attention output, and two MLP matrices;
# excludes vectors, embeddings and LM head. No model/GPU imports are needed.
ARCHITECTURES = {
    "160m": {"hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 12},
    "410m": {"hidden_size": 1024, "intermediate_size": 4096, "num_hidden_layers": 24},
    "1b": {"hidden_size": 2048, "intermediate_size": 8192, "num_hidden_layers": 16},
    "1.4b": {"hidden_size": 2048, "intermediate_size": 8192, "num_hidden_layers": 24},
}
RULE = "all three candidates and baselines are reported on every test; no selection by test outcome"
RESPONSE = "signed dL_c = loss(config)_c - dense_c, in nats; no transform or clipping"
INTERPOLATION_RULE = (
    "Bilinear in (log2(qmax), log2(g)), qmax=2**(b-1)-1, between the four dev "
    "anchors. Weights are not clipped: extend linearly outside either axis."
)


def matrix_n0(size):
    a = ARCHITECTURES[size]
    h, m = a["hidden_size"], a["intermediate_size"]
    return a["num_hidden_layers"] * (4 * h * h + 2 * h * m)


def path_label(path):
    path = Path(path).absolute()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def read_json(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def require_new(path):
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def write_new(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(f"WROTE {path}", flush=True)


def checked_losses(value, context):
    if not isinstance(value, dict) or any(cap not in value for cap in CAPS):
        raise ValueError(f"{context}: expected losses for math, code and qa")
    result = {}
    for cap in CAPS:
        loss = value[cap]
        if (isinstance(loss, bool) or not isinstance(loss, (int, float))
                or not np.isfinite(loss) or loss < 0):
            raise ValueError(f"{context}: {cap} loss must be finite, numeric and nonnegative")
        result[cap] = float(loss)
    return result


def measurement_path(directory, tag):
    return Path(directory) / tag.replace("@", "--") / "quant_group_losses.json"


def state_input(tag, dense, path):
    size, step = tag.removeprefix("pythia-").split("@step")
    return {"tag": tag, "size": size, "step": int(step), "N0": matrix_n0(size),
            "D0": int(step) * TOKENS_PER_STEP, "L0": dense, "path": path_label(path)}


def raw_features(state, cap):
    return [float(np.log(state["N0"])), state["L0"][cap], float(np.log(state["D0"]))]


def coordinates(config):
    match = re.fullmatch(r"b([0-9]+)_g([0-9]+)", config)
    if match is None:
        raise ValueError(f"Invalid quantization config: {config!r}")
    bits, group = map(int, match.groups())
    if bits < 2 or group <= 0:
        raise ValueError(f"Need bits >= 2 and group > 0: {config}")
    return float(np.log2(2 ** (bits - 1) - 1)), float(np.log2(group / 128))


def load_dev(directory):
    """Whitelist the exact 24 dev cells, ignoring all other response keys."""
    states, rows, hashes, missing = [], [], {}, []
    for tag in DEV_TAGS:
        path = measurement_path(directory, tag)
        if not path.is_file():
            missing.append(f"{path}: missing file")
            continue
        table, sha = read_json(path)
        absent = [key for key in ("dense", *DEV_CONFIGS) if key not in table]
        if absent:
            missing.append(f"{path}: missing {', '.join(absent)}")
            continue
        dense = checked_losses(table["dense"], f"{path}: dense")
        state = state_input(tag, dense, path)
        states.append(state)
        hashes[path_label(path)] = sha
        for config in DEV_CONFIGS:
            losses = checked_losses(table[config], f"{path}: {config}")
            for cap in CAPS:
                rows.append({"state": tag, "config": config, "capability": cap,
                             "phi_raw": raw_features(state, cap),
                             "dL": losses[cap] - dense[cap]})
    if missing:
        raise ValueError("Development requires all 24 cells (six states x four configs):\n"
                         + "\n".join(missing))
    return states, rows, hashes


def standardization(rows):
    raw = np.asarray([r["phi_raw"] for r in rows])
    center, scale = raw.mean(axis=0), raw.std(axis=0, ddof=0)
    constant = np.flatnonzero(scale == 0).tolist()
    scale[scale == 0] = 1.0
    return {"center": center.tolist(), "scale": scale.tolist(), "constant_features": constant,
            "u_center": float(np.mean([coordinates(r["config"])[0] for r in rows]))}


def phi(raw, stats):
    raw = np.asarray(raw, dtype=float)
    return np.concatenate((np.ones(raw.shape[:-1] + (1,)),
                           (raw - stats["center"]) / stats["scale"]), axis=-1)


def ridge_fit(x, y):
    return np.linalg.solve(x.T @ x + RIDGE * np.eye(x.shape[1]), x.T @ y)


def low_order_terms(x, v, u_center):
    u, v = np.broadcast_arrays(np.asarray(x) - u_center, np.asarray(v))
    return np.stack((np.ones_like(u), u, v, u * v, u * u), axis=-1)


def interpolate(anchors, config):
    """Four corners, exact bilinear interpolation and unclipped extension."""
    x, v = coordinates(config)
    xlo, vlo = coordinates("b3_g64")
    xhi, vhi = coordinates("b5_g256")
    tx, tv = (x - xlo) / (xhi - xlo), (v - vlo) / (vhi - vlo)
    return float((1 - tx) * (1 - tv) * anchors["b3_g64"]
                 + (1 - tx) * tv * anchors["b3_g256"]
                 + tx * (1 - tv) * anchors["b5_g64"]
                 + tx * tv * anchors["b5_g256"])


def fit_capability(rows, cap, stats):
    rows = [r for r in rows if r["capability"] == cap]
    z = phi([r["phi_raw"] for r in rows], stats)
    y = np.asarray([r["dL"] for r in rows])
    configs = np.asarray([r["config"] for r in rows])
    x, v = np.asarray([coordinates(c) for c in configs]).T
    # Batched independent 4x4 ridge solves. p-major, then q-major grid order
    # makes exact-SSE ties choose the smaller p, then the smaller q.
    p, q = np.meshgrid(P_GRID, Q_GRID, indexing="ij")
    shapes = np.exp2(-p.ravel()[:, None] * x + q.ravel()[:, None] * v)
    designs = shapes[:, :, None] * z[None, :, :]
    xt = designs.transpose(0, 2, 1)
    rhs = (xt @ y)[..., None]
    betas = np.linalg.solve(xt @ designs + RIDGE * np.eye(4), rhs)[..., 0]
    residuals = (designs @ betas[..., None])[..., 0] - y
    sses = np.sum(residuals ** 2, axis=1)
    best = int(np.argmin(sses))
    terms = low_order_terms(x, v, stats["u_center"])
    design = (terms[:, :, None] * z[:, None, :]).reshape(len(rows), 20)
    coefficients = ridge_fit(design, y).reshape(5, 4)
    anchors, means, medians = {}, {}, {}
    for config in DEV_CONFIGS:
        mask = configs == config
        if not np.any(mask):
            raise ValueError(f"Missing training anchor {config} for {cap}")
        anchors[config] = ridge_fit(z[mask], y[mask]).tolist()
        means[config] = float(np.mean(y[mask]))
        medians[config] = float(np.median(y[mask]))
    return {
        "separable": {"beta": betas[best].tolist(), "p": float(p.ravel()[best]),
                      "q": float(q.ravel()[best]), "dev_sse": float(sses[best])},
        "low_order_2d": {"coefficients": coefficients.tolist(),
                         "design_rank": int(np.linalg.matrix_rank(design)),
                         "dev_sse": float(np.sum((design @ coefficients.ravel() - y) ** 2))},
        "same_input_interpolation": {"anchors": anchors},
        "mean": {"anchors": means}, "median": {"anchors": medians}, "zero": {"value": 0.0},
    }


def fit_all(rows):
    stats = standardization(rows)
    return stats, {cap: fit_capability(rows, cap, stats) for cap in CAPS}


def predict_all(models, z, config, stats):
    x, v = coordinates(config)
    sep = models["separable"]
    terms = low_order_terms(x, v, stats["u_center"])
    anchors = {key: float(np.dot(beta, z))
               for key, beta in models["same_input_interpolation"]["anchors"].items()}
    return {
        "separable": float(np.dot(sep["beta"], z) * np.exp2(-sep["p"] * x + sep["q"] * v)),
        "low_order_2d": float(terms @ np.asarray(models["low_order_2d"]["coefficients"]) @ z),
        "same_input_interpolation": interpolate(anchors, config),
        "mean": interpolate(models["mean"]["anchors"], config),
        "median": interpolate(models["median"]["anchors"], config),
        "zero": float(models["zero"]["value"]),
    }


def mae_table(rows):
    table = []
    for name in METHODS:
        errors = {cap: [abs(r["predictions"][name] - r["dL"]) for r in rows
                        if r["capability"] == cap] for cap in CAPS}
        maes = {cap: float(np.mean(e)) if e else None for cap, e in errors.items()}
        table.append({"candidate": name, "mae": maes,
                      "n": {cap: len(e) for cap, e in errors.items()}})
    return table


def print_table(table, title):
    print(f"\n{title}: MAE (nats)")
    print(f"{'candidate':<26} {'math':>10} {'code':>10} {'qa':>10}")
    for row in table:
        values = [row["mae"][cap] for cap in CAPS]
        print(f"{row['candidate']:<26} " + " ".join(
            f"{v:10.6f}" if v is not None else f"{'N/A':>10}" for v in values))


def loso(states, rows):
    predictions, folds = [], []
    for state in states:
        tag = state["tag"]
        train = [r for r in rows if r["state"] != tag]
        held_out = [r for r in rows if r["state"] == tag]
        stats, models = fit_all(train)
        fold_rows = []
        for row in held_out:
            estimates = predict_all(models[row["capability"]], phi(row["phi_raw"], stats),
                                    row["config"], stats)
            fold_rows.append({**row, "predictions": estimates})
        predictions.extend(fold_rows)
        folds.append({"held_out": tag, "train_states": [s["tag"] for s in states if s["tag"] != tag],
                      "n_train_cells": len(train) // len(CAPS),
                      "n_held_out_cells": len(held_out) // len(CAPS),
                      "standardization": stats, "models": models, "mae": mae_table(fold_rows)})
    return mae_table(predictions), folds, predictions


def develop(directory=DATA_ROOT, out=OUT):
    path = Path(out) / "register.json"
    require_new(path)
    states, rows, hashes = load_dev(directory)
    table, folds, predictions = loso(states, rows)
    stats, models = fit_all(rows)
    register = {
        "schema_version": 1, "precommitted_rule": RULE, "response": RESPONSE,
        "dev_states": states, "dev_configs": list(DEV_CONFIGS), "n_dev_cells": 24,
        "n_dev_capability_rows": len(rows), "dev_rows": rows, "input_sha256": hashes,
        "test_sets": TEST_SETS, "methods": list(METHODS), "candidates": list(CANDIDATES),
        "baselines": list(BASELINES), "n_params_per_capability": N_PARAMS,
        "architectures": ARCHITECTURES, "tokens_per_step": TOKENS_PER_STEP,
        "feature_names": FEATURE_NAMES, "standardization": stats,
        "standardization_rule": (
            "Natural logs of N0 and D0; population mean/std pooled over training response rows "
            "and capabilities, as in v53. Constant feature scale=1. Refit on training states "
            "in every LOSO fold. u_center is the training mean of log2(qmax); v=log2(g/128). "
            "Only phi covariates are z-scored; configuration factors are as specified."),
        "ridge": {"lambda": RIDGE, "objective": "SSE + lambda * sum(coefficients**2)",
                  "penalize_intercept": True, "candidates": list(CANDIDATES),
                  "interpolation_interpretation": "Per-config OLS with the common fixed ridge, as in v53."},
        "grids": {"p": P_GRID.tolist(), "q": Q_GRID.tolist(),
                  "selection": "Minimum unpenalized dev SSE after ridge fitting; exact ties: smaller p, then q."},
        "candidate_definitions": {
            "separable": "(beta.phi) * qmax**(-p) * (g/128)**q",
            "low_order_2d": "phi.[a0 + a1*u + a2*v + a3*u*v + a4*u^2]; coefficients ordered by term then phi",
            "same_input_interpolation": "Independent four-vector per dev config. " + INTERPOLATION_RULE,
            "mean": "Per-config mean of training dL. " + INTERPOLATION_RULE,
            "median": "Per-config median of training dL. " + INTERPOLATION_RULE,
            "zero": "Constant zero response",
        },
        "low_order_terms": list(TERM_NAMES), "models": models, "loso_table": table,
        "loso_folds": folds, "loso_predictions": predictions,
    }
    write_new(path, register)
    print_table(table, "LOSO, six folds (24 held-out cells per capability)")
    return register


def joint_input(directory, fallback):
    """Prefer V54's own dense reference; V46 is a dense-only fallback."""
    path = measurement_path(directory, JOINT_TAG)
    if path.is_file():
        table, sha = read_json(path)
        if "dense" in table:
            return state_input(JOINT_TAG, checked_losses(table["dense"], path), path), sha
    fallback = Path(fallback)
    if fallback.is_file():
        table, sha = read_json(fallback)
        if table.get("tag", JOINT_TAG) != JOINT_TAG:
            raise ValueError(f"{fallback}: dense input is not for {JOINT_TAG}")
        dense = checked_losses(table.get("dense", table), fallback)
        return state_input(JOINT_TAG, dense, fallback), sha
    print(f"JOINT TEST OMITTED: no dense losses for {JOINT_TAG}; expected dense in "
          f"{path} or {fallback}. Writing bit and granularity predictions.", flush=True)
    return None, None


def freeze(directory=DATA_ROOT, out=OUT, joint_dense=JOINT_DENSE):
    out = Path(out)
    path = out / "predictions.json"
    require_new(path)
    register, register_sha = read_json(out / "register.json")
    if register.get("schema_version") != 1 or register.get("precommitted_rule") != RULE:
        raise ValueError("Unrecognized register schema or pre-committed rule")
    # Saved dev dense inputs are the same ones used at registration, even if
    # the V54 files have since been extended with held-out measurements.
    states = {s["tag"]: s for s in register["dev_states"]}
    joint, joint_sha = joint_input(directory, joint_dense)
    hashes = dict(register["input_sha256"])
    if joint is not None:
        states[JOINT_TAG] = joint
        hashes[joint["path"]] = joint_sha
    predictions, missing = [], []
    stats = register["standardization"]
    for test_set, spec in register["test_sets"].items():
        for tag in spec["states"]:
            if tag not in states:
                missing.append({"test_set": test_set, "state": tag,
                                "configs": spec["configs"], "reason": "missing dense input"})
                continue
            state = states[tag]
            for config in spec["configs"]:
                for cap in CAPS:
                    z = phi(raw_features(state, cap), stats)
                    predictions.append({"test_set": test_set, "state": tag, "config": config,
                                        "capability": cap, "L0": state["L0"][cap],
                                        "predictions": predict_all(register["models"][cap], z, config, stats)})
    frozen = {"schema_version": 1, "precommitted_rule": RULE, "response": RESPONSE,
              "test_sets": register["test_sets"], "methods": register["methods"],
              "states": list(states.values()), "predictions": predictions, "missing": missing,
              "provenance": {"register_sha256": register_sha, "input_sha256": hashes}}
    write_new(path, frozen)
    print(f"Frozen {len(predictions) // len(CAPS)} test cells for every candidate and baseline.")
    return frozen


def response_flags(values, expected):
    """Do not claim 'all cells near zero' from an incomplete test set."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    near = bool(np.all(np.abs(values) < 0.03)) if n else None
    complete = n == expected
    return {"n_observed": n, "n_expected": expected, "complete": complete,
            "near_zero_all_measured": near,
            "near_zero_all_cells": near if complete or near is False else None,
            "collapse_any_measured": bool(np.any(values > 4)) if n else None,
            "collapse_all_measured": bool(np.all(values > 4)) if n else None,
            "n_collapse": int(np.sum(values > 4)), "n_negative": int(np.sum(values < 0)),
            "min_dL": float(values.min()) if n else None,
            "max_dL": float(values.max()) if n else None}


def compare(directory=DATA_ROOT, out=OUT):
    out = Path(out)
    path = out / "compare.json"
    require_new(path)
    frozen, prediction_sha = read_json(out / "predictions.json")
    if frozen.get("schema_version") != 1 or frozen.get("precommitted_rule") != RULE:
        raise ValueError("Unrecognized predictions schema or pre-committed rule")
    predictions = {}
    for row in frozen["predictions"]:
        key = (row["test_set"], row["state"], row["config"], row["capability"])
        if key in predictions or set(row["predictions"]) != set(METHODS):
            raise ValueError(f"Duplicate or incomplete frozen prediction: {key}")
        predictions[key] = row
    cache, hashes, rows, missing, summaries = {}, {}, [], [], {}
    for test_set, spec in frozen["test_sets"].items():
        test_rows = []
        for tag in spec["states"]:
            source = measurement_path(directory, tag)
            if tag not in cache:
                if source.is_file():
                    table, sha = read_json(source)
                    cache[tag] = table
                    hashes[path_label(source)] = sha
                else:
                    cache[tag] = {}
            table = cache[tag]
            for config in spec["configs"]:
                if config not in table or "dense" not in table:
                    missing.append({"test_set": test_set, "state": tag, "config": config,
                                    "reason": "missing measurement or same-file dense reference"})
                    continue
                dense = checked_losses(table["dense"], f"{source}: dense")
                losses = checked_losses(table[config], f"{source}: {config}")
                for cap in CAPS:
                    prediction = predictions.get((test_set, tag, config, cap))
                    row = {"test_set": test_set, "state": tag, "config": config, "capability": cap,
                           "loss": losses[cap], "dense": dense[cap], "dL": losses[cap] - dense[cap]}
                    if prediction is None:
                        row.update(predictions=None, absolute_errors=None, reason="no frozen prediction")
                    else:
                        estimates = prediction["predictions"]
                        row.update(predictions=estimates,
                                   absolute_errors={name: abs(estimates[name] - row["dL"]) for name in METHODS},
                                   dense_difference_from_prediction_input=dense[cap] - prediction["L0"])
                    test_rows.append(row)
        rows.extend(test_rows)
        scored = [r for r in test_rows if r["predictions"] is not None]
        expected = len(spec["states"]) * len(spec["configs"])
        flags = response_flags([r["dL"] for r in test_rows], expected * len(CAPS))
        by_cap = {cap: response_flags([r["dL"] for r in test_rows if r["capability"] == cap], expected)
                  for cap in CAPS}
        summary = {"n_expected_cells": expected, "n_measured_cells": len(test_rows) // len(CAPS),
                   "n_scored_cells": len(scored) // len(CAPS), "mae_table": mae_table(scored),
                   "response_flags": flags, "response_flags_by_capability": by_cap}
        summaries[test_set] = summary
        print_table(summary["mae_table"], f"{test_set} ({summary['n_scored_cells']}/{expected} cells scored)")
        label = lambda value: "unknown" if value is None else ("yes" if value else "no")
        print(f"  Responses: near-zero (all planned |dL|<0.03)={label(flags['near_zero_all_cells'])}; "
              f"collapse (any measured dL>4)={label(flags['collapse_any_measured'])}; "
              f"measured={summary['n_measured_cells']}/{expected} cells")
    result = {"schema_version": 1, "precommitted_rule": RULE, "response": RESPONSE,
              "test_sets": summaries, "rows": rows, "missing": missing,
              "thresholds": {"near_zero": "abs(dL) < 0.03 for every planned capability/cell",
                             "collapse": "dL > 4; report any, all and count among measured responses"},
              "provenance": {"predictions_sha256": prediction_sha, "measurement_sha256": hashes,
                             "frozen_inputs": frozen["provenance"]}}
    write_new(path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("develop", "freeze", "compare"))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--joint-dense", type=Path, default=JOINT_DENSE,
                        help="dense-only fallback for joint test when V54 dense is unavailable")
    args = parser.parse_args(argv)
    try:
        if args.mode == "develop":
            develop(args.data_root, args.out)
        elif args.mode == "freeze":
            freeze(args.data_root, args.out, args.joint_dense)
        else:
            compare(args.data_root, args.out)
    except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
