#!/usr/bin/env python3
"""Develop pruning predictors and freeze predictions using only NumPy and CPU.

    python -B analysis/v53_prune_dev.py develop
    python -B analysis/v53_prune_dev.py predict pythia-410m@step48000
    python -B analysis/v53_prune_dev.py compare pythia-410m@step48000

Every output is created exclusively: existing files are never overwritten.
Responses are signed Delta L, relative to each source's true dense loss; dense
is an input/reference, not a pruning training row. Standardization follows v49:
population mean/std of [log(N0), L0c, log(D0)], pooled over training response rows
and capabilities, with a fresh training-only standardizer in each LOSO fold.

The shared ridge requirement takes precedence over the description of A2 as
OLS: all four candidates minimize SSE + 0.001 * ||coefficients||^2, including
the intercept coefficient. A2 fits each anchor independently. The source-free
strength baseline uses unregularized least squares, as in v49. No response
clipping, model loading, GPU libraries, random numbers, or timestamps are used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v53-prune-dev"
DEV_ROOT = ROOT / "results/v6-capability-geometry"
CAPS = ("math", "code", "qa")
CANDIDATES = ("power", "A1", "A2", "cont")
BASELINES = ("strength_only", "median_curve", "zero")
METHODS = CANDIDATES + BASELINES
N_PARAMS = {"A1": 4, "power": 5, "cont": 8, "A2": 20}
ZERO_AT_DENSE = {"A1", "power", "cont"}
ANCHORS = (0.9, 0.8, 0.7, 0.6, 0.55)
OFF_COARSE_D = (0.75, 0.65, 0.55)
TEST_D = (0.85, 0.675, 0.575)
GAMMA_GRID = np.arange(10, 121, dtype=float) / 20.0
RIDGE = 1e-3
TOKENS_PER_STEP = 2097152
TIE_TOLERANCE = 0.02
FEATURE_NAMES = ["1", "z(log N0)", "z(L0c)", "z(log D0)"]
# Exact dimensions and matrix counting convention from v36. Keeping these local
# avoids importing analysis modules with unrelated I/O or environment settings.
ARCHITECTURES = {
    "160m": {"hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 12},
    "410m": {"hidden_size": 1024, "intermediate_size": 4096, "num_hidden_layers": 24},
    "1b": {"hidden_size": 2048, "intermediate_size": 8192, "num_hidden_layers": 16},
    "1.4b": {"hidden_size": 2048, "intermediate_size": 8192, "num_hidden_layers": 24},
    "2.8b": {"hidden_size": 2560, "intermediate_size": 10240, "num_hidden_layers": 32},
    "6.9b": {"hidden_size": 4096, "intermediate_size": 16384, "num_hidden_layers": 32},
}
INTERPOLATION_RULE = (
    "Linear interpolation between adjacent predicted anchors; outside the anchor "
    "range, linear extension from the nearest boundary pair. No clipping."
)
SELECTION_RULE = (
    "Pick the candidate with the lowest LOSO MAE averaged equally over math, code, "
    "and qa, using all held-out pruning rows (row-weighted within capability). "
    "If the best two candidates are within 0.02 nats (inclusive), choose among "
    "those two the lower-parameter candidate that is continuous and zero at d=1. "
    "Parameter order: A1 (4) < power (5) < cont (8) < A2 (20). A2 is piecewise "
    "continuous but is not constrained to zero at d=1. Exact MAE ties are ordered "
    "by parameter count. Baselines and the off-coarse subset do not select the winner."
)


def matrix_n0(size, architectures=None):
    a = (ARCHITECTURES if architectures is None else architectures)[size]
    h, m = a["hidden_size"], a["intermediate_size"]
    return a["num_hidden_layers"] * (4 * h * h + 2 * h * m)


def parse_tag(tag):
    match = re.fullmatch(r"pythia-([0-9.]+[mb])@step([1-9][0-9]*)", tag)
    if match is None or match[1] not in ARCHITECTURES:
        raise ValueError(f"Invalid/unknown Pythia tag {tag!r}; expected pythia-410m@step48000")
    return match[1], int(match[2])


def read_json(path):
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    print(f"READ {path} (sha256={sha})", flush=True)
    return json.loads(raw), sha


def checked_losses(value, context):
    if not isinstance(value, dict) or any(cap not in value for cap in CAPS):
        raise ValueError(f"{context}: expected losses for math, code, and qa")
    out = {}
    for cap in CAPS:
        loss = value[cap]
        if isinstance(loss, bool) or not isinstance(loss, (int, float)):
            raise ValueError(f"{context}: {cap} loss must be numeric")
        if not np.isfinite(loss) or loss < 0:
            raise ValueError(f"{context}: {cap} loss must be finite and nonnegative")
        out[cap] = float(loss)
    return out


def parse_losses(table, context):
    if not isinstance(table, dict) or "1.0" not in table:
        raise ValueError(f"{context}: missing true dense key '1.0'")
    dense = checked_losses(table["1.0"], f"{context}: dense")
    measured = {}
    for key, value in table.items():
        if key.startswith("_") or key == "1.0":
            continue
        d = float(key)
        if not np.isfinite(d) or not 0 < d < 1:
            raise ValueError(f"{context}: invalid pruning density {key!r}")
        if d < 0.55:
            continue
        if d in measured:
            raise ValueError(f"{context}: duplicate density {d}")
        measured[d] = checked_losses(value, f"{context}: density {key}")
    return dense, measured


def raw_features(n0, d0, l0):
    return [float(np.log(n0)), l0, float(np.log(d0))]


def load_dev(directory=DEV_ROOT):
    states, rows, hashes = [], [], {}
    for state_dir in sorted(directory.iterdir()):
        name = state_dir.name
        if not state_dir.is_dir() or not name.startswith("pythia-") or name.startswith("pythia-2.8b"):
            continue
        tag = name.replace("--", "@")
        size, step = parse_tag(tag)
        path = state_dir / "prune_losses.json"
        if not path.is_file():
            print(f"SKIP {state_dir}: no prune_losses.json", flush=True)
            continue
        table, sha = read_json(path)
        dense, measured = parse_losses(table, path)
        if not measured:
            raise ValueError(f"{path}: no pruning measurements at density >= 0.55")
        try:
            relative = str(path.relative_to(ROOT))
        except ValueError:
            relative = str(path)
        hashes[relative] = sha
        n0, d0 = matrix_n0(size), step * TOKENS_PER_STEP
        states.append({"state": name, "tag": tag, "size": size, "step": step,
                       "N0": n0, "D0": d0, "L0": dense,
                       "densities": sorted(measured, reverse=True), "path": relative})
        for d in sorted(measured, reverse=True):
            for cap in CAPS:
                rows.append({"source": tag, "cap": cap, "d": d,
                             "phi_raw": raw_features(n0, d0, dense[cap]),
                             "y": measured[d][cap] - dense[cap]})
    if len(states) < 2:
        raise ValueError("LOSO requires at least two measured development sources")
    return states, rows, hashes


def zstats(rows):
    raw = np.asarray([r["phi_raw"] for r in rows], dtype=float)
    center, raw_scale = raw.mean(axis=0), raw.std(axis=0, ddof=0)
    # Constant covariates contribute zero; ridge still identifies coefficients.
    scale = np.where(raw_scale > 0, raw_scale, 1.0)
    return {"center": center.tolist(), "scale": scale.tolist(),
            "constant_features": np.flatnonzero(raw_scale == 0).tolist()}


def standardize(raw, stats):
    raw = np.asarray(raw, dtype=float)
    return np.concatenate((np.ones(raw.shape[:-1] + (1,)),
                           (raw - np.asarray(stats["center"])) / np.asarray(stats["scale"])), axis=-1)


def ridge_fit(x, y):
    return np.linalg.solve(x.T @ x + RIDGE * np.eye(x.shape[1]), x.T @ y)


def shape(d, gamma):
    return ((1.0 - np.asarray(d)) / 0.3) ** gamma


def linear_curve(anchors, density):
    """Exact anchors, adjacent interpolation, boundary-pair linear extension."""
    ds = sorted(float(k) for k in anchors)
    values = {float(k): float(v) for k, v in anchors.items()}
    if len(ds) < 2:
        raise ValueError("Linear interpolation/extension requires at least two anchors")
    if density in values:
        return values[density]
    i = int(np.clip(np.searchsorted(ds, density, side="right") - 1, 0, len(ds) - 2))
    lo, hi = ds[i:i + 2]
    return values[lo] + (values[hi] - values[lo]) * ((density - lo) / (hi - lo))


def fit_capability(rows, cap, stats):
    cr = [r for r in rows if r["cap"] == cap]
    z = standardize([r["phi_raw"] for r in cr], stats)
    y = np.asarray([r["y"] for r in cr])
    d = np.asarray([r["d"] for r in cr])
    best_power, best_strength = None, None
    for gamma in GAMMA_GRID:
        sh = shape(d, gamma)
        x = z * sh[:, None]
        beta = ridge_fit(x, y)
        sse = float(np.sum((x @ beta - y) ** 2))
        if best_power is None or sse < best_power["sse"]:
            best_power = {"beta": beta.tolist(), "gamma": float(gamma), "sse": sse}
        amplitude = float(sh @ y / (sh @ sh))
        sse_strength = float(np.sum((amplitude * sh - y) ** 2))
        if best_strength is None or sse_strength < best_strength["sse"]:
            best_strength = {"A": amplitude, "gamma": float(gamma), "sse": sse_strength}
    a2, counts = {}, {}
    for anchor in ANCHORS:
        mask = d == anchor
        if not np.any(mask):
            raise ValueError(f"{cap}: no training source measured at required A2 anchor {anchor}")
        a2[str(anchor)] = ridge_fit(z[mask], y[mask]).tolist()
        counts[str(anchor)] = int(mask.sum())
    strength = (1.0 - d)[:, None]
    cont = ridge_fit(np.concatenate((z * strength, z * strength ** 2), axis=1), y)
    return {
        "power": best_power,
        "A1": {"beta": ridge_fit(z * shape(d, 1.0)[:, None], y).tolist(), "gamma": 1.0},
        "A2": {"anchors": a2, "n_obs_by_anchor": counts},
        "cont": {"beta": cont[:4].tolist(), "zeta": cont[4:].tolist()},
        "strength_only": best_strength,
        "median_curve": {"anchors": {str(v): float(np.median(y[d == v])) for v in sorted(set(d))}},
        "zero": {"value": 0.0},
    }


def fit_all(rows):
    stats = zstats(rows)
    return stats, {cap: fit_capability(rows, cap, stats) for cap in CAPS}


def predict_all(fits, z, density):
    s = 1.0 - density
    return {
        "power": float(np.dot(fits["power"]["beta"], z) * shape(density, fits["power"]["gamma"])),
        "A1": float(np.dot(fits["A1"]["beta"], z) * shape(density, 1.0)),
        "A2": linear_curve({d: np.dot(b, z) for d, b in fits["A2"]["anchors"].items()}, density),
        "cont": float(np.dot(fits["cont"]["beta"], z) * s + np.dot(fits["cont"]["zeta"], z) * s ** 2),
        "strength_only": float(fits["strength_only"]["A"] * shape(density, fits["strength_only"]["gamma"])),
        "median_curve": linear_curve(fits["median_curve"]["anchors"], density),
        "zero": 0.0,
    }


def summarize_loso(predictions):
    table = []
    for subset in ("all", "off_coarse"):
        subset_rows = [r for r in predictions if subset == "all" or r["d"] in OFF_COARSE_D]
        for name in METHODS:
            entry = {"subset": subset, "candidate": name}
            for cap in CAPS:
                errors = [abs(r["predictions"][name] - r["y"]) for r in subset_rows if r["cap"] == cap]
                entry[f"{cap}_n"] = len(errors)
                entry[f"{cap}_mae"] = float(np.mean(errors)) if errors else None
            maes = [entry[f"{cap}_mae"] for cap in CAPS]
            entry["mean_mae"] = float(np.mean(maes)) if all(v is not None for v in maes) else None
            table.append(entry)
    return table


def loso(states, rows):
    predictions, folds = [], []
    for state in states:
        source = state["tag"]
        train = [r for r in rows if r["source"] != source]
        held_out = [r for r in rows if r["source"] == source]
        stats, models = fit_all(train)
        for row in held_out:
            z = standardize(row["phi_raw"], stats)
            predictions.append({k: row[k] for k in ("source", "cap", "d", "y")}
                               | {"predictions": predict_all(models[row["cap"]], z, row["d"])})
        folds.append({"held_out": source, "train_sources": sorted({r["source"] for r in train}),
                      "n_train_rows": len(train), "n_test_rows": len(held_out),
                      "standardization": stats,
                      "gamma": {cap: {name: models[cap][name]["gamma"]
                                      for name in ("power", "strength_only")} for cap in CAPS}})
        print(f"LOSO {source}: trained on {len(train)} rows; held out {len(held_out)} rows", flush=True)
    return summarize_loso(predictions), folds, predictions


def select_candidate(table):
    scores = {r["candidate"]: r["mean_mae"] for r in table
              if r["subset"] == "all" and r["candidate"] in CANDIDATES}
    if set(scores) != set(CANDIDATES) or not all(v is not None and np.isfinite(v) for v in scores.values()):
        raise ValueError("Selection requires finite all-row LOSO MAE for all four candidates")
    ranked = sorted(CANDIDATES, key=lambda name: (scores[name], N_PARAMS[name]))
    best, second = ranked[:2]
    gap = scores[second] - scores[best]
    selected = best
    tied = bool(gap <= TIE_TOLERANCE or np.isclose(gap, TIE_TOLERANCE, rtol=0, atol=1e-12))
    if tied:
        eligible = [name for name in ranked[:2] if name in ZERO_AT_DENSE]
        if eligible:
            selected = min(eligible, key=N_PARAMS.__getitem__)
    return selected, {"ranked_candidates": ranked, "mean_mae": scores,
                      "best_two_gap_nats": gap, "tie_rule_applied": tied}


def require_new(*paths):
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def write_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(content)
    print(f"WROTE {path}", flush=True)


def json_text(value):
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def csv_text(table):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["subset", "candidate", "math_n", "code_n", "qa_n",
                                               "math_mae", "code_mae", "qa_mae", "mean_mae"],
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(table)
    return stream.getvalue()


def print_table(table, title):
    print(f"\n{title} (MAE, nats)")
    print(f"{'candidate':<16} {'math':>10} {'code':>10} {'qa':>10} {'mean':>10}")
    for row in table:
        values = [row[f"{cap}_mae"] for cap in CAPS] + [row["mean_mae"]]
        cells = " ".join(f"{v:10.6f}" if v is not None else f"{'N/A':>10}" for v in values)
        print(f"{row['candidate']:<16} {cells}")


def develop():
    register_path, csv_path = OUT / "register.json", OUT / "loso_table.csv"
    require_new(register_path, csv_path)
    states, rows, hashes = load_dev()
    print(f"DEVELOP {len(states)} states; {len(rows)} capability/density rows", flush=True)
    table, folds, predictions = loso(states, rows)
    selected, selection_details = select_candidate(table)
    stats, models = fit_all(rows)
    register = {
        "schema_version": 1,
        "dev_state_rule": "Every measured pythia-* state except pythia-2.8b*; all 0.55 <= d < 1 rows.",
        "dev_states": states, "n_dev_states": len(states), "n_dev_rows": len(rows),
        "dev_hashes": hashes, "architectures": ARCHITECTURES, "tokens_per_step": TOKENS_PER_STEP,
        "response": "signed Delta L_c(d) = prune_loss_c(d) - dense_loss_c(1.0), in nats",
        "feature_names": FEATURE_NAMES,
        "standardization": stats | {
            "rule": "Natural logs in raw parameter/token units; mean and population std (ddof=0) pooled "
                    "over training response rows and capabilities, as in v49. Scale=1 for a constant "
                    "feature. Refit within each LOSO training fold; full-dev values used for prediction."},
        "ridge": {"lambda": RIDGE, "objective": "SSE + lambda * sum(coefficients**2)",
                  "penalize_intercept": True, "candidates": list(CANDIDATES),
                  "A2_interpretation": "Per-anchor linear regression with the same fixed ridge as all candidates."},
        "gamma_grid": GAMMA_GRID.tolist(),
        "gamma_selection": "Minimum unpenalized training SSE after fitting; exact ties use the smaller gamma.",
        "candidate_definitions": {
            "power": "(beta.phi) * ((1-d)/0.3)**gamma",
            "A1": "(beta.phi) * ((1-d)/0.3), gamma=1",
            "A2": "Independent beta.phi at each fixed anchor, using only sources measured there. " + INTERPOLATION_RULE,
            "cont": "(beta.phi)*(1-d) + (zeta.phi)*(1-d)**2",
            "strength_only": "Unregularized A*((1-d)/0.3)**gamma on all training rows per capability; same gamma grid.",
            "median_curve": "Median training response at every measured training density. " + INTERPOLATION_RULE,
            "zero": "0 at every density",
        },
        "n_params_per_capability": N_PARAMS, "A2_anchors": list(ANCHORS),
        "off_coarse_densities": list(OFF_COARSE_D), "test_densities": list(TEST_D),
        "models": models, "loso_table": table, "loso_folds": folds, "loso_predictions": predictions,
        "selected_candidate": selected, "selection_rule": SELECTION_RULE,
        "selection_details": selection_details,
    }
    # Serialize both before creating either artifact (also checks for NaN/Inf).
    register_text, table_text = json_text(register), csv_text(table)
    write_new(register_path, register_text)
    write_new(csv_path, table_text)
    for subset in ("all", "off_coarse"):
        print_table([r for r in table if r["subset"] == subset], f"LOSO {subset}")
    print(f"\nSelected: {selected}")


def predict(tag):
    size, step = parse_tag(tag)
    output_path = OUT / f"predictions_{tag}.json"
    require_new(output_path)
    dense_path = OUT / f"dense_{tag}.json"
    if not dense_path.is_file():
        raise FileNotFoundError(f"Missing dense evaluation: {dense_path}. "
                                "Write a JSON object {'math': ..., 'code': ..., 'qa': ...} first.")
    register_path = OUT / "register.json"
    if not register_path.is_file():
        raise FileNotFoundError(f"Missing frozen register: {register_path}; run 'develop' first.")
    reg, reg_sha = read_json(register_path)
    dense_values, dense_sha = read_json(dense_path)
    dense = checked_losses(dense_values, dense_path)
    n0 = matrix_n0(size, reg["architectures"])
    d0 = step * reg["tokens_per_step"]
    predictions = {}
    for cap in CAPS:
        z = standardize(raw_features(n0, d0, dense[cap]), reg["standardization"])
        predictions[cap] = {str(d): predict_all(reg["models"][cap], z, d) for d in reg["test_densities"]}
    out = {
        "schema_version": 1,
        "target": {"tag": tag, "size": size, "step": step, "N0": n0, "D0": d0, "L0": dense},
        "densities": reg["test_densities"], "response": reg["response"],
        "selected_candidate": reg["selected_candidate"], "predictions": predictions,
        "provenance": {"register_sha256": reg_sha, "dense_sha256": dense_sha},
    }
    write_new(output_path, json_text(out))


def compare(tag):
    parse_tag(tag)
    output_path = OUT / f"compare_{tag}.json"
    require_new(output_path)
    prediction_path = OUT / f"predictions_{tag}.json"
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Missing frozen predictions: {prediction_path}; run 'predict {tag}' first.")
    frozen, pred_sha = read_json(prediction_path)
    if frozen["target"]["tag"] != tag:
        raise ValueError(f"{prediction_path}: target tag does not match {tag}")
    measurement_path = DEV_ROOT / tag.replace("@", "--") / "prune_losses.json"
    table, measurement_sha = read_json(measurement_path)
    dense, measured = parse_losses(table, measurement_path)
    missing = [d for d in frozen["densities"] if d not in measured]
    if missing:
        raise ValueError(f"{measurement_path}: missing required test densities {missing}; compare requires the full panel")
    observed, errors, rows = {}, {}, []
    for cap in CAPS:
        observed[cap], errors[cap] = {}, {}
        for d in frozen["densities"]:
            key = str(d)
            y = measured[d][cap] - dense[cap]
            observed[cap][key] = y
            estimates = frozen["predictions"][cap][key]
            errors[cap][key] = {name: abs(estimates[name] - y) for name in METHODS}
            rows.append({"cap": cap, "d": d, "y": y, "predictions": estimates})
    summary = [r for r in summarize_loso(rows) if r["subset"] == "all"]
    mae = {r["candidate"]: {cap: r[f"{cap}_mae"] for cap in CAPS} | {"mean": r["mean_mae"]}
           for r in summary}
    out = {
        "schema_version": 1, "tag": tag, "densities": frozen["densities"],
        "selected_candidate": frozen["selected_candidate"], "response": frozen["response"],
        "dense": dense,
        "dense_difference_from_prediction_input": {cap: dense[cap] - frozen["target"]["L0"][cap] for cap in CAPS},
        "observed_delta_loss": observed, "absolute_errors": errors, "mae": mae,
        "provenance": {"predictions_sha256": pred_sha, "prune_losses_sha256": measurement_sha,
                       "prediction_inputs": frozen["provenance"]},
    }
    write_new(output_path, json_text(out))
    print_table(summary, tag)
    print(f"Selected: {frozen['selected_candidate']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("develop", help="LOSO selection and full-development refits")
    for name in ("predict", "compare"):
        mode = modes.add_parser(name)
        mode.add_argument("tag", help="e.g. pythia-410m@step48000")
    args = parser.parse_args(argv)
    try:
        if args.mode == "develop":
            develop()
        elif args.mode == "predict":
            predict(args.tag)
        else:
            compare(args.tag)
    except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
