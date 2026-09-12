#!/usr/bin/env python3
"""CPU-only capability-conditioning ablation on the frozen V53/V69/V70/V72 panels.

    python -B analysis/v76_cap_conditioning.py --selftest
    python -B analysis/v76_cap_conditioning.py
    python -B analysis/v76_cap_conditioning.py --check

Writes only results/v76-cap-conditioning/, including staged paper table and code
mirror. All estimators use the exact historical development snapshots. V76 is a
retrospective ablation on frozen panels, not a newly preregistered experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "results/v53-prune-dev/register.json").is_file())
OUT = ROOT / "results/v76-cap-conditioning"
CAPS = ("math", "code", "qa")
METHODS = ("A", "B", "C", "D")
LABELS = {"A": "Per capability", "B": "Shared + scale", "C": "Shared + offset", "D": "Shared only"}
N_BOOT = 5000
SEED = 0
PRUNE_TARGETS = ("pythia-410m@step48000", "pythia-1.4b@step112000", "pythia-6.9b@step80000")
REPEAT_TARGETS = ("pythia-2.8b@step16000", "pythia-2.8b@step143000")
QUANT_CONFIGS = tuple(f"b{b}_g{g}" for b in (3, 4, 5) for g in (64, 128, 256))
QUANT_TARGETS = ("pythia-410m@step143000", "pythia-1.4b@step16000", "pythia-1.4b@step112000")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(actual, expected, context):
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12, err_msg=context)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class Inputs:
    def __init__(self):
        self.hashes = {}
        self.cache = {}

    def read(self, rel, expected=None):
        rel = str(rel)
        raw = (ROOT / rel).read_bytes()
        actual = digest(raw)
        require(self.hashes.setdefault(rel, actual) == actual, f"Input changed during run: {rel}")
        if expected is not None:
            require(actual == expected, f"Historical input hash mismatch: {rel}")
        return raw

    def json(self, rel, expected=None):
        rel = str(rel)
        if rel not in self.cache or expected is not None:
            self.cache[rel] = json.loads(self.read(rel, expected))
        return self.cache[rel]

    def verify(self):
        for rel, expected in self.hashes.items():
            require(digest((ROOT / rel).read_bytes()) == expected, f"Input changed: {rel}")


def indexed(rows, fields):
    result = {tuple(r[k] for k in fields): r for r in rows}
    require(len(result) == len(rows), f"Duplicate rows indexed by {fields}")
    return result


def linear(anchors, value):
    xs = sorted(float(x) for x in anchors)
    require(len(xs) >= 2, "Need at least two curve anchors")
    ys = {float(x): float(y) for x, y in anchors.items()}
    i = int(np.clip(np.searchsorted(xs, value, side="right") - 1, 0, len(xs) - 2))
    lo, hi = xs[i:i + 2]
    return float(ys[lo] + (ys[hi] - ys[lo]) * (value - lo) / (hi - lo))


def quant_curve(anchors, config):
    match = re.fullmatch(r"b([345])_g([1-9][0-9]*)", config)
    require(match is not None, f"Unsupported quantization configuration: {config}")
    bit, group = map(int, match.groups())
    # Confirmation uses only exact b=3,4,5; V69's bilinear x interpolation
    # therefore reduces exactly to this adjacent log2(g/128) interpolation.
    v = math.log2(group / 128)
    value = linear({math.log2(g / 128): anchors[f"b{bit}_g{g}"] for g in (64, 128, 256)}, v)
    return max(0.0, value) if abs(v) > 1 else value


def weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    require(np.sum(weights) > 0, "Zero total weight")
    return float(values[np.searchsorted(np.cumsum(weights), np.sum(weights) / 2, side="left")])


def corrections(x, y, objective="ols"):
    """One fixed signed scale and one offset per capability; no test inputs."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    require(y.shape == (len(x), len(CAPS)) and x @ x > 1e-24, "Degenerate shared curve")
    if objective == "ols":
        scale = x @ y / (x @ x)
        offset = (y - x[:, None]).mean(axis=0)
    elif objective == "lad":
        mask = x != 0
        scale = np.array([weighted_median(y[mask, ci] / x[mask], np.abs(x[mask]))
                          for ci in range(len(CAPS))])
        offset = np.median(y - x[:, None], axis=0)
    else:
        raise ValueError(objective)
    return dict(zip(CAPS, map(float, scale))), dict(zip(CAPS, map(float, offset)))


def fit_anchors(cells, shared_order="median_of_means", correction_rows="anchors", objective="ols"):
    """cells have state, x and three aligned development responses in y."""
    indexed(cells, ("state", "x"))
    keys = sorted({r["x"] for r in cells}, key=str)
    medians, shared, counts = {}, {}, {}
    for key in keys:
        y = np.asarray([r["y"] for r in cells if r["x"] == key])
        medians[key] = np.median(y, axis=0)
        shared[key] = float(np.median(y.mean(axis=1)) if shared_order == "median_of_means"
                            else medians[key].mean())
        counts[key] = len(y)
    if correction_rows == "anchors":
        x, y = [shared[k] for k in keys], [medians[k] for k in keys]
    else:
        x, y = [shared[r["x"]] for r in cells], [r["y"] for r in cells]
    scales, offsets = corrections(x, y, objective)
    models = {"A": {c: {k: float(medians[k][i]) for k in keys} for i, c in enumerate(CAPS)},
              "B": {c: {k: scales[c] * shared[k] for k in keys} for c in CAPS},
              "C": {c: {k: shared[k] + offsets[c] for k in keys} for c in CAPS},
              "D": {c: dict(shared) for c in CAPS}}
    return {"shared_order": shared_order, "correction_rows": correction_rows,
            "objective": objective, "shared_anchors": shared, "scales": scales,
            "offsets": offsets, "n_states_by_anchor": counts, "models": models}


def fit_reuse(points):
    x = np.log1p([p["E"] for p in points])
    y = np.array([[p["delta"][c] for c in CAPS] for p in points])
    coefficients = x @ y / (x @ x)
    shared = float(x @ y.mean(axis=1) / (x @ x))
    require(abs(shared) > 1e-12, "Zero shared slope: scale parameterization is singular")
    scales, offsets = corrections(shared * x, y)
    scaled = shared * np.array([scales[c] for c in CAPS])
    close(scaled, coefficients, "Reuse-count A and B must be algebraically equivalent")
    return {"coefficients": dict(zip(CAPS, map(float, coefficients))), "shared_coefficient": shared,
            "scales": scales, "offsets": offsets,
            "max_coefficient_difference_A_B": float(np.max(np.abs(scaled - coefficients))),
            "equivalence": "a_c=(w.T y_c)/(w.T w); a=(w.T mean_c(y_c))/(w.T w); "
                           "s_c=((a*w).T y_c)/((a*w).T(a*w))=a_c/a, for a != 0. "
                           "Thus a*s_c*w=a_c*w; arbitrary signed scales are allowed."}


def load_pruning(inputs):
    base = "results/v53-prune-dev"
    reg = inputs.json(f"{base}/register.json")
    require(reg["n_dev_states"] == 17 and reg["n_dev_rows"] == 252, "Wrong V53 development size")
    dev_states = {s["tag"] for s in reg["dev_states"]}
    require(len(dev_states) == 17 and not dev_states.intersection(PRUNE_TARGETS + REPEAT_TARGETS),
            "Pruning development/confirmation overlap")
    snapshot = indexed(reg["loso_predictions"], ("source", "d", "cap"))
    cells = []
    for state in reg["dev_states"]:
        raw = inputs.json(state["path"], reg["dev_hashes"][state["path"]])
        for d in state["densities"]:
            y = [raw[str(d)][c] - raw["1.0"][c] for c in CAPS]
            for c, value in zip(CAPS, y):
                close(value, snapshot[state["tag"], d, c]["y"], "V53 development response")
            cells.append({"state": state["tag"], "x": str(d), "y": y})
    require(len(cells) == 84 and len(snapshot) == 252, "Incomplete V53 development grid")
    model = fit_anchors(cells)
    for c in CAPS:
        require(model["models"]["A"][c] == reg["models"][c]["median_curve"]["anchors"],
                "V53 median anchors differ from delivered model")
    # All fits precede loading confirmation responses.
    fits = anchor_sensitivities(cells, model)
    rows = []
    for tag in PRUNE_TARGETS:
        comp = inputs.json(f"{base}/compare_{tag}.json")
        pred = inputs.json(f"{base}/predictions_{tag}.json", comp["provenance"]["predictions_sha256"])
        require(pred["provenance"]["register_sha256"] == inputs.hashes[f"{base}/register.json"],
                "V53 predictions used a different register")
        require(comp["tag"] == tag and comp["densities"] == [0.85, 0.675, 0.575], "V53 panel changed")
        raw = inputs.json(f"results/v6-capability-geometry/{tag.replace('@', '--')}/prune_losses.json",
                          comp["provenance"]["prune_losses_sha256"])
        for d in comp["densities"]:
            for c in CAPS:
                y = comp["observed_delta_loss"][c][str(d)]
                close(y, raw[str(d)][c] - raw["1.0"][c], "V53 confirmation response")
                rows.append({"state": tag, "cluster": tag, "panel": "v53_confirmation", "x": str(d),
                             "capability": c, "actual": y,
                             "frozen_A": pred["predictions"][c][str(d)]["median_curve"]})
    base = "results/v72-prune-repeat"
    comp = inputs.json(f"{base}/compare.json")
    freeze = inputs.json(f"{base}/freeze.json", comp["provenance"]["freeze_sha256"])
    require(freeze["provenance"]["register_sha256"] == inputs.hashes["results/v53-prune-dev/register.json"],
            "V72 used a different V53 development snapshot")
    revisions = freeze["cache"]["revisions"]
    require({w['blob_sha256'] for w in revisions['step16000']['weights']} !=
            {w['blob_sha256'] for w in revisions['step143000']['weights']},
            "V72 freeze no longer records distinct weight identities")
    frozen_rows = indexed(freeze["predictions"], ("source", "density", "capability"))
    actual_rows = indexed(comp["rows"], ("source", "density", "capability"))
    expected = {(s, d, c) for s in REPEAT_TARGETS for d in (0.85, 0.75, 0.65) for c in CAPS}
    require(set(actual_rows) == set(frozen_rows) == expected, "V72 panel incomplete")
    for rel, expected_hash in comp["provenance"]["measurement_sha256"].items():
        inputs.read(rel, expected_hash)
    for key, r in actual_rows.items():
        require(r["predictions"] == frozen_rows[key]["predictions"], "V72 predictions changed")
        path = f"{base}/measurements/{r['source'].replace('@', '--')}/d{r['density']}/prune_losses.json"
        raw = inputs.json(path)
        close(r["observed_delta_loss"], raw[str(r["density"])][r["capability"]] - raw["1.0"][r["capability"]],
              "V72 confirmation response")
        rows.append({"state": r["source"], "cluster": r["source"], "panel": "v72_repeat",
                     "x": str(r["density"]), "capability": r["capability"],
                     "actual": r["observed_delta_loss"], "frozen_A": r["predictions"]["median_curve"]})
    require(len(rows) == 45, "Need 15 pruning state-density cells")
    return cells, fits, rows


def load_quantization(inputs):
    base = "results/v69-quant-confirm"
    dev = inputs.json(f"{base}/develop.json")
    snapshot = indexed(dev["dev_rows"], ("state", "config", "capability"))
    states = {s["tag"] for s in dev["dev_states"]}
    expected = {(s, q, c) for s in states for q in QUANT_CONFIGS for c in CAPS}
    require(len(states) == 6 and len(snapshot) == 162 and set(snapshot) == expected, "Need 54 V69 dev cells")
    canonical = json.dumps({"states": dev["dev_states"], "rows": dev["dev_rows"]}, sort_keys=True,
                           allow_nan=False, separators=(",", ":")).encode()
    require(digest(canonical) == dev["dev_subset_sha256"], "V69 development subset hash changed")
    cells = []
    for state in dev["dev_states"]:
        # V69 measurements append confirmation configurations to some same files:
        # verify the frozen development subset, not obsolete whole-file hashes.
        raw = inputs.json(state["path"])
        for config in QUANT_CONFIGS:
            y = [snapshot[state["tag"], config, c]["dL"] for c in CAPS]
            close(y, [raw[config][c] - raw["dense"][c] for c in CAPS], "V69 development subset")
            cells.append({"state": state["tag"], "x": config, "y": y})
    model = fit_anchors(cells)
    for c in CAPS:
        require(model["models"]["A"][c] == dev["models"][c]["median"]["anchors"], "V69 median anchors changed")
    fits = anchor_sensitivities(cells, model)
    freeze = inputs.json(f"{base}/freeze.json")
    comp = inputs.json(f"{base}/compare.json")
    require(freeze["provenance"]["develop_sha256"] == inputs.hashes[f"{base}/develop.json"] and
            comp["provenance"]["freeze_sha256"] == inputs.hashes[f"{base}/freeze.json"], "V69 freeze chain changed")
    require(comp["complete"] and comp["n_measured_cells"] == 21, "Incomplete V69 confirmation")
    require(freeze["models"] == dev["models"], "V69 frozen models differ from development")
    frozen_rows = indexed(freeze["predictions"], ("state", "config", "capability"))
    actual_rows = indexed(comp["rows"], ("state", "config", "capability"))
    expected_cells = {(s, f"b{b}_g{g}") for s in QUANT_TARGETS for b in (3, 4, 5) for g in (32, 512)}
    expected_cells |= {(QUANT_TARGETS[-1], f"b{b}_g128") for b in (3, 4, 5)}
    require(set(frozen_rows) == set(actual_rows) == {(s, q, c) for s, q in expected_cells for c in CAPS},
            "V69 confirmation grid changed")
    require(not {(r['state'], r['x']) for r in cells}.intersection(expected_cells), "V69 cell leakage")
    for rel, expected_hash in comp["provenance"]["measurement_sha256"].items():
        inputs.read(rel, expected_hash)
    rows = []
    for key, r in actual_rows.items():
        require(all(r[k] == v for k, v in frozen_rows[key].items()), "V69 frozen input/prediction changed")
        raw = inputs.json(f"results/v54-quant-group/{r['state'].replace('@', '--')}/quant_group_losses.json")
        close(r["dL"], raw[r["config"]][r["capability"]] - raw["dense"][r["capability"]], "V69 confirmation response")
        rows.append({"state": r["state"], "cluster": r["state"], "panel": r["test_set"], "x": r["config"],
                     "capability": r["capability"], "actual": r["dL"], "frozen_A": r["predictions"]["median"]})
    return cells, fits, rows


def load_distillation(inputs):
    base = "results/v70-distill-confirm"
    dev = inputs.json(f"{base}/develop.json")
    points = dev["points"]
    counts = Counter(p["cluster"] for p in points)
    require(len(points) == 100 and len(counts) == 25 and set(counts.values()) == {4}, "Need 25 dev trajectories")
    for p in points:
        raw = inputs.json(p["file"], dev["inputs_sha256"][p["file"]])
        close(p["E"], p["Tc"] / p["DU"], "Development reuse count")
        require(p["delta"] == raw["delta"] and p["Tc"] == raw["completion_tokens_seen"], "V70 development response changed")
    model = fit_reuse(points)
    for c in CAPS:
        close(model["coefficients"][c], dev["models"][c]["E"]["coef"][0], "V70 E coefficient")
    freeze = inputs.json(f"{base}/freeze.json")
    comp = inputs.json(f"{base}/compare.json")
    require(comp["complete"] and comp["freeze_sha256"] == inputs.hashes[f"{base}/freeze.json"], "V70 freeze chain changed")
    require(inputs.read(f"{base}/FREEZE_V70").decode().strip() == inputs.hashes[f"{base}/freeze.json"], "V70 sentinel mismatch")
    require(freeze["models"] == dev["models"] and
            freeze["inputs_sha256"][f"{base}/develop.json"] == inputs.hashes[f"{base}/develop.json"], "V70 frozen development changed")
    frozen_rows = indexed(freeze["predictions"], ("student", "pool", "T_planned", "capability"))
    actual_rows = indexed(comp["rows"], ("student", "pool", "T_planned", "capability"))
    expected = {(s, f"U200_s{seed}", t, c) for s in ("gemma3-270m", "gemma3-1b")
                for seed in range(31, 37) for t in (50000, 100000, 200000) for c in CAPS}
    require(set(frozen_rows) == set(actual_rows) == expected, "Need 12 confirmation trajectories / 36 checkpoints")
    require(not {p['pool'] for p in points}.intersection(r['pool'] for r in comp['rows']), "Distillation pool leakage")
    rows = []
    for key, r in actual_rows.items():
        require(all(r[k] == v for k, v in frozen_rows[key].items()), "V70 frozen prediction/input changed")
        raw = inputs.json(r["file"], comp["measurement_sha256"][r["file"]])
        c = r["capability"]
        close(r["actual"], raw["delta"][c], "V70 confirmation response")
        close(r["actual"], raw["post_training"][c] - raw["dense"][c], "V70 signed response")
        close(r["E_planned"], r["T_planned"] / r["DU"], "Frozen planned reuse count")
        w = math.log1p(r["E_planned"])
        shared = model["shared_coefficient"] * w
        predictions = {"A": model["coefficients"][c] * w, "B": model["scales"][c] * shared,
                       "C": shared + model["offsets"][c], "D": shared}
        close(predictions["A"], r["predictions"]["E"], "V70 delivered E prediction")
        close(predictions["A"], predictions["B"], "A/B reparameterization")
        rows.append({"state": r["student"], "student": r["student"], "panel": r["student"],
                     "trajectory": r["student"] + "|" + r["pool"], "cluster": r["pool"],
                     "x": str(r["T_planned"]), "E_planned": r["E_planned"], "T_actual": r["T_actual"],
                     "capability": c, "actual": r["actual"], "predictions": predictions})
    return points, model, rows


def anchor_sensitivities(cells, primary):
    return {"primary": primary,
            "mean_of_capability_medians": fit_anchors(cells, shared_order="mean_of_medians"),
            "development_rows_ols": fit_anchors(cells, correction_rows="development_rows"),
            "median_anchors_lad": fit_anchors(cells, objective="lad"),
            "development_rows_lad": fit_anchors(cells, correction_rows="development_rows", objective="lad")}


def predict_anchors(rows, fit, arm):
    curve = (lambda anchors, x: linear(anchors, float(x))) if arm == "pruning" else quant_curve
    result = []
    for r in rows:
        predictions = {m: curve(fit["models"][m][r["capability"]], r["x"]) for m in METHODS}
        close(predictions["A"], r["frozen_A"], f"{arm} delivered A prediction")
        result.append({**r, "predictions": predictions})
    return result


def score(rows):
    """Cell/checkpoint MAE; paired cluster bootstrap preserves unequal cell counts."""
    indexed(rows, ("state", "cluster", "x", "capability"))
    clusters = sorted({r["cluster"] for r in rows})
    n = len(clusters)
    require(n > 0, "Empty scoring panel")
    draws = np.random.default_rng(SEED).integers(n, size=(N_BOOT, n)) if n > 1 else None
    summaries = {}
    for cap in (*CAPS, "macro"):
        rr = [r for r in rows if cap == "macro" or r["capability"] == cap]
        sums, sizes, cluster_details = [], [], []
        for cluster in clusters:
            cr = [r for r in rr if r["cluster"] == cluster]
            errors = np.array([[abs(r["predictions"][m] - r["actual"]) for m in METHODS] for r in cr])
            require(len(errors) > 0 and np.isfinite(errors).all(), "Incomplete/nonfinite scoring cluster")
            sums.append(errors.sum(axis=0))
            sizes.append(len(errors))
            cluster_details.append({"cluster": cluster, "n_rows": len(errors),
                                    "mae": dict(zip(METHODS, map(float, errors.mean(axis=0))))})
        sums, sizes = np.asarray(sums), np.asarray(sizes)
        mae = sums.sum(axis=0) / sizes.sum()
        boot = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)[:, None] if n > 1 else None
        gains = {}
        for j, method in enumerate(METHODS[1:], 1):
            gain = float(mae[j] - mae[0])
            ci = list(map(float, np.quantile(boot[:, j] - boot[:, 0], [.025, .975]))) if n > 1 else None
            gains[method] = {"gain_nats": gain, "ci95_nats": ci,
                             "relative_gain_percent": float(100 * gain / mae[j]) if mae[j] > 0 else None,
                             "direction": "A_better" if gain > 1e-12 else "baseline_better" if gain < -1e-12 else "equivalent",
                             "interval_direction": None if ci is None else
                             "A_better" if ci[0] > 1e-12 else "baseline_better" if ci[1] < -1e-12 else "includes_zero"}
        summaries[cap] = {"n_rows": len(rr), "mae": dict(zip(METHODS, map(float, mae))),
                          "cluster_macro_mae": dict(zip(METHODS, map(float, (sums / sizes[:, None]).mean(axis=0)))),
                          "gains_of_A": gains, "clusters": cluster_details}
    return {"n_clusters": n, "cluster_ids": clusters, "n_rows": len(rows),
            "bootstrap_status": "unavailable: one cluster" if n == 1 else
                                "descriptive: fewer than six clusters" if n < 6 else "six independent sampled pools",
            "scores": summaries}


def all_panels(rows):
    panels = {"all": score(rows)}
    panels.update({p: score([r for r in rows if r["panel"] == p]) for p in sorted({r["panel"] for r in rows})})
    by_state = {s: score([r for r in rows if r["state"] == s]) for s in sorted({r["state"] for r in rows})}
    return panels, by_state


def outcome_duplicates(rows):
    signatures = {}
    for state in sorted({r["state"] for r in rows}):
        signature = tuple(sorted((r["x"], r["capability"], r["actual"]) for r in rows if r["state"] == state))
        signatures.setdefault(signature, []).append(state)
    return [states for states in signatures.values() if len(states) > 1]


def build(inputs):
    result = {"schema_version": 1, "experiment": "V76 value of capability conditioning",
              "status": "retrospective development-only ablation on previously frozen confirmation panels",
              "units": "MAE of signed post-intervention minus dense cross entropy, nats",
              "capabilities": list(CAPS), "variants": LABELS,
              "protocol": {
                  "equal_information": "Every variant has the same exact development labels and configuration inputs; zero confirmation-label fitting, calibration, selection, or tuning. No source descriptors.",
                  "A": "V53/V69 median over development states separately for each capability and anchor; V70 OLS a_c log(1+E). Reproduces frozen predictions.",
                  "shared_curve": "At each pruning density or quantization configuration: median over development states of their arithmetic mean signed response over the three capabilities. Same interpolation as A.",
                  "B": "Hold shared curve fixed; fit a separate unrestricted signed scale by unregularized least squares against each capability's development median anchors, equal weight per anchor.",
                  "C": "Hold shared curve fixed; offset is the mean capability-median-anchor residual, equal weight per anchor.",
                  "correction_formulas": "With m_kc the development capability median and h_k the shared anchor: s_c=sum_k(h_k*m_kc)/sum_k(h_k^2); b_c=mean_k(m_kc-h_k). No scale sign constraint, ridge, intercept in B, or slope refit in C.",
                  "D": "Shared curve only; no capability correction.",
                  "distillation": "All 25 trajectories x 4 checkpoints fit jointly with equal checkpoint weight, as in V70. Shared a is OLS on the capability-mean signed response; scales are OLS on the same 100 points. C and D are supplemental; the requested primary contrast is A versus B.",
                  "pruning_interpolation": "Adjacent linear interpolation/boundary-pair extension, no clipping, no synthetic dense training anchor.",
                  "quantization_interpolation": "V69 adjacent interpolation/extrapolation in log2(g/128) at each exact b=3,4,5; apply the V69 zero floor after extrapolating each variant's corrected anchors at g=32/512. Interior remains signed.",
                  "confirmation_weighting": "Equal cells/checkpoints within capability; macro is the equal mean of the three capability MAEs. Quantization states contribute 6,6,9 cells; equal-state sensitivity is also reported.",
                  "distillation_prediction_inputs": "Frozen planned E=T_planned/D_U; actual exposure overshoot does not replace planned predictions.",
                  "sensitivity": "Also report mean of capability medians (aggregation order), corrections fitted to individual development rows, and LAD corrections on anchors or development rows. No test-based choice among these conventions.",
              },
              "bootstrap": {"resamples": N_BOOT, "seed": SEED, "rng": "NumPy PCG64", "interval": "95% paired percentile",
                            "gain_sign": "baseline MAE minus A MAE; positive favors per-capability relations",
                            "unit": "Pruning/quantization: whole source state. Distillation: whole sampled pool, retaining both students and every budget.",
                            "pairing": "Same draws across variants and capabilities within each panel; macro uses the same draws.",
                            "weighting": "Resample clusters, sum their absolute errors and divide by resampled row counts (preserves cell-weighted estimand).",
                            "scope": "Conditional on fixed development fits, probes and training seed. Does not include development-fit or probe-item uncertainty. No interval for one-state panels; 2-5-state intervals are descriptive and cannot establish population generalization. No multiple-comparison adjustment."},
              "arms": {}}
    for name, loader in (("pruning", load_pruning), ("quantization", load_quantization)):
        cells, fits, raw_rows = loader(inputs)
        rows = predict_anchors(raw_rows, fits["primary"], name)
        panels, by_state = all_panels(rows)
        sensitivities = {}
        for label, fit in fits.items():
            if label != "primary":
                sr = predict_anchors(raw_rows, fit, name)
                sensitivities[label] = {"fit": fit, "panels": all_panels(sr)[0]}
        result["arms"][name] = {"n_development_states": len({r["state"] for r in cells}),
            "n_development_cells": len(cells), "n_development_capability_rows": len(cells) * 3,
            "development_states": sorted({r["state"] for r in cells}),
            "n_confirmation_states": len({r["state"] for r in rows}),
            "n_confirmation_cells": len(rows) // 3,
            "fit": fits["primary"], "panels": panels, "by_state": by_state,
            "sensitivity": sensitivities, "rows": rows}
        duplicates = outcome_duplicates(rows)
        result["arms"][name]["identical_recorded_outcome_vectors"] = duplicates
        if duplicates:
            # Audit sensitivity only: preserve the complete requested panel.
            removed = {state for group in duplicates for state in group[1:]}
            unique_rows = [r for r in rows if r['state'] not in removed]
            result["arms"][name]["unique_outcome_sensitivity"] = {
                "removed_states": sorted(removed), "reason": "Exact duplicate signed response vector at every matched configuration/capability; unknown cause. Preserve primary panel, also score one representative per identical outcome vector.",
                "panels": all_panels(unique_rows)[0], "n_confirmation_cells": len(unique_rows) // 3}
    points, model, rows = load_distillation(inputs)
    panels, by_state = all_panels(rows)
    result["arms"]["distillation"] = {"n_development_trajectories": 25, "n_development_checkpoints": len(points),
        "development_trajectories": sorted({p['cluster'] for p in points}),
        "n_confirmation_trajectories": len({r['trajectory'] for r in rows}),
        "n_confirmation_checkpoints": len(rows) // 3, "n_independent_confirmation_pools": 6,
        "fit": model, "panels": panels, "by_state": by_state, "rows": rows,
        "max_prediction_difference_A_B": max(abs(r['predictions']['A'] - r['predictions']['B']) for r in rows),
        "interpretation": "A and B are the same function family when a is nonzero and scales are unrestricted. A cannot demonstrate shape-conditioning value over B; any gain over D tests capability amplitude only."}
    result["conclusions"] = {
        "pruning": "Per-capability curves improve pooled MAE over shared + scale under every reported development fitting convention. QA drives the macro improvement; math/code gains are small and their primary state-bootstrap intervals include zero. Shared-only D has lower pooled math/code MAE than A.",
        "quantization": "The primary gain over shared + scale is small and its state-bootstrap interval includes zero. Its sign changes under the development-row LAD sensitivity; these cells do not establish an advantage for separate capability shapes.",
        "distillation": "No advantage over shared + scale is identifiable: the two requested reuse-count forms are algebraically equivalent.",
        "scope": "These conclusions concern these frozen panels and the stated development-only estimators. Small state counts and retrospective specification limit broader claims.",
        "v72_degenerate_intervals": "The two V72 state labels have exactly identical recorded signed response vectors at all three densities and capabilities, despite distinct weight hashes in freeze.json. Their source-free predictions and paired gains are therefore also identical. The cause is not established. The collapsed V72 intervals do not establish repeatability across distinct measured responses; a sensitivity counting this outcome vector once is reported."}
    result["validation"] = {"development_confirmation_disjoint_at_required_unit": True,
        "all_A_predictions_reproduce_frozen_artifacts": True, "confirmation_capability_rows": {"pruning": 45, "quantization": 63, "distillation": 108},
        "raw_responses_and_freeze_hashes_checked": True,
        "source_freeze_files_read_only": True, "cpu_only": True}
    result["inputs_sha256"] = dict(sorted(inputs.hashes.items()))
    result["outputs"] = {"summary_json": "results/v76-cap-conditioning/summary.json",
        "summary_md": "results/v76-cap-conditioning/summary.md",
        "staged_table": "results/v76-cap-conditioning/paper/paper/tables/cap_conditioning.tex",
        "staged_code_mirror": "results/v76-cap-conditioning/paper/code/analysis/v76_cap_conditioning.py",
        "requested_table_destination": "paper/paper/tables/cap_conditioning.tex",
        "requested_code_destination": "paper/code/analysis/v76_cap_conditioning.py",
        "paper_destination_status": "staged under results to obey explicit write-only scope"}
    return result


def number(value):
    return f"{0.0 if abs(value) < 5e-8 else value:.6f}"


def gain_text(gain):
    ci = gain["ci95_nats"]
    return number(gain["gain_nats"]) + (f" [{number(ci[0])}, {number(ci[1])}]" if ci else " [CI unavailable]")


def markdown(result):
    lines = ["# V76: Value of capability conditioning", "",
             "MAE in nats on the exact frozen confirmation panels. Positive gain is baseline MAE minus per-capability MAE. "
             "V76 is a retrospective ablation; its new shared variants were fitted only to historical development data.", "",
             "## Primary comparison", "",
             "| Arm | Capability | A: per-cap | B: shared + scale | C: shared + offset | D: shared | B − A [95% CI] | D − A [95% CI] |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm, data in result["arms"].items():
        for c in (*CAPS, "macro"):
            s = data["panels"]["all"]["scores"][c]
            lines.append("| " + " | ".join([arm, c, *[number(s["mae"][m]) for m in METHODS],
                                           gain_text(s["gains_of_A"]["B"]), gain_text(s["gains_of_A"]["D"])]) + " |")
    lines += ["", "## Interpretation", ""]
    for arm in ("pruning", "quantization"):
        s = result["arms"][arm]["panels"]["all"]["scores"]["macro"]
        gain = s["gains_of_A"]["B"]
        verb = "reduces" if gain["gain_nats"] > 0 else "increases"
        lines.append(f"- {arm.capitalize()}: A {verb} macro MAE relative to B by {abs(gain['gain_nats']):.6f} nats "
                     f"({abs(gain['relative_gain_percent']):.2f}%); signed gain {gain_text(gain)}. "
                     f"The panel has {result['arms'][arm]['panels']['all']['n_clusters']} state clusters, so its interval is descriptive.")
    lines += ["- Distillation: A and B are algebraically equivalent, with zero gain up to floating-point rounding. "
              "The comparison cannot distinguish per-capability shapes from constant capability scales when every curve is a single coefficient times log(1+E).", "",
              result["conclusions"]["pruning"], "", result["conclusions"]["quantization"], "",
              "## Data and fitting", "",
              "- Pruning: exact V53 register, 17 development states / 84 state-density cells / 252 capability rows. "
              "Confirmation: 410M@48k, 1.4B@112k, 6.9B@80k at d=0.85,0.675,0.575; "
              "2.8B@16k and @143k at d=0.85,0.75,0.65 (15 cells total).",
              "- Grouped quantization: 6 development states × 9 configurations = 54 cells; 21 frozen V69 confirmation cells. "
              "Two confirmation states appeared at different development configurations; the new 1.4B@112k state did not. "
              "All 21 cells are scored, with 6,6,9 cells per state.",
              "- Distillation: 25 development trajectories × 4 checkpoints; 12 confirmation trajectories × 3 budgets. "
              "Two students share six sampled pools; six pools are the independent bootstrap units. "
              "The development fits pool all students, including the four 4B trajectories.", ""]
    lines += [f"- **{key}**: {value}" for key, value in result["protocol"].items()]
    lines += ["", "Full anchors, correction constants, predictions, per-state MAEs, and input SHA256 hashes are in summary.json. "
              "B and C use the same full development information as A; parameter counts and label aggregation differ.", "",
              "## Frozen subpanels", "",
              "| Arm / panel | Clusters | Capability | A MAE | B MAE | C MAE | D MAE | B − A [95% CI] |",
              "|---|---:|---|---:|---:|---:|---:|---:|"]
    for arm, data in result["arms"].items():
        for label, panel in data["panels"].items():
            if label == "all":
                continue
            for c in (*CAPS, "macro"):
                s = panel["scores"][c]
                lines.append("| " + " | ".join([arm + " / " + label, str(panel["n_clusters"]), c,
                    *[number(s["mae"][m]) for m in METHODS], gain_text(s["gains_of_A"]["B"])]) + " |")
    lines += ["", "## Sensitivity to development fitting convention", "",
              "These are fixed alternative estimators, not selected using confirmation MAE. All reproduce the same frozen A predictions.", "",
              "| Arm | Convention | Math B − A | Code B − A | QA B − A | Macro B − A [95% CI] |",
              "|---|---|---:|---:|---:|---:|"]
    for arm in ("pruning", "quantization"):
        data = result["arms"][arm]
        for label, panel in [("primary", data['panels']['all'])] + [(k, v['panels']['all']) for k, v in data['sensitivity'].items()]:
            scores = panel["scores"]
            lines.append("| " + " | ".join([arm, label,
                *[number(scores[c]['gains_of_A']['B']['gain_nats']) for c in CAPS],
                gain_text(scores['macro']['gains_of_A']['B'])]) + " |")
    unique = result["arms"]["pruning"].get("unique_outcome_sensitivity")
    if unique:
        lines += ["", "## Repeated V72 outcome vector", "", result["conclusions"]["v72_degenerate_intervals"], "",
                  f"Sensitivity removes {', '.join(unique['removed_states'])}; the other V72 state remains. "
                  f"This leaves {unique['n_confirmation_cells']} cells across {unique['panels']['all']['n_clusters']} states. "
                  "The primary analysis above retains both requested state labels.", "",
                  "| Capability | A MAE | B MAE | B − A [95% CI] |", "|---|---:|---:|---:|"]
        for cap in (*CAPS, "macro"):
            s = unique['panels']['all']['scores'][cap]
            lines.append("| " + " | ".join([cap, number(s['mae']['A']), number(s['mae']['B']),
                                            gain_text(s['gains_of_A']['B'])]) + " |")
    q = result["arms"]["quantization"]["panels"]["all"]["scores"]["macro"]["cluster_macro_mae"]
    lines += ["", f"Quantization equal-state macro MAEs (instead of equal-cell): " + ", ".join(f"{m}={number(q[m])}" for m in METHODS) + ".",
              "", "## Distillation equivalence", "", result['arms']['distillation']['fit']['equivalence'], "",
              f"Maximum absolute A/B confirmation prediction difference: {result['arms']['distillation']['max_prediction_difference_A_B']:.3g} nats. "
              "B has no additional expressive restriction here; a nonzero shared amplitude and three signed scales reparameterize the three capability amplitudes.",
              "", "## Uncertainty and validation", ""]
    lines += [f"- {key}: {value}" for key, value in result['bootstrap'].items()]
    lines += ["- " + result["conclusions"]["v72_degenerate_intervals"],
              "- Verified exact development membership, confirmation membership, disjointness at the relevant unit, raw signed responses, "
              "historical hashes and frozen prediction chains. Independently reconstructed every A prediction across all 216 capability rows.",
              "- Checks use only NumPy and Python standard-library operations on CPU; no models are loaded.", "",
              "## Outputs", "",
              "Paper outputs are staged to respect the requested write-only scope:", "",
              "- `paper/paper/tables/cap_conditioning.tex` under this results directory (uses `[H]`).",
              "- `paper/code/analysis/v76_cap_conditioning.py` under this results directory (byte-identical code mirror).",
              "- Reproduce: `python -B analysis/v76_cap_conditioning.py`; verify without writes: append `--check`.", ""]
    return "\n".join(lines)


def latex(result):
    lines = [r"% Generated by analysis/v76_cap_conditioning.py; development-only fits.",
             r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Value of capability conditioning on frozen confirmation panels (MAE, nats). "
             r"A: per-capability response; B: shared response with a fixed capability scale; "
             r"C: shared response with a fixed capability offset; D: shared response only. "
             r"All use the same development information. Positive $\Delta_B=B-A$ favors A; "
             r"brackets give 95\% paired cluster-bootstrap intervals.}",
             r"\label{tab:cap_conditioning}", r"\begin{tabular}{llrrrrl}", r"\toprule",
             r"Arm & Capability & A & B & C & D & $\Delta_B$ [95\% CI] \\", r"\midrule"]
    names = {"pruning": "Pruning", "quantization": "Grouped quant.", "distillation": "Distillation"}
    for arm, data in result["arms"].items():
        for c in (*CAPS, "macro"):
            s = data["panels"]["all"]["scores"][c]
            g = s["gains_of_A"]["B"]
            ci = g["ci95_nats"]
            fmt = lambda x: f"{0.0 if abs(x) < .00005 else x:.4f}"
            interval = f"{fmt(g['gain_nats'])} [{fmt(ci[0])}, {fmt(ci[1])}]"
            lines.append(" & ".join([names[arm] if c == "math" else "", c,
                                     *[fmt(s['mae'][m]) for m in METHODS], interval]) + r" \\")
        if arm != "distillation":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              r"\begin{minipage}{0.99\linewidth}\footnotesize "
              r"Development: pruning 17 states (84 cells); quantization 54 cells; distillation 25 trajectories. "
              r"Confirmation: pruning 15 cells in 5 states; quantization 21 cells in 3 states; "
              r"distillation 12 trajectories sharing 6 pools. All capabilities receive equal macro weight. "
              r"Shared anchors are medians of statewise capability means; scales/offsets minimize development "
              r"median-anchor squared error. V53 interpolation and V69 boundary flooring are retained. "
              r"Distillation uses development OLS and frozen planned reuse counts. "
              r"Bootstrap: 5,000 resamples, seed 0, whole states or shared pools with all budgets retained; "
              r"fixed development fits. State intervals are descriptive because clusters are few. "
              r"The two V72 state labels record identical response vectors; counting that vector once "
              r"is reported as a sensitivity in the accompanying summary. "
              r"Distillation A and B are algebraically identical: $a_c=a s_c$ for nonzero $a$. "
              r"This retrospective ablation does not constitute a new preregistration.\end{minipage}",
              r"\end{table}", ""]
    return "\n".join(lines)


def selftest():
    # Known affine interpolation, extrapolation and V69 boundary floor.
    close(linear({0.6: 2., 0.8: 1.}, 0.7), 1.5, "Interior interpolation")
    close(linear({0.6: 2., 0.8: 1.}, 0.9), .5, "Boundary extension")
    anchors = {f"b{b}_g{g}": .1 + math.log2(g / 64) for b in (3, 4, 5) for g in (64, 128, 256)}
    close(quant_curve(anchors, "b4_g32"), 0., "Negative boundary floor")
    close(quant_curve(anchors, "b4_g512"), 3.1, "Positive boundary extrapolation")
    # Distinct aggregation orders, signed scales, and independently known OLS.
    cells = [{"state": str(i), "x": "0.7", "y": y} for i, y in enumerate([[0, 100, 0], [10, 0, 0], [100, 10, 0]])]
    close(fit_anchors(cells)["shared_anchors"]["0.7"], 100 / 3, "Median of means")
    close(fit_anchors(cells, shared_order="mean_of_medians")["shared_anchors"]["0.7"], 20 / 3, "Mean of medians")
    x = np.array([1., 2., 3.])
    y = x[:, None] * np.array([2., -3., 4.])
    scale, _ = corrections(x, y)
    close(list(scale.values()), [2., -3., 4.], "Signed OLS scales")
    points = [{"E": e, "delta": dict(zip(CAPS, np.log1p(e) * np.array([2., -3., 4.]))) } for e in (.5, 1., 3.)]
    model = fit_reuse(points)
    close(list(model['coefficients'].values()), [2., -3., 4.], "OLS reuse slopes")
    # Small exact support: unequal cluster sizes must keep the cell estimand;
    # capabilities share draws, single-cluster panels have no pretend interval.
    rows = [{"state": s, "cluster": s, "x": str(i), "capability": c,
             "actual": 0., "predictions": {"A": 0., "B": v, "C": v, "D": v}}
            for s, count, v in (("s0", 1, 1.), ("s1", 3, 3.)) for i in range(count) for c in CAPS]
    scores = score(rows)
    close(scores['scores']['math']['mae']['B'], 2.5, "Cell weighting")
    close(scores['scores']['math']['gains_of_A']['B']['ci95_nats'], [1., 3.], "Whole cluster bootstrap support")
    require(scores['scores']['math']['gains_of_A'] == scores['scores']['macro']['gains_of_A'], "Pairing across capabilities")
    require(score(rows[:3])['scores']['math']['gains_of_A']['B']['ci95_nats'] is None, "Single-cluster CI unavailable")
    try:
        score(rows + rows[:1])
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate confirmation rows accepted")
    print("PASS: interpolation/flooring, aggregation order, signed corrections, reuse equivalence, unequal-size paired cluster bootstrap, one-cluster guard, duplicate guard")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--check", action="store_true", help="Recompute and verify exact output bytes without writes")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    inputs = Inputs()
    script = inputs.read("analysis/v76_cap_conditioning.py")
    result = build(inputs)
    outputs = {OUT / "summary.json": (json.dumps(result, indent=2, allow_nan=False) + "\n").encode(),
               OUT / "summary.md": markdown(result).encode(),
               OUT / "paper/paper/tables/cap_conditioning.tex": latex(result).encode(),
               OUT / "paper/code/analysis/v76_cap_conditioning.py": script}
    inputs.verify()
    for path, raw in outputs.items():
        require(path.resolve().is_relative_to(OUT.resolve()) and not path.is_symlink(), f"Output scope violation: {path}")
        if args.check:
            require(path.read_bytes() == raw, f"Output differs from deterministic recomputation: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    inputs.verify()
    for arm, data in result["arms"].items():
        s = data["panels"]["all"]["scores"]["macro"]
        print(f"{arm}: MAE={s['mae']}; B-A={gain_text(s['gains_of_A']['B'])}")
    print(f"{'CHECKED' if args.check else 'WROTE'} four artifacts; validated 216 frozen capability rows; inputs unchanged; CPU only")


if __name__ == "__main__":
    main()
