#!/usr/bin/env python3
"""Deterministic CPU/NumPy shared-structure audit; exclusively creates new outputs.

Run: python -B analysis/v59_shared_structure.py
Optional --out-dir and --table permit independent reproducibility runs.
No torch, model construction, network, response clipping, or test-set selection.
The historical registers fix membership, never supply reported fitted scores.
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
import hashlib
import json
import os
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAPS = ("math", "code", "qa")
SEED = 59053
BOOT_DRAWS = 1000
FILES = {}
CONFIRM = ("pythia-410m@step48000", "pythia-1.4b@step112000", "pythia-6.9b@step80000")
PAIRS = tuple(f"pythia-{size}@step{step}" for size in ("1b", "6.9b") for step in (32000, 112000))


def record(path, role="data"):
    path = Path(path).absolute()
    raw = path.read_bytes()
    key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    sha = hashlib.sha256(raw).hexdigest()
    if key in FILES and FILES[key]["sha256"] != sha:
        raise ValueError(f"Input changed: {key}")
    item = FILES.setdefault(key, {"path": key, "sha256": sha, "bytes": len(raw), "roles": []})
    if role not in item["roles"]:
        item["roles"].append(role)
    return raw


def read_json(path, role="data"):
    return json.loads(record(path, role))


def reference_module(name):
    path = ROOT / "analysis" / f"{name}.py"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    # Compile the hashed source directly: no bytecode files and no module main().
    exec(compile(record(path, "reference_code"), str(path), "exec"), module.__dict__)
    return module


def metrics(rows, methods):
    out = {}
    for method in methods:
        out[method] = {}
        for cap in (*CAPS, "macro"):
            rr = [r for r in rows if cap == "macro" or r["cap"] == cap]
            e = np.asarray([r["predictions"][method] - r["y"] for r in rr])
            out[method][cap] = {"n": len(e), "mae": float(np.mean(abs(e))) if len(e) else None,
                                "bias": float(np.mean(e)) if len(e) else None}
    return out


def scored(rows, methods):
    return {"metrics": metrics(rows, methods), "predictions": rows,
            "by_source": {s: metrics([r for r in rows if r["source"] == s], methods)
                          for s in sorted({r["source"] for r in rows})}}


def compare(rows, shared, specific):
    mm = metrics(rows, (shared, specific))
    result = {}
    for cap in (*CAPS, "macro"):
        a, b = (mm[m][cap]["mae"] for m in (shared, specific))
        gap = a - b
        result[cap] = {"shared_mae": a, "specific_mae": b, "shared_minus_specific": gap,
                       "classification": "within 0.02" if abs(gap) <= .02 else
                       "worse" if gap > .02 else "better", "n": mm[shared][cap]["n"]}
    return result


def pruning_rows(p53, tag, densities, expected=None, confirmation=False):
    path = p53.DEV_ROOT / tag.replace("@", "--") / "prune_losses.json"
    table = read_json(path, "pruning_measurements")
    dense, measured = p53.parse_losses(table, path)
    size, step = p53.parse_tag(tag)
    n0, d0 = p53.matrix_n0(size), step * p53.TOKENS_PER_STEP
    inputs = dense
    if confirmation:
        inputs = p53.checked_losses(read_json(p53.OUT / f"dense_{tag}.json", "frozen_dense_input"), tag)
        assert inputs == dense, "Confirmation dense references differ"
    if expected:
        assert n0 == expected["N0"] and d0 == expected["D0"] and dense == expected["L0"]
    return [{"source": tag, "cap": c, "d": d, "y": measured[d][c] - dense[c],
             "phi_raw": p53.raw_features(n0, d0, inputs[c])} for d in densities for c in CAPS]


def power_grid(p53, rows, stats, cap, state_amplitude=False):
    rr = [r for r in rows if r["cap"] == cap]
    z = np.ones((len(rr), 1)) if state_amplitude else p53.standardize([r["phi_raw"] for r in rr], stats)
    y = np.asarray([r["y"] for r in rr])
    sh = p53.shape(np.asarray([r["d"] for r in rr])[None, :], p53.GAMMA_GRID[:, None])
    x = sh[:, :, None] * z[None, :, :]
    xt = x.transpose(0, 2, 1)
    beta = np.linalg.solve(xt @ x + p53.RIDGE * np.eye(z.shape[1]), (xt @ y)[..., None])[..., 0]
    sse = np.sum(((x @ beta[..., None])[..., 0] - y) ** 2, axis=1)
    return beta, sse


def power_models(p53, rows, stats, shared=False):
    grids = {c: power_grid(p53, rows, stats, c) for c in CAPS}
    joint = int(np.argmin(sum(grids[c][1] for c in CAPS))) if shared else None
    return {c: {"beta": grids[c][0][k].tolist(), "gamma": float(p53.GAMMA_GRID[k]),
                "sse": float(grids[c][1][k])}
            for c in CAPS for k in [joint if shared else int(np.argmin(grids[c][1]))]}


def calibrate_pruning(p53, models, stats, held):
    result, calibration = [], []
    for source in sorted({r["source"] for r in held}):
        sr = [r for r in held if r["source"] == source]
        mildest = max(r["d"] for r in sr)
        anchors = {r["cap"]: r for r in sr if r["d"] == mildest}
        amplitudes = {c: anchors[c]["y"] / p53.shape(mildest, models[c]["power"]["gamma"]) for c in CAPS}
        calibration.append({"source": source, "density": mildest, "compressed_measurements": 1,
                            "amplitudes": amplitudes, "observed": {c: anchors[c]["y"] for c in CAPS}})
        for row in sr:
            if row["d"] == mildest:
                continue
            cap = row["cap"]
            pr = p53.predict_all(models[cap], p53.standardize(row["phi_raw"], stats), row["d"])
            result.append({**row, "calibration_density": mildest, "predictions": {
                "K0": pr["power"], "K1": float(amplitudes[cap] * p53.shape(row["d"], models[cap]["power"]["gamma"])),
                "median_curve": pr["median_curve"], "strength_only": pr["strength_only"]}})
    return result, calibration


def pruning_cv(p53, rows, sharing=True):
    predictions, remaining, calibrations, folds = [], [], [], []
    for source in sorted({r["source"] for r in rows}):
        train = [r for r in rows if r["source"] != source]
        held = [r for r in rows if r["source"] == source]
        stats, models = p53.fit_all(train)
        shared = power_models(p53, train, stats, shared=True) if sharing else None
        for row in held:
            cap = row["cap"]
            z = p53.standardize(row["phi_raw"], stats)
            pr = p53.predict_all(models[cap], z, row["d"])
            # Same power evaluator and fitting path for shared-family and delivered-power entries.
            pr["shared_family"] = pr["power"]
            if shared:
                pr["shared_gamma"] = float(np.dot(shared[cap]["beta"], z) * p53.shape(row["d"], shared[cap]["gamma"]))
            predictions.append({**row, "predictions": pr})
        rr, cc = calibrate_pruning(p53, models, stats, held)
        remaining.extend(rr)
        calibrations.extend(cc)
        folds.append({"held_out": source, "train_sources": sorted({r["source"] for r in train}),
                      "standardization": stats, "models": models, "shared_gamma_models": shared})
    return {"all_rows": predictions, "remaining": remaining, "calibrations": calibrations, "folds": folds}


def intervals(residual_rows):
    out = {}
    for method in ("K0", "K1"):
        out[method] = {}
        for cap in CAPS:
            rr = [r for r in residual_rows if r["cap"] == cap]
            errors = np.asarray([abs(r["predictions"][method] - r["y"]) for r in rr])
            out[method][cap] = {"n_residuals": len(errors), "sources": sorted({r["source"] for r in rr}),
                               "absolute_errors": errors.tolist(),
                               "radii": {str(level): float(np.quantile(errors, level, method="higher"))
                                         for level in (.8, .95)}}
    return out


def interval_evaluation(rows, reference):
    out = {}
    for method in ("K0", "K1"):
        out[method] = {}
        for cap in (*CAPS, "macro"):
            rr = [r for r in rows if cap == "macro" or r["cap"] == cap]
            levels = {}
            for level in (.8, .95):
                radii = np.asarray([reference[method][r["cap"]]["radii"][str(level)] for r in rr])
                errors = np.asarray([abs(r["predictions"][method] - r["y"]) for r in rr])
                levels[str(level)] = {"n": len(rr), "covered": int(np.sum(errors <= radii)),
                                      "coverage": float(np.mean(errors <= radii)), "mean_width": float(np.mean(2 * radii))}
            out[method][cap] = levels
    return out


def bootstrap_pruning(p53, rows, stats, models):
    rng = np.random.default_rng(SEED)
    sources = sorted({r["source"] for r in rows})
    samples = {c: [] for c in CAPS}
    draws = []
    for draw in range(BOOT_DRAWS):
        indices = rng.integers(0, len(sources), len(sources))
        draws.append(indices.tolist())
        sampled = [r for i in indices for r in rows if r["source"] == sources[i]]
        bs = p53.zstats(sampled)
        fitted = power_models(p53, sampled, bs)
        for cap in CAPS:
            beta = np.asarray(fitted[cap]["beta"])
            # Map each refitted standardizer to full-dev phi before percentile CIs.
            slopes = beta[1:] * np.asarray(stats["scale"]) / bs["scale"]
            intercept = beta[0] + np.dot(beta[1:], (np.asarray(stats["center"]) - bs["center"]) / bs["scale"])
            samples[cap].append([float(intercept), *slopes.tolist(), fitted[cap]["gamma"]])
        if (draw + 1) % 250 == 0:
            print(f"Bootstrap {draw + 1}/{BOOT_DRAWS}", flush=True)
    names = ("beta_intercept", "beta_logN0", "beta_L0c", "beta_logD0", "gamma")
    result = {}
    for cap in CAPS:
        arr = np.asarray(samples[cap])
        point = [*models[cap]["power"]["beta"], models[cap]["power"]["gamma"]]
        result[cap] = {name: {"estimate": point[i], "ci95": np.quantile(arr[:, i], [.025, .975]).tolist(),
                              "bootstrap_median": float(np.median(arr[:, i]))} for i, name in enumerate(names)}
        result[cap]["gamma_boundary_fraction"] = float(np.mean((arr[:, -1] == p53.GAMMA_GRID[0]) | (arr[:, -1] == p53.GAMMA_GRID[-1])))
    return {"draws": BOOT_DRAWS, "seed": SEED, "cluster": "dev source; all densities/capabilities together",
            "method": "95% percentile CI; refit standardizer, beta and gamma each draw; map beta to full-dev phi",
            "parameter_basis": stats, "parameters": result, "samples": samples,
            "sample_parameter_order": names, "source_order": sources, "draw_source_indices": draws}


def quant_rows(p55, tag, configs, expected=None):
    path = p55.measurement_path(p55.DATA_ROOT, tag)
    table = read_json(path, "grouped_quantization_measurements")
    dense = p55.checked_losses(table["dense"], path)
    state = p55.state_input(tag, dense, path)
    if expected:
        assert all(state[k] == expected[k] for k in ("N0", "D0", "L0"))
    return [{"source": tag, "state": tag, "cap": c, "capability": c, "config": cfg,
             "y": p55.checked_losses(table[cfg], path)[c] - dense[c],
             "dL": table[cfg][c] - dense[c], "phi_raw": p55.raw_features(state, c)}
            for cfg in configs for c in CAPS]


def quant_shared_design(p55, rows, stats):
    z = p55.phi([r["phi_raw"] for r in rows], stats)
    uv = np.asarray([p55.coordinates(r["config"]) for r in rows])
    terms = p55.low_order_terms(uv[:, 0], uv[:, 1], stats["u_center"])
    # Three capability-specific 4-vector offsets + four shared 4-vector shape terms.
    x = np.zeros((len(rows), 28))
    for i, row in enumerate(rows):
        k = CAPS.index(row["cap"])
        x[i, k * 4:k * 4 + 4] = z[i]
    x[:, 12:] = (terms[:, 1:, None] * z[:, None, :]).reshape(len(rows), 16)
    return x


def quant_calibrate(p55, models, stats, rows, anchors):
    out, calibration = [], []
    for source in sorted({r["source"] for r in rows}):
        sr = [r for r in rows if r["source"] == source]
        ar = {r["cap"]: r for r in anchors if r["source"] == source}
        assert set(ar) == set(CAPS)
        cal = {"source": source, "config": ar["math"]["config"], "compressed_measurements": 1, "per_capability": {}}
        for cap in CAPS:
            anchor = ar[cap]
            ap = p55.predict_all(models[cap], p55.phi(anchor["phi_raw"], stats), anchor["config"], stats)
            # Exact amplitude calibration of the separable law; signed amplitudes allowed.
            sep = models[cap]["separable"]
            u0, v0 = p55.coordinates(anchor["config"])
            amplitude = anchor["y"] / np.exp2(-sep["p"] * u0 + sep["q"] * v0)
            denominator = ap["low_order_2d"]
            if abs(denominator) <= 1e-12:
                raise ValueError(f"Undefined 2D amplitude calibration: {source}/{cap}")
            ratio = anchor["y"] / denominator
            cal["per_capability"][cap] = {"observed": anchor["y"], "separable_amplitude": float(amplitude),
                                          "low2d_predicted_anchor": denominator, "low2d_multiplier": ratio}
            for row in sr:
                if row["cap"] != cap or row["config"] == anchor["config"]:
                    continue
                pr = p55.predict_all(models[cap], p55.phi(row["phi_raw"], stats), row["config"], stats)
                u, v = p55.coordinates(row["config"])
                out.append({**row, "calibration_config": anchor["config"], "predictions": {
                    "K0": pr["low_order_2d"], "K1": ratio * pr["low_order_2d"],
                    "separable_K0": pr["separable"], "separable_K1": float(amplitude * np.exp2(-sep["p"] * u + sep["q"] * v)),
                    "median_curve": pr["median"], "mean_curve": pr["mean"]}})
        calibration.append(cal)
    return out, calibration


def distill_phi_rows(points, counts):
    # D0 is unavailable: the log-D0 slot is constant zero, hence its z value is zero.
    return [{"source": p["cluster"], "cap": c, "y": p["delta"][c], "E": p["E"], "Tc": p["Tc"],
             "file": p["file"], "phi_raw": [float(np.log(counts[p["student"]])), p["L0"][c], 0.0]}
            for p in points for c in CAPS]


def distill_phi_fit(p53, d56, rows, kind):
    stats = p53.zstats(rows)
    candidates = []
    for ts in d56.T_GRID if kind == "F2" else (None,):
        fits = {}
        for cap in CAPS:
            rr = [r for r in rows if r["cap"] == cap]
            z = p53.standardize([r["phi_raw"] for r in rr], stats)
            e = np.log1p([r["E"] for r in rr])[:, None]
            x = z * e
            if ts is not None:
                sat = -np.expm1(-np.asarray([r["Tc"] for r in rr]) / ts)[:, None]
                x = np.concatenate((z * sat, x), axis=1)
            y = np.asarray([r["y"] for r in rr])
            beta = p53.ridge_fit(x, y)
            fits[cap] = {"beta": beta.tolist(), "sse": float(np.sum((x @ beta - y) ** 2)),
                         "design_rank": int(np.linalg.matrix_rank(x))}
        candidates.append({"T_star": ts, "fits": fits, "sse": sum(f["sse"] for f in fits.values())})
    return {"standardization": stats, **min(candidates, key=lambda f: f["sse"])}


def distill_phi_predict(p53, model, row):
    z = p53.standardize(row["phi_raw"], model["standardization"])
    x = z * np.log1p(row["E"])
    if model["T_star"] is not None:
        x = np.r_[z * -np.expm1(-row["Tc"] / model["T_star"]), x]
    return float(x @ model["fits"][row["cap"]]["beta"])


def distillation(p53, d56, register):
    d56.record_file = record
    d56.reg = read_json(d56.REG, "completion_token_register")
    points = [p for student in d56.DEV_STUDENTS for u in (75, 450) for seed in (11, 12, 13)
              for p in d56.points(student, u, seed, "p2v2")]
    assert points == register["data"]["dev_points"]
    assert len(points) == 48 and len({p["cluster"] for p in points}) == 12
    counts = register["refs"]["N"]
    refs = d56.descriptor_refs(points, counts)
    rows = distill_phi_rows(points, counts)
    predictions, folds = [], []
    for source in sorted({r["source"] for r in rows}):
        train = [r for r in rows if r["source"] != source]
        held = [r for r in rows if r["source"] == source]
        shared = distill_phi_fit(p53, d56, train, "F1")
        matched = distill_phi_fit(p53, d56, train, "F2")
        trp = [p for p in points if p["cluster"] != source]
        hp = [p for p in points if p["cluster"] == source]
        native_models, native_predictions = {}, {}
        for kind, descriptor in d56.FORMS:
            key = d56.form_key(kind, descriptor)
            model = d56.fit_a(trp, refs, kind, descriptor)
            native_models[key] = model
            pred = d56.predict_a(model, hp, refs)
            native_predictions[key] = {(p["file"], cap): float(pred[i, j]) for i, p in enumerate(hp) for j, cap in enumerate(CAPS)}
        for row in held:
            pr = {key: values[(row["file"], row["cap"])] for key, values in native_predictions.items()}
            pr.update(shared_family=distill_phi_predict(p53, shared, row), F2_matched_phi=distill_phi_predict(p53, matched, row))
            predictions.append({**row, "predictions": pr})
        folds.append({"held_out": source, "train_sources": sorted({r["source"] for r in train}),
                      "shared_model": shared, "matched_F2": matched, "delivered_models": native_models})
    native_keys = [d56.form_key(*f) for f in d56.FORMS]
    return {"predictions": predictions, "folds": folds, "metrics": metrics(predictions, ["shared_family", "F2_matched_phi", *native_keys]),
            "full_dev_models": {"shared_family": distill_phi_fit(p53, d56, rows, "F1"),
                                "F2_matched_phi": distill_phi_fit(p53, d56, rows, "F2"),
                                "delivered_forms": {d56.form_key(*f): d56.fit_a(points, refs, *f) for f in d56.FORMS}},
            "comparison": compare(predictions, "shared_family", "F2:L0"),
            "same_phi_comparison": compare(predictions, "shared_family", "F2_matched_phi"),
            "native_refs": refs, "points": points,
            "D0_limitation": "Pretraining D0 unavailable: log-D0 column held constant zero, standardized to zero; coefficient unidentifiable. N0 uses v56 recorded meta parameter counts, not Pythia matrix counts.",
            "native_comparator": "F2:L0; all ten v56 forms refitted. Native v56 descriptor/column scaling preserved; matched-phi F2 also supplied to isolate strength-form changes."}


def run_analysis(p53, p55, d56, r53, r55, r56):
    assert r53["n_dev_states"] == 17 and r53["selected_candidate"] == "power"
    assert len(r55["dev_states"]) == 6 and r55["dev_configs"] == list(p55.DEV_CONFIGS)
    rows = [r for s in r53["dev_states"] for r in pruning_rows(p53, s["tag"], s["densities"], s)]
    assert len(rows) == r53["n_dev_rows"]
    print("Pruning: refitting the registered 17-source LOSO panel.", flush=True)
    pcv = pruning_cv(p53, rows)
    stats, models = p53.fit_all(rows)
    # Historical numbers are used only as assertions, never as reported results.
    reproduction = {}
    table = p53.summarize_loso(pcv["all_rows"])
    delta = max(abs(a[f"{c}_mae"] - b[f"{c}_mae"]) for a, b in zip(table, r53["loso_table"]) for c in CAPS)
    assert delta < 1e-9
    reproduction["v53_max_loso_mae_difference"] = delta
    fitted = power_models(p53, rows, stats)
    assert all(np.allclose(fitted[c]["beta"], models[c]["power"]["beta"], atol=1e-9, rtol=1e-9)
               and fitted[c]["gamma"] == models[c]["power"]["gamma"] for c in CAPS)
    per_state = {c: [] for c in CAPS}
    for source in sorted({r["source"] for r in rows}):
        sr = [r for r in rows if r["source"] == source]
        for cap in CAPS:
            beta, sse = power_grid(p53, sr, stats, cap, state_amplitude=True)
            best = int(np.argmin(sse))
            per_state[cap].append({"source": source, "gamma": float(p53.GAMMA_GRID[best]),
                                   "amplitude": float(beta[best, 0]), "sse": float(sse[best]), "n": len(sr) // 3})
    distribution = {c: {"min": min(r["gamma"] for r in per_state[c]),
                         "median": float(np.median([r["gamma"] for r in per_state[c]])),
                         "max": max(r["gamma"] for r in per_state[c]),
                         "n_at_grid_boundary": sum(r["gamma"] in (p53.GAMMA_GRID[0], p53.GAMMA_GRID[-1]) for r in per_state[c])}
                    for c in CAPS}
    ci = bootstrap_pruning(p53, rows, stats, models)
    pm = ("K0", "K1", "median_curve", "strength_only")
    dev = scored(pcv["remaining"], pm)
    dev["calibration"] = pcv["calibrations"]
    pi_full = intervals(pcv["remaining"])
    conf_rows = [r for tag in CONFIRM for r in pruning_rows(p53, tag, p53.TEST_D, confirmation=True)]
    assert not set(CONFIRM) & {r["source"] for r in rows}
    cr, cc = calibrate_pruning(p53, models, stats, conf_rows)
    confirmation = {**scored(cr, pm), "calibration": cc, "intervals": interval_evaluation(cr, pi_full),
                    "train_sources": sorted({r["source"] for r in rows}), "interval_reference": "17_source_loso"}
    # The pair sources ARE in v53 dev. Preserve their ordinary OOF result, then
    # independently rebuild fits and the residual bank without any pair sources.
    pair_oof = scored([r for r in pcv["remaining"] if r["source"] in PAIRS], pm)
    pair_oof["calibration"] = [c for c in pcv["calibrations"] if c["source"] in PAIRS]
    pair_train = [r for r in rows if r["source"] not in PAIRS]
    pair_held = [r for r in rows if r["source"] in PAIRS]
    print("Pruning pairs: rebuilding LOSO residuals on 13 sources, excluding all four pairs.", flush=True)
    pair_cv = pruning_cv(p53, pair_train, sharing=False)
    pi_pair = intervals(pair_cv["remaining"])
    pst, pmod = p53.fit_all(pair_train)
    pr, cal = calibrate_pruning(p53, pmod, pst, pair_held)
    pairs = {**scored(pr, pm), "calibration": cal, "intervals": interval_evaluation(pr, pi_pair),
             "train_sources": sorted({r["source"] for r in pair_train}), "standardization": pst, "models": pmod,
             "interval_reference": "13_source_loso_excluding_all_pairs"}
    for reference, targets in ((pi_full, CONFIRM), (pi_pair, PAIRS)):
        for m in ("K0", "K1"):
            for c in CAPS:
                assert not set(reference[m][c]["sources"]) & set(targets)
    # A calibration response must never be included in its source's scored cells.
    for rr in (pcv["remaining"], cr, pr):
        assert all(r["d"] < r["calibration_density"] for r in rr)
    print("Quantization: six state folds, capability sharing, and one-cell calibration.", flush=True)
    qrows = [r for s in r55["dev_states"] for r in quant_rows(p55, s["tag"], p55.DEV_CONFIGS, s)]
    assert [{k: r[k] for k in ("state", "config", "capability", "phi_raw", "dL")} for r in qrows] == r55["dev_rows"]
    qp, qfolds, qremain, qcal = [], [], [], []
    qtests = {name: {"rows": [], "calibration": [], "folds": []} for name in r55["test_sets"]}
    for state in r55["dev_states"]:
        tag = state["tag"]
        train = [r for r in qrows if r["source"] != tag]
        held = [r for r in qrows if r["source"] == tag]
        qs, qm = p55.fit_all(train)
        x = quant_shared_design(p55, train, qs)
        shared_beta = p55.ridge_fit(x, np.asarray([r["y"] for r in train]))
        shared_pred = quant_shared_design(p55, held, qs) @ shared_beta
        for i, row in enumerate(held):
            pred = p55.predict_all(qm[row["cap"]], p55.phi(row["phi_raw"], qs), row["config"], qs)
            pred["shared_terms"] = float(shared_pred[i])
            qp.append({**row, "predictions": pred})
        anchors = [r for r in held if r["config"] == "b5_g256"]
        rr, ca = quant_calibrate(p55, qm, qs, held, anchors)
        qremain.extend(rr)
        qcal.extend(ca)
        qfolds.append({"held_out": tag, "train_sources": sorted({r["source"] for r in train}),
                       "standardization": qs, "models": qm, "shared_coefficients": shared_beta.tolist(),
                       "shared_design_rank": int(np.linalg.matrix_rank(x))})
        for name, spec in r55["test_sets"].items():
            if tag not in spec["states"]:
                continue
            held_test = quant_rows(p55, tag, spec["configs"], state)
            rr, ca = quant_calibrate(p55, qm, qs, held_test, anchors)
            qtests[name]["rows"].extend(rr)
            qtests[name]["calibration"].extend(ca)
            qtests[name]["folds"].append({"held_out_source": tag, "train_sources": sorted({r["source"] for r in train})})
    qmethods = ("K0", "K1", "separable_K0", "separable_K1", "median_curve", "mean_curve")
    qs, qm = p55.fit_all(qrows)
    jt = r55["test_sets"]["joint_test"]
    for tag in jt["states"]:
        jr = quant_rows(p55, tag, jt["configs"])
        # No b5_g256 measurement exists for the joint source. Its available
        # highest-bit cell is b5_g128: one calibration, two remaining test cells.
        ja = [r for r in jr if r["config"] == "b5_g128"]
        rr, ca = quant_calibrate(p55, qm, qs, jr, ja)
        qtests["joint_test"]["rows"].extend(rr)
        qtests["joint_test"]["calibration"].extend(ca)
        qtests["joint_test"]["folds"].append({"held_out_source": tag, "train_sources": sorted({r["source"] for r in qrows})})
    quant_tests = {}
    for name, value in qtests.items():
        quant_tests[name] = {**scored(value["rows"], qmethods), "calibration": value["calibration"], "folds": value["folds"]}
    # Also recompute the delivered full-six-state, held-configuration protocol.
    # Its calibration anchors for bit/granularity tests already entered fitting.
    delivered_tests = {}
    for name, spec in r55["test_sets"].items():
        rr, ca = [], []
        for tag in spec["states"]:
            held = quant_rows(p55, tag, spec["configs"])
            anchors = (quant_rows(p55, tag, ["b5_g256"]) if tag in p55.DEV_TAGS else
                       [r for r in held if r["config"] == "b5_g128"])
            one, cost = quant_calibrate(p55, qm, qs, held, anchors)
            rr.extend(one)
            ca.extend(cost)
        delivered_tests[name] = {**scored(rr, qmethods), "calibration": ca}
    qtable = p55.mae_table(qp)
    delta = max(abs(a["mae"][c] - b["mae"][c]) for a, b in zip(qtable, r55["loso_table"]) for c in CAPS)
    assert delta < 1e-9
    reproduction["v55_max_loso_mae_difference"] = delta
    all_x = quant_shared_design(p55, qrows, qs)
    all_beta = p55.ridge_fit(all_x, np.asarray([r["y"] for r in qrows]))
    print("Distillation: twelve run folds; all delivered forms refitted with NumPy.", flush=True)
    dist = distillation(p53, d56, r56)
    delta = max(abs(dist["metrics"][key][c]["mae"] - val["loco"]["metrics"][c]["mae"])
                for key, val in r56["part_a"].items() for c in CAPS)
    assert delta < 1e-9
    reproduction["v56_max_loro_mae_difference"] = delta
    return {
        "version": 59,
        "protocol": {"device": "CPU", "dependencies": "NumPy and Python standard library only", "numpy": np.__version__,
                     "seed": SEED, "ridge": 1e-3, "feature_names": p53.FEATURE_NAMES,
                     "standardization": "Natural log N0 and D0, dense L0; training-row/capability pooled population z scores; training-only per fold; all v53/v55 coefficients penalized.",
                     "Pythia_N0": "layers * (4*h*h + 2*h*intermediate); embeddings/head/vectors excluded", "tokens_per_step": p53.TOKENS_PER_STEP,
                     "pruning_gamma_grid": p53.GAMMA_GRID.tolist(), "quant_p_grid": p55.P_GRID.tolist(), "quant_q_grid": p55.Q_GRID.tolist(),
                     "metric": "Signed dL in nats, absolute errors pooled over held-out cells within capability; macro equally weights capabilities (complete panels).",
                     "classification": "abs(shared-specific)<=0.02: within 0.02; otherwise shared-specific>0.02: worse; otherwise better. Descriptive, not a significance test.",
                     "split_membership": "Registers whitelist exact dev states/cells/checkpoints; later measurements never expand dev membership.",
                     "strengths": {"pruning": "s=(1-d)/0.3; (beta.phi)*s**gamma", "quantization": "u=log2(qmax)-training_mean; v=log2(g/128); separable=(beta.phi)*2**(-p*(u+u_center)+q*v). Uncentered qmax factor preserves v55 ridge convention.", "distillation": "s=log(1+E); (beta.phi)*s; E=completion_tokens_seen/D_U_completion"},
                     "intervals": "Per-capability absolute LOSO errors, empirical 0.8/0.95 quantiles using higher order statistic; symmetric prediction +/- quantile; no clipping. Cell weighted, descriptive prediction intervals, no exchangeability/nominal coverage guarantee.",
                     "interval_score_cells": "Only densities remaining after the one-point calibration, for BOTH K0 and K1. Confirmation uses 17-source LOSO; pair evaluation uses new 13-source LOSO excluding all four pairs from every fit and residual bank.",
                     "calibration_cost": "One compressed configuration measured per source, evaluating all three capabilities; dense reference is already an input for K0 and K1. No training data or gamma tuning from target remaining cells."},
        "part1": {"pruning": {"comparison": compare(pcv["all_rows"], "shared_family", "power"), "delivered_forms_metrics": metrics(pcv["all_rows"], p53.METHODS), "predictions": pcv["all_rows"]},
                  "quantization": {"comparison": compare(qp, "separable", "low_order_2d"), "delivered_forms_metrics": metrics(qp, p55.METHODS), "predictions": qp},
                  "distillation": dist},
        "part2": {"pruning": {"sharing": compare(pcv["all_rows"], "shared_gamma", "power"),
                              "full_dev_standardization": stats, "full_dev_models": models, "full_dev_shared_gamma": power_models(p53, rows, stats, shared=True),
                              "folds": pcv["folds"], "gamma_distribution": distribution, "per_source_gamma": per_state,
                              "per_source_gamma_scope": "Descriptive within-source fits A_source,c*s**gamma_source,c, ridge 1e-3 on scalar amplitude. No LOSO score: an unseen source gamma cannot be fitted without responses.",
                              "parameter_counts": {"shared_gamma": 13, "per_capability_gamma": 15, "per_source_gamma": 102}, "bootstrap": ci},
                  "quantization": {"sharing": compare(qp, "shared_terms", "low_order_2d"), "folds": qfolds,
                                   "full_dev_standardization": qs, "full_dev_specific_models": qm,
                                   "shared_coefficients": all_beta.tolist(), "shared_design_rank": int(np.linalg.matrix_rank(all_x)),
                                   "definition": "dL_c=beta_0,c.phi+sum(k=1..4) beta_k.phi*t_k; t=(u,v,uv,u^2); beta_0,c varies with capability, other 4-vectors shared.",
                                   "parameter_counts": {"shared": 28, "specific": 60},
                                   "identifiability": "Only two dev bit levels: centered u^2 is constant, aliased with intercept; reported coefficients are ridge solutions, not separately identified terms."}},
        "part3": {"pruning": {"dev_loso": dev, "confirmation": confirmation, "pairs_17_source_loso": pair_oof, "pairs_independent_13_source": pairs,
                              "calibration_cost": {"measurements_per_source": 1, "capabilities_per_measurement": 3,
                                                   "dev_sources": 17, "confirmation_sources": 3, "pair_sources_reused_from_dev": 4,
                                                   "unique_compressed_measurements": 20},
                              "interval_references": {"17_source_loso": pi_full, "13_source_loso_excluding_all_pairs": pi_pair},
                              "pair_interval_training_folds": pair_cv["folds"], "pair_interval_residual_predictions": pair_cv["remaining"],
                              "pairs_caveat": "Pairs belong to the registered dev panel. Their 17-source OOF results are descriptive; independent PI evaluation excludes all four together and refits from the other 13 sources."},
                  "quantization": {"dev_loso": {**scored(qremain, qmethods), "calibration": qcal}, "source_held_out_tests": quant_tests,
                                   "calibration_cost": {"measurements_per_source": 1, "capabilities_per_measurement": 3,
                                                        "dev_b5_g256_reused_in_bit_and_granularity_tests": 6,
                                                        "joint_b5_g128": 1, "unique_compressed_measurements": 7},
                                   "delivered_configuration_split_tests": delivered_tests,
                                   "calibration_rule": "Fixed b5_g256 for six dev sources (5-bit endpoint, not the unique mildest cell: b5_g64 also exists). Joint source has no b5_g256; use available highest-bit b5_g128, score b3_g128/b4_g128 only.",
                                   "low2d_K1": "Multiply the frozen source-specific 2D curve by measured_anchor/predicted_anchor. This is a one-scalar amplitude correction conditional on x; it does not identify the 2D shape. No clipping; fail if denominator <=1e-12 in magnitude.",
                                   "separable_K1": "Exact amplitude=measured_anchor/[qmax**(-p)*(g/128)**q], with p,q fitted without target-source rows.",
                                   "test_protocol": "Primary calibration tests exclude all target-source dev rows (five-source fits for bit/granularity, six for joint). The delivered full-six-source configuration split is additionally reported in JSON, with calibration anchors already used in its bit/granularity training.",
                                   "intervals": "Not constructed for quantization; requested pruning intervals reported separately."}},
        "validation": {"historical_reproduction": reproduction, "calibration_cells_excluded_from_scores": True,
                       "PI_test_sources_disjoint_from_residual_bank_and_its_training_folds": True,
                       "batched_power_fit_matches_v53": True},
    }


def triplet(values, digits=3):
    return "/".join(f"{v:.{digits}f}" for v in values)


def mae3(result, method):
    return triplet([result["metrics"][method][c]["mae"] for c in CAPS])


def pi_cell(result, level):
    if "intervals" not in result:
        return "—"
    return " → ".join(f"{100*result['intervals'][m]['macro'][level]['coverage']:.1f}%/{result['intervals'][m]['macro'][level]['mean_width']:.3f}"
                      for m in ("K0", "K1"))


def report_tables(summary):
    p1, p2, p3 = (summary[f"part{i}"] for i in (1, 2, 3))
    t1 = ["| Part 1: family / comparator | Shared MAE M/C/Q | Specific MAE M/C/Q | Macro shared → specific | Verdict |",
          "|---|---:|---:|---:|---|"]
    comparisons = [("Pruning power / v53 power", p1["pruning"]["comparison"]),
                   ("Quant separable / v55 2D", p1["quantization"]["comparison"]),
                   ("Distill log(1+E) / v56 F2:L0", p1["distillation"]["comparison"]),
                   ("Distill log(1+E) / F2 same phi", p1["distillation"]["same_phi_comparison"])]
    for name, comp in comparisons:
        t1.append(f"| {name} | {triplet([comp[c]['shared_mae'] for c in CAPS])} | {triplet([comp[c]['specific_mae'] for c in CAPS])} | "
                  f"{comp['macro']['shared_mae']:.3f} → {comp['macro']['specific_mae']:.3f} | {comp['macro']['classification']} |")
    t2 = ["| Part 2: quantity | Math | Code | QA |",
          "|---|---:|---:|---:|"]
    for arm, label in (("pruning", "LOSO MAE: shared gamma → per-cap gamma"), ("quantization", "LOSO MAE: shared terms → per-cap terms")):
        comp = p2[arm]["sharing"]
        t2.append(f"| {label} | " + " | ".join(f"{comp[c]['shared_mae']:.3f} → {comp[c]['specific_mae']:.3f}" for c in CAPS) + " |")
    dist = p2["pruning"]["gamma_distribution"]
    t2.append("| Per-source gamma: min / median / max | " + " | ".join(triplet([dist[c][k] for k in ("min", "median", "max")], 2) for c in CAPS) + " |")
    sg = p2["pruning"]["full_dev_shared_gamma"]["math"]["gamma"]
    t2.append(f"| Full-dev gamma: shared → per-cap | " + " | ".join(f"{sg:.2f} → {p2['pruning']['full_dev_models'][c]['power']['gamma']:.2f}" for c in CAPS) + " |")
    boot = p2["pruning"]["bootstrap"]["parameters"]
    for name in ("beta_intercept", "beta_logN0", "beta_L0c", "beta_logD0", "gamma"):
        t2.append(f"| {name}: estimate [95% CI] | " + " | ".join(
            f"{boot[c][name]['estimate']:.3f} [{boot[c][name]['ci95'][0]:.3f}, {boot[c][name]['ci95'][1]:.3f}]" for c in CAPS) + " |")
    panels = [("Prune dev LOSO", p3["pruning"]["dev_loso"], "K0", "K1"),
              ("Prune confirmation", p3["pruning"]["confirmation"], "K0", "K1"),
              ("Prune pairs: 17-state OOF", p3["pruning"]["pairs_17_source_loso"], "K0", "K1"),
              ("Prune pairs: independent 13", p3["pruning"]["pairs_independent_13_source"], "K0", "K1")]
    for label, result in [("dev LOSO", p3["quantization"]["dev_loso"]),
                          *p3["quantization"]["source_held_out_tests"].items()]:
        panels.append((f"Quant 2D {label}", result, "K0", "K1"))
    for label, result in p3["quantization"]["source_held_out_tests"].items():
        panels.append((f"Quant sep {label}", result, "separable_K0", "separable_K1"))
    t3 = ["| Part 3: remaining cells | n/cap | K0 MAE M/C/Q | K1 MAE M/C/Q | Median MAE M/C/Q | Strength MAE M/C/Q | 80% PI: coverage/width K0 → K1 | 95% PI: coverage/width K0 → K1 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, result, k0, k1 in panels:
        strength = mae3(result, "strength_only") if "strength_only" in result["metrics"] else "—"
        t3.append(f"| {label} | {result['metrics'][k0]['math']['n']} | {mae3(result, k0)} | {mae3(result, k1)} | {mae3(result, 'median_curve')} | {strength} | {pi_cell(result, '0.8')} | {pi_cell(result, '0.95')} |")
    return ["\n".join(t) for t in (t1, t2, t3)], panels


def markdown(summary, tables):
    p3 = summary["part3"]
    gamma_dist = summary["part2"]["pruning"]["gamma_distribution"]
    texts = ["# V59 shared structure", "",
             "CPU/NumPy; deterministic seed 59053; MAE and widths in nats. M/C/Q = math/code/QA. "
             "Part 1 and sharing use every registered held-out dev cell: pruning 17 sources, quantization six states, distillation 12 runs (48 checkpoints). "
             "All delivered-form MAEs are recomputed, with historical numbers used only for reproduction assertions.", "", tables[0], "",
             "Shared family means the same functional template, with parameters fitted separately by method and capability; it does not mean equal exponents across methods. "
             "Pruning's shared and delivered forms coincide. Quantization uses qmax^(-p)(g/128)^q with v55's original ridge parameterization; "
             "u is centered for the 2D comparator. Distillation uses log(1+E) with v53-style phi. "
             "Its native delivered comparator is F2:L0; all ten v56 forms are refitted and stored in JSON. The extra F2 same-phi row holds features and ridge preprocessing fixed. "
             "D0 is unavailable for Gemma: its standardized column is zero, so its effect cannot be estimated. N0 uses the counts already recorded by v56. "
             "The native F2 comparison also changes feature/scaling conventions; use the same-phi row to isolate shape. "
             "Verdicts use macro MAE and the inclusive 0.02 threshold; capability verdicts are in JSON.", "", tables[1], "",
             "Pruning shares one gamma with separate four-vector amplitudes (13 vs 15 coefficients including exponents). "
             "Quantization shares the four nonconstant term coefficient vectors, retaining capability-specific four-vector offsets (28 vs 60). "
             "At two dev bit levels u^2 aliases the intercept; ridge identifies a solution but not unique unregularized term effects.", "",
             "'Distribution of parameters across models' = separately fitted scalar amplitude and gamma for each source/capability, using all its dev densities. "
             "These describe observed models, not predictions for an unseen gamma. Grid-boundary counts M/C/Q: "
             + "/".join(str(gamma_dist[c]["n_at_grid_boundary"]) for c in CAPS) + " of 17. "
             "'Parameter confidence interval' = 95% percentile intervals from 1000 cluster-bootstrap draws of 17 sources with replacement; "
             "all densities/capabilities travel together. Standardizer and gamma are refitted; beta is mapped back to the full-dev standardized basis before taking percentiles. "
             "The intervals are conditional on the delivered power family and gamma grid, not model-selection uncertainty.", "", tables[2], "",
             "Part 3 scores identical remaining cells for K0, K1 and baselines. K1 pruning fixes training-only gamma and sets "
             "A_c = observed_dL_c(d_cal)/[((1-d_cal)/0.3)^gamma_c]. The mildest density is 0.9 for 16 dev sources and 0.65 for 1B@96k; "
             "confirmation uses 0.85. Median and strength-only curves are refitted on each training fold, with v53 interpolation and unregularized strength-only fitting.", "",
             "The confirmation panel is 410M@48k, 1.4B@112k, 6.9B@80k at 0.675/0.575 after calibration. "
             "Pairs are 1B and 6.9B at 32k/112k, scoring 0.8/0.75/0.7/0.65/0.6/0.55 after calibration at 0.9. "
             + p3["pruning"]["pairs_caveat"], "",
             "Prediction intervals use per-capability absolute LOSO error quantiles (NumPy method='higher'), "
             "prediction +/- q80 or q95, evaluated without clipping. The compact table pools coverage and mean width over capabilities; "
             "JSON supplies per-capability coverage, widths, counts, radii, all residuals and per-source MAEs. "
             "K0 and K1 interval banks use the same post-calibration cells. No confirmation or pair source enters its own interval bank or any fit used to construct that bank. "
             "These are descriptive empirical intervals, not guaranteed nominal coverage; cells within sources are dependent.", "",
             "Calibration cost: one compressed measurement per source, evaluated on three capabilities. "
             "Pruning: 17 for the dev LOSO, 3 for confirmation, 4 for each pair evaluation (the same four measured anchors are reused); "
             "20 unique pruning sources total. Quantization: 6 b5_g256 measurements reused for dev/bit/granularity, plus one b5_g128 for the joint source. "
             "Dense reference losses are inputs for both K0 and K1 and are not counted as extra compressed measurements.", "",
             p3["quantization"]["calibration_rule"], "", p3["quantization"]["low2d_K1"], "",
             "Quant sep uses exact amplitude calibration of the separable law, with p and q from training. "
             "Small signed responses at the 5-bit anchor can make amplitude calibration unstable, especially when extrapolating toward 3 bits. "
             + p3["quantization"]["test_protocol"], "",
             "Input SHA256 values, source membership, every fold fit/prediction, bootstrap samples, calibration records, "
             "and all per-capability statistics are in summary.json. Outputs are created exclusively; existing files are never overwritten.", ""]
    return "\n".join(texts)


@proofread_table
def latex(summary, panels):
    lines = [r"\begin{table}[!htbp]", r"\centering\footnotesize\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}p{1.9cm}p{1.9cm}p{2.0cm}p{2.5cm}p{2.1cm}p{2.5cm}@{}}", r"\toprule",
             r"Law & What is shared & What varies & Held-out MAE shared vs specific & K0 vs K1 MAE & 80\% PI coverage \& width \\", r"\midrule"]
    for arm, law, shared, varies in (
        ("pruning", "Pruning power", "Power family", r"$\beta_c,\gamma_c$"),
        ("quantization", "Grouped RTN", "Separable family", r"$\beta_c,p_c,q_c$"),
        ("distillation", "Distill exposure", r"$\log(1+E)$ family", r"$\beta_c$; inactive $D_0$")):
        comp = summary["part1"][arm]["comparison"]["macro"]
        lines.append(f"{law} & {shared} & {varies} & {comp['shared_mae']:.3f} vs {comp['specific_mae']:.3f} & --- & ---" + r" \\")
    for arm, shared, varies in (("pruning", r"One $\gamma$", r"$\beta_c$"), ("quantization", "2D term vectors", "Capability offsets")):
        comp = summary["part2"][arm]["sharing"]["macro"]
        lines.append(f"{arm.title()} sharing & {shared} & {varies} & {comp['shared_mae']:.3f} vs {comp['specific_mae']:.3f} & --- & ---" + r" \\")
    gd = summary["part2"]["pruning"]["gamma_distribution"]
    for c in CAPS:
        val = "/".join(f"{gd[c][k]:.2f}" for k in ("min", "median", "max"))
        lines.append(f"Prune {c} & Power family & Source amplitudes, $\\gamma$ & $\\gamma$ min/med/max {val} & --- & ---" + r" \\")
    lines.append(r"\midrule")
    for label, result, k0, k1 in panels:
        a, b = (result["metrics"][m]["macro"]["mae"] for m in (k0, k1))
        name = label.replace("_", r"\_").replace("17-state OOF", "17-source OOF")
        share = r"Training $\gamma_c$" if label.startswith("Prune") else "Training curve"
        width = pi_cell(result, "0.8").replace("—", "---").replace("%", r"\%").replace(" → ", " to ")
        lines.append(f"{name} & {share} & One target amplitude & --- & {a:.3f} vs {b:.3f} & {width}" + r" \\")
    assert sum(line.endswith(r"\\") for line in lines) <= 30
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Shared structure and one-point calibration. MAE and PI widths are nats, pooled equally over capabilities. "
              r"Family comparisons use registered dev splits: 17 pruning sources, six quantization states, 12 distillation runs. "
              r"Specific forms: v53 power, v55 low-order 2D, v56 F2 with $L_0$. Distillation $D_0$ is unavailable (inactive column). "
              r"Gamma ranges describe within-source fits. Bootstrap parameter CIs are in the accompanying summary. "
              r"K1 costs one compressed measurement per source; scores exclude that cell. "
              r"PI entries give coverage/full width for K0 then K1, from per-capability absolute LOSO residual quantiles. "
              r"Confirmation excludes three new sources; independent pair evaluation excludes all four pair sources from fitting and its 13-source interval bank. "
              r"Quantization tests omit the target source from fitting; 2D K1 scales its predicted curve and sep K1 calibrates the separable amplitude.}",
              r"\label{tab:shared_structure}", r"\end{table}", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/v59-shared-structure")
    parser.add_argument("--table", type=Path, default=ROOT / "paper/paper/tables/shared_structure.tex")
    args = parser.parse_args(argv)
    paths = (args.out_dir / "summary.json", args.out_dir / "summary.md", args.table)
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    FILES.clear()
    record(__file__, "analysis_source")
    p53, p55, d56 = (reference_module(name) for name in ("v53_prune_dev", "v55_quant_group_fit", "v56_distill_forms"))
    r53 = read_json(p53.OUT / "register.json", "split_register")
    r55 = read_json(p55.OUT / "register.json", "split_register")
    r56 = read_json(d56.OUT / "summary.json", "distillation_data_and_forms_reference")
    summary = run_analysis(p53, p55, d56, r53, r55, r56)
    manifest = [FILES[k] for k in sorted(FILES)]
    summary["files_read"] = manifest
    summary["input_sha256"] = {f["path"]: f["sha256"] for f in manifest}
    summary["input_manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    tables, panels = report_tables(summary)
    summary["printed_tables"] = tables
    texts = (json.dumps(summary, indent=2, allow_nan=False) + "\n", markdown(summary, tables), latex(summary, panels))
    # Re-read the entire manifest before writing: ensure no input changed mid-run.
    for item in list(manifest):
        path = Path(item["path"])
        record(path if path.is_absolute() else ROOT / path, item["roles"][0])
    for path, content in zip(paths, texts):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
    print("\n" + "\n\n".join(tables), flush=True)
    print(f"\nWrote {paths[0]}, {paths[1]}, {paths[2]}; {len(manifest)} files hashed.", flush=True)
    return summary


if __name__ == "__main__":
    main()
