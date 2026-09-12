#!/usr/bin/env python3
"""CPU/numpy distillation forms and capability-conditioning audit.

Usage: python analysis/v56_distill_forms.py [dev|all]
The v50 points() path, filtering, ordering and first-four rule are preserved.
Only numpy fits models; transformers/accelerate instantiate meta weights solely
to reproduce v50's parameter counts. Cached configs suffice; no network/GPU use.
"""

import argparse
import glob
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re

# Set before numpy/transformers imports, including when imported for validation.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v56-distill-forms"
REG = ROOT / "results/v47-p2-register/register.json"
CAPS = ("math", "code", "qa")
DEV_STUDENTS = ("gemma3-270m", "gemma3-1b")
HF = {"gemma3-270m": "google/gemma-3-270m",
      "gemma3-1b": "google/gemma-3-1b-pt", "gemma3-4b": "google/gemma-3-4b-pt"}
T_REF = 35000.0
T_GRID = (17500, 35000, 70000, 140000, 280000)
RIDGE = 1e-3
FORMS = (("zero", None), ("constant", None), ("T-only", None),
         ("E-only", None), ("surface", "L0"), ("surface", "logN"),
         ("F1", "L0"), ("F1", "logN"), ("F2", "L0"), ("F2", "logN"))
reg = {}
FILES = {}
JSON_CACHE = {}
RUNS = []


def path_label(path):
    path = Path(path).absolute()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def record_file(path, role):
    """Hash the bytes actually consumed; reject changes during this analysis."""
    path = Path(path)
    raw = path.read_bytes()
    key = path_label(path)
    digest = hashlib.sha256(raw).hexdigest()
    if key in FILES and FILES[key]["sha256"] != digest:
        raise ValueError(f"Input changed while reading: {key}")
    FILES.setdefault(key, {"path": key, "sha256": digest, "bytes": len(raw), "roles": []})
    if role not in FILES[key]["roles"]:
        FILES[key]["roles"].append(role)
    return raw


def read_json(path, role):
    key = path_label(path)
    if key not in JSON_CACHE:
        JSON_CACHE[key] = json.loads(record_file(path, role))
    elif role not in FILES[key]["roles"]:
        FILES[key]["roles"].append(role)
    return JSON_CACHE[key]


def n_params(student):
    """Exactly v50's meta-device count (not the LoRA training manifest count)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.utils.hub import cached_file
    from accelerate import init_empty_weights

    cfg_path = cached_file(HF[student], "config.json", local_files_only=True)
    record_file(cfg_path, f"model_config:{student}")
    cfg = AutoConfig.from_pretrained(HF[student], local_files_only=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg)
    if any(p.device.type != "meta" for p in model.parameters()):
        raise ValueError("Parameter counting unexpectedly allocated non-meta weights")
    return sum(p.numel() for p in model.parameters())


def points(student, U, seed, suffix):
    """v50 data access verbatim in semantics, with dense losses/provenance added."""
    d = ROOT / f"results/v12-distill/{student}/gpt-5.6-luna_full_{U}_{suffix}_lora_dseed{seed}"
    DU = reg["pools"][f"U{U}_s{seed}"]["D_U_completion"]
    pts, documents = [], []
    if (d / "eval.json").is_file():
        documents.append((d / "eval.json", read_json(d / "eval.json", "run_eval")))
    for f in sorted(glob.glob(str(d / "trajectory/update-*/eval.json"))):
        j = read_json(f, "snapshot_eval")
        documents.append((Path(f), j))
        if j.get("processed_tokens", 0) > 0:
            pts.append({"student": student, "U": U, "seed": seed,
                        "Tc": j["completion_tokens_seen"], "DU": DU,
                        "E": j["completion_tokens_seen"] / DU, "delta": j["delta"],
                        "processed": j["processed_tokens"], "updates": j.get("updates"),
                        "file": path_label(f)})
    pts.sort(key=lambda p: p["Tc"])
    pts = pts[:4]
    dense_docs = [(p, j["dense"]) for p, j in documents if "dense" in j]
    cluster = f"{student}|U{U}_s{seed}"
    run = {"cluster": cluster, "student": student, "U": U, "seed": seed,
           "suffix": suffix, "path": path_label(d), "directory_exists": d.is_dir(),
           "n_points": len(pts), "found": bool(pts), "complete_four_points": len(pts) == 4,
           "dense_sources": [path_label(p) for p, _ in dense_docs]}
    if pts:
        if not dense_docs:
            raise ValueError(f"No dense losses in run or snapshot eval.json: {d}")
        dense = {c: float(dense_docs[0][1][c]) for c in CAPS}
        for f, values in dense_docs:
            if not all(np.isclose(values[c], dense[c], rtol=0, atol=1e-10) for c in CAPS):
                raise ValueError(f"Inconsistent dense losses: {f}")
        run["L0"] = dense
        for p in pts:
            p.update(L0=dense, cluster=cluster, suffix=suffix)
            if p["Tc"] <= 0 or DU <= 0:
                raise ValueError(f"Nonpositive completion-token exposure: {p['file']}")
            j = JSON_CACHE[p["file"]]
            for c in CAPS:
                if not np.isfinite(p["delta"][c]) or not np.isfinite(dense[c]):
                    raise ValueError(f"Nonfinite losses: {p['file']}")
                if "post_training" in j and not np.isclose(
                        p["delta"][c], j["post_training"][c] - dense[c], rtol=0, atol=1e-9):
                    raise ValueError(f"delta != post_training - dense: {p['file']}")
    RUNS.append(run)
    return pts


def descriptor_refs(dev, counts):
    # Equal weight per student, never per checkpoint/run. L0 is a pretreatment
    # descriptor, so these fixed DEV references are also used in student CV.
    values = {}
    for student in DEV_STUDENTS:
        rows = [p for p in dev if p["student"] == student]
        if rows:
            first = rows[0]["L0"]
            if not all(all(np.isclose(p["L0"][c], first[c], rtol=0, atol=1e-10)
                           for c in CAPS) for p in rows):
                raise ValueError(f"Student dense loss differs between runs: {student}")
            values[student] = first
    refs = {"N": counts, "L0_by_dev_student": values, "L0": {}, "logN": {}}
    for c in CAPS:
        arr = [v[c] for v in values.values()]
        refs["L0"][c] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
    arr = [math.log(counts[s]) for s in DEV_STUDENTS]
    refs["logN"] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
    refs["L0_reference_students"] = list(values)
    return refs


def z_value(p, cap, descriptor, refs):
    stats = refs["L0"][cap] if descriptor == "L0" else refs["logN"]
    value = p["L0"][cap] if descriptor == "L0" else math.log(refs["N"][p["student"]])
    return (value - stats["mean"]) / (stats["std"] or 1.0)


def basis(kind, descriptor, p, cap, refs, t_star=None):
    t = math.log1p(p["Tc"] / T_REF)
    e = math.log1p(p["E"])
    z = z_value(p, cap, descriptor, refs) if descriptor else 0.0
    if kind == "zero":
        return []
    if kind == "constant":
        return [1.0]
    if kind == "T-only":
        return [1.0, t]
    if kind == "E-only":
        return [1.0, e]
    if kind == "surface":
        return [1.0, t, e, z]
    if kind == "F1":
        return [e, z * e]
    if kind == "F2":
        s = -math.expm1(-p["Tc"] / t_star)
        return [s, z * s, e, z * e]
    raise ValueError(kind)


def ridge_fit(X, y, names, intercept):
    """min ||X_std beta-y||^2 + .001 ||beta_nonintercept||^2.

    Center when an intercept exists. Otherwise divide by training population
    standard deviations without centering, preserving F1/F2's zero at Tc=0.
    Constant columns have scale 1. Coefficients are also mapped to raw bases.
    """
    X = np.asarray(X, dtype=float).reshape(len(y), len(names))
    y = np.asarray(y, dtype=float)
    if not names:
        return {"coef": [], "columns": [], "n_params": 0, "rank": 0,
                "column_mean": [], "column_scale": [], "standardized_coef": [],
                "sse": float(y @ y), "ridge_objective": float(y @ y),
                "effective_df": 0.0, "intercept_unpenalized": False}
    mean = X.mean(axis=0) if intercept else np.zeros(X.shape[1])
    scale = X.std(axis=0)
    scale[scale < 1e-12] = 1.0
    penalty = np.ones(X.shape[1])
    if intercept:
        mean[0], scale[0], penalty[0] = 0.0, 1.0, 0.0
    Z = (X - mean) / scale
    augmented = np.vstack((Z, np.diag(np.sqrt(RIDGE * penalty))))
    beta = np.linalg.lstsq(augmented, np.r_[y, np.zeros(X.shape[1])], rcond=None)[0]
    coef = beta / scale
    if intercept:
        coef[0] -= mean @ coef
    residual = X @ coef - y
    gram = Z.T @ Z
    return {"coef": coef.tolist(), "columns": names, "n_params": len(names),
            "rank": int(np.linalg.matrix_rank(X)), "column_mean": mean.tolist(),
            "column_scale": scale.tolist(), "standardized_coef": beta.tolist(),
            "sse": float(residual @ residual),
            "ridge_objective": float(residual @ residual + RIDGE * (penalty * beta) @ beta),
            "effective_df": float(np.trace(np.linalg.solve(gram + RIDGE * np.diag(penalty), gram))),
            "intercept_unpenalized": intercept}


def metrics(actual, predicted):
    err = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {"n": len(err), "mae": float(np.abs(err).mean()) if len(err) else None,
            "bias": float(err.mean()) if len(err) else None,
            "sse": float(err @ err) if len(err) else None}


def prediction_rows(pts, predictions, fold=None):
    return [{"file": p["file"], "cluster": p["cluster"], "student": p["student"],
             "U": p["U"], "seed": p["seed"], "Tc": p["Tc"], "E": p["E"],
             "capability": c, "actual": float(p["delta"][c]),
             "predicted": float(predictions[i, k]), "fold": fold}
            for i, p in enumerate(pts) for k, c in enumerate(CAPS)]


def summarize(rows):
    result = {}
    for c in (*CAPS, "macro"):
        rr = [r for r in rows if c == "macro" or r["capability"] == c]
        result[c] = metrics([r["actual"] for r in rr], [r["predicted"] for r in rr])
    return result


def form_key(kind, descriptor):
    return f"{kind}:{descriptor}" if descriptor else kind


def fit_a(pts, refs, kind, descriptor):
    names = {"zero": [], "constant": ["a"], "T-only": ["a", "b_t"],
             "E-only": ["a", "b_e"], "surface": ["a", "b_t", "c_e", "d_z"],
             "F1": ["a_e", "lambda_ze"],
             "F2": ["a_s", "lambda_zs", "b_e", "mu_ze"]}[kind]
    candidates = []
    for ts in T_GRID if kind == "F2" else (None,):
        fits = {}
        for c in CAPS:
            X = [basis(kind, descriptor, p, c, refs, ts) for p in pts]
            fits[c] = ridge_fit(X, [p["delta"][c] for p in pts], names,
                                kind not in ("zero", "F1", "F2"))
        candidates.append({"T_star": ts, "sse": sum(f["sse"] for f in fits.values()), "fits": fits})
    best = min(candidates, key=lambda c: c["sse"])
    return {"kind": kind, "descriptor": descriptor, "T_star": best["T_star"],
            "fits": best["fits"], "grid_fits": candidates,
            "n_params": 3 * len(names), "n_params_per_capability": len(names),
            "discrete_hyperparameters": {"T_star_choices": len(candidates)},
            "training_sse": best["sse"]}


def predict_a(model, pts, refs):
    return np.array([[np.dot(basis(model["kind"], model["descriptor"], p, c, refs,
                                  model["T_star"]), model["fits"][c]["coef"])
                      for c in CAPS] for p in pts]).reshape(len(pts), 3)


def b_shape(kind, p, cap, refs, ts):
    if kind == "E-only":
        return basis("E-only", None, p, cap, refs)[1:]
    if kind == "T+E":
        return basis("surface", "L0", p, cap, refs)[1:3]
    return basis("F2", "L0", p, cap, refs, ts)


def b_subsets(kind):
    """Equal fitted coefficient counts, with nonnested allocation of offsets.

    Shared: 3 offsets + q common slopes. Per-cap: 1 common offset + q+2
    capability-specific slopes. No fake/dummy parameters. For T+E, one cap
    gets both terms and the other two get one each. For F2 each cap gets one
    of {s,z*s} and one of {e,z*e}. Subsets are training-SSE hyperparameters.
    """
    if kind == "E-only":
        return [(0, 1, 2)]
    if kind == "T+E":
        return sorted({tuple(sorted([2*c, 2*c+1] + [2*k + j for k, j in
                       zip([k for k in range(3) if k != c], choices)]))
                       for c in range(3) for choices in itertools.product(range(2), repeat=2)})
    return [tuple(4*c+j for c, pair in enumerate(pairs) for j in pair)
            for pairs in itertools.product(((0, 2), (0, 3), (1, 2), (1, 3)), repeat=3)]


def b_design(kind, mode, pts, refs, ts, subset=None):
    shape_names = {"E-only": ["e"], "T+E": ["t", "e"],
                   "F2": ["s", "z*s", "e", "z*e"]}[kind]
    q = len(shape_names)
    rows = []
    # Fixed orthonormal capability contrasts: span all three cap offsets.
    contrasts = ((1/math.sqrt(2), 1/math.sqrt(6)),
                 (-1/math.sqrt(2), 1/math.sqrt(6)), (0.0, -2/math.sqrt(6)))
    for p in pts:
        for k, c in enumerate(CAPS):
            shape = b_shape(kind, p, c, refs, ts)
            if mode == "shared":
                rows.append([1.0, *contrasts[k], *shape])
            else:
                block = np.zeros(3 * q)
                block[k*q:(k+1)*q] = shape
                rows.append([1.0, *block[list(subset)]])
    names = (["offset", "offset_math-code", "offset_math+code-2qa", *shape_names]
             if mode == "shared" else ["offset", *[f"{CAPS[j//q]}:{shape_names[j%q]}" for j in subset]])
    return np.asarray(rows).reshape(len(pts)*3, len(names)), names


def fit_b(pts, refs, kind, mode):
    candidates = []
    y = [p["delta"][c] for p in pts for c in CAPS]
    subsets = [None] if mode == "shared" else b_subsets(kind)
    for ts in T_GRID if kind == "F2" else (None,):
        for subset in subsets:
            X, names = b_design(kind, mode, pts, refs, ts, subset)
            fitted = ridge_fit(X, y, names, True)
            candidates.append({"T_star": ts, "subset": subset, "fit": fitted})
    best = min(candidates, key=lambda c: c["fit"]["sse"])
    return {"kind": kind, "mode": mode, **best, "n_params": best["fit"]["n_params"],
            "candidate_fits": candidates, "discrete_hyperparameters": {
                "subset_choices": len(subsets), "T_star_choices": 5 if kind == "F2" else 1},
            "training_sse": best["fit"]["sse"]}


def predict_b(model, pts, refs):
    X, _ = b_design(model["kind"], model["mode"], pts, refs, model["T_star"], model["subset"])
    return (X @ np.asarray(model["fit"]["coef"])).reshape(len(pts), 3)


def cross_validate(pts, refs, group, fit, predict, *args):
    labels = sorted({p[group] for p in pts})
    folds, rows = [], []
    for label in labels:
        train = [p for p in pts if p[group] != label]
        held = [p for p in pts if p[group] == label]
        if not train:
            folds.append({"held_out": label, "status": "unavailable: no training points"})
            continue
        model = fit(train, refs, *args)
        rr = prediction_rows(held, predict(model, held, refs), label)
        rows.extend(rr)
        folds.append({"held_out": label, "n_train_points": len(train), "n_held_points": len(held),
                      "train_clusters": sorted({p["cluster"] for p in train}),
                      "held_clusters": sorted({p["cluster"] for p in held}),
                      "fit": model, "metrics": summarize(rr)})
    return {"group": group, "n_folds": sum("fit" in f for f in folds),
            "folds": folds, "predictions": rows, "metrics": summarize(rows)}


def evaluate_model(dev, test, refs, fit, predict, args, student_cv=False):
    model = fit(dev, refs, *args)
    result = {"fit_all_dev": model,
              "dev_in_sample": summarize(prediction_rows(dev, predict(model, dev, refs))),
              "loco": cross_validate(dev, refs, "cluster", fit, predict, *args)}
    if student_cv:
        result["loso"] = cross_validate(dev, refs, "student", fit, predict, *args)
    if test:
        rows = prediction_rows(test, predict(model, test, refs))
        result["test"] = {"predictions": rows, "metrics": summarize(rows),
                          "by_student": {s: summarize([r for r in rows if r["student"] == s])
                                         for s in sorted({p["student"] for p in test})}}
    return result


def cell(m):
    return "NA" if m["mae"] is None else f"{m['mae']:.3f}/{m['bias']:+.3f}"


def markdown(summary):
    data = summary["data"]
    lines = ["# V56 distillation forms", "",
             f"Mode `{summary['mode']}`. Dev: {data['n_dev_runs_found']}/12 runs, "
             f"{data['n_dev_points']} checkpoints ({3*data['n_dev_points']} capability observations); "
             f"{data['n_complete_dev_runs']} runs have all four checkpoints. "
             f"Test: {data['n_test_runs_found']} runs, {data['n_test_points']} checkpoints.", "",
             "Tc = completion_tokens_seen; E = Tc / registered D_U_completion; "
             "delta = post_training - dense, in nats. A cluster is (student, U, data_seed), "
             "so all checkpoints/capabilities from a run stay together. Metrics pool held-out "
             "checkpoints; bias = prediction - actual. Missing runs are listed in JSON.", "",
             "L0 is standardized separately per capability, equally over available dev students; "
             "log N uses the two dev students. These fixed, pretreatment-only references are "
             "used in both CV variants and test prediction. Regression column scaling uses "
             "training data only. Ridge = 0.001; intercepts are unpenalized. F1/F2 columns "
             "are scaled without centering to preserve zero response at zero exposure.", "",
             "T_star is chosen from {17.5k, 35k, 70k, 140k, 280k} by pooled training SSE "
             "across capabilities, separately in every fold. Final fits use all dev points. "
             "Test outcomes never select a model, scale, timescale or subset.", "",
             "Part A: each cell is **MAE/signed bias**. P is the total coefficient count "
             "across all three capabilities; F2 additionally selects one five-choice timescale. "
             "LOCO leaves one run out; LOSO leaves one student out.", "",
             "| Form | P | LOCO math | LOCO code | LOCO qa | LOSO math | LOSO code | LOSO qa |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, result in summary["part_a"].items():
        cells = [cell(result[cv]["metrics"][c]) for cv in ("loco", "loso") for c in CAPS]
        lines.append(f"| {key} | {result['fit_all_dev']['n_params']} | " + " | ".join(cells) + " |")
    lines += ["", "Bases: t = log(1+Tc/35000), e = log(1+E), s = 1-exp(-Tc/T_star). "
              "T-only = a+b*t; E-only = a+b*e; surface = a+b*t+c*e+d*z; "
              "F1 = (a+lambda*z)*e; F2 = (a+lambda*z)*s+(b+mu*z)*e.", "",
              "With two dev students, standardized L0 and log N are identical up to sign "
              "within each capability; Part A cannot distinguish them on dev. LOSO has only "
              "one student in training, so descriptor effects are not separately identifiable "
              "from base coefficients. Ranks and ridge solutions are recorded in JSON.", "",
              "Part B uses primary L0 and **equal fitted coefficient counts**, with a trade "
              "between offsets and capability slopes. Shared shape: three capability offsets "
              "plus q shared slopes (q=1 for E-only, 2 for T+E, 4 for F2). Per-capability: "
              "one shared offset plus q+2 capability-specific slopes. E-only has all three "
              "E slopes (P=4). T+E gives one capability both slopes and each other capability "
              "one slope (P=5; 12 subsets). F2 gives each capability one of {s,z*s} and one "
              "of {e,z*e} (P=7; 64 subsets). These are reduced, nonnested per-capability "
              "models, not unrestricted fits. Subsets are chosen by training SSE within "
              "each fold; the extra discrete search is reported and is not an equal search "
              "budget. Both F2 models also select one five-choice T_star. Unrestricted "
              "per-capability offsets/slopes would require 6, 9 and 15 coefficients.", "",
              "Part B: LOCO MAE; gain = shared minus per-capability. `>0.02` uses "
              "unrounded nats and is a descriptive threshold, not a significance test.", "",
              "| Structure | P shared/per | Capability | Shared MAE | Per-cap MAE | Gain | >0.02 |",
              "|---|---:|---|---:|---:|---:|:---:|"]
    for kind, result in summary["part_b"].items():
        count = result["shared"]["fit_all_dev"]["n_params"]
        for c in (*CAPS, "macro"):
            v = result["comparison"][c]
            nums = ["NA" if v[k] is None else f"{v[k]:.3f}" for k in ("shared_mae", "per_capability_mae", "gain")]
            lines.append(f"| {kind} | {count}/{count} | {c} | " + " | ".join(nums) +
                         f" | {'yes' if v['beyond_0_02'] else 'no' if v['gain'] is not None else 'NA'} |")
    lines += ["", f"Per-capability gain >0.02 nats: {', '.join(summary['part_b_gain_over_0_02']) or 'none'}.",
              f"No gain >0.02 nats: {', '.join(summary['part_b_no_gain_over_0_02']) or 'none'}.", ""]
    if summary["mode"] == "all":
        lines.append("Test results (separate; dev fits frozen before reading test evals):")
        if not data["n_test_points"]:
            lines.append("No `_p2v2test` checkpoints present.")
        for key, result in summary["part_a"].items():
            if "test" in result:
                lines.append(f"- A {key}: " + "; ".join(f"{c} {cell(result['test']['metrics'][c])}" for c in CAPS))
        for kind, result in summary["part_b"].items():
            for mode in ("shared", "per_capability"):
                if "test" in result[mode]:
                    lines.append(f"- B {kind}/{mode}: " + "; ".join(
                        f"{c} {cell(result[mode]['test']['metrics'][c])}" for c in CAPS))
        lines.append("")
    lines += ["JSON contains every candidate fit, fold, prediction, parameter count, "
              "input point, dense-loss source, config hash and SHA256 of all analysis inputs. "
              "Results are deterministic for a fixed set of input files.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("dev", "all"), default="dev")
    args = parser.parse_args(argv)
    global reg
    FILES.clear()
    JSON_CACHE.clear()
    RUNS.clear()
    record_file(__file__, "analysis_source")
    record_file(ROOT / "analysis/v50_p2v2.py", "data_access_reference")
    reg = read_json(REG, "pool_completion_token_register")
    dev = [p for st in DEV_STUDENTS for U in (75, 450) for seed in (11, 12, 13)
           for p in points(st, U, seed, "p2v2")]
    found = sum(r["found"] for r in RUNS)
    complete = sum(r["complete_four_points"] for r in RUNS)
    print(f"Dev runs found: {found}/12; points: {len(dev)}; complete four-point runs: {complete}.", flush=True)
    counts = {s: n_params(s) for s in DEV_STUDENTS} if dev else {}
    print("N_S (v50 meta count): " + ", ".join(f"{s}={n}" for s, n in counts.items()), flush=True)
    refs = descriptor_refs(dev, counts) if dev else {}
    summary = {"version": 56, "mode": args.mode, "refs": refs,
               "protocol": {"caps": CAPS, "ridge": RIDGE, "T_ref": T_REF, "T_star_grid": T_GRID,
                            "cluster": "(student, U, data_seed)", "bias": "prediction - actual",
                            "selection": "training SSE only, reselected inside every CV fold",
                            "metric_weighting": "equal checkpoint weight; macro equals capability mean",
                            "test_used_for_fit": False,
                            "parameter_count": "continuous coefficients; discrete search counted separately",
                            "part_b_accounting": {
                                "shared": "3 capability offsets + q shared shape slopes",
                                "per_capability": "1 common offset + q+2 selected capability slopes",
                                "q": {"E-only": 1, "T+E": 2, "F2": 4},
                                "matched_counts": {"E-only": 4, "T+E": 5, "F2": 7},
                                "unrestricted_capability_offsets_and_slopes": {
                                    "E-only": 6, "T+E": 9, "F2": 15},
                                "subsets": {
                                    "E-only": "all three capability E slopes (1 choice)",
                                    "T+E": "one capability gets T and E; others one each (12 choices)",
                                    "F2": "each capability gets one of s,z*s and one of e,z*e (64 choices)"},
                                "limitation": "reduced nonnested models; equal coefficient count, unequal discrete search budget",
                                "threshold": "shared MAE minus per-capability MAE > 0.02 nats; descriptive"},
                            "runtime": {"numpy": np.__version__, "device": "CPU; parameter count on meta"}},
               "data": {"n_dev_runs_expected": 12, "n_dev_runs_found": found,
                        "n_complete_dev_runs": complete, "n_dev_points": len(dev),
                        "dev_points": dev, "test_points": [], "n_test_points": 0,
                        "n_test_runs_found": 0, "runs": RUNS}, "part_a": {}, "part_b": {},
               "part_b_gain_over_0_02": [], "part_b_no_gain_over_0_02": []}
    if not dev:
        summary["status"] = "no dev points; fits/CV unavailable"
    for kind, descriptor in FORMS if dev else ():
        key = form_key(kind, descriptor)
        summary["part_a"][key] = evaluate_model(dev, [], refs, fit_a, predict_a,
                                               (kind, descriptor), student_cv=True)
    for kind in ("E-only", "T+E", "F2") if dev else ():
        shared = evaluate_model(dev, [], refs, fit_b, predict_b, (kind, "shared"))
        per_cap = evaluate_model(dev, [], refs, fit_b, predict_b, (kind, "per_capability"))
        assert shared["fit_all_dev"]["n_params"] == per_cap["fit_all_dev"]["n_params"]
        comparison = {}
        for c in (*CAPS, "macro"):
            a, b = shared["loco"]["metrics"][c]["mae"], per_cap["loco"]["metrics"][c]["mae"]
            gain = a - b if a is not None and b is not None else None
            comparison[c] = {"shared_mae": a, "per_capability_mae": b, "gain": gain,
                             "beyond_0_02": gain > 0.02 if gain is not None else None}
            if c != "macro" and gain is not None:
                bucket = "part_b_gain_over_0_02" if gain > 0.02 else "part_b_no_gain_over_0_02"
                summary[bucket].append(f"{kind}/{c}")
        summary["part_b"][kind] = {"shared": shared, "per_capability": per_cap, "comparison": comparison}
    # Deliberately read test data only AFTER all fitting and model selection.
    if args.mode == "all":
        test = []
        for directory in sorted((ROOT / "results/v12-distill").glob("*/gpt-5.6-luna_full_*_p2v2test_lora_dseed*")):
            match = re.fullmatch(r"gpt-5\.6-luna_full_(\d+)_p2v2test_lora_dseed(\d+)", directory.name)
            if match:
                test.extend(points(directory.parent.name, int(match[1]), int(match[2]), "p2v2test"))
        summary["data"].update(test_points=test, n_test_points=len(test),
                               n_test_runs_found=sum(r["found"] for r in RUNS if r["suffix"] == "p2v2test"))
        if dev and test:
            for student in sorted({p["student"] for p in test} - counts.keys()):
                counts[student] = n_params(student)
            for family, predictor in ((summary["part_a"].values(), predict_a),
                                      ([r[m] for r in summary["part_b"].values()
                                        for m in ("shared", "per_capability")], predict_b)):
                for result in family:
                    rows = prediction_rows(test, predictor(result["fit_all_dev"], test, refs))
                    result["test"] = {"metrics": summarize(rows), "predictions": rows,
                                      "by_student": {s: summarize([r for r in rows if r["student"] == s])
                                                     for s in sorted({p["student"] for p in test})}}
    summary["files_read"] = [FILES[k] for k in sorted(FILES)]
    summary["input_manifest_sha256"] = hashlib.sha256(
        json.dumps(summary["files_read"], sort_keys=True).encode()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (OUT / "summary.md").write_text(markdown(summary))
    print("Part A: MAE/bias (nats); columns math, code, qa.")
    for key, result in summary["part_a"].items():
        print(f"  {key:14s} LOCO " + "  ".join(cell(result["loco"]["metrics"][c]) for c in CAPS)
              + " | LOSO " + "  ".join(cell(result["loso"]["metrics"][c]) for c in CAPS))
    print("Part B: shared -> per-capability LOCO MAE; gain >0.02 nats marked *.")
    for kind, result in summary["part_b"].items():
        p = result["shared"]["fit_all_dev"]["n_params"]
        cells = []
        for c in CAPS:
            v = result["comparison"][c]
            cells.append(f"{c} NA" if v["gain"] is None else
                         f"{c} {v['shared_mae']:.3f}->{v['per_capability_mae']:.3f}"
                         f" ({v['gain']:+.3f}){'*' if v['beyond_0_02'] else ''}")
        print(f"  {kind:6s} P={p}/{p}: " + "; ".join(cells))
    print("Part B matched counts trade capability offsets for selected slopes; subset search is extra.")
    if args.mode == "all":
        print(f"Test (never fit): {summary['data']['n_test_runs_found']} runs; {summary['data']['n_test_points']} points.")
    print(f"Wrote {path_label(OUT / 'summary.json')} and {path_label(OUT / 'summary.md')}; "
          f"{len(FILES)} files hashed.")
    return summary


if __name__ == "__main__":
    main()
