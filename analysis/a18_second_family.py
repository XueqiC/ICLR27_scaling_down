#!/usr/bin/env python3
"""A18: measurement-efficiency confirmation on a second family (OLMo-2), pre-registered in
results/a18-second-family/prereg.md.

Six development states and four test states of OLMo-2 (1B and 7B stage-1 checkpoints whose revision
names carry the pretraining token count). The development states are pruned at four densities
{0.9, 0.8, 0.7, 0.6}; the test states at three densities fixed in advance, {0.85, 0.75, 0.65}. Every
form is fitted on the full development grid (24 configurations) and on the reduced grid {0.9, 0.7}
(12 configurations, half the development configuration measurements) with the same labels, then
frozen before any test state is pruned. Losses are measured on the V6 probes (odd half, 64 per
capability) and on a new item set per capability that no earlier probe, sample, teacher trace or
few-shot exemplar used; per-item loss sums and token counts are stored, with the tokens evaluated
and the GPU seconds of every measurement, dense anchors included.

Stages, in order
  --items            build the new item sets (items.json)
  --hash             download each checkpoint into the project cache and record shard hashes; refuse
                     two states with identical weights (hashes.json)
  --run [--states]   GPU: measure dense and pruned losses (measurements/<tag>.json); test states are
                     refused until freeze.json exists; --purge removes a state's weight shards after
                     its measurement is written (public checkpoints, hashes kept)
  --freeze           fit every form on the development grids and write the test-state predictions
                     (freeze.json); refuses to overwrite
  --score            score the frozen predictions on the measured test states (score.json, summary.md)
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/a18-second-family"
CACHE = ROOT / "envs/hf_cache_a18"
os.environ.setdefault("HF_HUB_CACHE", str(CACHE / "hub"))

import numpy as np  # noqa: E402
from analysis import v53_prune_dev as p53  # noqa: E402

CAPS = ("math", "code", "qa")
DEV_DENSITIES = (0.9, 0.8, 0.7, 0.6)
REDUCED = (0.9, 0.7)
TEST_DENSITIES = (0.85, 0.75, 0.65)
N_PROBE, N_NEW, ITEM_SEED = 128, 64, 2027
REPOS = {"1B": "allenai/OLMo-2-0425-1B", "7B": "allenai/OLMo-2-1124-7B"}
STATES = {  # tag: (size, revision, role)
    "olmo2-1b@399B": ("1B", "stage1-step190000-tokens399B", "dev"),
    "olmo2-1b@1993B": ("1B", "stage1-step950000-tokens1993B", "dev"),
    "olmo2-1b@3608B": ("1B", "stage1-step1720000-tokens3608B", "dev"),
    "olmo2-7b@391B": ("7B", "stage1-step93000-tokens391B", "dev"),
    "olmo2-7b@1947B": ("7B", "stage1-step464000-tokens1947B", "dev"),
    "olmo2-7b@3507B": ("7B", "stage1-step836000-tokens3507B", "dev"),
    "olmo2-1b@1196B": ("1B", "stage1-step570000-tokens1196B", "test"),
    "olmo2-1b@2811B": ("1B", "stage1-step1340000-tokens2811B", "test"),
    "olmo2-7b@1167B": ("7B", "stage1-step278000-tokens1167B", "test"),
    "olmo2-7b@2727B": ("7B", "stage1-step650000-tokens2727B", "test"),
}
FORMS = ("power", "quadratic", "strength", "median", "per_density")
KAPPA_GRID = [round(k, 2) for k in np.arange(-0.9, 2.01, 0.05)]
LAMBDA_GRID = (0.0, 1e-3, 1e-2, 1e-1, 1.0)   # one ridge grid for every linear-in-feature form, chosen by leave-one-state-out


def read(p):
    return json.loads(Path(p).read_text())


def write_new(p, obj):
    p = Path(p)
    if p.exists():
        raise SystemExit(f"{p} exists; frozen artifacts are never overwritten")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))


def tokens_of(revision):
    return float(revision.rsplit("tokens", 1)[1].rstrip("B")) * 1e9


# ---------------------------------------------------------------- items
def build_items():
    from datasets import load_dataset
    from analysis.v6_capability_geometry import build_probes
    from analysis import v71_qa_scope as v71
    from analysis.v34_c17_scope_audit import normalize
    from analysis import v15_accuracy_link as v15
    probes = build_probes(N_PROBE)
    used = {c: {p["prompt"] for p in probes[c]} for c in CAPS}
    rng = np.random.default_rng(ITEM_SEED)
    items = {}
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    pool = [i for i in range(len(ds)) if f"Problem: {ds[i]['problem']}\nSolution:" not in used["math"]]
    pick = rng.choice(pool, size=N_NEW, replace=False)
    items["math"] = [{"index": int(i), "prompt": f"Problem: {ds[int(i)]['problem']}\nSolution:", "completion": " " + ds[int(i)]["solution"]} for i in pick]
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    pool = [i for i in range(len(ds)) if f"# Task: {ds[i]['text']}\n# Write a Python function.\n" not in used["code"]]
    pick = rng.choice(pool, size=N_NEW, replace=False)
    items["code"] = [{"index": int(i), "prompt": f"# Task: {ds[int(i)]['text']}\n# Write a Python function.\n", "completion": ds[int(i)]["code"]} for i in pick]
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "datasets"
    qa, provenance = v71.cached_dataset(cache, "framolfese/2WikiMultihopQA", None, "validation")
    register = read(ROOT / "results/v71-qa-scope/register.json")["sets"]["2wiki_new"]
    excluded = set(register["indices"]) | set(register["excluded_indices"])
    excluded |= {it["index"] for it in read(ROOT / "results/p1-qa-behaviour/plan.json")["items"]}
    questions, sources = v71.teacher_questions(ROOT / "results/traces-pilot")
    exemplars = {normalize(e["prompt"].split("Question: ")[-1].split("\nAnswer")[0]) for e in v15.FEWSHOT["qa"]["exemplars"]}
    probe_questions = {normalize(p["prompt"].split("Question: ")[-1].split("\nAnswer")[0]) for p in probes["qa"]}
    order = v71.fresh_indices(len(qa), excluded, n=len(qa) - len(excluded), seed=ITEM_SEED)
    chosen = []
    for i in order:
        row = qa[i]
        q = normalize(row["question"])
        if q in questions or q in exemplars or q in probe_questions:
            continue
        probe = v71.two_wiki_probe(row)
        chosen.append({"index": int(i), "id": row.get("_id", row.get("id", str(i))), "prompt": probe["prompt"], "completion": probe["completion"]})
        if len(chosen) == N_NEW:
            break
    if len(chosen) != N_NEW:
        raise SystemExit("not enough fresh question-answering items")
    items["qa"] = chosen
    write_new(OUT / "items.json", {"seed": ITEM_SEED, "n_per_capability": N_NEW, "items": items,
                                   "exclusions": {"v6_probes": {c: len(used[c]) for c in CAPS}, "v71_register_and_excluded": len(excluded - {it["index"] for it in read(ROOT / "results/p1-qa-behaviour/plan.json")["items"]}),
                                                  "p1_items": 384, "teacher_questions": len(questions), "exemplars": len(exemplars), "qa_source": provenance}})
    print("items written:", {c: len(v) for c, v in items.items()})


# ---------------------------------------------------------------- hashes
def snapshot(size, revision):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(REPOS[size], revision=revision, allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.txt"]))


def hash_states(tags=None):
    prior = read(OUT / "hashes.json") if (OUT / "hashes.json").exists() else {}
    for tag, (size, revision, role) in STATES.items():
        if tags and tag not in tags or tag in prior:
            continue
        path = snapshot(size, revision)
        shards = sorted(p for p in path.rglob("*.safetensors"))
        digest = hashlib.sha256()
        for shard in shards:
            with open(shard, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    digest.update(chunk)
        prior[tag] = {"repo": REPOS[size], "revision": revision, "role": role, "shards": [p.name for p in shards],
                      "bytes": sum(p.stat().st_size for p in shards), "sha256": digest.hexdigest(), "path": str(path)}
        (OUT / "hashes.json").write_text(json.dumps(prior, indent=1))
        print(tag, prior[tag]["sha256"][:12], round(prior[tag]["bytes"] / 1e9, 1), "GB", flush=True)
    digests = [v["sha256"] for v in prior.values()]
    if len(set(digests)) != len(digests):
        raise SystemExit("two states resolve to identical weights")


# ---------------------------------------------------------------- measurement
def measure_items(model, tok, items, device):
    import torch
    from analysis.v6_capability_geometry import completion_loss
    out = {}
    with torch.no_grad():
        for cap, rows in items.items():
            recs = []
            for it in rows:
                loss, n = completion_loss(model, tok, it["prompt"], it["completion"], device)
                recs.append({"loss_sum": float(loss), "tokens": int(n)})
            out[cap] = {"items": recs, "mean_loss": sum(r["loss_sum"] for r in recs) / max(sum(r["tokens"] for r in recs), 1),
                        "tokens": sum(r["tokens"] for r in recs)}
    return out


def run_states(tags, device, purge):
    import torch
    from analysis.v6_capability_geometry import (build_probes, load_text_causal_lm, language_weight_parameters,
                                                  _sample_abs_weights, apply_global_magnitude_pruning)
    hashes = read(OUT / "hashes.json")
    new_items = read(OUT / "items.json")["items"]
    probes = {c: v[1::2] for c, v in build_probes(N_PROBE).items()}
    probe_items = {c: [{"prompt": p["prompt"], "completion": p["completion"]} for p in probes[c]] for c in CAPS}
    (OUT / "measurements").mkdir(parents=True, exist_ok=True)
    for tag in tags:
        size, revision, role = STATES[tag]
        target = OUT / "measurements" / f"{tag.replace('@', '__')}.json"
        if target.exists():
            print(tag, "measured"); continue
        if role == "test" and not (OUT / "freeze.json").exists():
            raise SystemExit(f"{tag} is a test state; freeze.json must exist before it is pruned")
        if tag not in hashes:
            raise SystemExit(f"{tag} not hashed")
        t0 = time.time()
        model, tok = load_text_causal_lm(REPOS[size], torch.bfloat16, revision)
        model.to(device).eval()
        params = language_weight_parameters(model)
        scope = sum(p.numel() for _, p in params)
        block = sum(p.numel() for n, p in params if ".layers." in n or ".blocks." in n)
        abs_sample = _sample_abs_weights(params)
        densities = DEV_DENSITIES if role == "dev" else TEST_DENSITIES
        thresholds = {d: float(np.quantile(abs_sample, 1 - d)) for d in densities}
        del abs_sample
        record = {"tag": tag, "repo": REPOS[size], "revision": revision, "role": role, "size": size, "D0": tokens_of(revision),
                  "N0_block_matrices": block, "prune_scope_parameters": scope, "sha256": hashes[tag]["sha256"],
                  "densities": list(densities), "load_seconds": round(time.time() - t0, 1), "measurements": {}}
        t1 = time.time()
        record["measurements"]["1.0"] = {"probes": measure_items(model, tok, probe_items, device), "new": measure_items(model, tok, new_items, device),
                                         "seconds": round(time.time() - t1, 1)}
        print(tag, "dense", {c: round(record["measurements"]["1.0"]["probes"][c]["mean_loss"], 3) for c in CAPS}, flush=True)
        reference = [p.detach().clone().cpu() for _, p in params]
        for d in densities:
            t2 = time.time()
            apply_global_magnitude_pruning(model, d, reference_weights=reference, threshold=thresholds[d])
            record["measurements"][str(d)] = {"probes": measure_items(model, tok, probe_items, device), "new": measure_items(model, tok, new_items, device),
                                              "seconds": round(time.time() - t2, 1), "threshold": thresholds[d]}
            print(tag, d, {c: round(record["measurements"][str(d)]["probes"][c]["mean_loss"], 3) for c in CAPS}, flush=True)
        record["total_seconds"] = round(time.time() - t0, 1)
        record["tokens_evaluated"] = sum(m[s][c]["tokens"] for m in record["measurements"].values() for s in ("probes", "new") for c in CAPS)
        target.write_text(json.dumps(record, indent=1))
        del model, tok, reference, params
        gc.collect(); torch.cuda.empty_cache()
        if purge:
            path = Path(hashes[tag]["path"])
            for shard in path.rglob("*.safetensors"):
                real = shard.resolve()
                real.unlink(missing_ok=True); shard.unlink(missing_ok=True)
            print(tag, "weights removed after measurement; hash kept", flush=True)


# ---------------------------------------------------------------- fitting
def dev_rows(states):
    rows = []
    for tag, rec in states.items():
        dense = rec["measurements"]["1.0"]["probes"]
        for d in rec["densities"]:
            m = rec["measurements"][str(d)]["probes"]
            for c in CAPS:
                rows.append({"tag": tag, "cap": c, "d": float(d), "phi_raw": p53.raw_features(rec["N0_block_matrices"], rec["D0"], dense[c]["mean_loss"]),
                             "L0": dense[c]["mean_loss"], "y": m[c]["mean_loss"] - dense[c]["mean_loss"]})
    return rows


def ridge(x, y, lam):
    """Ridge on the standardized design as the least-squares solution of the augmented system
    [x; sqrt(lam) I] beta = [y; 0]; lam = 0 is the minimum-norm least-squares fit, which also covers a design
    with a constant (all-zero after standardization) column, as when every state shares one N0."""
    k = x.shape[1]
    xa = np.vstack([x, np.sqrt(lam) * np.eye(k)]); ya = np.concatenate([y, np.zeros(k)])
    return np.linalg.lstsq(xa, ya, rcond=None)[0]


def loso_lambda(cr, z, y, d, design):
    """Leave-one-state-out choice of the ridge weight for one linear-in-feature form (the same grid for every form)."""
    tags = sorted({r["tag"] for r in cr}); tag_of = np.asarray([r["tag"] for r in cr])
    best = None
    for lam in LAMBDA_GRID:
        err = []
        for held in tags:
            tr, te = tag_of != held, tag_of == held
            xtr, xte = design(z[tr], d[tr]), design(z[te], d[te])
            beta = ridge(xtr, y[tr], lam)
            err.extend(np.abs(xte @ beta - y[te]).tolist())
        mae = float(np.mean(err))
        if best is None or mae < best[1]:
            best = (lam, mae)
    return best


def fit_forms(rows, cap, stats):
    cr = [r for r in rows if r["cap"] == cap]
    z = p53.standardize([r["phi_raw"] for r in cr], stats)
    y = np.asarray([r["y"] for r in cr]); d = np.asarray([r["d"] for r in cr])
    fits = {}
    # power form: the exponent from the gamma grid and the ridge weight from the lambda grid, both by leave-one-state-out
    best = None
    for gamma in p53.GAMMA_GRID:
        design = lambda zz, dd, g=gamma: zz * p53.shape(dd, g)[:, None]
        lam, mae = loso_lambda(cr, z, y, d, design)
        if best is None or mae < best["loso_mae"]:
            beta = ridge(design(z, d), y, lam)
            best = {"beta": beta.tolist(), "gamma": float(gamma), "lambda": lam, "loso_mae": mae}
    fits["power"] = best
    best = None; s = p53.shape(d, 1.0)
    for kappa in KAPPA_GRID:
        design = lambda zz, dd, k=kappa: zz * (p53.shape(dd, 1.0) + k * p53.shape(dd, 1.0) ** 2)[:, None]
        lam, mae = loso_lambda(cr, z, y, d, design)
        if best is None or mae < best["loso_mae"]:
            beta = ridge(design(z, d), y, lam)
            best = {"beta": beta.tolist(), "kappa": float(kappa), "lambda": lam, "loso_mae": mae}
    fits["quadratic"] = best
    best = None
    for gamma in p53.GAMMA_GRID:
        sh = p53.shape(d, gamma); A = float(sh @ y / (sh @ sh)); sse = float(np.sum((A * sh - y) ** 2))
        if best is None or sse < best["sse"]:
            best = {"A": A, "gamma": float(gamma), "sse": sse}
    fits["strength"] = best
    fits["median"] = {"anchors": {str(v): float(np.median(y[d == v])) for v in sorted(set(d.tolist()))}}
    # per-density regression: one ridge weight for all densities, by leave-one-state-out within each density
    best = None
    for lam in LAMBDA_GRID:
        err = []
        for v in sorted(set(d.tolist())):
            sel = d == v; tags = np.asarray([r["tag"] for r in cr])[sel]; xz = z[sel]; yv = y[sel]
            for held in sorted(set(tags.tolist())):
                tr = tags != held
                beta = ridge(xz[tr], yv[tr], lam)
                err.extend(np.abs(xz[~tr] @ beta - yv[~tr]).tolist())
        mae = float(np.mean(err))
        if best is None or mae < best[1]:
            best = (lam, mae)
    anchors = {str(v): ridge(z[d == v], y[d == v], best[0]).tolist() for v in sorted(set(d.tolist()))}
    fits["per_density"] = {"anchors": anchors, "lambda": best[0], "loso_mae": best[1]}
    return fits


def predict(form, fit, z, d):
    s = p53.shape(d, 1.0)
    if form == "power":
        return float(np.dot(fit["beta"], z) * p53.shape(d, fit["gamma"]))
    if form == "quadratic":
        return float(np.dot(fit["beta"], z) * (s + fit["kappa"] * s ** 2))
    if form == "strength":
        return float(fit["A"] * p53.shape(d, fit["gamma"]))
    if form == "median":
        return p53.linear_curve(fit["anchors"], d)
    if form == "per_density":
        return p53.linear_curve({k: float(np.dot(b, z)) for k, b in fit["anchors"].items()}, d)
    raise ValueError(form)


def pythia_zero_calibration(N0, D0, L0, c, d):
    """Diagnostic: the Pythia development power form applied with its own standardization, no refit."""
    models = read(ROOT / "results/s3-selection-validation/inputs/locked_models.json")["prune"]
    z = p53.standardize(p53.raw_features(N0, D0, L0), models["standardization"])
    p = models["models"][c]["power"]
    return float(np.dot(p["beta"], z) * p53.shape(d, p["gamma"]))


def freeze():
    states = {tag: read(OUT / "measurements" / f"{tag.replace('@', '__')}.json") for tag, (_, _, role) in STATES.items() if role == "dev"}
    missing = [t for t, (_, _, role) in STATES.items() if role == "dev" and t not in states]
    if missing:
        raise SystemExit(f"development states not measured: {missing}")
    rows = dev_rows(states)
    grids = {"full": rows, "reduced": [r for r in rows if r["d"] in REDUCED]}
    out = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "development_states": list(states), "grids": {}, "test_predictions": {}}
    for name, grid in grids.items():
        stats = p53.zstats(grid)
        out["grids"][name] = {"n_configurations": len({(r["tag"], r["d"]) for r in grid}), "standardization": stats,
                              "fits": {c: fit_forms(grid, c, stats) for c in CAPS}}
    # test states: inputs are their dense anchors, measured when the state is pruned; predictions are written as
    # functions of (N0, D0, L0) evaluated at scoring time from the frozen coefficients, so the frozen object is the
    # coefficient set above plus the density list; the Pythia zero-calibration diagnostic is fixed the same way.
    out["test_states"] = {t: {"size": s, "revision": r} for t, (s, r, role) in STATES.items() if role == "test"}
    out["test_densities"] = list(TEST_DENSITIES)
    out["forms"] = list(FORMS)
    write_new(OUT / "freeze.json", out)
    digest = hashlib.sha256((OUT / "freeze.json").read_bytes()).hexdigest()
    (OUT / "freeze.sha256").write_text(digest + "\n")
    print("frozen", digest[:16], {n: g["n_configurations"] for n, g in out["grids"].items()})


BOOT_N, BOOT_SEED, PRIMARY = 1000, 2027, ("power_reduced", "per_density_full")


def item_bootstrap(fz, per_state):
    """Evaluation-sample uncertainty only: items are resampled with replacement, paired across the dense anchor and
    every density of a state, the token-weighted response of each cell is recomputed, and the mean absolute error
    of the primary pair over the test states is recomputed on each resample. States are not resampled (four states
    are a point estimate, as the prereg says). Returns the 95 percent interval of each predictor's error and of the
    difference reduced-grid power form minus full-grid regression, per capability and item block."""
    rng = np.random.default_rng(BOOT_SEED)
    recs = {tag: read(OUT / "measurements" / f"{tag.replace('@', '__')}.json") for tag in per_state}
    out = {}
    for c in CAPS:
        for block in ("probes", "new"):
            draws = {k: [] for k in PRIMARY}; diffs = []
            n_items = len(recs[next(iter(recs))]["measurements"]["1.0"][block][c]["items"])
            cells = {tag: {str(d): np.asarray([[it["loss_sum"], it["tokens"]] for it in recs[tag]["measurements"][str(d)][block][c]["items"]])
                           for d in ["1.0"] + list(fz["test_densities"])} for tag in per_state}
            preds = {tag: {str(x["d"]): {k: x["pred"][k] for k in PRIMARY} for x in per_state[tag][c]["cells"]} for tag in per_state}
            for _ in range(BOOT_N):
                idx = rng.integers(0, n_items, n_items)
                err = {k: [] for k in PRIMARY}
                for tag in per_state:
                    dense = cells[tag]["1.0"][idx]; l0 = dense[:, 0].sum() / dense[:, 1].sum()
                    for d in fz["test_densities"]:
                        m = cells[tag][str(d)][idx]; y = m[:, 0].sum() / m[:, 1].sum() - l0
                        for k in PRIMARY:
                            err[k].append(abs(preds[tag][str(d)][k] - y))
                for k in PRIMARY:
                    draws[k].append(float(np.mean(err[k])))
                diffs.append(draws[PRIMARY[0]][-1] - draws[PRIMARY[1]][-1])
            out[f"{c}_{block}"] = {k: {"ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for k, v in draws.items()}
            out[f"{c}_{block}"]["difference"] = {"mean": float(np.mean(diffs)), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    return {"n_resamples": BOOT_N, "seed": BOOT_SEED, "pair": list(PRIMARY), "cells": out}


def score():
    fz = read(OUT / "freeze.json")
    assert hashlib.sha256((OUT / "freeze.json").read_bytes()).hexdigest() == (OUT / "freeze.sha256").read_text().strip()
    results = {"per_state": {}, "summary": {}}
    for tag in fz["test_states"]:
        path = OUT / "measurements" / f"{tag.replace('@', '__')}.json"
        if not path.exists():
            print("not measured:", tag); continue
        rec = read(path)
        dense = rec["measurements"]["1.0"]
        entry = {}
        for c in CAPS:
            L0 = dense["probes"][c]["mean_loss"]; L0_new = dense["new"][c]["mean_loss"]
            raw = p53.raw_features(rec["N0_block_matrices"], rec["D0"], L0)
            cells = []
            for d in fz["test_densities"]:
                m = rec["measurements"][str(d)]
                cell = {"d": d, "y_probes": m["probes"][c]["mean_loss"] - L0, "y_new": m["new"][c]["mean_loss"] - L0_new, "pred": {}}
                for grid, g in fz["grids"].items():
                    z = p53.standardize(raw, g["standardization"])
                    for form in FORMS:
                        cell["pred"][f"{form}_{grid}"] = predict(form, g["fits"][c][form], z, d)
                cell["pred"]["pythia_zero_calibration"] = pythia_zero_calibration(rec["N0_block_matrices"], rec["D0"], L0, c, d)
                cells.append(cell)
            entry[c] = {"cells": cells,
                        "mae_probes": {k: float(np.mean([abs(x["pred"][k] - x["y_probes"]) for x in cells])) for k in cells[0]["pred"]},
                        "mae_new": {k: float(np.mean([abs(x["pred"][k] - x["y_new"]) for x in cells])) for k in cells[0]["pred"]}}
        entry["cost"] = {"total_seconds": rec["total_seconds"], "tokens_evaluated": rec["tokens_evaluated"], "load_seconds": rec["load_seconds"]}
        results["per_state"][tag] = entry
    if results["per_state"]:
        keys = next(iter(results["per_state"].values()))["math"]["mae_probes"].keys()
        for c in CAPS:
            results["summary"][c] = {k: {"probes": float(np.mean([e[c]["mae_probes"][k] for e in results["per_state"].values()])),
                                         "new": float(np.mean([e[c]["mae_new"][k] for e in results["per_state"].values()]))} for k in keys}
        results["bootstrap"] = item_bootstrap(fz, results["per_state"])
    (OUT / "score.json").write_text(json.dumps(results, indent=1))
    lines = ["# A18 second-family confirmation: mean absolute error over the test states (nats per token)", "",
             "| Capability | Predictor | Probes | New items |", "|---|---|---:|---:|"]
    for c in CAPS:
        for k, v in results["summary"].get(c, {}).items():
            lines.append(f"| {c} | {k} | {v['probes']:.3f} | {v['new']:.3f} |")
    if "bootstrap" in results:
        lines += ["", f"Item bootstrap ({BOOT_N} resamples, seed {BOOT_SEED}; states fixed): 95 percent interval of the mean absolute error and of the "
                  f"difference {PRIMARY[0]} minus {PRIMARY[1]}", "", "| Capability | Items | " + " | ".join(PRIMARY) + " | difference [95%] |", "|---|---|---|---|---|"]
        for key, v in results["bootstrap"]["cells"].items():
            c, block = key.split("_", 1)
            lines.append(f"| {c} | {block} | " + " | ".join(f"[{v[k]['ci95'][0]:.3f}, {v[k]['ci95'][1]:.3f}]" for k in PRIMARY)
                         + f" | {v['difference']['mean']:+.3f} [{v['difference']['ci95'][0]:+.3f}, {v['difference']['ci95'][1]:+.3f}] |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


# ---------------------------------------------------------------- cost accounting and the paper table
FORM_WORDS = {"power": "Compact power form", "quadratic": "Quadratic strength form", "strength": "Strength only",
              "median": "Median density curve", "per_density": "Per-density regression"}


def cell_tokens(m):
    return sum(m[block][c]["tokens"] for block in ("probes", "new") for c in CAPS)


def cost():
    """GPU seconds and evaluation tokens of the two development grids (dense anchors and loading included) and of
    every test state; written to cost.json."""
    out = {"development": {}, "test": {}}
    tot = {"full": {"seconds": 0.0, "tokens": 0, "configurations": 0}, "reduced": {"seconds": 0.0, "tokens": 0, "configurations": 0}}
    for tag, (_, _, role) in STATES.items():
        path = OUT / "measurements" / f"{tag.replace('@', '__')}.json"
        if not path.exists():
            continue
        rec = read(path); m = rec["measurements"]
        anchor = {"seconds": rec["load_seconds"] + m["1.0"]["seconds"], "tokens": cell_tokens(m["1.0"])}
        if role == "dev":
            entry = {"anchor": anchor}
            for grid, dens in (("full", DEV_DENSITIES), ("reduced", REDUCED)):
                sec = anchor["seconds"] + sum(m[str(d)]["seconds"] for d in dens)
                tok = anchor["tokens"] + sum(cell_tokens(m[str(d)]) for d in dens)
                entry[grid] = {"seconds": sec, "tokens": tok, "configurations": len(dens)}
                tot[grid]["seconds"] += sec; tot[grid]["tokens"] += tok; tot[grid]["configurations"] += len(dens)
            out["development"][tag] = entry
        else:
            out["test"][tag] = {"anchor": anchor, "seconds": rec["total_seconds"], "tokens": rec["tokens_evaluated"],
                                "configurations": len(TEST_DENSITIES)}
    tot["reduced_over_full"] = {k: tot["reduced"][k] / tot["full"][k] for k in ("seconds", "tokens", "configurations")} if tot["full"]["seconds"] else None
    out["totals"] = tot
    (OUT / "cost.json").write_text(json.dumps(out, indent=1))
    return out


def render():
    """The appendix table: mean absolute error of every frozen predictor over the four test states at the three
    unseen densities, on the probes and on the new items; per-state errors go to per_state.md."""
    from analysis.paper_table_layout import house_style, table_layout
    sc = read(OUT / "score.json"); c = cost(); n_states = len(sc["per_state"])
    rows = []
    for grid, word in (("reduced", "Reduced (12)"), ("full", "Full (24)")):
        for form in FORMS:
            key = f"{form}_{grid}"
            vals = [sc["summary"][cap][key][b] for b in ("probes", "new") for cap in CAPS]
            rows.append(" & ".join([FORM_WORDS[form], word] + [f"{v:.3f}" for v in vals]) + r" \\")
        rows.append(r"\midrule")
    key = "pythia_zero_calibration"
    vals = [sc["summary"][cap][key][b] for b in ("probes", "new") for cap in CAPS]
    rows.append(" & ".join(["Pythia coefficients, no refit", "None"] + [f"{v:.3f}" for v in vals]) + r" \\")
    frac = c["totals"]["reduced_over_full"]
    caption = (f"Second-family confirmation on OLMo-2: mean absolute error in nats per token over {n_states} test states "
               "(1B and 7B intermediate checkpoints used in no fit) at three densities fixed before the freeze, 0.85, 0.75 "
               "and 0.65, on the 64 development probes per capability and on 64 new items per capability. Every predictor is "
               "fitted once on the six development states, on the reduced grid of two densities per state or on the full grid "
               "of four, with the same labels, the same tuning grids and the same leave-one-state-out ridge rule, and frozen "
               "before any test state was pruned. The reduced grid costs "
               f"{100 * frac['seconds']:.0f} percent of the full grid's GPU seconds and {100 * frac['tokens']:.0f} percent of "
               "its evaluation tokens, dense anchors and model loading included.")
    text = ("% Generated by analysis/a18_second_family.py --table; pre-registered confirmation, do not edit.\n"
            "\\begin{table}[tb]\n\\centering\\footnotesize\n"
            f"\\caption{{{caption}}}\n\\label{{tab:second-family}}\n"
            "\\begin{tabular*}{\\textwidth}{llrrrrrr}\n\\toprule\n"
            "& & \\multicolumn{3}{c}{Probes} & \\multicolumn{3}{c}{New items} \\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
            "Predictor & Development grid & Math & Code & QA & Math & Code & QA \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    text = house_style(table_layout(text))
    target = ROOT / "paper/paper/tables/second_family.tex"
    target.write_text(text)
    lines = ["# A18 per-state mean absolute error (nats per token), primary comparison and competitors", "",
             "| State | Capability | Items | " + " | ".join(f"{FORM_WORDS[f]} ({g})" for g in ("reduced", "full") for f in FORMS) + " | Pythia, no refit |",
             "|---|---|---|" + "---:|" * (2 * len(FORMS) + 1)]
    for tag, e in sc["per_state"].items():
        for cap in CAPS:
            for b, word in (("mae_probes", "probes"), ("mae_new", "new")):
                lines.append(f"| {tag} | {cap} | {word} | " + " | ".join(f"{e[cap][b][f'{f}_{g}']:.3f}" for g in ("reduced", "full") for f in FORMS) + f" | {e[cap][b]['pythia_zero_calibration']:.3f} |")
    (OUT / "per_state.md").write_text("\n".join(lines) + "\n")
    print(text); print("cost", json.dumps(c["totals"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", action="store_true"); ap.add_argument("--hash", action="store_true")
    ap.add_argument("--run", action="store_true"); ap.add_argument("--states", default="")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--purge", action="store_true")
    ap.add_argument("--freeze", action="store_true"); ap.add_argument("--score", action="store_true")
    ap.add_argument("--table", action="store_true", help="cost.json, per_state.md and paper/paper/tables/second_family.tex from score.json")
    a = ap.parse_args()
    tags = [t for t in a.states.split(",") if t] or list(STATES)
    if a.items:
        build_items()
    if a.hash:
        hash_states(tags if a.states else None)
    if a.run:
        run_states(tags, a.device, a.purge)
    if a.freeze:
        freeze()
    if a.score:
        score()
    if a.table:
        render()
