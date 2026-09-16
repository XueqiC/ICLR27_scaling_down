#!/usr/bin/env python3
"""V69 quantization confirmation: CPU develop/freeze/compare, forward-only runner.

    python -B analysis/v69_quant_confirm.py --selftest
    python -B analysis/v69_quant_confirm.py develop
    python -B analysis/v69_quant_confirm.py freeze
    scripts/run_v69_grid.sh --dry-run
    python -B analysis/v69_quant_confirm.py compare

All 54 previously unblinded V54 cells are development data here. V55's original
split and its DEV_DONE/FREEZE_COMMITTED sentinels are historical, read-only
artifacts. V69 releases its 21 new cells only after its own write-once freeze.
Authoring modes and every --dry-run use CPU only and never load a model.
The explicit ``measure`` mode is reserved for scripts/run_v69_grid.sh.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import tempfile

sys.dont_write_bytecode = True
try:
    from . import v55_quant_group_fit as v55
except ImportError:
    import v55_quant_group_fit as v55

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / "results/v55-quant-group/register.json").is_file()),
            Path(__file__).resolve().parents[1])
OUT_REL = Path("results/v69-quant-confirm")
DATA_REL = Path("results/v54-quant-group")
CAPS = v55.CAPS
DEV_TAGS = v55.DEV_TAGS
BITS = (3, 4, 5)
GROUPS = (64, 128, 256)
DEV_CONFIGS = tuple(f"b{b}_g{g}" for b in BITS for g in GROUPS)
NEW_TAG = "pythia-1.4b@step112000"
TEST_TAGS = ("pythia-410m@step143000", "pythia-1.4b@step16000", NEW_TAG)
TERMS = {"low_order_2d": (0, 1, 2, 3, 4), "bilinear": (0, 1, 2, 3),
         "bit_only": (0, 1)}
METHODS = ("low_order_2d", "bilinear", "same_input_interpolation", "bit_only", "median", "zero")
N_PARAMS = dict(zip(METHODS, (20, 16, 36, 8, 9, 0)))
LABELS = dict(zip(METHODS, ("2-D surface", "Bilinear", "Piecewise interpolation",
                          "Bit only", "Per-config median", "Zero")))
SELECTION_RULE = (
    "Per capability: minimize LOSO macro MAE (equal weight per held-out state). "
    "Candidates within 0.02 nats of the minimum, inclusive, tie; choose the "
    "fewest coefficients, then lower MAE, then declared candidate order. "
    "All six candidates, including median and zero, are eligible. No test-based selection."
)
BOUNDARY_RULE = (
    "Piecewise bilinear interpolation in x=log2(qmax) and v=log2(g/128) on the "
    "3x3 measured grid. For g<64 use g=64,128; for g>256 use g=128,256, "
    "linearly extrapolating in v at the same b. Floor the extrapolated predicted "
    "signed dL at zero: max(0, dL_extrapolated). Interior predictions remain signed. "
    "Apply this identical rule to same-input interpolation and per-config medians. "
    "No bit extrapolation; only b=3,4,5 is in the confirmation protocol."
)
STANDARDIZATION_RULE = (
    "V55 population mean/std of [log(N0), L0c, log(D0)], pooled over training "
    "rows and capabilities; constant scale=1. Refit on five training states in "
    "each LOSO fold. u=log2(qmax)-training mean(log2(qmax)); v=log2(g/128). "
    "N0 counts transformer matrices excluding embeddings/head (V55), while "
    "V54 quantization includes embeddings/head. D0=step*2097152."
)
PROTOCOL = {"model_dtype": "bf16", "probe_seed": 0, "n_probe": 128,
            "probe_half": "odd-indexed [1::2]",
            "loss_protocol": "v10._measure_capability_losses (completion-token-weighted CE)",
            "quantization": "symmetric round-to-nearest (torch.round, ties to even)"}
CODE_INPUTS = ("analysis/v69_quant_confirm.py", "analysis/v55_quant_group_fit.py",
               "analysis/v54_quant_group.py", "analysis/v10_quantization.py",
               "analysis/v6_capability_geometry.py", "analysis/model_registry.py",
               "scripts/run_v69_grid.sh")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def label(root, path):
    path = Path(path).absolute()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def write_outputs(outputs, exclusive=False):
    """Publish complete files atomically; hard-link publication never clobbers a freeze."""
    if exclusive:
        for path in outputs:
            v55.require_new(path)
    for path, value in outputs.items():
        require(not path.is_symlink(), f"Refusing symlink output: {path}")
        text = value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            if exclusive:
                os.link(temporary, path)
            else:
                os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"WROTE {path}")


def test_cells():
    return [{"state": tag, "config": f"b{b}_g{g}",
             "test_set": "new_state_boundary" if tag == NEW_TAG else "development_state_boundary"}
            for tag in TEST_TAGS for b in BITS for g in (32, 512)] + [
                {"state": NEW_TAG, "config": f"b{b}_g128", "test_set": "new_state_interior"}
                for b in BITS]


def check_protocol(table, tag, context, probe_hash=None):
    meta = table.get("_meta", {})
    expected = {**PROTOCOL, "model_tag": tag, "hf_id": "EleutherAI/" + tag.split("@")[0],
                "revision": tag.split("@")[1]}
    for key, value in expected.items():
        require(meta.get(key) == value, f"{context}: incompatible protocol {key}")
    actual = meta.get("probe_sha256", "")
    require(re.fullmatch(r"[0-9a-f]{64}", actual) is not None, f"{context}: missing probe SHA256")
    if probe_hash is not None:
        require(actual == probe_hash, f"{context}: different probes")
    return actual


def load_dev(root):
    states, rows, hashes, probe_hash = [], [], {}, None
    for tag in DEV_TAGS:
        path = v55.measurement_path(root / DATA_REL, tag)
        table, sha = v55.read_json(path)
        require(all(c in table for c in ("dense", *DEV_CONFIGS)),
                f"Development requires all 54 cells: missing dense/config in {path}")
        probe_hash = check_protocol(table, tag, path, probe_hash)
        dense = v55.checked_losses(table["dense"], path)
        state = v55.state_input(tag, dense, path)
        state["path"] = label(root, path)
        states.append(state)
        hashes[label(root, path)] = sha
        for config in DEV_CONFIGS:
            losses = v55.checked_losses(table[config], f"{path}: {config}")
            for cap in CAPS:
                rows.append({"state": tag, "config": config, "capability": cap,
                             "phi_raw": v55.raw_features(state, cap), "dL": losses[cap] - dense[cap]})
    return states, rows, hashes, probe_hash


def grid_weights(config):
    """Local linear weights, with only the two nearest g anchors at a boundary."""
    x, v = v55.coordinates(config)
    xs = [v55.coordinates(f"b{b}_g128")[0] for b in BITS]
    require(xs[0] <= x <= xs[-1], f"Bit extrapolation is outside the V69 protocol: {config}")
    bi = int(np.clip(np.searchsorted(xs, x, side="right") - 1, 0, 1))
    gi = 0 if v <= 0 else 1
    tx = (x - xs[bi]) / (xs[bi + 1] - xs[bi])
    vs = (-1., 0., 1.)
    tv = (v - vs[gi]) / (vs[gi + 1] - vs[gi])
    return {f"b{BITS[bi + i]}_g{GROUPS[gi + j]}": float(wx * wv)
            for i, wx in enumerate((1 - tx, tx)) for j, wv in enumerate((1 - tv, tv))}, abs(v) > 1


def interpolate(anchors, config):
    weights, outside = grid_weights(config)
    value = float(sum(weight * anchors[key] for key, weight in weights.items()))
    return max(0., value) if outside else value


def design(rows, stats, method):
    z = v55.phi([r["phi_raw"] for r in rows], stats)
    if method in TERMS:
        x, v = np.array([v55.coordinates(r["config"]) for r in rows]).T
        terms = v55.low_order_terms(x, v, stats["u_center"])[:, TERMS[method]]
        return (terms[:, :, None] * z[:, None, :]).reshape(len(rows), -1)
    onehot = np.array([[float(r["config"] == c) for c in DEV_CONFIGS] for r in rows])
    if method == "same_input_interpolation":
        return (onehot[:, :, None] * z[:, None, :]).reshape(len(rows), -1)
    return onehot if method == "median" else np.empty((len(rows), 0))


def diagnostics(x):
    if x.shape[1] == 0:
        return {"n_rows": len(x), "n_coefficients": 0, "design_rank": 0,
                "condition_number": None, "condition_status": "not applicable (fixed zero)",
                "singular_values": []}
    s = np.linalg.svd(x, compute_uv=False)
    tol = s[0] * max(x.shape) * np.finfo(float).eps
    rank = int(np.sum(s > tol))
    return {"n_rows": len(x), "n_coefficients": x.shape[1], "design_rank": rank,
            "condition_number": float(s[0] / s[-1]) if rank == x.shape[1] else None,
            "condition_status": "finite" if rank == x.shape[1] else "infinite (rank deficient)",
            "singular_values": s.tolist(), "rank_tolerance": float(tol)}


def fit_all(rows):
    stats, models = v55.standardization(rows), {}
    for cap in CAPS:
        cr = [r for r in rows if r["capability"] == cap]
        y = np.array([r["dL"] for r in cr])
        models[cap] = {}
        for method in METHODS:
            x = design(cr, stats, method)
            model = {"diagnostics": diagnostics(x)}
            if method in TERMS:
                model.update(coefficients=v55.ridge_fit(x, y).reshape(-1, 4).tolist(),
                             terms=[v55.TERM_NAMES[i] for i in TERMS[method]])
            elif method == "same_input_interpolation":
                beta = v55.ridge_fit(x, y).reshape(9, 4)
                model["anchors"] = dict(zip(DEV_CONFIGS, beta.tolist()))
            elif method == "median":
                model["anchors"] = {c: float(np.median([r["dL"] for r in cr if r["config"] == c]))
                                    for c in DEV_CONFIGS}
            else:
                model["value"] = 0.
            models[cap][method] = model
    return stats, models


def predict_all(models, raw, config, stats):
    z = v55.phi(raw, stats)
    x, v = v55.coordinates(config)
    terms = v55.low_order_terms(x, v, stats["u_center"])
    estimates = {m: float(terms[list(indices)] @ np.asarray(models[m]["coefficients"]) @ z)
                 for m, indices in TERMS.items()}
    anchors = {c: float(np.dot(beta, z)) for c, beta in models["same_input_interpolation"]["anchors"].items()}
    estimates.update(same_input_interpolation=interpolate(anchors, config),
                     median=interpolate(models["median"]["anchors"], config), zero=0.)
    require(all(np.isfinite(v) for v in estimates.values()), "Nonfinite prediction")
    return {m: estimates[m] for m in METHODS}


def scores(rows):
    result = {}
    for method in METHODS:
        result[method] = {}
        for cap in CAPS:
            cr = [r for r in rows if r["capability"] == cap]
            by_state = {tag: float(np.mean([abs(r["predictions"][method] - r["dL"])
                                          for r in cr if r["state"] == tag]))
                        for tag in sorted({r["state"] for r in cr})}
            result[method][cap] = {
                "n": len(cr), "n_states": len(by_state), "state_mae": by_state,
                "macro_mae": float(np.mean(list(by_state.values()))) if by_state else None,
                "mae": float(np.mean([abs(r["predictions"][method] - r["dL"]) for r in cr])) if cr else None}
    return result


def select(table):
    selected = {}
    for cap in CAPS:
        best = min(table[m][cap]["macro_mae"] for m in METHODS)
        tied = [m for m in METHODS if table[m][cap]["macro_mae"] <= best + .02 + 1e-12]
        winner = min(tied, key=lambda m: (N_PARAMS[m], table[m][cap]["macro_mae"], METHODS.index(m)))
        selected[cap] = {"candidate": winner, "n_coefficients": N_PARAMS[winner],
                         "loso_macro_mae": table[winner][cap]["macro_mae"],
                         "minimum_loso_macro_mae": best, "tied_candidates": tied}
    return selected


def loso(rows):
    folds, predictions = [], []
    for tag in DEV_TAGS:
        train = [r for r in rows if r["state"] != tag]
        held = [r for r in rows if r["state"] == tag]
        stats, models = fit_all(train)
        pr = [{**r, "predictions": predict_all(models[r["capability"]], r["phi_raw"], r["config"], stats)}
              for r in held]
        predictions.extend(pr)
        folds.append({"held_out": tag, "train_states": [s for s in DEV_TAGS if s != tag],
                      "n_train_cells": len(train) // 3, "n_held_out_cells": len(held) // 3,
                      "standardization": stats, "models": models, "scores": scores(pr)})
    return {"scores": scores(predictions), "folds": folds, "predictions": predictions}


def tail_report():
    return {
        "rule": "V54 zero-pads each last contiguous input group, computes absmax using real entries "
                "plus neutral zeros, then trims. qmax=2**(b-1)-1. Grouping is along input width, "
                "not vocabulary/output width; embeddings and LM head have input width hidden_size.",
        "architectures": {size: {"hidden_size": a["hidden_size"], "intermediate_size": a["intermediate_size"],
                                  "test_state": size in ("410m", "1.4b"),
                                  "input_widths": {str(width): {str(g): {
                                      "n_groups": (width + g - 1) // g,
                                      "tail_real_entries": width % g, "padding": (-width) % g}
                                      for g in (32, 512)}
                                      for width in (a["hidden_size"], a["intermediate_size"])}}
                          for size, a in v55.ARCHITECTURES.items() if size in ("160m", "410m", "1.4b")},
        "conclusion": "All confirmation input widths (1024/4096 and 2048/8192) divide by 32 and 512: "
                      "no partial groups. The development-only 160M width 768 would have 256 real "
                      "tail entries and 256 padding zeros at g=512; its MLP input width 3072 divides exactly."}


def develop_markdown(d):
    lines = ["# V69 quantization development", "", "54 unblinded state/config cells; 162 capability responses. "
             "Six whole-state LOSO folds, 45 training and 9 held-out cells per fold/capability.",
             "", SELECTION_RULE, "", STANDARDIZATION_RULE, "", BOUNDARY_RULE,
             "", "Ridge 1e-3 penalizes all regression coefficients including intercepts. "
             "Median has nine scalar anchors (no ridge); zero has no coefficients. "
             "Condition numbers below are for the unregularized design, not the normal matrix. "
             "Median diagnostics describe its configuration indicator design.",
             "", "| Candidate | Coefficients | Math macro MAE | Code macro MAE | QA macro MAE |",
             "|---|---:|---:|---:|---:|"]
    for m in METHODS:
        lines.append(f"| {LABELS[m]} | {N_PARAMS[m]} | " + " | ".join(
            f"{d['loso']['scores'][m][c]['macro_mae']:.6f}" for c in CAPS) + " |")
    lines += ["", "Selections: " + "; ".join(f"{c}: {d['selected'][c]['candidate']}" for c in CAPS) + ".",
              "", "| Capability | Candidate | Full rank / coefficients | Condition number | Fold rank range | Fold condition range |",
              "|---|---|---:|---:|---:|---:|"]
    for cap in CAPS:
        for m in METHODS:
            diag = d["models"][cap][m]["diagnostics"]
            folds = [f["models"][cap][m]["diagnostics"] for f in d["loso"]["folds"]]
            ranks = [f["design_rank"] for f in folds]
            conditions = [f["condition_number"] for f in folds if f["condition_number"] is not None]
            condition = f"{diag['condition_number']:.4f}" if diag["condition_number"] is not None else diag["condition_status"]
            span = f"{min(conditions):.4f}–{max(conditions):.4f}" if len(conditions) == 6 else "See JSON (undefined or rank deficient)"
            lines.append(f"| {cap} | {LABELS[m]} | {diag['design_rank']} / {N_PARAMS[m]} | {condition} | "
                         f"{min(ranks)}–{max(ranks)} | {span} |")
    lines += ["", "Three measured bit levels remove V55's two-level u² aliasing; "
              "ranks and condition numbers are measured rather than assumed. Full singular spectra "
              "and every fold's coefficients/scaler are in develop.json.", "",
              d["tail_groups"]["rule"], "", d["tail_groups"]["conclusion"], "",
              "Confirmation panel: two development states and the new 1.4B@112000 state, each "
              "at b=3/4/5 × g=32/512 (18 cells), plus the new state at g=128 (3 cells). "
              "Every candidate is frozen for all 21 cells. No V54 sentinel is created or changed.", ""]
    return "\n".join(lines)


def develop(root, out):
    v55.require_new(out / "freeze.json")
    states, rows, hashes, probe_hash = load_dev(root)
    register_path = root / "results/v55-quant-group/register.json"
    old, old_sha = v55.read_json(register_path)
    require(old["ridge"]["lambda"] == v55.RIDGE == 1e-3 and old["ridge"]["penalize_intercept"], "V55 ridge changed")
    require([s["tag"] for s in old["dev_states"]] == list(DEV_TAGS), "Unexpected V55 states")
    old_rows = {(r["state"], r["config"], r["capability"]): r for r in rows}
    for r in old["dev_rows"]:
        require(r == old_rows[(r["state"], r["config"], r["capability"])], "Original V55 development input changed")
    hashes[label(root, register_path)] = old_sha
    stats, models = fit_all(rows)
    cv = loso(rows)
    d = {"schema_version": 1, "development_status": "all 54 V54 cells previously unblinded",
         "selection_rule": SELECTION_RULE, "boundary_rule": BOUNDARY_RULE,
         "response": "signed dL_c = config loss minus same-file dense loss, nats",
         "dev_states": states, "dev_configs": list(DEV_CONFIGS), "dev_rows": rows,
         "n_dev_cells": 54, "n_dev_capability_rows": len(rows), "methods": list(METHODS),
         "n_coefficients": N_PARAMS, "standardization": stats, "standardization_rule": STANDARDIZATION_RULE,
         "feature_names": v55.FEATURE_NAMES, "coefficient_order": "term or anchor, then phi",
         "ridge": {"lambda": v55.RIDGE, "penalize_intercept": True},
         "models": models, "loso": cv, "selected": select(cv["scores"]),
         "test_cells": test_cells(), "measurement_protocol": {**PROTOCOL, "probe_sha256": probe_hash},
         "tail_groups": tail_report(), "input_sha256": hashes,
         "dev_subset_sha256": object_digest({"states": states, "rows": rows}),
         "code_sha256": {p: digest(root / p) for p in CODE_INPUTS}}
    write_outputs({out / "develop.json": d, out / "develop.md": develop_markdown(d)})
    return d


def new_dense(root):
    paths = [root / f"results/v53-prune-dev/dense_{NEW_TAG}.json",
             root / f"results/v53-prune-dev/dense_dir_{NEW_TAG}/dense.json",
             root / "results/v6-capability-geometry" / NEW_TAG.replace("@", "--") / "prune_losses.json"]
    for path in paths:
        if not path.is_file():
            continue
        data, sha = v55.read_json(path)
        require(data.get("_tag", data.get("tag", NEW_TAG)) == NEW_TAG, f"Wrong dense state: {path}")
        require(data.get("_revision", data.get("revision", "step112000")) == "step112000", f"Wrong dense revision: {path}")
        dense = v55.checked_losses(data.get("dense", data.get("1.0", data)), path)
        state = v55.state_input(NEW_TAG, dense, path)
        state["path"] = label(root, path)
        return state, {label(root, path): sha}
    raise FileNotFoundError(f"Missing {NEW_TAG} dense losses in V53 or V6; freeze requires all 21 cells")


def default_hub_cache():
    hf_home = Path(os.environ.get("HF_HOME", str(Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface")))
    return Path(os.environ.get("HF_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub")))).expanduser()


def cached_weights(hub_cache):
    """Inspect cached config/index/tensor headers, without torch or weight loading."""
    repo = hub_cache / "models--EleutherAI--pythia-1.4b"
    ref = repo / "refs/step112000"
    revision = ref.read_text().strip()
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "Invalid cached revision")
    snapshot = repo / "snapshots" / revision
    config_path = snapshot / "config.json"
    config, _ = v55.read_json(config_path)
    require(all(config.get(k) == v for k, v in v55.ARCHITECTURES["1.4b"].items()), "Cached architecture mismatch")
    files = [ref, config_path]
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        manifest, _ = v55.read_json(index)
        weight_map = manifest["weight_map"]
        require(bool(weight_map), "Empty cached weight index")
        names = sorted(set(weight_map.values()))
        files.append(index)
    elif (snapshot / "model.safetensors").is_file():
        names, weight_map = ["model.safetensors"], None
    else:
        # Pythia revisions can also use the original PyTorch serialization.
        index = snapshot / "pytorch_model.bin.index.json"
        if index.is_file():
            manifest, _ = v55.read_json(index)
            names, weight_map = sorted(set(manifest["weight_map"].values())), None
            files.append(index)
        else:
            names, weight_map = ["pytorch_model.bin"], None
    widths, tensors = set(), {}
    for name in names:
        require(Path(name).name == name, "Invalid cached shard filename")
        path = snapshot / name
        require(path.is_file() and path.stat().st_size > 0, f"Missing cached weights: {path}")
        files.append(path)
        if path.suffix == ".safetensors":
            with path.open("rb") as stream:
                length = struct.unpack("<Q", stream.read(8))[0]
                require(0 < length < 100_000_000, "Invalid safetensors header size")
                header = json.loads(stream.read(length))
            for tensor, spec in header.items():
                if tensor == "__metadata__":
                    continue
                require(spec["data_offsets"][1] + 8 + length <= path.stat().st_size, f"Truncated shard: {path}")
                tensors[tensor] = name
                if len(spec["shape"]) >= 2:
                    require(len(spec["shape"]) == 2, "V54 grouped quantizer requires 2-D matrices")
                    widths.add(spec["shape"][1])
    if weight_map is not None:
        require(all(tensors.get(t) == f for t, f in weight_map.items()), "Missing indexed tensor in cached shards")
    if widths:
        require(widths == {2048, 8192}, f"Unexpected cached input widths: {widths}")
    tokenizer = snapshot / "tokenizer.json"
    require(tokenizer.is_file(), f"Missing cached tokenizer: {tokenizer}")
    files.append(tokenizer)
    return {"state": NEW_TAG, "revision": revision, "snapshot": str(snapshot),
            "weights_cached": True, "weight_files": names, "matrix_input_widths_from_headers": sorted(widths),
            "verification": "Complete local shards and config; safetensors index/header coverage checked when available. No model loaded.",
            "file_sha256": {str(p): digest(p) for p in files}}


def novelty_audit(root):
    """Reject prior quantization evidence for the new state in repository artifacts."""
    hashes, sections = {}, []
    markers = (NEW_TAG, NEW_TAG.replace("@", "--"))
    for directory in sorted((root / "results").glob("*quant*")):
        if directory.name == OUT_REL.name or not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.json")):
            raw = path.read_text()
            require(not any(m in label(root, path) or m in raw for m in markers),
                    f"New state already appears in prior quantization artifacts: {path}")
            hashes[label(root, path)] = digest(path)
    # V59 also fits quantization inside a multi-operation analysis.
    path = root / "results/v59-shared-structure/summary.json"
    if path.is_file():
        data, sha = v55.read_json(path)
        for part in ("part1", "part2", "part3"):
            if "quantization" in data.get(part, {}):
                require(not any(m in json.dumps(data[part]["quantization"]) for m in markers),
                        f"New state already appears in V59 {part} quantization")
                sections.append(f"{label(root, path)}:{part}.quantization")
        hashes[label(root, path)] = sha
    require(hashes, "No prior quantization artifacts available to audit")
    return {"state": NEW_TAG, "absent_from_prior_quantization_artifacts": True,
            "scope": "All JSON paths/content under results/*quant* except V69, plus V59 quantization sections. "
                     "This verifies available repository evidence, not unrecorded external fits.",
            "sections": sections, "input_sha256": hashes}


def verify_development(root, d):
    require(d["selection_rule"] == SELECTION_RULE and d["boundary_rule"] == BOUNDARY_RULE,
            "Frozen protocol differs from this implementation")
    require(d["test_cells"] == test_cells() and d["methods"] == list(METHODS), "Panel/candidates changed")
    require(d["selected"] == select(d["loso"]["scores"]), "Selection rule not followed")
    states, rows, _, probe_hash = load_dev(root)
    require(object_digest({"states": states, "rows": rows}) == d["dev_subset_sha256"],
            "Development cells/dense inputs changed; appending test keys alone is permitted")
    require(probe_hash == d["measurement_protocol"]["probe_sha256"], "Development probes changed")
    for rel, sha in d["code_sha256"].items():
        require(digest(root / rel) == sha, f"Code changed since develop: {rel}")
    register = "results/v55-quant-group/register.json"
    require(digest(root / register) == d["input_sha256"][register], "V55 register changed")


def freeze(root, out, hub_cache):
    path = out / "freeze.json"
    v55.require_new(path)
    d, develop_sha = v55.read_json(out / "develop.json")
    verify_development(root, d)
    # Check presence only: test losses must never be used to choose or fit anything.
    for cell in test_cells():
        source = v55.measurement_path(root / DATA_REL, cell["state"])
        table = v55.read_json(source)[0] if source.is_file() else {}
        require(cell["config"] not in table, f"Refusing retrospective freeze: already measured {cell}")
    audit = novelty_audit(root)
    new_state, dense_hashes = new_dense(root)
    cache = cached_weights(hub_cache)
    states = {s["tag"]: s for s in d["dev_states"]}
    states[NEW_TAG] = new_state
    predictions = []
    for cell in test_cells():
        state = states[cell["state"]]
        for cap in CAPS:
            raw = v55.raw_features(state, cap)
            estimates = predict_all(d["models"][cap], raw, cell["config"], d["standardization"])
            predictions.append({**cell, "capability": cap, "L0": state["L0"][cap], "phi_raw": raw,
                                "predictions": estimates,
                                "selected_candidate": d["selected"][cap]["candidate"],
                                "selected_prediction": estimates[d["selected"][cap]["candidate"]]})
    # Persist all candidates as well as a self-contained selected model per capability.
    f = {"schema_version": 1, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
         "selection_rule": SELECTION_RULE, "boundary_rule": BOUNDARY_RULE,
         "methods": list(METHODS), "n_test_cells": 21, "n_capability_predictions": 63,
         "test_cells": test_cells(), "states": [states[tag] for tag in TEST_TAGS],
         "standardization": d["standardization"], "standardization_rule": STANDARDIZATION_RULE,
         "models": d["models"], "selected": {
             cap: {**d["selected"][cap], "form": d["selected"][cap]["candidate"],
                   "model": d["models"][cap][d["selected"][cap]["candidate"]],
                   "standardization": d["standardization"]} for cap in CAPS},
         "predictions": predictions, "measurement_protocol": d["measurement_protocol"],
         "tail_groups": d["tail_groups"], "new_state_audit": audit, "cached_weights": cache,
         "provenance": {"develop_sha256": develop_sha, "dev_subset_sha256": d["dev_subset_sha256"],
                        "code_sha256": d["code_sha256"],
                        "input_sha256": {**d["input_sha256"], **audit["input_sha256"], **dense_hashes,
                                         **cache["file_sha256"], label(root, out / "develop.json"): develop_sha}}}
    write_outputs({path: f}, exclusive=True)
    return f


def load_freeze(root, out):
    f, sha = v55.read_json(out / "freeze.json")
    d, dev_sha = v55.read_json(out / "develop.json")
    require(dev_sha == f["provenance"]["develop_sha256"], "Develop artifact changed after freeze")
    verify_development(root, d)
    require(f["schema_version"] == 1 and f["test_cells"] == test_cells(), "Invalid freeze panel")
    for key in ("selection_rule", "boundary_rule", "methods", "measurement_protocol"):
        require(f[key] == d[key], f"Frozen {key} differs from develop")
    expected = {(r["state"], r["config"], c) for r in test_cells() for c in CAPS}
    actual = [(r["state"], r["config"], r["capability"]) for r in f["predictions"]]
    require(len(actual) == len(expected) and set(actual) == expected, "Incomplete/duplicate frozen predictions")
    require(all(set(r["predictions"]) == set(METHODS) for r in f["predictions"]), "Missing frozen candidate")
    require(f["models"] == d["models"] and f["standardization"] == d["standardization"], "Frozen models differ from develop")
    for cap in CAPS:
        chosen = d["selected"][cap]
        require(f["selected"][cap] == {**chosen, "form": chosen["candidate"],
                                     "model": d["models"][cap][chosen["candidate"]],
                                     "standardization": d["standardization"]}, "Frozen selected form changed")
    for r in f["predictions"]:
        require(r["selected_candidate"] == d["selected"][r["capability"]]["candidate"], "Frozen selection changed")
        require(all(np.isfinite(v) for v in r["predictions"].values()), "Nonfinite frozen predictions")
        require(r["selected_prediction"] == r["predictions"][r["selected_candidate"]], "Selected prediction mismatch")
    return f, sha


def measurement_plan(root, frozen=None):
    plan = []
    for cell in test_cells():
        path = v55.measurement_path(root / DATA_REL, cell["state"])
        table = v55.read_json(path)[0] if path.is_file() else {}
        if table and frozen is not None:
            check_protocol(table, cell["state"], path, frozen["measurement_protocol"]["probe_sha256"])
        measured = cell["config"] in table
        if measured:
            v55.checked_losses(table[cell["config"]], path)
            v55.checked_losses(table.get("dense"), path)
        plan.append({**cell, "path": label(root, path), "status": "skip" if measured else "pending",
                     "dense_pending": "dense" not in table})
    return plan


def measure(root, out, dry_run=False):
    if (out / "freeze.json").is_file():
        frozen, _ = load_freeze(root, out)
    else:
        require(dry_run, "freeze.json is required before any measurement")
        frozen = None
        print("Freeze absent: execution will be blocked until freeze.json exists.")
    plan = measurement_plan(root, frozen)
    for row in plan:
        print(f"{row['status'].upper():7} {row['state']} {row['config']} -> {row['path']}")
    pending = [r for r in plan if r["status"] == "pending"]
    dense_tags = sorted({r["state"] for r in pending if r["dense_pending"]})
    print(f"21 planned quantized cells: {len(pending)} pending, {21 - len(pending)} skipped; "
          f"{len(dense_tags)} same-file dense evaluations pending: {', '.join(dense_tags) or 'none'}.")
    if dry_run or not pending:
        return plan
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid) is not None,
            "Set CUDA_VISIBLE_DEVICES to one GPU UUID in the caller; V69 never chooses a GPU")
    # This is the ONLY model/GPU import in the operational workflow.
    try:
        from . import v54_quant_group as v54
    except ImportError:
        import v54_quant_group as v54
    for tag in TEST_TAGS:
        configs = [r["config"] for r in pending if r["state"] == tag]
        if configs:
            v54.run_quantization(tag, v54.parse_configs(",".join(configs)), device="cuda:0",
                                 model_dtype="bf16", n_probe=128, reference_device="cpu",
                                 out=v55.measurement_path(root / DATA_REL, tag).parent)
    return plan


def regime(value):
    return "near_zero" if abs(value) < .1 else "clear_damage"


def compare_markdown(d):
    lines = ["# V69 quantization confirmation", "", f"Measured {d['n_measured_cells']}/21 cells; "
             f"{len(d['rows'])}/63 capability responses. All errors use frozen predictions.", "",
             "Near-zero: measured |dL| < 0.1 nats. Clear-damage: |dL| >= 0.1, including negative "
             "responses; this is the complementary magnitude regime. Counts differ by capability. "
             "MAE below weights measured cells equally; JSON also reports macro MAE by state and "
             "separate development-state boundary, new-state boundary, and new-state interior panels.", "",
             "Selections (development LOSO only): " + "; ".join(
                 f"{c}: {d['selected'][c]['candidate']}" for c in CAPS) + ".", ""]
    for group, table in d["regimes"].items():
        lines += [f"## {group}", "", "| Candidate | Math MAE (n) | Code MAE (n) | QA MAE (n) |",
                  "|---|---:|---:|---:|"]
        for m in METHODS:
            cells = [f"{table[m][c]['mae']:.6f} ({table[m][c]['n']})" if table[m][c]["n"] else "N/A (0)" for c in CAPS]
            lines.append(f"| {LABELS[m]} | " + " | ".join(cells) + " |")
        lines.append("")
    if d["missing"]:
        lines += ["Incomplete panel; no final confirmation claim. Missing:", "",
                  *[f"- {r['state']} {r['config']}" for r in d["missing"]], ""]
    lines += ["The new state's frozen L0 comes from its pre-existing V53/V6 dense evaluation. "
              "Targets subtract V54's same-file dense reference; each difference from frozen L0 is "
              "reported in compare.json. No refitting or test-based selection occurs.", ""]
    return "\n".join(lines)


def compare_latex(d):
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\small",
             r"\caption{V69 quantization confirmation. MAE in nats from predictions frozen before "
             r"measurement; 21 state/configuration cells per capability. Near-zero means measured "
             r"$|\Delta L|<0.1$; clear-damage is its complement, including negative changes. "
             r"Parentheses give measured counts; bold marks the candidate selected by development "
             r"LOSO, including the 0.02-nat tie rule, independently of these test outcomes.}",
             r"\label{tab:quant_confirm}", r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
             r"Regime & Candidate & Math & Code & QA \\", r"\midrule"]
    for i, (group, table) in enumerate(d["regimes"].items()):
        if i:
            lines.append(r"\addlinespace")
        for j, m in enumerate(METHODS):
            values = []
            for c in CAPS:
                metric = table[m][c]
                value = f"{metric['mae']:.4f} ({metric['n']})" if metric["n"] else "--- (0)"
                values.append(r"\textbf{" + value + "}" if d["selected"][c]["candidate"] == m else value)
            lines.append(" & ".join([group.replace("_", " ").capitalize() if j == 0 else "", LABELS[m], *values]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if d["missing"]:
        lines.append(r"\par\smallskip Incomplete: " + str(d["n_measured_cells"]) + r"/21 cells measured.")
    lines += [r"\end{table}", ""]
    return "\n".join(lines)


def compare(root, out, allow_partial=False):
    f, sha = load_freeze(root, out)
    plan = measurement_plan(root, f)
    missing = [r for r in plan if r["status"] == "pending"]
    require(not missing or allow_partial, f"Missing {len(missing)}/21 measurements; compare writes nothing "
            "until complete (use --allow-partial for an explicitly incomplete report)")
    rows, hashes, tables = [], {}, {}
    for r in f["predictions"]:
        path = v55.measurement_path(root / DATA_REL, r["state"])
        if not path.is_file():
            continue
        if r["state"] not in tables:
            tables[r["state"]], hashes[label(root, path)] = v55.read_json(path)
        table = tables[r["state"]]
        if r["config"] not in table:
            continue
        cap = r["capability"]
        dense = v55.checked_losses(table["dense"], path)[cap]
        loss = v55.checked_losses(table[r["config"]], path)[cap]
        change = loss - dense
        rows.append({**r, "dense": dense, "loss": loss, "dL": change, "regime": regime(change),
                     "dense_difference_from_prediction_input": dense - r["L0"],
                     "absolute_errors": {m: abs(r["predictions"][m] - change) for m in METHODS}})
    result = {"schema_version": 1, "complete": not missing, "n_expected_cells": 21,
              "n_measured_cells": 21 - len(missing), "rows": rows, "missing": missing,
              "selected": f["selected"], "regime_rule": "near_zero: abs(measured dL)<0.1; clear_damage: abs(dL)>=0.1",
              "regimes": {"all": scores(rows), **{g: scores([r for r in rows if r["regime"] == g])
                                                   for g in ("near_zero", "clear_damage")}},
              "test_sets": {s: {"n_expected_cells": sum(c["test_set"] == s for c in test_cells()),
                                 "scores": scores([r for r in rows if r["test_set"] == s])}
                            for s in dict.fromkeys(r["test_set"] for r in test_cells())},
              "provenance": {"freeze_sha256": sha, "measurement_sha256": hashes}}
    write_outputs({out / "compare.json": result, out / "compare.md": compare_markdown(result),
                   root / "paper/paper/tables/quant_confirm.tex": compare_latex(result)})
    return result


def selftest():
    """CPU-only independent numerical checks and a disposable end-to-end panel."""
    import contextlib
    import copy
    import io
    import shutil
    from unittest.mock import patch

    def expect_error(function, message):
        try:
            function()
        except (ValueError, FileExistsError, FileNotFoundError) as exc:
            assert message in str(exc), str(exc)
        else:
            raise AssertionError(f"Expected error: {message}")

    def put(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    anchors = {f"b{b}_g{g}": float(v) for b in BITS for g, v in zip(GROUPS, (1, 3, 6))}
    for b in BITS:
        assert interpolate(anchors, f"b{b}_g32") == 0.
        assert interpolate(anchors, f"b{b}_g512") == 9.
        assert interpolate(anchors, f"b{b}_g128") == 3.
        assert np.isclose(interpolate(anchors, f"b{b}_g96"), 1 + 2 * np.log2(96 / 64))
    anchors["b4_g128"] = -2.
    assert interpolate(anchors, "b4_g128") == -2.  # No interior clipping.
    assert interpolate(anchors, "b3_g512") == 9.  # Other b anchors cannot affect it.
    expect_error(lambda: interpolate(anchors, "b2_g128"), "Bit extrapolation")
    assert regime(.099999) == regime(-.099999) == "near_zero"
    assert regime(.1) == regime(-.1) == "clear_damage"
    trial = {m: {c: {"macro_mae": 1.} for c in CAPS} for m in METHODS}
    for cap in CAPS:
        trial["low_order_2d"][cap]["macro_mae"] = .2
        trial["bit_only"][cap]["macro_mae"] = .22
    assert all(v["candidate"] == "bit_only" for v in select(trial).values())
    trial["bit_only"]["math"]["macro_mae"] = .220001
    assert select(trial)["math"]["candidate"] == "low_order_2d"

    with tempfile.TemporaryDirectory(prefix="v69-selftest-") as temporary:
        root, sink = Path(temporary), io.StringIO()
        out = root / OUT_REL
        for rel in CODE_INPUTS:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, root / rel)
        for i, tag in enumerate(DEV_TAGS):
            dense = {c: 3 + .3 * ci - .09 * i + .013 * (i + 1)**2 * (ci + 1)
                     for ci, c in enumerate(CAPS)}
            table = {"dense": dense, "_meta": {**PROTOCOL, "model_tag": tag,
                     "hf_id": "EleutherAI/" + tag.split("@")[0], "revision": tag.split("@")[1],
                     "probe_sha256": "a" * 64}}
            for config in DEV_CONFIGS:
                x, v = v55.coordinates(config)
                table[config] = {c: dense[c] + (.12 + .02 * i) * (x - 3)**2
                                 + .02 * v * (i + 1) - .04 * ci for ci, c in enumerate(CAPS)}
            put(v55.measurement_path(root / DATA_REL, tag), table)
        states, rows, _, _ = load_dev(root)
        assert len(states) == 6 and len(rows) == 162
        assert any(r["dL"] < 0 for r in rows)
        put(root / "results/v55-quant-group/register.json", {
            "ridge": {"lambda": .001, "penalize_intercept": True}, "dev_states": states,
            "dev_rows": [r for r in rows if r["config"] in v55.DEV_CONFIGS]})
        put(root / f"results/v53-prune-dev/dense_{NEW_TAG}.json", {
            "_tag": NEW_TAG, "_revision": "step112000", **dict.fromkeys(CAPS, 2.5)})
        cache = root / "cache"
        repo = cache / "models--EleutherAI--pythia-1.4b"
        snapshot = repo / "snapshots" / ("b" * 40)
        (repo / "refs").mkdir(parents=True)
        (repo / "refs/step112000").write_text("b" * 40)
        put(snapshot / "config.json", v55.ARCHITECTURES["1.4b"])
        put(snapshot / "tokenizer.json", {})
        header, offset = {}, 0
        for i, width in enumerate((2048, 8192)):
            header[f"weight{i}"] = {"dtype": "F32", "shape": [2, width],
                                    "data_offsets": [offset, offset + width * 8]}
            offset += width * 8
        raw_header = json.dumps(header).encode()
        (snapshot / "model.safetensors").write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + bytes(offset))

        stats, models = fit_all(rows)
        raw = np.array([r["phi_raw"] for r in rows])
        np.testing.assert_allclose(stats["center"], raw.mean(0))
        np.testing.assert_allclose(stats["scale"], raw.std(0))
        for cap in CAPS:
            cr = [r for r in rows if r["capability"] == cap]
            y = np.array([r["dL"] for r in cr])
            for m in METHODS:
                assert models[cap][m]["diagnostics"]["design_rank"] == N_PARAMS[m]
            for m in (*TERMS, "same_input_interpolation"):
                x = design(cr, stats, m)
                independent = np.linalg.lstsq(np.vstack((x, np.sqrt(.001) * np.eye(x.shape[1]))),
                                              np.r_[y, np.zeros(x.shape[1])], rcond=None)[0]
                actual = models[cap][m].get("coefficients", list(models[cap][m].get("anchors", {}).values()))
                np.testing.assert_allclose(np.asarray(actual).ravel(), independent, rtol=1e-8, atol=1e-10)
                assert np.isclose(models[cap][m]["diagnostics"]["condition_number"], np.linalg.cond(x))
            old = [r for r in cr if r["config"] in v55.DEV_CONFIGS]
            assert np.linalg.matrix_rank(design(old, v55.standardization(old), "low_order_2d")) == 16
        cv = loso(rows)
        changed = copy.deepcopy(rows)
        for r in changed:
            if r["state"] == DEV_TAGS[0]:
                r["dL"] += 20
                r["phi_raw"][1] += 200
        changed_cv = loso(changed)
        assert cv["folds"][0]["models"] == changed_cv["folds"][0]["models"]
        assert cv["folds"][0]["standardization"] == changed_cv["folds"][0]["standardization"]
        assert all(f["held_out"] not in f["train_states"] and f["n_train_cells"] == 45 for f in cv["folds"])

        with contextlib.redirect_stdout(sink):
            plan = measure(root, out, dry_run=True)
            assert len(plan) == 21 and not out.exists()
            expect_error(lambda: measure(root, out), "freeze.json is required")
            d = develop(root, out)
            first_source = v55.measurement_path(root / DATA_REL, TEST_TAGS[0])
            original = first_source.read_text()
            poisoned = json.loads(original)
            poisoned["b3_g32"] = "test response must not be read"
            put(first_source, poisoned)
            expect_error(lambda: freeze(root, out, cache), "already measured")
            first_source.write_text(original)
            frozen = freeze(root, out, cache)
            assert len(frozen["predictions"]) == 63
            assert all(set(r["predictions"]) == set(METHODS) for r in frozen["predictions"])
            before = (out / "freeze.json").read_bytes()
            expect_error(lambda: freeze(root, out, cache), "Refusing to overwrite")
            expect_error(lambda: develop(root, out), "Refusing to overwrite")
            expect_error(lambda: compare(root, out), "Missing 21/21")
            assert not (out / "compare.json").exists()
            for i, cell in enumerate(test_cells()):
                path = v55.measurement_path(root / DATA_REL, cell["state"])
                if path.is_file():
                    table = json.loads(path.read_text())
                else:
                    table = {"dense": dict.fromkeys(CAPS, 2.55), "_meta": {
                        **PROTOCOL, "model_tag": NEW_TAG, "hf_id": "EleutherAI/pythia-1.4b",
                        "revision": "step112000", "probe_sha256": "a" * 64}}
                table[cell["config"]] = {c: table["dense"][c] + (.05 if (i + ci) % 2 else -.2)
                                         for ci, c in enumerate(CAPS)}
                put(path, table)
            result = compare(root, out)
            assert result["complete"] and len(result["rows"]) == 63
            assert (out / "freeze.json").read_bytes() == before
            for group, table in result["regimes"].items():
                for m in METHODS:
                    for cap in CAPS:
                        cr = [r for r in result["rows"] if r["capability"] == cap
                              and (group == "all" or r["regime"] == group)]
                        assert table[m][cap]["n"] == len(cr)
                        assert table[m][cap]["mae"] == np.mean([abs(r["dL"] - r["predictions"][m]) for r in cr])
            assert all(np.isclose(r["dense_difference_from_prediction_input"], .05)
                       for r in result["rows"] if r["state"] == NEW_TAG)
            assert all(r["status"] == "skip" for r in measure(root, out, dry_run=True))
            assert r"\begin{table}[!htbp]" in (root / "paper/paper/tables/quant_confirm.tex").read_text()
            # Preserve append-only measurement compatibility, but reject dev mutations.
            table = json.loads(first_source.read_text())
            table["b3_g64"]["math"] += 1
            put(first_source, table)
            expect_error(lambda: load_freeze(root, out), "Development cells/dense inputs changed")
            table["b3_g64"]["math"] -= 1
            # Restore exactly, avoiding floating-point roundtrip changes.
            table["b3_g64"] = json.loads(original)["b3_g64"]
            del table["b3_g32"]
            put(first_source, table)
            partial = compare(root, out, allow_partial=True)
            assert not partial["complete"] and partial["n_measured_cells"] == 20

        # Reuse the real V54 quantizer on explicit CPU tensors, including tails.
        try:
            from . import v54_quant_group as v54
        except ImportError:
            import v54_quant_group as v54
        import torch
        generator = torch.Generator(device="cpu").manual_seed(69)
        for width in (17, 513, 768, 1024, 2048, 3072, 4096, 8192):
            sample = torch.randn(2, width, generator=generator, device="cpu")
            sample[0].zero_()
            for dtype in (torch.float32, torch.bfloat16):
                weight = sample.to(dtype)
                original_weight = weight.clone()
                for b in BITS:
                    for g in (32, 512):
                        actual = v54.fake_quantize_grouped(weight, b, g)
                        expected = torch.cat([v54.fake_quantize_per_output_channel(weight[:, start:start + g], b)
                                              for start in range(0, width, g)], dim=1)
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        assert actual.shape == weight.shape and actual.dtype == dtype and actual.device.type == "cpu"
                assert torch.equal(weight, original_weight)
        with contextlib.redirect_stdout(sink), patch.object(v54, "run_quantization") as runner:
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-12345678-1234-1234-1234-123456789abc"}):
                measure(root, out)  # Mocked: verifies resume dispatch without a model or GPU.
            assert runner.call_count == 1
            assert runner.call_args.args == (TEST_TAGS[0], [(3, 32)])
            assert runner.call_args.kwargs["reference_device"] == "cpu"
        assert not torch.cuda.is_initialized()
    print("PASS (CPU): ranks/condition, independent ridge, signed boundary interpolation, LOSO isolation, "
          "selection ties, freeze/no-overwrite, all-candidate regime scoring, resume dispatch, V54 fp32/bf16 tails.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", choices=("develop", "freeze", "compare", "measure"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, help="Default: ROOT/results/v69-quant-confirm")
    parser.add_argument("--hub-cache", type=Path, default=default_hub_cache())
    parser.add_argument("--dry-run", action="store_true", help="List planned work; no writes or model/GPU imports")
    parser.add_argument("--allow-partial", action="store_true", help="compare only: label an incomplete report")
    parser.add_argument("--selftest", action="store_true", help="CPU synthetic rows, workflow and quantizer tail tests")
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if not args.mode:
        parser.error("a mode or --selftest is required")
    root = args.root.resolve()
    out = args.out.resolve() if args.out else root / OUT_REL
    try:
        if args.mode == "measure":
            measure(root, out, args.dry_run)
        elif args.dry_run:
            names = {"develop": ("develop.json", "develop.md"), "freeze": ("freeze.json (write-once)",),
                     "compare": ("compare.json", "compare.md", "paper/paper/tables/quant_confirm.tex")}
            print(f"CPU {args.mode}: planned outputs {', '.join(names[args.mode])}; output directory {out}")
            if args.mode == "develop":
                print("Read 54 development cells; fit six candidates; six state folds; select by macro MAE/tie rule.")
            else:
                measure(root, out, dry_run=True)
        elif args.mode == "develop":
            develop(root, out)
        elif args.mode == "freeze":
            freeze(root, out, args.hub_cache)
        else:
            compare(root, out, args.allow_partial)
    except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
