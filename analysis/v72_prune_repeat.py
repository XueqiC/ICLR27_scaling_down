#!/usr/bin/env python3
"""V72 pruning repeatability; authoring and dry runs are CPU-only.

    python3 -B analysis/v72_prune_repeat.py --selftest
    python3 -B analysis/v72_prune_repeat.py freeze --dry-run
    scripts/run_v72_measure.sh --dry-run
    CUDA_VISIBLE_DEVICES=GPU-<UUID> scripts/run_v72_measure.sh
    python3 -B analysis/v72_prune_repeat.py compare

The runner obtains BOTH dense inputs with V53's v46_dense_eval.py, freezes
all predictions, then calls V53's v6.stage_prune for six separate cells.
No fitter, model loader, torch, datasets, or network is used during authoring.
All result artifacts are write-once; partial/malformed files are errors.

The requested step16000 and step143000 revisions are fixed, as are densities
0.85, 0.75, 0.65. Cached aliases of identical weights are NOT independent
source states: preflight reports them and real freeze/measurement refuses them.
No automatic substitution of a different checkpoint is permitted.
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
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
try:
    from . import v53_prune_dev as v53
except ImportError:
    import v53_prune_dev as v53

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / "results/v53-prune-dev/register.json").is_file()),
            Path(__file__).resolve().parents[1])
OUT_REL = Path("results/v72-prune-repeat")
REGISTER_REL = Path("results/v53-prune-dev/register.json")
TABLE_REL = Path("paper/paper/tables/prune_repeat.tex")
TAGS = ("pythia-2.8b@step16000", "pythia-2.8b@step143000")  # step64000 cache shares the step143000 blob (identity check 2026-09-11); step16000 is a distinct blob
DENSITIES = (0.85, 0.75, 0.65)
CAPS = v53.CAPS
METHODS = ("power", "A2", "median_curve", "zero")
LABELS = ("Power (V53 delivered)", "A2", "Median development curve", "Zero")
CODE_INPUTS = tuple(Path("analysis") / name for name in (
    "v53_prune_dev.py", "v46_dense_eval.py", "v12_distill.py",
    "v6_capability_geometry.py", "model_registry.py", "v72_prune_repeat.py"))
PROTOCOL = {
    "model_dtype": "bf16", "device": "cuda:0", "n_probe": 128,
    "probe_seed": 0, "probe_half": "odd-indexed [1::2]",
    "loss": "V6 completion-token-weighted CE, max_len=1024, nats",
    "dense_evaluator": "analysis/v46_dense_eval.py (as in run_v53_dense.sh)",
    "pruned_evaluator": "analysis/v6_capability_geometry.py:stage_prune (as in run_v53_measure.sh)",
    "pruning": "global magnitude, language matrices; sampled threshold, seed=0, sample_size=2000000",
    "reference_device": "cpu",
    "response": "signed pruned CE minus the same V6 run's true dense CE; no clipping",
    "dense_check_atol": 1e-6,
}
BUDGET = {
    "available_inputs_all_methods": ["N0", "D0", "dense L0c", "density", "frozen V53 development register"],
    "target_pruning_labels_before_prediction": 0,
    "target_calibration_points": 0,
    "refit_or_reselection": False,
    "source_free_baselines": ["median_curve", "zero"],
    "note": "Same available input budget; source-free baselines ignore source covariates. "
            "V53 A2 uses its delivered anchor subsets, median uses delivered development medians.",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def present(path):
    return path.exists() or path.is_symlink()


def dense_path(root, tag):
    return root / OUT_REL / f"dense_dir_{tag}" / "dense.json"


def cell_path(root, tag, density):
    return root / OUT_REL / "measurements" / tag.replace("@", "--") / f"d{density}" / "prune_losses.json"


def cache_root():
    base = Path(os.environ.get("HF_HOME", str(Path(os.environ.get(
        "XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface")))
    return Path(os.environ.get("HF_HUB_CACHE", os.environ.get(
        "HUGGINGFACE_HUB_CACHE", str(base / "hub")))).expanduser()


def snapshot_info(snapshot):
    """Read cache metadata only; prefer safetensors just as the loader does."""
    weights = []
    for single, index in (("model.safetensors", "model.safetensors.index.json"),
                          ("pytorch_model.bin", "pytorch_model.bin.index.json")):
        if (snapshot / single).is_file():
            weights = [single]
            break
        if (snapshot / index).is_file():
            weights = sorted(set(read(snapshot / index)["weight_map"].values()))
            break
    required = ["config.json", "tokenizer_config.json", *weights]
    complete = bool(weights) and (snapshot / "tokenizer.json").is_file()
    complete = complete and all((snapshot / f).is_file() and (snapshot / f).stat().st_size > 0
                                for f in required)
    identities = []
    for name in weights:
        path = snapshot / name
        if path.is_file():
            resolved = path.resolve()
            # HF LFS blobs are content-addressed. Avoid rereading multi-GB weights.
            digest = resolved.name if re.fullmatch(r"[0-9a-f]{64}", resolved.name) else sha256(path)
            identities.append({"file": name, "blob_sha256": digest, "bytes": path.stat().st_size})
    return {"complete": bool(complete), "weights": identities,
            "metadata_sha256": {f: sha256(snapshot / f) for f in required
                                if f not in weights and (snapshot / f).is_file()}}


def inspect_cache(hub):
    repo = hub / "models--EleutherAI--pythia-2.8b"
    revisions = {}
    for ref in sorted((repo / "refs").rglob("*")):
        if ref.is_file():
            commit = ref.read_text().strip()
            require(re.fullmatch(r"[0-9a-f]{40}", commit), f"Invalid cache ref: {ref}")
            revisions[str(ref.relative_to(repo / "refs"))] = {
                "commit": commit, **snapshot_info(repo / "snapshots" / commit)}
    duplicates = []
    selected = [tag.split("@")[1] for tag in TAGS]
    if all(revisions.get(rev, {}).get("complete") for rev in selected):
        signatures = [sorted((w["blob_sha256"], w["bytes"]) for w in revisions[rev]["weights"])
                      for rev in selected]
        if signatures[0] == signatures[1]:
            duplicates.append(selected)
    return {"hub": str(hub), "hf_id": "EleutherAI/pythia-2.8b", "revisions": revisions,
            "unreferenced_snapshots": sorted(p.name for p in (repo / "snapshots").glob("*")
                                            if p.is_dir() and p.name not in
                                            {v["commit"] for v in revisions.values()}),
            "duplicate_weight_revisions": duplicates}


def state_mentions(value, location="$", inherited_size=False):
    """Find literal tags and structured size/step records, excluding architecture metadata."""
    hits = []
    if isinstance(value, str):
        for tag in TAGS:
            step = tag.split("@step")[1]
            if re.search(r"pythia-2\.8b(?:@|--|[/_: -])+step" + step + r"(?!\d)", value):
                hits.append({"tag": tag, "location": location})
            elif inherited_size and re.fullmatch(r"(?:step)?" + step, value):
                hits.append({"tag": tag, "location": location})
    elif isinstance(value, (int, float)) and inherited_size:
        hits += [{"tag": tag, "location": location} for tag in TAGS
                 if value == int(tag.split("@step")[1])]
    elif isinstance(value, list):
        for i, item in enumerate(value):
            hits += state_mentions(item, f"{location}[{i}]", inherited_size)
    elif isinstance(value, dict):
        scalars = [str(v) for v in value.values() if isinstance(v, (str, int, float))]
        sizes = value.get("sizes", [])
        own_size = inherited_size or any(re.fullmatch(r"(?:EleutherAI/)?(?:pythia-)?2\.8b", s)
                                         for s in scalars + (sizes if isinstance(sizes, list) else []))
        for key, item in value.items():
            if key in ("architectures", "architecture", "candidate_definitions", "dev_state_rule"):
                continue
            hits += state_mentions(key, f"{location}.{key}", inherited_size)
            keyed_size = bool(re.fullmatch(r"(?:pythia-)?2\.8b", key))
            # Size context follows explicit source records, including sizes x steps grids.
            hits += state_mentions(item, f"{location}.{key}", own_size or keyed_size)
    return hits


def pruning_audit(root):
    """Audit persisted local pruning records, including mixed-arm fits and paper mirrors.

    Scan JSON/CSV/TSV (also gzip JSON) under results and both paper data roots.
    A record is relevant if its path/content mentions pruning, or its filename
    denotes a register/fit/freeze. Raw target pruning files are checked by
    existence separately: freeze never reads target outcome values.
    """
    files, hits, prior_measurements = {}, [], []
    for base in (root / "results", root / "paper/results", root / "paper/data_mirror"):
        for path in sorted(base.rglob("*")):
            if not path.is_file() or "v72-prune-repeat" in path.parts:
                continue
            if path.suffix not in (".json", ".csv", ".tsv") and not path.name.endswith(".json.gz"):
                continue
            label = str(path.relative_to(root))
            path_hits = state_mentions(label)
            if path_hits and "prun" in label.lower():
                prior_measurements.append(label)
                continue
            raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
            text = raw.decode("utf-8")
            if not ("prun" in (label + text).lower()
                    or re.search(r"register|fit|freeze", path.name, re.I)):
                continue
            if path.suffix in (".csv", ".tsv"):
                value = list(csv.DictReader(io.StringIO(text), delimiter="\t" if path.suffix == ".tsv" else ","))
            else:
                value = json.loads(text)
            files[label] = sha256(path)
            hits += [{"path": label, **hit} for hit in state_mentions(value)]
    # V72 uses one fresh output file per cell; even an empty pre-freeze file is disallowed.
    for tag in TAGS:
        directory = cell_path(root, tag, DENSITIES[0]).parents[1]
        prior_measurements += [str(p.relative_to(root)) for p in directory.rglob("prune_losses.json")]
    return {"scope": "Local result JSON/CSV/TSV/gzip JSON and paper mirrors; architecture-only mentions excluded",
            "files_sha256": files, "target_mentions": hits,
            "prior_target_pruning_files": sorted(set(prior_measurements)),
            "clean": not hits and not prior_measurements}


def load_dense(root, tag):
    path = dense_path(root, tag)
    record = read(path)
    require(record.get("tag") == tag and record.get("resolved") == "EleutherAI/pythia-2.8b"
            and record.get("revision") == tag.split("@")[1], f"Dense source identity mismatch: {path}")
    losses = v53.checked_losses(record.get("dense"), path)
    tokens = record.get("measurement_tokens", {})
    require(all(type(tokens.get(cap)) is int and tokens[cap] > 0 for cap in CAPS),
            f"Missing positive dense measurement token counts: {path}")
    return losses


def preflight(root, hub):
    cache, audit = inspect_cache(hub), pruning_audit(root)
    blockers = []
    for tag in TAGS:
        rev = tag.split("@")[1]
        if not cache["revisions"].get(rev, {}).get("complete"):
            blockers.append(f"Missing complete cached EleutherAI/pythia-2.8b {rev}")
    if cache["duplicate_weight_revisions"]:
        blockers.append("Requested revisions share identical cached weight blobs; two distinct source states required")
    if audit["target_mentions"]:
        blockers.append("Target state appears in a prior pruning/register/fit record: " +
                        json.dumps(audit["target_mentions"][:5]))
    if audit["prior_target_pruning_files"]:
        blockers.append("Target pruning measurements already exist before freeze: " +
                        ", ".join(audit["prior_target_pruning_files"]))
    return {"cache": cache, "pruning_audit": audit, "blockers": blockers}


def predictions(register, targets):
    rows = []
    for target in targets:
        for d in DENSITIES:
            for cap in CAPS:
                z = v53.standardize(v53.raw_features(target["N0"], target["D0"], target["L0"][cap]),
                                    register["standardization"])
                all_values = v53.predict_all(register["models"][cap], z, d)
                estimates = {method: all_values[method] for method in METHODS}
                require(all(np.isfinite(v) for v in estimates.values()), "Nonfinite frozen prediction")
                rows.append({"source": target["tag"], "density": d, "capability": cap,
                             "predictions": estimates})
    return rows


def freeze(root, hub, dry_run=False):
    path = root / OUT_REL / "freeze.json"
    v53.require_new(path)
    report = preflight(root, hub)
    targets, dense_hashes = [], {}
    register = read(root / REGISTER_REL)
    require(register["selected_candidate"] == "power", "V53 delivered candidate must be power")
    require(not state_mentions(register), "V53 register contains a requested target state")
    for tag in TAGS:
        source = dense_path(root, tag)
        if not source.is_file():
            report["blockers"].append(f"Dense losses must be measured first: {source}")
            continue
        dense = load_dense(root, tag)
        size, step = v53.parse_tag(tag)
        targets.append({"tag": tag, "size": size, "step": step, "L0": dense,
                        "N0": v53.matrix_n0(size, register["architectures"]),
                        "D0": step * register["tokens_per_step"]})
        dense_hashes[str(source.relative_to(root))] = sha256(source)
    if report["blockers"]:
        if dry_run:
            print(v53.json_text({"status": "blocked; no files written", **report}))
            return report
        raise ValueError("; ".join(report["blockers"]))
    artifact = {
        "schema_version": 1, "experiment": "v72-prune-repeat", "targets": targets,
        "densities": list(DENSITIES), "capabilities": list(CAPS), "methods": list(METHODS),
        "selected_candidate": "power", "protocol": PROTOCOL, "input_budget": BUDGET,
        "scoring": "Full 2 states x 3 densities x 3 capabilities; MAE and signed bias; "
                   "equal weight per state, density and capability; no missing/collapsed cells dropped",
        "predictions": predictions(register, targets),
        "delivered_models": {cap: {m: register["models"][cap][m] for m in METHODS} for cap in CAPS},
        "standardization": register["standardization"],
        "candidate_definitions": {m: register["candidate_definitions"][m] for m in METHODS},
        "cache": report["cache"], "pruning_audit": report["pruning_audit"],
        "provenance": {"register_sha256": sha256(root / REGISTER_REL),
                       "dense_sha256": dense_hashes,
                       "code_sha256": {str(p): sha256(root / p) for p in CODE_INPUTS}},
    }
    text = v53.json_text(artifact)
    if dry_run:
        print("DRY-RUN: validated 18 rows / 72 predictions; no files written")
    else:
        v53.write_new(path, text)
    return artifact


def load_freeze(root):
    frozen = read(root / OUT_REL / "freeze.json")
    require(frozen["densities"] == list(DENSITIES) and frozen["capabilities"] == list(CAPS)
            and frozen["methods"] == list(METHODS)
            and [t["tag"] for t in frozen["targets"]] == list(TAGS), "Frozen panel mismatch")
    require(frozen["protocol"] == PROTOCOL and frozen["input_budget"] == BUDGET,
            "Frozen protocol/input budget mismatch")
    prov = frozen["provenance"]
    require(sha256(root / REGISTER_REL) == prov["register_sha256"], "Frozen V53 register changed")
    for name, digest in (prov["dense_sha256"] | prov["code_sha256"]).items():
        require(sha256(root / name) == digest, f"Frozen input changed: {name}")
    for target in frozen["targets"]:
        require(load_dense(root, target["tag"]) == target["L0"], "Frozen dense input mismatch")
    expected = predictions(read(root / REGISTER_REL), frozen["targets"])
    require(frozen["predictions"] == expected, "Frozen predictions changed or incomplete")
    return frozen


def load_cell(root, tag, d, frozen):
    path = cell_path(root, tag, d)
    table = read(path)
    require(set(table) == {"1.0", str(d)}, f"Unexpected or missing densities: {path}")
    dense = v53.checked_losses(table["1.0"], path)
    losses = v53.checked_losses(table[str(d)], path)
    reference = next(t["L0"] for t in frozen["targets"] if t["tag"] == tag)
    require(all(abs(dense[c] - reference[c]) <= PROTOCOL["dense_check_atol"] for c in CAPS),
            f"Dense CE differs from pre-freeze input: {path}")
    return dense, losses


def summarize(rows):
    result = {}
    for method in METHODS:
        by_cap = {}
        for cap in CAPS:
            errors = [r["predictions"][method] - r["observed_delta_loss"] for r in rows
                      if r["capability"] == cap]
            by_cap[cap] = {"n": len(errors), "mae": float(np.mean(np.abs(errors))),
                           "bias": float(np.mean(errors))}
        result[method] = {"by_capability": by_cap,
                          "macro_mae": float(np.mean([v["mae"] for v in by_cap.values()])),
                          "macro_bias": float(np.mean([v["bias"] for v in by_cap.values()]))}
    return result


@proofread_table
def table_text(scores):
    lines = [r"\begin{table}[!htbp]", r"\centering\footnotesize",
             r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Predictor & Math MAE & Code MAE & QA MAE & Mean MAE \\", r"\midrule"]
    for method, label in zip(METHODS, LABELS):
        values = [scores[method]["by_capability"][cap]["mae"] for cap in CAPS]
        values.append(scores[method]["macro_mae"])
        lines.append(label + " & " + " & ".join(f"{v:.3f}" for v in values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Pruning repeatability on Pythia-2.8B (final checkpoint; the two cached revisions used carry identical weights, so the six cells per capability are three densities measured twice), "
              r"$d\in\{0.85,0.75,0.65\}$. Signed $\Delta L$ errors in nats; six cells per "
              r"capability, with equal state, density and capability weights. All four predictors "
              r"were frozen before target pruning measurements, using the delivered V53 register "
              r"and the same available inputs $(N_0,D_0,L_{0,c},d)$, with zero target pruning "
              r"calibration points. Source-free median and zero ignore source covariates. "
              r"A2 uses V53's fixed ridge anchor fits and linear interpolation. No refitting, "
              r"test-based selection or outcome censoring. Dense inputs are measured first; "
              r"responses use each V6 pruning run's checked dense reference.}",
              r"\label{tab:prune_repeat}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def compare(root, dry_run=False):
    output, table = root / OUT_REL / "compare.json", root / TABLE_REL
    v53.require_new(output, table)
    frozen = load_freeze(root)
    cells = {(tag, d): load_cell(root, tag, d, frozen) for tag in TAGS for d in DENSITIES}
    rows = []
    for row in frozen["predictions"]:
        cap = row["capability"]
        dense, losses = cells[row["source"], row["density"]]
        y = losses[cap] - dense[cap]
        rows.append({**row, "observed_delta_loss": y,
                     "absolute_errors": {m: abs(p - y) for m, p in row["predictions"].items()}})
    scores = summarize(rows)
    report = {"schema_version": 1, "n_rows": len(rows), "n_configurations": len(cells),
              "methods": list(METHODS), "input_budget": BUDGET, "rows": rows, "scores": scores,
              "by_state": {tag: summarize([r for r in rows if r["source"] == tag]) for tag in TAGS},
              "gain_of_power": {m: {cap: scores[m]["by_capability"][cap]["mae"] -
                                             scores["power"]["by_capability"][cap]["mae"] for cap in CAPS}
                                for m in METHODS if m != "power"},
              "provenance": {"freeze_sha256": sha256(root / OUT_REL / "freeze.json"),
                             "measurement_sha256": {str(cell_path(root, tag, d).relative_to(root)):
                                                    sha256(cell_path(root, tag, d)) for tag, d in cells}}}
    encoded, tex = v53.json_text(report), table_text(scores)
    if dry_run:
        print(tex)
        print("DRY-RUN: complete equal-budget comparison validated; no files written")
    else:
        v53.write_new(output, encoded)
        v53.write_new(table, tex)
    return report


def dense_command(tag, root):
    return [sys.executable, "-B", "analysis/v46_dense_eval.py", tag,
            str(dense_path(root, tag).parent)]


def prune_command(tag, d, root):
    # Only adapt output location and singleton density list. V6 owns all model
    # loading, probes, masks, restoration, dense and pruned loss measurement.
    code = ("import sys; from pathlib import Path; "
            "from analysis import v6_capability_geometry as v6; "
            "tag,d,out=sys.argv[1:]; "
            "name=v6.require_compliant(tag); _,rev=v6.resolve_model_and_revision(tag); "
            "v6.DENSITIES=[float(d)]; path=Path(out); path.mkdir(parents=True,exist_ok=True); "
            "v6.stage_prune(name,'cuda:0',128,path,reference_device='cpu',revision=rev)")
    return [sys.executable, "-B", "-c", code, tag, str(d), str(cell_path(root, tag, d).parent)]


def dispatch(command, log, root, hub):
    log.parent.mkdir(parents=True, exist_ok=True)
    # Append logs so an interrupted job without a result can be retried.
    env = dict(os.environ, HF_HUB_CACHE=str(hub), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    with log.open("a") as stream:
        stream.write("COMMAND " + shlex.join(command) + "\nCUDA_VISIBLE_DEVICES=" +
                     env["CUDA_VISIBLE_DEVICES"] + "\n")
        stream.flush()
        subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def measure(root, hub, dry_run=False):
    frozen_path = root / OUT_REL / "freeze.json"
    frozen = load_freeze(root) if present(frozen_path) else None
    report = None if frozen else preflight(root, hub)
    if report:
        print("CACHE " + json.dumps(report["cache"], sort_keys=True))
        print(f"PRUNING AUDIT: {len(report['pruning_audit']['files_sha256'])} files; "
              f"target mentions={len(report['pruning_audit']['target_mentions'])}; "
              f"prior target measurements={len(report['pruning_audit']['prior_target_pruning_files'])}")
        for reason in report["blockers"]:
            print("BLOCKED: " + reason)
    if not dry_run:
        require(re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
                             os.environ.get("CUDA_VISIBLE_DEVICES", "")),
                "Set CUDA_VISIBLE_DEVICES to one full GPU UUID (GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)")
        require(not report or not report["blockers"], "; ".join(report["blockers"]) if report else "")
    # An advisory lock prevents two invocations from racing V6's nonexclusive
    # writer. Dry runs never create locks, output directories, or log files.
    if not dry_run:
        import fcntl
        lock_path = root / OUT_REL / ".measure.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return run_plan(root, hub, False)
    return run_plan(root, hub, True)


def run_plan(root, hub, dry_run):
    frozen_path = root / OUT_REL / "freeze.json"
    frozen = load_freeze(root) if present(frozen_path) else None
    if not dry_run and frozen is None:
        check = preflight(root, hub)
        require(not check["blockers"], "; ".join(check["blockers"]))
    if not dry_run and frozen is not None:
        require(inspect_cache(hub) == frozen["cache"], "HF cache changed since freeze")
    for tag in TAGS:
        path, command = dense_path(root, tag), dense_command(tag, root)
        if present(path):
            load_dense(root, tag)
            print("SKIP " + str(path))
        elif dry_run:
            print("DENSE " + shlex.join(command))
        else:
            print("DENSE " + tag, flush=True)
            dispatch(command, root / OUT_REL / "logs" / f"dense_{tag}.log", root, hub)
            load_dense(root, tag)
    if frozen is None:
        if dry_run:
            print("FREEZE (requires both dense inputs and clean, distinct states): "
                  "python3 -B analysis/v72_prune_repeat.py freeze")
        else:
            freeze(root, hub)
            frozen = load_freeze(root)
    else:
        print("SKIP validated write-once freeze.json")
    for tag in TAGS:
        for d in DENSITIES:
            path, command = cell_path(root, tag, d), prune_command(tag, d, root)
            if present(path):
                require(frozen is not None, "Pruning output exists without a freeze")
                load_cell(root, tag, d, frozen)
                print("SKIP " + str(path))
            elif dry_run:
                print("PRUNE " + shlex.join(command))
            else:
                frozen = load_freeze(root)
                current = inspect_cache(hub)
                require(current == frozen["cache"], "HF cache changed since freeze")
                print(f"PRUNE {tag} d={d}", flush=True)
                dispatch(command, root / OUT_REL / "logs" / f"prune_{tag}_d{d}.log", root, hub)
                load_cell(root, tag, d, frozen)
    print("COMPARE: python3 -B analysis/v72_prune_repeat.py compare")


def selftest():
    """Synthetic integration checks in /tmp only; no empirical output publication."""
    import contextlib
    import copy
    import shutil
    from unittest.mock import patch

    def fails(call, match):
        try:
            call()
        except (OSError, ValueError) as exc:
            assert match in str(exc), str(exc)
        else:
            raise AssertionError(f"Expected failure: {match}")

    with tempfile.TemporaryDirectory(prefix="v72-selftest-") as tmp, contextlib.redirect_stdout(io.StringIO()):
        root, hub = Path(tmp) / "project", Path(tmp) / "hub"
        root.mkdir()
        for relative in (*CODE_INPUTS, REGISTER_REL):
            dest = root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, dest)
        repo = hub / "models--EleutherAI--pythia-2.8b"
        for i, tag in enumerate(TAGS):
            revision, commit = tag.split("@")[1], str(i + 1) * 40
            snapshot = repo / "snapshots" / commit
            snapshot.mkdir(parents=True)
            (repo / "refs").mkdir(exist_ok=True)
            (repo / "refs" / revision).write_text(commit)
            for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
                (snapshot / name).write_text("{}")
            (snapshot / "model.safetensors").write_bytes(f"synthetic-weights-{i}".encode())
        assert not preflight(root, hub)["blockers"]
        before = sorted(str(p) for p in root.rglob("*"))
        dry = freeze(root, hub, True)
        assert len(dry["blockers"]) == 2
        assert before == sorted(str(p) for p in root.rglob("*"))
        fails(lambda: freeze(root, hub), "Dense losses must be measured first")
        for tag in TAGS:
            path = dense_path(root, tag)
            path.parent.mkdir(parents=True)
            path.write_text(v53.json_text({"tag": tag, "resolved": "EleutherAI/pythia-2.8b",
                                          "revision": tag.split("@")[1],
                                          "dense": {"math": 1.3, "code": 1.4, "qa": 4.8},
                                          "measurement_tokens": dict.fromkeys(CAPS, 100)}))
        snapshots = sorted((repo / "snapshots").iterdir())
        second = snapshots[1] / "model.safetensors"
        original = second.read_bytes()
        second.write_bytes((snapshots[0] / "model.safetensors").read_bytes())
        fails(lambda: freeze(root, hub), "identical cached weight blobs")
        second.write_bytes(original)
        bad = root / "results/old-prune/fit.json"
        bad.parent.mkdir()
        for record in ({"train": [TAGS[0]]}, {"dev": {"sizes": ["2.8b"], "steps": [16000]}},
                       {"training": [{"size": "2.8b", "step": 143000}]}):
            bad.write_text(v53.json_text(record))
            fails(lambda: freeze(root, hub), "prior pruning/register/fit record")
        bad.unlink()
        early = cell_path(root, TAGS[0], DENSITIES[0])
        early.parent.mkdir(parents=True)
        early.write_text("not even valid JSON; must not read outcomes")
        fails(lambda: freeze(root, hub), "measurements already exist")
        early.unlink()
        # No development fit may run during prediction or compare.
        with patch.object(v53, "fit_all", side_effect=AssertionError("refit forbidden")), \
             patch.object(subprocess, "run", side_effect=AssertionError("GPU/process forbidden")):
            frozen = freeze(root, hub, True)
            assert not (root / OUT_REL / "freeze.json").exists()
            freeze(root, hub)
            frozen_path = root / OUT_REL / "freeze.json"
            original_frozen = frozen_path.read_bytes()
            fails(lambda: freeze(root, hub), "overwrite")
            assert len(frozen["predictions"]) == 18
            assert all(set(r["predictions"]) == set(METHODS) for r in frozen["predictions"])
            # Independent formula/linear-interpolation oracle, with signed predictions.
            register = read(root / REGISTER_REL)
            for row in frozen["predictions"]:
                cap, d = row["capability"], row["density"]
                target = next(t for t in frozen["targets"] if t["tag"] == row["source"])
                stats = register["standardization"]
                raw = np.array([np.log(target["N0"]), target["L0"][cap], np.log(target["D0"])])
                phi = np.r_[1., (raw - stats["center"]) / stats["scale"]]
                model = register["models"][cap]
                power = sum(b * z for b, z in zip(model["power"]["beta"], phi)) * ((1-d)/.3)**model["power"]["gamma"]
                assert abs(row["predictions"]["power"] - power) < 1e-12
                for method in ("A2", "median_curve"):
                    anchors = model[method]["anchors"]
                    ds = sorted(float(k) for k in anchors)
                    vals = [float(np.dot(anchors[str(x)], phi)) if method == "A2" else anchors[str(x)] for x in ds]
                    assert abs(row["predictions"][method] - np.interp(d, ds, vals)) < 1e-12
                assert row["predictions"]["zero"] == 0
            assert any(r["predictions"]["power"] < 0 for r in frozen["predictions"])
            fails(lambda: compare(root, True), "prune_losses.json")
            for tag in TAGS:
                for d in DENSITIES:
                    path = cell_path(root, tag, d)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    dense = load_dense(root, tag)
                    path.write_text(v53.json_text({"1.0": dense, str(d): {c: dense[c] + .25 for c in CAPS}}))
            before = sorted(str(p) for p in root.rglob("*"))
            report = compare(root, True)
            measure(root, hub, True)
            assert before == sorted(str(p) for p in root.rglob("*"))
            assert report["n_rows"] == 18 and report["n_configurations"] == 6
            assert abs(report["scores"]["zero"]["macro_mae"] - .25) < 1e-12
            assert all(v["n"] == 6 for s in report["scores"].values() for v in s["by_capability"].values())
            cell = cell_path(root, TAGS[0], .85)
            content = cell.read_bytes()
            value = read(cell); value["1.0"]["math"] += .01
            cell.write_text(v53.json_text(value))
            fails(lambda: compare(root, True), "Dense CE differs")
            cell.write_bytes(content)
            value = copy.deepcopy(frozen); value["predictions"].pop()
            frozen_path.write_text(v53.json_text(value))
            fails(lambda: load_freeze(root), "predictions changed or incomplete")
            frozen_path.write_bytes(original_frozen)
            compare(root)
            assert r"\begin{table}[!htbp]" in (root / TABLE_REL).read_text()
            assert frozen_path.read_bytes() == original_frozen
            fails(lambda: compare(root), "overwrite")
            dense = dense_path(root, TAGS[0]); dense.write_text(dense.read_text() + " ")
            fails(lambda: load_freeze(root), "Frozen input changed")
            # Exercise the real orchestration with only the dispatch boundary
            # replaced. Verify dense -> freeze -> prune ordering and partial resume.
            shutil.rmtree(root / OUT_REL)
            launches = []

            def fake_dispatch(command, log, project, cache):
                assert project == root and cache == hub
                if command[2] == "analysis/v46_dense_eval.py":
                    tag = command[3]
                    assert not (root / OUT_REL / "freeze.json").exists()
                    path = dense_path(root, tag)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(v53.json_text({"tag": tag, "resolved": "EleutherAI/pythia-2.8b",
                        "revision": tag.split("@")[1], "dense": dict.fromkeys(CAPS, 2.),
                        "measurement_tokens": dict.fromkeys(CAPS, 100)}))
                    launches.append("dense")
                else:
                    assert all(dense_path(root, tag).exists() for tag in TAGS)
                    load_freeze(root)
                    tag, d = command[-3], float(command[-2])
                    path = cell_path(root, tag, d)
                    assert not path.exists()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(v53.json_text({"1.0": dict.fromkeys(CAPS, 2.),
                                                  str(d): dict.fromkeys(CAPS, 2.25)}))
                    launches.append("prune")

            with patch(__name__ + ".dispatch", side_effect=fake_dispatch), \
                 patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-12345678-1234-1234-1234-123456789abc"}):
                measure(root, hub)
                assert launches == ["dense"] * 2 + ["prune"] * 6
                launches.clear()
                measure(root, hub)
                assert not launches
                cell_path(root, TAGS[1], .75).unlink()
                measure(root, hub)
                assert launches == ["prune"]
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
                    fails(lambda: measure(root, hub), "one full GPU UUID")
    assert "torch" not in sys.modules and "transformers" not in sys.modules
    print("PASS: CPU selftest; cache aliases, contamination, dense-first/write-once freeze, "
          "V53 formulas, 18 paired rows, missing/drifting outcomes, provenance, "
          "mocked runner order/partial resume/UUID, dry-run, [!htbp] table")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", choices=("freeze", "compare", "measure"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--hf-cache", type=Path, default=cache_root(), help="HF hub cache directory (read-only)")
    parser.add_argument("--dry-run", action="store_true", help="CPU validation/plan; no writes or model loading")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if args.mode is None:
        parser.error("a mode or --selftest is required")
    try:
        root, hub = args.root.resolve(), args.hf_cache.resolve()
        if args.mode == "freeze":
            freeze(root, hub, args.dry_run)
        elif args.mode == "compare":
            compare(root, args.dry_run)
        else:
            measure(root, hub, args.dry_run)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
