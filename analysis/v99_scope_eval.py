#!/usr/bin/env python3
"""V99: evaluate reuse trajectories on the frozen V71 QA panels.

First/last mean the numerically first/last saved trajectory updates (first is
update-0). --checkpoints adds named updates, or all. Each completed JSON embeds
its own trajectory's freshly measured update-0 and checkpoint-minus-update-0
deltas. T and D_U count supervised completion tokens, across training domains.

Only LoRA trajectories are supported. Update-0 must be the explicitly saved
dense baseline; every trained state requires a complete, successfully loaded
adapter. Models/data must already be local. Dry-run reads metadata and hashes
adapters, but loads no datasets/models, touches no accelerator, and writes nothing.
All result files are immutable and confined to results/v99-scope. Use a new
subdirectory to change distributions, dtype, device, or the frozen QA register.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

# Also prevent imported repository modules from creating bytecode outside scope.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from analysis import v67_musique_qa as v67
from analysis import v71_qa_scope as v71

OUTPUT_ROOT = ROOT / "results/v99-scope"
DISTRIBUTIONS = tuple(v71.SETS)
IDENTITY_KEYS = (
    "trajectory_run_id", "student", "resolved_student", "revision", "teacher",
    "recipe", "domains", "n_per_domain", "data_seed", "data_sampling_seed",
    "training_seed", "training_mode", "data_pool_sha256",
    "unique_data_pool_tokens", "unique_data_pool_examples",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scoped_path(path):
    path = Path(path).absolute()
    # Reject symlinks, including ones pointing back into scope: an immutable
    # result must never be redirected onto an existing result or input.
    require(not any(p.is_symlink() for p in (path, *path.parents)),
            f"Output path contains a symlink: {path}")
    path = path.resolve()
    require(path.is_relative_to(OUTPUT_ROOT.resolve()),
            f"Output must be under {OUTPUT_ROOT}: {path}")
    return path


def write_new_json(path, payload):
    """Publish complete JSON atomically, without replacing any existing file."""
    path = scoped_path(path)
    content = v71.json_text(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".v99-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Exclusive destination, including concurrent runs.
    finally:
        if temporary is not None:
            temporary.unlink()


def select_checkpoints(trajectory, checkpoints=()):
    """Return first + last + explicitly requested updates, sorted numerically."""
    directory = Path(trajectory)
    if directory.name != "trajectory":
        directory = directory / "trajectory"
    require(directory.is_dir(), f"Missing trajectory directory: {directory}")
    saved = {}
    for path in directory.iterdir():
        match = re.fullmatch(r"update-(\d+)", path.name)
        if not match or not path.is_dir():
            continue
        update = int(match[1])
        require(update not in saved, f"Duplicate update {update}: {directory}")
        if not (path / "eval.json").is_file():
            raise FileNotFoundError(f"Missing checkpoint metadata: {path / 'eval.json'}")
        saved[update] = path.resolve()
    require(saved and 0 in saved, f"Missing own update-0: {directory}")
    selected = {min(saved), max(saved)}
    for name in checkpoints:
        if name == "all":
            selected.update(saved)
        elif name in ("first", "last"):
            selected.add(min(saved) if name == "first" else max(saved))
        else:
            match = re.fullmatch(r"(?:update-)?(\d+)", str(name))
            require(match is not None, f"Invalid checkpoint selector: {name}")
            update = int(match[1])
            require(update in saved, f"Checkpoint {name} not saved in {directory}")
            selected.add(update)
    return [saved[update] for update in sorted(selected)]


def source_identity(path):
    return {"path": str(Path(path).resolve()), "sha256": v71.sha_file(path)}


def pool_accounting(run, baseline, selected, root=ROOT):
    """Recover completion D_U from a verified full distinct pass, or V67's register."""
    log_path = run / "train_log.json"
    log = v71.read_json(log_path)
    for key in ("student", "resolved_student", "n_per_domain", "data_sampling_seed",
                "training_seed", "unique_data_pool_tokens", "unique_data_pool_examples"):
        require(log[key] == baseline[key], f"Train log/baseline {key} mismatch: {run}")
    curve = log["loss_curve"]
    require(bool(curve), f"Empty training log: {log_path}")
    counts, processed, completion = {0: (0, 0)}, 0, 0
    epoch = 1
    for step, row in enumerate(curve, 1):
        require(row["step"] == step and row["epoch"] in (epoch, epoch + 1),
                f"Noncontiguous training log: {log_path}")
        epoch = row["epoch"]
        require(all(type(row[k]) is int and row[k] > 0 for k in ("tokens", "completion_tokens"))
                and row["completion_tokens"] <= row["tokens"], f"Invalid token increments: {log_path}")
        processed += row["tokens"]
        completion += row["completion_tokens"]
        require((row["processed_tokens"], row["completion_tokens_seen"]) == (processed, completion),
                f"Cumulative token mismatch: {log_path}, update {step}")
        require(row["unique_data_pool_tokens"] == baseline["unique_data_pool_tokens"],
                f"Pool changed: {log_path}")
        counts[step] = (processed, completion)
    for path, saved in selected:
        require(counts.get(saved["updates"]) ==
                (saved["processed_tokens"], saved["completion_tokens_seen"]),
                f"Checkpoint/log token mismatch: {path}")

    first = [r for r in curve if r["epoch"] == 1]
    complete = (bool(first) and log["training_examples"] == baseline["unique_data_pool_examples"]
                and first[-1]["unique_examples_seen"] == baseline["unique_data_pool_examples"]
                and sum(r["tokens"] for r in first) == first[-1]["unique_data_tokens"]
                == baseline["unique_data_pool_tokens"])
    du = sum(r["completion_tokens"] for r in first) if complete else None
    evidence = {**source_identity(log_path),
                "method": "sum completion_tokens over verified complete distinct first epoch",
                "first_complete_epoch_end_update": first[-1]["step"] if complete else None}

    # V67/V71 use this register for legacy P2 pools. Never match just U/seed
    # across teachers, recipes, or unrelated pool construction protocols.
    if (baseline["teacher"] == "gpt-5.6-luna" and baseline["recipe"] == "full"
            and "p2v2" in run.name):
        register_path = root / "results/v47-p2-register/register.json"
        register = v71.read_json(register_path)
        key = f"U{baseline['n_per_domain']}_s{baseline['data_seed']}"
        pool = register["pools"][key]
        require(pool["pool_processed_tokens"] == baseline["unique_data_pool_tokens"],
                f"Registered pool differs: {run}")
        require(du is None or du == pool["D_U_completion"], f"Registered D_U differs: {run}")
        du = pool["D_U_completion"]
        evidence["pool_register"] = {**source_identity(register_path), "pool_key": key}
        if not complete:
            evidence["method"] = "V67 registered D_U_completion; no complete distinct epoch"
    require(type(du) is int and 0 < du <= baseline["unique_data_pool_tokens"],
            f"Cannot establish supervised D_U from a complete distinct pool pass/register: {run}")
    for e in range(1, epoch):
        rows = [r for r in curve if r["epoch"] == e]
        if complete:
            require(sum(r["completion_tokens"] for r in rows) == du
                    and sum(r["tokens"] for r in rows) == baseline["unique_data_pool_tokens"],
                    f"Completed epochs disagree on pool counts: {run}")
    return du, evidence


def adapter_identity(checkpoint, saved):
    if saved["updates"] == 0:
        require(saved["snapshot_kind"] == "trajectory_baseline"
                and saved["processed_tokens"] == saved["completion_tokens_seen"] == 0,
                f"Update-0 must be an explicit untrained baseline: {checkpoint}")
        require(not (checkpoint / "adapter").exists(), f"Unexpected update-0 adapter: {checkpoint}")
        return {"adapter_path": None, "adapter_sha256": None, "adapter_files_sha256": {},
                "adapter_expected": False}
    adapter = v67.adapter_directory(checkpoint)
    config = v71.read_json(adapter / "adapter_config.json")
    require(config["peft_type"] == "LORA"
            and config["base_model_name_or_path"] == saved["resolved_student"],
            f"Adapter/base model mismatch: {adapter}")
    files = {p.name: v71.sha_file(p) for p in sorted(adapter.iterdir())
             if p.name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")}
    return {"adapter_path": str(adapter.resolve()), "adapter_sha256": v71.sha_json(files),
            "adapter_files_sha256": files, "adapter_expected": True}


def build_plan(trajectories, checkpoints, output_dir, root=ROOT):
    out = scoped_path(output_dir)
    plan, seen = [], set()
    for trajectory in trajectories:
        paths = select_checkpoints(trajectory, checkpoints)
        run = paths[0].parent.parent
        require(run not in seen, f"Duplicate trajectory: {run}")
        seen.add(run)
        selected = [(p, v71.read_json(p / "eval.json")) for p in paths]
        baseline = selected[0][1]
        require(baseline["training_mode"] == "lora", f"Only LoRA trajectories are supported: {run}")
        for path, saved in selected:
            require(int(path.name.split("-")[1]) == saved["updates"], f"Update/path mismatch: {path}")
            require(all(saved.get(k) == baseline.get(k) for k in IDENTITY_KEYS),
                    f"Checkpoint/own update-0 identity mismatch: {path}")
        du, evidence = pool_accounting(run, baseline, selected, root)
        run_key = re.sub(r"[^A-Za-z0-9_.-]", "_", run.name) + "-" + v71.sha_json(str(run))[:16]
        rows = []
        for path, saved in selected:
            row = {k: saved.get(k) for k in IDENTITY_KEYS}
            row.update(trajectory=str(run), checkpoint=str(path), checkpoint_name=path.name,
                       snapshot_kind=saved["snapshot_kind"],
                       eval_source=source_identity(path / "eval.json"),
                       pool_size=saved["n_per_domain"], pool_seed=saved["data_seed"],
                       updates=saved["updates"], actual_supervised_tokens=saved["completion_tokens_seen"],
                       completion_tokens_seen=saved["completion_tokens_seen"],
                       processed_tokens=saved["processed_tokens"], D_U=du, D_U_completion=du,
                       reuse=saved["completion_tokens_seen"] / du, D_U_source=evidence,
                       **adapter_identity(path, saved))
            output = scoped_path(out / run_key / f"{path.name}.json")
            rows.append({"identity": row, "output": str(output),
                         "action": "skip_existing" if output.exists() else "score"})
        plan.append({"trajectory": str(run), "checkpoints": rows})
    return plan


def load_protocol(register_path, distributions, device, dtype):
    register = v71.read_json(register_path)
    require(register["version"] == 71 and register["max_len"] == v67.MAX_LEN == v71.MAX_LEN
            and register["batch_size"] == 1 and register["n_per_set"] == v71.N,
            "V71 register/scoring protocol differs")
    for key in distributions:
        panel = register["sets"][key]
        require(all(panel[k] == v for k, v in v71.SETS[key].items())
                and panel["n"] == v71.N and len(panel["indices"]) == len(set(panel["indices"])) == v71.N,
                f"Invalid frozen V71 panel: {key}")
    return {"version": 99, "v71_register": source_identity(register_path),
            "distributions": list(distributions), "probes": {k: register["sets"][k] for k in distributions},
            "device": device, "dtype": dtype, "max_len": v67.MAX_LEN, "batch_size": 1,
            "loss_definition": register["loss"], "information_condition": register["information_condition"],
            "panel_builder": "analysis.v71_qa_scope.prepare_panels",
            "scorer": "analysis.v67_musique_qa.measure_qa",
            "reference_model": {"hf_id": register["hf_id"], "revision": register["revision"]},
            "code_sha256": {name: v71.sha_file(ROOT / "analysis" / name) for name in
                            ("v99_scope_eval.py", "v71_qa_scope.py", "v67_musique_qa.py",
                             "v6_capability_geometry.py", "v9_capability_regions.py", "v12_distill.py")}}


def prepare_registered_panels(protocol, cache_dir, data_file, trace_base, root=ROOT):
    """Run the original builders and reject any drift from the frozen register."""
    register = v71.read_json(protocol["v71_register"]["path"])
    require(v71.sha_file(protocol["v71_register"]["path"]) == protocol["v71_register"]["sha256"],
            "V71 register changed after planning")
    panels, metadata, sources = v71.prepare_panels(
        cache_dir, v67.local_data_file(data_file), trace_base, root)
    for key in DISTRIBUTIONS:
        require(metadata[key] == register["sets"][key]
                and v71.sha_json(panels[key]) == register["sets"][key]["prompt_reference_sha256"],
                f"Rebuilt panel differs from frozen V71 sample: {key}")
    # Cache relocation is harmless; source filenames and bytes must be identical.
    def source_hashes(items):
        return [{**{k: v for k, v in item.items() if k != "files"},
                 "files": {Path(p).name: h for p, h in item["files"].items()}} for item in items]
    require(source_hashes(sources) == source_hashes(register["dataset_sources"]),
            "Dataset sources differ from V71 register")
    return {key: panels[key] for key in protocol["distributions"]}


def local_model_path(identity, protocol):
    """Resolve to an existing snapshot before using the shared model loader."""
    name, revision = identity["resolved_student"], identity["revision"]
    reference = protocol["reference_model"]
    if revision is None and name == reference["hf_id"]:
        revision = reference["revision"]
    if Path(name).is_dir():
        path = Path(name).resolve()
    else:
        from huggingface_hub import try_to_load_from_cache
        config = try_to_load_from_cache(name, "config.json", revision=revision or "main")
        if not isinstance(config, str):
            raise FileNotFoundError(f"No local model config for {name}@{revision or 'main'}")
        path = Path(config).parent
    require((path / "config.json").is_file(), f"Missing local model config: {path}")
    return path, revision


def score_checkpoint(identity, panels, protocol):
    # Check again immediately before loading. Never catch a missing/failed
    # adapter and proceed with the base model.
    require(adapter_identity(Path(identity["checkpoint"]), identity) ==
            {k: identity[k] for k in ("adapter_path", "adapter_sha256", "adapter_files_sha256", "adapter_expected")},
            f"Adapter changed after planning: {identity['checkpoint']}")
    import torch
    from analysis.v12_distill import load_text_causal_lm

    model_path, revision = local_model_path(identity, protocol)
    base = tokenizer = model = None
    try:
        base, tokenizer = load_text_causal_lm(str(model_path), getattr(torch, protocol["dtype"]), None)
        tokenizer.truncation_side = "right"
        base.to(protocol["device"]).eval()
        model, loaded = base, False
        if identity["adapter_expected"]:
            from peft import PeftModel
            model = PeftModel.from_pretrained(base, identity["adapter_path"],
                                             local_files_only=True, is_trainable=False)
            loaded = isinstance(model, PeftModel) and bool(model.peft_config)
            require(loaded, f"Adapter was not actually loaded: {identity['adapter_path']}")
        losses = {}
        for key, probes in panels.items():
            loss, tokens = v67.measure_qa(model, tokenizer, probes, protocol["device"])
            require(math.isfinite(loss) and loss >= 0 and type(tokens) is int and tokens > 0,
                    f"Invalid conditional loss/count for {key}")
            losses[key] = {"loss": loss, "tokens": tokens, "n": len(probes),
                           "probe_identity": protocol["probes"][key]}
        return {**identity, "adapter_loaded": loaded, "model_local_path": str(model_path),
                "model_revision": revision, "model_config_sha256": v71.sha_file(model_path / "config.json"),
                "losses": losses}
    finally:
        del model, base, tokenizer
        gc.collect()
        if protocol["device"].startswith("cuda"):
            torch.cuda.empty_cache()


def completed_result(item, protocol):
    path = scoped_path(item["output"])
    if not path.exists():
        return None
    result = v71.read_json(path)
    require(result.get("status") == "complete" and result.get("protocol") == protocol,
            f"Existing output is incomplete or has a different protocol; use a new subdirectory: {path}")
    require(all(result.get(k) == v for k, v in item["identity"].items()),
            f"Existing output belongs to a changed checkpoint: {path}")
    baseline = result["update_0"]
    require(baseline["trajectory"] == result["trajectory"]
            and all(baseline.get(k) == result.get(k) for k in IDENTITY_KEYS)
            and baseline["updates"] == baseline["actual_supervised_tokens"] == baseline["processed_tokens"] == 0
            and baseline["D_U"] == result["D_U"] and baseline["adapter_path"] is None,
            f"Existing output has an invalid own update-0: {path}")
    for record in (result, baseline):
        require(record["adapter_loaded"] is (record["updates"] != 0)
                and set(record["losses"]) == set(protocol["distributions"]),
                f"Existing output has invalid adapter/distribution status: {path}")
        for key, cell in record["losses"].items():
            require(math.isfinite(cell["loss"]) and cell["loss"] >= 0
                    and type(cell["tokens"]) is int and cell["tokens"] > 0
                    and cell["n"] == protocol["probes"][key]["n"]
                    and cell["probe_identity"] == protocol["probes"][key],
                    f"Existing output has invalid measurement/probe identity: {path}")
    for key in protocol["distributions"]:
        require(result["losses"][key]["tokens"] == baseline["losses"][key]["tokens"]
                and result["delta_from_update_0"][key] == result["losses"][key]["loss"] - baseline["losses"][key]["loss"],
                f"Existing output has invalid baseline pairing/delta: {path}")
    return result


def evaluate(plan, protocol, output_dir, cache_dir, data_file, trace_base):
    out = scoped_path(output_dir)
    protocol_path = scoped_path(out / "protocol.json")
    if protocol_path.exists():
        require(v71.read_json(protocol_path) == protocol, "Existing protocol differs; use a new output subdirectory")
    existing = {item["output"]: completed_result(item, protocol)
                for run in plan for item in run["checkpoints"]}
    if all(value is not None for value in existing.values()):
        return {"scored": 0, "skipped": len(existing)}
    # Offline even if a previous experiment allowed downloads. Local paths are
    # passed to the model loader; no HF dataset cache writers are used.
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    panels = prepare_registered_panels(protocol, cache_dir, data_file, trace_base)
    if not protocol_path.exists():
        try:
            write_new_json(protocol_path, protocol)
        except FileExistsError:
            require(v71.read_json(protocol_path) == protocol, "Concurrent protocol differs")
    scored = skipped = 0
    for run in plan:
        baseline = None
        for item in run["checkpoints"]:
            previous = existing[item["output"]]
            if previous is not None:
                if item["identity"]["updates"] == 0:
                    baseline = {k: v for k, v in previous.items()
                                if k not in ("status", "protocol", "update_0", "delta_from_update_0")}
                skipped += 1
                continue
            measured = score_checkpoint(item["identity"], panels, protocol)
            if measured["updates"] == 0:
                baseline = measured
            require(baseline is not None, "Missing measured own update-0")
            require(measured["model_local_path"] == baseline["model_local_path"]
                    and measured["model_config_sha256"] == baseline["model_config_sha256"],
                    "Checkpoint/base model differs from own update-0")
            require(all(measured["losses"][k]["tokens"] == baseline["losses"][k]["tokens"]
                        for k in panels), "Completion token count differs from own update-0")
            result = {**measured, "status": "complete", "protocol": protocol, "update_0": baseline,
                      "delta_from_update_0": {k: measured["losses"][k]["loss"] - baseline["losses"][k]["loss"]
                                              for k in panels}}
            try:
                write_new_json(item["output"], result)
                scored += 1
            except FileExistsError:
                published = completed_result(item, protocol)
                if measured["updates"] == 0:
                    baseline = published["update_0"]
                skipped += 1
            print(f"V99: complete {item['output']}", flush=True)
    return {"scored": scored, "skipped": skipped}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectories", type=Path, nargs="+", required=True,
                        help="V12 run directories, or their trajectory/ directories")
    parser.add_argument("--checkpoints", nargs="+", default=[], metavar="UPDATE",
                        help="Add update-N or integer updates to first/last; 'all' selects every saved update")
    parser.add_argument("--distributions", nargs="+", choices=DISTRIBUTIONS, default=list(DISTRIBUTIONS))
    parser.add_argument("--device", default="cpu", help="Explicit scoring device (default: cpu)")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16",
                        help="Scoring dtype (default: bfloat16, as in V71)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Print plan; no data/model loading or writes")
    parser.add_argument("--register", type=Path, default=ROOT / "results/v71-qa-scope/register.json",
                        help="Existing frozen V71 register (read only)")
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get(
        "HF_DATASETS_CACHE", Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "datasets")))
    parser.add_argument("--data-file", type=Path, help="Existing MuSiQue answerable-dev JSONL; V67 lookup if omitted")
    parser.add_argument("--trace-base", type=Path, default=ROOT / "results/traces-pilot")
    args = parser.parse_args(argv)
    try:
        require(len(set(args.distributions)) == len(args.distributions), "Duplicate distributions")
        require(re.fullmatch(r"cpu|cuda(?::\d+)?", args.device), "--device must be cpu, cuda, or cuda:N")
        distributions = [key for key in DISTRIBUTIONS if key in args.distributions]
        protocol = load_protocol(args.register, distributions, args.device, args.dtype)
        plan = build_plan(args.trajectories, args.checkpoints, args.output_dir)
        if args.dry_run:
            protocol_path = scoped_path(args.output_dir / "protocol.json")
            if protocol_path.exists():
                require(v71.read_json(protocol_path) == protocol, "Existing protocol differs; use a new output subdirectory")
            for run in plan:
                for item in run["checkpoints"]:
                    completed_result(item, protocol)
            print(v71.json_text({"dry_run": True, "output_dir": str(scoped_path(args.output_dir)),
                                 "device": args.device, "dtype": args.dtype,
                                 "distributions": distributions, "v71_register": protocol["v71_register"],
                                 "probes": {k: {f: protocol["probes"][k][f] for f in
                                                 ("n", "seed", "prompt_reference_sha256")} for k in distributions},
                                 "trajectories": plan}), end="")
        else:
            print(v71.json_text(evaluate(plan, protocol, args.output_dir, args.cache_dir,
                                         args.data_file, args.trace_base)), end="")
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(1, f"V99: {exc}\n")


if __name__ == "__main__":
    main()
