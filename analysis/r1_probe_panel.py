#!/usr/bin/env python3
"""R1 per-item compression panel; run with ``python -m analysis.r1_probe_panel``.

The panel follows the frozen preregistration: V6(128)[1::2] for the primary
panel (3 x 64) and V9(128, seed=0)[1::2] for the eight task benchmarks
(8 x 64), 704 items in total; the manifest records the actual selection. V9's
builder requires C4 internally, but C4 is never measured here. The V9 Fisher
runs drew 64 samples per benchmark; the n=128 draw used here is a different
sample, so these response items are not the Fisher estimation items.

Config names: dense, prune_d0.9 ... prune_d0.5, quant_b8 ... quant_b3.
--out overrides the complete output directory (otherwise results/.../<tag>).
Dense is always required, even when omitted from --configs. Completed files
are validated and skipped; a completed run needs no model/dataset loading.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time

import numpy as np
import torch

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import v6_capability_geometry as v6
from analysis import v9_capability_regions as v9
from analysis import v10_quantization as v10

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = ("dense", "prune_d0.9", "prune_d0.8", "prune_d0.7",
                   "prune_d0.6", "prune_d0.5", "quant_b8", "quant_b6",
                   "quant_b5", "quant_b4", "quant_b3")
PRIMARY = {"math": "math500", "code": "mbpp", "qa": "2wiki"}
BENCHMARKS = ("gsm8k", "math500", "svamp", "humaneval", "mbpp", "2wiki",
              "hotpotqa", "triviaqa")
# 128 drawn, odd half measured: 64 items per benchmark, as preregistered.
V9_N = 128
V9_SEED = 0
ITEM_FIELDS = ("id", "set", "benchmark", "capability")
PROTOCOL = {
    "primary": {"builder": "v6.build_probes", "n": 128, "seed": 0},
    "benchmark": {"builder": "v9.build_probe_registry", "n": V9_N,
                  "seed": V9_SEED, "benchmarks": list(BENCHMARKS)},
    "selection": "[1::2]; index in the unsplit builder output",
    "item_sha256": "UTF-8(prompt + completion), before tokenization",
    "loss": "v6.completion_loss; no_grad; one item at a time; default max_len",
    "pruning": {"sample_size": v6.PRUNE_THRESHOLD_SAMPLE_SIZE, "seed": 0},
    "quantization": "v10._apply_fake_quantization; per-output-channel RTN",
    "restore_tolerance_nats_per_token": 1e-3,
}


# rsi.cs.fsu.edu reports the hostname sn4622131043 (rai is sn4622122543).
RSI_HOSTNAMES = ("sn4622131043",)


def is_rsi(hostname: str) -> bool:
    return hostname in RSI_HOSTNAMES or hostname.startswith("rsi")


def guard_model(model_id: str) -> None:
    hostname = socket.gethostname()
    # User directive 2026-11-07: Qwen experiments run on rsi only (not rai, never hpg or LONI).
    if "qwen" in model_id.lower() and not is_rsi(hostname):
        raise RuntimeError(f"Qwen runs only on rsi; refusing {model_id!r} on {hostname!r}")


def parse_configs(values) -> list[str]:
    if isinstance(values, str):
        values = [values]
    result = ["dense"]
    for value in values:
        for name in value.split(","):
            if name == "dense":
                continue
            if name.startswith("prune_d"):
                density = float(name.removeprefix("prune_d"))
                if not 0 < density < 1:
                    raise ValueError("Pruning density must be between zero and one")
                canonical = f"prune_d{density:g}"
                if canonical != name:
                    raise ValueError(f"Use canonical config name {canonical!r}")
            elif re.fullmatch(r"quant_b[0-9]+", name):
                bits = int(name.removeprefix("quant_b"))
                if not 2 <= bits <= 16 or name != f"quant_b{bits}":
                    raise ValueError("Quantization bits must be an integer from 2 to 16")
            else:
                raise ValueError(f"Unknown config {name!r}; use dense, prune_d0.7, quant_b4")
            if name not in result:
                result.append(name)
    return result


def build_panel() -> list[dict]:
    primary = v6.build_probes(128)
    registry = v9.build_probe_registry(V9_N, seed=V9_SEED)
    items = []
    groups = [("primary", benchmark, cap, primary[cap], 128)
              for cap, benchmark in PRIMARY.items()]
    groups += [("benchmark", name, v9.BENCHMARK_CAPABILITY[name], registry[name], V9_N)
               for name in BENCHMARKS]
    for set_name, benchmark, capability, samples, expected_n in groups:
        if len(samples) != expected_n:
            raise RuntimeError(f"Incomplete {set_name}:{benchmark}: {len(samples)} != {expected_n}")
        for index in range(1, len(samples), 2):
            sample = samples[index]
            prompt, completion = sample["prompt"], sample["completion"]
            items.append({"id": f"{set_name}:{benchmark}:{index}", "set": set_name,
                          "benchmark": benchmark, "capability": capability,
                          "prompt": prompt, "completion": completion,
                          "sha256": hashlib.sha256((prompt + completion).encode("utf-8")).hexdigest()})
    return items


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(model_id: str, revision: str, model) -> dict:
    """Inspect only cached files; never download extra files for provenance.

    Hash the first shard completely and verify its LFS blob name. Subsequent
    LFS shards use their content-addressed names. Non-LFS/local files are all
    hashed completely. The index enumerates every shard, including any towers
    discarded by V6's text-only loader.
    """
    from huggingface_hub import try_to_load_from_cache

    resolved_revision = getattr(model.config, "_commit_hash", None) or revision

    def cached(filename):
        if Path(model_id).is_dir():
            path = Path(model_id) / filename
            return path if path.is_file() else None
        value = try_to_load_from_cache(model_id, filename, revision=resolved_revision)
        return Path(value) if isinstance(value, str) and Path(value).is_file() else None

    config = cached("config.json")
    if config is None:
        raise RuntimeError("Loaded checkpoint's config.json is missing from the cache")
    single = cached("model.safetensors")
    index = cached("model.safetensors.index.json")
    if single is not None:
        names = ["model.safetensors"]
    elif index is not None:
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        raise RuntimeError("Checkpoint has no cached safetensors weights/index; cannot record shard identity")
    if not names or any(not name.endswith(".safetensors") for name in names):
        raise RuntimeError("Invalid safetensors shard index")
    shards = []
    for i, name in enumerate(names):
        path = cached(name)
        if path is None:
            raise RuntimeError(f"Missing cached safetensors shard: {name}")
        blob = path.resolve(strict=True)
        lfs_name = blob.parent.name == "blobs" and re.fullmatch(r"[0-9a-f]{64}", blob.name)
        full_hash = i == 0 or not lfs_name
        digest = sha256_file(blob) if full_hash else blob.name
        if lfs_name and digest != blob.name:
            raise RuntimeError(f"SHA256 does not match the cached blob filename: {blob}")
        shards.append({"filename": name, "sha256": digest, "blob_path": str(blob),
                       "hash_source": "full_file" if full_hash else "verified_lfs_filename"})
    return {"resolved_revision": resolved_revision,
            "config_sha256": sha256_file(config.resolve(strict=True)),
            "safetensors": shards}


def atomic_json(path: Path, value) -> None:
    """Replace only complete, fsynced JSON; interrupted temp files are ignored."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def runtime_metadata(device: str) -> dict:
    import transformers
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    return {"hostname": socket.gethostname(), "device": device,
            "gpu_name": torch.cuda.get_device_name(torch.device(device))
            if torch.device(device).type == "cuda" else None,
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def measure(model, tok, panel: list[dict], device: str) -> list[dict]:
    rows = []
    with torch.no_grad():
        for item in panel:
            nll, count = v6.completion_loss(model, tok, item["prompt"], item["completion"], device)
            nll, count = float(nll), int(count)
            if not math.isfinite(nll) or count <= 0:
                raise RuntimeError(f"Invalid completion loss for {item['id']}: {nll}, {count}")
            rows.append({**{key: item[key] for key in ITEM_FIELDS},
                         "nll_sum": nll, "n_tokens": count})
    return rows


def validate_result(path: Path, config: str, items: list[dict]) -> dict:
    data = json.loads(path.read_text())
    if data["config"] != config or len(data["items"]) != len(items):
        raise RuntimeError(f"Incomplete/mismatched result: {path}")
    if not isinstance(data["seconds"], (float, int)) or not math.isfinite(data["seconds"]) or data["seconds"] < 0:
        raise RuntimeError(f"Invalid duration: {path}")
    for row, item in zip(data["items"], items):
        if (any(row[key] != item[key] for key in ITEM_FIELDS)
                or type(row["n_tokens"]) is not int or row["n_tokens"] <= 0
                or not isinstance(row["nll_sum"], (float, int)) or not math.isfinite(row["nll_sum"])):
            raise RuntimeError(f"Invalid item in {path}: {item['id']}")
    return data


def check_expected_dense(rows: list[dict], path: Path | None) -> None:
    if path is None:
        return
    expected = json.loads(path.read_text())["1.0"]
    for cap in PRIMARY:
        selected = [row for row in rows if row["set"] == "primary" and row["capability"] == cap]
        actual = sum(row["nll_sum"] for row in selected) / sum(row["n_tokens"] for row in selected)
        target = float(expected[cap])
        if not math.isfinite(target) or abs(actual - target) > 0.01:
            raise RuntimeError(f"Dense {cap} mismatch: measured {actual:.8f}, expected {target:.8f} (>0.01 nats)")


def check_restored_dense(model, tok, panel, dense_rows, device) -> None:
    subset = [item for item in panel if item["set"] == "primary"][:8]
    baseline = {row["id"]: row for row in dense_rows}
    for row in measure(model, tok, subset, device):
        initial = baseline[row["id"]]
        difference = abs(row["nll_sum"] / row["n_tokens"] - initial["nll_sum"] / initial["n_tokens"])
        if row["n_tokens"] != initial["n_tokens"] or difference > 1e-3:
            raise RuntimeError(f"Dense restore mismatch for {row['id']}: {difference:.8f} nats/token")


def _identity_signature(identity: dict) -> dict:
    # Cache locations can change between hosts without changing the checkpoint.
    return {"resolved_revision": identity["resolved_revision"],
            "config_sha256": identity["config_sha256"],
            "safetensors": [(s["filename"], s["sha256"]) for s in identity["safetensors"]]}


def run_panel(model_id: str, revision: str, tag: str, *, out: Path | None = None,
              configs=DEFAULT_CONFIGS, device: str = "cuda:0",
              reference_device: str = "cpu", expect_dense: Path | None = None) -> Path:
    guard_model(model_id)  # Must precede model, tokenizer, dataset, or cache loading.
    if not revision or revision == "RESOLVE":
        raise ValueError("An explicit --revision is required; resolve RESOLVE in the driver first")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        raise ValueError("--tag must be a single safe directory name")
    if reference_device not in ("cpu", "cuda"):
        raise ValueError("--reference-device must be cpu or cuda")
    configs = parse_configs(configs)
    out = Path(out) if out is not None else ROOT / "results/r1-probe-panel" / tag
    out.mkdir(parents=True, exist_ok=True)
    # Serialize writers for a tag, including independently restarted drivers.
    with (out / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _run_locked(model_id, revision, tag, out, configs, device, reference_device, expect_dense)
    return out


def _run_locked(model_id, revision, tag, out, configs, device, reference_device, expect_dense):
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    settings = {"schema_version": 1, "model_id": model_id, "revision": revision,
                "tag": tag, "dtype": "bfloat16", "reference_device": reference_device,
                "protocol": PROTOCOL}
    saved = {}
    if manifest is not None:
        for key, value in settings.items():
            if manifest.get(key) != value:
                raise RuntimeError(f"Resume manifest mismatch for {key}; use a different output tag")
        for config in configs:
            path = out / f"{config}.json"
            if path.exists():
                saved[config] = validate_result(path, config, manifest["items"])
                print(f"[r1] {config}: exists, skipping", flush=True)
        if saved and "dense" not in saved:
            raise RuntimeError("Compressed results exist without their dense baseline")
        if "dense" in saved:
            check_expected_dense(saved["dense"]["items"], expect_dense)
        if len(saved) == len(configs):
            return
    elif any(out.glob("*.json")):
        raise RuntimeError("Results exist without manifest.json; refusing to mix unverified data")

    panel = build_panel()
    item_manifest = [{key: item[key] for key in (*ITEM_FIELDS, "sha256")} for item in panel]
    if manifest is not None and manifest["items"] != item_manifest:
        raise RuntimeError("Resume probe hashes/selection changed; use a different output tag")
    model, tok = v6.load_text_causal_lm(model_id, torch.bfloat16, revision)
    model.to(device).eval()
    identity = checkpoint_identity(model_id, revision, model)
    runtime = runtime_metadata(device)
    if manifest is not None:
        if _identity_signature(manifest["checkpoint"]) != _identity_signature(identity):
            raise RuntimeError("Resume checkpoint identity changed; use a different output tag")
        manifest["sessions"].append(runtime)
    else:
        manifest = {**settings, **runtime, "checkpoint": identity,
                    "items": item_manifest, "sessions": [runtime]}
    atomic_json(manifest_path, manifest)

    if "dense" in saved:
        dense_rows = saved["dense"]["items"]
        check_restored_dense(model, tok, panel, dense_rows, device)
    else:
        start = time.monotonic()
        dense_rows = measure(model, tok, panel, device)
        seconds = time.monotonic() - start
        check_expected_dense(dense_rows, expect_dense)
        atomic_json(out / "dense.json", {"config": "dense", "items": dense_rows, "seconds": seconds})
        print(f"[r1] dense: {len(dense_rows)} items in {seconds:.1f}s", flush=True)

    pending = [config for config in configs if config != "dense" and config not in saved]
    if not pending:
        return
    parameters = v6.language_weight_parameters(model)
    reference = "cpu" if reference_device == "cpu" else device
    dense_weights = [parameter.detach().to(reference, copy=True) for _, parameter in parameters]
    pruning = [config for config in pending if config.startswith("prune_d")]
    thresholds = {}
    if pruning:
        sample = v6._sample_abs_weights(parameters)
        thresholds = {name: float(np.quantile(sample, 1 - float(name.removeprefix("prune_d"))))
                      for name in pruning}
        del sample
        manifest.setdefault("pruning_thresholds", {}).update(thresholds)
        atomic_json(manifest_path, manifest)
    for config in pending:
        start = time.monotonic()
        try:
            if config.startswith("prune_d"):
                v6.apply_global_magnitude_pruning(model, float(config.removeprefix("prune_d")),
                                                 reference_weights=dense_weights, threshold=thresholds[config])
            else:
                v10._apply_fake_quantization(parameters, dense_weights, int(config.removeprefix("quant_b")))
            rows = measure(model, tok, panel, device)
        finally:
            v10._restore_dense_weights(parameters, dense_weights)
            check_restored_dense(model, tok, panel, dense_rows, device)
        if [row["n_tokens"] for row in rows] != [row["n_tokens"] for row in dense_rows]:
            raise RuntimeError(f"Token counts changed under {config}")
        seconds = time.monotonic() - start
        atomic_json(out / f"{config}.json", {"config": config, "items": rows, "seconds": seconds})
        print(f"[r1] {config}: {len(rows)} items in {seconds:.1f}s", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--model-id", dest="model_id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                        help="Space- or comma-separated names, e.g. dense,prune_d0.7,quant_b4")
    parser.add_argument("--out", type=Path, help="Complete output directory override")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--expect-dense", type=Path)
    args = parser.parse_args(argv)
    run_panel(**vars(args))


if __name__ == "__main__":
    main()
