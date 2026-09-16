#!/usr/bin/env python3
"""LONI-only forward wrapper: run the original V6 evaluator with frozen probes.

The --dry-run path is CPU-only and imports no torch/model libraries.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.a11_score import (OUT, ROOT, digest, load_registration, measurement_probe_hashes,
                                read, require, validate_losses, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    parser.add_argument("--device", default="cuda:0", choices=["cuda:0"])
    parser.add_argument("--n-probe", type=int, default=128, choices=[128])
    parser.add_argument("--probe-seed", type=int, default=0, choices=[0])
    parser.add_argument("--pruning-seed", type=int, default=0, choices=[0])
    parser.add_argument("--model-dtype", default="bf16", choices=["bf16"])
    parser.add_argument("--densities", default="0.9,0.85,0.8,0.75,0.7,0.65")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan, frozen = load_registration()
    target = next((t for t in plan["targets"] if t["state"] == args.state), None)
    require(target is not None, "State is outside registered A11 panel")
    require([float(x) for x in args.densities.split(",")] == plan["densities"], "Unregistered density grid")
    bundle = read(ROOT / plan["protocol"]["probe_file"])
    probes = bundle["probes"]
    require(set(probes) == set(plan["capabilities"]) and all(len(p) == 128 for p in probes.values()),
            "Need the full frozen 128-probe sets before odd-half selection")
    probe_hashes = measurement_probe_hashes(probes)
    require(probe_hashes == frozen["measurement_probe_sha256"], "Frozen measurement probes changed")
    directory = OUT / "measurements" / args.state.replace("@", "--")
    if args.dry_run:
        print(f"DRY RUN {args.state}: V6 stage_prune, bf16, cuda:0, dense + {args.densities}; "
              f"probe/pruning seeds=0; build 128 -> odd half 64/cap; output={directory}")
        return
    require(os.environ.get("SLURM_JOB_ID") is not None, "Forward measurement must run in the LONI Slurm job")
    require(not directory.exists(), f"Refusing to overwrite/reuse an existing state run: {directory}")
    directory.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    # Only now import GPU dependencies. Fitting code is never imported here.
    import torch
    from analysis import v6_capability_geometry as v6
    require(torch.cuda.is_available(), "No assigned CUDA GPU")
    require(v6.PRUNE_THRESHOLD_SAMPLE_SIZE == plan["protocol"]["threshold_sample_size"], "Threshold sample changed")
    model_name = v6.require_compliant(args.state)
    _, revision = v6.resolve_model_and_revision(args.state)
    require(model_name == target["hf_id"] and revision == target["revision"], "Model resolution mismatch")

    def frozen_probes(n, seed=0, *, include_secondary=False):
        require(n == 128 and seed == 0 and not include_secondary, "Unexpected V6 probe request")
        return probes

    original_loader = v6.load_text_causal_lm
    checkpoint = {}

    def checked_loader(name, dtype, rev=None):
        model, tokenizer = original_loader(name, dtype, rev)
        config = model.config
        architecture = {k: getattr(config, k) for k in target["architecture"]}
        require(architecture == target["architecture"], "Loaded architecture differs from registered N0")
        checkpoint.update(hf_id=name, revision=rev, commit_hash=getattr(config, "_commit_hash", None),
                          architecture=architecture, tokenizer_class=type(tokenizer).__name__)
        return model, tokenizer

    write_json(directory / "started.json", dict(target=target, started_utc=started,
               predictions_sha256=digest(OUT / "predictions.json"), slurm_job_id=os.environ["SLURM_JOB_ID"]), True)
    # V6 owns bf16 loading, tokenization, aggregation, sampled thresholds, and
    # dense restoration. Only redirect probes and the registered density grid.
    with patch.object(v6, "build_probes", frozen_probes), \
         patch.object(v6, "DENSITIES", plan["densities"]), \
         patch.object(v6, "load_text_causal_lm", checked_loader):
        v6.stage_prune(model_name, args.device, args.n_probe, directory,
                       reference_device="cuda", revision=revision)
    validate_losses(read(directory / "prune_losses.json"), plan)
    write_json(directory / "metadata.json", dict(status="complete", target=target,
               started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
               predictions_sha256=digest(OUT / "predictions.json"), protocol=plan["protocol"],
               probe_sha256=probe_hashes, checkpoint=checkpoint,
               prune_losses_sha256=digest(directory / "prune_losses.json"),
               slurm_job_id=os.environ["SLURM_JOB_ID"], gpu_name=torch.cuda.get_device_name(0),
               versions={name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "numpy", "tokenizers", "huggingface-hub")}), True)


if __name__ == "__main__":
    main()
