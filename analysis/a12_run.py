#!/usr/bin/env python3
"""Dispatch frozen V12 argv and add all six readouts; execution is explicit."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from analysis.a12_score import DEFAULT_PLAN, load_plan, require


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_commands(plan):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from analysis import v12_distill as v12
    outcomes = []
    old_argv = sys.argv
    try:
        for tr in plan["trajectories"]:
            if tr["v12_argv"] is None:
                outcomes.append({"trajectory_id": tr["trajectory_id"], "status": "blocked_no_realizable_command"})
                continue
            stream = io.StringIO()
            sys.argv = ["analysis/v12_distill.py", *tr["v12_argv"], "--dry-run"]
            with contextlib.redirect_stdout(stream):
                v12.main()  # Real ArgumentParser plus V12's actual design validation.
            result = json.loads(stream.getvalue())
            require(result["dry_run"] and result["student"] == tr["model"], "Wrong dry-run model")
            require(result["seed"] == tr.get("training_seed", plan["protocol"]["seed"]), "Wrong dry-run training seed")
            require(result["trajectory_tokens"] == [c["processed_tokens"] for c in tr["checkpoints"][1:]], "Wrong dry-run milestones")
            outcomes.append({"trajectory_id": tr["trajectory_id"], "status": "parser_dry_run_pass", "argv": sys.argv[1:], "result": result})
    finally:
        sys.argv = old_argv
    return {"valid_commands": sum(x["status"] == "parser_dry_run_pass" for x in outcomes),
            "blocked_commands": sum(x["status"] != "parser_dry_run_pass" for x in outcomes),
            "training": False, "model_weights_loaded": False, "outcomes": outcomes}


def execute(plan, tr):
    require(plan["launch_ready"] and all(t["status"] == "realized" for t in plan["trajectories"]),
            "A12 launch blocked: a complete, realized and frozen plan is required.")
    require(tr["status"] == "realized", f"Blocked trajectory: {tr['trajectory_id']}")
    for path, expected in plan["runtime_sha256"].items():
        require(sha(ROOT / path) == expected, f"Frozen runtime input changed: {path}")
    out = ROOT / tr["output_dir"]
    require(not out.exists(), f"Refusing to overwrite existing trajectory: {out}")
    from analysis import v12_distill as v12
    from analysis import v99_scope_eval as v99
    from transformers import AutoTokenizer
    from huggingface_hub import try_to_load_from_cache

    frozen_tokenizer = plan["tokenizers"][tr["student"]]
    config = try_to_load_from_cache(tr["model"], "config.json", revision=frozen_tokenizer["revision"])
    require(isinstance(config, str), f"Missing cached model snapshot: {tr['model']}")
    snapshot = Path(config).parent
    for name, expected in frozen_tokenizer["file_sha256"].items():
        require(sha(snapshot / name) == expected, f"Cached tokenizer differs: {name}")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    sampling_seed = tr["data_seed"] if tr["data_seed"] is not None else tr.get("training_seed", plan["protocol"]["seed"])
    records, _ = v12.load_sft_records("gpt-5.6-luna", v12.DOMAINS, tr["pool"]["n_per_domain"], "full",
                                     ROOT / "results/traces-pilot", sampling_seed, tr["data_seed"])
    pool_ids, supervision = [], {}
    for r in records:
        example = v12.tokenize_sft_example(tokenizer, r["prompt"], r["completion"])
        if not example["n_completion_tokens"]:
            continue
        identity = hashlib.sha256(json.dumps([r["prompt"], r["completion"]], ensure_ascii=False).encode()).hexdigest()
        supervision[identity] = int(example["n_completion_tokens"])
        pool_ids.append((identity, int(example["input_ids"].numel())))
    require(len(pool_ids) == len(supervision), "Duplicate source identities in pool")
    require(sum(supervision.values()) == tr["pool"]["D_U_pool"], "Supervised pool size changed")
    require(hashlib.sha256(json.dumps(sorted(pool_ids)).encode()).hexdigest() == tr["pool"]["data_pool_sha256"], "Pool hash changed")
    del tokenizer

    primary = json.loads((ROOT / plan["panels"]["primary"]).read_text())
    fresh = json.loads((ROOT / plan["panels"]["fresh"]).read_text())
    register = ROOT / plan["panels"]["register"]
    protocol = v99.load_protocol(register, v99.DISTRIBUTIONS, "cuda:0", "bfloat16")
    original_loader = v12.load_text_causal_lm
    original_probes, original_out, original_argv = v12.build_probes, v12.OUT_BASE, sys.argv

    def cached_loader(name, dtype, revision=None):
        require(name in (tr["model"], str(snapshot)), f"Unexpected model: {name}")
        model, tok = original_loader(str(snapshot), dtype, None)
        # PEFT metadata must identify the HF model, not a machine-local cache path.
        model.name_or_path = tr["model"]
        model.config._name_or_path = tr["model"]
        return model, tok

    def frozen_probes(n, seed=0):
        require(n == 128 and seed == 0, "Primary probe selection changed")
        return primary

    try:
        v12.load_text_causal_lm = cached_loader
        v12.build_probes = frozen_probes
        v12.OUT_BASE = out / "v12"
        sys.argv = ["analysis/v12_distill.py", *tr["v12_argv"]]
        v12.main()  # Exact frozen CLI and unchanged optimizer/training implementation.
    finally:
        v12.load_text_causal_lm = original_loader
        v12.build_probes, v12.OUT_BASE, sys.argv = original_probes, original_out, original_argv

    # Reuse V99's adapter integrity checks and V67's original scorer, using
    # frozen complete-pool D_U rather than V99's incomplete-first-epoch inference.
    baseline = None
    for cp in tr["checkpoints"]:
        path = ROOT / cp["eval_path"]
        ev = json.loads(path.read_text())
        require(ev["training_seed"] == tr.get("training_seed", plan["protocol"]["seed"])
                and ev["data_seed"] == tr["data_seed"], f"Registered seed mismatch: {path}")
        require(ev["updates"] == cp["update"] and ev["completion_tokens_seen"] == cp["actual_supervised_tokens"]
                and ev["processed_tokens"] == cp["processed_tokens"], f"Replayed boundary mismatch: {path}")
        require(ev["data_pool_sha256"] == tr["pool"]["data_pool_sha256"], f"Wrong pool: {path}")
        identity = {**ev, "revision": frozen_tokenizer["revision"], "checkpoint": str(path.parent),
                    **v99.adapter_identity(path.parent, ev)}
        scored = v99.score_checkpoint(identity, fresh, protocol)
        readouts = {}
        for readout, cap in (("math:MATH-500", "math"), ("code:MBPP", "code"), ("qa:2Wiki-probe", "qa")):
            readouts[readout] = {"loss": ev["post_training"][cap], "tokens": ev["measurement_tokens"][cap],
                "n": 64, "probe_sha256": plan["runtime_sha256"][plan["panels"]["primary"]]}
        for key, readout in (("2wiki_new", "qa:2Wiki-fresh"), ("musique", "qa:MuSiQue"), ("triviaqa", "qa:TriviaQA")):
            cell = scored["losses"][key]
            readouts[readout] = {"loss": cell["loss"], "tokens": cell["tokens"], "n": cell["n"],
                "probe_sha256": plan["runtime_sha256"][plan["panels"]["fresh"]]}
        if cp["update"] == 0:
            baseline = {r: v["loss"] for r, v in readouts.items()}
        require(baseline is not None, "Own update-0 missing")
        for readout, cell in readouts.items():
            cell["delta"] = cell["loss"] - baseline[readout]
        ev["a12"] = {"trajectory_id": tr["trajectory_id"], "predictions_sha256": plan["predictions_sha256"],
                     "D_U_pool": sum(supervision.values()), "D_U_seen": cp["D_U_seen"],
                     "readouts": readouts, "fresh_adapter_loaded": scored["adapter_loaded"],
                     "fresh_adapter_sha256": scored["adapter_sha256"], "fresh_protocol": protocol}
        v12.write_json_atomic(path, ev)
    v12.write_json_atomic(out / "complete.json", {"trajectory_id": tr["trajectory_id"], "status": "complete",
                                                 "predictions_sha256": plan["predictions_sha256"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--task", type=int, choices=range(18))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-commands", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    plan = load_plan(args.plan)
    if args.validate_commands:
        report = validate_commands(plan)
        (args.plan.parent / "parser-validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "outcomes"}))
    else:
        require(args.task is not None, "--execute requires --task")
        execute(plan, plan["trajectories"][args.task])


if __name__ == "__main__":
    main()
