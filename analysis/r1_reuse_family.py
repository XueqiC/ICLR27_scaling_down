#!/usr/bin/env python3
"""E6: replay the frozen Gemma matrix on two explicitly pinned Qwen Base students.

--plan resolves Hub metadata only, audits cached Gemma tokenization, and freezes
12 commands. --run --index I calls the unchanged V12 CLI. Completed runs skip;
interrupted runs are archived and replayed from seed 0 (V12 saves no optimizer).
--evaluate [--index I] selects saved checkpoints by supervised T, reuses their
V12 primary measurements, and scores the frozen V71 panels through V99/V67.
--analyse and --table work without models, including before any measurements.
All model/tokenizer access during execution is local-only. No training on --plan.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = Path("results/r1-reuse-family")
PREREG = Path("notes/rebuttal/prereg_r1_reuse_family.md")
TRAINER = Path("analysis/v12_distill.py")
TRAINER_SHA = "2d1d78b3562583a38fd15cda5f61ed9fbec7563ace7d44d9ddad7bd22225bc3a"
PREREG_SHA = "d66d798335be70ab5ec36f64939e955928442ac6cc6a1ecaf7a5e0cc08105f86"
LAUNCH = Path("scripts/run_matrix_gemma3-1b.sh")
A1 = Path("results/a1-development-table/development_table.csv")
REGISTER = Path("results/v71-qa-scope/register.json")
PRIMARY = Path("results/a12-data-requirement-confirmation/inputs/v6-probes.json")
FRESH = Path("results/a12-data-requirement-confirmation/inputs/fresh-qa-panels.json")
V15 = Path("results/v15-accuracy/gemma3-1b/dense/accuracy.json")
STUDENTS = ("Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base")
POOLS, SEEDS, BUDGETS = (66, 198, 594), (41, 42), (50000, 100000, 200000)
PRIMARY_KEYS = {"math": "MATH-500", "code": "MBPP", "qa": "2Wiki-probe"}
FRESH_KEYS = {"2wiki_new": "2Wiki-fresh", "musique": "MuSiQue", "triviaqa": "TriviaQA"}
DISTRIBUTIONS = (*PRIMARY_KEYS.values(), *FRESH_KEYS.values())
MILESTONES = [85000, 170000, 339000, 678000]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    temporary.replace(path)


def write(path, value):
    write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def frozen_checks(root=ROOT):
    require(sha(root / TRAINER) == TRAINER_SHA, "Frozen V12 trainer SHA mismatch")
    require(sha(root / PREREG) == PREREG_SHA, "Frozen E6 pre-registration SHA mismatch")
    require(read(root / "results/a1-development-table/summary.json")["code_sha256"][str(TRAINER)]
            == TRAINER_SHA, "A1 trainer identity differs")


def gemma_arguments(u, seed, root=ROOT):
    """Read the actual launch, preserving flag spelling, order, and defaults."""
    source = (root / LAUNCH).read_text().replace("\\\n", " ")
    match = re.search(r"python3 analysis/v12_distill\.py\s+(.*?)\s*>>", source, re.S)
    require(match is not None, "Cannot recover Gemma command from launch script")
    return shlex.split(match[1].replace("$U", str(u)).replace("$DS", str(seed)))


def flag_values(argv):
    result, current = {}, None
    for token in argv:
        if token.startswith("--"):
            require(token not in result, f"Repeated flag: {token}")
            current = token
            result[current] = []
        else:
            require(current is not None, f"Unattached argument: {token}")
            result[current].append(token)
    return result


def gemma_path(u, seed):
    return Path("results/v12-distill/gemma3-1b") / f"gpt-5.6-luna_full_{u}_matrix2_lora_dseed{seed}"


def check_recorded_protocol(u, seed, root=ROOT):
    ev, log = (read(root / gemma_path(u, seed) / name) for name in ("eval.json", "train_log.json"))
    expected = {"teacher": "gpt-5.6-luna", "recipe": "full", "training_mode": "lora",
                "n_per_domain": u, "data_seed": seed, "data_sampling_seed": seed,
                "training_seed": 0, "learning_rate": 1e-4, "epochs": 60,
                "output_suffix": "matrix2", "schedule_tokens": 678000,
                "stop_after_trajectory": True, "trajectory_tokens_requested": MILESTONES,
                "domains": ["math", "qa", "code"], "optimizer": "AdamW",
                "scheduler": "cosine", "warmup_ratio": .03, "probe_seed": 0,
                "n_probe_requested": 128}
    require(all(ev.get(k) == v for k, v in expected.items()), f"Gemma protocol differs: {u}/{seed}")
    require(log["learning_rate"] == 1e-4 and log["effective_batch_size_sequences"] == 16
            and log["max_len"] == 1024 and log["dtype"] == "bfloat16"
            and log["lora"] == {"r": 16, "alpha": 32, "dropout": 0.0,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
            f"Gemma training log differs: {u}/{seed}")
    flags = flag_values(gemma_arguments(u, seed, root))
    required = {"--student": ["gemma3-1b"], "--teacher": ["gpt-5.6-luna"],
                "--recipe": ["full"], "--training-mode": ["lora"], "--n-per-domain": [str(u)],
                "--data-seed": [str(seed)], "--seed": ["0"], "--lr": ["1e-4"],
                "--output-suffix": ["matrix2"], "--save-trajectory": [],
                "--trajectory-tokens": list(map(str, MILESTONES)), "--stop-after-trajectory": [],
                "--schedule-tokens": ["678000"], "--epochs": ["60"], "--device": ["cuda"]}
    require(flags == required, "Gemma launch flags changed")
    return ev, log


def source_hashes(records):
    # Exactly V12's example_id expression; no tokenizer-dependent lengths.
    return sorted(hashlib.sha256(json.dumps([r["prompt"], r["completion"]],
                                             ensure_ascii=False).encode()).hexdigest() for r in records)


def cached_tokenizer(model, revision="main"):
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoTokenizer

    config = try_to_load_from_cache(model, "tokenizer_config.json", revision=revision)
    require(isinstance(config, str), f"Tokenizer must already be cached: {model}@{revision}")
    snapshot = Path(config).parent
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    identity = {"model": model, "revision": snapshot.name, "class": type(tokenizer).__name__,
                "file_sha256": {p.name: sha(p) for p in sorted(snapshot.iterdir())
                                if p.name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                                              "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt")}}
    return tokenizer, identity


def price_pool(tokenizer, u, seed, root=ROOT, cache=None):
    """CPU accounting using V12 helpers; do not import A3 (it clears CUDA visibility)."""
    from analysis.v12_distill import load_sft_records, tokenize_sft_example, DOMAINS

    records, _ = load_sft_records("gpt-5.6-luna", DOMAINS, u, "full",
                                 root / "results/traces-pilot", seed, seed)
    cache = {} if cache is None else cache
    ledger = []
    for record in records:
        key = (record["prompt"], record["completion"])
        if key not in cache:
            ex = tokenize_sft_example(tokenizer, *key)
            cache[key] = {"example_id": source_hashes([record])[0],
                          "completion_tokens": int(ex["n_completion_tokens"]),
                          "processed_tokens": int(ex["input_ids"].numel())}
        if cache[key]["completion_tokens"]:
            ledger.append(cache[key])
    require(ledger and len({r["example_id"] for r in ledger}) == len(ledger),
            "Empty pool or duplicate source identities")
    pool_hash = hashlib.sha256(json.dumps(sorted(
        (r["example_id"], r["processed_tokens"]) for r in ledger)).encode()).hexdigest()
    return {"data_pool_sha256": pool_hash, "D_U_pool": sum(r["completion_tokens"] for r in ledger),
            "unique_data_pool_tokens": sum(r["processed_tokens"] for r in ledger), "_ledger": ledger}


def audit_gemma_pools(root=ROOT):
    from analysis.v12_distill import load_sft_records, DOMAINS

    tokenizer, identity = cached_tokenizer("google/gemma-3-1b-pt", read(root / REGISTER)["revision"])
    cache = {}
    with (root / A1).open() as stream:
        a1_runs = {r["run_id"] for r in csv.DictReader(stream)}
    pools = []
    for u in POOLS:
        for seed in SEEDS:
            ev, log = check_recorded_protocol(u, seed, root)
            require(str(gemma_path(u, seed).relative_to("results/v12-distill")) in a1_runs,
                    "Reference trajectory absent from development table")
            priced = price_pool(tokenizer, u, seed, root, cache)
            require(priced["data_pool_sha256"] == ev["data_pool_sha256"],
                    f"Reconstructed Gemma pool hash differs: U={u}, seed={seed}")
            records, _ = load_sft_records("gpt-5.6-luna", DOMAINS, u, "full",
                                         root / "results/traces-pilot", seed, seed)
            ids = source_hashes(records)
            require(ids == sorted(r["example_id"] for r in priced["_ledger"]),
                    "Gemma tokenization changed source membership")
            # Future/alternative V12 artifacts may expose these explicitly.
            exposed = []
            for payload in (ev, log):
                for key in ("sample_hashes", "example_ids", "sorted_sample_hashes"):
                    if key in payload:
                        require(sorted(payload[key]) == ids, f"Recorded {key} differs")
                        exposed.append(key)
            pools.append({"U": u, "pool_seed": seed, "gemma_run": str(gemma_path(u, seed)),
                          "sample_hashes": ids, "sample_hashes_sha256": digest(ids),
                          "gemma_data_pool_sha256": ev["data_pool_sha256"],
                          "gemma_D_U": priced["D_U_pool"], "sample_count": len(ids),
                          "recorded_sample_hash_fields": exposed,
                          "verification": "Replayed frozen V12 selection; sorted source hashes match the "
                          "Gemma ledger whose (hash, token length) digest matches recorded data_pool_sha256. "
                          "V12 currently records no individual sample hash list. Cross-tokenizer aggregate "
                          "pool hashes are not compared because they include token lengths."})
    return pools, identity


def verify_panels(root=ROOT):
    primary, fresh, register = read(root / PRIMARY), read(root / FRESH), read(root / REGISTER)
    old = read(root / V15)["records"]
    for cap in PRIMARY_KEYS:
        require(len(primary[cap]) == 128, f"Wrong primary count: {cap}")
        pairs = [(r["prompt"], r["completion"]) for r in primary[cap][1::2]]
        require(pairs == [(r["prompt"], r["reference_completion"]) for r in old if r["domain"] == cap],
                f"Primary {cap} does not match saved Gemma measurements")
    for key in FRESH_KEYS:
        require(len(fresh[key]) == register["n_per_set"] == 256
                and digest(fresh[key]) == register["sets"][key]["prompt_reference_sha256"],
                f"Fresh {key} differs from frozen V71 panel")
    return primary, fresh


def resolve_revision(model):
    from huggingface_hub import HfApi
    info = HfApi().model_info(model, revision="main", timeout=20)
    require(info.id == model and re.fullmatch(r"[0-9a-f]{40}", info.sha or ""),
            f"Hub did not resolve an explicit commit: {model}")
    return {"revision": info.sha, "method": "huggingface_hub.HfApi.model_info",
            "requested_revision": "main", "model_id": info.id}


def make_plan(root=ROOT, resolver=resolve_revision, pool_audit=None):
    frozen_checks(root)
    verify_panels(root)
    revisions = {model: resolver(model) for model in STUDENTS}
    pools, tokenizer = pool_audit if pool_audit is not None else audit_gemma_pools(root)
    runs = []
    for model in STUDENTS:
        revision = revisions[model]["revision"]
        require(re.fullmatch(r"[0-9a-f]{40}", revision), f"Unresolved revision: {model}")
        student = f"{model}@{revision}"
        tag = re.sub(r"[^A-Za-z0-9._-]+", "--", student).strip(".-_")
        for pool in pools:
            argv = gemma_arguments(pool["U"], pool["pool_seed"], root)
            reference = argv.copy()
            argv[argv.index("--student") + 1] = student
            command = ["python3", str(TRAINER), *argv]
            runs.append({"index": len(runs), "student": model, "revision": revision,
                         "U": pool["U"], "pool_seed": pool["pool_seed"], "training_seed": 0,
                         "gemma_run": pool["gemma_run"], "gemma_v12_argv": reference,
                         "v12_argv": argv, "command": command, "command_line": shlex.join(command),
                         "output_dir": str(Path("results/v12-distill") / tag / Path(pool["gemma_run"]).name),
                         "sample_hashes_sha256": pool["sample_hashes_sha256"]})
    sources = [PREREG, TRAINER, LAUNCH, A1, REGISTER, PRIMARY, FRESH, V15,
               Path("analysis/r1_reuse_family.py"), Path("scripts/rsi_r1_reuse.sh"),
               Path("results/a1-development-table/summary.json")]
    sources += [Path("analysis") / name for name in
                ("v6_capability_geometry.py", "v67_musique_qa.py", "v71_qa_scope.py",
                 "v99_scope_eval.py", "model_registry.py", "a3_corner_pool_search.py")]
    sources += [Path("results/traces-pilot") / f"gpt-5.6-luna_{d}.jsonl" for d in ("math", "qa", "code")]
    sources += [Path(p["gemma_run"]) / n for p in pools for n in ("eval.json", "train_log.json")]
    plan = {"version": 1, "experiment": "E6", "revisions": revisions, "runs": runs,
            "pools": pools, "gemma_tokenizer": tokenizer,
            "runtime_sha256": {str(p): sha(root / p) for p in sources},
            "budgets_supervised": list(BUDGETS), "trajectory_processed": MILESTONES,
            "schedule_processed": 678000, "primary": str(PRIMARY), "fresh": str(FRESH),
            "v71_register": str(REGISTER), "preregistration": str(PREREG),
            "selection": "nearest completion_tokens_seen among saved positive trajectory checkpoints; "
                         "ties choose lower T then lower update; no interpolation or best-loss selection",
            "response": "delta = checkpoint loss - own update-0 loss; E = supervised T / full supervised D_U",
            "resume": "skip validated complete runs; archive incomplete runs and replay unchanged seed-0 CLI",
            "third_student": "not included; no outcome-dependent expansion"}
    plan["plan_sha256"] = digest(plan)
    return plan


def validate_plan(plan, root=ROOT):
    frozen_checks(root)
    require(plan["plan_sha256"] == digest({k: v for k, v in plan.items() if k != "plan_sha256"}),
            "Plan identity changed")
    for path, expected in plan["runtime_sha256"].items():
        require(sha(root / path) == expected, f"Frozen runtime input changed: {path}")
    require([(r["student"], r["U"], r["pool_seed"]) for r in plan["runs"]]
            == [(m, u, s) for m in STUDENTS for u in POOLS for s in SEEDS], "Wrong 12-run design")
    for run in plan["runs"]:
        args = gemma_arguments(run["U"], run["pool_seed"], root)
        args[args.index("--student") + 1] = f"{run['student']}@{run['revision']}"
        require(args == run["v12_argv"], "V12 flags differ from Gemma except student")


def plan_file(root=ROOT):
    return root / OUT / "plan.json"


def load_plan(root=ROOT):
    require(plan_file(root).is_file(), "Run --plan first on a host with Hub metadata access")
    plan = read(plan_file(root))
    validate_plan(plan, root)
    return plan


def host_guard(model, hostname=None):
    # User directive 2026-11-07: Qwen runs on rsi only (not rai, never hpg or LONI); rsi's hostname is sn4622131043.
    host = socket.gethostname() if hostname is None else hostname
    if "qwen" in str(model).lower() and not (host in ("sn4622131043",) or host.startswith("rsi")):
        raise RuntimeError(f"Qwen execution requires the rsi host; refused {host}")


def offline_environment(model):
    host_guard(model)  # Cannot be bypassed by an inherited SDL_ALLOW_PRC.
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if "qwen" in model.lower():
        env["SDL_ALLOW_PRC"] = "1"
    return env


def verify_run_metadata(ev, run):
    require(ev["resolved_student"] == run["student"] and ev["revision"] == run["revision"]
            and ev["n_per_domain"] == run["U"] and ev["data_seed"] == run["pool_seed"]
            and ev["training_seed"] == 0 and ev["training_mode"] == "lora"
            and ev["teacher"] == "gpt-5.6-luna" and ev["recipe"] == "full"
            and ev["schedule_tokens"] == 678000 and ev["trajectory_tokens_requested"] == MILESTONES
            and ev["data_sampling_seed"] == run["pool_seed"] and ev["learning_rate"] == 1e-4
            and ev["optimizer"] == "AdamW" and ev["scheduler"] == "cosine"
            and ev["warmup_ratio"] == .03 and ev["stop_after_trajectory"] is True
            and ev["domains"] == ["math", "qa", "code"] and ev["probe_seed"] == 0
            and ev["n_probe_requested"] == 128 and ev["output_suffix"] == "matrix2",
            "Saved trajectory does not match the frozen run")


def qwen_pool(run, plan, root=ROOT):
    """Full tokenized pool denominator, even if training never finishes epoch 1."""
    tokenizer, identity = cached_tokenizer(run["student"], run["revision"])
    priced = price_pool(tokenizer, run["U"], run["pool_seed"], root)
    ids = sorted(r["example_id"] for r in priced["_ledger"])
    reference = next(p for p in plan["pools"] if (p["U"], p["pool_seed"]) == (run["U"], run["pool_seed"]))
    require(ids == reference["sample_hashes"], "Qwen/Gemma sorted sample membership differs")
    return {"D_U": priced["D_U_pool"], "data_pool_sha256": priced["data_pool_sha256"],
            "sample_hashes": ids, "membership_equal_to_gemma": True,
            "unique_data_pool_tokens": priced["unique_data_pool_tokens"],
            "tokenizer_revision": run["revision"], "tokenizer_sha256": identity["file_sha256"]}


def execute_run(plan, index, root=ROOT):
    run = plan["runs"][index]
    env = offline_environment(run["student"])
    validate_plan(plan, root)
    state = root / OUT / "runs" / str(index)
    out = root / run["output_dir"]
    with lock(state / "run.lock"):
        owner = {"plan_sha256": plan["plan_sha256"], "index": index, "v12_argv": run["v12_argv"]}
        if (state / "owner.json").exists():
            require(read(state / "owner.json") == owner, "Run ownership/plan mismatch")
        else:
            require(not out.exists(), f"Refusing to adopt unowned output: {out}")
            write(state / "owner.json", owner)
        if (out / "eval.json").exists():
            ev = read(out / "eval.json")
            verify_run_metadata(ev, run)
            require(read(out / "train_log.json")["status"] == "trained", "Incomplete final training log")
            pool = read(state / "pool.json")
            require(ev["data_pool_sha256"] == pool["data_pool_sha256"]
                    and digest(pool["sample_hashes"]) == run["sample_hashes_sha256"],
                    "Completed run pool differs from plan")
            print(f"run {index}: already complete", flush=True)
            return
        os.environ.update({k: env[k] for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")})
        pool = qwen_pool(run, plan, root)
        # Check V12's actual builder, then leave the trainer entirely unchanged.
        from analysis.v12_distill import build_probes
        require(build_probes(128, seed=0) == read(root / PRIMARY), "Runtime primary probes drifted")
        if out.exists():
            attempt = 1
            while (state / f"interrupted-{attempt:04d}").exists():
                attempt += 1
            out.rename(state / f"interrupted-{attempt:04d}")
        write(state / "pool.json", pool)
        print(run["command_line"], flush=True)
        subprocess.run([sys.executable, str(TRAINER), *run["v12_argv"]], cwd=root,
                       env=env, stdin=subprocess.DEVNULL, check=True)
        ev = read(out / "eval.json")
        verify_run_metadata(ev, run)
        require(ev["data_pool_sha256"] == pool["data_pool_sha256"], "Trained pool differs from preflight")


def select_checkpoints(run_dir):
    saved = [(p, read(p)) for p in sorted((run_dir / "trajectory").glob("update-*/eval.json"))]
    baseline = [(p, ev) for p, ev in saved if ev["updates"] == 0]
    positive = [(p, ev) for p, ev in saved if ev["updates"] > 0 and ev["completion_tokens_seen"] > 0]
    require(len(baseline) == 1 and positive, f"Missing baseline or trained checkpoints: {run_dir}")
    selected = {t: min(positive, key=lambda item: (abs(item[1]["completion_tokens_seen"] - t),
                                                   item[1]["completion_tokens_seen"], item[1]["updates"]))
                for t in BUDGETS}
    return baseline[0], selected


def evaluate_run(plan, index, root=ROOT):
    run = plan["runs"][index]
    env = offline_environment(run["student"])
    os.environ.update({k: env[k] for k in ("SDL_ALLOW_PRC", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
                       if k in env})
    from analysis import v99_scope_eval as v99
    _, fresh = verify_panels(root)
    state, out = root / OUT / "runs" / str(index), root / run["output_dir"]
    with lock(state / "run.lock"):
        require((out / "eval.json").exists(), f"Training is incomplete for index {index}")
        verify_run_metadata(read(out / "eval.json"), run)
        pool = qwen_pool(run, plan, root)
        require(read(state / "pool.json") == pool, "Saved pool/tokenizer identity changed")
        baseline, selected = select_checkpoints(out)
        log = read(out / "train_log.json")
        require(log["status"] == "trained", "Training log is not complete")
        counts = {0: (0, 0)}
        counts.update({r["step"]: (r["processed_tokens"], r["completion_tokens_seen"])
                       for r in log["loss_curve"]})
        protocol = v99.load_protocol(root / REGISTER, v99.DISTRIBUTIONS, "cuda", "bfloat16")
        checkpoints = {str(p): (p, ev) for p, ev in [baseline, *selected.values()]}
        scored = {}
        for path, ev in checkpoints.values():
            verify_run_metadata(ev, run)
            require(all(ev.get(k) == baseline[1].get(k) for k in v99.IDENTITY_KEYS),
                    "Checkpoint identity differs from own update-0")
            require(counts.get(ev["updates"]) == (ev["processed_tokens"], ev["completion_tokens_seen"]),
                    "Checkpoint token counts disagree with training log")
            require(ev["data_pool_sha256"] == pool["data_pool_sha256"], "Checkpoint pool mismatch")
            identity = {**ev, "checkpoint": str(path.parent), **v99.adapter_identity(path.parent, ev)}
            evidence = {"plan_sha256": plan["plan_sha256"], "eval_sha256": sha(path),
                        "adapter_sha256": identity["adapter_sha256"], "pool": digest(pool)}
            cache = state / "scores" / f"{path.parent.name}.json"
            if cache.exists():
                result = read(cache)
                require(result["evidence"] == evidence, "Cached score belongs to changed inputs")
            else:
                raw = v99.score_checkpoint(identity, fresh, protocol)
                losses = {PRIMARY_KEYS[k]: {"loss": ev["post_training"][k],
                           "tokens": ev["measurement_tokens"][k], "n": ev["measurement_samples"][k],
                           "scorer": "V12.measure_capability_losses (saved checkpoint measurement)"}
                          for k in PRIMARY_KEYS}
                losses.update({FRESH_KEYS[k]: v for k, v in raw["losses"].items()})
                result = {"evidence": evidence, "losses": losses,
                          "fresh_adapter_loaded": raw["adapter_loaded"], "protocol": protocol}
                write(cache, result)
            require(result["fresh_adapter_loaded"] == (ev["updates"] > 0), "Wrong scored state")
            scored[str(path)] = result["losses"]
        initial = scored[str(baseline[0])]
        rows = []
        for target, (path, ev) in selected.items():
            for distribution in DISTRIBUTIONS:
                loss = scored[str(path)][distribution]["loss"]
                dense = initial[distribution]["loss"]
                rows.append({"student": run["student"], "U": run["U"], "pool_seed": run["pool_seed"],
                             "target_T": target, "T": ev["completion_tokens_seen"], "D_U": pool["D_U"],
                             "E": ev["completion_tokens_seen"] / pool["D_U"], "distribution": distribution,
                             "initial_loss": dense, "loss": loss, "delta": loss - dense,
                             "checkpoint": str(path.relative_to(root)), "checkpoint_sha256": sha(path),
                             "budget_error": ev["completion_tokens_seen"] - target})
        write(root / OUT / "evaluations" / f"{index}.json",
              {"plan_sha256": plan["plan_sha256"], "index": index, "status": "complete", "rows": rows})
        print(f"evaluated run {index}: {len(rows)} cells", flush=True)


def spread(values):
    return {"mean": statistics.fmean(values), "range": [min(values), max(values)]} if values else None


def analyse_rows(rows):
    """No fitted threshold, tolerance, significance claim, or dropped negative result."""
    indexed = {}
    for row in rows:
        key = (row["student"], row["U"], row["pool_seed"], row["target_T"], row["distribution"])
        require(key[0] in STUDENTS and key[1] in POOLS and key[2] in SEEDS
                and key[3] in BUDGETS and key[4] in DISTRIBUTIONS, f"Unexpected cell: {key}")
        require(key not in indexed, f"Duplicate cell: {key}")
        require(all(math.isfinite(row[k]) for k in ("loss", "initial_loss", "delta", "T", "D_U", "E"))
                and row["T"] > 0 and row["D_U"] > 0, f"Invalid measurement: {key}")
        require(math.isclose(row["delta"], row["loss"] - row["initial_loss"], abs_tol=1e-12)
                and math.isclose(row["E"], row["T"] / row["D_U"], abs_tol=1e-12), "Response/accounting mismatch")
        indexed[key] = row
    cells, lookup = [], {}
    for student in STUDENTS:
        for target in BUDGETS:
            for distribution in DISTRIBUTIONS:
                for u in POOLS:
                    values = [indexed[student, u, seed, target, distribution] for seed in SEEDS
                              if (student, u, seed, target, distribution) in indexed]
                    complete = len(values) == 2
                    cell = {"student": student, "target_T": target, "distribution": distribution, "U": u,
                            "pool_seeds": [r["pool_seed"] for r in values], "complete": complete,
                            **{k: spread([r[k] for r in values]) for k in ("delta", "T", "D_U", "E")},
                            "net_gain_at_U594": (statistics.fmean(r["delta"] for r in values) < 0)
                            if complete and u == 594 else None}
                    cells.append(cell)
                    lookup[student, target, distribution, u] = cell

    def contrast(student, target, distribution):
        low, high = (lookup[student, target, distribution, u] for u in (66, 594))
        differences = [indexed[student, 66, s, target, distribution]["delta"]
                       - indexed[student, 594, s, target, distribution]["delta"] for s in SEEDS
                       if all((student, u, s, target, distribution) in indexed for u in (66, 594))]
        complete = low["complete"] and high["complete"]
        cost = statistics.fmean(differences) if complete else None
        return {"student": student, "target_T": target, "distribution": distribution,
                "reuse_cost": cost, "paired_seed_range": spread(differences)["range"] if differences else None,
                "direction": None if cost is None else "positive" if cost > 0 else "negative" if cost < 0 else "zero",
                "specified_direction_observed": None if cost is None else cost > 0}

    h6a = [contrast(s, 200000, "2Wiki-probe") for s in STUDENTS]
    size = None if any(r["reuse_cost"] is None for r in h6a) else h6a[1]["reuse_cost"] - h6a[0]["reuse_cost"]
    h6b = {"target_T": 200000, "distribution": "2Wiki-probe", "larger_minus_smaller_reuse_cost": size,
           "specified_direction_observed": None if size is None else size > 0}
    # H6c does not name a single budget in the preregistration: report all three.
    h6c = [contrast(s, t, d) for s in STUDENTS for t in BUDGETS for d in FRESH_KEYS.values()]
    gains = []
    for s in STUDENTS:
        for t in BUDGETS:
            for d in DISTRIBUTIONS:
                low, high = (lookup[s, t, d, u] for u in (66, 594))
                gains.append({"student": s, "target_T": t, "distribution": d,
                              "net_gain_at_U594": high["net_gain_at_U594"],
                              "crosses_from_U66_non_gain_to_U594_gain":
                              low["delta"]["mean"] >= 0 > high["delta"]["mean"]
                              if low["complete"] and high["complete"] else None})
    any_gains = []
    for s in STUDENTS:
        for t in BUDGETS:
            selected = [g for g in gains if g["student"] == s and g["target_T"] == t]
            complete = all(g["net_gain_at_U594"] is not None for g in selected)
            any_gains.append({"student": s, "target_T": t,
                              "any_distribution_net_gain_at_U594": any(g["net_gain_at_U594"] for g in selected)
                              if complete else None,
                              "gaining_distributions": [g["distribution"] for g in selected if g["net_gain_at_U594"]]})
    return {"status": "complete" if len(indexed) == 216 else "partial" if indexed else "not_measured",
            "measured_cells": len(indexed), "expected_cells": 216, "cells": cells,
            "H6a": h6a, "H6b": h6b, "H6c": h6c,
            "H6c_budget_reading": "All three registered budgets; preregistration does not restrict H6c to one.",
            "descriptive_math_code": [c for c in cells if c["target_T"] == 200000 and c["distribution"] in ("MATH-500", "MBPP")],
            "descriptive_net_gains": gains,
            "descriptive_any_net_gain": any_gains,
            "reading": "Strict signed directions; zero is neither positive nor a net gain. No tolerance or significance test. "
                       "Two pool seeds; checkpoints within a trajectory are dependent. Missing seeds leave hypotheses pending.",
            "rows": sorted(rows, key=lambda r: (r["student"], r["target_T"], r["distribution"], r["U"], r["pool_seed"]))}


def format_spread(value, digits=3):
    if value is None:
        return "--"
    return f"{value['mean']:.{digits}f} [{value['range'][0]:.{digits}f}, {value['range'][1]:.{digits}f}]"


def markdown(summary):
    lines = ["# E6 controlled reuse on a second student family", "",
             f"Status: {summary['status']}; {summary['measured_cells']}/216 seed-level cells.", "",
             summary["reading"], "", "delta = L(checkpoint) - L(own update-0); negative is gain. E = T / D_U, both supervised tokens.",
             "Reported cells are seed means [minimum, maximum]; unmeasured cells are --.", ""]
    for hypothesis in ("H6a", "H6b", "H6c"):
        lines += [f"## {hypothesis}", ""]
        if hypothesis == "H6c":
            lines += [summary["H6c_budget_reading"], ""]
        entries = summary[hypothesis] if isinstance(summary[hypothesis], list) else [summary[hypothesis]]
        for entry in entries:
            lines.append("- " + json.dumps(entry, sort_keys=True))
        lines.append("")
    lines += ["## All response cells", "", "| Student | Target T | Distribution | U | Seeds | Actual T | E | delta |",
              "|---|---:|---|---:|---|---|---|---|"]
    for c in summary["cells"]:
        lines.append(f"| {c['student'].split('/')[-1]} | {c['target_T']} | {c['distribution']} | {c['U']} | "
                     f"{','.join(map(str, c['pool_seeds'])) or '--'} | {format_spread(c['T'], 0)} | "
                     f"{format_spread(c['E'], 2)} | {format_spread(c['delta'])} |")
    lines += ["", "## Descriptive net gain and crossing at U=594", ""]
    lines.extend("- " + json.dumps(g, sort_keys=True) for g in summary["descriptive_any_net_gain"])
    lines.extend("- " + json.dumps(g, sort_keys=True) for g in summary["descriptive_net_gains"])
    return "\n".join(lines) + "\n"


def table_tex(summary):
    from analysis.paper_table_layout import house_style, table_layout
    cells = {(c["student"], c["target_T"], c["distribution"], c["U"]): c for c in summary["cells"]}
    tex = ["% Generated by analysis/r1_reuse_family.py; do not edit."]
    for i, student in enumerate(STUDENTS):
        tex += [r"\begin{table}[tb]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}",
                r"\caption{\rev{Controlled reuse on " + student.split("/")[-1] +
                r". Responses $\delta=L(S_{\rm KD})-L(S_0)$ are in nats per native token; negative is a gain. "
                r"Cells give pool-seed means [minimum, maximum] for seeds 41 and 42. "
                r"$U$ counts traces per domain; $T$ and $D_U$ count supervised tokens and $E=T/D_U$. "
                r"Checkpoints are nearest the indicated supervised budget. "
                + f"{summary['measured_cells']}/216 seed-level cells measured across both students. "
                + r"Dashes denote unmeasured cells; a superscript $\dagger$ denotes only one measured seed.}}",
                rf"\label{{tab:r1_reuse_family{'_large' if i else ''}}}",
                r"\begin{tabular}{lrrr}", r"\toprule",
                r"Readout & $U=66$ & $U=198$ & $U=594$ \\", r"\midrule"]
        for j, target in enumerate(BUDGETS):
            if j:
                tex.append(r"\midrule")
            tex.append(r"\multicolumn{4}{l}{Target $T=" + str(target // 1000) + r"$k} \\")
            for label, field, digits in ((r"Actual $T$", "T", 0), (r"Reuse $E$", "E", 2)):
                tex.append(label + " & " + " & ".join(format_spread(cells[student, target, "MATH-500", u][field], digits)
                                                      for u in POOLS) + r" \\")
            for distribution in DISTRIBUTIONS:
                entries = []
                for u in POOLS:
                    c = cells[student, target, distribution, u]
                    entries.append(format_spread(c["delta"]) + (r"$^{\dagger}$" if c["pool_seeds"] and not c["complete"] else ""))
                tex.append(distribution + " & " + " & ".join(entries) + r" \\")
        tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return house_style(table_layout("\n".join(tex) + "\n"))


def collect_summary(root=ROOT):
    rows = []
    files = sorted((root / OUT / "evaluations").glob("*.json"))
    plan = load_plan(root) if files else None
    for path in files:
        value = read(path)
        require(value["status"] == "complete" and value["plan_sha256"] == plan["plan_sha256"],
                f"Evaluation/plan mismatch: {path}")
        for row in value["rows"]:
            require(sha(root / row["checkpoint"]) == row["checkpoint_sha256"], "Analysed checkpoint changed")
        rows.extend(value["rows"])
    summary = analyse_rows(rows)
    summary["preregistration_sha256"] = PREREG_SHA
    summary["plan_sha256"] = plan["plan_sha256"] if plan else None
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("plan", "run", "evaluate", "analyse", "table"):
        mode.add_argument("--" + name, action="store_true")
    parser.add_argument("--index", type=int, choices=range(12))
    args = parser.parse_args(argv)
    frozen_checks()
    if args.plan:
        with lock(ROOT / OUT / "plan.lock"):
            plan = load_plan() if plan_file().exists() else make_plan()
            write(plan_file(), plan)
        print(f"Frozen {len(plan['runs'])} runs: {plan_file()}")
    elif args.run:
        require(args.index is not None, "--run requires --index")
        host_guard(STUDENTS[args.index // 6])  # Before imports, cache access, or writes.
        execute_run(load_plan(), args.index)
    elif args.evaluate:
        host_guard(STUDENTS[0])
        plan = load_plan()
        for index in range(12) if args.index is None else [args.index]:
            if args.index is None and not (ROOT / plan["runs"][index]["output_dir"] / "eval.json").exists():
                print(f"run {index}: not trained; pending")
                continue
            evaluate_run(plan, index)
    else:
        summary = collect_summary()
        write(ROOT / OUT / "summary.json", summary)
        write_text(ROOT / OUT / "summary.md", markdown(summary))
        if args.table:
            write_text(ROOT / "paper/paper/tables/r1_reuse_family.tex", table_tex(summary))
        print(f"{summary['status']}: {summary['measured_cells']}/216 cells")


if __name__ == "__main__":
    main()
