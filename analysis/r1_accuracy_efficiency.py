#!/usr/bin/env python3
"""E5: the frozen R1 accuracy-efficiency experiment (no measurements at import).

Run --plan, then --measure --dense-only for all six development states, then
--measure --state TAG for each state, --analyse, and --table. The shell driver
performs the pilot pass automatically. Completed densities require no model
loading; incomplete densities resume at the last atomically saved batch.

Inference, prompts, stopping, scoring and pruning are imported from V15/V6;
the linear estimators, LOSO tuning grids and interpolation are imported from
A18. The loss route uses A18's stored *probe* losses, never new loss runs.
Bootstrap uncertainty is conditional on the fitted predictors, as in A18:
the same item draw is paired across states, densities and dense anchors.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import unicodedata
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis import a18_second_family as a18
from analysis import v15_accuracy_link as v15
from analysis import v6_capability_geometry as v6
from analysis import v9_capability_regions as v9

OUT = ROOT / "results/r1-accuracy-efficiency"
PREREG = ROOT / "notes/rebuttal/prereg_r1_accuracy.md"
if not PREREG.exists():  # Identical copy in the paper checkout.
    PREREG = ROOT.parent / "notes/rebuttal/prereg_r1_accuracy.md"
CAPS = a18.CAPS
STATES, REPOS = a18.STATES, a18.REPOS
FULL, REDUCED, TEST = a18.DEV_DENSITIES, a18.REDUCED, a18.TEST_DENSITIES
N_ITEMS, SEED, BOOT_N = 256, 0, 2000
METHODS = ("compact", "per_density", "median", "pruning_law", "loss_route", "zero")
SCHEMA = 1


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(obj):
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def atomic_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_bytes(json_bytes(obj))
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def text_hashes(text):
    """Hash full text AND unwrapped tasks/questions, independent of prompt style.

    Checking only a rendered prompt would miss a teacher's 'Task:' wrapper or
    a QA probe with context. Answers/completions are deliberately not item IDs.
    """
    parts = [text]
    patterns = (
        r"(?:^|\n)(?:Problem|Question):\s*(.*?)(?=\n(?:Solution|Answer):|\Z)",
        r"(?:^|\n)(?:#\s*)?Task:\s*(.*?)(?=\n# Write a Python function\.|\Z)",
        r"here is your task:\s*(.*?)(?=\s*Your code should pass these tests:|\Z)",
    )
    for pattern in patterns:
        parts.extend(re.findall(pattern, text, flags=re.S | re.I))
    return {hashlib.sha256(normalized(p).encode()).hexdigest() for p in parts if normalized(p)}


def query_texts(obj):
    """Read nested probe/trace records, including chat-format teacher inputs."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"prompt", "question", "problem", "task", "text", "instruction", "input"} and isinstance(value, str):
                yield value
            elif key == "content" and obj.get("role") in {"user", "human"} and isinstance(value, str):
                yield value
            elif isinstance(value, (dict, list)):
                yield from query_texts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from query_texts(value)


def exclusion_sources(root=ROOT):
    """Rebuild mandatory registries; a partial registry is not an overlap audit."""
    sources = {}

    def add(name, obj):
        texts = list(query_texts(obj))
        if not texts:
            raise ValueError(f"No query text in exclusion source {name}")
        sources[name] = sorted(set().union(*(text_hashes(t) for t in texts)))

    for cap in CAPS:
        add(f"v15_fewshot/{cap}", v15.FEWSHOT[cap]["exemplars"])
    probes = v6.build_probes(128, seed=0, include_secondary=True)
    required = set(CAPS) | set(v6.SECONDARY_BENCHMARKS)
    if not required <= probes.keys() or any(len(probes[k]) != 128 for k in required):
        raise ValueError("Incomplete V6 primary/secondary exclusion probes")
    for key, rows in probes.items():
        add(f"v6_128_seed0/{key}/both_halves", rows)
    # The paper's V9 runs and CLI use 64; also honor any recorded other runs.
    specs = {(64, 0)}
    for path in sorted((root / "results/v9-capability-regions").glob("*/fisher_meta.json")):
        meta = read(path)
        specs.add((int(meta["n_probe_requested"]), int(meta["probe_seed"])))
    for n, seed in sorted(specs):
        registry = v9.build_probe_registry(n, seed=seed)
        if set(registry) != {spec.name for spec in v9.PROBE_REGISTRY}:
            raise ValueError("Incomplete V9 exclusion registry (a dataset was skipped)")
        for key, rows in registry.items():
            if key != "c4" and len(rows) != n:
                raise ValueError(f"Incomplete V9 {key} exclusion probes")
            add(f"v9_{n}_seed{seed}/{key}", rows)
    easy = v15.build_easy_probes(n=128, seed=0)
    if set(easy) != set(CAPS):
        raise ValueError("Incomplete V15 exclusion capabilities")
    for key, rows in easy.items():
        if len(rows) != 128:
            raise ValueError(f"Incomplete V15 {key} exclusion probes")
        add(f"v15_easy_128_seed0/{key}/both_halves", rows)
    # Include persisted probe panels and all JSON teacher inputs under data/.
    paths = set((root / "data").rglob("*.json")) | set((root / "data").rglob("*.jsonl"))
    paths |= set((root / "results").rglob("probes.json"))
    paths |= set((root / "results/v91-dense-stats/probes").glob("*.json"))
    # Current teacher inputs; _backup contains superseded, malformed outputs.
    paths |= set((root / "results/traces-pilot").glob("*.jsonl"))
    if (root / "results/a18-second-family/items.json").exists():
        paths.add(root / "results/a18-second-family/items.json")
    files = {}
    for path in sorted(paths):
        name = path.relative_to(root).as_posix()
        objects = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                   if path.suffix == ".jsonl" else read(path))
        texts = list(query_texts(objects))
        files[name] = {"sha256": digest(path), "query_texts": len(texts)}
        if texts:
            add(name, objects)
    return sources, files


def protocol():
    return {"shots": dict(v15.DOMAIN_DEFAULT_SHOTS), "max_new_tokens": dict(v15.MAX_NEW_TOKENS),
            "stop_sequences": {k: list(v) for k, v in v15.DOMAIN_STOP_SEQUENCES.items()},
            "decoding": "greedy, left padding, V15 left truncation and input length",
            "datasets": v15.ACCURACY_BENCHMARK_DATASETS["easy"],
            "split_note": "TriviaQA uses rc.nocontext validation, the labelled held-out split used by V15 easy.",
            "pruning_sample_size": v6.PRUNE_THRESHOLD_SAMPLE_SIZE, "pruning_seed": 0}


def build_plan(pools, exclusions, *, n=N_ITEMS, seed=SEED, files=None):
    """Pure deterministic draw after exclusion and text deduplication."""
    banned = set().union(*(set(v) for v in exclusions.values()))
    rng = np.random.default_rng(seed)
    items, counts = {}, {}
    for cap in CAPS:
        eligible, seen = [], set()
        for row in sorted(pools[cap], key=lambda r: (normalized(r["prompt"]), json_bytes(r))):
            hashes = text_hashes(row["prompt"])
            if hashes & (banned | seen):
                continue
            seen.update(hashes)
            eligible.append({**row, "text_hashes": sorted(hashes),
                             "id": hashlib.sha256(normalized(row["prompt"]).encode()).hexdigest()})
        counts[cap] = {"pool": len(pools[cap]), "eligible_unique": len(eligible)}
        if len(eligible) < n:
            raise ValueError(f"{cap}: only {len(eligible)} disjoint unique items; need {n}. Do not relax the frozen exclusions.")
        items[cap] = [eligible[int(i)] for i in rng.choice(len(eligible), n, replace=False)]
    result = {"schema_version": SCHEMA, "seed": seed, "n_per_capability": n, "items": items,
              "states": {t: {"size": s, "revision": r, "role": role, "repo": REPOS[s],
                              "densities": list(FULL if role == "dev" else TEST)} for t, (s, r, role) in STATES.items()},
              "reduced_densities": list(REDUCED), "protocol": protocol(),
              "hash_rule": "SHA256 of NFKC/casefold/whitespace-normalized full prompt and unwrapped question/task",
              "exclusion_hashes": exclusions, "exclusion_files": files or {}, "pool_counts": counts,
              "prereg_sha256": digest(PREREG)}
    validate_plan(result)
    return result


def validate_plan(plan):
    if plan["schema_version"] != SCHEMA or plan["protocol"] != protocol():
        raise ValueError("Plan schema/protocol differs from this implementation")
    banned = set().union(*(set(v) for v in plan["exclusion_hashes"].values()))
    for cap in CAPS:
        seen = set()
        rows = plan["items"][cap]
        if len(rows) != plan["n_per_capability"]:
            raise ValueError(f"Wrong item count: {cap}")
        for row in rows:
            hashes = text_hashes(row["prompt"])
            expected_id = hashlib.sha256(normalized(row["prompt"]).encode()).hexdigest()
            if hashes != set(row["text_hashes"]) or row["id"] != expected_id or hashes & (banned | seen):
                raise ValueError(f"Item hash mismatch, duplicate or overlap: {cap}")
            seen.update(hashes)


def load_plan(out=OUT):
    path = out / "plan.json"
    sha = digest(path)
    if sha != (out / "plan.sha256").read_text().strip():
        raise ValueError("Frozen plan digest mismatch")
    plan = read(path)
    validate_plan(plan)
    if plan["prereg_sha256"] != digest(PREREG):
        raise ValueError("Pre-registration digest mismatch")
    return plan, sha


def plan_stage(out=OUT):
    with lock(out / ".plan.lock"):
        if (out / "plan.json").exists():
            _, sha = load_plan(out)
            print(f"Plan already frozen: {sha}", flush=True)
            return
        exclusions, files = exclusion_sources()
        # min(n, dataset length) in V15 selects the entire labelled split.
        pools = v15.build_easy_probes(n=sys.maxsize, seed=SEED)
        plan = build_plan(pools, exclusions, files=files)
        atomic_write(out / "plan.json", plan)
        (out / "plan.sha256").write_text(digest(out / "plan.json") + "\n")
        print("Frozen 256 items per capability:", plan["pool_counts"], flush=True)


def measurement_path(out, state, density):
    return out / "measurements" / state / f"{float(density)}.json"


def synchronize(device):
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)


class TokenRecorder:
    """Transparent tokenizer proxy capturing the IDs decoded inside V15."""
    def __init__(self, tokenizer):
        object.__setattr__(self, "tokenizer", tokenizer)
        object.__setattr__(self, "input_tokens", [])
        object.__setattr__(self, "output_ids", [])

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __setattr__(self, name, value):
        if name in {"tokenizer", "input_tokens", "output_ids"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.tokenizer, name, value)

    def __call__(self, *args, **kwargs):
        encoded = self.tokenizer(*args, **kwargs)
        self.input_tokens = encoded["attention_mask"].sum(dim=1).tolist()
        return encoded

    def batch_decode(self, ids, **kwargs):
        self.output_ids = ids.detach().cpu().tolist()
        return self.tokenizer.batch_decode(ids, **kwargs)


def generated_length(ids, tokenizer, stops):
    """Count actual emission through EOS/delimiter, excluding subsequent pads."""
    eos = tokenizer.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos]))
    sequences = v15._encode_stop_sequences(tokenizer, stops)
    for end, token in enumerate(ids, 1):
        if token in eos or any(end >= len(s) and ids[end-len(s):end] == s for s in sequences):
            return end
    return len(ids)


def generate_batch(model, tokenizer, rows, cap, device):
    recorder = TokenRecorder(tokenizer)
    shots = v15.DOMAIN_DEFAULT_SHOTS[cap]
    outputs = v15.greedy_generate_batch(
        model, recorder, [v15.build_accuracy_prompt(cap, row, shots) for row in rows],
        device=device, max_new_tokens=v15.MAX_NEW_TOKENS[cap],
        input_max_length=max(v15.MAX_LEN // 2, v15.MAX_LEN - v15.MAX_NEW_TOKENS[cap]),
        truncation_side="left", stop_sequences=v15.DOMAIN_STOP_SEQUENCES[cap])
    result = []
    for row, text, ids, nin in zip(rows, outputs, recorder.output_ids, recorder.input_tokens, strict=True):
        nout = generated_length(ids, tokenizer, v15.DOMAIN_STOP_SEQUENCES[cap])
        result.append({"id": row["id"], "raw_generation_full": text,
                       "generation": v15.truncate_generation(cap, text),
                       "input_tokens": int(nin), "generated_tokens": nout,
                       "generated_token_ids": ids[:nout]})
    return result


def score_item(cap, item, generated):
    text = generated["generation"]
    if cap == "math":
        correct, extracted, prediction, target = v15.gsm8k_exact_match(text, str(item["answer"]))
        details = {"extracted_answer": extracted, "normalized_prediction": prediction, "normalized_reference": target}
    elif cap == "code":
        execution = v15.run_code_tests(text, item["test_list"], str(item.get("test_setup_code", "")))
        correct = execution["passed"]
        details = {"execution": execution, "executed_completion": v15.extract_code_completion(text)}
    else:
        prediction = v15.extract_triviaqa_answer(text)
        aliases = item.get("answer_aliases", [item["answer"]])
        correct = v15.qa_alias_exact_match(prediction, aliases)
        details = {"extracted_answer": prediction, "reference_aliases": aliases}
    return {**generated, "correct": bool(correct), **details}


def agreement_check(model, tokenizer, plan, device, batch_size):
    """Eight dense items: 3 math, 3 code, 2 QA; differences are diagnostic."""
    checks, tokens = [], 0
    synchronize(device)
    started = time.perf_counter()
    for cap, n in zip(CAPS, (3, 3, 2)):
        rows = plan["items"][cap][:n]
        batched = []
        for start in range(0, len(rows), max(2, batch_size)):
            batched.extend(generate_batch(model, tokenizer, rows[start:start+max(2, batch_size)], cap, device))
        single = [generate_batch(model, tokenizer, [row], cap, device)[0] for row in rows]
        for a, b in zip(batched, single, strict=True):
            tokens += a["generated_tokens"] + b["generated_tokens"]
            checks.append({"cap": cap, "id": a["id"], "agrees": a["raw_generation_full"] == b["raw_generation_full"],
                           "batched": a["raw_generation_full"], "unbatched": b["raw_generation_full"]})
    synchronize(device)
    rate = sum(row["agrees"] for row in checks) / len(checks)
    print(f"Dense batch agreement: {rate:.3f} ({sum(r['agrees'] for r in checks)}/{len(checks)})", flush=True)
    return {"n": len(checks), "agreement_rate": rate, "items": checks, "batch_size": max(2, batch_size),
            "generated_tokens": tokens, "seconds": time.perf_counter() - started}


def validate_measurement(record, plan, sha, state, density, *, complete=False):
    spec = plan["states"][state]
    if (record["schema_version"] != SCHEMA or record["plan_sha256"] != sha
            or record["state"] != state or record["density"] != float(density)
            or any(record[k] != spec[k] for k in ("size", "repo", "revision", "role"))):
        raise ValueError("Measurement identity/schema mismatch")
    if complete and record["status"] != "complete":
        raise ValueError(f"Incomplete measurement: {state}/{density}")
    for cap in CAPS:
        rows = record["items"][cap]
        expected = plan["items"][cap]
        if len(rows) > len(expected) or (record["status"] == "complete" and len(rows) != len(expected)):
            raise ValueError("Measurement item count mismatch")
        for row, item in zip(rows, expected):
            if (row["id"] != item["id"] or type(row["correct"]) is not bool
                    or not isinstance(row["raw_generation_full"], str)
                    or row["generated_tokens"] != len(row["generated_token_ids"])
                    or row["input_tokens"] < 1):
                raise ValueError("Measurement item schema/order mismatch")
    if record["status"] == "complete" and float(density) == 1.0 and record["batch_agreement"] is None:
        raise ValueError("Dense measurement lacks batch agreement audit")


def accuracy(record, cap):
    return float(np.mean([row["correct"] for row in record["items"][cap]]))


def pilot_gate(out, plan, sha):
    dense, inputs = {}, {}
    for state, spec in plan["states"].items():
        if spec["role"] != "dev":
            continue
        path = measurement_path(out, state, 1.0)
        if not path.exists():
            raise ValueError(f"Pilot requires every dense development state before pruning: missing {state}. Run --dense-only first.")
        record = read(path)
        validate_measurement(record, plan, sha, state, 1.0, complete=True)
        dense[state] = {c: accuracy(record, c) for c in CAPS}
        inputs[state] = digest(path)
    one_b = [t for t in dense if plan["states"][t]["size"] == "1B"]
    if len(one_b) != 3:
        raise ValueError("Pilot requires the three registered 1B development states")
    return {"dense_accuracy": dense, "dense_sha256": inputs,
            "seven_b_only": {c: all(dense[t][c] < 0.05 for t in one_b) for c in CAPS}}


def measure_state(state, *, out=OUT, device="cuda:0", batch_size=8, dense_only=False):
    import torch
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    plan, sha = load_plan(out)
    spec = plan["states"][state]
    densities = [1.0] + ([] if dense_only else spec["densities"])
    state_dir = out / "measurements" / state
    with lock(state_dir / ".lock"):
        pending = []
        for d in densities:
            path = measurement_path(out, state, d)
            if path.exists():
                record = read(path)
                validate_measurement(record, plan, sha, state, d)
                if record["status"] == "complete":
                    continue
            pending.append(d)
        if not pending:
            print(f"{state}: all requested densities complete", flush=True)
            return
        if 1.0 not in pending and not dense_only:
            pilot_gate(out, plan, sha)  # Refuse before loading, when possible.
        synchronize(device)
        started = time.perf_counter()
        model, tokenizer = v6.load_text_causal_lm(spec["repo"], torch.bfloat16, spec["revision"])
        model.to(device).eval()
        params = v6.language_weight_parameters(model)
        n0 = sum(p.numel() for name, p in params if ".layers." in name or ".blocks." in name)
        if n0 <= 0:
            raise ValueError("No A18 block-matrix parameters found")
        thresholds, reference = {}, None
        if any(d != 1 for d in pending):
            sample = v6._sample_abs_weights(params)
            thresholds = {d: float(np.quantile(sample, 1-d)) for d in pending if d != 1}
            reference = [p.detach().clone().cpu() for _, p in params]
        synchronize(device)
        run = {"id": uuid.uuid4().hex, "load_seconds": time.perf_counter()-started,
               "device": device, "batch_size": batch_size, "plan_sha256": sha}
        runs_path = state_dir / "loads.json"
        runs = read(runs_path) if runs_path.exists() else []
        runs.append(run)
        atomic_write(runs_path, runs)
        try:
            for d in pending:
                if d != 1:
                    gate = pilot_gate(out, plan, sha)
                    with lock(out / ".pilot.lock"):
                        if (out / "pilot.json").exists() and read(out / "pilot.json") != gate:
                            raise ValueError("Dense pilot inputs changed after pruning began")
                        atomic_write(out / "pilot.json", gate)
                path = measurement_path(out, state, d)
                record = read(path) if path.exists() else {
                    "schema_version": SCHEMA, "plan_sha256": sha, "state": state,
                    **{k: spec[k] for k in ("size", "repo", "revision", "role")},
                    "density": d, "N0_block_matrices": n0, "D0": a18.tokens_of(spec["revision"]),
                    "prune_scope_parameters": sum(p.numel() for _, p in params),
                    "threshold": thresholds.get(d), "status": "in_progress", "items": {c: [] for c in CAPS},
                    "seconds": 0.0, "load_run_ids": [], "batch_agreement": None}
                record["load_run_ids"].append(run["id"])
                synchronize(device)
                started = time.perf_counter()
                if d != 1:
                    if record["threshold"] != thresholds[d]:
                        raise ValueError("Pruning threshold changed on resume")
                    v6.apply_global_magnitude_pruning(model, d, reference_weights=reference, threshold=thresholds[d])
                synchronize(device)
                record["seconds"] += time.perf_counter()-started
                atomic_write(path, record)
                for cap in CAPS:
                    rows = plan["items"][cap]
                    for start in range(len(record["items"][cap]), len(rows), batch_size):
                        batch = rows[start:start+batch_size]
                        synchronize(device)
                        started = time.perf_counter()
                        generated = generate_batch(model, tokenizer, batch, cap, device)
                        scored = [score_item(cap, item, gen) for item, gen in zip(batch, generated, strict=True)]
                        synchronize(device)
                        record["seconds"] += time.perf_counter()-started
                        record["items"][cap].extend(scored)
                        atomic_write(path, record)
                if d == 1 and record["batch_agreement"] is None:
                    record["batch_agreement"] = agreement_check(model, tokenizer, plan, device, batch_size)
                    record["seconds"] += record["batch_agreement"]["seconds"]
                record["accuracy"] = {c: accuracy(record, c) for c in CAPS}
                record["generated_tokens"] = sum(r["generated_tokens"] for rows in record["items"].values() for r in rows)
                record["gpu_seconds"] = record["seconds"] if str(device).startswith("cuda") else 0.0
                record["status"] = "complete"
                validate_measurement(record, plan, sha, state, d, complete=True)
                atomic_write(path, record)
                print(state, d, record["accuracy"], f"{record['seconds']:.1f}s", flush=True)
        finally:
            del model, tokenizer, reference, params
            gc.collect()
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()


def accuracy_rows(states, *, role, cap, seven_b_only=False, grid="full"):
    rows = []
    for state, cells in states.items():
        dense = cells["1.0"]
        if dense["role"] != role or (seven_b_only and dense["size"] != "7B"):
            continue
        a0 = accuracy(dense, cap)
        densities = TEST if role == "test" else (REDUCED if grid == "reduced" else FULL)
        for d in densities:
            rows.append({"tag": state, "cap": cap, "d": d, "Acc0": a0,
                         "n_items": len(dense["items"][cap]),
                         "phi_raw": a18.p53.raw_features(dense["N0_block_matrices"], dense["D0"], a0),
                         "y": accuracy(cells[str(d)], cap)-a0})
    if not rows:
        raise ValueError(f"No {role} rows for {cap}/{grid}")
    return rows


def fit_pruning_law(rows):
    """One scale and exponent per capability; half-item floor for zero labels.

    As in Table 17, regress log accuracy ratios. Standardization, ridge grid
    and leave-one-state-out selection use A18; no test-state calibration.
    """
    d = np.array([r["d"] for r in rows])
    x = np.log(d)
    center, scale = float(x.mean()), float(x.std())
    z = np.column_stack([np.ones(len(rows)), (x-center)/scale])
    floor = 0.5 / rows[0]["n_items"]
    y = np.log([max(r["Acc0"]+r["y"], floor)/max(r["Acc0"], floor) for r in rows])
    lam, mae = a18.loso_lambda(rows, z, y, d, lambda zz, dd: zz)
    beta = a18.ridge(z, y, lam)
    alpha = float(beta[1]/scale)
    log_p0 = float(beta[0]-alpha*center)
    return {"P0": float(np.exp(log_p0)), "alpha": alpha, "lambda": lam,
            "loso_log_ratio_mae": mae, "zero_floor": floor}


def fit_logistic(losses, accuracies):
    """Table-17 logistic least squares, with a stable sigmoid at the endpoints."""
    from scipy.optimize import least_squares
    from scipy.special import expit
    losses, accuracies = np.asarray(losses), np.asarray(accuracies)
    if np.ptp(accuracies) == 0:
        return {"constant": float(accuracies[0])}
    mean = float(np.clip(accuracies.mean(), 1e-3, 1-1e-3))
    initial = [float(np.log(mean/(1-mean)) + losses.mean()), -1.0]
    result = least_squares(lambda p: expit(p[0]+p[1]*losses)-accuracies, initial)
    if not result.success:
        raise ValueError(f"Logistic link failed: {result.message}")
    return {"beta": result.x.tolist()}


def link_predict(link, loss):
    from scipy.special import expit
    if "constant" in link:
        return link["constant"]
    return float(expit(link["beta"][0]+link["beta"][1]*loss))


def fit_loss_route(states, loss_states, tags, cap):
    # A18's rows and standardization, restricted to the registered reduced grid.
    rows = [r for r in a18.dev_rows({t: loss_states[t] for t in tags}) if r["d"] in REDUCED]
    stats = a18.p53.zstats(rows)
    fit = a18.fit_forms(rows, cap, stats)["power"]
    losses, labels, pairs = [], [], []
    for tag in tags:
        for d in (1.0, *REDUCED):
            losses.append(loss_states[tag]["measurements"][str(d)]["probes"][cap]["mean_loss"])
            labels.append(accuracy(states[tag][str(d)], cap))
            pairs.append({"tag": tag, "d": d})
    return {"standardization": stats, "compact": fit, "link": fit_logistic(losses, labels),
            "generation_label_grid": "reduced+dense", "label_pairs": pairs,
            "loss_source": "A18 probes (odd half), stored measurements"}


def fit_grid(rows, cap, loss_route):
    stats = a18.p53.zstats(rows)
    forms = a18.fit_forms(rows, cap, stats)
    return {"standardization": stats, "compact": forms["power"],
            "per_density": forms["per_density"], "median": forms["median"],
            "pruning_law": fit_pruning_law(rows), "loss_route": loss_route, "zero": {},
            "training_rows": [{"tag": r["tag"], "d": r["d"]} for r in rows]}


def predict_six(fit, row, loss_state):
    z = a18.p53.standardize(row["phi_raw"], fit["standardization"])
    d, a0 = row["d"], row["Acc0"]
    law = fit["pruning_law"]
    route = fit["loss_route"]
    l0 = loss_state["measurements"]["1.0"]["probes"][row["cap"]]["mean_loss"]
    raw = a18.p53.raw_features(loss_state["N0_block_matrices"], loss_state["D0"], l0)
    loss_z = a18.p53.standardize(raw, route["standardization"])
    predicted_loss = l0 + a18.predict("power", route["compact"], loss_z, d)
    return {"compact": a18.predict("power", fit["compact"], z, d),
            "per_density": a18.predict("per_density", fit["per_density"], z, d),
            "median": a18.predict("median", fit["median"], z, d),
            "pruning_law": float(a0*(law["P0"]*d**law["alpha"]-1)),
            "loss_route": link_predict(route["link"], predicted_loss)-a0, "zero": 0.0}


def item_bootstrap(states, cells, *, n_resamples=BOOT_N, seed=SEED):
    """Paired item bootstrap, fixed states and fitted predictions (A18 reading)."""
    rng = np.random.default_rng(seed)
    result = {"n_resamples": n_resamples, "seed": seed, "states_fixed": True,
              "predictions_fixed": True, "shared_item_draw_across_states": True, "capabilities": {}}
    for cap in CAPS:
        targets, compact, comparator = [], [], []
        for cell in cells[cap]:
            measured = states[cell["tag"]]
            dense = measured["1.0"]["items"][cap]
            pruned = measured[str(cell["d"])]["items"][cap]
            if [r["id"] for r in dense] != [r["id"] for r in pruned]:
                raise ValueError("Bootstrap requires paired item order")
            targets.append([float(p["correct"])-float(b["correct"]) for p, b in zip(pruned, dense, strict=True)])
            compact.append(cell["predictions"]["reduced"]["compact"])
            comparator.append(cell["predictions"]["full"]["per_density"])
        targets = np.asarray(targets)
        compact, comparator = np.asarray(compact), np.asarray(comparator)
        draws = []
        for _ in range(n_resamples):
            indices = rng.integers(0, targets.shape[1], size=targets.shape[1])
            y = targets[:, indices].mean(axis=1)
            draws.append([float(np.abs(compact-y).mean()), float(np.abs(comparator-y).mean())])
        draws = np.asarray(draws)
        differences = draws[:, 0]-draws[:, 1]
        y = targets.mean(axis=1)
        result["capabilities"][cap] = {
            "compact_reduced_ci95": np.percentile(draws[:, 0], [2.5, 97.5]).tolist(),
            "per_density_full_ci95": np.percentile(draws[:, 1], [2.5, 97.5]).tolist(),
            "difference": {"estimate": float(np.abs(compact-y).mean()-np.abs(comparator-y).mean()),
                           "bootstrap_mean": float(differences.mean()),
                           "ci95": np.percentile(differences, [2.5, 97.5]).tolist()}}
    return result


def analyse_data(states, loss_states, *, n_resamples=BOOT_N, seed=SEED):
    one_b = [t for t, c in states.items() if c["1.0"]["role"] == "dev" and c["1.0"]["size"] == "1B"]
    if len(one_b) != 3:
        raise ValueError("Need all three 1B development dense anchors for pilot")
    seven_only = {c: all(accuracy(states[t]["1.0"], c) < 0.05 for t in one_b) for c in CAPS}
    fits, cells, mae = {}, {}, {}
    for cap in CAPS:
        rows = {g: accuracy_rows(states, role="dev", cap=cap, seven_b_only=seven_only[cap], grid=g)
                for g in ("reduced", "full")}
        tags = sorted({r["tag"] for r in rows["reduced"]})
        loss_route = fit_loss_route(states, loss_states, tags, cap)
        fits[cap] = {g: fit_grid(rows[g], cap, loss_route) for g in rows}
        cells[cap] = []
        for row in accuracy_rows(states, role="test", cap=cap, seven_b_only=seven_only[cap]):
            predictions = {g: predict_six(fits[cap][g], row, loss_states[row["tag"]]) for g in fits[cap]}
            if not all(np.isfinite(v) for pred in predictions.values() for v in pred.values()):
                raise ValueError("Non-finite test prediction")
            cells[cap].append({"tag": row["tag"], "d": row["d"], "observed_delta_accuracy": row["y"],
                               "predictions": predictions})
        mae[cap] = {g: {m: float(np.mean([abs(r["predictions"][g][m]-r["observed_delta_accuracy"])
                                         for r in cells[cap]])) for m in METHODS} for g in fits[cap]}
    bootstrap = item_bootstrap(states, cells, n_resamples=n_resamples, seed=seed)
    primary = {c: {"difference": bootstrap["capabilities"][c]["difference"]["estimate"],
                   "within_0.03": bootstrap["capabilities"][c]["difference"]["estimate"] <= 0.03}
               for c in ("math", "code")}
    return {"schema_version": SCHEMA, "pilot_seven_b_only": seven_only, "fits": fits, "test_cells": cells,
            "mae_accuracy": mae,
            "mae_accuracy_points": {c: {g: {m: 100*v for m, v in values.items()} for g, values in gs.items()}
                                    for c, gs in mae.items()},
            "bootstrap": bootstrap, "primary": {"threshold_accuracy": 0.03, "capabilities": primary,
            "reading_met": all(v["within_0.03"] for v in primary.values()),
            "definition": "compact-reduced test MAE minus per-density-full test MAE <= 0.03 on both math and code; CI reported separately"}}


def costs(states, loads):
    result = {}
    for grid, densities in (("reduced", REDUCED), ("full", FULL)):
        total = {"compressed_configurations": 0, "dense_anchors": 0, "gpu_seconds": 0.0,
                 "load_seconds": 0.0, "generated_tokens": 0, "agreement_generated_tokens": 0}
        for state, cells in states.items():
            if cells["1.0"]["role"] != "dev":
                continue
            # Every actual load (including pilot/restarts) is charged once to
            # each budget; density timing excludes these shared loads.
            load = sum(r["load_seconds"] for r in loads[state])
            total["load_seconds"] += load
            total["gpu_seconds"] += sum(r["load_seconds"] for r in loads[state] if r["device"].startswith("cuda"))
            total["compressed_configurations"] += len(densities)
            total["dense_anchors"] += 1
            for d in (1.0, *densities):
                rec = cells[str(d)]
                total["gpu_seconds"] += rec["gpu_seconds"]
                total["generated_tokens"] += rec["generated_tokens"]
                audit = rec.get("batch_agreement") or {}
                total["agreement_generated_tokens"] += audit.get("generated_tokens", 0)
        total["generated_tokens_including_agreement"] = total["generated_tokens"] + total["agreement_generated_tokens"]
        result[grid] = total
    result["policy"] = ("Six development states, all capabilities, dense anchors once; synchronized elapsed GPU allocation time "
                        "includes generation, scoring, pruning, dense agreement and every recorded load/preparation/reload. "
                        "Interrupted unsaved batches are not timed. Loss-route generation labels are reduced+dense, already charged; "
                        "stored A18 loss acquisition is reported separately and is not an additional generation label budget.")
    return result


def analyse_stage(out=OUT, loss_dir=None):
    plan, sha = load_plan(out)
    gate = pilot_gate(out, plan, sha)
    if not (out / "pilot.json").exists() or read(out / "pilot.json") != gate:
        raise ValueError("Missing or changed pre-pruning pilot record")
    loss_dir = Path(loss_dir) if loss_dir is not None else a18.OUT / "measurements"
    states, loss_states, loads, sources = {}, {}, {}, {}
    for state, spec in plan["states"].items():
        states[state] = {}
        for d in (1.0, *spec["densities"]):
            path = measurement_path(out, state, d)
            rec = read(path)
            validate_measurement(rec, plan, sha, state, d, complete=True)
            states[state][str(d)] = rec
            sources[str(path)] = digest(path)
        load_path = out / "measurements" / state / "loads.json"
        loads[state] = read(load_path)
        sources[str(load_path)] = digest(load_path)
        path = loss_dir / f"{state.replace('@', '__')}.json"
        loss = read(path)
        dense = states[state]["1.0"]
        if any(loss[k] != dense[k] for k in ("repo", "revision", "N0_block_matrices", "D0")):
            raise ValueError(f"A18 loss state mismatch: {state}")
        loss_states[state] = loss
        sources[str(path)] = digest(path)
    result = analyse_data(states, loss_states)
    result.update({"plan_sha256": sha, "input_sha256": sources, "costs": costs(states, loads)})
    # Stored loss-route inputs have an acquisition cost, even though no new
    # loss measurement is run by this experiment.
    result["costs"]["stored_a18_reduced_loss_seconds"] = sum(
        r["load_seconds"] + sum(r["measurements"][str(d)]["seconds"] for d in (1.0, *REDUCED))
        for t, r in loss_states.items() if plan["states"][t]["role"] == "dev")
    atomic_write(out / "analysis.json", result)
    print(json.dumps({"primary": result["primary"], "costs": result["costs"]}, indent=2))
    return result


def render_table(result, target):
    from analysis.paper_table_layout import house_style, table_layout
    names = {"compact": "Compact form", "per_density": "Per-density regression", "median": "Median density curve",
             "pruning_law": "Pruning law", "loss_route": "Loss route", "zero": "Zero change"}
    rows = []
    for grid in ("reduced", "full"):
        for method in METHODS:
            vals = [f"{result['mae_accuracy_points'][c][grid][method]:.2f}" for c in CAPS]
            rows.append(" & ".join([names[method], grid.capitalize(), *vals]) + r" \\")
        rows.append(r"\midrule")
    diff = [result["bootstrap"]["capabilities"][c]["difference"] for c in CAPS]
    rows.append(" & ".join([r"Compact $-$ regression", "Primary", *[f"{100*r['estimate']:.2f}" for r in diff]]) + r" \\")
    rows.append(" & ".join(["Paired item interval", r"95\%", *[f"[{100*r['ci95'][0]:.2f}, {100*r['ci95'][1]:.2f}]" for r in diff]]) + r" \\")
    rows.append(r"\midrule")
    for grid in ("reduced", "full"):
        cost = result["costs"][grid]
        detail = f"{cost['gpu_seconds']:.1f} GPU s; {cost['generated_tokens_including_agreement']:,} generated tokens"
        rows.append(f"Generation cost & {grid.capitalize()} & " + r"\multicolumn{3}{l}{" + detail + r"} \\")
    scope = ", ".join(f"{c}: {'7B only' if result['pilot_seven_b_only'][c] else '1B and 7B'}" for c in CAPS)
    reading = "met" if result["primary"]["reading_met"] else "not met"
    caption = ("E5: task-accuracy measurement efficiency on OLMo-2. Test MAE is in accuracy points "
               "(100 times accuracy); test states enter no fit. The primary comparison is compact on the reduced "
               "grid against per-density regression on the full grid. The paired item bootstrap uses 2,000 "
               "resamples with states and fitted predictors fixed. The pre-specified margin is 0.03 accuracy "
               f"(3 points) on mathematics and code; the reading is {reading}. Pilot scope: {scope}. "
               "The loss route always uses reduced-grid loss fits and reduced-grid plus dense generation labels, "
               "so its full-grid row repeats the same predictor. Costs include all six development states, "
               "dense anchors, loading, pruning and the eight-item dense batching check. "
               "Existing A18 loss acquisition is separate from these generation costs.")
    text = ("% Generated by analysis/r1_accuracy_efficiency.py; do not edit.\n"
            "\\begin{table}[tb]\n\\centering\n\\small\n"
            + "\\caption{\\rev{" + caption + "}}\n\\label{tab:r1-accuracy-efficiency}\n"
            + "\\begin{tabular}{llrrr}\n\\toprule\nForm & Grid & Math & Code & QA \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(house_style(table_layout(text)))
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group(required=True)
    for name in ("plan", "measure", "analyse", "table"):
        stage.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--state", choices=list(STATES))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dense-only", action="store_true", help="Pilot pass before any compressed run")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.plan:
        plan_stage()
    elif args.measure:
        if args.state is None:
            parser.error("--measure requires --state")
        os.environ.setdefault("HF_HUB_CACHE", str(a18.CACHE / "hub"))
        measure_state(args.state, device=args.device, batch_size=args.batch_size, dense_only=args.dense_only)
    elif args.analyse:
        analyse_stage()
    else:
        _, sha = load_plan()
        result = read(OUT / "analysis.json")
        if result["plan_sha256"] != sha or any(digest(p) != h for p, h in result["input_sha256"].items()):
            raise ValueError("Analysis inputs changed; rerun --analyse before --table")
        paper_root = ROOT / "paper" if (ROOT / "paper/paper").exists() else ROOT
        print(render_table(result, paper_root / "paper/tables/r1_accuracy_efficiency.tex"))


if __name__ == "__main__":
    main()
