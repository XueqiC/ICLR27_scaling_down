#!/usr/bin/env python3
"""A3b: CPU-only minimax search of achieved optimizer-update rectangles.

Price tokenizer ledgers and replay every boundary; never load weights or train.
Processed-token flags select exact boundaries, with three neighbors each side.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys

# Set before importing torch/transformers through the trainer. No CUDA queries.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

# Both selection and preparation come from the actual, unmodified trainer.
from analysis.v12_distill import (  # noqa: E402
    DOMAINS, EFFECTIVE_BATCH_SIZE, MAX_LEN, WARMUP_RATIO,
    load_sft_records, make_run_name, tokenize_sft_example,
)

OUT = ROOT / "results/a3-corner-pools"
TRAINER = Path("analysis/v12_distill.py")
A1 = Path("results/a1-development-table/summary.json")
STUDENTS = ("gemma3-1b", "gemma3-4b")
MODELS = {s: f"google/gemma-3-{s.split('-')[-1]}-pt" for s in STUDENTS}
TEACHER = "gpt-5.6-luna"
RECIPE = "full"
USED_SEEDS = frozenset((0, 1, 2, 3, 11, 12, 13, 21, 22, 23, 31, 32, 33,
                        34, 35, 36, 41, 42, 51, 52, 91))
RESERVED_SEEDS = frozenset(range(60, 64))
RECORDED_TOTALS = {(66, 41): 16962, (66, 42): 17104,
                   (132, 51): 34503, (132, 52): 34639}
D_A = 16962
T1 = 50563
SCHEDULE_TOKENS = 678000
EPOCHS = 60
RESIDUAL_LIMIT = Fraction(1, 200)
MIN_CONTRAST = Fraction(9, 5)
LADDER_RADIUS = 3
MISMATCH_NAMES = ("budget_low", "budget_high", "reuse_low", "reuse_high")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def exact(value):
    value = Fraction(value)
    return f"{value.numerator}/{value.denominator}"


def public(pool):
    return {k: v for k, v in pool.items() if not k.startswith("_")}


def verify_trainer_hash(root=ROOT):
    expected = read_json(root / A1)["code_sha256"][str(TRAINER)]
    actual = sha256(root / TRAINER)
    require(actual == expected,
            f"Frozen trainer SHA256 mismatch: expected {expected}, observed {actual}")
    return actual


def trajectory_contract(root=ROOT):
    """Inspect executable condition, not the misleading argparse help text."""
    tree = ast.parse((root / TRAINER).read_text())
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_train")
    conditions = [n for n in ast.walk(train) if isinstance(n, ast.While)
                  and "milestones[next_milestone]" in ast.unparse(n.test)]
    require(len(conditions) == 1, "Cannot uniquely identify the trajectory milestone condition")
    node = conditions[0]
    condition = ast.unparse(node.test)
    if "accounting.processed >= milestones[next_milestone]" in condition:
        unit = "processed_prompt_plus_completion_tokens"
    elif "completion_tokens_seen >= milestones[next_milestone]" in condition:
        unit = "supervised_completion_tokens"
    else:
        raise ValueError(f"Unknown trajectory accounting: {condition}")
    return {"trajectory_tokens_unit": unit, "condition": condition,
            "source": f"{TRAINER}:{node.lineno}",
            "schedule_tokens_unit": "processed_prompt_plus_completion_tokens"}


def load_tokenizer(student):
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoTokenizer

    model = MODELS[student]
    config = try_to_load_from_cache(model, "tokenizer_config.json")
    require(isinstance(config, str), f"No locally cached tokenizer for {model}")
    snapshot = Path(config).parent
    # Use a local path: some transformers versions query model_info even with
    # local_files_only=True when passed a remote model ID. No model class loaded.
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    filenames = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                 "special_tokens_map.json", "added_tokens.json")
    identity = {"model": model, "snapshot": str(snapshot),
                "revision": snapshot.name, "class": type(tokenizer).__name__,
                "file_sha256": {n: sha256(snapshot / n) for n in filenames
                                if (snapshot / n).is_file()}}
    return tokenizer, identity


class PoolCounter:
    def __init__(self, tokenizer, trace_base=ROOT / "results/traces-pilot"):
        self.tokenizer = tokenizer
        self.trace_base = trace_base
        self.cache = {}

    def price(self, n_per_domain, data_seed):
        # V12 uses data_seed for both the initial shuffle and the epoch shuffles.
        records, counts = load_sft_records(
            teacher=TEACHER, domains=DOMAINS, n_per_domain=n_per_domain,
            recipe=RECIPE, trace_base=self.trace_base,
            seed=data_seed if data_seed is not None else 0, data_seed=data_seed)
        ledger = []
        deleted = Counter()
        for record in records:
            key = (record["prompt"], record["completion"])
            if key not in self.cache:
                example = tokenize_sft_example(self.tokenizer, *key, max_len=MAX_LEN)
                require(example["input_ids"].device.type == "cpu", "Counter left CPU")
                example_id = hashlib.sha256(json.dumps(list(key), ensure_ascii=False).encode()).hexdigest()
                self.cache[key] = {
                    "example_id": example_id,
                    "completion_tokens": int(example["n_completion_tokens"]),
                    "processed_tokens": int(example["input_ids"].numel()),
                    "tokenized_sha256": hashlib.sha256(json.dumps({
                        k: example[k].tolist() for k in ("input_ids", "labels")
                    }, sort_keys=True).encode()).hexdigest(),
                }
            item = self.cache[key]
            if not item["completion_tokens"]:
                deleted[record["domain"]] += 1
                continue
            ledger.append({**item, "domain": record["domain"]})
        require(ledger, "Pool has no supervised completion tokens")
        require(len({x["example_id"] for x in ledger}) == len(ledger),
                "Duplicate source identities: distinct-pool and pass totals would differ")
        # Byte-for-byte expression used for eval_metadata.data_pool_sha256 in V12.
        pool_hash = hashlib.sha256(json.dumps(sorted(
            (x["example_id"], x["processed_tokens"]) for x in ledger)).encode()).hexdigest()
        total = sum(x["completion_tokens"] for x in ledger)
        domains = {}
        for domain in DOMAINS:
            tokens = sum(x["completion_tokens"] for x in ledger if x["domain"] == domain)
            domains[domain] = {
                **counts[domain], "tokenization_deleted_rows": deleted[domain],
                "prepared_examples": sum(x["domain"] == domain for x in ledger),
                "completion_tokens": tokens, "token_share": tokens / total,
                "token_share_exact": exact(Fraction(tokens, total)),
            }
        return {"n_per_domain": n_per_domain, "data_seed": data_seed,
                "D_U_pool": total, "unique_data_pool_tokens": sum(x["processed_tokens"] for x in ledger),
                "prepared_examples": len(ledger), "data_pool_sha256": pool_hash,
                "per_domain": domains, "_ledger": ledger}


def verify_recorded_pools(counter, student, summary=None):
    summary = read_json(ROOT / A1) if summary is None else summary
    verified = []
    for (n, seed), expected in RECORDED_TOTALS.items():
        pool = counter.price(n, seed)
        require(pool["D_U_pool"] == expected,
                f"Counter mismatch for {student}, n={n}, seed={seed}: "
                f"expected {expected}, observed {pool['D_U_pool']} supervised tokens; SEARCH ABORTED")
        runs = [r for r in summary["run_audit"] if r["core"] and r["U"] == n
                and r["run_id"].startswith(student + "/")
                and r["run_id"].endswith(f"_dseed{seed}")]
        require(len(runs) == 1, f"Missing/ambiguous recorded reference: {student}/{n}/{seed}")
        run = runs[0]
        require(run["D_U_pool"] == expected, "A1 reference total changed")
        require(run["pool_id"] == "sha256:" + pool["data_pool_sha256"],
                f"Recorded data_pool_sha256 mismatch: {run['run_id']}")
        verified.append({"student": student, "n_per_domain": n, "data_seed": seed,
                         "expected": expected, "observed": pool["D_U_pool"],
                         "data_pool_sha256": pool["data_pool_sha256"], "run_id": run["run_id"]})
    return verified


def recorded_inventory(root=ROOT, output=OUT):
    # Scan recorded JSON hash fields across results, including archived runs and
    # nested summaries. Exclude our generated plan so reruns cannot self-exclude.
    result = subprocess.run(["rg", "-l", '"(data_pool_sha256|pool_id|data_seed|data_sampling_seed|training_seed|pool_seed|seed)"\\s*:',
                             str(root / "results"), "-g", "*.json"],
                            text=True, capture_output=True)
    require(result.returncode in (0, 1), f"Recorded pool scan failed: {result.stderr}")
    files = {Path(p) for p in result.stdout.splitlines()}
    files.update((root / "results").rglob("train_log.json"))
    hashes, seeds, sources = set(), set(), {}

    def visit(node):
        if isinstance(node, dict):
            value = node.get("data_pool_sha256")
            if value is not None:
                require(isinstance(value, str) and len(value) == 64
                        and all(c in "0123456789abcdef" for c in value),
                        "Malformed recorded pool hash")
                hashes.add(value)
            pool_id = node.get("pool_id")
            if isinstance(pool_id, str) and pool_id.startswith("sha256:"):
                hashes.add(pool_id.removeprefix("sha256:"))
            for key in ("data_seed", "data_sampling_seed", "training_seed", "pool_seed", "seed"):
                if type(node.get(key)) is int:
                    seeds.add(node[key])
            for value in node.values():
                if isinstance(value, (list, dict)):
                    visit(value)
        elif isinstance(node, list):
            for value in node:
                if isinstance(value, (list, dict)):
                    visit(value)

    for path in sorted(files):
        if path.is_relative_to(output) or path.is_relative_to(root / "results/a3-corner-pools"):
            continue
        visit(read_json(path))
        sources[str(path.relative_to(root))] = sha256(path)
    # Include A1's canonical pool IDs as a separate recorded source.
    for run in read_json(root / A1)["run_audit"]:
        hashes.add(run["pool_id"].removeprefix("sha256:"))
    require(hashes, "No recorded pool hashes found")
    return {"hashes": sorted(hashes), "observed_seeds": sorted(seeds),
            "source_sha256": sources, "scope": "All JSON pool hashes/IDs and data/training/pool/generic seed fields under results, including archives and A1; excludes A3 outputs"}


def valid_seed(seed, observed=()):
    return 70 <= seed < 2**32 and seed not in USED_SEEDS | RESERVED_SEEDS | set(observed)


def add_drift(pool, reference):
    for domain in DOMAINS:
        entry = pool["per_domain"][domain]
        drift = 100 * (Fraction(entry["completion_tokens"], pool["D_U_pool"])
                       - Fraction(reference["per_domain"][domain]["completion_tokens"], reference["D_U_pool"]))
        entry["share_drift_pp"] = float(drift)
        entry["share_drift_pp_exact"] = exact(drift)
        entry["drift_caveat"] = abs(drift) > 2
    maximum = max(abs(Fraction(x["share_drift_pp_exact"])) for x in pool["per_domain"].values())
    pool["max_absolute_share_drift_pp"] = float(maximum)
    pool["max_absolute_share_drift_pp_exact"] = exact(maximum)
    return pool


def replay_accounting(pool):
    """Integer ledger only: V12 epoch shuffles/batches, no model/optimizer calls.

    This separately audits realizability of the algebraic targets. The pool
    order itself is returned by load_sft_records, never resampled here.
    """
    examples = pool["_ledger"]
    updates_per_epoch = math.ceil(len(examples) / EFFECTIVE_BATCH_SIZE)
    processed_pool = sum(x["processed_tokens"] for x in examples)
    schedule_updates = math.ceil(SCHEDULE_TOKENS / (processed_pool / updates_per_epoch))
    rows = []
    supervised = processed = 0
    for epoch in range(math.ceil(schedule_updates / updates_per_epoch)):
        order = list(range(len(examples)))
        random.Random(pool["data_seed"] + epoch).shuffle(order)
        for start in range(0, len(order), EFFECTIVE_BATCH_SIZE):
            batch = [examples[i] for i in order[start:start + EFFECTIVE_BATCH_SIZE]]
            supervised += sum(x["completion_tokens"] for x in batch)
            processed += sum(x["processed_tokens"] for x in batch)
            rows.append({"update": len(rows) + 1, "supervised_tokens": supervised,
                         "processed_tokens": processed, "epoch": epoch + 1})
            if len(rows) == schedule_updates:
                return rows, {"schedule_updates": schedule_updates,
                              "warmup_updates": int(WARMUP_RATIO * schedule_updates),
                              "updates_per_epoch": updates_per_epoch}
    raise ValueError("Accounting replay failed to reach schedule horizon")


def relative_gap(a, b):
    a, b = Fraction(a), Fraction(b)
    require(min(a, b) > 0, "Mismatch endpoints must be positive")
    return abs(a - b) / min(a, b)


def mismatch_fractions(t1_b, t2_b, t2_c, d_b, d_c):
    require(min(t1_b, t2_b, t2_c, d_b, d_c) > 0, "Rectangle coordinates must be positive")
    return dict(zip(MISMATCH_NAMES, (
        abs(Fraction(t1_b) - T1) / T1,
        relative_gap(t2_c, t2_b),
        relative_gap(Fraction(t1_b, d_b), Fraction(t2_c, d_c)),
        relative_gap(Fraction(T1, D_A), Fraction(t2_b, d_b)),
    )))


def rectangle(d_b, d_c, t1_b, t2_b, t2_c):
    """Use achieved integers. Ideal pool sizes are outputs of these boundaries."""
    mismatches = mismatch_fractions(t1_b, t2_b, t2_c, d_b, d_c)
    budget_contrast = Fraction(min(t2_b, t2_c), max(T1, t1_b))
    reuse_contrast = Fraction(T1, D_A) / Fraction(t1_b, d_b)
    corners = []
    for number, pool, t, d in ((1, "B", t1_b, d_b), (2, "A", T1, D_A),
                               (3, "C", t2_c, d_c), (4, "B", t2_b, d_b)):
        corners.append({"corner": number, "pool": pool, "D_U_pool": d,
                        "T_supervised": t, "E": t / d, "E_exact": exact(Fraction(t, d)),
                        "already_measured": number == 2})
    return {"D_A": D_A, "D_B": d_b, "D_C": d_c, "T1_A": T1,
            "T1_B": t1_b, "T2_B": t2_b, "T2_C": t2_c,
            "boundary_derived_D_B_exact": exact(Fraction(t2_b * D_A, T1)),
            "boundary_derived_D_C_exact": exact(Fraction(d_b * t2_c, t1_b)),
            "corners": corners,
            "mismatches": {k: {"fraction_exact": exact(v), "percent": float(100 * v),
                                "limit_percent": 0.5, "passed": v <= RESIDUAL_LIMIT}
                           for k, v in mismatches.items()},
            "worst_mismatch_exact": exact(max(mismatches.values())),
            "worst_mismatch_percent": float(100 * max(mismatches.values())),
            "budget_contrast_exact": exact(budget_contrast),
            "reuse_contrast_exact": exact(reuse_contrast),
            "contrast_passed": min(budget_contrast, reuse_contrast) >= MIN_CONTRAST,
            "tolerance_passed": all(v <= RESIDUAL_LIMIT for v in mismatches.values())}


def design_failures(design):
    failures = [f"{name}: {m['percent']:.9f}% exceeds the fixed 0.5% limit."
                for name in MISMATCH_NAMES
                if not (m := design["mismatches"][name])["passed"]]
    if not design["contrast_passed"]:
        failures.append("Budget/reuse contrast is below 1.8.")
    if failures:
        failures.insert(0, "No combination in the enumerated candidate space meets all four 0.5% limits; "
                        "the best achievable design is reported. Tolerance has not been widened.")
    return failures


def price_grids(counter, reference, n_values, seeds, inventory, label):
    """Replay every candidate's full horizon before hard domain/hash exclusions."""
    pools = []
    stats = dict(configurations_priced=0, update_boundaries_priced=0,
                 hash_collisions_excluded=0, domain_drift_excluded=0)
    for index, seed in enumerate(seeds):
        require(valid_seed(seed, inventory["observed_seeds"]), f"Used/reserved/invalid search seed: {seed}")
        for n in n_values:
            pool = add_drift(counter.price(n, seed), reference)
            rows, schedule = replay_accounting(pool)
            stats["configurations_priced"] += 1
            stats["update_boundaries_priced"] += len(rows)
            if pool["data_pool_sha256"] in inventory["hashes"]:
                stats["hash_collisions_excluded"] += 1
                continue
            if Fraction(pool["max_absolute_share_drift_pp_exact"]) > 2:
                stats["domain_drift_excluded"] += 1
                continue
            # Keep grids and summaries, not thousands of duplicate token ledgers.
            p = public(pool)
            p.update(_rows=rows, _schedule=schedule, coincides_with_recorded_pool=False)
            pools.append(p)
        if (index + 1) % 25 == 0:
            print(f"Pool {label}: {stats['configurations_priced']} pools / "
                  f"{stats['update_boundaries_priced']} update boundaries priced; "
                  f"{len(pools)} pass freshness and domain drift.", flush=True)
    require(pools, f"No fresh pool {label} candidates within 2 pp domain drift")
    stats["eligible_pools"] = len(pools)
    return pools, stats


def b_boundary_candidates(pool, ceiling=None):
    """Exact lower bounds for all eligible B pairs, screened with CPU arrays.

    The lower bound is max(budget_low, reuse_high). The other two residuals
    are nonnegative, so pairs above a completed design's worst error cannot win.
    Floating point is used only for conservative screening; ranking is rational.
    """
    rows = pool["_rows"][LADDER_RADIUS:-LADDER_RADIUS]
    ts = np.array([r["supervised_tokens"] for r in rows], dtype=np.int64)
    if not len(ts):
        return []
    low, high = ts[:, None], ts[None, :]
    d_b = pool["D_U_pool"]
    budget = np.abs(low - T1) / T1
    lhs, rhs = high * D_A, T1 * d_b
    reuse = np.abs(lhs - rhs) / np.minimum(lhs, rhs)
    lower = np.maximum(budget, reuse)
    valid = ((5 * high >= 9 * np.maximum(low, T1))
             & (5 * T1 * d_b >= 9 * D_A * low))
    if ceiling is None:
        bound = np.min(np.where(valid, lower, np.inf))
    else:
        bound = float(ceiling)
    if not np.isfinite(bound):
        return []
    candidates = []
    for i, j in np.argwhere(valid & (lower <= bound + 1e-12)):
        a, b = rows[int(i)], rows[int(j)]
        lb = max(abs(Fraction(a["supervised_tokens"] - T1, T1)),
                 relative_gap(Fraction(T1, D_A), Fraction(b["supervised_tokens"], d_b)))
        if ceiling is None or lb <= ceiling:
            candidates.append({"pool": pool, "low": a, "high": b, "lower_bound": lb})
    return sorted(candidates, key=lambda c: (c["lower_bound"], c["low"]["update"], c["high"]["update"]))


def candidate_key(candidate):
    b, c, d = candidate["B"], candidate["C"], candidate["design"]
    return (Fraction(d["worst_mismatch_exact"]),
            max(Fraction(p["max_absolute_share_drift_pp_exact"]) for p in (b, c)),
            b["n_per_domain"], b["data_seed"], c["n_per_domain"], c["data_seed"],
            candidate["B_low"]["update"], candidate["B_high"]["update"], candidate["C_high"]["update"])


class ConditionalCSearch:
    def __init__(self, pools):
        self.pools = pools
        self.rows = [p["_rows"][LADDER_RADIUS:-LADDER_RADIUS] for p in pools]
        require(all(self.rows), "A pool cannot supply a complete seven-boundary ladder")
        lengths = [len(rs) for rs in self.rows]
        self.starts = np.cumsum([0] + lengths[:-1])
        self.pool_indices = np.repeat(np.arange(len(pools)), lengths)
        self.tokens = np.array([r["supervised_tokens"] for rs in self.rows for r in rs], dtype=np.int64)
        self.totals = np.repeat([p["D_U_pool"] for p in pools], lengths)
        self.seeds = np.repeat([p["data_seed"] for p in pools], lengths)
        self.evaluations = 0

    def search(self, b, best_k=5):
        self.evaluations += 1
        pool = b["pool"]
        t1, t2, db = b["low"]["supervised_tokens"], b["high"]["supervised_tokens"], pool["D_U_pool"]
        high = np.abs(self.tokens - t2) / np.minimum(self.tokens, t2)
        lhs, rhs = t1 * self.totals, self.tokens * db
        reuse = np.abs(lhs - rhs) / np.minimum(lhs, rhs)
        scores = np.maximum(np.maximum(high, reuse), float(b["lower_bound"]))
        valid = (5 * self.tokens >= 9 * max(T1, t1)) & (self.seeds != pool["data_seed"])
        for i, other in enumerate(self.pools):
            if other["data_pool_sha256"] == pool["data_pool_sha256"]:
                valid[self.pool_indices == i] = False
        scores = np.where(valid, scores, np.inf)
        # Return the best distinct pools. Include numerical ties for exact rank.
        minima = np.minimum.reduceat(scores, self.starts)
        finite = minima[np.isfinite(minima)]
        if not len(finite):
            return []
        threshold = np.partition(finite, min(best_k, len(finite)) - 1)[min(best_k, len(finite)) - 1]
        ranked = []
        for index in np.flatnonzero(minima <= threshold + 1e-12):
            start = self.starts[index]
            local = scores[start:start + len(self.rows[index])]
            best = None
            for j in np.flatnonzero(local <= minima[index] + 1e-12):
                cp = self.rows[index][int(j)]
                other = self.pools[index]
                candidate = {"B": pool, "C": other, "B_low": b["low"], "B_high": b["high"],
                             "C_high": cp, "design": rectangle(db, other["D_U_pool"], t1, t2, cp["supervised_tokens"])}
                require(candidate["design"]["contrast_passed"], "Screen admitted an invalid contrast")
                if best is None or candidate_key(candidate) < candidate_key(best):
                    best = candidate
            ranked.append(best)
        return sorted(ranked, key=candidate_key)[:best_k]


def search_boundaries(b_pools, c_pools, best_k=5):
    """B first, then C conditional on achieved B; certify global minimax rank.

    Seed an incumbent with the best B-only lower bounds, then visit every B
    boundary pair capable of beating it. C is always conditioned on actual B
    coordinates, never on a doubled pool size or an idealized budget.
    """
    require(best_k > 0, "best_k must be positive")
    initial = [cs[0] for p in b_pools if (cs := b_boundary_candidates(p))]
    initial.sort(key=lambda b: (b["lower_bound"], Fraction(b["pool"]["max_absolute_share_drift_pp_exact"]),
                                b["pool"]["n_per_domain"], b["pool"]["data_seed"]))
    require(initial, "No B boundary pair meets contrast and ladder constraints")
    conditional = ConditionalCSearch(c_pools)
    by_b_pool, seen = {}, set()

    def complete(b):
        identity = (b["pool"]["n_per_domain"], b["pool"]["data_seed"])
        key = (*identity, b["low"]["update"], b["high"]["update"])
        if key in seen:
            return
        seen.add(key)
        matches = conditional.search(b, 1)
        if matches and (identity not in by_b_pool or candidate_key(matches[0]) < candidate_key(by_b_pool[identity])):
            by_b_pool[identity] = matches[0]

    for b in initial:
        complete(b)
        if len(by_b_pool) >= best_k:
            break
    require(by_b_pool, "No C boundary meets contrast, ladder, and distinct-seed constraints")
    ceiling = max(Fraction(c["design"]["worst_mismatch_exact"]) for c in by_b_pool.values())
    examined = 0
    for pool in b_pools:
        for b in b_boundary_candidates(pool, ceiling):
            complete(b)
            examined += 1
        ranked = sorted(by_b_pool.values(), key=candidate_key)
        if len(ranked) >= best_k:
            ceiling = Fraction(ranked[best_k - 1]["design"]["worst_mismatch_exact"])
    ranked = sorted(by_b_pool.values(), key=candidate_key)[:best_k]
    best = ranked[0]
    final_b = {"pool": best["B"], "low": best["B_low"], "high": best["B_high"],
               "lower_bound": max(Fraction(best["design"]["mismatches"][k]["fraction_exact"])
                                  for k in ("budget_low", "reuse_high"))}
    c_ranked = conditional.search(final_b, best_k)
    return best, ranked, c_ranked, {"B_pairs_completed": len(seen), "B_pairs_below_incumbent_bound": examined,
                                   "conditional_C_grid_searches": conditional.evaluations,
                                   "global_minimax_certified": True,
                                   "pruning_rule": "Only discard B pairs whose exact two-residual lower bound exceeds the current kth completed worst mismatch; include all ties."}


def public_candidate(candidate):
    return {"B": public(candidate["B"]), "C": public(candidate["C"]),
            "boundaries": {k: candidate[k] for k in ("B_low", "B_high", "C_high")},
            "design": candidate["design"],
            "max_domain_drift_pp": float(candidate_key(candidate)[1])}


def trajectory_plan(student, pool_name, pool, targets, corner_ids, contract, launchable, root=ROOT):
    rows, schedule = replay_accounting(pool)
    requested = {}
    for target, corner in zip(targets, corner_ids):
        update = target["update"]
        require(rows[update - 1] == target, "Chosen boundary differs from ledger replay")
        require(update > LADDER_RADIUS and update + LADDER_RADIUS <= len(rows), "Incomplete local ladder")
        for offset in range(-LADDER_RADIUS, LADDER_RADIUS + 1):
            row = rows[update + offset - 1]
            cp = requested.setdefault(row["update"], {**row, "targets": [], "corners": []})
            cp["targets"].append({"corner": corner, "offset_updates": offset})
            if offset == 0:
                cp["corners"].append(corner)
    checkpoints = [requested[u] for u in sorted(requested)]
    milestones = [r["processed_tokens"] for r in checkpoints]
    require(contract["trajectory_tokens_unit"] == "processed_prompt_plus_completion_tokens", "Frozen flag contract changed")
    suffix = f"a3b_corners_{'1_4_poolB' if pool_name == 'B' else '3_poolC'}"
    argv = ["python3", str(TRAINER), "--student", student, "--teacher", TEACHER,
            "--recipe", RECIPE, "--domains", ",".join(DOMAINS),
            "--n-per-domain", str(pool["n_per_domain"]), "--data-seed", str(pool["data_seed"]),
            "--seed", "0", "--trajectory-tokens", *map(str, milestones),
            "--stop-after-trajectory", "--schedule-tokens", str(SCHEDULE_TOKENS),
            "--epochs", str(EPOCHS), "--lr", "1e-4", "--training-mode", "lora",
            "--save-trajectory", "--output-suffix", suffix, "--device", "cuda"]
    run_name = make_run_name(TEACHER, RECIPE, pool["n_per_domain"], suffix, 0,
                             training_mode="lora", data_seed=pool["data_seed"])
    output = root / "results/v12-distill" / student / run_name
    require(not output.exists(), f"Proposed trajectory output already exists: {output}")
    return {"pool": pool_name, "corner_ids": corner_ids, "launchable": launchable,
            "command_line": shlex.join(argv), "argv": argv,
            "trajectory_tokens": milestones, "trajectory_tokens_unit": contract["trajectory_tokens_unit"],
            "target_boundaries": targets, "predicted_checkpoints": checkpoints,
            "predicted_actual_stop_supervised_tokens": checkpoints[-1]["supervised_tokens"],
            "schedule_tokens": SCHEDULE_TOKENS, **schedule, "epochs_flag": EPOCHS,
            "effective_epoch_limit": math.ceil(schedule["schedule_updates"] / schedule["updates_per_epoch"]),
            "output_directory": str(output.relative_to(root))}

def residual_contamination(design, root=ROOT):
    """Finite differences from adjacent recorded checkpoints, never fitted loss.

    Reuse gaps are expressed as the supervised displacement needed to match E
    at that corner's fixed D. These are directional sensitivity estimates; four
    edge residuals are dependent and must not be summed as independent bias.
    """
    source = Path("results/a1-development-table/development_table.csv")
    summary = read_json(root / A1)
    require(sha256(root / source) == summary["artifact_sha256"][source.name], "A1 table checksum mismatch")
    groups = defaultdict(list)
    with (root / source).open() as handle:
        for row in csv.DictReader(handle):
            if row["student_id"] in STUDENTS:
                groups[(row["student_id"], row["capability"], row["distribution"], row["run_id"])].append(row)
    t1, t2, tc, db, dc = (design[k] for k in ("T1_B", "T2_B", "T2_C", "D_B", "D_C"))
    displacements = {
        "budget_low": (Fraction(t1 - T1), Fraction(T1), 1),
        "budget_high": (Fraction(tc - t2), Fraction(t2), -1),
        "reuse_low": (Fraction(tc) - Fraction(dc * t1, db), Fraction(tc), -1),
        "reuse_high": (Fraction(t2) - Fraction(db * T1, D_A), Fraction(t2), 1),
    }
    channels = defaultdict(dict)
    for name, (delta, target, sign) in displacements.items():
        for (student, cap, dist, run), rows in sorted(groups.items()):
            rows.sort(key=lambda r: int(r["T_actual"]))
            for first, second in zip(rows, rows[1:]):
                a, b = int(first["T_actual"]), int(second["T_actual"])
                if not (0 < a <= target <= b and a < b):
                    continue
                slope = (float(second["loss"]) - float(first["loss"])) / (b - a)
                key = (student, cap, dist)
                entry = channels[key].setdefault(name, {"signed_equivalent_supervised_displacement_exact": exact(delta),
                                                        "second_difference_sign": sign, "adjacent_checkpoint_slopes": []})
                entry["adjacent_checkpoint_slopes"].append({
                    "run_id": run, "first_checkpoint": first["checkpoint_id"], "second_checkpoint": second["checkpoint_id"],
                    "T_first": a, "T_second": b, "loss_first": float(first["loss"]), "loss_second": float(second["loss"]),
                    "first_source": first["source_path"], "second_source": second["source_path"],
                    "native_token_nats_per_supervised_token": slope,
                    "signed_second_difference_sensitivity_nats": sign * float(delta) * slope})
    output = []
    for (student, cap, dist), residuals in sorted(channels.items()):
        require(set(residuals) == set(MISMATCH_NAMES), f"Missing measured slope for {student}/{dist}")
        for entry in residuals.values():
            estimates = [r["signed_second_difference_sensitivity_nats"] for r in entry["adjacent_checkpoint_slopes"]]
            entry["signed_sensitivity_range_nats"] = [min(estimates), max(estimates)]
            entry["max_absolute_sensitivity_nats"] = max(map(abs, estimates))
        output.append({"student": student, "capability": cap, "distribution": dist, "residuals": residuals})
    require(output and {r["student"] for r in output} == set(STUDENTS), "No recorded local budget slopes")
    return {"source": str(source), "source_sha256": sha256(root / source),
            "second_difference": "L4 - L3 - L2 + L1",
            "method": "For each existing same-run, same-distribution adjacent positive checkpoint pair bracketing the target T, slope=(loss_next-loss_prev)/(T_next-T_prev). Report the envelope across runs; no response enters pool ranking.",
            "equivalent_displacements": {k: exact(v[0]) for k, v in displacements.items()},
            "limitation": "Slopes follow fixed-pool trajectories, where T and E both change. These are local budget-equivalent sensitivities, not identified fixed-T reuse effects or rigorous bias bounds. The four edge corrections are dependent: do not sum them. Dense new ladders will measure local slopes on the chosen pools; reuse-only contamination still needs a surface assumption or matched-budget evidence.",
            "channels": output}


def build_plan(n_b=range(124, 141), n_c=range(248, 281), seed_max=269, best_k=5, output=OUT):
    trainer_hash = verify_trainer_hash()
    summary = read_json(ROOT / A1)
    contract = trajectory_contract()
    inventory = recorded_inventory(output=output)
    counters, tokenizers, checks, replay_checks = {}, {}, [], []
    for student in STUDENTS:
        tokenizer, identity = load_tokenizer(student)
        counter = PoolCounter(tokenizer)
        checks.extend(verify_recorded_pools(counter, student, summary))
        # Verify every logged update, not just the target or the final total.
        for check in checks:
            if check["student"] != student:
                continue
            rows, schedule = replay_accounting(counter.price(check["n_per_domain"], check["data_seed"]))
            path = Path("results/v12-distill") / check["run_id"] / "train_log.json"
            logged = read_json(ROOT / path)
            require(len(rows) == logged["total_updates_planned"] == schedule["schedule_updates"], "Recorded schedule changed")
            for predicted, actual in zip(rows, logged["loss_curve"]):
                require((predicted["update"], predicted["supervised_tokens"], predicted["processed_tokens"])
                        == (actual["step"], actual["completion_tokens_seen"], actual["tokens_seen"]), "Recorded update ledger mismatch")
            replay_checks.append({"source": str(path), "sha256": sha256(ROOT / path),
                                  "verified_updates": len(logged["loss_curve"])})
        counters[student], tokenizers[student] = counter, identity
    print("All four recorded totals, hashes and update ledgers verified for BOTH students; search gate passed.", flush=True)
    counter = counters[STUDENTS[0]]
    reference = add_drift(counter.price(66, 41), counter.price(66, 41))
    seeds = [s for s in range(70, seed_max + 1) if valid_seed(s, inventory["observed_seeds"])]
    require(seeds, "No permitted seeds in the requested search interval")
    b_pools, b_stats = price_grids(counter, reference, n_b, seeds, inventory, "B")
    c_pools, c_stats = price_grids(counter, reference, n_c, seeds, inventory, "C")
    chosen, b_best, c_best, certification = search_boundaries(b_pools, c_pools, best_k)
    design = chosen["design"]
    failures = design_failures(design)
    b, c = [add_drift(counter.price(chosen[name]["n_per_domain"], chosen[name]["data_seed"]), reference) for name in ("B", "C")]
    require(b["data_pool_sha256"] != c["data_pool_sha256"] and b["data_seed"] != c["data_seed"], "B and C must have distinct hashes and seeds")
    for pool in (b, c):
        require(pool["data_pool_sha256"] not in inventory["hashes"], "Chosen pool coincides with a recorded pool")
        require(valid_seed(pool["data_seed"], inventory["observed_seeds"]), "Chosen pool seed is forbidden")
        require(Fraction(pool["max_absolute_share_drift_pp_exact"]) <= 2, "Chosen pool exceeds 2 pp domain drift")
    students = {}
    for student in STUDENTS:
        for pool in (reference, b, c):
            other = counters[student].price(pool["n_per_domain"], pool["data_seed"])
            require(other["_ledger"] == pool["_ledger"], f"Student tokenization/order differs: {student}")
        path = Path("results/v12-distill") / student / "gpt-5.6-luna_full_66_matrix2_lora_dseed41/trajectory/update-00000038/eval.json"
        recorded = read_json(ROOT / path)
        require(recorded["completion_tokens_seen"] == T1
                and recorded["data_pool_sha256"] == reference["data_pool_sha256"], "Existing corner 2 changed")
        require(recorded["recipe"] == RECIPE and recorded["schedule_tokens"] == SCHEDULE_TOKENS
                and recorded["learning_rate"] == 1e-4 and recorded["training_mode"] == "lora"
                and recorded["epochs"] == EPOCHS and recorded["seed"] == 0, "Recorded matrix protocol changed")
        replay, _ = replay_accounting(reference)
        require(replay[37]["supervised_tokens"] == T1
                and replay[37]["processed_tokens"] == recorded["processed_tokens"], "Corner 2 ledger replay mismatch")
        students[student] = {"already_measured_corner_2": {"source": str(path), "sha256": sha256(ROOT / path),
                             "checkpoint": "update-00000038", "retrain": False, "supervised_tokens": T1,
                             "processed_tokens": recorded["processed_tokens"],
                             "requested_token_milestones": recorded["requested_token_milestones"]},
                            "new_trajectories": [
                                trajectory_plan(student, "B", b, [chosen["B_low"], chosen["B_high"]], [1, 4], contract, not failures),
                                trajectory_plan(student, "C", c, [chosen["C_high"]], [3], contract, not failures)]}
    contamination = residual_contamination(design)
    require(verify_trainer_hash() == trainer_hash, "Trainer changed during search")
    require(inventory == recorded_inventory(output=output), "Recorded run inventory changed during search; rerun")
    return {"schema_version": "a3b-corner-pool-search-v2", "status": "failed" if failures else "passed",
            "pool_search_status": "completed", "launchable": not failures,
            "device": "cpu", "training_run": False, "model_weights_loaded": False,
            "failures": failures, "trainer_sha256": trainer_hash,
            "source_sha256": {str(A1): sha256(ROOT / A1),
                              "analysis/a3_corner_pool_search.py": sha256(Path(__file__)),
                              contamination["source"]: contamination["source_sha256"],
                              **{str(p.relative_to(ROOT)): sha256(p) for p in counter.trace_base.glob(f"{TEACHER}_*.jsonl")}},
            "tokenizers": tokenizers, "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "tokenizers", "numpy")},
            "verification": {"passed_before_search": True, "recorded_pools": checks, "recorded_update_replays": replay_checks,
                             "chosen_token_ids_labels_and_order_equal_across_students": True,
                             "both_chosen_trainer_hashes_absent_from_every_recorded_hash": True},
            "flag_contract": contract,
            "protocol": {"teacher": TEACHER, "recipe": RECIPE, "training_mode": "lora", "learning_rate": 1e-4,
                         "training_seed": 0, "schedule_tokens": SCHEDULE_TOKENS, "epochs": EPOCHS,
                         "domains": list(DOMAINS), "max_len": MAX_LEN, "effective_batch_size": EFFECTIVE_BATCH_SIZE,
                         "warmup_ratio": WARMUP_RATIO, "new_trajectory_count": 4, "round_cap": 4,
                         "ladder_radius_updates": LADDER_RADIUS},
            "search": {"n_B": list(n_b), "n_C": list(n_c), "seed_min": 70, "seed_max": seed_max,
                       "seeds_enumerated": seeds, "used_seeds": sorted(USED_SEEDS), "reserved_seeds": sorted(RESERVED_SEEDS),
                       "ranking": "exact worst of all four achieved mismatches, then maximum B/C domain-share drift, then n/seed/update for determinism",
                       "B": {**b_stats, "best_candidates": [public_candidate(x) for x in b_best]},
                       "C_conditional_on_achieved_B": {**c_stats, "best_candidates": [public_candidate(x) for x in c_best]},
                       **certification},
            "recorded_pool_inventory": inventory, "pools": {"A": public(reference), "B": public(b), "C": public(c)},
            "design": design, "students": students, "residual_contamination": contamination}


def markdown(plan):
    lines = ["# A3b corrected corner pool plan", "",
             f"Status: **{plan['status'].upper()}{' — NOT LAUNCHABLE' if not plan.get('launchable') else ' — within all four 0.5% limits'}**.", "",
             "CPU only; local tokenizer files only; no training, GPU work, or model weights loaded.", ""]
    lines += [f"- {failure}" for failure in plan.get("failures", [])]
    if "design" not in plan:
        return "\n".join(lines) + "\n"
    d, search = plan["design"], plan["search"]
    lines += ["## Corrected objective", "",
              "Corner 2 is the existing A measurement: D_A=16962, T1_A=50563, E_2=50563/16962. "
              "The replayed boundaries determine ideal D_B=T2_B×16962/50563 and D_C=D_B×T2_C/T1_B. "
              "Actual D values are token-counted candidate pools; no doubled-pool target is imposed.", "",
              "| mismatch | definition | achieved % | limit % |",
              "|---|---|---:|---:|"]
    definitions = ("abs(T1_B−50563)/50563", "abs(T2_C−T2_B)/min(T2_C,T2_B)",
                   "abs(E_1−E_3)/min(E_1,E_3)", "abs(E_2−E_4)/min(E_2,E_4)")
    for name, formula in zip(MISMATCH_NAMES, definitions):
        lines.append(f"| {name} | {formula} | {d['mismatches'][name]['percent']:.9f} | 0.5 |")
    lines += ["", f"Worst mismatch: **{d['worst_mismatch_percent']:.9f}%**. "
              f"Conservative budget contrast min(T2_B,T2_C)/max(50563,T1_B)={float(Fraction(d['budget_contrast_exact'])):.9f}; "
              f"reuse contrast E_2/E_1={float(Fraction(d['reuse_contrast_exact'])):.9f}; both must be ≥1.8.", "",
              "## Boundary-grid search", "",
              "All four recorded pool totals (16962, 17104, 34503, 34639), trainer pool hashes, and every logged update boundary "
              "matched independently for both students before search.", "",
              f"Candidate space: B n/domain {min(search['n_B'])}–{max(search['n_B'])}; C {min(search['n_C'])}–{max(search['n_C'])}; "
              f"seeds 70–{search['seed_max']} excluding every observed/previously used seed and reserved 60–63. B and C also use distinct seeds.",
              "Every pool's full schedule grid is replayed. Targets must have three completed boundaries on each side; "
              "domain-share drift against A is a hard ≤2 percentage point constraint on every domain.",
              "Search B first, then C conditional on achieved B. Complete all B pairs whose two known residuals can improve the incumbent; "
              "this certifies the minimum of all four residuals over the enumerated grids. Rank by worst mismatch, then maximum B/C domain drift. "
              "The B table profiles each B against its best conditional C; the C table freezes the chosen B. Rows represent distinct pools.", ""]
    for label, result, pool_name in (("B (with its best conditional C)", search["B"], "B"),
                                     ("C conditional on chosen B", search["C_conditional_on_achieved_B"], "C")):
        lines += [f"### Best pools: {label}", "",
                  f"{result['configurations_priced']} pools; {result['update_boundaries_priced']} boundaries; "
                  f"{result['domain_drift_excluded']} pools excluded by drift; {result['hash_collisions_excluded']} by recorded hash.", "",
                  "| rank | n/seed | D | T1_B / T2_B / T2_C | budget low % | budget high % | reuse low % | reuse high % | worst % | max drift pp | paired n/seed |",
                  "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|"]
        for i, candidate in enumerate(result["best_candidates"], 1):
            p, other, design = candidate[pool_name], candidate["C" if pool_name == "B" else "B"], candidate["design"]
            errors = " | ".join(f"{design['mismatches'][k]['percent']:.9f}" for k in MISMATCH_NAMES)
            lines.append(f"| {i} | {p['n_per_domain']}/{p['data_seed']} | {p['D_U_pool']} | "
                         f"{design['T1_B']} / {design['T2_B']} / {design['T2_C']} | {errors} | "
                         f"{design['worst_mismatch_percent']:.9f} | {candidate['max_domain_drift_pp']:.6f} | {other['n_per_domain']}/{other['data_seed']} |")
        lines.append("")
    lines += [f"Completed {search['B_pairs_completed']} B boundary pairs and {search['conditional_C_grid_searches']} conditional C searches; "
              "all exclusions above the incumbent use a provable lower bound, not the 0.5% acceptance limit.", "",
              "## Chosen pools and achieved corners", ""]
    inv = plan["recorded_pool_inventory"]
    lines += [f"Both chosen trainer data_pool_sha256 values are distinct and **absent from every one of {len(inv['hashes'])} recorded hashes** "
              f"({len(inv['source_sha256'])} JSON sources, including archives). Inventory was checked again after search.", ""]
    for name, p in plan["pools"].items():
        lines.append(f"Pool {name}: n/domain={p['n_per_domain']}, seed={p['data_seed']}, D={p['D_U_pool']}; "
                     f"data_pool_sha256 `{p['data_pool_sha256']}`.")
    lines += ["", f"Boundary-derived ideal D_B={d['boundary_derived_D_B_exact']} ({float(Fraction(d['boundary_derived_D_B_exact'])):.6f}); "
              f"ideal D_C conditional on actual B={d['boundary_derived_D_C_exact']} ({float(Fraction(d['boundary_derived_D_C_exact'])):.6f}).", "",
              "| corner | pool | achieved supervised T | exact E=T/D | status |", "|---:|---|---:|---|---|"]
    for c in d["corners"]:
        lines.append(f"| {c['corner']} | {c['pool']} | {c['T_supervised']} | {c['E_exact']} | "
                     f"{'existing; do not retrain' if c['already_measured'] else 'predicted completed update'} |")
    lines += ["", "### Domain shares", "", "| pool | domain | selected / prepared | supervised tokens | share % | drift pp |",
              "|---|---|---:|---:|---:|---:|"]
    for name, p in plan["pools"].items():
        for domain in DOMAINS:
            entry = p["per_domain"][domain]
            lines.append(f"| {name} | {domain} | {entry['source_rows']} / {entry['prepared_examples']} | "
                         f"{entry['completion_tokens']} | {100 * entry['token_share']:.6f} | {entry['share_drift_pp']:+.6f} |")
    lines += ["", "## Residual sensitivity of the second difference", "",
              "Second difference: **L4−L3−L2+L1**, in native-token nats. Adjacent checkpoints in each existing run give "
              "s=(loss_next−loss_prev)/(T_next−T_prev), using pairs bracketing the relevant achieved budget. "
              "The table reports the largest absolute effect across these measured local slopes for each student/distribution.", "",
              "| residual | signed equivalent supervised displacement | contribution |", "|---|---:|---|"]
    contamination = plan["residual_contamination"]
    for name, sign in zip(MISMATCH_NAMES, ("+s×δT", "−s×δT", "−s×δT", "+s×δT")):
        lines.append(f"| {name} | {contamination['equivalent_displacements'][name]} | {sign} |")
    lines += ["", "Reuse-low displacement is T2_C−D_C×T1_B/D_B; reuse-high displacement is T2_B−D_B×50563/16962.", "",
              "| student | distribution | budget low nats | budget high nats | reuse low nats | reuse high nats |",
              "|---|---|---:|---:|---:|---:|"]
    for channel in contamination["channels"]:
        values = " | ".join(f"{channel['residuals'][k]['max_absolute_sensitivity_nats']:.8f}" for k in MISMATCH_NAMES)
        lines.append(f"| {channel['student']} | {channel['distribution']} | {values} |")
    lines += ["", contamination["limitation"], "",
              "All slope endpoints, losses, source paths, signed effects, and slope ranges are recorded in plan.json. "
              "Slopes are diagnostics only and never enter pool selection.", "",
              "## Processed thresholds and dense local ladders", "",
              f"Frozen condition at `{plan['flag_contract']['source']}`: `{plan['flag_contract']['condition']}`. "
              "--trajectory-tokens carries processed prompt+completion tokens. Each threshold below equals the processed total at its selected completed update, "
              "so it selects that boundary exactly. All supervised totals are replay predictions.", "",
              "Corner 2 was requested at 170000 processed tokens and landed at update 38: 171860 processed / 50563 supervised, for both students.",
              "Each target includes updates −3, −2, −1, 0, +1, +2, +3. The final three updates supply the upper side of the ladder; "
              "--stop-after-trajectory stops at the last ladder boundary. Intermediate probes preserve training state.",
              "--schedule-tokens 678000 keeps the processed-token schedule horizon: ceil(678000/(pool processed tokens/ceil(prepared examples/16))) updates. "
              "--epochs 60 is retained; with schedule_tokens, the effective epoch limit is derived from that horizon.", "",
              "## Four future trajectory commands", "",
              "Two new trajectories per student (B for corners 1/4, C for corner 3), four total: the round cap. "
              "Commands are emitted only; none was executed. The CUDA device occurs only in these future-training command strings. "
              "" + ("The plan passes the specified tolerances." if plan["launchable"] else "FAILED plan: diagnostic commands only; do not launch."), ""]
    for student, entry in plan["students"].items():
        lines += [f"### {student}", "", f"Reuse corner 2: `{entry['already_measured_corner_2']['source']}`; no new A trajectory.", ""]
        for tr in entry["new_trajectories"]:
            lines += [f"Pool {tr['pool']}, corners {tr['corner_ids']}; schedule {tr['schedule_updates']} updates, "
                      f"warmup {tr['warmup_updates']} updates, effective epoch limit {tr['effective_epoch_limit']}.", "",
                      "```bash", tr["command_line"], "```", ""]
    lines += ["### Requested boundaries (identical for both students)", ""]
    for tr in plan["students"][STUDENTS[0]]["new_trajectories"]:
        lines += [f"Pool {tr['pool']}:", "", "| update | processed threshold | predicted supervised total | target-relative update |",
                  "|---:|---:|---:|---|"]
        for cp in tr["predicted_checkpoints"]:
            label = ", ".join(f"corner {t['corner']}: {t['offset_updates']:+d}" for t in cp["targets"])
            lines.append(f"| {cp['update']} | {cp['processed_tokens']} | {cp['supervised_tokens']} | {label} |")
        lines.append("")
    lines += ["Use the same evaluation distributions at all four corners, including 2wiki_new. "
              "The existing corner-2 training-probe measurement is reused; its missing scope-QA evaluation remains necessary for that channel. "
              "This plan does not count unmeasured responses as observations.", "",
              f"Trainer remains byte-identical to A1: SHA256 `{plan['trainer_sha256']}`.",
              "Failure to meet any fixed 0.5% limit produces a failed, non-launchable plan and non-zero exit; "
              "counter/hash verification failures abort before search."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--seed-max", type=int, default=269)
    parser.add_argument("--best-k", type=int, default=5)
    parser.add_argument("--n-b", nargs=2, type=int, default=(124, 140), metavar=("MIN", "MAX"))
    parser.add_argument("--n-c", nargs=2, type=int, default=(248, 280), metavar=("MIN", "MAX"))
    args = parser.parse_args(argv)
    try:
        require(0 < args.n_b[0] <= args.n_b[1] and 0 < args.n_c[0] <= args.n_c[1], "Invalid n range")
        require(args.seed_max >= 70 and args.best_k > 0, "Invalid seed range/best_k")
        plan = build_plan(range(args.n_b[0], args.n_b[1] + 1), range(args.n_c[0], args.n_c[1] + 1),
                          args.seed_max, args.best_k, args.output_dir.resolve())
    except Exception as exc:
        plan = {"schema_version": "a3b-corner-pool-search-v2", "status": "failed", "launchable": False,
                "pool_search_status": "aborted", "device": "cpu", "training_run": False,
                "model_weights_loaded": False, "failures": [f"{type(exc).__name__}: {exc}"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "plan.md").write_text(markdown(plan))
    print(f"Plan {plan['status']}: {args.output_dir / 'plan.md'}", flush=True)
    for failure in plan.get("failures", []):
        print(failure, file=sys.stderr)
    return 0 if plan["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
