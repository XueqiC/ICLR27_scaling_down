#!/usr/bin/env python3
"""Build A1 once from recorded JSON; later stages use load_development_table.

CPU, standard library only. No model loading, tokenization, training or fitting.
Run: python3 -B analysis/a1_development_table.py
All generated artifacts stay in results/a1-development-table/.

The two measurement CSVs have exactly COLUMNS. The separately requested core
and reconstructed columns live in row_metadata.csv, keyed by ROW_KEY, so the
measurement schema does not acquire implicit extra columns. Empty CSV cells
and JSON null mean missing, including unlogged domain completion counts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
from itertools import combinations, product
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT_REL = Path("results/a1-development-table")
BASE_REL = Path("results/v12-distill")
HISTORICAL_REGISTER = Path("results/v89-intervention-retro/summary.json")
COLUMNS = (
    "run_id", "pool_id", "parent_pool_id", "pool_seed", "training_seed",
    "student_id", "weight_identity", "protocol_id", "U", "D_U_pool", "D_U_seen",
    "T_actual", "processed_tokens", "optimizer_steps", "T_domain_if_available",
    "capability", "distribution", "initial_loss", "loss", "delta",
    "checkpoint_id", "source_path",
)
ROW_KEY = ("run_id", "checkpoint_id", "capability", "distribution")
FLAG_COLUMNS = (*ROW_KEY, "core", "reconstructed", "reconstructed_fields",
                "baseline_source_path", "accounting_source_path")
INTEGER_COLUMNS = ("pool_seed", "training_seed", "U", "D_U_pool", "D_U_seen",
                   "T_actual", "processed_tokens", "optimizer_steps", "T_domain_if_available")
STUDENTS = ("gemma3-270m", "gemma3-1b", "gemma3-4b")
CAPABILITIES = ("math", "code", "qa")
ARCHIVE_SUFFIXES = ("_matrix_lora_dseed41", "_matrix_lora_dseed42")
CORE_SUFFIXES = ("_matrix2_lora_dseed41", "_matrix2_lora_dseed42",
                 "_132_critical_lora_dseed51", "_132_critical_lora_dseed52")
TOLERANCES = (0.005, 0.01, 0.05)
HALVING_WINDOW = (Fraction(17, 10), Fraction(23, 10))


def require(condition, message):
    """An assertion that also remains enabled under python -O."""
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def row_key(row):
    return tuple(row[k] for k in ROW_KEY)


def expected_membership():
    core = {f"{s}/gpt-5.6-luna_full_{u}_matrix2_lora_dseed{seed}"
            for s, u, seed in product(STUDENTS, (66, 198, 594), (41, 42))}
    core |= {f"{s}/gpt-5.6-luna_full_132_critical_lora_dseed{seed}"
             for s, seed in product(STUDENTS[1:], (51, 52))}
    external = {f"{s}/gpt-5.6-luna_full_{u}_p2v2_lora_dseed{seed}"
                for s, u, seed in product(STUDENTS[:2], (75, 450), (11, 12, 13))}
    external |= {f"{s}/gpt-5.6-luna_full_200_p2v3conf_lora_dseed{seed}"
                 for s, seed in product(STUDENTS[:2], range(31, 37))}
    external |= {f"{s}/gpt-5.6-luna_full_375_p2v2test_lora_dseed{seed}"
                 for s, seed in product(STUDENTS, (21, 22, 23))}
    external |= {f"gemma3-4b/gpt-5.6-luna_full_{u}_p2v2test_lora_dseed{seed}"
                 for u, seed in product((75, 450), (11, 12))}
    external |= {
        "gemma3-270m/gpt-5.6-luna_full_75_p2v2rep_lora_dseed11_seed1",
        "gemma3-1b/gpt-5.6-luna_full_450_p2v2rep_lora_dseed11_seed1",
        "gemma3-1b/gpt-5.6-luna_full_75_p2dev_lora_dseed11",
        "gemma3-1b/gpt-5.6-luna_full_200_p2v3conf_throughput_lora_dseed31",
    }
    archived = {f"gemma3-270m/gpt-5.6-luna_full_{u}_matrix_lora_dseed{seed}"
                for u, seed in ((66, 41), (66, 42), (198, 41), (198, 42), (594, 41))}
    return core, external, archived


def assert_membership(core, external, archived):
    groups = [list(g) for g in (core, external, archived)]
    expected = expected_membership()
    for label, group, wanted in zip(("core", "external", "archived"), groups, expected):
        require(len(group) == len(set(group)), f"Duplicate {label} membership")
        if label != "archived":
            require(not any(str(p).endswith(ARCHIVE_SUFFIXES) for p in group),
                    f"Archived launch entered {label}")
        require(set(group) == wanted,
                f"{label} membership changed: missing={sorted(wanted - set(group))}; "
                f"unexpected={sorted(set(group) - wanted)}")
    require(tuple(map(len, groups)) == (22, 41, 5), "Expected 22/41/5 trajectories")
    require(all(set(a).isdisjoint(b) for a, b in combinations(groups, 2)),
            "Membership groups overlap")


class Inputs:
    def __init__(self, root):
        self.root = Path(root)
        self.sha256 = {}

    def read(self, relative):
        relative = str(relative)
        require(not any(p.endswith(ARCHIVE_SUFFIXES) for p in Path(relative).parts),
                f"Archived contents must never be read: {relative}")
        raw = (self.root / relative).read_bytes()
        value = hashlib.sha256(raw).hexdigest()
        require(relative not in self.sha256 or self.sha256[relative] == value,
                f"Input changed during analysis: {relative}")
        self.sha256[relative] = value
        return json.loads(raw)

    def verify(self):
        for path, expected in self.sha256.items():
            require(hashlib.sha256((self.root / path).read_bytes()).hexdigest() == expected,
                    f"Input changed during analysis: {path}")


def discover_membership(inputs):
    directories = {str(p.relative_to(inputs.root / BASE_REL))
                   for p in (inputs.root / BASE_REL).glob("*/*") if p.is_dir()}
    core = sorted(p for p in directories if p.endswith(CORE_SUFFIXES))
    archived = sorted(p for p in directories if p.endswith(ARCHIVE_SUFFIXES))
    # V89 is used ONLY as a historical membership manifest, never as a source
    # of losses, budgets, pool sizes, deltas or pair matches.
    register = inputs.read(HISTORICAL_REGISTER)
    external = sorted(r["trajectory"] for r in register["runs"] if r["status"] == "included")
    assert_membership(core, external, archived)
    require(set(external) <= directories, "A historical trajectory directory is missing")
    return {"core": core, "external": external, "archived": archived,
            "other_directories_excluded": sorted(directories - set(core + external + archived)),
            "assertions_passed": True, "archived_contents_read": False}


def nonnegative_integer(value, label):
    require(type(value) is int and value >= 0, f"Invalid integer {label}: {value}")
    return value


def recorded_domain_tokens(payload, capability, total):
    """Only an explicitly logged cumulative supervised-domain ledger qualifies.

    Example counts, evaluation measurement_tokens, nominal targets and prompt
    tokens are intentionally not fallbacks. Even update-0 stays missing when
    no per-domain token total was logged, as required for the canonical schema.
    """
    ledger = payload.get("completion_tokens_seen_by_domain")
    if ledger is None:
        return None
    require(set(ledger) == set(CAPABILITIES), "Incomplete per-domain token ledger")
    require(sum(nonnegative_integer(v, k) for k, v in ledger.items()) == total,
            "Domain completion tokens disagree with T_actual")
    return ledger[capability]


def audit_accounting(log):
    """Prove a distinct first pass from every step and unique-example counters.

    If the first pass is incomplete, D_U_pool stays missing here; it may be
    recovered from another run ONLY after exact pool-hash equality is checked.
    D_U_seen before completion is the supervised sum of distinct encountered
    examples, not the prompt-inclusive unique_data_tokens field.
    """
    require(log["status"] == "trained", "Unfinished training log")
    curve = log["loss_curve"]
    require(curve, "Missing update ledger")
    unique = log["training_examples"] == log["unique_data_pool_examples"]
    batch = nonnegative_integer(log["effective_batch_size_sequences"], "batch size")
    require(batch > 0, "Empty batches")
    pool_n = log["unique_data_pool_examples"]
    counts = {0: {"processed_tokens": 0, "T_actual": 0, "D_U_seen": 0}}
    processed = supervised = first_supervised = 0
    epoch = 1
    epoch_rows = defaultdict(list)
    du = None
    for step, r in enumerate(curve, 1):
        require(r["step"] == step and r["epoch"] in (epoch, epoch + 1),
                "Missing/noncontiguous update or epoch")
        epoch = r["epoch"]
        epoch_rows[epoch].append(r)
        p = nonnegative_integer(r["tokens"], "tokens")
        t = nonnegative_integer(r["completion_tokens"], "completion_tokens")
        require(0 < t <= p, "Invalid completion increment")
        processed += p
        supervised += t
        for key in ("tokens_seen", "seen_tokens", "processed_tokens"):
            require(r[key] == processed, f"{key} disagrees at update {step}")
        require(r["completion_tokens_seen"] == supervised, "Supervised ledger disagrees")
        require(r["unique_data_pool_tokens"] == log["unique_data_pool_tokens"]
                and r["unique_data_pool_examples"] == pool_n, "Pool changed during run")
        if unique and epoch == 1:
            require(r["unique_examples_seen"] == min(step * batch, pool_n)
                    and r["unique_data_tokens"] == processed,
                    "First pass does not contain distinct examples exactly once")
            first_supervised = supervised
            if r["unique_examples_seen"] == pool_n:
                require(processed == log["unique_data_pool_tokens"], "Incomplete pool pass")
                du = first_supervised
        seen = first_supervised if unique else None
        if unique and epoch > 1:
            require(du is not None and r["unique_examples_seen"] == pool_n
                    and r["unique_data_tokens"] == log["unique_data_pool_tokens"],
                    "Repeated epoch without a verified complete first pass")
            seen = du
        counts[step] = {"processed_tokens": processed, "T_actual": supervised, "D_U_seen": seen}
    if du is not None:
        for e, rows in epoch_rows.items():
            if e < epoch or sum(r["tokens"] for r in rows) == log["unique_data_pool_tokens"]:
                require(sum(r["completion_tokens"] for r in rows) == du
                        and sum(r["tokens"] for r in rows) == log["unique_data_pool_tokens"],
                        "Complete epochs disagree on supervised pool tokens")
    require(log["updates"] == log["optimizer_steps"] == len(curve), "Final step mismatch")
    require(log["completion_tokens_seen"] == supervised, "Final supervised total mismatch")
    require(all(log[k] == processed for k in ("processed_tokens", "seen_tokens", "tokens_seen")),
            "Final processed total mismatch")
    snapshots = log["trajectory"]
    require(len(snapshots) == len({r["updates"] for r in snapshots}), "Duplicate snapshots")
    for r in snapshots:
        require(r["updates"] in counts, "Snapshot outside update ledger")
        c = counts[r["updates"]]
        require(r["completion_tokens_seen"] == c["T_actual"]
                and r["processed_tokens"] == c["processed_tokens"], "Snapshot accounting mismatch")
    return counts, du


def training_protocol(log, baseline, core):
    keys = ("teacher", "recipe", "training_mode", "learning_rate", "optimizer", "scheduler",
            "warmup_ratio", "effective_batch_size_sequences", "max_len", "dtype", "lora")
    protocol = {k: log[k] for k in keys}
    protocol["domains"] = log["domains"]
    protocol["schedule_tokens"] = baseline.get("schedule_tokens")
    protocol["schedule_rule"] = ("fixed processed-token horizon converted to updates using pool mean length"
                                 if protocol["schedule_tokens"] is not None else "fixed epoch horizon")
    if protocol["schedule_tokens"] is None:
        protocol["epochs"] = log["epochs"]
    # Actual horizon/warmup updates remain in the run audit. Pool-dependent
    # update rounding does not create a different nominal training protocol.
    if core:
        require(protocol["schedule_tokens"] == 678000 and baseline["stop_after_trajectory"]
                and log["stopped_after_trajectory"] and not log["trajectory_tokens_unreached"],
                "Core schedule/stop protocol changed")
        require(log["training_seed"] == 0 and log["learning_rate"] == 1e-4
                and log["teacher"] == "gpt-5.6-luna" and log["recipe"] == "full"
                and log["training_mode"] == "lora", "Core training protocol changed")
    prefix = "core:" if core else "external-diagnostic:"
    return prefix + digest(protocol)[:20], protocol


def probe_distribution(payload, cap):
    metadata = {k: payload[k] for k in (
        "probe_source", "probe_seed", "probe_half", "n_probe_requested", "loss_definition")}
    for k in ("measurement_benchmarks", "measurement_samples", "measurement_tokens"):
        metadata[k] = payload[k][cap]
    require(metadata["measurement_samples"] > 0 and metadata["measurement_tokens"] > 0,
            "Empty capability probe")
    return "training_probe:" + metadata["measurement_benchmarks"] + ":" + digest(metadata)[:16], metadata


def own_delta(loss, initial_loss):
    require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in (loss, initial_loss)),
            "Missing/nonfinite loss or own update-0")
    return loss - initial_loss


def load_run(inputs, run_id, core):
    require(not run_id.endswith(ARCHIVE_SUFFIXES), "Archived launch selected")
    base = BASE_REL / run_id
    log_path = base / "train_log.json"
    log = inputs.read(log_path)
    paths = sorted((inputs.root / base).glob("trajectory/update-*/eval.json"))
    docs = {int(p.parent.name.removeprefix("update-")): (str(p.relative_to(inputs.root)),
            inputs.read(p.relative_to(inputs.root))) for p in paths}
    require(0 in docs, f"Missing own update-0 for {run_id}")
    baseline_path, baseline = docs[0]
    counts, du = audit_accounting(log)
    final_path = base / "eval.json"
    final = inputs.read(final_path)
    final_step = final.get("updates", log["updates"])
    require(final_step == log["updates"], "Final evaluation update mismatch")
    aliases = []
    if final_step in docs:
        require(final["post_training"] == docs[final_step][1]["post_training"],
                "Conflicting final/snapshot losses")
        aliases.append(str(final_path))
    else:
        docs[final_step] = (str(final_path), final)
    require(set(docs) == {0, log["updates"], *(r["updates"] for r in log["trajectory"])},
            "Evaluation/log checkpoint set mismatch")
    if core:
        require(len(docs) == (15 if "_critical_" in run_id else 5), "Core checkpoint count changed")
    require(log["student"] == baseline["student"] == run_id.split("/")[0], "Student mismatch")
    u, seed = log["n_per_domain"], log["data_sampling_seed"]
    require(all(v["source_rows"] == u for v in log["trace_counts"].values()),
            "U was not the actual selected source-row count per domain")
    pool_hash = baseline["data_pool_sha256"]
    require(len(pool_hash) == 64, "Missing pool hash")
    protocol_id, protocol = training_protocol(log, baseline, core)
    for step, (path, payload) in docs.items():
        require(payload.get("updates", step) == step, "Checkpoint path/update mismatch")
        require((payload["student"], payload["n_per_domain"], payload["data_seed"],
                 payload["training_seed"], payload["data_pool_sha256"])
                == (log["student"], u, seed, log["training_seed"], pool_hash),
                f"Checkpoint identity mismatch: {path}")
        require(payload["resolved_student"] == log["resolved_student"], "Base model changed")
        for key, value in protocol.items():
            if key in payload and key not in ("domains",):
                require(payload[key] == value, f"Checkpoint protocol mismatch: {key}")
        for key, target in (("completion_tokens_seen", "T_actual"), ("processed_tokens", "processed_tokens")):
            if key in payload:
                require(payload[key] == counts[step][target], f"Eval/log {key} mismatch")
        require(payload["unique_data_pool_tokens"] == log["unique_data_pool_tokens"]
                and payload["unique_data_pool_examples"] == log["unique_data_pool_examples"],
                "Checkpoint pool accounting changed")
    return {"run_id": run_id, "core": core, "log": log, "log_path": str(log_path),
            "docs": docs, "baseline": baseline, "baseline_path": baseline_path,
            "counts": counts, "D_U_pool": du, "pool_id": "sha256:" + pool_hash,
            "D_U_source": str(log_path) if du is not None else None,
            "protocol_id": protocol_id, "protocol": protocol, "aliases": aliases}


def reconstruct_shared_pools(runs):
    known = {}
    for run in runs:
        if run["D_U_pool"] is not None:
            signature = (run["D_U_pool"], run["log"]["unique_data_pool_tokens"],
                         run["log"]["unique_data_pool_examples"])
            key = run["pool_id"]
            require(key not in known or known[key][0] == signature, "Identical pool hashes disagree")
            known[key] = signature, run["log_path"]
    for run in runs:
        if run["D_U_pool"] is None and run["pool_id"] in known:
            signature, path = known[run["pool_id"]]
            require(signature[1:] == (run["log"]["unique_data_pool_tokens"],
                                     run["log"]["unique_data_pool_examples"]), "Shared pool totals disagree")
            run["D_U_pool"], run["D_U_source"] = signature[0], path


def recorded_weight_identity(payload):
    """Retain a partial recorded identity honestly, without resolving a live HF tag.

    A resolved_student name is recorded, but it is not an immutable base-weight
    hash. Null revision/adapter hashes explicitly preserve that limitation.
    V99 records its actually loaded base revision and adapter digest.
    """
    revision = payload.get("model_revision") or payload.get("revision")
    source = ("model_revision" if payload.get("model_revision") else
              "revision" if payload.get("revision") else None)
    local_path = Path(payload.get("model_local_path") or "")
    # V99 sometimes leaves model_revision null but records the exact snapshot
    # directory actually loaded. This is an exact path extraction, not a lookup
    # of a current HF alias or an assumption about the original training run.
    if local_path.parent.name == "snapshots" and len(local_path.name) == 40 and all(
            c in "0123456789abcdef" for c in local_path.name):
        require(revision is None or revision == local_path.name, "Recorded model revision/path disagree")
        if revision is None:
            revision, source = local_path.name, "model_local_path"
    return canonical({"base_model": payload["resolved_student"],
                      "base_revision": revision, "base_revision_source": source,
                      "adapter_sha256": payload.get("adapter_sha256"),
                      "state": "base" if payload.get("updates") == 0 else "trained",
                      "checkpoint_ref": payload.get("checkpoint_name")})


def make_row(run, step, cap, distribution, loss, initial, path, payload, baseline_path):
    log, c = run["log"], run["counts"][step]
    row = dict.fromkeys(COLUMNS)
    row.update(run_id=run["run_id"], pool_id=run["pool_id"],
               parent_pool_id=payload.get("parent_pool_id", log.get("parent_pool_id")),
               pool_seed=log["data_sampling_seed"], training_seed=log["training_seed"],
               student_id=log["student"], weight_identity=recorded_weight_identity(payload),
               protocol_id=run["protocol_id"], U=log["n_per_domain"],
               D_U_pool=run["D_U_pool"], D_U_seen=c["D_U_seen"], T_actual=c["T_actual"],
               processed_tokens=c["processed_tokens"], optimizer_steps=step,
               T_domain_if_available=recorded_domain_tokens(run["docs"][step][1], cap, c["T_actual"]),
               capability=cap, distribution=distribution, initial_loss=initial, loss=loss,
               delta=own_delta(loss, initial), checkpoint_id=f"update-{step:08d}", source_path=path)
    # Give partial trained identities a unique saved checkpoint reference.
    identity = json.loads(row["weight_identity"])
    identity["checkpoint_ref"] = str(BASE_REL / run["run_id"] / "trajectory" / row["checkpoint_id"])
    if step not in {int(Path(p).parent.name.removeprefix("update-"))
                    for p, _ in run["docs"].values() if Path(p).parent.name.startswith("update-")}:
        identity["checkpoint_ref"] = str(BASE_REL / run["run_id"])
    row["weight_identity"] = canonical(identity)
    reconstructed = [k for k in ("D_U_pool", "D_U_seen") if row[k] is not None]
    if identity["base_revision_source"] == "model_local_path":
        reconstructed.append("weight_identity")
    if "completion_tokens_seen" not in payload and "actual_supervised_tokens" not in payload:
        reconstructed.append("T_actual")
    if "processed_tokens" not in payload:
        reconstructed.append("processed_tokens")
    flags = {**{k: row[k] for k in ROW_KEY}, "core": run["core"],
             "reconstructed": bool(reconstructed), "reconstructed_fields": ";".join(reconstructed),
             "baseline_source_path": baseline_path, "accounting_source_path": run["log_path"]}
    return row, flags


def training_rows(run, distributions):
    rows, metadata = [], []
    for step, (path, payload) in sorted(run["docs"].items()):
        for cap in CAPABILITIES:
            dist, spec = probe_distribution(payload, cap)
            base_dist, _ = probe_distribution(run["baseline"], cap)
            require(dist == base_dist, "Own update-0 uses a different distribution")
            distributions[dist] = spec
            row, flags = make_row(run, step, cap, dist, payload["post_training"][cap],
                                  run["baseline"]["post_training"][cap], path, payload, run["baseline_path"])
            if "delta" in payload:
                require(math.isclose(payload["delta"][cap], row["delta"], abs_tol=1e-10),
                        "Recorded delta disagrees with own update-0")
            rows.append(row)
            metadata.append(flags)
    return rows, metadata


def scope_rows(inputs, runs, distributions):
    by_run = {r["run_id"]: r for r in runs}
    rows, metadata, seen, aliases, omitted = [], [], {}, [], []
    paths = sorted((inputs.root / "results/v99-scope").rglob("update-*.json"))
    for path in paths:
        source = str(path.relative_to(inputs.root))
        result = inputs.read(source)
        require(result.get("status") == "complete", f"Incomplete V99 result: {source}")
        name = result["trajectory_run_id"]
        require(not name.endswith(ARCHIVE_SUFFIXES), "Archived run in V99")
        if name not in by_run:
            omitted.append(source)
            continue
        run, base = by_run[name], result["update_0"]
        require(base["trajectory_run_id"] == name and base["updates"] == 0
                and base["actual_supervised_tokens"] == base["processed_tokens"] == 0,
                "Scope baseline is not own update-0")
        protocol = result["protocol"]
        require(set(result["losses"]) == set(base["losses"]) == set(protocol["distributions"])
                == {"2wiki_new", "musique", "triviaqa"}, "Scope distribution set changed")
        for payload, suffix in ((result, ""), (base, "#/update_0")):
            step = payload["updates"]
            require(step in run["docs"], "V99 checkpoint absent from training log")
            require(payload["data_pool_sha256"] == run["baseline"]["data_pool_sha256"]
                    and payload["student"] == run["log"]["student"]
                    and payload["resolved_student"] == run["log"]["resolved_student"]
                    and payload["data_seed"] == run["log"]["data_sampling_seed"]
                    and payload["training_seed"] == run["log"]["training_seed"], "Scope identity mismatch")
            require(payload["actual_supervised_tokens"] == run["counts"][step]["T_actual"]
                    and payload["processed_tokens"] == run["counts"][step]["processed_tokens"]
                    and payload["D_U"] == run["D_U_pool"], "Scope accounting mismatch")
            require(payload["adapter_loaded"] is (step != 0), "Scope adapter not loaded correctly")
            for label, cell in payload["losses"].items():
                base_cell = base["losses"][label]
                require(cell["probe_identity"] == base_cell["probe_identity"] == protocol["probes"][label]
                        and cell["tokens"] == base_cell["tokens"] and cell["n"] == base_cell["n"],
                        "Scope baseline/checkpoint probes differ")
                spec = {"panel": cell["probe_identity"], "tokens": cell["tokens"], "n": cell["n"],
                        **{k: protocol[k] for k in ("loss_definition", "information_condition", "dtype",
                                                    "max_len", "batch_size", "scorer", "panel_builder")}}
                dist = label + ":" + digest(spec)[:16]
                distributions[dist] = spec
                row, flags = make_row(run, step, "qa", dist, cell["loss"], base_cell["loss"],
                                      source + suffix, payload, source + "#/update_0")
                if suffix == "":
                    require(result["delta_from_update_0"][label] == row["delta"], "Scope delta mismatch")
                key = row_key(row)
                if key in seen:
                    previous = seen[key]
                    require(all(previous[k] == row[k] for k in COLUMNS if k != "source_path"),
                            f"Conflicting repeated scope evaluation: {key}")
                    aliases.append({"key": list(key), "retained": previous["source_path"],
                                    "alias": row["source_path"]})
                else:
                    seen[key] = row
                    rows.append(row)
                    metadata.append(flags)
    return rows, metadata, {"files_read": len(paths), "deduplicated_aliases": aliases,
                            "nonmember_files_excluded": omitted}


def relative_mismatch(a, b):
    if a == b == 0:
        return 0.0
    require(min(a, b) > 0, "Positive budgets required")
    return abs(a - b) / min(a, b)


def budget_match(a, b, tolerance):
    require(0 <= tolerance <= 0.05, "Tolerance must be between 0 and 5%")
    return a > 0 and b > 0 and Fraction(abs(a - b), min(a, b)) <= Fraction(str(tolerance))


def budget_band(t):
    """Fixed absolute bands, never nominal checkpoint ordinals or sample quantiles."""
    for edge, name in ((37500, "[0,37500)"), (75000, "[37500,75000)"),
                       (150000, "[75000,150000)")):
        if t < edge:
            return name
    return "[150000,infinity)"


def intervention_pairs(rows, tolerances=TOLERANCES):
    """All eligible pairs; reverse pool changes are the negatives, not extra data.

    Excludes update-0 from positive-budget inventories. Does not thin by ordinal,
    interpolate, choose nearest neighbours, or aggregate repeated capabilities.
    """
    require(len(rows) == len({row_key(r) for r in rows}), "Duplicate measurement keys")
    require(all(r["protocol_id"].startswith("core:") for r in rows),
            "External diagnostics cannot enter the core intervention inventory")
    groups = defaultdict(list)
    for r in rows:
        require(not r["run_id"].endswith(ARCHIVE_SUFFIXES), "Archived run in pair input")
        if r["T_actual"] > 0:
            groups[r["student_id"], r["protocol_id"], r["capability"], r["distribution"]].append(r)
    pairs = []

    def add(kind, a, b, tolerance=None):
        t1, t2 = a["T_actual"], b["T_actual"]
        direction = (f"U={a['U']} fixed; T increases" if kind == "I_T" else
                     f"{a['U']}->{b['U']}" if kind == "I_U" else
                     f"U={a['U']}; pool seed {a['pool_seed']}->{b['pool_seed']}"
                     if kind == "same_U_pool_seed" else f"U={a['U']}; training seed changes")
        pairs.append({"kind": kind, "tolerance": tolerance, "first": list(row_key(a)),
                      "second": list(row_key(b)), "student_id": a["student_id"],
                      "protocol_id": a["protocol_id"], "capability": a["capability"],
                      "distribution": a["distribution"], "direction": direction,
                      "budget_band": budget_band(max(t1, t2)), "T_first": t1, "T_second": t2,
                      "budget_ratio": max(t1, t2) / min(t1, t2), "residual_tokens_signed": t2 - t1,
                      "residual_mismatch": relative_mismatch(t1, t2),
                      "effect": b["delta"] - a["delta"],
                      "both_final_checkpoints": a.get("_is_final", False) and b.get("_is_final", False)})

    for group in groups.values():
        for a, b in combinations(group, 2):
            if a["run_id"] == b["run_id"]:
                a, b = sorted((a, b), key=lambda r: r["T_actual"])
                if HALVING_WINDOW[0] <= Fraction(b["T_actual"], a["T_actual"]) <= HALVING_WINDOW[1]:
                    add("I_T", a, b)
                continue
            if a["U"] != b["U"]:
                a, b = sorted((a, b), key=lambda r: r["U"])
                require(a["pool_id"] != b["pool_id"], "Different U but same pool identity")
                kind = "I_U"
            elif a["pool_seed"] != b["pool_seed"]:
                a, b = sorted((a, b), key=lambda r: (r["pool_seed"], r["run_id"]))
                kind = "same_U_pool_seed"
            elif a["training_seed"] != b["training_seed"]:
                a, b = sorted((a, b), key=lambda r: (r["training_seed"], r["run_id"]))
                kind = "same_U_training_seed"
            else:
                continue
            for tolerance in tolerances:
                if budget_match(a["T_actual"], b["T_actual"], tolerance):
                    add(kind, a, b, tolerance)
    return sorted(pairs, key=lambda p: (p["kind"], p["tolerance"] or 0, p["first"], p["second"]))


def distribution_stats(values):
    values = sorted(values)
    if not values:
        return {"n": 0, **dict.fromkeys(("min", "p25", "median", "p75", "p95", "max", "mean"))}

    def quantile(q):
        index = (len(values) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (index - lo) * (values[hi] - values[lo])

    return {"n": len(values), "min": values[0], "p25": quantile(.25), "median": quantile(.5),
            "p75": quantile(.75), "p95": quantile(.95), "max": values[-1], "mean": statistics.mean(values)}


def inventory(pairs):
    groups = defaultdict(list)
    fields = ("kind", "tolerance", "budget_band", "student_id", "direction", "capability", "distribution")
    for p in pairs:
        groups[tuple(p[k] for k in fields)].append(p)

    def summarize(ps):
        physical = {(tuple(p["first"][:2]), tuple(p["second"][:2])): p for p in ps}
        unique = list(physical.values())
        return {"measurement_pairs": len(ps), "checkpoint_pairs": len(unique),
                "trajectories": len({key[0] for p in ps for key in (p["first"], p["second"])}),
                "both_final_checkpoint_pairs": sum(p["both_final_checkpoints"] for p in unique),
                "T_first": distribution_stats([p["T_first"] for p in unique]),
                "T_second": distribution_stats([p["T_second"] for p in unique]),
                "residual_mismatch_percent": distribution_stats([100 * p["residual_mismatch"] for p in unique]),
                "residual_tokens_signed": distribution_stats([p["residual_tokens_signed"] for p in unique]),
                "absolute_effect": distribution_stats([abs(p["effect"]) for p in ps]),
                "effect": distribution_stats([p["effect"] for p in ps])}

    strata = [{**dict(zip(fields, key)), **summarize(ps)} for key, ps in sorted(groups.items())]
    totals = []
    for kind in ("I_T", "I_U", "same_U_pool_seed", "same_U_training_seed"):
        for tol in ((None,) if kind == "I_T" else TOLERANCES):
            ps = [p for p in pairs if p["kind"] == kind and p["tolerance"] == tol]
            totals.append({"kind": kind, "tolerance": tol, **summarize(ps)})
    return {"totals": totals, "strata": strata}


def budget_only_intervention(predictor, t1, t2):
    return predictor(t2) - predictor(t1)


def verify_budget_only_zero(rows):
    exact = intervention_pairs(rows, (0.0,))
    exact_pool = [p for p in exact if p["kind"] == "I_U"]
    predictors = (lambda t: 7.0, lambda t: 2.3 * t, lambda t: math.log1p(t / 100000),
                  lambda t: 1 - math.exp(-t / 70000))
    budgets = {0, 1, 100000, *(r["T_actual"] for r in rows)}
    require(all(budget_only_intervention(f, t, t) == 0.0 for f in predictors for t in budgets),
            "Budget-only identity failed")
    require(all(p["T_first"] == p["T_second"] for p in exact_pool), "Nonexact pair in exact audit")
    return {"verified": True, "identity": "I_U[f(T)] = f(T) - f(T) = 0 identically at exactly matched T",
            "conditions": "Same predictor/parameters, student, protocol, capability and distribution at both endpoints.",
            "positive_exact_measurement_pairs": len(exact_pool),
            "positive_exact_checkpoint_pairs": len({(tuple(p["first"][:2]), tuple(p["second"][:2])) for p in exact_pool}),
            "zero_budget_anchors_excluded_from_inventory": True,
            "nonzero_tolerance_note": "At unequal T, a budget-only predictor may yield a nonzero effect solely from residual mismatch."}


def missing_summary(rows):
    return {k: sum(r[k] is None for r in rows) for k in COLUMNS}


def weight_identity_summary(rows):
    identities = [json.loads(r["weight_identity"]) for r in rows]
    return {"rows_without_recorded_base_revision": sum(i["base_revision"] is None for i in identities),
            "trained_rows_without_recorded_adapter_digest": sum(
                i["state"] == "trained" and i["adapter_sha256"] is None for i in identities),
            "rows_with_partial_identity": sum(i["base_revision"] is None or (
                i["state"] == "trained" and i["adapter_sha256"] is None) for i in identities)}


def build(root=ROOT):
    inputs = Inputs(root)
    membership = discover_membership(inputs)
    runs = [load_run(inputs, name, core) for core, names in
            ((True, membership["core"]), (False, membership["external"])) for name in names]
    reconstruct_shared_pools(runs)
    require(len({r["protocol_id"] for r in runs if r["core"]}) == 1, "Controlled protocols differ")
    rows, flags, distributions = [], [], {}
    for run in runs:
        rr, ff = training_rows(run, distributions)
        rows.extend(rr)
        flags.extend(ff)
    sr, sf, scope_audit = scope_rows(inputs, runs, distributions)
    rows.extend(sr)
    flags.extend(sf)
    require(len(rows) == len({row_key(r) for r in rows}), "Duplicate canonical row")
    require({row_key(r) for r in rows} == {row_key(f) for f in flags}, "Flag join is not one-to-one")
    core_names = set(membership["core"])
    core = sorted((r for r in rows if r["run_id"] in core_names), key=row_key)
    external = sorted((r for r in rows if r["run_id"] not in core_names), key=row_key)
    require({r["run_id"] for r in core} == core_names, "Missing core measurement run")
    require({r["run_id"] for r in external} == set(membership["external"]), "Missing diagnostic run")
    require(not any(r["run_id"].endswith(ARCHIVE_SUFFIXES) for r in rows), "Archived output row")
    last = {r["run_id"]: r["log"]["updates"] for r in runs}
    pair_rows = [{**r, "_is_final": r["optimizer_steps"] == last[r["run_id"]]} for r in core]
    pairs = intervention_pairs(pair_rows)
    report = {
        "schema_version": "a1-development-table-v1", "device": "cpu", "training": False,
        "columns": list(COLUMNS), "row_key": list(ROW_KEY), "membership": membership,
        "counts": {"core_runs": len(core_names), "external_runs": len(membership["external"]),
                   "archived_runs_excluded": len(membership["archived"]), "core_rows": len(core),
                   "external_rows": len(external), "core_training_probe_rows": len(core) - sum(r in core for r in sr),
                   "core_scope_rows": sum(r in core for r in sr),
                   "core_checkpoints": len({(r["run_id"], r["checkpoint_id"]) for r in core}),
                   "external_checkpoints": len({(r["run_id"], r["checkpoint_id"]) for r in external})},
        "missing_values": {"core": missing_summary(core), "external": missing_summary(external)},
        "weight_identity_completeness": {"core": weight_identity_summary(core),
                                         "external": weight_identity_summary(external)},
        "reconstructed_fields": dict(Counter(k for f in flags for k in f["reconstructed_fields"].split(";") if k)),
        "protocols": {r["protocol_id"]: r["protocol"] for r in runs},
        "distributions": distributions, "scope_audit": scope_audit,
        "inventory": inventory(pairs), "pairs": pairs, "budget_only_zero": verify_budget_only_zero(pair_rows),
        "run_audit": [{"run_id": r["run_id"], "core": r["core"], "pool_id": r["pool_id"],
                       "pool_seed": r["log"]["data_sampling_seed"], "U": r["log"]["n_per_domain"],
                       "prepared_examples": r["log"]["unique_data_pool_examples"],
                       "source_rows_by_domain": r["log"]["trace_counts"], "D_U_pool": r["D_U_pool"],
                       "D_U_source": r["D_U_source"], "D_U_reconstructed": r["D_U_pool"] is not None,
                       "actual_schedule_updates": r["log"]["total_updates_planned"],
                       "actual_warmup_updates": r["log"]["warmup_steps"],
                       "final_supervised_tokens": r["counts"][r["log"]["updates"]]["T_actual"],
                       "checkpoint_count": len(r["docs"]), "final_eval_aliases": r["aliases"]}
                      for r in runs],
        "field_notes": {
            "U": "Recorded number of source examples actually selected per domain before coverage/tokenization deletion; asserted against trace_counts.source_rows. Prepared unique example count is separately audited.",
            "D_U_pool": "Supervised completion tokens in one verified complete distinct pool pass; never unique_data_pool_tokens (prompt-inclusive). Shared-pool recovery requires identical recorded data_pool_sha256 and prepared counts.",
            "D_U_seen": "Unique supervised completion tokens actually encountered by this checkpoint. First distinct-pass cumulative completion sum; full pool sum after coverage. Update-0 is zero.",
            "T_actual": "Actual cumulative supervised completion tokens reconciled at every optimizer update. No milestone targets used as observations.",
            "T_domain_if_available": "Only completion_tokens_seen_by_domain qualifies. Unlogged domain totals remain missing, including anchors. Per-domain example counts and measurement_tokens never substitute.",
            "parent_pool_id": "Missing unless explicitly recorded. Same seed or different U does not prove nesting or parentage.",
            "weight_identity": "JSON of recorded base identifier, base revision, adapter digest and checkpoint reference. Null revision/digest means partial identity, not an immutable weight fingerprint. Training probes lack recorded base revisions. V99 revisions are retained or extracted exactly from its logged model_local_path snapshot directory (flagged reconstructed); adapter digests are retained where recorded. No live model alias is resolved.",
            "protocol_id": "Cohort-prefixed hash of recorded training settings and schedule rule. Actual pool-dependent rounded schedule/warmup update counts are in run_audit. External cohort prefix always differs from core.",
            "flags": "core and reconstructed are columns of row_metadata.csv, joined one-to-one on row_key; canonical measurement CSVs retain exactly the specified 22 columns.",
            "reconstruction": "D_U_pool and D_U_seen are exact algebraic reconstructions from logged completion increments and verified distinct-pass counters. No per-domain allocation, tokenization estimate, model loading or new evaluation is performed.",
        },
        "pair_rules": {
            "I_T": "All within-run positive-budget pairs with inclusive T_high/T_low in [1.7,2.3] (2 +/- 15%). Direction is increasing T.",
            "I_U": "All cross-run, different-U/pool pairs, same student/protocol/capability/distribution; increasing U orientation. Reverse direction is the negative of the stored effect, not an independent observation.",
            "matching": "Inclusive abs(T2-T1)/min(T1,T2) <= tolerance, checked with rational arithmetic; tolerances 0.005, 0.01, 0.05. All checkpoint combinations, no interpolation or ordinal matching.",
            "same_U": "Different pool seeds and different training seeds are separate inventories; neither is I_U. Core training seeds are all zero.",
            "strata": "Fixed actual-token bands by max(T_first,T_second): [0,37500), [37500,75000), [75000,150000), [150000,infinity); also student, pool direction, capability and distribution. No pooling across these strata for a claimed effect.",
            "counts": "measurement_pairs counts capability/distribution rows; checkpoint_pairs deduplicates physical endpoint pairs within each reported group. Strata are not independent replicates. Tolerances are nested, not additive.",
        },
        "input_sha256": inputs.sha256,
        "code_sha256": {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
                        for name in ("analysis/a1_development_table.py", "analysis/v12_distill.py")},
    }
    inputs.verify()
    return core, external, sorted(flags, key=row_key), report


def fmt(value):
    return "NA" if value is None else f"{value:.6g}"


def markdown(report):
    c = report["counts"]
    lines = ["# A1 canonical development table", "",
             f"CPU only; no training or model execution. **{c['core_runs']} core trajectories, "
             f"{c['core_checkpoints']} checkpoints, {c['core_rows']} rows** "
             f"({c['core_training_probe_rows']} training probes + {c['core_scope_rows']} fresh QA rows). "
             f"**{c['external_runs']} historical trajectories / {c['external_rows']} rows** are external diagnostics "
             f"in a separate file. All **{c['archived_runs_excluded']} archived launches** are excluded before reading their contents.", "",
             "Read `development_table.csv` via `load_development_table()`. Read the stored `pairs` in `summary.json` "
             "for later interventions; do not recompute from earlier result directories. "
             "`external_diagnostic_table.csv` requires `allow_external_diagnostic=True` in the reader and "
             "cannot enter the core pairing function. No historical rows are appended to the main CSV.", "",
             "The requested measurement schema is exactly 22 columns. `row_metadata.csv` supplies the "
             "separate `core`, `reconstructed` and `reconstructed_fields` columns, plus baseline/accounting "
             "provenance, using the same four-part row key. Its core flags are true for all main-table rows "
             "and false for all diagnostic rows. Missing CSV cells load as None; JSON uses null.", "",
             "## Membership assertions", "",
             "The explicit identity sets equal 18 matrix2 + four U=132 critical trajectories, 41 V89 historical "
             "members and five archived matrix launches. Unexpected/missing identities, duplicate members, "
             "overlap, archived input reads and archived output rows fail closed. Other repository runs are "
             "listed as excluded in the JSON manifest.", "",
             "## Values and provenance", ""]
    lines += [f"- **{key}**: {value}" for key, value in report["field_notes"].items()]
    lines += ["", "Loss displacement is the checkpoint loss minus that trajectory's own update-0 loss "
              "on exactly the same capability and distribution. V99's embedded own baselines are validated "
              "and duplicate baseline records are collapsed; differing losses/identities fail rather than "
              "silently selecting a rerun. All 2Wiki-new, MuSiQue and TriviaQA results found for selected "
              "runs are included. These fresh panels currently cover anchors and final checkpoints only.", "",
              "## Missing values", "", "| Column | Core missing | External missing |", "|---|---:|---:|"]
    for k in COLUMNS:
        lines.append(f"| {k} | {report['missing_values']['core'][k]} | {report['missing_values']['external'][k]} |")
    lines += ["", "Partial weight identities additionally contain explicit null base revisions/adapter digests; "
              "a nonmissing identity cell does not claim those fields were recorded. No parent pool "
              "is inferred from U or seed. Reconstruction counts across both tables: `" +
              canonical(report["reconstructed_fields"]) + "`.", "", "## Intervention-pair inventory", ""]
    identity_counts = report["weight_identity_completeness"]
    lines[-2:-2] = ["Partial weight identity rows: " +
                     f"{identity_counts['core']['rows_with_partial_identity']} core, " +
                     f"{identity_counts['external']['rows_with_partial_identity']} external. " +
                     "Their checkpoint reference and recorded model identifier are preserved; "
                     "unrecorded immutable weight identity fields remain null.", ""]
    lines += [f"- **{key}**: {value}" for key, value in report["pair_rules"].items()]
    lines += ["", "**Exactly matched budgets:** " + report["budget_only_zero"]["identity"] + ". "
              "This is an algebraic identity for any deterministic budget-only predictor with the same "
              "parameters at both endpoints; numerical checks also pass. Observed positive exact matches: "
              f"{report['budget_only_zero']['positive_exact_checkpoint_pairs']} checkpoint pairs / "
              f"{report['budget_only_zero']['positive_exact_measurement_pairs']} measurement pairs. "
              "Update-0 anchors do not count as budget interventions. At nonzero mismatch a budget-only "
              "effect need not vanish.", "",
              "### Totals (same-U seed questions remain separate)", "",
              "| Kind | Tolerance | Checkpoint pairs | Measurement pairs | Trajectories | Both final | Residual % min / median / p95 / max | Signed token residual min / median / max |",
              "|---|---:|---:|---:|---:|---:|---|---|"]
    for r in report["inventory"]["totals"]:
        m, t = r["residual_mismatch_percent"], r["residual_tokens_signed"]
        lines.append(f"| {r['kind']} | {fmt(100*r['tolerance'])+'%' if r['tolerance'] is not None else 'ratio 1.7–2.3'} | "
                     f"{r['checkpoint_pairs']} | {r['measurement_pairs']} | {r['trajectories']} | {r['both_final_checkpoint_pairs']} | "
                     + " / ".join(fmt(m[k]) for k in ("min", "median", "p95", "max")) + " | "
                     + " / ".join(fmt(t[k]) for k in ("min", "median", "max")) + " |")
    lines += ["", "### Actual-budget × student × direction × capability × distribution", "",
              "Each row retains its own effect distribution so numerous small early effects cannot hide "
              "large final-budget effects. Both-final counts distinguish actual endpoints within the "
              "highest band. Full residual values and endpoint keys are stored in `summary.json:pairs`; "
              "the JSON inventory also includes quartiles, means, signed residuals and actual T ranges. "
              "Reverse every U arrow to obtain a pool reduction with exactly the opposite signed effect; "
              "do not count that as another independent pair.", "",
              "| Kind | Tol % | Budget band | Student | Direction | Capability / distribution | Pairs | Both final | Residual % min / median / p95 / max | Effect median [min,max] |",
              "|---|---:|---|---|---|---|---:|---:|---|---|"]
    for r in report["inventory"]["strata"]:
        m, e = r["residual_mismatch_percent"], r["effect"]
        lines.append(f"| {r['kind']} | {fmt(100*r['tolerance']) if r['tolerance'] is not None else '—'} | "
                     f"{r['budget_band']} | {r['student_id']} | {r['direction']} | "
                     f"{r['capability']} / {r['distribution'].rsplit(':',1)[0]} | {r['measurement_pairs']} | "
                     f"{r['both_final_checkpoint_pairs']} | " +
                     " / ".join(fmt(m[k]) for k in ("min", "median", "p95", "max")) +
                     f" | {fmt(e['median'])} [{fmt(e['min'])},{fmt(e['max'])}] |")
    return "\n".join(lines) + "\n"


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: str(v).lower() if type(v) is bool else v for k, v in row.items()})


def write_outputs(core, external, flags, report, root=ROOT):
    out = Path(root).absolute() / OUT_REL
    require(not any(p.is_symlink() for p in (out, *out.parents)), "Output path contains a symlink")
    names = ("development_table.csv", "external_diagnostic_table.csv", "row_metadata.csv", "summary.json", "summary.md")
    require(not any((out / n).is_symlink() for n in names), "Output file is a symlink")
    out.mkdir(parents=True, exist_ok=True)
    for filename, columns, rows in ((names[0], COLUMNS, core), (names[1], COLUMNS, external),
                                    (names[2], FLAG_COLUMNS, flags)):
        write_csv(out / filename, columns, rows)
    report["artifact_sha256"] = {n: hashlib.sha256((out / n).read_bytes()).hexdigest() for n in names[:3]}
    (out / "summary.md").write_text(markdown(report), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return out


def load_development_table(directory=ROOT / OUT_REL, *, allow_external_diagnostic=False):
    """Canonical reader. No old logs, token accounting, delta or match recomputation.

    External diagnostics are only included after explicit caller opt-in. This
    prevents accidental mixing through a default loader or wildcard file read.
    """
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text())
    membership = report["membership"]
    assert_membership(*(membership[k] for k in ("core", "external", "archived")))
    require(report["columns"] == list(COLUMNS), "Canonical schema changed")
    names = ["development_table.csv"]
    if allow_external_diagnostic:
        names.append("external_diagnostic_table.csv")
    flag_path = directory / "row_metadata.csv"
    require(hashlib.sha256(flag_path.read_bytes()).hexdigest()
            == report["artifact_sha256"]["row_metadata.csv"], "Canonical row metadata hash mismatch")
    with flag_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(FLAG_COLUMNS), "Row metadata columns changed")
        flags = list(reader)
    flags_by_key = {row_key(f): f for f in flags}
    require(len(flags_by_key) == len(flags), "Duplicate row metadata keys")
    rows = []
    for name in names:
        raw = (directory / name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == report["artifact_sha256"][name], "Canonical table hash mismatch")
        with (directory / name).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames == list(COLUMNS), "Measurement columns changed")
            part = [{k: (None if v == "" else int(v) if k in INTEGER_COLUMNS else
                         float(v) if k in ("loss", "initial_loss", "delta") else v)
                     for k, v in r.items()} for r in reader]
        core = name == "development_table.csv"
        allowed = set(membership["core" if core else "external"])
        require({r["run_id"] for r in part} == allowed, "Canonical cohort membership changed")
        require(all(r["protocol_id"].startswith("core:" if core else "external-diagnostic:") for r in part),
                "Canonical protocol cohort changed")
        require(not any(r["run_id"].endswith(ARCHIVE_SUFFIXES) for r in part), "Archived canonical row")
        for r in part:
            flag = flags_by_key.get(row_key(r))
            require(flag is not None and flag["core"] == str(core).lower(), "Canonical core flag mismatch")
        rows.extend(part)
    require(len(rows) == len({row_key(r) for r in rows}), "Duplicate canonical keys")
    return rows


def load_intervention_pairs(directory=ROOT / OUT_REL, *, kind=None, tolerance=None):
    """Read the persisted A1 inventory; never rerun the matcher in later stages."""
    directory = Path(directory)
    keys = {row_key(r) for r in load_development_table(directory)}
    report = json.loads((directory / "summary.json").read_text())
    pairs = report["pairs"]
    require(all(tuple(p[endpoint]) in keys for p in pairs for endpoint in ("first", "second")),
            "Stored pair references a non-core or missing endpoint")
    return [p for p in pairs if (kind is None or p["kind"] == kind)
            and (tolerance is None or p["tolerance"] == tolerance)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-tables", action="store_true", help="Accepted for consistency; the report is always printed")
    parser.parse_args()
    core, external, flags, report = build()
    out = write_outputs(core, external, flags, report)
    print(markdown(report))
    print(f"Artifacts: {out}")


if __name__ == "__main__":
    main()
