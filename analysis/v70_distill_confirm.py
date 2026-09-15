#!/usr/bin/env python3
"""V70 CPU registration, development, write-once freeze and paired confirmation.

  python -B analysis/v70_distill_confirm.py register
  python -B analysis/v70_distill_confirm.py develop
  python -B analysis/v70_distill_confirm.py freeze
  scripts/run_v70_confirm.sh --dry-run
  python -B analysis/v70_distill_confirm.py compare

No mode in this module trains or loads model weights. `plan` is the shell
launcher's CPU preflight. `--dry-run` writes nothing; `--selftest` uses temporary
synthetic data. The 25 old uniform trajectories are now development data.
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
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import sys
import tempfile

sys.dont_write_bytecode = True
# This module is always CPU-only. A child process cannot change the shell's GPU selection.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
try:
    from . import v56_distill_forms as v56
except ImportError:
    import v56_distill_forms as v56

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / "results/v47-p2-register/register.json").is_file()),
            Path(__file__).resolve().parents[1])
REG_REL = Path("results/v47-p2-register/register.json")
OUT_REL = Path("results/v70-distill-confirm")
CAPS = ("math", "code", "qa")
STUDENTS = ("gemma3-270m", "gemma3-1b")
ALL_STUDENTS = (*STUDENTS, "gemma3-4b")
SEEDS = tuple(range(31, 37))
TARGETS = (50000, 100000, 200000)
OLD_TRIGGERS = (119000, 237000, 475000, 949000)
SCHEDULE = 1000000
T_REF = 35000.
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
CANDIDATES = tuple(k + s for k in ("constant", "T", "E", "joint") for s in ("", "+src")) + (
    "F1:L0", "F1:logN", "F2:L0", "F2:logN")
BASELINES = ("constant", "T-only", "E-only", "surface:L0", "surface:logN")
METHODS = tuple(dict.fromkeys((*CANDIDATES, *BASELINES)))
CODE_INPUTS = ("analysis/v70_distill_confirm.py", "analysis/v12_distill.py",
               "analysis/v50_p2v2.py", "analysis/v56_distill_forms.py",
               "analysis/v47_p2_register.py", "analysis/v6_capability_geometry.py",
               "analysis/model_registry.py", "scripts/run_v70_confirm.sh")
SELECTION_RULE = (
    "Per capability, lowest leave-one-trajectory-out (LOCO) MAE, equal weight per "
    "trajectory and per checkpoint within trajectory. All 12 candidates within "
    "0.02 nats (inclusive) of the minimum tie: choose fewer fitted parameters, "
    "then lower MAE, then declared candidate order. F2 counts four coefficients "
    "plus one fitted discrete T_star. Refit coefficients, descriptor references, "
    "column scaling and F2 T_star inside each training fold; T_star minimizes "
    "training SSE per capability on V56's fixed five-value grid."
)
BASELINE_RULE = (
    "Per student and capability, lowest development LOCO MAE among constant, "
    "intercept+T-only, intercept+E-only, and same-input surfaces with L0/logN. "
    "Exact ties: fewer parameters, then declared baseline order. Frozen before "
    "confirmation; no baseline selection or refitting on confirmation errors."
)
REF_RULE = (
    "T_ref=35000; D_ref=median training-point D_U; N_ref=geometric mean of unique "
    "training-student total meta-device parameter counts (V50 convention, not "
    "training-manifest or nominal sizes). L0 and logN use population mean/std "
    "with equal weight per unique training student. Recompute within each fold."
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def object_sha(obj):
    return sha(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def label(root, path):
    path = Path(path).absolute()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


class Inputs:
    def __init__(self, root):
        self.root, self.hashes = root, {}

    def read(self, path):
        path = Path(path)
        raw = path.read_bytes()
        key, digest = label(self.root, path), sha(raw)
        require(key not in self.hashes or self.hashes[key] == digest, f"Input changed while reading: {key}")
        self.hashes[key] = digest
        return raw

    def json(self, path):
        return json.loads(self.read(path))

    def verify(self):
        verify_hashes(self.root, self.hashes)


def verify_hashes(root, hashes):
    for key, digest in hashes.items():
        require(sha((root / key).read_bytes()) == digest, f"Frozen input changed: {key}")


def write(path, obj, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = obj if isinstance(obj, str) else json.dumps(obj, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)  # atomic write-once publication, including races
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def no_confirmation_started(root):
    for st in STUDENTS:
        matches = list((root / "results/v12-distill" / st).glob("gpt-5.6-luna_full_200_p2v3conf*"))
        require(not matches, f"Confirmation/throughput already started: {matches}")


def assert_unused_u(reg):
    """Walk every registered section, not just the original top-level pools."""
    def visit(obj, path="register"):
        if isinstance(obj, dict):
            if "U" in obj:
                require(int(obj["U"]) != 200, f"U=200 previously registered at {path}")
            for key, value in obj.items():
                if key != "v5_confirm":
                    visit(value, path + "/" + key)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                visit(value, path + f"/{i}")
    visit(reg)


def pool_milestones(pool):
    proc, comp = pool["pool_processed_tokens"], pool["D_U_completion"]
    require(proc > 0 and 0 < comp <= proc, "Invalid pool token counts")
    result = [{"planned_T_completion": t, "trigger_processed": (t * proc + comp - 1) // comp,
               "E_completion_convention": t / comp} for t in TARGETS]
    require(result[-1]["trigger_processed"] < SCHEDULE, "Confirmation exceeds 1M processed-token schedule")
    return result


def register_pools(root, dry_run=False):
    inputs = Inputs(root)
    reg = inputs.json(root / REG_REL)
    assert_unused_u(reg)  # before importing a tokenizer or sampling any new pool
    if "v5_confirm" in reg:
        section = reg["v5_confirm"]
        require(set(section["pools"]) == {f"U200_s{s}" for s in SEEDS}, "Existing v5_confirm pool set differs")
        verify_hashes(root, section["inputs_sha256"])
        for p in section["pools"].values():
            require(p["planned_milestones"] == pool_milestones(p), "Existing trigger rule differs")
        print("v5_confirm already registered; source hashes and triggers verified")
        return section
    no_confirmation_started(root)
    if dry_run:
        print("Would register six U=200 pools, seeds 31..36, using V12 sampling and actual shared-tokenizer completion counts")
        return None
    try:
        from . import v12_distill as v12
    except ImportError:
        import v12_distill as v12
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file
    source = Inputs(root)
    for rel in ("analysis/v12_distill.py", "analysis/v47_p2_register.py"):
        source.read(root / rel)
    for domain in ("math", "qa", "code"):
        source.read(root / f"results/traces-pilot/gpt-5.6-luna_{domain}.jsonl")
    tokenizers, snapshots = {}, {}
    for st in STUDENTS:
        revision = reg["student_snapshot"] if st == "gemma3-1b" else None
        kwargs = {"revision": revision, "local_files_only": True}
        config_path = cached_file(v56.HF[st], "tokenizer_config.json", **kwargs)
        # A local directory also prevents optional remote chat-template discovery.
        tokenizers[st] = AutoTokenizer.from_pretrained(str(Path(config_path).parent), local_files_only=True)
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            path = cached_file(v56.HF[st], name, **kwargs, _raise_exceptions_for_missing_entries=False)
            if path:
                source.read(path)
                snapshots[st] = Path(path).parent.name
    pools, ids_by_pool = {}, {}
    for seed in SEEDS:
        # seed=0 exactly matches V47. V12's launch shuffles by data_seed; order
        # does not affect the sampled multiset, counts or sorted identity hashes.
        records, counts = v12.load_sft_records("gpt-5.6-luna", ("math", "qa", "code"),
            200, "full", trace_base=root / "results/traces-pilot", seed=0, data_seed=seed)
        results = []
        for st, tokenizer in tokenizers.items():
            ids, lengths, proc, comp, domains = set(), [], 0, 0, {}
            for record in records:
                ex = v12.tokenize_sft_example(tokenizer, record["prompt"], record["completion"], max_len=v12.MAX_LEN)
                nc, ni = int(ex["n_completion_tokens"]), int(ex["input_ids"].numel())
                if not nc:
                    continue
                identity = sha(json.dumps([record["prompt"], record["completion"]], ensure_ascii=False).encode())
                ids.add(identity)
                lengths.append((identity, ni))
                proc += ni
                comp += nc
                domains[record["domain"]] = domains.get(record["domain"], 0) + 1
            results.append({"U": 200, "data_seed": seed, "role": "confirmation",
                "n_examples": len(ids), "n_training_examples": len(lengths),
                "pool_processed_tokens": proc, "D_U_completion": comp,
                "completion_processed_ratio": comp / proc, "domain_counts": domains,
                "id_set_sha256": sha("".join(sorted(ids)).encode()),
                "data_pool_sha256": sha(json.dumps(sorted(lengths)).encode()),
                "source_counts": counts})
        require(results[0] == results[1], f"Student tokenizers disagree for data seed {seed}")
        p = results[0]
        p["planned_milestones"] = pool_milestones(p)
        p["schedule_updates"] = math.ceil(SCHEDULE / (p["pool_processed_tokens"] / math.ceil(p["n_training_examples"] / 16)))
        p["planned_epochs"] = math.ceil(p["schedule_updates"] / math.ceil(p["n_training_examples"] / 16))
        key = f"U200_s{seed}"
        pools[key], ids_by_pool[key] = p, ids
        print(f"{key}: D_U={comp}, processed={proc}, triggers={[m['trigger_processed'] for m in p['planned_milestones']]}")
    require(len({p["id_set_sha256"] for p in pools.values()}) == 6, "New pools are not six distinct samples")
    source.verify()
    section = {"registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "unused_U_assertion": "U=200 absent from every pre-existing registered pool section",
        "sampling_rule": "V12 load_sft_records: random.Random(f'{data_seed}-{domain}').sample(eligible JSONL rows, U); full recipe; drop empty completions; V12 max_len=1024 tokenization",
        "trigger_rule": "ceil(T_completion * pool_processed_tokens / D_U_completion); completed-update crossings may overshoot in processed and completion tokens",
        "students": list(STUDENTS), "tokenizer_snapshots": snapshots, "shared_pool_verified": True,
        "protocol": {"schedule_tokens": SCHEDULE, "epochs_argument": 99, "training_seed": 0,
            "lora": {"r": 16, "alpha": 32, "dropout": 0., "targets": list(TARGET_MODULES)},
            "lr": 1e-4, "optimizer": "AdamW", "scheduler": "cosine", "warmup_ratio": .03,
            "effective_batch_size": 16, "max_len": 1024, "suffix": "p2v3conf"},
        "pools": pools, "overlap_jaccard": {a + "~" + b: len(ids_by_pool[a] & ids_by_pool[b]) / len(ids_by_pool[a] | ids_by_pool[b])
            for a, b in itertools.combinations(pools, 2)}, "inputs_sha256": source.hashes}
    inputs.verify()
    reg["v5_confirm"] = section
    write(root / REG_REL, reg)
    return section


def dev_specs():
    return ([(st, u, s, "p2v2") for st in STUDENTS for u in (75, 450) for s in (11, 12, 13)]
        + [(st, 375, s, "p2v2test") for st in ALL_STUDENTS for s in (21, 22, 23)]
        + [("gemma3-4b", u, s, "p2v2test") for u in (75, 450) for s in (11, 12)])


def run_dir(root, st, u, seed, suffix):
    return root / f"results/v12-distill/{st}/gpt-5.6-luna_full_{u}_{suffix}_lora_dseed{seed}"


def check_protocol(j, st, u, seed, suffix, triggers):
    expected = {"student": st, "teacher": "gpt-5.6-luna", "recipe": "full",
                "n_per_domain": u, "data_seed": seed, "training_seed": 0,
                "data_sampling_seed": seed, "output_suffix": "_" + suffix,
                "training_mode": "lora", "schedule_tokens": SCHEDULE,
                "learning_rate": 1e-4, "optimizer": "AdamW", "scheduler": "cosine",
                "warmup_ratio": .03, "trajectory_tokens_requested": list(triggers)}
    for key, value in expected.items():
        require(j.get(key) == value, f"Protocol mismatch {st}/U{u}/s{seed}: {key}")
    targets = {s.rsplit(".", 1)[-1] for s in j["training_manifest"]["target_modules"]}
    require(targets == set(TARGET_MODULES), "LoRA must cover all seven projections")


def read_trajectory(root, inputs, st, u, seed, suffix, pool, triggers):
    directory = run_dir(root, st, u, seed, suffix)
    final = inputs.json(directory / "eval.json")
    check_protocol(final, st, u, seed, suffix, triggers)
    log = inputs.json(directory / "train_log.json")
    require(log["status"] == "trained" and not log["trajectory_tokens_unreached"], f"Incomplete run: {directory}")
    require(log["lora"] == {"r": 16, "alpha": 32, "dropout": 0., "target_modules": list(TARGET_MODULES)}, "LoRA configuration changed")
    require(log["effective_batch_size_sequences"] == 16 and log["max_len"] == 1024, "Batch/truncation protocol changed")
    require(log["warmup_steps"] == int(.03 * log["total_updates_planned"]), "Warmup horizon changed")
    require(log["updates"] == log["total_updates_planned"], "Trajectory did not finish its fixed schedule")
    if "data_pool_sha256" in pool:
        require(final["data_pool_sha256"] == pool["data_pool_sha256"], "Confirmation pool sample changed")
        require(log["total_updates_planned"] == pool["schedule_updates"], "Schedule horizon changed")
    points = []
    for path in sorted(directory.glob("trajectory/update-*/eval.json")):
        j = inputs.json(path)
        if j.get("processed_tokens", 0) <= 0:
            continue
        check_protocol(j, st, u, seed, suffix, triggers)
        require(j["data_pool_sha256"] == final["data_pool_sha256"], "Pool changed within trajectory")
        require(j["dense"] == final["dense"], "Dense losses changed within trajectory")
        require(len(j["requested_token_milestones"]) == 1, "Multiple budgets collapsed into one update")
        trigger = j["requested_token_milestones"][0]
        require(trigger in triggers and j["processed_tokens"] >= trigger, "Invalid checkpoint trigger")
        require(j["milestone_overshoot_tokens"] == [j["processed_tokens"] - trigger], "Bad overshoot metadata")
        for c in CAPS:
            require(np.isfinite(j["delta"][c]) and np.isfinite(j["dense"][c]), "Nonfinite loss")
            require(np.isclose(j["delta"][c], j["post_training"][c] - j["dense"][c], atol=1e-9, rtol=0), "delta must be post minus dense")
        t, du = j["completion_tokens_seen"], pool["D_U_completion"]
        require(t > 0 and du > 0, "Nonpositive exposure")
        points.append({"student": st, "U": u, "seed": seed, "pool": f"U{u}_s{seed}",
            "suffix": suffix, "cluster": f"{st}|U{u}_s{seed}", "Tc": t, "DU": du, "E": t / du,
            "L0": j["dense"], "delta": j["delta"], "processed": j["processed_tokens"],
            "trigger_processed": trigger, "file": label(root, path), "data_pool_sha256": j["data_pool_sha256"]})
    points.sort(key=lambda p: p["Tc"])
    require([p["trigger_processed"] for p in points] == list(triggers), f"Missing/duplicate checkpoints: {directory}")
    return points


def load_dev(root, inputs):
    reg = inputs.json(root / REG_REL)
    assert_unused_u(reg)
    # Cached V50 meta-device counts reproduce the required convention without weights/imports.
    counts = inputs.json(root / "results/v50-p2v2/freeze.json")["refs"]["N"]
    require(set(counts) == set(ALL_STUDENTS) and all(n > 0 for n in counts.values()), "Invalid V50 counts")
    expected = {run_dir(root, *spec) for spec in dev_specs()}
    actual = {p for st in ALL_STUDENTS for suffix in ("p2v2", "p2v2test")
              for p in (root / "results/v12-distill" / st).glob(f"gpt-5.6-luna_full_*_{suffix}_lora_dseed*")
              if "_seed" not in p.name}
    require(actual == expected, f"Uniform development run set changed: missing={expected - actual}, extra={actual - expected}")
    pts = [p for st, u, s, suffix in dev_specs()
           for p in read_trajectory(root, inputs, st, u, s, suffix, reg["pools"][f"U{u}_s{s}"], OLD_TRIGGERS)]
    require(len(pts) == 100 and len({p["cluster"] for p in pts}) == 25, "Need 25 trajectories / 100 checkpoints")
    return pts, counts


def references(points, counts):
    students = sorted({p["student"] for p in points})
    dense = {}
    for st in students:
        rows = [p for p in points if p["student"] == st]
        dense[st] = rows[0]["L0"]
        require(all(p["L0"] == dense[st] for p in rows), f"Student dense loss changed: {st}")
    logn = [math.log(counts[s]) for s in students]
    return {"T_ref": T_REF, "D_ref": float(np.median([p["DU"] for p in points])),
        "N": counts, "N_ref": math.exp(float(np.mean(logn))), "L0_by_student": dense,
        "reference_students": students,
        "logN": {"mean": float(np.mean(logn)), "std": float(np.std(logn))},
        "L0": {c: {"mean": float(np.mean([dense[s][c] for s in students])),
                    "std": float(np.std([dense[s][c] for s in students]))} for c in CAPS}}


def basis(method, p, cap, refs, t_star=None):
    if method in CANDIDATES[:8]:
        u, w = math.log1p(p["Tc"] / refs["T_ref"]), math.log1p(p["E"])
        v, n = math.log(p["DU"] / refs["D_ref"]), math.log(refs["N"][p["student"]] / refs["N_ref"])
        kind = method.split("+")[0]
        cols = {"constant": [1.], "T": [u], "E": [w], "joint": [u, u*u, u*v]}[kind]
        return cols + ([n if kind == "constant" else n * (w if kind == "E" else u)] if "+src" in method else [])
    kind, _, descriptor = method.partition(":")
    return v56.basis(kind, descriptor or None, p, cap, refs, t_star)


def fit(points, cap, method, refs):
    y = np.asarray([p["delta"][cap] for p in points])
    trials = []
    for ts in v56.T_GRID if method.startswith("F2:") else (None,):
        x = np.asarray([basis(method, p, cap, refs, ts) for p in points])
        if method in CANDIDATES[:8]:
            b, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
            fitted = {"coef": b.tolist(), "rank": int(rank), "n_params": x.shape[1],
                      "sse": float(np.sum((x @ b - y)**2)), "estimator": "V50 OLS"}
        else:
            fitted = v56.ridge_fit(x, y, [f"b{i}" for i in range(x.shape[1])],
                                  not method.startswith(("F1:", "F2:")))
            fitted["estimator"] = "V56 standardized ridge, lambda=0.001"
        trials.append({**fitted, "T_star": ts})
    best = min(trials, key=lambda f: f["sse"])
    return {**best, "method": method, "selection_n_params": best["n_params"] + int(method.startswith("F2:")),
            "T_star_grid_sse": [{"T_star": f["T_star"], "sse": f["sse"]} for f in trials]}


def predict(model, point, cap, refs):
    return float(np.dot(basis(model["method"], point, cap, refs, model["T_star"]), model["coef"]))


def choose(scores, models, candidates, tolerance=.02):
    minimum = min(scores[m] for m in candidates)
    eligible = [m for m in candidates if scores[m] <= minimum + tolerance + 1e-12]
    winner = min(eligible, key=lambda m: (models[m]["selection_n_params"], scores[m], candidates.index(m)))
    return {"method": winner, "loco_mae": scores[winner], "minimum_mae": minimum,
            "tied_methods": eligible, "n_params": models[winner]["selection_n_params"]}


def loco(points, counts):
    folds, rows = [], []
    for cluster in sorted({p["cluster"] for p in points}):
        train, held = [p for p in points if p["cluster"] != cluster], [p for p in points if p["cluster"] == cluster]
        refs = references(train, counts)
        models = {c: {m: fit(train, c, m, refs) for m in METHODS} for c in CAPS}
        folds.append({"held_out": cluster, "train_clusters": sorted({p["cluster"] for p in train}),
                      "n_train_points": len(train), "refs": refs, "models": models})
        for p in held:
            for c in CAPS:
                rows.append({"cluster": cluster, "student": p["student"], "capability": c,
                    "Tc": p["Tc"], "actual": p["delta"][c],
                    "absolute_errors": {m: abs(predict(models[c][m], p, c, refs) - p["delta"][c]) for m in METHODS}})
    def scores(rr):
        clusters = sorted({r["cluster"] for r in rr})
        return {m: float(np.mean([np.mean([r["absolute_errors"][m] for r in rr if r["cluster"] == k]) for k in clusters])) for m in METHODS}
    return {"folds": folds, "rows": rows,
        "scores": {c: scores([r for r in rows if r["capability"] == c]) for c in CAPS},
        "scores_by_student": {st: {c: scores([r for r in rows if r["capability"] == c and r["student"] == st])
                                     for c in CAPS} for st in STUDENTS}}


def develop(root, out, dry_run=False):
    require(not (out / "freeze.json").exists(), "Refusing to redevelop after freeze.json")
    no_confirmation_started(root)
    inputs = Inputs(root)
    points, counts = load_dev(root, inputs)
    for rel in CODE_INPUTS:
        inputs.read(root / rel)
    refs = references(points, counts)
    cv = loco(points, counts)
    models = {c: {m: fit(points, c, m, refs) for m in METHODS} for c in CAPS}
    selected = {c: choose(cv["scores"][c], models[c], CANDIDATES) for c in CAPS}
    baseline = {st: {c: choose(cv["scores_by_student"][st][c], models[c], BASELINES, tolerance=0.)
                     for c in CAPS} for st in STUDENTS}
    result = {"schema_version": 1, "selection_rule": SELECTION_RULE, "baseline_rule": BASELINE_RULE,
        "reference_rule": REF_RULE, "candidates": list(CANDIDATES), "baselines": list(BASELINES),
        "development_structure": {"old_dev": 12, "old_test": 9, "held_out_4b": 4,
            "n_trajectories": 25, "checkpoints_per_trajectory": 4, "n_points": 100,
            "split_note": "All formerly observed uniform-protocol dev/test responses are development data for V70; seed-repeat and other protocols excluded"},
        "refs": refs, "points": points, "models": models, "loco": cv,
        "selected": selected, "strongest_baseline": baseline, "inputs_sha256": inputs.hashes}
    lines = ["# V70 development", "", "25 trajectories = 12 old dev + 9 old test + 4 held-out-4B; four checkpoints each (100 points).",
             "", SELECTION_RULE, "", BASELINE_RULE, "", REF_RULE, "",
             "| Capability | Selected | LOCO MAE | Parameters |", "|---|---|---:|---:|"]
    for c, s in selected.items():
        lines.append(f"| {c} | {s['method']} | {s['loco_mae']:.6f} | {s['n_params']} |")
    lines += ["", "| Method | Math LOCO MAE | Code LOCO MAE | QA LOCO MAE |", "|---|---:|---:|---:|"]
    for m in METHODS:
        lines.append("| " + m + " | " + " | ".join(f"{cv['scores'][c][m]:.6f}" for c in CAPS) + " |")
    lines += ["", "Frozen comparison baselines (development LOCO within student):", ""]
    for st in STUDENTS:
        lines.append(f"- {st}: " + "; ".join(f"{c}: {baseline[st][c]['method']}" for c in CAPS))
    inputs.verify()
    if not dry_run:
        write(out / "develop.json", result)
        write(out / "develop.md", "\n".join(lines) + "\n")
    print(f"{'DRY RUN ' if dry_run else ''}develop: 25 trajectories / 100 points; " + str({c: s['method'] for c, s in selected.items()}))
    return result


def freeze(root, out, dry_run=False):
    require(not (out / "freeze.json").exists(), "Refusing to overwrite freeze.json")
    require(not (out / "FREEZE_V70").exists(), "FREEZE_V70 already exists without a usable new freeze")
    no_confirmation_started(root)
    inputs = Inputs(root)
    dev = inputs.json(out / "develop.json")
    verify_hashes(root, dev["inputs_sha256"])
    require(dev["selection_rule"] == SELECTION_RULE and dev["baseline_rule"] == BASELINE_RULE, "Selection rules changed")
    reg = inputs.json(root / REG_REL)
    assert_unused_u(reg)
    require("v5_confirm" in reg, "Run register before freeze (and before develop)")
    section = reg["v5_confirm"]
    verify_hashes(root, section["inputs_sha256"])
    require(set(section["pools"]) == {f"U200_s{s}" for s in SEEDS}, "Need all six registered confirmation pools")
    predictions = []
    for st in STUDENTS:
        for seed in SEEDS:
            pool = section["pools"][f"U200_s{seed}"]
            require(pool["planned_milestones"] == pool_milestones(pool), "Registered triggers changed")
            for milestone in pool["planned_milestones"]:
                t, du = milestone["planned_T_completion"], pool["D_U_completion"]
                p = {"student": st, "Tc": t, "DU": du, "E": t / du, "L0": dev["refs"]["L0_by_student"][st]}
                for c in CAPS:
                    predictions.append({"student": st, "pool": f"U200_s{seed}", "data_seed": seed,
                        "T_planned": t, "trigger_processed": milestone["trigger_processed"],
                        "DU": du, "E_planned": t / du, "capability": c,
                        "selected": dev["selected"][c]["method"],
                        "strongest_baseline": dev["strongest_baseline"][st][c]["method"],
                        "predictions": {m: predict(dev["models"][c][m], p, c, dev["refs"]) for m in METHODS}})
    result = {"schema_version": 1, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": SELECTION_RULE, "baseline_rule": BASELINE_RULE, "reference_rule": REF_RULE,
        "selected": dev["selected"], "strongest_baseline": dev["strongest_baseline"],
        "models": dev["models"], "refs": dev["refs"], "confirmation_register": section,
        "prediction_rule": "Primary errors use the stored planned-T predictions; report actual completion exposure and overshoot without replacing predictions. No confirmation fitting/selection.",
        "bootstrap": {"n_resamples": 5000, "seed": 0, "rng": "NumPy PCG64",
            "unit": "pool sample; six pools per student, three budgets kept together",
            "pairing": "same resampled pool indices across all methods, capabilities and both students",
            "weighting": "mean over three budgets within each pool, then equal mean over six pools",
            "interval": "95% percentile", "sign": "baseline absolute error minus selected absolute error; positive favors selected"},
        "predictions": predictions, "n_predictions": len(predictions),
        "inputs_sha256": {**dev["inputs_sha256"], **section["inputs_sha256"], **inputs.hashes}}
    require(len(predictions) == 108, "Expected 2 students x 6 pools x 3 budgets x 3 capabilities")
    verify_hashes(root, result["inputs_sha256"])
    if not dry_run:
        write(out / "freeze.json", result, exclusive=True)
        write(out / "FREEZE_V70", sha((out / "freeze.json").read_bytes()) + "\n", exclusive=True)
    print(f"{'DRY RUN ' if dry_run else ''}freeze: 108 capability predictions; write-once freeze.json + SHA256 sentinel FREEZE_V70")
    return result


def load_freeze(root, out):
    path, sentinel = out / "freeze.json", out / "FREEZE_V70"
    require(path.is_file() and sentinel.is_file(), f"freeze.json and FREEZE_V70 required: {out}")
    raw = path.read_bytes()
    digest = sha(raw)
    require(sentinel.read_text().strip() == digest, "FREEZE_V70 does not match freeze.json SHA256")
    frozen = json.loads(raw)
    require(frozen["selection_rule"] == SELECTION_RULE and frozen["baseline_rule"] == BASELINE_RULE, "Frozen rules differ from this implementation")
    verify_hashes(root, frozen["inputs_sha256"])
    reg = json.loads((root / REG_REL).read_bytes())
    require(reg["v5_confirm"] == frozen["confirmation_register"], "Registered confirmation inputs changed")
    return frozen, digest


def plan(root, out, throughput=False, dry_run=False, student=None):
    if dry_run:
        reg = json.loads((root / REG_REL).read_bytes())
        assert_unused_u(reg)
        require("v5_confirm" in reg, "Run register to compute actual pool token triggers before planning")
        section = reg["v5_confirm"]
    else:
        frozen, _ = load_freeze(root, out)
        section = frozen["confirmation_register"]
    specs = [("gemma3-1b", SEEDS[0])] if throughput else [(st, s) for st in STUDENTS for s in SEEDS if student in (None, st)]
    for st, seed in specs:
        p = section["pools"][f"U200_s{seed}"]
        suffix = "p2v3conf_throughput" if throughput else "p2v3conf"
        directory = run_dir(root, st, 200, seed, suffix)
        milestones = p["planned_milestones"][:1] if throughput else p["planned_milestones"]
        triggers = [m["trigger_processed"] for m in milestones]
        command = ["python3", "-B", "analysis/v12_distill.py", "--student", st,
            "--teacher", "gpt-5.6-luna", "--recipe", "full", "--training-mode", "lora",
            "--n-per-domain", "200", "--data-seed", str(seed), "--seed", "0", "--epochs", "99",
            "--lr", "1e-4", "--schedule-tokens", str(SCHEDULE), "--save-trajectory",
            "--trajectory-tokens", *map(str, triggers), "--output-suffix", "_" + suffix, "--device", "cuda:0"]
        if throughput:
            command.append("--stop-after-trajectory")
        status = "pending"
        if (directory / "eval.json").is_file():
            final = json.loads((directory / "eval.json").read_bytes())
            check_protocol(final, st, 200, seed, suffix, triggers)
            require(final["data_pool_sha256"] == p["data_pool_sha256"], f"Existing output has different pool: {directory}")
            log = json.loads((directory / "train_log.json").read_bytes())
            require(not final["trajectory_tokens_unreached"] and log["status"] == "trained", "Incomplete existing output")
            require(log["total_updates_planned"] == p["schedule_updates"], "Existing output schedule mismatch")
            require(bool(log.get("stopped_after_trajectory")) == throughput, "Existing output stop policy differs")
            if not throughput:
                read_trajectory(root, Inputs(root), st, 200, seed, suffix, p, triggers)
            status = "skip"
        elif directory.exists():
            require(dry_run, f"Partial trajectory exists; V12 cannot resume optimizer state: {directory}. Preserve it and use a fresh directory before relaunching.")
            status = "blocked-partial"
        yield {"student": st, "seed": seed, "status": status, "command": command,
               "directory": str(directory), "log": str(root / f"logs/v70_{suffix}_{st}_U200_s{seed}.log"),
               "throughput": throughput}


def paired_summary(rows):
    draws = np.random.default_rng(0).integers(0, 6, size=(5000, 6))
    groups = []
    for st in STUDENTS:
        for c in CAPS:
            rr = [r for r in rows if r["student"] == st and r["capability"] == c]
            require(len(rr) == 18, f"Need 18 checkpoints / 6 pools for {st}/{c}")
            selected = {r["selected"] for r in rr}
            baseline = {r["strongest_baseline"] for r in rr}
            require(len(selected) == len(baseline) == 1, "Selections must be frozen across confirmation budgets")
            selected, baseline = selected.pop(), baseline.pop()
            clusters = []
            for seed in SEEDS:
                pr = sorted([r for r in rr if r["data_seed"] == seed], key=lambda r: r["T_planned"])
                require([r["T_planned"] for r in pr] == list(TARGETS), f"Need all three budgets in pool {seed}")
                clusters.append({"pool": f"U200_s{seed}", "n_checkpoints": 3,
                    "T_planned": list(TARGETS), "T_actual": [r["T_actual"] for r in pr],
                    "mae": {m: float(np.mean([r["absolute_errors"][m] for r in pr])) for m in METHODS}})
            errors = np.asarray([[p["mae"][selected], p["mae"][baseline]] for p in clusters])
            means = errors.mean(0)
            boot = errors[draws].mean(1)
            diff = boot[:, 1] - boot[:, 0]
            relative = None
            if means[1] > 0 and np.all(boot[:, 1] > 0):
                relative = {"estimate": float((means[1] - means[0]) / means[1]),
                            "ci95": np.quantile(diff / boot[:, 1], [.025, .975]).tolist()}
            groups.append({"student": st, "capability": c, "selected": selected,
                "strongest_baseline": baseline, "n_pools": 6, "n_checkpoints": 18, "clusters": clusters,
                "candidate_mae": float(means[0]), "baseline_mae": float(means[1]),
                "paired_difference": {"estimate": float(means[1] - means[0]), "ci95": np.quantile(diff, [.025, .975]).tolist()},
                "relative_improvement": relative,
                "all_method_mae": {m: float(np.mean([p["mae"][m] for p in clusters])) for m in METHODS}})
    return groups


@proofread_table
def compare_text(result, latex=False):
    if latex:
        lines = [r"\begin{table}[H]", r"\centering", r"\small",
            r"\caption{V70 distillation confirmation at $U=200$ and supervised $T=50,100,200$k. "
            r"Each student/capability has six sampled pools, with three dependent checkpoints per pool "
            r"(18 points, not 18 replicates). MAE and paired improvement (baseline minus selected "
            r"absolute error) are in nats. Both forms and strongest baselines were chosen by "
            r"development LOCO before confirmation. Brackets: 95\% percentile intervals from "
            r"5,000 paired pool-cluster bootstrap resamples, seed 0; all budgets stay together.}",
            r"\label{tab:distill_confirm}", r"\begin{tabular}{@{}ll ll rrr@{}}", r"\toprule",
            r"Student & Cap. & Selected & Baseline & MAE & Base MAE & Improvement [95\% CI] \\", r"\midrule"]
        for g in result["groups"]:
            d = g["paired_difference"]
            lo, hi = d["ci95"]
            fields = [g["student"].replace("gemma3-", ""), g["capability"], g["selected"], g["strongest_baseline"],
                      f"{g['candidate_mae']:.4f}", f"{g['baseline_mae']:.4f}", f"{d['estimate']:.4f} [{lo:.4f}, {hi:.4f}]"]
            lines.append(" & ".join(fields) + r" \\")
        return "\n".join([*lines, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    lines = ["# V70 distillation confirmation", "", "Each student × capability: **6 pools × 3 dependent budgets = 18 points**. "
        "The same six pool samples (data seeds 31..36) are reused across students. Training seed is fixed at 0.", "",
        "Paired difference is baseline absolute error minus selected absolute error, so positive values favor the selected form. "
        "Intervals use 5,000 pool-cluster bootstrap resamples, seed 0, keeping all three budgets together and sharing draws across students/capabilities. "
        "They describe resampling of these six observed pools; they do not estimate training-seed variability. Pools can overlap in source examples (Jaccard values are registered).", "",
        "Predictions are the stored planned-T values. Actual token counts and overshoot are recorded in compare.json. "
        "The strongest baseline was fixed using development LOCO; no confirmation-based selection occurs.", "",
        "| Student | Cap. | Selected | Frozen baseline | MAE | Baseline MAE | Paired difference [95% CI] |",
        "|---|---|---|---|---:|---:|---:|"]
    for g in result["groups"]:
        d = g["paired_difference"]
        lo, hi = d["ci95"]
        lines.append(f"| {g['student']} | {g['capability']} | {g['selected']} | {g['strongest_baseline']} | {g['candidate_mae']:.6f} | {g['baseline_mae']:.6f} | {d['estimate']:.6f} [{lo:.6f}, {hi:.6f}] |")
    return "\n".join(lines) + "\n"


def compare(root, out, dry_run=False):
    frozen, digest = load_freeze(root, out)
    missing = [str(run_dir(root, st, 200, seed, "p2v3conf")) for st in STUDENTS for seed in SEEDS
               if not (run_dir(root, st, 200, seed, "p2v3conf") / "eval.json").is_file()]
    if dry_run and missing:
        print(f"DRY RUN compare: {len(missing)}/12 trajectories pending; will require all six pools and three budgets per student. "
              "Outputs: compare.json/md and paper/paper/tables/distill_confirm.tex [H]. No writes.")
        return {"complete": False, "missing": missing}
    inputs = Inputs(root)
    points = {}
    for st in STUDENTS:
        for seed in SEEDS:
            pool = frozen["confirmation_register"]["pools"][f"U200_s{seed}"]
            triggers = [m["trigger_processed"] for m in pool["planned_milestones"]]
            for p in read_trajectory(root, inputs, st, 200, seed, "p2v3conf", pool, triggers):
                require(p["L0"] == frozen["refs"]["L0_by_student"][st], "Confirmation dense descriptor changed")
                points[st, seed, p["trigger_processed"]] = p
    rows = []
    for row in frozen["predictions"]:
        p = points[row["student"], row["data_seed"], row["trigger_processed"]]
        actual = p["delta"][row["capability"]]
        rows.append({**row, "actual": actual, "T_actual": p["Tc"],
            "T_actual_minus_planned": p["Tc"] - row["T_planned"], "processed_actual": p["processed"],
            "processed_overshoot": p["processed"] - row["trigger_processed"], "file": p["file"],
            "signed_errors": {m: v - actual for m, v in row["predictions"].items()},
            "absolute_errors": {m: abs(v - actual) for m, v in row["predictions"].items()}})
    require(len(rows) == 108 and len(points) == 36, "Incomplete confirmation panel")
    result = {"schema_version": 1, "complete": True, "bootstrap": frozen["bootstrap"],
        "structure": "2 students sharing 6 sampled pools, 3 budgets per trajectory, 3 capabilities; 6 independent pool clusters per student/capability",
        "freeze_sha256": digest, "measurement_sha256": inputs.hashes, "rows": rows, "groups": paired_summary(rows)}
    inputs.verify()
    if not dry_run:
        write(out / "compare.json", result)
        write(out / "compare.md", compare_text(result))
        write(root / "paper/paper/tables/distill_confirm.tex", compare_text(result, latex=True))
    print(f"{'DRY RUN ' if dry_run else ''}compare: six pools x three budgets per student/capability; 5,000 paired cluster resamples")
    return result


def selftest():
    """Independent formula/CV/bootstrap tests and disposable file workflow; no training."""
    import contextlib
    import copy
    import io
    import shutil

    def fails(fn, text):
        try:
            fn()
        except (ValueError, FileExistsError, FileNotFoundError) as exc:
            assert text in str(exc), str(exc)
        else:
            raise AssertionError(f"Expected rejection: {text}")

    fails(lambda: assert_unused_u({"v2": {"pools": {"old": {"U": 200, "role": "test"}}}}), "previously registered")
    assert_unused_u({"pools": {"old": {"U": 75}}, "v5_confirm": {"pools": {"new": {"U": 200}}}})
    p = {"pool_processed_tokens": 180003, "D_U_completion": 53117}
    for m in pool_milestones(p):
        trigger, t = m["trigger_processed"], m["planned_T_completion"]
        assert (trigger - 1) * p["D_U_completion"] < t * p["pool_processed_tokens"] <= trigger * p["D_U_completion"]
    scores = {m: 10. for m in CANDIDATES}
    models = {m: {"selection_n_params": 4} for m in CANDIDATES}
    scores["joint"], scores["constant"] = .1, .12
    models["constant"]["selection_n_params"] = 1
    assert choose(scores, models, CANDIDATES)["method"] == "constant"
    scores["constant"] = .120001
    assert choose(scores, models, CANDIDATES)["method"] == "joint"

    with tempfile.TemporaryDirectory(prefix="v70-selftest-") as temporary:
        root, sink = Path(temporary), io.StringIO()
        out = root / OUT_REL
        for rel in CODE_INPUTS:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, path)
        counts = {st: n for st, n in zip(ALL_STUDENTS, (435870336, 1301875840, 4971331952))}
        write(root / "results/v50-p2v2/freeze.json", {"refs": {"N": counts}})
        pools = {f"U{u}_s{s}": {"U": u, "data_seed": s, "D_U_completion": u * 265 + s,
                 "pool_processed_tokens": u * 900 + s} for _, u, s, _ in dev_specs()}
        conf = {}
        for s in SEEDS:
            pool = {"U": 200, "data_seed": s, "D_U_completion": 53000 + s,
                    "pool_processed_tokens": 180000 + s, "schedule_updates": 210,
                    "data_pool_sha256": str(s) * 32}
            pool["planned_milestones"] = pool_milestones(pool)
            conf[f"U200_s{s}"] = pool
        section = {"pools": conf, "inputs_sha256": {}, "shared_pool_verified": True}
        write(root / REG_REL, {"pools": pools, "v5_confirm": section})

        def trajectory(st, u, s, suffix, pool, triggers):
            directory = run_dir(root, st, u, s, suffix)
            idx = ALL_STUDENTS.index(st)
            dense = {c: 4. - .7 * idx + .2 * ci for ci, c in enumerate(CAPS)}
            metadata = {"student": st, "teacher": "gpt-5.6-luna", "recipe": "full", "n_per_domain": u,
                "data_seed": s, "training_seed": 0, "data_sampling_seed": s, "output_suffix": "_" + suffix,
                "training_mode": "lora", "schedule_tokens": SCHEDULE, "learning_rate": 1e-4,
                "optimizer": "AdamW", "scheduler": "cosine", "warmup_ratio": .03,
                "trajectory_tokens_requested": list(triggers), "dense": dense,
                "training_manifest": {"target_modules": list(TARGET_MODULES)},
                "data_pool_sha256": pool.get("data_pool_sha256", "a" * 64), "trajectory_tokens_unreached": []}
            for k, trigger in enumerate(triggers):
                tc = (trigger + 19) * pool["D_U_completion"] / pool["pool_processed_tokens"]
                delta = {c: .1 * (ci+1) * math.log1p(tc / pool["D_U_completion"]) * (idx-1)
                         + .015 * math.sin(s+ci) for ci, c in enumerate(CAPS)}
                j = {**metadata, "processed_tokens": trigger + 19, "completion_tokens_seen": tc,
                     "requested_token_milestones": [trigger], "milestone_overshoot_tokens": [19], "delta": delta,
                     "post_training": {c: dense[c] + delta[c] for c in CAPS}}
                write(directory / f"trajectory/update-{k+1:08d}/eval.json", j)
            write(directory / "eval.json", j)
            write(directory / "train_log.json", {"status": "trained", "trajectory_tokens_unreached": [],
                "lora": {"r": 16, "alpha": 32, "dropout": 0., "target_modules": list(TARGET_MODULES)},
                "effective_batch_size_sequences": 16, "max_len": 1024, "warmup_steps": 6,
                "total_updates_planned": 210, "updates": 210})

        for st, u, s, suffix in dev_specs():
            trajectory(st, u, s, suffix, pools[f"U{u}_s{s}"], OLD_TRIGGERS)
        points, counts = load_dev(root, Inputs(root))
        refs = references(points, counts)
        assert refs["T_ref"] == 35000 and refs["D_ref"] == np.median([p["DU"] for p in points])
        for c in CAPS:
            for m in CANDIDATES[:8]:
                p = points[0]
                u, w = np.log1p(p["Tc"] / 35000), np.log1p(p["E"])
                v, n = np.log(p["DU"] / refs["D_ref"]), np.log(counts[p["student"]] / refs["N_ref"])
                k = m.split("+")[0]
                expected = {"constant": [1.], "T": [u], "E": [w], "joint": [u, u*u, u*v]}[k]
                if "+src" in m:
                    expected += [n * ({"constant": 1., "E": w}.get(k, u))]
                np.testing.assert_allclose(basis(m, p, c, refs), expected)
            for m in METHODS:
                model = fit(points, c, m, refs)
                x = np.asarray([basis(m, p, c, refs, model["T_star"]) for p in points])
                y = np.asarray([p["delta"][c] for p in points])
                if m in CANDIDATES[:8]:
                    independent = np.linalg.lstsq(x, y, rcond=None)[0]
                else:
                    scale, center = np.asarray(model["column_scale"]), np.asarray(model["column_mean"])
                    z = (x - center) / scale
                    penalty = np.ones(x.shape[1]); penalty[0] = 0 if model["intercept_unpenalized"] else 1
                    beta = np.linalg.solve(z.T @ z + .001 * np.diag(penalty), z.T @ y)
                    independent = beta / scale
                    if model["intercept_unpenalized"]:
                        independent[0] -= center @ independent
                np.testing.assert_allclose(model["coef"], independent, atol=1e-8)
        with contextlib.redirect_stdout(sink):
            d = develop(root, out)
            assert len(d["loco"]["folds"]) == 25
            for fold in d["loco"]["folds"]:
                assert fold["held_out"] not in fold["train_clusters"] and fold["n_train_points"] == 96
            poisoned = copy.deepcopy(points)
            held = d["loco"]["folds"][0]["held_out"]
            for p in poisoned:
                if p["cluster"] == held:
                    p["delta"]["math"] += 999
                    p["DU"] *= 20
            # The held-out trajectory cannot influence any fitted fold state.
            train = [p for p in poisoned if p["cluster"] != held]
            r = references(train, counts)
            assert r == d["loco"]["folds"][0]["refs"]
            assert fit(train, "math", "F2:L0", r) == d["loco"]["folds"][0]["models"]["math"]["F2:L0"]
            assert len(list(plan(root, out, dry_run=True))) == 12
            pilot = list(plan(root, out, throughput=True, dry_run=True))[0]
            assert pilot["student"] == "gemma3-1b" and "--stop-after-trajectory" in pilot["command"]
            assert pilot["command"][pilot["command"].index("--schedule-tokens") + 1] == "1000000"
            fails(lambda: list(plan(root, out)), "FREEZE_V70 required")
            f = freeze(root, out, dry_run=True)
            assert not (out / "freeze.json").exists() and len(f["predictions"]) == 108
            freeze(root, out)
            before = (out / "freeze.json").read_bytes()
            fails(lambda: freeze(root, out), "overwrite")
            fails(lambda: develop(root, out), "redevelop")
            assert len(list(plan(root, out))) == 12
            fails(lambda: compare(root, out), "eval.json")
            assert not (out / "compare.json").exists()
            for st in STUDENTS:
                for s in SEEDS:
                    pool = conf[f"U200_s{s}"]
                    trajectory(st, 200, s, "p2v3conf", pool, [m["trigger_processed"] for m in pool["planned_milestones"]])
            result = compare(root, out, dry_run=True)
            assert not (out / "compare.json").exists()
            result = compare(root, out)
            assert (out / "freeze.json").read_bytes() == before
            assert r"\begin{table}[H]" in (root / "paper/paper/tables/distill_confirm.tex").read_text()
            assert all(r["status"] == "skip" for r in plan(root, out))
            assert all(g["n_pools"] == 6 and g["n_checkpoints"] == 18 for g in result["groups"])
            for g in result["groups"]:
                differences = [p["mae"][g["strongest_baseline"]] - p["mae"][g["selected"]] for p in g["clusters"]]
                rng = np.random.default_rng(0)
                # Independent explicit cluster draw implementation.
                samples = [np.mean(np.asarray(differences)[rng.choice(6, 6, replace=True)]) for _ in range(5000)]
                np.testing.assert_allclose(g["paired_difference"]["ci95"], np.quantile(samples, [.025, .975]), atol=1e-14)
                np.testing.assert_allclose(g["paired_difference"]["estimate"], np.mean(differences), atol=1e-14)
            frozen_path = out / "freeze.json"
            frozen_path.write_bytes(before + b" ")
            fails(lambda: load_freeze(root, out), "SHA256")
            frozen_path.write_bytes(before)
            input_path = root / "analysis/v50_p2v2.py"
            input_path.write_bytes(input_path.read_bytes() + b"\n")
            fails(lambda: load_freeze(root, out), "Frozen input changed")
    assert "torch" not in sys.modules, "Selftest must not import models or training"
    print("PASS (CPU, no training): formulas, independent OLS/ridge, 25-fold isolation, ties, pool triggers, write-once freeze, provenance, dry-run, paired six-pool bootstrap, [H] table")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", choices=("register", "develop", "freeze", "compare", "plan"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="CPU validation without writes or training")
    parser.add_argument("--throughput", action="store_true", help="plan only: isolated 1B run to first checkpoint")
    parser.add_argument("--student", choices=STUDENTS, help="plan only: select a confirmation lane")
    parser.add_argument("--json", action="store_true", help="plan only: machine-readable launch plan")
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if not args.mode:
        parser.error("a mode or --selftest is required")
    root = args.root.resolve()
    out = args.out.resolve() if args.out else root / OUT_REL
    try:
        if args.mode == "register":
            register_pools(root, args.dry_run)
        elif args.mode == "develop":
            develop(root, out, args.dry_run)
        elif args.mode == "freeze":
            freeze(root, out, args.dry_run)
        elif args.mode == "compare":
            compare(root, out, args.dry_run)
        else:
            rows = list(plan(root, out, args.throughput, args.dry_run, args.student))
            if args.json:
                print(json.dumps(rows))
            else:
                for r in rows:
                    print(f"{r['status']}: {shlex.join(r['command'])} > {shlex.quote(r['log'])} 2>&1")
    except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
