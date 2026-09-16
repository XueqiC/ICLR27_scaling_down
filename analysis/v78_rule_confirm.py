#!/usr/bin/env python3
"""V78 locked selection rule and independent panel (authoring is CPU only).

    python -B analysis/v78_rule_confirm.py plan --dry-run
    python -B analysis/v78_rule_confirm.py --selftest
    python -B analysis/v78_rule_confirm.py freeze --stage independent
    # Download fresh snapshots into the private --snapshot-root, then:
    python -B analysis/v78_rule_confirm.py verify
    CUDA_VISIBLE_DEVICES=GPU-<uuid> python -B analysis/v78_rule_confirm.py measure --stage dense
    python -B analysis/v78_rule_confirm.py freeze --stage finalize
    CUDA_VISIBLE_DEVICES=GPU-<uuid> python -B analysis/v78_rule_confirm.py measure --stage configs
    python -B analysis/v78_rule_confirm.py compare

No downloading, training, or GPU access occurs outside explicit measure mode.
Every --dry-run is read-only and never imports the measurement libraries.
Freezes and measurements are write-once, atomically published with SHA256
sidecars. Finalization evaluates frozen fits with dense anchors; it never fits.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / "results/v64-selection-feasible/summary.json").is_file()),
            Path(__file__).resolve().parents[1])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from analysis import final_rule as rule
from analysis import v64_selection_feasible as v64
from analysis import v53_prune_dev as prune
from analysis import v55_quant_group_fit as group
from analysis import v69_quant_confirm as v69
from analysis import v36_pythia_controlled_fit as v36
from analysis import v39_distill_controlled as distill

OUT = ROOT / "results/v78-rule-confirm"
TAGS = ("pythia-160m@step32000", "pythia-410m@step32000",
        "pythia-1.4b@step32000", "pythia-1b@step64000")
STUDENTS = ("pythia-160m@step64000", "pythia-410m@step64000")
CAPS, OBJECTIVES, BUDGETS = v64.CAPS, v64.OBJECTIVES, v64.BUDGETS
POLICIES = ("locked-rule", "v64-law", "quant-only", "prune-only", "distill-only", "cheapest")
V77 = "results/v77-model-arch/weight_identity.json"
CODE = ("analysis/final_rule.py", "analysis/v78_rule_confirm.py",
        "analysis/v64_selection_feasible.py", "analysis/v53_prune_dev.py",
        "analysis/v55_quant_group_fit.py", "analysis/v69_quant_confirm.py",
        "analysis/v36_pythia_controlled_fit.py", "analysis/v38_prospective_step.py",
        "analysis/v39_distill_controlled.py", "analysis/prediction_audit.py",
        "analysis/v6_capability_geometry.py", "analysis/v10_quantization.py",
        "analysis/v54_quant_group.py", "analysis/model_registry.py",
        "paper/docs/A100_EXPLORATION_PLAN.md")
PROTOCOL = {**v69.PROTOCOL, "qa_distribution": "2Wiki", "reference_device": "cpu",
            "pruning": "v6 sampled global magnitude threshold, seed 0, 2000000 weights"}
READING_RULE = (
    "The locked rule is confirmed for a capability if its regret is within the v64 "
    "map's and below quant-only's on the new panel; otherwise it is reported as retrospective."
)
READING_DETAIL = (
    "Operational rule, fixed before measurement: mean regret over the same 4 states x 17 "
    "budgets (equal state and budget weights), locked <= v64 and locked < quant-only, "
    "with no fitted margin. All three policies are feasible on the full planned panel. "
    "Missing measurements prevent confirmation; never silently shrink the panel. "
    "The max map uses max_c(L_c-source_L0c), as in v64. Quant-only pools channel and "
    "grouped RTN and uses locked predictions; its v64-prediction variant is also reported. "
    "KD candidates reuse two historical v39 outcomes, not new independent KD experiments. "
    "Both students are purged from both KD fits, as in v64. Constants are v39's arithmetic "
    "mean-delta baseline on the remaining development students; math uses its +D0 linear form. "
    "QA claims apply only to 2Wiki. No-clear-winner sets reuse v64's frozen development LOSO "
    "MAE thresholds for both maps, unchanged; they are retrospective heuristics, not "
    "calibrated uncertainty for the locked predictors. Storage is nominal matrix storage: "
    "d, b/16, (b+16/g)/16, and N_student/N_source; sparse index overhead is excluded."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return v69.digest(path)


def label(path):
    path = Path(path).absolute()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def hash_inputs(paths):
    return {label(p): digest(p) for p in paths}


def check_hashes(hashes):
    for name, expected in hashes.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        require(digest(path) == expected, f"Input SHA256 changed: {path}")


def json_text(obj):
    return json.dumps(obj, indent=2, allow_nan=False, default=v64.to_json) + "\n"


def publish(path, obj):
    text = json_text(obj)
    sha = hashlib.sha256(text.encode()).hexdigest()
    v69.write_outputs({path: text, path.with_suffix(path.suffix + ".sha256"): sha + "\n"}, exclusive=True)


def sealed(path):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == path.with_suffix(path.suffix + ".sha256").read_text().strip(),
            f"Artifact SHA256 changed: {path}")
    return json.loads(raw)


def candidate_grid(tag):
    require(tag in TAGS, f"Unregistered panel state: {tag}")
    source = v64.state_info(tag, prune)
    def config(method, law, key, ratio, **extra):
        return {"id": f"{method}:{key}", "method": method, "law": law, "config": key,
                "r": float(ratio), "target_state": tag, **extra}
    configs = [config("dense", "dense", "source", 1.)]
    configs += [config("prune", "prune_power", f"d{d:g}", d, d=d) for d in (.9, .8, .7, .6)]
    configs += [config("quant", "quant_channel", f"channel_b{b}", b / 16, bit=b) for b in v64.BITS]
    configs += [config("quant", "quant_group", f"b{b}_g{g}", (b + 16 / g) / 16,
                       bit=b, group_size=g) for b in (3, 4, 5) for g in (64, 128, 256)]
    if tag == TAGS[-1]:
        for student in STUDENTS:
            info = v64.state_info(student, prune)
            configs.append(config("distill", "distill_linear", student, info["N0"] / source["N0"],
                                  target_state=student))
    return {**source, "state_status": "new_stage", "configs": configs}


def plan(out, snapshot_root):
    return {"panel": [candidate_grid(t) for t in TAGS], "budgets": BUDGETS,
            "objectives": list(OBJECTIVES), "policies": list(POLICIES),
            "new_evaluations": {"dense": 4, "compressed": 72}, "reused_kd_outcomes": 2,
            "snapshot_root": str(snapshot_root), "output": str(out), "protocol": PROTOCOL,
            "sequence": ["freeze --stage independent", "download fresh snapshots", "verify",
                         "measure --stage dense", "freeze --stage finalize", "measure --stage configs", "compare"],
            "download_contract": "Use a new private Hugging Face hub cache with force_download=True for "
                "each exact EleutherAI model/revision. Include weights, config and tokenizer files. "
                "verify reads refs/stepN -> snapshots/<commit> and hashes actual bytes of every blob. "
                "No network or snapshot download is performed by this script.",
            "reading_rule": READING_RULE, "reading_detail": READING_DETAIL}


def all_hashes(value):
    """Conservative: every SHA256 anywhere in the V77 identity artifact."""
    if isinstance(value, str):
        return {value.lower()} if re.fullmatch(r"[0-9a-fA-F]{64}", value) else set()
    if isinstance(value, dict):
        return set().union(*(all_hashes(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(all_hashes(v) for v in value)) if value else set()
    return set()


def inspect_snapshots(snapshot_root, identity_path):
    identity = json.loads(identity_path.read_bytes())
    forbidden = all_hashes(identity)
    require(forbidden, "V77 identity has no hashes; cannot verify independence")
    historical_cache = identity.get("cache_root")
    if historical_cache:
        require(snapshot_root.resolve() != Path(historical_cache).resolve(), "Use a fresh private snapshot cache")
    records, seen = [], set()
    hashes = hash_inputs([identity_path])
    for tag in TAGS:
        base, revision = tag.split("@")
        repo = snapshot_root / ("models--EleutherAI--" + base)
        ref = repo / "refs" / revision
        commit = ref.read_text().strip()
        require(re.fullmatch(r"[0-9a-f]{40}", commit), f"Invalid snapshot commit: {ref}")
        snapshot = repo / "snapshots" / commit
        require((snapshot / "config.json").is_file(), f"Missing config: {snapshot}")
        config = json.loads((snapshot / "config.json").read_bytes())
        size, _ = prune.parse_tag(tag)
        require(config.get("model_type") == "gpt_neox" and all(
            config.get(k) == v for k, v in prune.ARCHITECTURES[size].items()),
            f"Snapshot architecture does not match {tag}")
        files = sorted(p for p in snapshot.iterdir() if p.is_file())
        weights = [p for p in files if p.suffix in (".safetensors", ".bin")]
        require(weights, f"No downloaded weight blobs: {snapshot}")
        require(any((snapshot / name).is_file() for name in
                    ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin",
                     "pytorch_model.bin.index.json")), f"No loadable model weights: {snapshot}")
        for index in snapshot.glob("*.index.json"):
            data = json.loads(index.read_bytes())
            if "weight_map" in data:
                for name in set(data["weight_map"].values()):
                    require(Path(name).name == name and (snapshot / name).is_file(),
                            f"Missing or invalid shard {name} in {index}")
        hashes.update(hash_inputs([ref, *files]))
        blobs = []
        for path in weights:
            sha = hashes[label(path)]
            require(sha not in forbidden, f"REFUSE: {tag}/{path.name} coincides with a V77 hash {sha}")
            require(sha not in seen, f"REFUSE: repeated weight blob within the new panel: {tag}/{path.name}")
            seen.add(sha)
            blobs.append({"file": path.name, "sha256": sha, "bytes": path.stat().st_size,
                          "blob_path": str(path.resolve())})
        records.append({"tag": tag, "hf_id": "EleutherAI/" + base, "revision": revision,
                        "commit": commit, "snapshot_path": str(snapshot.absolute()), "weights": blobs})
    check_hashes(hashes)
    return {"status": "VERIFIED", "rule": "No actual downloaded weight-blob SHA256 matches any V77 hash",
            "records": records, "input_sha256": hashes, "identity_sha256": digest(identity_path)}


def verify(out, snapshot_root):
    publish(out / "verification.json", inspect_snapshots(snapshot_root, ROOT / V77))


def development_objects():
    """Use frozen development snapshots, never glob mutable confirmation data.

    V53/V69 delivered fits are reused verbatim. V64's exact frozen row roster
    is reconstructed for fit_fold (v53/v36/v55/v39 fit functions unchanged).
    The two selectable KD outcomes are excluded from fitting in both policies.
    """
    paths = [ROOT / p for p in (*CODE, V77, "results/v53-prune-dev/register.json",
                              "results/v69-quant-confirm/develop.json",
                              "results/v64-selection-feasible/summary.json",
                              "results/v39-distill-controlled/summary.json")]
    hashes = hash_inputs(paths)
    def read(rel):
        return json.loads((ROOT / rel).read_bytes())
    p = read("results/v53-prune-dev/register.json")
    g = read("results/v69-quant-confirm/develop.json")
    old = read("results/v64-selection-feasible/summary.json")
    reference = read("results/v39-distill-controlled/summary.json")
    require(reference["panel"]["run"] == distill.RUN, "Unexpected v39 recipe")
    rosters = [s["tag"] for s in p["dev_states"]] + [s["tag"] for s in old["states"]]
    rosters += [r["state"] for r in g["dev_rows"]]
    require(not set(TAGS) & set(rosters), "Confirmation state in a development roster")
    require(g["boundary_rule"] == v69.BOUNDARY_RULE, "V69 boundary rule changed")
    require(all(g["measurement_protocol"].get(k) == v for k, v in v69.PROTOCOL.items()),
            "Development and confirmation measurement protocols differ")
    prows, qrows, grows, drows = [], [], [], []
    for s in old["states"]:
        for q in s["configs"]:
            for cap in CAPS:
                anchor = q["anchor"][cap]
                raw = prune.raw_features(s["N0"], s["D0"], anchor)
                delta = q["actual"][cap] - anchor
                if q["law"] == "prune_power" and q["d"] >= .55:
                    prows.append({"source": s["tag"], "cap": cap, "d": q["d"], "phi_raw": raw, "y": delta})
                elif q["law"] == "quant_channel":
                    qrows.append({"row_id": f"{s['tag']}|{cap}|{q['bit']}", "cell": s["tag"],
                                  "arm": "quantization", "capability": cap, "config": q["bit"],
                                  "N0": s["N0"], "D0": s["D0"], "L0": anchor, "observed": delta})
                elif q["law"] == "quant_group" and s["tag"] in group.DEV_TAGS and q["config"] in group.DEV_CONFIGS:
                    grows.append({"state": s["tag"], "config": q["config"], "capability": cap,
                                  "phi_raw": raw, "dL": delta})
    require(len(old["students"]) == reference["panel"]["n_cells"], "Incomplete v39 cohort")
    for s in old["students"].values():
        require(s["tag"] not in TAGS, "Confirmation student in development")
        for cap in CAPS:
            drows.append({**v64.state_info(s["tag"], prune), "cap": cap, "L0": s["dense"][cap],
                          "delta": s["delta"][cap]})
    modules = (prune, group, v36, distill, None)
    fold = v64.fit_fold(candidate_grid(TAGS[-1]), prows, qrows, grows, drows, modules)
    constant = {c: float(np.mean([r["delta"] for r in drows if r["cap"] == c and r["tag"] not in STUDENTS]))
                for c in CAPS}
    locked = {"prune": {"standardization": p["standardization"], "models": p["models"]},
              "grouped": {"standardization": g["standardization"], "models": g["models"]},
              "channel": {"models": fold["quant_channel"]["models"],
                          "median": {c: {str(b): float(np.median([r["observed"] for r in qrows
                              if r["capability"] == c and r["config"] == b])) for b in v64.BITS} for c in CAPS}},
              "distill": {"linear": fold["distill_linear"]["models"], "constant": constant,
                          "excluded_students": list(STUDENTS)}}
    students = {t: {k: s[k] for k in ("tag", "N0", "D0", "dense", "path", "teacher")}
                for t, s in old["students"].items() if t in STUDENTS}
    for s in students.values():
        path = ROOT / s["path"]
        hashes.update(hash_inputs([path]))
        require(hashes[label(path)] == old["input_sha256"][s["path"]], "Reused student artifact changed")
    check_hashes(hashes)
    return {"locked": locked, "v64": fold, "students": students,
            "maes": old["law_loso_mae"], "development_roster": sorted(set(rosters)), "input_sha256": hashes,
            "probe_sha256": g["measurement_protocol"]["probe_sha256"],
            "fit_rule": "Delivered V53 and V69 fits; V64 frozen paired rows; v39 excludes both eligible students",
            "versions": {"python": sys.version, "numpy": np.__version__}}


def measurement_path(out, tag, config):
    filename = config["id"].replace(":", "__").replace("@", "--") + ".json"
    return out / "measurements" / tag.replace("@", "--") / filename


def refuse_compressed(out):
    for tag in TAGS:
        for q in candidate_grid(tag)["configs"]:
            if q["method"] not in ("dense", "distill"):
                require(not measurement_path(out, tag, q).exists(), "Compressed results precede final freeze")


def predict_config(source, q, models, dense=None):
    """All candidate/capability entries, including explicit unresolved L0s."""
    arm = {"prune_power": "pruning", "quant_channel": "per-channel",
           "quant_group": "grouped", "distill_linear": "distillation", "dense": "dense"}[q["law"]]
    predictions = {p: {} for p in ("locked-rule", "v64-law")}
    for cap in CAPS:
        inputs = {**source, **q, "L0": dense[cap] if dense is not None else 0.,
                  "models": models["locked"], "qa_distribution": "2Wiki"}
        if q["method"] == "distill":
            inputs["student"] = models["students"][q["target_state"]]
        value = rule.predict(arm, cap, "new_stage", inputs)
        independent = q["method"] == "distill"
        predictions["locked-rule"][cap] = {"absolute_loss": value if dense is not None or independent else None,
            "delta": value - dense[cap] if dense is not None and not independent else
                     (value if not independent else value - inputs["student"]["dense"][cap]),
            "anchor": "student_dense" if independent else "source_dense",
            "status": "FROZEN" if dense is not None or independent else "WAITING_FOR_DENSE"}
        if dense is None and not independent:
            predictions["v64-law"][cap] = {"absolute_loss": None, "delta": 0. if arm == "dense" else None,
                "anchor": "source_dense", "status": "WAITING_FOR_DENSE"}
        else:
            anchor = inputs["student"]["dense"] if independent else dense
            public = {**q, "anchor": anchor}
            val = v64.predict_config(public, source, models["v64"], (prune, group, v36, distill, None))[cap]
            predictions["v64-law"][cap] = {"absolute_loss": val, "delta": val - anchor[cap],
                "anchor": "student_dense" if independent else "source_dense", "status": "FROZEN"}
    return {**q, "predictions": predictions}


def build_maps(states, maes):
    maps = {policy: {c: [] for c in OBJECTIVES} for policy in ("locked-rule", "v64-law")}
    for policy in maps:
        for source in states:
            configs = [{**q, "predicted": {c: q["predictions"][policy][c]["absolute_loss"] for c in CAPS}}
                       for q in source["configs"]]
            require(all(q["predicted"][c] is not None for q in configs for c in CAPS), "Maps require dense anchors")
            for cap in OBJECTIVES:
                for budget in BUDGETS:
                    feasible = [q for q in configs if q["r"] <= budget + 1e-12]
                    selected = {"MAP": v64.choose(feasible, cap, source["dense"]) if feasible else None,
                                "cheapest": min(feasible, key=lambda q: (q["r"], *v64.tie_key(q))) if feasible else None}
                    for m in ("quant", "prune", "distill"):
                        within = [q for q in feasible if q["method"] == m]
                        selected[m + "-only"] = v64.choose(within, cap, source["dense"]) if within else None
                    ambiguity = v64.ambiguity(feasible, cap, source["dense"], maes)
                    maps[policy][cap].append({"state": source["tag"], "budget": budget,
                        "n_feasible": len(feasible), "policies": {name: {
                            "status": "FEASIBLE" if q else "INFEASIBLE", "config_id": q["id"] if q else None,
                            "method": q["method"] if q else None, "r": q["r"] if q else None,
                            "predicted_score": v64.score(q, cap, source["dense"], "predicted") if q else None}
                            for name, q in selected.items()}, **ambiguity,
                        "candidate_config_ids": [r["config_id"] for r in ambiguity["methods"]
                                                 if r["method"] in ambiguity["candidate_methods"]]})
    return maps


def reading_markdown():
    return READING_RULE + "\n\n" + READING_DETAIL + "\n\nStatus: awaiting independent panel measurements.\n"


def freeze_independent(out):
    require(not (out / "freeze-independent.json").exists(), "Independent freeze is write-once")
    require(not any((out / "measurements").rglob("*.json")), "Independent freeze must precede dense measurements")
    models = development_objects()
    states = []
    for tag in TAGS:
        s = candidate_grid(tag)
        states.append({**s, "configs": [predict_config(s, q, models) for q in s["configs"]]})
    data = {"stage": "dense-independent", "models": models, "states": states, "protocol": PROTOCOL,
            "reading_rule": READING_RULE, "reading_detail": READING_DETAIL,
            "maps_status": "WAITING_FOR_DENSE: absolute-loss maps finalized before compressed measurements",
            "input_sha256": models["input_sha256"]}
    text = reading_markdown()
    if (out / "compare.md").exists():
        require((out / "compare.md").read_text() == text, "Pre-stated compare.md reading rule changed")
    else:
        v69.write_outputs({out / "compare.md": text}, exclusive=True)
    v69.write_outputs({out / "reading-rule.md": text}, exclusive=True)
    publish(out / "freeze-independent.json", data)


def checked_measurement(out, source, q, verification, independent_sha, final_sha=None):
    path = measurement_path(out, source["tag"], q)
    value = sealed(path)
    require(value["state"] == source["tag"] and value["config"] == q, f"Measurement identity mismatch: {path}")
    require(value["protocol"] == PROTOCOL, f"Measurement protocol changed: {path}")
    require(value["verification_sha256"] == verification and value["independent_sha256"] == independent_sha,
            f"Measurement freeze/weight provenance mismatch: {path}")
    if q["method"] != "dense":
        require(final_sha is not None and value["final_sha256"] == final_sha, f"Missing final freeze provenance: {path}")
    require(re.fullmatch(r"[0-9a-f]{64}", value["probe_sha256"]), f"Missing probe hash: {path}")
    return prune.checked_losses(value["losses"], path), value["probe_sha256"]


def freeze_finalize(out):
    require(not (out / "freeze.json").exists(), "Final freeze is write-once")
    refuse_compressed(out)
    initial_path, verify_path = out / "freeze-independent.json", out / "verification.json"
    initial, verification = sealed(initial_path), sealed(verify_path)
    check_hashes(initial["input_sha256"])
    require(verification["identity_sha256"] == initial["input_sha256"][V77], "V77 identity changed")
    states, paths, probes = [], [initial_path, verify_path, out / "reading-rule.md"], set()
    for tag in TAGS:
        s = candidate_grid(tag)
        dense_q = s["configs"][0]
        dense, probe = checked_measurement(out, s, dense_q, digest(verify_path), digest(initial_path))
        probes.add(probe)
        paths.append(measurement_path(out, tag, dense_q))
        states.append({**s, "dense": dense,
            "configs": [predict_config(s, q, initial["models"], dense) for q in s["configs"]]})
    require(len(probes) == 1, "Different dense probe sets across states")
    require(probes == {initial["models"]["probe_sha256"]}, "Dense probes differ from development")
    data = {"stage": "final", "states": states, "maps": build_maps(states, initial["models"]["maes"]),
            "probe_sha256": probes.pop(), "protocol": PROTOCOL,
            "reading_rule": READING_RULE, "reading_detail": READING_DETAIL,
            "input_sha256": {**initial["input_sha256"], **hash_inputs(paths)}}
    check_hashes(data["input_sha256"])
    publish(out / "freeze.json", data)


def require_gpu_uuid():
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value),
            "measure requires caller-set CUDA_VISIBLE_DEVICES containing one full GPU UUID")
    return value


def measure(out, stage):
    gpu_uuid = require_gpu_uuid()
    initial_path, verify_path = out / "freeze-independent.json", out / "verification.json"
    initial, verification = sealed(initial_path), sealed(verify_path)
    check_hashes(initial["input_sha256"])
    check_hashes(verification["input_sha256"])
    require(verification["identity_sha256"] == initial["input_sha256"][V77], "V77 identity changed")
    final_path = out / "freeze.json"
    final = sealed(final_path) if stage == "configs" else None
    if final:
        check_hashes(final["input_sha256"])
    else:
        refuse_compressed(out)
    # This is the only import boundary for torch/transformers/datasets.
    import gc
    import torch
    from analysis import v6_capability_geometry as v6
    from analysis import v10_quantization as v10
    from analysis import v54_quant_group as v54
    require(torch.cuda.is_available(), "CUDA is unavailable")
    probes = {c: p[1::2] for c, p in v6.build_probes(PROTOCOL["n_probe"], seed=0).items()}
    require(set(probes) == set(CAPS) and all(probes.values()), "Incomplete measurement probes")
    probe_sha = hashlib.sha256(json.dumps(probes, sort_keys=True).encode()).hexdigest()
    require(probe_sha == initial["models"]["probe_sha256"], "Measurement probes differ from development")
    if final:
        require(probe_sha == final["probe_sha256"], "Dense/config probes differ")
    for record in verification["records"]:
        tag = record["tag"]
        source = candidate_grid(tag)
        wanted = [q for q in source["configs"] if (q["method"] == "dense" if stage == "dense"
                                                   else q["method"] not in ("dense", "distill"))]
        pending = []
        for q in wanted:
            if measurement_path(out, tag, q).exists():
                _, existing_probe = checked_measurement(out, source, q, digest(verify_path), digest(initial_path),
                                                       digest(final_path) if final else None)
                require(existing_probe == probe_sha, "Existing file has different probes")
                print(f"SKIP {tag} {q['id']}")
            else:
                pending.append(q)
        if not pending:
            continue
        # Load the verified local snapshot, never a moving remote revision.
        model, tokenizer = v6.load_text_causal_lm(record["snapshot_path"], torch.bfloat16)
        parameters, weights = [], []
        try:
            model.to("cuda:0").eval().requires_grad_(False)
            parameters = v6.language_weight_parameters(model)
            weights = [p.detach().cpu().clone() for _, p in parameters] if stage == "configs" else []
            thresholds = {}
            if any(q["method"] == "prune" for q in pending):
                sample = v6._sample_abs_weights(parameters)
                thresholds = {q["d"]: float(np.quantile(sample, 1 - q["d"]))
                              for q in pending if q["method"] == "prune"}
            for q in pending:
                try:
                    if q["method"] == "prune":
                        v6.apply_global_magnitude_pruning(model, q["d"], reference_weights=weights,
                                                         threshold=thresholds[q["d"]])
                    elif q["law"] == "quant_channel":
                        v10._apply_fake_quantization(parameters, weights, q["bit"])
                    elif q["law"] == "quant_group":
                        v54._apply_grouped_quantization(parameters, weights, q["bit"], q["group_size"])
                    losses = v10._measure_capability_losses(model, tokenizer, probes, "cuda:0")
                    prune.checked_losses(losses, f"{tag}/{q['id']}")
                finally:
                    if weights:
                        v10._restore_dense_weights(parameters, weights)
                publish(measurement_path(out, tag, q), {"state": tag, "config": q, "losses": losses,
                    "protocol": PROTOCOL, "probe_sha256": probe_sha, "gpu_uuid": gpu_uuid,
                    "verification_sha256": digest(verify_path), "independent_sha256": digest(initial_path),
                    "final_sha256": digest(final_path) if final else None})
        finally:
            del parameters, weights, model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()


def evaluate(frozen, actuals):
    """Compare sealed selections to a complete measured oracle; no refitting."""
    cells, tables, coverage, verdict = {}, {}, {}, {}
    states = {s["tag"]: s for s in frozen["states"]}
    for cap in OBJECTIVES:
        cells[cap] = []
        for locked, old in zip(frozen["maps"]["locked-rule"][cap], frozen["maps"]["v64-law"][cap]):
            tag, budget = locked["state"], locked["budget"]
            require((tag, budget) == (old["state"], old["budget"]), "Map cell mismatch")
            s = states[tag]
            measured = [{**q, "actual": actuals[tag][q["id"]]} for q in s["configs"]]
            feasible = [q for q in measured if q["r"] <= budget + 1e-12]
            oracle = v64.choose(feasible, cap, s["dense"], "actual")
            best = v64.score(oracle, cap, s["dense"], "actual")
            lookup = {q["id"]: q for q in feasible}
            policies = {"locked-rule": locked["policies"]["MAP"], "v64-law": old["policies"]["MAP"],
                        **{p: locked["policies"][p] for p in POLICIES[2:]},
                        **{"v64-" + p: old["policies"][p] for p in ("quant-only", "prune-only", "distill-only")}}
            results = {}
            for policy, choice in policies.items():
                selected = lookup.get(choice["config_id"])
                require(choice["config_id"] is None or selected is not None, "Frozen selection is infeasible")
                regret = v64.score(selected, cap, s["dense"], "actual") - best if selected else None
                require(regret is None or regret >= -1e-12, "Negative regret")
                results[policy] = {**choice, "regret": regret,
                    "actual_score": best + regret if selected else None,
                    "oracle_method_agreement": selected["method"] == oracle["method"] if selected else None}
            sets = {}
            for name, entry in (("locked-rule", locked), ("v64-law", old)):
                sets[name] = {"candidate_methods": entry["candidate_methods"],
                    "candidate_config_ids": entry["candidate_config_ids"],
                    "no_clear_winner_heuristic": entry["no_clear_winner_heuristic"],
                    "oracle_method_covered": oracle["method"] in entry["candidate_methods"],
                    "oracle_config_covered": oracle["id"] in entry["candidate_config_ids"]}
            cells[cap].append({"state": tag, "budget": budget, "oracle_id": oracle["id"],
                "oracle_method": oracle["method"], "oracle_score": best, "policies": results, "candidate_sets": sets})
        tables[cap] = []
        for name in cells[cap][0]["policies"]:
            values = [r["policies"][name] for r in cells[cap]]
            own = [r for r in values if r["regret"] is not None]
            tables[cap].append({"policy": name, "n_cells": len(values), "n_feasible": len(own),
                "mean_regret": v64.mean_or_none(r["regret"] for r in own),
                "method_agreement": v64.mean_or_none(r["oracle_method_agreement"] for r in own)})
        summary = {r["policy"]: r for r in tables[cap]}
        matched = all(summary[p]["n_feasible"] == len(TAGS) * len(BUDGETS)
                      for p in ("locked-rule", "v64-law", "quant-only"))
        require(matched, "Confirmation requires the full common feasible panel")
        a, b, c = (summary[p]["mean_regret"] for p in ("locked-rule", "v64-law", "quant-only"))
        verdict[cap] = "confirmed" if a <= b and a < c else "retrospective"
        coverage[cap] = {}
        for policy in ("locked-rule", "v64-law"):
            sets = [r["candidate_sets"][policy] for r in cells[cap]]
            ambiguous = [r for r in sets if r["no_clear_winner_heuristic"]]
            coverage[cap][policy] = {"n_cells": len(sets), "n_no_clear_winner": len(ambiguous),
                "method_coverage": v64.mean_or_none(r["oracle_method_covered"] for r in sets),
                "config_coverage": v64.mean_or_none(r["oracle_config_covered"] for r in sets),
                "method_coverage_no_clear_winner": v64.mean_or_none(r["oracle_method_covered"] for r in ambiguous)}
    return {"cells": cells, "tables": tables, "candidate_set_coverage": coverage, "verdict": verdict,
            "reading_rule": READING_RULE, "reading_detail": READING_DETAIL}


def report_markdown(result):
    lines = [READING_RULE, "", READING_DETAIL, ""]
    for cap in OBJECTIVES:
        lines += [f"{cap}: **{result['verdict'][cap]}**", "",
                  "| Policy | Feasible / 68 | Mean regret (nats) | Method agreement |",
                  "|---|---:|---:|---:|"]
        for r in result["tables"][cap]:
            lines.append(f"| {r['policy']} | {r['n_feasible']} | {v64.number(r['mean_regret'])} | "
                         f"{v64.number(r['method_agreement'], 3)} |")
        lines += ["", "| Candidate set | Method coverage | Exact config coverage | Ambiguous cells |",
                  "|---|---:|---:|---:|"]
        for p, r in result["candidate_set_coverage"][cap].items():
            lines.append(f"| {p} | {r['method_coverage']:.3f} | {r['config_coverage']:.3f} | {r['n_no_clear_winner']} |")
        lines.append("")
    return "\n".join(lines)


def latex(result):
    lines = [r"\begin{table}[!htbp]", r"\centering\small",
             r"\caption{Independent V78 selection panel. Regret is in nats; agreement and heuristic "
             r"method coverage are percentages. QA is restricted to 2Wiki; KD reuses v39 students.}",
             r"\label{tab:rule-confirm}", r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Capability & Policy & Feasible & Regret & Agreement & Set coverage \\", r"\midrule"]
    for cap in OBJECTIVES:
        for r in result["tables"][cap]:
            if r["policy"] not in ("locked-rule", "v64-law", "quant-only", "cheapest"):
                continue
            cov = result["candidate_set_coverage"][cap].get(r["policy"])
            lines.append(f"{cap if cap != 'multi' else 'Max'} & {r['policy']} & {r['n_feasible']}/68 & "
                         f"{v64.number(r['mean_regret'], 4)} & {v64.number(100*r['method_agreement'], 1)} & "
                         f"{v64.number(100*cov['method_coverage'], 1) if cov else '---'}" + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\par\smallskip\begin{minipage}{\linewidth}\small",
              r"The max objective is $\max_c(L_c-L_{0c})$. Quant-only pools both RTN arms. "
              r"All four headline policies share the same 68 feasible cells. Heuristic sets use "
              r"frozen v64 development errors, not calibrated uncertainty. Verdicts: " +
              "; ".join(f"{c}: {result['verdict'][c]}" for c in CAPS) + ".",
              r"\end{minipage}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def figure(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    colors = ("#d77a3d", "#467db1", "#60a27b", "#bcbcbc")
    symbols = dict(zip(v64.METHODS, ("^", "s", "D", "o")))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    labels = [f"{v64.state_info(t, prune)['size']} @ {v64.state_info(t, prune)['step']//1000}k / {p}"
              for t in TAGS for p in ("locked", "v64")]
    for ax, cap in zip(axes.flat, OBJECTIVES):
        entries = result["cells"][cap]
        matrix = np.empty((8, len(BUDGETS)), dtype=int)
        for i, entry in enumerate(entries):
            state, x = divmod(i, len(BUDGETS))
            for j, policy in enumerate(("locked-rule", "v64-law")):
                y = state * 2 + j
                matrix[y, x] = v64.METHODS.index(entry["policies"][policy]["method"])
                ax.plot(x, y, marker=symbols[entry["oracle_method"]], ms=4,
                        mfc="white", mec="#222222", mew=.65, linestyle="none")
                if entry["candidate_sets"][policy]["no_clear_winner_heuristic"]:
                    ax.add_patch(Rectangle((x-.5, y-.5), 1, 1, fill=False, hatch="//", lw=0, edgecolor="#555555"))
        ax.imshow(matrix, cmap=ListedColormap(colors), vmin=-.5, vmax=3.5, aspect="auto", zorder=-1)
        ax.set_yticks(range(8), labels, fontsize=8)
        ticks = list(range(0, len(BUDGETS), 2))
        ax.set_xticks(ticks, [f"{BUDGETS[i]:.2f}" for i in ticks], fontsize=8)
        ax.set_xlabel("Nominal storage budget")
        ax.set_title({"qa": "QA (2Wiki)", "multi": "Max capability loss increase"}.get(cap, cap.title()))
    handles = [Patch(facecolor=c, label=m.title()) for c, m in zip(colors, v64.METHODS)]
    handles += [Line2D([], [], marker=symbols[m], ls="", mfc="white", mec="black", label="Oracle " + m)
                for m in v64.METHODS]
    fig.legend(handles=handles, loc="outside upper center", ncols=4, fontsize=9)
    fig.suptitle("Predicted method colours; symbols identify measured oracle; hatching: heuristic candidate set", fontsize=10)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return buffer.getvalue()


def compare(out, paper_root=None):
    paper_root = ROOT / "paper" if paper_root is None else paper_root
    frozen_path, initial_path, verify_path = out / "freeze.json", out / "freeze-independent.json", out / "verification.json"
    frozen, initial = sealed(frozen_path), sealed(initial_path)
    check_hashes(frozen["input_sha256"])
    sealed(verify_path)
    actuals, hashes = {}, hash_inputs([frozen_path, initial_path, verify_path])
    for source in frozen["states"]:
        tag = source["tag"]
        actuals[tag] = {}
        for q in candidate_grid(tag)["configs"]:
            if q["method"] == "distill":
                path = ROOT / initial["models"]["students"][q["target_state"]]["path"]
                ev = json.loads(path.read_bytes())
                require(ev["student"] == q["target_state"] and ev["training_mode"] == "lora", "Student identity changed")
                actuals[tag][q["id"]] = prune.checked_losses(ev["post_training"], path)
            else:
                path = measurement_path(out, tag, q)
                actuals[tag][q["id"]], probe = checked_measurement(out, candidate_grid(tag), q,
                    digest(verify_path), digest(initial_path), digest(frozen_path))
                require(probe == frozen["probe_sha256"], "Measurement probes differ from frozen dense")
            hashes.update(hash_inputs([path]))
    result = evaluate(frozen, actuals)
    result["input_sha256"] = {**frozen["input_sha256"], **hashes}
    check_hashes(result["input_sha256"])
    # The write-once reading-rule.md and initial freeze preserve preregistration;
    # compare.md keeps that rule first and adds the measured results.
    v69.write_outputs({out / "compare.md": report_markdown(result), out / "compare.json": result,
                       paper_root / "paper/tables/rule_confirm.tex": latex(result)})
    pdf = figure(result)
    for path in (out / "figs/rule_maps.pdf", paper_root / "paper/figs/rule_maps.pdf"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pdf)
    print(report_markdown(result))


def selftest():
    """CPU tests of rule branches, frozen predictions, oracle and release gates."""
    from unittest.mock import patch
    def fails(fn, needle):
        try:
            fn()
        except (ValueError, FileNotFoundError, FileExistsError) as exc:
            assert needle in str(exc), str(exc)
        else:
            raise AssertionError("Expected refusal: " + needle)
    models = development_objects()
    original = json_text(models)
    states = []
    for i, tag in enumerate(TAGS):
        s = candidate_grid(tag)
        assert len(s["configs"]) == (21 if tag == TAGS[-1] else 19)
        dense = {c: 2. + j + i / 10 for j, c in enumerate(CAPS)}
        independent = [predict_config(s, q, models) for q in s["configs"]]
        full = [predict_config(s, q, models, dense) for q in s["configs"]]
        for a, b in zip(independent, full):
            for c in CAPS:
                x, y = a["predictions"]["locked-rule"][c], b["predictions"]["locked-rule"][c]
                assert np.isclose(x["delta"], y["delta"])
                assert (x["absolute_loss"] is not None) == (a["method"] == "distill")
        states.append({**s, "dense": dense, "configs": full})
    assert json_text(models) == original
    s = candidate_grid(TAGS[-1])
    base = {**s, "L0": 3., "d": .85, "models": models["locked"], "bit": 5, "config": "b4_g128"}
    for c in CAPS:
        f = models["locked"]["prune"]
        expected = prune.predict_all(f["models"][c], prune.standardize(prune.raw_features(s["N0"], s["D0"], 3.),
                                     f["standardization"]), .85)
        for status in ("new_size", "new_stage", "new_state", "seen_size_unseen_density"):
            key = "power" if status == "seen_size_unseen_density" and c != "qa" else "median_curve"
            assert np.isclose(rule.predict("pruning", c, status, base), 3 + expected[key])
        with v64.quant_vocabulary(v36):
            for b in v64.BITS:
                x = {**base, "bit": b, "config": b}
                f = models["locked"]["channel"]["models"][c]
                public = {k: x[k] for k in (*f["input_fields"], "config")}
                assert np.isclose(rule.predict("per-channel", c, "seen_state", x), 3 + v36.predict(f, [public])[0])
        for status in ("seen_state", "new_state"):
            for config in ("b3_g32", "b4_g128", "b5_g512"):
                f = models["locked"]["grouped"]
                expected = v69.predict_all(f["models"][c], prune.raw_features(s["N0"], s["D0"], 3.),
                                          config, f["standardization"])
                key = "same_input_interpolation" if status == "seen_state" and c != "qa" else "median"
                assert np.isclose(rule.predict("grouped", c, status, {**base, "config": config}), 3 + expected[key])
    fails(lambda: rule.predict("pruning", "math", "seen_state", base), "unseen density")
    fails(lambda: rule.predict("per-channel", "math", "new_state", {**base, "bit": 2}), "seen bit")
    fails(lambda: rule.predict("grouped", "math", "new_state", {**base, "config": "b2_g64"}), "Bit extrapolation")
    kd = {**base, "student": models["students"][STUDENTS[0]], "qa_distribution": "MuSiQue"}
    fails(lambda: rule.predict("distill", "qa", "new_source", kd), "2Wiki")
    kd["qa_distribution"] = "2Wiki"
    for c in ("code", "qa"):
        assert np.isclose(rule.predict("distill", c, "new_source", kd),
                          kd["student"]["dense"][c] + models["locked"]["distill"]["constant"][c])
    fails(lambda: rule.predict("distill", "math", "new_source", {**kd, "D0": 1}), "same stage")
    fails(lambda: rule.predict("prune", "qa", "new_stage", {**base, "d": .59}), "domain")
    assert set(STUDENTS) <= set(models["v64"]["distill_linear"]["excluded_states"])
    maps = build_maps(states, models["maes"])
    assert all(len(maps[p][c]) == 68 for p in maps for c in OBJECTIVES)
    assert maps["locked-rule"]["math"][0]["policies"]["prune-only"]["status"] == "INFEASIBLE"
    assert maps["locked-rule"]["math"][0]["policies"]["distill-only"]["config_id"] is None
    # Grouped scale overhead excludes g64 int3 from the first budget.
    assert .2 < next(q["r"] for q in states[0]["configs"] if q["config"] == "b3_g64")
    # Multi is the worst capability increase, not the largest absolute loss.
    trial = [{"id": "a", "method": "quant", "r": .2, "predicted": dict(math=2., code=2., qa=8.)},
             {"id": "b", "method": "quant", "r": .2, "predicted": dict(math=4., code=4., qa=6.)}]
    assert v64.choose(trial, "multi", dict(math=1., code=1., qa=7.))["id"] == "a"
    actual = {s["tag"]: {q["id"]: {c: q["predictions"]["locked-rule"][c]["absolute_loss"] for c in CAPS}
                          for q in s["configs"]} for s in states}
    result = evaluate({"states": states, "maps": maps}, actual)
    assert all(next(r for r in result["tables"][c] if r["policy"] == "locked-rule")["mean_regret"] == 0.
               for c in OBJECTIVES)
    assert all(result["candidate_set_coverage"][c]["locked-rule"]["method_coverage"] == 1. for c in OBJECTIVES)
    assert all(next(r for r in result["tables"][c] if r["policy"] == "locked-rule")["method_agreement"] == 1.
               for c in OBJECTIVES)
    assert "\\begin{table}[!htbp]" in latex(result)
    assert figure(result).startswith(b"%PDF")
    with tempfile.TemporaryDirectory(prefix="v78-selftest-") as tmp:
        out = Path(tmp)
        with redirect_stdout(io.StringIO()):
            publish(out / "sealed.json", {"x": 1})
        fails(lambda: publish(out / "sealed.json", {}), "overwrite")
        (out / "sealed.json").write_text('{}')
        fails(lambda: sealed(out / "sealed.json"), "SHA256 changed")
        fails(lambda: freeze_finalize(out), "freeze-independent")
        cache = out / "cache"
        for i, tag in enumerate(TAGS):
            base, revision = tag.split("@")
            repo = cache / ("models--EleutherAI--" + base)
            snapshot = repo / "snapshots" / (str(i) * 40)
            snapshot.mkdir(parents=True)
            (repo / "refs").mkdir()
            (repo / "refs" / revision).write_text(str(i) * 40)
            size, _ = prune.parse_tag(tag)
            (snapshot / "config.json").write_text(json_text({"model_type": "gpt_neox", **prune.ARCHITECTURES[size]}))
            (snapshot / "model.safetensors").write_bytes(f"synthetic-{i}".encode())
        identity = out / "identity.json"
        identity.write_text(json_text({"weights": [{"sha256": "f" * 64}]}))
        assert inspect_snapshots(cache, identity)["status"] == "VERIFIED"
        first = next(cache.rglob("model.safetensors"))
        identity.write_text(json_text({"weights": [{"sha256": digest(first)}]}))
        fails(lambda: inspect_snapshots(cache, identity), "coincides with a V77")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            fails(require_gpu_uuid, "UUID")
        # Full fixture lifecycle. Only synthetic dense/config losses are written;
        # development fits stay fixed and publication stays under /tmp.
        fixture = out / "panel"
        with redirect_stdout(io.StringIO()), patch(__name__ + ".development_objects", return_value=models):
            freeze_independent(fixture)
            initial_path = fixture / "freeze-independent.json"
            verify_path = fixture / "verification.json"
            initial_sha = digest(initial_path)
            publish(verify_path, {"input_sha256": {}, "identity_sha256": models["input_sha256"][V77]})
            verify_sha = digest(verify_path)
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-12345678-1234-1234-1234-123456789abc"}):
                fails(lambda: measure(fixture, "configs"), "freeze.json")
            fails(lambda: freeze_finalize(fixture), "dense__source")
            def measurement(s, q, loss, final_sha=None):
                publish(measurement_path(fixture, s["tag"], q), {"state": s["tag"], "config": q,
                    "losses": loss, "protocol": PROTOCOL, "probe_sha256": models["probe_sha256"],
                    "verification_sha256": verify_sha, "independent_sha256": initial_sha,
                    "final_sha256": final_sha})
            for s in states:
                measurement(s, candidate_grid(s["tag"])["configs"][0], s["dense"])
            # Replacing the fitter with an exception proves finalization never refits.
            with patch(__name__ + ".development_objects", side_effect=AssertionError("refit")):
                freeze_finalize(fixture)
            frozen_path = fixture / "freeze.json"
            final_sha = digest(frozen_path)
            frozen = sealed(frozen_path)
            assert frozen["maps"] == maps
            fails(lambda: freeze_finalize(fixture), "write-once")
            fails(lambda: compare(fixture, out / "paper"), "prune__")
            for s in states:
                for q in candidate_grid(s["tag"])["configs"]:
                    if q["method"] not in ("dense", "distill"):
                        measurement(s, q, actual[s["tag"]][q["id"]], final_sha)
            fails(lambda: refuse_compressed(fixture), "precede final freeze")
            compare(fixture, out / "paper")
            assert (fixture / "figs/rule_maps.pdf").read_bytes().startswith(b"%PDF")
            assert (fixture / "compare.md").read_text().startswith(READING_RULE)
            assert "\\begin{table}[!htbp]" in (out / "paper/paper/tables/rule_confirm.tex").read_text()
            # compare may be repeated without mutating its frozen reading rule.
            compare(fixture, out / "paper")
            first = measurement_path(fixture, TAGS[0], candidate_grid(TAGS[0])["configs"][1])
            first.write_text('{}')
            fails(lambda: compare(fixture, out / "paper"), "SHA256 changed")
    assert "torch" not in sys.modules, "CPU selftest imported GPU libraries"
    print("SELFTEST PASS: rules, boundary interpolation, purity, 78 candidates, 544 map cells, "
          "oracle/regret, figure, full freeze/compare lifecycle, hash collision refusal, "
          "write-once and CUDA UUID gates; no GPU imports")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", default="plan", choices=("plan", "verify", "freeze", "measure", "compare"))
    parser.add_argument("--stage", choices=("independent", "finalize", "dense", "configs"))
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    out = args.output_root.absolute()
    snapshot_root = args.snapshot_root or out / "snapshots"
    try:
        if args.dry_run or args.mode == "plan" and not args.selftest:
            print(json_text({**plan(out, snapshot_root), "requested_mode": args.mode, "requested_stage": args.stage,
                             "dry_run": args.dry_run}))
        elif args.selftest:
            selftest()
        elif args.mode == "verify":
            verify(out, snapshot_root)
        elif args.mode == "freeze":
            require(args.stage in ("independent", "finalize"), "freeze requires --stage independent or finalize")
            (freeze_independent if args.stage == "independent" else freeze_finalize)(out)
        elif args.mode == "measure":
            require(args.stage in ("dense", "configs"), "measure requires --stage dense or configs")
            measure(out, args.stage)
        else:
            compare(out)
    except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
