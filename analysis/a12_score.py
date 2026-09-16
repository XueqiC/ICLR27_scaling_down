#!/usr/bin/env python3
"""CPU-only scoring of A12 against frozen predictions; no fitting or model loads."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "results/a12-data-requirement-confirmation/plan.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    require(digest(plan["predictions"]) == plan["predictions_sha256"], "Frozen predictions SHA256 mismatch")
    require(digest(plan["decisions"]) == plan["decisions_sha256"], "Frozen decisions SHA256 mismatch")
    require(len(plan["trajectories"]) == 18 and len({t["trajectory_id"] for t in plan["trajectories"]}) == 18,
            "Expected 18 unique trajectories")
    expected = {(t["trajectory_id"], b, r, m) for t in plan["trajectories"] for b in (0, 50000, 100000, 200000)
                for r in plan["readouts"] for m in plan["methods"]}
    keys = [(p["trajectory_id"], p["nominal_supervised_tokens"], p["readout"], p["method"]) for p in plan["predictions"]]
    require(set(keys) == expected and len(keys) == len(expected), "Missing or duplicate frozen prediction")
    require(build_decisions(plan) == plan["decisions"], "Decisions do not follow the frozen prediction rule")
    return plan


def classify(points, tau):
    """A10's observed replicate envelopes and monotone crossing conventions."""
    if len(points) != 3 or any(p["count"] != 3 for p in points):
        return {"status": "incomplete", "interval": None}
    points = sorted(points, key=lambda p: p["D_U_min"])
    if any(b["mean"] > a["mean"] + 1e-12 for a, b in zip(points, points[1:])):
        return {"status": "non_monotone", "interval": None}
    good = [p["max"] <= tau for p in points]
    bad = [p["min"] > tau for p in points]
    if any(not (a or b) for a, b in zip(good, bad)):
        return {"status": "replicate_ambiguous", "interval": None}
    if all(good):
        return {"status": "upper_bound", "interval": [0, points[0]["D_U_max"]]}
    if all(bad):
        return {"status": "lower_bound", "interval": [points[-1]["D_U_min"], None]}
    first = good.index(True)
    return {"status": "crossed", "interval": [points[first-1]["D_U_min"], points[first]["D_U_max"]]}


def verdict_interval(groups):
    if any(len(g) != 3 or any(p["verdict"] == "abstain" for p in g) for g in groups):
        return {"status": "abstain", "interval": None}
    statuses = []
    for group in groups:
        values = {p["verdict"] for p in group}
        if len(values) != 1:
            return {"status": "replicate_ambiguous", "interval": None}
        statuses.append(next(iter(values)))
    if any(a == "pass" and b == "fail" for a, b in zip(statuses, statuses[1:])):
        return {"status": "non_monotone", "interval": None}
    if all(s == "pass" for s in statuses):
        return {"status": "upper_bound", "interval": [0, max(p["D_U"] for p in groups[0])]}
    if all(s == "fail" for s in statuses):
        return {"status": "lower_bound", "interval": [min(p["D_U"] for p in groups[-1]), None]}
    first = statuses.index("pass")
    return {"status": "crossed", "interval": [min(p["D_U"] for p in groups[first-1]), max(p["D_U"] for p in groups[first])]}


def build_decisions(plan):
    trajectories = {t["trajectory_id"]: t for t in plan["trajectories"]}
    groups = defaultdict(list)
    for p in plan["predictions"]:
        if p["nominal_supervised_tokens"]:
            for th in p["thresholds"]:
                for kind in ("boundary", "delta"):
                    groups[p["student"], p["nominal_supervised_tokens"], p["readout"], th["tau"], kind, p["method"]].append(
                        {"trajectory_id": p["trajectory_id"], "D_U": p["D_U"],
                         "position": trajectories[p["trajectory_id"]]["position"], "verdict": th[kind + "_verdict"]})
    decisions = []
    for key, values in sorted(groups.items()):
        by_pos = [[v for v in values if v["position"] == pos] for pos in ("below", "near", "above")]
        rec = next((g for g in by_pos if len(g) == 3 and all(v["verdict"] == "pass" for v in g)), None)
        decisions.append(dict(zip(("student", "nominal_supervised_tokens", "readout", "tau", "kind", "method"), key),
            predicted_crossing=verdict_interval(by_pos), recommendation_position=rec[0]["position"] if rec else None,
            recommended_trajectory_ids=[v["trajectory_id"] for v in rec] if rec else [],
            recommended_D_U=statistics.mean(v["D_U"] for v in rec) if rec else None))
    for d in decisions:
        baseline = [x for x in decisions if all(x[k] == d[k] for k in ("student", "nominal_supervised_tokens", "readout", "tau", "kind"))
                    and x["method"] in ("fixed_reuse", "student_isotonic")]
        fallback = any(b["recommended_D_U"] is None for b in baseline)
        if fallback:
            upper = [t for t in plan["trajectories"] if t["student"] == d["student"] and t["position"] == "above"]
            conservative = statistics.mean(t["pool"]["D_U_pool"] if t["pool"] else t["D_U_target"] for t in upper)
            comparator_ids = [t["trajectory_id"] for t in upper]
        else:
            conservative_row = max(baseline, key=lambda b: b["recommended_D_U"])
            conservative = conservative_row["recommended_D_U"]
            comparator_ids = conservative_row["recommended_trajectory_ids"]
        d.update(conservative_D_U=conservative, conservative_uses_unverified_ceiling=fallback,
                 conservative_trajectory_ids=comparator_ids,
                 independent_data_ratio=d["recommended_D_U"] / conservative if d["recommended_D_U"] else None,
                 independent_data_savings_fraction=1-d["recommended_D_U"] / conservative if d["recommended_D_U"] else None)
    return decisions


def interval_comparison(predicted, measured):
    a, b = predicted["interval"], measured["interval"]
    if a is None or b is None:
        return None
    alo, ahi = a[0], math.inf if a[1] is None else a[1]
    blo, bhi = b[0], math.inf if b[1] is None else b[1]
    overlap = max(alo, blo) <= min(ahi, bhi)
    distance = 0. if overlap else math.log(blo / ahi) if ahi < blo else math.log(alo / bhi)
    return {"overlap": overlap, "log_interval_separation": distance,
            "censored": predicted["status"] != "crossed" or measured["status"] != "crossed"}


def read_measurements(plan, root):
    measured, missing = {}, []
    for tr in plan["trajectories"]:
        if tr["status"] != "realized":
            missing.append({"trajectory_id": tr["trajectory_id"], "reason": tr["status"]})
            continue
        pool = tr["pool"]
        logpath = root / tr["native_run_dir"] / "train_log.json"
        if not logpath.is_file():
            missing.append({"trajectory_id": tr["trajectory_id"], "reason": "missing train_log.json"})
            continue
        log = json.loads(logpath.read_text())
        require(log.get("status") == "trained", f"Unfinished training: {logpath}")
        for key in ("lora", "learning_rate", "optimizer", "scheduler", "warmup_ratio", "dtype", "max_len", "effective_batch_size_sequences"):
            require(log.get(key) == plan["protocol"][key], f"Wrong training recipe {key}: {logpath}")
        counts, processed, supervised = {0: (0, 0)}, 0, 0
        for i, row in enumerate(log["loss_curve"], 1):
            require(row["step"] == i, f"Noncontiguous log: {logpath}")
            processed += row["tokens"]
            supervised += row["completion_tokens"]
            require((row["processed_tokens"], row["completion_tokens_seen"]) == (processed, supervised), f"Token ledger mismatch: {logpath}")
            counts[i] = processed, supervised
        baseline = None
        for cp in tr["checkpoints"]:
            path = root / cp["eval_path"]
            if not path.is_file():
                missing.append({"trajectory_id": tr["trajectory_id"], "reason": f"missing {path}"})
                continue
            ev = json.loads(path.read_text())
            for key, value in {"student": tr["student"], "resolved_student": tr["model"],
                    "data_seed": tr["data_seed"], "training_seed": tr.get("training_seed", plan["protocol"]["seed"]),
                    "n_per_domain": pool["n_per_domain"],
                    "data_pool_sha256": pool["data_pool_sha256"], "training_mode": "lora",
                    "teacher": "gpt-5.6-luna", "recipe": "full", "learning_rate": 1e-4,
                    "schedule_tokens": 678000, "updates": cp["update"],
                    "completion_tokens_seen": cp["actual_supervised_tokens"], "processed_tokens": cp["processed_tokens"]}.items():
                require(ev.get(key) == value, f"Wrong {key}: {path}")
            require(counts.get(cp["update"]) == (cp["processed_tokens"], cp["actual_supervised_tokens"]), f"Checkpoint/log mismatch: {path}")
            extra = ev.get("a12")
            if extra is None:
                missing.append({"trajectory_id": tr["trajectory_id"], "reason": f"missing six-readout evaluation: {path}"})
                continue
            require(extra["predictions_sha256"] == plan["predictions_sha256"] and extra["trajectory_id"] == tr["trajectory_id"], f"Wrong preregistration: {path}")
            require(extra["D_U_pool"] == pool["D_U_pool"] and extra["D_U_seen"] == cp["D_U_seen"], f"Wrong independent data accounting: {path}")
            require(extra["fresh_adapter_loaded"] is (cp["update"] != 0), f"Fresh readout used wrong adapter: {path}")
            require(ev.get("dense") == {cap: ev["post_training"][cap] - ev["delta"][cap] for cap in ("math", "code", "qa")}
                    or all(math.isclose(ev["dense"][cap], ev["post_training"][cap] - ev["delta"][cap], abs_tol=1e-10)
                           for cap in ("math", "code", "qa")), f"Dense/delta mismatch: {path}")
            require(set(extra["readouts"]) == set(plan["readouts"]), f"Readout set differs: {path}")
            if cp["update"] == 0:
                baseline = extra["readouts"]
            require(baseline is not None, f"Missing own update-0: {path}")
            for readout, cell in extra["readouts"].items():
                require(math.isfinite(cell["loss"]) and cell["loss"] >= 0 and cell["tokens"] > 0, f"Invalid loss: {path}")
                base = baseline[readout]
                require(cell["tokens"] == base["tokens"] and cell["probe_sha256"] == base["probe_sha256"], f"Probe changed: {path}")
                require(cell["n"] == (64 if readout in ("math:MATH-500", "code:MBPP", "qa:2Wiki-probe") else 256), f"Wrong sample count: {path}")
                source = plan["panels"]["primary" if readout in ("math:MATH-500", "code:MBPP", "qa:2Wiki-probe") else "fresh"]
                require(cell["probe_sha256"] == plan["runtime_sha256"][source], f"Wrong frozen probes: {path}")
                delta = cell["loss"] - base["loss"]
                require(math.isclose(delta, cell["delta"], abs_tol=1e-10), f"Delta/own-reference mismatch: {path}")
                cap = {"math:MATH-500": "math", "code:MBPP": "code", "qa:2Wiki-probe": "qa"}.get(readout)
                if cap:
                    require(cell["loss"] == ev["post_training"][cap] and math.isclose(delta, ev["delta"][cap], abs_tol=1e-10), f"V12 primary mismatch: {path}")
                measured[tr["trajectory_id"], cp["nominal_supervised_tokens"], readout] = {
                    "delta": delta, "loss": cell["loss"], "D_U": pool["D_U_pool"], "D_U_seen": cp["D_U_seen"],
                    "T": cp["actual_supervised_tokens"], "position": tr["position"], "student": tr["student"]}
    return measured, missing


def score(plan, measured, missing):
    errors, decisions = [], []
    for p in plan["predictions"]:
        observed = measured.get((p["trajectory_id"], p["nominal_supervised_tokens"], p["readout"]))
        if observed is None or not p["nominal_supervised_tokens"]:
            continue
        for th in p["thresholds"]:
            passed = observed["delta"] <= th["tau"]
            errors.append({**{k: p[k] for k in ("trajectory_id", "student", "nominal_supervised_tokens", "readout", "method")},
                "tau": th["tau"], "measured_delta": observed["delta"], "predicted_delta": p["delta_prediction"],
                "signed_delta_error": None if p["delta_prediction"] is None else p["delta_prediction"] - observed["delta"],
                "delta_correct": None if th["delta_verdict"] == "abstain" else (th["delta_verdict"] == "pass") == passed,
                "boundary_correct": None if th["boundary_verdict"] == "abstain" else (th["boundary_verdict"] == "pass") == passed})
    for d in plan["decisions"]:
        values = [v for (tid, b, r), v in measured.items() if v["student"] == d["student"] and b == d["nominal_supervised_tokens"] and r == d["readout"]]
        points = []
        for pos in ("below", "near", "above"):
            group = [v for v in values if v["position"] == pos]
            if group:
                points.append({"count": len(group), "mean": statistics.mean(v["delta"] for v in group),
                    "min": min(v["delta"] for v in group), "max": max(v["delta"] for v in group),
                    "D_U_min": min(v["D_U"] for v in group), "D_U_max": max(v["D_U"] for v in group)})
        actual = classify(points, d["tau"])
        span = (max(v["T"] for v in values) / min(v["T"] for v in values) - 1) if values else None
        if span is not None and span > plan["matched_budget_span_tolerance"]:
            actual = {"status": "unmatched_budgets", "interval": None}
        rec = [measured.get((tid, d["nominal_supervised_tokens"], d["readout"])) for tid in d["recommended_trajectory_ids"]]
        success = all(v["delta"] <= d["tau"] for v in rec) if len(rec) == 3 and all(v is not None for v in rec) else None
        comparator = [measured.get((tid, d["nominal_supervised_tokens"], d["readout"])) for tid in d["conservative_trajectory_ids"]]
        comparator_success = all(v["delta"] <= d["tau"] for v in comparator) if len(comparator) == 3 and all(v is not None for v in comparator) else None
        decisions.append({**d, "measured_crossing": actual, "actual_budget_relative_span": span,
            "interval_comparison": interval_comparison(d["predicted_crossing"], actual),
            "recommendation_meets_loss_constraint": success,
            "conservative_meets_loss_constraint": comparator_success,
            "credited_savings_fraction": d["independent_data_savings_fraction"] if success is True and comparator_success is True else None,
            "conservative_D_U_seen": statistics.mean(v["D_U_seen"] for v in comparator) if all(v is not None for v in comparator) else None,
            "recommended_D_U_seen": statistics.mean(v["D_U_seen"] for v in rec) if rec and all(v is not None for v in rec) else None})
    groups = defaultdict(list)
    for e in errors:
        groups[e["student"], e["nominal_supervised_tokens"], e["readout"], e["tau"], e["method"]].append(e)
    metrics = []
    common = defaultdict(set)
    for e in errors:
        if e["signed_delta_error"] is not None:
            common[e["trajectory_id"], e["nominal_supervised_tokens"], e["readout"], e["tau"]].add(e["method"])
    for key, rows in sorted(groups.items()):
        signed = [r["signed_delta_error"] for r in rows if r["signed_delta_error"] is not None]
        paired = [abs(r["signed_delta_error"]) for r in rows if r["signed_delta_error"] is not None and
                  len(common[r["trajectory_id"], r["nominal_supervised_tokens"], r["readout"], r["tau"]]) == 3]
        m = dict(zip(("student", "nominal_supervised_tokens", "readout", "tau", "method"), key), n=len(rows),
                 delta_mae=statistics.mean(abs(x) for x in signed) if signed else None,
                 delta_signed_error=statistics.mean(signed) if signed else None,
                 common_support_delta_mae=statistics.mean(paired) if paired else None, common_support_n=len(paired))
        for kind in ("delta", "boundary"):
            correct = [r[kind + "_correct"] for r in rows if r[kind + "_correct"] is not None]
            m[kind + "_accuracy"] = statistics.mean(correct) if correct else None
            m[kind + "_coverage"] = len(correct) / len(rows)
        metrics.append(m)
    return {"status": "complete" if not missing and plan["launch_ready"] else "incomplete",
            "predictions_sha256": plan["predictions_sha256"], "missing": missing,
            "measured_readout_cells": len(measured), "expected_readout_cells": 432,
            "metrics": metrics, "checkpoint_errors": errors, "decisions": decisions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root holding returned run files")
    parser.add_argument("--output", type=Path, default=DEFAULT_PLAN.parent / "score.json")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    plan = load_plan(args.plan)
    measured, missing = read_measurements(plan, args.root)
    result = score(plan, measured, missing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "measured_readout_cells", "expected_readout_cells")}))
    if result["status"] != "complete" and not args.allow_incomplete:
        raise SystemExit("A12 incomplete or blocked; no confirmation verdict issued")


if __name__ == "__main__":
    main()
