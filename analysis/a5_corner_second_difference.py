#!/usr/bin/env python3
"""A5: the preregistered, fit-free corner contrast, from saved JSON only.

Standard library/CPU only; no model, tokenizer, training, or evaluation imports.
Default missing primary corners are PENDING. With --students, missing primary
inputs for any explicitly named student fail. Missing secondary corners remain
PENDING without affecting the primary. Present but invalid inputs always fail.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

if __package__:
    from .a1_development_table import own_delta, probe_distribution, require
else:
    from a1_development_table import own_delta, probe_distribution, require

ROOT = Path(__file__).resolve().parents[1]
PLAN = Path("results/a3-corner-pools/plan.json")
PREREG = Path("docs/prereg/corner_second_difference_prereg.md")
OUT = Path("results/a5-corner-second-difference")
REPORT = Path("docs/CORNER_SECOND_DIFFERENCE_REPORT.md")
STUDENTS = ("gemma3-1b", "gemma3-4b")
READOUTS = {
    "qa": ("QA probe, 2Wiki, measured in the trajectory", "2WikiMultihopQA", "PRIMARY"),
    "math": ("MATH-500 probe", "MATH-500", "SECONDARY, marginal"),
    "code": ("MBPP probe", "MBPP", "underpowered"),
}
FRESH = ("2Wiki, fresh sample", "TriviaQA", "MuSiQue")
FRESH_KEYS = ("2wiki_new", "triviaqa", "musique")
EDGES = ("budget_low", "budget_high", "reuse_low", "reuse_high")
FORMULA = "I = delta_3 - delta_4 - delta_1 + delta_2"
TOTAL_LIMITATION = (
    "Total contamination and the requested comparison of |I| with that total are "
    "unimplementable from plan.json without an additional surface assumption. "
    "The plan supplies four dependent, fixed-pool budget-equivalent sensitivities "
    "and explicitly says 'do not sum them'. They do not identify fixed-budget "
    "reuse effects or a joint bias bound. Each is reported below; no total is "
    "invented, no new slopes are estimated, and no corner is interpolated."
)


class Inputs:
    """Read immutable checkpoint snapshots once and retain their byte hashes."""

    def __init__(self, root):
        self.root = Path(root)
        self.sha256 = {}
        self.cache = {}

    def read_text(self, relative):
        relative = str(relative)
        if relative not in self.cache:
            raw = (self.root / relative).read_bytes()
            self.cache[relative] = raw.decode("utf-8")
            self.sha256[relative] = hashlib.sha256(raw).hexdigest()
        return self.cache[relative]

    def read(self, relative):
        return json.loads(self.read_text(relative))

    def verify(self):
        for path, expected in self.sha256.items():
            require(hashlib.sha256((self.root / path).read_bytes()).hexdigest() == expected,
                    f"Input changed during analysis: {path}")


def parse_prereg(document):
    """Read the amendment's table, never the superseded pooled-QA table."""
    heading = "## Amendment, same day, still before any corner was trained"
    require(document.count(heading) == 1, "Missing or ambiguous prereg amendment")
    for clause in ("delta_3 - delta_4 - delta_1 + delta_2", "twice the noise",
                   "two students agree in sign", "within that band on both students",
                   "size-dependent interaction", "MATH-500 stays", "MBPP stays underpowered"):
        require(clause in document, f"Unimplementable prereg rule: missing {clause!r}")
    amended = document.split(heading, 1)[1]
    rows = {}
    for line in amended.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().strip("*") for c in line.strip().strip("|").split("|")]
        if cells[0] not in {r[0] for r in READOUTS.values()} | set(FRESH):
            continue
        require(len(cells) == 5 and cells[0] not in rows, "Malformed prereg table")
        pairs, noise, prediction, ratio = int(cells[1]), *map(float, cells[2:])
        require(pairs > 0 and all(math.isfinite(v) and v > 0 for v in (noise, prediction, ratio)),
                "Invalid registered power or noise")
        rows[cells[0]] = dict(matched_seed_pairs=pairs, noise_on_I=noise,
                             predicted_disagreement=prediction, registered_ratio=ratio,
                             twice_noise=2 * noise)
    require(set(rows) == {r[0] for r in READOUTS.values()} | set(FRESH),
            "Missing distribution in prereg amendment table")
    return {
        "source": str(PREREG), "amendment": heading.removeprefix("## "),
        "formula": FORMULA, "interval_definition": "I +/- twice registered noise; not a calibrated confidence interval",
        "readouts": {cap: dict(rows[label], label=label, benchmark=benchmark, role=role)
                     for cap, (label, benchmark, role) in READOUTS.items()},
        "fresh_sample_readouts": {label: dict(rows[label], role="SECONDARY") for label in FRESH},
    }


def second_difference(corners):
    """Keys 1..4 are the registered corners; preserve exact numeric types."""
    require(set(corners) == {1, 2, 3, 4}, "Exactly four corner responses are required")
    return corners[3] - corners[4] - corners[1] + corners[2]


def corner_specs(plan, student):
    """Select only checkpoints explicitly marked as corners in the frozen plan."""
    design = {c["corner"]: c for c in plan["design"]["corners"]}
    require(set(design) == {1, 2, 3, 4}, "Invalid plan corner set")
    entry = plan["students"][student]
    old = entry["already_measured_corner_2"]
    expected = (f"results/v12-distill/{student}/gpt-5.6-luna_full_66_matrix2_lora_dseed41"
                "/trajectory/update-00000038/eval.json")
    require(old["checkpoint"] == "update-00000038" and old["source"] == expected,
            "Plan corner 2 is not the registered existing update-00000038")
    specs = {2: dict(design[2], source=old["source"], update=38,
                     processed_tokens=old["processed_tokens"], expected_sha256=old["sha256"])}
    require(design[2]["T_supervised"] == old["supervised_tokens"], "Plan corner 2 budget discrepancy")
    for trajectory in entry["new_trajectories"]:
        pool = trajectory["pool"]
        require(pool in ("B", "C"), "Unexpected new pool in plan")
        for cp in trajectory["predicted_checkpoints"]:
            for number in cp["corners"]:
                require(number not in specs and number in design, "Duplicate/unknown planned corner")
                require(design[number]["pool"] == pool and
                        cp["supervised_tokens"] == design[number]["T_supervised"],
                        f"Plan corner {number} budget/pool discrepancy")
                require(any(t["corner"] == number and t["offset_updates"] == 0 for t in cp["targets"]),
                        f"Plan selected a neighbour for corner {number}")
                require(any(t["update"] == cp["update"] and
                            t["supervised_tokens"] == cp["supervised_tokens"] and
                            t["processed_tokens"] == cp["processed_tokens"]
                            for t in trajectory["target_boundaries"]), "Plan target boundary discrepancy")
                source = str(Path(trajectory["output_directory"]) / "trajectory" /
                             f"update-{cp['update']:08d}" / "eval.json")
                specs[number] = dict(design[number], source=source, update=cp["update"],
                                     processed_tokens=cp["processed_tokens"])
    require(set(specs) == {1, 2, 3, 4}, "Missing planned corner")
    for number, spec in specs.items():
        require(spec["pool"] == {1: "B", 2: "A", 3: "C", 4: "B"}[number], "Plan corner pool changed")
        require(spec["D_U_pool"] == plan["pools"][spec["pool"]]["D_U_pool"], "Plan pool size discrepancy")
        require(Fraction(spec["E_exact"]) == Fraction(spec["T_supervised"], spec["D_U_pool"]),
                "Plan reuse discrepancy")
        path = Path(spec["source"])
        spec["baseline_source"] = str(path.parent.parent / "update-00000000/eval.json")
        spec["run_id"] = str(path.parents[2].relative_to("results/v12-distill"))
        require(spec["run_id"].split("/")[0] == student, "Wrong student in planned run")
    return specs


def check_payload(payload, spec, student, plan, *, baseline=False):
    pool = plan["pools"][spec["pool"]]
    wanted = {
        "student": student, "trajectory_run_id": spec["run_id"],
        "updates": 0 if baseline else spec["update"],
        "completion_tokens_seen": 0 if baseline else spec["T_supervised"],
        "processed_tokens": 0 if baseline else spec["processed_tokens"],
        "data_pool_sha256": pool["data_pool_sha256"],
        "data_sampling_seed": pool["data_seed"], "n_per_domain": pool["n_per_domain"],
        "training_seed": plan["protocol"]["training_seed"],
    }
    for key in ("teacher", "recipe", "training_mode", "schedule_tokens", "learning_rate"):
        wanted[key] = plan["protocol"][key]
    for key, expected in wanted.items():
        observed = payload.get(key)
        require(type(observed) is type(expected) and observed == expected,
                f"{key} discrepancy: predicted {expected!r}, observed {observed!r}")


def measure_corner(inputs, plan, student, spec):
    payload = inputs.read(spec["source"])
    check_payload(payload, spec, student, plan)
    if "expected_sha256" in spec:
        require(inputs.sha256[spec["source"]] == spec["expected_sha256"],
                "Existing corner 2 checksum discrepancy against plan")
    result = dict(spec, status="VERIFIED", achieved_update=payload["updates"],
                  achieved_supervised_tokens=payload["completion_tokens_seen"],
                  achieved_E=payload["completion_tokens_seen"] / spec["D_U_pool"],
                  achieved_E_exact=str(Fraction(payload["completion_tokens_seen"], spec["D_U_pool"])),
                  responses={})
    if not (inputs.root / spec["baseline_source"]).is_file():
        result["status"] = "PENDING"
        return result
    baseline = inputs.read(spec["baseline_source"])
    check_payload(baseline, spec, student, plan, baseline=True)
    require(payload["resolved_student"] == baseline["resolved_student"], "Initial student identity discrepancy")
    for cap, (_, benchmark, _) in READOUTS.items():
        dist, metadata = probe_distribution(payload, cap)
        base_dist, _ = probe_distribution(baseline, cap)
        require(dist == base_dist and metadata["measurement_benchmarks"] == benchmark,
                f"{cap}: own initial student uses a different distribution")
        loss, initial = payload["post_training"][cap], baseline["post_training"][cap]
        delta = own_delta(loss, initial)
        if "delta" in payload:
            require(math.isclose(payload["delta"][cap], delta, rel_tol=0, abs_tol=1e-10),
                    f"{cap}: saved delta disagrees with own initial student")
        result["responses"][cap] = dict(distribution=dist, initial_loss=initial, loss=loss, delta=delta)
    return result


def residual_mismatches(corners, plan):
    """The exact four A3 residual definitions, checked against the frozen plan."""
    t = {i: Fraction(c["achieved_supervised_tokens"]) for i, c in corners.items()}
    e = {i: t[i] / corners[i]["D_U_pool"] for i in corners}
    fractions = {
        "budget_low": abs(t[1] - t[2]) / t[2],
        "budget_high": abs(t[3] - t[4]) / min(t[3], t[4]),
        "reuse_low": abs(e[1] - e[3]) / min(e[1], e[3]),
        "reuse_high": abs(e[2] - e[4]) / min(e[2], e[4]),
    }
    for key, value in fractions.items():
        require(value == Fraction(plan["design"]["mismatches"][key]["fraction_exact"]),
                f"Achieved {key} discrepancy against plan")
    return {k: dict(fraction_exact=str(v), percent=float(100 * v)) for k, v in fractions.items()}


def sensitivity(plan, student, cap, distribution, statistic):
    source = plan["residual_contamination"]
    require(source["second_difference"] == "L4 - L3 - L2 + L1",
            "Unrecognized plan sensitivity sign convention")
    channels = [c for c in source["channels"] if (c["student"], c["capability"], c["distribution"])
                == (student, cap, distribution)]
    require(len(channels) == 1, "Missing/ambiguous plan sensitivity for exact student and distribution")
    channel = channels[0]
    require(set(channel["residuals"]) == set(EDGES), "Incomplete plan residual sensitivities")
    residuals = {}
    for edge in EDGES:
        old = channel["residuals"][edge]
        displacement = Fraction(old["signed_equivalent_supervised_displacement_exact"])
        require(displacement == Fraction(source["equivalent_displacements"][edge]),
                "Plan residual displacement discrepancy")
        sign = -old["second_difference_sign"]  # A3 uses -I; no change to magnitudes.
        slopes = []
        for stored in old["adjacent_checkpoint_slopes"]:
            slope = stored["native_token_nats_per_supervised_token"]
            require(math.isfinite(slope), "Nonfinite plan slope")
            contribution = sign * float(displacement) * slope
            require(math.isclose(contribution, -stored["signed_second_difference_sensitivity_nats"],
                                 rel_tol=1e-12, abs_tol=1e-15), "Plan signed sensitivity discrepancy")
            slopes.append(dict(stored, signed_sensitivity_for_registered_I=contribution))
        require(slopes, "No stored adjacent checkpoint slopes")
        values = [s["signed_sensitivity_for_registered_I"] for s in slopes]
        maximum = max(map(abs, values))
        require(math.isclose(maximum, old["max_absolute_sensitivity_nats"], rel_tol=1e-12, abs_tol=1e-15),
                "Plan sensitivity envelope discrepancy")
        residuals[edge] = dict(
            signed_equivalent_supervised_displacement_exact=str(displacement),
            sign_for_registered_I=sign, signed_range_nats=[min(values), max(values)],
            max_absolute_sensitivity_nats=maximum, stored_slopes=slopes,
            abs_I_exceeds_individual_sensitivity=abs(statistic) > maximum,
            abs_I_over_individual_sensitivity=abs(statistic) / maximum if maximum else None,
        )
    return dict(residuals=residuals, total_contamination_nats=None,
                abs_I_vs_total_contamination=None, limitation=TOTAL_LIMITATION,
                plan_limitation=source["limitation"], plan_method=source["method"],
                plan_sign_convention=source["second_difference"], sign_multiplier=-1)


def registered_decision(values, noise, *, underpowered=False):
    """Two separate student results; no mean and no post-hoc noise adjustment."""
    if set(values) != set(STUDENTS):
        return dict(status="PENDING", explanation="The registered decision needs both students and cannot be reached yet.")
    first, second = (values[s] for s in STUDENTS)
    threshold = 2 * noise
    if first * second < 0:
        status, explanation = "SIZE_DEPENDENT_INTERACTION", "Students disagree in sign: size-dependent interaction; do not average them."
    elif all(abs(v) > threshold for v in values.values()) and first * second > 0:
        status, explanation = "REJECTED", "Both students exceed twice the registered noise and agree in sign. Additivity is rejected; reuse strength must depend on budget in this regime."
    elif all(abs(v) <= threshold for v in values.values()):
        if underpowered:
            status, explanation = "UNDERPOWERED", "underpowered: a null cannot resolve the registered effect and is not support for additivity."
        else:
            status, explanation = "SURVIVES", "Additivity survives its one fit-free test. Failure of the three fitted structures is attributed to the form of the individual terms or to irreducible variation rather than to a missing interaction."
    else:
        status, explanation = "UNRESOLVED", "The students do not jointly satisfy rejection or survival. The registration specifies no conclusive outcome for this mixed-band case."
    if underpowered and status != "UNDERPOWERED":
        explanation += " MBPP remains underpowered whatever the outcome; a null cannot support additivity."
    return dict(status=status, explanation=explanation)


def fresh_sample_readout(inputs, specs, student, registration):
    """Use exact planned V99 corners; keep secondary status out of the decision."""
    result = dict(status="COMPLETE", role="SECONDARY", distributions=list(FRESH),
                  corners={}, readouts={}, missing_corners=[], failures=[],
                  explanation="SECONDARY only: the registered decision is carried by the PRIMARY "
                  "QA probe. These fresh-sample results never change that decision. Their noise "
                  "estimate is weak: only two matched seed pairs per distribution.")
    for number, spec in sorted(specs.items()):
        pattern = (f"results/v99-scope/a5-corners-{student}-pool{spec['pool']}"
                   f"/*/update-{spec['update']:08d}.json")
        corner = dict(pool=spec["pool"], update=spec["update"],
                      predicted_supervised_tokens=spec["T_supervised"],
                      source_pattern=pattern, status="PENDING", responses={})
        result["corners"][number] = corner
        paths = sorted(inputs.root.glob(pattern))
        if not paths:
            result["missing_corners"].append(number)
            continue
        try:
            require(len(paths) == 1, f"Ambiguous scope corner: {', '.join(map(str, paths))}")
            corner["source"] = str(paths[0].relative_to(inputs.root))
            payload = inputs.read(corner["source"])
            corner["actual_supervised_tokens"] = payload.get("actual_supervised_tokens")
            for key, expected in {
                "actual_supervised_tokens": spec["T_supervised"],
                "student": student, "trajectory_run_id": spec["run_id"],
                "updates": spec["update"], "status": "complete",
            }.items():
                observed = payload.get(key)
                require(type(observed) is type(expected) and observed == expected,
                        f"{key} discrepancy: predicted {expected!r}, observed {observed!r}")
            responses = {key: payload["delta_from_update_0"][key] for key in FRESH_KEYS}
            require(all(type(v) in (int, float) and math.isfinite(v) for v in responses.values()),
                    "Invalid/nonfinite fresh-sample delta_from_update_0")
            corner.update(status="VERIFIED", responses=responses)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            corner.update(status="FAILED", discrepancy=str(exc))
            result["failures"].append(f"{student} SECONDARY corner {number} "
                                      f"({corner.get('source', pattern)}): {exc}")
    if result["failures"]:
        result["status"] = "FAILED"
    elif result["missing_corners"]:
        result["status"] = "PENDING"
    else:
        for key, label in zip(FRESH_KEYS, FRESH):
            registered = registration["fresh_sample_readouts"][label]
            raw = {i: c["responses"][key] for i, c in result["corners"].items()}
            statistic = second_difference(raw)
            if not math.isfinite(statistic):
                result["status"] = "FAILED"
                result["failures"].append(f"{student} SECONDARY {key}: Nonfinite second difference")
                result["readouts"].clear()
                break
            band = registered["twice_noise"]
            result["readouts"][key] = dict(
                **registered, label=label, distribution=key, raw_responses=raw,
                I=statistic, abs_I=abs(statistic), interval=[statistic - band, statistic + band],
                exceeds_twice_noise=abs(statistic) > band,
            )
    return result


def build(root=ROOT, students=None):
    explicit = students is not None
    selected = tuple(students) if explicit else STUDENTS
    require(selected and len(set(selected)) == len(selected) and set(selected) <= set(STUDENTS),
            "--students must name distinct registered students")
    inputs = Inputs(root)
    registration = parse_prereg(inputs.read_text(PREREG))
    plan = inputs.read(PLAN)
    require(plan["status"] == "passed" and plan["launchable"], "A3 plan did not pass")
    report = dict(
        schema_version="a5-corner-second-difference-v1", status="COMPLETE",
        generated_at_utc=datetime.now(timezone.utc).isoformat(), device="cpu",
        model_weights_loaded=False, training_run=False, new_evaluation_run=False, fits_performed=0,
        selected_students=list(selected), explicitly_requested_students=list(selected) if explicit else [],
        preregistration=registration, plan_source=str(PLAN), students={}, decisions={}, failures=[],
        planned_residual_mismatches=deepcopy(plan["design"]["mismatches"]),
        limitations=[TOTAL_LIMITATION],
    )
    for student in STUDENTS:
        if student not in selected:
            report["students"][student] = dict(status="NOT_REQUESTED", corners={}, readouts={})
            continue
        specs = corner_specs(plan, student)
        entry = dict(status="COMPLETE", corners={}, readouts={}, missing_checkpoints=[], failures=[])
        report["students"][student] = entry
        for number, spec in sorted(specs.items()):
            for key in ("source", "baseline_source"):
                if not (inputs.root / spec[key]).is_file() and spec[key] not in entry["missing_checkpoints"]:
                    entry["missing_checkpoints"].append(spec[key])
            if not (inputs.root / spec["source"]).is_file():
                entry["corners"][number] = dict(spec, status="PENDING", responses={})
                continue
            try:
                entry["corners"][number] = measure_corner(inputs, plan, student, spec)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                error = f"{student} corner {number} ({spec['source']}): {exc}"
                entry["failures"].append(error)
                entry["corners"][number] = dict(spec, status="FAILED", responses={}, discrepancy=str(exc))
        if entry["failures"]:
            entry["status"] = "FAILED"
        elif entry["missing_checkpoints"]:
            entry["status"] = "PENDING"
        if explicit and entry["missing_checkpoints"]:
            entry["failures"].append(f"Explicitly requested {student}: missing checkpoint(s): " + ", ".join(entry["missing_checkpoints"]))
        try:
            if entry["status"] == "COMPLETE":
                entry["residual_mismatches"] = residual_mismatches(entry["corners"], plan)
                for cap, registered in registration["readouts"].items():
                    raw = {i: c["responses"][cap] for i, c in entry["corners"].items()}
                    require(len({r["distribution"] for r in raw.values()}) == 1,
                            f"{cap}: different distributions across corners")
                    statistic = second_difference({i: r["delta"] for i, r in raw.items()})
                    require(math.isfinite(statistic), "Nonfinite second difference")
                    noise = registered["noise_on_I"]
                    entry["readouts"][cap] = dict(
                        **registered, distribution=raw[1]["distribution"], raw_responses=raw,
                        I=statistic, abs_I=abs(statistic), interval=[statistic - 2 * noise, statistic + 2 * noise],
                        exceeds_twice_noise=abs(statistic) > 2 * noise,
                        contamination=sensitivity(plan, student, cap, raw[1]["distribution"], statistic),
                    )
        except (ValueError, KeyError, TypeError, OSError) as exc:
            entry["status"] = "FAILED"
            entry["failures"].append(f"{student}: {exc}")
        report["failures"].extend(entry["failures"])
        entry["fresh_sample_check"] = fresh_sample_readout(inputs, specs, student, registration)
        report["failures"].extend(entry["fresh_sample_check"]["failures"])
    for cap, registered in registration["readouts"].items():
        values = {s: e["readouts"][cap]["I"] for s, e in report["students"].items()
                  if e["status"] == "COMPLETE"}
        report["decisions"][cap] = dict(role=registered["role"], **registered_decision(
            values, registered["noise_on_I"], underpowered=cap == "code"))
    if report["failures"]:
        report["status"] = "FAILED"
    elif any(e["status"] != "COMPLETE" for e in report["students"].values()):
        report["status"] = "PENDING"
    inputs.verify()
    report["source_sha256"] = inputs.sha256
    report["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return report


def student_table(report):
    lines = ["| Student | Distribution / role | I | I ± 2 × noise | 2 × noise | Individual band |",
             "|---|---|---:|---|---:|---|"]
    for student, entry in report["students"].items():
        if entry["status"] != "COMPLETE":
            lines.append(f"| {student} | {entry['status']} | — | — | — | — |")
            continue
        for row in entry["readouts"].values():
            lo, hi = row["interval"]
            band = "outside" if row["exceeds_twice_noise"] else "inside"
            lines.append(f"| {student} | {row['benchmark']} / {row['role']} | {row['I']:+.10f} | "
                         f"[{lo:+.10f}, {hi:+.10f}] | {row['twice_noise']:.4f} | {band} |")
    lines += ["", "### Fresh-sample SECONDARY", "",
              "Noise estimate is weak: only two matched seed pairs per distribution. "
              "The PRIMARY QA probe alone carries the registered decision; SECONDARY results never change it.", "",
              "| Student | Distribution / role | I | I ± 2 × noise | 2 × noise | Individual band |",
              "|---|---|---:|---|---:|---|"]
    for student, entry in report["students"].items():
        secondary = entry.get("fresh_sample_check", {"status": "NOT_REQUESTED"})
        if secondary["status"] != "COMPLETE":
            lines.append(f"| {student} | SECONDARY {secondary['status']} | — | — | — | — |")
            continue
        for row in secondary["readouts"].values():
            lo, hi = row["interval"]
            band = "outside" if row["exceeds_twice_noise"] else "inside"
            lines.append(f"| {student} | {row['distribution']} / SECONDARY | {row['I']:+.10f} | "
                         f"[{lo:+.10f}, {hi:+.10f}] | {row['twice_noise']:.4f} | {band} |")
    return "\n".join(lines)


def render(report):
    lines = ["# A5: Corner second difference", "", f"Status: **{report['status']}**.", ""]
    if "preregistration" not in report:
        return "\n".join(lines + ["Analysis failed before valid results could be produced.", ""] +
                          [f"- {f}" for f in report["failures"]]) + "\n"
    if any(e["status"] != "COMPLETE" for e in report["students"].values()):
        lines += ["**The registered decision needs both students and cannot be reached yet.**", ""]
    lines += [student_table(report), "", "## Registered analysis", "",
              f"`{FORMULA}`. Delta is checkpoint loss minus the same trajectory's own update-0 loss, "
              "using A1's delta and distribution-identity functions. Units are native-token nats/token. "
              "Students are evaluated separately and never averaged.", "",
              "On an exact rectangle, A(T) cancels within each reuse contrast and B*h(E) cancels "
              "between budgets, for any h. The achieved rectangle's residuals are reported below.", "",
              f"Source: `{PREREG}`, including its same-day amendment. SHA256: "
              f"`{report['source_sha256'][str(PREREG)]}`. Numeric noise and power figures are parsed "
              "from the amendment table at runtime. The original pooled-QA figure is superseded.", "",
              "| Readout | Role | Matched seed pairs | Noise on I | Prediction | Registered ratio |",
              "|---|---|---:|---:|---:|---:|"]
    fresh_rows = [dict(row, label=label) for label, row in
                  report["preregistration"]["fresh_sample_readouts"].items()]
    for row in (*report["preregistration"]["readouts"].values(), *fresh_rows):
        lines.append(f"| {row['label']} | {row['role']} | {row['matched_seed_pairs']} | "
                     f"{row['noise_on_I']} | {row['predicted_disagreement']} | {row['registered_ratio']} |")
    lines += ["", "The displayed intervals are I ± twice the registered noise, the registered decision band "
              "translated to I. They are not calibrated confidence intervals; the noise is a propagated "
              "pool-seed variability measure, not a fitted standard error. Pool B enters twice with a minus "
              "sign, so a constant pool-B offset enters I at double weight.", "",
              "Rejection requires both students outside the band with the same sign. Both inside the band "
              "gives survival for the powered/marginal readouts; opposite signs are reported as a "
              "size-dependent interaction. A same-sign mixed-band case remains unresolved. "
              "MBPP is underpowered whatever it shows; a null there never supports additivity.", "",
              "## Registered decisions", ""]
    for cap, decision in report["decisions"].items():
        lines.append(f"- **{READOUTS[cap][1]} ({decision['role']}): {decision['status']}**. {decision['explanation']}")
    lines += ["", "## Checkpoints, achieved budgets and reuse", "",
              "Targets below are the plan's predicted completed-update coordinates, not nominal "
              "prompt-inclusive milestones. E = achieved supervised T / the plan's supervised pool size D. "
              "D is identified by exact pool-hash equality. PENDING entries show predictions only. "
              "The imperfect equality of budgets/reuse across paired corners is checked separately.", "",
              "| Student | Corner / pool | Predicted update | Achieved update | Target T | Achieved T | D | Target E | Achieved E | Status |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for student, entry in report["students"].items():
        for number, c in entry["corners"].items():
            lines.append(f"| {student} | {number} / {c['pool']} | {c['update']} | {c.get('achieved_update', '—')} | "
                         f"{c['T_supervised']} | {c.get('achieved_supervised_tokens', '—')} | {c['D_U_pool']} | "
                         f"{c['E']:.12f} | {format(c['achieved_E'], '.12f') if 'achieved_E' in c else '—'} | {c['status']} |")
    lines += ["", "### Sources and missing checkpoints", ""]
    for student, entry in report["students"].items():
        lines += [f"#### {student}: {entry['status']}", ""]
        for number, c in entry["corners"].items():
            lines.append(f"- Corner {number}: `{c['source']}`; own initial: `{c['baseline_source']}`.")
        if entry.get("missing_checkpoints"):
            lines += ["", "Missing checkpoints (including required own initial measurements):", ""]
            lines += [f"- `{p}`" for p in entry["missing_checkpoints"]]
        lines += [""]
    lines += ["## Four raw responses", "",
              "The following deltas are the four response values in I. Initial and checkpoint losses "
              "are included to audit their provenance. All printed raw values retain round-trip precision.", "",
              "| Student | Distribution | Corner | Own initial loss | Checkpoint loss | Delta |",
              "|---|---|---:|---:|---:|---:|"]
    for student, entry in report["students"].items():
        for cap in READOUTS:
            for number, corner in entry["corners"].items():
                if cap in corner["responses"]:
                    r = corner["responses"][cap]
                    lines.append(f"| {student} | {READOUTS[cap][1]} | {number} | {r['initial_loss']!r} | {r['loss']!r} | {r['delta']!r} |")
    lines += ["", "## Four residual mismatches", "",
              "| Residual | Definition | Plan / verified achieved fraction | Percent |",
              "|---|---|---|---:|"]
    definitions = ("abs(T1 − T2) / T2", "abs(T3 − T4) / min(T3,T4)",
                   "abs(E1 − E3) / min(E1,E3)", "abs(E2 − E4) / min(E2,E4)")
    for edge, definition in zip(EDGES, definitions):
        r = report["planned_residual_mismatches"][edge]
        lines.append(f"| {edge} | {definition} | {r['fraction_exact']} | {r['percent']:.12f}% |")
    complete = [s for s, e in report["students"].items() if e["status"] == "COMPLETE"]
    lines += ["", "Verified against achieved coordinates for: " + (", ".join(complete) or "none yet") +
              ". Incomplete students retain planned residuals only.", "", "## Residual contamination", "",
              "**Unimplementable total:** " + TOTAL_LIMITATION, "",
              "The stored plan uses L4 − L3 − L2 + L1, opposite to the registered I. "
              "Signed sensitivities are multiplied by −1; their absolute magnitudes are unchanged. "
              "Each row uses only stored adjacent-checkpoint slopes, multiplied by the plan's exact "
              "equivalent supervised displacement. The range is the envelope across those stored runs. "
              "JSON retains every source slope and checkpoint pair.", "",
              "| Student | Distribution | Residual | Equivalent ΔT | Signed sensitivity range for I | Max absolute sensitivity | abs(I) | abs(I) / individual sensitivity |",
              "|---|---|---|---|---|---:|---:|---:|"]
    for student, entry in report["students"].items():
        if entry["status"] != "COMPLETE":
            continue
        for row in entry["readouts"].values():
            for edge, r in row["contamination"]["residuals"].items():
                lo, hi = r["signed_range_nats"]
                ratio = r["abs_I_over_individual_sensitivity"]
                lines.append(f"| {student} | {row['benchmark']} | {edge} | "
                             f"{r['signed_equivalent_supervised_displacement_exact']} | [{lo:.10g}, {hi:.10g}] | "
                             f"{r['max_absolute_sensitivity_nats']:.10g} | {row['abs_I']:.10g} | "
                             f"{format(ratio, '.6g') if ratio is not None else 'undefined (zero sensitivity)'} |")
    lines += ["", "These individual comparisons are descriptive and cannot establish that I exceeds total "
              "contamination. No sensitivity is added to or subtracted from I or the registered noise band.", "",
              "## Fresh-sample SECONDARY check", "",
              f"`{FORMULA}`, using each saved scope file's `delta_from_update_0`. "
              "Every available scope corner's `actual_supervised_tokens` is checked against the "
              "plan's predicted corner budget before its deltas are used. Missing scope corners "
              "leave only the SECONDARY check PENDING; endpoints are never substituted.", ""]
    for student, entry in report["students"].items():
        if "fresh_sample_check" in entry:
            secondary = entry["fresh_sample_check"]
            lines.append(f"- {student}: **{secondary['status']}**. {secondary['explanation']}")
    lines += ["", "| Student | Corner / pool | Update | Predicted T | Scope actual T | Status |",
              "|---|---|---:|---:|---:|---|"]
    for student, entry in report["students"].items():
        for number, c in entry.get("fresh_sample_check", {}).get("corners", {}).items():
            lines.append(f"| {student} | {number} / {c['pool']} | {c['update']} | "
                         f"{c['predicted_supervised_tokens']} | {c.get('actual_supervised_tokens', '—')} | {c['status']} |")
    lines += ["", "### Scope sources and raw deltas", ""]
    for student, entry in report["students"].items():
        for number, c in entry.get("fresh_sample_check", {}).get("corners", {}).items():
            lines.append(f"- {student} corner {number} ({c['status']}): "
                         f"`{c.get('source', c['source_pattern'])}`; "
                         f"`delta_from_update_0`: {json.dumps(c['responses'], sort_keys=True)}.")
    lines += ["", "## Discrepancies and execution", ""]
    lines += ([f"- {f}" for f in report["failures"]] if report["failures"] else
              ["No discrepancies in the available selected checkpoints: update, achieved supervised total, "
               "processed total, pool identity, protocol, own baseline, saved delta and probe identity passed."])
    lines += ["", "CPU and standard-library analysis of saved JSON only. No fitting, training, model loading "
              "or new evaluation. SHA256 input provenance is in summary.json. Existing corner 2 is checked "
              "against the SHA256 recorded in plan.json.", "",
              "```bash", "python -B -m pytest -q tests/test_a5_corner_second_difference.py",
              "python -B analysis/a5_corner_second_difference.py",
              "# Require all corners for an explicitly named student:",
              "python -B analysis/a5_corner_second_difference.py --students gemma3-1b", "```", "",
              "Default: both students, missing checkpoints PENDING, exit zero. Explicit --students: "
              "missing primary checkpoints are still listed as PENDING but make the overall run FAILED with "
              "a non-zero exit. Malformed records and discrepancies fail in either mode. "
              "Missing SECONDARY corners remain PENDING without failing the PRIMARY; secondary "
              "discrepancies fail the run but do not change the primary readouts or registered decision. "
              "An unrequested student is NOT_REQUESTED and cannot contribute to a registered decision.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--students", nargs="+", choices=STUDENTS, default=None)
    parser.add_argument("--root", type=Path, default=ROOT, help="Input repository root")
    parser.add_argument("--output-dir", type=Path, help="Defaults to ROOT/results/a5-corner-second-difference")
    parser.add_argument("--report-path", type=Path, help="Defaults to ROOT/docs/CORNER_SECOND_DIFFERENCE_REPORT.md")
    args = parser.parse_args(argv)
    try:
        report = build(args.root, args.students)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        report = dict(schema_version="a5-corner-second-difference-v1", status="FAILED", failures=[str(exc)])
    output = args.output_dir or args.root / OUT
    paper_report = args.report_path or args.root / REPORT
    output.mkdir(parents=True, exist_ok=True)
    paper_report.parent.mkdir(parents=True, exist_ok=True)
    markdown = render(report)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "summary.md").write_text(markdown)
    paper_report.write_text(markdown)
    if "students" in report:
        print(student_table(report))
        if any(e["status"] != "COMPLETE" for e in report["students"].values()):
            print("\nThe registered decision needs both students and cannot be reached yet.")
    for failure in report["failures"]:
        print(f"FAILED: {failure}", file=sys.stderr)
    return 1 if report["status"] == "FAILED" else 0


if __name__ == "__main__":
    sys.exit(main())
