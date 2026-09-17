#!/usr/bin/env python3
"""Retrospective S3 pristine-student control; CPU arithmetic, no model execution.

The standalone paper mirror uses the same three small scoring routines as
analysis/s3_common.py. Tests check their ordering, feasibility and live parity.
Only the requested table and results/s3-selection-control outputs are written.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

try:
    from .paper_table_layout import house_style
except ImportError:  # run as a script
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from paper_table_layout import house_style

from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "s3-selection-validation"
CONTROL = "s3-selection-control"
CAPS = ("math", "code", "qa")
OBJECTIVES = (*CAPS, "multi")
BUDGETS = [i / 100 for i in range(20, 101, 5)]
METHODS = ("prune", "quant", "distill", "dense")
EPS = 1e-12
S0_ID = "dense:student_s0"
KD_ID = "distill:new_s3"
# The published plan has scrubbed host paths; its selection fields are identical.
# Pin both known byte streams instead of silently accepting a changed freeze.
PLAN_HASHES = {
    "302efd715f307d6127e1025796f8a15419820790534c5a3df44b28af7d2139d9": "original frozen plan",
    "83e9d0879757ddc7aa72b3cbec0d991f8e61ddb06a09c903e7901410ddc94a51": "published plan with scrubbed host paths",
}
SCORE_SHA256 = "11fe915389a8e2d26067ef048860531e583c68fc6cdcd420cc71b97425feb810"
NAMES = {
    "pythia-410m--step120000": "Pythia 410M",
    "pythia-1.4b--step120000": "Pythia 1.4B",
    "gemma3-1b": "Gemma 3, 1B",
    "gemma3-4b": "Gemma 3, 4B",
}
LABELS = {"math": "Mathematics", "code": "Code", "qa": "Question answering",
          "multi": "Largest increase"}
SETS = {
    "quantization": "Quantization only",
    "quantization_plus_s0": "Quantization plus pristine student",
    "quantization_plus_distilled": "Quantization plus distilled student",
    "registered": "All twelve registered candidates",
    "quantization_plus_both_students": "Quantization plus both students",
    "registered_plus_s0": "All twelve registered candidates plus pristine student",
}
DECOMPOSITION = {
    "s0_opportunity": "Q - Q0",
    "distillation_change": "Q0 - QD (signed replacement of pristine by distilled student)",
    "other_registered_methods": "QD - A (pruning and dense reference)",
    "registered_opportunity": "Q - A = s0_opportunity + distillation_change + other_registered_methods",
    "distillation_added_retaining_s0": "Q0 - Q0D (nonnegative gain from adding distilled student)",
    "other_methods_retaining_s0": "Q0D - A0",
    "augmented_opportunity": "Q - A0 = s0_opportunity + distillation_added_retaining_s0 + other_methods_retaining_s0",
}
IMPLICATION = (
    "Additional measured opportunity from this round is confined to question answering; "
    "mathematics and code benefits cannot be attributed to its distillation data, since "
    "all four students' losses worsened on those capabilities relative to their pristine anchors."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def score(value, objective, dense):
    return max(value[c] - dense[c] for c in CAPS) if objective == "multi" else value[objective]


def feasible(configs, budget):
    return [q for q in configs if q["r"] <= budget + EPS]


def choose(configs, objective, dense, field="actual"):
    if not configs:
        return None
    return min(configs, key=lambda q: (score(q[field], objective, dense),
               METHODS.index(q["method"]), q["r"], q["id"]))


def input_directory(root):
    # Prefer the published mirror to potentially stale bootstrap copies there.
    mirror = root / "data_mirror" / EXPERIMENT
    return mirror if mirror.is_dir() else root / "results" / EXPERIMENT


def table_path(root):
    return root / ("paper/tables/s3_control.tex" if (root / "data_mirror").is_dir()
                   else "paper/paper/tables/s3_control.tex")


class Inputs:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.hashes = {}

    def read(self, name, sealed=False):
        path = self.directory / name
        digest = sha(path)
        self.hashes[f"results/{EXPERIMENT}/{name}"] = digest
        if sealed:
            require(Path(str(path) + ".sha256").read_text().strip() == digest,
                    f"Input seal changed: {path}")
        return json.loads(path.read_text())


def pristine_candidate(ref, anchor):
    """Use the plan's nominal text-decoder matrix ratio for BOTH student states."""
    ratio = ref["student"]["N0"] / ref["N0"]
    distilled = [q for q in ref["candidates"] if q["method"] == "distill"]
    require(len(distilled) == 1 and distilled[0]["r"] == ratio,
            "Merged student storage differs from plan student.N0 / reference.N0")
    require(S0_ID not in {q["id"] for q in ref["candidates"]}, "S0 was already registered")
    return dict(id=S0_ID, method="dense", r=ratio, actual=copy.deepcopy(anchor["losses"]),
                retrospective=True)


def candidate_sets(configs, s0):
    quant = [q for q in configs if q["method"] == "quant"]
    distilled = [q for q in configs if q["method"] == "distill"]
    return dict(quantization=quant, quantization_plus_s0=quant + [s0],
                quantization_plus_distilled=quant + distilled, registered=configs,
                quantization_plus_both_students=quant + [s0] + distilled,
                registered_plus_s0=configs + [s0])


def oracle(configs, objective, dense, budget):
    available = feasible(configs, budget)
    best = choose(available, objective, dense)
    return {"candidate": best["id"] if best else None,
            "loss": score(best["actual"], objective, dense) if best else None,
            "feasible": bool(available), "feasible_ids": [q["id"] for q in available],
            "reason_if_infeasible": None if available else "No candidate meets r <= budget + 1e-12."}


def evaluate_cell(configs, s0, dense, objective, budget, frozen_map):
    oracles = {name: oracle(qs, objective, dense, budget)
               for name, qs in candidate_sets(configs, s0).items()}
    q, q0, qd, a, q0d, a0 = [oracles[k]["loss"] for k in SETS]

    def gap(left, right):
        return left - right if left is not None and right is not None else None

    decomposition = dict(s0_opportunity=gap(q, q0), distillation_change=gap(q0, qd),
        other_registered_methods=gap(qd, a), registered_opportunity=gap(q, a),
        distillation_added_retaining_s0=gap(q0, q0d), other_methods_retaining_s0=gap(q0d, a0),
        augmented_opportunity=gap(q, a0))
    available = {q["id"]: q for q in feasible(configs, budget)}
    policies = {}
    for name in ("rule", "quant-only"):
        selected = frozen_map[name]["selected"]
        config = available.get(selected)
        valid = config is not None and (name != "quant-only" or config["method"] == "quant")
        loss = score(config["actual"], objective, dense) if valid else None
        policies[name] = dict(selected=selected, feasible=valid, actual_loss=loss,
            regret_registered=gap(loss, a), regret_registered_plus_s0=gap(loss, a0),
            reason_if_infeasible=None if valid else "Frozen policy made no feasible selection.")
    return dict(retrospective=True, objective=objective, budget=budget, oracles=oracles,
                decomposition=decomposition, policies=policies)


def denominator(rows, predicate, reason):
    included = [r["budget"] for r in rows if predicate(r)]
    excluded = [{"budget": r["budget"], "reason": reason(r)} for r in rows if not predicate(r)]
    return dict(n_total=len(rows), n_included=len(included), included_budgets=included, excluded=excluded)


def aggregate(rows):
    common = denominator(rows,
        lambda r: all(o["feasible"] for o in r["oracles"].values())
                  and all(p["feasible"] for p in r["policies"].values()),
        lambda r: "Infeasible candidate sets: " + ", ".join(k for k, v in r["oracles"].items() if not v["feasible"])
                  + "; infeasible frozen policies: " + ", ".join(k for k, v in r["policies"].items() if not v["feasible"]))
    paired = [r for r in rows if r["budget"] in common["included_budgets"]]
    result = dict(retrospective=True, objective=rows[0]["objective"], n_budgets=len(rows),
                  common_denominator=common, candidate_sets={}, policies={})
    for name in SETS:
        d = denominator(rows, lambda r: r["oracles"][name]["feasible"],
                        lambda r: r["oracles"][name]["reason_if_infeasible"])
        defined = [r["oracles"][name]["loss"] for r in rows if r["oracles"][name]["feasible"]]
        result["candidate_sets"][name] = dict(denominator=d,
            mean_over_feasible_budgets=mean(defined) if defined else None,
            mean_over_common_budgets=mean(r["oracles"][name]["loss"] for r in paired) if paired else None)
    result["mean_decomposition"] = {
        name: mean(r["decomposition"][name] for r in paired) if paired else None for name in DECOMPOSITION}
    for name in ("rule", "quant-only"):
        d = denominator(rows, lambda r: r["policies"][name]["feasible"],
                        lambda r: r["policies"][name]["reason_if_infeasible"])
        result["policies"][name] = dict(denominator=d, common_denominator=copy.deepcopy(common),
            mean_regret_registered=mean(r["policies"][name]["regret_registered"] for r in paired) if paired else None,
            mean_regret_registered_plus_s0=mean(r["policies"][name]["regret_registered_plus_s0"] for r in paired) if paired else None)
    return result


def analyze(root=None, directory=None):
    root = Path(root) if root is not None else ROOT
    inputs = Inputs(directory if directory is not None else input_directory(root))
    plan = inputs.read("plan_v2.json")
    frozen_score = inputs.read("score_v2.json")
    plan_hash = sha(inputs.directory / "plan_v2.json")
    require(plan_hash in PLAN_HASHES, "Unknown or modified frozen plan")
    require(sha(inputs.directory / "score_v2.json") == SCORE_SHA256, "Frozen score changed")
    require(frozen_score["status"] == "COMPLETE" and plan["final_freeze"], "S3 is not complete/frozen")
    require(plan["budgets"] == BUDGETS and len(plan["references"]) == 4, "Wrong registered grid")
    maps = {(m["reference"], m["objective"], m["budget"]): m for m in plan["maps"]}
    old_rows = {(r["reference"], r["objective"], r["budget"]): r for r in frozen_score["rows"]}
    require(len(maps) == len(old_rows) == 272, "Missing or duplicate frozen cells")
    references, rows = [], []
    for index, ref in enumerate(plan["references"]):
        units = "nats per native " + ref["family"] + " token"
        anchors = {}
        for role in ("reference", "student"):
            name = f"anchors/{ref['id']}__{role}.json"
            anchor = inputs.read(name, sealed=True)
            require(anchor["pristine"] and not anchor["training_performed"] and not anchor["compression_applied"],
                    "Anchor is not pristine")
            require(anchor["reference"] == ref["id"] and anchor["role"] == role, "Wrong anchor identity")
            require(anchor["probe_sha256"] == plan["protocol"]["probe_sha256"] and
                    anchor["registration_sha256"] == plan["registration_sha256"], "Wrong anchor protocol")
            owner = ref if role == "reference" else ref["student"]
            require(anchor["losses"] == owner["dense"] and anchor["model"] == owner["model"]
                    and anchor["revision"] == owner["revision"], "Anchor differs from plan")
            require(anchor["units"] == units and anchor["family"] == ref["family"], "Wrong anchor units")
            anchors[role] = anchor
        require(anchors["student"]["tokenizer_sha256"] == anchors["reference"]["tokenizer_sha256"],
                "Student and reference tokenizers differ")
        configs = copy.deepcopy(ref["candidates"])
        require(len(configs) == 12 and len({q["id"] for q in configs}) == 12, "Incomplete registered panel")
        for config in configs:
            name = (f"students/{ref['id']}/measurement.json" if config["method"] == "distill"
                    else f"measurements/{ref['id']}/{config['id'].replace(':', '__')}.json")
            value = inputs.read(name, sealed=True)
            require(value["reference"] == ref["id"] and value["candidate"] == config["id"], "Wrong measurement")
            require(value["plan_sha256"] == frozen_score["plan_sha256"] and
                    value["probe_sha256"] == plan["protocol"]["probe_sha256"] and value["fresh_s3_run"],
                    "Measurement is not from the frozen S3 round")
            require(value["family"] == ref["family"] and value["units"] == units, "Wrong measurement units")
            require(set(value["losses"]) == set(CAPS) and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0
                for x in value["losses"].values()), "Invalid measured losses")
            config.update(actual=value["losses"], measurement_file=f"results/{EXPERIMENT}/{name}")
        dense = next(q["actual"] for q in configs if q["id"] == "dense:source")
        require(dense == anchors["reference"]["losses"], "Reference anchor/measurement mismatch")
        s0 = pristine_candidate(ref, anchors["student"])
        s0["measurement_file"] = f"results/{EXPERIMENT}/anchors/{ref['id']}__student.json"
        kd = next(q for q in configs if q["id"] == KD_ID)
        transfer = {c: dict(s0_loss=s0["actual"][c], distilled_loss=kd["actual"][c],
            distilled_minus_s0=kd["actual"][c] - s0["actual"][c], units=units,
            anchor_file=s0["measurement_file"], measurement_file=kd["measurement_file"],
            json_pointer=f"/losses/{c}", retrospective=True) for c in CAPS}
        rr = []
        for objective in OBJECTIVES:
            for budget in BUDGETS:
                key = ref["id"], objective, budget
                row = evaluate_cell(configs, s0, dense, objective, budget, maps[key])
                old = old_rows[key]
                require(row["oracles"]["registered"]["candidate"] == old["oracle_id"] and
                        row["oracles"]["registered"]["loss"] == old["oracle_loss"] and
                        row["oracles"]["quantization"]["candidate"] == old["quant_oracle_id"] and
                        row["decomposition"]["registered_opportunity"] == old["opportunity"],
                        "Recomputed registered oracle differs from frozen score")
                for policy in ("rule", "quant-only"):
                    current, prior = row["policies"][policy], old["policies"][policy]
                    require(current["selected"] == prior["selected"] and current["feasible"] == prior["feasible"]
                            and current["actual_loss"] == prior["actual_loss"]
                            and current["regret_registered"] == prior["regret"], "Frozen policy/score mismatch")
                row.update(reference=ref["id"], family=ref["family"], units=units)
                rr.append(row)
        summaries = [aggregate([r for r in rr if r["objective"] == c]) for c in OBJECTIVES]
        require(all(s["common_denominator"]["included_budgets"] == plan["paired_decision_budgets"]
                    for s in summaries), "Retrospective common denominator differs from paired grid")
        references.append(dict(retrospective=True, reference=ref["id"], family=ref["family"], units=units,
            student_model=ref["student"]["model"], student_revision=ref["student"]["revision"],
            storage=dict(student_N0=ref["student"]["N0"], reference_N0=ref["N0"],
                s0_ratio=s0["r"], distilled_ratio=kd["r"], source=f"results/{EXPERIMENT}/plan_v2.json",
                numerator_pointer=f"/references/{index}/student/N0", denominator_pointer=f"/references/{index}/N0",
                student_feasible_budgets=[b for b in BUDGETS if feasible([s0], b)],
                student_infeasible_budgets=[b for b in BUDGETS if not feasible([s0], b)],
                n_student_feasible=sum(bool(feasible([s0], b)) for b in BUDGETS), n_budgets=len(BUDGETS)),
            transfer=transfer, candidates=configs + [s0],
            candidate_sets={k: [q["id"] for q in v] for k, v in candidate_sets(configs, s0).items()},
            objectives=summaries))
        rows.extend(rr)
    return dict(status="RETROSPECTIVE_ONLY", retrospective=True, analysis="A19 pristine-student control",
        budgets=BUDGETS, n_budgets=17, n_references=4, n_objectives=4, n_cells=len(rows),
        input_directory=str(inputs.directory.relative_to(root)) if inputs.directory.is_relative_to(root) else str(inputs.directory),
        input_sha256=inputs.hashes, plan_copy=PLAN_HASHES[plan_hash],
        frozen_plan_sha256=frozen_score["plan_sha256"], frozen_score_sha256=SCORE_SHA256,
        frozen_score_reproduction="All 272 oracle cells and both frozen policies reproduce exactly.",
        storage_rule="r(S0) = references[i].student.N0 / references[i].N0 = r(S_KD). These are the plan's nominal text-decoder matrix counts. "
                     "Merging LoRA preserves the student's matrix dimensions, so pristine and merged students have the same nominal storage. "
                     "This convention does not count checkpoint file sizes, adapter overhead, embeddings, or model-name parameter totals.",
        tie_rule="Exact loss, then pruning, quantization, distillation, dense; then smaller r, then lexicographic candidate id. "
                 "S0 has method=dense because it is pristine; its id is dense:student_s0. The priority matches analysis/s3_common.py.",
        feasibility_rule="r <= budget + 1e-12; no loss threshold. Infeasible losses and regrets are null, never zero.",
        objective_rule="Single capabilities use measured completion-token-weighted cross entropy. Largest increase is max_c(L_candidate,c - L_dense_reference,c) before minimization.",
        frozen_policy_note="The rule's own selection was frozen without S0 in the candidate set. Both policies retain the selected ids in plan_v2.json maps; S0 enters only this retrospective comparison. The quantization-only policy is prediction-selected, distinct from the measured quantization oracle.",
        averaging_rule="All decomposition and paired policy means use the same 16 of 17 budgets, 0.25 through 1.00 in steps of 0.05, per reference and objective. The 0.20 budget is excluded because quantization is infeasible. Budgets are correlated operating points, never independent replicates; no confidence intervals or cross-family pooling.",
        oracle_symbols=dict(Q="quantization", Q0="quantization_plus_s0", QD="quantization_plus_distilled",
                            A="registered", Q0D="quantization_plus_both_students", A0="registered_plus_s0"),
        decomposition_definitions=DECOMPOSITION, candidate_set_labels=SETS,
        interpretation=IMPLICATION,
        attribution_limit="The before/after comparison describes this measured training round; it does not isolate the causal effect of data from training, optimization, or teacher choice, and uses one new student run per reference.",
        references=references, rows=rows)


def fmt(value):
    return "--" if value is None else f"{value:.6f}"


def markdown(result):
    lines = ["# Retrospective S3 pristine-student control (A19)", "", "**RETROSPECTIVE ONLY. CPU arithmetic on existing measurements.**", ""]
    for key in ("frozen_policy_note", "storage_rule", "tie_rule", "feasibility_rule", "objective_rule", "averaging_rule"):
        lines += [result[key], ""]
    lines += ["The four requested oracles are Q (quantization), Q0 (quantization plus pristine student), QD (quantization plus distilled student), and A (all twelve registered candidates). "
              "Q0D retains both students; A0 adds the pristine student to all registered candidates. The latter two expose gains that survive retaining S0.", ""]
    lines += [f"- `{key}`: {value}." for key, value in DECOMPOSITION.items()]
    lines += ["", "A negative distillation change means that replacing the pristine student loses opportunity. "
              "The nonnegative gain retaining S0 is reported separately. Other registered methods can contribute too, so their residual is never credited to distillation.", "",
              "## Retrospective storage and transfer", "",
              "Transfer is L(S_KD) − L(S0); positive values are degradation. Every value is in nats per native token of its stated family. "
              "Each loss is read from `/losses/<capability>` in the linked anchor or measurement JSON. Full precision and input SHA256 hashes are in summary.json.", ""]
    for ref in result["references"]:
        st = ref["storage"]
        lines += [f"### {NAMES[ref['reference']]} ({ref['reference']}; {ref['units']})", "",
            f"Pristine student: `{ref['student_model']}@{ref['student_revision']}`. "
            f"Nominal ratio: {st['student_N0']} / {st['reference_N0']} = {st['s0_ratio']:.17g} for both student states. "
            f"Student feasible: {st['n_student_feasible']}/17 budgets; excluded for student storage: {st['student_infeasible_budgets']}.", "",
            "| Capability | Pristine loss | Distilled loss | Transfer | Anchor source | Measurement source |",
            "|---|---:|---:|---:|---|---|"]
        for cap in CAPS:
            t = ref["transfer"][cap]
            links = [f"[{Path(t[k]).name}](../{t[k].removeprefix('results/')})" for k in ("anchor_file", "measurement_file")]
            lines.append(f"| {LABELS[cap]} | {fmt(t['s0_loss'])} | {fmt(t['distilled_loss'])} | "
                         f"{t['distilled_minus_s0']:+.6f} | {links[0]} | {links[1]} |")
    lines += ["", "## Retrospective opportunity decomposition", "",
              "Each row averages exactly 16 common budgets (25%–100%); all numbers use the reference family's native-token nats. "
              "Before rounding, pristine + signed distillation change + other methods = registered opportunity. "
              "Pristine + added distillation retaining S0 + other methods retaining S0 = augmented opportunity.", "",
              "| Reference | Objective | Budgets | Pristine opportunity | Signed distillation change | Other methods | Registered opportunity | Added distillation retaining S0 | Other methods retaining S0 | Augmented opportunity |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ref in result["references"]:
        for s in ref["objectives"]:
            d = s["mean_decomposition"]
            values = [fmt(d[k]) for k in DECOMPOSITION]
            lines.append(f"| {NAMES[ref['reference']]} | {LABELS[s['objective']]} | {s['common_denominator']['n_included']}/17 | " + " | ".join(values) + " |")
    lines += ["", "## Retrospective denominators and measured oracle means", "",
              "The excluded-budget list below applies separately to each objective. All six sets have six quantization candidates; "
              "the complete registered set has twelve candidates and its augmentation has thirteen. "
              "An empty excluded list means all seventeen budgets are feasible.", "",
              "| Reference | Candidate set | Candidate count | Feasible / 17 | Excluded budgets and reason |",
              "|---|---|---:|---:|---|"]
    for ref in result["references"]:
        for name in SETS:
            d = ref["objectives"][0]["candidate_sets"][name]["denominator"]
            excluded = "; ".join(f"{x['budget']:.0%}: {x['reason']}" for x in d["excluded"]) or "None"
            lines.append(f"| {NAMES[ref['reference']]} | {SETS[name]} | {len(ref['candidate_sets'][name])} | {d['n_included']}/17 | {excluded} |")
    lines += ["", "Oracle means below use the common 16 budgets even when a set is feasible at 20%. "
              "summary.json also records means over each set's own feasible budgets with its explicit denominator.", "",
              "| Reference | Objective | Q | Q0 | QD | A | Q0D | A0 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for ref in result["references"]:
        for s in ref["objectives"]:
            lines.append(f"| {NAMES[ref['reference']]} | {LABELS[s['objective']]} | " +
                         " | ".join(fmt(s["candidate_sets"][k]["mean_over_common_budgets"]) for k in SETS) + " |")
    lines += ["", "## Retrospective frozen-policy regret", "",
              "The selected candidates remain frozen. Regret is measured selected loss minus A or A0 as labeled. "
              "Each mean uses the common 16 budgets. Gemma 3, 1B's rule selects at 17/17 budgets; "
              "the other rules and all quantization-only policies select at 16/17. The sole excluded budget for pairing is 20%. "
              "All 272 budget cells, including oracle candidate identities, measured losses, both policy selections, "
              "infeasibility reasons and regrets, are retained in summary.json.", "",
              "| Reference | Objective | Mean budgets | Rule versus A | Quantization policy versus A | Rule versus A0 | Quantization policy versus A0 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for ref in result["references"]:
        for s in ref["objectives"]:
            values = [fmt(s["policies"][p][k]) for k in ("mean_regret_registered", "mean_regret_registered_plus_s0") for p in ("rule", "quant-only")]
            lines.append(f"| {NAMES[ref['reference']]} | {LABELS[s['objective']]} | 16/17 | " + " | ".join(values) + " |")
    lines += ["", "## Retrospective interpretation and provenance", "", result["interpretation"], "",
              result["attribution_limit"], "", result["frozen_score_reproduction"], "",
              f"Frozen plan SHA256: `{result['frozen_plan_sha256']}`.", "",
              f"Frozen score SHA256: `{result['frozen_score_sha256']}`.", "",
              "Regenerate with `PYTHONDONTWRITEBYTECODE=1 python analysis/s3_control_table.py`. "
              "The generator reads the validation directory and never writes there. "
              "The public generator reads its existing data mirror, whose plan scrubs host paths; "
              "both known plan byte streams are pinned and the registered score is reproduced exactly.", ""]
    return "\n".join(lines)


def render(result=None):
    return house_style(_render(result))


def _render(result=None):
    result = analyze() if result is None else result
    rows = []
    for ref in result["references"]:
        if rows:
            rows.append(r"\midrule")
        for i, s in enumerate(ref["objectives"]):
            d = s["mean_decomposition"]
            values = [fmt(d[k]) for k in ("s0_opportunity", "distillation_change",
                                          "other_registered_methods", "registered_opportunity")]
            rows.append(f"{NAMES[ref['reference']] if i == 0 else ''} & {LABELS[s['objective']]} & " +
                        " & ".join(values) + r" \\")
    caption = (
        "This analysis is retrospective. The pristine student $S_0$ was not in the frozen candidate set, "
        "and the rule's selections remain unchanged. Both student states use the registered nominal "
        "student-to-reference matrix storage ratio. Pristine opportunity is the best measured quantization "
        "loss minus the best loss after adding the pristine student. Distillation change is the signed "
        "difference between the best loss with the pristine student and the best loss with the distilled "
        "student, with quantization available in both cases. A negative change means lost opportunity. "
        "Other methods measure the further gain from registered pruning and the dense reference. "
        "These three columns sum to registered opportunity before rounding. All means use the same sixteen of seventeen "
        "budgets, from 25\\% to 100\\%; 20\\% is excluded because quantization is infeasible. "
        "Sets containing either student are feasible at all seventeen budgets only for Gemma 3, 1B; "
        "all other sets are feasible at sixteen. Budgets are operating points, not independent replicates. "
        "Losses are in nats per native token within each family. Largest increase takes the maximum "
        "capability loss increase relative to the dense reference before selecting a candidate. "
        "Both Pythia references use step 120000.")
    return ("% RETROSPECTIVE ONLY. Generated by analysis/s3_control_table.py; do not edit.\n"
            "\\begin{table}[!htbp]\n\\centering\\footnotesize\n"
            f"\\caption{{{caption}}}\n\\label{{tab:s3-retrospective-control}}\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llrrrr@{}}\n\\toprule\n"
            "Reference & Objective & \\shortstack{Pristine\\\\opportunity} & "
            "\\shortstack{Distillation\\\\change} & \\shortstack{Other\\\\methods} & "
            "\\shortstack{Registered\\\\opportunity} \\\\\n\\midrule\n" +
            "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")


def generate(root=None):
    root = Path(root) if root is not None else ROOT
    result = analyze(root)
    output = root / "results" / CONTROL
    table = table_path(root)
    rendered = render(result)
    output.mkdir(parents=True, exist_ok=True)
    table.parent.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(canonical(result))
    (output / "summary.md").write_text(markdown(result))
    (output / "s3_control.tex").write_text(rendered)
    table.write_text(rendered)
    # Confirm every input byte stream remains identical after output generation.
    for name, digest in result["input_sha256"].items():
        path = input_directory(root) / name.removeprefix(f"results/{EXPERIMENT}/")
        require(sha(path) == digest, f"Read-only input changed: {name}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = generate()
    print(f"RETROSPECTIVE ONLY: wrote {CONTROL}/summary.json, summary.md, s3_control.tex and {table_path(ROOT)}")
    print(result["interpretation"])


if __name__ == "__main__":
    main()
