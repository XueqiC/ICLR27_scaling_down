#!/usr/bin/env python3
"""Selection value, retrospective (rebuttal round, 2026-11-07). CPU, recorded losses only.

Run: python -B analysis/r1_selection_value.py
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import numpy as np
from analysis import s3_policy_ablation as s3
from analysis import v53_prune_dev as p53
from analysis import v55_quant_group_fit as v55
from analysis.paper_table_layout import house_style, table_layout

ID = "r1-selection-value"
STATUS = "retrospective (rebuttal round, 2026-11-07)"
CAPS, OBJECTIVES = s3.CAPS, s3.OBJECTIVES


def read(root, relative, hashes):
    raw = (root / relative).read_bytes()
    hashes[str(relative)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def descriptor(q):
    return {k: q[k] for k in ("id", "method", "r", "config", "d", "bit", "group_size", "target_state") if k in q} if q else None


def analyze_cell(reference, family, objective, budget, dense, configs, chosen, fields):
    """The gap is to the best candidate of a DIFFERENT method, not an arbitrary runner-up.

    Pair errors are descriptive. A uniform bound over every feasible candidate is
    checked separately; exact configuration recovery additionally needs the gap
    to the runner-up among ALL configurations.
    """
    available = s3.common.feasible(configs, budget)
    oracle = s3.common.choose(available, objective, dense, "actual")
    score = lambda q, field="actual": s3.common.score(q[field], objective, dense)
    other = s3.common.choose([q for q in available if q["method"] != oracle["method"]],
                            objective, dense, "actual") if oracle else None
    second = s3.common.choose([q for q in available if q["id"] != oracle["id"]],
                             objective, dense, "actual") if oracle else None
    gap = score(other)-score(oracle) if other else None
    config_gap = score(second)-score(oracle) if second else None
    priority = chosen["priority"]
    row = dict(reference=reference, family=family, objective=objective, budget=budget,
        n_feasible=len(available), oracle=descriptor(oracle), second_method=descriptor(other),
        oracle_loss=score(oracle) if oracle else None, second_method_loss=score(other) if other else None,
        cross_method_gap=gap, configuration_gap=config_gap,
        rank_switch=bool(oracle["method"] != priority["method"]) if oracle and priority else None,
        policies={}, candidates=[])
    for q in available:
        row["candidates"].append(dict(**descriptor(q), actual_loss=score(q),
            predicted={name: score(q, field) for name, field in fields.items()}))
    for name, q in chosen.items():
        record = dict(chosen=descriptor(q), actual_loss=score(q) if q else None,
                      regret=score(q)-score(oracle) if q and oracle else None)
        if name in fields and oracle:
            field = fields[name]
            errors = {t["id"]: abs(score(t, field)-score(t)) for t in available}
            maximum = max(errors.values())
            chosen_error = errors[q["id"]] if q else None
            oracle_error = errors[oracle["id"]]
            record.update(chosen_abs_error=chosen_error, oracle_abs_error=oracle_error,
                max_feasible_abs_error=maximum,
                chosen_below_half_gap=chosen_error < gap/2 if gap is not None and q else None,
                oracle_below_half_gap=oracle_error < gap/2 if gap is not None else None,
                pair_below_half_gap=max(chosen_error, oracle_error) < gap/2 if gap is not None and q else None,
                uniform_method_certificate=2*maximum < gap if gap is not None else None,
                uniform_configuration_certificate=2*maximum < config_gap if config_gap is not None else None,
                recovered_method=q["method"] == oracle["method"] if q else None,
                recovered_configuration=q["id"] == oracle["id"] if q else None)
            if record["uniform_method_certificate"]:
                assert record["recovered_method"]
            if record["uniform_configuration_certificate"]:
                assert record["recovered_configuration"]
        row["policies"][name] = record
    row["median_regret_gain_over_priority"] = (row["policies"]["priority"]["regret"] -
        row["policies"]["median"]["regret"]) if oracle else None
    return row


def independent(root, hashes, v64):
    base = Path("results/s3-selection-validation")
    plan = read(root, base / "plan_v2.json", hashes)
    sealed = read(root, base / "score_v2.json", hashes)
    ablation = read(root, "results/s3-policy-ablation/summary.json", hashes)
    models = read(root, base / "inputs/locked_models.json", hashes)
    v70 = read(root, "results/v70-distill-confirm/freeze.json", hashes)
    priority = s3.development_priority(v64)
    predictions = {(r["reference"], r["candidate"]): r for r in plan["predictions"]}
    maps = {(r["reference"], r["objective"], r["budget"]): r for r in plan["maps"]}
    old = {(r["reference"], r["objective"], r["budget"]): r for r in ablation["rows"]}
    sealed = {(r["reference"], r["objective"], r["budget"]): r for r in sealed["rows"]}
    rows = []
    for ref in plan["references"]:
        configs = copy.deepcopy(ref["candidates"])
        preds = s3.predictions(ref, models, v70)
        for q in configs:
            path = s3.measure_path(ref, q, base)
            q["actual"] = s3.common.losses(read(root, path, hashes)["losses"])
            q["median"], q["relations"] = preds[q["id"]]["median"], preds[q["id"]]["relations"]
            q["rule"] = {c: predictions[ref["id"], q["id"]]["losses"][c]["absolute_loss"] for c in CAPS}
        for objective in OBJECTIVES:
            for budget in plan["budgets"]:
                available = s3.common.feasible(configs, budget)
                lookup = {q["id"]: q for q in available}
                key = ref["id"], objective, budget
                m = maps[key]
                chosen = {"priority": s3.choose_priority(available, objective, budget, priority),
                    "priority_count": s3.choose_priority(available, objective, budget, priority, "order_count"),
                    "median": s3.common.choose(available, objective, ref["dense"], "median"),
                    "relations": s3.common.choose(available, objective, ref["dense"], "relations"),
                    "rule": lookup.get(m["rule"]["selected"]),
                    "quant-only": lookup.get(m["quant-only"]["selected"])}
                row = analyze_cell(ref["id"], ref["family"], objective, budget, ref["dense"], configs, chosen,
                                   {"median": "median", "rule": "rule", "relations": "relations"})
                for name in chosen:
                    got, expected = row["policies"][name]["regret"], old[key]["policies"][name]["regret"]
                    assert (got is None and expected is None) or abs(got-expected) < 1e-9
                    if name in ("rule", "quant-only"):
                        expected = sealed[key]["policies"][name]["regret"]
                        assert (got is None and expected is None) or abs(got-expected) < 1e-9
                    selected = chosen[name]["id"] if chosen[name] else None
                    assert selected == old[key]["policies"][name]["selected"]
                rows.append(row)
    return rows


def heldout_medians(state, fold, states, students):
    """Use the exact recorded training cohorts, excluding the target and selectable students."""
    configs = copy.deepcopy(state["configs"])
    for q in configs:
        q["median"], q["rule"] = {}, q["predicted"]
        for c in CAPS:
            if q["method"] == "dense":
                delta = 0.0
            elif q["law"] == "prune_power":
                delta = p53.linear_curve(fold["prune"]["models"][c]["median_curve"]["anchors"], q["d"])
            elif q["law"] == "quant_group":
                delta = v55.interpolate(fold["quant_group"]["models"][c]["median"]["anchors"], q["config"])
            elif q["law"] == "quant_channel":
                cohort = set(fold["quant_channel"]["train_states"])
                assert state["tag"] not in cohort
                values = [t["actual"][c]-t["anchor"][c] for s in states if s["tag"] in cohort
                          for t in s["configs"] if t["law"] == q["law"] and t["bit"] == q["bit"]]
                assert values
                delta = float(np.median(values))
            elif q["law"] == "distill_linear":
                cohort = fold["distill_linear"]["train_states"]
                assert state["tag"] not in cohort and q["target_state"] not in cohort
                # The fixed student baseline is the training constant (mean delta), as S3.
                delta = statistics.mean(students[t]["delta"][c] for t in cohort)
            else:
                raise ValueError(q["law"])
            q["median"][c] = q["anchor"][c]+delta
    return configs


def heldout(v64):
    rows, cohorts = [], {}
    folds = {f["held_out"]: f for f in v64["folds"]}
    cells = {(r["state"], o, r["budget"]): r for o, rs in v64["cells"].items() for r in rs}
    for state in v64["states"]:
        # Priority is also held out: never use this reference's oracle to rank its methods.
        dev = {"cells": {o: [r for r in rs if r["state"] != state["tag"]] for o, rs in v64["cells"].items()}}
        priority = s3.development_priority(dev)
        configs = heldout_medians(state, folds[state["tag"]], v64["states"], v64["students"])
        cohorts[state["tag"]] = {k: v["train_states"] for k, v in folds[state["tag"]].items() if k != "held_out"}
        for objective in OBJECTIVES:
            for budget in v64["budgets"]:
                available = s3.common.feasible(configs, budget)
                chosen = {"priority": s3.choose_priority(available, objective, budget, priority),
                    "median": s3.common.choose(available, objective, state["dense"], "median"),
                    "rule": s3.common.choose(available, objective, state["dense"], "rule")}
                row = analyze_cell(state["tag"], "pythia", objective, budget, state["dense"], configs, chosen,
                                   {"median": "median", "rule": "rule"})
                previous = cells[state["tag"], objective, budget]
                assert row["oracle"]["id"] == previous["policies"]["ORACLE"]["config_id"]
                assert row["policies"]["rule"]["chosen"]["id"] == previous["policies"]["MAP"]["config_id"]
                assert abs(row["policies"]["rule"]["regret"]-previous["policies"]["MAP"]["regret"]) < 1e-9
                rows.append(row)
    return rows, cohorts


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def aggregate(rows):
    out = {}
    for objective in OBJECTIVES:
        all_rows = [r for r in rows if r["objective"] == objective]
        valid = [r for r in all_rows if r["oracle"]]
        switched = [r for r in valid if r["rank_switch"]]
        gaps = [r for r in valid if r["cross_method_gap"] is not None]
        out[objective] = dict(n_total=len(all_rows), n_feasible=len(valid), n_switch=len(switched),
            switch_frequency=len(switched)/len(valid) if valid else None, n_cross_method_gap=len(gaps),
            error_conditions={key: dict(count=sum(r["policies"]["median"][key] for r in gaps),
                denominator=len(gaps), frequency=mean([r["policies"]["median"][key] for r in gaps]))
                for key in ("chosen_below_half_gap", "oracle_below_half_gap", "pair_below_half_gap", "uniform_method_certificate")},
            by_family={})
        for family in sorted({r["family"] for r in valid}):
            cr = [r for r in valid if r["family"] == family]
            out[objective]["by_family"][family] = dict(n=len(cr), n_switch=sum(r["rank_switch"] for r in cr),
                gain_switch=mean([r["median_regret_gain_over_priority"] for r in cr if r["rank_switch"]]),
                gain_non_switch=mean([r["median_regret_gain_over_priority"] for r in cr if not r["rank_switch"]]),
                mean_regret={name: mean([r["policies"][name]["regret"] for r in cr]) for name in cr[0]["policies"]},
                n_policy_feasible={name: sum(r["policies"][name]["regret"] is not None for r in cr) for name in cr[0]["policies"]})
    return out


def build(root=ROOT):
    hashes = {}
    v64 = read(root, "results/v64-selection-feasible/summary.json", hashes)
    rows = independent(root, hashes, v64)
    hrows, cohorts = heldout(v64) if v64.get("states") and v64.get("folds") else ([], {})
    return dict(id=ID, status=STATUS, input_sha256=hashes,
        definitions={"rank_switch": "Oracle method differs from the development-priority chosen method.",
            "cross_method_gap": "Best loss among methods other than the oracle method minus oracle loss; null if only one method is feasible.",
            "errors": "Absolute objective prediction errors on the chosen and oracle candidates; multi uses max capability increase before taking absolute error.",
            "condition": "Pair errors below gap/2 are descriptive, not a uniform guarantee. 2*max_feasible_error < cross_method_gap guarantees the correct METHOD. Exact configuration recovery requires the separately reported configuration_gap to every other candidate.",
            "priority": "S3 development oracle win-rate order and largest feasible storage; the H.1 priority excludes the target reference's cells. S3's candidate_methods eligibility convention is retained.",
            "heldout_rule": "H.1 rule column replays its recorded law-based MAP, not the later S3 frozen final rule. H.1 medians are added retrospectively from each recorded training fold; student uses the training mean constant as S3.",
            "averaging": "All feasible budgets retained, including single-method budgets; infeasible cells retained with nulls. Native-nat means reported separately by family; switch counts and error-condition rates may pool families.",
            "H1": "V64 retains V60 candidate predictions and fixes infeasible policy handling; do not double count the two versions."},
        panels={"independent": dict(rows=rows, by_objective=aggregate(rows)),
                "heldout_pythia_H1": dict(status="recorded" if hrows else "per-cell data absent", rows=hrows,
                                         training_cohorts=cohorts, by_objective=aggregate(hrows))})


def fmt(value):
    return "--" if value is None else f"{value:.4f}"


def render(summary):
    md = [f"# {ID}", "", STATUS, "", *summary["definitions"].values(), ""]
    for panel, p in summary["panels"].items():
        md += [f"## {panel}", "", "| Objective | Switches/feasible | Median chosen < gap/2 | Both errors < gap/2 | Uniform method bound |",
               "|---|---:|---:|---:|---:|"]
        body = []
        for obj, a in p["by_objective"].items():
            conditions = a["error_conditions"]
            count = lambda k: f"{conditions[k]['count']}/{conditions[k]['denominator']}"
            md.append(f"| {obj} | {a['n_switch']}/{a['n_feasible']} | {count('chosen_below_half_gap')} | " +
                      f"{count('pair_below_half_gap')} | {count('uniform_method_certificate')} |")
            for family, f in a["by_family"].items():
                body.append(" & ".join([family, obj, f"{f['n_switch']}/{f['n']}", fmt(f["gain_switch"]),
                    fmt(f["gain_non_switch"]), count("pair_below_half_gap"), count("uniform_method_certificate")]) + r" \\")
        md += ["", "| Family | Objective | Switches | Gain in switch cells | Gain in non-switch cells | Pair condition (pooled) | Uniform condition (pooled) |",
               "|---|---|---:|---:|---:|---:|---:|", *["| " + x.removesuffix(r" \\").replace(" & ", " | ")+" |" for x in body], ""]
        md += ["| Reference | Objective | Budget | Oracle method/config | Other method/config | Switch | Gap | Priority regret | Median regret | Rule regret | Median errors chosen/oracle | Rule errors chosen/oracle |",
               "|---|---|---:|---|---|---|---:|---:|---:|---:|---|---|"]
        for r in p["rows"]:
            name = lambda q: q["id"] if q else "infeasible"
            err = lambda policy: "/".join(fmt(r["policies"][policy].get(k)) for k in ("chosen_abs_error", "oracle_abs_error"))
            md.append("| "+" | ".join([r["reference"], r["objective"], str(r["budget"]), name(r["oracle"]),
                name(r["second_method"]), str(r["rank_switch"]), fmt(r["cross_method_gap"]),
                *[fmt(r["policies"][k]["regret"]) for k in ("priority", "median", "rule")], err("median"), err("rule")])+" |")

    objectives = {"math": "Mathematics", "code": "Code", "qa": "Question answering", "multi": "Largest increase"}
    families = {"gemma3": "Gemma 3", "pythia": "Pythia"}
    body = []
    for panel, title in (("independent", "Independent validation (four references)"),
                         ("heldout_pythia_H1", "Held-out Pythia selection maps (Fig.~\\ref{fig:selection})")):
        body += [r"\midrule", r"\multicolumn{6}{l}{\emph{" + title + r"}} \\"]
        for objective, aggregate in summary["panels"][panel]["by_objective"].items():
            condition = aggregate["error_conditions"]["pair_below_half_gap"]
            for family, values in aggregate["by_family"].items():
                labels = ([objectives[objective], families[family]] if panel == "independent" else
                          [r"\multicolumn{2}{l}{" + objectives[objective] + "}"])
                gains = ["--" if values[key] is None else f"{values[key]:.3f}".replace("-0.000", "0.000")
                         for key in ("gain_switch", "gain_non_switch")]
                body.append(" & ".join([*labels, f"{values['n_switch']} of {values['n']}", *gains,
                                        f"{condition['count']} of {condition['denominator']}"]) + r" \\")
    caption = (
        "Selection gains from development medians over development-priority choices. "
        "Gains are mean reductions in regret in nats per native token, separately by family and by whether the oracle "
        "method differs from the priority method. All feasible cells enter the switch counts. "
        "The last column counts cells where median prediction errors on both the chosen and oracle candidates are "
        "below half the gap between the oracle and the best candidate from another method. "
        "These descriptive counts pool families within each objective, repeat for each family, and exclude cells with "
        "only one feasible method; they do not certify selection. Dashes denote empty subsets. "
        "Retrospective analysis of recorded results (rebuttal round)."
    )
    widths = (20, 13, 12, 20, 15, 20)
    columns = "".join(r">{\raggedright\arraybackslash}p{\dimexpr " + f"{w / 100:.3f}" +
                      r"\textwidth-" + f"{10 * w / 100:.3f}" + r"\tabcolsep\relax}" for w in widths)
    tex = ("% Generated by analysis/r1_selection_value.py; do not edit.\n"
           "\\begin{table}[tb]\n\\centering\\small\\setlength{\\tabcolsep}{3pt}\n"
           "\\caption{\\rev{" + caption + "}}\n\\label{tab:r1-selection}\n"
           "\\begin{tabular}{@{}" + columns + "@{}}\n\\toprule\n" +
           r"Objective & Family & Method switches & Gain of medians over priority where the method switches & Gain where it does not & Cells where both median errors are below half the gap \\" +
           "\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    tex = house_style(table_layout(tex))
    tex = tex.replace("Math &", "Mathematics &").replace("QA &", "Question answering &")
    tex = tex.replace("{Math}", "{Mathematics}").replace("{QA}", "{Question answering}")
    return "\n".join(md)+"\n", tex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", action="store_true", help="Render the recorded summary without refitting")
    args = parser.parse_args()
    out = ROOT / "results" / ID
    summary = json.loads((out / "summary.json").read_text()) if args.table else build()
    md, tex = render(summary)
    out.mkdir(parents=True, exist_ok=True)
    if not args.table:
        (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    (out / "summary.md").write_text(md)
    (ROOT / "paper/paper/tables" / (ID+".tex")).write_text(tex)
    print(json.dumps({k: v["by_objective"] for k, v in summary["panels"].items()}, indent=2))


if __name__ == "__main__":
    main()
