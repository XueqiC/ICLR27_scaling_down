#!/usr/bin/env python3
"""Retrospective policy ablation on the independent selection validation (S3). CPU only.

    python -B analysis/s3_policy_ablation.py

Reads the sealed S3 plan, its measured outcomes, the fitted objects the final rule
used, the distillation forms of Section 4 and the development selection panel, and
scores five selection policies on the same 4 references x 4 objectives x 17 budgets:

  priority   development-fixed method order per objective and budget (from the
             289-cell development panel), largest feasible storage within the method;
             no numeric prediction at all
  median     development medians per configuration for pruning and quantization and
             the development constant for the student, every capability
  relations  the source-conditioned relations of Section 4 where their inputs exist
             (Pythia references; Gemma has no disclosed D0 and falls back to median),
             and the selected distillation forms for the student on every reference
  rule       the final rule as fixed before the round (from the sealed maps)
  quant-only quantization restricted to its own family

plus the all-method and quantization-only oracles and the decomposition
L(Q^)-L(A*) = [L(Q^)-L(Q*)] + [L(Q*)-L(A*)], and L(Q*)-L(A^) for each policy.
Nothing is refit; no S3 file is written. Outputs go to results/s3-policy-ablation/.
"""
from __future__ import annotations

import collections
import copy
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.final_rule import predict as rule_predict  # noqa: E402
from analysis import v70_distill_confirm as v70  # noqa: E402

S3 = ROOT / "results/s3-selection-validation"
OUT = ROOT / "results/s3-policy-ablation"
# The S3 scoring conventions, restated so this module needs no private S3 runtime;
# the test checks the sealed policies cell for cell against score_v2.json.
CAPS = ("math", "code", "qa")
OBJECTIVES = (*CAPS, "multi")
BUDGETS = [i / 100 for i in range(20, 101, 5)]
METHODS = ("prune", "quant", "distill", "dense")  # tie priority
EPS = 1e-12
POLICIES = ("priority", "priority_count", "median", "relations", "rule", "quant-only")


class common:  # namespace matching analysis.s3_common
    CAPS, OBJECTIVES, BUDGETS, METHODS = CAPS, OBJECTIVES, BUDGETS, METHODS

    @staticmethod
    def losses(value):
        if set(value) != set(CAPS) or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                              and x >= 0 and x == x for x in value.values()):
            raise ValueError("Invalid loss record")
        return value

    @staticmethod
    def score(value, objective, dense):
        return max(value[c] - dense[c] for c in CAPS) if objective == "multi" else value[objective]

    @staticmethod
    def feasible(configs, budget):
        return [q for q in configs if q["r"] <= budget + EPS]

    @staticmethod
    def choose(configs, objective, dense, field):
        if not configs:
            return None
        return min(configs, key=lambda q: (common.score(q[field], objective, dense),
                   METHODS.index(q["method"]), q["r"], q["id"]))


def measure_path(ref, q, directory):
    if q["method"] == "distill":
        return directory / "students" / ref["id"] / "measurement.json"
    return directory / "measurements" / ref["id"] / (q["id"].replace(":", "__") + ".json")


def read(path):
    return json.loads(Path(path).read_text())


def development_priority(v64):
    """Method order per objective and budget from the development panel.

    ``rate`` ranks methods by how often they were the oracle among the development
    cells in which they had a feasible candidate; ``count`` ranks by raw oracle
    wins, which penalises a method that was a candidate on fewer states. Ties keep
    the fixed method order. Both are development-only quantities.
    """
    order = {}
    for objective in OBJECTIVES:
        for budget in BUDGETS:
            wins, offered = collections.Counter(), collections.Counter()
            for cell in v64["cells"][objective]:
                if abs(cell["budget"] - budget) > 1e-9:
                    continue
                wins[cell["oracle_method"]] += 1
                methods = cell["candidate_methods"]
                for m in (methods if isinstance(methods, list) else list(methods)):
                    offered[m] += 1
            rate = {m: (wins[m] / offered[m]) if offered[m] else 0.0 for m in METHODS}
            by_rate = sorted(METHODS, key=lambda m: (-rate[m], -wins[m], METHODS.index(m)))
            by_count = sorted(METHODS, key=lambda m: (-wins[m], METHODS.index(m)))
            order[objective, budget] = {"order": by_rate, "order_count": by_count,
                                        "wins": {m: wins[m] for m in METHODS},
                                        "offered": {m: offered[m] for m in METHODS},
                                        "rate": {m: round(rate[m], 4) for m in METHODS}}
    return order


def student_point(ref):
    s = ref["student"]
    T = s["checkpoint"]["supervised_tokens"]
    DU = s["pool"]["D_U_pool"]
    # The selected forms (E, joint) carry no student-size term; the sentinel makes
    # v70's basis evaluate log(N/N_ref) = 0 without touching the stored refs.
    return {"Tc": T, "E": T / DU, "DU": DU, "student": "s3"}


def student_refs(v70_freeze):
    refs = v70_freeze["refs"]
    return {**refs, "N": {**refs["N"], "s3": refs["N_ref"]}}


def predictions(ref, models, v70_freeze):
    """Per candidate: median-only and Section-4 predictions; the rule's come from the plan."""
    out = {}
    for q in ref["candidates"]:
        median, relation, note = {}, {}, []
        for c in CAPS:
            inputs = {**ref, **q, "student": ref["student"], "models": models,
                      "L0": ref["dense"][c], "qa_distribution": "2Wiki"}
            if q["method"] == "distill":
                anchor = ref["student"]["dense"][c]
                median[c] = anchor + models["distill"]["constant"][c]
                sel = v70_freeze["selected"][c]["method"]
                model = v70_freeze["models"][c][sel]
                relation[c] = anchor + v70.predict(model, student_point(ref), c, student_refs(v70_freeze))
            elif q["method"] == "dense":
                median[c] = relation[c] = float(ref["dense"][c])
            else:
                median[c] = rule_predict(q["arm"], c, "new_stage", inputs)
                if ref["D0"] is None:
                    relation[c] = median[c]
                    note.append(f"{c}: no D0, median used")
                else:
                    status = "seen_size_unseen_density" if q["arm"] == "pruning" else "seen_state"
                    relation[c] = rule_predict(q["arm"], c, status, inputs)
        out[q["id"]] = {"median": median, "relations": relation, "relations_note": note}
    return out


def choose_priority(available, objective, budget, priority, key="order"):
    for method in priority[objective, budget][key]:
        pool = [q for q in available if q["method"] == method]
        if pool:
            return max(pool, key=lambda q: (q["r"], -METHODS.index(q["method"]), q["id"]))
    return None


NAMES = {"pythia-410m--step120000": "Pythia 410M, step 120000", "pythia-1.4b--step120000": "Pythia 1.4B, step 120000",
         "gemma3-1b": "Gemma 3, 1B", "gemma3-4b": "Gemma 3, 4B"}
LABELS = {"math": "Mathematics", "code": "Code", "qa": "Question answering", "multi": "Largest increase"}
TABLE = ROOT / "paper/paper/tables/s3_ablation.tex"


def render(summary):
    """Appendix table: mean regret per policy beside the opportunity and the rule's gain over the quantization oracle."""
    from analysis.paper_table_layout import house_style
    fmt = lambda x: f"{0.0 if abs(x) < 5e-4 else x:.3f}"  # no signed zero
    rows = []
    order = list(NAMES)
    for i, rid in enumerate(order):
        for j, objective in enumerate(OBJECTIVES):
            e = summary["per_reference"][f"{rid}|{objective}"]
            cells = [NAMES[rid] if j == 0 else "", LABELS[objective], fmt(e["mean_opportunity"]),
                     *[fmt(e[n]["mean_regret"]) for n in ("priority", "median", "relations", "rule", "quant-only")],
                     fmt(e["rule"]["mean_quant_residual"])]
            rows.append(" & ".join(cells) + r" \\")
        if i < len(order) - 1:
            rows.append(r"\midrule")
    caption = (
        "Retrospective policy ablation on the independent selection validation. Every policy chooses from the same "
        "twelve candidates at the same seventeen budgets; entries are mean regret against the all-method optimum in "
        "nats per native token over the sixteen budgets at which both the rule and quantization find a candidate, "
        "with the opportunity of Table~\\ref{tab:s3-selection} repeated. The development-priority policy ranks "
        "methods by their oracle win rate on the development panel and takes the largest feasible storage within "
        "the method, with no numeric prediction; the median policy predicts every candidate from development medians "
        "and the student from the development constant; the relations policy uses the source-conditioned relations "
        "of Section~\\ref{sec:laws} where their inputs exist and the selected distillation forms; the final rule and "
        "quantization-only are the policies fixed before the round. The last column is the loss of the best "
        "quantization candidate minus that of the rule's choice, the gain the rule realises even against a perfect "
        "quantization selection. The three added policies were defined after the round from development data alone.")
    text = ("% Generated by analysis/s3_policy_ablation.py; retrospective, do not edit.\n"
            "\\begin{table}[tb]\n\\centering\\footnotesize\n"
            f"\\caption{{{caption}}}\n\\label{{tab:s3-ablation}}\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llrrrrrrr@{}}\n\\toprule\n"
            "Reference & Objective & Opportunity & Priority & Median & Relations & Rule & Quantization only & "
            "Rule gain over quantization oracle \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    return house_style(text)


def main():
    plan = read(S3 / "plan_v2.json")
    score = read(S3 / "score_v2.json")
    models = read(S3 / "inputs/locked_models.json")
    v70_freeze = read(ROOT / "results/v70-distill-confirm/freeze.json")
    v64 = read(ROOT / "results/v64-selection-feasible/summary.json")
    priority = development_priority(v64)
    maps = {(m["reference"], m["objective"], m["budget"]): m for m in plan["maps"]}
    sealed = {(r["reference"], r["objective"], r["budget"]): r for r in score["rows"]}
    rows, checks = [], 0
    for ref in plan["references"]:
        configs = copy.deepcopy(ref["candidates"])
        for q in configs:
            q["actual"] = common.losses(read(measure_path(ref, q, S3))["losses"])
        preds = predictions(ref, models, v70_freeze)
        for q in configs:
            q["median"], q["relations"] = preds[q["id"]]["median"], preds[q["id"]]["relations"]
        dense = ref["dense"]
        for objective in OBJECTIVES:
            for budget in BUDGETS:
                available = common.feasible(configs, budget)
                quant = [q for q in available if q["method"] == "quant"]
                oracle = common.choose(available, objective, dense, "actual")
                q_oracle = common.choose(quant, objective, dense, "actual")
                lookup = {q["id"]: q for q in available}
                m = maps[ref["id"], objective, budget]
                chosen = {
                    "priority": choose_priority(available, objective, budget, priority),
                    "priority_count": choose_priority(available, objective, budget, priority, "order_count"),
                    "median": common.choose(available, objective, dense, "median"),
                    "relations": common.choose(available, objective, dense, "relations"),
                    "rule": lookup.get(m["rule"]["selected"]),
                    "quant-only": lookup.get(m["quant-only"]["selected"]),
                }
                best = common.score(oracle["actual"], objective, dense) if oracle else None
                qbest = common.score(q_oracle["actual"], objective, dense) if q_oracle else None
                row = {"reference": ref["id"], "family": ref["family"], "objective": objective, "budget": budget,
                       "oracle_id": oracle["id"] if oracle else None, "oracle_loss": best,
                       "quant_oracle_id": q_oracle["id"] if q_oracle else None, "quant_oracle_loss": qbest,
                       "opportunity": (qbest - best) if best is not None and qbest is not None else None,
                       "paired": bool(oracle) and bool(q_oracle), "policies": {}}
                for name, q in chosen.items():
                    actual = common.score(q["actual"], objective, dense) if q else None
                    row["policies"][name] = {"selected": q["id"] if q else None, "actual_loss": actual,
                        "regret": (actual - best) if actual is not None and best is not None else None,
                        "quant_residual": (qbest - actual) if actual is not None and qbest is not None else None}
                s = sealed[ref["id"], objective, budget]
                for name in ("rule", "quant-only"):
                    a, b = row["policies"][name]["regret"], s["policies"][name]["regret"]
                    if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-9):
                        raise ValueError(f"Sealed {name} regret differs at {ref['id']}/{objective}/{budget}")
                    checks += 1
                if row["opportunity"] is not None and abs(row["opportunity"] - (s["opportunity"] or 0)) > 1e-9:
                    raise ValueError("Opportunity differs from the sealed score")
                rows.append(row)
    # Aggregates over the paired budgets (both oracles exist), per reference and objective.
    per = {}
    for ref in plan["references"]:
        for objective in OBJECTIVES:
            cells = [r for r in rows if r["reference"] == ref["id"] and r["objective"] == objective and r["paired"]]
            entry = {"n_paired": len(cells), "mean_opportunity": statistics.mean(r["opportunity"] for r in cells)}
            for name in POLICIES:
                regs = [r["policies"][name]["regret"] for r in cells]
                if any(v is None for v in regs):
                    raise ValueError(f"Policy {name} infeasible on a paired cell")
                entry[name] = {"mean_regret": statistics.mean(regs),
                               "mean_quant_residual": statistics.mean(r["policies"][name]["quant_residual"] for r in cells)}
            q = entry["quant-only"]
            entry["decomposition"] = {"quant_gap": q["mean_regret"],
                                      "quant_internal": q["mean_regret"] - entry["mean_opportunity"],
                                      "opportunity": entry["mean_opportunity"]}
            per[ref["id"], objective] = entry
    pooled = {}
    for objective in OBJECTIVES:
        entries = [per[r["id"], objective] for r in plan["references"]]
        pooled[objective] = {"mean_opportunity": statistics.mean(e["mean_opportunity"] for e in entries),
                             **{name: statistics.mean(e[name]["mean_regret"] for e in entries) for name in POLICIES},
                             "quant_residual_of_rule": statistics.mean(e["rule"]["mean_quant_residual"] for e in entries)}
    fallback = sum(1 for ref in plan["references"] if ref["D0"] is None)
    summary = {
        "status": "RETROSPECTIVE",
        "note": "Retrospective analysis on the sealed S3 outcomes; the final rule and quantization-only entries are "
                "the sealed policies and are checked against score_v2.json. The other three policies were defined "
                "after the round from development data alone and did not exist when the round was registered.",
        "policies": {"priority": "development-fixed method order per objective and budget: methods ranked by their "
                                 "oracle win rate among the development cells in which they had a feasible candidate "
                                 "(289-cell panel), ties in the fixed method order; within the chosen method the largest "
                                 "feasible nominal storage; no numeric prediction",
                     "priority_count": "as priority, but ranked by raw oracle wins, which penalises a method that was a "
                                       "candidate on fewer development states",
                     "median": "development per-configuration medians for pruning and quantization; the development "
                               "constant for the student on every capability",
                     "relations": "Section 4 source-conditioned relations where inputs exist (pruning power form, per-bit "
                                  "regression, grouped interpolation; Pythia references only, Gemma references have no "
                                  "disclosed D0 and use the median), and the selected distillation forms for the student",
                     "rule": "the final rule as fixed before the round", "quant-only": "quantization restricted to its family"},
        "development_priority": {f"{o}@{b:.2f}": v for (o, b), v in priority.items()},
        "references_without_D0": fallback, "sealed_checks": checks,
        "per_reference": {f"{k[0]}|{k[1]}": v for k, v in per.items()}, "pooled": pooled, "rows": rows,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = ["# S3 policy ablation (retrospective)", "", summary["note"], "",
             "| Reference | Objective | Opportunity | " + " | ".join(POLICIES) + " | L(Q*)-L(rule) |",
             "|---|---|---:|" + "---:|" * (len(POLICIES) + 1)]
    for (rid, objective), e in per.items():
        lines.append("| " + " | ".join([rid, objective, f"{e['mean_opportunity']:.4f}",
                     *[f"{e[n]['mean_regret']:.4f}" for n in POLICIES], f"{e['rule']['mean_quant_residual']:.4f}"]) + " |")
    lines += ["", "## Pooled over references (equal weight)", "", "| Objective | Opportunity | " + " | ".join(POLICIES) + " |",
              "|---|---:|" + "---:|" * len(POLICIES)]
    for objective, p in pooled.items():
        lines.append("| " + " | ".join([objective, f"{p['mean_opportunity']:.4f}", *[f"{p[n]:.4f}" for n in POLICIES]]) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    TABLE.write_text(render(summary))
    print("\n".join(lines))
    print(f"wrote {TABLE}")


if __name__ == "__main__":
    main()
