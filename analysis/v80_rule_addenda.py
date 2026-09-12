#!/usr/bin/env python3
"""CPU-only V81 labels from the sealed V78 selections; no fitting or measurement.

Run: python -B analysis/v80_rule_addenda.py
Updates the V80 addenda in place. Writes only results/v81-rule-labels/,
the two named paper table files/three PDFs,
and a byte-identical mirror of this script in paper/analysis/.
V78 inputs, including their directory metadata inventory, are checked unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "results/v78-rule-confirm/compare.json").is_file())
OUT = ROOT / "results/v81-rule-labels"
V78 = ROOT / "results/v78-rule-confirm"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPLCONFIGDIR"] = str(OUT / ".mplconfig")
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"
sys.path.insert(0, str(ROOT))

import numpy as np
from analysis import final_rule, plot_fig1_responses as style
from analysis import v64_selection_feasible as v64
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text

CAPS = v64.CAPS
OBJECTIVES = v64.OBJECTIVES
POLICIES = ("locked-rule", "v64-law", "quant-only", "cheapest")
POLICY_LABELS = {
    "locked-rule": "Frozen selection rule",
    "v64-law": "Source-conditioned predictor",
    "quant-only": "Quantization-only",
    "prune-only": "Pruning-only",
    "distill-only": "Distillation-only",
    "cheapest": "Cheapest feasible",
}
INTERNAL_NOTE = "Policy internal names: locked-rule, v64-law (respectively)."
MAPS = POLICIES[:2]
METHODS = v64.METHODS
TITLES = {"math": "Math", "code": "Code", "qa": "QA (2Wiki)", "multi": "Multi (max ΔL)"}
TABLE_LABELS = {**TITLES, "multi": "Multi"}
COLORS = (style.COLORS["pruning"], style.COLORS["quantization"],
          style.COLORS["distillation"], "#c4c4c4")
METHOD_LABELS = ("Prune", "Quant", "Distill", "Dense")
SYMBOLS = dict(zip(METHODS, ("^", "s", "D", "o")))
FIG_NAMES = ("rule_maps_main", "rule_regret", "rule_maps_full")
TABLE = "paper/paper/tables/rule_confirm_by_state.tex"
CONFIRM_TABLE = "paper/paper/tables/rule_confirm.tex"
MIRROR = "paper/analysis/v80_rule_addenda.py"
PAPER_OUTPUTS = {TABLE, CONFIRM_TABLE, MIRROR, *(f"paper/paper/figs/{n}.pdf" for n in FIG_NAMES)}
FIGURE_PAD = .04


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path = path.absolute()
    require(path.is_relative_to(OUT) or str(path.relative_to(ROOT)) in PAPER_OUTPUTS,
            f"Output outside V81 allowlist: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode()
    path.write_bytes(data)


def dump(data):
    return json.dumps(data, indent=2, allow_nan=False) + "\n"


def inventory():
    """Include snapshots without reading multi-GB weights or following symlinks."""
    result = {}
    for path in [V78, *sorted(V78.rglob("*"))]:
        s = path.lstat()
        result[str(path.relative_to(ROOT))] = [s.st_mode, s.st_size, s.st_mtime_ns,
                                               os.readlink(path) if path.is_symlink() else None]
    return result


def load_inputs():
    compare = json.loads((V78 / "compare.json").read_bytes())
    freeze = json.loads((V78 / "freeze.json").read_bytes())
    initial = json.loads((V78 / "freeze-independent.json").read_bytes())
    hashes = {**compare["input_sha256"]}
    hashes["results/v78-rule-confirm/compare.json"] = sha(V78 / "compare.json")
    for name, expected in hashes.items():
        require(sha(ROOT / name) == expected, f"Frozen input hash mismatch: {name}")
    for name in ("freeze", "freeze-independent"):
        require(sha(V78 / f"{name}.json") == (V78 / f"{name}.json.sha256").read_text().strip(),
                f"Invalid V78 seal: {name}")
    for name in ("analysis/plot_fig1_responses.py", "paper/paper/iclr2027_conference.sty"):
        hashes[name] = sha(ROOT / name)
    return compare, freeze, initial, hashes


def indexed(rows):
    result = {(r["state"], r["budget"]): r for r in rows}
    require(len(result) == len(rows), "Duplicate state/budget cell")
    return result


def validate(compare, freeze, initial):
    """Re-evaluate frozen predictions, feasible choices, oracle and heuristic sets."""
    states = freeze["states"]
    require(len(states) == 4 and len(v64.BUDGETS) == 17, "Unexpected V78 panel")
    expected = {(s["tag"], b) for s in states for b in v64.BUDGETS}
    actuals = {}
    for s in states:
        actuals[s["tag"]] = {}
        quant = [q for q in s["configs"] if q["method"] == "quant"]
        require({q["id"] for q in quant} ==
                {f"quant:channel_b{b}" for b in (8, 6, 5, 4, 3)} |
                {f"quant:b{b}_g{g}" for b in (3, 4, 5) for g in (64, 128, 256)},
                "Quantization candidate grid changed")
        for q in s["configs"]:
            if q["method"] == "distill":
                path = ROOT / initial["models"]["students"][q["target_state"]]["path"]
                actual = json.loads(path.read_bytes())["post_training"]
            else:
                path = V78 / "measurements" / s["tag"].replace("@", "--") / (
                    q["id"].replace(":", "__").replace("@", "--") + ".json")
                actual = json.loads(path.read_bytes())["losses"]
            actuals[s["tag"]][q["id"]] = actual
            for c in CAPS:
                inputs = {**s, **q, "L0": s["dense"][c], "models": initial["models"]["locked"],
                          "qa_distribution": "2Wiki"}
                if q["method"] == "distill":
                    inputs["student"] = initial["models"]["students"][q["target_state"]]
                arm = {"quant_channel": "per-channel", "quant_group": "grouped",
                       "prune_power": "pruning", "distill_linear": "distillation", "dense": "dense"}[q["law"]]
                np.testing.assert_allclose(final_rule.predict(arm, c, "new_stage", inputs),
                    q["predictions"]["locked-rule"][c]["absolute_loss"], rtol=0, atol=1e-12)
    for cap in OBJECTIVES:
        entries = indexed(compare["cells"][cap])
        require(set(entries) == expected, f"Incomplete compare panel: {cap}")
        for policy in MAPS:
            maps = indexed(freeze["maps"][policy][cap])
            require(set(maps) == expected, f"Incomplete frozen map: {cap}/{policy}")
            for s in states:
                configs = [{**q, "predicted": {c: q["predictions"][policy][c]["absolute_loss"] for c in CAPS},
                            "actual": actuals[s["tag"]][q["id"]]} for q in s["configs"]]
                for b in v64.BUDGETS:
                    row, frozen = entries[s["tag"], b], maps[s["tag"], b]
                    feasible = [q for q in configs if q["r"] <= b + 1e-12]
                    oracle = v64.choose(feasible, cap, s["dense"], "actual")
                    require((oracle["id"], oracle["method"]) == (row["oracle_id"], row["oracle_method"]),
                            "Measured oracle differs")
                    np.testing.assert_allclose(row["oracle_score"], v64.score(oracle, cap, s["dense"], "actual"),
                                               rtol=0, atol=1e-12)
                    for name in ("MAP", "quant-only", "cheapest"):
                        pool = [q for q in feasible if q["method"] == "quant"] if name == "quant-only" else feasible
                        choice = (min(pool, key=lambda q: (q["r"], *v64.tie_key(q))) if name == "cheapest"
                                  else v64.choose(pool, cap, s["dense"]))
                        require(choice["id"] == frozen["policies"][name]["config_id"], "Frozen selection differs")
                        output_name = policy if name == "MAP" else name
                        if policy == "v64-law" and name != "MAP":
                            if name == "cheapest":
                                continue
                            output_name = "v64-" + name
                        selected = row["policies"][output_name]
                        for key, value in frozen["policies"][name].items():
                            require(selected[key] == value, "Compare selection differs from freeze")
                        require(selected["status"] == "FEASIBLE", "Missing common feasible cell")
                        np.testing.assert_allclose(selected["regret"],
                            v64.score(choice, cap, s["dense"], "actual") - row["oracle_score"], rtol=0, atol=1e-12)
                        require(selected["oracle_method_agreement"] == (choice["method"] == oracle["method"]),
                                "Oracle method agreement differs")
                    ambiguity = v64.ambiguity(feasible, cap, s["dense"], initial["models"]["maes"])
                    candidate = row["candidate_sets"][policy]
                    for key in ("candidate_methods", "no_clear_winner_heuristic"):
                        require(candidate[key] == frozen[key] == ambiguity[key], "Candidate set differs")
                    ids = [r["config_id"] for r in ambiguity["methods"] if r["method"] in ambiguity["candidate_methods"]]
                    require(candidate["candidate_config_ids"] == frozen["candidate_config_ids"] == ids,
                            "Candidate configuration set differs")
                    require(candidate["oracle_method_covered"] == (oracle["method"] in candidate["candidate_methods"]),
                            "Method coverage differs")
                    require(candidate["oracle_config_covered"] == (oracle["id"] in ids), "Configuration coverage differs")
    return {"states": 4, "budgets_per_state": 17, "cells_per_objective": 68,
            "quant_candidates_per_state": 14, "all_frozen_locked_predictions_reproduced": True,
            "selections_oracles_regrets_and_candidate_sets_reproduced": True,
            "fit_or_measurement_performed": False}


def kd_provenance(initial):
    models = initial["models"]
    locked, fold = models["locked"]["distill"], models["v64"]["distill_linear"]
    students = models["students"]
    require(set(students) == {"pythia-160m@step64000", "pythia-410m@step64000"}, "Unexpected KD students")
    require(set(students) == set(locked["excluded_students"]), "Locked KD exclusions differ")
    require(set(students) <= set(fold["excluded_states"]), "V64 KD exclusions differ")
    require(not set(students) & set(fold["train_states"]), "Selectable student leaked into KD fit")
    require(locked["linear"] == fold["models"] and fold["n_train_students"] == 7, "KD fit differs")
    development = json.loads((ROOT / "results/v64-selection-feasible/summary.json").read_bytes())["students"]
    training = [s for t, s in development.items() if t not in students]
    require({s["tag"] for s in training} == set(fold["train_states"]), "KD baseline roster differs")
    for c in CAPS:
        np.testing.assert_allclose(locked["constant"][c], np.mean([s["delta"][c] for s in training]),
                                   rtol=0, atol=1e-12)
    return {"source": "pythia-1b@step64000", "students": students,
            "original_v39_cohort_size": len(development), "v78_train_students": fold["train_states"],
            "excluded_states": fold["excluded_states"], "v78_train_count": fold["n_train_students"],
            "post_training_outcomes_in_v78_prediction_fit": False,
            "historical_outcomes_reused_for_scoring": True, "teacher": "gpt-5.6-luna"}


def aggregate(compare, freeze):
    per_state, pooled, sets = [], {}, []
    for cap in OBJECTIVES:
        rows = compare["cells"][cap]
        pooled[cap] = {}
        for s in freeze["states"]:
            own = [r for r in rows if r["state"] == s["tag"]]
            choices = sorted({r["policies"]["locked-rule"]["config_id"] for r in own})
            per_state.append({"state": s["tag"], "objective": cap, "n_budgets": len(own),
                "mean_regret": {p: float(np.mean([r["policies"][p]["regret"] for r in own])) for p in POLICIES},
                "locked_distinct_configurations": len(choices), "locked_config_ids": choices})
        for p in POLICIES:
            mean = float(np.mean([r["policies"][p]["regret"] for r in rows]))
            ref = next(r for r in compare["tables"][cap] if r["policy"] == p)
            require(ref["n_feasible"] == 68, "Unexpected infeasible headline policy")
            np.testing.assert_allclose(mean, ref["mean_regret"], rtol=0, atol=1e-12)
            pooled[cap][p] = mean
        for p in MAPS:
            candidates = [r["candidate_sets"][p] for r in rows]
            counts = {str(k): sum(len(r["candidate_methods"]) == k for r in candidates) for k in range(1, 5)}
            require(sum(counts.values()) == 68, "Candidate-size counts do not cover the panel")
            ambiguous = [r for r in candidates if r["no_clear_winner_heuristic"]]
            record = {"objective": cap, "policy": p, "n_cells": 68, "size_counts": counts,
                "size_percent": {k: 100 * n / 68 for k, n in counts.items()},
                "n_no_clear_winner": len(ambiguous),
                "no_clear_winner_size_counts": {str(k): sum(len(r["candidate_methods"]) == k for r in ambiguous)
                                                 for k in range(1, 5)},
                "oracle_method_covered": sum(r["oracle_method_covered"] for r in candidates),
                "oracle_config_covered": sum(r["oracle_config_covered"] for r in candidates),
                "method_coverage": float(np.mean([r["oracle_method_covered"] for r in candidates])),
                "config_coverage": float(np.mean([r["oracle_config_covered"] for r in candidates])),
                "method_coverage_no_clear_winner": float(np.mean([r["oracle_method_covered"] for r in ambiguous]))}
            reference = compare["candidate_set_coverage"][cap][p]
            for key in ("n_no_clear_winner", "method_coverage", "config_coverage", "method_coverage_no_clear_winner"):
                np.testing.assert_allclose(record[key], reference[key], rtol=0, atol=1e-12)
            sets.append(record)
    return {"per_state": per_state, "pooled_mean_regret": pooled, "candidate_sets": sets,
            "verdict": compare["verdict"]}


def state_label(tag):
    size, step = tag.removeprefix("pythia-").split("@step")
    return f"{size.upper()} @ {int(step)//1000}k"


def single_choice_agreement(compare):
    """Compare one selected configuration with the oracle on identical feasible cells."""
    records = []
    for cap in OBJECTIVES:
        for reference in compare["tables"][cap]:
            policy = reference["policy"]
            rows = [r for r in compare["cells"][cap]
                    if r["policies"][policy]["status"] == "FEASIBLE"]
            exact, method = 0, 0
            for row in rows:
                choice = row["policies"][policy]
                same_config = choice["config_id"] == row["oracle_id"]
                same_method = choice["method"] == row["oracle_method"]
                require(choice["oracle_method_agreement"] == same_method,
                        f"Stored method agreement differs: {cap}/{policy}")
                require(not same_config or same_method,
                        f"Exact configuration match without method match: {cap}/{policy}")
                exact += same_config
                method += same_method
            n = len(rows)
            require(n == reference["n_feasible"] and n > 0, "Agreement denominator differs")
            np.testing.assert_allclose(method / n, reference["method_agreement"], rtol=0, atol=1e-12)
            require(exact <= method, f"Exact agreement exceeds method agreement: {cap}/{policy}")
            records.append({"objective": cap, "policy": policy, "n_feasible": n,
                            "exact_configuration_matches": exact, "method_matches": method,
                            "exact_configuration_agreement": exact / n, "method_agreement": method / n})
    return records


def latex_confirm(compare):
    lines = [r"\begin{table}[H]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Independent V78 selection panel. Regret is in nats. Agreement is the percentage "
        r"of single choices matching the oracle method; set coverage is the percentage of heuristic "
        r"candidate sets containing that method. QA is restricted to 2Wiki; KD reuses V39 students.}",
        r"\label{tab:rule-confirm}", r"\begin{tabularx}{\linewidth}{lXrrrr}", r"\toprule",
        r"Capability & Policy & Feasible & Regret & \shortstack{Method\\agreement} & "
        r"\shortstack{Set contains\\oracle method} \\", r"\midrule"]
    for cap in OBJECTIVES:
        for r in compare["tables"][cap]:
            if r["policy"] not in POLICIES:
                continue
            cov = compare["candidate_set_coverage"][cap].get(r["policy"])
            coverage = f"{100*cov['method_coverage']:.1f}" if cov else "---"
            lines.append(f"{TABLE_LABELS[cap]} & {POLICY_LABELS[r['policy']]} & {r['n_feasible']}/68 & "
                         f"{v64.number(r['mean_regret'], 4)} & {100*r['method_agreement']:.1f} & {coverage}" + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabularx}", r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize",
        r"The multi objective is $\max_c(L_c-L_{0c})$. Quantization-only pools both RTN arms. "
        r"All four headline policies share the same 68 feasible cells. Heuristic sets use frozen "
        r"development errors; set coverage is retrospective, not calibrated uncertainty. Verdicts: " +
        "; ".join(f"{TABLE_LABELS[c]}: {compare['verdict'][c]}" for c in CAPS) + ". " + INTERNAL_NOTE,
        r"\end{minipage}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def latex(data, freeze):
    lines = [r"\begin{table}[H]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
        r"\caption{V78 regret by source state and objective. Each entry is the mean over all 17 nominal storage budgets (0.20--1.00, step 0.05), in nats. $K$ counts distinct configurations selected by the Frozen selection rule across these budgets.}",
        r"\label{tab:rule-confirm-by-state}", r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"State & Objective & \shortstack{Frozen selection\\rule} & "
        r"\shortstack{Source-conditioned\\predictor} & Quantization-only & "
        r"\shortstack{Cheapest\\feasible} & $K$ \\", r"\midrule"]
    for s in freeze["states"]:
        for i, cap in enumerate(OBJECTIVES):
            r = next(r for r in data["per_state"] if r["state"] == s["tag"] and r["objective"] == cap)
            values = " & ".join(f"{r['mean_regret'][p]:.6f}" for p in POLICIES)
            lines.append(f"{state_label(s['tag']) if i == 0 else ''} & {TABLE_LABELS[cap]} & {values} & "
                         f"{r['locked_distinct_configurations']}" + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize",
        r"All four policies are feasible on all 68 state--budget cells per objective. "
        r"Multi minimizes $\max_c[L_c(M)-L_c(M_0)]$. QA uses 2Wiki. "
        r"The 1B/64k state includes two historical V39 KD students, excluded from the V78 prediction fits. " + INTERNAL_NOTE,
        r"\end{minipage}", r"\end{table}", "", r"\begin{table}[H]", r"\centering\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Heuristic candidate-set sizes and oracle coverage on all 68 cells per objective. "
        r"Size columns count methods in the no-clear-winner candidate set, including singleton sets. "
        r"Both coverage columns describe the set, not the single choice, and report count/68 (percent).}",
        r"\label{tab:rule-confirm-candidate-sizes}", r"\begin{tabularx}{\linewidth}{lXrrrrrr}", r"\toprule",
        r"& & \multicolumn{4}{c}{Set size (methods)} & \multicolumn{2}{c}{Oracle coverage} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}",
        r"Objective & Policy & 1 & 2 & 3 & 4 & \shortstack{Set contains\\oracle method} & "
        r"\shortstack{Set contains\\oracle configuration} \\", r"\midrule"]
    for cap in OBJECTIVES:
        for i, p in enumerate(MAPS):
            r = next(r for r in data["candidate_sets"] if r["objective"] == cap and r["policy"] == p)
            counts = " & ".join(str(r["size_counts"][str(k)]) for k in range(1, 5))
            coverage = " & ".join(f"{r[key]}/68 ({100*r[key]/68:.1f})"
                                  for key in ("oracle_method_covered", "oracle_config_covered"))
            lines.append(f"{TABLE_LABELS[cap] if i == 0 else ''} & {POLICY_LABELS[p]} & {counts} & {coverage}" + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabularx}", r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize",
        r"Sets reuse frozen development LOSO MAE thresholds; coverage is retrospective, not calibrated uncertainty. "
        r"On this panel the no-clear-winner flag holds exactly for sets of size $\geq2$. "
        r"The Source-conditioned predictor for Code at 1B/64k and budget 1.00 has four candidates "
        r"(quantization, dense, pruning, distillation); it is retained explicitly. "
        r"Set contains oracle method tests whether the measured oracle's method belongs to the candidate methods. "
        r"Set contains oracle configuration tests whether its exact configuration ID belongs to the set of "
        r"one predicted best configuration per included method. Neither column tests the single selected configuration. "
        + INTERNAL_NOTE,
        r"\end{minipage}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def quote(path, first, last):
    source = (ROOT / path).read_text().splitlines()
    return f"`{path}:{first}–{last}` (verbatim):\n\n```text\n" + "\n".join(
        f"{i}: {source[i-1]}" for i in range(first, last + 1)) + "\n```\n"


def quote_block(path, first_text, last_text):
    """Locate exact source lines so the verbatim coverage evidence survives relabeling."""
    source = (ROOT / path).read_text().splitlines()
    first = next(i for i, line in enumerate(source) if first_text in line)
    last = next(i for i in range(first, len(source)) if last_text in source[i])
    return quote(path, first + 1, last + 1)


def markdown(data, provenance):
    parts = ["# V81 selection-confirmation labels", "",
        "Generated on CPU from the sealed `results/v78-rule-confirm/compare.json` and freezes. "
        "All four states × 17 budgets are retained per objective. No fitting, measurements, GPU "
        "access, V78 edits, or commits are performed.", "",
        "## Candidate-set coverage: definitions and verbatim code", "",
        "In `tab:rule-confirm-candidate-sizes`, **Method coverage** (now **Set contains oracle method**) "
        "is the fraction of the 68 state–budget cells whose heuristic `candidate_methods` contains "
        "the measured oracle's method. **Exact config coverage** (now **Set contains oracle configuration**) "
        "is the fraction whose `candidate_config_ids` contains the measured oracle's exact configuration ID. "
        "The configuration set contains one predicted best configuration per included method. "
        "Both columns describe the **no-clear-winner candidate set**, not the single chosen configuration.", "",
        "The denominator is **all 68 cells, including singleton sets**, as in V78 and V80. It is not "
        "restricted to cells flagged no-clear-winner. The separate `method_coverage_no_clear_winner` "
        "statistic uses only flagged cells. On this panel that flag holds exactly when set size ≥2. "
        "The best predicted method is always included; each additional method is included when its "
        "predicted gap is strictly below the maximum of the two frozen development LOSO MAEs. "
        "These are retrospective heuristic sets, not calibrated uncertainty.", "",
        quote_block("analysis/v64_selection_feasible.py", "def ambiguity(",
                    '\"threshold\": records[1]'),
        quote_block("analysis/v78_rule_confirm.py", '\"candidate_config_ids\": [r[\"config_id\"]',
                    'if r[\"method\"] in ambiguity[\"candidate_methods\"]]'),
        quote_block("analysis/v78_rule_confirm.py", 'sets[name] = {',
                    '\"oracle_config_covered\": oracle'),
        quote_block("analysis/v78_rule_confirm.py", 'coverage[cap][policy] = {',
                    '\"method_coverage_no_clear_winner\": v64.mean_or_none'),
        "V80's aggregation, retained verbatim in the updated generator:", "",
        quote_block("analysis/v80_rule_addenda.py", '\"oracle_method_covered\": sum(',
                    '\"method_coverage_no_clear_winner\": float(np.mean'),
        "| Objective | Policy | Set contains oracle method | Set contains oracle configuration |",
        "|---|---|---:|---:|"]
    for r in data["candidate_sets"]:
        values = " | ".join(f"{r[k]}/68 ({100*r[k]/68:.1f}%)"
                            for k in ("oracle_method_covered", "oracle_config_covered"))
        parts.append(f"| {TABLE_LABELS[r['objective']]} | {POLICY_LABELS[r['policy']]} | {values} |")
    parts += ["", "## Single chosen configuration: exact agreement ≤ method agreement", "",
        "For each feasible cell, exact agreement is `choice['config_id'] == row['oracle_id']`; "
        "method agreement is `choice['method'] == row['oracle_method']`. Exact agreement implies "
        "method agreement in every cell. Counts and percentages below use the same feasible-cell "
        "denominator for each pair, and method agreement reproduces `compare.json`'s table statistic. "
        "**The inequality holds for every objective and every policy in compare.json**, including "
        "the three additional source-conditioned method-only variants. The six named policies follow; "
        "all nine are recorded in `summary.json`.", "",
        "| Objective | Policy | Exact-configuration agreement | Method agreement |",
        "|---|---|---:|---:|"]
    for r in data["single_choice_agreement"]:
        if r["policy"] not in POLICY_LABELS:
            continue
        n = r["n_feasible"]
        values = " | ".join(f"{r[k]}/{n} ({100*r[k]/n:.1f}%)"
                            for k in ("exact_configuration_matches", "method_matches"))
        parts.append(f"| {TABLE_LABELS[r['objective']]} | {POLICY_LABELS[r['policy']]} | {values} |")
    parts += ["", "## Regenerated artifacts and checks", "",
        "- `paper/paper/figs/rule_maps_main.pdf`: original four-panel layout, exact title `QA (2Wiki)`, "
        "saved with `bbox_inches='tight'` and `pad_inches=0.04` to retain edge ticks and row labels.",
        "- `paper/paper/figs/rule_maps_full.pdf`: row suffixes `frozen` and `source-cond.`; "
        "all oracle markers and no-clear-winner hatching retained.",
        "- `paper/paper/figs/rule_regret.pdf`: logarithmic axis, 16 bars, each mean printed above its "
        "bar with three significant digits; numeric labels rotated to fit within 3.2 inches. "
        "Bar tops and annotations use the unchanged means over the 68 cells.",
        "- `paper/paper/tables/rule_confirm.tex` and `rule_confirm_by_state.tex`: reader-facing policy "
        "names and explicit candidate-set coverage definitions; internal policy names appear only "
        "in table footnotes. Numeric entries and candidate-set size counts remain unchanged.",
        "- V78 predictions, selections, oracles, regrets, candidate sets, seals, input hashes, and "
        "directory inventory are verified. Figure text extents, legend separation, annotation "
        "counts, label overlaps, and saved regret width are checked in `validation.json`.",
        "- `analysis/v80_rule_addenda.py` is updated in place and mirrored byte-for-byte to "
        "`paper/analysis/v80_rule_addenda.py`. Reproduce with `python -B analysis/v80_rule_addenda.py`. "
        "PDF/PNG previews and both table files are also saved under `results/v81-rule-labels/`; "
        "SHA256 provenance is in `manifest.json`.", "",
        "V78 verdicts are unchanged: Math, Code, and QA confirmed; Multi retrospective. "
        "QA is restricted to 2Wiki. The two historical distillation students remain excluded "
        f"from the prediction fits ({provenance['v78_train_count']} development students).", ""]
    return "\n".join(parts)


def setup_style():
    font_manager.fontManager.ttflist.sort(key=lambda e: str(Path(e.fname).resolve()))
    font_manager.fontManager._findfont_cached.cache_clear()
    style.setup_style()


def method_handles():
    return [Patch(facecolor=c, label=m) for c, m in zip(COLORS, METHOD_LABELS)]


def map_axis(ax, entries, states, policies, cap, *, full):
    rows = indexed(entries)
    matrix = np.empty((len(states) * len(policies), len(v64.BUDGETS)), dtype=int)
    n_markers = n_hatches = 0
    for i, s in enumerate(states):
        for j, p in enumerate(policies):
            y = i * len(policies) + j
            for x, b in enumerate(v64.BUDGETS):
                r = rows[s["tag"], b]
                method = r["policies"][p]["method"]
                matrix[y, x] = METHODS.index(method)
                if full and r["candidate_sets"][p]["no_clear_winner_heuristic"]:
                    ax.add_patch(Rectangle((x-.5, y-.5), 1, 1, facecolor="none", edgecolor="#222222",
                                           hatch="///", lw=0, zorder=2))
                    n_hatches += 1
                if full or method != r["oracle_method"]:
                    ax.plot(x, y, marker=SYMBOLS[r["oracle_method"]] if full else "o", ls="",
                            ms=3 if full else 2.8, mfc="white", mec="#111111", mew=.65, zorder=3)
                    n_markers += 1
    ax.imshow(matrix, cmap=ListedColormap(COLORS), vmin=-.5, vmax=3.5,
              aspect="auto", interpolation="nearest", zorder=0)
    ticks = [0, 4, 8, 12, 16] if full else [0, 8, 16]
    ax.set_xticks(ticks, [str(round(v64.BUDGETS[x] * 100)) for x in ticks])
    ax.set_yticks(range(len(matrix)))
    ax.set_xticks(np.arange(-.5, len(v64.BUDGETS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(matrix), 1), minor=True)
    ax.grid(which="minor", color="white", lw=.35, alpha=.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", which="major", length=0, pad=3)
    ax.set_title(TITLES[cap], pad=6, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return {"matrix_shape": list(matrix.shape), "oracle_markers": n_markers,
            "hatched_cells": n_hatches, "predicted_method_matrix": matrix.tolist()}


def check_layout(fig, legends):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    saved_box = fig.get_tightbbox(renderer).padded(FIGURE_PAD).transformed(fig.dpi_scale_trans)
    visible = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    for t in visible:
        require(t.get_fontsize() >= 8, f"Text smaller than 8 pt: {t.get_text()}")
        box = t.get_window_extent(renderer)
        require(box.x0 >= saved_box.x0 and box.y0 >= saved_box.y0 and
                box.x1 <= saved_box.x1 and box.y1 <= saved_box.y1,
                f"Clipped text: {t.get_text()}")
        require(t.get_text().lower() not in POLICY_LABELS,
                f"Internal policy name in figure: {t.get_text()}")
    legend_boxes = [l.get_window_extent(renderer) for l in legends]
    axis_boxes = [a.get_tightbbox(renderer) for a in fig.axes]
    for i, box in enumerate(legend_boxes):
        require(not any(box.overlaps(b) for b in axis_boxes), "Legend overlaps panel or labels")
        require(not any(box.overlaps(b) for b in legend_boxes[i+1:]), "Legends overlap")
        require(not any(box.overlaps(t.get_window_extent(renderer)) for t in fig.texts),
                "Legend overlaps shared axis label")
    for i, box in enumerate(axis_boxes):
        require(not any(box.overlaps(b) for b in axis_boxes[i+1:]), "Panels or axis labels overlap")
    return {"size_inches": list(fig.get_size_inches()),
            "saved_size_inches": [saved_box.width / fig.dpi, saved_box.height / fig.dpi],
            "minimum_font_points": min(t.get_fontsize() for t in visible),
            "text_within_saved_bbox": True, "reader_facing_policy_labels": True,
            "legends_outside_panels": True, "no_panel_or_legend_overlap": True,
            "bbox_inches": "tight", "pad_inches": FIGURE_PAD}


def save_figure(fig, name, legends, details):
    layout = check_layout(fig, legends)
    if name == "rule_regret":
        require(layout["saved_size_inches"][0] <= 3.2, "Regret figure exceeds 3.2 inches")
    import io
    for ext in ("pdf", "png"):
        buffer = io.BytesIO()
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else {}
        fig.savefig(buffer, format=ext, dpi=300, metadata=metadata,
                    bbox_inches="tight", pad_inches=FIGURE_PAD)
        write(OUT / f"figs/{name}.{ext}", buffer.getvalue())
        if ext == "pdf":
            write(ROOT / f"paper/paper/figs/{name}.pdf", buffer.getvalue())
    plt.close(fig)
    return {**layout, **details}


def figures(compare, freeze, data):
    setup_style()
    states = freeze["states"]
    validation = {}
    fig, axes = plt.subplots(1, 4, figsize=(5.5, 2.1), sharey=True)
    fig.subplots_adjust(left=.16, right=.985, top=.79, bottom=.40, wspace=.13)
    details = {}
    for ax, cap in zip(axes, OBJECTIVES):
        details[cap] = map_axis(ax, compare["cells"][cap], states, ("locked-rule",), cap, full=False)
        require(details[cap]["oracle_markers"] == sum(not r["policies"]["locked-rule"]["oracle_method_agreement"]
                                                      for r in compare["cells"][cap]), "Wrong main markers")
    axes[0].set_yticklabels([state_label(s["tag"]) for s in states])
    fig.text(.57, .24, "Nominal storage budget (% of dense matrix storage)", ha="center", fontsize=8)
    handles = method_handles() + [Line2D([], [], marker="o", ms=3, mfc="white", mec="#111111", ls="",
                                         label="Oracle method differs")]
    legends = [fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .025), ncol=5,
                          frameon=False, columnspacing=1., handlelength=1.1, handletextpad=.4)]
    validation["rule_maps_main"] = save_figure(fig, "rule_maps_main", legends, {"panels": details})

    fig, ax = plt.subplots(figsize=(3.1, 3.6))
    fig.subplots_adjust(left=.22, right=.97, top=.91, bottom=.37)
    x, width = np.arange(len(OBJECTIVES)), .18
    bar_colors = (style.COLORS["measured"], style.COLORS["pruning"], style.COLORS["quantization"], "#a9a9a9")
    floor = 1e-6
    annotations = []
    for i, (p, color) in enumerate(zip(POLICIES, bar_colors)):
        values = [data["pooled_mean_regret"][c][p] for c in OBJECTIVES]
        require(all(v > floor for v in values), "Log axis floor would hide a regret mean")
        bars = ax.bar(x + (i-1.5) * width, np.array(values)-floor, width=width*.94, bottom=floor,
                      color=color, label=POLICY_LABELS[p], zorder=3)
        for cap, bar, value in zip(OBJECTIVES, bars, values):
            label = format(value, ".3g")
            ax.annotate(label, (bar.get_x() + bar.get_width()/2, value),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        rotation=90, fontsize=8, annotation_clip=False)
            annotations.append({"objective": cap, "policy": p, "value": value, "label": label})
    ax.set_yscale("log")
    ax.set_ylim(floor, 80)
    ax.set_yticks([1e-6, 1e-4, 1e-2, 1], ["1e-6", "1e-4", "1e-2", "1"])
    ax.minorticks_off()
    ax.set_ylabel("Mean regret (nats; log scale)", fontsize=8, labelpad=3)
    ax.set_xticks(x, ["Math", "Code", "QA\n(2Wiki)", "Multi"])
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=2, pad=2)
    ax.grid(axis="y", color="#dddddd", lw=.5, zorder=0)
    legends = [fig.legend(handles=ax.get_legend_handles_labels()[0], labels=[POLICY_LABELS[p] for p in POLICIES],
                          loc="lower center", bbox_to_anchor=(.51, .015), ncol=1, frameon=False,
                          columnspacing=1., handlelength=1.1, handletextpad=.5)]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    annotation_boxes = [t.get_window_extent(renderer) for t in ax.texts]
    require(len(annotation_boxes) == len(ax.patches) == 16, "Missing regret value labels")
    require(not any(a.overlaps(b) for i, a in enumerate(annotation_boxes) for b in annotation_boxes[i+1:]),
            "Regret value labels overlap")
    validation["rule_regret"] = save_figure(fig, "rule_regret", legends,
        {"scale": "log", "axis_floor": floor, "bar_count": len(ax.patches), "means": data["pooled_mean_regret"],
         "annotations": annotations, "value_label_significant_digits": 3, "value_labels_do_not_overlap": True})

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.8), sharey=True)
    fig.subplots_adjust(left=.25, right=.985, top=.93, bottom=.255, wspace=.12, hspace=.37)
    details = {}
    for ax, cap in zip(axes.flat, OBJECTIVES):
        details[cap] = map_axis(ax, compare["cells"][cap], states, MAPS, cap, full=True)
        require(details[cap]["oracle_markers"] == 136, "Full map must show all oracle marks")
        expected_hatches = sum(r["candidate_sets"][p]["no_clear_winner_heuristic"]
                               for r in compare["cells"][cap] for p in MAPS)
        require(details[cap]["hatched_cells"] == expected_hatches, "Wrong full-map hatching")
        for y in (1.5, 3.5, 5.5):
            ax.axhline(y, color="white", lw=.8, zorder=4)
    labels = [f"{state_label(s['tag'])} / {p}" for s in states for p in ("frozen", "source-cond.")]
    for ax in axes[:, 0]:
        ax.set_yticklabels(labels)
    fig.text(.61, .185, "Nominal storage budget (% of dense matrix storage)", ha="center", fontsize=8)
    oracle_handles = [Line2D([], [], marker=SYMBOLS[m], ms=3.5, ls="", mfc="white", mec="#111111",
                             label="Oracle " + label.lower()) for m, label in zip(METHODS, METHOD_LABELS)]
    hatch = [Patch(facecolor="white", edgecolor="#222222", hatch="///", label="No clear winner (heuristic)")]
    legends = [fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, bottom), ncol=ncol,
                          frameon=False, columnspacing=1.3, handlelength=1.5, handletextpad=.5)
               for handles, bottom, ncol in ((method_handles(), .115, 4), (oracle_handles, .065, 4), (hatch, .015, 1))]
    validation["rule_maps_full"] = save_figure(fig, "rule_maps_full", legends, {"panels": details})
    return validation


def main():
    before = inventory()
    compare, freeze, initial, hashes = load_inputs()
    checks = validate(compare, freeze, initial)
    provenance = kd_provenance(initial)
    data = aggregate(compare, freeze)
    data["single_choice_agreement"] = single_choice_agreement(compare)
    checks["single_choice_exact_agreement_le_method_agreement"] = True
    checks["single_choice_policy_objectives_checked"] = len(data["single_choice_agreement"])
    require(all((len(r["candidate_methods"]) >= 2) == r["no_clear_winner_heuristic"]
                for rows in compare["cells"].values() for row in rows for r in row["candidate_sets"].values()),
            "No-clear-winner flag differs from size >= 2; update table note")
    checks["figures"] = figures(compare, freeze, data)
    table = latex(data, freeze)
    write(ROOT / TABLE, table)
    write(OUT / "rule_confirm_by_state.tex", table)
    confirm_table = latex_confirm(compare)
    write(ROOT / CONFIRM_TABLE, confirm_table)
    write(OUT / "rule_confirm.tex", confirm_table)
    write(OUT / "summary.json", dump({**data, "kd_provenance": provenance}))
    write(OUT / "summary.md", markdown(data, provenance))
    write(ROOT / MIRROR, Path(__file__).read_bytes())
    for name, expected in hashes.items():
        require(sha(ROOT / name) == expected, f"Input changed during build: {name}")
    require(inventory() == before, "V78 directory changed during build")
    checks["v78_inventory_unchanged"] = True
    checks["input_hashes_unchanged"] = True
    checks["script_mirror_identical"] = sha(ROOT / MIRROR) == sha(Path(__file__))
    require("torch" not in sys.modules, "CPU label build imported GPU libraries")
    checks["cpu_only_no_torch_import"] = True
    write(OUT / "validation.json", dump(checks))
    outputs = [ROOT / p for p in PAPER_OUTPUTS]
    outputs += [OUT / n for n in ("summary.json", "summary.md", "validation.json",
                                  "rule_confirm.tex", "rule_confirm_by_state.tex")]
    outputs += [OUT / f"figs/{n}.{ext}" for n in FIG_NAMES for ext in ("pdf", "png")]
    write(OUT / "manifest.json", dump({"input_sha256": hashes, "v78_inventory": before,
        "script_sha256": sha(Path(__file__)), "output_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(outputs)},
        "versions": {"python": sys.version, "numpy": np.__version__, "matplotlib": plt.matplotlib.__version__}}))
    print("V81 complete: 16 state/objective rows, 8 candidate distributions, 3 figures, 2 table files, mirrored script.")
    print("Verified all V78 inputs and directory inventory unchanged; no fits or measurements performed.")


if __name__ == "__main__":
    main()
