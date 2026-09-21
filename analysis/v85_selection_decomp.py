#!/usr/bin/env python3
"""V85: decompose sealed V78 regrets and transcribe the selection rule (CPU/stdlib).

    python -B analysis/v85_selection_decomp.py [--check]

Reads V78 without changing or recomputing its selections, oracle, or outcomes.
Writes results/v85-selection-decomp/, the two paper tables, and this script's
paper/code/analysis/ mirror. --check verifies those outputs without writing.
The rule implementation and its fitted objects are read, never imported/refit.
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
import ast
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean, median

try:
    from . import provenance
except ImportError:
    import provenance


ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "results/v78-rule-confirm/compare.json").is_file())
OUT = "results/v85-selection-decomp"
TABLES = "paper/paper/tables"
COMPARE = "results/v78-rule-confirm/compare.json"
FREEZE = "results/v78-rule-confirm/freeze-independent.json"
RULE = "analysis/final_rule.py"
DELIVERED = f"{TABLES}/final_deliverables.tex"
CAPS = ("math", "code", "qa")
OBJECTIVES = (*CAPS, "multi")
POLICIES = ("locked-rule", "v64-law", "quant-only", "cheapest")
NAMES = {"math": "Math", "code": "Code", "qa": "QA", "multi": "Multi (max)",
         "locked-rule": "Frozen rule", "v64-law": "Source-conditioned predictor",
         "quant-only": "Quantization-only", "cheapest": "Cheapest"}
STATES = ("pythia-160m@step32000", "pythia-410m@step32000",
          "pythia-1.4b@step32000", "pythia-1b@step64000")
STUDENTS = ("pythia-160m@step64000", "pythia-410m@step64000")
SUBSETS = {"step32k": STATES[:3], "1b_step64k": STATES[3:], "all": STATES}
SUBSET_NAMES = {"step32k": "Three step-32k states", "1b_step64k": "1B@64k", "all": "All four states"}
NEW = ("new_state", "new_size", "new_stage", "new_source")
CODE = (RULE, "analysis/v78_rule_confirm.py", "analysis/v53_prune_dev.py",
        "analysis/v36_pythia_controlled_fit.py", "analysis/v55_quant_group_fit.py",
        "analysis/v69_quant_confirm.py", "analysis/v39_distill_controlled.py",
        "analysis/v64_selection_feasible.py")
# Two of those files were edited after V78 was frozen, in 80a41ce, and the whole change is
# \begin{table}[H] becoming \begin{table}[!htbp] in the LaTeX each one emits. No computation is
# touched, so the frozen results are still the results these files produced. The exact bytes V78
# recorded are preserved beside its evidence and checked here, so the provenance check verifies the
# implementation that produced the numbers rather than whichever float specifier the manuscript
# currently wants. See results/v78-rule-confirm/frozen_implementation/README.md.
FROZEN_CODE = {
    "analysis/v78_rule_confirm.py":
        "results/v78-rule-confirm/frozen_implementation/v78_rule_confirm.py",
    "analysis/v69_quant_confirm.py":
        "results/v78-rule-confirm/frozen_implementation/v69_quant_confirm.py",
    "analysis/v64_selection_feasible.py":
        "results/v78-rule-confirm/frozen_implementation/v64_selection_feasible.py",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, context):
    require(math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-13),
            f"Inconsistent {context}: {a!r} != {b!r}")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Inputs:
    def __init__(self):
        self.raw = {}

    def text(self, path):
        if path not in self.raw:
            self.raw[path] = (ROOT / FROZEN_CODE.get(path, path)).read_bytes()
        return self.raw[path].decode()

    def json(self, path):
        return json.loads(self.text(path))

    def hashes(self):
        return {p: sha(raw) for p, raw in sorted(self.raw.items())}

    def unchanged(self):
        for path, raw in self.raw.items():
            require((ROOT / FROZEN_CODE.get(path, path)).read_bytes() == raw,
                    f"Input changed during generation: {path}")


def decompose(comp, frozen):
    """Equal-weight means of original state-budget regrets, with no filtering."""
    require(set(comp["cells"]) == set(OBJECTIVES), "Unexpected objectives")
    panel = {s["tag"]: s for s in frozen["states"]}
    require(set(panel) == set(STATES), "Unexpected frozen panel")
    for tag, state in panel.items():
        configs = state["configs"]
        require(state["state_status"] == "new_stage", "V78 state status changed")
        require(Counter(q["method"] for q in configs) ==
                Counter(dense=1, prune=4, quant=14, **({"distill": 2} if tag == STATES[-1] else {})),
                f"Unexpected candidate set: {tag}")
        require({q["target_state"] for q in configs if q["method"] == "distill"} ==
                (set(STUDENTS) if tag == STATES[-1] else set()), "Historical KD roster changed")

    budgets = [round(.2 + .05 * i, 2) for i in range(17)]
    expected = {(s, b) for s in STATES for b in budgets}
    aggregates, paired = {}, []
    for cap in OBJECTIVES:
        rows = comp["cells"][cap]
        require(len(rows) == 68 and {(r["state"], r["budget"]) for r in rows} == expected,
                f"Incomplete/duplicate state-budget grid: {cap}")
        for index, row in enumerate(rows):
            record = {"objective": cap, "state": row["state"], "budget": row["budget"],
                      "subset": "step32k" if row["state"] in STATES[:3] else "1b_step64k",
                      "source_pointer": f"{COMPARE}#/cells/{cap}/{index}",
                      "oracle_id": row["oracle_id"]}
            configs = {q["id"]: q for q in panel[row["state"]]["configs"]}
            for policy in POLICIES:
                entry = row["policies"][policy]
                value = entry["regret"]
                require(entry["status"] == "FEASIBLE" and isinstance(value, (int, float))
                        and not isinstance(value, bool) and math.isfinite(value) and value >= -1e-12,
                        f"Missing/nonfinite/infeasible regret: {cap}/{index}/{policy}")
                require(entry["config_id"] in configs, "Selection absent from frozen candidate set")
                q = configs[entry["config_id"]]
                require(q["method"] == entry["method"] and q["r"] <= row["budget"] + 1e-12,
                        "Selection method/storage mismatch")
                close(value, entry["actual_score"] - row["oracle_score"], "cell regret")
                record[policy] = value
            record["frozen_minus_quant"] = record["locked-rule"] - record["quant-only"]
            paired.append(record)
        for subset, states in SUBSETS.items():
            rr = [r for r in rows if r["state"] in states]
            means = {p: mean(r["policies"][p]["regret"] for r in rr) for p in POLICIES}
            diff = mean(r["policies"]["locked-rule"]["regret"] -
                        r["policies"]["quant-only"]["regret"] for r in rr)
            close(diff, means["locked-rule"] - means["quant-only"], "paired mean difference")
            aggregates.setdefault(subset, {})[cap] = {
                "n_cells": len(rr), "mean_regret": means, "frozen_minus_quant": diff,
                "selected_methods": {p: dict(sorted(Counter(r["policies"][p]["method"]
                                                           for r in rr).items())) for p in POLICIES},
                "frozen_quant_different_config_cells": sum(
                    r["policies"]["locked-rule"]["config_id"] != r["policies"]["quant-only"]["config_id"]
                    for r in rr),
                "difference_by_frozen_selected_method": {
                    method: sum(r["policies"]["locked-rule"]["regret"] -
                                r["policies"]["quant-only"]["regret"] for r in rr
                                if r["policies"]["locked-rule"]["method"] == method) / len(rr)
                    for method in sorted({r["policies"]["locked-rule"]["method"] for r in rr})}}
        reported = {r["policy"]: r for r in comp["tables"][cap]}
        for policy in POLICIES:
            require(reported[policy]["n_cells"] == reported[policy]["n_feasible"] == 68,
                    "Published full-panel denominator changed")
            close(aggregates["all"][cap]["mean_regret"][policy], reported[policy]["mean_regret"],
                  f"published mean: {cap}/{policy}")
            close(aggregates["all"][cap]["mean_regret"][policy],
                  .75 * aggregates["step32k"][cap]["mean_regret"][policy] +
                  .25 * aggregates["1b_step64k"][cap]["mean_regret"][policy], "51:17 decomposition")
    return aggregates, paired


def source_excerpt(inputs, path, name):
    source = inputs.text(path)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return {"path": path, "first_line": node.lineno, "last_line": node.end_lineno,
            "verbatim": ast.get_source_segment(source, node)}


def rule_spec(inputs, frozen):
    """Literal branch labels/expressions from final_rule, plus fitted-data provenance."""
    models = frozen["models"]
    locked = models["locked"]
    ppath = "results/v53-prune-dev/register.json"
    gpath = "results/v69-quant-confirm/develop.json"
    qpath = "results/v64-selection-feasible/summary.json"
    p, g, q = (inputs.json(path) for path in (ppath, gpath, qpath))
    require(locked["prune"] == {k: p[k] for k in ("standardization", "models")}, "V53 frozen objects differ")
    require(locked["grouped"] == {k: g[k] for k in ("standardization", "models")}, "V69 frozen objects differ")
    qroster = {str(b): sorted(s["tag"] for s in q["states"]
                             if any(c["law"] == "quant_channel" and c["bit"] == b for c in s["configs"]))
               for b in (8, 6, 5, 4, 3)}
    for cap in CAPS:
        for bit in qroster:
            deltas = [c["actual"][cap] - c["anchor"][cap] for s in q["states"] for c in s["configs"]
                      if c["law"] == "quant_channel" and str(c["bit"]) == bit]
            close(median(deltas), locked["channel"]["median"][cap][bit], "frozen channel median")
        fit = locked["channel"]["models"][cap]
        require(fit["input_fields"] == ["N0", "L0", "D0"], "Channel regression inputs changed")
        require(fit == models["v64"]["quant_channel"]["models"][cap], "Channel regression object changed")
        close(mean(s["delta"][cap] for tag, s in q["students"].items() if tag not in STUDENTS),
              locked["distill"]["constant"][cap], "frozen KD arithmetic mean")
    require(set(locked["distill"]["excluded_students"]) == set(STUDENTS), "KD exclusions changed")
    require(locked["distill"]["linear"]["math"]["with_d0"], "KD math must use +D0 linear form")

    development = {
        "P": {"artifact": ppath, "states": p["dev_states"], "n_capability_rows": p["n_dev_rows"],
              "description": "V53 register: 17 states, 84 density-state responses per capability; "
                             "all measured 0.55 <= d < 1 rows, excluding 2.8B. "
                             "Power uses ridge 0.001 and the development gamma grid; median is per density."},
        "Q": {"artifact": qpath, "states_by_bit": qroster,
              "description": "V64 frozen paired channel responses: 16/16/4/17/17 states at "
                             "8/6/5/4/3 bits (70 responses per capability across 17 states). "
                             "V36 per-bit source OLS via V64 fit_quant, or per-bit arithmetic median."},
        "G": {"artifact": gpath, "states": [s["tag"] for s in g["dev_states"]],
              "configs": g["dev_configs"],
              "description": "V69 develop: 160M, 410M, 1.4B at 16k and 143k; "
                             "bits 3,4,5 x groups 64,128,256 (54 responses per capability). "
                             "Per-anchor source ridge 0.001 or per-configuration median."},
        "K": {"artifact": qpath + "#/students", "original_panel": "results/v39-distill-controlled/summary.json",
              "train_states": models["v64"]["distill_linear"]["train_states"],
              "excluded_students": list(STUDENTS),
              "description": "V39 fixed-recipe Pythia panel as frozen in V64: 160M, 410M, 1.4B "
                             "at 16k,64k,143k, excluding 160M@64k and 410M@64k from both fits "
                             "(7 students per capability). Teacher gpt-5.6-luna, full pool 600, "
                             "2 epochs, strict LoRA, seed 0. Math uses +D0 OLS; code/QA use mean delta."}}
    require(len(development["K"]["train_states"]) == 7, "Unexpected KD training denominator")
    rows = []

    def add(arm, cap, status, status_label, predictor, expression, dev, anchor="source"):
        require(expression in inputs.text(RULE), f"Rule expression is not verbatim: {expression}")
        rows.append({"arm": arm, "capability": cap, "state_status": list(status),
                     "state_label": status_label, "predictor_verbatim": predictor,
                     "code_verbatim": expression, "development": dev, "reference_anchor": anchor})

    for status, label in ((('seen_size_unseen_density',), "Seen source state; new $d$"), (NEW, "New size or new stage")):
        for cap in CAPS:
            power = status != NEW and cap != "qa"
            add("pruning", cap, status, label, "v53 power" if power else "v53 median curve",
                'delta = float(np.dot(p["beta"], z) * prune.shape(d, p["gamma"]))' if power else
                'delta = prune.linear_curve(f["models"][c]["median_curve"]["anchors"], d)', "P")
    for status, label in ((('seen_state',), "Seen state"), (NEW, "New state")):
        for cap in CAPS:
            add("per-channel", cap, status, label, "v36 regression" if status != NEW else "per-bit dev median",
                'delta = channel_delta(f["models"][c], inputs)' if status != NEW else
                'delta = f["median"][c][str(inputs["bit"])]', "Q")
    for status, label in ((('seen_state',), "Seen state"), (NEW, "New state")):
        for cap in CAPS:
            interp = status != NEW and cap != "qa"
            add("grouped", cap, status, label, "v69 interpolation" if interp else "per-config median",
                'anchors = {k: float(np.dot(beta, z)) for k, beta in\n'
                '                       f["models"][c]["same_input_interpolation"]["anchors"].items()}' if interp else
                'anchors = f["models"][c]["median"]["anchors"]', "G")
    for cap in CAPS:
        add("distillation", cap, NEW, "New source", "linear (math)" if cap == "math" else "constant",
            'float(distill._predict(f["linear"][c], [{**s, "L0": anchor}])[0])' if cap == "math" else
            'float(f["constant"][c])', "K", "student")
    for cap in CAPS:
        add("dense", cap, (*NEW, "seen_state", "seen_size_unseen_density"), "Any valid status",
            "dense reference", 'value = float(inputs["L0"])', "None")
    source = inputs.text(RULE)
    excerpts = [source_excerpt(inputs, RULE, n) for n in ("channel_delta", "predict")]
    for path, functions in {
        "analysis/v78_rule_confirm.py": ("development_objects",),
        "analysis/v53_prune_dev.py": ("raw_features", "standardize", "shape", "linear_curve"),
        "analysis/v69_quant_confirm.py": ("grid_weights", "interpolate"),
        "analysis/v39_distill_controlled.py": ("_design", "_predict"),
        "analysis/v64_selection_feasible.py": ("fit_quant", "fit_fold", "score", "tie_key", "choose"),
    }.items():
        excerpts.extend(source_excerpt(inputs, path, n) for n in functions)
    return {"docstring_verbatim": ast.get_docstring(ast.parse(source), clean=False),
            "rows": rows, "development": development, "source_excerpts": excerpts,
            "frozen_distillation": locked["distill"], "group_boundary_rule_verbatim": g["boundary_rule"],
            "differences_from_delivered": [
                "Pruning: final_deliverables allows power or A2 on seen sizes; final_rule selects power "
                "for math/code and the median curve for QA. New size or stage uses the median for every capability.",
                "Per-channel: final_deliverables delivers per-bit source regression on new states and "
                "explicitly flags the selection-rule exception. final_rule uses per-bit development "
                "medians on new states; source regression is only the seen-state branch.",
                "Grouped: both use source interpolation for seen-state math/code and medians for QA/new states. "
                "The exact rule applies V69 interpolation/boundary handling to median anchors as well.",
                "Distillation: final_deliverables reports exposure a_c log(1+E) for math/code and a joint "
                "budget-pool form for QA on development students. final_rule instead uses V39 fixed-recipe "
                "student-state +D0 linear math and per-capability arithmetic-mean constants for code/QA, "
                "anchored to the initial student's dense loss; QA requires 2Wiki."]}


def json_text(value):
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def number(value, signed=False):
    """Six decimals preserve the tiny math/code gaps; zero is never negative."""
    return format(0. if abs(value) < .5e-6 else value, "+.6f" if signed else ".6f")


@proofread_table
def decomp_table(aggregates):
    lines = [r"% Generated by analysis/v85_selection_decomp.py; original V78 cell regrets.",
             r"\begin{table}[!htbp]", r"\centering\small",
             r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.08}",
             r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
             r"Objective & Policy & Step-32k & 1B@64k & All states \\",
             r" & Cells per objective & 51 & 17 & 68 \\", r"\midrule"]
    for cap in OBJECTIVES:
        for i, policy in enumerate((*POLICIES, "difference")):
            label = "Frozen minus quant." if policy == "difference" else NAMES[policy]
            vals = [aggregates[s][cap]["frozen_minus_quant"] if policy == "difference" else
                    aggregates[s][cap]["mean_regret"][policy] for s in SUBSETS]
            lines.append(f"{NAMES[cap] if i == 0 else ''} & {label} & " +
                         " & ".join(number(v, policy == "difference") for v in vals) + r" \\")
        if cap != OBJECTIVES[-1]:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}",
        r"\caption{Selection regret decomposition (native-token nats; lower is better). "
        r"Step-32k comprises 160M, 410M, and 1.4B: pruning and quantization candidates "
        r"with newly measured outcomes. 1B@64k additionally includes two historical "
        r"distillation candidates. The original dense candidate is retained at budget 1.0 "
        r"in every state. Means weight each state--budget cell equally over 17 storage "
        r"budgets (0.20--1.00 in steps of 0.05); all-state means weight the subsets 3:1. "
        r"Quantization-only pools per-channel and grouped candidates using frozen-rule "
        r"predictions. Multi minimizes $\max_c(L_c-L_{0,c}^{\mathrm{source}})$. "
        r"Differences are computed before rounding; negative favors the frozen rule.}",
        r"\label{tab:rule_decomp}", r"\end{table}"])
    return "\n".join(lines) + "\n"


PREDICTOR_NAMES = {
    "v53 power": "Pruning power form",
    "v53 median curve": "Pruning development median curve",
    "v36 regression": "Per-bit source regression",
    "per-bit dev median": "Per-bit development median",
    "v69 interpolation": "Piecewise source interpolation",
    "per-config median": "Per-configuration development median",
    "linear (math)": "Linear source regression",
    "linear (Math)": "Linear source regression",
    "constant": "Frozen constant",
}


@proofread_table
def locked_table(spec):
    lines = [r"% Generated from analysis/final_rule.py. Predictor labels below quote its docstring/code."]
    lines += ["% " + line for line in spec["docstring_verbatim"].splitlines()]
    lines += [r"\begin{table}[!htbp]", r"\centering\small",
              r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.05}",
              r"\begin{tabular}{@{}llll@{}}", r"\toprule",
              r"Arm & Cap. & State status & Response predictor \\", r"\midrule"]
    arms = {"pruning": "Pruning", "per-channel": "Channel RTN", "grouped": "Grouped RTN",
            "distillation": "Distillation", "dense": "Dense"}
    previous = None
    for row in spec["rows"]:
        arm = row["arm"]
        if previous is not None and previous != arm:
            lines.append(r"\addlinespace")
        for line in row["code_verbatim"].splitlines():
            lines.append("% " + line)
        anchor = r"$L_{0,c}$" if row["reference_anchor"] == "source" else r"$L_{S0,c}$"
        predictor = PREDICTOR_NAMES.get(row["predictor_verbatim"], row["predictor_verbatim"]) \
            if arm != "dense" else "0 (no loss change)"
        lines.append(" & ".join((arms[arm], NAMES[row["capability"]], row["state_label"],
                                 predictor)) + r" \\")
        previous = arm
    lines += [r"\bottomrule", r"\end{tabular}",
        r"\caption{Exact locked selection rule in \texttt{analysis/final\_rule.py}. "
        r"Each absolute prediction adds the listed response to a dense anchor: the source dense loss "
        r"$L_{0,c}$ for pruning and quantization, and the initial student's dense loss $L_{S0,c}$ for "
        r"distillation. Every method is fitted on its own development panel, described below.}",
        r"\label{tab:locked_rule}", r"\end{table}",
        r"\begingroup\small", r"\noindent\textbf{Development data.} "
        r"P: V53 register, 17 Pythia states, 84 density responses per capability "
        r"($0.55\le d<1$, excluding 2.8B). "
        r"Q: V64 frozen paired channel rows; 16/16/4/17/17 states at 8/6/5/4/3 bits. "
        r"G: V69 development grid, 160M/410M/1.4B at 16k/143k, "
        r"$b\in\{3,4,5\}$, $g\in\{64,128,256\}$. "
        r"K: V39 fixed-recipe Pythia students frozen in V64: 160M/410M/1.4B at "
        r"16k/64k/143k, excluding both selectable students (160M@64k, 410M@64k), "
        r"leaving seven. Recipe: \texttt{gpt-5.6-luna}, full pool 600, two epochs, "
        r"strict LoRA, seed 0. Full rosters and literal code are in the V85 artifact.",
        r"\par\noindent\textbf{Exact forms.} "
        r"The V53 power response is $(\beta_c^\top z)[(1-d)/0.3]^{\gamma_c}$, "
        r"with $z=(1,\allowbreak z(\log N_0),\allowbreak z(L_{0,c}),\allowbreak z(\log D_0))$ and development "
        r"standardization. Its median curve linearly interpolates adjacent per-density "
        r"medians. V36 regression is per-bit OLS on "
        r"$(1,\allowbreak z(\log(N_0/10^9)),\allowbreak z(L_{0,c}),\allowbreak z(\log(D_0/10^9)))$; "
        r"the new-state branch takes the median signed response at that bit. "
        r"V69 interpolation uses per-configuration source-regression anchors "
        r"$\beta_{c,b,g}^\top z$; its median branch uses per-configuration "
        r"development medians. Both pass the anchors through the same piecewise "
        r"bilinear interpolation in $(\log_2(2^{b-1}-1),\log_2(g/128))$. "
        r"Outside the group grid, use the nearest boundary pair and floor only the "
        r"extrapolated response at zero; no bit extrapolation. "
        r"V53 power and V69 source-anchor fits use ridge $10^{-3}$ including the intercept. "
        r"Distillation math uses "
        r"$\beta_0+\beta_1z(\log N_S)+\beta_2z(L_{S0,c})+\beta_3z(\log D_S)$; "
        r"code and QA use the per-capability arithmetic mean development response "
        r"(not a median).",
        r"\par\noindent\textbf{Domain and selection.} "
        r"New status means \texttt{new\_state}, \texttt{new\_size}, "
        r"\texttt{new\_stage}, or \texttt{new\_source}, taking precedence over seen size. "
        r"Pruning requires $0.6\le d\le0.9$; channel RTN requires a development bit. "
        r"Distillation requires a smaller same-stage student ($N_S<N_0$, $D_S=D_0$), "
        r"and QA requires \texttt{qa\_distribution='2Wiki'}. Undefined cells are rejected. "
        r"Dense accepts every valid status. Selection minimizes predicted absolute loss "
        r"or $\max_c(\widehat L_c-L_{0,c})$ over storage-feasible candidates; multi "
        r"has no separately fitted predictor. V78 invokes the new-stage branch for all four states.",
        r"\par\noindent\textbf{Differences from the delivered-predictor table.} "
        r"\texttt{tables/final\_deliverables.tex} allows power or A2 for seen-size "
        r"pruning; this rule uses power for math/code and medians for QA and new "
        r"size/stage. The channel-quantization branches now agree: both take the "
        r"per-bit source regression on states in the fit and the per-bit development "
        r"median on new ones. This rule was frozen with that branch before the "
        r"confirmation panel was measured, and is unchanged. "
        r"Grouped branches agree, with the boundary rule specified above. "
        r"Its distillation exposure and joint budget--pool forms are not used here: "
        r"selection uses the fixed-recipe student-state math form and code/QA constants.",
        r"\endgroup"]
    return "\n".join(lines) + "\n"


def summary(aggregates, spec, hashes):
    lines = ["# V85 selection decomposition and exact locked rule", "",
             "Generated on CPU with the Python standard library; no fitting or measurement. "
             "V78 selections, oracle candidates, and outcomes are read without modification.", "",
             "Regret is the selected candidate's measured objective minus the original feasible oracle. "
             "Every state has 17 equally weighted budgets (0.20 to 1.00 by 0.05). "
             "The 51-cell subset has three step-32k states (160M, 410M, 1.4B); "
             "its compressed candidates are newly measured pruning and quantization. "
             "The 17-cell subset is 1B@64k, with newly measured pruning/quantization and "
             "two reused historical distillation outcomes (160M@64k and 410M@64k). "
             "Both subsets retain the original dense candidate, feasible at budget 1.0. "
             "Thus 51/17/68 are state-budget cells **per objective**, not counts of experiments.", "",
             "Policy keys: frozen rule = `locked-rule`; source-conditioned predictor = `v64-law`; "
             "quantization-only = `quant-only` (both channel and grouped RTN, ranked by locked predictions); "
             "cheapest = `cheapest`. Multi uses `max_c(L_c - source_L0c)` over math/code/QA, "
             "not their average. QA is 2Wiki. All four policies are feasible in every cell. "
             "Differences use unrounded regrets; negative favors the frozen rule.", "",
             "| Subset | Objective | Cells | Frozen rule | Source-conditioned predictor | Quantization-only | Cheapest | Frozen minus quant. |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for subset in SUBSETS:
        for cap in OBJECTIVES:
            row = aggregates[subset][cap]
            lines.append("| " + " | ".join((SUBSET_NAMES[subset], NAMES[cap], str(row["n_cells"]),
                         *(number(row["mean_regret"][p]) for p in POLICIES),
                         number(row["frozen_minus_quant"], True))) + " |")
    lines += ["", "The pooled result weights the 51-cell and 17-cell subsets 3:1. "
              "The pooled math and code advantage over quantization-only comes almost entirely from "
              "1B@64k. On the newly measured step-32k subset, the math difference is "
              f"{aggregates['step32k']['math']['frozen_minus_quant']:.12g} nats and the code difference is "
              f"{aggregates['step32k']['code']['frozen_minus_quant']:.12g} nats (code is slightly worse). "
              "The frozen rule selects quantization in 48 cells and dense in three for each of math/code; "
              "their tiny differences arise at the dense budget. QA selects pruning in 27 cells and "
              "quantization in 24, giving a material improvement within the newly measured subset.", "",
              "This is a descriptive decomposition of the original panel, not a distillation-removal "
              "ablation or a newly precommitted confirmation criterion. 1B@64k differs in source state "
              "as well as candidate availability, so its contribution cannot be identified solely with "
              "distillation. V78's original full-panel verdicts remain math/code/QA confirmed and "
              "multi retrospective; multi's frozen-rule regret exceeds the source-conditioned predictor.", "",
              "## Exact rule and delivered-predictor differences", "",
              "`locked_rule.tex` lists each capability separately in all seven compression arm/status "
              "cases, plus the dense reference (24 rows). Equivalent new-status aliases share a row. "
              "Every V78 state uses `new_stage`; hence V78 uses pruning median curves, channel per-bit "
              "medians, grouped per-configuration medians, and the distillation branch. "
              "Absolute losses add the predicted response to source dense loss for pruning/RTN, "
              "and to initial student dense loss for distillation.", ""]
    lines += ["- " + d for d in spec["differences_from_delivered"]]
    lines += ["", "The distillation constants are arithmetic means on seven retained students: code "
              f"`{spec['frozen_distillation']['constant']['code']:.17g}`, QA "
              f"`{spec['frozen_distillation']['constant']['qa']:.17g}` nats. "
              "The stored math constant is unused; math calls the +D0 linear model. "
              "Exact coefficients, centers, scales, per-bit rosters, and source excerpts are in `rule_spec.json`.", ""]
    lines += [f"- **{key}**: {value['description']} Source: `{value['artifact']}`."
              for key, value in spec["development"].items()]
    lines += ["", "Verbatim module docstring from `analysis/final_rule.py`:", "", "```text",
              spec["docstring_verbatim"], "```", "", "Verbatim evaluator (the imported helper "
              "implementations and fit construction are also captured in `rule_spec.json`):", "", "```python",
              next(e["verbatim"] for e in spec["source_excerpts"] if e["path"] == RULE and
                   e["verbatim"].startswith("def predict(")), "```", "", "## Reproduction and validation", "",
              "Run `python -B analysis/v85_selection_decomp.py`; verify determinism with "
              "`python -B analysis/v85_selection_decomp.py --check`. The mirrored entry point "
              "`python -B paper/code/analysis/v85_selection_decomp.py --check` resolves the same repository inputs.", "",
              "Generation validates the full 4 x 17 grid for each objective, frozen candidate rosters, "
              "policy feasibility, actual-minus-oracle regrets, paired differences, the 51:17 weighted "
              "identity, and reproduction of V78's published all-state means. It checks implementation "
              "hashes against V78, verbatim rule expressions, V53/V69 object identity, channel medians, "
              "and distillation mean constants/exclusions without refitting. No test cells are dropped.", "",
              "Outputs: `decomposition.json` (full precision, method counts and contributions), "
              "`decomposition.csv` (12 aggregate rows), `cell_regrets.csv` (272 original cells with JSON "
              "pointers), `rule_spec.json` (24 rule rows, provenance, literal code), this summary, "
              "and the two requested LaTeX tables. Implementations that changed after V78 only in "
              "presentation are read from the preserved frozen copies beside its evidence, so the "
              "excerpts below quote the code that produced these numbers.", "",
              "## Input SHA256", ""]
    lines += [f"- `{path}`: `{value}`" for path, value in hashes.items()]
    return "\n".join(lines) + "\n"


def build_outputs():
    inputs = Inputs()
    comp, frozen = inputs.json(COMPARE), inputs.json(FREEZE)
    require(sha(inputs.raw[FREEZE]) == comp["input_sha256"][FREEZE], "V78 freeze hash mismatch")
    for path in CODE:
        inputs.text(path)
        require(provenance.matches(comp["input_sha256"][path], sha(inputs.raw[path])),
                f"Implementation changed since V78: {path}")
    inputs.text(DELIVERED)
    aggregates, paired = decompose(comp, frozen)
    spec = rule_spec(inputs, frozen)
    for path, raw in inputs.raw.items():
        if path in comp["input_sha256"]:
            require(sha(raw) == comp["input_sha256"][path], f"Development input changed since V78: {path}")
    for path, preserved in FROZEN_CODE.items():
        require(provenance.matches(comp["input_sha256"][path], sha((ROOT / preserved).read_bytes())),
                f"Preserved frozen implementation differs from the V78 record: {path}")
    flat = [{"subset": subset, "objective": cap, "n_cells": row["n_cells"],
             **row["mean_regret"], "frozen_minus_quant": row["frozen_minus_quant"]}
            for subset, caps in aggregates.items() for cap, row in caps.items()]
    hashes = inputs.hashes()
    return inputs, {
        f"{OUT}/decomposition.json": json_text({"units": "native-token nats", "weighting": "equal state-budget cells",
            "policy_names": {p: NAMES[p] for p in POLICIES}, "subset_states": SUBSETS,
            "aggregates": aggregates, "original_full_panel_verdict": comp["verdict"], "input_sha256": hashes}),
        f"{OUT}/decomposition.csv": csv_text(flat), f"{OUT}/cell_regrets.csv": csv_text(paired),
        f"{OUT}/rule_spec.json": json_text({**spec, "input_sha256": hashes}),
        f"{OUT}/summary.md": summary(aggregates, spec, hashes),
        f"{TABLES}/rule_decomp.tex": decomp_table(aggregates),
        f"{TABLES}/locked_rule.tex": locked_table(spec),
        "paper/code/analysis/v85_selection_decomp.py": (ROOT / "analysis/v85_selection_decomp.py").read_text()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify generated artifacts without writing")
    args = parser.parse_args()
    inputs, outputs = build_outputs()
    inputs.unchanged()
    for relative, content in outputs.items():
        path = ROOT / relative
        require(not path.resolve().is_relative_to((ROOT / "results/v78-rule-confirm").resolve()),
                "Refusing to write inside V78")
        if args.check:
            require(path.is_file() and path.read_bytes() == content.encode(), f"Stale/missing output: {relative}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        print(f"{'CHECKED' if args.check else 'WROTE'} {relative}")
    inputs.unchanged()
    print("Validated 272 original cells, 48 policy means, 12 paired differences, and 24 rule rows.")


if __name__ == "__main__":
    main()
