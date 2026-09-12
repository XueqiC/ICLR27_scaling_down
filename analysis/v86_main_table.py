#!/usr/bin/env python3
"""V86: frozen candidates, all frozen alternatives, and delivered rules (CPU only).

Derived from v84_main_table.py; retains exact JSON SOURCE pointers and INPUT hashes.
Run: python3 analysis/v86_main_table.py [--check | --regenerate-locked-tables]
Writes only the two main paper tables and results/v86-main-table/summary.{md,json}.
The regeneration option calls V85's validated in-memory builders, writing only
locked_rule.tex and rule_decomp.tex, without touching V85 result artifacts.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean
import sys

try:
    from . import provenance
except ImportError:
    import provenance

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "paper/paper/tables"
OUT = ROOT / "results/v86-main-table"
CAPS = ("math", "code", "qa")
LABELS = {
    "power": "Power", "A1": "A1", "A2": "Per-density regression",
    "cont": "Continuous regression", "median_curve": "Median dev. curve",
    "strength_only": "Strength-only", "zero": "Zero",
    "same_input_interpolation": "Interp.", "low_order_2d": "Surface",
    "bilinear": "Bilinear", "bit_only": "Bit-only", "median": "Per-config. median",
    "E": "$E$", "T": "$T$", "joint": "Joint", "constant": "Constant",
    "T-only": "$T$-only", "E-only": "$E$-only",
    "surface:L0": "Surface $L_0$", "surface:logN": "Surface $\\log N$",
    "F1:L0": "$F_1(L_0)$", "F1:logN": "$F_1(\\log N)$",
    "F2:L0": "$F_2(L_0)$", "F2:logN": "$F_2(\\log N)$",
    "joint+src": "Joint+src", "constant+src": "Constant+src",
    "T+src": "$T$+src", "E+src": "$E$+src",
}
# Compact one-line method stacks; summary.md spells out these names.
TABLE_LABELS = {**LABELS, "A2": "A2", "median_curve": "Median", "median": "Median",
                "constant+src": "Const.+src"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, context):
    require(math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-13),
            f"Inconsistent comparison: {context}: {actual} != {expected}")


@dataclass(frozen=True)
class Number:
    value: float
    source: str


class Comparison:
    def __init__(self, relative):
        self.path = relative
        raw = (ROOT / relative).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.data = json.loads(raw)

    def ref(self, *keys):
        return self.path + "#/" + "/".join(
            str(k).replace("~", "~0").replace("/", "~1") for k in keys)

    def number(self, *keys):
        value = self.data
        for key in keys:
            value = value[key]
        require(isinstance(value, (float, int)) and not isinstance(value, bool)
                and math.isfinite(value), f"Invalid number: {self.ref(*keys)}")
        return Number(float(value), self.ref(*keys))


def average(numbers):
    require(bool(numbers), "Cannot average an empty test set")
    return Number(mean(n.value for n in numbers),
                  "mean(" + "; ".join(n.source for n in numbers) + ")")


@dataclass
class Row:
    key: str
    arm: str
    predictor: str
    test: str
    scores: dict
    candidates: dict
    delivered: dict
    timing: str
    scope: str
    metadata: str
    n: int
    rationale: str
    selection_source: str
    roster_source: str
    recorded_baselines: dict | None = None

    @property
    def alternatives(self):
        return {c: [m for m in self.scores[c] if m != self.candidates[c]] for c in CAPS}

    @property
    def baselines(self):
        # Rank only AFTER pooling; dict order resolves exact ties deterministically.
        return {c: min(self.alternatives[c], key=lambda m: self.scores[c][m].value) for c in CAPS}

    def traces(self):
        yield f"{self.key} test object and counts: {self.metadata}; n/capability={self.n}"
        yield f"{self.key} frozen candidate: {self.selection_source}"
        yield f"{self.key} full predictor roster: {self.roster_source}; exclude candidate per capability"
        for cap in CAPS:
            for method, number in self.scores[cap].items():
                yield f"{self.key} {cap} {method} MAE = {number.value:.17g} <- {number.source}"
            a, b = (self.scores[cap][m] for m in (self.candidates[cap], self.baselines[cap]))
            yield (f"{self.key} {cap} candidate={a.value:.17g}; alternative ({self.baselines[cap]})="
                   f"{b.value:.17g}; gain={b.value-a.value:.17g} <- ({b.source}) - ({a.source}); "
                   "each displayed number rounded independently to 4 decimals")
            if self.delivered[cap] is not None:
                d = self.scores[cap][self.delivered[cap]]
                yield (f"{self.key} {cap} delivered ({self.delivered[cap]}) MAE={d.value:.17g} <- "
                       f"{d.source}; delivered gain={b.value-d.value:.17g} <- ({b.source}) - ({d.source})")
        yield (f"{self.key} delivery timing: {self.timing}; rule mapping/timing from supplied V86 "
               "specification and paper/docs/RESULTS_LEDGER.md; new stages use the new-state branch "
               "in analysis/final_rule.py#/predict (read only)")


def same(method):
    return dict.fromkeys(CAPS, method)


def row_scores(comp, rows, methods, cap_key="capability", error_key="absolute_errors"):
    scores = {}
    for cap in CAPS:
        indices = [i for i, r in rows if r[cap_key] == cap]
        scores[cap] = {m: average([comp.number("rows", i, error_key, m) for i in indices]) for m in methods}
    return scores


def paired_rows(comp, freeze, identity, observed):
    """Preserve V83's prediction-identity and absolute-error validation for ALL forms."""
    key = lambda r: tuple(r[k] for k in identity)
    frozen = {key(r): r for r in freeze.data["predictions"]}
    measured = {key(r): r for r in comp.data["rows"]}
    require(len(frozen) == len(freeze.data["predictions"]), "Duplicate frozen rows")
    require(len(measured) == len(comp.data["rows"]), "Duplicate measured rows")
    require(frozen.keys() == measured.keys(), "Incomplete frozen comparison")
    for ident, row in measured.items():
        require(row["predictions"] == frozen[ident]["predictions"], f"Changed predictions: {ident}")
        require(row["absolute_errors"].keys() == row["predictions"].keys(), f"Missing forms: {ident}")
        for method, prediction in row["predictions"].items():
            close(abs(prediction-row[observed]), row["absolute_errors"][method], f"{ident}/{method}")


def build_rows():
    inputs, rows = [], []

    def read(path):
        comp = Comparison("results/" + path)
        inputs.append(comp)
        return comp

    panel = [read(f"v53-prune-dev/compare_pythia-{tag}.json") for tag in
             ("410m@step48000", "1.4b@step112000", "6.9b@step80000")]
    register = read("v53-prune-dev/register.json")
    methods = tuple(panel[0].data["mae"])
    require(register.data["selected_candidate"] == "power", "C35 frozen selection changed")
    for comp in panel:
        d = comp.data
        pred = read(comp.path.removeprefix("results/").replace("compare_", "predictions_"))
        require(d["provenance"]["predictions_sha256"] == pred.sha256, "C35 prediction hash")
        require(pred.data["provenance"]["register_sha256"] == register.sha256, "C35 register hash")
        require(comp.path.endswith("compare_" + d["tag"] + ".json"), "C35 state tag changed")
        require(d["selected_candidate"] == pred.data["selected_candidate"] == "power", "C35 candidate")
        require(sorted(d["densities"]) == [.575, .675, .85], "C35 density grid changed")
        require(set(d["mae"]) == set(methods), "C35 predictor roster differs across states")
        for cap in CAPS:
            for density in d["densities"]:
                ds = str(density)
                require(set(d["absolute_errors"][cap][ds]) == set(methods), "C35 missing forms")
                for method in methods:
                    close(abs(pred.data["predictions"][cap][ds][method]-d["observed_delta_loss"][cap][ds]),
                          d["absolute_errors"][cap][ds][method], f"C35/{cap}/{ds}/{method}")
            for method in methods:
                close(mean(d["absolute_errors"][cap][str(x)][method] for x in d["densities"]),
                      d["mae"][method][cap], f"C35/{cap}/{method}")
    scores = {c: {m: average([p.number("mae", m, c) for p in panel]) for m in methods} for c in CAPS}
    rows.append(Row("C35", "Pruning", "Power (frozen)",
        r"Three checkpoints: 410M@48k, 1.4B@112k, 6.9B@80k; $d=0.85,\allowbreak0.675,\allowbreak0.575$",
        scores, same("power"), same("median_curve"), "fixed after test; reused frozen in Sec. 5",
        "New stages of seen sizes",
        "; ".join(p.ref("tag") + ", " + p.ref("densities") for p in panel)
        + "; 3 states = count(files); 9 cells = sum(len(densities))", 9,
        "Pool all nine cells equally, then minimize pooled MAE over all six non-power predictors. "
        "All three checkpoints are new states for the locked rule; deliver the median development curve.",
        register.ref("selected_candidate"), "; ".join(p.ref("mae") for p in panel)))

    comp, freeze = read("v72-prune-repeat/compare.json"), read("v72-prune-repeat/freeze.json")
    require(provenance.matches(comp.data["provenance"]["freeze_sha256"], freeze.sha256),
            "C52 freeze hash")
    require(freeze.data["selected_candidate"] == "power", "C52 candidate changed")
    paired_rows(comp, freeze, ("source", "density", "capability"), "observed_delta_loss")
    rr = list(enumerate(comp.data["rows"]))
    require(len(rr) == 18, "C52 must contain all 18 capability rows")
    states = sorted({r["source"] for _, r in rr})
    require(states == ["pythia-2.8b@step143000", "pythia-2.8b@step16000"], "C52 labels changed")
    methods = tuple(comp.data["scores"])
    require(set(methods) == set(freeze.data["methods"]), "C52 missing methods")
    for state in states:
        for cap in CAPS:
            require(sorted(r["density"] for _, r in rr if r["source"] == state and r["capability"] == cap)
                    == [.65, .75, .85], "C52 grid incomplete")
    scores = row_scores(comp, rr, methods)
    for cap in CAPS:
        for method in methods:
            stored = comp.number("scores", method, "by_capability", cap, "mae")
            close(scores[cap][method].value, stored.value, f"C52/{cap}/{method}")
            scores[cap][method] = stored
    rows.append(Row("C52", "Pruning", "Power (frozen)",
        r"2.8B repeat: one state, 16k/143k labels; $d=0.85,\allowbreak0.75,\allowbreak0.65$",
        scores, same("power"), same("median_curve"), "fixed after test", "One state measured twice",
        comp.ref("rows") + " (source, density, capability); learned-state identity: RESULTS_LEDGER.md C52", 6,
        "Pool both revision labels equally, preserving the six-record MAE per capability. "
        "The two labels load identical weights; they are not independent model states.",
        freeze.ref("selected_candidate"), comp.ref("scores")))

    comp, freeze = read("v69-quant-confirm/compare.json"), read("v69-quant-confirm/freeze.json")
    require(comp.data["complete"] and len(comp.data["rows"]) == 63, "Quantization incomplete")
    require(comp.data["provenance"]["freeze_sha256"] == freeze.sha256, "Quantization freeze hash")
    require(comp.data["selected"] == freeze.data["selected"], "Quantization selection changed")
    paired_rows(comp, freeze, ("state", "config", "capability"), "dL")
    methods = tuple(comp.data["regimes"]["all"])
    require(set(methods) == set(freeze.data["methods"]), "Quantization missing frozen methods")
    candidates = {c: comp.data["selected"][c]["candidate"] for c in CAPS}
    require(candidates == dict(zip(CAPS, ("low_order_2d", "median", "zero"))), "Frozen D changed")
    for r in comp.data["rows"]:
        require(r["selected_candidate"] == candidates[r["capability"]], "Row candidate differs from D")
        close(r["selected_prediction"], r["predictions"][r["selected_candidate"]], "Selected prediction")
    for new in (False, True):
        subsets = ("new_state_boundary", "new_state_interior") if new else ("development_state_boundary",)
        rr = [(i, r) for i, r in enumerate(comp.data["rows"]) if r["test_set"] in subsets]
        states = {"pythia-1.4b@step112000"} if new else {"pythia-410m@step143000", "pythia-1.4b@step16000"}
        groups = (32, 128, 512) if new else (32, 512)
        for cap in CAPS:
            actual = [(r["state"], r["config"]) for _, r in rr if r["capability"] == cap]
            expected = [(s, f"b{b}_g{g}") for s in states for b in (3, 4, 5) for g in groups]
            require(sorted(actual) == sorted(expected), "Quantization grid incomplete/duplicated")
        scores = row_scores(comp, rr, methods)
        for subset in subsets:
            subrows = [(i, r) for i, r in rr if r["test_set"] == subset]
            ss = row_scores(comp, subrows, methods)
            for cap in CAPS:
                for method in methods:
                    metric = comp.data["test_sets"][subset]["scores"][method][cap]
                    require(metric["n"] == sum(r["capability"] == cap for _, r in subrows), "Subset count")
                    close(ss[cap][method].value, metric["mae"], f"Quantization/{subset}/{cap}/{method}")
        rows.append(Row("C46" if new else "C44", "Grouped quant.", "Frozen D: surface / median / zero",
            (r"New state: 1.4B@112k; $b=3,4,5$; $g=32,128,512$" if new else
             r"Dev. states: 410M@143k, 1.4B@16k; $b=3,4,5$; $g=32,512$"),
            scores, candidates.copy(), same("median") if new else
            {c: "median" if c == "qa" else "same_input_interpolation" for c in CAPS},
            "fixed after test", "New state" if new else "New group sizes; seen states",
            comp.ref("rows") + " filtered by test_set in " + repr(subsets)
            + "; states=unique(state); bits and group sizes parsed from config", 9 if new else 12,
            "Candidate D is the development-selected surface/median/zero, not the retrospective delivered rule. "
            + ("Pool six boundary and three interior cells equally (2:1 stratum weights)." if new else
               "Reproduce V74's twelve equally weighted boundary cells on the two development states."),
            comp.ref("selected"), comp.ref("regimes", "all")))

    comp, freeze = read("v70-distill-confirm/compare.json"), read("v70-distill-confirm/freeze.json")
    require(comp.data["complete"] and len(comp.data["rows"]) == 108, "Distillation incomplete")
    require(comp.data["freeze_sha256"] == freeze.sha256, "Distillation freeze hash")
    require(comp.data["bootstrap"] == freeze.data["bootstrap"], "Bootstrap protocol changed")
    paired_rows(comp, freeze, ("student", "pool", "T_planned", "capability"), "actual")
    require(len(comp.data["groups"]) == 6, "Expected six distillation groups")
    for student, label, key in (("gemma3-270m", "270M", "C47"), ("gemma3-1b", "1B", "C48")):
        scores, candidates, recorded, selection_refs, roster_refs = {}, {}, {}, [], []
        for cap in CAPS:
            matches = [(i, g) for i, g in enumerate(comp.data["groups"])
                       if g["student"] == student and g["capability"] == cap]
            require(len(matches) == 1, "Distillation group missing/duplicated")
            i, g = matches[0]
            candidate, baseline = g["selected"], g["strongest_baseline"]
            require(candidate == freeze.data["selected"][cap]["method"], "Frozen candidate changed")
            require(baseline == freeze.data["strongest_baseline"][student][cap]["method"], "Recorded baseline changed")
            candidates[cap], recorded[cap] = candidate, baseline
            selection_refs.append(comp.ref("groups", i, "selected"))
            roster_refs.append(comp.ref("groups", i, "all_method_mae"))
            methods = tuple(g["all_method_mae"])
            rr = [r for r in comp.data["rows"] if r["student"] == student and r["capability"] == cap]
            require(len(rr) == g["n_checkpoints"] == 18 and g["n_pools"] == 6, "Distillation count")
            require(sorted((r["pool"], r["T_planned"]) for r in rr) ==
                    [(f"U200_s{s}", t) for s in range(31, 37) for t in (50000, 100000, 200000)],
                    "Distillation pools/budgets changed")
            require(all(set(r["absolute_errors"]) == set(methods) for r in rr), "Missing recorded forms")
            require([cl["pool"] for cl in g["clusters"]] == [f"U200_s{s}" for s in range(31, 37)], "Pool order")
            scores[cap] = {}
            for method in methods:
                for cluster in g["clusters"]:
                    close(mean(r["absolute_errors"][method] for r in rr if r["pool"] == cluster["pool"]),
                          cluster["mae"][method], f"{key}/{cap}/{method}/{cluster['pool']}")
                stored = comp.number("groups", i, "all_method_mae", method)
                close(mean(cl["mae"][method] for cl in g["clusters"]), stored.value, f"{key}/{cap}/{method}")
                close(mean(r["absolute_errors"][method] for r in rr), stored.value, f"{key}/pooled/{cap}/{method}")
                scores[cap][method] = stored
            close(scores[cap][candidate].value, g["candidate_mae"], "Stored distillation candidate MAE")
            close(scores[cap][baseline].value, g["baseline_mae"], "Recorded baseline MAE")
            close(g["baseline_mae"]-g["candidate_mae"], g["paired_difference"]["estimate"], "Recorded gain")
        rows.append(Row(key, "Distillation", r"Frozen $E$ / $E$ / joint",
            rf"Gemma-3-{label}; 6 pools, $U=200$; $T=50,100,200$k", scores, candidates,
            candidates.copy(), "fixed before test", "Seen student; new pools; 2Wiki QA",
            comp.ref("groups") + " filtered by student=" + student
            + "; n_pools, n_checkpoints; rows/pool encodes U; rows/T_planned gives token budgets", 18,
            "Equal weight over six pools and three planned budgets per pool. Compare the frozen selected form "
            "with the minimum of every other recorded form; report the development-fixed baseline mismatch. "
            "E is zero-anchored reuse; E-only has an intercept. No old baseline CI is reassigned to the new minimum.",
            "; ".join(selection_refs), "; ".join(roster_refs), recorded))

    comp = read("v50-p2v2/compare_test.json")
    for student, label in (("gemma3-270m", "270M"), ("gemma3-1b", "1B"), ("gemma3-4b", "4B")):
        rr = [(i, r) for i, r in enumerate(comp.data["rows"]) if r["student"] == student and r["role"] == "test_pool"]
        methods = tuple(rr[0][1]["abs"])
        require(all(set(r["abs"]) == set(methods) == set(r["pred_at_actual"]) for _, r in rr), "U375 forms")
        scores = row_scores(comp, rr, methods, cap_key="cap", error_key="abs")
        for cap in CAPS:
            cr = [(i, r) for i, r in rr if r["cap"] == cap]
            require(len(cr) == 12 and {r["pool"] for _, r in cr} == {f"U375_s{s}" for s in (21, 22, 23)}, "U375 pools")
            for pool in {r["pool"] for _, r in cr}:
                require(sum(r["pool"] == pool for _, r in cr) == 4, "U375 missing budget")
            keys = [f"{student}|test_pool|{cap}|T{t}" for t in (35000, 70000, 140000, 280000)]
            for method in methods:
                for _, r in cr:
                    close(abs(r["pred_at_actual"][method]-r["actual"]), r["abs"][method], "U375 error")
                require(all(set(comp.data["summary"][k]["mae"]) == set(methods) for k in keys), "U375 summary forms")
                stored = average([comp.number("summary", k, "mae", method) for k in keys])
                close(scores[cap][method].value, stored.value, f"U375/{student}/{cap}/{method}")
                scores[cap][method] = stored
            # Zero is also a recorded predictor, in a separate field; do not silently omit it.
            for _, r in cr:
                close(abs(r["actual"]), r["abs_zero"], "U375 zero error")
            zero = average([comp.number("summary", k, "mae_zero") for k in keys])
            close(mean(r["abs_zero"] for _, r in cr), zero.value, "U375 pooled zero")
            scores[cap]["zero"] = zero
        rows.append(Row("C38-" + label, "Distill. context", "Frozen joint+src",
            rf"Gemma-3-{label}; 3 pools, $U=375$; $T=35,70,140,280$k", scores,
            same("joint+src"), same(None), "rule stated after test", "Held-out size" if label == "4B" else "Seen student; new pools",
            comp.ref("rows") + " filtered by student=" + student + ", role=test_pool; pool encodes U; "
            + comp.ref("summary") + " keys give four planned budgets", 12,
            "Earlier context, not delivered. All eight forms plus the separately recorded zero reference are "
            "eligible before excluding joint+src. Average four equal-size budget summaries (three pools each), "
            "validated against twelve raw errors. Predictions use actual achieved Tc/E as in V50.",
            "V86 specification; paper/docs/RESULTS_LEDGER.md C38 (headline rule stated after test)",
            comp.ref("summary") + "/*/mae and /*/mae_zero"))
    require([r.key for r in rows] == ["C35", "C52", "C44", "C46", "C47", "C48", "C38-270M", "C38-1B", "C38-4B"], "Row order")
    for row in rows:
        for cap in CAPS:
            require(row.candidates[cap] in row.scores[cap] and len(row.alternatives[cap]) > 0, "Candidate/alternatives")
            require(set(row.alternatives[cap]) | {row.candidates[cap]} == set(row.scores[cap]), "Incomplete alternatives")
            require(all(math.isfinite(n.value) and n.value >= 0 for n in row.scores[cap].values()), "Invalid MAE")
    return rows, inputs


def stack(values):
    return r"\newline ".join(values)


def render_tex(rows, inputs, context=False):
    # Presentation only: keep the shared row records and four-decimal summaries
    # unchanged, including the test metadata imported by Figure 2.
    test_objects = {
        "C35": r"3 checkpoints (410M@48k, 1.4B@112k, 6.9B@80k); $d$ 0.85/0.675/0.575",
        "C52": r"2.8B, one state (16k/143k labels); $d$ 0.85/0.75/0.65",
        "C44": r"Dev. 410M@143k, 1.4B@16k; $b$ 3--5; $g$ 32, 512",
        "C46": r"New 1.4B@112k; $b$ 3--5; $g$ 32/128/512",
        "C47": r"270M; 6 pools $U=200$; $T$ 50/100/200k",
        "C48": r"1B; 6 pools $U=200$; $T$ 50/100/200k",
        **{f"C38-{label}": rf"{label}; 3 pools $U=375$; $T$ 35/70/140/280k"
           for label in ("270M", "1B", "4B")},
    }
    timing_labels = {
        "fixed after test; reused frozen in Sec. 5": "after test; reused in Sec. 5",
        "fixed after test": "after test",
        "fixed before test": "before test",
        "rule stated after test": "after test",
    }
    # Exactly textwidth: subtract all twelve internal tabcolseps proportionally.
    widths = (.127, .230, .097, .163, .073, .130, .180)
    close(sum(widths), 1, "column width sum")
    columns = "@{}" + "".join(r">{\raggedright\arraybackslash}p{\dimexpr " + f"{w:.3f}" +
        r"\textwidth-" + f"{12*w:.3f}" + r"\tabcolsep\relax}" for w in widths) + "@{}"
    lines = ["% Generated by analysis/v86_main_table.py; do not edit by hand."]
    lines += [f"% INPUT {c.path} sha256={c.sha256}" for c in inputs]
    lines += [r"\begin{table}[H]" if context else r"\begin{table}[t]", r"\centering\scriptsize",
              r"\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{1.06}",
              r"\begin{tabular}{" + columns + "}", r"\toprule",
              r"Arm and frozen candidate & Test object & Candidate MAE & Strongest frozen alternative (name, MAE) & Gain & Delivered rule & Scope \\",
              r"\midrule"]
    for i, row in enumerate(rows):
        if not context and i in (2, 4):
            lines.append(r"\midrule")
        elif i:
            lines.append(r"\addlinespace[2pt]")
        lines += ["% SOURCE " + trace.replace("rounded independently to 4 decimals",
                                             "rounded independently to 3 decimals")
                  for trace in row.traces()]
        av = [row.scores[c][row.candidates[c]].value for c in CAPS]
        bv = [row.scores[c][row.baselines[c]].value for c in CAPS]
        if all(d is None for d in row.delivered.values()):
            delivered = "Not delivered"
        else:
            delivered = stack("same" if row.delivered[c] == row.candidates[c] else
                f"{TABLE_LABELS[row.delivered[c]]} ${row.scores[c][row.delivered[c]].value:.3f}$"
                for c in CAPS)
        predictor = "Frozen D" if row.key in ("C44", "C46") else row.predictor
        cells = [stack((row.arm, predictor)), test_objects[row.key],
                 stack(f"${v:.3f}$" for v in av),
                 stack(f"{TABLE_LABELS[row.baselines[c]]} ${v:.3f}$" for c, v in zip(CAPS, bv)),
                 stack(f"${b-a:+.3f}$" for a, b in zip(av, bv)), delivered,
                 r"\emph{" + timing_labels[row.timing] + r"}\newline " + row.scope]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
        r"\caption{\scriptsize " + ("Earlier distillation context. " if context else "Frozen candidates and delivered rules. ") +
        r"MAE in nats per token; stacks: math/code/QA. Strongest: lowest pooled test MAE among all "
        r"predictors frozen in that round except the candidate (hindsight ranking). Gain = alternative "
        r"minus candidate; negative = alternative better. Delivered rules fixed after a test are "
        r"retrospective for it. " +
        (r"The joint+src rule was stated after test; not delivered.}" if context else
         r"Frozen D = surface/median/zero.}"),
        r"\label{tab:main_context}" if context else r"\label{tab:main_final}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def row_record(row):
    caps = {}
    for c in CAPS:
        a, b, d = row.candidates[c], row.baselines[c], row.delivered[c]
        av, bv = row.scores[c][a].value, row.scores[c][b].value
        dv = row.scores[c][d].value if d is not None else None
        caps[c] = {
            "candidate": a, "candidate_mae": av,
            "alternative_set": row.alternatives[c], "strongest_alternative": b, "alternative_mae": bv,
            "gain": bv-av, "delivered": d, "delivered_mae": dv,
            "delivered_gain": bv-dv if dv is not None else None,
            "delivered_differs": d is not None and d != a,
            "all_method_mae": {m: n.value for m, n in row.scores[c].items()},
            "mae_sources": {m: n.source for m, n in row.scores[c].items()},
            "display": {"candidate_mae": f"{av:.4f}", "alternative_mae": f"{bv:.4f}",
                        "gain": f"{bv-av:+.4f}", "delivered_mae": f"{dv:.4f}" if dv is not None else None},
        }
    return {"row": row.key, "arm": row.arm, "test_tex": row.test, "n_per_capability": row.n,
            "capabilities": caps, "delivered_timing": row.timing, "scope": row.scope,
            "candidate_source": row.selection_source, "roster_source": row.roster_source,
            "test_source": row.metadata, "rationale": row.rationale, "source_traces": list(row.traces())}


def check_block(rows, inputs):
    indexed = {r.key: r for r in rows}
    checks = []
    expected = [("C46", "qa", "zero", "0.2217"), ("C46", "qa", "median", "0.1412"),
                ("C46", "math", "median", "0.0882"), ("C46", "math", "same_input_interpolation", "0.1412")]
    for key, cap, method, target in expected:
        n = indexed[key].scores[cap][method]
        require(f"{n.value:.4f}" == target, f"Required check {key}/{cap}/{method}")
        checks.append({"row": key, "capability": cap, "method": method,
                       "actual": n.value, "expected_4dp": target, "passed": True, "source": n.source})
    # Check the rendered V74 table itself, not just hard-coded rounded values.
    v74_path = TABLES / "quant_threeway.tex"
    v74_raw = v74_path.read_bytes()
    section = v74_raw.decode().split(r"\addlinespace")[0]
    dev_checks = []
    for label, methods, targets in (
        ("FROZEN development-selected", indexed["C44"].candidates, ("0.2149", "0.5567", "0.4570")),
        ("Piecewise interpolation", same("same_input_interpolation"), ("0.0653", "0.1163", "0.4766")),
        ("Per-configuration median", same("median"), ("0.4998", "0.5567", "0.4579")),
    ):
        line = next(line for line in section.splitlines() if "& " + label + " &" in line)
        table_numbers = tuple(re.findall(r"\d+\.\d{4}", line))
        values = tuple(indexed["C44"].scores[c][methods[c]].value for c in CAPS)
        require(tuple(f"{v:.4f}" for v in values) == table_numbers == targets, f"V74 mismatch: {label}")
        dev_checks.append({"label": label, "actual": values, "expected_4dp": targets, "passed": True,
                           "source": "paper/paper/tables/quant_threeway.tex: " + line})
    mismatches = []
    baseline_audit = []
    for row in rows:
        if row.recorded_baselines is None:
            continue
        for cap in CAPS:
            recorded, strongest = row.recorded_baselines[cap], row.baselines[cap]
            rv, sv = (row.scores[cap][m].value for m in (recorded, strongest))
            entry = {"row": row.key, "capability": cap, "recorded_strongest_baseline": recorded,
                     "recorded_mae": rv, "minimum_over_all_alternatives": strongest, "minimum_mae": sv,
                     "matches_minimum": math.isclose(rv, sv, rel_tol=1e-11, abs_tol=1e-13),
                     "recorded_source": row.scores[cap][recorded].source,
                     "minimum_source": row.scores[cap][strongest].source}
            baseline_audit.append(entry)
            if not entry["matches_minimum"]:
                mismatches.append(entry)
    # V84 is read-only and deterministic; its reconstruction equals the pre-V86 table.
    import v84_main_table as previous
    old_rows, old_inputs = previous.build_rows()
    old_keys = {"C35": "C35", "C52": "C50/C52", "C44": "C46-dev", "C46": "C46-new",
                "C47": "C48-270M", "C48": "C48-1B", **{f"C38-{s}": f"C38-{s}" for s in ("270M", "1B", "4B")}}
    old_by_key = {r.key: r for r in old_rows}
    differences = []
    for row in rows:
        old = old_by_key[old_keys[row.key]]
        changed_caps, before = [], {}
        for cap in CAPS:
            a, b = old.candidates[cap], old.baselines[cap]
            av, bv = old.scores[cap][a].value, old.scores[cap][b].value
            before[cap] = {"candidate": a, "candidate_mae": av, "alternative": b, "alternative_mae": bv, "gain": bv-av}
            if (a != row.candidates[cap] or b != row.baselines[cap]
                or not math.isclose(av, row.scores[cap][row.candidates[cap]].value, abs_tol=1e-13)
                or not math.isclose(bv, row.scores[cap][row.baselines[cap]].value, abs_tol=1e-13)):
                changed_caps.append(cap)
        reasons = {
            "C35": "All six non-power predictors replace the restricted A2/median set; new-stage delivery is now median, fixed after test and reused frozen in Sec. 5.",
            "C52": "All three non-power predictors are checked; median remains strongest. Delivery is explicitly median, fixed after test; one learned state remains one state.",
            "C44": "Frozen D replaces retrospective interpolation as candidate; all other five predictors enter the alternative set. Delivered interpolation/interpolation/median is separate, fixed after test. Corrected row ID to C44.",
            "C46": "Frozen D replaces retrospective median as candidate; all other five predictors enter the alternative set. Delivered median is separate, fixed after test; boundary/interior pooling remains 2:1.",
            "C47": "All fifteen nonselected forms replace the recorded development-fixed baseline; all three recorded baselines differ from the full-set minimum. Selected/delivered form remains fixed before test. Corrected row ID to C47.",
            "C48": "All fifteen nonselected forms replace the recorded development-fixed baseline; all three recorded baselines differ from the full-set minimum. Selected/delivered form remains fixed before test.",
        }
        reason = reasons.get(row.key, "Moved to appendix main_context; all recorded forms and the separate zero reference replace the constant-only comparison. Not delivered; rule stated after test.")
        differences.append({"row": row.key, "previous_row": old.key, "changed_capabilities": changed_caps,
                            "numerical_roles_changed": bool(changed_caps), "why": reason, "previous_values": before})
    return {"four_required_numbers": checks, "v74_development_checks": dev_checks,
            "v74_table_sha256": hashlib.sha256(v74_raw).hexdigest(),
            "distillation_recorded_baseline_audit": baseline_audit,
            "distillation_baseline_mismatches": mismatches, "mismatch_count": len(mismatches),
            "previous_main_table": {"generator": "analysis/v84_main_table.py",
                "sha256": hashlib.sha256(previous.render_tex(old_rows, old_inputs).encode()).hexdigest(),
                "changes": differences},
            "complete_alternative_sets": True, "equal_cell_weights": True,
            "new_state_boundary_interior_weight_ratio": "6:3 (2:1)",
            "main_rows": 6, "context_rows": 3}


def printable_rows(report, context=False):
    lines = ["| Test | Candidate MAE (M/C/Q) | Strongest frozen alternative: name, MAE (M/C/Q) | Signed gain (M/C/Q) | Delivered rule: name, MAE (M/C/Q); timing |",
             "|---|---|---|---|---|"]
    for row in report["context_rows" if context else "main_rows"]:
        caps = [row["capabilities"][c] for c in CAPS]
        candidate = " / ".join(r["display"]["candidate_mae"] for r in caps)
        alternative = " / ".join(f"{r['strongest_alternative']} {r['display']['alternative_mae']}" for r in caps)
        gain = " / ".join(r["display"]["gain"] for r in caps)
        delivered = " / ".join(f"{r['delivered']} {r['display']['delivered_mae']}" if r["delivered"] else "not delivered" for r in caps)
        lines.append(f"| {row['row']} | {candidate} | {alternative} | {gain} | {delivered}; {row['delivered_timing']} |")
    return lines


def printable_checks(checks):
    lines = ["CHECK BLOCK"]
    for check in checks["four_required_numbers"]:
        lines.append(f"PASS {check['row']} {check['capability']} {check['method']}: {check['expected_4dp']}")
    for check in checks["v74_development_checks"]:
        lines.append("PASS V74 development " + check["label"] + ": " + "/".join(check["expected_4dp"]))
    lines.append("PASS full recorded alternative sets; equal cell weights; new-state boundary/interior weights 6:3.")
    lines.append(f"AUDIT {checks['mismatch_count']}/6 recorded distillation baselines differ from the full-set minimum:")
    for m in checks["distillation_baseline_mismatches"]:
        lines.append(f"  {m['row']} {m['capability']}: recorded {m['recorded_strongest_baseline']} {m['recorded_mae']:.4f}; "
                     f"minimum {m['minimum_over_all_alternatives']} {m['minimum_mae']:.4f}")
    for change in checks["previous_main_table"]["changes"]:
        lines.append(f"V84 -> {change['row']}: numeric roles changed in {','.join(change['changed_capabilities']) or 'none'}. {change['why']}")
    return lines


def render_summary(report):
    lines = ["# V86 frozen candidate evaluation", "",
        "CPU/stdlib; no fitting. All compare/freeze inputs are read-only. "
        "Regenerate: `python3 analysis/v86_main_table.py`; check: add `--check`. "
        "The figure imports the same rows and checks its values against summary.json.", "",
        "MAE is in nats per token. Each stack is math/code/QA. All methods are pooled before "
        "choosing the minimum over every recorded alternative, excluding the candidate. "
        "This is a test-MAE ranking of frozen predictors, not a pre-test selection of the strongest. "
        "Gain = alternative minus candidate; negative favors the alternative. Delivery choices fixed "
        "after test are retrospective for that test. QA is the primary 2Wiki conditional-loss probe.", "",
        "Names: A2 = per-density regression; median_curve = median development curve; "
        "median = per-configuration median; interp. = same-input interpolation; surface = 2-D surface. "
        "Distillation E is the zero-anchored reuse form, while E-only includes an intercept. "
        "Other form names retain the frozen file identifiers. Context includes the separate zero reference.", "",
        *printable_rows(report), "", "## Appendix context", "", *printable_rows(report, True), "",
        "## Checks and changes from the previous table", "", "```text", *printable_checks(report["checks"]), "```", "",
        "The V84 generator reconstructs the pre-change main_final.tex byte for byte; its SHA-256 is "
        + report["checks"]["previous_main_table"]["sha256"] + ". All rows now separate delivery and its timing; "
        "display precision is four decimals (V84 used three), independently rounded from full precision.", "",
        "## Input SHA-256", ""]
    lines += [f"- `{p}`: `{h}`" for p, h in report["input_sha256"].items()]
    for row in report["main_rows"] + report["context_rows"]:
        lines += ["", "## " + row["row"], "", row["rationale"], "",
                  "| Capability | Full alternative set |", "|---|---|"]
        for cap in CAPS:
            r = row["capabilities"][cap]
            lines.append(f"| {cap} | " + "; ".join(f"`{m}` ({r['all_method_mae'][m]!r})" for m in r["alternative_set"]) + " |")
        lines += ["", "```text", *row["source_traces"], "```"]
    return "\n".join(lines) + "\n"


def unchanged(inputs):
    for comp in inputs:
        require(hashlib.sha256((ROOT/comp.path).read_bytes()).hexdigest() == comp.sha256,
                f"Input changed during generation: {comp.path}")


def write_output(path, content, check=False):
    require(path.parent == TABLES or path.parent == OUT, f"Output outside authorized locations: {path}")
    require(not path.is_symlink() and all(not p.is_symlink() for p in path.parents if p != ROOT),
            f"Refusing symlink output: {path}")
    if check:
        require(path.is_file() and path.read_text() == content, f"Missing/stale output: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    print(f"{'CHECKED' if check else 'WROTE'} {path.relative_to(ROOT)}")


def regenerate_locked_tables(check=False):
    """Run all V85 validations but never its entry point that writes V85 results."""
    import v85_selection_decomp as v85
    inputs, outputs = v85.build_outputs()
    inputs.unchanged()
    for name in ("locked_rule.tex", "rule_decomp.tex"):
        content = outputs[f"paper/paper/tables/{name}"]
        require(r"\begin{table}[H]" in content, "V85 placement must remain [H]")
        write_output(TABLES/name, content, check)
    inputs.unchanged()
    print("V85 validation passed: 272 original cells, 48 policy means, 12 paired differences, 24 rule rows; only two paper tables written.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate outputs without writes")
    parser.add_argument("--regenerate-locked-tables", action="store_true", help="V85 table-only regeneration entry point")
    args = parser.parse_args()
    if args.regenerate_locked_tables:
        regenerate_locked_tables(args.check)
        return
    rows, inputs = build_rows()
    checks = check_block(rows, inputs)
    report = {"schema_version": 1, "units": "MAE in nats per token", "capability_order": CAPS,
              "gain_definition": "strongest frozen alternative MAE minus frozen candidate MAE",
              "input_sha256": {c.path: c.sha256 for c in inputs},
              "main_rows": [row_record(r) for r in rows[:6]], "context_rows": [row_record(r) for r in rows[6:]], "checks": checks}
    outputs = {TABLES/"main_final.tex": render_tex(rows[:6], inputs),
               TABLES/"main_context.tex": render_tex(rows[6:], inputs, context=True),
               OUT/"summary.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
               OUT/"summary.md": render_summary(report)}
    unchanged(inputs)
    for path, content in outputs.items():
        write_output(path, content, args.check)
    unchanged(inputs)
    print("\n".join([*printable_rows(report), "", *printable_rows(report, True), "", *printable_checks(checks)]))


if __name__ == "__main__":
    main()
