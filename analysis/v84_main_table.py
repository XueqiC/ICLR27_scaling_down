#!/usr/bin/env python3
"""Generate main Table 1 from frozen-comparison results, using only the CPU/stdlib.

Run: python -B analysis/v84_main_table.py [--check]
Writes only paper/paper/tables/main_final.tex and main_final_sources.md.
No fitting, measurements, result writes, or imports of existing table generators.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "paper/paper/tables"
CAPS = ("math", "code", "qa")
LABELS = {
    "power": "Power", "A2": "A2", "median_curve": "Median",
    "same_input_interpolation": "Interp.", "low_order_2d": "Surface",
    "median": "Median", "E": "$E$", "joint": "Joint",
    "T-only": "$T$-only", "E-only": "$E$-only",
    "surface:L0": "Surf. $L_0$", "surface:logN": "Surf. $\\log N$",
    "joint+src": "Joint+src", "constant": "Constant",
}
FROZEN = "frozen before measurement"
AFTER_TEST = "rule stated after test"


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
    baselines: dict
    provenance: str
    scope: str
    metadata: str
    rationale: str

    def traces(self):
        yield f"{self.key} test object: {self.metadata}"
        for cap in CAPS:
            for method, number in self.scores[cap].items():
                yield (f"{self.key} {cap} {method} MAE = {number.value:.17g} <- "
                       f"{number.source}")
            candidate = self.scores[cap][self.candidates[cap]]
            baseline = self.scores[cap][self.baselines[cap]]
            yield (f"{self.key} {cap} candidate = {candidate.value:.17g}; "
                   f"baseline ({self.baselines[cap]}) = {baseline.value:.17g}; "
                   f"gain = {baseline.value - candidate.value:.17g} <- "
                   f"({baseline.source}) - ({candidate.source}); "
                   "round each displayed number independently to 3 decimals")


def same(method):
    return dict.fromkeys(CAPS, method)


def row_scores(comp, rows, methods, cap_key="capability", error_key="absolute_errors"):
    """Average observed comparison errors, retaining exact JSON pointers."""
    scores = {}
    for cap in CAPS:
        indices = [i for i, r in rows if r[cap_key] == cap]
        scores[cap] = {m: average([comp.number("rows", i, error_key, m) for i in indices])
                       for m in methods}
    return scores


def build_rows():
    inputs = []

    def read(path):
        comp = Comparison("results/" + path)
        inputs.append(comp)
        return comp

    rows = []
    panel = [read(f"v53-prune-dev/compare_pythia-{tag}.json") for tag in
             ("410m@step48000", "1.4b@step112000", "6.9b@step80000")]
    scores = {c: {} for c in CAPS}
    for comp in panel:
        d = comp.data
        require(comp.path.endswith("compare_" + d["tag"] + ".json"), "C35 state tag changed")
        require(d["selected_candidate"] == "power", "C35 selected form changed")
        require(sorted(d["densities"]) == [.575, .675, .85], "C35 density grid changed")
        for cap in CAPS:
            for method in ("power", "A2", "median_curve"):
                close(mean(d["absolute_errors"][cap][str(x)][method] for x in d["densities"]),
                      d["mae"][method][cap], f"{comp.path}/{cap}/{method}")
    for cap in CAPS:
        for method in ("power", "A2", "median_curve"):
            scores[cap][method] = average([c.number("mae", method, cap) for c in panel])
    rows.append(Row(
        "C35", "Pruning", "Power form",
        r"3 new states: 410M@48k, 1.4B@112k, 6.9B@80k; $d=0.85,\allowbreak 0.675,\allowbreak 0.575$",
        scores, same("power"),
        {c: min(("A2", "median_curve"), key=lambda m: scores[c][m].value) for c in CAPS},
        FROZEN + r"; baseline: retrospective", "QA fails",
        "; ".join(c.ref("tag") + ", " + c.ref("densities") for c in panel)
        + "; 3 states = count(files); 9 cells/capability = sum(len(densities))",
        "Equal-cell pooled MAE over three states and three densities each. Strongest baseline is "
        "the lower pooled confirmation MAE of A2 and median_curve per capability, chosen AFTER pooling, "
        "not the minimum error at each target. Predictions and power selection were frozen; "
        "the strongest-baseline ranking uses test MAE (retrospective)."))

    comp = read("v72-prune-repeat/compare.json")
    rr = list(enumerate(comp.data["rows"]))
    require(len(rr) == 18, "C50 repeat must contain all 18 capability rows")
    states = sorted({r["source"] for _, r in rr})
    require(states == ["pythia-2.8b@step143000", "pythia-2.8b@step16000"], "C50 states changed")
    for state in states:
        for cap in CAPS:
            require(sorted(r["density"] for _, r in rr if r["source"] == state
                           and r["capability"] == cap) == [.65, .75, .85], "C50 grid incomplete")
    scores = row_scores(comp, rr, ("power", "A2", "median_curve", "zero"))
    for cap in CAPS:
        for method in scores[cap]:
            stored = comp.number("scores", method, "by_capability", cap, "mae")
            close(scores[cap][method].value, stored.value, f"C50/{cap}/{method}")
            scores[cap][method] = stored
    rows.append(Row(
        "C50/C52", "Pruning", r"Median curve delivered; power candidate",
        r"2.8B: 1 state repeated (16k/143k labels); $d=0.85,\allowbreak 0.75,\allowbreak 0.65$",
        scores, same("power"), same("median_curve"),
        FROZEN + "; delivery: retrospective", "One state only",
        comp.ref("rows") + " fields source, density, capability; "
        + comp.ref("n_configurations") + " = 6 recorded cells; unique learned-state count = 1 "
        "per paper/docs/RESULTS_LEDGER.md C52 (weight identity correction to C50)",
        "Power remains the scored candidate; median_curve is both the baseline and delivered predictor. "
        "Its delivery was decided retrospectively. Preserve the published six-record MAEs. "
        "C52 identifies the two revision labels as one learned state measured twice; do not relabel "
        "the prediction inputs or treat the repeats as independent states. A2 and zero are included "
        "in the source audit; median_curve has the lowest MAE on every capability."))
    require(all(scores[c]["median_curve"].value < scores[c][m].value
                for c in CAPS for m in ("power", "A2", "zero")), "C50 median ordering changed")

    comp = read("v69-quant-confirm/compare.json")
    require(comp.data["complete"] and len(comp.data["rows"]) == 63, "C46 incomplete")
    for new_state in (False, True):
        subsets = ("new_state_boundary", "new_state_interior") if new_state else ("development_state_boundary",)
        rr = [(i, r) for i, r in enumerate(comp.data["rows"]) if r["test_set"] in subsets]
        expected_states = {"pythia-1.4b@step112000"} if new_state else {
            "pythia-410m@step143000", "pythia-1.4b@step16000"}
        groups = (32, 128, 512) if new_state else (32, 512)
        require({r["state"] for _, r in rr} == expected_states, "C46 state set changed")
        for cap in CAPS:
            actual = [(r["state"], r["config"]) for _, r in rr if r["capability"] == cap]
            expected = [(s, f"b{b}_g{g}") for s in expected_states for b in (3, 4, 5) for g in groups]
            require(sorted(actual) == sorted(expected), "C46 grid incomplete or duplicated")
        methods = ("same_input_interpolation", "low_order_2d", "median")
        scores = row_scores(comp, rr, methods)
        for subset in subsets:
            subset_scores = row_scores(comp, [(i, r) for i, r in rr if r["test_set"] == subset], methods)
            for cap in CAPS:
                for method in methods:
                    close(subset_scores[cap][method].value,
                          comp.data["test_sets"][subset]["scores"][method][cap]["mae"],
                          f"C46/{subset}/{cap}/{method}")
        candidate = "median" if new_state else "same_input_interpolation"
        baseline = same("same_input_interpolation") if new_state else {
            c: min(("low_order_2d", "median"), key=lambda m: scores[c][m].value) for c in CAPS}
        rows.append(Row(
            "C46-new" if new_state else "C46-dev", "Grouped quant.",
            "Per-config. median" if new_state else "Interpolation; QA: median",
            (r"1 new state: 1.4B@112k; $b=3,4,5$; $g=32,128,512$" if new_state else
             r"2 dev. states: 410M@143k, 1.4B@16k; $b=3,4,5$; $g=32,512$"),
            scores, same(candidate), baseline, FROZEN + "; " + AFTER_TEST,
            "New state only" if new_state else "QA: median wins",
            comp.ref("rows") + " filtered by test_set in " + repr(subsets)
            + "; states = unique(state); configurations = unique(config); bits/group sizes parsed from config",
            ("Pool all nine new-state cells equally (six boundary plus three interior), not an equal "
             "average of the two subset MAEs. Median is compared with the named same-input interpolation "
             "reference, matching the confirmation figure; this is NOT an all-method test oracle." if new_state else
             "Twelve boundary cells, equally weighted. Score frozen interpolation on all capabilities; "
             "take the lower confirmation MAE of surface and median per capability as baseline. The "
             "delivered rule uses interpolation for math/code and median for QA; the negative QA gain "
             "here deliberately retains interpolation's failure.")
            + " All predictions were frozen, but the delivered rule was stated after test. The actual "
            "development-frozen selector was surface/median/zero for math/code/QA; this row must not "
            "be read as its prospective success. The median uses the same frozen interpolation/boundary "
            "rule on per-configuration development medians. Baseline sets are explicitly restricted "
            "to the named comparisons, not all six frozen quantization methods."))

    comp = read("v70-distill-confirm/compare.json")
    require(comp.data["complete"] and len(comp.data["rows"]) == 108, "C48 incomplete")
    require(len(comp.data["groups"]) == 6, "C48 group count changed")
    for student, label in (("gemma3-270m", "270M"), ("gemma3-1b", "1B")):
        scores, candidates, baselines = {}, {}, {}
        for cap in CAPS:
            matches = [(i, g) for i, g in enumerate(comp.data["groups"])
                       if g["student"] == student and g["capability"] == cap]
            require(len(matches) == 1, "C48 group missing/duplicated")
            i, g = matches[0]
            candidate, baseline = g["selected"], g["strongest_baseline"]
            candidates[cap], baselines[cap] = candidate, baseline
            require(candidate == ("joint" if cap == "qa" else "E"), "C48 selected form changed")
            scores[cap] = {candidate: comp.number("groups", i, "candidate_mae"),
                           baseline: comp.number("groups", i, "baseline_mae")}
            rr = [r for r in comp.data["rows"] if r["student"] == student and r["capability"] == cap]
            require(len(rr) == g["n_checkpoints"] == 18 and g["n_pools"] == 6, "C48 count changed")
            require(sorted((r["pool"], r["T_planned"]) for r in rr) ==
                    [(f"U200_s{s}", t) for s in range(31, 37) for t in (50000, 100000, 200000)],
                    "C48 pools/budgets changed")
            for method in (candidate, baseline):
                close(mean(r["absolute_errors"][method] for r in rr), scores[cap][method].value,
                      f"C48/{student}/{cap}/{method}")
            close(scores[cap][baseline].value - scores[cap][candidate].value,
                  g["paired_difference"]["estimate"], f"C48/{student}/{cap}/gain")
        rows.append(Row(
            "C48-" + label, "Distillation", r"$E$ / $E$ / joint",
            rf"Gemma-3-{label}; 6 pools, $U=200$; $T=50,100,200$k",
            scores, candidates, baselines, FROZEN,
            "Code, QA gain" if label == "270M" else "Code gain only",
            comp.ref("groups") + " filtered by student=" + student
            + " (student, n_pools, n_checkpoints); " + comp.ref("rows")
            + " filtered by student=" + student + " (pool encodes U; T_planned / 1000 gives k tokens)",
            "Use selected and strongest_baseline from each stored group, fixed by development before "
            "measurement; never reselect from all_method_mae using confirmation results. Equal weight "
            "over six pools and three planned budgets per pool. E is the selected zero-anchored "
            "a*log(1+E) form; E-only is the distinct intercept-bearing c+b*log(1+E) baseline. "
            "QA scope is primary 2Wiki conditional loss. Gains are point estimates; the scope wording "
            "agrees with the stored paired pool-cluster intervals. C48 used the documented dense-drift "
            "comparison wrapper; V84 consumes its stored comparison without recomputing measurements."))

    comp = read("v50-p2v2/compare_test.json")
    for student, label in (("gemma3-270m", "270M"), ("gemma3-1b", "1B"), ("gemma3-4b", "4B")):
        rr = [(i, r) for i, r in enumerate(comp.data["rows"])
              if r["student"] == student and r["role"] == "test_pool"]
        scores = row_scores(comp, rr, ("joint+src", "constant"), cap_key="cap", error_key="abs")
        for cap in CAPS:
            cr = [r for _, r in rr if r["cap"] == cap]
            require(len(cr) == 12 and {r["pool"] for r in cr} == {f"U375_s{s}" for s in (21, 22, 23)},
                    "C38 test pool count changed")
            for pool in {r["pool"] for r in cr}:
                require(sum(r["pool"] == pool for r in cr) == 4, "C38 missing budget")
            for method in scores[cap]:
                keys = [f"{student}|test_pool|{cap}|T{t}" for t in (35000, 70000, 140000, 280000)]
                stored = average([comp.number("summary", k, "mae", method) for k in keys])
                close(scores[cap][method].value, stored.value, f"C38/{student}/{cap}/{method}")
                scores[cap][method] = stored
        rows.append(Row(
            "C38-" + label, "Distill. context", "Joint+src",
            rf"Gemma-3-{label}; 3 pools, $U=375$; $T=35,70,140,280$k",
            scores, same("joint+src"), same("constant"), FROZEN + "; " + AFTER_TEST,
            "Size transfer fails" if label == "4B" else "Seen student",
            comp.ref("rows") + " filtered by student=" + student + ", role=test_pool "
            "(student; unique(pool) gives U375 and 3 pools); " + comp.ref("summary")
            + " keys " + student + "|test_pool|CAP|T{35000,70000,140000,280000} give planned budgets",
            "Earlier context only: joint+src versus the frozen per-capability constant, not a test-selected "
            "best form. All forms were frozen before measurement; the headline selection rule (lowest "
            "in-sample development MAE at freeze) was stated after test. Filter role=test_pool to exclude "
            "the held-out student's development-pool runs. Average four budget summaries with three pools "
            "each, validated against the twelve raw errors. The compare file evaluates frozen functions "
            "at actual achieved Tc and E; displayed T values are the planned budgets. "
            + ("4B was held out of development; preserve all three negative gains." if label == "4B" else
               "The student was seen in development, but these pools were unseen.")))
    return rows, inputs


def stack(values):
    return r"\newline ".join(values)


def render_tex(rows, inputs):
    # Seven p-column widths sum to textwidth minus the twelve internal tabcolseps.
    widths = (0.145, 0.210, 0.100, 0.185, 0.09, 0.155, 0.115)
    close(sum(widths), 1, "column width sum")
    columns = "@{}" + "".join(
        r">{\raggedright\arraybackslash}p{\dimexpr " + f"{w:.3f}" +
        r"\textwidth-" + f"{12*w:.3f}" + r"\tabcolsep\relax}" for w in widths) + "@{}"
    lines = ["% Generated by analysis/v84_main_table.py; do not edit by hand."]
    lines += [f"% INPUT {c.path} sha256={c.sha256}" for c in inputs]
    lines += [r"\begin{table}[t]", r"\centering\scriptsize",
              r"\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{1.06}",
              r"\begin{tabular}{" + columns + "}", r"\toprule",
              r"Arm / Final predictor & Test object (states, configurations) & Candidate MAE & Strongest same-information baseline (name, MAE) & Gain & Provenance & Scope note \\",
              r"\midrule"]
    for i, row in enumerate(rows):
        if i in (2, 4, 6):
            lines.append(r"\midrule")
        lines += ["% SOURCE " + trace for trace in row.traces()]
        candidate = [row.scores[c][row.candidates[c]].value for c in CAPS]
        baseline = [row.scores[c][row.baselines[c]].value for c in CAPS]
        cells = [stack((row.arm, row.predictor)), row.test,
                 stack(f"${v:.3f}$" for v in candidate),
                 stack(f"{LABELS[row.baselines[c]]} ${v:.3f}$" for c, v in zip(CAPS, baseline)),
                 stack(f"${b-a:+.3f}$" for a, b in zip(candidate, baseline)),
                 row.provenance, row.scope]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\scriptsize Latest frozen confirmations and earlier distillation context. MAE and signed gain "
              r"(baseline minus candidate) are in nats per token; each numeric stack is math / code / QA. "
              r"Strongest is within the named comparison set: pruning panel, better of A2 and median; "
              r"pruning repeat, median; quantization development states, better of surface and median; "
              r"quantization new state, interpolation. Better-of rankings use test MAE. Distillation "
              r"confirmation baselines were fixed from development; context uses the frozen constant. "
              r"The repeat scores power while delivering median; development-state quantization scores "
              r"interpolation throughout, with median delivered for QA. All predictions were frozen "
              r"before measurement; delivery and headline-rule timing are stated separately. "
              r"The two 2.8B revision labels identify one learned state (C52).}",
              r"\label{tab:main_final}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def render_sources(rows, inputs):
    lines = ["# Main Table 1 sources (V84)", "",
             "Regenerate with `python -B analysis/v84_main_table.py`; verify without writes with "
             "`python -B analysis/v84_main_table.py --check`. Python standard library only; no fitting or GPU. "
             "The only generated files are `paper/paper/tables/main_final.tex` and this document.", "",
             "All numerical metrics come from the comparison JSON files below. `#/...` is an exact JSON "
             "Pointer (RFC 6901); `mean(...)` denotes equal weighting. Every score and gain has a `% SOURCE` "
             "comment in the TeX, including unused members of named comparison sets. Gains are computed at "
             "full precision and then rounded independently to three decimals, so subtracting the displayed "
             "rounded MAEs can differ by 0.001. Negative gains are retained. Counts, state/revision names, "
             "pool sizes, and configuration values are traced in each row's test-object comment.", "",
             "Process provenance comes from `paper/docs/RESULTS_LEDGER.md`, entries C35, C38, C44/C46, "
             "C47/C48, C50/C52 (not inferred from a favorable score or filesystem timestamps). "
             "Freeze commits recorded there: C35 `40a51b7`, C38 `33c706c`, C46 `6199353`, "
             "C48 `118d1f8`, C50 `6089efd`. V84 does not refit or reclassify predictions. "
             "The one-learned-state correction for C50 is from C52; compare.json retains the original "
             "two revision labels. Test-time baseline rankings and post-test delivery choices are "
             "retrospective even when their constituent predictions were frozen before measurement.", "",
             "The named baseline sets match Figure 2 and the requested C38 constant comparison. "
             "In particular, the quantization new-state row is median versus interpolation, not a claim "
             "that interpolation beats every other frozen method. The caption limits 'strongest' to "
             "these named sets. There is no per-target oracle, cross-capability averaging, or new "
             "significance calculation. QA refers to the primary 2Wiki conditional-loss probe.", "",
             "## Input fingerprints", "", "| Comparison JSON | SHA-256 |", "|---|---|"]
    lines += [f"| `{c.path}` | `{c.sha256}` |" for c in inputs]
    lines += ["", "## Displayed values", "",
              "| Row | Capability | Candidate | MAE | Baseline | MAE | Gain |",
              "|---|---|---|---:|---|---:|---:|"]
    for row in rows:
        for cap in CAPS:
            a, b = row.candidates[cap], row.baselines[cap]
            av, bv = row.scores[cap][a].value, row.scores[cap][b].value
            lines.append(f"| {row.key} | {cap} | {a} | {av:.3f} | {b} | {bv:.3f} | {bv-av:+.3f} |")
    for row in rows:
        lines += ["", "## " + row.key, "", row.rationale, "",
                  "Provenance: " + row.provenance + ".", "", "```text", *row.traces(), "```"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate existing outputs without writes")
    args = parser.parse_args()
    rows, inputs = build_rows()
    outputs = {"main_final.tex": render_tex(rows, inputs),
               "main_final_sources.md": render_sources(rows, inputs)}
    for name, content in outputs.items():
        path = TABLES / name
        require(not path.is_symlink(), f"Refusing symlink output: {path}")
        if args.check:
            require(path.is_file() and path.read_text() == content, f"Missing/stale output: {path}")
        else:
            path.write_text(content)
        print(f"{'CHECKED' if args.check else 'WROTE'} {path.relative_to(ROOT)}")
    print(f"Validated {len(rows)} rows / {len(rows)*len(CAPS)} capability comparisons from {len(inputs)} JSON files.")


if __name__ == "__main__":
    main()
