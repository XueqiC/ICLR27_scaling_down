#!/usr/bin/env python3
"""Generate the frozen-prediction table from frozen results; no fitting or oracle ranking.

Run: python -B analysis/final_prediction_table.py --stdout
Every cell has an executable JSON-pointer recipe in main_prediction_v2_sources.md.
"""
from __future__ import annotations


import sys
sys.dont_write_bytecode = True

from dataclasses import dataclass, field
import json
import re
from statistics import mean
from math import isclose

if __package__:
    from .paper_artifacts import ROOT, CAPS, Artifacts, frozen_run, output_path, legacy_rows
else:
    from paper_artifacts import ROOT, CAPS, Artifacts, frozen_run, output_path, legacy_rows

P53 = "results/v53-prune-dev/register.json"
Q69 = "results/v69-quant-confirm/develop.json"
Q55 = "results/v55-quant-group/register.json"
D70 = "results/v70-distill-confirm/compare.json"
R74 = "results/v74-quant-threeway/quant_threeway.json"
S86 = "results/v86-main-table/summary.json"
E11 = "results/a11-efficiency-confirmation/summary.json"
E11_STATE = "results/a11-efficiency-confirmation/per_state.json"

# Presentation vocabulary only. Coefficient counts and scores still come from
# JSON fields; each mapped phrase carries the source values in its recipe.
RELATIONS = {
    "power": "Five-parameter power form",
    "low_order_2d": "Source regression across bit widths (twenty coefficients)",
    "median": "Development median",
    "median_curve": "Source-free median density curve",
    "same_input_interpolation": "Interpolation",
    "zero": "no change",
}
BASELINES = {
    "A2": "per-density regression", "median_curve": "development median", "median": "development median",
    "bilinear": "bilinear regression", "zero": "no change",
    "T-only": "budget regression", "E-only": "reuse regression",
    "surface:L0": "loss regression", "surface:logN": "size regression",
}
RANGES = {
    "prune_new": "Three unseen checkpoints pruned to densities 0.575, 0.675 and 0.85",
    "quant_bit": "Pythia 160 million to 1.4 billion at unseen bit width 4, group sizes 64 and 256",
    "quant_group": "Pythia 410 million and 1.4 billion at unseen group sizes 32 and 512, bit widths 3 to 5",
    "quant_new": "An unseen 1.4 billion stage at bit widths 3 to 5, group sizes 32 to 512",
    "pool": "Gemma 270 million and 1 billion distilled on six new pools at 50 to 200 thousand tokens",
    "efficiency": "Four unseen Pythia states pruned at six densities",
}
HEADERS = ("Prediction task", "Frozen candidate", "Candidate error (nats)",
           "Delivered relation", "Delivered error (nats)", "Development baseline",
           "Development measurements")
COLUMN_WIDTHS = (".20", ".15", ".10", ".16", ".10", ".16", ".13")
TABLE_FONT = r"\footnotesize\fontsize{8}{9.5}\selectfont"
CAPTION_FONT = r"\footnotesize\fontsize{8.5}{10}\selectfont"
CAPTION = (
    "Errors are mean absolute errors in nats per token, in math, code and question answering order. "
    "The first predictor of a row was frozen before its measurement; a delivered rule marked as later "
    "was chosen after seeing the result. Interpolation is piecewise on the measured grid, and a median "
    "curve uses no source inputs. The last column counts the development configuration measurements "
    "per capability behind the first predictor. Distillation entries give the {students} students in "
    "that order."
)

NUMBER_PATTERN = r"[-+]?\d+(?:\.\d+)?|\b(?:one|three|four|five|six|seventeen|twenty|half)\b"


class MissingValue(ValueError):
    """An incomplete task is omitted instead of printing an absent value."""


def pointer(path, *keys):
    return path + "#/" + "/".join(str(k).replace("~", "~0").replace("/", "~1") for k in keys)


def resolve(audit, ref):
    path, ptr = ref.split("#", 1)
    value = audit.data[path] if path in audit.data else audit.read(path)
    for key in ptr.removeprefix("/").split("/") if ptr else []:
        key = key.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def value(*sources, op="identity", fmt=None, **presentation):
    return {"sources": list(sources), "op": op, "format": fmt, **presentation}


def mapped(audit, label, *sources):
    return value(*sources, op="label", label=label,
                 expected=[resolve(audit, ref) for ref in sources])


def relation(audit, name, *sources):
    return mapped(audit, RELATIONS[name], *sources)


def baseline(audit, name, *sources):
    return mapped(audit, BASELINES[name], *sources)


def tested_range(audit, name, *sources):
    return mapped(audit, RANGES[name], *sources)


def evaluate(audit, part):
    vals = [resolve(audit, ref) for ref in part["sources"]]
    if any(v is None for v in vals):
        raise MissingValue("Frozen value is absent")
    op = part["op"]
    if op == "identity":
        result = vals[0]
    elif op == "mean":
        result = mean(vals)
    elif op == "weighted_mean":
        result = sum(v * n for v, n in zip(vals[::2], vals[1::2])) / sum(vals[1::2])
    elif op == "length":
        result = len(vals[0])
    elif op == "per_capability":
        count, capabilities = vals
        if set(capabilities) != set(CAPS) or count % len(capabilities):
            raise ValueError("Development rows must divide evenly across capabilities")
        result = count // len(capabilities)
    elif op == "keys":
        result = ", ".join(vals[0])
    elif op == "unique":
        result = ", ".join(str(v) for v in sorted(set(vals)))
    elif op == "join":
        result = ", ".join(str(v) for v in vals[0])
    elif op == "shared":
        if not vals or any(v != vals[0] for v in vals[1:]):
            raise ValueError("Grouped presentation requires identical frozen values")
        result = vals[0]
    elif op == "label":
        if vals != part["expected"]:
            raise ValueError("Presentation label no longer matches its frozen sources")
        result = part["label"]
    else:
        raise ValueError(f"Unknown source operation: {op}")
    if result is None:
        raise MissingValue("Frozen value is absent")
    return format(result, part["format"]) if part["format"] else str(result)


def tex_escape(text):
    escaped = "".join({"\\": r"\textbackslash{}", "_": r"\_", "&": r"\&", "%": r"\%",
                    "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}.get(c, c) for c in text)
    return escaped


def render_text(text):
    return r"\newline ".join("".join(
        token if token.startswith("$") else tex_escape(token)
        for token in re.split(r"(\$[^$]+\$)", line)) for line in text.split("\n"))


def baseline_names(names):
    """Keep all frozen selections; only replace their abbreviated display names."""
    names = list(dict.fromkeys(BASELINES[name] for name in names))
    return ", ".join(names)


@dataclass
class Cell:
    parts: list
    context: list = field(default_factory=list)
    note: str = ""
    compact_scores: str = ""

    def render(self, audit):
        plain = self.plain(audit)
        if not self.compact_scores:
            return render_text(plain)
        # Measure in the actual document font and column, without squeezing type.
        # Score cells retain three capability-ordered lines when the triplet is too wide.
        prefix, scores = plain.split("\n", 1) if self.compact_scores == "comparison" else ("", plain)
        return ((render_text(prefix) + r"\newline ") if prefix else "") + r"\TableOneErrors" + "".join(
            "{" + render_text(line) + "}" for line in scores.split("\n"))

    def plain(self, audit):
        return "".join(p if isinstance(p, str) else evaluate(audit, p) for p in self.parts)

    def record(self, audit):
        return {"parts": self.parts, "context": self.context, "note": self.note,
                "compact_scores": self.compact_scores,
                "rendered": self.render(audit)}


def cell(*parts, refs=(), note=""):
    return Cell(list(parts), list(refs), note)


def number(path, *keys):
    return value(pointer(path, *keys), fmt=".2f")


def scored_number(n):
    # Reuse V86's source-carrying Number, including its equal-weight means.
    if n.source.startswith("mean("):
        refs = n.source[5:-1].split("; ")
        return value(*refs, op="mean", fmt=".2f")
    return value(n.source, fmt=".2f")


def cap_lines(fn, separator="; ", label_separator=" "):
    parts = []
    for cap in CAPS:
        if parts:
            parts.append(separator)
        parts += [("QA" if cap == "qa" else cap.capitalize()) + label_separator, *fn(cap)]
    return parts


def score_parts(fn):
    parts = []
    for cap in CAPS:
        if parts:
            parts.append("\n")
        parts.extend(fn(cap))
    return parts


def baseline_parts(audit, fn):
    """Name the comparison once; its header supplies the capability order."""
    entries = [fn(cap) for cap in CAPS]
    names = [evaluate(audit, entry[0]) for entry in entries]
    sources = [ref for entry in entries for ref in entry[0]["sources"]]
    if len(set(names)) == 1:
        label = names[0].capitalize()
    elif names == ["per-density regression", "per-density regression", "development median"]:
        label = "Per-density regression; question answering: median"
    elif names == ["bilinear regression", "no change", "development median"]:
        label = "Bilinear regression, no change and the median"
    elif names == ["budget regression, loss regression", "reuse regression", "reuse regression, size regression"]:
        label = "Regressions on budget and loss, reuse, and reuse and size"
    else:
        grouped = {}
        for cap, name in zip(CAPS, names):
            grouped.setdefault(name, []).append("QA" if cap == "qa" else cap.capitalize())
        label = "; ".join(", ".join(caps) + ": " + name for name, caps in grouped.items())
    scores = dict(zip(CAPS, (entry[1:] for entry in entries)))
    parts = [mapped(audit, label, *sources), "\n"]
    for cap in CAPS:
        if cap != CAPS[0]:
            parts.append("\n")
        parts.extend(scores[cap])
    return parts


def composite_label(names):
    """Translate the two frozen mixed predictors into ordinary words."""
    if list(names.values()) == ["E", "E", "joint"]:
        return "Reuse forms, with a budget and pool form for question answering"
    if list(names.values()) == ["low_order_2d", "median", "zero"]:
        return "Source regression, the median and no change"
    if list(names.values()) == ["same_input_interpolation", "same_input_interpolation", "median"]:
        return "Interpolate math and code; use the median for question answering"
    raise ValueError(f"Unmapped mixed predictor: {names}")


def delivered_cells(audit, names, refs, scores, *, note=""):
    """Keep rule identity/timing separate; incomplete tasks are omitted below."""
    if len(set(names.values())) == 1:
        label = RELATIONS[names[CAPS[0]]]
    else:
        label = composite_label(names)
    return [
        cell(mapped(audit, label, *refs), note=note),
        cell(*score_parts(scores), refs=refs, note=note),
    ]


def paired_interval(audit, path, *keys):
    """Read stored endpoints only; missing intervals never trigger estimation."""
    parent = resolve(audit, pointer(path, *keys))
    interval = parent.get("paired_difference", {}).get("ci95")
    if interval is None:
        return []
    if len(interval) != 2 or any(v is None for v in interval):
        raise ValueError("Incomplete stored paired interval")
    # Four decimals preserve the small, nonzero code intervals and narrow math
    # intervals; the MAEs retain the table's two-decimal convention.
    return [" [", value(pointer(path, *keys, "paired_difference", "ci95", 0), fmt=".4f"),
            ", ", value(pointer(path, *keys, "paired_difference", "ci95", 1), fmt=".4f"), "]"]


def efficiency_row(audit):
    """Authenticate the registered panel and reconcile both recorded score levels."""
    summary = audit.read(E11)
    states = audit.read(E11_STATE)
    plan_path = "results/a11-efficiency-confirmation/plan.json"
    predictions_path = "results/a11-efficiency-confirmation/predictions.json"
    plan = audit.read(plan_path)
    frozen = audit.read(predictions_path)
    for path in (E11, plan_path, predictions_path):
        if not audit.matches_digest(states["input_sha256"][path], audit.inputs[path]):
            raise ValueError("Efficiency record digest mismatch: " + path)
    if (summary["status"] != "complete" or states["status"] != "complete"
            or summary["predictions_sha256"] != states["predictions_sha256"]
            or not audit.matches_digest(summary["predictions_sha256"], audit.inputs[predictions_path])
            or summary["measurement_sha256"] != states["measurement_sha256"]):
        raise ValueError("Efficiency confirmation is incomplete or has mismatched provenance")
    targets = states["state_order"]
    excluded = set(plan["exclusion_audit"]["a9_states"] + plan["exclusion_audit"]["selection_rule_states"])
    if (targets != [t["state"] for t in plan["targets"]] or len(set(targets)) != 4
            or excluded.intersection(targets) or plan["exclusion_audit"]["target_overlap"]):
        raise ValueError("Efficiency states must be outside every earlier fit and test")
    budget = states["development_measurements_per_capability"]
    candidate, comparator = summary["decision"]["candidate"], summary["decision"]["comparator"]
    if (candidate != "power_18" or comparator != "A2_36"
            or budget[candidate] * 2 != budget[comparator]
            or budget["median_curve_36"] != budget[comparator]):
        raise ValueError("Efficiency confirmation must retain the registered half-budget comparison")
    # The delivered QA fallback is the original full-development median, not
    # the half-budget sensitivity curve or whichever test error happens to win.
    methods = (candidate, comparator, "median_curve_36")
    if any(budget[m] != frozen["coefficients"][m]["budget_per_capability"] for m in methods):
        raise ValueError("Efficiency development budget disagrees with frozen coefficients")
    cells = {(r["state"], r["density"], r["capability"]): r for r in summary["cells"]}
    expected = {(s, d, c) for s in targets for d in plan["densities"] for c in CAPS}
    if len(plan["densities"]) != 6 or set(cells) != expected or len(cells) != len(summary["cells"]):
        raise ValueError("Efficiency confirmation must contain exactly the registered cells")
    records = {(r["state"], r["capability"]): r for r in states["records"]}
    if set(records) != {(s, c) for s in targets for c in CAPS} or len(records) != len(states["records"]):
        raise ValueError("Efficiency per-state records have missing or duplicate groups")
    for (state, cap), record in records.items():
        densities = record["densities"]
        if densities != plan["densities"] or record["n_cells"] != len(densities):
            raise ValueError("Efficiency per-state density membership mismatch")
        for method in methods:
            errors = [cells[state, d, cap]["absolute_errors"][method] for d in densities]
            if (errors != record["absolute_errors"][method]
                    or not isclose(mean(errors), record["mae"][method], abs_tol=1e-12)):
                raise ValueError("Efficiency per-state errors disagree with summary cells")
            for d, error in zip(densities, errors):
                observed = cells[state, d, cap]
                if not isclose(error, abs(observed["predicted_delta_L"][method] - observed["observed_delta_L"]), abs_tol=1e-12):
                    raise ValueError("Efficiency absolute error disagrees with recorded prediction")
    for cap in CAPS:
        if summary["by_capability"][cap] != states["by_capability"][cap]:
            raise ValueError("Efficiency aggregate records disagree")
        if summary["by_capability"][cap]["n_cells"] != len(targets) * len(plan["densities"]):
            raise ValueError("Efficiency aggregate cell count mismatch")
        for method in methods:
            if not isclose(mean(records[s, cap]["mae"][method] for s in targets),
                           summary["by_capability"][cap]["mae"][method], abs_tol=1e-12):
                raise ValueError("Efficiency aggregate error disagrees with per-state means")
    def errors(method_for_cap):
        return score_parts(lambda c: [number(E11, "by_capability", c, "mae", method_for_cap(c))])
    score_refs = [pointer(E11_STATE, "records", i, "mae") for i in range(len(states["records"]))]
    score_note = "Stored unweighted per-capability MAEs, checked against all four per-state means and the identical six-density cells; no refit or test ranking."
    return [
        cell(tested_range(audit, "efficiency", pointer(E11_STATE, "state_order"),
                         pointer(E11_STATE, "records", 0, "densities")),
             refs=[pointer(plan_path, "created_utc"), pointer(plan_path, "exclusion_audit"),
                   pointer(E11, "predictions_sha256")]),
        cell(mapped(audit, "Compact power form using half the measurements",
                    pointer(E11, "decision", "candidate"),
                    pointer(E11_STATE, "development_measurements_per_capability", candidate),
                    pointer(E11_STATE, "development_measurements_per_capability", comparator))),
        cell(*errors(lambda c: candidate), refs=score_refs, note=score_note),
        cell(mapped(audit, "The same power form, with a median density curve for question answering",
                    pointer(E11, "decision", "primary_capabilities"), pointer(E11, "decision", "candidate"),
                    pointer(E11, "decision", "descriptive_capabilities"),
                    pointer(E11_STATE, "development_measurements_per_capability", "median_curve_36")),
             note="Section 6 measurement-efficiency scope and Appendix E retain the source-free median for QA. Use the full-development median_curve_36 (36 measurements), not median_curve_18; the last column counts the first predictor only. This is the specified delivery, not a claim that the median wins this test."),
        cell(*errors(lambda c: "median_curve_36" if c == "qa" else candidate), refs=score_refs, note=score_note),
        cell(mapped(audit, "Per-density regression at full budget", pointer(E11, "decision", "comparator")),
             " (", value(pointer(E11_STATE, "development_measurements_per_capability", comparator)), ")\n",
             *errors(lambda c: comparator), refs=[pointer(E11, "decision"), *score_refs], note=score_note),
        cell(value(pointer(E11_STATE, "development_measurements_per_capability", candidate)),
             refs=[pointer(E11_STATE, "development_measurements_per_capability", comparator)],
             note="18 recorded development configuration measurements per capability for power_18, half the comparator's 36; excludes dense anchors and test measurements."),
    ]


def build(audit):
    rows = legacy_rows(audit)
    prune = audit.data[P53]
    quant = audit.read(Q69)
    q55 = audit.read(Q55)
    comp55 = audit.read("results/v55-quant-group/compare.json")
    audit.read("results/a5-corner-second-difference/summary.json")
    audit.read("results/a7-closeout-audit/summary.json")
    v46 = audit.read("results/v46-p1-newsource/compare.json")
    f46 = audit.read("results/v46-p1-newsource/predictions_frozen.json")
    for r in v46["pruning"]:
        for method, pred in f46["pruning"][f"{r['cap']}|{r['d']}"].items():
            # V46 comparison fields are absolute errors, not predictions.
            if abs(abs(pred - r["actual"]) - r[method]["abs"]) > 1e-10:
                raise ValueError("V46 frozen prediction/error mismatch")
    audit.read("results/v47-p2-register/register.json")
    missing47 = "results/v47-p2-register/freeze.json"
    if audit.path(missing47).exists():
        audit.read(missing47)
    else:
        audit.rule("V47 original freeze.json is absent: that original test is omitted here. "
                   "The registered v5_confirm section is paired with the available V70 freeze.")
    v78 = audit.read("results/v78-rule-confirm/freeze.json")
    audit.read("results/v78-rule-confirm/compare.json")
    for state in v78["states"]:
        for cap in CAPS:
            path = f"results/v93-confirm-inputs/{state['tag'].replace('@', '--')}/{cap}/descriptor_bv.json"
            descriptor = audit.read(path)
            audit.rule(f"{path}#/aggregates: dense descriptors only; no frozen response prediction or error interval. "
                       "Not scored as a predictor.")
    audit.rule("V78 is a prospective decision test with frozen configuration predictions; its regret "
               "and test oracle are not substituted for relation MAE or a development baseline.")
    audit.rule("V46 is an earlier new-state test; its freeze identity is checked. Its strongest development "
               "baseline is not recorded in the supplied V46 pair, so it is not mixed into the later V53 score.")
    audit.rule("V70 paired_difference.ci95 is a baseline-minus-candidate interval, never an MAE interval.")
    delivery = audit.read(S86)
    audit.read(R74)
    delivery_indices = {r["row"]: i for i, r in enumerate(delivery["main_rows"])}
    def delivery_refs(key):
        i = delivery_indices[key]
        return [pointer(S86, "main_rows", i, "delivered_timing"),
                *[pointer(S86, "main_rows", i, "capabilities", c, "delivered") for c in CAPS]]

    result = []
    pbase = {}
    for cap in CAPS:
        eligible = [(i, r) for i, r in enumerate(prune["loso_table"])
                    if r["subset"] == "all" and r["candidate"] != "power"]
        pbase[cap] = min(eligible, key=lambda ir: ir[1][f"{cap}_mae"])
    power = cell(relation(audit, "power", pointer(P53, "candidate_definitions", "power"),
                          pointer(P53, "n_params_per_capability", "power"), pointer(P53, "n_dev_states")),
                 refs=[pointer(P53, "feature_names")])
    original_panel = "results/v40-prune-strength/register.json"
    audit.read(original_panel)
    audit.omit("Development-only density, budget and data-reuse tasks are outside this frozen-prediction table.")
    audit.rule("V72 repeats use the same weights; no displayed score uses those repeats.")
    row = rows["C35"]
    ps = [p for p in audit.data if p.startswith("results/v53-prune-dev/compare_")]
    if len(ps) != 3 or any(audit.data[p]["tag"] in {s["tag"] for s in prune["dev_states"]} for p in ps):
        raise ValueError("Pruning test must contain three checkpoints outside every fit")
    result.append([
        cell(tested_range(audit, "prune_new", *[pointer(p, k) for p in ps for k in ("tag", "densities")]),
             refs=[row.selection_source, pointer(P53, "dev_states"), pointer(ps[0], "densities")],
             note="All three test checkpoints are absent from every development fit. The 6.9B size occurs at other stages in the expanded register."),
        power,
        cell(*score_parts(lambda c: [scored_number(row.scores[c]["power"])])),
        *delivered_cells(audit, dict.fromkeys(CAPS, "median_curve"), delivery_refs("C35"),
                         lambda c: [scored_number(row.scores[c]["median_curve"])],
                         note="New-state source-free median curve, fixed after test; scores use the same three checkpoints and densities as the frozen candidate."),
        cell(*baseline_parts(audit, lambda c: [
                 baseline(audit, pbase[c][1]["candidate"], pointer(P53, "loso_table", pbase[c][0], "candidate")),
                 scored_number(row.scores[c][pbase[c][1]["candidate"]])]),
             refs=[pointer(P53, "loso_table")], note="Minimum development LOSO MAE excluding power, per capability: " +
             " / ".join(BASELINES[pbase[c][1]["candidate"]] for c in CAPS) + "; no test ranking.")])

    # V55 predates V69 and is the actual unseen-bit test. V69 bits were all seen.
    bp = "results/v55-quant-group/compare.json"
    entries = comp55["test_sets"]["bit_test"]["mae_table"]
    candidate_index = next(i for i,r in enumerate(entries) if r["candidate"] == "low_order_2d")
    base55 = {c: min(((i,r) for i,r in enumerate(q55["loso_table"]) if r["candidate"] in q55["baselines"]),
                    key=lambda ir: ir[1]["mae"][c]) for c in CAPS}
    def bit_baseline(c):
        i, r = base55[c]
        j = next(j for j,e in enumerate(entries) if e["candidate"] == r["candidate"])
        return [baseline(audit, r["candidate"], pointer(Q55,"loso_table",i,"candidate")),
                number(bp,"test_sets","bit_test","mae_table",j,"mae",c)]
    bit_scores = score_parts(lambda c: [number(bp,"test_sets","bit_test","mae_table",candidate_index,"mae",c)])
    bit_delivery_note = (
        "The low-order surface is a tested candidate, not a delivered relation (Appendix E.1: design rank 16 of 20). "
        "The later delivered rule has no matching stored score for this earlier bit test, "
        "whose configurations entered its development grid. V55 alternatives use different "
        "models and boundary rules and are not substituted for that later rule."
    )
    bit_cells = {(s, cfg, c) for s in q55["test_sets"]["bit_test"]["states"]
                 for cfg in q55["test_sets"]["bit_test"]["configs"] for c in CAPS}
    later_cells = {(r["state"], r["config"], r["capability"])
                   for r in audit.data["results/v69-quant-confirm/compare.json"]["rows"]}
    if bit_cells & later_cells or not set(q55["test_sets"]["bit_test"]["configs"]) <= set(quant["dev_configs"]):
        raise ValueError("Earlier bit-test membership changed; re-audit delivered-score availability")
    bit_delivery_refs = [pointer(R74, "recommendation_rule"), pointer(Q69, "dev_configs"),
                         pointer(Q69, "boundary_rule"), pointer(Q55, "dev_configs"),
                         pointer(Q55, "candidate_definitions", "same_input_interpolation"),
                         pointer(bp, "test_sets", "bit_test", "mae_table"),
                         pointer("results/v69-quant-confirm/compare.json", "test_sets")]
    result.append([
        cell(tested_range(audit, "quant_bit", pointer(Q55,"test_sets","bit_test","states"),
                         pointer(Q55,"test_sets","bit_test","configs")), refs=[pointer(Q55,"precommitted_rule")]),
        cell(relation(audit, "low_order_2d", pointer(Q55,"candidate_definitions","low_order_2d"),
                      pointer(Q55,"n_params_per_capability","low_order_2d")), refs=[pointer(Q55,"feature_names")]),
        cell(*bit_scores),
        cell(mapped(audit, "Interpolate math and code on seen states; use the median otherwise",
                    pointer(R74, "recommendation_rule")),
             refs=bit_delivery_refs, note=bit_delivery_note),
        cell("No stored error on these test cells", refs=bit_delivery_refs, note=bit_delivery_note),
        cell(*baseline_parts(audit, bit_baseline), refs=[pointer(Q55,"loso_table")], note="Development minimum over registered baselines mean/median/zero: " +
             " / ".join(BASELINES[base55[c][1]["candidate"]] for c in CAPS) + ".")])
    for key in ("C44", "C46"):
        row = rows[key]
        selected_baseline = {c: min((m for m in quant["methods"] if m != row.candidates[c]),
                           key=lambda m:quant["loso"]["scores"][m][c]["macro_mae"]) for c in CAPS}
        qp = "results/v69-quant-confirm/compare.json"
        rr = [(i,r) for i,r in enumerate(audit.data[qp]["rows"])
              if (r["test_set"] == "development_state_boundary") == (key=="C44")]
        # Reuse stored subset MAEs after the legacy loader verifies them against
        # every frozen prediction. New-state strata are weighted by stored n.
        def quant_score(c, method):
            subsets = ("development_state_boundary",) if key == "C44" else ("new_state_boundary", "new_state_interior")
            if len(subsets) == 1:
                part = number(qp, "test_sets", subsets[0], "scores", method, c, "mae")
            else:
                part = value(*[pointer(qp, "test_sets", s, "scores", method, c, field)
                               for s in subsets for field in ("mae", "n")], op="weighted_mean", fmt=".2f")
            if evaluate(audit, part) != format(row.scores[c][method].value, ".2f"):
                raise ValueError("Stored quantization score differs from the frozen-cell mean")
            return part
        range_refs, seen_values = [], set()
        for i, r in rr:
            for k in ("state", "config"):
                if (k, r[k]) not in seen_values:
                    range_refs.append(pointer(qp, "rows", i, k))
                    seen_values.add((k, r[k]))
        result.append([
            cell(tested_range(audit, "quant_group" if key=="C44" else "quant_new",
                             *range_refs),
                 refs=[pointer("results/v69-quant-confirm/freeze.json","frozen_at_utc"),
                       *[pointer(qp,"rows",i,"test_set") for i,_ in rr]]),
            cell(mapped(audit, composite_label(row.candidates), *[pointer(Q69,"selected",c,"candidate") for c in CAPS]),
                 refs=[pointer(Q55,"candidate_definitions","low_order_2d"),pointer(Q69,"models"),
                       pointer(Q69,"feature_names"),pointer(Q69,"dev_configs")],
                 note="Frozen candidates unchanged: Math uses the source surface, Code the per-configuration median, QA zero. Median interpolation is source-free and differs from the delivered same-input interpolation."),
            cell(*score_parts(lambda c:[quant_score(c, row.candidates[c])])),
            *delivered_cells(audit, {c: "median" if key == "C46" or c == "qa" else "same_input_interpolation" for c in CAPS},
                             [*delivery_refs(key), pointer(R74,"recommendation_rule")],
                             lambda c: [quant_score(c, "median" if key == "C46" or c == "qa" else "same_input_interpolation")],
                             note="Post-test rule on the identical frozen cells, never a test-error minimum. New-state errors pool boundary and interior cells equally per cell."),
            cell(*baseline_parts(audit, lambda c: [
                     baseline(audit, selected_baseline[c], pointer(Q69,"loso","scores",selected_baseline[c],c)),
                     quant_score(c, selected_baseline[c])]),
                 refs=[pointer(Q69,"loso","scores")],note="Minimum development LOSO macro MAE excluding the selected candidate, per capability: " +
                 " / ".join(BASELINES[selected_baseline[c]] for c in CAPS) + ".")])

    f70 = "results/v70-distill-confirm/freeze.json"
    groups = audit.data[D70]["groups"]
    students = audit.data[f70]["confirmation_register"]["students"]
    pool_inds = {c: [next(i for i,g in enumerate(groups) if g["student"]==s and g["capability"]==c)
                     for s in students] for c in CAPS}
    def pool_numbers(c, key):
        parts = []
        for i in pool_inds[c]:
            if parts:
                parts.append(", ")
            parts.append(number(D70,"groups",i,key))
        return parts
    pool_relation = cell(mapped(audit, composite_label({c: audit.data[f70]["selected"][c]["method"] for c in CAPS}),
                                *[pointer(f70,"selected",c,k) for c in CAPS for k in ("method", "n_params")]),
                         refs=[pointer(f70,"selected"), pointer(f70,"models")],
                         note="The frozen and delivered forms are identical: one-coefficient zero-anchored reuse for Math/Code, joint budget/pool for QA; fixed before test.")
    result.append([
        cell(tested_range(audit, "pool", pointer(f70,"confirmation_register","students"),
                         pointer(f70,"confirmation_register","pools"),pointer(D70,"groups",0,"clusters",0,"T_planned")),
             refs=[pointer(f70,"confirmation_register","unused_U_assertion"),
                    pointer(f70,"frozen_at_utc"),pointer("results/v47-p2-register/register.json","v5_confirm","registered_at_utc")]),
        pool_relation,
        cell(*score_parts(lambda c: pool_numbers(c,"candidate_mae")),
             refs=[pointer(f70,"confirmation_register","students"),pointer(f70,"bootstrap")],
             note="Scores follow student order 270M, 1B; no student averaging. Stored paired baseline-minus-candidate intervals do not fit beside these scores and remain in the appendix tables; each baseline identity is checked by the reused frozen loader."),
        cell(mapped(audit, "The same predictor", *pool_relation.parts[0]["sources"]),
             refs=pool_relation.context, note=pool_relation.note),
        cell(*score_parts(lambda c: pool_numbers(c,"candidate_mae")), refs=delivery_refs("C47") + delivery_refs("C48"),
             note="Same candidate, same cells; repeat the stored MAEs without the candidate-versus-baseline intervals."),
        cell(*baseline_parts(audit, lambda c: [
                 mapped(audit, baseline_names(groups[i]["strongest_baseline"] for i in pool_inds[c]),
                        *[pointer(f70,"strongest_baseline",groups[i]["student"],c,"method") for i in pool_inds[c]]),
                 *pool_numbers(c,"baseline_mae")]), refs=[pointer(f70,"baseline_rule"),pointer(f70,"strongest_baseline")],
             note="Development-fixed baseline identities, student order 270M, 1B: " +
             " / ".join(baseline_names(groups[i]["strongest_baseline"] for i in pool_inds[c]) for c in CAPS) + ".")])
    # Configuration measurements exclude dense anchors, repeats and test cells.
    # V53 counts scalar capability rows, whereas V55/V69 count configurations.
    # In V69 the QA zero rule still used the development panel for selection.
    d70 = "results/v70-distill-confirm/develop.json"
    development = audit.read(d70)
    if not audit.matches_digest(audit.data[f70]["inputs_sha256"][d70], audit.inputs[d70]):
        raise ValueError("Distillation development does not match its frozen registration")
    if prune["n_dev_rows"] != len(CAPS) * sum(len(s["densities"]) for s in prune["dev_states"]):
        raise ValueError("Pruning development row count disagrees with registered cells")
    if development["development_structure"]["n_points"] != len(development["points"]):
        raise ValueError("Distillation development count disagrees with registered points")
    counts = [
        cell(value(pointer(P53, "n_dev_rows"), pointer(P53, "models"), op="per_capability"),
             refs=[pointer(P53, "dev_states")],
             note="252 recorded scalar rows / three capability models = 84 state-density configurations per capability; cross-checked against dev_states[*].densities. This is the V53 17-state fit, not A9/A11's 36-cell panel. Dense anchors excluded."),
        cell(value(pointer(Q55, "n_dev_cells")), refs=[pointer(Q55, "dev_states"), pointer(Q55, "dev_configs")],
             note="Recorded configuration count; six states times four configurations per capability. Dense anchors excluded."),
        *[cell(value(pointer(Q69, "n_dev_cells")), refs=[pointer(Q69, "dev_states"), pointer(Q69, "dev_configs"), pointer(Q69, "selection_rule")],
               note="Recorded configuration count; six states times nine configurations per capability. Includes development selection of QA's zero rule, which fits no coefficients. The same development panel supports both tests; counts are not additive.") for _ in range(2)],
        cell(value(pointer(d70, "development_structure", "n_points")),
             refs=[pointer(d70, "development_structure"), pointer(d70, "points"), pointer(f70, "inputs_sha256", d70)],
             note="100 registered checkpoints (25 trajectories times four) per capability, pooled across development students for the shared fit; not 100 per test student. The freeze authenticates develop.json; dense anchors excluded."),
    ]
    for row, count in zip(result, counts):
        row.append(count)
    result.append(efficiency_row(audit))
    complete = []
    for row in result:
        try:
            for c in row:
                if not c.plain(audit).strip():
                    raise MissingValue("Frozen cell is empty")
                if not c.context and not any(isinstance(p, dict) for p in c.parts):
                    raise ValueError("Cell lacks JSON provenance")
        except MissingValue:
            audit.omit("Task omitted because a frozen value is absent: " + row[0].plain(audit))
        else:
            for index in (2, 4):
                if len(row[index].plain(audit).splitlines()) == len(CAPS):
                    row[index].compact_scores = "ordered"
            row[5].compact_scores = "comparison"
            complete.append(row)
    return complete


def student_label(audit, ref):
    return mapped(audit, {"gemma3-270m": "270M", "gemma3-1b": "1B"}[resolve(audit, ref)], ref)


def caption_cell(audit):
    before, after = CAPTION.split("{students}")
    return cell(before, mapped(audit, "270 million and 1 billion",
                              pointer("results/v70-distill-confirm/freeze.json","confirmation_register","students")), after, refs=[
        pointer(P53,"feature_names"), pointer(Q55,"feature_names"),
        pointer("results/v70-distill-confirm/freeze.json","reference_rule"),
        pointer(P53,"candidate_definitions","power"), pointer(Q55,"candidate_definitions","low_order_2d"),
        pointer("results/v70-distill-confirm/freeze.json","models")])


def render_table(rows, audit, *, sidecar="main_prediction_v2_sources.md"):
    # Empty outer padding leaves two tabcolseps per interior gap.
    gaps = 2 * (len(HEADERS) - 1)
    columns = "".join(
        r">{\raggedright\arraybackslash\hspace{0pt}}p{\dimexpr " + width +
        r"\textwidth-" + f"{gaps * float(width):.2f}" + r"\tabcolsep\relax}"
        for width in COLUMN_WIDTHS)
    lines = [f"% Generated from frozen JSON; see {sidecar}.",
             r"\begin{table*}[!htbp]\normalfont", r"\centering" + TABLE_FONT + r"\linespread{0.85}\selectfont",
             r"\setlength{\tabcolsep}{1.5pt}", r"\renewcommand{\arraystretch}{0.92}",
             r"\setlength{\abovecaptionskip}{4pt}",
             r"\newcommand{\TableOneErrors}[3]{\setbox0=\hbox{#1 / #2 / #3}%",
             r"\ifdim\wd0>\linewidth #1\newline #2\newline #3\else\box0\fi}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}" + columns + "@{}}",
             r"\toprule", " & ".join(render_text(h) for h in HEADERS) + r" \\", r"\midrule"]
    body = [" & ".join(c.render(audit) for c in row) + r" \\" for row in rows]
    lines += [("\n" + r"\midrule" + "\n").join(body)]
    lines += [r"\bottomrule", r"\end{tabular*}", r"\caption{"+CAPTION_FONT+" "+caption_cell(audit).render(audit)+"}",
              r"\label{tab:main-prediction-v2}", r"\end{table*}"]
    return "\n".join(lines)+"\n"


def numeric_inventory(records, audit):
    """List every printed numeric occurrence, including numbers in test labels."""
    entries = []
    for record in records:
        for index, part in enumerate(record["parts"]):
            if isinstance(part, str):
                if re.search(r"\d", part):
                    raise ValueError("Printed numeric literal lacks a JSON recipe")
                continue
            displayed = evaluate(audit, part)
            for match in re.finditer(NUMBER_PATTERN, displayed, re.I):
                location = {"row": record["row"], "column": record["column"]} if "row" in record else {"location": "caption"}
                entries.append({**location, "part": index,
                                "number": match.group(), "displayed": displayed,
                                "op": part["op"], "sources": part["sources"]})
    return entries


def generate(root=ROOT, *, write_tex=True):
    with frozen_run(root) as access:
        audit = Artifacts(root)
        rows = build(audit)
        caption = caption_cell(audit)
        sidecar = "main_prediction_v2_sources.md" if write_tex else "main_prediction_v2_preview_sources.md"
        tex = render_table(rows, audit, sidecar=sidecar)
        records = [{"row":i,"column":j,**c.record(audit)} for i,row in enumerate(rows) for j,c in enumerate(row)]
        numbers = numeric_inventory([*records, caption.record(audit)], audit)
        side = ["# Main prediction table: cell sources", "", "Columns: " + "; ".join(HEADERS) + ".",
                "Rendered layout: one row per frozen prediction task. Cell coordinates match the seven printed columns; every cell is populated.",
                "Development measurements are distinct development configuration measurements per capability for fitting or selecting the tested candidate, excluding dense anchors and held-out measurements. Counts do not describe the post-test delivered predictor. V53 divides recorded scalar rows by the recorded capability-model count; V55/V69 use n_dev_cells; V70 uses development_structure.n_points authenticated by freeze.json. Shared development sets are not additive across rows.",
                "Numeric error cells use one math / code / question answering line if it fits the actual column, otherwise three lines in that order. The earlier bit test explicitly states that the delivered error is not stored. Distillation pairs follow student order 270 million, 1 billion as stated in the caption. Baseline names precede their scores, in the capability order named in the caption. Development or registration selection pointers are recorded in each baseline cell's note/context. Short task labels retain the full state and configuration definitions in their source recipes.",
                "The caption identifies deliveries preceding distillation as chosen after testing; first predictors were frozen before measurement. Delivered scores reuse the same frozen test cells, not test-error winners. For the earlier bit test the delivered interpolation/median rule has no matching stored score; its development includes those cells. The source surface is not delivered, and the older interpolation's different model and boundary rules cannot supply the missing error.",
                "The efficiency confirmation reads summary.json and per_state.json, authenticates their frozen prediction identity, and checks all per-state means against the same recorded cells. The first predictor uses the reduced budget; the registered regression comparison and the delivered QA median use the full budget. Each new cell has its own executable recipe and context pointers.",
                "Stored V70 paired_difference.ci95 endpoints are omitted from Table 1 because they do not fit on the same line as the score. They remain in the appendix tables. The sign is baseline minus candidate; these are not MAE intervals. No refits, resampling or invented intervals.",
                "All indices are zero-based JSON pointers. `mean` is equal-weight arithmetic; `weighted_mean` pairs each stored subset MAE with its stored cell count, preserving equal cell weights. No refits or resampling.",
                "Numeric values are formatted directly from the following executable source recipes. Context pointers justify textual labels and freeze identities; incomplete tasks are omitted.",
                "The `label` operation uses the generator's explicit presentation mappings; `expected` preserves the source values and must match before rendering `label`. Counts always use their own JSON fields.",
                "The `shared` operation prints a common value only after checking that all grouped source values are identical.",
                "", "## Loader reuse and limits", "", *audit.notes, "", "## Fields per cell", ""]
        for r in records:
            refs = r["context"] + [s for p in r["parts"] if isinstance(p,dict) for s in p["sources"]]
            side += [f"- Cell ({r['row']}, {r['column']}): " + "; ".join(f"`{s}`" for s in dict.fromkeys(refs))]
        side += ["", "## Machine-readable cell recipes", "", "```json",json.dumps(records,indent=2),"```",
                 "", "## Caption recipe", "", "```json",json.dumps(caption.record(audit),indent=2),"```",
                 "", "## Every printed number", ""]
        side += [(f"- Cell ({n['row']}, {n['column']})" if "row" in n else "- Caption") +
                 f", `{n['number']}` ({n['op']}): " +
                 "; ".join(f"`{s}`" for s in n["sources"]) for n in numbers]
        side += ["", "## Machine-readable numeric inventory", "", "```json",json.dumps(numbers,indent=2),"```",
                 "", "## Input hashes", ""]
        side += [f"- `{p}`: `{h}`" for p,h in sorted(audit.inputs.items())]
        outputs = [(sidecar, "\n".join(side)+"\n")]
        if write_tex:
            outputs.insert(0, ("main_prediction_v2.tex", tex))
        for name, content in outputs:
            path=output_path(root,"tables",name);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
        return rows,audit,access


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", action="store_true", help="Print LaTeX and write a preview sidecar, preserving the existing .tex/source pair")
    args = parser.parse_args()
    rows, audit, _ = generate(write_tex=not args.stdout)
    print(render_table(rows, audit, sidecar="main_prediction_v2_preview_sources.md" if args.stdout else "main_prediction_v2_sources.md"), end="")
