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

# Presentation vocabulary only. Coefficient counts and scores still come from
# JSON fields; each mapped phrase carries the source values in its recipe.
RELATIONS = {
    "power": "Five-parameter power form",
    "low_order_2d": "Source regression across bit widths (twenty coefficients)",
    "median": "Development median",
    "median_curve": "Source-free median density curve",
    "same_input_interpolation": "Interpolation",
    "zero": "no change",
    "E": "Reuse form",
    "joint": "Budget and pool form",
}
# The appendix retains this word; the main table maps the same stored delivery
# timing to a dagger beside the delivered relation.
RETROSPECTIVE = "Retrospective"
BASELINES = {
    "A2": "per-density regression", "median_curve": "development median", "median": "development median",
    "bilinear": "bilinear regression", "zero": "no change",
    "T-only": "budget regression", "E-only": "reuse regression",
    "surface:L0": "loss regression", "surface:logN": "size regression",
}
RANGES = {
    "prune_new": "Three unseen checkpoints pruned to densities 0.575, 0.675 and 0.85",
    "quant_group": "Pythia 410 million and 1.4 billion at unseen group sizes 32 and 512, bit widths 3 to 5",
    "quant_new": "An unseen 1.4 billion stage at bit widths 3 to 5, group sizes 32 to 512",
    "pool": "Gemma 270 million and 1 billion distilled on six new pools at 50 to 200 thousand tokens",
    "pool_270m": "Gemma 270 million distilled on six new pools at 50 to 200 thousand tokens",
    "pool_1b": "Gemma 1 billion distilled on six new pools at 50 to 200 thousand tokens",
}
SHORT_RANGES = {
    "prune_new": "Three new checkpoints",
    "quant_group": "Group sizes 32 and 512",
    "quant_new": "A new 1.4 billion stage",
    "pool_270m": "Six new pools, 270 million student",
    "pool_1b": "Six new pools, 1 billion student",
}
METHODS = {"prune_new": "Pruning", "quant_group": "Quantization", "quant_new": "Quantization",
           "pool_270m": "Distillation", "pool_1b": "Distillation"}
SHORT_RELATIONS = {
    ("median_curve",) * 3: "Median density curve",
    ("same_input_interpolation", "same_input_interpolation", "median"): "Interpolation; median for QA",
    ("median",) * 3: "Development median",
    ("E", "E", "joint"): "Reuse; budget and pool for QA",
}
# Every task is assembled once with eight cells: task, frozen candidate, its
# error, the delivered relation, its error, the strongest development baseline
# and the development budget and improvement. The main table prints the delivered relation
# against its baseline; the appendix candidates table prints how the frozen
# candidate became the delivered relation.
HEADERS = ("Test", "Delivered relation", "Math", "Code", "QA", "Math", "Code", "QA")
# Method lines are short; a natural-width span keeps tabular* from adding the table width to the last column.
GROUP_SPEC = r"@{}l@{}"
CANDIDATE_COLUMNS = (0, 1, 2, 3, 5)
CANDIDATE_HEADERS = ("Prediction task", "Pre-specified candidate", "Candidate error (nats)",
                     "Delivered relation", "Strongest development baseline")
CANDIDATE_WIDTHS = (".24", ".22", ".12", ".22", ".20")
TABLE_FONT = r"\footnotesize\fontsize{8}{9.5}\selectfont"
CAPTION_FONT = r"\footnotesize\fontsize{8.5}{10}\selectfont"
CAPTION = (
    "Prediction at equal development budget. Test: configurations or states unseen by the relation. "
    "Delivered relation: predictor recommended on all evidence; a dagger marks selection after seeing test results. "
    "Error: test mean absolute error in nats per token, per capability (QA: question answering). "
    "Gain: strongest development baseline error minus delivered error; positive favours the delivered relation. "
    "Baselines are chosen inside the development folds and scored on the same cells. "
    "$n$, on each method line: development configuration measurements per capability. "
    "Appendix Table~\\ref{tab:main-prediction-candidates} names baselines and pre-specified candidates; "
    "Fig.~\\ref{fig:generalization} compares each row with its baseline."
)
CANDIDATE_CAPTION = (
    "Pre-specified candidates for Table~\\ref{tab:main-prediction-v2}, their errors on the same cells, "
    "delivered relations, and strongest development baselines. Errors are mean absolute errors in "
    "nats per token; errors and baseline names follow math, code and question answering order unless qualified. "
    "The distillation rows are the {students} students."
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


def tested_range(audit, name, *sources, short=False):
    return mapped(audit, (SHORT_RANGES if short else RANGES)[name], *sources)


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
    elif op == "difference":
        # Signed improvement: the baseline value minus the delivered value, each evaluated by its own recorded part.
        b_part, d_part = part["parts"]
        b_val = float(evaluate(audit, {**b_part, "format": None}))
        d_val = float(evaluate(audit, {**d_part, "format": None}))
        result = b_val - d_val
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
    # Mathematics and cross-references pass through; everything else is escaped.
    protected = r"(\$[^$]+\$|(?:Fig\.|Table|Section|Appendix|Eq\.)?~?\\ref\{[^}]*\})"
    return r"\newline ".join(signed("".join(
        token if token.startswith("$") or "\\ref{" in token else tex_escape(token)
        for token in re.split(protected, line))) for line in text.split("\n"))


def signed(line):
    """Typeset the sign of a signed score as math, so minus and plus have one width and centred
    improvements keep their decimal points aligned; every other line is returned unchanged."""
    match = re.fullmatch(r"([+-])(\d+\.\d+)", line)
    return f"${match.group(1)}${match.group(2)}" if match else line


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
    capabilities: dict = field(default_factory=dict)

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


def cell(*parts, refs=(), note="", capabilities=None):
    return Cell(list(parts), list(refs), note, capabilities=capabilities or {})


def number(path, *keys):
    return value(pointer(path, *keys), fmt=".3f")


def scored_number(n):
    # Reuse V86's source-carrying Number, including its equal-weight means.
    if n.source.startswith("mean("):
        refs = n.source[5:-1].split("; ")
        return value(*refs, op="mean", fmt=".3f")
    return value(n.source, fmt=".3f")


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
    elif names == ["budget regression", "reuse regression", "reuse regression"]:
        label = "Budget regression; reuse regression for code and question answering"
    elif names == ["loss regression", "reuse regression", "size regression"]:
        label = "Regressions on loss, reuse and size"
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


def baseline_cell(audit, fn, *, refs=(), note=""):
    """Retain the original per-capability identities before making a summary."""
    entries = {cap: fn(cap) for cap in CAPS}
    return cell(*baseline_parts(audit, entries.__getitem__), refs=refs, note=note,
                capabilities={cap: cell({**entry[0], "label": evaluate(audit, entry[0]).capitalize()},
                                        *entry[1:], refs=refs, note=note)
                              for cap, entry in entries.items()})


def composite_label(names):
    """Translate the two frozen mixed predictors into ordinary words."""
    if list(names.values()) == ["E", "E", "joint"]:
        return "Reuse forms, with a budget and pool form for question answering"
    if list(names.values()) == ["low_order_2d", "median", "zero"]:
        return "Source regression, the median and no change"
    if list(names.values()) == ["same_input_interpolation", "same_input_interpolation", "median"]:
        return "Interpolate math and code; use the median for question answering"
    raise ValueError(f"Unmapped mixed predictor: {names}")


def delivered_cells(audit, names, refs, scores, *, note="", timing=None):
    """Keep rule identity/timing separate; incomplete tasks are omitted below."""
    if len(set(names.values())) == 1:
        label = RELATIONS[names[CAPS[0]]]
    else:
        label = composite_label(names)
    identity = [mapped(audit, label, *refs)]
    if timing is not None and resolve(audit, timing).startswith("fixed after test"):
        identity += ["\n", mapped(audit, RETROSPECTIVE, timing)]
    capabilities = {}
    for cap in CAPS:
        ref, = [ref for ref in refs if ref.endswith(f"/capabilities/{cap}/delivered")]
        if resolve(audit, ref) != names[cap]:
            raise ValueError("Delivered identity differs from its frozen selection")
        capabilities[cap] = cell(relation(audit, names[cap], ref), refs=refs, note=note)
    return [
        cell(*identity, note=note, capabilities=capabilities),
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
        return [delivery_timing(key),
                *[pointer(S86, "main_rows", i, "capabilities", c, "delivered") for c in CAPS]]

    def delivery_timing(key):
        return pointer(S86, "main_rows", delivery_indices[key], "delivered_timing")

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
                         timing=delivery_timing("C35"),
                         note="New-state source-free median curve, fixed after test; scores use the same three checkpoints and densities as the frozen candidate."),
        baseline_cell(audit, lambda c: [
                 baseline(audit, pbase[c][1]["candidate"], pointer(P53, "loso_table", pbase[c][0], "candidate")),
                 scored_number(row.scores[c][pbase[c][1]["candidate"]])],
             refs=[pointer(P53, "loso_table")], note="Minimum development LOSO MAE excluding power, per capability: " +
             " / ".join(BASELINES[pbase[c][1]["candidate"]] for c in CAPS) + "; no test ranking.")])

    # The earlier unseen-bit-width test predates the confirmation round and is the
    # only genuine unseen-bit test; the later round's bit widths were all seen. It
    # is reported in the candidate-form appendix rather than as a row here: its
    # cells were added to the later development grid, so the delivered rule can
    # show no independent error on them, and the earlier interpolation carries
    # different models and boundary rules and cannot stand in for it. The
    # membership audit that licenses that statement still runs, and the numbers
    # the appendix prints are recorded below from the same frozen JSON.
    bp = "results/v55-quant-group/compare.json"
    entries = comp55["test_sets"]["bit_test"]["mae_table"]
    base55 = {c: min(((i, r) for i, r in enumerate(q55["loso_table"]) if r["candidate"] in q55["baselines"]),
                     key=lambda ir: ir[1]["mae"][c]) for c in CAPS}
    bit_cells = {(s, cfg, c) for s in q55["test_sets"]["bit_test"]["states"]
                 for cfg in q55["test_sets"]["bit_test"]["configs"] for c in CAPS}
    later_cells = {(r["state"], r["config"], r["capability"])
                   for r in audit.data["results/v69-quant-confirm/compare.json"]["rows"]}
    if bit_cells & later_cells or not set(q55["test_sets"]["bit_test"]["configs"]) <= set(quant["dev_configs"]):
        raise ValueError("Earlier bit-test membership changed; re-audit delivered-score availability")

    def bit_mae(candidate, cap):
        index = next(j for j, e in enumerate(entries) if e["candidate"] == candidate)
        return format(resolve(audit, pointer(bp, "test_sets", "bit_test", "mae_table", index, "mae", cap)), ".3f")

    audit.omit(
        "The earlier unseen-bit-width test moves to the candidate-form appendix: its cells entered the "
        "later development grid, so the delivered rule has no independent error on them. Frozen "
        "low-order surface: " + " / ".join(bit_mae("low_order_2d", c) for c in CAPS) + " nats against " +
        " / ".join(bit_mae(base55[c][1]["candidate"], c) for c in CAPS) + " for the development-selected "
        "baseline (" + " / ".join(BASELINES[base55[c][1]["candidate"]] for c in CAPS) + "), from " +
        str(resolve(audit, pointer(Q55, "n_dev_cells"))) +
        " development configuration measurements per capability.")
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
                               for s in subsets for field in ("mae", "n")], op="weighted_mean", fmt=".3f")
            if evaluate(audit, part) != format(row.scores[c][method].value, ".3f"):
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
                             timing=delivery_timing(key),
                             note="Post-test rule on the identical frozen cells, never a test-error minimum. New-state errors pool boundary and interior cells equally per cell."),
            baseline_cell(audit, lambda c: [
                     baseline(audit, selected_baseline[c], pointer(Q69,"loso","scores",selected_baseline[c],c)),
                     quant_score(c, selected_baseline[c])],
                 refs=[pointer(Q69,"loso","scores")],note="Minimum development LOSO macro MAE excluding the selected candidate, per capability: " +
                 " / ".join(BASELINES[selected_baseline[c]] for c in CAPS) + ".")])

    f70 = "results/v70-distill-confirm/freeze.json"
    groups = audit.data[D70]["groups"]
    students = audit.data[f70]["confirmation_register"]["students"]
    pool_inds = {c: [next(i for i,g in enumerate(groups) if g["student"]==s and g["capability"]==c)
                     for s in students] for c in CAPS}
    def pool_numbers(c, key, j):
        """One student's frozen number for capability c (student j of the confirmation register)."""
        return [number(D70,"groups",pool_inds[c][j],key)]
    pool_relation = cell(mapped(audit, composite_label({c: audit.data[f70]["selected"][c]["method"] for c in CAPS}),
                                *[pointer(f70,"selected",c,k) for c in CAPS for k in ("method", "n_params")]),
                         refs=[pointer(f70,"selected"), pointer(f70,"models")],
                         note="The frozen and delivered forms are identical: one-coefficient zero-anchored reuse for Math/Code, joint budget/pool for QA; fixed before test.")
    for j, student in enumerate(students):
        label = {"gemma3-270m": "pool_270m", "gemma3-1b": "pool_1b"}[resolve(audit, pointer(f70,"confirmation_register","students",j))]
        word = {"gemma3-270m": "270M", "gemma3-1b": "1B"}[resolve(audit, pointer(f70,"confirmation_register","students",j))]
        result.append([
            cell(tested_range(audit, label, pointer(f70,"confirmation_register","students",j),
                             pointer(f70,"confirmation_register","pools"),pointer(D70,"groups",0,"clusters",0,"T_planned")),
                 refs=[pointer(f70,"confirmation_register","unused_U_assertion"),
                        pointer(f70,"frozen_at_utc"),pointer("results/v47-p2-register/register.json","v5_confirm","registered_at_utc")]),
            pool_relation,
            cell(*score_parts(lambda c: pool_numbers(c,"candidate_mae",j)),
                 refs=[pointer(f70,"confirmation_register","students"),pointer(f70,"bootstrap")],
                 note="Scores of one student, " + word + "; no student averaging. Stored paired baseline-minus-candidate intervals do not fit beside these scores and remain in the frozen record."),
            cell(mapped(audit, "The same predictor", *pool_relation.parts[0]["sources"]),
                 refs=pool_relation.context, note=pool_relation.note,
                 capabilities={c: cell(relation(audit, resolve(audit, pointer(f70, "selected", c, "method")),
                                                 pointer(f70, "selected", c, "method")),
                                       refs=pool_relation.context, note=pool_relation.note) for c in CAPS}),
            cell(*score_parts(lambda c: pool_numbers(c,"candidate_mae",j)), refs=delivery_refs("C47") + delivery_refs("C48"),
                 note="Same candidate, same cells; repeat the stored MAEs without the candidate-versus-baseline intervals."),
            baseline_cell(audit, lambda c: [
                     mapped(audit, baseline_names([groups[pool_inds[c][j]]["strongest_baseline"]]),
                            pointer(f70,"strongest_baseline",groups[pool_inds[c][j]]["student"],c,"method")),
                     *pool_numbers(c,"baseline_mae",j)], refs=[pointer(f70,"baseline_rule"),pointer(f70,"strongest_baseline")],
                 note="Development-fixed baseline identities for this student: " +
                 " / ".join(baseline_names([groups[pool_inds[c][j]]["strongest_baseline"]]) for c in CAPS) + ".")])
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
        *[cell(value(pointer(Q69, "n_dev_cells")), refs=[pointer(Q69, "dev_states"), pointer(Q69, "dev_configs"), pointer(Q69, "selection_rule")],
               note="Recorded configuration count; six states times nine configurations per capability. Includes development selection of QA's zero rule, which fits no coefficients. The same development panel supports both tests; counts are not additive.") for _ in range(2)],
        cell(value(pointer(d70, "development_structure", "n_points")),
             refs=[pointer(d70, "development_structure"), pointer(d70, "points"), pointer(f70, "inputs_sha256", d70)],
             note="100 registered checkpoints (25 trajectories times four) per capability, pooled across development students for the shared fit; not 100 per test student. The freeze authenticates develop.json; dense anchors excluded."),
    ]
    counts.append(counts[-1])   # the second student row shares the distillation development count
    for row, count in zip(result, counts):
        row.append(count)
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
            # The baseline cell names the comparison, then its three scores.
            if len(row[5].plain(audit).splitlines()) == len(CAPS) + 1:
                row[5].compact_scores = "comparison"
            # Signed improvement per capability: baseline minus delivered, from the two cells' own recorded parts.
            d_parts = [q for q in row[4].parts if isinstance(q, dict) and q.get("format")][-len(CAPS):]
            b_parts = [q for q in row[5].parts if isinstance(q, dict) and q.get("format")][-len(CAPS):]
            if len(d_parts) == len(CAPS) and len(b_parts) == len(CAPS):
                parts = []
                for d, b in zip(d_parts, b_parts):
                    if parts:
                        parts.append("\n")
                    parts.append({"sources": list(d["sources"]) + list(b["sources"]), "op": "difference",
                                  "parts": [b, d], "format": "+.3f"})
                improvement = Cell(parts, list(row[4].context) + list(row[5].context),
                                   "baseline error minus delivered error, per capability", "ordered")
            else:
                improvement = Cell(["not comparable"], list(row[4].context) + list(row[5].context),
                                   "the two cells do not list one score per capability")
            while len(row) < 8:
                row.append(improvement)
            complete.append(row)
    return complete


def printed(rows, columns):
    """Select the cells one table prints, in its column order."""
    return [[row[i] for i in columns] for row in rows]


def student_label(audit, ref):
    return mapped(audit, {"gemma3-270m": "270M", "gemma3-1b": "1B"}[resolve(audit, ref)], ref)


def caption_cell(audit, text=CAPTION):
    parts = [text]
    if "{students}" in text:
        before, after = text.split("{students}")
        parts = [before, mapped(audit, "270 million and 1 billion",
                               pointer("results/v70-distill-confirm/freeze.json", "confirmation_register", "students")), after]
    return cell(*parts, refs=[
        pointer(P53,"feature_names"), pointer(Q55,"feature_names"),
        pointer("results/v70-distill-confirm/freeze.json","reference_rule"),
        pointer(P53,"candidate_definitions","power"), pointer(Q55,"candidate_definitions","low_order_2d"),
        pointer("results/v70-distill-confirm/freeze.json","models")])


def compact_delivered(audit, delivered):
    """Summarize the resolved capability identities, retaining their sources."""
    refs = [ref for cap in CAPS for ref in delivered.capabilities[cap].parts[0]["sources"]]
    names = tuple(resolve(audit, ref) for ref in refs)
    parts = [mapped(audit, SHORT_RELATIONS[names], *refs)]
    for part in delivered.parts:
        if isinstance(part, dict) and part.get("label") == RETROSPECTIVE:
            parts.append(mapped(audit, r"$^{\dagger}$", *part["sources"]))
    return cell(*parts, refs=delivered.context, note=delivered.note)


def compact_baseline(audit, baseline_):
    """Print names only, in capability order or with explicit qualifications."""
    identities = [baseline_.capabilities[cap].parts[0] for cap in CAPS]
    names = [evaluate(audit, part).replace("Development median", "Median") for part in identities]
    if len(set(names)) == 1:
        label = names[0]
    elif names[0] == names[1]:
        label = names[0] + "; " + names[2].lower() + " for QA"
    elif names[1] == names[2]:
        label = names[0] + "; " + names[1].lower() + " for Code and QA"
    else:
        label = "; ".join([names[0], *[name.lower() for name in names[1:]]])
    return cell(mapped(audit, label, *[ref for part in identities for ref in part["sources"]]),
                refs=baseline_.context, note=baseline_.note)


def candidate_rows(rows, audit):
    selected = printed(rows, CANDIDATE_COLUMNS)
    return [[*row[:-1], compact_baseline(audit, row[-1])] for row in selected]


def main_rows(rows, audit):
    """Physical body rows: an italic method heading followed by one row per task.

    Move the original score recipes intact, including the unrounded operands of
    each improvement; never subtract already formatted table values.
    """
    physical = []
    previous_method = None
    counts = {}
    for row in rows:
        task_key = next(key for key in SHORT_RANGES if RANGES[key] == row[0].plain(audit))
        counts.setdefault(METHODS[task_key], set()).add(row[6].plain(audit))
    if any(len(values) != 1 for values in counts.values()):
        raise ValueError("Tasks of one method must share their development measurement count")
    for row in rows:
        task, delivered, scores, count, improvement = (row[i] for i in (0, 3, 4, 6, 7))
        task_key = next(key for key in SHORT_RANGES if RANGES[key] == task.plain(audit))
        sources = task.parts[0]["sources"]
        method = METHODS[task_key]
        if method != previous_method:
            # The method line carries the development count its tasks share.
            physical.append([cell(mapped(audit, method, *sources), ", $n={}$".format(""), refs=task.context + count.context)])
            physical[-1][0].parts[-1:] = [", $n=", *count.parts, "$"]
            previous_method = method
        d_parts = [p for p in scores.parts if isinstance(p, dict)]
        i_parts = [p for p in improvement.parts if isinstance(p, dict)]
        if len(d_parts) != len(CAPS) or len(i_parts) != len(CAPS):
            raise ValueError("Main table requires one error and improvement per capability")
        physical.append([
            cell(tested_range(audit, task_key, *sources, short=True), refs=task.context, note=task.note),
            compact_delivered(audit, delivered),
            *[cell(d, refs=scores.context, note=scores.note) for d in d_parts],
            *[cell(delta, refs=improvement.context, note=improvement.note) for delta in i_parts],
        ])
    return physical


def render_group(content):
    return r"\multicolumn{8}{" + GROUP_SPEC + r"}{\textit{" + content + "}}"


def main_records(rows, audit):
    records = []
    for i, row in enumerate(main_rows(rows, audit)):
        for j, c in enumerate(row):
            record = {"table": "tab:main-prediction-v2", "row": i, "column": j, **c.record(audit)}
            if len(row) == 1:
                record.update(column_span=8, rendered=render_group(record["rendered"]))
            records.append(record)
    return records


def render_table(rows, audit, *, sidecar="main_prediction_v2_sources.md"):
    # Short one-line labels in natural-width text columns; equal-precision scores in
    # right-aligned columns share decimal positions; \extracolsep{\fill} spreads the spare width.
    column_spec = "ll" + "r" * 6
    lines = [f"% Generated from frozen JSON; see {sidecar}.",
             r"\begin{table*}[t]\normalfont", r"\centering" + TABLE_FONT + r"\linespread{0.85}\selectfont",
             r"\setlength{\tabcolsep}{1.5pt}", r"\renewcommand{\arraystretch}{0.88}",
             r"\setlength{\abovecaptionskip}{4pt}",
             r"\caption{" + CAPTION_FONT + " " + caption_cell(audit).render(audit) + "}",
             r"\label{tab:main-prediction-v2}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}" + column_spec + "@{}}",
             r"\toprule",
             r"& & \multicolumn{3}{c}{Error} & \multicolumn{3}{c}{Gain over baseline} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             " & ".join(render_text(h) for h in HEADERS) + r" \\", r"\midrule"]
    for i, row in enumerate(main_rows(rows, audit)):
        if len(row) == 1:
            if i:
                lines.append(r"\midrule")
            lines.append(render_group(row[0].render(audit)) + r" \\")
        else:
            lines.append(" & ".join(c.render(audit) for c in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def _render_candidates_table(rows, audit, *, sidecar,
                             headers, widths, label, caption, environment):
    rows = candidate_rows(rows, audit)
    # Empty outer padding leaves two tabcolseps per interior gap.
    gaps = 2 * (len(headers) - 1)
    column_spec = "".join(
        r">{\raggedright\arraybackslash\hspace{0pt}}p{\dimexpr " + width +
        r"\textwidth-" + f"{gaps * float(width):.2f}" + r"\tabcolsep\relax}"
        for width in widths)
    lines = [f"% Generated from frozen JSON; see {sidecar}.",
             r"\begin{" + environment + r"}[t]\normalfont", r"\centering" + TABLE_FONT + r"\linespread{0.85}\selectfont",
             r"\setlength{\tabcolsep}{1.5pt}", r"\renewcommand{\arraystretch}{0.88}",
             r"\setlength{\abovecaptionskip}{4pt}",
             r"\newcommand{\TableOneErrors}[3]{\setbox0=\hbox{#1 / #2 / #3}%",
             r"\ifdim\wd0>\linewidth #1\newline #2\newline #3\else\box0\fi}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}" + column_spec + "@{}}",
             r"\toprule", " & ".join(render_text(h) for h in headers) + r" \\", r"\midrule"]
    body = [" & ".join(c.render(audit) for c in row) + r" \\" for row in rows]
    lines += [("\n" + r"\midrule" + "\n").join(body)]
    lines += [r"\bottomrule", r"\end{tabular*}",
              r"\caption{"+CAPTION_FONT+" "+caption_cell(audit, caption).render(audit)+"}",
              r"\label{" + label + "}", r"\end{" + environment + "}"]
    return "\n".join(lines)+"\n"


def render_candidates(rows, audit, *, sidecar="main_prediction_v2_sources.md"):
    return _render_candidates_table(rows, audit, sidecar=sidecar,
                        headers=CANDIDATE_HEADERS, widths=CANDIDATE_WIDTHS,
                        label="tab:main-prediction-candidates", caption=CANDIDATE_CAPTION,
                        environment="table")


def numeric_inventory(records, audit):
    """List every printed numeric occurrence, including numbers in test labels."""
    entries = []
    # A cross-reference label is not a printed number.
    unreferenced = lambda text: re.sub(r"\\ref\{[^}]*\}", "", text)
    for record in records:
        for index, part in enumerate(record["parts"]):
            if isinstance(part, str):
                if re.search(r"\d", unreferenced(part)):
                    raise ValueError("Printed numeric literal lacks a JSON recipe")
                continue
            displayed = evaluate(audit, part)
            for match in re.finditer(NUMBER_PATTERN, displayed, re.I):
                location = {"row": record["row"], "column": record["column"]} if "row" in record else {"location": "caption"}
                entries.append({"table": record["table"], **location, "part": index,
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
        candidates_tex = render_candidates(rows, audit, sidecar=sidecar)
        candidate_caption = caption_cell(audit, CANDIDATE_CAPTION)
        records = main_records(rows, audit)
        caption_record = {"table": "tab:main-prediction-v2", **caption.record(audit)}
        candidate_caption_record = {"table": "tab:main-prediction-candidates", **candidate_caption.record(audit)}
        candidate_records = [{"table": "tab:main-prediction-candidates", "row":i,"column":j,**c.record(audit)}
                             for i,row in enumerate(candidate_rows(rows, audit)) for j,c in enumerate(row)]
        numbers = numeric_inventory([*records, caption_record, *candidate_records, candidate_caption_record], audit)
        side = ["# Main prediction table: cell sources", "", "Columns: " + "; ".join(HEADERS) + ".",
                "Rendered layout: nine columns: Test, Delivered relation, Error (Math, Code, QA), Gain over baseline (Math, Code, QA), n. "
                "Italic method headings span all nine columns, followed by one row per task. "
                "Coordinates are zero-based physical body rows and columns, excluding headers and rules. With all tasks present, method headings occupy rows 0, 2 and 5 at column 0 with column_span 9; task rows occupy rows 1, 3, 4, 6 and 7. Every cell is populated. "
                "The appendix candidates table prints columns " + "; ".join(CANDIDATE_HEADERS) + " from the same assembled rows; its recipes follow the main table's.",
                "Development measurements are distinct development configuration measurements per capability for fitting or selecting the tested candidate, excluding dense anchors and held-out measurements. Counts do not describe the post-test delivered predictor. V53 divides recorded scalar rows by the recorded capability-model count; V55/V69 use n_dev_cells; V70 uses development_structure.n_points authenticated by freeze.json. Shared development sets are not additive across rows.",
                "Each task row has a short test label (column 0), a delivered identity summarized from its three capability selections (column 1), delivered errors (columns 2--4), signed gains (columns 5--7), and a development measurement count (column 8). Gains retain both recorded operands and compute baseline minus delivered before rounding. The appendix's baseline column preserves each capability's development-selected identity and selection pointers; its original four columns are unchanged. Short test labels retain the full state and configuration definitions in their source recipes.",
                "A dagger follows the main-table delivered relation only when the stored delivered timing says fixed after test; the appendix retains the word Retrospective. First predictors were frozen before measurement. Delivered scores reuse the same frozen test cells, not test-error winners. For the earlier bit test the delivered interpolation/median rule has no matching stored score; its development includes those cells. The source surface is not delivered, and the older interpolation's different model and boundary rules cannot supply the missing error.",
                "Stored V70 paired_difference.ci95 endpoints are omitted from Table 1 because they do not fit on the same line as the score. They remain in the appendix tables. The sign is baseline minus candidate; these are not MAE intervals. No refits, resampling or invented intervals.",
                "All indices are zero-based. The numeric inventory identifies the table, physical row/column and recipe part, so appendix and main-table coordinates are distinct. `mean` is equal-weight arithmetic; `weighted_mean` pairs each stored subset MAE with its stored cell count, preserving equal cell weights. No refits or resampling.",
                "Numeric values are formatted directly from the following executable source recipes. Context pointers justify textual labels and freeze identities; incomplete tasks are omitted.",
                "The `label` operation uses the generator's explicit presentation mappings; `expected` preserves the source values and must match before rendering `label`. Counts always use their own JSON fields.",
                "The `shared` operation prints a common value only after checking that all grouped source values are identical.",
                "", "## Loader reuse and limits", "", *audit.notes, "", "## Fields per cell", ""]
        for r in records:
            refs = r["context"] + [s for p in r["parts"] if isinstance(p,dict) for s in p["sources"]]
            side += [f"- Cell ({r['row']}, {r['column']}): " + "; ".join(f"`{s}`" for s in dict.fromkeys(refs))]
        side += ["", "## Machine-readable cell recipes", "", "```json",json.dumps(records,indent=2),"```",
                 "", "## Caption recipe", "", "```json",json.dumps(caption_record,indent=2),"```",
                 "", "## Candidates table cell recipes", "", "```json",json.dumps(candidate_records,indent=2),"```",
                 "", "## Candidates table caption recipe", "", "```json",json.dumps(candidate_caption_record,indent=2),"```",
                 "", "## Every printed number", ""]
        side += [f"- {n['table']}: " + (f"Cell ({n['row']}, {n['column']})" if "row" in n else "Caption") +
                 f", `{n['number']}` ({n['op']}): " +
                 "; ".join(f"`{s}`" for s in n["sources"]) for n in numbers]
        side += ["", "## Machine-readable numeric inventory", "", "```json",json.dumps(numbers,indent=2),"```",
                 "", "## Input hashes", ""]
        side += [f"- `{p}`: `{h}`" for p,h in sorted(audit.inputs.items())]
        outputs = [(sidecar, "\n".join(side)+"\n")]
        if write_tex:
            outputs.insert(0, ("main_prediction_v2.tex", tex))
            outputs.insert(1, ("main_prediction_candidates.tex", candidates_tex))
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
