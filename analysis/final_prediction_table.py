#!/usr/bin/env python3
"""Generate the five-task frozen-prediction table from frozen results; no fitting or oracle ranking.

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
    "power": "Power form",
    "low_order_2d": "Source surface",
    "median": "Median",
    "median_curve": "Source-free median density curve",
    "same_input_interpolation": "Interpolation",
    "zero": "zero change",
    "E": r"$a_c\log(1+E)$",
    "joint": "Joint budget/pool",
}
BASELINES = {
    "A2": "per-density", "median_curve": "development median", "median": "development median",
    "bilinear": "bilinear", "zero": "zero change",
    "T-only": "budget", "E-only": "reuse",
    "surface:L0": "loss surface", "surface:logN": "size surface",
}
RANGES = {
    "prune_new": "three checkpoints outside every fit; includes 6.9B",
    "quant_bit": "160M–1.4B; bit 4; groups 64 and 256",
    "quant_group": "410M at step 143k, 1.4B at step 16k; bits 3 to 5; groups 32 and 512",
    "quant_new": "1.4B at step 112k; bits 3 to 5; groups 32, 128, and 512",
    "pool": "270M, 1B pairs; six pools; 50k–200k tokens",
}
HEADERS = ("Task (test set)", "Frozen candidate", "Candidate error (Math / Code / QA)",
           "Delivered relation", "Delivered error (Math / Code / QA)",
           "Baseline error (Math / Code / QA)")
COLUMN_WIDTHS = (".20", ".17", ".245", ".145", ".12", ".12")
CAPTION = (
    "Frozen prediction tasks by method. Frozen candidates, delivered relations, and baselines "
    "selected within development folds report mean absolute errors in native-token nats "
    "(Math / Code / QA) on the same cells. Brackets give stored paired intervals of baseline "
    "minus candidate. Daggers mark delivered rules fixed after the test. "
    "Distillation entries list the 270M and 1B students in that order."
)


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

    def render(self, audit):
        return render_text(self.plain(audit))

    def plain(self, audit):
        return "".join(p if isinstance(p, str) else evaluate(audit, p) for p in self.parts)

    def record(self, audit):
        return {"parts": self.parts, "context": self.context, "note": self.note,
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
            parts.append(" / ")
        parts.extend(fn(cap))
    return parts


def baseline_parts(audit, fn):
    """Name shared baselines once; otherwise retain capability/student order."""
    entries = [fn(cap) for cap in CAPS]
    names = [evaluate(audit, entry[0]) for entry in entries]
    sources = [ref for entry in entries for ref in entry[0]["sources"]]
    label = names[0] if len(set(names)) == 1 else " / ".join(names)
    parts = [mapped(audit, label, *sources), "\n"]
    for i, entry in enumerate(entries):
        if i:
            parts.append(" / ")
        parts.extend(entry[2:])
    return parts


def composite_label(names):
    """Spell out every capability in a mixed relation, with compact line breaks."""
    labels = {c: "Reuse" if names[c] == "E" else RELATIONS[names[c]] for c in CAPS}
    return ";\n".join(("QA" if c == "qa" else c.capitalize()) + ": " +
                       labels[c][:1].lower() + labels[c][1:] for c in CAPS)


def delivered_cells(audit, names, refs, scores, *, posthoc=False, note=""):
    """Keep rule identity/timing separate; incomplete tasks are omitted below."""
    if len(set(names.values())) == 1:
        label = RELATIONS[names[CAPS[0]]]
    else:
        label = composite_label(names)
    marker = r"$^{\dagger}$" if posthoc else ""
    return [
        cell(mapped(audit, label, *refs), marker, note=note),
        cell(*score_parts(scores), marker, refs=refs, note=note),
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
        return [pointer(S86, "main_rows", i, "delivered_timing"),
                *[pointer(S86, "main_rows", i, "capabilities", c, "delivered") for c in CAPS]]

    result = []
    pbase = {}
    for cap in CAPS:
        eligible = [(i, r) for i, r in enumerate(prune["loso_table"])
                    if r["subset"] == "all" and r["candidate"] != "power"]
        pbase[cap] = min(eligible, key=lambda ir: ir[1][f"{cap}_mae"])
    power = cell(relation(audit, "power", pointer(P53, "candidate_definitions", "power")),
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
        cell("Pruning, new state\n", tested_range(audit, "prune_new", *[pointer(p, "tag") for p in ps]),
             refs=[row.selection_source, pointer(P53, "dev_states"), pointer(ps[0], "densities")],
             note="All three test checkpoints are absent from every development fit. The 6.9B size occurs at other stages in the expanded register."),
        power,
        cell(*score_parts(lambda c: [scored_number(row.scores[c]["power"])])),
        *delivered_cells(audit, dict.fromkeys(CAPS, "median_curve"), delivery_refs("C35"),
                         lambda c: [scored_number(row.scores[c]["median_curve"])], posthoc=True,
                         note="New-state source-free median curve, fixed after test; scores use the same three checkpoints and densities as the frozen candidate."),
        cell(*score_parts(lambda c: [scored_number(row.scores[c][pbase[c][1]["candidate"]])]),
             refs=[pointer(P53, "loso_table")], note="Minimum development LOSO MAE excluding power, per capability: " +
             " / ".join(BASELINES[pbase[c][1]["candidate"]] for c in CAPS) + "; no test ranking.")])

    # V55 predates V69 and is the actual unseen-bit test. V69 bits were all seen.
    bp = "results/v55-quant-group/compare.json"
    entries = comp55["test_sets"]["bit_test"]["mae_table"]
    candidate_index = next(i for i,r in enumerate(entries) if r["candidate"] == "low_order_2d")
    base55 = {c: min(((i,r) for i,r in enumerate(q55["loso_table"]) if r["candidate"] in q55["baselines"]),
                    key=lambda ir: ir[1]["mae"][c]) for c in CAPS}
    def bit_baseline(c):
        _, r = base55[c]
        j = next(j for j,e in enumerate(entries) if e["candidate"] == r["candidate"])
        return [number(bp,"test_sets","bit_test","mae_table",j,"mae",c)]
    bit_scores = score_parts(lambda c: [number(bp,"test_sets","bit_test","mae_table",candidate_index,"mae",c)])
    bit_delivery_note = (
        "The delivered relation for this task is the source surface as tested; its errors "
        "reuse the candidate's exact frozen JSON fields. No post-test dagger applies. "
        "The later delivered rule has no matching stored score for this earlier bit test, "
        "whose configurations entered its development grid. V55 alternatives use different "
        "models and boundary rules and are not substituted for that later rule."
    )
    result.append([
        cell("Quantization, unseen bit-width\n", tested_range(audit, "quant_bit", pointer(Q55,"test_sets","bit_test","states"),
                         pointer(Q55,"test_sets","bit_test","configs")), refs=[pointer(Q55,"precommitted_rule")]),
        cell(relation(audit, "low_order_2d", pointer(Q55,"candidate_definitions","low_order_2d")), refs=[pointer(Q55,"feature_names")]),
        cell(*bit_scores),
        cell(mapped(audit, "Source surface (as tested)", pointer(Q55,"candidate_definitions","low_order_2d")),
             refs=[pointer(Q55,"precommitted_rule"), pointer(R74,"recommendation_rule"),
                   pointer(Q69,"dev_configs"), pointer(Q55,"dev_configs")], note=bit_delivery_note),
        cell(*bit_scores, note=bit_delivery_note),
        cell(*score_parts(bit_baseline), refs=[pointer(Q55,"loso_table")], note="Development minimum over registered baselines mean/median/zero: " +
             " / ".join(BASELINES[base55[c][1]["candidate"]] for c in CAPS) + ".")])
    for key,task in (("C44","Quantization, unseen group size"),("C46","Quantization, new state")):
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
            cell(task, "\n", tested_range(audit, "quant_group" if key=="C44" else "quant_new",
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
                             posthoc=True, note="Post-test rule on the identical frozen cells, never a test-error minimum. New-state errors pool boundary and interior cells equally per cell."),
            cell(*score_parts(lambda c:[quant_score(c, selected_baseline[c])]),
                 refs=[pointer(Q69,"loso","scores")],note="Minimum development LOSO macro MAE excluding the selected candidate, per capability: " +
                 " / ".join(BASELINES[selected_baseline[c]] for c in CAPS) + ".")])

    f70 = "results/v70-distill-confirm/freeze.json"
    groups = audit.data[D70]["groups"]
    students = audit.data[f70]["confirmation_register"]["students"]
    pool_inds = {c: [next(i for i,g in enumerate(groups) if g["student"]==s and g["capability"]==c)
                     for s in students] for c in CAPS}
    def pool_numbers(c, key, intervals=False):
        parts = []
        for i in pool_inds[c]:
            if parts:
                parts.append(",\n" if intervals else ", ")
            parts.append(number(D70,"groups",i,key))
            if intervals:
                parts.extend(paired_interval(audit, D70, "groups", i))
        return parts
    pool_relation = cell(mapped(audit, composite_label({c: audit.data[f70]["selected"][c]["method"] for c in CAPS}),
                                *[pointer(f70,"selected",c,"method") for c in CAPS]),
                         refs=[pointer(f70,"selected"), pointer(f70,"models")],
                         note="The frozen and delivered forms are identical: one-coefficient zero-anchored reuse for Math/Code, joint budget/pool for QA; fixed before test.")
    result.append([
        cell("Distillation, new pool\n", tested_range(audit, "pool", pointer(f70,"confirmation_register","students"),
                         pointer(f70,"confirmation_register","pools"),pointer(D70,"groups",0,"clusters",0,"T_planned")),
             refs=[pointer(f70,"confirmation_register","unused_U_assertion"),
                    pointer(f70,"frozen_at_utc"),pointer("results/v47-p2-register/register.json","v5_confirm","registered_at_utc")]),
        pool_relation,
        cell(*score_parts(lambda c: pool_numbers(c,"candidate_mae", intervals=True)),
             refs=[pointer(f70,"confirmation_register","students"),pointer(f70,"bootstrap")],
             note="Comma-separated scores follow student order 270M, 1B; no student averaging. Brackets are stored 95% paired baseline-minus-candidate intervals, not MAE intervals; each baseline identity is checked by the reused frozen loader."),
        pool_relation,
        cell(*score_parts(lambda c: pool_numbers(c,"candidate_mae")), refs=delivery_refs("C47") + delivery_refs("C48"),
             note="Same candidate, same cells; repeat the stored MAEs without the candidate-versus-baseline intervals."),
        cell(*score_parts(lambda c: pool_numbers(c,"baseline_mae")), refs=[pointer(f70,"baseline_rule"),pointer(f70,"strongest_baseline")],
             note="Development-fixed baseline identities, student order 270M, 1B: " +
             " / ".join(baseline_names(groups[i]["strongest_baseline"] for i in pool_inds[c]) for c in CAPS) + ".")])
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
            complete.append(row)
    return complete


def caption_cell(audit):
    before, students, after = re.split(r"(270M and 1B)", CAPTION)
    return cell(before, mapped(audit, students,
                              pointer("results/v70-distill-confirm/freeze.json","confirmation_register","students")),
                after, refs=[
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
             r"\begin{table*}[!htbp]\normalfont", r"\centering\footnotesize\linespread{0.85}\selectfont",
             r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{0.92}",
             r"\setlength{\abovecaptionskip}{4pt}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}" + columns + "@{}}",
             r"\toprule", " & ".join(HEADERS) + r" \\", r"\midrule"]
    lines += [" & ".join(c.render(audit) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular*}", r"\caption{\footnotesize "+caption_cell(audit).render(audit)+"}",
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
            for match in re.finditer(r"[-+]?\d+(?:\.\d+)?|\b(?:three|six)\b", displayed):
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
                "Rendered layout: one row per frozen prediction task. Cell coordinates match the six printed columns; every cell is populated.",
                "Slash-separated scores follow Math / Code / QA order. Comma-separated distillation pairs follow student order 270M, 1B. Baseline identities and their development selection pointers are recorded in each baseline cell's note/context.",
                "Daggers mark post-test delivery decisions; frozen candidate predictions remain distinct. Delivered scores reuse the same frozen test cells, not test-error winners. For the earlier bit test, the delivered relation is the source surface as tested and repeats the candidate's stored errors without a dagger. The later delivered model has no matching stored score and its development includes those cells.",
                "Only stored V70 paired_difference.ci95 endpoints are printed, after their own candidate MAE. The sign is baseline minus candidate; the baseline identities match the frozen development selection. These are not MAE intervals. Missing intervals stay absent; no resampling or invented intervals.",
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
