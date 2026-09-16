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

# Presentation vocabulary only. Coefficient counts and scores still come from
# JSON fields; each mapped phrase carries the source values in its recipe.
RELATIONS = {
    "power": r"$A_c(\mathbf x)\,((1-d)/0.3)^{\gamma_c}$",
    "low_order_2d": "per-bit source regression",
    "median": r"piecewise interpolation over the $(b,g)$ grid",
    "zero": "zero change",
    "E": r"$a_c\log(1+E)$",
    "joint": r"$u(a+bu+qv)$",
}
BASELINES = {
    "A2": "per-density", "median_curve": "development median", "median": "development median",
    "bilinear": "bilinear", "zero": "zero change",
    "T-only": "budget", "E-only": "reuse",
    "surface:L0": "loss surface", "surface:logN": "size surface",
}
RANGES = {
    "prune_new": "new stages; 6.9B outside size range",
    "quant_bit": "160M–1.4B; bit 4; groups 64, 256",
    "quant_group": "410M, 1.4B; bits 3–5; groups 32, 512",
    "quant_new": "1.4B, step 112000; groups 32, 128, 512",
    "pool": "270M, 1B pairs; six pools; 50000–200000 tokens",
}
HEADERS = ("Task", "Relation (parameters)", "Tested range", "Error (nats)", "Baseline (nats)")
COLUMN_WIDTHS = (".19", ".29", ".17", ".12", ".23")
CAPTION = (
    "Frozen predictions by method and task: relation with parameter count, tested range, "
    "mean absolute error in native-token nats on the training probes (Math / Code / QA) "
    "against the baseline selected inside the development folds. "
    "All rows are predictions frozen before measurement."
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

    result = []
    pbase = {}
    for cap in CAPS:
        eligible = [(i, r) for i, r in enumerate(prune["loso_table"])
                    if r["subset"] == "all" and r["candidate"] != "power"]
        pbase[cap] = min(eligible, key=lambda ir: ir[1][f"{cap}_mae"])
    power = cell(relation(audit, "power", pointer(P53, "candidate_definitions", "power")),
                 " (", value(pointer(P53, "n_params_per_capability", "power")), ")",
                 refs=[pointer(P53, "feature_names")])
    original_panel = "results/v40-prune-strength/register.json"
    audit.read(original_panel)
    audit.omit("Development-only density, budget and data-reuse tasks are outside this frozen-prediction table.")
    audit.rule("V72 repeats use the same weights; no displayed score uses those repeats.")
    row = rows["C35"]
    ps = [p for p in audit.data if p.startswith("results/v53-prune-dev/compare_")]
    result.append([
        cell("Pruning,\nnew source state", refs=[row.selection_source]), power,
        cell(tested_range(audit, "prune_new", *[pointer(p, "tag") for p in ps], pointer(original_panel, "dev", "sizes")),
             refs=[pointer(ps[0], "densities")],
             note="Outside the size range of the original nine-state panel; V53's expanded register itself includes 6.9B."),
        cell(*score_parts(lambda c: [scored_number(row.scores[c]["power"])])),
        cell(*baseline_parts(audit, lambda c: [baseline(audit, pbase[c][1]["candidate"], pointer(P53, "loso_table", pbase[c][0], "candidate")),
                                  " ", scored_number(row.scores[c][pbase[c][1]["candidate"]])]),
             refs=[pointer(P53, "loso_table")], note="Minimum development LOSO MAE excluding power, per capability; no test ranking.")])

    # V55 predates V69 and is the actual unseen-bit test. V69 bits were all seen.
    bp = "results/v55-quant-group/compare.json"
    entries = comp55["test_sets"]["bit_test"]["mae_table"]
    candidate_index = next(i for i,r in enumerate(entries) if r["candidate"] == "low_order_2d")
    base55 = {c: min(((i,r) for i,r in enumerate(q55["loso_table"]) if r["candidate"] in q55["baselines"]),
                    key=lambda ir: ir[1]["mae"][c]) for c in CAPS}
    def bit_baseline(c):
        i, r = base55[c]
        j = next(j for j,e in enumerate(entries) if e["candidate"] == r["candidate"])
        return [baseline(audit, r["candidate"], pointer(Q55,"loso_table",i,"candidate")), " ", number(bp,"test_sets","bit_test","mae_table",j,"mae",c)]
    result.append([
        cell("Quantization,\nunseen bit-width", refs=[pointer(Q55,"test_sets","bit_test"), pointer(Q55,"precommitted_rule")]),
        cell(relation(audit, "low_order_2d", pointer(Q55,"candidate_definitions","low_order_2d")), " (", value(pointer(Q55,"n_params_per_capability","low_order_2d")), ")",
             refs=[pointer(Q55,"feature_names")]),
        cell(tested_range(audit, "quant_bit", pointer(Q55,"test_sets","bit_test","states"),
                         pointer(Q55,"test_sets","bit_test","configs"))),
        cell(*score_parts(lambda c:[number(bp,"test_sets","bit_test","mae_table",candidate_index,"mae",c)])),
        cell(*baseline_parts(audit, bit_baseline), refs=[pointer(Q55,"loso_table")], note="Development minimum over registered baselines mean/median/zero.")])
    for key,task in (("C44","Quantization, unseen group size"),("C46","Quantization, new state")):
        row = rows[key]
        selected_baseline = {c: min((m for m in quant["methods"] if m != row.candidates[c]),
                           key=lambda m:quant["loso"]["scores"][m][c]["macro_mae"]) for c in CAPS}
        qp = "results/v69-quant-confirm/compare.json"
        rr = [(i,r) for i,r in enumerate(audit.data[qp]["rows"])
              if (r["test_set"] == "development_state_boundary") == (key=="C44")]
        result.append([
            cell(task.replace(", ", ",\n", 1), refs=[pointer("results/v69-quant-confirm/freeze.json","frozen_at_utc"),
                       *[pointer(qp,"rows",i,"test_set") for i,_ in rr]]),
            cell(*cap_lines(lambda c:[relation(audit, quant["selected"][c]["candidate"], pointer(Q69,"selected",c,"candidate")), " (",
                                      value(pointer(Q69,"selected",c,"n_coefficients")), ")"], separator="; ", label_separator=": "),
                 refs=[pointer(Q55,"candidate_definitions","low_order_2d"),pointer(Q69,"models"),
                       pointer(Q69,"feature_names"),pointer(Q69,"dev_configs")],
                 note="Frozen candidates unchanged: the source regression has 20 coefficients; Code interpolates nine per-configuration medians; QA is identically zero. This is not the retrospective same-input interpolation selection."),
            cell(tested_range(audit, "quant_group" if key=="C44" else "quant_new",
                             *[pointer(qp,"rows",i,k) for i,_ in rr for k in ("state","config")])),
            cell(*score_parts(lambda c:[scored_number(row.scores[c][row.candidates[c]])])),
            cell(*baseline_parts(audit, lambda c:[baseline(audit, selected_baseline[c], pointer(Q69,"loso","scores",selected_baseline[c],c)), " ",scored_number(row.scores[c][selected_baseline[c]])]),
                 refs=[pointer(Q69,"loso","scores")],note="Minimum development LOSO macro MAE excluding the selected candidate, per capability.")])

    f70 = "results/v70-distill-confirm/freeze.json"
    groups = audit.data[D70]["groups"]
    students = audit.data[f70]["confirmation_register"]["students"]
    pool_inds = {c: [next(i for i,g in enumerate(groups) if g["student"]==s and g["capability"]==c)
                     for s in students] for c in CAPS}
    def pool_numbers(c, key):
        return sum(([number(D70,"groups",i,key), ","] for i in pool_inds[c]), [])[:-1]
    def pool_baseline(c):
        names = [groups[i]["strongest_baseline"] for i in pool_inds[c]]
        label = baseline_names(names)
        return [mapped(audit,label,*[pointer(D70,"groups",i,"strongest_baseline") for i in pool_inds[c]]),
                " ",*pool_numbers(c,"baseline_mae")]
    result.append([
        cell("Distillation,\nnew pool",refs=[pointer(f70,"confirmation_register","unused_U_assertion"),
                    pointer(f70,"frozen_at_utc"),pointer("results/v47-p2-register/register.json","v5_confirm","registered_at_utc")]),
        cell("Math, Code: ", relation(audit, "E", *[pointer(f70,"selected",c,"method") for c in ("math","code")]),
             " (", value(*[pointer(f70,"selected",c,"n_params") for c in ("math","code")], op="shared"), "); QA: ",
             relation(audit, "joint", pointer(f70,"selected","qa","method")),
             " (", value(pointer(f70,"selected","qa","n_params")), ")",
             refs=[pointer(f70,"models"),pointer(f70,"reference_rule"),pointer(f70,"prediction_rule")],
             note="E and joint are the frozen V50/V70 design identifiers. E has one coefficient; joint has three coefficients on u, u*u, u*log(D_U/D_ref). T_star is null, not fitted, for these selected forms."),
        cell(tested_range(audit, "pool", pointer(f70,"confirmation_register","students"),
                         pointer(f70,"confirmation_register","pools"),pointer(D70,"groups",0,"clusters",0,"T_planned"))),
        cell(*score_parts(lambda c: pool_numbers(c,"candidate_mae")),
             refs=[pointer(f70,"confirmation_register","students")],
             note="Comma-separated scores follow student order 270M, 1B; no student averaging."),
        cell(*baseline_parts(audit,pool_baseline),refs=[pointer(f70,"baseline_rule"),pointer(f70,"strongest_baseline"),
                        *[pointer(D70,"groups",i,"paired_difference") for i in range(len(groups))]],
             note="Only MAEs are printed, paired in student order 270M, 1B. Comma-separated names follow the same student order; a shared name is printed once. Loss-surface and size-surface labels distinguish the stored initial-loss (surface:L0) and size (surface:logN) variants, without changing either selection. Stored paired_difference.ci95 estimates baseline-minus-candidate gain, not an MAE interval; it is not displayed in the MAE columns.")])
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


def caption_cell():
    return cell(CAPTION, refs=[
        pointer(P53,"feature_names"), pointer(Q55,"feature_names"),
        pointer("results/v70-distill-confirm/freeze.json","reference_rule"),
        pointer(P53,"candidate_definitions","power"), pointer(Q55,"candidate_definitions","low_order_2d"),
        pointer("results/v70-distill-confirm/freeze.json","models")])


def render_table(rows, audit, *, sidecar="main_prediction_v2_sources.md"):
    # With empty outer padding, the four column gaps consume eight tabcolseps.
    columns = "".join(
        r">{\raggedright\arraybackslash\hspace{0pt}}p{\dimexpr " + width +
        r"\textwidth-" + f"{8 * float(width):.2f}" + r"\tabcolsep\relax}"
        for width in COLUMN_WIDTHS)
    lines = [f"% Generated from frozen JSON; see {sidecar}.",
             r"\begin{table*}[!htbp]\normalfont", r"\centering\footnotesize\linespread{0.9}\selectfont",
             r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{0.92}",
             r"\setlength{\abovecaptionskip}{4pt}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}" + columns + "@{}}",
             r"\toprule", " & ".join(HEADERS) + r" \\", r"\midrule"]
    lines += [" & ".join(c.render(audit) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular*}", r"\caption{\footnotesize "+caption_cell().render(audit)+"}",
              r"\label{tab:main-prediction-v2}", r"\end{table*}"]
    return "\n".join(lines)+"\n"


def generate(root=ROOT, *, write_tex=True):
    with frozen_run(root) as access:
        audit = Artifacts(root)
        rows = build(audit)
        caption = caption_cell()
        sidecar = "main_prediction_v2_sources.md" if write_tex else "main_prediction_v2_preview_sources.md"
        tex = render_table(rows, audit, sidecar=sidecar)
        records = [{"row":i,"column":j,**c.record(audit)} for i,row in enumerate(rows) for j,c in enumerate(row)]
        side = ["# Main prediction table: cell sources", "", "Columns: task, relation (parameters), tested range, error (nats), development baseline (nats).",
                "Rendered layout: one row per complete frozen prediction task; status is shared in the caption. Cell coordinates match the five printed columns.",
                "Baseline entries follow math, code and QA order. Slash-separated scores and baseline names follow Math / Code / QA order; shared names appear once. Comma-separated pool scores and differing names follow student order 270M, 1B. Per-density denotes per-density regression; budget and reuse denote the corresponding one-input baselines; loss surface and size surface distinguish initial-loss and student-size response surfaces.",
                "All indices are zero-based JSON pointers. `mean` is equal-weight arithmetic only; no refits or resampling.",
                "Numeric values are formatted directly from the following executable source recipes. Context pointers justify textual labels and freeze identities; incomplete tasks are omitted.",
                "The `label` operation uses the generator's explicit presentation mappings; `expected` preserves the source values and must match before rendering `label`. Counts always use their own JSON fields.",
                "The `shared` operation prints a common value only after checking that all grouped source values are identical.",
                "", "## Loader reuse and limits", "", *audit.notes, "", "## Fields per cell", ""]
        for r in records:
            refs = r["context"] + [s for p in r["parts"] if isinstance(p,dict) for s in p["sources"]]
            side += [f"- Cell ({r['row']}, {r['column']}): " + "; ".join(f"`{s}`" for s in dict.fromkeys(refs))]
        side += ["", "## Machine-readable cell recipes", "", "```json",json.dumps(records,indent=2),"```",
                 "", "## Caption recipe", "", "```json",json.dumps(caption.record(audit),indent=2),"```",
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
