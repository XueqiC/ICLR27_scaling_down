#!/usr/bin/env python3
"""Main-text paired MAEs from frozen predictions; CPU only, no fits/resampling.

Also regenerates the unchanged per-cell appendix as generalization_cells.
Paired gain intervals are reflected about the *fixed* baseline MAE for display;
they are never labelled as marginal MAE or per-cell prediction intervals.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from statistics import mean

sys.dont_write_bytecode = True

if __package__:
    from . import plot_fig_generalization_cells as cells
    from .paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot, save_figure, write_notes, output_path
else:
    import plot_fig_generalization_cells as cells
    from paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot, save_figure, write_notes, output_path

P53 = "results/v53-prune-dev/register.json"
Q69 = "results/v69-quant-confirm/develop.json"
D70 = "results/v70-distill-confirm/compare.json"
F70 = "results/v70-distill-confirm/freeze.json"
GREEN, GREY = "#24835b", "#777777"
MARKERS = {"math": "o", "code": "s", "qa": "D"}
PANEL_SIZES = {"a": (2.3, 2.0), "b": (1.5, 2.0), "c": (1.7, 2.0)}
KINDS = {"a": "full", "b": "full", "c": "panel"}
LEGEND_SIZE = (5.5, .3)
FIGSIZE = (5.5, 2.32)
if __package__:
    from .paper_figure_style import apply_style, panel_axes, legend_row, save_panel, write_caption, combine_panels
else:
    from paper_figure_style import apply_style, panel_axes, legend_row, save_panel, write_caption, combine_panels
SCOPE_NOTE = ("No frozen response-law predictions on fresh evaluation distributions.\n"
              "Only registered corner contrasts are available below.")


def resolve(audit, reference):
    path, pointer = reference.split("#", 1)
    value = audit.data[path] if path in audit.data else audit.read(path)
    for key in pointer.removeprefix("/").split("/") if pointer else []:
        key = key.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def check_close(actual, expected, context):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"Frozen score mismatch: {context}: {actual} != {expected}")


def gain_whisker(baseline_mae, interval):
    """Translate a stored baseline-minus-relation interval, without resampling."""
    if interval is None:
        return None
    lo, hi = interval
    if not all(math.isfinite(x) for x in (baseline_mae, lo, hi)) or lo > hi:
        raise ValueError("Invalid stored paired gain interval")
    return [baseline_mae - hi, baseline_mae - lo]


def baseline_choices(audit):
    # Same development-only selection as the main prediction table. Alternative
    # relation forms are eligible comparators; the plotted relation is excluded.
    prune = audit.data[P53]
    eligible = [(i, r) for i, r in enumerate(prune["loso_table"])
                if r["subset"] == "all" and r["candidate"] != "power"]
    pruning = {}
    for cap in CAPS:
        i, row = min(eligible, key=lambda ir: ir[1][f"{cap}_mae"])
        pruning[cap] = (row["candidate"], f"{P53}#/loso_table/{i}/{cap}_mae")
    quant = audit.read(Q69)
    quantization = {}
    for cap in CAPS:
        selected = quant["selected"][cap]["candidate"]
        baseline = min((m for m in quant["methods"] if m != selected),
                       key=lambda m: quant["loso"]["scores"][m][cap]["macro_mae"])
        quantization[cap] = (baseline, f"{Q69}#/loso/scores/{baseline}/{cap}/macro_mae")
    return pruning, quantization


def summarize(part):
    """Score both predictors on exactly the same cells, with equal cell weights."""
    first = part[0]
    baseline = first["baseline"]
    if any(r["baseline"] != baseline or r["relation"] != first["relation"] for r in part):
        raise ValueError("Cannot pool different predictor selections")
    relation_mae = mean(abs(r["predicted"] - r["measured"]) for r in part)
    baseline_mae = None if baseline is None else mean(
        abs(r["baseline_prediction"] - r["measured"]) for r in part)
    return {"kind": "mae", "panel": first["panel"], "group": first["group"],
            "stratum": first["stratum"], "capability": first["capability"],
            "status": first["status"], "relation": first["relation"], "baseline": baseline,
            "relation_mae": relation_mae, "baseline_mae": baseline_mae, "n": len(part),
            "below_baseline": None if baseline_mae is None else relation_mae < baseline_mae,
            "paired_gain_interval": None, "whisker": None, "interval_source": None,
            "baseline_note": first.get("baseline_note"), "cells": part}


def build(audit):
    source_cells = cells.build(audit)
    # Keep the cell figure's original disclosure in its own sidecar, and give
    # this encoding its own interval/selection notes.
    audit.notes = []
    pruning, quantization = baseline_choices(audit)
    groups = defaultdict(list)
    result = []
    for original in source_cells:
        if original["measurement_interval"] is not None:
            result.append({"kind": "corner", **original})
            continue
        r = {**original, "baseline": None, "baseline_prediction": None,
             "baseline_source": None, "selection_source": None, "stratum": ""}
        cap, source = r["capability"], r["source"]
        if source.startswith("results/v53-prune-dev/"):
            r["relation"] = "power"
            r["baseline"], r["selection_source"] = pruning[cap]
            r["baseline_source"] = source.rsplit("/", 1)[0] + "/" + r["baseline"]
        elif source.startswith("results/v72-prune-repeat/"):
            r["relation"], r["stratum"] = "power", "V72"
            r["baseline"], r["selection_source"] = pruning[cap]
            r["baseline_source"] = source + "/predictions/" + r["baseline"]
        elif source.startswith("results/v46-p1-newsource/"):
            r["relation"], r["stratum"] = "power", "V46"
            r["baseline_note"] = "Development-selected baseline unavailable in the V46 freeze."
        elif source.startswith("results/v69-quant-confirm/"):
            r["relation"] = r["candidate"]
            r["baseline"], r["selection_source"] = quantization[cap]
            if r["relation"] != audit.data[Q69]["selected"][cap]["candidate"]:
                raise ValueError("V69 development/freeze selection mismatch")
            r["baseline_source"] = source + "/predictions/" + r["baseline"]
        elif source.startswith(D70):
            stored = resolve(audit, source)
            student = stored["student"]
            r["stratum"], r["student"], r["relation"] = student, student, r["candidate"]
            r["selection_source"] = f"{F70}#/strongest_baseline/{student}/{cap}/method"
            r["baseline"] = resolve(audit, r["selection_source"])
            if stored["strongest_baseline"] != r["baseline"] or stored["selected"] != r["relation"]:
                raise ValueError("V70 development/freeze selection mismatch")
            r["baseline_source"] = source + "/predictions/" + r["baseline"]
        elif source.startswith("results/v78-rule-confirm/"):
            r["relation"] = "locked-rule"
            r["baseline_note"] = ("No development-selected prediction baseline stored for these cells; "
                                  "v64-law and policy regret are different comparisons.")
        else:
            raise ValueError(f"Unclassified frozen prediction: {source}")
        if r["baseline_source"] is not None:
            r["baseline_prediction"] = resolve(audit, r["baseline_source"])
        groups[r["panel"], r["group"], r["stratum"], cap].append(r)

    for part in groups.values():
        row = summarize(part)
        if part[0]["source"].startswith(D70):
            student, cap = row["stratum"], row["capability"]
            i, stored = next((i, g) for i, g in enumerate(audit.data[D70]["groups"])
                             if (g["student"], g["capability"]) == (student, cap))
            if (stored["selected"], stored["strongest_baseline"], stored["n_checkpoints"]) != (
                    row["relation"], row["baseline"], row["n"]):
                raise ValueError("V70 interval group/selection mismatch")
            for field, key in (("relation_mae", "candidate_mae"), ("baseline_mae", "baseline_mae")):
                check_close(row[field], stored[key], f"V70/{student}/{cap}/{key}")
            check_close(row["baseline_mae"] - row["relation_mae"],
                        stored["paired_difference"]["estimate"], "V70 paired difference")
            row["paired_gain_interval"] = stored["paired_difference"]["ci95"]
            row["whisker"] = gain_whisker(row["baseline_mae"], row["paired_gain_interval"])
            row["interval_source"] = f"{D70}#/groups/{i}/paired_difference/ci95"
        result.append(row)

    audit.rule("Same prediction cells and row families as generalization_cells. MAE is mean absolute "
               "prediction-minus-measurement error in native-token nats. Paired markers use identical cells "
               "and equal cell weights; n is the cell count per capability, not independent sample size.")
    audit.rule("Baselines use minimum development LOSO MAE excluding the selected relation: V53 "
               "loso_table[subset=all] for V53/V72; V69 loso.scores[*][cap].macro_mae for V69. "
               "V70 uses freeze.strongest_baseline[student][cap].method, never confirmation ranking.")
    audit.rule("V46 has no recorded development-selected baseline; V78 has no such baseline predictions "
               "on its plotted cells. Keep these relation MAEs grey and label baseline unavailable. "
               "Do not choose on test errors, transport the later V53 selection to V46, or call V78's "
               "v64-law a development-selected baseline. Split V46/V72 within the inside-range row to "
               "avoid pairing a subset baseline with a full-row relation MAE.")
    audit.rule("V70 remains split by student within the new-pool budgets row: paired_difference.ci95 "
               "covers one student/capability's 18 checkpoints (6 equally sized pools). Never average "
               "interval endpoints across students. Draw the stored baseline-minus-relation gain interval "
               "[lo, hi] as [baseline MAE - hi, baseline MAE - lo]. This is a translated paired gain "
               "interval at a fixed baseline reference, not a marginal MAE confidence interval.")
    audit.rule("A2 frozen_prediction_error_interval covers its development holdouts, not any of these "
               "V70 or A5/A7 corner cells. No A2 holdout row existed in the cell figure; none is added "
               "and no A2 interval is transplanted. No refitting, resampling, or invented interval.")
    audit.rule("Corner rows are unchanged: signed prediction-minus-measurement on a separate symlog "
               "axis, original capability colours, circles filled iff within the stored band, triangles "
               "for the 4B development student. Whiskers are A5 registered +/-2-noise bands, not CIs. "
               "Primary QA additivity failed to reject. Source panel C (lower group in display panel c) shows only the registered additive "
               "corner predictions (zero); no frozen response-law predictions on fresh distributions exist.")
    audit.rule("V46 .55 is outside its original coarse .6--.9 range; .65 is inside. V72 repeats two "
               "revision labels with identical weights; both records retained, not independent states. "
               "V78 uses each frozen configuration once. V93 descriptors only verify state identity.")
    return result


def row_label(group, stratum):
    labels = {
        "Pruning: density inside range": "Prune in-range",
        "Pruning: density outside range": "Prune V46 out",
        "Quantization: new group size": "Quant group",
        "Distillation: new-pool budgets": "Distill",
        "Pythia: new stages (power)": "New stages",
        "Pythia: new quantization state": "New quant state",
        "Pythia: locked rule on new states": "Locked rule",
    }
    label = labels[group]
    if stratum == "V46" and "inside" in group:
        label = "Prune V46 in"
    elif stratum.startswith("gemma3-"):
        label += " " + {"gemma3-270m": "270M", "gemma3-1b": "1B"}[stratum]
    return label


def draw_maes(ax, rows, panel, limits):
    from matplotlib.ticker import NullFormatter
    order = list(dict.fromkeys((r["group"], r["stratum"]) for r in rows if r["panel"] == panel))
    labels = []
    for y, key in enumerate(order):
        part = [r for r in rows if r["panel"] == panel and (r["group"], r["stratum"]) == key]
        for r in part:
            # In the narrow middle panel, row labels sit above each marker group.
            yy = y + (.16 if panel == "B" else 0) + (CAPS.index(r["capability"]) - 1) * (.16 if panel == "B" else .30)
            marker = MARKERS[r["capability"]]
            color = GREEN if r["below_baseline"] else GREY
            candidate, baseline = r["relation_mae"], r["baseline_mae"]
            if baseline is not None:
                ax.plot([baseline, candidate], [yy, yy], color=".68", zorder=1)
                ax.plot(baseline, yy, marker=marker, mfc="white", mec=GREY, ls="", zorder=3)
            if r["whisker"] is not None:
                lo, hi = r["whisker"]
                ax.hlines(yy, lo, hi, color=color, lw=2, zorder=2)
                ax.vlines([lo, hi], yy-.08, yy+.08, color=color, lw=2, zorder=2)
            ax.plot(candidate, yy, marker=marker, color=color, ls="", zorder=4)
        counts = {r["n"] for r in part}
        if len(counts) != 1:
            raise ValueError("A shared row count requires equal counts per capability")
        labels.append(row_label(*key))
        if y < len(order)-1:
            ax.axhline(y+.5, color=".93", lw=.6, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(*limits)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set(yticks=range(len(order)), yticklabels=labels,
           ylim=(len(order)-.5, -.5), xlabel="MAE (nats)")
    ax.set_xticks([.1, 1])
    ax.tick_params(axis="y", length=0, pad=4)
    ax.grid(axis="x", alpha=.15)
    ax.spines["left"].set_visible(False)


def draw_corners(ax, rows):
    from matplotlib.ticker import NullFormatter
    rows = sorted(rows, key=lambda r: (r["panel"] == "C", r["panel"]))
    groups = list(dict.fromkeys(r["group"] for r in rows))
    for y, group in enumerate(groups):
        part = [r for r in rows if r["group"] == group]
        for k, r in enumerate(part):
            jitter = 0 if len(part) == 1 else -.26 + .52*k/(len(part)-1)
            color = COLORS[r["capability"]]
            dev = r["status"] == "development"
            lo, hi = r["residual_interval"]
            e = r["residual"]
            ax.errorbar(e, y+jitter, xerr=[[e-lo], [hi-e]], fmt="none", color=color, capsize=3, lw=2)
            ax.plot(e, y+jitter, marker="^" if dev else "o", ls="", color=color,
                    mfc=color if r["within"] else "white", alpha=.85)
    ax.axvline(0, color=".5", lw=1.2)
    ax.axhline(1.5, color=".7", lw=.6, zorder=0)
    ax.set_xscale("symlog", linthresh=.03)
    ax.set_xlim(-2.5, 2.5)
    ax.set_xticks([-1, 0, 1], labels=["−1", "0", "1"])
    ax.set_xticks([-.1, .1], minor=True)
    ax.xaxis.set_minor_formatter(NullFormatter())
    labels = {"Distillation: corner budgets (1B)": "1B",
              "4B DEVELOPMENT: corner budgets": "4B dev.",
              "2wiki_new": "2Wiki", "musique": "MuSiQue", "triviaqa": "TriviaQA"}
    ax.set(yticks=range(len(groups)), yticklabels=[labels[g] for g in groups],
           ylim=(len(groups)-.5, -.5), xlabel="Pred. − obs.\n(nats)")
    ax.tick_params(axis="y", length=0, pad=3)
    ax.grid(axis="x", alpha=.15)
    ax.spines["left"].set_visible(False)


def mae_limits(rows):
    maes = [r for r in rows if r["kind"] == "mae"]
    values = [v for r in maes for v in (r["relation_mae"], r["baseline_mae"], *(r["whisker"] or [])) if v is not None]
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("Log MAE axis requires positive values; never add a pseudocount or clip an interval")
    return min(values)/1.4, max(values)*1.4


def draw_panel(fig, rows, panel):
    if panel in "ab":
        ax = panel_axes(fig, PANEL_SIZES[panel], left=1.24 if panel == "a" else .08,
                        bottom=.51, right=.08, top=.04)
        draw_maes(ax, [r for r in rows if r["kind"] == "mae"], panel.upper(), mae_limits(rows))
        if panel == "b":
            from matplotlib.transforms import ScaledTranslation
            for text in ax.get_yticklabels():
                text.set_horizontalalignment("left")
                text.set_verticalalignment("center")
                text.set_transform(ax.get_yaxis_transform() +
                                   ScaledTranslation(0, .14, fig.dpi_scale_trans))
    else:
        ax = panel_axes(fig, PANEL_SIZES[panel], left=.84, bottom=.72, right=.06)
        draw_corners(ax, [r for r in rows if r["kind"] == "corner"])
        ax.xaxis.set_label_coords(.22, -.22)
    return ax


def draw_legend(fig):
    from matplotlib.lines import Line2D
    if __package__:
        from .paper_figure_style import legend_strip
    else:
        from paper_figure_style import legend_strip
    handles = [Line2D([], [], color=COLORS[c], marker=MARKERS[c], ls="",
                      label="QA" if c == "qa" else c.title()) for c in CAPS]
    handles += [Line2D([], [], color=GREY, mfc="white", marker="o", ls="", label="Base"),
                Line2D([], [], color=GREY, marker="o", ls="", label="Rel."),
                Line2D([], [], color=GREEN, marker="o", ls="", label="Lower"),
                Line2D([], [], color=".25", marker="o", ls="", label="1B"),
                Line2D([], [], color=".25", marker="^", ls="", label="4B dev.")]
    legend = legend_strip(fig, handles)
    for text in legend.get_texts()[-2:]:
        text.set_fontsize(13)  # Corner panel retains its original panel style.
    return legend


def plot(rows, plt):
    return combine_panels(plt, [
        ("full", (0, .32, *PANEL_SIZES["a"]), lambda f: draw_panel(f, rows, "a")),
        ("full", (2.3, .32, *PANEL_SIZES["b"]), lambda f: draw_panel(f, rows, "b")),
        ("panel", (3.8, .32, *PANEL_SIZES["c"]), lambda f: draw_panel(f, rows, "c")),
        ("full", (0, 0, *LEGEND_SIZE), draw_legend),
    ], FIGSIZE)


CAPTION_TEXT = """Generalization to new configurations of seen states (a), new sources
or students (b), and registered corner contrasts for both students and fresh
evaluation distributions (c). MAE axes are logarithmic in native-token nats.
Paired markers compare the relation and development-selected baseline on identical
cells with equal cell weights. Math, Code and QA use circles, squares and diamonds
in the MAE panels. Hollow markers denote baselines; filled relation markers are
green only when the relation MAE is lower. Cell counts per capability are in
generalization_mae_pairs.md and the record sidecars. V46 and the Pythia locked-rule
row have no stored development-selected baseline: their unpaired relation markers
remain grey. V46/V72 and the 270M/1B new-pool budgets remain separate.
Paired gain CIs are translated about the fixed baseline MAE: a stored
baseline-minus-relation interval [lo, hi] is drawn at [baseline MAE - hi,
baseline MAE - lo]. These are paired gain intervals, not marginal MAE confidence
intervals; no intervals are averaged across students.
Corner whiskers are registered ±2-noise bands, not CIs. Corners: filled in band /
hollow outside. The combined corner panel uses signed prediction-minus-measurement with a
symmetric-log axis and unchanged limits of -2.5 to 2.5 native-token nats.
Circles denote 1B; triangles denote the 4B development student. Capability colours
apply in (c), and all fresh-distribution contrasts below its thin separator are QA.
Primary QA additivity: failed to reject. Math is size-dependent and code unresolved.
No frozen response-law predictions on fresh evaluation distributions. Only
registered corner contrasts are available on fresh distributions, with additive predictions zero.
No A2 development-holdout interval is transplanted to these confirmation cells.
The three panels are 2.3, 1.5 and 1.7 inches wide and 2 inches high in one
5.5-inch row. Panel (c) stacks 1B, 4B development, 2Wiki, MuSiQue and TriviaQA.
The shared key is fig3_legend.pdf: Base = baseline, Rel. = relation, Lower =
lower relation MAE. Short pruning labels distinguish V72 in-range from V46
in-range and out-of-range. Panel (b)'s row labels sit above their markers.
"""


def format_pairs(rows):
    lines = ["| Panel / row | Capability | Relation | MAE | Development baseline | MAE | Cells |",
             "|---|---|---|---:|---|---:|---:|"]
    for r in sorted((r for r in rows if r["kind"] == "mae"), key=lambda r: (r["panel"], r["group"], r["stratum"], CAPS.index(r["capability"]))):
        label = r["group"] + (f" ({r['stratum']})" if r["stratum"] else "")
        baseline = "unavailable" if r["baseline_mae"] is None else f"{r['baseline_mae']:.8f}"
        lines.append(f"| {r['panel']} / {label} | {r['capability']} | {r['relation']} | "
                     f"{r['relation_mae']:.8f} | {r['baseline'] or 'unavailable'} | {baseline} | {r['n']} |")
    return "\n".join(lines)


def generate(root=ROOT):
    _, _, appendix_access = cells.generate(root)
    with frozen_run(root) as access:
        audit = Artifacts(root)
        plt = pyplot(root)
        rows = build(audit)
        audit.rule("Display mapping: fig3_a = MAE A, fig3_b = MAE B, fig3_c = all corner A+B+C. "
                   "Frozen panel fields, counts and numbers are unchanged; a thin separator precedes fresh distributions.")
        for letter in "abc":
            apply_style(KINDS[letter])
            fig = plt.figure(figsize=PANEL_SIZES[letter])
            draw_panel(fig, rows, letter)
            if letter in "ab":
                panel_rows = [r for r in rows if r["kind"] == "mae" and r["panel"] == letter.upper()]
            else:
                panel_rows = [r for r in rows if r["kind"] == "corner"]
            save_panel(fig, f"fig3_{letter}", KINDS[letter], audit, panel_rows)
            write_caption(f"fig3_{letter}", audit, CAPTION_TEXT)
            plt.close(fig)
        apply_style("full")
        fig = plt.figure(figsize=LEGEND_SIZE)
        draw_legend(fig)
        save_panel(fig, "fig3_legend", "full", audit, [])
        write_caption("fig3_legend", audit, CAPTION_TEXT)
        plt.close(fig)
        for suffix in (".pdf", "_data.json", "_caption.txt", "_sources.md"):
            output_path(root, "figs", "fig3_d" + suffix).unlink(missing_ok=True)
        fig = plot(rows, plt)
        save_figure(fig, "generalization", audit)
        write_notes("generalization", audit, rows)
        write_caption("generalization", audit, CAPTION_TEXT)
        output_path(root, "figs", "generalization_mae_pairs.md").write_text(
            "# Generalization: per-row MAE pairs\n\n"
            "Native-token nats; cell counts are per capability. Baselines are selected on development "
            "evidence only. Unavailable baselines remain unscored.\n\n" + format_pairs(rows) + "\n")
        plt.close(fig)
        access[0].update(appendix_access[0])
        access[1].update(appendix_access[1])
        return rows, audit, access


if __name__ == "__main__":
    rows, _, _ = generate()
    print(format_pairs(rows))
