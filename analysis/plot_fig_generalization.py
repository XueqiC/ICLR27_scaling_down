#!/usr/bin/env python3
"""Main-text paired MAEs from frozen predictions; CPU only, no fits/resampling.

The legacy per-cell appendix remains independently reproducible.
Paired gain intervals are reflected about the *fixed* baseline MAE for display;
they are never labelled as marginal MAE or per-cell prediction intervals.
"""
from __future__ import annotations

if __package__:
    from .paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp
else:
    from paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp

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
P40 = "results/v40-prune-strength/register.json"
F46 = "results/v46-p1-newsource/predictions_frozen.json"
Q69 = "results/v69-quant-confirm/develop.json"
D70_DEV = "results/v70-distill-confirm/develop.json"
D70 = "results/v70-distill-confirm/compare.json"
F70 = "results/v70-distill-confirm/freeze.json"
GREY = PALETTE["reference"]
MARKERS = {c: "o" for c in CAPS}
PANEL_SIZES = {"a": (5.5, 1.), "b": (5.5, 1.35)}
KINDS = {p: "double" for p in "ab"}
LEGEND_SIZE = (5.5, .3)
FIGSIZE = (5.5, 2.65)
SUBCAPTIONS = {
    "a": "New configurations of model states that entered the fit.",
    "b": "Model states that entered no fit.",
}
ROW_ORDER = (
    ("Pruning: density inside range", "V72"),
    ("Pruning: density inside range", "V46"),
    ("Pruning: density outside range", "V46"),
    ("Quantization: new group size", ""),
    ("Distillation: new-pool budgets", "gemma3-270m"),
    ("Distillation: new-pool budgets", "gemma3-1b"),
    ("Pythia: new stages (power)", ""),
    ("Pythia: new quantization state", ""),
)
SUBROW_OFFSETS = {"math": -.25, "code": 0., "qa": .25}
AXIS_MARGINS = dict(left=2.62, bottom=.32, right=.08, top=.025)
MAE_TICKS = (.03, .1, .3, 1.)
if __package__:
    from .paper_figure_style import apply_style, finish_panel, panel_axes, save_panel as _save_panel, combine_panels
else:
    from paper_figure_style import apply_style, finish_panel, panel_axes, save_panel as _save_panel, combine_panels


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


def fit_membership(audit, cell):
    """Check source states against the development records for that frozen fit.

    Neither the legacy panel/group nor a confirmation test-set label determines
    membership. Distillation's source is the pretrained student, not the
    adapted checkpoint produced by the held-out pool/budget configuration.
    """
    source, cap = cell["source"], cell["capability"]
    if source.startswith("results/v53-prune-dev/"):
        path = source.split("#", 1)[0]
        state_source = f"{path}#/target/tag"
        state = resolve(audit, state_source)
        configuration = float(source.split("/")[-2])
        fit_source = f"{P53}#/dev_states"
        states = resolve(audit, fit_source)
        seen = any(s["tag"] == state for s in states)
        config_seen = any(s["tag"] == state and configuration in s["densities"] for s in states)
    elif source.startswith("results/v72-prune-repeat/"):
        stored = resolve(audit, source)
        state_source = source + "/source"
        state, configuration = stored["source"], stored["density"]
        fit_source = f"{P53}#/dev_states"
        states = resolve(audit, fit_source)
        seen = any(s["tag"] == state for s in states)
        config_seen = any(s["tag"] == state and configuration in s["densities"] for s in states)
    elif source.startswith("results/v46-p1-newsource/"):
        state_source = f"{F46}#/target"
        target = resolve(audit, state_source)
        state = f"pythia-{target['size']}@step{target['step']}"
        configuration = resolve(audit, source)["d"]
        fit_source = f"{P40}#/dev"
        dev = resolve(audit, fit_source)
        states = {f"pythia-{size}@step{step}" for size in dev["sizes"] for step in dev["steps"]}
        recorded = {path.split("/")[-2].replace("--", "@")
                    for path in resolve(audit, f"{F46}#/provenance/dev_file_hashes")
                    if path.endswith("/prune_losses.json")}
        if states != recorded:
            raise ValueError("V46 original fit state manifest mismatch")
        seen = state in states
        config_seen = seen and configuration in dev["seen_densities"]
    elif source.startswith("results/v69-quant-confirm/"):
        stored = resolve(audit, source)
        state_source = source + "/state"
        state, configuration = stored["state"], stored["config"]
        fit_source = f"{Q69}#/dev_rows"
        dev = [r for r in resolve(audit, fit_source) if r["capability"] == cap]
        seen = any(r["state"] == state for r in dev)
        config_seen = any((r["state"], r["config"]) == (state, configuration) for r in dev)
    elif source.startswith(D70):
        stored = resolve(audit, source)
        state_source = source + "/student"
        state, configuration = stored["student"], stored["pool"]
        fit_source = f"{D70_DEV}#/points"
        dev = resolve(audit, fit_source)
        seen = any(r["student"] == state for r in dev)
        config_seen = any((r["student"], r["pool"]) == (state, configuration) for r in dev)
    else:
        raise ValueError(f"No frozen fit membership evidence: {source}")
    if config_seen:
        raise ValueError(f"Generalization cell configuration already entered the fit: {source}")
    return dict(model_state=state, model_state_source=state_source,
                model_state_seen_in_fit=seen, configuration=configuration,
                configuration_seen_in_fit=config_seen, fit_membership_source=fit_source)


def summarize(part):
    """Score both predictors on exactly the same cells, with equal cell weights."""
    first = part[0]
    baseline = first["baseline"]
    if any(r["baseline"] != baseline or r["relation"] != first["relation"] for r in part):
        raise ValueError("Cannot pool different predictor selections")
    if any(r["model_state_seen_in_fit"] != first["model_state_seen_in_fit"] for r in part):
        raise ValueError("Cannot pool seen and unseen model states")
    relation_mae = mean(abs(r["predicted"] - r["measured"]) for r in part)
    baseline_mae = None if baseline is None else mean(
        abs(r["baseline_prediction"] - r["measured"]) for r in part)
    return {"kind": "mae", "panel": first["panel"], "group": first["group"],
            "stratum": first["stratum"], "capability": first["capability"],
            "status": first["status"], "relation": first["relation"], "baseline": baseline,
            "model_state_seen_in_fit": first["model_state_seen_in_fit"],
            "model_states": sorted({r["model_state"] for r in part}),
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
            continue
        if original["group"] == "Pythia: locked rule on new states":
            continue
        r = {**original, "baseline": None, "baseline_prediction": None,
             "baseline_source": None, "selection_source": None, "stratum": ""}
        r.update(fit_membership(audit, original))
        r["panel"] = "A" if r["model_state_seen_in_fit"] else "B"
        cap, source = r["capability"], r["source"]
        if source.startswith("results/v53-prune-dev/"):
            r["relation"] = "power"
            r["baseline"], r["selection_source"] = pruning[cap]
            r["baseline_source"] = source.rsplit("/", 1)[0] + "/" + r["baseline"]
        elif source.startswith("results/v72-prune-repeat/"):
            r["relation"], r["stratum"] = "power", "V72"
            r["previous_row_label"] = "Pruning, unseen densities of development states"
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

    result.sort(key=lambda r: (r["panel"], ROW_ORDER.index((r["group"], r["stratum"])),
                               CAPS.index(r["capability"])))
    audit.rule("Panels classify the source MODEL STATE against the development records of its own "
               "frozen fit: A = state entered the fit, configuration did not; B = state entered no fit. "
               "Cell sidecars record state identity, membership, configuration and evidence pointers. "
               "V53/V72 use V53 dev_states; V46 uses the original V40 dev grid cross-checked with "
               "its frozen dev_file_hashes, not the later V53 fit; V69 uses capability-specific "
               "dev_rows; V70 uses development points' source students and pools.")
    audit.rule("Corrected old label: 'Pruning, unseen densities of development states' was V72's "
               "pythia-2.8b@step16000 and pythia-2.8b@step143000, both excluded from V53 dev_states. "
               "It is now 'Pruning, held-out size inside the density range' in B. No eligible "
               "seen-state pruning-density row exists in these records. V46's inside/outside rows "
               "also move to B; their new-checkpoint labels were correct. V69's group-size row and "
               "both V70 new-pool student rows remain in A; V53's new stages and V69's new state remain in B.")
    audit.rule("Same loss-prediction cells as generalization_cells, excluding locked-rule selection rows and corner contrasts. MAE is mean absolute "
               "prediction-minus-measurement error in native-token nats. Paired markers use identical cells "
               "and equal cell weights; n is the cell count per capability, not independent sample size.")
    audit.rule("Baselines use minimum development LOSO MAE excluding the selected relation: V53 "
               "loso_table[subset=all] for V53/V72; V69 loso.scores[*][cap].macro_mae for V69. "
               "V70 uses freeze.strongest_baseline[student][cap].method, never confirmation ranking.")
    audit.rule("V46 has no recorded development-selected baseline. Keep its relation MAEs "
               "capability-coloured and label baseline unavailable. Do not transport the later V53 "
               "selection to V46. Keep V46/V72 as separate inside-range rows with identical paired cells. "
               "Locked-rule selection rows are excluded from this loss-prediction MAE figure.")
    audit.rule("V70 remains split by student within the new-pool budgets row: paired_difference.ci95 "
               "covers one student/capability's 18 checkpoints (6 equally sized pools). Never average "
               "interval endpoints across students. Draw the stored baseline-minus-relation gain interval "
               "[lo, hi] as [baseline MAE - hi, baseline MAE - lo]. This is a translated paired gain "
               "interval at a fixed baseline reference, not a marginal MAE confidence interval.")
    audit.rule("A2 frozen_prediction_error_interval covers its development holdouts, not any of these "
               "V70 or A5/A7 corner cells. No A2 holdout row existed in the cell figure; none is added "
               "and no A2 interval is transplanted. No refitting, resampling, or invented interval.")
    audit.rule("Corner rows retain their frozen values in the separate corner_contrasts appendix. "
               "V46 .55 is outside its original coarse .6--.9 range; .65 is inside. V72 repeats two "
               "revision labels with identical weights; both records retained, not independent states.")
    audit.rule("Display: Math, Code and QA occupy categorical sub-rows at -0.25, 0 and +0.25 "
               "row units. Intervals retain their exact x coordinates. Hollow markers have "
               "thin white outer edges. A relation diamond is drawn above its baseline circle; "
               "coincidence means abs(relation MAE - baseline MAE) / baseline MAE <= 0.02. "
               "Each record's display fields document this coincidence and its sub-row. "
               "Residual cross-capability glyph collisions use the shared 1.5% horizontal display "
               "offset limit, with minimum total displacement on a 0.25-point grid. "
               "Coincident pairs move together. Marker sidecars retain true MAEs and display offsets; "
               "connecting segment endpoints follow the displayed markers.")
    for panel in "AB":
        part = [r for r in result if r["panel"] == panel]
        order = list(dict.fromkeys((r["group"], r["stratum"]) for r in part))
        for r in part:
            key = r["group"], r["stratum"]
            offset = SUBROW_OFFSETS[r["capability"]]
            gap = relative_pair_gap(r)
            r["display"] = dict(row_label=row_label(*key), row_index=order.index(key),
                                subrow_offset=offset, y=order.index(key)+offset,
                                relative_pair_gap=gap, coincident_within_2_percent=is_coincident(r),
                                top_marker="diamond" if is_coincident(r) else None)
    return result


def relative_pair_gap(row):
    baseline = row["baseline_mae"]
    return None if baseline is None else abs(row["relation_mae"] - baseline) / baseline


def is_coincident(row):
    gap = relative_pair_gap(row)
    return gap is not None and (gap <= .02 or math.isclose(gap, .02, abs_tol=1e-12))


def row_label(group, stratum):
    labels = {
        "Pruning: density inside range": "Pruning, held-out size inside the density range",
        "Pruning: density outside range": "Pruning, new checkpoints outside the range",
        "Quantization: new group size": "Quantization, unseen group sizes",
        "Distillation: new-pool budgets": "Distillation, new pools,",
        "Pythia: new stages (power)": "Pruning, new training stages of a seen size",
        "Pythia: new quantization state": "Quantization, new model state",
    }
    label = labels[group]
    if stratum == "V46" and "inside" in group:
        label = "Pruning, new checkpoints inside the density range"
    elif stratum.startswith("gemma3-"):
        label += " " + {"gemma3-270m": "270M", "gemma3-1b": "1B"}[stratum] + " student"
    return label


def draw_maes(ax, rows, panel, limits):
    from matplotlib.ticker import NullLocator
    order = list(dict.fromkeys((r["group"], r["stratum"]) for r in rows if r["panel"] == panel))
    labels = []
    for y, key in enumerate(order):
        part = [r for r in rows if r["panel"] == panel and (r["group"], r["stratum"]) == key]
        for r in part:
            yy = y + SUBROW_OFFSETS[r["capability"]]
            marker = MARKERS[r["capability"]]
            color = COLORS[r["capability"]]
            candidate, baseline = r["relation_mae"], r["baseline_mae"]
            if baseline is not None:
                segment, = ax.plot([baseline, candidate], [yy, yy], color=color if r["below_baseline"] else GREY, zorder=1)
                segment._mae_segment = (yy, r["capability"])
                point, = ax.plot(baseline, yy, marker=marker, mfc=PALETTE["white"], color=color, mec=color, ls="", zorder=3)
                point._mae_pair = dict(capability=r["capability"], role="baseline",
                                       coincident_within_2_percent=is_coincident(r))
            if r["whisker"] is not None:
                lo, hi = r["whisker"]
                midpoint = (lo + hi) / 2
                ax.errorbar(midpoint, yy, xerr=[[midpoint-lo], [hi-midpoint]],
                            fmt="none", color=color, lw=1.1, capsize=2, zorder=2)
            point, = ax.plot(candidate, yy, marker="D", mfc=PALETTE["transparent"], color=color, ls="", zorder=4)
            point._mae_pair = dict(capability=r["capability"], role="relation",
                                   coincident_within_2_percent=is_coincident(r))
        counts = {r["n"] for r in part}
        if len(counts) != 1:
            raise ValueError("A shared row count requires equal counts per capability")
        labels.append(row_label(*key))
        if y < len(order)-1:
            ax.axhline(y+.5, color=PALETTE["background"], lw=.6, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(*limits)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set(yticks=range(len(order)), yticklabels=labels,
           ylim=(len(order)-.5, -.5), xlabel="Mean absolute error (nats)")
    ax.set_xticks(MAE_TICKS, labels=["0.03", "0.1", "0.3", "1"])
    ax.tick_params(axis="y", length=0, pad=4)
    ax.grid(axis="x", alpha=.15)
    ax.spines["left"].set_visible(False)


def mae_limits(rows):
    maes = [r for r in rows if r["kind"] == "mae"]
    values = [v for r in maes for v in (r["relation_mae"], r["baseline_mae"], *(r["whisker"] or [])) if v is not None]
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("Log MAE axis requires positive values; never add a pseudocount or clip an interval")
    return min(values)/1.4, max(values)*1.4


def draw_panel(fig, rows, panel):
    ax = panel_axes(fig, PANEL_SIZES[panel], **AXIS_MARGINS)
    draw_maes(ax, rows, panel.upper(), mae_limits(rows))
    return finish_panel(ax)


def prepare_mae_markers(fig):
    """Retain shared glyph sizes/audit, explicit sub-rows and diamonds on top.

    The shared pass only separates centres and orders by glyph area. This
    figure separates complete glyphs across capabilities, keeping near-equal
    pairs together and preserving the author's diamond-on-top encoding.
    """
    import matplotlib.patheffects as effects
    if __package__:
        from .paper_figure_style import prepare_figure, _update_marker_position, verify_marker_pixels
    else:
        from paper_figure_style import prepare_figure, _update_marker_position, verify_marker_pixels
    if getattr(fig, "_mae_prepared", False):
        return
    metadata = {(i, line.get_marker(), *line.get_xydata()[0]): line._mae_pair
                for i, ax in enumerate(fig.axes) for line in ax.lines
                if hasattr(line, "_mae_pair")}
    prepare_figure(fig)
    for ax in fig.axes:
        for point in ax.lines:
            if not getattr(point, "_palette_point", False):
                continue
            record = point._marker_record
            record.update(metadata[record["axis"], record["marker"], record["x"], record["y"]])
            _update_marker_position(point, 0.)
            connector = getattr(point, "_dodge_connector", None)
            if connector is not None:
                connector.remove()
                del point._dodge_connector
            point.set_zorder(4 if record["role"] == "relation" else 3)
            point.set_path_effects([
                effects.Stroke(linewidth=2*point.get_markeredgewidth(), foreground=PALETTE["white"]),
                effects.Normal(),
            ])
            record.update(zorder=point.get_zorder(), white_outer_edge=True,
                          subrow_offset=SUBROW_OFFSETS[record["capability"]],
                          fallback_to_true=False)
    separate_capability_markers(fig)
    fig._marker_pixel_audit = verify_marker_pixels(fig)
    fig._mae_prepared = True


def separate_capability_markers(fig):
    """Resolve the few residual glyph-edge collisions at the actual export DPI.

    Search bounded horizontal offsets, minimizing their total absolute size.
    Include white edges in collision checks; never separate a coincident pair.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import RendererAgg
    if __package__:
        from .paper_figure_style import MAX_MARKER_DISPLACEMENT, _update_marker_position
    else:
        from paper_figure_style import MAX_MARKER_DISPLACEMENT, _update_marker_position
    width, height = map(int, fig.bbox.size)
    renderer = RendererAgg(width, height, fig.dpi)

    def footprint(points, dx):
        pixels = set()
        for point in points:
            _update_marker_position(point, dx)
            if max(point._marker_record["data_displacement_fraction"]) > MAX_MARKER_DISPLACEMENT + 1e-12:
                return None
            renderer.clear()
            point.draw(renderer)
            alpha = np.asarray(renderer.buffer_rgba())[:, :, 3]
            pixels.update(np.flatnonzero(alpha > .2*255).tolist())
        return pixels

    for ax in fig.axes:
        groups = defaultdict(lambda: defaultdict(list))
        for point in ax.lines:
            if getattr(point, "_palette_point", False):
                r = point._marker_record
                key = r["capability"], "pair" if r["coincident_within_2_percent"] else r["role"]
                groups[round(r["y"])][key].append(point)
        bound = MAX_MARKER_DISPLACEMENT * ax.bbox.width * 72 / fig.dpi
        offsets = [0.] + [sign*k*.25 for k in range(1, int(bound/.25)+1) for sign in (-1, 1)]
        for row, markers in groups.items():
            keys = list(markers)
            choices = {key: [(0., footprint(points, 0.))] for key, points in markers.items()}
            zero_collision = any(a[0] != b[0] and choices[a][0][1] & choices[b][0][1]
                                 for i, a in enumerate(keys) for b in keys[:i])
            if not zero_collision:
                continue
            for key, points in markers.items():
                for dx in offsets[1:]:
                    pixels = footprint(points, dx)
                    if pixels is not None:
                        choices[key].append((dx, pixels))
            best = None

            def search(chosen, cost):
                nonlocal best
                if best is not None and cost >= best[0]:
                    return
                if len(chosen) == len(keys):
                    best = cost, chosen
                    return
                key = keys[len(chosen)]
                for candidate in choices[key]:
                    if any(key[0] != previous[0] and candidate[1] & value[1]
                           for previous, value in zip(keys, chosen)):
                        continue
                    search(chosen + [candidate], cost + abs(candidate[0]))

            search([], 0.)
            if best is None:
                raise ValueError(f"Capability glyphs cannot be separated within the shared display bound: row {row}")
            for key, (dx, _) in zip(keys, best[1]):
                for point in markers[key]:
                    _update_marker_position(point, dx)
        endpoints = {(p._marker_record["y"], p._marker_record["capability"], p._marker_record["role"]):
                     p._marker_record["drawn_coordinate"][0]
                     for p in ax.lines if getattr(p, "_palette_point", False)}
        for line in ax.lines:
            if hasattr(line, "_mae_segment"):
                y, cap = line._mae_segment
                line.set_xdata([endpoints[y, cap, role] for role in ("baseline", "relation")])


def save_panel(fig, stem, kind, audit, records):
    """Use the shared exporter, replacing its generic dodge policy in sidecars."""
    if __package__:
        from .paper_figure_style import DODGE_CAPTION
    else:
        from paper_figure_style import DODGE_CAPTION
    prepare_mae_markers(fig)
    _save_panel(fig, stem, kind, audit, records)
    audit.notes[:] = [note for note in audit.notes if DODGE_CAPTION not in note]
    write_notes(stem, audit, records)


def write_caption(stem, audit, text):
    # The categorical sub-row policy is already explained in this caption.
    output_path(audit.root, "figs", f"{stem}_caption.txt").write_text(text.strip() + "\n")


def draw_legend(fig):
    from matplotlib.lines import Line2D
    if __package__:
        from .paper_figure_style import legend_strip
    else:
        from paper_figure_style import legend_strip
    handles = [Line2D([], [], color=COLORS[c], marker=MARKERS[c], ls="",
                      label="QA" if c == "qa" else c.title()) for c in CAPS]
    handles += [Line2D([], [], color=GREY, mfc=PALETTE["white"], marker="o", ls="", label="Baseline"),
                Line2D([], [], color=GREY, marker="D", mfc=PALETTE["transparent"], ls="", label="Relation"),
                Line2D([], [], color=GREY, ls="-", label="Lower of pair")]
    return legend_strip(fig, handles)


def plot(rows, plt):
    fig = combine_panels(plt, [
        (KINDS["a"], (0, PANEL_SIZES["b"][1], *PANEL_SIZES["a"]), lambda f: draw_panel(f, rows, "a")),
        (KINDS["b"], (0, 0, *PANEL_SIZES["b"]), lambda f: draw_panel(f, rows, "b")),
        ("legend", (0, PANEL_SIZES["a"][1]+PANEL_SIZES["b"][1], *LEGEND_SIZE), draw_legend),
    ], FIGSIZE)
    prepare_mae_markers(fig)
    return fig


CAPTION_TEXT = """New configurations of model states that entered the fit (a), and
model states that entered no fit (b). Membership is checked against the development
records of the fit that produced each frozen prediction, not a later fit or a row label.
Panel a contains unseen quantization group sizes and the two distillation new-pool
student rows. Panel b contains pruning at a held-out size, new checkpoints inside
and outside the density range, new training stages of a seen size, and a new
quantization model state. MAE axes are logarithmic in native-token nats and share
one range. Paired markers compare relation and development-selected baseline
on identical cells with equal cell weights. Math, Code and QA retain their
capability hues. A hollow circle denotes the development-selected baseline;
a hollow diamond denotes the frozen relation. The connecting segment uses the capability colour when
the relation MAE is lower, and reference grey otherwise. Cell counts per capability
are in generalization_mae_pairs.md and record sidecars.
Pruning, new checkpoints inside the density range and Pruning, new checkpoints
outside the range are frozen new-state pruning predictions. These rows have no stored
development-selected baseline and retain unpaired capability-coloured markers.
The former label 'Pruning, unseen densities of development states' contradicted the
records: both V72 Pythia-2.8B targets were excluded from the V53 fit. The corrected
row is 'Pruning, held-out size inside the density range' in panel b; no seen-state
pruning-density row remains. Its two revision labels have identical measured
weights and remain repeated records, not independent source states. Distillation
classifies the pretrained source student; new pools and budgets are configurations.
V46 is checked against its original V40 fit, even though its target entered V53 later.
Within each row, Math, Code and QA occupy sub-rows at -0.25, 0 and +0.25 row units;
each connecting segment and interval follows its capability's sub-row. Markers
retain thin white outer edges. Small horizontal display offsets, bounded by the
shared 1.5% limit, separate any remaining glyph-edge collisions across capabilities;
true MAEs and offsets are recorded in the sidecars. Connecting segments follow
the displayed markers. When a baseline and relation
coincide within 2 percent of the baseline MAE, the diamond is drawn on top and the
coincidence is recorded in the sidecar.
Paired gain CIs are translated about the fixed baseline
MAE: [lo, hi] for baseline-minus-relation is drawn at
[baseline MAE - hi, baseline MAE - lo]. These are paired gain intervals, not
marginal MAE confidence intervals; no intervals are averaged across students.
No A2 development-holdout interval is transplanted to confirmation cells.
Two 5.5-inch-wide panels are stacked with matching left margins: fig3_a is
1.0 inches high (three rows), and fig3_b is 1.35 inches high (five rows).
The 5.5 x 0.3-inch fig3_legend.pdf strip sits above them:
Baseline, Relation and Lower of pair distinguish the paired MAEs. The locked-rule
selection row is excluded. Corner second differences and fresh-distribution
contrasts are shown separately in corner_contrasts_a.pdf.
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
    with frozen_run(root) as access:
        audit = Artifacts(root)
        plt = pyplot(root)
        rows = build(audit)
        audit.rule("Display mapping: fig3_a = new configurations of fitted model states (MAE A), "
                   "fig3_b = model states absent from their frozen fits (MAE B). "
                   "Locked-rule selection rows and corner contrasts are excluded from MAEs.")
        for letter in "ab":
            apply_style(KINDS[letter])
            fig = plt.figure(figsize=PANEL_SIZES[letter])
            draw_panel(fig, rows, letter)
            panel_rows = [r for r in rows if r["panel"] == letter.upper()]
            save_panel(fig, f"fig3_{letter}", KINDS[letter], audit, panel_rows)
            write_caption(f"fig3_{letter}", audit, SUBCAPTIONS[letter] + "\n\n" + CAPTION_TEXT)
            plt.close(fig)
        apply_style("legend")
        fig = plt.figure(figsize=LEGEND_SIZE)
        draw_legend(fig)
        save_panel(fig, "fig3_legend", "legend", audit, [])
        write_caption("fig3_legend", audit, CAPTION_TEXT)
        plt.close(fig)
        if __package__:
            from .paper_figure_archive import archive_panels
        else:
            from paper_figure_archive import archive_panels
        archive_panels(root, ("fig3_c", "fig3_d"))
        fig = plot(rows, plt)
        save_figure(fig, "generalization", audit)
        write_notes("generalization", audit, rows)
        write_caption("generalization", audit, CAPTION_TEXT)
        output_path(root, "figs", "generalization_mae_pairs.md").write_text(
            "# Generalization: per-row MAE pairs\n\n"
            "Native-token nats; cell counts are per capability. Baselines are selected on development "
            "evidence only. Unavailable baselines remain unscored.\n\n" + format_pairs(rows) + "\n")
        plt.close(fig)
        return rows, audit, access


if __name__ == "__main__":
    rows, _, _ = generate()
    print(format_pairs(rows))
    for letter in "ab":
        width, height = PANEL_SIZES[letter]
        print(f"fig3_{letter}: {width:g} x {height:g} in")
        for label in dict.fromkeys(r["display"]["row_label"] for r in rows if r["panel"] == letter.upper()):
            print(f"  {label}")
    print(f"fig3_legend: {LEGEND_SIZE[0]:g} x {LEGEND_SIZE[1]:g} in")
