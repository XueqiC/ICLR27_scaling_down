#!/usr/bin/env python3
"""V86 Figure 2, derived from plot_fig2_confirm.py (CPU/Agg; no fitting).

Run: python3 analysis/plot_fig2_candidates.py
Writes paper/paper/figs/frozen_candidates.{pdf,png} and
results/v86-main-table/fig_notes.md. Frozen inputs are read only.
Uses V86's shared table rows, full alternative sets, and provenance validation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "v86-fig2-mpl"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np

from v86_main_table import ROOT, OUT, CAPS, build_rows, row_record, unchanged, require

# The PDF is already 0.9 of the style's 5.5-inch textwidth, so inclusion does not
# shrink any glyph. Preserve the original 2.4-inch height and 7 pt lower bound.
WIDTH, HEIGHT, FONT = 4.95, 2.4, 7.2
CAP_LABELS = {"math": "Math", "code": "Code", "qa": "QA"}
POSITIVE, NEGATIVE = "#2166ac", "#bd542c"


def close(actual, expected, context):
    require(np.allclose(actual, expected, rtol=1e-11, atol=1e-13),
            f"Score/provenance mismatch: {context}: {actual} != {expected}")


def validate_stored_intervals(inputs):
    """Retain V83's six paired pool-bootstrap checks; these CIs are not replotted.

Stored intervals compare the development-fixed baseline, which may differ from
V86's full-set test minimum. Reusing them for the new bars would be incorrect.
"""
    comp = next(c.data for c in inputs if c.path == "results/v70-distill-confirm/compare.json")
    require(comp["bootstrap"]["n_resamples"] == 5000 and comp["bootstrap"]["seed"] == 0,
            "Unexpected bootstrap protocol")
    draws = np.random.default_rng(0).integers(0, 6, size=(5000, 6))
    for group in comp["groups"]:
        methods = (group["selected"], group["strongest_baseline"])
        errors = np.asarray([[cl["mae"][m] for m in methods] for cl in group["clusters"]])
        close(errors.mean(0), [group["candidate_mae"], group["baseline_mae"]], "Stored cluster mean")
        boot = errors[draws].mean(1)
        close(np.quantile(boot[:, 1]-boot[:, 0], [.025, .975]), group["paired_difference"]["ci95"],
              f"Stored pool-cluster CI/{group['student']}/{group['capability']}")
        lo, hi = group["paired_difference"]["ci95"]
        require(lo <= group["paired_difference"]["estimate"] <= hi, "Stored CI does not bracket estimate")


def style():
    families = ["Liberation Serif", "Nimbus Roman", "Times New Roman"]
    font = font_manager.findfont(font_manager.FontProperties(family=families, weight="bold"))
    plt.rcParams.update({
        "font.family": "serif", "font.serif": families, "font.weight": "bold",
        "font.size": FONT, "axes.titlesize": 8, "axes.titleweight": "bold",
        "axes.labelsize": FONT, "xtick.labelsize": FONT, "ytick.labelsize": FONT,
        "legend.fontsize": FONT, "legend.title_fontsize": FONT,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.linewidth": .5, "xtick.major.size": 2, "xtick.major.width": .5,
        "xtick.major.pad": 2, "ytick.major.pad": 3, "axes.unicode_minus": True,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })
    return font


def draw_panel(fig, index, bars, title, headers, limits, ticks):
    left = .32 + index * 1.65
    ax = fig.add_axes([left/WIDTH, .51/HEIGHT, 1.24/WIDTH, 1.41/HEIGHT])
    ax.set(xlim=limits, ylim=(7.05, -.95), xticks=ticks)
    positions = [.55, 1.55, 2.55, 4.55, 5.55, 6.55]
    ax.set_yticks(positions, [CAP_LABELS[b["cap"]] for b in bars])
    ax.tick_params(axis="y", length=0)
    fig.text((left+.49)/WIDTH, 2.035/HEIGHT, title, fontsize=8, ha="center", va="center")
    for y, header in zip((-.73, 3.28), headers):
        ax.text(-.23, y, header, transform=ax.get_yaxis_transform(), va="center", fontsize=FONT)
    for low, high in ((.25, 2.85), (4.25, 6.85)):
        for tick in ticks:
            ax.vlines(tick, low, high, color="#3e3e3e" if tick == 0 else "#e8e8e8",
                      lw=.7 if tick == 0 else .45, zorder=0)
    annotations, diamonds = [], []
    for y, bar in zip(positions, bars):
        gain = bar["gain"]
        color = POSITIVE if gain >= 0 else NEGATIVE
        ax.barh(y, gain, height=.26, color=color, edgecolor=color, linewidth=.4, zorder=2)
        if bar["delivered_differs"]:
            diamond, = ax.plot(bar["delivered_gain"], y, marker="D", linestyle="none", markersize=4.4,
                               markerfacecolor="none", markeredgecolor="#161616", markeredgewidth=.9, zorder=4)
            diamonds.append((diamond, bar, y))
        annotation = ax.annotate(f"{bar['candidate_mae']:.3f} / {bar['alternative_mae']:.3f}",
                                (gain, y), xytext=(0, 3.0), textcoords="offset points",
                                ha="left" if gain < 0 else "right", va="bottom", fontsize=FONT)
        annotations.append(annotation)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for annotation in annotations:
        box = annotation.get_window_extent(renderer)
        if box.x0 < ax.bbox.x0 or box.x1 > ax.bbox.x1:
            annotation.set_ha("right" if annotation.get_ha() == "left" else "left")
    return ax, annotations, diamonds


def validate_figure(fig, axes, annotations, panels, diamonds, inclusion_scale):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    for text in texts:
        require(text.get_fontsize()*inclusion_scale >= 7, f"Font below 7 pt at 0.9 textwidth: {text.get_text()}")
        box = text.get_window_extent(renderer)
        require(box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1+1 and box.y1 <= fig.bbox.y1+1,
                f"Text outside page: {text.get_text()}")
    for ax, labels, bars, markers in zip(axes, annotations, panels, diamonds):
        require(len(ax.patches) == len(labels) == len(bars) == 6, "Missing bars or MAE labels")
        require(len(markers) == sum(b["delivered_differs"] for b in bars), "Missing/extra delivered diamonds")
        for patch, label, bar in zip(ax.patches, labels, bars):
            close(patch.get_width(), bar["gain"], "Signed bar width")
            require(label.get_text() == f"{bar['candidate_mae']:.3f} / {bar['alternative_mae']:.3f}", "Wrong MAE label")
            box = label.get_window_extent(renderer)
            require(box.x0 >= ax.bbox.x0-1 and box.x1 <= ax.bbox.x1+1, f"MAE label outside panel: {label.get_text()}")
            for value in (0, bar["gain"], bar["delivered_gain"]):
                require(ax.get_xlim()[0] <= value <= ax.get_xlim()[1], "Clipped bar/diamond")
        for marker, bar, y in markers:
            close(marker.get_xdata(), [bar["alternative_mae"]-bar["delivered_mae"]], "Diamond gain")
            close(marker.get_ydata(), [y], "Diamond row")
            require(marker.get_markerfacecolor() == "none", "Delivered diamond must be hollow")
    boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in texts]
    for i, (name, box) in enumerate(boxes):
        for other, other_box in boxes[i+1:]:
            require(not box.overlaps(other_box), f"Overlapping labels: {name!r}, {other!r}")
    close(fig.get_size_inches(), [WIDTH, HEIGHT], "Physical figure size")
    require(HEIGHT <= 2.4, "Figure exceeds original height")
    return min(t.get_fontsize()*inclusion_scale for t in texts)


def write_notes(panels, inputs, font, minimum, scale, summary_sha):
    lines = ["# V86 Figure 2: Frozen candidate evaluation", "",
        "Build: `python3 analysis/plot_fig2_candidates.py` (CPU; NumPy and matplotlib Agg; no fitting).", "",
        f"Outputs: `paper/paper/figs/frozen_candidates.pdf` and `.png`, {WIDTH} × {HEIGHT} inches; "
        f"300 dpi PNG, {round(WIDTH*300)} × {round(HEIGHT*300)} pixels. No bounding-box crop.", "",
        "## Font and geometry check", "",
        f"PASS: minimum visible glyph size **{minimum:g} pt at 0.9 textwidth**. "
        f"The style declares a 5.5-inch text width; 0.9 textwidth = {WIDTH} inches. "
        f"PDF inclusion scale = {scale:g}; minimum source font = {FONT:g} pt. "
        "Figure height remains at the original 2.4-inch maximum. "
        "Checks cover page and panel bounds, text collisions, signed bar lengths, MAE annotations, "
        "all 18 bars, and all 11 hollow diamonds, including their gain arithmetic and positions.",
        f"Font: `{font}`; bold Times-like serif, embedded TrueType (pdf.fonttype=42).", "",
        "## Definitions and timing", "",
        "Bars = MAE(strongest frozen alternative) − MAE(frozen candidate), in nats per token. "
        "Blue is positive; orange is negative. Numbers above endpoints are candidate / alternative MAE "
        "at three decimals. Each arm has its own linear x scale. All negative gains are retained.", "",
        "A hollow diamond appears for every capability where the delivered method differs from the "
        "candidate. Its x value is the SAME alternative MAE minus delivered MAE. "
        "Pruning and quantization delivery was fixed after these tests (retrospective); "
        "C35's median was later reused frozen in Sec. 5. Distillation delivery equals its frozen "
        "candidate and was fixed before test, so it has no diamond.", "",
        "The six groups and every candidate, alternative, MAE, and gain are imported from "
        "`analysis/v86_main_table.py` and checked against the generated summary.json. "
        "Quantization candidate is frozen D (surface/median/zero), not interpolation or delivered median. "
        "Strongest ranks all frozen forms except that candidate AFTER equal-cell pooling. "
        "The new-state quantization strata have 6:3 weights. Distillation considers all 16 recorded forms "
        "before excluding the selected form; six recorded strongest_baseline labels differ from the test minimum.", "",
        "The original six stored 95% pool-cluster bootstrap intervals are independently reproduced "
        "using 5,000 PCG64 seed-0 draws over six pools, retaining three budgets together. They belong "
        "to the development-fixed baselines and are not intervals for these newly ranked alternatives; "
        "no interval is plotted or repurposed. Full prediction identity, hash chains, grid completeness, "
        "all-form error arithmetic, and pooled MAE checks are inherited from the shared V86 builder.", "",
        "## Exact plotted numbers", "",
        "| Test | Capability | Candidate | Alternative | n | Candidate MAE | Alternative MAE | Bar gain | Delivered | Delivered MAE | Diamond gain (if different) |",
        "|---|---|---|---|---:|---:|---:|---:|---|---:|---:|"]
    for bars in panels:
        for b in bars:
            diamond = repr(b["delivered_gain"]) if b["delivered_differs"] else "—"
            lines.append(f"| {b['row']} | {b['cap']} | {b['candidate']} | {b['strongest_alternative']} | {b['n']} | "
                         f"{b['candidate_mae']!r} | {b['alternative_mae']!r} | {b['gain']!r} | {b['delivered']} | "
                         f"{b['delivered_mae']!r} | {diamond} |")
    lines += ["", "## Complete alternative sets and sources", ""]
    for bars in panels:
        for b in bars:
            lines.append(f"- {b['row']} / {b['cap']}: " + "; ".join(f"`{m}`={b['all_method_mae'][m]!r}" for m in b["alternative_set"]))
            for m in dict.fromkeys((b["candidate"], b["strongest_alternative"], b["delivered"])):
                lines.append(f"  - `{m}`: `{b['mae_sources'][m]}`")
    lines += ["", "## Input SHA-256", "", f"- `results/v86-main-table/summary.json`: `{summary_sha}`"]
    lines.extend(f"- `{c.path}`: `{c.sha256}`" for c in inputs)
    (OUT/"fig_notes.md").write_text("\n".join(lines)+"\n")


def main():
    font = style()
    rows, inputs = build_rows()
    validate_stored_intervals(inputs)
    summary_raw = (OUT/"summary.json").read_bytes()
    summary = json.loads(summary_raw)
    records = [row_record(r) for r in rows[:6]]
    require(records == summary["main_rows"], "Figure/table definitions or numbers diverged; regenerate V86 table")
    require(summary["input_sha256"] == {c.path: c.sha256 for c in inputs}, "Table input hashes changed")
    panels = [[dict(row["capabilities"][c], row=row["row"], cap=c, n=row["n_per_capability"])
               for row in records[start:start+2] for c in CAPS] for start in (0, 2, 4)]
    style_text = (ROOT/"paper/paper/iclr2027_conference.sty").read_text()
    textwidth = float(re.search(r"\\textwidth\s+([\d.]+)\s+true\s+in", style_text)[1])
    scale = .9*textwidth/WIDTH
    close(scale, 1, "0.9 textwidth inclusion scale")
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    fig.text(.5, 2.275/HEIGHT, "Frozen candidate evaluation", ha="center", va="center", fontsize=10)
    specifications = [
        ("Pruning", ("C35 · three checkpoints", "C52 · 2.8B repeat"), (-.68, .06), [-.6, -.3, 0]),
        ("Grouped quantization", ("C44 · development states", "C46 · new state"), (-.49, .11), [-.4, -.2, 0]),
        ("Distillation", ("C47 · Gemma-3-270M", "C48 · Gemma-3-1B"), (-.21, .018), [-.2, -.1, 0]),
    ]
    axes, annotations, diamonds = [], [], []
    for i, (bars, spec) in enumerate(zip(panels, specifications)):
        ax, labels, markers = draw_panel(fig, i, bars, *spec)
        axes.append(ax); annotations.append(labels); diamonds.append(markers)
    fig.text(.5, .275/HEIGHT, "Gain in nats/token; endpoint MAEs: candidate / alternative", ha="center", va="center", fontsize=FONT)
    fig.legend(handles=[Patch(facecolor=NEGATIVE, label="Bar: frozen candidate gain"),
                        Line2D([], [], marker="D", linestyle="none", markerfacecolor="none", markeredgecolor="#161616",
                               markersize=4.4, label="Diamond: delivered rule gain")],
               loc="center", bbox_to_anchor=(.5, .11/HEIGHT), frameon=False, ncol=2,
               handlelength=1.1, handletextpad=.5, columnspacing=1.4, borderpad=0)
    minimum = validate_figure(fig, axes, annotations, panels, diamonds, scale)
    require(sum(len(m) for m in diamonds) == 11, "Expected 11 delivered-rule diamonds")
    unchanged(inputs)
    require((OUT/"summary.json").read_bytes() == summary_raw, "Table summary changed during plotting")
    for extension in ("pdf", "png"):
        path = ROOT/f"paper/paper/figs/frozen_candidates.{extension}"
        require(not path.is_symlink(), f"Refusing symlink output: {path}")
        fig.savefig(path, dpi=300)
        print(f"WROTE {path.relative_to(ROOT)}")
    write_notes(panels, inputs, font, minimum, scale, hashlib.sha256(summary_raw).hexdigest())
    plt.close(fig)
    unchanged(inputs)
    print(f"PASS 18 signed bars, 11 hollow diamonds; table values identical; all six stored CI checks; min glyph {minimum:g} pt at 0.9 textwidth.")


if __name__ == "__main__":
    main()
