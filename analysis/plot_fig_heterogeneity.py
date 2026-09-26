#!/usr/bin/env python3
"""Original heterogeneous twelve-model panel, frozen V51 values and loaders."""
from __future__ import annotations

if __package__:
    from .paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp
else:
    from paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp

import math

if __package__:
    from .paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot
    from .paper_figure_style import finish_panel, panel_axes
    from .paper_panel_exports import ref, capability_handles, export
    from . import v51_panel_tables as tables
else:
    from paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot
    from paper_figure_style import finish_panel, panel_axes
    from paper_panel_exports import ref, capability_handles, export
    import v51_panel_tables as tables

SOURCE = "results/v51-panel/panel.json"
PANEL_SIZE = (5.5, 1.8)
CAPABILITY_MARKERS = {"math": "o", "code": "s", "qa": "^"}
SHORT_NAMES = {"Gemma-3": "G3", "Gemma-4": "G4", "OLMo-3": "OL3", "Qwen3": "Q3", "Muse": "Muse"}
CAPTION = """Heterogeneity across the original twelve-model panel in one wide panel: the
models run along the x axis, ordered by family/series and increasing size within each series,
with sizes as tick labels and family names beneath. In each model's slot, pruning at density
d=0.7 sits on the left (filled markers) and per-output-channel symmetric RTN int4 on the right
(open markers); Math circles, Code squares and QA triangles are offset by -0.075, 0 and +0.075
slots within each method. A thin grey bar spans each method's three capability loss changes for
that model. Positive values are worse; negative values are improvements. Losses use native-token
nats, so comparisons across tokenizers require caution. Values are read directly from V51
panel.json (dl07 and dl4), the frozen source of panel_prune.tex and panel_quant.tex, and checked
against the original V51 load_prune/load_quant arithmetic. The two prospective Qwen3 additions are
excluded by cohort=panel. Thin grey separators divide series. The loss axis is symmetric log with
a 0.1-nat linear threshold and linscale 0.6; a solid horizontal line marks zero. No seed averaging
or uncertainty intervals. The panel is 5.5 x 1.8 inches at text width; markers are 3.8 pt, filled
with 0.4-pt white edges or open with 0.8-pt coloured edges. The shared key fig8_legend.pdf sits
above the panel.
"""


def model_size(model):
    suffix = model.rsplit("-", 1)[1]
    return float(suffix[:-1]) * (1e-3 if suffix.endswith("M") else 1.)


def build(audit):
    raw = audit.read(SOURCE)
    models = {name: model for model, name, _, cohort in tables.MODELS if cohort == "panel"}
    selected = sorted(((i, r) for i, r in enumerate(raw) if r["cohort"] == "panel"),
                      key=lambda pair: (pair[1]["series"], model_size(pair[1]["model"])))
    if len(selected) != 12 or {r["model"] for _, r in selected} != set(models):
        raise ValueError("Expected the original twelve V51 models")
    rows = []
    for order, (i, r) in enumerate(selected):
        model = models[r["model"]]
        pr, qu = tables.load_prune(model, root=audit.root, read=audit.read), tables.load_quant(model, root=audit.root, read=audit.read)
        if pr is None or qu is None:
            raise ValueError(f"Missing original V51 losses: {model}")
        label = SHORT_NAMES[r["series"]] + " " + r["model"].rsplit("-", 1)[1]
        for panel, field, check in (("a", "dl07", pr[1][.7]), ("b", "dl4", qu[4])):
            for cap in CAPS:
                if not math.isclose(r[field][cap], check[cap], abs_tol=1e-12):
                    raise ValueError(f"V51 saved/loader mismatch: {model}/{field}/{cap}")
                rows.append({"panel": panel, "model": r["model"], "series": r["series"], "order": order,
                             "label": label, "capability": cap, "delta": r[field][cap],
                             "delta_source": ref(SOURCE, i, field, cap),
                             "model_source": ref(SOURCE, i, "model"), "cohort_source": ref(SOURCE, i, "cohort")})
    audit.rule("delta read verbatim from V51 panel.json dl07/dl4. Reused import-safe V51 "
               "load_prune/load_quant with audited reads to verify dense subtraction; no table generator invoked. "
               "Keep cohort=panel; sort series then numeric model size; panel a density=.7, b bit=4.")
    return rows


FAMILY_NAMES = {"Gemma-3": "Gemma 3", "Gemma-4": "Gemma 4", "Muse": "Muse", "OLMo-3": "OLMo 3", "Qwen3": "Qwen3"}
METHOD_OFFSETS = {"a": -.2, "b": .2}          # pruning left, int4 right within each model slot
CAPABILITY_OFFSETS = dict(zip(CAPS, (-.075, 0., .075)))


def draw_panel(fig, rows):
    """One wide panel: the twelve models along the x axis, pruning (filled) and per-channel int4
    (open) side by side in each model slot, loss change on a symmetric log axis."""
    from matplotlib.ticker import NullLocator
    ax = panel_axes(fig, PANEL_SIZE, left=.46, bottom=.36, right=.04, top=.04)
    order = sorted((r for r in rows if r["panel"] == "a" and r["capability"] == "math"), key=lambda r: r["order"])
    for panel, offset in METHOD_OFFSETS.items():
        part = [r for r in rows if r["panel"] == panel]
        for slot in sorted({r["order"] for r in part}):
            values = [r["delta"] for r in part if r["order"] == slot]
            ax.vlines(slot + offset, min(values), max(values), color=PALETTE["dense"], lw=.7, zorder=2)
        for cap in CAPS:
            series = sorted((r for r in part if r["capability"] == cap), key=lambda r: r["order"])
            # Collections keep the exact loss coordinates through export's marker-dodging pass.
            filled = panel == "a"
            ax.scatter([r["order"] + offset + CAPABILITY_OFFSETS[cap] for r in series], [r["delta"] for r in series],
                       marker=CAPABILITY_MARKERS[cap], s=3.8**2, zorder=3,
                       facecolors=COLORS[cap] if filled else PALETTE["white"],
                       edgecolors=PALETTE["white"] if filled else COLORS[cap], linewidths=.4 if filled else .8)
    for a, b in zip(order, order[1:]):
        if a["series"] != b["series"]:
            ax.vlines(a["order"] + .5, -1.25, 5.9, color=PALETTE["grid"], lw=.5, zorder=0)
    ax.set_xticks(range(12), [r["model"].rsplit("-", 1)[1] for r in order])
    ax.tick_params(axis="x", labelsize=7.5, length=0, pad=2)
    for series in dict.fromkeys(r["series"] for r in order):
        slots = [r["order"] for r in order if r["series"] == series]
        ax.text((min(slots) + max(slots)) / 2, -.15, FAMILY_NAMES[series], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.5, fontweight="bold")
    ax.set_yscale("symlog", linthresh=.1, linscale=.6)
    ax.set(xlim=(-.55, 11.55), ylim=(-1.25, 5.9), ylabel="Loss change (nats)")
    ax.set_yticks([-1, -.1, 0, .1, 1, 5], ["\u22121", "\u22120.1", "0", "0.1", "1", "5"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="y", labelsize=7.5)
    ax.hlines(0, -.55, 11.55, color=PALETTE["reference"], lw=.7, zorder=1)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.3, lw=.4)
    return finish_panel(ax)


def draw_legend(fig):
    from matplotlib.lines import Line2D
    if __package__:
        from .paper_figure_style import legend_strip
    else:
        from paper_figure_style import legend_strip
    handles = [Line2D([], [], ls="none", marker=CAPABILITY_MARKERS[c], color=COLORS[c], markeredgecolor=PALETTE["white"],
                      label={"math": "Math", "code": "Code", "qa": "QA"}[c]) for c in CAPS]
    handles += [Line2D([], [], ls="none", marker="o", color=PALETTE["reference"], markeredgecolor=PALETTE["white"],
                       label="Pruning at $d=0.7$"),
                Line2D([], [], ls="none", marker="o", markerfacecolor=PALETTE["white"],
                       markeredgecolor=PALETTE["reference"], label="Per-channel int4")]
    legend = legend_strip(fig, handles)
    for handle in legend.legend_handles:
        handle.set_markeredgewidth(.4 if handle.get_markerfacecolor() != PALETTE["white"] else .8)
    return legend


def generate(root=ROOT):
    with frozen_run(root) as access:
        audit, plt = Artifacts(root), pyplot(root)
        rows = build(audit)
        panels = [("fig8_a", PANEL_SIZE, lambda f: draw_panel(f, rows), rows, CAPTION)]
        export(plt, audit, "heterogeneity", panels, CAPTION, width=5.5,
               legend=("fig8_legend", (5.5, .3),
                       draw_legend, [], CAPTION))
        return rows, audit, access


if __name__ == "__main__":
    generate()
