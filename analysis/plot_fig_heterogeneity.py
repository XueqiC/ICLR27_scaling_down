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
PANEL_SIZES = {"a": (3.0, 1.9), "b": (2.45, 1.9)}
CAPABILITY_MARKERS = {"math": "o", "code": "s", "qa": "^"}
SHORT_NAMES = {"Gemma-3": "G3", "Gemma-4": "G4", "OLMo-3": "OL3", "Qwen3": "Q3", "Muse": "Muse"}
CAPTION = """Heterogeneity across the original twelve-model panel, ordered by
family/series and increasing size within each series, from top to bottom.
Horizontal dot plots show Math,
Code and QA loss changes from each model's own dense loss. Panel (a) is pruning
at density d=0.7; panel (b) is per-output-channel symmetric RTN int4. Positive
values are worse; negative values are improvements. Losses use native-token
nats, so comparisons across tokenizers require caution. Values are read directly
from V51 panel.json (dl07 and dl4), the frozen source of panel_prune.tex and
panel_quant.tex, and checked against the original V51 load_prune/load_quant
arithmetic. The two prospective Qwen3 additions are excluded by cohort=panel.
Full model names appear on the left panel; aligned rows identify the same
models on the right. Thin grey separators divide series and faint guides mark
model rows. Math circles, Code squares and QA triangles use vertical offsets
of -0.18, 0 and +0.18 rows, respectively; horizontal values are unchanged.
A thin grey spread bar on each model row spans the smallest to the largest
of its three capability loss changes.
The loss axis is symmetric log with a 0.1-nat linear threshold and linscale 0.6;
a solid vertical line marks zero. No seed averaging or uncertainty intervals.
The 3.0 x 1.9-inch and 2.45 x 1.9-inch panels share vertical margins and form
one row at 5.5-inch text width. Markers are 3.8 pt with 0.4-pt white edges;
full model labels are horizontal at 7.5 pt.
The shared key fig8_legend.pdf sits above the panels.
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


def draw_panel(fig, rows, panel):
    from matplotlib.ticker import NullLocator
    ax = panel_axes(fig, PANEL_SIZES[panel], left=1.04 if panel == "a" else .08,
                    bottom=.31, right=.06, top=.04)
    part = [r for r in rows if r["panel"] == panel]
    # A thin bar from the smallest to the largest of each model's three responses shows its spread.
    for order in sorted({r["order"] for r in part}):
        values = [r["delta"] for r in part if r["order"] == order]
        ax.hlines(order, min(values), max(values), color=PALETTE["dense"], lw=.7, zorder=2)
    for offset, cap in zip((-.18, 0, .18), CAPS):
        series = sorted((r for r in part if r["capability"] == cap), key=lambda r: r["order"])
        # Collections preserve the exact loss coordinates and explicit row
        # offsets through export's legacy Line2D marker-dodging pass.
        ax.scatter([r["delta"] for r in series], [r["order"] + offset for r in series],
                   marker=CAPABILITY_MARKERS[cap], color=COLORS[cap], s=3.8**2,
                   edgecolors=PALETTE["white"], linewidths=.4, zorder=3)
    order = sorted((r for r in part if r["capability"] == "math"), key=lambda r: r["order"])
    for a, b in zip(order, order[1:]):
        if a["series"] != b["series"]:
            ax.hlines(a["order"]+.5, -1.25, 5.9, color=PALETTE["grid"], lw=.5, zorder=0)
    labels = [r["series"].replace("-", " ") + " " + r["model"].rsplit("-", 1)[1] for r in order]
    ax.set_yticks(range(12), labels)
    ax.tick_params(axis="y", labelsize=7.5, length=0, labelleft=panel == "a")
    ax.set_xscale("symlog", linthresh=.1, linscale=.6)
    ax.set(xlim=(-1.25, 5.9), ylim=(11.55, -.55), xlabel="Loss change (nats)")
    ax.set_xticks([-1, -.1, 0, .1, 1, 5], ["−1", "−0.1", "0", "0.1", "1", "5"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", labelsize=7.5)
    ax.vlines(0, -.55, 11.55, color=PALETTE["reference"], lw=.7, zorder=1)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.3, lw=.4)
    return finish_panel(ax)


def draw_legend(fig):
    if __package__:
        from .paper_figure_style import legend_strip
    else:
        from paper_figure_style import legend_strip
    handles = capability_handles()
    for cap, handle in zip(CAPS, handles):
        handle.set_linestyle("none")
        handle.set_marker(CAPABILITY_MARKERS[cap])
        handle.set_markeredgecolor(PALETTE["white"])
    legend = legend_strip(fig, handles)
    for handle in legend.legend_handles:
        handle.set_markeredgewidth(.4)
    return legend


def generate(root=ROOT):
    with frozen_run(root) as access:
        audit, plt = Artifacts(root), pyplot(root)
        rows = build(audit)
        panels = [(f"fig8_{p}", PANEL_SIZES[p], lambda f, p=p: draw_panel(f, rows, p),
                   [r for r in rows if r["panel"] == p],
                   ("Pruning at density 0.7.\n" if p == "a" else "Per-channel int4.\n") + CAPTION) for p in "ab"]
        export(plt, audit, "heterogeneity", panels, CAPTION, width=5.5,
               legend=("fig8_legend", (5.5, .3),
                       draw_legend, [], CAPTION))
        return rows, audit, access


if __name__ == "__main__":
    generate()
