#!/usr/bin/env python3
"""Original heterogeneous twelve-model panel, frozen V51 values and loaders."""
from __future__ import annotations

import math

if __package__:
    from .paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot
    from .paper_figure_style import panel_axes
    from .paper_panel_exports import ref, capability_handles, axes_defaults, export
    from . import v51_panel_tables as tables
else:
    from paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot
    from paper_figure_style import panel_axes
    from paper_panel_exports import ref, capability_handles, axes_defaults, export
    import v51_panel_tables as tables

SOURCE = "results/v51-panel/panel.json"
PANEL_SIZE = (5.5, 2.7)
SHORT_NAMES = {"Gemma-3": "G3", "Gemma-4": "G4", "OLMo-3": "OL3", "Qwen3": "Q3", "Muse": "Muse"}
CAPTION = """Heterogeneity across the original twelve-model panel, ordered by
family/series and increasing size within each series. Grouped bars show Math,
Code and QA loss changes from each model's own dense loss. Panel (a) is pruning
at density d=0.7; panel (b) is per-output-channel symmetric RTN int4. Positive
values are worse; negative values are improvements. Losses use native-token
nats, so comparisons across tokenizers require caution. Values are read directly
from V51 panel.json (dl07 and dl4), the frozen source of panel_prune.tex and
panel_quant.tex, and checked against the original V51 load_prune/load_quant
arithmetic. The two prospective Qwen3 additions are excluded by cohort=panel.
Tick abbreviations: G3 = Gemma-3, G4 = Gemma-4, OL3 = OLMo-3, Q3 = Qwen3;
Muse is written in full. The complete display names and JSON pointers are in
the per-panel data sidecars. No seed averaging or uncertainty intervals.
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
        label = SHORT_NAMES[r["series"]] + "-" + r["model"].rsplit("-", 1)[1]
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
    import numpy as np
    ax = panel_axes(fig, PANEL_SIZE, left=.58, bottom=1.16, right=.05, top=.06)
    part = [r for r in rows if r["panel"] == panel]
    for offset, cap in zip((-.26, 0, .26), CAPS):
        series = sorted((r for r in part if r["capability"] == cap), key=lambda r: r["order"])
        ax.bar(np.array([r["order"] for r in series]) + offset, [r["delta"] for r in series],
               width=.24, color=COLORS[cap], linewidth=.6, edgecolor="white")
    order = sorted((r for r in part if r["capability"] == "math"), key=lambda r: r["order"])
    for a, b in zip(order, order[1:]):
        if a["series"] != b["series"]:
            ax.axvline(a["order"]+.5, color=".8", lw=1, zorder=0)
    ax.set_xticks(range(12), [r["label"] for r in order], rotation=65, ha="right", rotation_mode="anchor")
    ax.set(xlim=(-.65, 11.65), xlabel="Model", ylabel="Δ loss (nats)")
    axes_defaults(ax)
    ax.margins(y=.18)
    ax.legend(handles=capability_handles(), loc="upper right", ncol=3)
    return ax


def generate(root=ROOT):
    with frozen_run(root) as access:
        audit, plt = Artifacts(root), pyplot(root)
        rows = build(audit)
        panels = [(f"fig8_{p}", PANEL_SIZE, lambda f, p=p: draw_panel(f, rows, p),
                   [r for r in rows if r["panel"] == p],
                   ("Pruning at density 0.7.\n" if p == "a" else "Per-channel int4.\n") + CAPTION) for p in "ab"]
        export(plt, audit, "heterogeneity", panels, CAPTION, columns=1)
        return rows, audit, access


if __name__ == "__main__":
    generate()
