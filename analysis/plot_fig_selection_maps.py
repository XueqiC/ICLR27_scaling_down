#!/usr/bin/env python3
"""Shared-style sibling of V80 rule_maps_main; saved V78 selections only."""
from __future__ import annotations

if __package__:
    from .paper_artifacts import ROOT, Artifacts, frozen_run, pyplot
    from .paper_figure_style import panel_axes, legend_row
    from .paper_panel_exports import ref, export
else:
    from paper_artifacts import ROOT, Artifacts, frozen_run, pyplot
    from paper_figure_style import panel_axes, legend_row
    from paper_panel_exports import ref, export

COMPARE = "results/v78-rule-confirm/compare.json"
FREEZE = "results/v78-rule-confirm/freeze.json"
OBJECTIVES = ("math", "code", "qa", "multi")
METHODS = ("prune", "quant", "distill", "dense")
# Preserve V80's categorical method palette and marker semantics.
METHOD_COLORS = ("#2166ac", "#d97718", "#22834a", "#c4c4c4")
PANEL_SIZE = (2.7, 1.95)
LEGEND_SIZE = (5.5, .32)
CAPTION = """Frozen selection maps on four fresh source states, for Math (a),
Code (b), QA restricted to 2Wiki (c), and the maximum capability loss-change
objective (d). Each panel has the same four states and 17 nominal storage
budgets, from 20% through 100% of dense matrix storage in 5% increments.
Row labels give Pythia parameter size / pretraining step (k=1,000).
Cell colour is the frozen rule's predicted chosen method: pruning, quantization,
distillation or dense. A white circle means the measured oracle method differs;
unmarked cells agree at method level and need not agree at configuration level.
The method colours and mismatch semantics reproduce V80 rule_maps_main.
Selections, oracles and scores are read verbatim from V78 compare.json, with
the chosen method/configuration checked against freeze.json. Nothing is refit,
rescored or selected again. All 68 state-budget cells per objective are retained.
QA remains limited to 2Wiki; the multi-objective result remains retrospective.
The shared key is supplied as fig7_legend.pdf.
"""


def build(audit):
    compare, freeze = audit.read(COMPARE), audit.read(FREEZE)
    if audit.read(FREEZE + ".sha256").strip() != audit.inputs[FREEZE]:
        raise ValueError("V78 freeze seal mismatch")
    states = [s["tag"] for s in freeze["states"]]
    if len(states) != 4:
        raise ValueError("Expected four fresh V78 source states")
    rows = []
    for panel, cap in zip("abcd", OBJECTIVES):
        frozen = {(r["state"], r["budget"]): (i, r["policies"]["MAP"])
                  for i, r in enumerate(freeze["maps"]["locked-rule"][cap])}
        budgets = sorted({r["budget"] for r in compare["cells"][cap]})
        if len(budgets) != 17 or len(compare["cells"][cap]) != 68:
            raise ValueError("Expected 4 x 17 V78 selection cells")
        seen = set()
        for i, r in enumerate(compare["cells"][cap]):
            key = (r["state"], r["budget"])
            if key in seen:
                raise ValueError("Duplicate V78 state/budget cell")
            seen.add(key)
            chosen = r["policies"]["locked-rule"]
            fi, expected = frozen[key]
            if any(chosen[k] != expected[k] for k in ("method", "config_id", "predicted_score")):
                raise ValueError("Saved chosen configuration differs from frozen map")
            agreement = chosen["method"] == r["oracle_method"]
            if agreement != chosen["oracle_method_agreement"]:
                raise ValueError("V78 oracle agreement flag mismatch")
            rows.append({"panel": panel, "objective": cap, "state": r["state"],
                         "row": states.index(r["state"]), "column": budgets.index(r["budget"]),
                         "budget": r["budget"], "method": chosen["method"], "method_index": METHODS.index(chosen["method"]),
                         "oracle_method": r["oracle_method"], "oracle_agreement": agreement,
                         "budget_source": ref(COMPARE, "cells", cap, i, "budget"),
                         "method_source": ref(COMPARE, "cells", cap, i, "policies", "locked-rule", "method"),
                         "agreement_source": ref(COMPARE, "cells", cap, i, "policies", "locked-rule", "oracle_method_agreement"),
                         "state_source": ref(FREEZE, "states", states.index(r["state"]), "tag"),
                         "frozen_choice_source": ref(FREEZE, "maps", "locked-rule", cap, fi, "policies", "MAP")})
        if seen != {(s, b) for s in states for b in budgets}:
            raise ValueError("Incomplete V78 grid")
    audit.rule("Reuse V80 rule_maps_main semantics: colour indexes (prune,quant,distill,dense); "
               "white circle iff locked-rule.oracle_method_agreement is false. Direct V78 fields; "
               "freeze seal and frozen chosen method/configuration/score verified; no rule execution.")
    return rows


def draw_panel(fig, rows, panel):
    import numpy as np
    from matplotlib.colors import ListedColormap
    ax = panel_axes(fig, PANEL_SIZE, left=1.02, bottom=.53, right=.04, top=.03)
    part = [r for r in rows if r["panel"] == panel]
    matrix = np.empty((4, 17), dtype=int)
    for r in part:
        matrix[r["row"], r["column"]] = r["method_index"]
        if not r["oracle_agreement"]:
            ax.plot(r["column"], r["row"], "o", mfc="white", mec="#111111", ls="", zorder=3,
                    clip_on=False)  # Preserve the full 7-pt ring at the last budget.
    ax.imshow(matrix, cmap=ListedColormap(METHOD_COLORS), vmin=-.5, vmax=3.5,
              aspect="auto", interpolation="nearest", zorder=0)
    first = sorted((r for r in part if r["column"] == 0), key=lambda r: r["row"])
    labels = []
    for r in first:
        size, step = r["state"].removeprefix("pythia-").split("@step")
        labels.append(f"{size.upper()}/{int(step)//1000}k")
    ax.set_yticks(range(4), labels)
    ticks = [0, 8, 16]
    ax.set_xticks(ticks, [f"{100*next(r['budget'] for r in part if r['column']==i):g}" for i in ticks])
    ax.get_xticklabels()[-1].set_ha("right")
    ax.set_xticks(np.arange(-.5, 17), minor=True)
    ax.set_yticks(np.arange(-.5, 4), minor=True)
    ax.grid(which="minor", color="white", lw=.6, alpha=.6)
    ax.tick_params(which="both", length=0)
    ax.set_xlabel("Storage (%)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def draw_legend(fig):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=m) for c, m in zip(METHOD_COLORS, ("Prune", "Quant", "Distill", "Dense"))]
    handles += [Line2D([], [], marker="o", mfc="white", mec="#111111", ls="", label="Oracle differs")]
    return legend_row(fig, handles)


def generate(root=ROOT):
    with frozen_run(root) as access:
        audit, plt = Artifacts(root), pyplot(root)
        rows = build(audit)
        panels = [(f"fig7_{p}", PANEL_SIZE, lambda f, p=p: draw_panel(f, rows, p),
                   [r for r in rows if r["panel"] == p], f"Objective: {c}.\n" + CAPTION)
                  for p, c in zip("abcd", OBJECTIVES)]
        panels.append(("fig7_legend", LEGEND_SIZE, draw_legend, [], CAPTION))
        export(plt, audit, "selection_maps", panels, CAPTION, columns=2)
        return rows, audit, access


if __name__ == "__main__":
    generate()
