#!/usr/bin/env python3
"""Compact CPU-only input/form evidence; run from any working directory.

Reuse Fig. 2's row representation and audited reader, and Fig. 3's source-file
loaders and matched density scoring. Only Panel A has stored intervals.
"""

if __package__:
    from .paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp
else:
    from paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp
import sys

sys.dont_write_bytecode = True
if __package__:
    from .plot_fig2_gains import CAPS, COLORS, checked_close, row, setup_style
    from .plot_fig3_transfer import AppendAudit, load_confirmation, load_v49, regime_rows, score_cell
else:
    from plot_fig2_gains import CAPS, COLORS, checked_close, row, setup_style
    from plot_fig3_transfer import AppendAudit, load_confirmation, load_v49, regime_rows, score_cell
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

MARKERS = {"math": "o", "code": "s", "qa": "^"}
OFFSETS = {"math": -.23, "code": 0, "qa": .23}


def load_panels(audit):
    v36 = audit.read("results/v36b-input-comparison/summary.json")
    v42 = audit.read("results/v42-prune-sameinput/summary.json")
    v49 = load_v49(audit)
    v53 = load_confirmation(audit)
    v56 = audit.read("results/v56-distill-forms/summary.json")
    panel_a, panel_b, panel_c = [], [], []
    for key, added in (("combined_vs_D0", "L0"), ("combined_vs_L0", "D0")):
        for method, label in (("pruning", "Prune"), ("quantization", "Quant")):
            # Same v36b comparison.input_pairs loading as plot_fig2_gains.py.
            pairs = {c: v36["results"][method][c]["comparison"]["input_pairs"][key] for c in CAPS}
            for cap, pair in pairs.items():
                checked_close(pair["improvement"], pair["reference_mae"] - pair["mae"],
                              f"v36b input gain/{method}/{key}/{cap}")
                audit.rule(f"Panel A {method}/{key}/{cap}: MAE({pair['reference']}) − MAE({pair['model']}) "
                           f"= {pair['improvement']:+.9f}; stored 95% interval={pair['improvement_ci95']}; "
                           f"n={pair['n']}, held-out stage clusters={pair['n_groups']}.")
            panel_a.append(row(f"{label} +{added}", {"method": method, "origin": "L"},
                               {c: pairs[c]["improvement"] for c in CAPS},
                               {c: pairs[c]["improvement_ci95"] for c in CAPS}))
    audit.rule("Panel A: v36b leave-one-stage-out; combined_vs_D0 adds L0 to {N0,D0}; "
               "combined_vs_L0 adds D0 to {N0,L0}. Use only stored improvement_ci95; no resampling or fitting.")

    prune = {"method": "pruning", "origin": "P"}
    for baseline in ("A2", "A1"):
        rr = row("v42 pooled", prune,
                 {c: v42["by_cap"][c]["mae"][baseline] - v42["by_cap"][c]["mae"]["cand"] for c in CAPS})
        rr["baseline"] = baseline
        panel_b.append(rr)
    for size, label in (("1b", "1B pair"), ("6.9b", "6.9B pair")):
        source_maes = []
        for step in (32000, 112000):
            payload = v49[size, step]["protocols"]["A"]
            rows = regime_rows(payload["rows_prune"], payload["rows_quant"])[0]
            per_cap = {}
            for cap in CAPS:
                selected = [r for r in rows if r["cap"] == cap]
                if not selected:
                    raise ValueError(f"Missing v49 interpolation rows: {size}@{step}/{cap}")
                per_cap[cap] = {b: score_cell(selected, "power", (b,)) for b in ("A2", "A1")}
                audit.rule(f"Panel B v49 {size}@{step}/{cap}: protocol A, "
                           f"{len(selected)} rows at densities {sorted({r['d'] for r in selected})}; "
                           "matched candidate and reference absolute errors checked against predictions.")
            source_maes.append(per_cap)
        for baseline in ("A2", "A1"):
            rr = row(label, prune,
                     {c: float(np.mean([s[c][baseline]["gain"] for s in source_maes])) for c in CAPS})
            rr["baseline"] = baseline
            panel_b.append(rr)
    for baseline in ("A2", "A1"):
        rr = row("Confirm.", prune,
                 {c: float(np.mean([p["mae"][baseline][c] - p["mae"]["power"][c] for p in v53.values()]))
                  for c in CAPS})
        rr["baseline"] = baseline
        panel_b.append(rr)
    for rr in panel_b:
        audit.rule(f"Panel B {rr['label']}: MAE({rr['baseline']}) − MAE(power), per capability: {rr['values']}.")
    audit.rule("Panel B: v42 uses by_cap.mae[A2/A1] − mae[cand] for its pooled test. "
               "Each v49 size pair gives equal weight to the 32k and 112k source states, after averaging "
               "protocol-A rows_prune with 0.6 <= d <= 0.9 separately per state/capability. "
               "Confirmation gives equal weight to 410M@48k, 1.4B@112k and 6.9B@80k, using saved mae "
               "over d=.85/.675/.575. Capabilities and test cohorts remain separate; both A2 and A1 are shown.")

    for structure, payload in v56["part_b"].items():
        values = {}
        for cap in CAPS:
            shared = payload["shared"]["loco"]["metrics"][cap]["mae"]
            per_cap = payload["per_capability"]["loco"]["metrics"][cap]["mae"]
            comparison = payload["comparison"][cap]
            checked_close([comparison["shared_mae"], comparison["per_capability_mae"], comparison["gain"]],
                          [shared, per_cap, shared - per_cap], f"v56 Part B LOCO/{structure}/{cap}")
            values[cap] = comparison["gain"]
            audit.rule(f"Panel C {structure}/{cap}: LOCO shared MAE={shared:.9f}, "
                       f"per-capability MAE={per_cap:.9f}, gain={values[cap]:+.9f}.")
        panel_c.append(row(structure, {"method": "distillation", "origin": "L"}, values))
    audit.rule("Panel C: every v56 Part B structure (E-only, T+E, F2), shared minus per-capability LOCO MAE. "
               "The comparison is checked against each mode's loco.metrics; no in-sample metrics are substituted. "
               "LOCO holds out a (student, U, data_seed) run cluster.")
    audit.omit("Panels B and C have no stored intervals for these comparisons; markers only. "
               "Pair/confirmation aggregation does not imply independent training trajectories or a pooled cross-test score.")
    return panel_a, panel_b, panel_c


def forest_axis(fig, bounds, labels, xlim, ticks):
    ax = fig.add_axes(bounds)
    ax.axvline(0, color=PALETTE["reference"], lw=.8, zorder=1)
    for i in range(len(labels)):
        if i % 2 == 0:
            ax.axhspan(i - .5, i + .5, color=PALETTE["background"], zorder=-2)
    ax.set_ylim(len(labels) - .5, -.5)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.tick_params(axis="x", length=2, pad=2)
    ax.grid(axis="x", color=PALETTE["background"], lw=.5)
    ax.spines["left"].set_visible(False)
    return ax


def plot_points(ax, rr, y, color, hollow=False, dodge=0):
    for cap in CAPS:
        yy = y + OFFSETS[cap] + dodge
        if cap in rr["intervals"]:
            lo, hi = rr["intervals"][cap]
            ax.hlines(yy, lo, hi, color=color, lw=.8)
            ax.vlines([lo, hi], yy - .055, yy + .055, color=color, lw=.7)
        ax.plot(rr["values"][cap], yy, marker=MARKERS[cap], ms=3.8, ls="none", color=color,
                mfc=PALETTE["white"] if hollow else color, mew=.8, zorder=3)


def main():
    if __package__:
        from .plot_paper_appendix import generate
    else:
        from plot_paper_appendix import generate
    return generate("input_form_compact")


if __name__ == "__main__":
    main()
