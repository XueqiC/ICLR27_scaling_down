#!/usr/bin/env python3
"""Compact CPU-only input/form evidence; run from any working directory.

Reuse Fig. 2's row representation and audited reader, and Fig. 3's source-file
loaders and matched density scoring. Only Panel A has stored intervals.
"""
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
    ax.axvline(0, color="#444444", lw=.8, zorder=1)
    for i in range(len(labels)):
        if i % 2 == 0:
            ax.axhspan(i - .5, i + .5, color="#f5f5f5", zorder=-2)
    ax.set_ylim(len(labels) - .5, -.5)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.tick_params(axis="x", length=2, pad=2)
    ax.grid(axis="x", color="#ececec", lw=.5)
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
                mfc="white" if hollow else color, mew=.8, zorder=3)


def main():
    setup_style()
    audit = AppendAudit("inputs", "Compact input/form figure: adding inputs, form, and capability shape")
    panel_a, panel_b, panel_c = load_panels(audit)
    fig = plt.figure(figsize=(6.5, 3.2))
    ax_a = forest_axis(fig, [.125, .32, .205, .47], [r["label"] for r in panel_a], (-.48, 2.95), [0, 1, 2])
    ax_b = forest_axis(fig, [.485, .32, .215, .47], list(dict.fromkeys(r["label"] for r in panel_b)),
                       (-.30, .55), [-.2, 0, .2, .4])
    ax_c = forest_axis(fig, [.82, .32, .165, .47], [r["label"] for r in panel_c], (-.25, .49), [-.2, 0, .2, .4])
    for i, rr in enumerate(panel_a):
        plot_points(ax_a, rr, i, COLORS[rr["method"]])
        ax_a.get_yticklabels()[i].set_color(COLORS[rr["method"]])
    for i, rr in enumerate(panel_b):
        plot_points(ax_b, rr, i // 2, COLORS[rr["baseline"]], hollow=rr["baseline"] == "A2",
                    dodge=-.035 if rr["baseline"] == "A2" else .035)
    for i, rr in enumerate(panel_c):
        plot_points(ax_c, rr, i, COLORS["distillation"])

    for x, title, subtitle in ((.025, "A  adding inputs", "v36b · leave-one-stage-out"),
                               (.385, "B  form on the same inputs", "MAE(A2/A1) − MAE(power)"),
                               (.765, "C  capability-specific\nshape", "")):
        fig.text(x, .975, title, fontsize=9, va="top", linespacing=1.05)
        if subtitle:
            fig.text(x, .865, subtitle, fontsize=8, va="bottom")
    fig.text(.025, .22, "L0 added to {N0,D0}\nD0 added to {N0,L0}", fontsize=8, va="top", linespacing=1.15)
    baseline_legend = [Line2D([], [], color=COLORS[b], marker="o", mfc="white" if b == "A2" else COLORS[b],
                             ls="", ms=4, label=b) for b in ("A2", "A1")]
    fig.legend(handles=baseline_legend, loc="upper left", bbox_to_anchor=(.405, .25), ncol=2,
               frameon=False, handletextpad=.3, columnspacing=1, borderpad=0)
    fig.text(.405, .13, "Pairs: 32k + 112k\nConfirm.: 3 states", fontsize=8, va="top", linespacing=1.1)
    fig.text(.765, .22, "v56 · LOCO\nShared − per-capability MAE", fontsize=8, va="top", linespacing=1.15)
    cap_legend = [Line2D([], [], color="#444444", marker=MARKERS[c], ls="", ms=4,
                        label={"math": "Math", "code": "Code", "qa": "QA"}[c]) for c in CAPS]
    fig.legend(handles=cap_legend, loc="lower left", bbox_to_anchor=(.012, .006), ncol=3,
               frameon=False, handletextpad=.3, columnspacing=.7, borderpad=0)
    fig.text(.985, .027, "MAE gain (nats/token); positive = better · A: stored 95% intervals",
             ha="right", fontsize=8)
    # Guard against silently clipping points/intervals if upstream results change.
    for ax, rows in ((ax_a, panel_a), (ax_b, panel_b), (ax_c, panel_c)):
        lo, hi = ax.get_xlim()
        for rr in rows:
            for cap in CAPS:
                values = [rr["values"][cap], *rr["intervals"].get(cap, [])]
                if not all(lo < value < hi for value in values):
                    raise ValueError(f"Expand compact axis limits: {rr['label']}/{cap}: {values}")
    audit.rule("Layout: 6.5 × 3.2 inches, all text >=8 pt at native size, 300 dpi PNG, embedded TrueType PDF; "
               "method/reference colours from Fig. 1. Circle=Math, square=Code, triangle=QA. "
               "Zero lines on every panel; positive gains favour added inputs, power, or per-capability shape, respectively.")
    audit.save(fig, "input_form_compact")
    audit.finish()


if __name__ == "__main__":
    main()
