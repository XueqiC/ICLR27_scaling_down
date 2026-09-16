#!/usr/bin/env python3
"""CPU-only paired input/form gains; three panels, each split math / code / QA."""

if __package__:
    from .paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp
else:
    from paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp
import sys

sys.dont_write_bytecode = True  # Keep generated files within the requested allowlist.
if __package__:
    from .plot_fig1_responses import Audit, CAPS, CAP_LABEL, COLORS, ROOT, checked_close, setup_style
else:
    from plot_fig1_responses import Audit, CAPS, CAP_LABEL, COLORS, ROOT, checked_close, setup_style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def short_test(group):
    """Abbreviate the inspected test strings; never use labels to derive numeric values."""
    src, test = group["src"], group["test"]
    if src.startswith("v36b"):
        return f"v36b {group['method']} · step LOO"
    if src.startswith("v39"):
        return "v39 distillation · step LOO"
    if src.startswith("v38"):
        return "v38 160M/1.4B @96k · prune"
    if src.startswith("v44"):
        return "v44 3 sizes @96k · " + ("bits ≥4" if "ge4" in test else "int3")
    if src.startswith("v42"):
        return "v42 unseen d=.65/.55 · prune"
    if src.startswith("v41"):
        return "v41 Gemma-3-1B · U225"
    if src.startswith(("v46", "v49")):
        version = "v46" if src.startswith("v46") else "v49"
        size = "6.9B" if "6.9b" in src else "1B"
        stage = "96k" if version == "v46" else "32k+112k"
        if group["method"] == "pruning":
            regime = "d=.55" if "extrap" in test else ("d=.65" if version == "v46" else "d=.9–.6")
        else:
            regime = "int3" if "int3" in test else ("int4" if version == "v46" else "bits ≥4")
        return f"{version} {size}@{stage} · {regime}"
    return test  # Newer registered groups retain their full saved identity.


def row(label, group, values, intervals=None, counts=None, comparators=None):
    return {"label": label, "method": group["method"], "origin": group["origin"].split("·")[0],
            "values": values, "intervals": intervals or {}, "counts": counts or {},
            "comparators": comparators or {}}


def draw_panel(fig, grid, title, rows, xlabel, annotate=None):
    axes = []
    for j, cap in enumerate(CAPS):
        ax = fig.add_subplot(grid[0, j])
        axes.append(ax)
        ax.axvline(0, color=PALETTE["reference"], lw=.8, zorder=0)
        for y, rr in enumerate(rows):
            if y % 2 == 0:
                ax.axhspan(y - .5, y + .5, color=PALETTE["background"], zorder=-2)
            gain = rr["values"][cap]
            color = COLORS[rr["method"]]
            marker = "o" if rr["origin"] == "P" else "s"
            ci = rr["intervals"].get(cap)
            if ci is not None:
                lo, hi = ci
                ax.hlines(y, lo, hi, color=color, lw=1)
                ax.vlines([lo, hi], y - .11, y + .11, color=color, lw=.8)
            ax.plot(gain, y, marker=marker, ms=4.2, linestyle="none", color=color,
                    markerfacecolor=PALETTE["white"] if rr["origin"] == "P" else color, zorder=4)
            if annotate == "n":
                label_x = max(gain, ci[1]) if ci is not None else gain
                ax.annotate(f"n={rr['counts'][cap]}", (label_x, y), xytext=(5, 0), textcoords="offset points",
                            va="center", fontsize=8)
            elif annotate == "comparator":
                name = rr["comparators"][cap].replace("A1_gamma1", "A1")
                ax.annotate(name, (gain, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)
        ax.set_ylim(len(rows) - .5, -.5)
        ax.set_yticks(range(len(rows)), [rr["label"] for rr in rows] if j == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.set_title(CAP_LABEL[cap], fontsize=9, pad=5)
        ax.locator_params(axis="x", nbins=4)
        ax.grid(axis="x", color=PALETTE["background"], lw=.5)
        vals = [v for rr in rows for v in (rr["values"][cap], *rr["intervals"].get(cap, []))]
        lo, hi = min(0, min(vals)), max(0, max(vals))
        span = max(hi - lo, .015)
        ax.set_xlim(lo - .12 * span, hi + (.53 if annotate else .12) * span)
    box = axes[0].get_position()
    fig.text(.035, box.y1 + .029, title, fontsize=11, va="bottom")
    right = axes[-1].get_position().x1
    fig.text((box.x0 + right) / 2, box.y0 - .027, xlabel, ha="center", fontsize=8)
    return axes


def load_panels(audit):
    groups = audit.read("results/v52-prediction-tables/groups.json")
    # Preserve the twenty tests in this historical appendix figure. Later V50
    # additions belong to the multi-student confirmation, not this cohort.
    groups = [g for g in groups if "v50-p2v2" not in g["src"]]
    if len(groups) != 20:
        raise ValueError("Expected the twenty original input/form test groups")
    v36 = audit.read("results/v36b-input-comparison/summary.json")
    v39 = audit.read("results/v39-distill-controlled/summary.json")
    v42 = audit.read("results/v42-prune-sameinput/summary.json")
    panel_a, panel_b, panel_c = [], [], []
    for g in groups:
        audit.rule(f"Group: axis={g['axis']}; method={g['method']}; test={g['test']}; "
                   f"origin={g['origin']}; recorded upstream source={g['src']}; recorded n={g['n']}.")
        src = g["src"]
        if src.startswith("v36b"):
            comp = {c: v36["results"][g["method"]][c]["comparison"] for c in CAPS}
            pair = {c: comp[c]["input_pairs"]["combined_vs_L0"] for c in CAPS}
            counts = {c: len({r["cell"] for r in comp[c]["records"]}) for c in CAPS}
            for c in CAPS:
                checked_close(pair[c]["improvement"], g["same"][c][1] - g["cand"][c], f"v36 no-D0/{c}")
                other = comp[c]["input_pairs"]["combined_vs_D0"]
                audit.rule(f"v36b {g['method']}/{c}: n={counts[c]} unique source states; "
                           f"{pair[c]['n_groups']} held-out step clusters for stored CI. "
                           f"Adding L0 instead (combined_vs_D0, not Panel A): {other['improvement']:+.9f}, "
                           f"CI={other['improvement_ci95']}.")
            panel_a.append(row(short_test(g), g, {c: pair[c]["improvement"] for c in CAPS},
                               {c: pair[c]["improvement_ci95"] for c in CAPS}, counts))
        elif src.startswith("v39"):
            values = {c: v39["by_cap"][c]["step"]["mae_noD0"] - v39["by_cap"][c]["step"]["mae_D0"] for c in CAPS}
            panel_a.append(row(short_test(g), g, values, counts={c: v39["panel"]["n_cells"] for c in CAPS}))
        elif src.startswith(("v46", "v49")) and all(
                g["same"][c] and g["same"][c][0] in ("noD0", "noD0_N0_L0") for c in CAPS):
            # Count actual source comparison files, not strength/configuration rows.
            paths = sorted(ROOT.glob("results/" + src))
            for path in paths:
                audit.read(path)
            if not paths:
                raise ValueError(f"Cannot count source states for {src}")
            panel_a.append(row(short_test(g), g, {c: g["same"][c][1] - g["cand"][c] for c in CAPS},
                               counts={c: len(paths) for c in CAPS}))
        elif src.startswith(("v46", "v49")):
            audit.omit(f"Panel A: {short_test(g)} has no no-D0 pruning predictor; same-input A1/A2 "
                       "is not a D0 ablation, so no point is substituted.")

        # v52.improvement is min(same, simple) - candidate, NOT simple - candidate.
        gains = {c: g["simple"][c][1] - g["cand"][c] for c in CAPS}
        for c in CAPS:
            references = [g[k][c][1] for k in ("same", "simple") if g[k][c] is not None]
            checked_close(g["improvement"][c], min(references) - g["cand"][c], f"v52 recorded gain/{c}")
            if not np.isclose(gains[c], g["improvement"][c]):
                audit.rule(f"Panel B {short_test(g)}/{c}: simple={g['simple'][c][0]}, "
                           f"simple-only gain={gains[c]:+.9f}; stored improvement={g['improvement'][c]:+.9f} "
                           "uses a different reference and is not plotted.")
        cis = {}
        if src.startswith("v36b"):
            for c in CAPS:
                metric = v36["results"][g["method"]][c]["comparison"]["core_table"]["N0_D0_L0"]
                checked_close(metric["improvement"], gains[c], f"v36 simple gain/{c}")
                cis[c] = metric["improvement_ci95"]
        panel_b.append(row(short_test(g), g, gains, cis))

    prune_template = {"method": "pruning", "origin": "P"}
    for baseline in ("A2", "A1"):
        panel_c.append(row(f"v42 unseen densities · vs {baseline}", prune_template,
                           {c: v42["by_cap"][c]["mae"][baseline] - v42["by_cap"][c]["mae"]["cand"] for c in CAPS},
                           comparators={c: baseline for c in CAPS}))
    for g in groups:
        if g["method"] == "pruning" and g["src"].startswith(("v46", "v49")):
            panel_c.append(row(short_test(g), g, {c: g["same"][c][1] - g["cand"][c] for c in CAPS},
                               comparators={c: g["same"][c][0] for c in CAPS}))
    audit.rule("Panel A: full vs no-D0, v36b leave-one-step-out and v39 step only; frozen v46/v49 "
               "quantization groups with an explicitly named noD0 line. n counts unique evaluated "
               "source states, not configurations. v36b uses 9 unique cells and 3 step clusters; "
               "v39 uses 9 panel cells; v46/v49 have 1/2 source-state files. "
               "Different steps on one trajectory are not independent trained models.")
    audit.rule("Panel B: all 20 groups, including main=false tests; one point per group/capability, "
               "separate test rows, method colour and origin shape (P hollow circle; L filled square). "
               "Plot simple[cap][1] - cand[cap]; no pooling across tests or panels. "
               "Intervals are stored v36b core_table.N0_D0_L0.improvement_ci95 only.")
    audit.rule("Panel C: v42 by_cap.mae[A2 or A1] - mae[cand]; v46/v49 pruning "
               "same[cap][1] - cand[cap], with the selected reference printed next to each point. "
               "v49 'same' can select continuous 2-term as well as A1/A2; selection stays as recorded.")
    audit.omit("Spec conflict: v52 improvement subtracts candidate from min(same-input, simple), "
               "so it cannot directly represent the requested source-free comparison in Panel B. "
               "The simple and candidate MAEs provide that comparison exactly; all discrepancies are listed above.")
    audit.omit("Spec conflict: Panel C's written cand-minus-baseline subtraction opposes the required "
               "positive=candidate-better convention. All Panel C differences are reversed and labelled explicitly.")
    audit.omit("No CI is supplied for v39 or the frozen prospective points; none is invented. "
               "The files enumerate source states but cannot establish statistically independent sources; "
               "n is explicitly labelled source states. The v36b narrative contains stale six-cell wording; "
               "counts come from its 9 unique records.cell values, not that prose.")
    return panel_a, panel_b, panel_c

def main():
    if __package__:
        from .plot_paper_appendix import generate
    else:
        from plot_paper_appendix import generate
    generate("input_form_gain")


if __name__ == "__main__":
    main()
