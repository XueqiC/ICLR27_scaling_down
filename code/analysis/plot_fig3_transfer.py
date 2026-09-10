#!/usr/bin/env python3
"""CPU-only transfer matrix using saved protocol summaries and matched v46 rows."""
import sys

sys.dont_write_bytecode = True
if __package__:
    from .plot_fig1_responses import Audit, CAPS, CAP_LABEL, COLORS, ROOT, checked_close, setup_style
else:
    from plot_fig1_responses import Audit, CAPS, CAP_LABEL, COLORS, ROOT, checked_close, setup_style
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

SOURCES = (("1b", 32000), ("1b", 96000), ("1b", 112000), ("6.9b", 32000), ("6.9b", 112000))
REGIMES = ("Prune interp\nd=.9–.6", "Prune extrap\nd=.55", "Quantization\nbits ≥4", "Quantization\nint3")
BASELINES = (("strength_only", "median_curve", "zero"), ("per_bit_mean", "per_bit_median", "zero"))
SHORT = {"strength_only": "strength", "median_curve": "median", "zero": "zero",
         "per_bit_mean": "bit mean", "per_bit_median": "bit median"}


def regime_rows(prune, quant):
    return ([r for r in prune if .6 <= r["d"] <= .9],
            [r for r in prune if np.isclose(r["d"], .55)],
            [r for r in quant if r["bit"] >= 4],
            [r for r in quant if r["bit"] == 3])


def score_cell(rows, candidate, baseline_names, stored=None):
    if not rows:
        return None
    # Every metric is evaluated on the same source/regime rows; baseline chosen once per cell.
    names = [candidate] + [name for name in baseline_names if all(name in r for r in rows)]
    mae = {name: float(np.mean([r[name]["abs"] for r in rows])) for name in names}
    for name in names:
        checked_close([r[name]["abs"] for r in rows],
                      [abs(r[name]["pred"] - r["actual"]) for r in rows], f"row absolute errors/{name}")
    if stored is not None:
        for name in names:
            checked_close(mae[name], stored[name], f"saved regime MAE/{name}")
        mae = {name: stored[name] for name in names}
    baseline = min((name for name in names if name != candidate), key=lambda name: mae[name])
    return {"gain": mae[baseline] - mae[candidate], "candidate": mae[candidate],
            "baseline": baseline, "baseline_mae": mae[baseline], "n_rows": len(rows)}


def main():
    setup_style()
    audit = Audit(3, "Limits of transfer by source, protocol, and regime")
    data = {}
    for size, step in SOURCES:
        if step == 96000:
            continue
        path = ROOT / f"results/v49-p1v2/compare_pythia-{size}@step{step}.json"
        if path.exists():
            data[size, step] = audit.read(path)
        else:
            audit.omit(f"{path.relative_to(ROOT)} is missing: both protocol rows marked not evaluated.")
    v46 = audit.read("results/v46-p1-newsource/compare.json")
    v41 = audit.read("results/v41-distill-newpool/summary.json")
    cells = {p: [[None] * 4 for _ in SOURCES] for p in ("A", "B")}
    for protocol in ("A", "B"):
        for i, (size, step) in enumerate(SOURCES):
            if step == 96000:
                if protocol == "B":
                    continue
                rows = regime_rows(v46["pruning"], v46["quantization"])
                summaries = [None] * 4
            else:
                payload = data.get((size, step), {}).get("protocols", {}).get(protocol)
                if payload is None:
                    continue
                rows = regime_rows(payload["rows_prune"], payload["rows_quant"])
                s = payload["summary"]
                summaries = [s["prune"]["interp_0.9-0.6"], s["prune"]["extrap_0.55"],
                             s["quant_ge4"], s["quant"]["int3"]]
            for j, rr in enumerate(rows):
                candidate = "power" if j < 2 else ("full_N0_L0_D0" if step == 96000 else "full")
                cell = score_cell(rr, candidate, BASELINES[0 if j < 2 else 1], summaries[j])
                cells[protocol][i][j] = cell
                if cell:
                    audit.rule(f"Protocol {protocol}, {size}@{step}, {REGIMES[j].replace(chr(10), ' ')}: "
                               f"{cell['n_rows']} rows, equal weight across math/code/QA; candidate={candidate}, "
                               f"MAE={cell['candidate']:.9f}; baseline={cell['baseline']}, "
                               f"MAE={cell['baseline_mae']:.9f}; gain={cell['gain']:+.9f}.")
    audit.rule("Read v49 summary.prune.interp_0.9-0.6, summary.prune.extrap_0.55, "
               "summary.quant_ge4 and summary.quant.int3; verify each against its exact underlying rows. "
               "Strongest source-free baseline minimizes aggregate MAE once per source/protocol/regime, "
               "not separately per capability or observation. Pruning candidates: strength_only, "
               "median_curve, zero; quantization: per_bit_mean, per_bit_median, zero where present.")
    audit.rule("For v46 1B@96k, compute MAE from pruning/quantization rows; its actual interpolation "
               "coverage is only d=.65 and its ≥4-bit coverage is only int4. Place these in protocol A "
               "as requested, labelled with an asterisk. The historical v46 freeze is preserved, not refit as v49.")
    audit.omit("1B@96k protocol B was not evaluated: four cells are hatched grey. "
               "The v46 A cells do not contain the full v49 density/bit ladders; unmeasured strengths "
               "are not filled in. Protocol summaries aggregate three capabilities as specified for this matrix.")
    audit.rule("Separate distillation panel: v41.by_cap[cap].mae.constant minus mae.E; "
               "U225 unseen pool, ordered math/code/QA. This candidate is E-only, with no post-test "
               "switch to constant in math/code. No confidence intervals are available.")
    gains = [c["gain"] for matrix in cells.values() for rr in matrix for c in rr if c]
    limit = max(abs(v) for v in gains)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    cmap = plt.get_cmap("RdBu")
    fig = plt.figure(figsize=(8.4, 7.1))
    grid = fig.add_gridspec(2, 2, left=.135, right=.985, bottom=.12, top=.885,
                          height_ratios=[3.5, 1.25], hspace=.64, wspace=.10)
    axes = []
    for col, protocol in enumerate(("A", "B")):
        ax = fig.add_subplot(grid[0, col])
        axes.append(ax)
        ax.set_xlim(-.5, 3.5)
        ax.set_ylim(4.5, -.5)
        ax.set_title(f"Protocol {protocol}", fontsize=11, pad=9)
        for i, rr in enumerate(cells[protocol]):
            for j, cell in enumerate(rr):
                if cell is None:
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="#e1e1e1",
                                           edgecolor="#bbbbbb", hatch="///", lw=.6))
                    ax.text(j, i, "not\nevaluated", ha="center", va="center", fontsize=8,
                            bbox={"facecolor": "#e1e1e1", "edgecolor": "none", "pad": 1})
                    continue
                face = cmap(norm(cell["gain"]))
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=face, edgecolor="white", lw=1))
                text_color = "white" if abs(cell["gain"]) > .63 * limit else "#171717"
                ax.text(j, i - .23, f"{cell['gain']:+.3f}", ha="center", va="center", color=text_color, fontsize=10)
                ax.text(j, i + .03, f"MAE {cell['candidate']:.3f}", ha="center", va="center", color=text_color, fontsize=8)
                ax.text(j, i + .28, SHORT[cell["baseline"]], ha="center", va="center", color=text_color, fontsize=8)
        labels = [f"{'1B' if s == '1b' else '6.9B'}@{t // 1000}k" + ("*" if t == 96000 else "") for s, t in SOURCES]
        ax.set_yticks(range(5), labels if col == 0 else [])
        ax.set_xticks(range(4), REGIMES)
        ax.tick_params(length=0, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
    color_ax = fig.add_axes([.31, .955, .5, .017])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_ax, orientation="horizontal")
    cb.ax.tick_params(labelsize=8, length=2, pad=1)
    fig.text(.56, .982, "MAE(simple) − MAE(candidate): positive = candidate better", ha="center", fontsize=9)
    distill = fig.add_subplot(grid[1, :])
    distill.axvline(0, color="#444444", lw=.8)
    for i, cap in enumerate(CAPS):
        mae = v41["by_cap"][cap]["mae"]
        gain = mae["constant"] - mae["E"]
        distill.plot(gain, i, marker="o", mfc="white", mec=COLORS["E"], ms=6, linestyle="none")
        distill.annotate(f"{gain:+.3f}  (E MAE {mae['E']:.3f}; constant {mae['constant']:.3f})",
                        (gain, i), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8)
        audit.rule(f"U225 {cap}: E MAE={mae['E']:.9f}, constant MAE={mae['constant']:.9f}, gain={gain:+.9f}.")
    distill.set_ylim(2.6, -.6)
    distill.set_yticks(range(3), [CAP_LABEL[c] for c in CAPS])
    distill.set_xlim(-.2, 1.35)
    distill.set_title("Distillation · unseen U225 pool (6 runs, Gemma-3-1B)", fontsize=10, loc="left")
    distill.set_xlabel("MAE(constant) − MAE(E-only), nats/token; positive = E-only better")
    distill.grid(axis="x", color="#eeeeee", lw=.5)
    fig.text(.5, .018, "*1B@96k: only d=.65/.55 and int4/int3; original v46 freeze. Cell lines: gain / candidate MAE / selected simple baseline.\n"
             "Pythia cells average math, code and QA. All values are nats per native token; the distillation panel is separate.",
             ha="center", fontsize=8, linespacing=1.5)
    audit.save(fig, "transfer_limits")
    audit.finish()


if __name__ == "__main__":
    main()
