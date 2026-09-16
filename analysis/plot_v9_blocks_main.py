#!/usr/bin/env python3
"""Readable main-text Figure 1: 4 representative models (one per family + 30B) x 2 decisive metrics
(residual log-Fisher cosine; top-0.1% Jaccard), large bold Times-family fonts, shared colorbar per metric.
The full 7x4 grid stays in the appendix (plot_v9_blocks.py). Data unchanged (results/v9-capability-regions)."""

if __package__:
    from .paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp, neutral_cmap
else:
    from paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp, neutral_cmap
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import analysis.plot_v9_blocks as P
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif", "Times New Roman"],
                     "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold", "pdf.fonttype": 42})
MODELS = [("Qwen--Qwen3-4B", "Qwen3-4B"), ("gemma3-4b", "Gemma3-4B"), ("olmo3-7b", "OLMo3-7B"), ("muse-30b", "Muse-30B")]
METRICS = [("shared_component_removed_cosine", "Residual log-Fisher cosine", neutral_cmap()), ("top_0.1pct_jaccard", "Top-0.1% Jaccard", neutral_cmap())]
recs = []
for tag, label in MODELS:
    payload = json.loads((P.OUT_BASE / tag / "similarity.json").read_text()); names = P._benchmark_order(payload)
    recs.append({"label": label, "names": names, "caps": payload["benchmark_capability"],
                 "mats": {m: P._matrix(payload, m, names) for m, _, _ in METRICS}})
norms = {m: Normalize(*P._color_limits([r["mats"][m] for r in recs])) for m, _, _ in METRICS}
fig, axes = plt.subplots(len(MODELS), len(METRICS), figsize=(7.4, 3.55 * len(MODELS)))
for i, r in enumerate(recs):
    for j, (m, title, cmap) in enumerate(METRICS):
        ax = axes[i, j]; mat = r["mats"][m]; n = len(r["names"])
        im = ax.imshow(mat, cmap=cmap, norm=norms[m], interpolation="nearest")
        for pos in P._separator_positions(r["names"], r["caps"]):
            ax.axhline(pos, color=PALETTE["black"], lw=1.6); ax.axvline(pos, color=PALETTE["black"], lw=1.6)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(r["names"], rotation=60, ha="right", fontsize=11, fontweight="bold")
        ax.set_yticklabels(r["names"], fontsize=11, fontweight="bold")
        if i == 0: ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
        if j == 0: ax.set_ylabel(r["label"], fontsize=14, fontweight="bold", labelpad=12)
        ax.tick_params(length=3)
fig.subplots_adjust(left=0.17, right=0.80, top=0.965, bottom=0.075, wspace=0.42, hspace=0.42)
# both colorbars at the far right of the grid (no overlap with tick labels), labeled by metric
span_top = axes[0, -1].get_position(); span_bot = axes[-1, -1].get_position()
for j, (m, title, cmap) in enumerate(METRICS):
    cax = fig.add_axes([span_top.x1 + 0.015 + j * 0.075, span_bot.y0, 0.02, span_top.y1 - span_bot.y0])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norms[m], cmap=cmap), cax=cax); cb.ax.tick_params(labelsize=11)
    for t in cb.ax.get_yticklabels(): t.set_fontweight("bold")
out = P.FIGURE_DIR; out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "v9_blocks_main.png", dpi=300, bbox_inches="tight"); fig.savefig(out / "v9_blocks_main.pdf", bbox_inches="tight")
print("wrote", out / "v9_blocks_main.png")
