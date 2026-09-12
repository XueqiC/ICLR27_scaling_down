#!/usr/bin/env python3
"""Appendix: one readable figure PER MODEL (4 metrics in a row), for every model with similarity.json.
Bold Times-family fonts. Emits figs/v9_blocks_<tag>.{png,pdf} and paper/paper/figs/appendix_blocks.tex (figure envs)."""
import json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import analysis.plot_v9_blocks as P
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif", "Times New Roman"], "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold", "pdf.fonttype": 42})
LABEL = {"Qwen--Qwen3-0.6B": "Qwen3-0.6B", "Qwen--Qwen3-1.7B": "Qwen3-1.7B", "Qwen--Qwen3-4B": "Qwen3-4B", "gemma3-270m": "Gemma3-270M", "gemma3-1b": "Gemma3-1B",
         "gemma3-4b": "Gemma3-4B", "gemma3-12b": "Gemma3-12B", "gemma4-31b": "Gemma4-31B", "muse-30b": "Muse-30B", "olmo3-7b": "OLMo3-7B"}
CMAP = {"raw_cosine": "coolwarm", "log_cosine": "coolwarm", "shared_component_removed_cosine": "coolwarm", "top_0.1pct_jaccard": "viridis"}
tags = [t for t in P._selected_model_tags(None) if t in LABEL]
recs = {t: json.loads((P.OUT_BASE / t / "similarity.json").read_text()) for t in tags}
norms = {m: Normalize(*P._color_limits([P._matrix(recs[t], m, P._benchmark_order(recs[t])) for t in tags])) for m, _ in P.METRICS}
envs = []
for t in tags:
    payload = recs[t]; names = P._benchmark_order(payload); caps = payload["benchmark_capability"]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2))
    for j, (m, title) in enumerate(P.METRICS):
        ax = axes[j]; mat = P._matrix(payload, m, names); n = len(names)
        im = ax.imshow(mat, cmap=CMAP[m], norm=norms[m], interpolation="nearest")
        for pos in P._separator_positions(names, caps): ax.axhline(pos, color="black", lw=1.5); ax.axvline(pos, color="black", lw=1.5)
        ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=10, fontweight="bold")
        ax.set_yticklabels(names if j == 0 else [], fontsize=10, fontweight="bold"); ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=9)
        for tk in cb.ax.get_yticklabels(): tk.set_fontweight("bold")
    fig.suptitle(LABEL[t], fontsize=15, fontweight="bold", y=1.02); fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.26, wspace=0.35)
    safe = t.replace("--", "-"); fig.savefig(P.FIGURE_DIR / f"v9_blocks_{safe}.png", dpi=220, bbox_inches="tight"); fig.savefig(P.FIGURE_DIR / f"v9_blocks_{safe}.pdf", bbox_inches="tight"); plt.close(fig)
    envs.append("\\begin{figure}[h]\n\\centering\n\\includegraphics[width=\\linewidth]{figs/v9_blocks_%s.pdf}\n\\caption{Capability regions in gradient space for %s: pairwise similarity of per-benchmark gradient signatures under four metrics (raw Fisher cosine, log-Fisher cosine, shared-component-removed log-Fisher cosine, top-0.1\\%% coordinate Jaccard). Black lines separate the math, code, QA, and control blocks.}\n\\label{fig:blocks_%s}\n\\end{figure}" % (safe, LABEL[t], safe.replace(".", "").replace("-", "_")))
(P.ROOT / "paper/paper/figs/appendix_blocks.tex").write_text("\n\n".join(envs) + "\n")
import shutil; [shutil.copy(P.FIGURE_DIR / f"v9_blocks_{t.replace('--','-')}.pdf", P.ROOT / "paper/paper/figs/") for t in tags]
print(len(tags), "per-model figures ->", P.ROOT / "paper/paper/figs/appendix_blocks.tex")
