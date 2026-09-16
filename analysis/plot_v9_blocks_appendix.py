#!/usr/bin/env python3
"""All ten included capability-region maps, from frozen similarity matrices."""
from __future__ import annotations

import numpy as np

if __package__:
    from .paper_artifacts import ROOT, Artifacts, frozen_run, pyplot, output_path
    from .paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, neutral_cmap, panel_axes, compact_number
    from .plot_paper_appendix import publish, capability_keys, key
else:
    from paper_artifacts import ROOT, Artifacts, frozen_run, pyplot, output_path
    from paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, neutral_cmap, panel_axes, compact_number
    from plot_paper_appendix import publish, capability_keys, key

LABEL = {"Qwen--Qwen3-0.6B": "Qwen3-0.6B", "Qwen--Qwen3-1.7B": "Qwen3-1.7B", "Qwen--Qwen3-4B": "Qwen3-4B",
         "gemma3-12b": "Gemma3-12B", "gemma3-1b": "Gemma3-1B", "gemma3-270m": "Gemma3-270M", "gemma3-4b": "Gemma3-4B",
         "gemma4-31b": "Gemma4-31B", "muse-30b": "Muse-30B", "olmo3-7b": "OLMo3-7B"}
ORDER = ("gsm8k", "math500", "svamp", "humaneval", "mbpp", "2wiki", "hotpotqa", "triviaqa", "c4")
SHORT = ("GSM8K", "MATH", "SVAMP", "HEval", "MBPP", "2Wiki", "Hotpot", "Trivia", "C4")
METRICS = (("raw_cosine", "Raw Fisher cosine"), ("log_cosine", "Log-Fisher cosine"),
           ("shared_component_removed_cosine", "Residual log-Fisher cosine"), ("top_0.1pct_jaccard", "Top-0.1% Jaccard"))
SIZE = (2.7, 2.4)


def generate(root=ROOT):
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator, FuncFormatter
    with frozen_run(root) as access:
        audit=Artifacts(root);plt=pyplot(root)
        data={tag:audit.read(f"results/v9-capability-regions/{tag}/similarity.json") for tag in LABEL}
        matrices={(tag,m):np.array([[r["matrices"][m][a][b] for b in ORDER] for a in ORDER])
                  for tag,r in data.items() for m,_ in METRICS}
        norms={m:Normalize(min(matrices[tag,m].min() for tag in data),max(matrices[tag,m].max() for tag in data)) for m,_ in METRICS}
        environments=[]
        for index,(tag,label) in enumerate(LABEL.items()):
            r=data[tag];assert set(r["benchmarks"])==set(ORDER)
            stem="v9_blocks_"+tag.replace("--","-")
            specs=[]
            for metric,title in METRICS:
                matrix=matrices[tag,metric]
                assert matrix.shape==(9,9) and np.isfinite(matrix).all()
                def draw(f,matrix=matrix,metric=metric,r=r):
                    ax=panel_axes(f,SIZE,left=.54,bottom=.63,right=.58,top=.08)
                    im=ax.imshow(matrix,cmap=neutral_cmap(),norm=norms[metric],interpolation="nearest",aspect="equal")
                    ax.set_xticks(range(9),SHORT,rotation=55,ha="right")
                    ax.set_yticks(range(9),SHORT)
                    for direction in (ax.get_xticklabels(),ax.get_yticklabels()):
                        for text,name in zip(direction,ORDER):text.set_color(QA_COLORS.get(name, CAPABILITY_COLORS.get(r["benchmark_capability"][name],PALETTE["reference"])))
                    for boundary in (2.5,4.5,7.5):
                        ax.axhline(boundary,color=PALETTE["black"])
                        ax.axvline(boundary,color=PALETTE["black"])
                    ax.tick_params(length=0)
                    cax=f.add_axes((2.20/2.7,.64/2.4,.06/2.7,1.5/2.4))
                    cb=f.colorbar(im,cax=cax)
                    cb.locator=MaxNLocator(3);cb.formatter=FuncFormatter(compact_number);cb.update_ticks()
                specs.append((title,SIZE,draw,[{"model":tag,"metric":metric,"benchmarks":list(ORDER),"matrix":matrix.tolist(),"normalization":[norms[metric].vmin,norms[metric].vmax]}]))
            caption=f"Capability regions in gradient space for {label}: pairwise similarity of per-benchmark gradient signatures under four metrics (raw Fisher cosine, log-Fisher cosine, shared-component-removed log-Fisher cosine, top-0.1% coordinate Jaccard). Black lines separate the math, code, QA, and control blocks."
            publish(plt,audit,stem,specs,capability_keys()+[key("Control",marker=""),key("Darker = higher",marker="")],caption+" Benchmark labels: MATH = math500; HEval = HumanEval; Hotpot = HotpotQA; Trivia = TriviaQA. Axis-label hue denotes capability; matrix intensity denotes similarity. Each metric uses one unchanged shared scale across all ten models.",columns=2)
            env=[r"\begin{figure}[p]",r"\centering",rf"\includegraphics[width=5.5in]{{figs/{stem}_legend.pdf}}\\[1pt]"]
            for i,(_,title) in enumerate(METRICS):
                if i:env.append(r"\\[3pt]" if i%2==0 else r"\hfill")
                env.append(rf"\begin{{subfigure}}[t]{{2.7in}}\centering\includegraphics[width=\linewidth]{{figs/{stem}_{chr(97+i)}.pdf}}\subcaption{{{title.replace('%',chr(92)+'%')}}}\end{{subfigure}}")
            env += [r"\caption{"+caption.replace("%",r"\%")+"}"]
            if index==0:env += [r"\label{fig:blocks}"]
            env += [r"\label{fig:blocks_"+tag.replace("--","-").replace(".","").replace("-","_")+"}",r"\end{figure}"]
            environments.append("\n".join(env))
        output_path(root,"figs","appendix_blocks.tex").write_text("\n\n".join(environments)+"\n")
        return data,audit,access


if __name__=="__main__":
    generate()
