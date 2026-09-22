#!/usr/bin/env python3
"""Generate the heterogeneous-panel tables and count macros from raw result files (CPU only).
Inputs: results/v6-capability-geometry/<model>/prune_losses.json, results/v10-quantization/<model>/quant_losses.json.
Outputs: paper/paper/tables/panel_prune.tex, paper/paper/tables/panel_quant.tex, paper/paper/tables/counts.tex, results/v51-panel/panel.json"""

try:
    from .paper_table_text import table_text as publication_table_text
except ImportError:  # Direct scripts and file-based imports.
    try:
        from analysis.paper_table_text import table_text as publication_table_text
    except ImportError:
        from paper_table_text import table_text as publication_table_text

import json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; R6 = ROOT / "results/v6-capability-geometry"; R10 = ROOT / "results/v10-quantization"
OUT = ROOT / "results/v51-panel"; TAB = ROOT / "paper/paper/tables"
C = ("math", "code", "qa")
# display name, family, series, cohort (panel = original heterogeneous 12; prosp = later prospective additions)
MODELS = [("Qwen3-0.6B", "Qwen3-0.6B", "Qwen3", "panel"), ("Qwen3-1.7B", "Qwen3-1.7B", "Qwen3", "panel"), ("Qwen3-4B", "Qwen3-4B", "Qwen3", "panel"),
          ("Qwen3-8B", "Qwen3-8B", "Qwen3", "prosp"), ("Qwen3-14B", "Qwen3-14B", "Qwen3", "prosp"),
          ("gemma3-270m", "Gemma-3-270M", "Gemma-3", "panel"), ("gemma3-1b", "Gemma-3-1B", "Gemma-3", "panel"), ("gemma3-4b", "Gemma-3-4B", "Gemma-3", "panel"),
          ("gemma3-12b", "Gemma-3-12B", "Gemma-3", "panel"), ("gemma3-27b", "Gemma-3-27B", "Gemma-3", "panel"), ("gemma4-31b", "Gemma-4-31B", "Gemma-4", "panel"),
          ("muse-30b", "Muse-30B", "Muse", "panel"), ("olmo3-7b", "OLMo-3-7B", "OLMo-3", "panel"), ("olmo3-32b", "OLMo-3-32B", "OLMo-3", "panel")]
QUANT_DIR = {"Qwen3-0.6B": "Qwen--Qwen3-0.6B", "Qwen3-1.7B": "Qwen--Qwen3-1.7B", "Qwen3-4B": "Qwen--Qwen3-4B"}
PRUNE_DIR = {"Qwen3-8B": "Qwen--Qwen3-8B", "Qwen3-14B": "Qwen--Qwen3-14B"}
def load_prune(m, *, root=ROOT, read=None):
    p = Path(root) / "results/v6-capability-geometry" / PRUNE_DIR.get(m, m) / "prune_losses.json"
    if not p.exists(): return None
    j = read(p) if read else json.loads(p.read_text()); dense = j["1.0"]
    dens = sorted([float(k) for k in j if not k.startswith("_") and k != "1.0"], reverse=True)
    key = {float(k): k for k in j if not k.startswith("_")}
    dl = {d: {c: j[key[d]][c] - dense[c] for c in C} for d in dens}
    return dens, dl
def load_quant(m, *, root=ROOT, read=None):
    p = Path(root) / "results/v10-quantization" / QUANT_DIR.get(m, m) / "quant_losses.json"
    if not p.exists(): return None
    j = read(p) if read else json.loads(p.read_text()); dense = j["dense"]
    return {int(b): {c: j[b][c] - dense[c] for c in C} for b in j if b.isdigit()}


def short_capability_subheaders(text):
    """Author's preferences (2026-09-22): capability sub-headers read Math / Code / QA and the panel column names stay short."""
    for long, short in (("Mathematics & Code & Question answering", "Math & Code & QA"),
                        ("Mathematics loss change at density 0.5", "Math at density 0.5"),
                        ("Density where mathematics loss has risen by one nat", "Threshold density"),
                        ("Question answering least affected; most affected, out of measured densities", "QA least; most affected"),
                        ("Smallest question-answering loss change (density)", "Lowest QA change (density)"),
                        ("Loss change at 4 bits per weight", "Loss change at 4 bits"),
                        ("Loss change at 3 bits per weight", "Loss change at 3 bits")):
        text = text.replace(long, short)
    return text


def main():
    OUT.mkdir(exist_ok=True)
    rows = []; fam = lambda v: f"{v:+.2f}"
    for m, name, series, cohort in MODELS:
        pr = load_prune(m); qu = load_quant(m); row = dict(model=name, series=series, cohort=cohort)
        if pr:
            dens, dl = pr; cliff = next((d for d in dens if dl[d]["math"] >= 1), None)
            pre = [d for d in dens if cliff is None or d > cliff]
            least = [min(C, key=lambda c: dl[d][c]) for d in pre]; most = [max(C, key=lambda c: dl[d][c]) for d in pre]
            row.update(n_densities=len(dens), d_min=min(dens), dl07={c: dl[0.7][c] for c in C} if 0.7 in dl else None,
                       dl05_math=dl[0.5]["math"] if 0.5 in dl else None, cliff=cliff, pre_cliff_densities=pre,
                       qa_least_damaged=sum(l == "qa" for l in least), qa_most_damaged=sum(x == "qa" for x in most), n_pre=len(pre),
                       qa_sign_mild=("improves" if dl[max(dens)]["qa"] < 0 else "degrades"),
                       qa_min_dl=min(dl[d]["qa"] for d in dens), qa_min_at=min(dens, key=lambda d: dl[d]["qa"]),
                       max_gap_mild=max(abs(dl[d]["qa"] - min(dl[d]["math"], dl[d]["code"])) for d in pre) if pre else None)
        if qu:
            row.update(quant_bits=sorted(qu), dl4={c: qu[4][c] for c in C} if 4 in qu else None, dl3={c: qu[3][c] for c in C} if 3 in qu else None)
        rows.append(row)
    json.dump(rows, open(OUT / "panel.json", "w"), indent=1)
    # ---- pruning panel table
    L = [r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}lrrrrlcl@{}}", r"\toprule",
         r"Model & \multicolumn{3}{c}{$\Delta L_c$ at $d=0.7$} & math@0.5 & $d^{*}_{\mathrm{math}}$ & QA least$\,|\,$worst & min QA ($d$) \\",
         r"\cmidrule(lr){2-4} & math & code & QA & & & & \\", r"\midrule"]
    for r in rows:
        if "dl07" not in r: continue
        d7 = r["dl07"]; least = (f"{r['qa_least_damaged']}/{r['n_pre']}$\\,|\\,${r['qa_most_damaged']}/{r['n_pre']}" if r["n_pre"] else "--")


        cliff = f"{r['cliff']:.2f}" if r["cliff"] else f"none by {r['d_min']:.2f}"
        star = "" if r["cohort"] == "panel" else ""
        L.append(f"{r['model']}{star} & {fam(d7['math'])} & {fam(d7['code'])} & {fam(d7['qa'])} & "
                 f"{(fam(r['dl05_math']) if r['dl05_math'] is not None else '--')} & {cliff} & {least} & {fam(r['qa_min_dl'])} ({r['qa_min_at']:.2f}) \\\\")
    L += [r"\bottomrule\end{tabular}",
          r"\caption{Per-capability pruning damage on the heterogeneous panel (nats/token, native tokenizer units; within-model differences). "
          r"$d^{*}_{\mathrm{math}}$ is the largest measured density with $\Delta L_{\mathrm{math}}\ge 1$; models are grouped by series (Qwen3, Gemma-3, Gemma-4, Muse, OLMo-3); ``math@0.5'' is $\Delta L_{\mathrm{math}}$ at $d=0.5$; ``QA least$\,|\,$worst'' counts, over the pre-cliff densities, how often QA is the least-damaged and how often the most-damaged capability "
          r"the last column is the most negative QA response and the density at which it occurs. "
          r"Generated by \texttt{v51\_panel\_tables.py} from \texttt{prune\_losses.json}.}",
          r"\label{tab:panel_prune}\end{table}"]
    (TAB / "panel_prune.tex").write_text(short_capability_subheaders(publication_table_text("\n".join(L) + "\n")))
    # ---- quantization panel table
    L = [r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}llrrrrrrl@{}}", r"\toprule",
         r"Model & Series & \multicolumn{3}{c}{$\Delta L_c$ at int4} & \multicolumn{3}{c}{$\Delta L_c$ at int3} & bits \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8} & & math & code & QA & math & code & QA & \\", r"\midrule"]
    for r in rows:
        if "dl4" not in r or r["dl4"] is None: continue
        d4, d3 = r["dl4"], r["dl3"]; star = "" if r["cohort"] == "panel" else ""
        L.append(f"{r['model']}{star} & {r['series']} & {fam(d4['math'])} & {fam(d4['code'])} & {fam(d4['qa'])} & "
                 f"{fam(d3['math'])} & {fam(d3['code'])} & {fam(d3['qa'])} & {','.join(str(b) for b in sorted(r['quant_bits'], reverse=True))} \\\\")
    L += [r"\bottomrule\end{tabular}",
          r"\caption{Per-capability quantization damage (per-output-channel symmetric RTN, weights only; nats/token). int8 and int6 are within 0.01 nats of dense for every listed model and are omitted. "
          r"Generated by \texttt{v51\_panel\_tables.py} from \texttt{quant\_losses.json}.}", r"\label{tab:panel_quant}\end{table}"]
    (TAB / "panel_quant.tex").write_text(short_capability_subheaders(publication_table_text("\n".join(L) + "\n")))
    # ---- counts
    panel = [r for r in rows if r["cohort"] == "panel" and "dl07" in r]; quant = [r for r in rows if r.get("dl3")]
    collapse = [r["model"] for r in quant if min(r["dl3"].values()) >= 4]; nocollapse = [r["model"] for r in quant if max(r["dl3"].values()) < 4]
    series = sorted({r["series"] for r in panel}); families = sorted({s.split("-")[0] for s in series})
    qa_least_models = [r["model"] for r in panel if r["n_pre"] and r["qa_least_damaged"] == r["n_pre"]]
    qa_worst_models = [r["model"] for r in panel if r["n_pre"] and r["qa_most_damaged"] >= max(1, r["n_pre"] - 1)]
    counts = dict(n_panel_prune=len(panel), n_prune_all=len([r for r in rows if "dl07" in r]), n_quant=len(quant), n_quant_collapse=len(collapse),
                  quant_no_collapse=nocollapse, n_series=len(series), series=series, n_families=len(families), families=families,
                  qa_least_models=qa_least_models, qa_worst_models=qa_worst_models)
    json.dump(counts, open(OUT / "counts.json", "w"), indent=1)
    mac = [f"\\newcommand{{\\nPanelPrune}}{{{counts['n_panel_prune']}}}", f"\\newcommand{{\\nQuantModels}}{{{counts['n_quant']}}}",
           f"\\newcommand{{\\nQuantCollapse}}{{{counts['n_quant_collapse']}}}", f"\\newcommand{{\\nSeries}}{{{counts['n_series']}}}", f"\\newcommand{{\\nFamilies}}{{{counts['n_families']}}}"]
    (TAB / "counts.tex").write_text("% generated by analysis/v51_panel_tables.py\n" + "\n".join(mac) + "\n")
    print(json.dumps(counts, indent=1)); print("wrote panel_prune.tex, panel_quant.tex, counts.tex")


if __name__ == "__main__":
    main()
