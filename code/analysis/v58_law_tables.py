#!/usr/bin/env python3
"""Tables for the restructured paper (CPU): Table 1 models-and-domains; appendix coefficient table for the grouped-RTN 2D
form (v55); LOSO/selection tables for v53, v55; v56 forms + capability-conditioning tables."""
import json, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; TAB = ROOT / "paper/paper/tables"; C = ("math", "code", "qa")
# ---- Table 1: predictive models and their domains
rows = [
 ("Pruning", r"$A_c(\mathbf x)\,((1-d)/0.3)^{\gamma_c}$ (Eq.~\ref{eq:power}); same-input alternative A2 (per-$d$ regression + interpolation)", "5 (A2: 20)", r"$d$; 17 Pythia states (160M--6.9B, 16k--143k)", "0", r"$d\in[0.55,0.9]$ interp.\ on new states; $d=0.55$ extrap.\ fails", r"Table~\ref{tab:round3_coef}"),
 ("Quantization, fixed $b$", r"per-bit regression $\beta_{c,b}\cdot\phi(\mathbf x)$", "4 per $b$", r"new state at seen $b\in\{8,6,4,3\}$; 9 dev states", "0", r"seen $b$; 5-bit rule fails; near-zero regime uninformative", "register v38/v49"),
 ("Quantization, $(b,g)$", r"$\phi(\mathbf x)\cdot[1,u,v,uv,u^2]$ (Eq.~\ref{eq:quant2d})", "20", r"$b\in\{3,5\}\times g\in\{64,256\}$; 6 dev states", "0", r"unseen $b=4$, $g=128$ on dev states; new state mixed", r"Table~\ref{tab:quant2d_coef}"),
 ("Distillation, source axis", r"$\delta_c=\beta_c\cdot\phi(\mathbf x)$ (initial student)", "4", "9 Pythia students, fixed pool/budget", "0", "held-out stage; math only", "register v39"),
 ("Distillation, pool axis", r"$a_c+b_c\log E$ (and constant, $\log T$, joint)", "2", "Gemma-3-1B; pools U75/U600", "0", "unseen pool U225: QA only; multi-student test frozen", "register v41/v50"),
]
L = [r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}p{1.8cm}p{3.6cm}p{0.8cm}p{2.5cm}p{0.7cm}p{2.2cm}p{1.2cm}@{}}", r"\toprule",
     r"Arm & Predictor & $P$/cap. & Varied setting; dev.\ states & Target cal. & Tested range & Coeff. \\", r"\midrule"]
for r in rows: L.append(" & ".join(r) + r" \\")
L += [r"\bottomrule\end{tabular}", r"\caption{Predictive models and their domains. Inputs are always $\mathbf x=(N_0,D_0,L_{0,c})$ through $\phi(\mathbf x)=[1,\tilde{\log N_0},\tilde L_{0,c},\tilde{\log D_0}]$; no prediction uses a measurement of the compressed target (target calibration count 0). Ranges summarize \S\ref{sec:general}; coefficients and standardization constants are in Appendix~\ref{sec:app_coef}.}", r"\label{tab:models}\end{table}"]
(TAB / "models_domains.tex").write_text("\n".join(L) + "\n")
# ---- quant 2D coefficients (v55)
r5 = json.load(open(ROOT / "results/v55-quant-group/register.json")); st = r5["standardization"]
L = [r"\begin{table}[H]\centering\footnotesize", r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule", r"capability & term & $\beta_{0}$ & $\beta_{\log N_0}$ & $\beta_{L_0}$ & $\beta_{\log D_0}$ \\", r"\midrule"]
for c in C:
    coef = r5["models"][c]["low_order_2d"]["coefficients"]
    for name, row in zip(("1", "$u$", "$v$", "$uv$", "$u^2$"), coef): L.append(f"{c} & {name} & " + " & ".join(f"{b:+.3f}" for b in row) + r" \\")
    if c != "qa": L.append(r"\midrule")
L += [r"\bottomrule\end{tabular}", r"\caption{Grouped-quantization form $\widehat{\Delta L}_c=\sum_k \beta_{c,k}\cdot\phi(\mathbf x)\,t_k(b,g)$ with terms $t_k\in\{1,u,v,uv,u^2\}$, $u=\log_2 q_{\max}(b)$ centred at " + f"{r5['models']['math'].get('u_center', r5.get('u_center', float('nan'))) if isinstance(r5['models']['math'].get('u_center'), float) else 'the development mean'}" + r", $v=\log_2(g/128)$; standardization centers " + ", ".join(f"{v:.3f}" for v in st["center"]) + " and scales " + ", ".join(f"{v:.3f}" for v in st["scale"]) + r" for $(\log N_0, L_{0,c}, \log D_0)$; ridge $10^{-3}$. Register: \texttt{v55-quant-group/register.json}.}", r"\label{tab:quant2d_coef}\end{table}"]
(TAB / "quant2d_coef.tex").write_text("\n".join(L) + "\n")
# ---- v53 LOSO table
rows = list(csv.DictReader(open(ROOT / "results/v53-prune-dev/loso_table.csv")))
L = [r"\begin{table}[H]\centering\footnotesize", r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule", r"subset & candidate & math & code & QA & mean \\", r"\midrule"]
for r in rows: L.append(f"{r['subset'].replace('_',' ')} & {r['candidate'].replace('_',' ')} & {float(r['math_mae']):.3f} & {float(r['code_mae']):.3f} & {float(r['qa_mae']):.3f} & {float(r['mean_mae']):.3f} \\\\")
L += [r"\bottomrule\end{tabular}", r"\caption{Pruning development on 17 Pythia states: leave-one-source-out MAE (nats) over all held-out rows (`all') and over the densities off the coarse grid (0.75, 0.65, 0.55). Pre-committed rule: lowest mean over capabilities among the four source-conditioned candidates; a gap below 0.02 nats between the best two is resolved toward the continuous, zero-at-$d{=}1$ form with fewer parameters, which selected the power form over A2 (gap 0.015). Baselines do not select.}", r"\label{tab:v53_loso}\end{table}"]
(TAB / "v53_loso.tex").write_text("\n".join(L) + "\n")
# ---- v55 LOSO table
lo = r5["loso_table"]
L = [r"\begin{table}[H]\centering\footnotesize", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r"candidate & math & code & QA \\", r"\midrule"]
def getm(entry, c):
    if isinstance(entry, dict): return entry.get(c, entry.get("mae", {}).get(c))
    return None
if isinstance(lo, dict):
    for k, v in lo.items(): L.append(f"{k.replace('_',' ')} & " + " & ".join(f"{float(getm(v,c)):.3f}" for c in C) + r" \\")
elif isinstance(lo, list):
    for e in lo: L.append(f"{str(e.get('candidate','')).replace('_',' ')} & " + " & ".join(f"{float(e.get(c, e.get('mae',{}).get(c))):.3f}" for c in C) + r" \\")
L += [r"\bottomrule\end{tabular}", r"\caption{Grouped quantization: leave-one-state-out MAE (nats) on the 24 development cells (six states). The late-stage 160M state, which collapses at 3 bits, dominates every held-out error, so the development set does not rank the forms; all three candidates and the baselines were frozen and tested (\S\ref{sec:unseen_settings}).}", r"\label{tab:v55_loso}\end{table}"]
(TAB / "v55_loso.tex").write_text("\n".join(L) + "\n")
# ---- v56 forms tables from summary.md (already tabulated there)
md = (ROOT / "results/v56-distill-forms/summary.md").read_text()
def md_table(txt, head):
    i = txt.index(head); j = txt.index("\n\n", i + 1); block = txt[i:j].strip().split("\n"); return [l for l in block if l.startswith("|")]
tA = md_table(md, "| Form | P |"); tB = md_table(md, "| Structure | P shared/per |")
def to_tex(lines, caption, label, colspec):
    hdr = [c.strip() for c in lines[0].strip("|").split("|")]; out = [r"\begin{table}[H]\centering\scriptsize\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{" + colspec + "}", r"\toprule", " & ".join(h.replace("_", r"\_") for h in hdr) + r" \\", r"\midrule"]
    for l in lines[2:]:
        cells = [c.strip().replace("_", r"\_").replace("**", "") for c in l.strip("|").split("|")]; out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule\end{tabular}", r"\caption{" + caption + "}", f"\\label{{{label}}}\\end{{table}}"]; return "\n".join(out) + "\n"
tex = to_tex(tA, r"Distillation development matrix (12 runs, 48 points): held-out MAE/signed bias (nats) per form and capability under leave-one-run-out (LOCO) and leave-one-student-out (LOSO). Forms: $t=\log(1+T_c/35000)$, $e=\log(1+E)$, $s=1-e^{-T_c/T_*}$; T-only $a+bt$; E-only $a+be$; surface $a+bt+ce+dz$; F1 $(a+\lambda z)e$; F2 $(a+\lambda z)s+(b+\mu z)e$, $z$ = standardized $L_{0,c}$ (or $\log N_S$, identical up to sign with two students). Under LOSO the descriptor forms degenerate (one student in training) and fall back to zero.", "tab:v56_forms", "@{}lr" + "r" * (len(tA[0].split("|")) - 4) + "@{}")
tex += to_tex(tB, r"Capability conditioning at matched parameter count (leave-one-run-out MAE, nats): shared response shape with per-capability offsets versus per-capability models with the same total count; gain $>0.02$ nats is a descriptive threshold.", "tab:v56_condition", "@{}lll" + "r" * (len(tB[0].split("|")) - 5) + "@{}")
(TAB / "v56_forms.tex").write_text(tex)
print("wrote models_domains, quant2d_coef, v53_loso, v55_loso, v56_forms")
