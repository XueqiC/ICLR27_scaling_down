#!/usr/bin/env python3
"""Authoritative prediction-comparison tables (CPU only). Supersedes the presentation of v45_main_table.py (kept unchanged).
Each test = one group of three lines: candidate / strongest same-input baseline / strongest simple baseline, per capability.
Origin code = prediction origin (P frozen prospective, L leave-one-out on an existing panel) · same-input baseline origin ·
simple baseline origin (F frozen or pre-specified together with the candidate, R computed after the test, - none) ·
selection (A all pre-specified forms reported, S per-capability choice made after the test, D chosen on the development set).
Improvement = MAE(strongest baseline) - MAE(candidate), positive = candidate better. Outputs JSON/CSV + two LaTeX tables."""
import json, csv, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"; OUT = R / "v52-prediction-tables"; OUT.mkdir(exist_ok=True)
TAB = ROOT / "paper/tables"; CAPS = ("math", "code", "qa"); J = lambda p: json.loads((R / p).read_text())
NAMES = {"config_median": "per-config median", "config_mean": "per-config mean", "zero": "zero change", "so": "strength-only",
         "med": "median curve", "strength_only": "strength-only", "median_curve": "median curve", "A1": "A1 ($\\gamma{=}1$)",
         "A1_gamma1": "A1 ($\\gamma{=}1$)", "A2": "A2 (per-$d$ regr.)", "cont": "continuous 2-term", "noD0": "no-$D_0$",
         "noD0_N0_L0": "no-$D_0$", "per_bit_median": "per-bit median", "per_bit_mean": "per-bit mean", "mean_base": "constant (mean)",
         "med_base": "constant (median)", "constant": "constant", "T": "$T$-only", "2D": "$T{+}E$ joint", "E": "$E$-only"}
groups = []
def add(axis, method, test, unseen, cand_name, cand, same, simple, origin, src, freeze, n, ci=None, note="", main=True):
    """cand/same/simple: dict cap->mae ; same/simple may carry per-cap names via tuples (name, mae)."""
    groups.append(dict(axis=axis, method=method, test=test, unseen=unseen, cand_name=cand_name, cand=cand, same=same, simple=simple,
                       origin=origin, src=src, freeze=freeze, n=n, ci=ci or {}, note=note, main=main))
def best(d, keys):  # returns cap -> (name, mae) with the smallest mae among keys
    return {c: min(((k, d[c][k]) for k in keys if k in d[c] and d[c][k] is not None), key=lambda t: t[1]) for c in CAPS}
# ---------- SOURCE AXIS ----------
v36 = J("v36b-input-comparison/summary.json")["results"]
for arm, label in (("pruning", "pruning"), ("quantization", "quantization")):
    cand, same, simple, ci = {}, {}, {}, {}
    for c in CAPS:
        cmp = v36[arm][c]["comparison"]; core = cmp["core_table"]
        cand[c] = core["N0_D0_L0"]["mae"]; same[c] = ("noD0_N0_L0", core["N0_L0"]["mae"])
        s = cmp["strongest_simple_baseline"]; simple[c] = (s, cmp["baseline_candidates"][s]["mae"])
        lo, hi = core["N0_D0_L0"]["improvement_ci95"]; ci[c] = "ci_excl0" if lo > 0 else "ci_incl0"
    add("source", label, f"source transfer at fixed {'$d$' if arm=='pruning' else '$b$'}: $3\\times3$ Pythia panel, leave-one-step-out",
        "held-out pretraining stage (in-range)", "config-indicator OLS $\\{N_0,L_0,D_0\\}$", cand, same, simple, "L·F·R·A",
        "v36b-input-comparison/summary.json", "retrospective", "36 cells / 3 folds", ci)
v38 = J("v38-prospective/compare.json")["arms"]["pruning"]
add("source", "pruning", "new stage 96k on 160M and 1.4B, all measured $d$", "stage 96k (interpolation between 64k and 143k)",
    "config-indicator OLS $\\{N_0,L_0,D_0\\}$", {c: v38[c]["mae_full_N0_D0_L0"] for c in CAPS}, {c: ("noD0_N0_L0", v38[c]["mae_noD0_N0_L0"]) for c in CAPS},
    {c: ("config_median", v38[c]["mae_baseline_config_median"]) for c in CAPS}, "P·F·R·A", "v38-prospective/compare.json", "4e0bf2c (analysis repo)", "8 cells per capability")
v44 = J("v44-quant-partition/summary.json")["partition"]
for reg, lab in (("ge4", "$b\\ge4$ (near-zero regime)"), ("int3", "int3 (collapse regime)")):
    add("source", "quantization", f"new stage 96k on 160M, 410M, 1.4B; {lab}", "stage 96k (interpolation)", "config-indicator OLS $\\{N_0,L_0,D_0\\}$",
        {c: v44[f"{c}_{reg}"]["cand_mae"] for c in CAPS}, {c: None for c in CAPS}, {c: ("per_bit_median", v44[f"{c}_{reg}"]["median_mae"]) for c in CAPS},
        "P·–·R·A", "v44-quant-partition/summary.json", "4e0bf2c + 20542e2 (analysis repo)", "3 sources")
v39 = J("v39-distill-controlled/summary.json")["by_cap"]
add("source", "distillation", "$\\delta_c$ source transfer: $3\\times3$ LoRA panel, leave-one-step-out", "held-out pretraining stage (in-range)",
    "linear $\\{N_0,L_0,D_0\\}$", {c: v39[c]["step"]["mae_D0"] for c in CAPS}, {c: ("noD0_N0_L0", v39[c]["step"]["mae_noD0"]) for c in CAPS},
    {c: (v39[c]["step"]["strongest_baseline"], v39[c]["step"]["mae_baseline"]) for c in CAPS}, "L·F·R·A", "v39-distill-controlled/summary.json", "retrospective", "9 cells / 3 folds")
# ---------- CONFIGURATION AXIS ----------
v42 = J("v42-prune-sameinput/summary.json")["by_cap"]
m = {c: v42[c]["mae"] for c in CAPS}
add("config", "pruning", "unseen $d$ 0.65 and 0.55, dev sources + 410M@96k", "densities 0.65 (interp.) and 0.55 (extrap.) pooled",
    "shared power $A_c(\\mathbf x)((1-d)/0.3)^{\\gamma_c}$", {c: m[c]["cand"] for c in CAPS}, best(m, ("A1", "A2")), best(m, ("so", "med", "zero")),
    "P·R·F·A", "v42-prune-sameinput/summary.json", "20542e2 (analysis repo)", "24 points")
v46 = J("v46-p1-newsource/compare.json")
def pool(rows, keys):
    out = {c: {} for c in CAPS}
    for c in CAPS:
        rr = [r for r in rows if r["cap"] == c]
        for k in keys: out[c][k] = sum(r[k]["abs"] for r in rr) / len(rr) if rr else None
    return out
for d, lab in ((0.65, "$d=0.65$ (interp.)"), (0.55, "$d=0.55$ (extrap.)")):
    p = pool([r for r in v46["pruning"] if abs(r["d"] - d) < 1e-9], ("power", "A2", "A1_gamma1", "strength_only", "median_curve", "zero"))
    add("config", "pruning", f"new source 1B@96k, {lab}", "in-range size; stage 96k",
        "shared power (frozen v40)", {c: p[c]["power"] for c in CAPS}, best(p, ("A2", "A1_gamma1")), best(p, ("strength_only", "median_curve", "zero")),
        "P·F·F·A", "v46-p1-newsource/compare.json", "b1bf631 (paper repo)", "1 source-state", main=False)
for b in (4, 3):
    p = pool([r for r in v46["quantization"] if r["bit"] == b], ("full_N0_L0_D0", "noD0_N0_L0", "per_bit_median", "zero"))
    add("config", "quantization", f"new source 1B@96k, int{b}", "in-range size; stage 96k; seen bit", "config-indicator OLS $\\{N_0,L_0,D_0\\}$",
        {c: p[c]["full_N0_L0_D0"] for c in CAPS}, {c: ("noD0_N0_L0", p[c]["noD0_N0_L0"]) for c in CAPS}, best(p, ("per_bit_median", "zero")),
        "P·F·F·A", "v46-p1-newsource/compare.json", "b1bf631 (paper repo)", "1 source-state", main=False)
for size, lab, unseen, fz in (("1b", "1B@32k+112k", "in-range size; stages 32k, 112k", "cd9ed8f (paper repo)"),
                              ("6.9b", "6.9B@32k+112k", "size $5\\times$ beyond dev range; stages 32k, 112k", "950312f (paper repo)")):
    files = sorted(glob.glob(str(R / f"v49-p1v2/compare_pythia-{size}@step*.json")))
    if len(files) != 2: continue
    A = [json.loads(Path(f).read_text())["protocols"]["A"] for f in files]
    pr = [r for a in A for r in a["rows_prune"]]; qr = [r for a in A for r in a["rows_quant"]]
    for reg, sel, rl in (("interp", lambda r: 0.6 <= r["d"] <= 0.9, "$d$ 0.9--0.6 (interp.)"), ("extrap", lambda r: abs(r["d"] - 0.55) < 1e-9, "$d=0.55$ (extrap.)")):
        p = pool([r for r in pr if sel(r)], ("power", "A2", "A1", "cont", "strength_only", "median_curve", "zero"))
        add("config", "pruning", f"new source {lab}, {rl}", unseen, "shared power (frozen v49)",
            {c: p[c]["power"] for c in CAPS}, best(p, ("A2", "A1", "cont")), best(p, ("strength_only", "median_curve", "zero")), "P·F·F·A",
            f"v49-p1v2/compare_pythia-{size}@step*.json", fz, "2 source-states")
    for reg, sel, rl in (("ge4", lambda r: r["bit"] >= 4, "$b\\ge4$ (incl.\\ 5-bit rule)"), ("int3", lambda r: r["bit"] == 3, "int3")):
        p = pool([r for r in qr if sel(r)], ("full", "noD0", "per_bit_mean", "per_bit_median", "zero"))
        add("config", "quantization", f"new source {lab}, {rl}", unseen, "config-indicator OLS $\\{N_0,L_0,D_0\\}$", {c: p[c]["full"] for c in CAPS},
            {c: ("noD0", p[c]["noD0"]) for c in CAPS}, best(p, ("per_bit_mean", "per_bit_median", "zero")), "P·F·F·A", f"v49-p1v2/compare_pythia-{size}@step*.json", fz, "2 source-states")
v41 = J("v41-distill-newpool/summary.json")["by_cap"]; m = {c: v41[c]["mae"] for c in CAPS}
add("config", "distillation", "unseen pool U225 from endpoint pools U75/U600 (Gemma-3-1B)", "pool (reuse count in-range; sampling protocol differs)",
    "$E$-only (reuse count)", {c: m[c]["E"] for c in CAPS}, best(m, ("T", "2D")), {c: ("constant", m[c]["constant"]) for c in CAPS}, "P·F·F·A",
    "v41-distill-newpool/summary.json", "0bbbaa8 (analysis repo)", "6 runs (3 pools $\\times$ 2 seeds)",
    note="per-capability recommendation (E for QA, constant for math/code) is a post-test selection")
# ---------- derived fields ----------
for g in groups:
    g["improvement"] = {}; g["outcome"] = {}
    for c in CAPS:
        cands = [v[1] for v in (g["same"][c], g["simple"][c]) if v]; strongest = min(cands) if cands else None
        g["improvement"][c] = (strongest - g["cand"][c]) if strongest is not None else None
        g["outcome"][c] = None if strongest is None else ("better" if g["cand"][c] < strongest else "not better")
json.dump(groups, open(OUT / "groups.json", "w"), indent=1)
with open(OUT / "rows.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["axis", "method", "test", "unseen", "line", "name", "math", "code", "qa", "origin", "src", "freeze", "n"])
    for g in groups:
        w.writerow([g["axis"], g["method"], g["test"], g["unseen"], "candidate", g["cand_name"], *[g["cand"][c] for c in CAPS], g["origin"], g["src"], g["freeze"], g["n"]])
        for key in ("same", "simple"):
            w.writerow([g["axis"], g["method"], g["test"], g["unseen"], key, "|".join(sorted({g[key][c][0] for c in CAPS if g[key][c]})), *[(g[key][c][1] if g[key][c] else "") for c in CAPS], "", "", "", ""])
# ---------- LaTeX ----------
def fmt(v): return "--" if v is None else f"{v:.3f}"
SHORT = {"config-indicator OLS $\\{N_0,L_0,D_0\\}$": "config-indicator OLS", "linear $\\{N_0,L_0,D_0\\}$": "linear $\\{N_0,L_0,D_0\\}$",
         "shared power $A_c(\\mathbf x)((1-d)/0.3)^{\\gamma_c}$": "shared power form", "shared power (frozen v40)": "shared power form", "shared power (frozen v49)": "shared power form"}
def table(sel, label, caption, methods_rule=True):
    L = [r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.02}",
         r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.1cm}>{\raggedright\arraybackslash}p{3.1cm}rrr@{}}", r"\toprule",
         r"Test; \emph{unseen axis}; [origin] & Line & math & code & QA \\", r"\midrule"]
    cur = None
    for g in [g for g in groups if sel(g)]:
        if cur and cur != g["method"] and methods_rule: L.append(r"\midrule")
        cur = g["method"]
        mins = {c: min([g["cand"][c]] + [v[1] for v in (g["same"][c], g["simple"][c]) if v]) for c in CAPS}
        def cell(v, c, extra=""):
            t = fmt(v); t = r"\textbf{" + t + "}" if v is not None and abs(v - mins[c]) < 5e-4 else t; return t + extra
        cc = [cell(g["cand"][c], c, (f" ({g['improvement'][c]:+.2f})" if g["improvement"][c] is not None else "") + (r"$^{*}$" if g["ci"].get(c) == "ci_excl0" else "")) for c in CAPS]
        head = f"\\textbf{{{g['method']}}}: {g['test']}\\newline{{\\itshape {g['unseen']}}} [{g['origin']}]"
        L.append(f"{head} & cand.: {SHORT.get(g['cand_name'], g['cand_name'])} & {cc[0]} & {cc[1]} & {cc[2]} \\\\")
        for key, lab in (("same", "same-input"), ("simple", "simple")):
            names = sorted({NAMES.get(g[key][c][0], g[key][c][0]) for c in CAPS if g[key][c]})
            nm = "; ".join(names) if names else "none available"
            L.append(f" & {lab}: {nm} & " + " & ".join(cell(g[key][c][1], c) if g[key][c] else "--" for c in CAPS) + r" \\")
        L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule\end{tabular}", r"\caption{" + caption + "}", f"\\label{{{label}}}\\end{{table}}"]
    return "\n".join(L) + "\n"
LEG = (r" MAE in nats/token per capability; bold = smallest in the group; the number in parentheses is the improvement of the candidate over the strongest listed baseline "
       r"(positive = candidate better); $^{*}$ = bootstrap 95\% interval of that improvement excludes zero (computed only for the leave-one-out panels). Origin code: prediction origin "
       r"(P frozen prospective, L leave-one-out) $\cdot$ same-input baseline origin $\cdot$ simple baseline origin (F pre-specified with the candidate, R computed after the test, -- none) $\cdot$ "
       r"selection (A: all pre-specified forms reported). Where several same-input or simple baselines exist, the strongest per capability is shown and named; that choice is retrospective. "
       r"Config-indicator OLS uses $\{N_0,L_0,D_0\}$ with one coefficient set per configuration. Generated by \texttt{v52\_prediction\_tables.py}.")
(TAB / "pred_source.tex").write_text(table(lambda g: g["axis"] == "source", "tab:pred_source", r"Source axis: does the pre-compression source state predict the response at a fixed intervention setting?" + LEG))
(TAB / "pred_config_prune.tex").write_text(table(lambda g: g["axis"] == "config" and g["method"] == "pruning" and g["main"], "tab:pred_config_prune", r"Configuration axis, pruning: does a relation fit at seen densities and sources predict unseen densities, sizes and stages? Protocol A (full development panel) for the two P1-v2 pairs; protocol B in Appendix Table~\ref{tab:p1v2}." + LEG))
(TAB / "pred_config_qd.tex").write_text(table(lambda g: g["axis"] == "config" and g["method"] != "pruning" and g["main"], "tab:pred_config_qd", r"Configuration axis, quantization and distillation: unseen sizes and stages at seen bit-widths (protocol A; the 5-bit prediction is a fixed interpolation rule), and an unseen distillation pool." + LEG))
(TAB / "pred_config_full.tex").write_text(table(lambda g: g["axis"] == "config", "tab:pred_config_full", r"Configuration axis, all registered tests including the single-stage Pythia-1B@96k prospective (per capability)." + LEG))
print(len(groups), "groups ->", OUT, TAB / "pred_source.tex", TAB / "pred_config_prune.tex", TAB / "pred_config_qd.tex")
for g in groups: print(f"{g['axis']:6s} {g['method']:12s} {g['test'][:58]:58s} cand={[round(g['cand'][c],3) for c in CAPS]} impr={[round(g['improvement'][c],3) if g['improvement'][c] is not None else None for c in CAPS]} {g['origin']}")
