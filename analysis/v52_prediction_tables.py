#!/usr/bin/env python3
"""Authoritative prediction-comparison tables (CPU only). Supersedes the presentation of v45_main_table.py.
Each test = one group of three lines: candidate / strongest same-input baseline / strongest simple baseline, per capability.
Origin code = prediction origin (P frozen before measurement, R retrospective, including leave-one-out) · same-input baseline origin ·
simple baseline origin (F frozen or pre-specified together with the candidate, R computed after the test, - none) ·
selection (A all pre-specified forms reported, S per-capability choice made after the test).
Origin records the process only: a frozen prediction stays P whether it wins, ties, or loses; outcome is separate.
Improvement = MAE(strongest baseline) - MAE(candidate), positive = candidate better. Outputs JSON/CSV + two LaTeX tables."""

try:
    from .paper_table_text import table_text as publication_table_text
except ImportError:  # Direct scripts and file-based imports.
    try:
        from analysis.paper_table_text import table_text as publication_table_text
    except ImportError:
        from paper_table_text import table_text as publication_table_text

import json, csv, glob, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"; OUT = R / "v52-prediction-tables"; OUT.mkdir(exist_ok=True)
TAB = ROOT / "paper/paper/tables"; CAPS = ("math", "code", "qa"); J = lambda p: json.loads((R / p).read_text())
NAMES = {"config_median": "per-config median", "config_mean": "per-config mean", "zero": "zero change", "so": "strength-only",
         "med": "median curve", "strength_only": "strength-only", "median_curve": "median curve", "A1": "A1 ($\\gamma{=}1$)",
         "A1_gamma1": "A1 ($\\gamma{=}1$)", "A2": "A2 (per-$d$ regr.)", "cont": "continuous 2-term", "noD0": "no-$D_0$",
         "noD0_N0_L0": "no-$D_0$", "per_bit_median": "per-bit median", "per_bit_mean": "per-bit mean", "mean_base": "constant (mean)",
         "med_base": "constant (median)", "constant": "constant", "T": "$T$-only", "2D": "$T{+}E$ joint", "E": "$E$-only"}
groups = []
def add(axis, method, test, unseen, cand_name, cand, same, simple, origin, src, freeze, n, ci=None, note="", main=True):
    """cand/same/simple: dict cap->mae ; same/simple may carry per-cap names via tuples (name, mae)."""
    # Supply origin from the recorded process, independently of errors and improvements.
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
        "held-out pretraining stage (in-range)", "config-indicator OLS $\\{N_0,L_0,D_0\\}$", cand, same, simple, "R·F·R·A",
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
    {c: (v39[c]["step"]["strongest_baseline"], v39[c]["step"]["mae_baseline"]) for c in CAPS}, "R·F·R·A", "v39-distill-controlled/summary.json", "retrospective", "9 cells / 3 folds")
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
# Match v62_p2v2_test_table.py: average summary rows within (student, role, cap).
v50_freeze = J("v50-p2v2/freeze.json"); v50_test = J("v50-p2v2/compare_test.json")
selected = {c: min(v50_freeze["fits"][c], key=lambda form: v50_freeze["fits"][c][form]["dev_mae"]) for c in CAPS}
assert all(form == "joint+src" for form in selected.values()), "Frozen dev-selected headline form changed"
agg = collections.defaultdict(lambda: collections.defaultdict(list)); zero = collections.defaultdict(list); npts = collections.Counter()
for key, row in v50_test["summary"].items():
    student, role, cap, budget = key.split("|")
    for form, mae in row["mae"].items(): agg[(student, role, cap)][form].append(mae)
    zero[(student, role, cap)].append(row["mae_zero"]); npts[(student, role, cap)] += row["n_pools"]
for student, label in (("gemma3-270m", "Gemma-3-270M"), ("gemma3-1b", "Gemma-3-1B"),
                       ("gemma3-4b", "held-out student Gemma-3-4B")):
    cand, same, simple = {}, {}, {}
    for c in CAPS:
        key = (student, "test_pool", c)
        mae = {form: sum(values) / len(values) for form, values in agg[key].items()}
        cand[c] = mae[selected[c]]; same[c] = ("constant", mae["constant"])
        simple[c] = ("zero", sum(zero[key]) / len(zero[key]))
        assert npts[key] == 12, f"Unexpected test point count: {key}"
    add("config", "distillation (uniform absolute-exposure protocol)",
        f"multi-student distillation, unseen pool U375, {label}",
        "pool; held-out student" if student == "gemma3-4b" else "pool",
        "joint form with source term (dev-selected, R)", cand, same, simple, "P/F/F/A",
        "results/v50-p2v2/compare_test.json (c06c856)", "freeze.json 33c706c", npts[(student, "test_pool", CAPS[0])],
        note="Headline form minimizes frozen per-capability dev_mae; selection rule stated after the tests (R). 12 points per capability.")
# ---------- derived fields ----------
for g in groups:
    g["improvement"] = {}; g["outcome"] = {}
    for c in CAPS:
        cands = [v[1] for v in (g["same"][c], g["simple"][c]) if v]; strongest = min(cands) if cands else None
        g["improvement"][c] = (strongest - g["cand"][c]) if strongest is not None else None
        g["outcome"][c] = None if strongest is None else ("better" if g["cand"][c] < strongest else "not better")
json.dump(groups, open(OUT / "groups.json", "w"), indent=1)
(TAB / "groups.json").write_bytes((OUT / "groups.json").read_bytes())
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
def table(sel, label, caption, methods_rule=True, page_break_method=None):
    L = [r"\begin{table}[p]\centering\footnotesize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.02}",
         r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.1cm}>{\raggedright\arraybackslash}p{3.1cm}rrr@{}}", r"\toprule",
         r"Test; \emph{unseen axis}; [origin] & Line & math & code & QA \\", r"\midrule"]
    header = L.copy()
    cur = None
    for g in [g for g in groups if sel(g)]:
        if cur and cur != g["method"] and g["method"] == page_break_method:
            # Continue the same table on a second float; retain one caption and table number.
            L += [r"\bottomrule\end{tabular}", r"\end{table}"] + header
            cur = None
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
       r"(P frozen before measurement regardless of outcome, R retrospective, including leave-one-out) $\cdot$ same-input baseline origin $\cdot$ simple baseline origin (F pre-specified with the candidate, R computed after the test, -- none) $\cdot$ "
       r"selection (A: all pre-specified forms reported; S: per-capability choice made after the test). Where several same-input or simple baselines exist, the strongest per capability is shown and named; that choice is retrospective. "
       r"Config-indicator OLS uses $\{N_0,L_0,D_0\}$ with one coefficient set per configuration. Generated by \texttt{v52\_prediction\_tables.py}.")
(TAB / "pred_source.tex").write_text(publication_table_text(table(lambda g: g["axis"] == "source", "tab:pred_source", r"Source axis: does the pre-compression source state predict the response at a fixed intervention setting?" + LEG)))
(TAB / "pred_config_prune.tex").write_text(publication_table_text(table(lambda g: g["axis"] == "config" and g["method"] == "pruning", "tab:pred_config_prune", r"Configuration axis, pruning: does a relation fit at seen densities and sources predict unseen densities, sizes and stages? Protocol A (full development panel) for the two P1-v2 pairs and the single-stage 1B@96k prospective; protocol B in Table~\ref{tab:p1v2}. Format, bold, improvement in parentheses, and origin code as in Table~\ref{tab:pred_source}.")))
(TAB / "pred_config_qd.tex").write_text(publication_table_text(table(lambda g: g["axis"] == "config" and g["method"] != "pruning", "tab:pred_config_qd", r"Configuration axis, quantization and distillation (continued from the preceding table block): unseen sizes and stages at seen bit-widths (protocol A; the 5-bit prediction is a fixed interpolation rule), and unseen distillation pools. The U375 headline is the frozen form with the lowest per-capability development MAE (joint with a source term for every capability); this selection rule was stated after the tests (R). Each U375 group has 12 points per capability; Gemma-3-4B is a held-out student. Format, bold, improvement in parentheses, and origin code as in Table~\ref{tab:pred_source}.", page_break_method="distillation (uniform absolute-exposure protocol)")))

# ---------- compact main-text table: one row per test ----------
CODE = {"config_median": "med", "config_mean": "mean", "zero": "0", "so": "so", "med": "med", "strength_only": "so", "median_curve": "med", "A1": "A1", "A1_gamma1": "A1",
        "A2": "A2", "cont": "ct", "noD0": "nD", "noD0_N0_L0": "nD", "per_bit_median": "pbm", "per_bit_mean": "pba", "mean_base": "c", "med_base": "c", "constant": "c", "T": "T", "2D": "TE", "E": "E"}
SHORT = [  # (substring of g["test"], short label, unseen tag)
    ("source transfer at fixed $d$", "prune: fixed $d$, $3{\\times}3$, leave-one-stage-out", "stage"),
    ("source transfer at fixed $b$", "quant: fixed $b$, $3{\\times}3$, leave-one-stage-out", "stage"),
    ("new stage 96k on 160M and 1.4B", "prune: stage 96k (160M, 1.4B), all $d$", "stage"),
    ("$b\\ge4$ (near-zero regime)", "quant: new stage 96k (160M/410M/1.4B), $b\\ge4$", "stage"),
    ("int3 (collapse regime)", "quant: new stage 96k (160M/410M/1.4B), int3", "stage"),
    ("$\\delta_c$ source transfer", "distill: $\\delta_c$, $3{\\times}3$ LoRA, leave-one-stage-out", "stage"),
    ("unseen $d$ 0.65 and 0.55", "prune: $d$ 0.65/0.55, dev sources + 410M@96k", "$d$"),
    ("1B@96k, $d=0.65$", "prune: 1B@96k, $d=0.65$", "size, stage, $d$"),
    ("1B@96k, $d=0.55$", "prune: 1B@96k, $d=0.55$", "size, stage, $d$"),
    ("1B@96k, int4", "quant: 1B@96k, int4", "size, stage"),
    ("1B@96k, int3", "quant: 1B@96k, int3", "size, stage"),
    ("1B@32k+112k, $d$ 0.9--0.6", "prune: 1B@32k+112k, $d$ 0.9--0.6", "size, stage"),
    ("1B@32k+112k, $d=0.55$", "prune: 1B@32k+112k, $d=0.55$", "size, stage, $d$"),
    ("6.9B@32k+112k, $d$ 0.9--0.6", "prune: 6.9B@32k+112k, $d$ 0.9--0.6", "size$\\uparrow$, stage"),
    ("6.9B@32k+112k, $d=0.55$", "prune: 6.9B@32k+112k, $d=0.55$", "size$\\uparrow$, stage, $d$"),
    ("1B@32k+112k, $b\\ge4$", "quant: 1B@32k+112k, $b\\ge4$ (5-bit rule incl.)", "size, stage, $b$"),
    ("1B@32k+112k, int3", "quant: 1B@32k+112k, int3", "size, stage"),
    ("6.9B@32k+112k, $b\\ge4$", "quant: 6.9B@32k+112k, $b\\ge4$", "size$\\uparrow$, stage, $b$"),
    ("6.9B@32k+112k, int3", "quant: 6.9B@32k+112k, int3", "size$\\uparrow$, stage"),
    ("unseen pool U225", "distill: unseen pool U225 (Gemma-3-1B)", "pool"),
    ("unseen pool U375, Gemma-3-270M", "KD 270M, pool 375", "pool"),
    ("unseen pool U375, Gemma-3-1B", "KD 1B, pool 375", "pool"),
    ("unseen pool U375, held-out student Gemma-3-4B", "KD 4B, pool 375", "student, pool"),
]
def compact():
    L = [r"\begin{table}[!t]\centering\footnotesize\setlength{\tabcolsep}{2.5pt}\renewcommand{\arraystretch}{1.02}",
         r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.15cm}>{\raggedright\arraybackslash}p{1.4cm}lccc@{}}", r"\toprule",
         r"Test & unseen & origin & math & code & QA \\", r"\midrule"]
    last_axis = None; last_method = None
    for g in groups:
        if g["axis"] != last_axis:
            L.append(r"\multicolumn{6}{@{}l}{\textit{" + ("Source axis: seen setting, unseen source state" if g["axis"] == "source" else "Configuration axis: unseen setting, size, stage, or pool") + r"}} \\")
            last_axis = g["axis"]; last_method = None
        if last_method and last_method != g["method"]: L.append(r"\addlinespace[1.5pt]")
        last_method = g["method"]
        short, tag = next(((sh, tg) for sub, sh, tg in SHORT if sub in g["test"]), (g["test"], ""))
        cells = []
        for c in CAPS:
            cand = g["cand"][c]; alts = [v for v in (g["same"][c], g["simple"][c]) if v]
            bname, bmae = min(alts, key=lambda t: t[1]) if alts else ("--", None)
            cs = f"{cand:.3f}"; bs = f"{bmae:.3f}" if bmae is not None else "--"
            if bmae is not None and cand < bmae: cs = r"\textbf{" + cs + "}"
            elif bmae is not None: bs = r"\textbf{" + bs + "}"
            cells.append(f"{cs}/{bs}{{\\scriptsize\\,{CODE.get(bname, bname)}}}")
        L.append(f"{short} & {tag} & {g['origin'].replace('·','')} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule\end{tabular}",
          r"\caption{\scriptsize Prediction tests, all three arms (MAE in nats/token per capability, candidate\,/\,strongest baseline, bold = better). The candidate is the source-conditioned predictor of each arm "
          r"(configuration-indicator regression on the source axis and for quantization; shared power form for pruning on the configuration axis; $E$-only form for the U225 distillation pool test; joint form with a source term for the multi-student U375 tests). "
          r"For U375, the headline is the frozen form with the lowest per-capability development MAE; this selection rule was stated after the tests (R). Each U375 group has 12 points per capability; ``student'' marks the held-out Gemma-3-4B. "
          r"The baseline shown is the stronger of the same-input and the simple baseline per capability, named by code: nD = no-$D_0$ variant, med = per-configuration median or median development curve, "
          r"0 = zero change, so = strength-only curve, A1 = $\gamma{=}1$ power form, A2 = per-density regression with fixed interpolation, ct = continuous two-term form, pbm/pba = per-bit median/mean, "
          r"c = constant, TE = joint $T{+}E$ form. ``unseen'' lists what the test holds out (size$\uparrow$ = five times beyond the development range). Origin code, four letters: prediction (P frozen before measurement regardless of outcome, R retrospective, including leave-one-out), same-input baseline, simple baseline (F pre-specified with the candidate, R computed after the test, -- none), selection (A: all pre-specified forms reported; S: per-capability choice made after the test). "
          r"Improvements, both baselines separately, and bootstrap intervals are in Appendix Tables~\ref{tab:pred_source}--\ref{tab:pred_config_qd}. Generated by \texttt{v52\_prediction\_tables.py}.}",
          r"\label{tab:pred_main}\end{table}"]
    return "\n".join(L) + "\n"
(TAB / "pred_main.tex").write_text(compact())

print(len(groups), "groups ->", OUT, TAB / "pred_source.tex", TAB / "pred_config_prune.tex", TAB / "pred_config_qd.tex")
for g in groups: print(f"{g['axis']:6s} {g['method']:12s} {g['test'][:58]:58s} cand={[round(g['cand'][c],3) for c in CAPS]} impr={[round(g['improvement'][c],3) if g['improvement'][c] is not None else None for c in CAPS]} {g['origin']}")
