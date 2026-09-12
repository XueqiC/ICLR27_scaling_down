#!/usr/bin/env python3
"""Closeout main prediction table. CPU-only; reads ONLY existing result JSONs (no weights/GPU).

One row per (method, capability, test) for the frozen/registered prediction tests that carry the paper's
claims. Every number is read from a results/v*/summary.json (or compare.json) so the manuscript table is
machine-traceable. Emits CSV + JSON + a booktabs LaTeX table (paper/paper/tables/main_prediction.tex).

Columns: method | capability | test | candidate (free params) | inputs/budget | split | interp/extrap |
cand MAE | strongest compatible baseline (name, MAE) | paired improvement = base-cand | status | verdict.
Status: FROZEN-PROSPECTIVE (prediction registered before measurement), LOO-RETRO (held-out on an existing
panel), RETRO-DIAG (baseline/diagnostic added in the closeout, original prediction untouched).
"""
import json, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT_DIR = R / "v45-main-table"
TEX = ROOT / "paper/paper/tables/main_prediction.tex"
CAPS = ("math", "code", "qa")

def J(p): return json.loads((R / p).read_text())

def rows():
    out = []
    # --- Pruning: source-transfer at FIXED density (controlled panel, leave-one-step-out), v36b
    v36b = J("v36b-input-comparison/summary.json")["results"]["pruning"]
    for c in CAPS:
        cmp = v36b[c]["comparison"]; full = cmp["core_table"]["N0_D0_L0"]; strong = cmp["strongest_simple_baseline"]
        base_mae = cmp["baseline_candidates"][strong]["mae"]
        out.append(dict(method="pruning", capability=c, test="source transfer, fixed $d$ (3$\\times$3, leave-one-step-out)",
            candidate="config-indicator OLS {N0,L0,D0} (16 coeffs/arm-cap)", inputs="K0", split="source", ie="interp",
            cand_mae=full["mae"], base_name=f"per-config {strong.replace('config_','')}", base_mae=base_mae, status="LOO-RETRO",
            ci=f"[{full['improvement_ci95'][0]:+.2f},{full['improvement_ci95'][1]:+.2f}] (3 step-folds)"))
    # --- Pruning: strength axis, v40 frozen + v42 same-input A2
    v42 = J("v42-prune-sameinput/summary.json")["by_cap"]
    for c in CAPS:
        m = v42[c]["mae"]
        out.append(dict(method="pruning", capability=c, test="unseen $d$ 0.65/0.55, incl.\\ new source",
            candidate="A_c(x)*((1-d)/0.3)^g_c, shared g (5 params/cap)", inputs="K0", split="strength+source",
            ie="interp(0.65)+extrap(0.55)", cand_mae=m["cand"], base_name="A2: per-$d$ regr.\\ + lin.\\ interp",
            base_mae=m["A2"], status="FROZEN-PROSPECTIVE (v40) / A2 RETRO-DIAG", ci="point est.; per-source in PRUNE_SAMEINPUT"))
    # --- Quantization: per-bit partition on new @96k sources, v44
    v44 = J("v44-quant-partition/summary.json")["partition"]
    for c in CAPS:
        for reg, lab in (("ge4", r"$\ge$4-bit"), ("int3", "int3")):
            p = v44[f"{c}_{reg}"]
            out.append(dict(method="quantization", capability=c, test=f"source transfer, fixed bit, {lab} (3 new sources)",
                candidate="config-indicator OLS {N0,L0,D0}", inputs="K0", split="source", ie="interp",
                cand_mae=p["cand_mae"], base_name="per-bit median", base_mae=p["median_mae"],
                status="FROZEN(v38 160M/1.4B)+v40(410M) / RETRO-DIAG table", ci="point est.; per-source in QUANT_PARTITION"))
    # --- Distillation: controlled source-transfer (3x3 LoRA panel), v39
    v39 = J("v39-distill-controlled/summary.json")["by_cap"]
    for c in CAPS:
        s = v39[c]["step"]
        out.append(dict(method="distillation", capability=c, test="$\\delta$ source transfer (3$\\times$3 LoRA, leave-one-step-out)",
            candidate="linear {N0,L0,D0} (4 params)", inputs="K0", split="source", ie="interp",
            cand_mae=s["mae_D0"], base_name=f"constant ({s['strongest_baseline'].replace('_base','').replace('med','median')})", base_mae=s["mae_baseline"],
            status="LOO-RETRO", ci="point est.; 3 step-folds"))
    # --- Distillation: new pool U=225, v41 frozen + v43 residual
    v41 = J("v41-distill-newpool/summary.json")["by_cap"]
    for c in CAPS:
        m = v41[c]["mae"]; best = "E" if c == "qa" else "constant"
        out.append(dict(method="distillation", capability=c, test="unseen pool U225 from U75/U600",
            candidate=f"{best} (4 candidates frozen = P; per-cap choice made after test = R)", inputs="K0", split="config(pool)", ie="interp (E in range)",
            cand_mae=m[best], base_name="constant" if best != "constant" else "E-only", base_mae=m["constant"] if best != "constant" else m["E"],
            status="P (4 frozen candidates, v41) + R (per-cap selection; residuals v43)", ci="3 pools x 2 seeds; bias-dominated (DISTILL_RESIDUALS)"))
    # --- P-new (A100 supplement): pythia-1b@step96000, predictions frozen before measurement (v46)
    try:
        c46 = J("v46-p1-newsource/compare.json")
        for d in (0.65, 0.55):
            rows_ = [r for r in c46["pruning"] if r["d"] == d]
            m = lambda k: sum(r[k]["abs"] for r in rows_) / len(rows_)
            same = min(("A2", "A1_gamma1"), key=m); strongest = min(("A2", "A1_gamma1", "strength_only", "median_curve", "zero"), key=m)
            out.append(dict(method="pruning", capability="all", test=f"NEW source pythia-1b@96k, d={d} ({'interp' if d==0.65 else 'extrap'})",
                candidate="frozen shared power (v40)", inputs="K0", split="source+strength", ie="interp" if d == 0.65 else "extrap",
                cand_mae=m("power"), base_name=f"strongest: {strongest}" + (f" (same-input best {same} {m(same):.3f})" if strongest != same else " (same-input)"),
                base_mae=m(strongest), status="P-NEW (v46, frozen b1bf631)", ci="one source-state; 3 caps"))
        for b in (4, 3):
            rows_ = [r for r in c46["quantization"] if r["bit"] == b]
            m = lambda k: sum(r[k]["abs"] for r in rows_) / len(rows_)
            strongest = min(("noD0_N0_L0", "per_bit_median", "zero"), key=m)
            out.append(dict(method="quantization", capability="all", test=f"NEW source pythia-1b@96k, int{b} (seen bit)",
                candidate="frozen config-indicator {N0,L0,D0}", inputs="K0", split="source", ie="interp",
                cand_mae=m("full_N0_L0_D0"), base_name=f"strongest: {strongest}", base_mae=m(strongest),
                status="P-NEW (v46, frozen b1bf631)", ci="one source-state; 3 caps"))
    except FileNotFoundError:
        pass
    # --- P1-v2 (A100 round v2): 1B (in-range size) and 6.9B (~5x size extrapolation) at steps 32k+112k, protocol A,
    #     predictions frozen before measurement (v49; cd9ed8f for 1B, 950312f for 6.9B). Per-stage values: tables/p1v2.tex.
    import glob as _glob
    for size, lab in (("1b", "NEW size 1B (in-range), 32k+112k"), ("6.9b", "NEW size 6.9B (5$\\times$ extrap.), 32k+112k")):
        files = sorted(_glob.glob(str(R / f"v49-p1v2/compare_pythia-{size}@step*.json")))
        if len(files) != 2:
            continue
        S = [json.loads(Path(f).read_text())["protocols"]["A"]["summary"] for f in files]
        mp = lambda reg, k: sum(x["prune"][reg][k] for x in S) / len(S)
        for reg, ie, tl in (("interp_0.9-0.6", "interp", "$d$ 0.9--0.6"), ("extrap_0.55", "extrap", "$d$=0.55")):
            others = ("power", "cont", "strength_only", "median_curve", "zero"); strongest = min(others, key=lambda k: mp(reg, k))
            out.append(dict(method="pruning", capability="all", test=f"{lab}, {tl}", candidate="A2: per-$d$ source regr.\\ + fixed interp (frozen v49)",
                inputs="K0", split="source+strength", ie=ie, cand_mae=mp(reg, "A2"), base_name=f"strongest: {strongest}",
                base_mae=mp(reg, strongest), status="P-NEW (v49, frozen)", ci="mean of 2 stages; 3 caps; protocol A"))
        mq = lambda key, k: sum((x["quant_ge4"] if key == "ge4" else x["quant"]["int3"])[k] for x in S) / len(S)
        for key, tl in (("ge4", "$\\ge$4-bit"), ("int3", "int3")):
            others = ("noD0", "per_bit_mean", "per_bit_median", "zero"); strongest = min(others, key=lambda k: mq(key, k))
            out.append(dict(method="quantization", capability="all", test=f"{lab}, {tl}", candidate="config-indicator OLS {N0,L0,D0}",
                inputs="K0", split="source", ie="interp", cand_mae=mq(key, "full"), base_name=f"strongest: {strongest}",
                base_mae=mq(key, strongest), status="P-NEW (v49, frozen)", ci="mean of 2 stages; 3 caps; protocol A"))
    # LaTeX-ready candidate forms (math mode); plain 'candidate' stays for CSV/JSON
    tex_form = {
        "config-indicator OLS {N0,L0,D0} (16 coeffs/arm-cap)": r"config-indicator OLS $\{N_0,L_0,D_0\}$ (16 coeffs)",
        "A_c(x)*((1-d)/0.3)^g_c, shared g (5 params/cap)": r"$A_c(\mathbf x)\,((1-d)/0.3)^{\gamma_c}$, shared $\gamma_c$ (5 params)",
        "config-indicator OLS {N0,L0,D0}": r"config-indicator OLS $\{N_0,L_0,D_0\}$",
        "linear {N0,L0,D0} (4 params)": r"linear $\{N_0,L_0,D_0\}$ (4 params)",
    }
    for r in out:
        r["improvement"] = r["base_mae"] - r["cand_mae"]
        r["candidate_tex"] = tex_form.get(r["candidate"], r["candidate"])
    return out

def verdict(r):
    imp = r["improvement"]
    if r["status"].startswith("P-NEW"):
        return "new-source frozen: candidate better" if imp > 0 else "new-source frozen: candidate NOT better"
    if r["method"] == "pruning" and "UNSEEN density" in r["test"]:
        return "source cand > strength-only; A2 better pt-est, no extra adv. for power form" if imp <= 0.06 else "candidate pt-est > same-input baseline"
    if r["method"] == "quantization":
        return "near-zero region, no demonstrated gain" if "4-bit" in r["test"] else ("int3 real; transfer unstable" if imp < 0 else "int3 real; candidate helps")
    if r["method"] == "distillation" and "U=225" in r["test"]:
        return "E-only > constant (bias-dominated misfit)" if r["capability"] == "qa" else "constant best; E/T over-extrapolate"
    if r["method"] == "distillation":
        return "D0 pt-est improvement (math; code within noise)" if imp > 0 else "constant as good/better"
    return "pt-est improvement (CI excl. 0 only for code)" if imp > 0 else "does not beat strongest baseline"

def main():
    rs = rows()
    for r in rs: r["verdict"] = verdict(r)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "main_table.json").write_text(json.dumps(rs, indent=2))
    with (OUT_DIR / "main_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0].keys())); w.writeheader(); w.writerows(rs)
    TEX.parent.mkdir(parents=True, exist_ok=True)
    L = [r"\begin{sidewaystable}[p]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}llp{4.6cm}lrp{3.2cm}rrl@{}}", r"\toprule",
         r"Method & Cap. & Test & Split & Cand. & Strongest same-budget baseline & Base. & Impr. & St. \\",
         r"\midrule"]
    prev = None
    for r in rs:
        if prev and prev != r["method"]:
            L.append(r"\midrule")
        prev = r["method"]
        _DN = {"A1_gamma1": "A1 (gamma=1)", "strength_only": "strength-only", "median_curve": "median curve", "noD0_N0_L0": "no-D0",
               "per_bit_median": "per-bit median", "per_bit_mean": "per-bit mean", "full_N0_L0_D0": "full", "config_median": "median", "mean_base": "mean", "med_base": "median"}
        dn = lambda t: __import__("functools").reduce(lambda acc, kv: acc.replace(kv[0], kv[1]), _DN.items(), str(t)).replace("_", r"\_")
        esc = lambda s: str(s).replace("_", r"\_").replace("&", r"\&").replace(">=", r"$\ge$").replace("^", r"\^{}")
        # Status records the process only; improvement and verdict record the outcome.
        st = "P" if r["status"].startswith(("P-NEW", "FROZEN")) else "P/R" if r["status"].startswith("P (4") else "R"
        ab = {"pruning": "prune", "quantization": "quant", "distillation": "distill"}
        L.append(f"{ab[r['method']]} & {r['capability']} & {r['test']} & "
                 f"{esc(r['split'])}/{esc(r['ie']).replace('interp(0.65)+extrap(0.55)','int.+ext.').replace('interp (E in range)','interp')} & {r['cand_mae']:.3f} & {dn(r['base_name'])} & {r['base_mae']:.3f} & "
                 f"{r['improvement']:+.3f} & {st} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Main prediction table (capability loss, nats/token); candidate forms and inputs (all K0) are given in \S\ref{sec:twoaxes}. Impr.\ $=$ baseline MAE $-$ candidate MAE "
          r"(positive $=$ candidate better) against the strongest baseline at the \emph{same} information budget (K0). "
          r"Three statements are kept separate: whether adding $D_0$ improves on the model without it, whether the full-input model improves on the strongest same-budget baseline, and whether a frozen model succeeds on a new source state; Impr.\ reports point-estimate error reduction, and significance is stated in the text. Status: P $=$ frozen prospective (prediction registered before measurement, regardless of outcome); "
          r"R $=$ retrospective (leave-one-out on an existing panel or baseline/diagnostic added at closeout with the original prediction untouched); P/R $=$ the four candidates were frozen (P) but the per-capability choice among them was made after the test (R). "
          r"Every number is generated from \texttt{results/v*/summary.json} by \texttt{analysis/v45\_main\_table.py}; "
          r"per-source, per-density, and per-pool breakdowns are in the corresponding docs. Point estimates; the "
          r"controlled panels have three size/step/pool clusters, so intervals are panel-conditional (see text).}",
          r"\label{tab:main}", r"\end{sidewaystable}"]
    TEX.write_text("\n".join(L) + "\n")
    print(f"{len(rs)} rows -> {OUT_DIR}/main_table.{{json,csv}} and {TEX}")
    for r in rs:
        print(f"  {r['method']:12} {r['capability']:4} cand={r['cand_mae']:.3f} base={r['base_mae']:.3f} "
              f"impr={r['improvement']:+.3f} | {r['verdict']}")

if __name__ == "__main__":
    main()
