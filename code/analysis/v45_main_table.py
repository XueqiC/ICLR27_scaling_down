#!/usr/bin/env python3
"""Closeout main prediction table. CPU-only; reads ONLY existing result JSONs (no weights/GPU).

One row per (method, capability, test) for the frozen/registered prediction tests that carry the paper's
claims. Every number is read from a results/v*/summary.json (or compare.json) so the manuscript table is
machine-traceable. Emits CSV + JSON + a booktabs LaTeX table (paper/tables/main_prediction.tex).

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
TEX = ROOT / "paper/tables/main_prediction.tex"
CAPS = ("math", "code", "qa")

def J(p): return json.loads((R / p).read_text())

def rows():
    out = []
    # --- Pruning: source-transfer at FIXED density (controlled panel, leave-one-step-out), v36b
    v36b = J("v36b-input-comparison/summary.json")["results"]["pruning"]
    for c in CAPS:
        cmp = v36b[c]["comparison"]; full = cmp["core_table"]["N0_D0_L0"]; strong = cmp["strongest_simple_baseline"]
        base_mae = cmp["baseline_candidates"][strong]["mae"]
        out.append(dict(method="pruning", capability=c, test="source-transfer @ fixed d (3x3 Pythia, leave-one-step-out)",
            candidate="config-indicator OLS {N0,L0,D0} (16 coeffs/arm-cap)", inputs="K0", split="source", ie="interp",
            cand_mae=full["mae"], base_name=f"per-config {strong}", base_mae=base_mae, status="LOO-RETRO",
            ci=f"[{full['improvement_ci95'][0]:+.2f},{full['improvement_ci95'][1]:+.2f}] (3 step-folds)"))
    # --- Pruning: strength axis, v40 frozen + v42 same-input A2
    v42 = J("v42-prune-sameinput/summary.json")["by_cap"]
    for c in CAPS:
        m = v42[c]["mae"]
        out.append(dict(method="pruning", capability=c, test="UNSEEN density d=0.65/0.55 incl. new source 410M@96k",
            candidate="A_c(x)*((1-d)/0.3)^g_c, shared g (5 params/cap)", inputs="K0", split="strength+source",
            ie="interp(0.65)+extrap(0.55)", cand_mae=m["cand"], base_name="A2 per-density regr+lin interp (same input)",
            base_mae=m["A2"], status="FROZEN-PROSPECTIVE (v40) / A2 RETRO-DIAG", ci="point est.; per-source in PRUNE_SAMEINPUT"))
    # --- Quantization: per-bit partition on new @96k sources, v44
    v44 = J("v44-quant-partition/summary.json")["partition"]
    for c in CAPS:
        for reg, lab in (("ge4", ">=4-bit"), ("int3", "int3")):
            p = v44[f"{c}_{reg}"]
            out.append(dict(method="quantization", capability=c, test=f"source-transfer @ fixed bit, {lab} (3 new @96k sources)",
                candidate="config-indicator OLS {N0,L0,D0}", inputs="K0", split="source", ie="interp",
                cand_mae=p["cand_mae"], base_name="per-bit median", base_mae=p["median_mae"],
                status="FROZEN(v38 160M/1.4B)+v40(410M) / RETRO-DIAG table", ci="point est.; per-source in QUANT_PARTITION"))
    # --- Distillation: controlled source-transfer (3x3 LoRA panel), v39
    v39 = J("v39-distill-controlled/summary.json")["by_cap"]
    for c in CAPS:
        s = v39[c]["step"]
        out.append(dict(method="distillation", capability=c, test="delta source-transfer (3x3 LoRA, leave-one-step-out)",
            candidate="linear {N0,L0,D0} (4 params)", inputs="K0", split="source", ie="interp",
            cand_mae=s["mae_D0"], base_name=f"constant ({s['strongest_baseline']})", base_mae=s["mae_baseline"],
            status="LOO-RETRO", ci="point est.; 3 step-folds"))
    # --- Distillation: new pool U=225, v41 frozen + v43 residual
    v41 = J("v41-distill-newpool/summary.json")["by_cap"]
    for c in CAPS:
        m = v41[c]["mae"]; best = "E" if c == "qa" else "constant"
        out.append(dict(method="distillation", capability=c, test="UNSEEN pool U=225 from U75/U600 endpoints",
            candidate=f"{best}-only (frozen on endpoints)", inputs="K0", split="config(pool)", ie="interp (E in range)",
            cand_mae=m[best], base_name="constant" if best != "constant" else "E-only", base_mae=m["constant"] if best != "constant" else m["E"],
            status="FROZEN-PROSPECTIVE (v41) / residual RETRO-DIAG (v43)", ci="3 pools x 2 seeds; bias-dominated (DISTILL_RESIDUALS)"))
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
    if r["method"] == "pruning" and "UNSEEN density" in r["test"]:
        return "source-info wins vs strength-only; power form NOT nec. (A2 ties/beats)" if imp <= 0.06 else "candidate > same-input baseline"
    if r["method"] == "quantization":
        return "near-zero region, no demonstrated gain" if "4-bit" in r["test"] else ("int3 real; transfer unstable" if imp < 0 else "int3 real; candidate helps")
    if r["method"] == "distillation" and "U=225" in r["test"]:
        return "E-only > constant (bias-dominated misfit)" if r["capability"] == "qa" else "constant best; E/T over-extrapolate"
    if r["method"] == "distillation":
        return "D0 helps (math only)" if imp > 0 else "constant as good/better"
    return "beats baseline" if imp > 0 else "does not beat strongest baseline"

def main():
    rs = rows()
    for r in rs: r["verdict"] = verdict(r)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "main_table.json").write_text(json.dumps(rs, indent=2))
    with (OUT_DIR / "main_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0].keys())); w.writeheader(); w.writerows(rs)
    TEX.parent.mkdir(parents=True, exist_ok=True)
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}llp{3.1cm}p{2.6cm}lrlrrl@{}}", r"\toprule",
         r"Method & Cap. & Test & Candidate (inputs) & Split & Cand. & Strongest compatible baseline & Base. & Impr. & Status \\",
         r"\midrule"]
    prev = None
    for r in rs:
        if prev and prev != r["method"]:
            L.append(r"\midrule")
        prev = r["method"]
        esc = lambda s: str(s).replace("_", r"\_").replace("&", r"\&").replace(">=", r"$\ge$").replace("^", r"\^{}")
        st = "P" if r["status"].startswith("FROZEN") else ("L" if r["status"].startswith("LOO") else "R")
        L.append(f"{r['method']} & {r['capability']} & {esc(r['test'])} & {r['candidate_tex']} ({r['inputs']}) & "
                 f"{esc(r['split'])}/{esc(r['ie'])} & {r['cand_mae']:.3f} & {esc(r['base_name'])} & {r['base_mae']:.3f} & "
                 f"{r['improvement']:+.3f} & {st} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Main prediction table (capability loss, nats/token). Impr.\ $=$ baseline MAE $-$ candidate MAE "
          r"(positive $=$ candidate better) against the strongest baseline at the \emph{same} information budget (K0). "
          r"Status: P $=$ frozen prospective (prediction registered before measurement); L $=$ leave-one-out on an existing "
          r"panel (retrospective); R $=$ baseline/diagnostic added at closeout with the original prediction untouched. "
          r"Every number is generated from \texttt{results/v*/summary.json} by \texttt{analysis/v45\_main\_table.py}; "
          r"per-source, per-density, and per-pool breakdowns are in the corresponding docs. Point estimates; the "
          r"controlled panels have three size/step/pool clusters, so intervals are panel-conditional (see text).}",
          r"\label{tab:main}", r"\end{table}"]
    TEX.write_text("\n".join(L) + "\n")
    print(f"{len(rs)} rows -> {OUT_DIR}/main_table.{{json,csv}} and {TEX}")
    for r in rs:
        print(f"  {r['method']:12} {r['capability']:4} cand={r['cand_mae']:.3f} base={r['base_mae']:.3f} "
              f"impr={r['improvement']:+.3f} | {r['verdict']}")

if __name__ == "__main__":
    main()
