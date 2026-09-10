#!/usr/bin/env python3
"""Aggregate P1-v2 compare files (all measured new sources) into one machine-generated table (JSON/CSV/LaTeX).
Rows: source x protocol x regime (prune interp 0.9-0.6 / prune extrap 0.55 / quant >=4-bit / quant int3 / int5 rule),
columns: MAE per candidate (mean over 3 capabilities), best candidate, strongest same-budget baseline, improvement.
Labels: size = 'in-range' (1B) or 'extrapolation ~5x' (6.9B); stage = 'interp' (A) or 'extrap' (B, only for 112k)."""
import json, csv, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results/v49-p1v2"
GE = "$\\ge$"; BS = "\\\\"
PC = ("power", "A2", "A1", "cont", "strength_only", "median_curve", "zero"); QC = ("full", "noD0", "per_bit_mean", "per_bit_median", "zero")
rows = []
def _key(f):
    t = Path(f).stem.replace("compare_", ""); return ("6.9b" in t, int(t.split("step")[1]))
for f in sorted(glob.glob(str(OUT / "compare_*.json")), key=_key):
    c = json.loads(Path(f).read_text()); tag = c["tag"]; size = "6.9b" if "6.9b" in tag else "1b"; step = int(tag.split("step")[1])
    size_lab = "size extrapolation (~5x)" if size == "6.9b" else "size in-range"
    for P, res in c["protocols"].items():
        stage_lab = "stage interp" if (P == "A" or step <= 64000) else "stage EXTRAPOLATION (fit on <=64k)"
        s = res["summary"]
        for reg, key in (("prune interp d0.9-0.6", ("prune", "interp_0.9-0.6")), ("prune extrap d0.55", ("prune", "extrap_0.55"))):
            m = s[key[0]][key[1]]; best = min(PC, key=lambda k: m[k]); same_best = min(("A2", "A1", "cont"), key=lambda k: m[k])
            rows.append({"source": tag, "protocol": P, "size": size_lab, "stage": stage_lab, "regime": reg, **{k: round(m[k], 4) for k in PC},
                         "best": best, "best_same_input_alt": same_best, "power_minus_bestalt": round(m["power"] - m[same_best], 4), "strength_only_vs_best": round(m["strength_only"] - m[best], 4)})
        for reg, key in (("quant >=4-bit", "quant_ge4"), ("quant int3", None), ("quant int5 (rule)", None)):
            m = s[key] if key else s["quant"]["int3" if "int3" in reg else "int5"]
            best = min(QC, key=lambda k: m[k])
            rows.append({"source": tag, "protocol": P, "size": size_lab, "stage": stage_lab, "regime": reg, **{k: round(m[k], 4) for k in QC}, "best": best,
                         "full_minus_bestbase": round(m["full"] - min(m[k] for k in QC if k != "full"), 4)})
(OUT / "p1v2_table.json").write_text(json.dumps(rows, indent=2))
if rows:
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("source", "protocol", "size", "stage", "regime", "best"), k))
    with (OUT / "p1v2_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow({k: r.get(k, "") for k in keys}) for r in rows]
    L = [r"\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{1.6pt}", r"\begin{tabular}{@{}lllrrrrrrrl@{}}", r"\toprule",
         r"Source & Prot. & Regime & power & A2 & A1 & cont & str.-only & median & zero & best \\", r"\midrule"]
    for r in rows:
        if r["regime"].startswith("prune"):
            src = r['source'].replace('pythia-','').replace('@step','@'); best = r['best'].replace('_','-')
            L.append(f"{src} & {r['protocol']} & {r['regime']} & {r['power']:.3f} & {r['A2']:.3f} & {r['A1']:.3f} & {r['cont']:.3f} & {r['strength_only']:.3f} & {r['median_curve']:.3f} & {r['zero']:.3f} & {best} " + BS)
    L += [r"\midrule", r"Source & Prot. & Regime & full & no-D0 & mean & median & zero & & & best \\", r"\midrule"]
    for r in rows:
        if r["regime"].startswith("quant"):
            src = r['source'].replace('pythia-','').replace('@step','@'); reg = r['regime'].replace('>=', GE); best = r['best'].replace('_','-')
            L.append(f"{src} & {r['protocol']} & {reg} & {r['full']:.3f} & {r['noD0']:.3f} & {r['per_bit_mean']:.3f} & {r['per_bit_median']:.3f} & {r['zero']:.3f} & & & {best} " + BS)
    size_note = ("1B sources are in-range sizes and 6.9B is a $\\sim$5$\\times$ size extrapolation. "
                 if any("6.9b" in r["source"] for r in rows) else "The 1B source is an in-range size. ")
    cap = ("\\caption{P1-v2 new-source frozen prospectives (MAE over three capabilities, nats/token). Protocol A fits on the "
           "full nine-state development panel; protocol B on step$\\le$64k only, so its 112k rows are training-stage "
           "extrapolations. " + size_note + "Predictions were committed before measurement; per-capability and per-point "
           "values are in the results ledger.}")
    L += [r"\bottomrule\end{tabular}", cap, r"\label{tab:p1v2}\end{table}"]
    (ROOT / "paper/tables/p1v2.tex").write_text("\n".join(L) + "\n")
print(f"{len(rows)} rows from {len(glob.glob(str(OUT/'compare_*.json')))} sources -> p1v2_table.{{json,csv}} + paper/tables/p1v2.tex")
