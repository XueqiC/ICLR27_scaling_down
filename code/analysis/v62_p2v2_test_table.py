#!/usr/bin/env python3
"""P2-v2 multi-student distillation test table -> paper/tables/p2v2_test.tex.
Reads results/v50-p2v2/{freeze,compare_test}.json. Headline column = frozen form with the lowest in-sample dev MAE at
freeze time (joint+src for every capability; rule stated after the tests -> origin code P/F/F/A for the prediction, R for
the rule). Rows: student x role (unseen pool U375 seeds 21-23; held-out student at dev pools when present) x capability;
MAE over all (seed, budget) points; zero-change baseline; best frozen form retrospectively (R)."""
import json, collections; from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
f = json.loads((ROOT / "results/v50-p2v2/freeze.json").read_text()); d = json.loads((ROOT / "results/v50-p2v2/compare_test.json").read_text())
CAPS = ("math", "code", "qa"); sel = {c: min(f["fits"][c], key=lambda k: f["fits"][c][k]["dev_mae"]) for c in CAPS}
agg = collections.defaultdict(lambda: collections.defaultdict(list)); zero = collections.defaultdict(list); npts = collections.Counter()
for k, v in d["summary"].items():
    st, role, cap, T = k.split("|")
    for c, m in v["mae"].items(): agg[(st, role, cap)][c].append(m)
    zero[(st, role, cap)].append(v["mae_zero"]); npts[(st, role, cap)] += v["n_pools"]
SN = {"gemma3-270m": "Gemma-3-270M", "gemma3-1b": "Gemma-3-1B", "gemma3-4b": "Gemma-3-4B"}
RN = {"test_pool": "unseen pool ($U{=}375$)", "dev_pool_heldout_student": "dev pools ($U{=}75,450$), held-out student"}
order = [("gemma3-270m", "test_pool"), ("gemma3-1b", "test_pool"), ("gemma3-4b", "test_pool"), ("gemma3-4b", "dev_pool_heldout_student")]
lines = []
for st, role in order:
    if not any((st, role, c) in agg for c in CAPS): continue
    for i, c in enumerate(CAPS):
        key = (st, role, c); row = {cand: sum(v) / len(v) for cand, v in agg[key].items()}; best = min(row, key=row.get); z = sum(zero[key]) / len(zero[key])
        lab = f"{SN[st]}, {RN[role]}" if i == 0 else ""
        lines.append(f"{lab} & {c.upper() if c=='qa' else c.capitalize()} & {row[sel[c]]:.3f} & {row['constant']:.3f} & {z:.3f} & {best.replace('+src', '$+$src')} ({row[best]:.3f}) & {npts[key]} \\\\")
    lines.append("\\addlinespace")
tex = r"""\begin{table}[H]
\centering\small
\caption{Multi-student distillation test on the uniform absolute-exposure protocol. Eight forms per capability were frozen from the twelve development runs (Gemma-3-270M and 1B, $U\in\{75,450\}$, three data seeds) before any test run; the headline column is the frozen form with the lowest development error at freeze time (joint with a source term $k\log(N_S/N_{\mathrm{ref}})$ for every capability; this selection rule was stated after the tests). MAE in nats over all test points of the row (three data seeds times four budgets per trajectory), the frozen per-capability constant (the same-input baseline of this arm), the zero-change baseline, and the best of the eight frozen forms chosen after the fact (R). Gemma-3-4B is a held-out student whose $\log N_S$ lies $1.9$ above the reference against a development span of $\pm0.55$.}
\label{tab:p2v2_test}
\begin{tabular}{llccccr}
\toprule
Student, pools & Cap. & Dev-selected & Constant & Zero & Best frozen (R) & $n$ \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(lines).removesuffix("\n\\addlinespace")
(ROOT / "paper/tables/p2v2_test.tex").write_text(tex); print("WROTE paper/tables/p2v2_test.tex"); print(tex)
