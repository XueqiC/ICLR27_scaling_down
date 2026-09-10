#!/usr/bin/env python3
"""P3 primary-vs-secondary check table (v48 measurements) -> paper/tables/p3_check.tex. Values: loss change from dense,
nats per native token, primary / secondary, Gemma-3-1B, six pre-registered states."""
import json; from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
m = json.loads((ROOT / "results/v48-p3-measure/measurements.json").read_text()); d0 = m[0]
name = {"prune_d0.7": "magnitude pruning $d=0.7$", "rtn_int4": "RTN int4",
        "kd_U75_s11_update-00000025": "KD, $U=75$, first milestone", "kd_U75_s11_update-00000049": "KD, $U=75$, second milestone",
        "kd_U450_s11_update-00000026": "KD, $U=450$, first milestone", "kd_U450_s11_update-00000050": "KD, $U=450$, second milestone"}
rows = []
for r in m[1:]:
    cells = [f"${r['primary'][c]-d0['primary'][c]:+.2f}$ / ${r['secondary'][c]-d0['secondary'][c]:+.2f}$" for c in ("math", "code", "qa")]
    rows.append(f"{name[r['state']]} & " + " & ".join(cells) + r" \\")
tok = d0["primary_tokens"], d0["secondary_tokens"]
tex = r"""\begin{table}[H]
\centering\small
\caption{Primary versus secondary benchmark on Gemma-3-1B (P3 check). Loss change from dense in nats per native token on the main-protocol probe / on an independent benchmark of the same capability (math: SVAMP; code: HumanEval; QA: TriviaQA; 128 samples, fixed probe seed, existing references). The six states were fixed before any secondary measurement. Scored completion tokens, primary / secondary: math %d / %d, code %d / %d, QA %d / %d.}
\label{tab:p3_check}
\begin{tabular}{lccc}
\toprule
State & Math & Code & QA \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (tok[0]["math"], tok[1]["math"], tok[0]["code"], tok[1]["code"], tok[0]["qa"], tok[1]["qa"], "\n".join(rows))
(ROOT / "paper/tables/p3_check.tex").write_text(tex); print("WROTE paper/tables/p3_check.tex")
