#!/usr/bin/env python3
"""MuSiQue scope table from results/v67-musique-qa/measurements.json -> paper/paper/tables/musique_scope.tex.
Rows: student x pool group (final checkpoints of U375 test pools, U450 dev pools, U75 dev pools; seed repeats included);
mean delta from dense on MuSiQue and on the primary 2Wiki probe, with the range over trajectories."""

try:
    from .paper_table_text import table_text as publication_table_text
except ImportError:  # Direct scripts and file-based imports.
    try:
        from analysis.paper_table_text import table_text as publication_table_text
    except ImportError:
        from paper_table_text import table_text as publication_table_text

import json, collections, re; from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / "results/v67-musique-qa/measurements.json").read_text())
rows = rows["rows"] if isinstance(rows, dict) and "rows" in rows else rows
dense = {r["student"]: r for r in rows if r.get("checkpoint") == "dense" or r.get("run") == "dense"}
grp = collections.defaultdict(list)
for r in rows:
    if r in dense.values(): continue
    m = re.search(r"_full_(\d+)_", r["run"]); U = int(m.group(1)) if m else None
    last = r.get("selection") in (None, "last_trajectory") or "update-0000019" in str(r.get("checkpoint", "")) or "update-00000200" in str(r.get("checkpoint", ""))
    key = (r["student"], U, "final" if last else "early")
    dm = r["musique_loss"] - dense[r["student"]]["musique_loss"]; dp = r["primary_qa_loss"] - r["dense_primary_qa_loss"]
    grp[key].append((dm, dp, r.get("E")))
SN = {"gemma3-270m": "270M", "gemma3-1b": "1B", "gemma3-4b": "4B"}
lines = []
for key in sorted(grp, key=lambda k: (["gemma3-270m", "gemma3-1b", "gemma3-4b"].index(k[0]), k[1] or 0, k[2])):
    st, U, stage = key; v = grp[key]; dm = [x[0] for x in v]; dp = [x[1] for x in v]; E = [x[2] for x in v if x[2] is not None]
    role = {375: "unseen pool", 450: "dev pool", 75: "dev pool"}[U]
    lines.append(f"{SN[st]} & $U{{=}}{U}$ ({role}), {'final' if stage=='final' else 'early'} & {min(E):.1f}--{max(E):.1f} & {len(v)} & ${sum(dp)/len(dp):+.2f}$ [{min(dp):+.2f}, {max(dp):+.2f}] & ${sum(dm)/len(dm):+.2f}$ [{min(dm):+.2f}, {max(dm):+.2f}] \\\\")
tex = r"""\begin{table}[H]
\centering\small
\caption{Distillation QA response on the primary probe and on an independent multi-hop set. Change in conditional loss from the student's dense loss (nats per native token) on the 2Wiki measurement probe and on 128 MuSiQue answerable-dev questions with supporting paragraphs supplied (same context condition, fixed probe seed), for every trajectory of the uniform protocol at its final checkpoint (three data seeds per pool size; seed repeats included) and for the early checkpoints of the P3 states. Mean over trajectories and the range in brackets.}
\label{tab:musique_scope}
\begin{tabular}{llcccc}
\toprule
Student & Pool, checkpoint & $E$ & $n$ & $\Delta$ 2Wiki (primary) & $\Delta$ MuSiQue \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(lines)
(ROOT / "paper/paper/tables/musique_scope.tex").write_text(publication_table_text(tex)); print(tex)
