#!/usr/bin/env python3
"""R1 (rebuttal round, retrospective): exact benchmark-label permutation test of the V9 gradient block structure.

Reads the stored shared-component-removed cosine matrices (results/v9-capability-regions/<model>/similarity.json;
the shared component is the coordinatewise geometric mean over all nine benchmarks, C4 included, and uses no
capability label) and reports, per model:
  * the block score (mean within-capability minus mean cross-capability cosine) of the task partition and its
    exact permutation p-value over all distinct relabelings with the same group sizes, with SVAMP in mathematics
    (280 relabelings of 8 task benchmarks) and with SVAMP left out (105 relabelings of 7);
  * the same test for the worked-solution versus short-answer partition (35 relabelings);
  * the pair cosines that explain the result.
Writes results/r1-gradient-blocks/summary.json and paper/paper/tables/gradient_blocks.tex. CPU only, seconds.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/v9-capability-regions"
OUT = ROOT / "results/r1-gradient-blocks"
TABLE = ROOT / "paper/paper/tables/gradient_blocks.tex"
TASK = ["gsm8k", "math500", "svamp", "humaneval", "mbpp", "2wiki", "hotpotqa", "triviaqa"]
LABEL = {"gsm8k": "math", "math500": "math", "svamp": "math", "humaneval": "code", "mbpp": "code",
         "2wiki": "qa", "hotpotqa": "qa", "triviaqa": "qa"}
FORMAT = {"gsm8k": "worked", "math500": "worked", "humaneval": "worked", "mbpp": "worked",
          "svamp": "short", "2wiki": "short", "hotpotqa": "short", "triviaqa": "short"}
NAMES = {"Qwen--Qwen3-0.6B": "Qwen3 0.6B", "Qwen--Qwen3-1.7B": "Qwen3 1.7B", "Qwen--Qwen3-4B": "Qwen3 4B",
         "gemma3-270m": "Gemma 3 270M", "gemma3-1b": "Gemma 3 1B", "gemma3-4b": "Gemma 3 4B",
         "gemma3-12b": "Gemma 3 12B", "gemma4-31b": "Gemma 4 31B", "muse-30b": "Muse 30B", "olmo3-7b": "OLMo 3 7B"}
ORDER = list(NAMES)


def block_score(m: np.ndarray, labels: list[str]) -> float:
    pairs = list(itertools.combinations(range(len(labels)), 2))
    within = [m[i, j] for i, j in pairs if labels[i] == labels[j]]
    cross = [m[i, j] for i, j in pairs if labels[i] != labels[j]]
    return float(np.mean(within) - np.mean(cross))


def exact_permutation(m: np.ndarray, labels: list[str]) -> tuple[float, float, int]:
    """Observed score, exact p (share of distinct same-size partitions scoring at least as high), count."""
    observed = block_score(m, labels)
    seen, scores = set(), []
    for perm in set(itertools.permutations(labels)):
        part = frozenset(frozenset(i for i, x in enumerate(perm) if x == g) for g in set(labels))
        if part not in seen:
            seen.add(part)
            scores.append(block_score(m, list(perm)))
    return observed, float(np.mean([s >= observed - 1e-12 for s in scores])), len(scores)


def analyse_model(sim: dict) -> dict:
    cos = sim["matrices"]["shared_component_removed_cosine"]
    m = np.array([[cos[a][b] for b in TASK] for a in TASK])
    idx = {b: i for i, b in enumerate(TASK)}
    keep = [b for b in TASK if b != "svamp"]
    mk = m[np.ix_([idx[b] for b in keep], [idx[b] for b in keep])]
    mean = lambda xs, ys: float(np.mean([m[idx[x], idx[y]] for x in xs for y in ys]))
    s_all, p_all, n_all = exact_permutation(m, [LABEL[b] for b in TASK])
    s_keep, p_keep, n_keep = exact_permutation(mk, [LABEL[b] for b in keep])
    s_fmt, p_fmt, n_fmt = exact_permutation(m, [FORMAT[b] for b in TASK])
    return {"task_with_svamp": {"score": s_all, "p": p_all, "n_partitions": n_all},
            "task_without_svamp": {"score": s_keep, "p": p_keep, "n_partitions": n_keep},
            "worked_vs_short": {"score": s_fmt, "p": p_fmt, "n_partitions": n_fmt},
            "cos_gsm8k_math500": float(m[idx["gsm8k"], idx["math500"]]),
            "cos_worked_math_code": mean(["gsm8k", "math500"], ["humaneval", "mbpp"]),
            "cos_worked_math_qa": mean(["gsm8k", "math500"], ["2wiki", "hotpotqa", "triviaqa"]),
            "cos_svamp_qa": mean(["svamp"], ["2wiki", "hotpotqa", "triviaqa"]),
            "cos_svamp_worked_math": mean(["svamp"], ["gsm8k", "math500"])}


def main() -> None:
    rows = {}
    for tag in ORDER:
        path = SRC / tag / "similarity.json"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        rows[tag] = analyse_model(json.loads(path.read_text()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({"status": "retrospective (rebuttal round, 2026-11-06)",
                                                   "metric": "shared_component_removed_cosine", "models": rows}, indent=1))
    body = []
    for tag in ORDER:
        r = rows[tag]
        body.append(f"{NAMES[tag]} & {r['cos_gsm8k_math500']:+.2f} & {r['cos_worked_math_code']:+.2f} & "
                    f"{r['cos_worked_math_qa']:+.2f} & {r['cos_svamp_qa']:+.2f} & {r['task_without_svamp']['p']:.3f} & "
                    f"{r['task_with_svamp']['p']:.3f} & {r['worked_vs_short']['p']:.3f} \\\\")
    TABLE.write_text(
        "% Generated by analysis/r1_gradient_blocks.py; do not edit.\n"
        "\\begin{table}[tb]\n\\centering\\footnotesize\n"
        "\\caption{\\rev{Benchmark-level gradient block structure, retrospective. Cosines of the shared-component-removed "
        "log-Fisher signatures between GSM8K and MATH-500, between those two worked-solution mathematics benchmarks and "
        "the code or question-answering benchmarks (means over pairs), and between SVAMP and the question-answering "
        "benchmarks; exact permutation $p$-values of the task partition over all relabelings with the same group sizes, "
        "without SVAMP (105 partitions of seven benchmarks) and with it (280 of eight), and of the partition into "
        "worked-solution and short-answer references (35 of eight).}}\n\\label{tab:gradient-blocks}\n"
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrrrr@{}}\n\\toprule\n"
        "Model & GSM8K--MATH & Math--code & Math--QA & SVAMP--QA & $p$, no SVAMP & $p$, task & $p$, format \\\\\n\\midrule\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    print(TABLE.read_text())


if __name__ == "__main__":
    main()
