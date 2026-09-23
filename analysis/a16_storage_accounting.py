#!/usr/bin/env python3
"""A16: byte accounting for the S3 candidates beside the paper's nominal storage ratio.

The paper's nominal storage ratio counts stored matrix parameters: density d for pruning, b/16 for
b-bit quantization plus the 16-bit group scales for grouped quantization, and the student's matrix
parameter count for distillation, all relative to the dense reference in 16-bit weights. This
script prices the same candidates in bytes under explicit storage formats and reports the ratio of
those bytes to the dense reference's 16-bit bytes:

  dense        16-bit weights: 2 bytes per matrix parameter.
  pruning      unstructured sparsity in two formats: a bitmap (2 bytes per kept weight plus one
               bit per position) and CSR (2 bytes per kept weight, a 16-bit column index per kept
               weight, and a 32-bit row pointer per row).
  quantization b bits per weight packed exactly, plus one 16-bit scale per output row (per-channel)
               or per group of g weights (grouped).
  distillation the student's matrix parameters in 16-bit weights (adapters merged).

Matrix shapes come from the checkpoints loaded on CPU with the V6 loader, so the parameter scope
is the one every compression in the paper acts on (all weight matrices, embeddings included).

    python3 analysis/a16_storage_accounting.py            # results/a16-storage-accounting/{summary.json,summary.md}
    python3 analysis/a16_storage_accounting.py --table    # also paper/paper/tables/storage_accounting.tex
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/a16-storage-accounting"
PLAN = ROOT / "results/s3-selection-validation/plan_v2.json"
TABLE = ROOT / "paper/paper/tables/storage_accounting.tex"
NAMES = {"pythia-410m--step120000": "Pythia 410M, step 120000", "pythia-1.4b--step120000": "Pythia 1.4B, step 120000",
         "gemma3-1b": "Gemma 3, 1B", "gemma3-4b": "Gemma 3, 4B"}


def shapes(model_id, revision=None):
    """(rows, cols) of every weight matrix in the language model, loaded on CPU in bf16."""
    import torch
    from analysis.v6_capability_geometry import language_weight_parameters, load_text_causal_lm
    model, _ = load_text_causal_lm(model_id, torch.bfloat16, revision)
    out = [tuple(p.shape[:2]) if p.dim() == 2 else (p.shape[0], int(p.numel() // p.shape[0])) for _, p in language_weight_parameters(model)]
    del model
    return out


def dense_bytes(shp):
    return sum(2 * r * c for r, c in shp)


def prune_bytes(shp, d):
    n = sum(r * c for r, c in shp); rows = sum(r for r, _ in shp)
    kept = d * n
    bitmap = 2 * kept + n / 8
    csr = 2 * kept + 2 * kept + 4 * rows
    return {"bitmap": bitmap, "csr": csr}


def quant_bytes(shp, bits, group=None):
    n = sum(r * c for r, c in shp); rows = sum(r for r, _ in shp)
    scales = 2 * (n / group if group else rows)
    return n * bits / 8 + scales


def price(candidate, ref_shapes, student_shapes):
    cid = candidate["id"]
    base = dense_bytes(ref_shapes)
    if cid.startswith("dense"):
        actual = {"bytes": base}
    elif cid.startswith("prune:d"):
        d = float(cid.split("d")[1]); b = prune_bytes(ref_shapes, d)
        actual = {"bytes": b["bitmap"], "bytes_csr": b["csr"]}
    elif cid.startswith("quant:channel_b"):
        actual = {"bytes": quant_bytes(ref_shapes, int(cid.split("_b")[1]))}
    elif cid.startswith("quant:b"):
        m = re.match(r"quant:b(\d+)_g(\d+)", cid); actual = {"bytes": quant_bytes(ref_shapes, int(m.group(1)), int(m.group(2)))}
    elif cid.startswith("distill"):
        actual = {"bytes": dense_bytes(student_shapes)}
    else:
        raise ValueError(cid)
    row = {"id": cid, "nominal": candidate["r"], "actual": actual["bytes"] / base}
    if "bytes_csr" in actual:
        row["actual_csr"] = actual["bytes_csr"] / base
    return row


def main(write_table):
    plan = json.loads(PLAN.read_text())
    refs = plan["references"] if "references" in plan else plan["a12_candidates"]
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"formats": __doc__.split("\n\n")[1], "references": {}}
    for ref in refs:
        rs = shapes(ref["model"], ref.get("revision")); ss = shapes(ref["student"]["model"], None)
        rows = [price(c, rs, ss) for c in ref["candidates"]]
        summary["references"][ref["id"]] = {"model": ref["model"], "student": ref["student"]["model"],
                                            "matrix_parameters": sum(r * c for r, c in rs),
                                            "student_matrix_parameters": sum(r * c for r, c in ss),
                                            "rows": sum(r for r, _ in rs), "candidates": rows}
        print(ref["id"], "done", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = ["# A16 storage accounting (bytes beside nominal storage)", "",
             "| Reference | Candidate | Nominal | Bytes, bitmap or packed | Bytes, CSR |", "|---|---|---:|---:|---:|"]
    for rid, e in summary["references"].items():
        for r in e["candidates"]:
            lines.append(f"| {rid} | {r['id']} | {r['nominal']:.4f} | {r['actual']:.4f} | {r.get('actual_csr', float('nan')):.4f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if write_table:
        TABLE.write_text(render(summary))
        print("wrote", TABLE)


CANDIDATE_WORDS = {"dense:source": "Dense reference", "prune:d0.9": "Pruning, density 0.9", "prune:d0.8": "Pruning, density 0.8",
                   "prune:d0.7": "Pruning, density 0.7", "prune:d0.6": "Pruning, density 0.6",
                   "quant:channel_b8": "Per-channel 8 bits", "quant:channel_b6": "Per-channel 6 bits", "quant:channel_b4": "Per-channel 4 bits",
                   "quant:b4_g32": "Grouped 4 bits, group 32", "quant:b4_g128": "Grouped 4 bits, group 128",
                   "quant:b3_g32": "Grouped 3 bits, group 32", "distill:new_s3": "Distilled student"}


def render(summary):
    """One row per candidate: nominal ratio and bytes as a fraction of the dense reference (identical across references except for the student)."""
    from analysis.paper_table_layout import house_style, table_layout
    refs = list(summary["references"])
    def cand(r, cid):
        return next(x for x in summary["references"][r]["candidates"] if x["id"] == cid)
    rows = []
    for cid, word in CANDIDATE_WORDS.items():
        nominal = [cand(r, cid)["nominal"] for r in refs]; actual = [cand(r, cid)["actual"] for r in refs]
        if cid.startswith("distill"):
            cells = [word, ", ".join(f"{v:.3f}" for v in nominal), ", ".join(f"{v:.3f}" for v in actual)]
        elif cid.startswith("prune"):
            csr = [cand(r, cid)["actual_csr"] for r in refs]
            cells = [word, f"{nominal[0]:.3f}", f"bitmap {max(actual):.3f}; compressed sparse row {max(csr):.3f}"]
        else:
            cells = [word, f"{nominal[0]:.3f}", f"{max(actual):.3f}"]
        rows.append(" & ".join(cells) + r" \\")
    caption = ("Storage of the selection candidates in bytes, as a fraction of the dense reference stored in 16-bit weights, beside "
               "the nominal storage ratio the selection uses. Pruning is priced in a bitmap format, 16-bit kept weights plus one bit "
               "per position, and in compressed sparse row format, 16-bit kept weights with 16-bit column indices and 32-bit row "
               "pointers; quantization packs its bits exactly and adds one 16-bit scale per output row or per group; the distilled "
               "student is stored in 16-bit weights. Byte ratios agree across the four references to three decimals except for the "
               "student, whose four values follow the order Pythia 410M, Pythia 1.4B, Gemma 3 1B, Gemma 3 4B, the four references of "
               "the validation.")
    text = ("% Generated by analysis/a16_storage_accounting.py --table; do not edit.\n"
            "\\begin{table}[tb]\n\\centering\\footnotesize\n"
            f"\\caption{{{caption}}}\n\\label{{tab:storage-accounting}}\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lll@{}}\n\\toprule\n"
            "Candidate & Nominal storage ratio & Bytes as a fraction of the dense reference \\\\\n\\midrule\n" +
            "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    return house_style(table_layout(text))


def regret_under_bytes():
    """Re-run the selection with the feasible set counted in bytes, and report regret against the byte-feasible oracle.

    The policies are re-decided from the development-median predictions (on these four new states the
    final rule's pruning and quantization predictions are those medians; its student prediction differs
    only in the mathematics term, which never selects the student), once under the nominal ratios and
    once under the byte ratios, so that the two regrets are like for like.
    """
    import copy
    import statistics
    from analysis import s3_common as common
    from analysis import s3_policy_ablation as ab
    summary = json.loads((OUT / "summary.json").read_text())
    plan = json.loads(PLAN.read_text())
    models = json.loads((ab.S3 / "inputs/locked_models.json").read_text())
    v70_freeze = json.loads((ROOT / "results/v70-distill-confirm/freeze.json").read_text())
    price = {rid: {c["id"]: c["actual"] for c in e["candidates"]} for rid, e in summary["references"].items()}
    out = {"per_reference": {}, "pooled": {}}
    for ref in plan["references"]:
        configs = copy.deepcopy(ref["candidates"])
        for q in configs:
            q["actual"] = common.losses(json.loads(Path(ab.measure_path(ref, q, ab.S3)).read_text())["losses"])
        preds = ab.predictions(ref, models, v70_freeze)
        for q in configs:
            q["median"] = preds[q["id"]]["median"]
        dense = ref["dense"]
        entry = {}
        for objective in ab.OBJECTIVES:
            cells = {"nominal": [], "bytes": []}
            for budget in ab.BUDGETS:
                for unit in ("nominal", "bytes"):
                    cfgs = copy.deepcopy(configs)
                    if unit == "bytes":
                        for q in cfgs:
                            q["r"] = price[ref["id"]][q["id"]]
                    available = common.feasible(cfgs, budget)
                    quant = [q for q in available if q["method"] == "quant"]
                    oracle = common.choose(available, objective, dense, "actual")
                    q_oracle = common.choose(quant, objective, dense, "actual")
                    if not oracle or not q_oracle:
                        continue
                    best = common.score(oracle["actual"], objective, dense)
                    rule = common.choose(available, objective, dense, "median")
                    qonly = common.choose(quant, objective, dense, "median")
                    cells[unit].append({"budget": budget, "opportunity": common.score(q_oracle["actual"], objective, dense) - best,
                                        "rule_regret": common.score(rule["actual"], objective, dense) - best,
                                        "quant_only_regret": common.score(qonly["actual"], objective, dense) - best,
                                        "rule_id": rule["id"]})
            entry[objective] = {unit: {"n_paired": len(c), "mean_opportunity": statistics.mean(x["opportunity"] for x in c),
                                       "rule_regret": statistics.mean(x["rule_regret"] for x in c),
                                       "quant_only_regret": statistics.mean(x["quant_only_regret"] for x in c),
                                       "student_chosen": sum(1 for x in c if x["rule_id"].startswith("distill"))}
                                for unit, c in cells.items()}
        out["per_reference"][ref["id"]] = entry
    for objective in ab.OBJECTIVES:
        out["pooled"][objective] = {unit: {k: statistics.mean(out["per_reference"][r][objective][unit][k] for r in out["per_reference"])
                                          for k in ("mean_opportunity", "rule_regret", "quant_only_regret")}
                                    for unit in ("nominal", "bytes")}
    summary["regret_under_bytes"] = out
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    for objective, e in out["pooled"].items():
        print(objective, {u: {k: round(v, 4) for k, v in d.items()} for u, d in e.items()})
    for rid, e in out["per_reference"].items():
        print(rid, "qa", {u: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()} for u, d in e["qa"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", action="store_true"); ap.add_argument("--feasibility", action="store_true")
    ap.add_argument("--regret", action="store_true")
    a = ap.parse_args()
    if a.feasibility:
        feasibility()
    elif a.regret:
        regret_under_bytes()
    else:
        main(a.table)
