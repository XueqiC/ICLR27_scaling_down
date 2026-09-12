#!/usr/bin/env python3
"""P2 registration (CPU, BEFORE any training): sample the nine pre-registered pools, tokenize to get pool
processed tokens and D_U (supervised completion tokens per pass), check pool-set distinctness by sample-ID
hash, fix planned epochs (smallest int with epochs*pool_processed >= 1.0M), planned E at the two frozen
milestones, and SAVE the OLD frozen v41 predictors' predictions for the six dev pools at the planned points."""
import json, hashlib, math, itertools
import numpy as np
from pathlib import Path
import analysis.v12_distill as v12
import analysis.v41_distill_newpool as v41
from transformers import AutoTokenizer

ROOT = v12.ROOT if hasattr(v12, "ROOT") else Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parents[1] / "results/v47-p2-register"; OUT.mkdir(parents=True, exist_ok=True)
SNAP = "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29"
DEV = [(75, s) for s in (11, 12, 13)] + [(450, s) for s in (11, 12, 13)]
TEST = [(375, s) for s in (21, 22, 23)]
TRIG = (500_000, 1_000_000)           # processed-token triggers (historical protocol)
T_MILESTONES = (148_000, 294_000)     # frozen supervised-completion targets from old logs (~0.295 ratio)
tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt", revision=SNAP)

def pool(U, seed):
    recs, _ = v12.load_sft_records("gpt-5.6-luna", ["math", "qa", "code"], U, "full", seed=0, data_seed=seed)
    ids, proc, compl, dom = set(), 0, 0, {}
    for r in recs:
        ex = v12.tokenize_sft_example(tok, r["prompt"], r["completion"], max_len=v12.MAX_LEN)
        n_c = int(ex["n_completion_tokens"]); n_in = int(ex["input_ids"].numel())
        if n_c == 0: continue
        ids.add(hashlib.sha256(json.dumps([r["prompt"], r["completion"]], ensure_ascii=False).encode()).hexdigest())
        proc += n_in; compl += n_c; dom[r["domain"]] = dom.get(r["domain"], 0) + 1
    return {"U": U, "data_seed": seed, "n_examples": len(ids), "pool_processed_tokens": proc, "D_U_completion": compl,
            "domain_counts": dom, "ids": sorted(ids)}

pools = {f"U{U}_s{s}": pool(U, s) for U, s in DEV + TEST}
# distinctness / overlap by ID sets
def jacc(a, b): a, b = set(a), set(b); return len(a & b) / len(a | b)
overlap = {}
for k1, k2 in itertools.combinations(pools, 2):
    overlap[f"{k1}~{k2}"] = round(jacc(pools[k1]["ids"], pools[k2]["ids"]), 3)
ep = v41._endpoint_rows(); fits = {c: {k: v41._fit(ep, c, k) for k in ("constant", "T", "E", "2D")} for c in v41.CAPS}
reg = {"student_snapshot": SNAP, "protocol": {"lora": {"r": 16, "alpha": 32, "dropout": 0.0,
        "targets": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]}, "lr": 1e-4, "optimizer": "AdamW",
        "scheduler": "cosine", "warmup_ratio": 0.03, "recipe": "full", "training_mode": "lora", "training_seed": 0,
        "processed_triggers": TRIG, "T_milestones_completion": T_MILESTONES,
        "epoch_rule": "smallest integer epochs with epochs*pool_processed >= 1.0e6"},
       "pools": {}, "overlap_jaccard": overlap, "old_predictor_predictions_dev": {}}
for k, p in pools.items():
    epochs = max(1, math.ceil(1.0e6 / p["pool_processed_tokens"]))
    planned = [{"trigger_processed": t, "planned_T_completion": T_MILESTONES[i],
                "E_processed_convention": t / p["pool_processed_tokens"],
                "E_completion_convention": T_MILESTONES[i] / p["D_U_completion"]} for i, t in enumerate(TRIG)]
    reg["pools"][k] = {**{kk: vv for kk, vv in p.items() if kk != "ids"}, "id_set_sha256": hashlib.sha256("".join(p["ids"]).encode()).hexdigest(),
                       "planned_epochs": epochs, "planned_milestones": planned, "role": "dev" if p["U"] in (75, 450) else "test"}
    if p["U"] in (75, 450):  # old frozen predictors (v41 endpoint fits; E in v41's processed convention)
        preds = {}
        for i, m in enumerate(planned):
            pt = {"Tc": m["planned_T_completion"], "E": m["E_processed_convention"]}
            preds[f"milestone{i+1}"] = {c: {kind: float(v41._predict([pt], c, kind, fits[c][kind])[0]) for kind in fits[c]} for c in v41.CAPS}
        reg["old_predictor_predictions_dev"][k] = preds
(OUT / "register.json").write_text(json.dumps(reg, indent=2))
for k, p in reg["pools"].items():
    print(f"{k}: n={p['n_examples']} proc={p['pool_processed_tokens']} D_U={p['D_U_completion']} epochs={p['planned_epochs']} "
          f"E@1M(proc)={p['planned_milestones'][1]['E_processed_convention']:.2f} dom={p['domain_counts']}")
same_u = {k: v for k, v in overlap.items() if k.split('~')[0].split('_')[0] == k.split('~')[1].split('_')[0]}
print("same-U pool overlaps (Jaccard):", same_u)
print("wrote", OUT / "register.json")
