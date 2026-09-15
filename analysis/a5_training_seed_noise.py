#!/usr/bin/env python3
"""Training-seed noise on delta, from two runs of the SAME pool (B) differing only in --seed.
Not a corner. Reports per-distribution |delta_seed1 - delta_seed0| at every shared checkpoint,
the implied sigma on a single delta, and the implied sigma on the second difference I where the
single-delta sigma enters four times (two of them from pool B, same sign)."""
import json, glob, os, math, statistics, sys
R = "results/v12-distill/gemma3-1b"
S0 = f"{R}/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory"
S1 = sorted(glob.glob(f"{R}/*a3b_corners_1_4_poolB_tseed1*/trajectory"))[0]
def load(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/update-*/eval.json")):
        j = json.load(open(f))
        if j["processed_tokens"] > 0:
            out[os.path.basename(os.path.dirname(f))] = (j["completion_tokens_seen"], j["delta"])
    return out
a, b = load(S0), load(S1)
shared = sorted(set(a) & set(b))
if len(shared) < 8:
    print(f"only {len(shared)} shared checkpoints; seed-1 run incomplete"); sys.exit(2)
rows = []
for ck in shared:
    T0, d0 = a[ck]; T1, d1 = b[ck]
    assert T0 == T1, (ck, T0, T1)
    rows.append((ck, T0, {c: d1[c] - d0[c] for c in d0}))
print(f"shared checkpoints: {len(shared)} (identical supervised budgets asserted)")
print(f"{'checkpoint':16s} {'T':>7s} {'qa':>9s} {'math':>9s} {'code':>9s}")
for ck, T, d in rows:
    print(f"{ck:16s} {T:7d} {d['qa']:9.4f} {d['math']:9.4f} {d['code']:9.4f}")
print()
reg = {"qa": 0.1608, "math": 0.0088, "code": 0.0254}
print(f"{'cap':5s} {'median|diff|':>12s} {'sigma_delta':>12s} {'sigma_I':>9s} {'registered pool-seed sigma_I':>29s}")
res = {}
for c in ("qa", "math", "code"):
    v = sorted(abs(d[c]) for _, _, d in rows)
    med = statistics.median(v)
    sigma = med / 0.9539 / math.sqrt(2)   # median|X1-X2| for two draws of N(0,sigma)
    sI = 2 * sigma                         # four deltas, pool B twice with the same sign
    res[c] = {"n": len(v), "median_abs_diff": med, "sigma_delta": sigma, "sigma_I": sI, "registered_pool_seed_sigma_I": reg[c]}
    print(f"{c:5s} {med:12.5f} {sigma:12.5f} {sI:9.5f} {reg[c]:29.4f}")
os.makedirs("results/a5-training-seed-noise", exist_ok=True)
json.dump({"seed0": S0, "seed1": S1, "shared_checkpoints": shared, "per_checkpoint": [(ck, T, d) for ck, T, d in rows], "summary": res,
           "note": "Training-seed noise only; pool held fixed. Not a corner and not part of the registered decision."},
          open("results/a5-training-seed-noise/summary.json", "w"), indent=2)
print("wrote results/a5-training-seed-noise/summary.json")
