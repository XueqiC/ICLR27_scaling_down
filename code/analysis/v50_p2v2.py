#!/usr/bin/env python3
"""P2-v2 freeze/predict for the multi-student x pool x exposure matrix (CPU).
Dev: gemma3-270m, gemma3-1b x U{75,450} x data-seeds {11,12,13}, 4 checkpoints each (48 points), uniform
absolute-exposure protocol (suffix _p2v2). Candidates per capability (OLS, linear in parameters; bases fixed):
  constant : d = a                      | +source: a + k*n
  T-only   : d = a*u, u=log(1+T/T_ref)  | +source: (a + k*n)*u
  E-only   : d = a*w, w=log(1+E)        | +source: (a + k*n)*w
  joint    : d = u*(a + b*u + q*v), v=log(D_U/D_ref) | +source: u*((a + k*n) + b*u + q*v)
n = log(N_S/N_ref). T_ref = 35k (first milestone), D_ref = median dev D_U, N_ref = geometric mean of dev student
sizes (total parameters, meta-device count). No candidate selection at freeze; all 8 forms are frozen and
predicted for the 9 test trajectories (270m/1b/4b x U375 seeds 21,22,23) at planned T {35k,70k,140k,280k}.
Modes: freeze | compare (after tests)."""
import sys, json, glob, hashlib, datetime, math
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; REG = ROOT / "results/v47-p2-register"; OUT = ROOT / "results/v50-p2v2"; OUT.mkdir(parents=True, exist_ok=True)
reg = json.loads((REG / "register.json").read_text()); CAPS = ("math", "code", "qa")
DEV_STUDENTS = ("gemma3-270m", "gemma3-1b"); TEST_STUDENTS = ("gemma3-270m", "gemma3-1b", "gemma3-4b")
T_PLANNED = (35000, 70000, 140000, 280000); T_REF = 35000.0
HF = {"gemma3-270m": "google/gemma-3-270m", "gemma3-1b": "google/gemma-3-1b-pt", "gemma3-4b": "google/gemma-3-4b-pt"}

def n_params(student):
    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights
    cfg = AutoConfig.from_pretrained(HF[student])
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg)
    return sum(p.numel() for p in m.parameters())

def points(student, U, seed, suffix):
    d = ROOT / f"results/v12-distill/{student}/gpt-5.6-luna_full_{U}_{suffix}_lora_dseed{seed}"; DU = reg["pools"][f"U{U}_s{seed}"]["D_U_completion"]
    pts = []
    for f in sorted(glob.glob(str(d / "trajectory/update-*/eval.json"))):
        j = json.loads(Path(f).read_text())
        if j.get("processed_tokens", 0) > 0:
            pts.append({"student": student, "U": U, "seed": seed, "Tc": j["completion_tokens_seen"], "DU": DU, "E": j["completion_tokens_seen"] / DU, "delta": j["delta"], "processed": j["processed_tokens"], "updates": j.get("updates")})
    pts.sort(key=lambda p: p["Tc"]); return pts[:4]

def basis(kind, src, p, refs):
    u = math.log1p(p["Tc"] / T_REF); w = math.log1p(p["E"]); v = math.log(p["DU"] / refs["D_ref"]); n = math.log(refs["N"][p["student"]] / refs["N_ref"])
    if kind == "constant": cols = [1.0] + ([n] if src else [])
    elif kind == "T": cols = [u] + ([n * u] if src else [])
    elif kind == "E": cols = [w] + ([n * w] if src else [])
    elif kind == "joint": cols = [u, u * u, u * v] + ([n * u] if src else [])
    return np.array(cols)

def fit_all(pts, refs):
    fits = {}
    for cap in CAPS:
        y = np.array([p["delta"][cap] for p in pts]); fits[cap] = {}
        for kind in ("constant", "T", "E", "joint"):
            for src in (False, True):
                X = np.array([basis(kind, src, p, refs) for p in pts]); b, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
                fits[cap][f"{kind}{'+src' if src else ''}"] = {"coef": b.tolist(), "rank": int(rank), "n_params": X.shape[1], "dev_mae": float(np.mean(np.abs(X @ b - y)))}
    return fits

def freeze():
    pts = [p for st in DEV_STUDENTS for U in (75, 450) for s in (11, 12, 13) for p in points(st, U, s, "p2v2")]
    assert len(pts) == 48, f"expected 48 dev points, got {len(pts)}"
    N = {st: n_params(st) for st in TEST_STUDENTS}
    refs = {"N": N, "N_ref": float(np.exp(np.mean([np.log(N[s]) for s in DEV_STUDENTS]))), "D_ref": float(np.median([p["DU"] for p in pts])), "T_ref": T_REF}
    fits = fit_all(pts, refs)
    preds = {}
    TEST_POOLS = [(st, 375, s) for st in TEST_STUDENTS for s in (21, 22, 23)] + [("gemma3-4b", U, s) for U in (75, 450) for s in (11, 12)]  # v3: 4B held out at dev pools
    for st, U, s in TEST_POOLS:
        DU = reg["pools"][f"U{U}_s{s}"]["D_U_completion"]; key = f"{st}|U{U}_s{s}"; preds[key] = {}
        for T in T_PLANNED:
            p = {"student": st, "Tc": T, "DU": DU, "E": T / DU}
            preds[key][str(T)] = {cap: {k: float(basis(k.split("+")[0], "+src" in k, p, refs) @ np.array(fits[cap][k]["coef"])) for k in fits[cap]} for cap in CAPS}
    out = {"frozen_at_utc": datetime.datetime.utcnow().isoformat() + "Z", "refs": refs, "n_dev_points": len(pts), "dev_points": pts, "fits": fits, "test_predictions_planned": preds,
           "dev_hash": hashlib.sha256(json.dumps(pts, sort_keys=True).encode()).hexdigest()}
    (OUT / "freeze.json").write_text(json.dumps(out, indent=2, default=float))
    for cap in CAPS: print(cap, {k: round(v["dev_mae"], 3) for k, v in fits[cap].items()})
    print("frozen ->", OUT / "freeze.json")

def compare():
    fz = json.loads((OUT / "freeze.json").read_text()); refs = fz["refs"]; rows = []
    RUNS = [(st, 375, s, "test_pool") for st in TEST_STUDENTS for s in (21, 22, 23)] + [("gemma3-4b", U, s, "dev_pool_heldout_student") for U in (75, 450) for s in (11, 12)]
    for st, U, s, role in RUNS:
            for p in points(st, U, s, "p2v2test"):
                p["role"] = role
                for cap in CAPS:
                    act = p["delta"][cap]; pa = {k: float(basis(k.split("+")[0], "+src" in k, p, refs) @ np.array(fz["fits"][cap][k]["coef"])) for k in fz["fits"][cap]}
                    rows.append({"student": st, "pool": f"U{U}_s{s}", "role": role, "Tc": p["Tc"], "E": p["E"], "cap": cap, "actual": act, "pred_at_actual": pa, "abs": {k: abs(v - act) for k, v in pa.items()}, "signed": {k: v - act for k, v in pa.items()}, "abs_zero": abs(act)})
    summ = {}
    for st, role in sorted({(r["student"], r["role"]) for r in rows}):
        for cap in CAPS:
            for i, T in enumerate(T_PLANNED):
                pool_rows = [r for r in rows if r["student"] == st and r["role"] == role and r["cap"] == cap]
                rr = sorted(pool_rows, key=lambda r: abs(r["Tc"] - T))[:len({r["pool"] for r in pool_rows})]
                if not rr: continue
                keys = list(rr[0]["abs"]); summ[f"{st}|{role}|{cap}|T{T}"] = {"mae": {k: float(np.mean([r["abs"][k] for r in rr])) for k in keys}, "bias": {k: float(np.mean([r["signed"][k] for r in rr])) for k in keys}, "mae_zero": float(np.mean([r["abs_zero"] for r in rr])), "n_pools": len(rr)}
    (OUT / "compare_test.json").write_text(json.dumps({"rows": rows, "summary": summ}, indent=2, default=float)); print(json.dumps({k: {"const": round(v["mae"]["constant"], 3), "E": round(v["mae"]["E"], 3), "joint+src": round(v["mae"]["joint+src"], 3), "zero": round(v["mae_zero"], 3)} for k, v in summ.items()}, indent=1))

if __name__ == "__main__":
    {"freeze": freeze, "compare": compare}[sys.argv[1]]()
