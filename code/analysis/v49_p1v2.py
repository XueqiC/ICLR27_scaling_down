#!/usr/bin/env python3
"""P1 v2: joint size x stage x strength prospective for NEW Pythia sources. CPU only.
Two protocols: P1-A (fit on full 9-state dev) and P1-B (fit ONLY on dev step<=64k; 143k never used).
Pruning candidates (all K0): power (shared-shape, refit per protocol), A2 per-density source regression with the
FIXED interpolation rule, A1 (gamma=1), NEW continuous candidate dL=(beta.phi)s+(zeta.phi)s^2 (s=1-d), strength-only,
median-curve, zero. Quantization: config-indicator full/no-D0, per-bit mean/median, zero; 5-bit via the
pre-registered rule = linear interpolation of the 4- and 6-bit predictions ("interpolation-rule baseline").
Modes: freeze  -> fit + write results/v49-p1v2/register.json (no target data). predict TAG -> needs dense.json
for the target (K0 L0) -> writes predictions_<tag>.json. compare TAG -> after measurement."""
import sys, json, hashlib, itertools, datetime
import numpy as np
from pathlib import Path
import analysis.v36_pythia_controlled_fit as v36
import analysis.v40_prune_strength_axis as v40
OUT = v36.ROOT / "results/v49-p1v2"; OUT.mkdir(parents=True, exist_ok=True)
CAPS = ("math", "code", "qa"); SEEN_D = (0.9, 0.8, 0.7, 0.6); TEST_D = (0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55)
BITS_DEV = (8, 6, 4, 3); TEST_B = (8, 6, 5, 4, 3); GAMMA = np.round(np.arange(0.4, 4.01, 0.05), 3)
PROTO = {"A": (16000, 64000, 143000), "B": (16000, 64000)}

def dev_rows(steps):
    rows = [r for r in v40.load_dev() if r["step"] in steps]
    return rows
def zstats(rows):
    phi = np.array([r["phi"] for r in rows]); return phi.mean(0), phi.std(0)
def Z(phi_raw, c, s): return np.concatenate([[1.0], (np.asarray(phi_raw) - c) / s])
def fit_power(rows, cap, c, s):
    Zm = np.array([Z(r["phi"], c, s) for r in rows if r["cap"] == cap]); y = np.array([r["y"] for r in rows if r["cap"] == cap])
    d = np.array([r["d"] for r in rows if r["cap"] == cap]); best = None
    for g in GAMMA:
        X = Zm * v40._shape(d, g)[:, None]; b, *_ = np.linalg.lstsq(X, y, rcond=None); sse = float(((X @ b - y) ** 2).sum())
        if best is None or sse < best[0]: best = (sse, float(g), b.tolist())
    return {"gamma": best[1], "beta": best[2]}
def fit_A1(rows, cap, c, s):
    X = np.array([Z(r["phi"], c, s) * v40._shape(r["d"], 1.0) for r in rows if r["cap"] == cap]); y = np.array([r["y"] for r in rows if r["cap"] == cap])
    return np.linalg.lstsq(X, y, rcond=None)[0].tolist()
def fit_A2(rows, cap, c, s):
    out = {}
    for d0 in SEEN_D:
        X = np.array([Z(r["phi"], c, s) for r in rows if r["cap"] == cap and r["d"] == d0]); y = np.array([r["y"] for r in rows if r["cap"] == cap and r["d"] == d0])
        out[str(d0)] = np.linalg.lstsq(X, y, rcond=None)[0].tolist()
    return out
def fit_cont(rows, cap, c, s):
    """dL = (beta.phi) s + (zeta.phi) s^2, s = 1-d; 8 params/cap; single lstsq."""
    X = np.array([np.concatenate([Z(r["phi"], c, s) * (1 - r["d"]), Z(r["phi"], c, s) * (1 - r["d"]) ** 2]) for r in rows if r["cap"] == cap])
    y = np.array([r["y"] for r in rows if r["cap"] == cap]); b, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    return {"coef": b.tolist(), "rank": int(rank), "n_params": X.shape[1], "n_obs": len(y)}
def fit_strength(rows, cap):
    y = np.array([r["y"] for r in rows if r["cap"] == cap]); d = np.array([r["d"] for r in rows if r["cap"] == cap]); best = None
    for g in GAMMA:
        sh = v40._shape(d, g); A = float((sh * y).sum() / (sh * sh).sum()); sse = float(((A * sh - y) ** 2).sum())
        if best is None or sse < best[0]: best = (sse, float(g), A)
    return {"gamma": best[1], "A": best[2]}
def fit_median(rows, cap):
    meds = {d: float(np.median([r["y"] for r in rows if r["cap"] == cap and r["d"] == d])) for d in SEEN_D}; best = None
    for p in GAMMA:
        sh = np.array([v40._shape(d, p) for d in SEEN_D]); yv = np.array([meds[d] for d in SEEN_D]); cc = float((sh * yv).sum() / (sh * sh).sum())
        sse = float(((cc * sh - yv) ** 2).sum())
        if best is None or sse < best[0]: best = (sse, float(p), cc)
    return {"p": best[1], "c": best[2]}
def A2_predict(coefs, z, d):
    pred = {float(k): float(np.array(v) @ z) for k, v in coefs.items()}; ds = sorted(pred)
    if d in pred: return pred[d]
    lo = [x for x in ds if x < d]; hi = [x for x in ds if x > d]
    if lo and hi: a, b = lo[-1], hi[0]                       # in-range: adjacent fitted densities
    elif hi: a, b = ds[0], ds[1]                             # below range: two lowest (0.6, 0.7)
    else: a, b = ds[-2], ds[-1]                              # above range: two highest
    return pred[a] + (pred[b] - pred[a]) / (b - a) * (d - a)

def freeze():
    reg = {"frozen_at_utc": datetime.datetime.utcnow().isoformat() + "Z", "protocols": {}}
    for P, steps in PROTO.items():
        rows = dev_rows(steps); c, s = zstats(rows)
        qtrain, qh = v36.load_grid(sizes=("160m", "410m", "1.4b")); qtrain = [r for r in qtrain if r["step"] in steps]
        pr = {"dev_steps": steps, "n_dev_rows": len(rows), "center": c.tolist(), "scale": s.tolist(), "prune": {}, "quant": {}}
        for cap in CAPS:
            pr["prune"][cap] = {"power": fit_power(rows, cap, c, s), "A1": fit_A1(rows, cap, c, s), "A2": fit_A2(rows, cap, c, s),
                                "cont": fit_cont(rows, cap, c, s), "strength_only": fit_strength(rows, cap), "median_curve": fit_median(rows, cap)}
            tr = [r for r in qtrain if r["arm"] == "quantization" and r["capability"] == cap]
            fB = v36.fit_direct(tr, input_fields=("N0", "L0", "D0")); fA = v36.fit_direct(tr, input_fields=("N0", "L0"))
            pr["quant"][cap] = {"full": {k: fB[k] for k in ("center", "scale", "coefficients", "feature_names", "arm", "with_d0", "input_fields")},
                                "noD0": {k: fA[k] for k in ("center", "scale", "coefficients", "feature_names", "arm", "with_d0", "input_fields")},
                                "per_bit_mean": {str(b): float(np.mean([r["observed"] for r in tr if r["config"] == b])) for b in BITS_DEV},
                                "per_bit_median": {str(b): float(np.median([r["observed"] for r in tr if r["config"] == b])) for b in BITS_DEV}}
        reg["protocols"][P] = pr
    reg["dev_hashes"] = qh; reg["rules"] = {"A2": "in-range linear between adjacent fitted densities; out-of-range linear extension from boundary pair; anchors are predictions",
                                            "bit5": "linear interpolation of the 4- and 6-bit predictions (interpolation-rule baseline)"}
    (OUT / "register.json").write_text(json.dumps(reg, indent=2, default=float)); print("frozen ->", OUT / "register.json")

def predict(tag):
    size, step = tag.split("@"); step = int(step.replace("step", ""))
    dense = json.loads((OUT / f"dense_{tag}.json").read_text())["dense"]; reg = json.loads((OUT / "register.json").read_text())
    N0, D0 = v36.matrix_n0(size), step * v36.TOKENS_PER_STEP; out = {"target": {"tag": tag, "N0": N0, "D0": D0, "dense": dense}, "protocols": {}}
    for P, pr in reg["protocols"].items():
        c, s = np.array(pr["center"]), np.array(pr["scale"]); res = {"prune": {}, "quant": {}}
        for cap in CAPS:
            z = Z([np.log(N0), dense[cap], np.log(D0)], c, s); f = pr["prune"][cap]
            for d in TEST_D:
                sd = 1 - d; res["prune"][f"{cap}|{d}"] = {
                    "power": float(np.array(f["power"]["beta"]) @ z) * v40._shape(d, f["power"]["gamma"]),
                    "A2": A2_predict(f["A2"], z, d), "A1": float(np.array(f["A1"]) @ z) * v40._shape(d, 1.0),
                    "cont": float(np.concatenate([z * sd, z * sd ** 2]) @ np.array(f["cont"]["coef"])),
                    "strength_only": f["strength_only"]["A"] * v40._shape(d, f["strength_only"]["gamma"]),
                    "median_curve": f["median_curve"]["c"] * v40._shape(d, f["median_curve"]["p"]), "zero": 0.0}
            q = pr["quant"][cap]; row = {"arm": "quantization", "capability": cap, "N0": N0, "D0": D0, "L0": dense[cap]}
            def cls(fit, fields, b):
                return float(v36.predict({**fit, "with_d0": "D0" in fields}, [v36.basic_input({**row, "config": b}, input_fields=fields)])[0])
            for b in TEST_B:
                if b in BITS_DEV:
                    res["quant"][f"{cap}|{b}"] = {"full": cls(q["full"], ("N0", "L0", "D0"), b), "noD0": cls(q["noD0"], ("N0", "L0"), b),
                                                  "per_bit_mean": q["per_bit_mean"][str(b)], "per_bit_median": q["per_bit_median"][str(b)], "zero": 0.0}
                else:  # 5-bit rule
                    lo = {k: v for k, v in res["quant"][f"{cap}|4"].items()}; hi = {k: v for k, v in res["quant"][f"{cap}|6"].items()}
                    res["quant"][f"{cap}|{b}"] = {k: 0.5 * (lo[k] + hi[k]) for k in lo}; res["quant"][f"{cap}|{b}"]["rule"] = "interp(4,6)"
        out["protocols"][P] = res
    out["provenance"] = {"register_sha": hashlib.sha256((OUT / "register.json").read_bytes()).hexdigest(), "written_utc": datetime.datetime.utcnow().isoformat() + "Z"}
    (OUT / f"predictions_{tag}.json").write_text(json.dumps(out, indent=2, default=float)); print("predictions ->", OUT / f"predictions_{tag}.json")

if __name__ == "__main__":
    {"freeze": freeze, "predict": lambda: predict(sys.argv[2])}[sys.argv[1]]()
