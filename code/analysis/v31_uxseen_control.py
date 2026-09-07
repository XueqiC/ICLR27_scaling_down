#!/usr/bin/env python3
"""V31: unique-data x seen-tokens control (distillation).

Answers the advisor question: is the distillation capability response driven by
UNIQUE-DATA diversity or by cumulative SEEN-TOKEN exposure? We hold seen-tokens
fixed and vary the unique-example pool size U in {75, 600}, evaluating capability
loss at matched processed-token milestones from the v12 trajectory snapshots.

Each of the six continuous runs (U in {75,600} x seed in {0,1,2}, gemma3-1b, teacher
gpt-5.6-luna, recipe full, LoRA) saved a trajectory checkpoint at ~500k and ~1000k
processed tokens. delta_c = L_c(S_KD) - L_c(S0). The two checkpoints of one run are
PAIRED observations (one trajectory), never independent seeds. The paired contrast is
U=75 minus U=600 at matched seen-tokens; a positive value means the small (heavily
repeated) pool damages capability more than the large pool at the same training volume.

CPU-only re-analysis of saved JSON; writes results/v31-uxseen/summary.json.
"""
from __future__ import annotations
import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/v12-distill/gemma3-1b"
OUT = ROOT / "results/v31-uxseen"
CAPS = ("math", "code", "qa")
SEEDS = ((("", 0)), ("_seed1", 1), ("_seed2", 2))
POOLS = (75, 600)
MILESTONES = (500_000, 1_000_000)
T_CRIT = {1: 12.71, 2: 4.303, 3: 3.182}  # two-sided 95%, df=n-1


def _snap(sd: Path):
    for c in ("snapshot.json", "eval.json"):
        if (sd / c).exists():
            return json.load(open(sd / c))
    js = list(sd.glob("*.json"))
    return json.load(open(js[0])) if js else None


def _delta(d, cap):
    return d["delta"][cap] if d and "delta" in d and cap in d["delta"] else None


def _tokens(d):
    for k in ("processed_tokens", "tokens_seen", "seen_tokens"):
        if d and k in d:
            return d[k]
    return None


def series(run):
    pts = {}
    for sd in sorted((BASE / run / "trajectory").glob("update-*")):
        d = _snap(sd)
        t = _tokens(d)
        if isinstance(t, (int, float)) and t > 0:
            pts[t] = d
    return pts


def nearest(pts, target):
    ks = list(pts)
    return min(ks, key=lambda k: abs(k - target)) if ks else None


def ci(vals):
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0.0, 0.0
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    se = sd / math.sqrt(n)
    return m, sd, T_CRIT[n - 1] * se


def main():
    runs = {}
    for U in POOLS:
        for suf, s in SEEDS:
            runs[(U, s)] = series(f"gpt-5.6-luna_full_{U}_uxseen{suf}")

    out = {"design": {
        "student": "gemma3-1b", "teacher": "gpt-5.6-luna", "recipe": "full",
        "pools_U": list(POOLS), "seeds": [s for _, s in SEEDS],
        "milestones_seen_tokens": list(MILESTONES),
        "note": "U=75 reaches a milestone via repetition (many epochs); U=600 via more unique data. "
                "Paired contrast = U75 - U600 at matched seen-tokens. delta_c = L_c(KD) - L_c(S0).",
        "checkpoints_paired_not_independent": True,
    }, "milestones": {}}

    for T in MILESTONES:
        block = {}
        for cap in CAPS:
            d75, d600, pair = [], [], []
            for s in (0, 1, 2):
                p75, p6 = runs[(75, s)], runs[(600, s)]
                k75, k6 = nearest(p75, T), nearest(p6, T)
                a, b = _delta(p75[k75], cap), _delta(p6[k6], cap)
                if a is None or b is None:
                    continue
                d75.append(a); d600.append(b); pair.append(a - b)
            m75, _, h75 = ci(d75)
            m6, _, h6 = ci(d600)
            mp, sdp, hp = ci(pair)
            block[cap] = {
                "U75_deltaL_mean": m75, "U75_ci95_halfwidth": h75,
                "U600_deltaL_mean": m6, "U600_ci95_halfwidth": h6,
                "paired_U75_minus_U600_mean": mp,
                "paired_ci95": [mp - hp, mp + hp],
                "paired_seed_sd": sdp,
                "excludes_zero": bool((mp - hp) * (mp + hp) > 0),
                "n_seeds": len(pair),
            }
            out["milestones"][str(T)] = block

    # --- D_U pool-token accounting and reuse count E = T / D_U (advisor) ---
    du = {}
    for U in POOLS:
        tl = BASE / f"gpt-5.6-luna_full_{U}_uxseen" / "train_log.json"
        d = json.load(open(tl)) if tl.exists() else {}
        du[U] = {"D_U_pool_tokens": d.get("unique_data_pool_tokens"),
                 "pool_examples": d.get("unique_data_pool_examples"),
                 "processed_tokens_total": d.get("tokens_seen"),
                 "supervised_tokens_total": d.get("completion_tokens_seen")}
    out["D_U_accounting"] = du
    out["reuse_count_E"] = {str(U): {str(T): (T / du[U]["D_U_pool_tokens"] if du[U]["D_U_pool_tokens"] else None)
                                     for T in MILESTONES} for U in POOLS}
    out["same_E_U75_milestones"] = {str(T): round((T / du[600]["D_U_pool_tokens"]) * du[75]["D_U_pool_tokens"])
                                     for T in MILESTONES}
    out["seed_subset_caveat"] = ("data_selection='first n rows' with per-seed shuffle only -> all 3 seeds "
                                 "SHARE the same U-subset; intervals reflect training/shuffle randomness on a "
                                 "FIXED subset, not data-subset resampling. Next round: stratified pool resample "
                                 "by domain+length, separate data-subset seed from training seed.")

    # --- T-direction paired CI per pool: delta(1000k)-delta(500k) ---
    Tlo, Thi = MILESTONES
    tdir = {}
    for U in POOLS:
        tdir[str(U)] = {}
        for cap in CAPS:
            vals = []
            for s in (0, 1, 2):
                p = runs[(U, s)]
                a, b = _delta(p[nearest(p, Thi)], cap), _delta(p[nearest(p, Tlo)], cap)
                if a is not None and b is not None:
                    vals.append(a - b)
            m, sd, h = ci(vals)
            tdir[str(U)][cap] = {"mean": m, "ci95": [m - h, m + h],
                                 "excludes_zero": bool((m - h) * (m + h) > 0)}
    out["T_direction_paired"] = tdir

    # --- interaction I_c = [dU75(Thi)-dU75(Tlo)] - [dU600(Thi)-dU600(Tlo)] per-seed ---
    inter = {}
    for cap in CAPS:
        vals = []
        for s in (0, 1, 2):
            p7, p6 = runs[(75, s)], runs[(600, s)]
            d7 = _delta(p7[nearest(p7, Thi)], cap) - _delta(p7[nearest(p7, Tlo)], cap)
            d6 = _delta(p6[nearest(p6, Thi)], cap) - _delta(p6[nearest(p6, Tlo)], cap)
            vals.append(d7 - d6)
        m, sd, h = ci(vals)
        inter[cap] = {"mean": m, "ci95": [m - h, m + h], "excludes_zero": bool((m - h) * (m + h) > 0)}
    out["interaction_I_c"] = inter

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=2))
    # console table
    for T in MILESTONES:
        print(f"\n=== matched seen-tokens ~{T} ===")
        for cap in CAPS:
            b = out["milestones"][str(T)][cap]
            print(f"  {cap:4s}: U75 {b['U75_deltaL_mean']:+.3f}  U600 {b['U600_deltaL_mean']:+.3f}  "
                  f"| U75-U600 {b['paired_U75_minus_U600_mean']:+.3f} "
                  f"CI[{b['paired_ci95'][0]:+.3f},{b['paired_ci95'][1]:+.3f}] "
                  f"{'EXCL0' if b['excludes_zero'] else 'incl0'}")
    print(f"\nwrote {OUT/'summary.json'}")


if __name__ == "__main__":
    main()
