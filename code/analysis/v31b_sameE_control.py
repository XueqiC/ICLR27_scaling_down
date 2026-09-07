#!/usr/bin/env python3
"""V31b: matched REUSE-COUNT (E=T/D_U) control for distillation.

v31 showed a large pool-size effect at matched SEEN-TOKENS, but that confounds pool size
with reuse count E=T/D_U (8x gap). Here we compare U=75 at its own E-matched checkpoints
(T=62775->E~0.94, 125550->E~1.87) against U=600 at T=500k (E~0.94) and 1000k (E~1.87).
If matching E collapses the pool-size gap, a reuse-count coordinate is preferred.

U=75 same-E runs live under *_uxseenE; U=600 under *_uxseen. delta_c = L_c(KD)-L_c(S0).
CPU-only; writes results/v31-uxseen/sameE_summary.json.
"""
from __future__ import annotations
import json, math
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "results/v12-distill/gemma3-1b"
OUT = Path(__file__).resolve().parents[1] / "results/v31-uxseen/sameE_summary.json"
CAPS = ("math", "code", "qa")
DU = {75: 67027, 600: 533869}
# (E label, U75 target tokens, U600 target tokens)
PAIRS = (("0.94", 62775, 500000), ("1.87", 125550, 1000000))
T_CRIT = {1: 12.71, 2: 4.303, 3: 3.182}


def _snap(sd):
    for c in ("snapshot.json", "eval.json"):
        if (sd / c).exists():
            return json.load(open(sd / c))
    js = list(sd.glob("*.json"))
    return json.load(open(js[0])) if js else None


def _d(d, cap):
    return d["delta"][cap] if d and "delta" in d and cap in d["delta"] else None


def _tk(d):
    for k in ("processed_tokens", "tokens_seen", "seen_tokens"):
        if d and k in d:
            return d[k]


def series(run):
    o = {}
    for s in sorted((BASE / run / "trajectory").glob("update-*")):
        d = _snap(s); t = _tk(d)
        if isinstance(t, (int, float)) and t > 0:
            o[t] = d
    return o


def near(p, t):
    return min(p, key=lambda k: abs(k - t))


def ci(v):
    n = len(v); m = sum(v) / n
    if n < 2:
        return m, 0.0, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    return m, sd, T_CRIT[n - 1] * sd / math.sqrt(n)


def main():
    u75 = {s: series(f"gpt-5.6-luna_full_75_uxseenE{'' if s == 0 else '_seed' + str(s)}") for s in (0, 1, 2)}
    u600 = {s: series(f"gpt-5.6-luna_full_600_uxseen{'' if s == 0 else '_seed' + str(s)}") for s in (0, 1, 2)}
    out = {"design": "matched reuse-count E=T/D_U; U75(_uxseenE) vs U600(_uxseen); delta_c=L(KD)-L(S0)",
           "D_U": DU, "reuse_pairs": [{"E": e, "U75_T": a, "U600_T": b} for e, a, b in PAIRS],
           "matched_E": {}}
    for E, t75, t600 in PAIRS:
        blk = {}
        for cap in CAPS:
            a75, a6, pair = [], [], []
            for s in (0, 1, 2):
                x = _d(u75[s][near(u75[s], t75)], cap)
                y = _d(u600[s][near(u600[s], t600)], cap)
                if x is None or y is None:
                    continue
                a75.append(x); a6.append(y); pair.append(x - y)
            m75, _, h75 = ci(a75); m6, _, h6 = ci(a6); mp, _, hp = ci(pair)
            blk[cap] = {"U75": m75, "U75_hw": h75, "U600": m6, "U600_hw": h6,
                        "U75_minus_U600": mp, "ci95": [mp - hp, mp + hp],
                        "excludes_zero": bool((mp - hp) * (mp + hp) > 0)}
        out["matched_E"][E] = blk
    OUT.write_text(json.dumps(out, indent=2))
    for E, _, _ in PAIRS:
        print(f"\n=== matched reuse E~{E} ===")
        for cap in CAPS:
            b = out["matched_E"][E][cap]
            print(f"  {cap:4s}: U75 {b['U75']:+.3f}  U600 {b['U600']:+.3f} | "
                  f"U75-U600 {b['U75_minus_U600']:+.3f} CI[{b['ci95'][0]:+.3f},{b['ci95'][1]:+.3f}] "
                  f"{'EXCL0' if b['excludes_zero'] else 'incl0'}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
