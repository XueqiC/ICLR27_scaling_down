#!/usr/bin/env python3
"""V3: measurement-layer audit on the 8B-teacher campaign item data.

Three tests the prior work never ran (irt_audit.md §a/§c/§h):
  (c) trained-half vs transfer-half split: distillation traces came from
      gsm8k/hotpotqa/humaneval; each domain panel is 50% that benchmark,
      50% a never-trained sibling (math500/2wiki/mbpp). If distillation
      "domain gains" concentrate on the trained half, they are partly
      format/distribution fit, not capability.
  (a) 2PL vs 1PL: fit both jointly; report discrimination spread, AIC,
      and theta rank agreement — was equal-discrimination defensible?
  (b) DIF: per item, does the distilled group over/under-perform relative
      to base respondents at matched ability? Flags item-level bias of the
      panel, split by trained/transfer half.

Data: local eval JSONs of the 8B-teacher campaign (M0 distill / M1
answer_only / M2 soft_kd / M3 onpolicy, sizes 0.6-8B, seeds 1-3) + dense
bases. Outputs: results/v3-measurement-audit/{summary.md, *.json}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "prior/xueqi-intern-backup/docs/scaling-down/results/runs"
OUT = ROOT / "results/v3-measurement-audit"

DOMAINS = {
    "math": {"trained": "gsm8k", "transfer": "math500"},
    "qa": {"trained": "hotpotqa", "transfer": "2wiki"},
    "code": {"trained": "humaneval", "transfer": "mbpp"},
}
# run dirs holding eval files for each task
TASK_DIRS = {
    "gsm8k": ["raw-qwen3-pilots/scaling-pilot-qwen3-8b-nonthinking-2026-07-12",
              "answer-only-qwen3-2026-07-16", "soft-kd-qwen3-2026-07-19",
              "onpolicy-qwen3-2026-07-21"],
    "math500": ["math500-qwen3-2026-07-22"],
    "hotpotqa": ["hotpotqa-qwen3-2026-07-15", "answer-only-qwen3-2026-07-16",
                 "soft-kd-qwen3-2026-07-19", "onpolicy-qwen3-2026-07-21"],
    "2wiki": ["2wiki-qwen3-2026-07-22"],
    "humaneval": ["humaneval-qwen3-2026-07-15", "answer-only-qwen3-2026-07-16",
                  "soft-kd-qwen3-2026-07-19", "onpolicy-qwen3-2026-07-21"],
    "mbpp": ["mbpp-qwen3-2026-07-22"],
}
RECIPES = {"distill": "M0", "answer_only": "M1", "soft_kd": "M2",
           "onpolicy": "M3", "gold": None, "fullft": None}
FN = re.compile(r"^(?P<recipe>[a-z_0-9]+?)(?:-(?P<task>gsm8k|hotpotqa|"
                r"humaneval|math500|2wiki|mbpp))?-(?P<size>\d+p?\d*b)"
                r"-s(?P<seed>\d)\.json$")


def read_items(path: Path, task: str) -> dict[str, int]:
    with open(path) as fh:
        d = json.load(fh)
    out = {}
    for it in d["items"]:
        iid = (f"{task}:{it['task_id']}" if it.get("task_id")
               else f"{task}:{int(it['i'])}")
        s = it.get("score")
        out[iid] = int(float(s) >= 0.5) if s not in (None, "") else 0
    return out


def collect(task: str) -> dict[tuple, dict[str, int]]:
    """respondent key -> {item_id: 0/1}. Key: ('base', size) or
    (recipe_label, size, seed)."""
    resp: dict[tuple, dict[str, int]] = {}
    for dname in TASK_DIRS[task]:
        d = RUNS / dname
        for f in sorted((d / "eval-base").glob("base-*.json")):
            size = f.stem.split("-")[1]
            resp[("base", size)] = read_items(f, task)
        for f in sorted((d / "eval").glob("*.json")):
            m = FN.match(f.name)
            if not m:
                continue
            if m.group("task") not in (None, task):
                continue
            # task-titled dirs have no task in filename; arm dirs do
            if m.group("task") is None and dname.split("-")[0] not in (
                    task, "raw", "scaling"):
                # e.g. 'distill-0p6b-s1.json' inside math500 dir -> ok,
                # dir prefix names the task; keep (task passed by caller)
                pass
            rec = RECIPES.get(m.group("recipe"))
            if rec is None:
                continue
            key = (rec, m.group("size"), int(m.group("seed")))
            resp.setdefault(key, {}).update(read_items(f, task))
    return resp


def split_table(dom: str) -> tuple[list[str], dict]:
    cfg = DOMAINS[dom]
    tr = collect(cfg["trained"])
    tf = collect(cfg["transfer"])
    lines = [f"### {dom}: trained ({cfg['trained']}) vs transfer "
             f"({cfg['transfer']})",
             "| recipe | n | Δacc trained | Δacc transfer | trained−transfer "
             "[90% CI] | 读法 |", "|---|---|---|---|---|---|"]
    out = {}
    rng = np.random.default_rng(0)
    for rec in ("M0", "M1", "M2", "M3"):
        pairs = []
        for key in tr:
            if key[0] != rec or key not in tf:
                continue
            base_tr = tr.get(("base", key[1]))
            base_tf = tf.get(("base", key[1]))
            if not base_tr or not base_tf:
                continue
            a_tr = np.mean(list(tr[key].values()))
            a_tf = np.mean(list(tf[key].values()))
            b_tr = np.mean(list(base_tr.values()))
            b_tf = np.mean(list(base_tf.values()))
            pairs.append((a_tr - b_tr, a_tf - b_tf))
        if len(pairs) < 4:
            continue
        p = np.asarray(pairs)
        diff = p[:, 0] - p[:, 1]
        boots = [np.mean(diff[rng.integers(0, len(diff), len(diff))])
                 for _ in range(2000)]
        lo, hi = np.quantile(boots, [0.05, 0.95])
        sig = "训练半区偏高" if lo > 0 else ("迁移半区偏高" if hi < 0 else "无分裂")
        lines.append(f"| {rec} | {len(pairs)} | {p[:,0].mean():+.3f} | "
                     f"{p[:,1].mean():+.3f} | {diff.mean():+.3f} "
                     f"[{lo:+.3f},{hi:+.3f}] | {sig} |")
        out[rec] = {"n": len(pairs), "d_trained": float(p[:, 0].mean()),
                    "d_transfer": float(p[:, 1].mean()),
                    "gap": float(diff.mean()), "ci90": [float(lo), float(hi)]}
    lines.append("")
    return lines, out


def build_matrix(dom: str):
    cfg = DOMAINS[dom]
    both = {}
    for task in (cfg["trained"], cfg["transfer"]):
        for key, items in collect(task).items():
            both.setdefault(key, {}).update(items)
    keys = sorted(both, key=str)
    iids = sorted({i for v in both.values() for i in v})
    Y = np.full((len(keys), len(iids)), -1, dtype=int)
    idx = {i: j for j, i in enumerate(iids)}
    for r, key in enumerate(keys):
        for i, y in both[key].items():
            Y[r, idx[i]] = y
    return keys, iids, Y


def fit_irt(Y: np.ndarray, two_pl: bool):
    n_r, n_i = Y.shape
    mask = Y >= 0
    Yc = np.clip(Y, 0, 1)

    def unpack(v):
        theta = v[:n_r]
        b = v[n_r:n_r + n_i]
        loga = v[n_r + n_i:] if two_pl else np.zeros(n_i)
        return theta, b, loga

    def nll(v):
        theta, b, loga = unpack(v)
        a = np.exp(loga)
        z = a[None, :] * (theta[:, None] - b[None, :])
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        ll = np.where(mask, Yc * np.log(p) + (1 - Yc) * np.log(1 - p), 0.0)
        pen = 0.1 * (theta @ theta + b @ b)
        if two_pl:
            pen += 2.0 * (loga @ loga)  # log a ~ N(0, 0.5)
        return -(ll.sum()) + pen

    v0 = np.zeros(n_r + n_i + (n_i if two_pl else 0))
    res = minimize(nll, v0, method="L-BFGS-B",
                   options={"maxiter": 2000, "maxfun": 200000})
    theta, b, loga = unpack(res.x)
    theta = theta - theta.mean()
    z = np.exp(loga)[None, :] * (theta[:, None] - (b - b.mean() * 0)[None, :])
    p = np.clip(1 / (1 + np.exp(-z)), 1e-9, 1 - 1e-9)
    ll = float(np.where(mask, Yc * np.log(p) + (1 - Yc) * np.log(1 - p),
                        0).sum())
    return {"theta": theta, "b": b, "loga": loga, "ll": ll,
            "converged": bool(res.success)}


def dif_scan(keys, iids, Y, theta, b, dom):
    """Per item: MLE group offset gamma for distilled vs base at fixed
    theta/b; Wald z. Positive gamma = distilled group over-performs."""
    grp = np.array([0 if k[0] == "base" else 1 for k in keys])
    use_r = np.array([k[0] in ("base", "M0", "M1", "M2", "M3")
                      for k in keys])
    out = []
    for j, iid in enumerate(iids):
        rows = (Y[:, j] >= 0) & use_r
        if rows.sum() < 20 or len(set(grp[rows])) < 2:
            continue
        y = Y[rows, j]
        t = theta[rows]
        g = grp[rows]
        from scipy.optimize import minimize_scalar

        def nll(gam):
            z = t - b[j] + gam * g
            p = np.clip(1 / (1 + np.exp(-z)), 1e-9, 1 - 1e-9)
            return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        r = minimize_scalar(nll, bounds=(-6, 6), method="bounded")
        gam = float(r.x)
        z = t - b[j] + gam * g
        p = 1 / (1 + np.exp(-z))
        info = float(np.sum(p * (1 - p) * g))
        se = 1 / np.sqrt(info) if info > 0 else np.inf
        out.append({"item": iid, "gamma": gam, "z": gam / se,
                    "half": ("trained" if iid.split(":")[0] ==
                             DOMAINS[dom]["trained"] else "transfer")})
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# V3: measurement-layer audit (8B-teacher campaign)",
             "", "## (c) trained-half vs transfer-half split", ""]
    split_out = {}
    for dom in DOMAINS:
        ls, o = split_table(dom)
        lines += ls
        split_out[dom] = o

    lines += ["## (a) 1PL vs 2PL and (b) DIF", ""]
    irt_out = {}
    for dom in DOMAINS:
        keys, iids, Y = build_matrix(dom)
        f1 = fit_irt(Y, two_pl=False)
        f2 = fit_irt(Y, two_pl=True)
        from scipy.stats import spearmanr
        rho = float(spearmanr(f1["theta"], f2["theta"]).statistic)
        n_i = len(iids)
        aic1 = 2 * (len(keys) + n_i) - 2 * f1["ll"]
        aic2 = 2 * (len(keys) + 2 * n_i) - 2 * f2["ll"]
        a = np.exp(f2["loga"])
        dif = dif_scan(keys, iids, Y, f1["theta"], f1["b"], dom)
        flag = [d for d in dif if abs(d["gamma"]) > 1 and abs(d["z"]) > 2.58]
        n_tr = sum(1 for d in flag if d["half"] == "trained")
        lines.append(
            f"### {dom}: {len(keys)} respondents × {n_i} items | "
            f"2PL a: median {np.median(a):.2f}, IQR "
            f"[{np.quantile(a,.25):.2f},{np.quantile(a,.75):.2f}], "
            f"SD(log a) {np.std(f2['loga']):.2f} | ΔAIC(2PL−1PL) "
            f"{aic2-aic1:+.0f} | Spearman(θ1,θ2) {rho:.3f} | "
            f"DIF flags |γ|>1,z>2.58: {len(flag)}/{len(dif)} "
            f"({n_tr} trained-half)")
        worst = sorted(flag, key=lambda d: -abs(d["gamma"]))[:5]
        for d in worst:
            lines.append(f"  - {d['item']} γ={d['gamma']:+.2f} "
                         f"z={d['z']:+.1f} ({d['half']})")
        irt_out[dom] = {"n_resp": len(keys), "n_items": n_i,
                        "sd_loga": float(np.std(f2["loga"])),
                        "aic1": aic1, "aic2": aic2, "spearman": rho,
                        "dif_flags": len(flag), "dif_total": len(dif),
                        "dif_trained": n_tr}
        lines.append("")

    with open(OUT / "split.json", "w") as fh:
        json.dump(split_out, fh, indent=1)
    with open(OUT / "irt_dif.json", "w") as fh:
        json.dump(irt_out, fh, indent=1)
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
