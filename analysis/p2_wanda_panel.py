#!/usr/bin/env python3
"""P2: one Wanda pruning panel against the V6 magnitude panel (retrospective, small).

Wanda (Sun et al., 2024) scores each weight by |W_ij| * ||X_j||_2, where ||X_j||_2 is the
L2 norm of input feature j over a calibration set, and prunes per output row (each row keeps
its top-k scores). It applies to the linear layers inside the transformer blocks; embeddings
and the LM head stay dense, as in the reference implementation. Density here is the retained
fraction of the weights in that scope.

Stages
  --run       GPU: for each model, collect activation norms on the calibration set, then
              measure the V6 capability losses (same probes as `stage_prune`: the odd half of
              `build_probes(n_probe)`) dense and at each density, restoring dense weights in
              between. Writes results/p2-wanda-panel/<tag>/wanda_losses.json.
  --analyse   CPU: pair each Wanda cell with the V6 magnitude cell at the same density, and
              fit the pruning response shape: loss change = A * (1 - d)^alpha per capability,
              with alpha fixed from the magnitude cells (scale refit only) against alpha free;
              report the mean absolute error of each fit and of a one-point (K1) calibration.
              Writes summary.json / summary.md and paper/paper/tables/wanda_panel.tex.

Calibration set: the first 128 documents of the C4 English validation split with at least
512 tokens, truncated to 512 tokens (the Wanda default is 128 x 2048 from C4 train; the
shorter context keeps the pass cheap on a shared card and the norms are per-feature averages).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/p2-wanda-panel"
MODELS = {"qwen3-4b": "Qwen/Qwen3-4B", "gemma3-4b": "google/gemma-3-4b-pt"}
V6_DIRS = {"qwen3-4b": "Qwen3-4B", "gemma3-4b": "gemma3-4b"}
DENSITIES = (0.9, 0.8, 0.7)
CAPS = ("math", "code", "qa")
N_CALIB, CALIB_TOKENS, N_PROBE, SEED = 128, 512, 128, 0


def read(p):
    return json.loads(Path(p).read_text())


def calibration_texts(tok, n=N_CALIB, tokens=CALIB_TOKENS):
    """First n C4 validation documents with at least `tokens` tokens, truncated."""
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    ids = []
    for row in ds:
        enc = tok(row["text"], add_special_tokens=False)["input_ids"]
        if len(enc) >= tokens:
            ids.append(enc[:tokens])
        if len(ids) >= n:
            break
    return ids


def block_linears(model):
    """Linear layers inside the transformer blocks: name -> module (embeddings and LM head excluded)."""
    import torch
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" not in name and "embed" not in name and ".layers." in name:
            out[name] = mod
    if not out:
        raise RuntimeError("no block linear layers found")
    return out


def activation_norms(model, tok, linears, device, calib_ids):
    """Per-layer sqrt(sum over calibration tokens of x_j^2 / n_tokens) for every input feature j."""
    import torch
    sums = {n: torch.zeros(m.in_features, dtype=torch.float64, device=device) for n, m in linears.items()}
    counts = {n: 0 for n in linears}
    hooks = []
    for name, mod in linears.items():
        def hook(module, inputs, name=name):
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1]).to(torch.float64)
            sums[name] += (x * x).sum(dim=0)
            counts[name] += x.shape[0]
        hooks.append(mod.register_forward_pre_hook(hook))
    model.eval()
    with torch.no_grad():
        for ids in calib_ids:
            model(torch.tensor([ids], device=device))
    for h in hooks:
        h.remove()
    return {n: torch.sqrt(sums[n] / max(counts[n], 1)).to(torch.float32) for n in linears}


def apply_wanda(linears, norms, density, reference):
    """Per-output-row pruning by |W| * ||X||; restores from `reference` first so densities are independent."""
    import torch
    with torch.no_grad():
        for name, mod in linears.items():
            w = reference[name].to(mod.weight.device, non_blocking=True)
            metric = w.abs().to(torch.float32) * norms[name].to(mod.weight.device)[None, :]
            k = max(1, int(round(density * w.shape[1])))
            idx = torch.topk(metric, k, dim=1).indices
            mask = torch.zeros_like(metric, dtype=torch.bool).scatter_(1, idx, True)
            mod.weight.copy_(w * mask)


def measure(model, tok, probes, device):
    import torch
    from analysis.v6_capability_geometry import completion_loss
    res = {}
    with torch.no_grad():
        for cap, samples in probes.items():
            tot, ntok = 0.0, 0
            for s in samples:
                loss, n = completion_loss(model, tok, s["prompt"], s["completion"], device)
                tot += float(loss); ntok += n
            res[cap] = tot / max(ntok, 1)
    return res


def run(tags, device, densities=DENSITIES):
    """Measure the requested densities; an existing record is extended (its dense anchor is kept)."""
    import torch
    from analysis.v6_capability_geometry import build_probes, load_text_causal_lm
    probes = {c: v[1::2] for c, v in build_probes(N_PROBE).items()}
    for tag in tags:
        out = OUT / tag; out.mkdir(parents=True, exist_ok=True)
        target = out / "wanda_losses.json"
        previous = read(target) if target.exists() else None
        todo = [d for d in densities if not (previous and str(d) in previous["losses"])]
        if not todo:
            print(f"[wanda] {tag}: nothing to measure"); continue
        t0 = time.time()
        model, tok = load_text_causal_lm(MODELS[tag], torch.bfloat16, None)
        model.to(device).eval()
        linears = block_linears(model)
        scope = sum(m.weight.numel() for m in linears.values())
        total = sum(p.numel() for _, p in model.named_parameters() if p.dim() >= 2)
        calib = calibration_texts(tok)
        norms = activation_norms(model, tok, linears, device, calib)
        reference = {n: m.weight.detach().clone().cpu() for n, m in linears.items()}
        results = dict(previous["losses"]) if previous else {"1.0": measure(model, tok, probes, device)}
        print(f"[wanda] {tag} dense:", results["1.0"], flush=True)
        for d in todo:
            apply_wanda(linears, norms, d, reference)
            results[str(d)] = measure(model, tok, probes, device)
            print(f"[wanda] {tag} d={d}:", results[str(d)], flush=True)
        record = {"model": MODELS[tag], "tag": tag, "method": "wanda", "scope": "block linear layers",
                  "scope_parameters": scope, "matrix_parameters_total": total, "n_layers": len(linears),
                  "calibration": {"dataset": "allenai/c4 en validation (streaming)", "documents": len(calib),
                                  "tokens_per_document": CALIB_TOKENS},
                  "probes": {"n_probe": N_PROBE, "half": "odd", "seed": SEED},
                  "densities": sorted({float(k) for k in results if k != "1.0"}, reverse=True),
                  "losses": results, "seconds": (previous or {}).get("seconds", 0) + time.time() - t0}
        target.write_text(json.dumps(record, indent=1))
        del model, tok, reference, norms; gc.collect(); torch.cuda.empty_cache()


# ---------------------------------------------------------------- analysis
def fit_power(ds, ys, alpha=None):
    """Least squares for y = A * (1-d)^alpha on log scale; y must be positive. Returns (A, alpha)."""
    import numpy as np
    x = np.log(1 - np.asarray(ds, float)); y = np.log(np.asarray(ys, float))
    if alpha is None:
        X = np.stack([np.ones_like(x), x], 1)
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(np.exp(coef[0])), float(coef[1])
    logA = float(np.mean(y - alpha * x))
    return float(np.exp(logA)), float(alpha)


def analyse():
    import numpy as np
    summary = {"models": {}, "shape": {}}
    lines = ["# P2 Wanda panel (retrospective)", ""]
    for tag in MODELS:
        rec = read(OUT / tag / "wanda_losses.json")
        mag = read(ROOT / "results/v6-capability-geometry" / V6_DIRS[tag] / "prune_losses.json")
        cells = {}
        for d in sorted((float(k) for k in rec["losses"] if k != "1.0"), reverse=True):
            cells[str(d)] = {c: {"wanda": rec["losses"][str(d)][c] - rec["losses"]["1.0"][c],
                                 "magnitude": (mag[str(d)][c] - mag["1.0"][c]) if str(d) in mag else None} for c in CAPS}
        summary["models"][tag] = {"dense_wanda_run": rec["losses"]["1.0"], "dense_v6": mag["1.0"],
                                  "scope_parameters": rec["scope_parameters"], "matrix_parameters_total": rec["matrix_parameters_total"],
                                  "cells": cells}
    # shape: per capability, alpha from magnitude cells (both models pooled, positive deltas only), scale per model
    for c in CAPS:
        entry = {}
        ds, ys = [], []
        for tag in MODELS:
            for d in summary["models"][tag]["cells"]:
                y = summary["models"][tag]["cells"][d][c]["magnitude"]
                if y is not None and y > 0: ds.append(float(d)); ys.append(y)
        A_m, alpha_m = fit_power(ds, ys) if len(ys) >= 3 else (float("nan"), float("nan"))
        entry["alpha_magnitude_pooled"] = alpha_m
        for tag in MODELS:
            wd, wy = [], []
            for d in sorted(summary["models"][tag]["cells"], key=float, reverse=True):
                y = summary["models"][tag]["cells"][d][c]["wanda"]
                if y > 0: wd.append(float(d)); wy.append(y)
            row = {"positive_cells": len(wy)}
            if len(wy) >= 2 and not math.isnan(alpha_m):
                A_fixed, _ = fit_power(wd, wy, alpha=alpha_m)
                row["scale_refit_alpha_fixed"] = {"A": A_fixed, "mae": float(np.mean([abs(A_fixed * (1 - d) ** alpha_m - y) for d, y in zip(wd, wy)]))}
                A_free, alpha_free = fit_power(wd, wy)
                row["free_fit"] = {"A": A_free, "alpha": alpha_free, "mae": float(np.mean([abs(A_free * (1 - d) ** alpha_free - y) for d, y in zip(wd, wy)]))}
                # K1: calibrate the scale on the mildest density, predict the others
                d0, y0 = wd[0], wy[0]
                A_k1 = y0 / (1 - d0) ** alpha_m
                others = [(d, y) for d, y in zip(wd, wy) if d != d0]
                row["k1_mildest"] = {"A": A_k1, "mae": float(np.mean([abs(A_k1 * (1 - d) ** alpha_m - y) for d, y in others])) if others else None}
            entry[tag] = row
        summary["shape"][c] = entry
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    lines.append("| Model | Density | Capability | Magnitude dL | Wanda dL |"); lines.append("|---|---|---|---:|---:|")
    for tag in MODELS:
        for d in sorted(summary["models"][tag]["cells"], key=float, reverse=True):
            for c in CAPS:
                cell = summary["models"][tag]["cells"][d][c]
                lines.append(f"| {tag} | {d} | {c} | {'/' if cell['magnitude'] is None else format(cell['magnitude'], '+.3f')} | {cell['wanda']:+.3f} |")
    lines += ["", "| Capability | alpha (magnitude, pooled) | Model | scale-refit MAE | free-fit MAE (alpha) | K1 MAE |", "|---|---:|---|---:|---:|---:|"]
    for c in CAPS:
        e = summary["shape"][c]
        for tag in MODELS:
            r = e[tag]
            if "free_fit" in r:
                lines.append(f"| {c} | {e['alpha_magnitude_pooled']:.2f} | {tag} | {r['scale_refit_alpha_fixed']['mae']:.3f} | {r['free_fit']['mae']:.3f} ({r['free_fit']['alpha']:.2f}) | {r['k1_mildest']['mae'] if r['k1_mildest']['mae'] is not None else float('nan'):.3f} |")
            else:
                lines.append(f"| {c} | {e['alpha_magnitude_pooled']:.2f} | {tag} | too few positive cells ({r['positive_cells']}) | | |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


TABLE = ROOT / "paper/paper/tables/wanda_panel.tex"
TAG_WORDS = {"qwen3-4b": "Qwen3, 4B", "gemma3-4b": "Gemma 3, 4B"}


def render(summary):
    """Appendix table: loss change under magnitude and Wanda pruning at each density, per capability."""
    from analysis.paper_table_layout import house_style, table_layout
    rows = []
    tags = list(summary["models"])
    for i, tag in enumerate(tags):
        cells = summary["models"][tag]["cells"]
        for j, d in enumerate(sorted(cells, key=float, reverse=True)):
            row = [TAG_WORDS[tag] if j == 0 else "", d]
            for c in CAPS:
                m = cells[d][c]["magnitude"]; w = cells[d][c]["wanda"]
                row.append("/" if m is None else f"{m:+.2f}"); row.append(f"{w:+.2f}")
            rows.append(" & ".join(row) + r" \\")
        if i < len(tags) - 1:
            rows.append(r"\midrule")
    caption = ("Loss change in nats per native token from the dense model under two pruning criteria at the same retained "
               "density, on the V6 probes: global magnitude pruning of every weight matrix (the panel's criterion) and Wanda, "
               "which scores each weight by its magnitude times the norm of its input activation over 128 C4 documents and "
               "prunes each output row of the block linear layers, leaving embeddings dense. A slash marks a density the "
               "magnitude panel did not measure.")
    text = ("% Generated by analysis/p2_wanda_panel.py --table; retrospective, do not edit.\n"
            "\\begin{table}[tb]\n\\centering\\footnotesize\n"
            f"\\caption{{{caption}}}\n\\label{{tab:wanda-panel}}\n"
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llrrrrrr@{}}\n\\toprule\n"
            " & & \\multicolumn{2}{c}{Math} & \\multicolumn{2}{c}{Code} & \\multicolumn{2}{c}{QA} \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
            "Model & Density & Magnitude & Wanda & Magnitude & Wanda & Magnitude & Wanda \\\\\n\\midrule\n" +
            "\n".join(rows) + "\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n")
    return house_style(table_layout(text))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true"); ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--models", nargs="*", default=list(MODELS)); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--densities", default=",".join(str(d) for d in DENSITIES), help="comma list")
    a = ap.parse_args()
    if a.run:
        run(a.models, a.device, tuple(float(x) for x in a.densities.split(",") if x))
    if a.analyse:
        analyse()
    if a.table:
        TABLE.write_text(render(read(OUT / "summary.json"))); print("wrote", TABLE)
