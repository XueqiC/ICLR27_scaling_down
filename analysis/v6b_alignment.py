#!/usr/bin/env python3
"""V6b: first-order alignment + full-Fisher quadratic damage prediction.

V6 (diagonal) pilot finding: normalized diagonal spectra collapse
across capabilities — the diagonal quadratic form carries no capability
discrimination — yet measured damage IS capability-specific (QA improves
under mild pruning). Missing pieces this script adds, both computable in
one streaming pass without storing any large object:

For pruning perturbation δw_d = −w·mask_d and per-sample gradient g_i of
capability c:  a_i(d) = g_i · δw_d = −Σ_{j∈pruned(d)} g_ij w_j.
  * first-order term:  mean_i a_i(d)  → ḡ_c·δw_d  (SIGNED; can predict
    loss improvement — invisible to any PSD quadratic form)
  * full quadratic:    mean_i a_i(d)² → δw_dᵀ F_c δw_d  (empirical
    capability Fisher INCLUDING off-diagonals)
Prediction: ΔL_c(d) ≈ mean_a + ½·mean_a² (2nd-order Taylor with full F).

a_i(d) for all densities at once via |w|-quantile bins: per sample, bin
(g·w) by |w| and take cumulative sums below each density threshold.

Usage: python3 v6b_alignment.py --model gemma3-4b --device cuda:0
       [--model-dtype fp32|bf16] [--n-probe 128]
Writes results/v6-capability-geometry/<tag>/alignment.json + report_b.md.
Requires the V6 prune_losses.json of the same tag for comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .model_registry import require_compliant
    from .v6_capability_geometry import (
        DENSITIES, build_probes, completion_loss, language_weight_parameters,
        load_text_causal_lm, model_output_tag,
    )
except ImportError:  # direct execution: python analysis/v6b_alignment.py
    from model_registry import require_compliant
    from v6_capability_geometry import (
        DENSITIES, build_probes, completion_loss, language_weight_parameters,
        load_text_causal_lm, model_output_tag,
    )

ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "results/v6-capability-geometry"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-270m",
                    help="registry tag or raw Hugging Face model id")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-probe", type=int, default=128)
    ap.add_argument("--model-dtype", default="fp32",
                    choices=["fp32", "bf16"])
    args = ap.parse_args()
    model_name = require_compliant(args.model)
    tag = model_output_tag(args.model, model_name)
    out = OUT_BASE / tag
    out.mkdir(parents=True, exist_ok=True)
    device = args.device

    dtype = torch.float32 if args.model_dtype == "fp32" else torch.bfloat16
    model, tok = load_text_causal_lm(model_name, dtype)
    model.to(device).train()
    model.requires_grad_(False)
    params = language_weight_parameters(model)
    for _, p in params:
        p.requires_grad_(True)

    n_tot = sum(p.numel() for _, p in params)

    # per-parameter density thresholds from the global |w| distribution
    rng = np.random.default_rng(0)
    sample = []
    for _, p in params:
        flat = p.detach().reshape(-1)
        k = max(int(2e6 * p.numel() / n_tot), 100)
        idx = torch.from_numpy(
            rng.integers(0, flat.numel(), size=min(k, flat.numel())))
        sample.append(flat[idx.to(flat.device)].abs().float().cpu())
    sample = torch.cat(sample).numpy()
    thresholds = {d: float(np.quantile(sample, 1 - d)) for d in DENSITIES}
    # fixed masks are too large to store; recompute per param on the fly

    # even-indexed probes (same split as the fisher stage)
    probes = {c: v[0::2] for c, v in build_probes(args.n_probe).items()}

    result: dict[str, dict] = {}
    for cap, samples in probes.items():
        A = []  # per sample: a_i(d) for each density
        for s in samples:
            model.zero_grad(set_to_none=True)
            loss, n_tok = completion_loss(model, tok, s["prompt"],
                                          s["completion"], device)
            if n_tok == 0:
                continue
            (loss / n_tok).backward()
            a = {d: 0.0 for d in DENSITIES}
            with torch.no_grad():
                for _, p in params:
                    if p.grad is None:
                        continue
                    gw = (p.grad.float() * p.detach().float())
                    absw = p.detach().float().abs()
                    for d in DENSITIES:
                        m = absw < thresholds[d]
                        a[d] += float(gw[m].sum())
            # delta_w = -w_masked -> a_i(d) = -(sum of g*w over mask)
            A.append({d: -v for d, v in a.items()})
        arr = {d: np.array([r[d] for r in A]) for d in DENSITIES}
        np.savez_compressed(out / f"align_samples_{cap}.npz",
                            **{str(d): arr[d] for d in DENSITIES})
        result[cap] = {
            str(d): {"first_order": float(arr[d].mean()),
                     "quad_full": float((arr[d] ** 2).mean()),
                     "quad_trimmed": float(
                         (np.sort(arr[d] ** 2)[:max(1, int(0.9 * arr[d].size))]).mean()),
                     "n": int(arr[d].size)} for d in DENSITIES}
        print(f"[align] {cap}: {len(A)} samples", flush=True)

    (out / "alignment.json").write_text(json.dumps(result, indent=1))

    # report vs measured
    losses_f = out / "prune_losses.json"
    lines = [f"# V6b report — {model_name}", "",
             "pred = first_order + 0.5*quad_full (2nd-order Taylor, "
             "full capability Fisher)", ""]
    if losses_f.exists():
        losses = json.loads(losses_f.read_text())
        dense = losses["1.0"]
        caps = list(result.keys())
        lines.append("| d | " + " | ".join(
            f"{c}: 1st | ½quad | pred | meas" for c in caps) + " |")
        lines.append("|---|" + "---|" * (4 * len(caps)))
        for d in DENSITIES:
            row = [f"| {d}"]
            for c in caps:
                e = result[c][str(d)]
                pred = e["first_order"] + 0.5 * e["quad_full"]
                meas = (losses[str(d)][c] - dense[c]
                        if str(d) in losses else float("nan"))
                row.append(f" {e['first_order']:+.3f} | "
                           f"{0.5*e['quad_full']:.3f} | {pred:+.3f} | "
                           f"{meas:+.3f}")
            lines.append(" |".join(row) + " |")
    (out / "report_b.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
