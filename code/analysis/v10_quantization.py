#!/usr/bin/env python3
"""V10: measure capability-specific loss damage from weight quantization.

For every requested bit-width, all language weight matrices are fake-quantized
with symmetric, per-output-channel scales.  The model is evaluated on the
held-out (odd-indexed) half of the V6 probes, then restored to its dense
weights.  The resulting damage is checked against

    Delta L_c(b) = q_c * 4**(-b).

Artifacts are written under ``results/v10-quantization/<model_tag>/``.

Example:
  python3 analysis/v10_quantization.py --model gemma3-270m \
      --device cuda:0 --model-dtype bf16 --bits 8,6,4,3
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

try:
    from .v6_capability_geometry import (
        build_probes,
        completion_loss,
        language_weight_parameters,
        load_text_causal_lm,
        model_output_tag,
        require_compliant,
    )
except ImportError:  # direct execution: python analysis/v10_quantization.py
    from v6_capability_geometry import (
        build_probes,
        completion_loss,
        language_weight_parameters,
        load_text_causal_lm,
        model_output_tag,
        require_compliant,
    )


ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "results/v10-quantization"
DEFAULT_BITS = "8,6,4,3"


def parse_bits(value: str) -> list[int]:
    """Parse and validate a comma-separated list of quantization bit-widths."""
    try:
        bits = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--bits must be a comma-separated list of integers"
        ) from exc
    if not bits:
        raise argparse.ArgumentTypeError("--bits must contain at least one value")
    if any(bit < 2 for bit in bits):
        raise argparse.ArgumentTypeError("all --bits values must be at least 2")
    if len(set(bits)) != len(bits):
        raise argparse.ArgumentTypeError("--bits values must be unique")
    return bits


def fake_quantize_per_output_channel(
    weight: torch.Tensor, bits: int
) -> torch.Tensor:
    """Return a symmetric row-wise fake-quantized copy of ``weight``.

    Dimension zero is the output-channel dimension.  All arithmetic remains
    in ``weight.dtype`` and on ``weight.device``.  A zero-valued output
    channel uses a safe divisor and remains exactly zero.
    """
    if weight.ndim < 2:
        raise ValueError("per-output-channel quantization needs a weight matrix")
    if not weight.is_floating_point():
        raise TypeError("fake quantization requires a floating-point tensor")
    if not isinstance(bits, int) or isinstance(bits, bool) or bits < 2:
        raise ValueError("bits must be an integer of at least 2")

    quant_max = 2 ** (bits - 1) - 1
    reduce_dims = tuple(range(1, weight.ndim))
    max_abs = weight.abs().amax(dim=reduce_dims, keepdim=True)
    scale = max_abs / quant_max
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.round(weight / safe_scale)
    quantized.clamp_(-quant_max, quant_max)
    quantized.mul_(scale)
    return quantized


def fit_exponential_decay(
    bits: Sequence[int | float], deltas: Sequence[int | float]
) -> dict[str, float | int | None]:
    """Fit ``log(delta) = intercept + slope * bits`` for positive damage.

    The returned ``base`` is ``exp(-slope)``, so a fitted base near four is
    evidence for ``delta = q * 4**(-bits)``.  Non-finite and non-positive
    damage observations are excluded.  At least two observations are needed.
    """
    x = np.asarray(bits, dtype=np.float64)
    damage = np.asarray(deltas, dtype=np.float64)
    if x.ndim != 1 or damage.ndim != 1 or x.shape != damage.shape:
        raise ValueError("bits and deltas must be one-dimensional and equal length")

    valid = np.isfinite(x) & np.isfinite(damage) & (damage > 0)
    x = x[valid]
    log_damage = np.log(damage[valid])
    n_points = int(x.size)
    empty_fit: dict[str, float | int | None] = {
        "n_points": n_points,
        "intercept": None,
        "slope": None,
        "q": None,
        "base": None,
        "r2": None,
    }
    if n_points < 2:
        return empty_fit

    design = np.column_stack((np.ones_like(x), x))
    intercept, slope = np.linalg.lstsq(design, log_damage, rcond=None)[0]
    fitted = design @ np.array([intercept, slope])
    residual_sum = float(np.sum((log_damage - fitted) ** 2))
    total_sum = float(np.sum((log_damage - log_damage.mean()) ** 2))
    if total_sum == 0.0:
        r2 = 1.0 if residual_sum <= np.finfo(np.float64).eps else 0.0
    else:
        r2 = 1.0 - residual_sum / total_sum

    try:
        base = math.exp(-float(slope))
    except OverflowError:
        base = math.inf
    try:
        coefficient = math.exp(float(intercept))
    except OverflowError:
        coefficient = math.inf
    return {
        "n_points": n_points,
        "intercept": float(intercept),
        "slope": float(slope),
        "q": coefficient,
        "base": base,
        "r2": float(r2),
    }


def _measure_capability_losses(
    model,
    tokenizer,
    probes: dict[str, list[dict]],
    device: str,
) -> dict[str, float]:
    """Compute completion-token-weighted mean CE for every capability."""
    losses: dict[str, float] = {}
    with torch.no_grad():
        for capability, samples in probes.items():
            total_loss = 0.0
            total_tokens = 0
            for sample in samples:
                loss, n_tokens = completion_loss(
                    model,
                    tokenizer,
                    sample["prompt"],
                    sample["completion"],
                    device,
                )
                total_loss += float(loss)
                total_tokens += n_tokens
            losses[capability] = total_loss / max(total_tokens, 1)
    return losses


def _restore_dense_weights(
    parameters: Sequence[tuple[str, torch.Tensor]],
    dense_weights: Sequence[torch.Tensor],
) -> None:
    with torch.no_grad():
        for (_, parameter), dense in zip(parameters, dense_weights):
            parameter.copy_(dense)


def _apply_fake_quantization(
    parameters: Sequence[tuple[str, torch.Tensor]],
    dense_weights: Sequence[torch.Tensor],
    bits: int,
) -> None:
    with torch.no_grad():
        for (_, parameter), dense in zip(parameters, dense_weights):
            parameter.copy_(fake_quantize_per_output_channel(dense, bits))


def _format_optional(value: float | int | None, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def write_report(
    out: Path,
    model_name: str,
    bits: Sequence[int],
    losses: dict[str, dict[str, float]],
) -> None:
    """Write loss-damage tables and per-capability exponential fit checks."""
    capabilities = list(losses["dense"])
    dense = losses["dense"]
    lines = [
        f"# V10 quantization report — {model_name}",
        "",
        "All language weight matrices use symmetric round-to-nearest fake "
        "quantization with one scale per output channel. Losses use the "
        "odd-indexed measurement half of each V6 probe set.",
        "",
        "## Dense losses",
        "",
        "| capability | dense loss |",
        "|---|---:|",
    ]
    for capability in capabilities:
        lines.append(f"| {capability} | {dense[capability]:.6f} |")

    lines.extend(
        [
            "",
            "## Quantization loss damage",
            "",
            "Damage is `Delta L_c(b) = L_c(b) - L_c(dense)`.",
            "",
            "| bits | " + " | ".join(f"Delta L_{cap}" for cap in capabilities) + " |",
            "|---:|" + "---:|" * len(capabilities),
        ]
    )
    damage_by_capability: dict[str, list[float]] = {
        capability: [] for capability in capabilities
    }
    for bit in bits:
        row = [f"| {bit}"]
        for capability in capabilities:
            damage = losses[str(bit)][capability] - dense[capability]
            damage_by_capability[capability].append(damage)
            row.append(f"{damage:+.6f}")
        lines.append(" | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## Exponential fit check",
            "",
            "Least-squares fit of `ln(Delta L_c)` against bit-width, using "
            "only positive damage. The proposed `4^(-b)` law predicts slope "
            "`-ln(4) = -1.3863` and fitted base `4`.",
            "",
            "| capability | positive points | q_c | slope | fitted base | R^2 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for capability in capabilities:
        fit = fit_exponential_decay(bits, damage_by_capability[capability])
        lines.append(
            f"| {capability} | {fit['n_points']} | "
            f"{_format_optional(fit['q'])} | "
            f"{_format_optional(fit['slope'])} | "
            f"{_format_optional(fit['base'])} | "
            f"{_format_optional(fit['r2'])} |"
        )

    (out / "report.md").write_text("\n".join(lines) + "\n")


def run_quantization(
    model_name: str,
    device: str,
    model_dtype: str,
    n_probe: int,
    bits: Sequence[int],
    out: Path,
) -> None:
    """Load one model and run dense plus fake-quantized measurements."""
    dtype = torch.float32 if model_dtype == "fp32" else torch.bfloat16
    model, tokenizer = load_text_causal_lm(model_name, dtype)
    model.to(device).eval()
    probes = {
        capability: capability_probes[1::2]
        for capability, capability_probes in build_probes(n_probe).items()
    }
    parameters = language_weight_parameters(model)
    if not parameters:
        raise ValueError("No language weight matrices found for quantization")

    dense_weights = [parameter.detach().clone() for _, parameter in parameters]
    results: dict[str, dict[str, float]] = {
        "dense": _measure_capability_losses(model, tokenizer, probes, device)
    }
    print(f"[quantize] dense: {results['dense']}", flush=True)

    try:
        for bit in bits:
            _apply_fake_quantization(parameters, dense_weights, bit)
            results[str(bit)] = _measure_capability_losses(
                model, tokenizer, probes, device
            )
            print(f"[quantize] b={bit}: {results[str(bit)]}", flush=True)
            _restore_dense_weights(parameters, dense_weights)
    finally:
        _restore_dense_weights(parameters, dense_weights)

    (out / "quant_losses.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    write_report(out, model_name, bits, results)
    del dense_weights, parameters, model, tokenizer
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="gemma3-270m",
        help="model-registry tag or raw Hugging Face model id",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--n-probe", type=int, default=128)
    parser.add_argument(
        "--bits",
        type=parse_bits,
        default=parse_bits(DEFAULT_BITS),
        help=f"comma-separated bit-widths (default: {DEFAULT_BITS})",
    )
    args = parser.parse_args()

    model_name = require_compliant(args.model)
    tag = model_output_tag(args.model, model_name)
    out = OUT_BASE / tag
    out.mkdir(parents=True, exist_ok=True)
    run_quantization(
        model_name=model_name,
        device=args.device,
        model_dtype=args.model_dtype,
        n_probe=args.n_probe,
        bits=args.bits,
        out=out,
    )


if __name__ == "__main__":
    main()
