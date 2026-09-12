#!/usr/bin/env python3
"""V54: group-wise symmetric round-to-nearest weight quantization.

Reuse V10's model loader, language-weight scope (including embeddings and the
LM head), and completion-token-weighted losses on the odd-indexed measurement
half of the V6 probes. Only forward passes are performed. Completed dense and
config measurements are saved individually and reused on subsequent runs.

Example:
  python3 analysis/v54_quant_group.py --model pythia-410m@step16000 \
      --configs b3_g64,b5_g256 --device cuda:0 --model-dtype bf16 \
      --reference-device cpu
  python3 analysis/v54_quant_group.py --selftest
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch
from torch.nn import functional as F

try:
    from .eval_records import measure_capability_losses as _recorded_losses, with_records
except ImportError:
    from eval_records import measure_capability_losses as _recorded_losses, with_records

try:
    from .v10_quantization import (
        _apply_fake_quantization,
        _measure_capability_losses,
        _restore_dense_weights,
        build_probes,
        fake_quantize_per_output_channel,
        language_weight_parameters,
        load_text_causal_lm,
        model_output_tag,
        require_compliant,
        resolve_model_and_revision,
    )
except ImportError:  # direct execution: python analysis/v54_quant_group.py
    from v10_quantization import (
        _apply_fake_quantization,
        _measure_capability_losses,
        _restore_dense_weights,
        build_probes,
        fake_quantize_per_output_channel,
        language_weight_parameters,
        load_text_causal_lm,
        model_output_tag,
        require_compliant,
        resolve_model_and_revision,
    )


ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "results/v54-quant-group"
PROBE_SEED = 0  # build_probes(n_probe)'s default in V10
CAPABILITIES = ("math", "code", "qa")
PADDING_RULE = (
    "Zero-pad the last input group in each row to group_size; zeros cannot "
    "change the absmax of the real entries. Trim padding after quantization. "
    "group_size=0 or None uses V10 per-output-channel quantization."
)


def parse_configs(value: str) -> list[tuple[int, int]]:
    """Parse unique b<bits>_g<group> configs; g0 means per-output-channel."""
    configs = []
    for part in value.split(","):
        match = re.fullmatch(r"b([0-9]+)_g([0-9]+)", part.strip())
        if match is None:
            raise argparse.ArgumentTypeError(
                "--configs must contain comma-separated b<bits>_g<group> values"
            )
        bits, group_size = map(int, match.groups())
        if bits < 2:
            raise argparse.ArgumentTypeError("config bits must be at least 2")
        if (bits, group_size) in configs:
            raise argparse.ArgumentTypeError("--configs values must be unique")
        configs.append((bits, group_size))
    return configs


def _symmetric_grouped(
    weight: torch.Tensor, bits: int, group_size: int | None
) -> torch.Tensor:
    """Quantize contiguous input groups independently in each [out, in] row.

    Each group uses qmax = 2**(bits-1)-1 and scale = absmax/qmax, with
    round-to-nearest (torch.round), symmetric clamping, and a safe divisor for
    zero groups. All arithmetic stays in the input dtype and on its device,
    exactly as in V10. The input is never modified.
    """
    if group_size is not None and (
        not isinstance(group_size, int)
        or isinstance(group_size, bool)
        or group_size < 0
    ):
        raise ValueError("group_size must be a non-negative integer or None")
    if group_size is None or group_size == 0:
        return fake_quantize_per_output_channel(weight, bits)
    if weight.ndim != 2:
        raise ValueError("grouped quantization needs a 2-D [out, in] weight")
    if weight.shape[1] == 0:
        raise ValueError("grouped quantization needs a non-empty input dimension")

    n_out, n_in = weight.shape
    padding = (-n_in) % group_size
    padded = F.pad(weight, (0, padding)) if padding else weight
    # Every flattened group becomes one V10 output channel. Zero padding is
    # neutral for absmax, so even the last scale depends only on real weights.
    groups = padded.reshape(n_out * ((n_in + padding) // group_size), group_size)
    quantized = fake_quantize_per_output_channel(groups, bits)
    return quantized.reshape(n_out, n_in + padding)[:, :n_in].contiguous()


def _groups(weight, group_size):
    if weight.ndim != 2 or not weight.is_floating_point() or not weight.shape[1]:
        raise ValueError("grouped quantization needs a nonempty floating-point [out, in] matrix")
    if group_size is not None and (not isinstance(group_size, int)
                                  or isinstance(group_size, bool) or group_size < 0):
        raise ValueError("group_size must be a non-negative integer or None")
    size = group_size or weight.shape[1]
    padding = (-weight.shape[1]) % size
    padded = F.pad(weight, (0, padding)) if padding else weight
    valid = F.pad(torch.ones_like(weight, dtype=torch.bool), (0, padding)) if padding else torch.ones_like(weight, dtype=torch.bool)
    return padded.reshape(-1, size), valid.reshape(-1, size), padding


def _asymmetric_parameters(groups, bits):
    # Include zero: one-sided and constant groups have a defined integer zero
    # point; zero padding cannot alter the range of even a partial group.
    x = groups.float()
    low = x.amin(-1, keepdim=True).clamp(max=0)
    high = x.amax(-1, keepdim=True).clamp(min=0)
    scale = (high - low) / (2**bits - 1)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    zero = torch.round(-low / safe).clamp(0, 2**bits - 1)
    return scale, safe, zero


def fake_quantize_grouped(weight, bits, group_size, mode="symmetric"):
    """Default executes the original V54 function byte-for-byte.

    Asymmetric uses FP32 affine unsigned codes with an integer zero point;
    dequantized weights are cast back to the input dtype.
    """
    if mode == "symmetric":
        return _symmetric_grouped(weight, bits, group_size)
    if mode != "asymmetric":
        raise ValueError("mode must be symmetric or asymmetric")
    if not isinstance(bits, int) or isinstance(bits, bool) or bits < 2:
        raise ValueError("bits must be an integer of at least 2")
    groups, _, padding = _groups(weight, group_size)
    scale, safe, zero = _asymmetric_parameters(groups, bits)
    codes = (torch.round(groups.float() / safe) + zero).clamp(0, 2**bits - 1)
    result = ((codes - zero) * scale).to(weight.dtype)
    return result.reshape(weight.shape[0], weight.shape[1] + padding)[:, :weight.shape[1]].contiguous()


def quantization_statistics(weight, quantized, bits, group_size, mode="symmetric"):
    """Summary moments over real groups; padding never counts as a weight.

    Clipped means an integer code actually changed at the clamp, distinct
    from ordinary rounding or endpoint movement due to a rounded zero point.
    """
    if mode not in ("symmetric", "asymmetric"):
        raise ValueError("mode must be symmetric or asymmetric")
    groups, valid, _ = _groups(weight, group_size)
    qgroups, _, _ = _groups(quantized, group_size)
    if mode == "symmetric":
        qmax = 2 ** (bits - 1) - 1
        scale = groups.abs().amax(-1, keepdim=True) / qmax
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        zero = torch.zeros_like(scale)
        codes = torch.round(groups / safe)
        qmin = -qmax
    else:
        qmin, qmax = 0, 2**bits - 1
        scale, safe, zero = _asymmetric_parameters(groups, bits)
        codes = torch.round(groups.float() / safe) + zero
    counts = valid.sum(-1)
    clipped = (((codes < qmin) | (codes > qmax)) & valid).sum(-1)
    square_error = ((qgroups.double() - groups.double()).square() * valid).sum(-1)

    def stats(values):
        x = values.double()
        return {"min": float(x.min()), "mean": float(x.mean()),
                "max": float(x.max()), "std": float(x.std(unbiased=False))}

    return {"mode": mode, "n_groups": groups.shape[0], "n_weights": int(counts.sum()),
            "step": stats(scale), "zero_point": stats(zero),
            "clipping_rule": f"clamp rounded integer codes to [{qmin}, {qmax}]; fraction counts codes changed by clamp; excludes padding",
            "fraction_weights_clipped": float(clipped.sum() / counts.sum()),
            "fraction_clipped_per_group": stats(clipped / counts),
            "rms_rounding_error_per_group": stats((square_error / counts).sqrt()),
            "rms_rounding_error": float((square_error.sum() / counts.sum()).sqrt()),
            "statistics_weighting": "min/mean/max/population std over groups; overall RMS and clipping weighted by real weight count",
            "arithmetic_dtype": str(weight.dtype) if mode == "symmetric" else "torch.float32 (dequantized cast to input dtype)"}


def _apply_grouped_quantization(
    parameters: Sequence[tuple[str, torch.Tensor]],
    dense_weights: Sequence[torch.Tensor],
    bits: int,
    group_size: int | None,
    mode: str = "symmetric",
) -> dict:
    """Follow V10's reference-weight/copy path, with grouped scales."""
    statistics = {}
    with torch.no_grad():
        for (name, parameter), dense in zip(parameters, dense_weights):
            quantized = fake_quantize_grouped(dense, bits, group_size, mode)
            statistics[name] = quantization_statistics(dense, quantized, bits, group_size, mode)
            parameter.copy_(quantized)
    return statistics


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _save_results(path: Path, results: dict) -> None:
    """Atomically checkpoint each completed measurement without losing others."""
    results["_meta"]["updated_at_utc"] = _utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(with_records(results), indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_quantization(
    model_tag: str,
    configs: Sequence[tuple[int, int]],
    device: str = "cuda:0",
    model_dtype: str = "fp32",
    n_probe: int = 128,
    reference_device: str = "cuda",
    out: Path | None = None,
    mode: str = "symmetric",
) -> None:
    """Measure dense once, then missing configs; resume only matching protocols."""
    if n_probe < 2:
        raise ValueError("n_probe must be at least 2 for a measurement half")
    if model_dtype not in ("fp32", "bf16"):
        raise ValueError("model_dtype must be fp32 or bf16")
    if mode not in ("symmetric", "asymmetric"):
        raise ValueError("mode must be symmetric or asymmetric")
    model_name = require_compliant(model_tag)
    _, revision = resolve_model_and_revision(model_tag)
    out = Path(out) if out is not None else OUT_BASE / model_output_tag(model_tag, model_name)
    path = out / "quant_group_losses.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(results, dict) or not isinstance(results.get("_meta", {}), dict):
        raise ValueError(f"Invalid result object in {path}")

    protocol = {
        "hf_id": model_name,
        "revision": revision,
        "model_dtype": model_dtype,
        "probe_seed": PROBE_SEED,
        "n_probe": n_probe,
        "probe_half": "odd-indexed [1::2]",
        "loss_protocol": "v10._measure_capability_losses (completion-token-weighted CE)",
        "quantization": f"{mode} round-to-nearest (torch.round, ties to even)",
        "padding_rule": PADDING_RULE,
    }
    if mode == "asymmetric":
        protocol["padding_rule"] = "Zero-pad partial groups; include zero in every min/max range; trim padding; g0/None uses the full row."
    meta = results.setdefault("_meta", {})
    for key, value in protocol.items():
        if key in meta and meta[key] != value:
            raise ValueError(f"Cannot merge {path}: metadata {key!r} differs")
    pending = []
    for bits, group_size in configs:
        key = f"b{bits}_g{group_size}"
        if key in results:
            print(f"[quant-group] tag={model_tag} config={key} skipped (already present)", flush=True)
        else:
            pending.append((key, bits, group_size))
    if not pending and "dense" in results:
        return  # No model or dataset loading, and no rewrite on a completed run.

    meta.update(protocol)
    meta.setdefault("model_tag", model_tag)
    meta.setdefault("created_at_utc", _utc_now())
    config_meta = meta.setdefault("configs", {})
    dtype = torch.float32 if model_dtype == "fp32" else torch.bfloat16
    model, tokenizer = load_text_causal_lm(model_name, dtype, revision)
    dense_weights = []
    parameters = []
    try:
        model.to(device).eval()
        model.requires_grad_(False)
        probes = {
            capability: samples[1::2]
            for capability, samples in build_probes(n_probe, seed=PROBE_SEED).items()
        }
        if set(probes) != set(CAPABILITIES) or any(not samples for samples in probes.values()):
            raise ValueError("Expected non-empty math, code, and qa measurement probes")
        probe_hash = hashlib.sha256(
            json.dumps(probes, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if meta.get("probe_sha256", probe_hash) != probe_hash:
            raise ValueError("Cannot merge measurements made with different probes")
        meta["probe_sha256"] = probe_hash
        meta["measurement_samples"] = {cap: len(samples) for cap, samples in probes.items()}
        parameters = language_weight_parameters(model)
        if not parameters:
            raise ValueError("No language weight matrices found for quantization")
        ref_device = "cpu" if str(reference_device).startswith("cpu") else device
        dense_weights = [p.detach().clone().to(ref_device) for _, p in parameters]
        print(f"[quant-group] tag={model_tag} revision={revision} reference_device={ref_device}", flush=True)

        if "dense" not in results:
            _synchronize(device)
            started_at = _utc_now()
            start = time.perf_counter()
            results["dense"] = _recorded_losses(model, tokenizer, probes, device)
            _synchronize(device)
            elapsed = time.perf_counter() - start
            meta["dense"] = {
                "started_at_utc": started_at,
                "completed_at_utc": _utc_now(),
                "wall_time_s": elapsed,
                "device": device,
            }
            _save_results(path, results)
            print(f"[quant-group] tag={model_tag} config=dense wall_time_s={elapsed:.6f} losses={results['dense']}", flush=True)
        else:
            print(f"[quant-group] tag={model_tag} dense skipped (already present)", flush=True)

        for key, bits, group_size in pending:
            _synchronize(device)
            started_at = _utc_now()
            start = time.perf_counter()
            try:
                statistics = _apply_grouped_quantization(parameters, dense_weights, bits, group_size, mode)
                losses = _recorded_losses(model, tokenizer, probes, device)
            finally:
                # Restore even if applying quantization or evaluating a probe fails.
                _restore_dense_weights(parameters, dense_weights)
                _synchronize(device)
            elapsed = time.perf_counter() - start
            results[key] = losses
            config_meta[key] = {
                "bits": bits,
                "group_size": group_size,
                "qmax": 2 ** (bits - 1) - 1 if mode == "symmetric" else 2**bits - 1,
                "mode": mode,
                "quantization_statistics": statistics,
                "started_at_utc": started_at,
                "completed_at_utc": _utc_now(),
                "wall_time_s": elapsed,
                "timing_scope": "quantize + measure + restore (excludes model/probe loading and dense eval)",
                "device": device,
                "reference_device": ref_device,
            }
            _save_results(path, results)
            print(f"[quant-group] tag={model_tag} config={key} wall_time_s={elapsed:.6f} losses={losses}", flush=True)
    finally:
        del dense_weights, parameters, model, tokenizer
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def selftest() -> None:
    """CPU-only, exact equivalence checks on a deterministic random 8x64 tensor."""
    generator = torch.Generator(device="cpu").manual_seed(PROBE_SEED)
    weight = torch.randn(8, 64, generator=generator, device="cpu")
    weight[0].zero_()  # Also exercise the safe divisor for a zero row.
    for dtype in (torch.float32, torch.bfloat16):
        sample = weight.to(dtype)
        original = sample.clone()
        for bits in (2, 3, 4, 5, 8):
            expected = fake_quantize_per_output_channel(sample, bits)
            for group_size in (None, 0, 64):
                actual = fake_quantize_grouped(sample, bits, group_size)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert actual.dtype == sample.dtype and actual.device == sample.device
        assert torch.equal(sample, original)
    print("selftest passed (CPU): g=None, g=0, and g=64 exactly equal V10 on 8x64 fp32/bf16")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="model-registry tag or HF id, optionally @revision")
    parser.add_argument("--configs", type=parse_configs, help="comma-separated b<bits>_g<group> values")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--n-probe", type=int, default=128)
    parser.add_argument("--reference-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--selftest", action="store_true", help="run CPU unit checks without loading a model")
    parser.add_argument("--mode", choices=("symmetric", "asymmetric"), default="symmetric")
    parser.add_argument("--output-dir", type=Path, help="use a separate directory for asymmetric measurements")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if args.model is None or args.configs is None:
        parser.error("--model and --configs are required unless --selftest is used")
    if args.n_probe < 2:
        parser.error("--n-probe must be at least 2")
    run_quantization(
        model_tag=args.model,
        configs=args.configs,
        device=args.device,
        model_dtype=args.model_dtype,
        n_probe=args.n_probe,
        reference_device=args.reference_device,
        mode=args.mode,
        out=args.output_dir,
    )


if __name__ == "__main__":
    main()
