#!/usr/bin/env python3
"""V23: cross-benchmark validity of capability LOSS under compression.

The endpoint is E_i[sum target-token CE_i / n_target_tokens_i]. Also retain
sum_i CE_i / sum_i n_i (the legacy V6/V12 corpus-token mean) for comparison.
Both use V6 completion_loss, fixed zero-shot prompt/reference spans, and only
the odd-indexed measurement half. No generation or correctness scoring occurs.

Each invocation includes an uncompressed reference and independently reloads
the same source for every pruning/quantization cell. Both flags together run
two separate method ladders, not composed compression. Quantization is V15's
per-output-channel symmetric fake quantization, not packed integer inference.
--checkpoint loads a full HF checkpoint/path; --adapter reuses a V12 full-FT
delta or PEFT adapter. With neither, the source is the resolved --model.

Examples (measurement commands are for a separately launched GPU worker):
  python analysis/v23_loss_validity.py --model gemma3-270m --dry-run
  python analysis/v23_loss_validity.py --model gemma3-270m \
      --prune-density 0.9 0.8 0.7 0.6 --quant-bits 8 6 4 3 --device cuda:0
  python analysis/v23_loss_validity.py --model gemma3-270m \
      --adapter results/v12-distill/gemma3-270m/RUN/adapter --dry-run
  python analysis/v23_loss_validity.py --summarize

Outputs: results/v23-loss-validity/<model>/<checkpoint>/loss_validity.json;
--summarize reads those files and writes paper/docs/LOSS_VALIDITY.md only.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    from .v6_capability_geometry import (
        SECONDARY_BENCHMARKS, apply_global_magnitude_pruning, build_probes,
        completion_loss, load_text_causal_lm, model_output_tag, require_compliant,
    )
    from .v12_distill import seed_everything, write_json_atomic
    from .v15_accuracy_link import _apply_quantization, _load_adapter
except ImportError:  # direct script execution
    from v6_capability_geometry import (
        SECONDARY_BENCHMARKS, apply_global_magnitude_pruning, build_probes,
        completion_loss, load_text_causal_lm, model_output_tag, require_compliant,
    )
    from v12_distill import seed_everything, write_json_atomic
    from v15_accuracy_link import _apply_quantization, _load_adapter

ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "results/v23-loss-validity"
REPORT = ROOT / "paper/docs/LOSS_VALIDITY.md"
CAPABILITY_PAIRS = {
    "math": ("math", "math_gsm8k"),
    "code": ("code", "code_humaneval"),
    "qa": ("qa", "qa_hotpotqa"),
}
BENCHMARKS = {
    "math": ("HuggingFaceH4/MATH-500", None, "test"),
    "code": ("google-research-datasets/mbpp", "full", "test"),
    "qa": ("framolfese/2WikiMultihopQA", None, "validation"),
    **SECONDARY_BENCHMARKS,
}
NAMES = {"math": "MATH-500", "code": "MBPP", "qa": "2WikiMultihopQA",
         "math_gsm8k": "GSM8K", "code_humaneval": "HumanEval",
         "qa_hotpotqa": "HotpotQA"}
LOSS_DEFINITION = "mean_over_examples(sum_target_token_CE / n_target_tokens)"
CLIFF_DELTA = 1.0


def measurement_probes(n_probe: int = 128, seed: int = 0) -> dict:
    if n_probe < 2:
        raise ValueError("--n-probe must be at least 2 for a measurement half")
    all_probes = build_probes(n_probe, seed=seed, include_secondary=True)
    probes = {key: all_probes[key][1::2] for key in BENCHMARKS}
    if any(not samples for samples in probes.values()):
        raise ValueError("Every benchmark needs a nonempty measurement half")
    return probes


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def measure_losses(model, tokenizer, probes: dict, device: str,
                   max_len: int = 1024, loss_function=None) -> dict:
    """Length-normalize each example before averaging; expose legacy weighting."""
    loss_function = loss_function or completion_loss
    measured = {}
    model.eval()
    with torch.no_grad():
        for key, samples in probes.items():
            items = []
            for index, sample in enumerate(samples):
                loss, n_tokens = loss_function(
                    model, tokenizer, sample["prompt"], sample["completion"],
                    device, max_len=max_len,
                )
                total = float(loss)
                if n_tokens <= 0 or not math.isfinite(total):
                    raise RuntimeError(f"{key} item {index}: empty/nonfinite target loss")
                items.append({"measurement_index": index,
                              "probe_sha256": _digest(sample),
                              "sum_ce": total, "n_tokens": n_tokens,
                              "loss": total / n_tokens})
            if not items:
                raise ValueError(f"No measurement probes for {key}")
            tokens = sum(item["n_tokens"] for item in items)
            measured[key] = {
                "benchmark": NAMES[key], "dataset": BENCHMARKS[key][0],
                "config": BENCHMARKS[key][1], "split": BENCHMARKS[key][2],
                "L_c": float(np.mean([item["loss"] for item in items])),
                "token_weighted_L_c": sum(item["sum_ce"] for item in items) / tokens,
                "n_samples": len(items), "n_tokens": tokens, "items": items,
            }
    return measured


def compression_cells(prune_density=None, quant_bits=None) -> list[dict]:
    densities, bits = list(prune_density or []), list(quant_bits or [])
    if any(not 0 < d <= 1 for d in densities):
        raise ValueError("--prune-density values must be in (0, 1]")
    if any(not isinstance(b, int) or not 2 <= b <= 16 for b in bits):
        raise ValueError("--quant-bits values must be integers in [2, 16]")
    if len(set(densities)) != len(densities) or len(set(bits)) != len(bits):
        raise ValueError("Compression ladders must not contain duplicate values")
    cells = [{"method": "baseline", "severity": 0.0, "suffix": "",
              "prune_density": None, "quant_bits": None}]
    cells.extend({"method": "prune", "severity": 1 - d,
                  "suffix": f"prune-d{d}", "prune_density": d, "quant_bits": None}
                 for d in sorted(densities, reverse=True) if d < 1)
    cells.extend({"method": "quant", "severity": 4.0 ** -b,
                  "suffix": f"quant-b{b}", "prune_density": None, "quant_bits": b}
                 for b in sorted(bits, reverse=True))
    return cells


def resolve_source(model_request: str, checkpoint: str | None,
                   adapter: Path | None) -> tuple[str, str, str]:
    """Guard aliases, resolved IDs, and any available local source metadata."""
    if checkpoint and adapter:
        raise ValueError("--checkpoint and --adapter are mutually exclusive")
    resolved = require_compliant(require_compliant(model_request))
    source = require_compliant(require_compliant(checkpoint)) if checkpoint else resolved
    if Path(source).is_dir():
        source = str(Path(source).resolve())
        config = json.loads((Path(source) / "config.json").read_text())
        for field in ("_name_or_path", "model_type"):
            if config.get(field):
                require_compliant(require_compliant(str(config[field])))
        for architecture in config.get("architectures", []):
            require_compliant(str(architecture))
    if adapter is not None:
        manifest = adapter / "delta_manifest.json"
        if not manifest.is_file():
            manifest = adapter / "adapter_config.json"
        config = json.loads(manifest.read_text())
        base = config.get("base_model", config.get("base_model_name_or_path"))
        if not base or require_compliant(require_compliant(str(base))) != resolved:
            raise ValueError("Adapter base must match the resolved --model")
        label = adapter.parent.name if adapter.name == "adapter" else adapter.name
    else:
        label = (Path(source).name if Path(source).is_dir() else source) if checkpoint else "dense"
    return resolved, source, model_output_tag(label, label)


def evaluate_cell(source: str, resolved_model: str, adapter: Path | None,
                  cell: dict, probes: dict, device: str, max_len: int) -> tuple[dict, dict]:
    """Reload the source each time, so transformations never accumulate."""
    seed_everything(0)
    model, tokenizer = load_text_causal_lm(source, torch.bfloat16)
    try:
        if adapter is not None:
            model = _load_adapter(model, adapter, resolved_model)
        model.to(device).eval()
        threshold = None
        if cell["prune_density"] is not None:
            threshold = apply_global_magnitude_pruning(model, cell["prune_density"], seed=0)
        if cell["quant_bits"] is not None:
            _apply_quantization(model, cell["quant_bits"])
        measured = measure_losses(model, tokenizer, probes, device, max_len)
        return measured, {"tokenizer": str(getattr(tokenizer, "name_or_path", source)),
                          "pruning_threshold": threshold}
    finally:
        del model, tokenizer
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_sweep(*, model_request: str, checkpoint: str | None = None,
              adapter: Path | None = None, prune_density=None, quant_bits=None,
              n_probe: int = 128, probe_seed: int = 0, max_len: int = 1024,
              device: str = "cuda:0", output_base: Path = OUT_BASE,
              dry_run: bool = False) -> list[Path]:
    cells = compression_cells(prune_density, quant_bits)
    if max_len < 2 or not 0 <= probe_seed < 2**32:
        raise ValueError("max length must be >=2 and probe seed in [0, 2**32)")
    resolved, source, base_label = resolve_source(model_request, checkpoint, adapter)
    tag = model_output_tag(model_request, resolved)
    probes = measurement_probes(n_probe, probe_seed)
    paths = [Path(output_base) / tag /
             (f"{base_label}_{cell['suffix']}" if cell["suffix"] else base_label) /
             "loss_validity.json" for cell in cells]
    if dry_run:
        print(f"Resolved model: {resolved}; checkpoint: {source}; adapter: {adapter}")
        print(f"Loss: {LOSS_DEFINITION}; measurement half v[1::2]; "
              f"probe seed={probe_seed}; max_len={max_len} (half prompt, half target)")
        for key, samples in probes.items():
            print(f"{key}: {NAMES[key]}, {BENCHMARKS[key]}, "
                  f"{len(samples)} measurement probes (requested total {n_probe})")
            print(json.dumps(samples[0], ensure_ascii=False, indent=2))
        for cell, path in zip(cells, paths):
            print(f"[would measure] {cell['method']} severity={cell['severity']:g} -> {path}")
        return paths

    protocol = {"loss_definition": LOSS_DEFINITION,
                "legacy_loss_definition": "sum_all_target_CE / sum_all_target_tokens",
                "probe_source": "analysis.v6_capability_geometry.build_probes(include_secondary=True)",
                "probe_half": "measurement (odd indices, v[1::2])",
                "probe_seed": probe_seed, "n_probe_requested": n_probe,
                "max_len": max_len, "prompt_max_tokens": max_len // 2,
                "target_max_tokens": max_len // 2, "chat_template": False,
                "dtype": "bfloat16", "compression_seed": 0,
                "probe_sha256": _digest(probes)}
    baseline = None
    baseline_tokenizer = None
    for cell, path in zip(cells, paths):
        measured, metadata = evaluate_cell(source, resolved, adapter, cell, probes, device, max_len)
        if baseline is None:
            baseline, baseline_tokenizer = measured, metadata["tokenizer"]
        if metadata["tokenizer"] != baseline_tokenizer:
            raise RuntimeError("Tokenizer changed across compression cells")
        capabilities = {}
        for capability, keys in CAPABILITY_PAIRS.items():
            capabilities[capability] = {}
            for role, key in zip(("primary", "secondary"), keys):
                current, reference = measured[key], baseline[key]
                if ([item["n_tokens"] for item in current["items"]] !=
                        [item["n_tokens"] for item in reference["items"]]):
                    raise RuntimeError(f"{key}: target token counts changed across cells")
                capabilities[capability][role] = {
                    **current, "baseline_L_c": reference["L_c"],
                    "delta_L_c": current["L_c"] - reference["L_c"],
                    "baseline_token_weighted_L_c": reference["token_weighted_L_c"],
                    "delta_token_weighted_L_c": (current["token_weighted_L_c"] -
                                                 reference["token_weighted_L_c"]),
                }
        payload = {"version": 23, "model": model_request, "resolved_model": resolved,
                   "model_tag": tag, "source_checkpoint": source,
                   "adapter": str(adapter.resolve()) if adapter else None,
                   "checkpoint": path.parent.name, "baseline_checkpoint": base_label,
                   "baseline_sha256": _digest(baseline), "protocol": protocol,
                   "protocol_sha256": _digest(protocol), "compression": cell,
                   "quantization": "V15 symmetric per-output-channel fake quantization",
                   **metadata, "capabilities": capabilities}
        write_json_atomic(path, payload)
        print(f"[write] {path}", flush=True)
    return paths


def response_agreement(primary, secondary) -> dict:
    """Describe paired nonbaseline deltas; undefined correlations remain null."""
    from scipy.stats import kendalltau, spearmanr

    x, y = np.asarray(primary, dtype=float), np.asarray(secondary, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite([x, y]).all():
        raise ValueError("Response arrays must be paired finite vectors")
    stats = {"n": len(x), "pearson": None, "spearman": None, "kendall_tau": None,
             "ordering_agreement": None, "sign_agreement": None, "n_order_pairs": 0}
    if len(x) >= 3 and np.ptp(x) > 1e-12 and np.ptp(y) > 1e-12:
        stats.update(pearson=float(np.corrcoef(x, y)[0, 1]),
                     spearman=float(spearmanr(x, y).statistic),
                     kendall_tau=float(kendalltau(x, y).statistic))
    if len(x):
        sign = lambda a: np.where(np.abs(a) <= 1e-12, 0, np.sign(a))
        stats["sign_agreement"] = float(np.mean(sign(x) == sign(y)))
        i, j = np.triu_indices(len(x), 1)
        dx, dy = sign(x[j] - x[i]), sign(y[j] - y[i])
        untied = (dx != 0) & (dy != 0)
        stats["n_order_pairs"] = int(untied.sum())
        if untied.any():
            stats["ordering_agreement"] = float(np.mean(dx[untied] == dy[untied]))
    return stats


def law_transfer(severity, primary, secondary) -> dict:
    """Transfer a primary anchored power shape, calibrating only target scale.

    Source shape sees all eligible PRIMARY cells. Only the two mildest
    SECONDARY cells calibrate a nonnegative scale; remaining secondary cells
    are held out. This tests cross-benchmark transfer, not unseen-severity
    extrapolation by the source law. The first >1 nat delta on either benchmark
    and every stronger point are excluded from this perturbative diagnostic.
    """
    order = np.argsort(severity)
    s, x, y = (np.asarray(v, dtype=float)[order] for v in (severity, primary, secondary))
    if len(s) != len(x) or len(x) != len(y) or np.any(s <= 0):
        raise ValueError("Law transfer needs paired positive-severity cells")
    cliffs = np.flatnonzero((x > CLIFF_DELTA) | (y > CLIFF_DELTA))
    cut = int(cliffs[0]) if len(cliffs) else len(s)
    s, x, y = s[:cut], x[:cut], y[:cut]
    result = {"status": "insufficient: need >=4 pre-cliff compression cells",
              "n_eligible": len(s), "n_calibration": 2, "n_heldout": 0}
    if len(s) < 4:
        return result
    # Normalize severity for numerical stability; quant's coordinate is 4^-b.
    z = s / s[-1]
    candidates = []
    for gamma in np.geomspace(0.1, 4.0, 161):
        shape = z ** gamma
        amplitude = float(shape @ x / (shape @ shape))
        candidates.append((float(np.mean((x - amplitude * shape) ** 2)),
                           float(gamma), amplitude, amplitude * shape))
    mse, gamma, amplitude, prediction = min(candidates, key=lambda row: row[0])
    denominator = float(prediction[:2] @ prediction[:2])
    if denominator <= 1e-24:
        result["status"] = "unidentified: primary response is zero"
        return result
    scale = max(0.0, float(prediction[:2] @ y[:2]) / denominator)
    predicted, observed = scale * prediction[2:], y[2:]
    mae = float(np.mean(np.abs(predicted - observed)))
    rms = float(np.sqrt(np.mean(observed ** 2)))
    result.update(status="ok", gamma=gamma, primary_amplitude=amplitude,
                  primary_fit_rmse=math.sqrt(mse), secondary_scale=scale,
                  n_heldout=len(observed), heldout_mae=mae,
                  heldout_relative_mae=mae / rms if rms > 1e-12 else None,
                  zero_response_mae=float(np.mean(np.abs(observed))),
                  calibration_severities=s[:2].tolist(), heldout_severities=s[2:].tolist(),
                  heldout_observed=observed.tolist(), heldout_predicted=predicted.tolist())
    return result


def write_summary(output_base: Path = OUT_BASE, report: Path = REPORT) -> Path:
    """Read V23 artifacts only; never rewrite or augment measurement inputs."""
    groups = defaultdict(list)
    files = sorted(Path(output_base).glob("*/*/loss_validity.json"))
    for path in files:
        payload = json.loads(path.read_text())
        if payload["version"] != 23 or payload["protocol"]["loss_definition"] != LOSS_DEFINITION:
            raise ValueError(f"Unsupported loss-validity protocol in {path}")
        if payload["compression"]["method"] == "baseline":
            continue
        # Never pool different sources, probe panels, tokenizers or dense anchors.
        identity = (payload["model_tag"], payload["baseline_checkpoint"],
                    payload["source_checkpoint"], payload["adapter"],
                    payload["tokenizer"], payload["protocol_sha256"],
                    payload["baseline_sha256"], payload["compression"]["method"])
        groups[identity].append(payload)
    lines = ["# Capability-loss measurement validity", "",
             "Endpoint: capability-conditioned teacher-forced LOSS. "
             "Primary/secondary pairs: MATH-500/GSM8K, MBPP/HumanEval, "
             "2WikiMultihopQA/HotpotQA. No accuracy targets are fitted.", "",
             "Preview with `python analysis/v23_loss_validity.py --model gemma3-270m "
             "--prune-density 0.9 0.8 0.7 0.6 --quant-bits 8 6 4 3 --dry-run`. "
             "For a separately launched measurement worker, replace `--dry-run` with "
             "`--device cuda:0`. An uncompressed reference is included automatically. "
             "Use `--adapter PATH` for V12/PEFT artifacts or `--checkpoint PATH_OR_HF_ID` "
             "for a full checkpoint. Rebuild this report with "
             "`python analysis/v23_loss_validity.py --summarize`.", "",
             "Each L_c is the mean of per-example target CE divided by the scored "
             "target length. JSON also retains the legacy corpus-token-weighted mean. "
             "Both use the V6 measurement half, zero-shot prompt/target tokenization "
             "and fixed truncation; primary values are freshly measured, not imported "
             "from incompatible historical means. GSM8K keeps worked solutions, "
             "HumanEval keeps the canonical continuation, and HotpotQA keeps short answers.", "",
             "Deltas subtract each benchmark's own uncompressed source baseline. "
             "Correlations use compression cells only, allowing different baselines/scales. "
             "Pruning and quantization are separate trajectories; dense anchors are not "
             "scored. Pearson/Spearman/Kendall need >=3 nonconstant paired responses. "
             "Ordering agreement excludes pairs tied on either benchmark; sign agreement "
             "retains negative deltas. N/A means undefined or insufficient data.", "",
             "Law transfer: fit ΔL_primary = a (s/s_max)^γ on all eligible primary cells "
             "(γ grid 0.1–4); calibrate a nonnegative secondary scale on its two mildest "
             "cells, then score the stronger held-out secondary cells. This is benchmark "
             "transfer with target calibration, not zero-shot transfer or source-severity "
             "extrapolation. s=1-density for pruning, s=4^-bits for quantization. At least "
             "four cells are needed. Exclude the first ΔL>1 nat on either benchmark and "
             "all stronger cells from the law diagnostic; correlations retain the full sweep. "
             "MAE is in secondary nats/token; relative MAE divides by held-out secondary "
             "RMS ΔL. Zero MAE is the no-response baseline. This power-shape check is "
             "a validity diagnostic, not a validation of every paper law.", "",
             "Groups keep model/checkpoint, tokenizer, probe protocol and baseline hashes "
             "separate. Correlation alone does not establish capability validity. These "
             "are descriptive point estimates; no seed/item uncertainty is claimed. "
             "Test/validation splits are used for secondary benchmarks; checkpoint training "
             "overlap must still be audited before claiming held-out benchmark validity.", "",
             f"Measurement files: {len(files)}. Compression trajectories: {len(groups)}.", ""]
    if not groups:
        lines += ["**PENDING:** No measured compression trajectories are available. "
                  "The implementation and dry-run do not establish measurement validity.", ""]
    fmt = lambda value: "N/A" if value is None else f"{value:.4g}"
    for identity, payloads in sorted(groups.items(), key=lambda pair: str(pair[0])):
        payloads.sort(key=lambda row: row["compression"]["severity"])
        severities = [row["compression"]["severity"] for row in payloads]
        if len(set(severities)) != len(severities):
            raise ValueError(f"Duplicate compression severity in {identity}")
        model, checkpoint, _, _, _, protocol_hash, baseline_hash, method = identity
        lines += [f"## {model} / {checkpoint} / {method}", "",
                  f"Protocol `{protocol_hash[:12]}`; baseline `{baseline_hash[:12]}`.", "",
                  "| Capability | Cells | Pearson | Spearman | Kendall | Order agreement (pairs) | Sign agreement |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        transfers = []
        for capability in CAPABILITY_PAIRS:
            primary = [p["capabilities"][capability]["primary"]["delta_L_c"] for p in payloads]
            secondary = [p["capabilities"][capability]["secondary"]["delta_L_c"] for p in payloads]
            stats = response_agreement(primary, secondary)
            lines.append(f"| {capability} | {stats['n']} | {fmt(stats['pearson'])} | "
                         f"{fmt(stats['spearman'])} | {fmt(stats['kendall_tau'])} | "
                         f"{fmt(stats['ordering_agreement'])} ({stats['n_order_pairs']}) | "
                         f"{fmt(stats['sign_agreement'])} |")
            transfers.append((capability, law_transfer(severities, primary, secondary)))
        lines += ["", "| Capability | Law transfer | Pre-cliff / held-out | γ | Scale | MAE | Relative MAE | Zero MAE |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for capability, fit in transfers:
            values = " | ".join(fmt(fit.get(key)) for key in
                                ("gamma", "secondary_scale", "heldout_mae",
                                 "heldout_relative_mae", "zero_response_mae"))
            lines.append(f"| {capability} | {fit['status']} | "
                         f"{fit['n_eligible']} / {fit['n_heldout']} | {values} |")
        lines += ["", "| Checkpoint | Capability | Primary L_c | Primary ΔL | Secondary L_c | Secondary ΔL |",
                  "|---|---|---:|---:|---:|---:|"]
        for payload in payloads:
            for capability, pair in payload["capabilities"].items():
                p, s = pair["primary"], pair["secondary"]
                lines.append(f"| {payload['checkpoint']} | {capability} | {fmt(p['L_c'])} | "
                             f"{fmt(p['delta_L_c'])} | {fmt(s['L_c'])} | {fmt(s['delta_L_c'])} |")
        lines.append("")
    report = Path(report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gemma3-270m")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", help="full checkpoint directory or HF model id")
    source.add_argument("--adapter", type=Path, help="existing V12 delta or PEFT adapter")
    parser.add_argument("--prune-density", type=float, nargs="+",
                        help="retained densities; each independently applied to the source")
    parser.add_argument("--quant-bits", type=int, nargs="+",
                        help="V15 fake-quantization bit widths; separate from the prune sweep")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-probe", type=int, default=128)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--output-base", type=Path, default=OUT_BASE)
    parser.add_argument("--report", type=Path, default=REPORT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    if args.summarize:
        print(write_summary(args.output_base, args.report))
        return
    try:
        run_sweep(model_request=args.model, checkpoint=args.checkpoint,
                  adapter=args.adapter, prune_density=args.prune_density,
                  quant_bits=args.quant_bits, n_probe=args.n_probe,
                  probe_seed=args.probe_seed, max_len=args.max_len,
                  device=args.device, output_base=args.output_base, dry_run=args.dry_run)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
