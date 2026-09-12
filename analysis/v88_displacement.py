"""Paired final-logit displacement measurements; no vocabulary-by-token buffers.

Examples (measurement commands are opt-in; CPU is the default device)::

    python -m analysis.v88_displacement --pilot --states pythia-410m@step16000
    python -m analysis.v88_displacement --full --states STATE1 STATE2 \
        --configs prune_d0.9,b3_g64,b4_g0 --device DEVICE
    python -m analysis.v88_displacement --summarize
    python -m analysis.v88_displacement --selftest

Probe JSON maps capabilities to already-selected measurement samples. Without
it, use V6 build_probes(seed=0), then its odd-indexed measurement half. Saved
config dictionaries (including pruning thresholds) can be passed in a JSON list
with --configs-json to replay the exact transform in a fresh output directory.

The stipulated shrinkage predictor omits 0.5*eps**2*Var_p(z), even for pure
shrinkage. It is recorded as specified, not equated to the Taylor polynomial.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time
from types import FunctionType, SimpleNamespace

import torch
from torch.utils._python_dispatch import TorchDispatchMode

try:
    from . import descriptor_bv, model_registry, v6_capability_geometry as v6
    from .eval_records import AGGREGATION_RULE, aggregate_records
    from .tiny_test_model import TinyModel, TinyTokenizer, PROBES
    from .v10_quantization import fake_quantize_per_output_channel
    from .v54_quant_group import fake_quantize_grouped, PROBE_SEED
except ImportError:  # python analysis/v88_displacement.py
    import descriptor_bv
    import model_registry
    import v6_capability_geometry as v6
    from eval_records import AGGREGATION_RULE, aggregate_records
    from tiny_test_model import TinyModel, TinyTokenizer, PROBES
    from v10_quantization import fake_quantize_per_output_channel
    from v54_quant_group import fake_quantize_grouped, PROBE_SEED

ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "results/v88-displacement"
METRICS = (
    "dense_ce", "compressed_ce", "measured_delta", "first_order",
    "second_order", "second_order_prediction", "B", "V", "eps_hat",
    "sigma2_hat", "shrinkage_prediction", "r_norm2", "z_norm2",
    "top16_r2_share", "p_r2", "top16_p_r2",
)
FORMULAS = {
    "measured_delta": "CE(z+r,y) - CE(z,y)",
    "first_order": "sum(p*r) - r[y]",
    "second_order": "0.5 * max(sum(p*r*r) - sum(p*r)**2, 0)",
    "second_order_prediction": "first_order + second_order (Taylor polynomial; remainder O(||r||^3))",
    "B": "z[y] - sum(p*z); descriptor_bv.reduce_final_logits",
    "V": "1 - sum(p*p); descriptor_bv.reduce_final_logits",
    "eps_hat": "-sum(r*z)/sum(z*z); 0 when sum(z*z)==0 (unidentifiable)",
    "sigma2_hat": "Var_p(s), s=r+eps_hat*z; variance clamped at zero for roundoff",
    "shrinkage_prediction": "eps_hat*B + 0.5*sigma2_hat*V",
    "r_norm2": "sum(r*r), unweighted squared Euclidean norm",
    "z_norm2": "sum(z*z), unweighted squared Euclidean norm",
    "top16_r2_share": "sum_{top min(16,vocab) abs(r)} p*r*r / sum(p*r*r); 0 if denominator==0; clamp to [0,1] for roundoff",
}
SHRINKAGE_CAVEAT = (
    "The stipulated shrinkage prediction omits 0.5*eps_hat**2*Var_p(z). "
    "For pure shrinkage its difference from the Taylor polynomial is exactly "
    "that term. Also sigma2_hat is a p-weighted residual variance, not the "
    "coordinate variance of isotropic noise: before projection E[Var_p(eta)] "
    "= sigma_coordinate**2*V. No fitted coefficients or silent rescaling."
)


def digest(value):
    # Match V54's probe hashing serialization exactly.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _rebind(function, **overrides):
    """Private globals, as in eval_records; never patch a frozen module."""
    result = FunctionType(function.__code__, {**function.__globals__, **overrides},
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def scored_inputs(tokenizer, sample, device="cpu", max_len=1024):
    """Execute V6's EXACT tokenization/shift/mask using shape-only dummy CE.

    Only integer IDs and a token-length zero vector are allocated. V6 itself
    returns the scored suffix length. No model forward or vocabulary is needed.
    """
    if max_len < 2:
        raise ValueError("max_len must be at least 2")
    captured = {}

    def shape_model(input_ids, use_cache=False):
        captured["ids"] = input_ids
        return SimpleNamespace(logits=input_ids.unsqueeze(-1))

    def shape_ce(logits, targets, reduction):
        assert reduction == "none"
        return torch.zeros_like(targets, dtype=torch.float32)

    scorer = _rebind(v6.completion_loss, F=SimpleNamespace(cross_entropy=shape_ce))
    _, count = scorer(shape_model, tokenizer, sample["prompt"], sample["completion"],
                      device, max_len=max_len)
    return captured["ids"], count


class FP32MatmulGuard(TorchDispatchMode):
    """Avoid fp32 cuBLAS even in bf16 models' float32 rotary embeddings.

    Transformers RoPE multiplies [batch, frequencies, 1] by [batch, 1, positions].
    That is an outer product, exactly expressible by broadcasting multiplication.
    Any other fp32 matrix product fails before dispatch, including on CPU.
    """
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func._schema.name.split("::")[-1]
        products = {"mm", "bmm", "addmm", "addbmm", "baddbmm", "mv", "_scaled_mm"}
        tensors = [value for value in args if isinstance(value, torch.Tensor)]
        if name in products and any(value.dtype == torch.float32 for value in tensors):
            if (name in ("mm", "bmm") and not kwargs and args[0].shape[-1] == 1
                    and args[1].shape[-2] == 1):
                return args[0] * args[1]
            raise RuntimeError(f"V88 forbids fp32 matrix multiplication: {name}; "
                               "only singleton-axis outer products can use elementwise multiplication")
        return func(*args, **kwargs)


class TokenForward:
    """One final-logit vector from a causal prefix, preserving model transforms.

    use_cache=False matches V6. A restricted output head prevents even native
    bf16 sequence logits. Prefix recomputation trades speed for strict memory
    and no-cache semantics. Unsupported heads fail before a forward.
    """
    def __init__(self, model):
        self.model = model
        self.hook = None
        self.kwargs = {}
        self.tiny = type(model) is TinyModel
        parameters = inspect.signature(model.forward).parameters
        if self.tiny:
            self.method = "TinyModel context-free transition: last input only"
        elif "logits_to_keep" in parameters:
            self.kwargs = {"logits_to_keep": 1}
            self.method = "logits_to_keep=1; causal prefix; use_cache=False"
        elif "num_logits_to_keep" in parameters:
            self.kwargs = {"num_logits_to_keep": 1}
            self.method = "num_logits_to_keep=1; causal prefix; use_cache=False"
        else:
            getter = getattr(model, "get_output_embeddings", None)
            head = getter() if getter else getattr(model, "lm_head", None)
            if not isinstance(head, torch.nn.Linear):
                raise ValueError("A last-token logits API or Linear output head is required; "
                                 "refusing a full vocabulary-by-token forward")

            def last_hidden(module, args):
                hidden = args[0]
                if hidden.ndim != 3 or hidden.shape[0] != 1:
                    raise ValueError("Expected a [1, tokens, hidden] output-head input")
                return (hidden[:, -1:, :], *args[1:])

            self.hook = head.register_forward_pre_hook(last_hidden)
            self.method = "Linear output-head last-hidden pre-hook; causal prefix; use_cache=False"

    def __call__(self, prefix):
        ids = prefix[:, -1:] if self.tiny else prefix
        with FP32MatmulGuard():
            logits = self.model(input_ids=ids, use_cache=False, **self.kwargs).logits
        if logits.ndim != 3 or logits.shape[:2] != (1, 1):
            raise ValueError("Model did not honor the single-token output restriction")
        return logits[0, 0]

    def close(self):
        if self.hook is not None:
            self.hook.remove()


def reduce_token(dense_logits, compressed_logits, target):
    """All vocabulary arithmetic float32 AFTER forwards; no matrix products."""
    if dense_logits.ndim != 1 or compressed_logits.shape != dense_logits.shape:
        raise ValueError("reduce_token requires two equal one-dimensional logit vectors")
    z, compressed = dense_logits.float(), compressed_logits.float()
    r = compressed - z
    p = torch.softmax(z, dim=-1, dtype=torch.float32)
    pr = (p * r).sum(dtype=torch.float32)
    pr2 = (p * r.square()).sum(dtype=torch.float32)
    variance = (pr2 - pr.square()).clamp_min(0)
    r_norm2, z_norm2 = r.square().sum(), z.square().sum()
    eps = -(r * z).sum() / torch.where(z_norm2 > 0, z_norm2, torch.ones_like(z_norm2))
    residual = r + eps * z
    sigma2 = ((p * residual.square()).sum() - (p * residual).sum().square()).clamp_min(0)
    y = torch.as_tensor(target, device=z.device, dtype=torch.long).reshape(1)
    b, v = descriptor_bv.reduce_final_logits(z.unsqueeze(0), y, chunk_tokens=1)
    dense_ce = torch.logsumexp(z, dim=0) - z[target]
    compressed_ce = torch.logsumexp(compressed, dim=0) - compressed[target]
    first = pr - r[target]
    second = 0.5 * variance
    top = r.abs().topk(min(16, r.numel()), sorted=False).indices
    top_mass = (p[top] * r[top].square()).sum()
    share = (top_mass / torch.where(pr2 > 0, pr2, torch.ones_like(pr2))).clamp(0, 1)
    values = {
        "dense_ce": dense_ce, "compressed_ce": compressed_ce,
        "measured_delta": compressed_ce - dense_ce,
        "first_order": first, "second_order": second,
        "second_order_prediction": first + second,
        "B": b, "V": v, "eps_hat": eps, "sigma2_hat": sigma2,
        "shrinkage_prediction": eps * b + 0.5 * sigma2 * v,
        "r_norm2": r_norm2, "z_norm2": z_norm2,
        "top16_r2_share": share, "p_r2": pr2, "top16_p_r2": top_mass,
    }
    result = {key: float(value) for key, value in values.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Non-finite logit displacement statistic")
    return result


@torch.no_grad()
def measure_pair(dense, compressed, tokenizer, probes, device="cpu", max_len=1024):
    """Pair forwards on each scored token; retain Python scalar sample sums only."""
    if dense is compressed:
        raise ValueError("Dense and compressed must be separate models")
    rows = []
    dense.eval()
    compressed.eval()
    forwards = []
    try:
        for model in (dense, compressed):
            forwards.append(TokenForward(model))
        for capability, samples in probes.items():
            for index, sample in enumerate(samples):
                ids, count = scored_inputs(tokenizer, sample, device, max_len)
                sums = dict.fromkeys(METRICS, 0.0)
                nonzero_displacements = 0
                for target_position in range(ids.shape[1] - count, ids.shape[1]):
                    prefix = ids[:, :target_position]
                    z = forwards[0](prefix)
                    q = forwards[1](prefix)
                    values = reduce_token(z, q, ids[0, target_position])
                    del z, q
                    for key in METRICS:
                        sums[key] += values[key]
                    nonzero_displacements += values["p_r2"] > 0
                rows.append({
                    "capability": capability, "sample_index": index,
                    "sample_id": str(sample.get("sample_id", sample.get("id", digest(sample)))),
                    "sample_sha256": digest(sample), "scored_token_count": count,
                    "nonzero_displacement_token_count": nonzero_displacements,
                    "sums": sums,
                    **{key: value / count if count else None for key, value in sums.items()},
                })
        methods = [forward.method for forward in forwards]
    finally:
        for forward in forwards:
            forward.close()
    pooled = {}
    for capability, samples in probes.items():
        group = [row for row in rows if row["capability"] == capability]
        count = sum(row["scored_token_count"] for row in group)
        sums = {key: sum(row["sums"][key] for row in group) for key in METRICS}
        values = {}
        for key in METRICS:
            records = [{"capability": capability, "summed_nll": row["sums"][key],
                        "scored_token_count": row["scored_token_count"]} for row in group]
            values[key] = aggregate_records(records).get(capability, 0.0)
        nonzero = sum(row["nonzero_displacement_token_count"] for row in group)
        pooled[capability] = {
            "sample_count": len(samples), "scored_sample_count": sum(row["scored_token_count"] > 0 for row in group),
            "scored_token_count": count, "sums": sums, **values,
            "nonzero_displacement_token_count": nonzero,
            "top16_r2_share_nonzero_mean": sums["top16_r2_share"] / nonzero if nonzero else None,
            "top16_r2_mass_share": sums["top16_p_r2"] / sums["p_r2"] if sums["p_r2"] else None,
        }
    return {"per_capability": pooled, "per_sample": rows, "forward_methods": methods}


def normalize_config(config):
    if isinstance(config, str):
        match = re.fullmatch(r"(?:prune_d|p)([0-9.eE+-]+)", config)
        quant = re.fullmatch(r"b([0-9]+)(?:_g([0-9]+))?", config)
        if match:
            config = {"kind": "prune", "density": float(match[1])}
        elif quant:
            group = int(quant[2] or 0)
            config = {"kind": "grouped_rtn" if group else "per_channel_rtn",
                      "bits": int(quant[1]), "group_size": group}
        else:
            raise ValueError(f"Invalid config {config!r}; use prune_d0.9, b3_g64, or b4_g0")
    config = dict(config)
    kind = config.get("kind")
    if kind == "prune":
        density = float(config["density"])
        if not 0 < density <= 1:
            raise ValueError("density must be in (0, 1]")
        threshold = config.get("threshold")
        if threshold is not None and (not math.isfinite(float(threshold)) or float(threshold) < 0):
            raise ValueError("stored threshold must be finite and nonnegative")
        seed = config.get("seed", PROBE_SEED)
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("pruning seed must be a nonnegative integer")
        return {"id": f"prune_d{density}", "kind": kind, "density": density,
                "seed": seed, "threshold": float(threshold) if threshold is not None else None}
    if kind not in ("grouped_rtn", "per_channel_rtn"):
        raise ValueError(f"Unknown compression kind {kind!r}")
    bits, group = config["bits"], config.get("group_size", 0)
    if not isinstance(bits, int) or isinstance(bits, bool) or not 2 <= bits <= 16:
        raise ValueError("bits must be an integer from 2 to 16")
    if not isinstance(group, int) or isinstance(group, bool) or group < 0:
        raise ValueError("group_size must be a nonnegative integer")
    if kind == "per_channel_rtn" and group:
        raise ValueError("per_channel_rtn must have group_size=0")
    if config.get("mode", "symmetric") != "symmetric":
        raise ValueError("Only the frozen default symmetric RTN mode is supported")
    return {"id": f"b{bits}_g{group}", "kind": "grouped_rtn" if group else "per_channel_rtn",
            "bits": bits, "group_size": group, "mode": "symmetric"}


def compress_copy(dense, config):
    """Clone once; retain original bf16 tensors and record the realized threshold."""
    config = normalize_config(config)
    compressed = copy.deepcopy(dense)
    if config["kind"] == "prune":
        threshold = v6.apply_global_magnitude_pruning(
            compressed, config["density"], seed=config["seed"], threshold=config["threshold"],
            reference_weights=[p.detach() for _, p in v6.language_weight_parameters(dense)])
        config["threshold"] = threshold if math.isfinite(threshold) else None
        config["threshold_rule"] = "identity at density=1; otherwise abs(w)>threshold, V6 proportional sampled quantile"
        config["threshold_sample_size"] = v6.PRUNE_THRESHOLD_SAMPLE_SIZE
    else:
        with torch.no_grad():
            for _, parameter in v6.language_weight_parameters(compressed):
                if config["kind"] == "grouped_rtn":
                    quantized = fake_quantize_grouped(parameter, config["bits"], config["group_size"])
                else:
                    quantized = fake_quantize_per_output_channel(parameter, config["bits"])
                parameter.copy_(quantized)
    return compressed, config


@torch.no_grad()
def _streaming_sanity(model, tokenizer, model_name):
    """Keep the frozen loader's sanity guard without its full sequence logits."""
    ids = tokenizer(v6.SANITY_PROMPT, return_tensors="pt").input_ids
    if ids.shape[1] < 2:
        raise RuntimeError(f"Checkpoint-integrity failure for {model_name}: sanity prompt too short")
    ids = ids.to(next(model.parameters()).device)
    model.eval()
    forward = TokenForward(model)
    try:
        losses = []
        for pos in range(1, ids.shape[1]):
            z = forward(ids[:, :pos]).float()
            losses.append(float(torch.logsumexp(z, 0) - z[ids[0, pos]]))
        loss = sum(losses) / len(losses)
    finally:
        forward.close()
    if not math.isfinite(loss) or loss > v6.SANITY_CE_LIMIT:
        raise RuntimeError(f"Checkpoint-integrity failure for {model_name}: sanity CE={loss}")
    return loss


def load_model(model_id, dtype, revision):
    loader = _rebind(v6.load_text_causal_lm, _checkpoint_sanity_forward=_streaming_sanity)
    return loader(model_id, dtype, revision)


def _output_path(path):
    path = Path(path).absolute()
    resolved = path.resolve()
    # Check lexical and resolved locations, including symlinks out of results.
    for candidate in (path, resolved):
        if candidate.is_relative_to(ROOT / "results") and not candidate.is_relative_to(OUT_BASE):
            raise ValueError("results output must be under results/v88-displacement/")
    if path.is_relative_to(OUT_BASE) and not resolved.is_relative_to(OUT_BASE):
        raise ValueError("Output symlink escapes results/v88-displacement/")
    if path.is_symlink():
        raise ValueError("Refusing symlink output")
    return resolved


def write_json(path, payload, *, replace=False):
    path = _output_path(path)
    data = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace:
        with path.open("x") as handle:
            handle.write(data)
    else:
        path.write_text(data)


def _weight_hash(model, model_id, revision):
    """Hash tiny in-memory weights; otherwise inspect local HF blob names only."""
    tensors = list(model.state_dict().items())
    if sum(t.numel() * t.element_size() for _, t in tensors) <= 1_048_576:
        sha = hashlib.sha256()
        for name, tensor in tensors:
            sha.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
            sha.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return sha.hexdigest(), "named state_dict tensor metadata and bytes (<=1 MiB)"
    try:
        from huggingface_hub import try_to_load_from_cache
        for name in ("model.safetensors", "pytorch_model.bin"):
            cached = try_to_load_from_cache(model_id, name, revision=revision or "main")
            if isinstance(cached, str):
                blob = Path(cached).resolve().name
                if re.fullmatch(r"[0-9a-f]{64}", blob):
                    return blob, f"local Hugging Face LFS blob name: {name} (no weight-file read)"
    except (ImportError, ValueError, OSError):
        pass
    return None, "Unavailable cheaply; no full checkpoint read/hash (may be sharded)"


def provenance():
    names = ("v88_displacement.py", "descriptor_bv.py", "eval_records.py", "model_registry.py",
             "v6_capability_geometry.py", "v10_quantization.py", "v54_quant_group.py")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
            "code_sha256": {name: hashlib.sha256((ROOT / "analysis" / name).read_bytes()).hexdigest()
                            for name in names},
            "torch_version": torch.__version__, "python_version": platform.python_version(),
            "aggregation_rule": AGGREGATION_RULE, "probability_reduction_dtype": "float32",
            "sample_accumulation": "Python float sums of float32 token reductions",
            "logit_source": "model(...).logits after final architecture transforms",
            "token_mask": "v6.completion_loss code executed with private shape-only CE",
            "matmul_rule": "All model forwards: reject fp32 matrix products before dispatch; "
                           "RoPE singleton-axis outer products use elementwise multiplication, including on CPU.",
            "formulas": FORMULAS, "shrinkage_caveat": SHRINKAGE_CAVEAT,
            "memory_rule": "one vector per model per scored token; no retained logit/displacement arrays",
            "forward_caveat": "Causal prefixes with no cache; output-head shape differs from full V6 forward. "
                              "Floating-point kernel differences and V6 native-dtype CE rounding can affect reproduction."}


def run_state(state, configs, probes=None, *, device="cpu", dtype="bfloat16",
              output_dir=OUT_BASE, revision=None, n_probe=128, max_len=1024,
              model_loader=None, selftest=False):
    """Measure each config with two simultaneously resident models; never resume stale CE."""
    output_dir = _output_path(output_dir)
    dtype_map = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                 "fp32": torch.float32, "float32": torch.float32}
    if dtype not in dtype_map:
        raise ValueError("dtype must be bfloat16 or float32 (CPU diagnostics only)")
    torch_dtype = dtype_map[dtype]
    target_device = torch.device(device)
    if target_device.type != "cpu" and torch_dtype != torch.bfloat16:
        raise ValueError("Non-CPU measurement requires bfloat16; no fp32 model matmul")
    model_id, embedded_revision = model_registry.resolve_model_and_revision(state)
    if revision is not None and embedded_revision is not None and revision != embedded_revision:
        raise ValueError("Conflicting state and explicit revisions")
    revision = revision if revision is not None else embedded_revision
    model_registry.require_compliant(state)
    tag = v6.model_output_tag(state, model_id, revision)
    configs = [normalize_config(config) for config in configs]
    if not configs or len({config["id"] for config in configs}) != len(configs):
        raise ValueError("At least one config is required; config IDs must be unique")
    paths = [_output_path(output_dir / tag / (config["id"] + ".json")) for config in configs]
    if any(path.exists() for path in paths):
        raise FileExistsError("Measurement exists; use a fresh output directory (never overwrite JSON)")
    probe_source = "provided measurement probes (no additional split)"
    if probes is None:
        if n_probe < 2:
            raise ValueError("n_probe must be at least 2")
        probes = {cap: rows[1::2] for cap, rows in v6.build_probes(n_probe, seed=PROBE_SEED).items()}
        probe_source = "v6.build_probes, odd-indexed [1::2]"
    if not probes or any(not isinstance(rows, list) for rows in probes.values()):
        raise ValueError("probes must map capabilities to sample lists")
    dense, tokenizer = (model_loader or load_model)(model_id, torch_dtype, revision)
    dense.to(device=target_device, dtype=torch_dtype).eval().requires_grad_(False)
    resolved_revision = getattr(getattr(dense, "config", None), "_commit_hash", None)
    weight_hash, weight_hash_basis = _weight_hash(dense, model_id, resolved_revision or revision)
    device_name = (platform.processor() or platform.machine()) if target_device.type == "cpu" else (
        torch.cuda.get_device_name(target_device) if target_device.type == "cuda" else str(target_device))
    common_provenance = provenance()
    outputs = []
    for config, path in zip(configs, paths):
        started = time.perf_counter()
        compressed, realized_config = compress_copy(dense, config)
        try:
            measured = measure_pair(dense, compressed, tokenizer, probes, str(target_device), max_len)
        finally:
            del compressed
        payload = {
            "schema_version": 1, "experiment": "v88-displacement", "selftest": selftest,
            "state": state, "state_tag": tag, "model_id": model_id,
            "revision": revision, "resolved_revision": resolved_revision,
            "weight_sha256": weight_hash, "weight_sha256_basis": weight_hash_basis,
            "config": realized_config, "dtype": str(torch_dtype).removeprefix("torch."),
            "device": str(target_device), "device_name": device_name,
            "probe_sha256": digest(probes), "probe_seed": PROBE_SEED,
            "probe_source": probe_source, "n_probe": n_probe if probes is not None and probe_source.startswith("v6.") else None,
            "max_len": max_len,
            "scored_token_counts": {cap: data["scored_token_count"] for cap, data in measured["per_capability"].items()},
            **measured, "provenance": {**common_provenance, "wall_time_s": time.perf_counter() - started},
        }
        write_json(path, payload)
        outputs.append(path)
        print(f"wrote {path}", flush=True)
    return outputs


def _regime(delta):
    return "mild" if abs(delta) < 0.1 else "moderate" if abs(delta) <= 0.5 else "severe"


def _prediction_stats(rows, key):
    errors = [abs(row[key] - row["measured_delta"]) for row in rows]
    relative = [error / abs(row["measured_delta"]) for row, error in zip(rows, errors)
                if row["measured_delta"] != 0]
    sign = lambda x: (x > 0) - (x < 0)
    return {"mae": statistics.mean(errors) if errors else None,
            "median_relative_error": statistics.median(relative) if relative else None,
            "relative_error_count": len(relative),
            "zero_damage_count": sum(row["measured_delta"] == 0 for row in rows),
            "sign_agreement": statistics.mean(sign(row[key]) == sign(row["measured_delta"]) for row in rows) if rows else None}


def _distribution(values):
    values = [value for value in values if value is not None]
    return {"count": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def frozen_reproductions(payload, frozen_root=ROOT / "results"):
    """Read only direct frozen state/config matches; differences are never corrections."""
    config = payload["config"]
    if config["kind"] == "prune":
        directory, filename = "v6-capability-geometry", "prune_losses.json"
    else:
        directory, filename = "v54-quant-group", "quant_group_losses.json"
    aliases = [key for key, entry in model_registry.MODEL_REGISTRY.items() if entry["hf_id"] == payload["model_id"]]
    tags = {payload["state_tag"], v6.model_output_tag(payload["model_id"], payload["model_id"], payload["revision"])}
    tags.update(v6.model_output_tag(alias, payload["model_id"], payload["revision"]) for alias in aliases)
    comparisons = []
    for path in sorted((Path(frozen_root) / directory).glob(f"*/{filename}")):
        frozen = json.loads(path.read_text())
        meta = frozen.get("_meta", {})
        if "hf_id" in meta:
            if meta["hf_id"] != payload["model_id"] or meta.get("revision") != payload["revision"]:
                continue
        elif path.parent.name not in tags:
            continue
        if config["kind"] == "prune":
            matches = [key for key in frozen if not key.startswith("_") and _float_equal(key, config["density"])]
            dense_key = "1.0"
        else:
            matches, dense_key = [config["id"]], "dense"
            if meta.get("quantization", "symmetric").startswith("asymmetric"):
                continue
            if meta.get("configs", {}).get(config["id"], {}).get("mode", "symmetric") != "symmetric":
                continue
        for key in matches:
            for cap, row in payload["per_capability"].items():
                if row["scored_token_count"] == 0 or cap not in frozen.get(key, {}) or cap not in frozen.get(dense_key, {}):
                    continue
                old = frozen[key][cap] - frozen[dense_key][cap]
                comparisons.append({
                    "state": payload["state"], "revision": payload["revision"],
                    "state_tag": payload["state_tag"], "config": config["id"], "capability": cap,
                    "frozen_path": str(path), "frozen_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "frozen_delta": old, "recomputed_delta": row["measured_delta"],
                    "difference": row["measured_delta"] - old,
                    "probe_sha256_match": meta["probe_sha256"] == payload["probe_sha256"] if "probe_sha256" in meta else None,
                    "frozen_dtype": meta.get("model_dtype", "bf16" if directory.startswith("v6-") else None),
                    "threshold_verified": None,
                    "interpretation": "Reproduction check only; frozen measurements are unchanged. "
                                      "V6 lacks probe hashes/stored thresholds; native CE dtype and forward shapes may differ.",
                })
    return comparisons


def _float_equal(value, expected):
    try:
        return float(value) == expected
    except ValueError:
        return False


def summarize(output_dir=OUT_BASE, *, frozen_root=ROOT / "results", include_selftest=False):
    output_dir = _output_path(output_dir)
    rows, comparisons, inputs = [], [], []
    for path in sorted(output_dir.glob("*/*.json")):
        payload = json.loads(path.read_text())
        if payload.get("experiment") != "v88-displacement" or (payload.get("selftest") and not include_selftest):
            continue
        inputs.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        comparisons.extend(frozen_reproductions(payload, frozen_root))
        for cap, row in payload["per_capability"].items():
            if row["scored_token_count"]:
                rows.append({"state": payload["state"], "revision": payload["revision"],
                             "resolved_revision": payload.get("resolved_revision"),
                             "state_tag": payload["state_tag"], "model_id": payload["model_id"],
                             "config": payload["config"]["id"],
                             "capability": cap, "dtype": payload["dtype"], "device": payload["device"],
                             "probe_sha256": payload["probe_sha256"], **row})
    regimes = {}
    for name in ("mild", "moderate", "severe"):
        group = [row for row in rows if _regime(row["measured_delta"]) == name]
        regimes[name] = {
            "count": len(group),
            "second_order_prediction": _prediction_stats(group, "second_order_prediction"),
            "shrinkage_prediction": _prediction_stats(group, "shrinkage_prediction"),
            "isotropy": {key: _distribution([row[key] for row in group]) for key in
                         ("top16_r2_share", "top16_r2_share_nonzero_mean", "top16_r2_mass_share")},
        }
    summary = {"schema_version": 1, "experiment": "v88-displacement-summary",
               "unit": "one pooled (state, config, capability) measurement, equally weighted",
               "regime_boundaries": "mild |dL|<0.1; moderate 0.1<=|dL|<=0.5; severe |dL|>0.5",
               "relative_error_rule": "abs(pred-measured)/abs(measured); zero measured values excluded and counted",
               "sign_rule": "sign(0)=0; exact sign equality, no tolerance",
               "shrinkage_caveat": SHRINKAGE_CAVEAT, "inputs": inputs, "regimes": regimes,
               "measurements": rows, "reproduction_checks": comparisons,
               "provenance": provenance()}
    write_json(output_dir / "summary.json", summary, replace=True)
    fmt = lambda x: "n/a" if x is None else f"{x:.6g}"
    lines = ["# V88 displacement summary", "", summary["unit"] + ".", "",
             summary["regime_boundaries"], "", summary["relative_error_rule"], "", SHRINKAGE_CAVEAT, "",
             "| Regime | N | Prediction | MAE | Median relative error | Sign agreement |",
             "|---|---:|---|---:|---:|---:|"]
    for name, data in regimes.items():
        for key in ("second_order_prediction", "shrinkage_prediction"):
            stats = data[key]
            lines.append(f"| {name} | {data['count']} | {key} | {fmt(stats['mae'])} | "
                         f"{fmt(stats['median_relative_error'])} | {fmt(stats['sign_agreement'])} |")
    lines += ["", "Top-16 |r| share of p-weighted r² (larger shares indicate concentration; not an isotropy proof).",
              "Zero displacement has share 0; nonzero-token means and pooled mass ratios are also reported.", "",
              "| Regime | Mean token share | Median token share | Mean nonzero share | Mean mass share |",
              "|---|---:|---:|---:|---:|"]
    for name, data in regimes.items():
        iso = data["isotropy"]
        lines.append(f"| {name} | {fmt(iso['top16_r2_share']['mean'])} | {fmt(iso['top16_r2_share']['median'])} | "
                     f"{fmt(iso['top16_r2_share_nonzero_mean']['mean'])} | {fmt(iso['top16_r2_mass_share']['mean'])} |")
    lines += ["", "## Frozen reproduction checks", "", "Differences are recomputed minus frozen; no frozen value is corrected.", "",
              "| State | Config | Capability | Recomputed dL | Frozen dL | Difference | Probe hash match |",
              "|---|---|---|---:|---:|---:|---|"]
    for row in comparisons:
        lines.append(f"| {row['state_tag']} | {row['config']} | {row['capability']} | {fmt(row['recomputed_delta'])} | "
                     f"{fmt(row['frozen_delta'])} | {fmt(row['difference'])} | {row['probe_sha256_match']} |")
    if not comparisons:
        lines.append("\nNo matching frozen measurements.")
    lines += ["", "V6 probe hashes and realized thresholds may be unavailable. Forward shapes and CE arithmetic differ; "
              "see summary.json for source hashes and protocol metadata."]
    markdown = _output_path(output_dir / "summary.md")
    markdown.write_text("\n".join(lines) + "\n")
    return summary


def selftest(output_dir):
    """Offline bf16 tiny-model integration check. Synthetic outputs are marked."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        paths = run_state("v88-tiny@seed87", ["prune_d0.8", "b3_g4"], PROBES,
                          output_dir=output_dir, device="cpu", selftest=True,
                          model_loader=lambda model_id, dtype, revision: (TinyModel(dtype), TinyTokenizer()))
        for path in paths:
            payload = json.loads(path.read_text())
            assert payload["scored_token_counts"] == {"math": 8, "code": 2, "qa": 3}
            assert len(payload["per_sample"]) == 4
            for row in payload["per_sample"]:
                assert row["second_order_prediction"] == (row["sums"]["second_order_prediction"] / row["scored_token_count"])
        print("V88 CPU selftest: passed (bf16 tiny model, two configs)")
        return paths
    finally:
        torch.set_num_threads(previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for mode in ("pilot", "full", "summarize", "selftest"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--states", "--model", nargs="+")
    parser.add_argument("--revision")
    configs = parser.add_mutually_exclusive_group()
    configs.add_argument("--configs", default="prune_d0.9,b3_g64")
    configs.add_argument("--configs-json", type=Path)
    parser.add_argument("--probe-set", type=Path)
    parser.add_argument("--n-probe", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("bf16", "bfloat16", "fp32", "float32"), default="bfloat16")
    parser.add_argument("--output-dir", type=Path, default=OUT_BASE)
    args = parser.parse_args(argv)
    if args.selftest:
        if torch.device(args.device).type != "cpu":
            parser.error("--selftest is CPU only")
        selftest(args.output_dir)
    elif args.summarize:
        summary = summarize(args.output_dir)
        print(f"Summarized {len(summary['measurements'])} capability measurements in {args.output_dir}")
    else:
        if not args.states:
            parser.error("--states is required for measurement")
        configs = json.loads(args.configs_json.read_text()) if args.configs_json else args.configs.split(",")
        if args.pilot and (len(args.states) != 1 or len(configs) != 2):
            parser.error("--pilot requires exactly one state and two configs")
        probes = json.loads(args.probe_set.read_text()) if args.probe_set else None
        for state in args.states:
            run_state(state, configs, probes, device=args.device, dtype=args.dtype,
                      output_dir=args.output_dir, revision=args.revision,
                      n_probe=args.n_probe, max_len=args.max_len)


if __name__ == "__main__":
    main()
