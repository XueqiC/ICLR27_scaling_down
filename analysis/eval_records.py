"""V87 additive evaluation records around the frozen V6/V10/V12 protocols.

``python -m analysis.eval_records distill <V12 arguments>`` or
``python -m analysis.eval_records quantize <V10 arguments>`` enables records.
The original entry points remain frozen. See descriptor_bv.schema.md.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from types import FunctionType
from unittest.mock import patch

import numpy as np
import torch

try:
    from . import v10_quantization as v10
    from . import v12_distill as v12
    from .v6_capability_geometry import completion_loss
except ImportError:
    import v10_quantization as v10
    import v12_distill as v12
    from v6_capability_geometry import completion_loss

KEY = "per_example_v87"
AGGREGATION_RULE = "sum_i sum_scored_tokens(value_i) / sum_i scored_token_count_i; pooled tokens, not mean of sample means"
BYTE_AGGREGATION_RULE = "sum_i sum_scored_tokens(value_i) / sum_i scored_reference_byte_count_i; pooled bytes"
BENCHMARKS = {"math": "MATH-500", "code": "MBPP", "qa": "2WikiMultihopQA"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def record(sample, capability, index, summed_nll, n_tokens, distribution=None):
    prompt, reference = sample["prompt"], sample["completion"]
    identity = digest([prompt, reference])
    return {"sample_id": str(sample.get("sample_id", sample.get("id", identity))),
            "sample_index": index, "capability": capability,
            "distribution": str(sample.get("distribution", distribution or BENCHMARKS.get(capability, capability))),
            "summed_nll": float(summed_nll), "scored_token_count": int(n_tokens),
            "reference_byte_count": len(reference.encode("utf-8")),
            "prompt_sha256": digest(prompt), "reference_sha256": digest(reference),
            "prompt_reference_sha256": identity}


class RecordedLosses(dict):
    """All original dictionary keys/values, with non-serialized record metadata."""
    def __init__(self, values, records):
        super().__init__(values)
        self.records = records


def _v10_with_scorer(scorer):
    # Rebind only the scorer in a private globals dictionary. Execute the EXACT
    # frozen aggregator's code, without editing its source or patching globals.
    original = v10._measure_capability_losses
    return FunctionType(original.__code__, {**original.__globals__, "completion_loss": scorer},
                        original.__name__, original.__defaults__, original.__closure__)


def aggregate_records(records, value_key="summed_nll", count_key="scored_token_count"):
    """Execute V10's original accumulation order and zero-denominator rule."""
    probes = {}
    for row in records:
        probes.setdefault(row["capability"], []).append({"prompt": row, "completion": ""})

    def scorer(model, tokenizer, row, completion, device):
        return row[value_key], row[count_key]

    return _v10_with_scorer(scorer)(None, None, probes, "cpu")


def measure_capability_losses(model, tokenizer, probes, device, *, distribution=None):
    """Frozen V10 evaluation plus one record per scored example, same forwards."""
    records = []
    iterator = iter((cap, i, sample) for cap, samples in probes.items()
                    for i, sample in enumerate(samples))

    def scorer(*args, **kwargs):
        loss, count = completion_loss(*args, **kwargs)
        cap, i, sample = next(iterator)
        if count:
            records.append(record(sample, cap, i, loss, count, distribution))
        return loss, count

    values = _v10_with_scorer(scorer)(model, tokenizer, probes, device)
    return RecordedLosses(values, records)


def measure_distillation(model, tokenizer, probes, device, max_len=v12.MAX_LEN,
                         loss_function=completion_loss, *, distribution=None):
    records = []
    iterator = iter((cap, i, sample) for cap in v12.CAPABILITIES
                    for i, sample in enumerate(probes[cap]))

    def scorer(*args, **kwargs):
        loss, count = loss_function(*args, **kwargs)
        cap, i, sample = next(iterator)
        if count:
            records.append(record(sample, cap, i, loss, count, distribution))
        return loss, count

    losses, counts = _ORIGINAL_DISTILL(model, tokenizer, probes, device, max_len, scorer)
    return RecordedLosses(losses, records), counts


_ORIGINAL_DISTILL = v12.measure_capability_losses


def with_records(payload, *, placement="top"):
    """Add a single top-level key; never alter old scalar or capability keys."""
    if not isinstance(payload, dict):
        return payload
    evaluations = {key: value.records for key, value in payload.items()
                   if isinstance(value, RecordedLosses)}
    if not evaluations:
        return payload
    container = payload if placement == "top" else payload["dense"]
    previous = container.get(KEY, {})
    metadata = {"schema_version": 1, "aggregation_rule": AGGREGATION_RULE,
                "reference_bytes_rule": "UTF-8 bytes of original full reference, before legacy truncation",
                "evaluations": {**previous.get("evaluations", {}), **evaluations}}
    if placement == "dense":
        # Legacy V17/V14 iterate top-level V10 bit keys and cast them to numbers.
        # Put V10 metadata under dense so those readers keep accepting the file.
        return {**payload, "dense": {**payload["dense"], KEY: metadata}}
    return {**payload, KEY: metadata}


class _RecordingJSON:
    def __init__(self, placement="top"):
        self.placement = placement
        self.previous_evaluations = {}

    def __getattr__(self, name):
        return getattr(json, name)

    def dumps(self, payload, *args, **kwargs):
        result = with_records(payload, placement=self.placement)
        if self.placement == "dense" and isinstance(result, dict) and "dense" in result:
            meta = result["dense"].get(KEY)
            if meta is not None:
                meta["evaluations"] = {**self.previous_evaluations, **meta["evaluations"]}
        return json.dumps(result, *args, **kwargs)

    def loads(self, value, *args, **kwargs):
        result = json.loads(value, *args, **kwargs)
        if self.placement == "dense" and isinstance(result, dict):
            self.previous_evaluations = result.get("dense", {}).get(KEY, {}).get("evaluations", {})
        return result


@contextmanager
def preserve_cpu_training_state(model):
    """V12 state-preservation semantics for CPU, with no accelerator queries."""
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("CPU state adapter needs a CPU model")
    modes = [(module, module.training) for module in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        for module, training in modes:
            module.training = training


def seed_cpu(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)


def cpu_optimizer_health_check(optimizer):
    if any(p.device.type != "cpu" for group in optimizer.param_groups for p in group["params"]):
        raise ValueError("CPU optimizer adapter received a non-CPU parameter")


@contextmanager
def recording_evaluation(distribution=None):
    """Scoped entry-point adapter; restores all bindings even on failure.

    Only the target modules' bindings change (never the shared json module).
    Use this around a whole sequential V12/V10 run, not concurrent threads.
    """
    def distill(*args, **kwargs):
        return measure_distillation(*args, **kwargs, distribution=distribution)

    # Preserve V10's original aggregator for the private rebinding above.
    def quant(*args, **kwargs):
        return measure_capability_losses(*args, **kwargs, distribution=distribution)

    with patch.object(v12, "measure_capability_losses", distill), \
         patch.object(v12, "json", _RecordingJSON()), \
         patch.object(v10, "json", _RecordingJSON("dense")):
        # Clone the runner's globals, so nested calls select our evaluator while
        # the immutable evaluator remains available for exact aggregation.
        runner = v10.run_quantization
        wrapped = FunctionType(runner.__code__, {**runner.__globals__,
                               "_measure_capability_losses": quant}, runner.__name__,
                               runner.__defaults__, runner.__closure__)
        wrapped.__kwdefaults__ = runner.__kwdefaults__
        with patch.object(v10, "run_quantization", wrapped):
            yield


def read_eval(path):
    """Read old or additive eval JSON verbatim; absent records stay absent."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("evaluation must be a JSON object")
    metadata = payload.get(KEY, payload.get("dense", {}).get(KEY))
    if metadata is not None:
        for name, rows in metadata["evaluations"].items():
            for row in rows:
                count = row["scored_token_count"]
                if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                    raise ValueError("scored records need positive integer token counts")
                if not math.isfinite(row["summed_nll"]) or row["summed_nll"] < 0:
                    raise ValueError("scored records need finite nonnegative NLL")
            actual = aggregate_records(rows)
            expected = {cap: value for cap, value in payload[name].items() if cap != KEY}
            if set(actual) - set(expected):
                raise ValueError(f"per-example capability mismatch: {name}")
            for cap, value in expected.items():
                if not math.isclose(actual.get(cap, 0.0), value, rel_tol=0, abs_tol=1e-9):
                    raise ValueError(f"per-example aggregate mismatch: {name}/{cap}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("distill", "quantize"))
    parser.add_argument("--distribution")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", "--output-base", type=Path)
    args, remaining = parser.parse_known_args()
    if torch.device(args.device).type != "cpu":
        parser.error("V87 entry point is CPU only")
    root = Path(__file__).resolve().parents[1]
    output = (args.output_dir or root / "results/v87-prep" / args.mode).resolve()
    if output.is_relative_to(root / "results") and not output.is_relative_to(root / "results/v87-prep"):
        parser.error("results output must be under results/v87-prep/")
    with recording_evaluation(args.distribution), \
         patch.object(v12, "preserve_training_state", preserve_cpu_training_state), \
         patch.object(v12, "seed_everything", seed_cpu), \
         patch.object(torch.optim.Optimizer, "_cuda_graph_capture_health_check", cpu_optimizer_health_check), \
         patch.object(v12, "OUT_BASE", output), patch.object(v10, "OUT_BASE", output), \
         patch.object(sys, "argv", [sys.argv[0], *remaining, "--device", args.device]):
        (v12 if args.mode == "distill" else v10).main()


if __name__ == "__main__":
    main()
