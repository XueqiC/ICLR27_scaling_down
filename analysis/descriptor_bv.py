"""V87 final-logit B/V, using the frozen conditional scorer and aggregation.

Example (local model/tokenizer and JSON probes, CPU by default)::

    python -m analysis.descriptor_bv --model /path/to/model --revision REV \
        --probe-set probes.json --distribution source --dtype bf16 \
        --output-dir results/v87-prep/dense

Probe JSON maps capability names to lists of {prompt, completion, sample_id?}.
See descriptor_bv.schema.md for units, provenance and the shrinkage sign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch

try:
    from .eval_records import (AGGREGATION_RULE, BYTE_AGGREGATION_RULE,
                               aggregate_records, digest, record)
    from .v6_capability_geometry import completion_loss
except ImportError:
    from eval_records import (AGGREGATION_RULE, BYTE_AGGREGATION_RULE,
                              aggregate_records, digest, record)
    from v6_capability_geometry import completion_loss

ROOT = Path(__file__).resolve().parents[1]


class _CaptureFinal:
    """Keep only the current forward's native output, released after reduction.

    Capture model(...).logits, NOT lm_head output: any final softcap or other
    architectural transform has already happened here.
    """
    def __init__(self, model):
        self.model = model

    def __call__(self, **kwargs):
        output = self.model(**kwargs)
        self.logits = output.logits
        self.ids = kwargs["input_ids"]
        return output


def reduce_final_logits(logits, targets, chunk_tokens=32, with_w=False):
    """Reduce bounded token chunks; all vocabulary reductions are float32.

    Returns the summed B and V, and with with_w=True also the summed
    W = Var_p(z), the second-order sensitivity of the loss to shrinking the logits.
    The default two-value return is kept so existing callers are unaffected.
    """
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    b_sum = v_sum = w_sum = 0.0
    for start in range(0, targets.numel(), chunk_tokens):
        z = logits[start:start + chunk_tokens].float()
        y = targets[start:start + chunk_tokens]
        p = torch.softmax(z, dim=-1, dtype=torch.float32)
        pz = (p * z).sum(-1, dtype=torch.float32)
        b = z.gather(-1, y[:, None]).squeeze(-1) - pz
        v = 1.0 - p.square().sum(-1, dtype=torch.float32)
        # Python scalar accumulation avoids retaining any vocabulary tensors.
        b_sum += float(b.sum(dtype=torch.float32))
        v_sum += float(v.sum(dtype=torch.float32))
        if with_w:
            w = (p * z.square()).sum(-1, dtype=torch.float32) - pz.square()
            w_sum += float(w.clamp_min(0).sum(dtype=torch.float32))
    return (b_sum, v_sum, w_sum) if with_w else (b_sum, v_sum)


def score_sample(model, tokenizer, sample, capability, index, distribution,
                 max_len=1024, chunk_tokens=32):
    capture = _CaptureFinal(model)
    with torch.no_grad():
        loss, count = completion_loss(capture, tokenizer, sample["prompt"],
                                      sample["completion"],
                                      str(next(model.parameters()).device), max_len=max_len)
        result = record(sample, capability, index, loss, count, distribution)
        # V6 returns the size of its scored suffix. Reuse that result directly:
        # no duplicate BOS, truncation, shift or prompt-mask implementation.
        if count:
            logits = capture.logits[0, -count - 1:-1]
            targets = capture.ids[0, -count:]
            b, v, w = reduce_final_logits(logits, targets, chunk_tokens, with_w=True)
            try:
                try:
                    from .v27_scoring_and_units import target_byte_count
                except ImportError:
                    from v27_scoring_and_units import target_byte_count
                byte_count = target_byte_count(tokenizer, sample["completion"], targets.tolist())
                byte_error = None
            except (ValueError, AttributeError, NotImplementedError) as exc:
                byte_count, byte_error = None, str(exc)
        else:
            b = v = w = 0.0
            byte_count, byte_error = 0, None
    result.update({"B_sum": b, "V_sum": v, "W_sum": w,
                   "B": b / count if count else None,
                   "V": v / count if count else None,
                   "W": w / count if count else None,
                   "L": float(loss) / count if count else None,
                   "scored_reference_byte_count": byte_count,
                   "byte_count_error": byte_error,
                   "B_per_byte": b / byte_count if byte_count else None,
                   "V_per_byte": v / byte_count if byte_count else None,
                   "L_per_byte": float(loss) / byte_count if byte_count else None,
                   "forward_logits_dtype": str(capture.logits.dtype),
                   "probability_reduction_dtype": "torch.float32"})
    return result


def describe(model, tokenizer, probes, distribution, *, model_id, revision,
             max_len=1024, chunk_tokens=32, input_hashes=None, allow_gpu=False):
    # Forward passes only. The device guard exists so a preparation step cannot take a
    # shared accelerator by accident; measuring a real panel needs it lifted on purpose.
    if not allow_gpu and any(t.device.type != "cpu"
                             for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError("descriptor extraction defaults to CPU; pass allow_gpu=True to use an accelerator")
    rows = []
    training = model.training
    model.eval()
    try:
        for cap, samples in probes.items():
            for i, sample in enumerate(samples):
                rows.append(score_sample(model, tokenizer, sample, cap, i, distribution,
                                         max_len, chunk_tokens))
    finally:
        model.train(training)
    aggregates = {}
    for row in rows:
        aggregates.setdefault(row["capability"], {}).setdefault(row["distribution"], {})
    for cap, distributions in aggregates.items():
        for label, values in distributions.items():
            group = [r for r in rows if r["capability"] == cap and r["distribution"] == label]
            for name, key in (("L", "summed_nll"), ("B", "B_sum"), ("V", "V_sum"), ("W", "W_sum")):
                values[name] = aggregate_records(group, key)[cap]
                values[name + "_per_byte"] = (
                    aggregate_records(group, key, "scored_reference_byte_count")[cap]
                    if all(r["scored_reference_byte_count"] is not None for r in group)
                    and sum(r["scored_reference_byte_count"] for r in group) else None)
            values.update({"n_samples": len(group),
                           "n_scored_samples": sum(r["scored_token_count"] > 0 for r in group),
                           "scored_token_count": sum(r["scored_token_count"] for r in group),
                           "scored_reference_byte_count": (sum(r["scored_reference_byte_count"] for r in group)
                               if all(r["scored_reference_byte_count"] is not None for r in group) else None)})
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {"schema_version": 1, "model_id": model_id, "model_revision": revision,
            "resolved_model_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
            "commit": commit, "device": str(next(model.parameters()).device), "max_len": max_len,
            "chunk_tokens": chunk_tokens, "probability_reduction_dtype": "torch.float32",
            "aggregation_rule": AGGREGATION_RULE, "byte_aggregation_rule": BYTE_AGGREGATION_RULE,
            "input_hashes": {"probes_sha256": digest(probes), **(input_hashes or {})},
            "code_sha256": {name: hashlib.sha256((ROOT / "analysis" / name).read_bytes()).hexdigest()
                             for name in ("descriptor_bv.py", "eval_records.py", "v6_capability_geometry.py",
                                          "v10_quantization.py", "v27_scoring_and_units.py")},
            "logit_source": "model(input_ids, use_cache=False).logits after final architecture transforms",
            "shrinkage_identity": "B = +d CE((1-eps)*z)/d eps at eps=0; negative derivative is -B",
            "per_sample": rows, "aggregates": aggregates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--probe-set", type=Path)
    parser.add_argument("--distribution")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "float32", "bfloat16"), default="fp32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/v87-prep")
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--chunk-tokens", type=int, default=32)
    parser.add_argument("--allow-gpu", action="store_true",
                        help="permit a non-CPU device; forward passes only, never training")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if torch.device(args.device).type != "cpu" and not args.allow_gpu:
        parser.error("descriptor extraction defaults to CPU; pass --allow-gpu to use an accelerator")
    output = args.output_dir.resolve()
    allowed = (ROOT / "results/v87-prep", ROOT / "results/v91-dense-stats",
               ROOT / "results/v93-confirm-inputs")
    if output.is_relative_to(ROOT / "results") and not any(output.is_relative_to(a) for a in allowed):
        parser.error("results output must be under results/v87-prep/ or results/v91-dense-stats/")
    if args.selftest:
        try:
            from .descriptor_bv_selftest import selftest
        except ImportError:
            from descriptor_bv_selftest import selftest
        payload = selftest()
        name = "descriptor_selftest.json"
    else:
        if not all((args.model, args.revision, args.probe_set, args.distribution)):
            parser.error("--model, --revision, --probe-set and --distribution are required")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtype = torch.bfloat16 if args.dtype in ("bf16", "bfloat16") else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                  torch_dtype=dtype, local_files_only=True, device_map=None).to(
                        args.device if args.allow_gpu else "cpu")
        payload = describe(model, tokenizer, json.loads(args.probe_set.read_text()), args.distribution,
                           model_id=args.model, revision=args.revision, max_len=args.max_len,
                           chunk_tokens=args.chunk_tokens,
                           input_hashes={"probe_file_sha256": hashlib.sha256(args.probe_set.read_bytes()).hexdigest(),
                                         "tokenizer_vocab_sha256": digest(tokenizer.get_vocab())}, allow_gpu=args.allow_gpu)
        name = "descriptor_bv.json"
    # A caller may use temporary/local directories, but never another results subtree.
    output.mkdir(parents=True, exist_ok=True)
    (output / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
