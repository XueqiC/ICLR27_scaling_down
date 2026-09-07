#!/usr/bin/env python3
"""V27: conditioning losses, surface-format sensitivity, and exact byte units.

MAIN: mean per-example L_full = -log p(r,y|x) / target tokens.
AUXILIARY: L_direct = -log p(y|x) / |y|; L_given = -log p(y|x,r) / |y|;
paired surface-format sensitivity; corpus token/byte NLL; legacy V6 loss.
L_given is answer GIVEN reference reasoning, not a direct-answer loss.

Uses V6 zero-shot prompts and odd measurement indices, including its secondary
benchmarks. Complete r/y spans are encoded separately with no target BOS/EOS;
overlength pairs are excluded explicitly, never truncated into a different
conditioning question. The legacy V6 endpoint is retained separately.

Examples (GPU measurements are launched separately):
  python analysis/v27_scoring_and_units.py --model gemma3-270m --dry-run
  python analysis/v27_scoring_and_units.py --model gemma3-270m --quant-bits 4
  python analysis/v27_scoring_and_units.py --model gemma3-270m \
      --adapter PATH/adapter --mode format-control --format-conditioning L_given
  python analysis/v27_scoring_and_units.py --mode recompute \
      --reuse-results results/v23-loss-validity/MODEL/CELL/loss_validity.json

Recomputation preserves the entire input payload and appends units_v27_v1 to
item groups in a new output file. Aggregate-only inputs are refused. V23 item
NLL is reused; missing bytes require exact probe/hash/token-count verification
and a tokenizer, never a model. Dry runs load neither datasets nor tokenizers
nor models and create no files. Default output: results/v27-scoring-units/.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

try:
    from .v6_capability_geometry import (
        apply_global_magnitude_pruning, load_text_causal_lm, model_output_tag,
        require_compliant,
    )
    from .v12_distill import tokenize_sft_example, write_json_atomic
    from .v15_accuracy_link import _apply_quantization, _load_adapter
    from .v23_loss_validity import (
        BENCHMARKS, NAMES, _digest, measurement_probes, resolve_source,
    )
except ImportError:
    from v6_capability_geometry import (
        apply_global_magnitude_pruning, load_text_causal_lm, model_output_tag,
        require_compliant,
    )
    from v12_distill import tokenize_sft_example, write_json_atomic
    from v15_accuracy_link import _apply_quantization, _load_adapter
    from v23_loss_validity import (
        BENCHMARKS, NAMES, _digest, measurement_probes, resolve_source,
    )

OUT_BASE = Path(__file__).resolve().parents[1] / "results/v27-scoring-units"
VERSION_FIELD = "units_v27_v1"
LOSSES = ("L_full", "L_direct", "L_given")
METRICS = {
    "MAIN": {"name": "L_full", "definition": "mean_i[-log p(r_i,y_i|x_i) / n_target_tokens_i]", "units": "nats/token"},
    "AUXILIARY": {
        "L_direct": "mean_i[-log p(y_i|x_i) / n_y_tokens_i]: direct answer, reasoning absent",
        "L_given": "mean_i[-log p(y_i|x_i,r_i) / n_y_tokens_i]: answer GIVEN correct reference reasoning",
        "format_control": "paired loss differences and within-item range, fixed conditioning/content and surface-only target wrappers",
        "legacy_v6": "original completion, half-length prompt/target truncation; both example and corpus token means retained",
        VERSION_FIELD: "token_normalized=sum_i NLL_i/sum_i target_tokens_i; byte_normalized=sum_i NLL_i/sum_i UTF8_target_bytes_i; example_token_normalized=mean_i NLL_i/n_i",
    },
    "sign": "NLL is negative total target log probability; positive loss in nats (natural logarithm)",
    "comparability": "Benchmarks remain separate within capability; all three conditioning losses use the same eligible items. Format controls use a separately matched cohort. No average across different conditioning or benchmark groups.",
}
SUPPORT = {
    "math": "MATH-500: r before final balanced boxed answer, y is boxed answer plus punctuation; unsupported/mismatched/nonterminal boxes excluded",
    "math_gsm8k": "GSM8K: r before final #### delimiter, y includes #### and final answer",
    "code": "MBPP: y is the entire reference code; no supplied separate reasoning, r empty, all three losses identical",
    "code_humaneval": "HumanEval: y is canonical code continuation; no supplied reasoning, r empty, all three losses identical",
    "qa": "2WikiMultihopQA: y is reference answer, r empty; supplied context stays in x, all three losses identical",
    "qa_hotpotqa": "HotpotQA: y is reference answer, r empty; supplied context stays in x, all three losses identical",
}
FORMAT_PREFIXES = {
    "canonical": "", "newline": "\n", "blank_line": "\n\n",
    "answer_prefix": "Answer: ", "equivalent_prefix": "The answer is ",
}


def decompose_probe(key: str, sample: dict) -> dict:
    """Retain the original reference text exactly as r+y; never invent CoT."""
    completion = sample["completion"]
    if not completion.strip():
        return {"supported": False, "reason": "empty reference"}
    if key == "math":
        matches = list(re.finditer(r"\\boxed\s*\{", completion))
        if not matches:
            return {"supported": False, "reason": "no boxed final answer"}
        match = matches[-1]
        depth, end = 1, match.end()
        while end < len(completion) and depth:
            # Escaped braces are LaTeX literals, not nesting delimiters.
            backslashes = len(completion[:end]) - len(completion[:end].rstrip("\\"))
            if backslashes % 2 == 0:
                depth += (completion[end] == "{") - (completion[end] == "}")
            end += 1
        if depth:
            return {"supported": False, "reason": "unbalanced final box"}
        answer = completion[match.end():end - 1]
        if re.sub(r"\s+", "", answer) != re.sub(r"\s+", "", str(sample["answer"])):
            return {"supported": False, "reason": "boxed answer differs from reference answer"}
        if not re.fullmatch(r"[\s.$,!;:]*", completion[end:]):
            return {"supported": False, "reason": "information after final box; cannot isolate final answer safely"}
        split = match.start()
    elif key == "math_gsm8k":
        split = completion.rfind("####")
        if split < 0 or completion[split + 4:].strip() != str(sample["answer"]).strip():
            return {"supported": False, "reason": "missing/mismatched #### final answer"}
    elif key in SUPPORT:
        split = 0
    else:
        return {"supported": False, "reason": "no registered decomposition"}
    reasoning, answer = completion[:split], completion[split:]
    return {"supported": True, "r": reasoning, "y": answer,
            "reasoning_available": bool(reasoning.strip()),
            "empty_reasoning_identity": not bool(reasoning),
            "reconstruction_exact": reasoning + answer == completion}


def encode(tokenizer, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return list(ids)


def prompt_ids(tokenizer, prompt: str, max_len: int) -> list[int]:
    # Use the exact V6 prompt truncation and Gemma BOS guard.
    example = tokenize_sft_example(tokenizer, prompt, "", max_len)
    ids = example["input_ids"].tolist()
    if not ids:
        raise ValueError("Prompt must supply a conditioning token")
    return ids


def prepare_ranges(tokenizer, sample: dict, decomposition: dict, max_len: int) -> dict:
    p = prompt_ids(tokenizer, sample["prompt"], max_len)
    r = encode(tokenizer, decomposition["r"])
    y = encode(tokenizer, decomposition["y"])
    if not y:
        raise ValueError("empty answer token span")
    if len(p) + len(r) + len(y) > max_len:
        raise ValueError("complete reasoning+answer exceeds max_len; all conditioning comparisons excluded")
    return {
        "L_full": (p, r + y, decomposition["r"] + decomposition["y"]),
        "L_direct": (p, y, decomposition["y"]),
        "L_given": (p + r, y, decomposition["y"]),
    }


def score_ids(model, context: list[int], target: list[int], device: str) -> float:
    """Mask context explicitly; the first target is predicted by its last token."""
    if not context or not target:
        raise ValueError("Scoring requires nonempty context and target tokens")
    ids = torch.tensor([context + target], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits
        # Sum in fp32 even for bf16 checkpoints.
        prediction = logits[:, len(context) - 1:-1].float()
        total = F.cross_entropy(prediction.transpose(1, 2), ids[:, len(context):], reduction="sum")
    result = float(total)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Nonfinite or negative target NLL")
    return result


def target_byte_count(tokenizer, text: str, scored_ids: list[int]) -> int:
    """Count original UTF-8 bytes of exactly the scored prefix, or refuse.

    In particular, do not divide truncated NLL by the bytes of a full reference
    or assign full Unicode-character bytes to a partial byte-fallback token.
    """
    all_ids = encode(tokenizer, text)
    if scored_ids == all_ids:
        return len(text.encode("utf-8"))
    if not scored_ids or all_ids[:len(scored_ids)] != scored_ids:
        raise ValueError("target is not the expected token prefix")
    try:
        encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoding["offset_mapping"]
        end = int(offsets[len(scored_ids) - 1][1])
        if int(offsets[len(scored_ids)][0]) < end:
            raise ValueError("truncation splits a Unicode character")
        prefix = text[:end]
        if encode(tokenizer, prefix) != scored_ids:
            raise ValueError("offset prefix does not round-trip to scored tokens")
    except (NotImplementedError, TypeError, KeyError):
        prefix = tokenizer.decode(scored_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if not text.startswith(prefix) or encode(tokenizer, prefix) != scored_ids:
            raise ValueError("cannot reconstruct exact target bytes")
    return len(prefix.encode("utf-8"))


def recompute_units(items: list[dict]) -> dict:
    """A ratio of totals requires example-level numerators and denominators."""
    if not items:
        raise ValueError("Per-example NLL and lengths required; aggregate conversion is forbidden")
    totals, tokens, byte_counts = [], [], []
    for item in items:
        total = item.get("sum_ce", item.get("total_nll"))
        count = item.get("n_tokens")
        if total is None and "loss" in item and count is not None:
            total = float(item["loss"]) * count  # per-example mean, never aggregate
        if total is None or type(count) is not int or count <= 0 or not math.isfinite(total) or total < 0:
            raise ValueError("Each item needs finite nonnegative total NLL and positive integer n_tokens")
        byte_count = item.get("target_bytes")
        if type(byte_count) is not int or byte_count <= 0:
            raise ValueError("Each item needs exact positive target_bytes; aggregate conversion is forbidden")
        totals.append(float(total))
        tokens.append(count)
        byte_counts.append(byte_count)
    return {"version": "v27-units-v1", "log_base": "e", "n_examples": len(items),
            "sum_nll": sum(totals), "sum_target_tokens": sum(tokens),
            "sum_target_bytes": sum(byte_counts),
            "token_normalized": sum(totals) / sum(tokens), "token_units": "nats/token",
            "byte_normalized": sum(totals) / sum(byte_counts), "byte_units": "nats/UTF8-byte",
            "example_token_normalized": float(np.mean([n / t for n, t in zip(totals, tokens)]))}


def summarize(items: list[dict]) -> dict:
    if not items:
        return {"status": "no eligible probes", "items": [], "L_c": None, VERSION_FIELD: None}
    units = recompute_units(items)
    return {"status": "ok", "L_c": units["example_token_normalized"],
            "original_units": "nats/token (mean per example)",
            "token_weighted_L_c": units["token_normalized"], VERSION_FIELD: units, "items": items}


def _item(model, context, target, text, device, identity) -> dict:
    return {**identity, "sum_ce": score_ids(model, context, target, device),
            "n_tokens": len(target), "target_bytes": len(text.encode("utf-8")),
            "target_sha256": _digest(text)}


def measure_benchmarks(model, tokenizer, probes: dict, device: str, max_len: int,
                       mode: str = "all", format_conditioning: str = "L_full") -> dict:
    model.eval()
    measured = {}
    for key, samples in probes.items():
        scoring = {name: [] for name in LOSSES}
        formats = {name: [] for name in FORMAT_PREFIXES}
        legacy, support, excluded, format_excluded = [], [], [], []
        for index, sample in enumerate(samples):
            identity = {"measurement_index": index, "v6_probe_index": 2 * index + 1,
                        "probe_sha256": _digest(sample)}
            decomposition = decompose_probe(key, sample)
            support.append({**identity, **decomposition})
            if mode in ("all", "scoring"):
                example = tokenize_sft_example(tokenizer, sample["prompt"], sample["completion"], max_len)
                ids = example["input_ids"].tolist()
                n = int(example["n_completion_tokens"])
                if n <= 0:
                    raise ValueError(f"{key} item {index}: empty legacy target")
                byte_count = target_byte_count(tokenizer, sample["completion"], ids[-n:])
                legacy.append({**identity, "sum_ce": score_ids(model, ids[:-n], ids[-n:], device),
                               "n_tokens": n, "target_bytes": byte_count})
            if not decomposition["supported"]:
                excluded.append({**identity, "reason": decomposition["reason"]})
                continue
            try:
                ranges = prepare_ranges(tokenizer, sample, decomposition, max_len)
            except ValueError as exc:
                excluded.append({**identity, "reason": str(exc)})
                continue
            if mode in ("all", "scoring"):
                cache = {}
                for name, (context, target, text) in ranges.items():
                    cache_key = (tuple(context), tuple(target))
                    if cache_key not in cache:
                        cache[cache_key] = _item(model, context, target, text, device, identity)
                    scoring[name].append(dict(cache[cache_key]))
            if mode in ("all", "format-control"):
                context, canonical_ids, content = ranges[format_conditioning]
                # Prefixes wrap the identical body, leaving all internal code
                # whitespace, reasoning, answer values and context unchanged.
                variants = {name: (encode(tokenizer, prefix) + canonical_ids, prefix + content)
                            for name, prefix in FORMAT_PREFIXES.items()}
                if any(len(context) + len(ids) > max_len for ids, _ in variants.values()):
                    format_excluded.append({**identity, "reason": "at least one surface variant exceeds max_len; exclude item from every format"})
                    continue
                for name, (ids, text) in variants.items():
                    formats[name].append(_item(model, context, ids, text, device,
                                               {**identity, "information_sha256": _digest(content),
                                                "context_tokens_sha256": _digest(context)}))
        result = {"capability": key.split("_")[0], "benchmark": NAMES[key],
                  "dataset": BENCHMARKS[key], "n_measurement_probes": len(samples),
                  "decomposition_policy": SUPPORT[key], "probe_decompositions": support,
                  "conditioning_exclusions": excluded}
        if mode in ("all", "scoring"):
            result["scoring"] = {name: summarize(items) for name, items in scoring.items()}
            result["legacy_v6"] = summarize(legacy)
        if mode in ("all", "format-control"):
            paired = []
            for index, canonical in enumerate(formats["canonical"]):
                values = {name: items[index]["sum_ce"] / items[index]["n_tokens"] for name, items in formats.items()}
                paired.append({"measurement_index": canonical["measurement_index"],
                               "range_nats_per_token": max(values.values()) - min(values.values()),
                               "delta_from_canonical": {name: value - values["canonical"] for name, value in values.items()}})
            result["format_control"] = {
                "conditioning": format_conditioning, "surface_prefixes": FORMAT_PREFIXES,
                "fixed_information": "identical reference body and context; only outer newlines/semantically equivalent prefixes vary; code indentation retained",
                "variants": {name: summarize(items) for name, items in formats.items()},
                "paired_items": paired, "format_exclusions": format_excluded,
                "mean_within_item_range_nats_per_token": float(np.mean([p["range_nats_per_token"] for p in paired])) if paired else None,
            }
        measured[key] = result
        print(f"[measure] {key}: {len(samples)} probes, {len(excluded)} conditioning exclusions", flush=True)
    return measured


def _item_groups(value, path=()):
    if isinstance(value, dict):
        if isinstance(value.get("items"), list) and value["items"]:
            yield path, value
        for key, child in value.items():
            if key != "items":
                yield from _item_groups(child, (*path, key))


def reuse_result(input_path: Path, output_base: Path, dry_run: bool = False) -> Path:
    """Read-only V23/V27 per-example reuse, with original units/fields intact."""
    original = json.loads(input_path.read_text(encoding="utf-8"))
    for field in ("model", "resolved_model", "source_checkpoint", "tokenizer"):
        if original.get(field):
            require_compliant(require_compliant(str(original[field])))
    groups = list(_item_groups(original))
    if not groups:
        raise ValueError("No per-example records; cannot convert aggregate means to bytes")
    needs_bytes = any("target_bytes" not in item for _, group in groups for item in group["items"])
    if needs_bytes and original.get("version") != 23:
        raise ValueError("Missing target_bytes: only V23 has a registered exact reconstruction protocol")
    path_hash = hashlib.sha256(str(input_path.resolve()).encode()).hexdigest()[:12]
    output = output_base / "recomputed" / f"{input_path.stem}-{path_hash}.json"
    if input_path.resolve() == output.resolve():
        raise ValueError("Recomputation output must not overwrite its input")
    if dry_run:
        print(json.dumps({"input": str(input_path), "output": str(output),
                          "item_groups": len(groups), "requires_tokenizer_and_probe_verification": needs_bytes,
                          "requires_model": False}, indent=2))
        return output
    if output.exists():
        raise FileExistsError(output)
    payload = copy.deepcopy(original)
    tokenizer, probes = None, None
    if needs_bytes:
        from transformers import AutoTokenizer
        protocol = original["protocol"]
        if (protocol.get("probe_half") != "measurement (odd indices, v[1::2])"
                or protocol.get("chat_template") is not False
                or protocol.get("prompt_max_tokens") != protocol["max_len"] // 2
                or protocol.get("target_max_tokens") != protocol["max_len"] // 2):
            raise ValueError("Unsupported V23 scoring/truncation protocol")
        probes = measurement_probes(protocol["n_probe_requested"], protocol["probe_seed"])
        if _digest(probes) != protocol["probe_sha256"]:
            raise ValueError("V23 probe hash mismatch; cannot reconstruct byte denominators")
        tokenizer = AutoTokenizer.from_pretrained(require_compliant(original["tokenizer"]))
        if getattr(tokenizer, "truncation_side", "right") != "right":
            raise ValueError("Only right-truncated V6 targets can be reconstructed")
    for path, group in _item_groups(payload):
        items = copy.deepcopy(group["items"])
        for item in items:
            if "target_bytes" in item:
                continue
            key = next((k for k in BENCHMARKS if BENCHMARKS[k][0] == group.get("dataset")), None)
            if key is None:
                raise ValueError(f"Unknown probe dataset in {path}")
            sample = probes[key][item["measurement_index"]]
            if _digest(sample) != item["probe_sha256"]:
                raise ValueError("Per-example probe hash mismatch")
            example = tokenize_sft_example(tokenizer, sample["prompt"], sample["completion"], original["protocol"]["max_len"])
            if example["n_completion_tokens"] != item["n_tokens"]:
                raise ValueError("Stored token count differs from reconstructed target; refusing conversion")
            ids = example["input_ids"].tolist()[-item["n_tokens"]:]
            item["target_bytes"] = target_byte_count(tokenizer, sample["completion"], ids)
        group[VERSION_FIELD] = {**recompute_units(items), "per_example_denominators": [
            {key: item[key] for key in ("measurement_index", "n_tokens", "target_bytes") if key in item}
            for item in items]}
    payload["v27_recomputation"] = {"version": 27, "metric_definitions": METRICS,
                                    "input": str(input_path.resolve()), "input_sha256": _digest(original),
                                    "measurement_scope": "units of the stored scoring endpoint only; no new conditioning losses or format controls inferred",
                                    "nll_source": "stored per-example NLL; no model inference",
                                    "byte_source": "stored exact bytes or verified V23 target reconstruction",
                                    "original_fields_and_units_preserved": True}
    write_json_atomic(output, payload)
    return output


def run_measurement(*, model_request: str, checkpoint=None, adapter=None,
                    prune_density=None, quant_bits=None, n_probe=128, probe_seed=0,
                    max_len=1024, device="cuda:0", mode="all", format_conditioning="L_full",
                    output_base=OUT_BASE, dry_run=False) -> Path:
    if prune_density is not None and not 0 < prune_density <= 1:
        raise ValueError("--prune-density must be in (0, 1]")
    if quant_bits is not None and (type(quant_bits) is not int or not 2 <= quant_bits <= 16):
        raise ValueError("--quant-bits must be an integer in [2, 16]")
    if n_probe < 2 or max_len < 2 or not 0 <= probe_seed < 2**32:
        raise ValueError("n_probe/max_len must be >=2 and probe_seed in [0, 2**32)")
    if mode not in ("all", "scoring", "format-control") or format_conditioning not in LOSSES:
        raise ValueError("Unknown measurement mode or format conditioning")
    resolved, source, label = resolve_source(model_request, checkpoint, adapter)
    if checkpoint or adapter is not None:
        # Different trajectories often have the same update-NNNNNNNN basename.
        # Keep their checkpoints distinct across data pools and training seeds.
        label += "-" + _digest({"source": source,
                                "adapter": str(adapter.resolve()) if adapter else None})[:12]
    if prune_density is not None:
        label += f"_prune-d{prune_density}"
    if quant_bits is not None:
        label += f"_quant-b{quant_bits}"
    tag = model_output_tag(model_request, resolved)
    output = Path(output_base) / tag / label / f"{mode}-{format_conditioning}-scoring-units.json"
    protocol = {"probe_source": "V6 build_probes(include_secondary=True)",
                "probe_half": "measurement (odd indices, v[1::2])", "probe_seed": probe_seed,
                "n_probe_requested": n_probe, "max_len": max_len,
                "prompt_max_tokens": max_len // 2, "target_policy": "complete r+y must fit remaining context; paired exclusions",
                "tokenization": "V6 prompt/BOS, no chat template; separately encode r and y without special tokens; exact same y IDs for direct/given/full",
                "legacy_v6_max_len": max_len, "legacy_target_max_tokens": max_len // 2,
                "format_conditioning": format_conditioning,
                "format_policy": "identical body tokens; prefixes encoded separately without special tokens",
                "byte_encoding": "UTF-8", "compression_order": "adapter, magnitude pruning, V15 fake quantization",
                "dtype": "bfloat16"}
    print(json.dumps({"metric_definitions": METRICS, "protocol": protocol,
                      "decomposition_support": SUPPORT, "output": str(output),
                      "source_checkpoint": source, "adapter": str(adapter) if adapter else None,
                      "dry_run": dry_run}, indent=2), flush=True)
    if dry_run:
        return output
    if output.exists():
        raise FileExistsError(output)
    probes = measurement_probes(n_probe, probe_seed)
    model, tokenizer = load_text_causal_lm(source, torch.bfloat16)
    try:
        if getattr(tokenizer, "truncation_side", "right") != "right":
            raise ValueError("V27 requires right truncation for legacy V6 target bytes")
        if adapter is not None:
            model = _load_adapter(model, adapter, resolved)
        model.to(device).eval()
        threshold = apply_global_magnitude_pruning(model, prune_density, seed=0) if prune_density is not None else None
        if quant_bits is not None:
            _apply_quantization(model, quant_bits)
        measured = measure_benchmarks(model, tokenizer, probes, device, max_len, mode, format_conditioning)
        payload = {"version": 27, "metric_definitions": METRICS, "protocol": protocol,
                   "model": model_request, "resolved_model": resolved, "source_checkpoint": source,
                   "adapter": str(adapter.resolve()) if adapter else None,
                   "prune_density": prune_density, "pruning_threshold": threshold,
                   "quant_bits": quant_bits, "mode": mode,
                   "tokenizer": str(getattr(tokenizer, "name_or_path", source)),
                   "probe_sha256": _digest(probes), "benchmarks": measured}
        write_json_atomic(output, payload)
    finally:
        del model, tokenizer
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint")
    source.add_argument("--adapter", type=Path)
    parser.add_argument("--prune-density", type=float)
    parser.add_argument("--quant-bits", type=int)
    parser.add_argument("--n-probe", type=int, default=128)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("all", "scoring", "format-control", "recompute"), default="all")
    parser.add_argument("--format-conditioning", choices=LOSSES, default="L_full")
    parser.add_argument("--reuse-results", type=Path, nargs="+")
    parser.add_argument("--output-base", type=Path, default=OUT_BASE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "recompute":
        if not args.reuse_results:
            parser.error("--mode recompute requires --reuse-results")
        if args.model or args.checkpoint or args.adapter or args.prune_density is not None or args.quant_bits is not None:
            parser.error("recompute uses model/compression metadata from its input files")
        print(json.dumps({"metric_definitions": METRICS}, indent=2))
        for path in args.reuse_results:
            print(reuse_result(path, args.output_base, args.dry_run))
    else:
        if not args.model or args.reuse_results:
            parser.error("measurement requires --model; --reuse-results belongs to --mode recompute")
        kwargs = vars(args).copy()
        kwargs["model_request"] = kwargs.pop("model")
        kwargs.pop("reuse_results")
        print(run_measurement(**kwargs))


if __name__ == "__main__":
    main()
