#!/usr/bin/env python3
"""V6: capability geometry pilot — spectrum-based zero-fit damage prediction.

Pipeline per model:
  1. Build capability probe sets (uncontaminated halves):
     math=MATH-500, code=MBPP, qa=2WikiMultihopQA. n samples each,
     prompt + reference completion.
  2. Diagonal capability Fisher: F_c[i] = mean over probe samples of
     grad_i(CE on reference tokens)^2, at the dense checkpoint w0.
  3. Capability spectra: s_c = F_c * w0^2 in global magnitude-pruning
     order -> predicted damage D_c(d) = cumulative s_c mass pruned at
     density d (zero free parameters). Also tr(F_c) for the int8/int4
     prediction and pairwise spectrum overlaps.
  4. Generate magnitude-pruned checkpoints at the prior grid densities,
     measure per-capability CE loss (and optionally item accuracy) ->
     empirical Delta L_c(d).
  5. Compare: rank/shape agreement of D_c(d) vs Delta L_c(d); is the
     transfer curve phi shared across capabilities (T1) or split (T2)?

Usage:
  python3 v6_capability_geometry.py --model gemma3-4b \
      --device cuda:0 --n-probe 128 --stage all
Stages: fisher | prune | report | all. Intermediate artifacts under
results/v6-capability-geometry/<model_tag>/.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

try:
    from .model_registry import MODEL_REGISTRY, require_compliant
except ImportError:  # direct execution: python analysis/v6_capability_geometry.py
    from model_registry import MODEL_REGISTRY, require_compliant

ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "results/v6-capability-geometry"

# densities matching the prior src8b ladder plus cliff region
DENSITIES = [0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3]
PRUNE_THRESHOLD_SAMPLE_SIZE = 2_000_000
SANITY_PROMPT = "The capital of France is"
SANITY_CE_LIMIT = 12.0
_TIED_WEIGHT_MISSING_ALLOWLIST = frozenset({"lm_head.weight"})


def model_output_tag(requested: str, resolved: str) -> str:
    """Return a safe single-directory tag for a registry tag or raw HF id."""
    source = requested if requested in MODEL_REGISTRY else resolved
    tag = re.sub(r"[^A-Za-z0-9._-]+", "--", source).strip(".-_")
    return tag or "model"


class _TextCausalLMAdapter(torch.nn.Module):
    """Language-only view of a multimodal conditional-generation model."""

    def __init__(self, language_model, lm_head, text_config) -> None:
        super().__init__()
        self.language_model = language_model
        self.lm_head = lm_head
        self.config = text_config

    def forward(self, input_ids: torch.Tensor, **kwargs):
        kwargs.pop("logits_to_keep", None)
        outputs = self.language_model(input_ids=input_ids, **kwargs)
        hidden = (outputs.last_hidden_state
                  if hasattr(outputs, "last_hidden_state") else outputs[0])
        logits = self.lm_head(hidden)
        multiplier = getattr(self.config, "output_multiplier", None)
        if multiplier is not None:
            logits = logits * multiplier
        softcap = getattr(self.config, "final_logit_softcapping", None)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        return SimpleNamespace(
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
        )


def _language_only_view(model):
    """Drop vision/audio towers from a conditional-generation checkpoint."""
    container = getattr(model, "model", None)
    language_model = getattr(container, "language_model", None)
    lm_head = getattr(model, "lm_head", None)
    if language_model is None or lm_head is None:
        return model
    text_config = getattr(model.config, "text_config", language_model.config)
    return _TextCausalLMAdapter(language_model, lm_head, text_config)


def _loading_info_error(model_name: str, loading_info: dict,
                        config) -> RuntimeError | None:
    """Build a hard checkpoint-integrity error for unloaded parameters."""
    missing = list(loading_info.get("missing_keys") or [])
    mismatched = list(loading_info.get("mismatched_keys") or [])
    text_config = getattr(config, "text_config", None)
    tied = (bool(getattr(config, "tie_word_embeddings", False))
            or bool(getattr(text_config, "tie_word_embeddings", False)))
    allowed_missing = _TIED_WEIGHT_MISSING_ALLOWLIST if tied else frozenset()
    disallowed_missing = [key for key in missing if key not in allowed_missing]
    if not disallowed_missing and not mismatched:
        return None

    def examples(values: list) -> str:
        return ", ".join(repr(value) for value in values[:5]) or "none"

    return RuntimeError(
        f"Checkpoint-integrity failure for {model_name!r}: "
        f"missing_keys={len(missing)} "
        f"(disallowed={len(disallowed_missing)}, examples: "
        f"{examples(disallowed_missing)}, "
        f"allowed_tied={len(missing) - len(disallowed_missing)}); "
        f"mismatched_keys={len(mismatched)} "
        f"(examples: {examples(mismatched)}). Refusing to analyze a model "
        "whose checkpoint weights were not loaded exactly."
    )


def _raise_for_loading_info(model_name: str, loading_info: dict,
                            config) -> None:
    error = _loading_info_error(model_name, loading_info, config)
    if error is not None:
        raise error


def _checkpoint_sanity_forward(model, tok, model_name: str) -> float:
    """Return tiny-prompt next-token CE, rejecting random-level models."""
    encoded = tok(SANITY_PROMPT, return_tensors="pt")
    input_ids = encoded.input_ids
    if input_ids.shape[1] < 2:
        raise RuntimeError(
            f"Checkpoint-integrity failure for {model_name!r}: sanity "
            f"prompt tokenized to only {input_ids.shape[1]} token(s)."
        )
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    input_ids = input_ids.to(model_device)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits
        ce = F.cross_entropy(
            logits[:, :-1].float().transpose(1, 2), input_ids[:, 1:],
            reduction="mean",
        )
    mean_ce = float(ce)
    if not np.isfinite(mean_ce) or mean_ce > SANITY_CE_LIMIT:
        raise RuntimeError(
            f"Checkpoint-integrity failure for {model_name!r}: sanity "
            f"forward mean next-token CE is {mean_ce:.3f} nats on "
            f"{SANITY_PROMPT!r}, above the random-level guard of "
            f"{SANITY_CE_LIMIT:.1f}."
        )
    return mean_ce


def load_text_causal_lm(model_name: str, dtype: torch.dtype):
    """Load only the causal text LM, including from multimodal checkpoints.

    AutoModelForCausalLM gets first chance with the checkpoint's text config.
    Families that only expose a ConditionalGeneration class are loaded through
    the matching multimodal auto class and immediately reduced to their
    language-model component plus LM head.
    """
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(model_name)
    text_config = getattr(config, "text_config", config)
    load_kwargs = {"config": text_config, "dtype": dtype,
                   "low_cpu_mem_usage": True,
                   "output_loading_info": True}
    first_error: Exception | None = None
    try:
        model, loading_info = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs)
        unexpected = list(loading_info.get("unexpected_keys") or [])
        if len(unexpected) > 10:
            first_error = RuntimeError(
                f"Primary text loader returned {len(unexpected)} "
                f"unexpected checkpoint keys (examples: "
                f"{unexpected[:5]!r}); retrying with a multimodal loader."
            )
            del model
            model = None
            gc.collect()
        else:
            _raise_for_loading_info(model_name, loading_info, text_config)
    except (KeyError, ValueError) as exc:
        first_error = exc
        model = None

    if model is None:
        fallback_kwargs = {"config": config, "dtype": dtype,
                           "low_cpu_mem_usage": True,
                           "output_loading_info": True}
        loaders = []
        for auto_name in ("AutoModelForMultimodalLM",
                          "AutoModelForImageTextToText",
                          "AutoModelForVision2Seq"):
            loader = getattr(transformers, auto_name, None)
            if loader is not None and loader not in loaders:
                loaders.append(loader)
        architecture = ((getattr(config, "architectures", None) or [None])[0])
        architecture_cls = (getattr(transformers, architecture, None)
                            if architecture else None)
        if architecture_cls is not None and architecture_cls not in loaders:
            loaders.append(architecture_cls)
        fallback_errors = []
        for loader in loaders:
            try:
                model, loading_info = loader.from_pretrained(
                    model_name, **fallback_kwargs)
                # Multimodal loaders may legitimately leave checkpoint-only
                # vision/audio keys unexpected. Missing or shape-mismatched
                # model parameters are never safe to accept.
                _raise_for_loading_info(model_name, loading_info, config)
                break
            except (KeyError, ValueError) as exc:
                fallback_errors.append(exc)
        if model is None:
            detail = fallback_errors[-1] if fallback_errors else first_error
            raise RuntimeError(
                f"Could not load the text LM for {model_name!r}; install a "
                f"Transformers version supporting its architecture: {detail}"
            ) from detail

    model = _language_only_view(model)
    if any(re.search(r"vision|visual|perception|audio|projector", name,
                     re.IGNORECASE)
           for name, _ in model.named_parameters()):
        raise RuntimeError(
            f"Refusing to analyze {model_name!r}: a non-language tower "
            "remained in the Fisher/pruning parameter scope."
        )
    tok = AutoTokenizer.from_pretrained(model_name)
    _checkpoint_sanity_forward(model, tok, model_name)
    gc.collect()
    return model, tok


def _sample_abs_weights(
    params: list[tuple[str, torch.Tensor]],
    sample_size: int = PRUNE_THRESHOLD_SAMPLE_SIZE,
    seed: int = 0,
) -> np.ndarray:
    """Take a deterministic, proportional sample of the global |w| values."""
    n_tot = sum(p.numel() for _, p in params)
    if n_tot == 0:
        raise ValueError("Cannot sample pruning thresholds without weights.")
    rng = np.random.default_rng(seed)
    sample = []
    for _, p in params:
        flat = p.detach().reshape(-1)
        k = max(int(sample_size * p.numel() / n_tot), 100)
        idx = torch.from_numpy(
            rng.integers(0, flat.numel(), size=min(k, flat.numel())))
        sample.append(flat[idx.to(flat.device)].abs().float().cpu())
    return torch.cat(sample).numpy()


def language_weight_parameters(model) -> list[tuple[str, torch.Tensor]]:
    """Weight matrices in the language model (embeddings and LM head included)."""
    return [(name, param) for name, param in model.named_parameters()
            if param.dim() >= 2]


def apply_global_magnitude_pruning(
    model,
    density: float,
    *,
    seed: int = 0,
    reference_weights: list[torch.Tensor] | None = None,
    threshold: float | None = None,
) -> float:
    """Apply V6 sampled-threshold global magnitude pruning in place.

    ``density`` is the retained weight fraction.  Optional reference weights
    let callers apply several densities to the same dense checkpoint, as the
    V6 pruning ladder does.  Supplying its precomputed threshold avoids
    resampling that checkpoint for every rung.
    """
    if not 0.0 < density <= 1.0:
        raise ValueError("prune density must be in (0, 1]")
    parameters = language_weight_parameters(model)
    if not parameters:
        raise ValueError("No language weight matrices found for pruning")
    if reference_weights is None:
        reference_weights = [parameter.detach() for _, parameter in parameters]
    if len(reference_weights) != len(parameters):
        raise ValueError("reference weights do not match pruning parameters")
    if any(reference.shape != parameter.shape
           for (_, parameter), reference in zip(parameters, reference_weights)):
        raise ValueError("reference weight shapes do not match pruning parameters")

    if density == 1.0:
        pruning_threshold = -math.inf
    elif threshold is None:
        sample_parameters = [
            (name, reference)
            for (name, _), reference in zip(parameters, reference_weights)
        ]
        absolute_sample = _sample_abs_weights(sample_parameters, seed=seed)
        pruning_threshold = float(np.quantile(absolute_sample, 1.0 - density))
    else:
        pruning_threshold = float(threshold)

    with torch.no_grad():
        for (_, parameter), reference in zip(parameters, reference_weights):
            parameter.copy_(reference * (reference.abs() > pruning_threshold))
    return pruning_threshold


SECONDARY_BENCHMARKS = {
    "math_gsm8k": ("openai/gsm8k", "main", "test"),
    "code_humaneval": ("openai/openai_humaneval", None, "test"),
    "qa_hotpotqa": ("hotpotqa/hotpot_qa", "distractor", "validation"),
}


def secondary_probe(key: str, row: dict) -> dict:
    """Render held-out references without generation, answer leakage or code tests.

    Math retains the entire worked solution, matching MATH-500's target span.
    HumanEval retains the canonical continuation verbatim (including indentation).
    QA uses the supplied context and a short reference answer, as in 2Wiki.
    """
    if key == "math_gsm8k":
        solution = str(row["answer"])
        if "####" not in solution:
            raise ValueError("GSM8K reference answer must contain the #### delimiter")
        answer = solution.rsplit("####", 1)[1].strip()
        if not answer:
            raise ValueError("GSM8K reference has an empty final answer")
        return {"prompt": f"Problem: {row['question']}\nSolution:",
                "completion": " " + solution, "answer": answer}
    if key == "code_humaneval":
        return {"prompt": str(row["prompt"]),
                "completion": str(row["canonical_solution"]),
                "task_id": str(row["task_id"])}
    if key == "qa_hotpotqa":
        context = row["context"]
        if isinstance(context, dict):
            pairs = zip(context.get("title", []),
                        context.get("sentences", context.get("content", [])))
        elif isinstance(context, (list, tuple)):
            pairs = context
        else:
            raise ValueError("HotpotQA context must contain title/sentence pairs")
        parts = []
        for title, sentences in pairs:
            body = (" ".join(str(s) for s in sentences)
                    if isinstance(sentences, (list, tuple)) else str(sentences))
            parts.append(f"{title}: {body}")
        context_text = "\n".join(parts)[:4000]
        answer = str(row["answer"])
        return {"prompt": (f"Context:\n{context_text}\n\nQuestion: "
                           f"{row['question']}\nAnswer:"),
                "completion": " " + answer, "answer": answer}
    raise ValueError(f"Unknown secondary benchmark {key!r}")


def build_probes(n: int, seed: int = 0, *,
                 include_secondary: bool = False) -> dict[str, list[dict]]:
    """Build deterministic probes; consumers select calibration/measurement halves.

    Opt-in secondary keys are appended after the original math/code/qa keys.
    Their independent RNGs never change the legacy sampling sequence or prompts.
    """
    from datasets import load_dataset
    rng = np.random.default_rng(seed)
    probes: dict[str, list[dict]] = {}

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    probes["math"] = [
        {"prompt": f"Problem: {ds[int(i)]['problem']}\nSolution:",
         "completion": " " + ds[int(i)]["solution"],
         "answer": ds[int(i)]["answer"]}
        for i in idx]

    ds = load_dataset("google-research-datasets/mbpp", "full",
                      split="test")
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    probes["code"] = [
        {"prompt": (f"# Task: {ds[int(i)]['text']}\n# Write a Python "
                    f"function.\n"),
         "completion": ds[int(i)]["code"],
         "test_list": ds[int(i)].get("test_list", []),
         "test_setup_code": ds[int(i)].get("test_setup_code", ""),
         "challenge_test_list": ds[int(i)].get("challenge_test_list", [])}
        for i in idx]

    ds = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    qa = []
    for i in idx:
        row = ds[int(i)]
        ctx = row.get("context")
        if isinstance(ctx, dict):
            parts = []
            for title, sents in zip(ctx.get("title", []),
                                    ctx.get("sentences",
                                            ctx.get("content", []))):
                body = " ".join(sents) if isinstance(sents, list) else str(sents)
                parts.append(f"{title}: {body}")
            ctx_text = "\n".join(parts)[:4000]
        else:
            ctx_text = str(ctx)[:4000]
        qa.append({"prompt": (f"Context:\n{ctx_text}\n\nQuestion: "
                              f"{row['question']}\nAnswer:"),
                   "completion": " " + str(row["answer"]),
                   "answer": str(row["answer"])})
    probes["qa"] = qa
    if include_secondary:
        for key, (dataset, config, split) in SECONDARY_BENCHMARKS.items():
            ds = (load_dataset(dataset, config, split=split) if config is not None
                  else load_dataset(dataset, split=split))
            secondary_rng = np.random.default_rng(seed)
            idx = secondary_rng.choice(len(ds), size=min(n, len(ds)), replace=False)
            probes[key] = [secondary_probe(key, ds[int(i)]) for i in idx]
    return probes


def completion_loss(model, tok, prompt: str, completion: str,
                    device: str, max_len: int = 1024):
    """CE on completion tokens only. Returns (sum_loss, n_tokens) graph-
    carrying tensor when grad is enabled.

    These are base/pretrained checkpoints, so tokenization is deliberately
    direct (no chat template). Gemma tokenizers need exactly one prompt BOS;
    completion tokens never receive their own BOS or other special tokens.
    """
    p_ids = tok(prompt, return_tensors="pt", truncation=True,
                max_length=max_len // 2,
                add_special_tokens=True).input_ids
    tokenizer_name = (f"{getattr(tok, 'name_or_path', '')} "
                      f"{type(tok).__name__}").lower()
    bos_id = getattr(tok, "bos_token_id", None)
    if ("gemma" in tokenizer_name and bos_id is not None
            and (p_ids.shape[1] == 0 or int(p_ids[0, 0]) != bos_id)):
        bos = torch.tensor([[bos_id]], dtype=p_ids.dtype,
                           device=p_ids.device)
        p_ids = torch.cat([bos, p_ids], dim=1)[:, :max_len // 2]
    c_ids = tok(completion, return_tensors="pt", truncation=True,
                max_length=max_len // 2, add_special_tokens=False).input_ids
    ids = torch.cat([p_ids, c_ids], dim=1)[:, -max_len:].to(device)
    n_p = min(p_ids.shape[1], ids.shape[1] - c_ids.shape[1]) if \
        ids.shape[1] > c_ids.shape[1] else 0
    logits = model(input_ids=ids, use_cache=False).logits[:, :-1]
    targets = ids[:, 1:]
    lp = F.cross_entropy(logits.transpose(1, 2), targets,
                         reduction="none")[0]
    start = max(n_p - 1, 0)
    loss = lp[start:].sum()
    return loss, int(lp.shape[0] - start)


def stage_fisher(model_name: str, device: str, n_probe: int,
                 out: Path, fisher_device: str = "",
                 model_dtype: str = "fp32") -> None:
    fisher_device = fisher_device or device
    dtype = torch.float32 if model_dtype == "fp32" else torch.bfloat16
    model, tok = load_text_causal_lm(model_name, dtype)
    model.to(device).train()  # train mode but we only need grads
    model.requires_grad_(False)
    params = language_weight_parameters(model)
    for _, p in params:
        p.requires_grad_(True)

    # even-indexed probes estimate Fisher; odd-indexed ones are held out
    # for the prune-stage loss measurement (no shared sampling noise)
    probes = {c: v[0::2] for c, v in build_probes(n_probe).items()}
    (out / "probes.json").write_text(json.dumps(
        {k: len(v) for k, v in probes.items()}))

    n_tot = sum(p.numel() for _, p in params)

    # |w| quantile bin edges from a proportional sample (binned spectra
    # keep 32B-scale models in bounded memory: no argsort, no full sort)
    rng = np.random.default_rng(0)
    sample = []
    for _, p in params:
        flat = p.detach().reshape(-1)
        k = max(int(1e6 * p.numel() / n_tot), 100)
        idx = torch.from_numpy(
            rng.integers(0, flat.numel(), size=min(k, flat.numel())))
        sample.append(flat[idx.to(flat.device)].abs().float().cpu())
    sample = torch.cat(sample).numpy()
    n_bins = 200_000
    edges = np.quantile(sample, np.linspace(0, 1, n_bins + 1)[1:-1])
    edges = np.unique(edges).astype(np.float64)
    del sample

    def bin_index(t: torch.Tensor) -> np.ndarray:
        return np.searchsorted(edges, t.abs().float().cpu().numpy()
                               .reshape(-1))

    counts = np.zeros(len(edges) + 1, dtype=np.int64)
    for _, p in params:
        counts += np.bincount(bin_index(p.detach()),
                              minlength=len(edges) + 1)
    meta = {"model": model_name, "n_params": int(n_tot),
            "param_names": [n for n, _ in params], "counts": {},
            "n_bins": int(len(edges) + 1)}
    masses = {}

    # one capability at a time: single Fisher buffer, bin, free
    for cap, samples in probes.items():
        fisher = [torch.zeros(p.shape, device=fisher_device,
                              dtype=torch.float32) for _, p in params]
        count = 0
        for s in samples:
            model.zero_grad(set_to_none=True)
            loss, n_tok = completion_loss(model, tok, s["prompt"],
                                          s["completion"], device)
            if n_tok == 0:
                continue
            (loss / n_tok).backward()
            del loss
            with torch.no_grad():
                for buf, (_, p) in zip(fisher, params):
                    if p.grad is not None:
                        g2 = (p.grad.float() ** 2)
                        buf += g2.to(fisher_device)
                        del g2
            count += 1
        model.zero_grad(set_to_none=True)
        meta["counts"][cap] = count
        mass = np.zeros(len(edges) + 1)
        tr = 0.0
        with torch.no_grad():
            for buf, (_, p) in zip(fisher, params):
                # Reuse the CPU Fisher buffer when it already lives there;
                # this avoids another full-size fp32 allocation.
                f = buf.detach().to(device="cpu", dtype=torch.float32)
                f.div_(max(count, 1))
                tr += float(f.sum())
                weight_sq = p.detach().to(
                    device="cpu", dtype=torch.float32, copy=True)
                weight_sq.square_()
                f.mul_(weight_sq)
                spectrum = f.numpy().reshape(-1)
                mass += np.bincount(bin_index(p.detach()), weights=spectrum,
                                    minlength=len(edges) + 1)
                del f, weight_sq, spectrum
        del fisher
        if params:
            del buf
        masses[cap] = mass
        meta[f"tr_fisher_{cap}"] = tr
        gc.collect()
        print(f"[fisher] {cap}: {count} samples done", flush=True)

    np.savez_compressed(out / "spectrum_bins.npz", edges=edges,
                        counts=counts,
                        **{f"mass_{c}": m for c, m in masses.items()})
    (out / "fisher_meta.json").write_text(json.dumps(meta, indent=1))
    model.zero_grad(set_to_none=True)
    del params, model, tok
    gc.collect()
    torch.cuda.empty_cache() if device.startswith("cuda") else None


def stage_prune(model_name: str, device: str, n_probe: int,  # noqa: C901
                out: Path) -> None:
    """Measure per-capability CE loss on dense + pruned variants."""
    model, tok = load_text_causal_lm(model_name, torch.bfloat16)
    model.to(device).eval()
    probes = {c: v[1::2] for c, v in build_probes(n_probe).items()}

    params = language_weight_parameters(model)
    abs_sample = _sample_abs_weights(params)
    thresholds = {
        d: float(np.quantile(abs_sample, 1 - d)) for d in DENSITIES
    }
    del abs_sample

    def measure() -> dict[str, float]:
        res = {}
        with torch.no_grad():
            for cap, samples in probes.items():
                tot, ntok = 0.0, 0
                for s in samples:
                    loss, n = completion_loss(model, tok, s["prompt"],
                                              s["completion"], device)
                    tot += float(loss)
                    ntok += n
                res[cap] = tot / max(ntok, 1)
        return res

    results = {"1.0": measure()}
    print("[prune] dense:", results["1.0"], flush=True)
    dense_weights = [p.detach().clone() for _, p in params]

    for d in DENSITIES:
        thresh = thresholds[d]
        apply_global_magnitude_pruning(
            model,
            d,
            reference_weights=dense_weights,
            threshold=thresh,
        )
        results[str(d)] = measure()
        print(f"[prune] d={d}:", results[str(d)], flush=True)

    merged = {}
    prev_path = out / "prune_losses.json"
    if prev_path.exists():
        try:
            merged.update(json.loads(prev_path.read_text()))
        except Exception:
            pass
    merged.update(results)
    prev_path.write_text(json.dumps(merged, indent=1))
    del dense_weights, params, model, tok
    gc.collect()
    torch.cuda.empty_cache() if device.startswith("cuda") else None


def stage_report(out: Path) -> None:
    meta = json.loads((out / "fisher_meta.json").read_text())
    losses = json.loads((out / "prune_losses.json").read_text())
    caps = [c for c in ("math", "code", "qa")]
    n = meta["n_params"]
    lines = [f"# V6 report — {meta['model']}", "",
             f"params in prune scope: {n:,}", "",
             "| d | " + " | ".join(
                 f"pred D_{c} | meas ΔL_{c}" for c in caps) + " |",
             "|---|" + "---|" * (2 * len(caps))]
    binned = (out / "spectrum_bins.npz").exists()
    if binned:
        z = np.load(out / "spectrum_bins.npz")
        cum_n = np.cumsum(z["counts"])
        cum_mass = {c: np.cumsum(z[f"mass_{c}"]) for c in caps}

        def pred_at(c: str, d: float) -> float:
            k = (1 - d) * n
            j = int(np.searchsorted(cum_n, k))
            return float(cum_mass[c][min(j, len(cum_n) - 1)])
        top_share = {}
        for c in caps:
            j = int(np.searchsorted(cum_n, 0.99 * n))
            total = cum_mass[c][-1]
            top_share[c] = 1 - cum_mass[c][min(j, len(cum_n) - 1)] / total
    else:
        spectra = {c: np.load(out / f"spectrum_{c}.npy") for c in caps}
        cum = {c: np.concatenate([[0], np.cumsum(spectra[c])])
               for c in caps}

        def pred_at(c: str, d: float) -> float:
            return float(cum[c][min(int((1 - d) * n), n)])
        top_share = {c: float(np.sort(spectra[c])
                              [-max(1, int(0.01 * n)):].sum()
                              / spectra[c].sum()) for c in caps}
    dense = losses["1.0"]
    rows = []
    for d in DENSITIES:
        key = str(d)
        if key not in losses:
            continue
        row = [f"| {d}"]
        for c in caps:
            pred = pred_at(c, d)
            meas = losses[key][c] - dense[c]
            row.append(f" {pred:.3g} | {meas:+.3f}")
            rows.append({"d": d, "cap": c, "pred": pred, "meas": meas})
        lines.append(" |".join(row) + " |")
    # phi sharing check: spearman between pred and meas pooled vs per-cap
    from scipy.stats import spearmanr
    pooled = spearmanr([r["pred"] for r in rows],
                       [r["meas"] for r in rows]).statistic
    lines += ["", f"pooled Spearman(pred, meas) = {pooled:.3f}"]
    for c in caps:
        rc = [r for r in rows if r["cap"] == c]
        rho = spearmanr([r["pred"] for r in rc],
                        [r["meas"] for r in rc]).statistic
        lines.append(f"  {c}: Spearman = {rho:.3f}, tr_F = "
                     f"{meta[f'tr_fisher_{c}']:.3g}")
    lines += ["", "spectrum concentration: top-1% magnitude-coordinate "
              "mass share"]
    for c in caps:
        lines.append(f"  {c}: {top_share[c]:.1%} of Fisher-weight mass "
                     f"in top 1% coordinates")
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-270m",
                    help="registry tag or raw Hugging Face model id")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-probe", type=int, default=128)
    ap.add_argument("--densities", default="",
                    help="comma list overriding the default density grid")
    ap.add_argument("--stage", default="all",
                    choices=["fisher", "prune", "report", "all"])
    ap.add_argument("--fisher-device", default="",
                    help="device for Fisher accumulators (cpu for big models)")
    ap.add_argument("--model-dtype", default="fp32",
                    choices=["fp32", "bf16"])
    args = ap.parse_args()
    if args.densities:
        global DENSITIES
        DENSITIES = [float(x) for x in args.densities.split(",") if x]
    model_name = require_compliant(args.model)
    tag = model_output_tag(args.model, model_name)
    out = OUT_BASE / tag
    out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("fisher", "all"):
        stage_fisher(model_name, args.device, args.n_probe, out,
                     args.fisher_device, args.model_dtype)
    if args.stage in ("prune", "all"):
        stage_prune(model_name, args.device, args.n_probe, out)
    if args.stage in ("report", "all"):
        stage_report(out)


if __name__ == "__main__":
    main()
