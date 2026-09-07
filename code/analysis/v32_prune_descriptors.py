#!/usr/bin/env python3
"""Dense-only pruning descriptors and CPU signed-amplitude development audit.

extract loads dense weights; only --with-activations permits forward passes.
predict reads JSON only. See paper/docs/PRUNE_DESCRIPTORS.md for the protocol.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

try:
    from . import model_registry as registry
    from . import prediction_audit as audit
    from . import v28_new_source_prediction as v28
except ImportError:
    import model_registry as registry
    import prediction_audit as audit
    import v28_new_source_prediction as v28

import numpy as np

ROOT = audit.ROOT
OUT = ROOT / "results/v32-descriptors"
REPORT = ROOT / "paper/docs/PRUNE_DESCRIPTORS.md"
SCHEMA_VERSION = 1
DENSITIES = v28.PRUNE_DEV
STATS = ("mean", "variance", "entropy", "skew", "max", "min", "gini")
WEIGHT_KEYS = tuple(f"abs_weight_sample.{s}" for s in STATS) + tuple(
    f"retention.{d}.{s}" for d in DENSITIES for s in STATS)
ACTIVATION_KEYS = tuple(f"abs_activation_{moment}.{s}"
                        for moment in ("mean", "variance") for s in STATS)
VERSIONS = ("V1", "V2", "V3")
TARGET = "Qwen3-8B"
CHUNK_SIZE = 1_000_000


def geometry():
    # Keep torch/transformers/datasets out of the CPU prediction path.
    if __package__:
        from . import v6_capability_geometry as module
    else:
        import v6_capability_geometry as module
    return module


def distribution(values):
    """Population moments; entropy of normalized mass (nats), ordinary Gini.

    Equal layer weighting, not parameter weighting. Constant/zero inputs have
    skew=0; zero total mass has entropy=Gini=0. Entropy is not histogram entropy.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Descriptors need a nonempty finite nonnegative vector")
    mean, var = float(x.mean()), float(x.var())
    if x.max() == x.min():
        mean, var = float(x[0]), 0.  # Avoid spurious skew for constants such as 0.8.
    mass = x / x.sum() if x.sum() else np.zeros_like(x)
    positive = mass[mass > 0]
    ordered = np.sort(x)
    gini = (2 * np.dot(np.arange(1, len(x)+1), ordered) / (len(x)*x.sum())
            - (len(x)+1)/len(x)) if x.sum() else 0.
    return {"mean": mean, "variance": var,
            "entropy": float(-np.sum(positive*np.log(positive))),
            "skew": float(np.mean((x-mean)**3)/var**1.5) if var > 0 else 0.,
            "max": float(x.max()), "min": float(x.min()), "gini": float(gini)}


def layer_name(parameter_name):
    """Aggregate matrices in each decoder block; other matrix owners stand alone."""
    match = re.match(r"^(.*?(?:layers|blocks|h)\.\d+)(?:\.|$)", parameter_name)
    return match[1] if match else parameter_name.rpartition(".")[0]


def is_embedding_or_head(name):
    return name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight")


def weight_descriptors(model):
    """Read-only exact V6 mask counts using its unchanged sample and strict > rule."""
    import torch
    v6 = geometry()
    parameters = v6.language_weight_parameters(model)
    sample = v6._sample_abs_weights(parameters, seed=0)
    if not np.isfinite(sample).all():
        raise ValueError("Nonfinite dense weights")
    thresholds = {str(d): float(np.quantile(sample, 1-d)) for d in DENSITIES}
    layers = {}
    with torch.no_grad():
        for name, param in parameters:
            block = layers.setdefault(layer_name(name), {
                "n_weights": 0, "retained_counts": {str(d): 0 for d in DENSITIES}})
            block["n_weights"] += param.numel()
            flat = param.detach().reshape(-1)
            for chunk in flat.split(CHUNK_SIZE):
                absolute = chunk.abs()  # Preserve dtype: exactly V6's scalar comparison.
                if not torch.isfinite(absolute).all().item():
                    raise ValueError(f"Nonfinite dense weights: {name}")
                for d, threshold in thresholds.items():
                    block["retained_counts"][d] += int((absolute > threshold).sum().item())
    features = {f"abs_weight_sample.{k}": v for k, v in distribution(sample).items()}
    for block in layers.values():
        block["retention"] = {d: n/block["n_weights"] for d, n in block["retained_counts"].items()}
    for d in thresholds:
        features.update({f"retention.{d}.{k}": v for k, v in
                         distribution([b["retention"][d] for b in layers.values()]).items()})
    n_weights = sum(p.numel() for _, p in parameters)
    n0 = sum(p.numel() for n, p in parameters if p.ndim == 2 and not is_embedding_or_head(n))
    if n0 <= 0:
        raise ValueError("No non-embedding language matrices for N0")
    return {"N0": n0, "features": features, "layers": layers,
            "n_layers": len(layers), "n_weights": n_weights,
            "weight_bytes": sum(p.numel()*p.element_size() for _, p in parameters),
            "thresholds": thresholds,
            "achieved_density": {d: sum(b["retained_counts"][d] for b in layers.values())/n_weights
                                 for d in thresholds},
            "protocol": {"rule": "v6._sample_abs_weights; np.quantile(1-d); abs(w)>threshold",
                         "seed": 0, "sample_size": v6.PRUNE_THRESHOLD_SAMPLE_SIZE,
                         "actual_sample_size": len(sample), "densities": list(DENSITIES),
                         "scope": "v6.language_weight_parameters (embeddings/head included)",
                         "layer_unit": "decoder block, otherwise matrix-owning module"}}


def activation_descriptors(model, batches, *, protocol):
    """Streaming |block output| moments. No labels, loss, gradients, or pruning.

    Batches are unpadded single sequences. Hooks and each module's training flag
    are restored even on failure. Tiny CPU modules can exercise this function.
    """
    import torch
    names = sorted({layer_name(n) for n, _ in geometry().language_weight_parameters(model)
                    if not is_embedding_or_head(n)})
    modules = dict(model.named_modules())
    if not names or any(n not in modules for n in names):
        raise ValueError("Cannot locate activation layers")
    states = {name: [0, 0., 0.] for name in names}  # count, mean, centered sum of squares
    hooks = []
    training = [(module, module.training) for module in modules.values()]

    def hook(name):
        def collect(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            elif hasattr(output, "last_hidden_state"):
                output = output.last_hidden_state
            if not torch.is_tensor(output):
                raise ValueError(f"Activation layer {name} did not return a tensor")
            for chunk in output.detach().reshape(-1).split(CHUNK_SIZE):
                values = chunk.float().abs()
                if not torch.isfinite(values).all().item():
                    raise ValueError(f"Nonfinite activations: {name}")
                if not values.numel():
                    continue
                var, mean = torch.var_mean(values, correction=0)
                count, old_mean, m2 = states[name]
                n, m, v = values.numel(), mean.item(), var.item()
                delta = m-old_mean
                states[name] = [count+n, old_mean+delta*n/(count+n),
                                m2+v*n+delta**2*count*n/(count+n)]
        return collect

    forwards, tokens = 0, 0
    started = time.perf_counter()
    try:
        model.eval()
        for name in names:
            hooks.append(modules[name].register_forward_hook(hook(name)))
        with torch.inference_mode():
            for batch in batches:
                ids = batch["input_ids"]
                if ids.ndim != 2 or ids.shape[0] != 1 or not ids.numel():
                    raise ValueError("Activation probes must be nonempty single sequences")
                if "attention_mask" in batch and not batch["attention_mask"].bool().all():
                    raise ValueError("Activation probes must be unpadded")
                model(**batch)
                forwards += 1
                tokens += ids.numel()
    finally:
        for handle in hooks:
            handle.remove()
        for module, state in training:
            module.training = state
    if not forwards or any(s[0] == 0 for s in states.values()):
        raise ValueError("Every activation layer must be visited")
    layers = {n: {"count": s[0], "mean": s[1], "variance": max(0., s[2]/s[0])}
              for n, s in states.items()}
    features = {f"abs_activation_{moment}.{k}": v
                for moment in ("mean", "variance")
                for k, v in distribution([s[moment] for s in layers.values()]).items()}
    return {"features": features, "layers": layers, "protocol": protocol,
            "cost": {"extra_dense_forwards": forwards, "input_tokens": tokens,
                     "seconds": time.perf_counter()-started}}


def load_dense_weights(hf_id, dtype, device_map=None):
    """V6 loader conventions and integrity checks, without its mandatory forward.

    Load on host first (like V6). Multimodal fallback is reduced to the text LM.
    No tokenizer/dataset is loaded unless activations are requested.
    ``device_map`` (e.g. "auto") enables accelerate CPU/GPU offload so 27B+
    models fit on a single card; weights stay materialized (CPU or GPU) and the
    forward-only descriptors read them identically.
    """
    import transformers
    v6 = geometry()
    config = transformers.AutoConfig.from_pretrained(hf_id)
    text_config = getattr(config, "text_config", config)
    candidates = [(transformers.AutoModelForCausalLM, text_config, True)]
    for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        if hasattr(transformers, name):
            candidates.append((getattr(transformers, name), config, False))
    architecture = (getattr(config, "architectures", None) or [""])[0]
    if architecture and hasattr(transformers, architecture):
        candidates.append((getattr(transformers, architecture), config, False))
    errors = []
    for loader, cfg, primary in candidates:
        try:
            _extra = {"device_map": device_map} if device_map else {}
            model, info = loader.from_pretrained(hf_id, config=cfg, dtype=dtype,
                                                low_cpu_mem_usage=True, output_loading_info=True,
                                                **_extra)
        except (KeyError, ValueError) as exc:
            errors.append(str(exc))
            continue
        if primary and len(info.get("unexpected_keys") or []) > 10:
            del model
            gc.collect()
            errors.append("Primary loader left more than 10 unexpected checkpoint keys")
            continue
        v6._raise_for_loading_info(hf_id, info, cfg)
        model = v6._language_only_view(model)
        if any(re.search(r"vision|visual|perception|audio|projector", n, re.I)
               for n, _ in model.named_parameters()):
            raise RuntimeError("Non-language tower remains in descriptor scope")
        return model.eval()
    raise RuntimeError(f"Cannot load dense text weights for {hf_id}: {errors}")


def shared_probe_batches(tokenizer, device, forwards=12, max_length=512):
    """First even-index probes, balanced round robin; fixed V6 n=128, seed=0."""
    import torch
    if forwards <= 0 or forwards % 3 or forwards > 192 or max_length < 2:
        raise ValueError("Use 3..192 activation forwards divisible by 3 and max_length>=2")
    probes = geometry().build_probes(128, seed=0)
    selected = [{"capability": cap, "probe_index": 2*i, **probes[cap][2*i]}
                for i in range(forwards//3) for cap in audit.CAPABILITIES]
    protocol = {"probe_builder": "v6.build_probes", "n_probe": 128, "seed": 0,
                "selection": "first even indices, math/code/qa round robin",
                "requested_forwards": forwards, "max_length": max_length,
                "tokenization": "direct prompt+completion; prompt BOS only; halves truncated",
                "probe_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()}

    def batches():
        for row in selected:
            prompt = tokenizer(row["prompt"], return_tensors="pt", truncation=True,
                               max_length=max_length//2, add_special_tokens=True).input_ids
            bos = getattr(tokenizer, "bos_token_id", None)
            tokenizer_name = f"{getattr(tokenizer, 'name_or_path', '')} {type(tokenizer).__name__}".lower()
            if "gemma" in tokenizer_name and bos is not None and (not prompt.numel() or prompt[0, 0] != bos):
                prompt = torch.cat([torch.tensor([[bos]], dtype=prompt.dtype), prompt], dim=1)[:, :max_length//2]
            completion = tokenizer(row["completion"], return_tensors="pt", truncation=True,
                                   max_length=max_length//2, add_special_tokens=False).input_ids
            yield {"input_ids": torch.cat([prompt, completion], dim=1).to(device), "use_cache": False}
    return batches(), protocol


def loss_path(model, root=ROOT):
    candidates = [root / "results" / base / tag / "prune_losses.json"
                  for base in ("v6-capability-geometry", "v9-capability-regions")
                  for tag in (model, "Qwen--"+model)]
    existing = [p for p in candidates if p.is_file()]
    if len(existing) > 1:
        anchors = [audit.read_json(p)["1.0"] for p in existing]
        if any(a != anchors[0] for a in anchors[1:]):
            raise ValueError(f"Conflicting dense anchors: {model}")
    return existing[0] if existing else None


def dense_anchor(path):
    # Never copy compressed observations into a descriptor file.
    if path is None:
        return None
    block = audit.read_json(path)["1.0"]
    return {c: audit.finite(block[c]) for c in audit.CAPABILITIES}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def extract(args):
    hf_id = registry.require_compliant(args.model)  # Before cache check or model access.
    if args.model not in registry.MODEL_REGISTRY:
        raise ValueError("extract requires a model_registry tag")
    if args.with_activations and (args.activation_forwards <= 0 or args.activation_forwards % 3
                                 or args.activation_forwards > 192 or args.activation_max_length < 2):
        raise ValueError("Use 3..192 activation forwards divisible by 3 and max_length>=2")
    destination = args.output_dir / args.model / "features.json"
    previous = audit.read_json(destination) if destination.is_file() else None
    upgrading = previous is not None
    if previous is not None:
        validate_features(previous, args.model)
        if previous["hf_id"] != hf_id or previous["dtype"] != args.dtype:
            raise ValueError("Existing descriptors use a different checkpoint or dtype")
        activation = previous.get("activations")
        if args.with_activations and activation:
            p = activation["protocol"]
            if p["requested_forwards"] != args.activation_forwards or p["max_length"] != args.activation_max_length:
                raise ValueError("Existing activation budget differs; use another output directory")
        if not args.with_activations or activation:
            return previous
    import torch
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("extract requires a GPU; use synthetic unit tests in a CPU sandbox")
    started = time.perf_counter()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if getattr(args, "offload", False):
        # Full CPU residency (materialized, no meta/disk) so weight-descriptor reads work for 27B+.
        # rai has ~1TB RAM; retention needs only weights. Activations (if requested) run on CPU.
        model = load_dense_weights(hf_id, dtype, device_map={"": "cpu"})
    else:
        model = load_dense_weights(hf_id, dtype).to(args.device)
        torch.cuda.synchronize()
    load_seconds = time.perf_counter()-started
    revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    if upgrading and previous.get("revision") and previous["revision"] != revision:
        raise ValueError("Checkpoint revision changed since weight extraction; use another output directory")
    path = args.prune_losses or loss_path(args.model)
    if previous is None:
        started = time.perf_counter()
        weights = weight_descriptors(model)
        meta = audit.read_json(v28.METADATA)
        expected = meta["models"].get(args.model, meta.get("qwen3_8b") if args.model == TARGET else None)
        if expected and weights["N0"] != expected["N0"]:
            raise ValueError("Extracted N0 disagrees with v28 metadata")
        previous = {"schema_version": SCHEMA_VERSION, "model": args.model, "hf_id": hf_id,
                    "revision": revision,
                    "dtype": args.dtype, "N0": weights["N0"],
                    "family": registry.MODEL_REGISTRY[args.model]["family"],
                    "dense_L_c": dense_anchor(path), "dense_loss_source": str(path) if path else None,
                    "weights": weights, "activations": None,
                    "cost": {"weights": {"load_seconds": load_seconds,
                                         "descriptor_seconds": time.perf_counter()-started,
                                         "weight_bytes": weights["weight_bytes"],
                                         "extra_dense_forwards": 0}}}
        write_json(destination, previous)  # Preserve completed weights if activation extraction fails.
    if args.with_activations:
        from transformers import AutoTokenizer
        batches, protocol = shared_probe_batches(AutoTokenizer.from_pretrained(hf_id), args.device,
                                                args.activation_forwards, args.activation_max_length)
        previous["activations"] = activation_descriptors(model, batches, protocol=protocol)
        previous["activations"]["cost"]["reload_seconds"] = load_seconds if upgrading else 0.
        write_json(destination, previous)
    return previous


def validate_features(payload, model):
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("model") != model:
        raise ValueError(f"Descriptor schema/model mismatch: {model}")
    for group, keys in (("weights", WEIGHT_KEYS), ("activations", ACTIVATION_KEYS)):
        if payload.get(group) is None:
            if group == "weights":
                raise ValueError(f"Missing weight descriptors: {model}")
            continue
        features = payload[group]["features"]
        if set(features) != set(keys):
            raise ValueError(f"Unexpected {group} feature schema: {model}")
        for value in features.values():
            audit.finite(value)
    if payload.get("activations") and payload["activations"]["cost"]["extra_dense_forwards"] <= 0:
        raise ValueError("Activation measurement cost must be positive")


def descriptor_input(row, version):
    """Explicit allowlist: no outcome, own coefficient, or raw layer metadata."""
    result = v28.basic_input(row)
    features = row.get("descriptor_features", {})
    keys = () if version == "V1" else WEIGHT_KEYS
    if version == "V3":
        keys += ACTIVATION_KEYS
    if version not in VERSIONS:
        raise ValueError("Unknown input version")
    result["descriptor_features"] = {key: audit.finite(features[key]) for key in keys}
    return result


def fit_mapping(rows, labels, version):
    """Augment the unchanged V28 ridge helper via its exact Schur complement.

    V28 owns the base fit, centering, family coding, intercept and lambda=1.
    Residualizing added columns through that helper and solving their penalized
    block is algebraically the joint ridge fit (not a residual-only heuristic).
    Only source-fold nonconstant descriptors are selected; lambda is fixed.
    """
    rows = [descriptor_input(r, version) for r in rows]
    base = v28.fit_mapping(rows, labels)
    keys = sorted(rows[0]["descriptor_features"])
    values = np.array([[r["descriptor_features"][k] for k in keys] for r in rows])
    selected = np.flatnonzero(values.std(axis=0) > 1e-12) if keys else np.array([], dtype=int)
    keys = [keys[i] for i in selected]
    values = values[:, selected]
    center, scale = values.mean(axis=0), values.std(axis=0)
    z = (values-center)/scale
    extra = np.empty(0)
    if keys:
        x = np.array([[math.log(r["N0"]/1e9), r["dense_loss"]] for r in rows])
        families = np.array([[float(r["family"] == f) for f in base["families"]] for r in rows])
        design = np.column_stack([np.ones(len(rows)), (x-base["center"])/base["scale"],
                                  families-base["family_frequencies"]])
        residual_y = np.asarray(labels)-design @ base["coefficients"]
        residual_z = z - np.column_stack([
            design @ v28.fit_mapping(rows, column)["coefficients"] for column in z.T])
        extra = np.linalg.solve(z.T @ residual_z + np.eye(len(keys)), z.T @ residual_y)
        base = v28.fit_mapping(rows, np.asarray(labels)-z @ extra)
    return {"version": version, "base": base, "descriptor_keys": keys,
            "descriptor_center": center.tolist(), "descriptor_scale": scale.tolist(),
            "descriptor_coefficients": extra.tolist(),
            "selection": "source-fold variance > 1e-12; fixed feature allowlist; no outcome selection"}


def predict_mapping(fit, target):
    row = descriptor_input(target, fit["version"])
    z = (np.array([row["descriptor_features"][k] for k in fit["descriptor_keys"]])
         - fit["descriptor_center"])/fit["descriptor_scale"]
    return float(v28.predict_mapping(fit["base"], row) + z @ fit["descriptor_coefficients"])


def observed_amplitude(row, gamma):
    """Scoring label only: signed least-squares projection on the imported shape."""
    x = np.array([v28.shape("pruning", d, gamma) for d in DENSITIES])
    y = np.array([audit.finite(row["deltas"][str(d)]) for d in DENSITIES])
    return float(y @ x / (x @ x))


def evaluate_fold(train, target, versions):
    shape_fit = v28.fit_arm(train, "pruning")
    labels = [shape_fit["development_coefficient_labels"][r["model"]] for r in train]
    fits = {v: fit_mapping(train, labels, v) for v in versions}
    # Predict from allowlisted dense-only inputs BEFORE projecting target outcomes.
    predictions = {v: predict_mapping(f, descriptor_input(target, v)) for v, f in fits.items()}
    predictions.update(zero=0., population_mean=shape_fit["mean_coefficient"])
    record = {"model": target["model"], "capability": target["capability"],
              "observed": observed_amplitude(target, shape_fit["gamma"]), "predictions": predictions}
    fold = {"held_out": target["model"], "gamma": shape_fit["gamma"],
            "shape_fit_sse": shape_fit["shape_fit_sse"],
            "train_models": [r["model"] for r in train],
            "source_amplitude_labels": shape_fit["development_coefficient_labels"], "fits": fits,
            "target_compressed_points_used_for_prediction": 0}
    return record, fold


def summarize(records, n_boot):
    predictions = {n: [r["predictions"][n] for r in records] for n in records[0]["predictions"]}
    metrics, errors, draws = audit.compare_predictions(records, predictions, "V1", n_boot=n_boot)
    names = list(predictions)
    signs = np.column_stack([np.sign(predictions[n]) == np.sign([r["observed"] for r in records])
                             for n in names]).astype(float)
    sign_draws = audit.bootstrap_means(signs, [r["model"] for r in records], n_boot=n_boot)
    for i, name in enumerate(names):
        metrics[name]["improvement"] = {
            ref: {"mae_gain": float((errors[:, names.index(ref)]-errors[:, i]).mean()),
                  "ci95": audit.interval(draws[:, names.index(ref)]-draws[:, i])}
            for ref in ("V1", "zero", "population_mean")}
        metrics[name].update(sign_agreement=float(signs[:, i].mean()),
                             sign_agreement_ci95=audit.interval(sign_draws[:, i]))
        negative = [j for j, r in enumerate(records) if r["observed"] < 0]
        metrics[name]["negative_amplitude_sign_agreement"] = (
            float(signs[negative, i].mean()) if negative else None)
        metrics[name]["n_negative_models"] = len(negative)
    return metrics


def evaluate_lomo(rows, versions=VERSIONS, n_boot=10000):
    if not versions or versions[0] != "V1" or tuple(versions) != VERSIONS[:len(versions)]:
        raise ValueError("Versions must be nested: V1 [V2 [V3]]")
    if len(rows) < 3 or len({r["model"] for r in rows}) != len(rows):
        raise ValueError("LOMO needs at least three distinct models per capability")
    records, folds = [], []
    for split in audit.splits(rows, "model"):
        train = [rows[i] for i in split["train_indices"]]
        target, = [rows[i] for i in split["test_indices"]]
        record, fold = evaluate_fold(train, target, versions)
        records.append(record)
        folds.append(fold)
    return {"records": records, "folds": folds, "metrics": summarize(records, n_boot)}


def load_panel(metadata, descriptor_dir, versions, root=ROOT):
    """Fixed 12-model dev whitelist. Missing requested groups fail, never shrink cohorts."""
    panels = {c: [] for c in audit.CAPABILITIES}
    target_rows, costs, paths, protocols = {}, {}, [], []
    specs = {**metadata["models"], TARGET: {"family": "qwen3", "N0": metadata["qwen3_8b"]["N0"],
                                        "hf_id": registry.MODEL_REGISTRY[TARGET]["hf_id"]}}
    for model, info in specs.items():
        path = loss_path(model, root)
        if path is None:
            if model == TARGET:
                continue
            raise ValueError(f"Missing pruning labels: {model}")
        payload = audit.read_json(path)
        dense = dense_anchor(path)
        paths.append(path)
        features, cost = {}, {"V1": {"extra_dense_forwards": 0, "dense_L_c": "reused existing measurement"}}
        descriptor_path = descriptor_dir / model / "features.json"
        if len(versions) > 1:
            if not descriptor_path.is_file():
                if model == TARGET:
                    target_rows["status"] = "pending target descriptors"
                    continue
                raise ValueError(f"Missing descriptors for {model}; run the documented GPU extraction commands")
            descriptor = audit.read_json(descriptor_path)
            validate_features(descriptor, model)
            if any(descriptor[k] != info[k] for k in ("N0", "family", "hf_id")):
                raise ValueError(f"Descriptor metadata mismatch: {model}")
            if descriptor["dense_L_c"] is not None and descriptor["dense_L_c"] != dense:
                raise ValueError(f"Descriptor dense anchor mismatch: {model}")
            features.update(descriptor["weights"]["features"])
            cost["V2"] = {**cost["V1"], **descriptor["cost"]["weights"], "measurement": "weights only"}
            signature = {k: v for k, v in descriptor["weights"]["protocol"].items() if k != "actual_sample_size"}
            signature["dtype"] = descriptor["dtype"]
            if "V3" in versions:
                activation = descriptor.get("activations")
                if not activation:
                    if model == TARGET:
                        target_rows["status"] = "pending target activations"
                        continue
                    raise ValueError(f"Missing activations for {model}; extract --with-activations")
                features.update(activation["features"])
                cost["V3"] = {"weights": cost["V2"], **activation["cost"], "measurement": "weights + dense activations"}
                signature["activations"] = activation["protocol"]
            protocols.append(signature)
            paths.append(descriptor_path)
        costs[model] = cost
        for cap in audit.CAPABILITIES:
            row = {"model": model, "family": info["family"], "N0": info["N0"], "capability": cap,
                   "dense_loss": dense[cap], "descriptor_features": features,
                   "deltas": {str(d): audit.finite(payload[str(d)][cap])-dense[cap] for d in DENSITIES}}
            if model == TARGET:
                target_rows[cap] = row
            else:
                panels[cap].append(row)
    if protocols and any(p != protocols[0] for p in protocols[1:]):
        raise ValueError("Descriptor measurement protocols differ between models")
    return panels, target_rows, costs, paths


def extraction_commands():
    tags = list(audit.read_json(v28.METADATA)["models"]) + [TARGET]
    lines = []
    for tag in tags:
        prefix = "SDL_ALLOW_PRC=1 " if registry.is_prc_model(tag) else ""
        lines.append(f"{prefix}python analysis/v32_prune_descriptors.py extract --model {tag} --device cuda:0 --dtype bf16 --with-activations --activation-forwards 12 --activation-max-length 512")
    return lines


def render_report(summary):
    lines = ["# Pruning descriptors and signed amplitude prediction", "",
             "Advisor development audit. GPU extraction has not been run by this implementation task. "
             "Numeric results below use only the versions and source JSON listed in the audit artifact.", "",
             "## Definition and fold discipline", "",
             "We import V28's shape and shape fitter: ΔL_c(d)=a_c((1-d)/0.3)^gamma_c. "
             "Both fitted labels and predictions are signed; no log amplitude, absolute-value target, or clipping. "
             "Each outer model holdout refits shared gamma_c on the other development models at d=0.9/0.8/0.7/0.6. "
             "The held-out label is the least-squares projection of its curve onto that fixed source shape, used only for scoring. "
             "Consequently the estimand is amplitude MAE, not density-cell loss MAE.", "",
             "V1 uses log(N0/1e9), family, dense L_c. V2 adds group (a); V3 adds group (b). "
             "V28's unchanged ridge helper supplies lambda=1 and an unpenalized intercept. Added columns use an exact "
             "Schur-complement extension of the same joint ridge objective. The fixed feature allowlist, nonconstant-column "
             "selection, scaling and family vocabulary use source rows only. Unseen families get V28's zero correction. "
             "No feature/hyperparameter is chosen from outer-fold outcomes. All versions use identical model folds; "
             "missing requested descriptors stop the audit rather than change the cohort.", "",
             "The 12 development models are the V28 metadata whitelist. Qwen3-8B is excluded from every development fold "
             "and scored separately after fitting all 12. Its existing curve is retrospective advisor evidence, not a new prospective test. "
             "The registry resolves Qwen3-8B to Qwen/Qwen3-8B; the legacy loss JSON does not establish Base versus post-trained "
             "checkpoint identity. Extraction records its actual HF ID/revision; this limits interpretation of that comparison.", "",
             "95% intervals use paired whole-model bootstrap draws with fits held fixed. Positive gain means comparator MAE "
             "minus candidate MAE. These intervals describe this panel, not retraining or measurement uncertainty. "
             "Sign agreement uses sign(a), with exact zero as a third class; negative-only agreement is also saved in JSON.", "",
             "## Measurement and cost", "",
             "Group (a) imports V6's deterministic proportional sample (seed 0, requested size 2,000,000) and quantiles. "
             "Retention counts use the exact strict abs(w)>threshold comparison in the original weight dtype. "
             "Ties and sampling mean achieved density can differ from requested density. We never mask or change the dense weights. "
             "Threshold scope includes embeddings/head as in V6; N0 excludes them and counts only 2-D language matrices as in V28.", "",
             "Layers are decoder blocks (layers.N/blocks.N/h.N); other matrix-owning modules, including embeddings/head, "
             "are separate layers. Each density has equally weighted layer-retention mean, population variance, entropy, skew, "
             "max, min and Gini. The same statistics describe sampled absolute weights. Entropy is the natural-log Shannon entropy "
             "of normalized nonnegative mass; zero mass has entropy/Gini zero, constant vectors have skew zero. "
             "Raw counts, layer sizes, thresholds and achieved densities are retained for inspection.", "",
             "Group (b) hooks dense decoder-block outputs (matrix owners for other architectures), excluding embeddings/head. "
             "It streams per-layer mean/variance of absolute activations, then summarizes each with the seven statistics. "
             "The default budget is 12 unpadded, non-generating forwards: first four even-index probes per capability from "
             "V6 build_probes(128, seed=0), round robin. Direct prompt/completion tokenization, prompt BOS only, each half "
             "truncated to 256 tokens. No backward pass or compressed loss is needed. Protocol, probe hash, input-token count, "
             "forward count and elapsed time are saved; mismatched extraction protocols are rejected.", "",
             "| Version | Incremental measurement beyond existing dense L_c |", "|---|---|",
             "| V1 | Metadata and existing dense L_c; 0 extra forwards |",
             "| V2 | V1 + dense weights only; 0 extra forwards; bytes/load/descriptor time recorded |",
             "| V3 | V2 + measured N dense forwards (default N=12); activation time/tokens recorded separately |", "",
             "The weight loader reuses V6 checkpoint-integrity checks without its unconditional sanity forward. "
             "Weight-only extraction does not load a tokenizer or probe dataset. Existing compatible features are reused; "
             "a later --with-activations call adds group (b), recording reload cost. Existing prune_losses.json is read only. "
             "Schema version 1 stores model, HF ID/revision, dtype, N0, family, optional dense_L_c, weights, activations and cost.", "",
             "## Results", "",
             f"Computed versions: {', '.join(summary['versions'])}. Bootstrap resamples: {summary['n_boot']}.", ""]
    missing = [v for v in VERSIONS if v not in summary["versions"]]
    if missing:
        lines += [f"Pending GPU descriptors: {', '.join(missing)}. No descriptor improvement is claimed before extraction.", ""]
    lines += ["| Capability | Predictor | Signed a MAE [95% CI] | Gain over V1 | Gain over zero | Gain over mean | Sign agreement [95% CI] |",
              "|---|---|---|---|---|---|---|"]
    for cap, result in summary["development"].items():
        for name, m in result["metrics"].items():
            gains = [audit.with_ci(m["improvement"][r]["mae_gain"], m["improvement"][r]["ci95"])
                     for r in ("V1", "zero", "population_mean")]
            lines.append(f"| {cap} | {name} | {audit.with_ci(m['mae'], m['mae_ci95'])} | "
                         + " | ".join(gains) + f" | {audit.with_ci(m['sign_agreement'], m['sign_agreement_ci95'])} |")
    lines += ["", "### Qwen3-8B separate retrospective check", ""]
    target = summary["target"]
    if "status" in target:
        lines += [target["status"], ""]
    else:
        lines += ["One model: report individual signed labels/predictions and correctness; no population CI.", "",
                  "| Capability | Observed signed a | Predictor | Predicted signed a | Correct sign |", "|---|---|---|---|---|"]
        for cap, result in target.items():
            row = result["record"]
            for name, value in row["predictions"].items():
                lines.append(f"| {cap} | {row['observed']:.5f} | {name} | {value:.5f} | {bool(np.sign(value)==np.sign(row['observed']))} |")
    lines += ["", "## GPU commands (emitted only; not executed)", "",
              "Run from the repository root in an allocated GPU environment with the checkpoint available. "
              "Gemma and OLMo are hpg-eligible. Muse is also non-PRC and allowed by the existing registry/hpg wrapper. "
              "The existing hpg wrapper uses its tf5 environment for Gemma4/Muse. Qwen commands are rai-only: "
              "SDL_ALLOW_PRC=1 must never be set on hpg. These are 12 dev models plus the separate Qwen3-8B target. "
              "Commands include the optional 12-forward group so all three versions can be compared. "
              "For weights only, omit --with-activations (the budget flags have no effect without it).", "", "```bash",
              *extraction_commands(), "```", "",
              "After extraction, run the CPU audit:", "", "```bash",
              "python analysis/v32_prune_descriptors.py predict --versions V1 V2 V3 --n-boot 10000", "```", "",
              "Before extraction, the existing-data baseline can be reproduced with:", "", "```bash",
              "python analysis/v32_prune_descriptors.py predict --versions V1 --n-boot 10000", "```", "",
              "Machine-readable folds, labels, predictions, costs and input SHA256 hashes are written to "
              "results/v32-descriptors/prediction/summary.json; the same report is saved there as report.md.", ""]
    return "\n".join(lines)


def predict(args):
    metadata = audit.read_json(v28.METADATA)
    panels, target, costs, paths = load_panel(metadata, args.descriptor_dir, args.versions)
    inputs = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in [v28.METADATA, *paths]}
    summary = {"schema_version": SCHEMA_VERSION, "versions": args.versions, "n_boot": args.n_boot,
               "coefficient_definition": "v28 signed a_c; shape=v28.shape('pruning', d, gamma_c)",
               "development": {c: evaluate_lomo(rows, tuple(args.versions), args.n_boot) for c, rows in panels.items()},
               "cost_by_model_and_version": costs, "input_sha256": inputs}
    summary["target"] = {}
    if all(c in target for c in audit.CAPABILITIES):
        for cap in audit.CAPABILITIES:
            record, fold = evaluate_fold(panels[cap], target[cap], args.versions)
            summary["target"][cap] = {"record": record, "fold": fold}
    else:
        summary["target"] = {"status": target.get("status", "pending target pruning labels")}
    report = render_report(summary)
    for path, digest in inputs.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Input changed during analysis: {path}")
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "report.md").write_text(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    return summary


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    modes = ap.add_subparsers(dest="mode", required=True)
    ex = modes.add_parser("extract", help="GPU dense-only descriptors; never prune")
    ex.add_argument("--model", required=True)
    ex.add_argument("--device", default="cuda:0")
    ex.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ex.add_argument("--output-dir", type=Path, default=OUT)
    ex.add_argument("--prune-losses", type=Path)
    ex.add_argument("--with-activations", action="store_true")
    ex.add_argument("--offload", action="store_true",
                    help="load with device_map='auto' (accelerate CPU/GPU offload) so 27B+ models "
                         "fit on one card; forward-only, results identical")
    ex.add_argument("--activation-forwards", type=int, default=12)
    ex.add_argument("--activation-max-length", type=int, default=512)
    pr = modes.add_parser("predict", help="CPU LOMO from saved JSON")
    pr.add_argument("--descriptor-dir", type=Path, default=OUT)
    pr.add_argument("--output-dir", type=Path, default=OUT / "prediction")
    pr.add_argument("--report", type=Path, default=REPORT)
    pr.add_argument("--versions", nargs="+", choices=VERSIONS, default=list(VERSIONS))
    pr.add_argument("--n-boot", type=int, default=10000)
    return ap


def main():
    args = parser().parse_args()
    if args.mode == "extract":
        result = extract(args)
        print(f"{result['model']}: {args.output_dir / result['model'] / 'features.json'}")
    else:
        predict(args)
        print(args.report)


if __name__ == "__main__":
    main()
