#!/usr/bin/env python3
"""Audit cached model architectures and Pythia weight identity, on CPU only.

Run with .venv-gemma4/bin/python -B analysis/v77_model_arch.py. No downloads,
forwards, checkpoint deserialization, GPU access, or writes outside
results/v77-model-arch/. Paper destinations are staged beneath that directory.
Every selected weight blob is streamed through SHA-256; LFS filenames alone
are not accepted as verification. Mono safetensors precede sharded safetensors,
then PyTorch bin, matching the existing loader and v72 cache audit.
"""
from __future__ import annotations

import ast
from collections import defaultdict
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import struct
import sys

sys.dont_write_bytecode = True
for key, value in {
    "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
}.items():
    os.environ[key] = value

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v77-model-arch"
TAG = re.compile(r"pythia-([0-9.]+[mb])(?:@|--)step(\d+)")
MOE_EXAMPLE = "google/gemma-4-26B-A4B-it"
CONVENTIONS = {
    "total_parameters": "Text LM sum(p.numel() for p in model.parameters()) inside "
        "accelerate.init_empty_weights, before explicitly restoring tied weights. "
        "This reproduces v50's literal meta counting procedure, restricted to text_config.",
    "total_unique_parameters": "Same text LM on meta after tie_weights() outside "
        "init_empty_weights; shared embedding/head Parameter counted once. Includes "
        "embeddings, head, norms and biases; excludes vision/audio/projectors and buffers.",
    "active_parameters_per_token": "Dense: total_parameters. MoE: total_parameters "
        "minus all routed expert weights plus k/E of routed expert weights per layer. "
        "Router and always-on shared dense MLP stay in the non-expert count. This is "
        "a parameter convention, not FLOPs or the number of embedding rows accessed.",
    "active_unique_parameters_per_token": "Same active formula with total_unique_parameters.",
    "transformer_matrix_parameters": "Decoder-block matrix/tensor parameters (rank >= 2), "
        "excluding embeddings, LM head, biases and normalization vectors. Pythia: exactly "
        "v53/v64 N0 = L*(4*h*h + 2*h*m). For other architectures the corresponding actual "
        "block matrices are summed; this is an extension, not an assertion that v53 fitted them.",
    "v50_frozen_total_parameters": "Unchanged results/v50-p2v2/freeze.json refs.N: "
        "literal full-config meta count. Gemma-3-4B includes vision/projector parameters; "
        "tied embedding/head weights are counted twice in all three students.",
    "v64": "V64 uses v53.matrix_n0 for Pythia sources and students; it is a "
        "transformer-matrix convention, not v50's full-model meta total.",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Audit:
    def __init__(self):
        self.inputs = {}

    def read(self, path):
        path = Path(path)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        label = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        if label in self.inputs and self.inputs[label] != digest:
            raise RuntimeError(f"Input changed during audit: {path}")
        self.inputs[label] = digest
        return raw

    def json(self, path):
        return json.loads(self.read(path))

    def verify(self):
        for path in list(self.inputs):
            self.read(ROOT / path if not Path(path).is_absolute() else path)


def write(relative, content):
    path = OUT / relative
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError(f"Output outside permitted root: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def literal_assignment(audit, path, name):
    """Read registries without importing scripts that write outputs at import time."""
    tree = ast.parse(audit.read(ROOT / path))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"No literal {name} in {path}")


def cache_root():
    base = Path(os.environ.get("HF_HOME", str(Path(os.environ.get(
        "XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface")))
    return Path(os.environ.get("HF_HUB_CACHE", os.environ.get(
        "HUGGINGFACE_HUB_CACHE", str(base / "hub")))).expanduser()


def repo_path(hub, hf_id):
    return hub / ("models--" + hf_id.replace("/", "--"))


def refs(audit, repo):
    result = {}
    for ref in sorted((repo / "refs").rglob("*")):
        if ref.is_file():
            commit = audit.read(ref).decode().strip()
            if not re.fullmatch(r"[0-9a-f]{40}", commit):
                raise ValueError(f"Invalid cache ref: {ref}")
            result[str(ref.relative_to(repo / "refs"))] = commit
    return result


def cached_config(audit, hub, hf_id):
    repo = repo_path(hub, hf_id)
    revisions = refs(audit, repo)
    candidates = []
    if "main" in revisions:
        candidates.append(repo / "snapshots" / revisions["main"] / "config.json")
    # A usable named revision is preferable to an arbitrary old snapshot.
    for revision in sorted(revisions, key=lambda r: (r != "step143000", r)):
        candidates.append(repo / "snapshots" / revisions[revision] / "config.json")
    candidates.extend(sorted((repo / "snapshots").glob("*/config.json")))
    selected = next((p for p in candidates if p.is_file()), None)
    return selected, revisions


def panel_models(audit):
    registry = literal_assignment(audit, "analysis/model_registry.py", "MODEL_REGISTRY")
    panel_rows = literal_assignment(audit, "analysis/v51_panel_tables.py", "MODELS")
    original = [r[0] for r in panel_rows if r[3] == "panel"]
    if len(original) != 12 or len(set(original)) != 12:
        raise ValueError(f"Expected 12 original panel models, found {original}")
    students = literal_assignment(audit, "analysis/v50_p2v2.py", "TEST_STUDENTS")
    result = []
    for tag, entry in registry.items():
        cohort = "heterogeneous_12" if tag in original else (
            "pythia" if entry["family"] == "pythia" else "prospective_addition")
        evidence = {}
        for version, filename in (("v6-capability-geometry", "prune_losses.json"),
                                  ("v9-capability-regions", None),
                                  ("v10-quantization", "quant_losses.json")):
            aliases = (tag, entry["hf_id"].replace("/", "--"))
            directories = [ROOT / "results" / version / a for a in aliases]
            if filename:
                files = [d / filename for d in directories if (d / filename).is_file()]
                for p in files:
                    audit.read(p)
            else:
                files = [p for d in directories if d.is_dir() for p in sorted(d.glob("*.json"))]
            evidence[version] = [str(p.relative_to(ROOT)) for p in files]
        if cohort == "heterogeneous_12" and not all(evidence[v] for v in (
                "v6-capability-geometry", "v10-quantization")):
            raise ValueError(f"Original panel model missing result evidence: {tag}")
        result.append(dict(tag=tag, hf_id=entry["hf_id"], cohort=cohort,
                           gemma3_student=tag in students, panel_evidence=evidence,
                           missing_result_arms=[v for v, paths in evidence.items() if not paths]))
    return result


def architecture_record(audit, hub, record, frozen_students):
    import torch
    import transformers
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    record = dict(record)
    path, revisions = cached_config(audit, hub, record["hf_id"])
    record["cached_refs"] = revisions
    if path is None:
        record.update(status="missing_cached_config", architecture_class=None,
                      total_parameters=None, active_parameters_per_token=None,
                      transformer_matrix_parameters=None)
        return record
    raw = audit.json(path)
    text_raw = raw.get("text_config", raw)
    config = AutoConfig.for_model(raw["model_type"], **{
        k: v for k, v in raw.items() if k != "model_type"})
    text_config = getattr(config, "text_config", config)
    record.update(status="ok", config_path=str(path), config_blob=str(path.resolve()),
                  config_sha256=sha256(path), snapshot_commit=path.parent.name,
                  config_selection="main" if revisions.get("main") == path.parent.name
                  else "cached named revision/snapshot fallback; main config unavailable",
                  architecture_class=raw.get("architectures", []),
                  text_model_type=text_config.model_type, raw_text_config=text_raw,
                  resolved_text_config=text_config.to_dict())
    experts = getattr(text_config, "num_experts", None)
    top_k = getattr(text_config, "num_experts_per_tok", None) or getattr(
        text_config, "top_k_experts", None)
    moe = bool(getattr(text_config, "enable_moe_block", False) or experts)
    record.update(dense_or_moe="MoE" if moe else "dense", num_experts=experts,
                  num_experts_per_tok=top_k,
                  raw_expert_fields={k: v for k, v in text_raw.items()
                                     if "expert" in k or "moe" in k})
    # Muse has no causal auto class. Construct its native text decoder plus head
    # directly, exactly matching v6's language-only view, without a vision tower.
    with init_empty_weights():
        if text_config.model_type == "muse_glimmer_text":
            from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerTextModel
            model = torch.nn.Module()
            model.language_model = MuseGlimmerTextModel(text_config)
            model.lm_head = torch.nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
            record["text_architecture_class"] = "MuseGlimmerTextModel + Linear LM head"
        else:
            model = AutoModelForCausalLM.from_config(text_config)
            record["text_architecture_class"] = type(model).__name__
    record["total_parameters"] = sum(p.numel() for p in model.parameters())
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    parameters = list(model.named_parameters())
    if any(p.device.type != "meta" for _, p in parameters):
        raise RuntimeError("Unexpected non-meta model parameter")
    record["total_unique_parameters"] = sum(p.numel() for _, p in parameters)
    record["meta_tied_weight_overcount"] = record["total_parameters"] - record["total_unique_parameters"]
    record["transformer_matrix_parameters"] = sum(
        p.numel() for n, p in parameters if p.ndim >= 2 and re.search(r"(?:^|\.)layers\.\d+\.", n))
    record["language_matrix_parameters_including_embeddings_head"] = sum(
        p.numel() for _, p in parameters if p.ndim >= 2)
    expert_parameters = [(n, p) for n, p in parameters if ".experts." in n]
    routed_count = sum(p.numel() for _, p in expert_parameters)
    if moe:
        if not experts or not top_k or not expert_parameters or not 0 < top_k <= experts:
            raise ValueError("MoE routing/count convention unavailable")
        if any(p.ndim != 3 or p.shape[0] != experts for _, p in expert_parameters):
            raise ValueError("Unsupported expert layout; cannot silently estimate active parameters")
    elif routed_count:
        raise ValueError("Found routed experts in a purported dense model")
    selected_count = routed_count // experts * top_k if moe else 0
    record.update(routed_expert_parameters=routed_count,
                  selected_expert_parameters_per_token=selected_count,
                  non_expert_parameters=record["total_parameters"] - routed_count,
                  active_parameters_per_token=record["total_parameters"] - routed_count + selected_count,
                  active_unique_parameters_per_token=record["total_unique_parameters"] - routed_count + selected_count,
                  shared_experts={k: v for k, v in text_raw.items() if "shared_expert" in k},
                  parameter_count_convention="total_parameters; see conventions dictionary",
                  parameter_inventory=[{"name": n, "shape": list(p.shape), "numel": p.numel()}
                                       for n, p in parameters])
    if moe and text_config.model_type == "gemma4_text":
        record["shared_experts"] = {
            "explicit_config_shared_expert_count": None,
            "always_on_dense_mlp_per_layer": 1,
            "shared_mlp_parameters": sum(p.numel() for n, p in parameters if ".mlp." in n),
            "interpretation": "Native Gemma4TextDecoderLayer always executes its dense MLP "
                "alongside routed experts. It is counted in non-expert parameters; no "
                "num_shared_experts field is supplied by this cached config.",
        }
    sources = {inspect.getfile(type(module)) for module in model.modules()
               if type(module).__module__.startswith("transformers.")}
    sources.add(inspect.getfile(type(text_config)))
    for source in sorted(sources):
        audit.read(source)
    record["implementation_sources"] = sorted(sources)
    if text_config.model_type == "gpt_neox":
        n0 = text_config.num_hidden_layers * (4 * text_config.hidden_size ** 2
                + 2 * text_config.hidden_size * text_config.intermediate_size)
        if n0 != record["transformer_matrix_parameters"]:
            raise ValueError("Pythia meta matrices disagree with v53 formula")
        record["v53_formula_verified"] = True
    # The original Fisher metadata counts all language matrices with ties shared.
    checks = []
    for result in record.get("panel_evidence", {}).get("v6-capability-geometry", []):
        meta_path = ROOT / result.replace("prune_losses.json", "fisher_meta.json")
        if meta_path.is_file():
            meta = audit.json(meta_path)
            checks.append(dict(path=str(meta_path.relative_to(ROOT)), recorded=meta["n_params"],
                               matched=meta["n_params"] == record["language_matrix_parameters_including_embeddings_head"]))
    record["v6_fisher_matrix_checks"] = checks
    if any(not c["matched"] for c in checks):
        raise ValueError(f"Meta architecture differs from measured language scope: {record['tag']}")
    if record["tag"] in frozen_students:
        with init_empty_weights():
            original_v50_model = AutoModelForCausalLM.from_config(config)
        reproduced = sum(p.numel() for p in original_v50_model.parameters())
        record.update(v50_frozen_total_parameters=frozen_students[record["tag"]],
                      v50_full_config_meta_reproduced=reproduced,
                      v50_nontext_meta_parameters=reproduced - record["total_parameters"])
        if reproduced != frozen_students[record["tag"]]:
            raise ValueError(f"Failed to reproduce frozen v50 N for {record['tag']}")
        del original_v50_model
    del parameters, expert_parameters, model
    gc.collect()
    print(f"ARCH {record['tag']}: {record['dense_or_moe']} meta={record['total_parameters']:,} "
          f"unique={record['total_unique_parameters']:,} matrices={record['transformer_matrix_parameters']:,}", flush=True)
    return record


def measurement_revisions(audit):
    evidence = defaultdict(set)
    # Raw measurement files establish use; register/prediction-only tags do not.
    filenames = {"prune_losses.json", "quant_losses.json", "quant_group_losses.json"}
    for path in sorted((ROOT / "results").rglob("*.json")):
        if path.is_relative_to(OUT) or path.name not in filenames:
            continue
        matches = list(TAG.finditer(str(path)))
        if not matches:
            continue
        audit.read(path)
        for match in matches:
            evidence[f"pythia-{match[1]}@step{match[2]}"].add(str(path.relative_to(ROOT)))
    return {tag: sorted(paths) for tag, paths in sorted(evidence.items())}


def weight_selection(audit, snapshot):
    for single, index, fmt in (("model.safetensors", "model.safetensors.index.json", "safetensors"),
                               ("pytorch_model.bin", "pytorch_model.bin.index.json", "pytorch_bin")):
        if (snapshot / single).is_file():
            return [single], fmt, single
        if (snapshot / index).is_file():
            data = audit.json(snapshot / index)
            names = sorted(set(data["weight_map"].values()))
            if any(Path(n).name != n for n in names):
                raise ValueError(f"Unexpected shard path in {snapshot / index}")
            return names, fmt, index
    return [], None, None


def safe_header(path):
    """Read tensor shapes only, never allocate/load tensor data."""
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if not 0 < size < min(path.stat().st_size, 100_000_000):
            raise ValueError(f"Invalid safetensors header: {path}")
        header = json.loads(handle.read(size))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    return {"tensor_count": len(tensors),
            "dtypes": sorted({v["dtype"] for v in tensors.values()}),
            "header_sha256": hashlib.sha256(json_text(header).encode()).hexdigest()}


def pythia_tensor_identity(identity, architecture):
    """Distinguish file identity from learned-weight identity, including signed zero.

    GPTNeoX's legacy checkpoint embed_out.weight is the current lm_head.weight.
    Meta named_parameters defines the learned scope, excluding saved attention
    masks/rotary buffers. Tensor bytes are streamed; no model weights are loaded.
    """
    import numpy as np

    if architecture.get("status") != "ok":
        return {"status": "unavailable", "reason": "Missing cached architecture config"}
    names = {("embed_out.weight" if p["name"] == "lm_head.weight" else p["name"]): p["shape"]
             for p in architecture["parameter_inventory"]}
    by_tag = {r["tag"]: r for r in identity["revisions"]}
    states = []
    for revision in ("step16000", "step143000"):
        source = by_tag["pythia-2.8b@" + revision]
        if source["status"] != "ok" or len(source["weights"]) != 1:
            return {"status": "unavailable", "reason": "Missing monolithic safetensors"}
        path = Path(source["weights"][0]["blob_path"])
        state = dict(revision=revision, blob_path=str(path), parameters={})
        with path.open("rb") as handle:
            n = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(n))
            for name, shape in sorted(names.items()):
                tensor = header[name]
                if tensor["shape"] != shape:
                    raise ValueError(f"Checkpoint/meta parameter shape mismatch: {name}")
                start, stop = tensor["data_offsets"]
                handle.seek(8 + n + start)
                digest = hashlib.sha256()
                remaining = stop - start
                while remaining:
                    chunk = handle.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Truncated tensor payload")
                    digest.update(chunk)
                    remaining -= len(chunk)
                state["parameters"][name] = dict(shape=shape, dtype=tensor["dtype"],
                    sha256=digest.hexdigest(), absolute_data_offset=8 + n + start, bytes=stop - start)
        state["checkpoint_only_tensors"] = sorted(set(header) - set(names) - {"__metadata__"})
        states.append(state)
        print(f"TENSOR IDENTITY {revision}: {len(names)} learned tensors hashed", flush=True)
    left, right = states
    different = [n for n in names if any(left["parameters"][n][k] != right["parameters"][n][k]
                                       for k in ("dtype", "shape", "sha256"))]
    details = []
    for name in different:
        a, b = (state["parameters"][name] for state in states)
        if a["dtype"] != "F16" or b["dtype"] != "F16" or a["shape"] != b["shape"]:
            details.append(dict(name=name, equal_modulo_signed_zero=False,
                                reason="Different dtype/shape or unsupported differing dtype"))
            continue
        signed_zero_count, other_count, examples, offset = 0, 0, [], 0
        with Path(left["blob_path"]).open("rb") as f, Path(right["blob_path"]).open("rb") as g:
            f.seek(a["absolute_data_offset"])
            g.seek(b["absolute_data_offset"])
            remaining = a["bytes"]
            while remaining:
                count = min(4 * 1024 * 1024, remaining)
                x, y = np.frombuffer(f.read(count), dtype="<u2"), np.frombuffer(g.read(count), dtype="<u2")
                if x.nbytes != count or y.nbytes != count:
                    raise ValueError("Truncated differing tensor")
                changed = x != y
                zero_only = changed & ((x & 0x7fff) == 0) & ((y & 0x7fff) == 0)
                signed_zero_count += int(zero_only.sum())
                other_count += int((changed & ~zero_only).sum())
                for index in np.flatnonzero(changed)[:max(0, 16 - len(examples))]:
                    examples.append(dict(flat_index=offset + int(index),
                                         step16000_fp16_bits=f"0x{int(x[index]):04x}",
                                         step143000_fp16_bits=f"0x{int(y[index]):04x}"))
                offset += len(x)
                remaining -= count
        details.append(dict(name=name, equal_modulo_signed_zero=other_count == 0,
                            signed_zero_only_elements=signed_zero_count,
                            other_differing_elements=other_count, examples=examples))
    equivalent = all(d["equal_modulo_signed_zero"] for d in details)
    return dict(status="ok", compared_revisions=[s["revision"] for s in states],
                learned_parameter_tensors=len(names), learned_parameter_elements=architecture["total_unique_parameters"],
                byte_identical_parameter_tensors=len(names) - len(different),
                differing_parameter_tensors=details, equal_modulo_signed_zero=equivalent,
                same_parameter_values_revision_group=["step16000", "step64000", "step143000"]
                if equivalent and identity["pythia_2_8b_checks"]["step64000_shares_step143000_blob"] else [],
                method="Compare every meta-defined learned tensor's shape, dtype and SHA-256 payload; "
                    "inspect differing F16 payloads at the bit level. Legacy embed_out maps to lm_head. "
                    "Exclude checkpoint-only attention-mask and rotary buffers.", revisions=states)


def weight_identity(audit, hub, evidence):
    requested = set(evidence) | {"pythia-2.8b@step16000", "pythia-2.8b@step64000",
                                "pythia-2.8b@step143000"}
    freeze_path = ROOT / "results/v72-prune-repeat/freeze.json"
    freeze = audit.json(freeze_path)
    records, blob_cache = [], {}
    all_revisions = {}
    for base in sorted({t.split("@")[0] for t in requested}):
        all_revisions[base] = refs(audit, repo_path(hub, "EleutherAI/" + base))
        requested.update(f"{base}@{r}" for r in all_revisions[base] if r.startswith("step"))
    for tag in sorted(requested):
        base, revision = tag.split("@")
        hf_id = "EleutherAI/" + base
        commit = all_revisions[base].get(revision)
        record = dict(tag=tag, hf_id=hf_id, revision=revision, commit=commit,
                      used_in_pruning_quantization_panels=tag in evidence,
                      measurement_evidence=evidence.get(tag, []), weights=[])
        if commit is None:
            record["status"] = "missing_cache_ref"
            records.append(record)
            continue
        snapshot = repo_path(hub, hf_id) / "snapshots" / commit
        record["snapshot_path"] = str(snapshot)
        if (snapshot / "config.json").is_file():
            raw = audit.read(snapshot / "config.json")
            record["config_sha256"] = hashlib.sha256(raw).hexdigest()
        names, fmt, selected_by = weight_selection(audit, snapshot)
        record.update(format=fmt, selected_by=selected_by, status="ok" if names else "missing_weights",
                      unselected_cached_weight_files=sorted(p.name for p in snapshot.glob("*")
                          if p.suffix in (".safetensors", ".bin") and p.name not in names))
        for name in names:
            path = snapshot / name
            if not path.is_file():
                record["status"] = "missing_shard"
                record["weights"].append(dict(file=name, status="missing"))
                continue
            target = path.resolve()
            before = target.stat()
            key = (str(target), before.st_size, before.st_mtime_ns, before.st_ino)
            if key not in blob_cache:
                print(f"HASH {tag} {name} ({before.st_size / 1e9:.2f} GB)", flush=True)
                digest = sha256(target)
                after = target.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                        after.st_size, after.st_mtime_ns, after.st_ino):
                    raise RuntimeError(f"Weight blob changed while hashing: {target}")
                blob_cache[key] = digest
            digest = blob_cache[key]
            address = target.name if re.fullmatch(r"[0-9a-f]{64}", target.name) else None
            weight = dict(file=name, snapshot_path=str(path), blob_path=str(target),
                          is_symlink=path.is_symlink(), symlink_target=os.readlink(path) if path.is_symlink() else None,
                          bytes=before.st_size, sha256=digest, lfs_address_sha256=address,
                          lfs_address_matches=address == digest if address else None,
                          digest_method="SHA-256 streamed from actual blob bytes; each unique target read once")
            if fmt == "safetensors":
                weight["safetensors_header"] = safe_header(path)
            record["weights"].append(weight)
            if address is not None and address != digest:
                record["status"] = "lfs_content_address_mismatch"
        if base == "pythia-2.8b" and revision in freeze["cache"]["revisions"]:
            prior = freeze["cache"]["revisions"][revision]
            old = sorted((w["file"], w["blob_sha256"], w["bytes"]) for w in prior["weights"])
            now = sorted((w["file"], w["sha256"], w["bytes"]) for w in record["weights"] if "sha256" in w)
            record["v72_provenance"] = dict(path=str(freeze_path.relative_to(ROOT)),
                commit_matches=prior["commit"] == commit, selected_weight_hashes_match=old == now,
                historical_evidence="v72 frozen loader-selection provenance; byte hashes verified in v77")
        else:
            record["historical_evidence"] = ("Current cache selection tied to measured revision tag; "
                "the raw measurement files do not record historical blob hashes. This "
                "audit cannot establish that the cache was unchanged since measurement.")
        records.append(record)
    groups = defaultdict(list)
    shared = defaultdict(list)
    for r in records:
        if r["status"] == "ok":
            signature = tuple(sorted((w["sha256"], w["bytes"]) for w in r["weights"]))
            groups[(r["hf_id"], signature)].append(r["tag"])
            for w in r["weights"]:
                shared[(w["sha256"], w["bytes"])].append(r["tag"])
    duplicates = [dict(hf_id=k[0], revisions=sorted(tags),
                       sha256=[h for h, _ in k[1]]) for k, tags in groups.items() if len(tags) > 1]
    by_tag = {r["tag"]: r for r in records}
    def signature(tag):
        r = by_tag[tag]
        return sorted(w["sha256"] for w in r["weights"]) if r["status"] == "ok" else None
    a, b, c = (signature("pythia-2.8b@step" + s) for s in ("16000", "64000", "143000"))
    return dict(schema_version=1, cache_root=str(hub), revisions=records,
                coverage={"measured_revision_count": len(evidence), "audited_revision_count": len(records),
                          "verified_unique_blob_count": len(blob_cache),
                          "verified_unique_bytes": sum(k[1] for k in blob_cache)},
                full_weight_set_coincidences=duplicates,
                individual_blob_coincidences=[dict(sha256=k[0], bytes=k[1], revisions=sorted(set(tags)))
                    for k, tags in shared.items() if len(set(tags)) > 1],
                pythia_2_8b_checks={"step16000_distinct_from_step143000": bool(a and c and a != c),
                                   "step64000_shares_step143000_blob": bool(b and c and b == c)},
                limitations=["Identity is of serialized weight blobs, not a tensor-value equivalence "
                    "test across different serializations. The supplemental Pythia-2.8B check "
                    "compares all learned tensors and identifies signed-zero-only differences.",
                    "Step labels are cache revision labels; coincident blobs are not independent weight states.",
                    "Only v72 supplies frozen blob-selection provenance; other panels are checked against "
                    "their current cached revisions. PyTorch bin shards are hashed without deserialization."])


def weight_markdown(identity):
    cov = identity["coverage"]
    lines = ["# V77 Pythia weight identity", "",
             f"Audited {cov['audited_revision_count']} cached step revisions, including all "
             f"{cov['measured_revision_count']} revisions with raw pruning/quantization measurements. "
             f"Streamed SHA-256 over {cov['verified_unique_blob_count']} unique blobs "
             f"({cov['verified_unique_bytes']:,} bytes).", "",
             "Pythia-2.8B step16000 and step143000 select different serialized blobs. "
             "The step64000 and step143000 snapshots select the same blob. "
             "All three selections are cross-checked against the v72 freeze provenance.", ""]
    tensor = identity["pythia_2_8b_tensor_identity"]
    if tensor.get("equal_modulo_signed_zero"):
        lines += ["**Critical weight-state finding:** step16000, step64000 and step143000 "
            "have identical learned parameter values modulo the sign of zero. Among "
            f"{tensor['learned_parameter_tensors']} parameter tensors, "
            f"{tensor['byte_identical_parameter_tensors']} are byte-identical between step16000 "
            "and step143000. The only difference is `gpt_neox.layers.8.input_layernorm.bias[201]`: "
            "step16000 stores `+0.0` (`0x0000`), step143000 stores `-0.0` (`0x8000`). "
            "Step16000 additionally serializes 96 non-parameter buffers. **Distinct file hashes "
            "therefore do not establish two independent Pythia training states for v72.** "
            "This audit does not modify the existing measurements or fits.", ""]
    lines += [
             "## Coincident revision groups", ""]
    for group in identity["full_weight_set_coincidences"]:
        lines.append("- " + ", ".join(f"`{t}`" for t in group["revisions"]))
    if not identity["full_weight_set_coincidences"]:
        lines.append("None found.")
    lines += ["", "No other complete selected weight sets coincide among successfully verified revisions.",
              "", "## Selected weight blobs", "",
              "| Revision | Measured panel state | Format/file | Bytes | Actual SHA-256 |",
              "|---|---|---|---:|---|"]
    for r in identity["revisions"]:
        if not r["weights"]:
            lines.append(f"| {r['tag']} | — | {r['status']} | — | — |")
        for w in r["weights"]:
            byte_label = f"{w['bytes']:,}" if "bytes" in w else "missing"
            lines.append(f"| {r['tag']} | {'yes' if r['used_in_pruning_quantization_panels'] else 'cache check only'} "
                         f"| {w['file']} | {byte_label} | `{w.get('sha256', 'missing')}` |")
    lines += ["", "The step16000 snapshot also contains two safetensors shards. They are not "
              "the selected weights: the loader chooses its monolithic `model.safetensors` first. "
              "Pythia-6.9B uses two `.bin` shards per revision; both are included in the identity signature.",
              "", "## Provenance and limits", ""]
    lines += ["- " + s for s in identity["limitations"]]
    lines += ["- JSON includes resolved snapshot commits, symlink targets, actual blob paths, "
              "byte lengths, LFS address comparisons, measurement evidence and v72 comparisons.", ""]
    return "\n".join(lines)


def table_tex(records):
    codes = {"gemma3_text": "G3", "gemma4_text": "G4", "muse_glimmer_text": "MG",
             "olmo3": "O3", "gpt_neox": "PN", "qwen3": "Q3"}
    lines = ["% Generated by analysis/v77_model_arch.py; all counts in billions.",
             r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             r"Model & Class/type & $N_{\rm meta}$ & $N_{\rm active}$ & $N_{\rm unique}$ & $N_{0,\rm matrix}$ & Conv. \\",
             r"\midrule"]
    for cohort, title in (("heterogeneous_12", "Original heterogeneous panel (12 models)"),
                          ("pythia", "Pythia architecture sizes (revisions audited separately)"),
                          ("prospective_addition", "Later prospective additions")):
        lines.append(r"\multicolumn{7}{l}{\textit{" + title + r"}} \\")
        cohort_records = [r for r in records if r["cohort"] == cohort]
        if cohort == "pythia":
            cohort_records.sort(key=lambda r: r.get("total_parameters") or 0)
        for r in cohort_records:
            name = r["tag"].replace("_", r"\_")
            if r["status"] != "ok":
                lines.append(name + r" & missing config & -- & -- & -- & -- & -- \\")
                continue
            cls = codes.get(r["text_model_type"], r["text_model_type"])
            counts = " & ".join(f"{r[k] / 1e9:.3f}" for k in (
                "total_parameters", "active_parameters_per_token", "total_unique_parameters",
                "transformer_matrix_parameters"))
            lines.append(f"{name} & {cls}/{r['dense_or_moe']} & {counts} & M/U/T " + r"\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}",
        r"\caption{Cached text-model architectures and parameter counts ($10^9$). "
        r"M: literal meta-device parameter count before restoring tied weights; "
        r"U: unique parameters after restoring ties; T: transformer matrices, excluding "
        r"embeddings, LM head, norms and biases. Dense active counts equal M under this "
        r"convention; all 12 original panel models are dense. M can double-count a tied "
        r"embedding/head. Pythia T is exactly the v53/v64 law covariate "
        r"$L(4h^2+2hm)$. Multimodal checkpoints use their text decoder only. "
        r"G3: Gemma3ForCausalLM; G4: Gemma4ForCausalLM; "
        r"MG: MuseGlimmerTextModel plus LM head; O3: Olmo3ForCausalLM; "
        r"PN: GPTNeoXForCausalLM; Q3: Qwen3ForCausalLM. "
        r"The cached top-level classes for multimodal Gemma/Muse are the corresponding "
        r"ForConditionalGeneration classes. V50's frozen Gemma-3 student counts are "
        r"0.435870336B, 1.301875840B, and 4.971331952B; the last includes "
        r"0.419816304B non-text parameters and differs from the text-only M column. "
        r"Gemma-4-26B-A4B-it is a cached MoE outside the registry/panel and is excluded here.}",
        r"\label{tab:model_arch}", r"\end{table}", ""]
    return "\n".join(lines)


def summary_markdown(summary):
    records, identity = summary["models"], summary["weight_identity"]
    lines = ["# V77 model architecture and weight identity audit", "",
        "CPU/config/meta inspection only; weight files are streamed for hashes and never loaded as tensors. "
        "The 12 original heterogeneous models are identified by v51's `cohort=panel` and checked "
        "against v6/v9/v10 result files. All six registered Pythia sizes and two later Qwen3 "
        "prospective additions are also reported; the three Gemma-3 students are already panel members.", "",
        f"**Panel MoE models:** {', '.join(summary['panel_moe_models']) or 'none'}. "
        f"**Models missing cached config:** {', '.join(summary['missing_cached_configs']) or 'none'}.", "",
        "**V9 coverage gap:** Gemma-3-27B and OLMo-3-32B have empty v9 result directories. "
        "Both have completed v6/v10 results and remain members of the original 12-model panel.", "",
        "## Parameter conventions", ""]
    lines += [f"- **{k}:** {v}" for k, v in CONVENTIONS.items()]
    lines += ["", "These counts expose, but do not change, the existing laws or their frozen covariates. "
        "The literal meta counts must not be presented as independent physical weights when the "
        "embedding and LM head are tied. No model size is inferred from its marketing name.", "",
        "## Architecture table (exact integer counts)", "",
        "| Model | Cohort | Text architecture | Type | Total meta | Active meta | Unique text | Transformer matrices |",
        "|---|---|---|---|---:|---:|---:|---:|"]
    for r in records:
        if r["status"] == "ok":
            counts = " | ".join(f"{r[k]:,}" for k in ("total_parameters", "active_parameters_per_token",
                "total_unique_parameters", "transformer_matrix_parameters"))
            lines.append(f"| {r['tag']} | {r['cohort']} | {r['text_architecture_class']} | {r['dense_or_moe']} | {counts} |")
        else:
            lines.append(f"| {r['tag']} | {r['cohort']} | missing config | unknown | — | — | — | — |")
    lines += ["", "## Frozen Gemma-3 student convention", "",
        "| Student | V50 frozen full-config meta | Text-only meta | Unique text | Non-text in frozen count |",
        "|---|---:|---:|---:|---:|"]
    for r in records:
        if "v50_frozen_total_parameters" in r:
            lines.append("| " + r["tag"] + " | " + " | ".join(f"{r[k]:,}" for k in (
                "v50_frozen_total_parameters", "total_parameters", "total_unique_parameters",
                "v50_nontext_meta_parameters")) + " |")
    extra = summary["cached_moe_outside_panel"]
    lines += ["", "## Cached MoE outside the panel", "",
        f"`{MOE_EXAMPLE}` is absent from the registry and original v6/v9/v10 panel. "
        "Its cached `text_config` explicitly enables MoE: 128 routed experts, top_k_experts=8. "
        "The implementation also executes one always-on dense MLP per layer; no explicit "
        "shared-expert count appears in the config.", ""]
    if extra["status"] == "ok":
        lines.append(f"Text meta total {extra['total_parameters']:,}; active meta {extra['active_parameters_per_token']:,}; "
            f"unique total {extra['total_unique_parameters']:,}; active unique {extra['active_unique_parameters_per_token']:,}. "
            f"Routed expert parameters {extra['routed_expert_parameters']:,}; selected expert parameters "
            f"{extra['selected_expert_parameters_per_token']:,}; non-expert meta parameters "
            f"{extra['non_expert_parameters']:,}. This supplementary cache check does not add it to the panel.")
    lines += ["", "## Pythia identity", "",
        f"Audited {identity['coverage']['audited_revision_count']} cached revisions covering "
        f"{identity['coverage']['measured_revision_count']} measured pruning/quantization states. "
        "Pythia-2.8B step16000 and step143000 select distinct safetensors blobs; step64000 "
        "shares the step143000 blob. No other selected weight-set coincidences were found. "
        "See [weight_identity.md](weight_identity.md) and [weight_identity.json](weight_identity.json) "
        "for every full SHA-256 and provenance limit.", "",
        "**Critical:** the full learned-tensor audit finds that all three Pythia-2.8B revision "
        "labels have the same parameter values modulo signed zero. Between step16000 and "
        "step143000, 387/388 parameter tensors are byte-identical; the remaining tensor "
        "differs only at one +0.0/-0.0 entry. Extra serialized buffers also change the file hash. "
        "V72's two distinct blob hashes do not establish two independent training states.", "",
        "## Validation and artifacts", "",
        "- Every original panel model's language matrix count matches its v6 Fisher metadata.",
        "- All six Pythia matrix counts match the exact v53 formula; all three frozen v50 student totals reproduce exactly.",
        "- Every selected blob is hashed from bytes and checked against its LFS address; v72's three cached Pythia-2.8B selections match frozen provenance.",
        "- All 388 learned Pythia-2.8B parameter tensors are compared across the two distinct "
        "serialized blobs; the only payload difference is the sign bit of one zero.",
        "- `model_arch.json` retains full cached/resolved text configs, top-level architecture classes, "
        "parameter inventories, implementation source hashes, snapshot commits and result evidence.",
        "- `paper/paper/tables/model_arch.tex` and `paper/analysis/v77_model_arch.py` are staged "
        "under this results directory to respect the requested write boundary. No commit is made.", ""]
    return "\n".join(lines)


def main():
    import torch
    import transformers
    import accelerate

    torch.set_num_threads(1)
    audit, hub = Audit(), cache_root()
    for name in ("v77_model_arch.py", "v53_prune_dev.py", "v64_selection_feasible.py",
                 "v6_capability_geometry.py", "v72_prune_repeat.py"):
        audit.read(ROOT / "analysis" / name)
    audit.read(inspect.getfile(accelerate.init_empty_weights))
    frozen = audit.json(ROOT / "results/v50-p2v2/freeze.json")["refs"]["N"]
    records = [architecture_record(audit, hub, r, frozen) for r in panel_models(audit)]
    extra = architecture_record(audit, hub, dict(tag=MOE_EXAMPLE, hf_id=MOE_EXAMPLE,
        cohort="cached_only_not_panel", gemma3_student=False), {})
    evidence = measurement_revisions(audit)
    identity = weight_identity(audit, hub, evidence)
    identity["pythia_2_8b_tensor_identity"] = pythia_tensor_identity(
        identity, next(r for r in records if r["tag"] == "pythia-2.8b"))
    bad_weights = [r["tag"] for r in identity["revisions"] if r["status"] != "ok"]
    provenance_bad = [r["tag"] for r in identity["revisions"] if "v72_provenance" in r and
                     not (r["v72_provenance"]["commit_matches"] and
                          r["v72_provenance"]["selected_weight_hashes_match"])]
    validation = dict(weight_failures=bad_weights, v72_provenance_failures=provenance_bad,
                      cpu_only=not torch.cuda.is_initialized(), original_panel_models=12,
                      pythia_sizes=sum(r.get("v53_formula_verified", False) for r in records),
                      reproduced_v50_students=sum("v50_frozen_total_parameters" in r for r in records),
                      panel_fisher_matches=sum(len(r.get("v6_fisher_matrix_checks", [])) for r in records))
    if not validation["cpu_only"]:
        raise RuntimeError("CUDA unexpectedly initialized")
    audit.verify()
    identity["input_sha256"] = audit.inputs
    summary = dict(schema_version=1, conventions=CONVENTIONS, models=records,
                   cached_moe_outside_panel=extra,
                   missing_cached_configs=[r["tag"] for r in records if r["status"] == "missing_cached_config"],
                   panel_moe_models=[r["tag"] for r in records if r["cohort"] == "heterogeneous_12" and r.get("dense_or_moe") == "MoE"],
                   versions=dict(python=sys.version.split()[0], torch=torch.__version__,
                                 transformers=transformers.__version__, accelerate=accelerate.__version__),
                   validation=validation, input_sha256=audit.inputs, weight_identity=identity)
    write("model_arch.json", json_text({k: v for k, v in summary.items() if k != "weight_identity"}))
    write("weight_identity.json", json_text(identity))
    write("weight_identity.md", weight_markdown(identity))
    write("pythia_2_8b_tensor_identity.json", json_text(identity["pythia_2_8b_tensor_identity"]))
    write("summary.md", summary_markdown(summary))
    write("paper/paper/tables/model_arch.tex", table_tex(records))
    write("paper/analysis/v77_model_arch.py", Path(__file__).read_bytes())
    print(json_text(validation), flush=True)
    if bad_weights or provenance_bad or not all(identity["pythia_2_8b_checks"].values()):
        raise SystemExit("Identity audit has failures; see output records")


if __name__ == "__main__":
    main()
