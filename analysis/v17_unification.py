#!/usr/bin/env python3
"""V17: perturbative, overlap-aware cross-method unification test.

This CPU-only analysis assembles pruning, quantization, and two distillation
delta definitions in one table.  Its common coordinates are fixed:

* unstructured pruning at retained density ``d``: ``r_storage=d``,
  ``r_active=1`` (the dense kernels still execute every weight);
* quantization at ``b`` bits: ``r_storage=b/16``, ``r_active=1`` (the dense
  artifact is the bf16, 16-bit anchor);
* distillation with student size ``N_S`` and dense reference ``N_0``:
  ``r_storage=N_S/N_0``, ``r_active=N_S/N_0``.

For the V16 API-teacher runs, teacher parameter counts are undisclosed.  The
distillation normalization uses a registered same-family source model as
``N_0``; it is not a claim about API-teacher size.  Cross-method analysis uses
``distill_source = L_c(distilled student) - L_c(source dense)``.  The earlier
style-residualized, student-self-referenced ``dL_cap`` remains in the assembled
table as ``distill_self`` for the validity discussion but never enters the
method-selection map or matched comparisons.  The only registered clean
source ladder in the current artifacts is Gemma-3 270M/1B/4B/12B against
Gemma-3 27B.  Student and reference storage counts use full checkpoint
parameter counts where they are available; any embedding-inclusive V6
fallback is marked in the output table and report.

All laws are fitted only before the first capability-specific cliff.  For each
model/method/capability trajectory, the cliff begins at the first coordinate
whose damage exceeds ``--precliff-cap`` while moving toward more compression;
that point and every more-compressed point are excluded from fitting and
evaluation.

The three pre-registered hypotheses are fitted without form search:

* M1: one curve for every method, capability, and family,
  ``A * (r_storage**(-alpha) - 1)``.  There is no capability scale.
* M2: the M1 curve plus zero-sum additive method and family effects.
* M3: separate audit-compatible perturbative reduced forms: a pruning power
  branch, a fixed-base quantization branch, and a distillation capacity
  intercept plus log-size term fitted to ``distill_source``.

The shared-law claim is assessed only on pre-cliff rows whose storage ratios
are within ``--storage-match-tolerance`` of another method.  Held-out metrics
on all pre-cliff rows are retained as diagnostics, but are not allowed to turn
cross-method extrapolation through empty storage regimes into a unification
claim.  In-sample MAE is emitted and printed as ``NON-DECISIONAL``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

try:
    from .model_registry import MODEL_REGISTRY
except ImportError:  # direct execution: python analysis/v17_unification.py
    from model_registry import MODEL_REGISTRY


ROOT = Path(__file__).resolve().parents[1]
PRUNE_BASE = ROOT / "results/v6-capability-geometry"
QUANT_BASE = ROOT / "results/v10-quantization"
DISTILL_BASE = ROOT / "results/v16-style-residual"
OUT_BASE = ROOT / "results/v17-unification"

# V17's original heterogeneous 12-model panel (unification_table.csv), before
# the prospective Qwen and controlled Pythia campaigns. Shared V6/V10 storage
# does not enroll later experiments in this historical fit. Excluded paths are
# reported in the audit; missing counts for an in-scope model still fail closed.
V17_MODELS = frozenset((
    "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "gemma3-270m", "gemma3-1b",
    "gemma3-4b", "gemma3-12b", "gemma3-27b", "gemma4-31b", "muse-30b",
    "olmo3-7b", "olmo3-32b",
))
V17_BITS = frozenset((3., 4., 6., 8., 16.))

DEFAULT_PRECLIFF_CAP = 1.0
DEFAULT_STORAGE_MATCH_TOLERANCE = 0.03
DEFAULT_LOSS_BUDGETS = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)

CAPABILITIES = ("math", "code", "qa")
DISTILL_SELF = "distill_self"
DISTILL_SOURCE = "distill_source"
METHODS = ("pruning", "quantization", DISTILL_SOURCE)
TABLE_METHODS = (*METHODS, DISTILL_SELF)
METHOD_ORDER = {
    method: index
    for index, method in enumerate(
        ("pruning", "quantization", DISTILL_SELF, DISTILL_SOURCE)
    )
}
EPS = np.finfo(np.float64).eps

# Source-referenced distillation is intentionally opt-in.  Same-family model
# availability alone is not evidence that a student was distilled from that
# source.  In particular, OLMo-3 7B is only a single observed student and has no
# clean distilled-from-OLMo-3-32B run, so it must not acquire a fabricated
# source delta merely because the 32B dense losses happen to exist on disk.
DISTILLATION_SOURCE_LADDERS: dict[str, dict[str, object]] = {
    "gemma3": {
        "source_model": "gemma3-27b",
        "students": (
            "gemma3-270m",
            "gemma3-1b",
            "gemma3-4b",
            "gemma3-12b",
        ),
        "run_suffix": "full_600",
        "preferred_teacher": "gpt-5.6-luna",
    }
}

# Exact full-checkpoint tensor counts for the Gemma-3 ladder.  The multimodal
# 4B/12B/27B checkpoints include the vision tower because r_storage describes
# the stored artifact.  These yield the registered ladder ratios 0.010, 0.036,
# 0.157, and 0.445 after rounding to three decimals.
DOCUMENTED_CHECKPOINT_PARAMS: dict[str, int] = {
    "gemma3-270m": 268_098_176,
    "gemma3-1b": 999_885_952,
    "gemma3-4b": 4_300_079_472,
    "gemma3-12b": 12_187_325_040,
    "gemma3-27b": 27_432_406_640,
}

# V6's n_params is the number of 2-D language weights and includes token
# embeddings.  These counts subtract the tied embedding matrix using the
# documented architecture dimensions.  Models absent from both dictionaries
# fall back to V6 n_params, and that fallback is identified in
# ``n0_count_source``.
DOCUMENTED_NON_EMBEDDING_PARAMS: dict[str, int] = {
    "Qwen3-0.6B": 440_401_920,
    "Qwen3-1.7B": 1_409_286_144,
    "Qwen3-4B": 3_633_315_840,
    "gemma4-31b": 29_286_727_680,
    "olmo3-7b": 6_476_005_376,
}

TABLE_FIELDS = (
    "method",
    "model",
    "family",
    "capability",
    "raw_coordinate_name",
    "raw_coordinate",
    "r_storage",
    "r_active",
    "delta_L_c",
    "dL_cap",
    "delta_L_source_c",
    "distilled_L_c",
    "reference_L_c",
    "delta_reference",
    "source_reference_status",
    "source_loss_path",
    "precliff_status",
    "precliff_cap",
    "cliff_r_storage",
    "overlap_method_count",
    "overlap_methods",
    "model_size_params",
    "n0_params",
    "n0_model",
    "n0_count_source",
    "is_baseline",
    "teacher",
    "recipe",
    "trace_examples_per_domain",
    "run",
    "source_path",
)


def canonical_model_tag(tag: str) -> str:
    """Normalize model directory spellings shared by V6/V10/V16."""
    value = str(tag).strip()
    if value.startswith("Qwen--"):
        return value.removeprefix("Qwen--")
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    aliases = {
        "gemma-3-270m": "gemma3-270m",
        "gemma-3-1b-pt": "gemma3-1b",
        "gemma-3-4b-pt": "gemma3-4b",
        "gemma-3-12b-pt": "gemma3-12b",
        "gemma-3-27b-pt": "gemma3-27b",
        "gemma-4-31B": "gemma4-31b",
        "Olmo-3-1025-7B": "olmo3-7b",
        "Olmo-3-1125-32B": "olmo3-32b",
        "Muse-Glimmer-30B": "muse-30b",
    }
    return aliases.get(value, value)


def model_family(model: str, resolved_name: str = "") -> str:
    """Map every analysis model to one of the paper's five family labels."""
    tag = canonical_model_tag(model)
    registry = MODEL_REGISTRY.get(tag, {})
    registered = str(registry.get("family", ""))
    text = f"{tag} {resolved_name} {registered}".lower()
    if "qwen3" in text:
        return "qwen3"
    if "gemma4" in text or "gemma-4" in text:
        return "gemma4"
    if "gemma3" in text or "gemma-3" in text:
        return "gemma3"
    if "olmo3" in text or "olmo-3" in text:
        return "olmo3"
    if "muse" in text or "glimmer" in text:
        return "muse"
    raise ValueError(f"No paper family mapping for model {model!r}")


def coordinate_mapping(
    method: str,
    raw_coordinate: float,
    *,
    n0_params: int | float | None = None,
) -> dict[str, float | str]:
    """Return the fixed raw/common coordinates for one compression cell."""
    value = float(raw_coordinate)
    if not math.isfinite(value):
        raise ValueError("raw coordinate must be finite")
    if method == "pruning":
        if not 0.0 < value <= 1.0:
            raise ValueError("pruning density must lie in (0, 1]")
        return {
            "raw_coordinate_name": "density",
            "raw_coordinate": value,
            "r_storage": value,
            # These artifacts use unstructured zeroing without sparse kernels:
            # storage falls, but active parameters / dense-kernel work do not.
            "r_active": 1.0,
        }
    if method == "quantization":
        if value <= 0.0 or value > 16.0:
            raise ValueError("quantization bits must lie in (0, 16]")
        return {
            "raw_coordinate_name": "bits",
            "raw_coordinate": value,
            "r_storage": value / 16.0,
            "r_active": 1.0,
        }
    if method in {"distillation", DISTILL_SELF, DISTILL_SOURCE}:
        if n0_params is None:
            raise ValueError("distillation coordinates require n0_params")
        reference = float(n0_params)
        if value <= 0.0 or not math.isfinite(reference) or reference <= 0.0:
            raise ValueError("student and reference parameter counts must be positive")
        ratio = value / reference
        if ratio > 1.0 + 1e-12:
            raise ValueError("distillation student cannot exceed its dense reference")
        return {
            "raw_coordinate_name": "student_params",
            "raw_coordinate": value,
            "r_storage": ratio,
            "r_active": ratio,
        }
    raise ValueError(f"unknown compression method {method!r}")


def leave_one_out_splits(
    rows: Sequence[Mapping[str, object]], group_key: str
) -> list[dict[str, object]]:
    """Return deterministic index splits that hold out one whole group."""
    if not rows:
        return []
    if any(group_key not in row for row in rows):
        raise KeyError(f"split key {group_key!r} is absent from at least one row")
    groups = sorted({str(row[group_key]) for row in rows})
    splits = []
    for group in groups:
        test = [index for index, row in enumerate(rows) if str(row[group_key]) == group]
        train = [
            index for index, row in enumerate(rows) if str(row[group_key]) != group
        ]
        if train and test:
            splits.append(
                {
                    "held_out": group,
                    "train_indices": train,
                    "test_indices": test,
                }
            )
    return splits


def leave_largest_model_out_splits(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Within each multi-model family, hold out every cell of its largest model."""
    by_family: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        family = str(row["family"])
        model = str(row["model"])
        size = float(row["model_size_params"])
        previous = by_family[family].get(model)
        if previous is not None and not math.isclose(previous, size, rel_tol=0.0):
            raise ValueError(f"inconsistent size for model {model}")
        by_family[family][model] = size
    splits = []
    for family, model_sizes in sorted(by_family.items()):
        if len(model_sizes) < 2:
            continue
        largest = max(model_sizes, key=lambda model: (model_sizes[model], model))
        test = [index for index, row in enumerate(rows) if row["model"] == largest]
        train = [index for index, row in enumerate(rows) if row["model"] != largest]
        if train and test:
            splits.append(
                {
                    "held_out": largest,
                    "family": family,
                    "train_indices": train,
                    "test_indices": test,
                }
            )
    return splits


def _json_load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _deduplicated_artifacts(base: Path, filename: str) -> list[tuple[str, Path]]:
    """Deduplicate aliases such as Qwen3-1.7B and Qwen--Qwen3-1.7B."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    if base.is_dir():
        for path in sorted(base.glob(f"*/{filename}")):
            grouped[canonical_model_tag(path.parent.name)].append(path)
    selected = []
    for model, paths in sorted(grouped.items()):
        if len(paths) > 1:
            payloads = [_json_load(path) for path in paths]
            if any(payload != payloads[0] for payload in payloads[1:]):
                joined = ", ".join(str(path) for path in paths)
                raise ValueError(
                    f"conflicting duplicate artifacts for {model}: {joined}"
                )
        preferred = min(
            paths,
            key=lambda path: (path.parent.name != model, str(path)),
        )
        selected.append((model, preferred))
    return selected


def _registry_parameter_count(model: str) -> int | None:
    entry = MODEL_REGISTRY.get(model, {})
    for key in ("non_embedding_params", "n0_params", "n_params"):
        value = entry.get(key)
        if value is not None:
            return int(value)
    return None


def _model_count(
    model: str, metadata: Mapping[str, object] | None = None
) -> tuple[int, str]:
    registered = _registry_parameter_count(model)
    if registered is not None:
        return registered, "analysis/model_registry.py"
    if model in DOCUMENTED_CHECKPOINT_PARAMS:
        return DOCUMENTED_CHECKPOINT_PARAMS[model], "documented_checkpoint_total_dict"
    if model in DOCUMENTED_NON_EMBEDDING_PARAMS:
        return DOCUMENTED_NON_EMBEDDING_PARAMS[model], "documented_non_embedding_dict"
    if metadata is not None and metadata.get("n_params") is not None:
        return int(metadata["n_params"]), "v6_fisher_meta_embedding_inclusive_fallback"
    raise ValueError(f"No parameter count is available for model {model!r}")


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_pruning_rows(base: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    notes: list[str] = []
    for model, path in _deduplicated_artifacts(base, "prune_losses.json"):
        if model not in V17_MODELS:
            notes.append(f"Excluded later campaign outside the V17 12-model panel: {path}")
            continue
        payload = _json_load(path)
        if "1.0" not in payload or not isinstance(payload["1.0"], dict):
            raise ValueError(f"pruning artifact lacks the 1.0 dense anchor: {path}")
        metadata_path = path.parent / "fisher_meta.json"
        metadata = _json_load(metadata_path) if metadata_path.is_file() else {}
        family = model_family(model, str(metadata.get("model", "")))
        n0, count_source = _model_count(model, metadata)
        dense = payload["1.0"]
        for density_text, losses in payload.items():
            if density_text.startswith("_"):
                notes.append(f"Ignored pruning metadata key {density_text!r} in {path}")
                continue
            if not isinstance(losses, dict):
                raise ValueError(f"invalid pruning loss row {density_text!r}: {path}")
            density = _finite_float(density_text, f"density in {path}")
            coordinates = coordinate_mapping("pruning", density)
            for capability in CAPABILITIES:
                if capability not in dense or capability not in losses:
                    continue
                delta = _finite_float(
                    losses[capability], "pruned loss"
                ) - _finite_float(dense[capability], "dense pruning loss")
                rows.append(
                    {
                        "method": "pruning",
                        "model": model,
                        "family": family,
                        "capability": capability,
                        **coordinates,
                        "delta_L_c": delta,
                        "dL_cap": "",
                        "delta_L_source_c": "",
                        "distilled_L_c": "",
                        "reference_L_c": _finite_float(
                            dense[capability], "dense pruning loss"
                        ),
                        "delta_reference": "own_dense",
                        "source_reference_status": "not_applicable",
                        "source_loss_path": "",
                        "model_size_params": n0,
                        "n0_params": n0,
                        "n0_model": model,
                        "n0_count_source": count_source,
                        "is_baseline": math.isclose(density, 1.0, abs_tol=1e-12),
                        "teacher": "",
                        "recipe": "",
                        "trace_examples_per_domain": "",
                        "run": "",
                        "source_path": _relative_path(path),
                    }
                )
    if not rows:
        notes.append(f"No pruning artifacts found below {base}")
    return rows, notes


def _load_quantization_rows(base: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    notes: list[str] = []
    for model, path in _deduplicated_artifacts(base, "quant_losses.json"):
        if model not in V17_MODELS:
            notes.append(f"Excluded later campaign outside the V17 12-model panel: {path}")
            continue
        payload = _json_load(path)
        if "dense" not in payload or not isinstance(payload["dense"], dict):
            raise ValueError(f"quantization artifact lacks the dense anchor: {path}")
        metadata_path = PRUNE_BASE / model / "fisher_meta.json"
        metadata = _json_load(metadata_path) if metadata_path.is_file() else {}
        family = model_family(model, str(metadata.get("model", "")))
        n0, count_source = _model_count(model, metadata)
        dense = payload["dense"]
        for bit_text, losses in payload.items():
            if bit_text.startswith("_"):
                notes.append(f"Ignored quantization metadata key {bit_text!r} in {path}")
                continue
            if not isinstance(losses, dict):
                raise ValueError(f"invalid quantization loss row {bit_text!r}: {path}")
            bits = (
                16.0
                if bit_text == "dense"
                else _finite_float(bit_text, f"bits in {path}")
            )
            coordinates = coordinate_mapping("quantization", bits)
            if bits not in V17_BITS:
                notes.append(f"Excluded later bit configuration outside the V17 ladder: {path}/{bit_text}")
                continue
            for capability in CAPABILITIES:
                if capability not in dense or capability not in losses:
                    continue
                delta = _finite_float(
                    losses[capability], "quantized loss"
                ) - _finite_float(dense[capability], "dense quantization loss")
                rows.append(
                    {
                        "method": "quantization",
                        "model": model,
                        "family": family,
                        "capability": capability,
                        **coordinates,
                        "delta_L_c": delta,
                        "dL_cap": "",
                        "delta_L_source_c": "",
                        "distilled_L_c": "",
                        "reference_L_c": _finite_float(
                            dense[capability], "dense quantization loss"
                        ),
                        "delta_reference": "own_dense",
                        "source_reference_status": "not_applicable",
                        "source_loss_path": "",
                        "model_size_params": n0,
                        "n0_params": n0,
                        "n0_model": model,
                        "n0_count_source": count_source,
                        "is_baseline": bit_text == "dense",
                        "teacher": "",
                        "recipe": "",
                        "trace_examples_per_domain": "",
                        "run": "",
                        "source_path": _relative_path(path),
                    }
                )
    if not rows:
        notes.append(f"No quantization artifacts found below {base}")
    return rows, notes


def _available_model_counts(
    pruning_rows: Sequence[Mapping[str, object]]
) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    for row in pruning_rows:
        result[str(row["model"])] = (
            int(row["model_size_params"]),
            str(row["n0_count_source"]),
        )
    for model in MODEL_REGISTRY:
        try:
            result.setdefault(model, _model_count(model))
        except ValueError:
            pass
    return result


def source_referenced_deltas(
    distilled_losses: Mapping[str, object],
    source_dense_losses: Mapping[str, object],
) -> dict[str, float]:
    """Subtract a dense source loss from a distilled-student loss by capability."""
    deltas: dict[str, float] = {}
    for capability in CAPABILITIES:
        if capability not in distilled_losses or capability not in source_dense_losses:
            continue
        deltas[capability] = _finite_float(
            distilled_losses[capability], f"distilled {capability} loss"
        ) - _finite_float(
            source_dense_losses[capability], f"source dense {capability} loss"
        )
    return deltas


def _distilled_loss_block(
    payload: Mapping[str, object], path: Path
) -> Mapping[str, object]:
    """Return V16's compressed/distilled teacher-forced capability losses."""
    compressed = payload.get("compressed")
    if not isinstance(compressed, Mapping):
        compressed = payload.get("distilled")
    if not isinstance(compressed, Mapping) or not isinstance(
        compressed.get("L_c"), Mapping
    ):
        raise ValueError(f"distillation artifact lacks compressed L_c: {path}")
    losses = compressed["L_c"]
    assert isinstance(losses, Mapping)
    return losses


def _preferred_source_runs(paths: Sequence[Path]) -> dict[str, Path]:
    """Choose one registered full-budget source run per student.

    Some students have both API-teacher variants at ``full_600``.  Both
    artifact layouts are valid inputs, but a size ladder must contain one
    point per student.  Prefer the ladder's registered teacher and use a
    deterministic path tie-breaker when only another teacher is available.
    """
    candidates: dict[str, list[tuple[bool, Path]]] = defaultdict(list)
    for path in paths:
        payload = _json_load(path)
        student = canonical_model_tag(
            str(
                payload.get("student_tag")
                or payload.get("student")
                or path.parents[1].name
            )
        )
        family = model_family(student, str(payload.get("resolved_student", "")))
        ladder = DISTILLATION_SOURCE_LADDERS.get(family)
        registered_students = (
            {str(value) for value in ladder.get("students", ())}
            if ladder
            else set()
        )
        if ladder is None or student not in registered_students:
            continue
        run_suffix = str(ladder.get("run_suffix", ""))
        if not path.parent.name.endswith(run_suffix):
            continue
        preferred_teacher = str(ladder.get("preferred_teacher", ""))
        teacher = str(payload.get("teacher") or "")
        is_preferred = teacher == preferred_teacher or path.parent.name.startswith(
            f"{preferred_teacher}_"
        )
        candidates[student].append((is_preferred, path))

    selected: dict[str, Path] = {}
    for student, student_candidates in candidates.items():
        selected[student] = min(
            student_candidates,
            key=lambda item: (not item[0], str(item[1])),
        )[1]
    return selected


def _load_distillation_rows(
    base: Path,
    pruning_rows: Sequence[Mapping[str, object]],
    prune_base: Path = PRUNE_BASE,
) -> tuple[list[dict], list[str], dict[str, str], dict[str, object]]:
    rows: list[dict] = []
    notes: list[str] = []
    normalization_references: dict[str, str] = {}
    counts = _available_model_counts(pruning_rows)
    paths = sorted(base.glob("*/*/residual.json")) if base.is_dir() else []
    preferred_source_runs = _preferred_source_runs(paths)

    discovered_students: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        payload = _json_load(path)
        student = canonical_model_tag(
            str(
                payload.get("student_tag")
                or payload.get("student")
                or path.parents[1].name
            )
        )
        discovered_students[
            model_family(student, str(payload.get("resolved_student", "")))
        ].add(student)

    prune_artifacts = dict(
        _deduplicated_artifacts(Path(prune_base), "prune_losses.json")
    )
    source_cache: dict[str, tuple[Mapping[str, object], Path]] = {}
    for ladder in DISTILLATION_SOURCE_LADDERS.values():
        source_model = str(ladder["source_model"])
        source_path = prune_artifacts.get(source_model)
        if source_path is None:
            notes.append(
                f"Source reference unavailable for {source_model}: no prune_losses.json"
            )
            continue
        source_payload = _json_load(source_path)
        dense = source_payload.get("1.0")
        if not isinstance(dense, Mapping):
            notes.append(
                f"Source reference unavailable for {source_model}: missing dense key '1.0'"
            )
            continue
        source_cache[source_model] = (dense, source_path)

    source_students: set[str] = set()
    for path in paths:
        payload = _json_load(path)
        student = canonical_model_tag(
            str(
                payload.get("student_tag")
                or payload.get("student")
                or path.parents[1].name
            )
        )
        family = model_family(student, str(payload.get("resolved_student", "")))
        if student not in counts:
            try:
                counts[student] = _model_count(student)
            except ValueError as exc:
                notes.append(f"Skipped {path}: {exc}")
                continue
        family_models = {
            model: count
            for model, (count, _) in counts.items()
            if model_family(model) == family
        }
        if not family_models:
            notes.append(
                f"Skipped {path}: no dense reference model for family {family}"
            )
            continue
        ladder = DISTILLATION_SOURCE_LADDERS.get(family)
        if ladder is not None:
            n0_model = str(ladder["source_model"])
            if n0_model not in counts:
                notes.append(
                    f"Skipped {path}: no parameter count for registered source {n0_model}"
                )
                continue
        else:
            n0_model = max(
                family_models, key=lambda model: (family_models[model], model)
            )
        n0, n0_source = counts[n0_model]
        student_size, _ = counts[student]
        coordinates = coordinate_mapping(DISTILL_SELF, student_size, n0_params=n0)
        normalization_references[family] = n0_model
        deltas = payload.get("deltas")
        if not isinstance(deltas, dict):
            raise ValueError(f"distillation artifact lacks deltas: {path}")
        distilled_losses = _distilled_loss_block(payload, path)

        run_suffix = str(ladder.get("run_suffix", "")) if ladder else ""
        registered_students = (
            {str(value) for value in ladder.get("students", ())} if ladder else set()
        )
        source_eligible = bool(
            ladder
            and student in registered_students
            and preferred_source_runs.get(student) == path
            and n0_model in source_cache
        )
        if source_eligible:
            source_losses, source_loss_path = source_cache[n0_model]
            source_deltas = source_referenced_deltas(distilled_losses, source_losses)
            source_status = (
                f"source_referenced_{len(registered_students)}_point_ladder"
            )
        else:
            source_losses = {}
            source_loss_path = Path()
            source_deltas = {}
            if (
                ladder
                and student in registered_students
                and path.parent.name.endswith(run_suffix)
            ):
                source_status = "self_only_nonpreferred_source_run"
            elif ladder and student in registered_students:
                source_status = "self_only_not_full_600"
            elif ladder:
                source_status = "self_only_not_registered_source_student"
            elif len(discovered_students[family]) == 1:
                source_status = "single_point_no_clean_same_family_source"
            else:
                source_status = "no_registered_clean_same_family_source_ladder"

        for capability in CAPABILITIES:
            if capability not in deltas or not isinstance(deltas[capability], dict):
                continue
            if "dL_cap" not in deltas[capability]:
                raise ValueError(
                    f"distillation delta lacks dL_cap for {capability}: {path}"
                )
            self_delta = _finite_float(
                deltas[capability]["dL_cap"], "style-residualized dL_cap"
            )
            distilled_loss = _finite_float(
                distilled_losses[capability], "distilled capability loss"
            )
            source_delta = source_deltas.get(capability)
            common = {
                "model": student,
                "family": family,
                "capability": capability,
                **coordinates,
                "dL_cap": self_delta,
                "delta_L_source_c": source_delta if source_delta is not None else "",
                "distilled_L_c": distilled_loss,
                "model_size_params": student_size,
                "n0_params": n0,
                "n0_model": n0_model,
                "n0_count_source": n0_source,
                "is_baseline": False,
                "teacher": str(payload.get("teacher") or ""),
                "recipe": str(payload.get("recipe") or ""),
                "trace_examples_per_domain": payload.get("n_per_domain", ""),
                "run": str(payload.get("run_name") or path.parent.name),
                "source_path": _relative_path(path),
                "source_reference_status": source_status,
            }
            rows.append(
                {
                    **common,
                    "method": DISTILL_SELF,
                    "delta_L_c": self_delta,
                    "reference_L_c": _finite_float(
                        payload["dense"]["L_c"][capability],
                        "student dense capability loss",
                    ),
                    "delta_reference": "student_own_dense_style_residualized",
                    "source_loss_path": (
                        _relative_path(source_loss_path) if source_eligible else ""
                    ),
                }
            )
            if source_delta is not None:
                source_students.add(student)
                rows.append(
                    {
                        **common,
                        "method": DISTILL_SOURCE,
                        "delta_L_c": source_delta,
                        "reference_L_c": _finite_float(
                            source_losses[capability], "source dense capability loss"
                        ),
                        "delta_reference": "same_family_source_dense",
                        "source_loss_path": _relative_path(source_loss_path),
                    }
                )
    if not rows:
        notes.append(f"No usable distillation residual artifacts found below {base}")
    source_audit: dict[str, object] = {
        "source_students": sorted(source_students),
        "n_source_students": len(source_students),
        "registered_ladders": {
            family: {
                "source_model": str(ladder["source_model"]),
                "students": list(ladder["students"]),
                "n_points": len(ladder["students"]),
            }
            for family, ladder in DISTILLATION_SOURCE_LADDERS.items()
        },
        "family_status": {
            family: (
                "source_referenced_4_point_ladder"
                if family in DISTILLATION_SOURCE_LADDERS
                else "single_point_no_clean_same_family_source"
                if len(students) == 1
                else "no_registered_clean_same_family_source_ladder"
            )
            for family, students in sorted(discovered_students.items())
        },
        "selected_source_runs": {
            student: _relative_path(path)
            for student, path in sorted(preferred_source_runs.items())
        },
    }
    return rows, notes, normalization_references, source_audit


def assemble_table(
    prune_base: Path = PRUNE_BASE,
    quant_base: Path = QUANT_BASE,
    distill_base: Path = DISTILL_BASE,
) -> tuple[list[dict], dict[str, object]]:
    """Load all three methods and return the sorted long table plus audit."""
    pruning, prune_notes = _load_pruning_rows(Path(prune_base))
    quantization, quant_notes = _load_quantization_rows(Path(quant_base))
    distillation, distill_notes, references, source_audit = _load_distillation_rows(
        Path(distill_base), pruning, Path(prune_base)
    )
    rows = pruning + quantization + distillation
    rows.sort(
        key=lambda row: (
            METHOD_ORDER[str(row["method"])],
            str(row["family"]),
            str(row["model"]),
            str(row["capability"]),
            -float(row["r_storage"]),
            str(row["run"]),
        )
    )
    students = sorted(
        {str(row["model"]) for row in distillation if row["method"] == DISTILL_SELF}
    )
    source_students = sorted(
        {str(row["model"]) for row in distillation if row["method"] == DISTILL_SOURCE}
    )
    audit = {
        "notes": prune_notes + quant_notes + distill_notes,
        "distillation_students": students,
        "n_distillation_students": len(students),
        "distillation_reference_models": references,
        "distillation_source_students": source_students,
        "n_distillation_source_students": len(source_students),
        "distillation_source_audit": source_audit,
        "parameter_count_fallback_rows": sum(
            "embedding_inclusive_fallback" in str(row["n0_count_source"])
            for row in rows
        ),
    }
    return rows, audit


def apply_precliff_filter(
    rows: Sequence[Mapping[str, object]],
    cap: float = DEFAULT_PRECLIFF_CAP,
) -> tuple[list[dict], dict[str, object]]:
    """Annotate cells and identify the perturbative prefix of each trajectory.

    Trajectories are capability-specific because the paper's cliff is a
    capability-conditioned event.  At a repeated coordinate (currently only
    possible for distillation recipes), one cap crossing marks the entire
    coordinate as the cliff; ordering recipes at identical compression would
    be arbitrary.  The crossing coordinate itself is post-cliff.
    """
    cap = float(cap)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("pre-cliff cap must be a finite positive number")

    annotated = [dict(row) for row in rows]
    trajectories: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(annotated):
        row["precliff_cap"] = cap
        row["cliff_r_storage"] = ""
        row["overlap_method_count"] = ""
        row["overlap_methods"] = ""
        if bool(row.get("is_baseline", False)):
            row["precliff_status"] = "baseline"
            continue
        key = (str(row["model"]), str(row["method"]), str(row["capability"]))
        trajectories[key].append(index)

    cliff_by_trajectory: dict[tuple[str, str, str], float | None] = {}
    for key, indices in trajectories.items():
        by_coordinate: dict[float, list[int]] = defaultdict(list)
        for index in indices:
            by_coordinate[float(annotated[index]["r_storage"])].append(index)

        crossed = False
        cliff_coordinate: float | None = None
        for coordinate in sorted(by_coordinate, reverse=True):
            coordinate_indices = by_coordinate[coordinate]
            if not crossed and any(
                float(annotated[index]["delta_L_c"]) > cap
                for index in coordinate_indices
            ):
                crossed = True
                cliff_coordinate = coordinate
            status = "post_cliff" if crossed else "pre_cliff"
            for index in coordinate_indices:
                annotated[index]["precliff_status"] = status
        cliff_by_trajectory[key] = cliff_coordinate
        if cliff_coordinate is not None:
            for index in indices:
                annotated[index]["cliff_r_storage"] = cliff_coordinate

    candidate_counts = Counter(
        str(row["method"])
        for row in annotated
        if not bool(row.get("is_baseline", False))
    )
    precliff_counts = Counter(
        str(row["method"])
        for row in annotated
        if row.get("precliff_status") == "pre_cliff"
    )
    postcliff_counts = Counter(
        str(row["method"])
        for row in annotated
        if row.get("precliff_status") == "post_cliff"
    )
    total_candidates = sum(candidate_counts.values())
    total_postcliff = sum(postcliff_counts.values())
    audit = {
        "cap": cap,
        "grouping": ["model", "method", "capability"],
        "crossing_rule": "first delta_L_c > cap toward lower r_storage; crossing included as post-cliff",
        "n_trajectories": len(trajectories),
        "n_trajectories_with_cliff": sum(
            value is not None for value in cliff_by_trajectory.values()
        ),
        "candidate_cells": dict(sorted(candidate_counts.items())),
        "precliff_cells": dict(sorted(precliff_counts.items())),
        "postcliff_cells": dict(sorted(postcliff_counts.items())),
        "total_candidate_cells": total_candidates,
        "total_precliff_cells": total_candidates - total_postcliff,
        "total_postcliff_cells": total_postcliff,
        "postcliff_fraction": (
            total_postcliff / total_candidates if total_candidates else 0.0
        ),
    }
    return annotated, audit


def _analysis_rows(rows: Sequence[dict], include_distillation: bool) -> list[dict]:
    """Return only non-baseline perturbative cells eligible for any fit."""
    eligible_methods = {"pruning", "quantization"}
    if include_distillation:
        eligible_methods.add(DISTILL_SOURCE)
    return [
        row
        for row in rows
        if not bool(row["is_baseline"])
        and row.get("precliff_status") == "pre_cliff"
        and row["method"] in eligible_methods
    ]


def overlap_aware_rows(
    rows: Sequence[dict],
    tolerance: float = DEFAULT_STORAGE_MATCH_TOLERANCE,
) -> tuple[list[dict], dict[str, object]]:
    """Return pre-cliff rows with nearby storage support from another method."""
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("storage match tolerance must be finite and positive")

    method_coordinates: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        method_coordinates[str(row["method"])].append(float(row["r_storage"]))
    for method in method_coordinates:
        method_coordinates[method] = sorted(set(method_coordinates[method]))

    selected: list[dict] = []
    comparable_counts: Counter[str] = Counter()
    for row in rows:
        method = str(row["method"])
        coordinate = float(row["r_storage"])
        comparable = sorted(
            other
            for other, coordinates in method_coordinates.items()
            if other != method
            and any(
                abs(coordinate - candidate) < tolerance for candidate in coordinates
            )
        )
        row["overlap_method_count"] = 1 + len(comparable) if comparable else 1
        row["overlap_methods"] = ";".join([method, *comparable])
        if comparable:
            selected.append(row)
            comparable_counts[method] += 1

    ranges = {
        method: {
            "min": min(coordinates),
            "max": max(coordinates),
            "n_distinct_coordinates": len(coordinates),
        }
        for method, coordinates in sorted(method_coordinates.items())
        if coordinates
    }
    pairwise: dict[str, dict[str, object]] = {}
    present_methods = sorted(ranges, key=lambda method: METHOD_ORDER.get(method, 99))
    for left_index, left in enumerate(present_methods):
        for right in present_methods[left_index + 1 :]:
            low = max(float(ranges[left]["min"]), float(ranges[right]["min"]))
            high = min(float(ranges[left]["max"]), float(ranges[right]["max"]))
            pairwise[f"{left}__{right}"] = {
                "continuous_range_intersection": [low, high] if low <= high else None,
                "has_comparable_observed_coordinates": any(
                    abs(a - b) < tolerance
                    for a in method_coordinates[left]
                    for b in method_coordinates[right]
                ),
            }
    all_method_intersection: list[float] | None = None
    if len(ranges) >= 2:
        low = max(float(value["min"]) for value in ranges.values())
        high = min(float(value["max"]) for value in ranges.values())
        if low <= high:
            all_method_intersection = [low, high]

    audit = {
        "tolerance": tolerance,
        "comparison_rule": "at least one other method has pre-cliff data with absolute r_storage difference < tolerance",
        "ranges": ranges,
        "pairwise": pairwise,
        "all_method_continuous_intersection": all_method_intersection,
        "n_comparable_rows": len(selected),
        "comparable_rows_by_method": dict(sorted(comparable_counts.items())),
    }
    return selected, audit


def _shared_basis(r_storage: np.ndarray, alpha: float) -> np.ndarray:
    safe = np.clip(np.asarray(r_storage, dtype=np.float64), 1e-12, 1.0)
    return np.exp(np.clip(-alpha * np.log(safe), -100.0, 100.0)) - 1.0


def _effect_design(levels: Sequence[str], values: Sequence[str]) -> np.ndarray:
    """Zero-sum/effect-coded design; unseen values receive zero correction."""
    if len(levels) <= 1:
        return np.zeros((len(values), 0), dtype=np.float64)
    columns = {level: index for index, level in enumerate(levels[:-1])}
    reference = levels[-1]
    design = np.zeros((len(values), len(levels) - 1), dtype=np.float64)
    for row_index, value in enumerate(values):
        if value in columns:
            design[row_index, columns[value]] = 1.0
        elif value == reference:
            design[row_index, :] = -1.0
        # A level absent from training gets the population-average effect zero.
    return design


def _effect_values(
    levels: Sequence[str], coefficients: Sequence[float]
) -> dict[str, float]:
    if not levels:
        return {}
    if len(levels) == 1:
        return {levels[0]: 0.0}
    values = {level: float(value) for level, value in zip(levels[:-1], coefficients)}
    values[levels[-1]] = -float(np.sum(coefficients))
    return values


def _fit_m1(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(rows) < 2:
        return {"status": "too_few_rows", "n_params": 2}
    r = np.asarray([row["r_storage"] for row in rows], dtype=np.float64)
    y = np.asarray([row["delta_L_c"] for row in rows], dtype=np.float64)
    scale = max(float(np.median(np.abs(y))), float(np.std(y)), 1e-6)
    h0 = _shared_basis(r, 1.0)
    a0 = float(np.dot(h0, y) / max(float(np.dot(h0, h0)), EPS))

    def residual(theta: np.ndarray) -> np.ndarray:
        return (theta[0] * _shared_basis(r, theta[1]) - y) / scale

    result = least_squares(
        residual,
        x0=np.array([a0, 1.0]),
        bounds=(np.array([-1e6 * scale, 0.05]), np.array([1e6 * scale, 8.0])),
        max_nfev=20_000,
    )
    return {
        "status": "ok" if result.success else "fit_failed",
        "n_params": 2,
        "parameters": {"A": float(result.x[0]), "alpha": float(result.x[1])},
        "optimizer_message": str(result.message),
    }


def _fit_m2(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(rows) < 2:
        return {"status": "too_few_rows", "n_params": None}
    methods = sorted({str(row["method"]) for row in rows})
    families = sorted({str(row["family"]) for row in rows})
    method_design = _effect_design(methods, [str(row["method"]) for row in rows])
    family_design = _effect_design(families, [str(row["family"]) for row in rows])
    r = np.asarray([row["r_storage"] for row in rows], dtype=np.float64)
    y = np.asarray([row["delta_L_c"] for row in rows], dtype=np.float64)
    scale = max(float(np.median(np.abs(y))), float(np.std(y)), 1e-6)
    n_effects = method_design.shape[1] + family_design.shape[1]
    h0 = _shared_basis(r, 1.0)
    a0 = float(np.dot(h0, y) / max(float(np.dot(h0, h0)), EPS))

    def residual(theta: np.ndarray) -> np.ndarray:
        cursor = 2
        prediction = theta[0] * _shared_basis(r, theta[1])
        if method_design.shape[1]:
            prediction = (
                prediction
                + method_design @ theta[cursor : cursor + method_design.shape[1]]
            )
            cursor += method_design.shape[1]
        if family_design.shape[1]:
            prediction = (
                prediction
                + family_design @ theta[cursor : cursor + family_design.shape[1]]
            )
        return (prediction - y) / scale

    lower = np.concatenate(
        [np.array([-1e6 * scale, 0.05]), np.full(n_effects, -1e6 * scale)]
    )
    upper = np.concatenate(
        [np.array([1e6 * scale, 8.0]), np.full(n_effects, 1e6 * scale)]
    )
    result = least_squares(
        residual,
        x0=np.concatenate([np.array([a0, 1.0]), np.zeros(n_effects)]),
        bounds=(lower, upper),
        max_nfev=20_000,
    )
    cursor = 2
    method_coefficients = result.x[cursor : cursor + method_design.shape[1]]
    cursor += method_design.shape[1]
    family_coefficients = result.x[cursor : cursor + family_design.shape[1]]
    return {
        "status": "ok" if result.success else "fit_failed",
        "n_params": int(2 + n_effects),
        "parameters": {
            "A": float(result.x[0]),
            "alpha": float(result.x[1]),
            "method_offsets": _effect_values(methods, method_coefficients),
            "family_offsets": _effect_values(families, family_coefficients),
            "unseen_group_offset": 0.0,
        },
        "optimizer_message": str(result.message),
    }


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    x = np.clip(np.asarray(value, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _fit_m3_pruning(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(rows) < 3:
        return {"status": "too_few_rows", "n_params": 2}
    r = np.asarray([row["r_storage"] for row in rows], dtype=np.float64)
    y = np.asarray([row["delta_L_c"] for row in rows], dtype=np.float64)
    x = 1.0 - r
    scale = max(float(np.median(np.abs(y))), float(np.std(y)), 1e-6)
    positive = y[y > 0.0]
    amplitude = float(np.median(positive)) if positive.size else scale

    def prediction(theta: np.ndarray) -> np.ndarray:
        amplitude_value, gamma = theta
        return amplitude_value * np.power(np.clip(x, 0.0, 1.0), gamma)

    def residual(theta: np.ndarray) -> np.ndarray:
        return (prediction(theta) - y) / scale

    result = least_squares(
        residual,
        x0=np.array([amplitude, 1.5]),
        bounds=(
            np.array([-1e4 * scale, 0.1]),
            np.array([1e4 * scale, 8.0]),
        ),
        max_nfev=30_000,
    )
    names = ("smooth_amplitude", "gamma")
    return {
        "status": "ok" if result.success else "fit_failed",
        "n_params": 2,
        "parameters": {name: float(value) for name, value in zip(names, result.x)},
        "optimizer_message": str(result.message),
    }


def _quant_bases(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    smooth = np.power(4.0, 4.0 - bits) - np.power(4.0, 4.0 - 16.0)
    cliff = _sigmoid((3.5 - bits) / 0.15) - _sigmoid((3.5 - 16.0) / 0.15)
    return smooth, cliff


def _fit_m3_quantization(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(rows) < 2:
        return {"status": "too_few_rows", "n_params": 1}
    bits = np.asarray([row["raw_coordinate"] for row in rows], dtype=np.float64)
    y = np.asarray([row["delta_L_c"] for row in rows], dtype=np.float64)
    smooth, _ = _quant_bases(bits)
    denominator = float(np.dot(smooth, smooth))
    if denominator <= EPS:
        return {"status": "rank_deficient", "n_params": 1}
    coefficient = float(np.dot(smooth, y) / denominator)
    return {
        "status": "ok",
        "n_params": 1,
        "parameters": {
            "q_at_4bit": coefficient,
        },
    }


def _fit_m3_distillation(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    r = np.asarray([row["r_storage"] for row in rows], dtype=np.float64)
    y = np.asarray([row["delta_L_c"] for row in rows], dtype=np.float64)
    if len(rows) < 2 or np.unique(np.round(r, 15)).size < 2:
        return {"status": "too_few_student_sizes", "n_params": 2}
    design = np.column_stack([np.ones_like(r), np.log(r)])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    log_size_slope = float(coefficients[1])
    return {
        "status": "ok",
        "n_params": 2,
        "n_student_sizes": int(np.unique(np.round(r, 15)).size),
        "fit_form": "capacity_floor - size_exponent * log(r_storage)",
        "parameters": {
            "capacity_floor": float(coefficients[0]),
            "size_exponent": -log_size_slope,
            # Retained as an explicit signed coefficient for downstream
            # compatibility: log_size_slope == -size_exponent.
            "log_size_slope": log_size_slope,
        },
    }


def _fit_m3(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    fitters: dict[
        str, Callable[[Sequence[Mapping[str, object]]], dict[str, object]]
    ] = {
        "pruning": _fit_m3_pruning,
        "quantization": _fit_m3_quantization,
        DISTILL_SOURCE: _fit_m3_distillation,
    }
    fits: dict[str, dict[str, object]] = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        if method_rows:
            fits[method] = fitters[method](method_rows)
    usable_params = sum(
        int(fit["n_params"]) for fit in fits.values() if fit.get("status") == "ok"
    )
    status = (
        "ok"
        if fits and all(fit.get("status") == "ok" for fit in fits.values())
        else "partial"
    )
    return {"status": status, "n_params": usable_params, "method_fits": fits}


def fit_hypothesis(
    hypothesis: str, rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    if hypothesis == "M1":
        return {"hypothesis": hypothesis, **_fit_m1(rows)}
    if hypothesis == "M2":
        return {"hypothesis": hypothesis, **_fit_m2(rows)}
    if hypothesis == "M3":
        return {"hypothesis": hypothesis, **_fit_m3(rows)}
    raise ValueError(f"unknown hypothesis {hypothesis!r}")


def predict_hypothesis(
    fit: Mapping[str, object], row: Mapping[str, object]
) -> float | None:
    """Predict one row, returning None when a method-specific law is absent."""
    if fit.get("status") not in {"ok", "partial"}:
        return None
    hypothesis = fit["hypothesis"]
    r = float(row["r_storage"])
    if hypothesis in {"M1", "M2"}:
        parameters = fit["parameters"]
        assert isinstance(parameters, Mapping)
        value = float(parameters["A"]) * float(
            _shared_basis(np.array([r]), float(parameters["alpha"]))[0]
        )
        if hypothesis == "M2":
            method_offsets = parameters["method_offsets"]
            family_offsets = parameters["family_offsets"]
            assert isinstance(method_offsets, Mapping) and isinstance(
                family_offsets, Mapping
            )
            value += float(method_offsets.get(str(row["method"]), 0.0))
            value += float(family_offsets.get(str(row["family"]), 0.0))
        return value
    method = str(row["method"])
    method_fits = fit.get("method_fits", {})
    assert isinstance(method_fits, Mapping)
    method_fit = method_fits.get(method)
    if not isinstance(method_fit, Mapping) or method_fit.get("status") != "ok":
        return None
    return _predict_m3_method(method, method_fit, row)


def _predict_m3_method(
    method: str,
    method_fit: Mapping[str, object],
    row: Mapping[str, object],
) -> float | None:
    """Predict one row from one method-specific perturbative branch."""
    if method_fit.get("status") != "ok":
        return None
    parameters = method_fit["parameters"]
    assert isinstance(parameters, Mapping)
    if method == "pruning":
        r = float(row["r_storage"])
        x = 1.0 - r
        a = float(parameters["smooth_amplitude"])
        gamma = float(parameters["gamma"])
        return float(a * max(x, 0.0) ** gamma)
    if method == "quantization":
        bits = np.array([float(row["raw_coordinate"])])
        smooth, _ = _quant_bases(bits)
        return float(float(parameters["q_at_4bit"]) * smooth[0])
    if method in {"distillation", DISTILL_SOURCE}:
        r = float(row["r_storage"])
        if "size_exponent" in parameters:
            return float(
                float(parameters["capacity_floor"])
                - float(parameters["size_exponent"]) * math.log(r)
            )
        return float(
            float(parameters["capacity_floor"])
            + float(parameters["log_size_slope"]) * math.log(r)
        )
    return None


def _metrics(
    records: Sequence[Mapping[str, object]], total_cells: int
) -> dict[str, object]:
    if not records:
        return {
            "n_cells": 0,
            "total_test_cells": total_cells,
            "coverage": 0.0,
            "mae": None,
            "sign_accuracy": None,
            "n_sign_correct": 0,
        }
    observed = np.asarray([record["observed"] for record in records], dtype=np.float64)
    predicted = np.asarray(
        [record["predicted"] for record in records], dtype=np.float64
    )
    correct = int(np.sum(np.sign(observed) == np.sign(predicted)))
    return {
        "n_cells": len(records),
        "total_test_cells": total_cells,
        "coverage": len(records) / total_cells if total_cells else 0.0,
        "mae": float(np.mean(np.abs(predicted - observed))),
        "sign_accuracy": correct / len(records),
        "n_sign_correct": correct,
    }


def _prediction_record(
    row: Mapping[str, object], predicted: float, held_out: str
) -> dict[str, object]:
    return {
        "held_out": held_out,
        "method": row["method"],
        "model": row["model"],
        "family": row["family"],
        "capability": row["capability"],
        "r_storage": row["r_storage"],
        "raw_coordinate": row["raw_coordinate"],
        "run": row["run"],
        "observed": float(row["delta_L_c"]),
        "predicted": float(predicted),
        "absolute_error": abs(float(predicted) - float(row["delta_L_c"])),
        "sign_correct": bool(
            np.sign(float(predicted)) == np.sign(float(row["delta_L_c"]))
        ),
    }


def evaluate_splits(
    hypothesis: str,
    rows: Sequence[dict],
    split_name: str,
    splits: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Fit on each training fold and pool only its genuinely held-out predictions."""
    all_predictions: list[dict[str, object]] = []
    folds = []
    total_test_cells = 0
    for split in splits:
        train = [rows[int(index)] for index in split["train_indices"]]
        test = [rows[int(index)] for index in split["test_indices"]]
        held_out = str(split["held_out"])
        total_test_cells += len(test)
        fit = fit_hypothesis(hypothesis, train)
        predictions = []
        for row in test:
            prediction = predict_hypothesis(fit, row)
            if prediction is not None and math.isfinite(prediction):
                predictions.append(_prediction_record(row, prediction, held_out))
        all_predictions.extend(predictions)
        fold_metrics = _metrics(predictions, len(test))
        fold_status = (
            "ok"
            if fold_metrics["coverage"] == 1.0
            else ("not_estimable" if not predictions else "partial")
        )
        folds.append(
            {
                "held_out": held_out,
                **({"family": split["family"]} if "family" in split else {}),
                "n_train": len(train),
                "n_test": len(test),
                "status": fold_status,
                "fit": fit,
                "metrics": fold_metrics,
                "predictions": predictions,
            }
        )
    aggregate = _metrics(all_predictions, total_test_cells)
    status = (
        "ok"
        if aggregate["coverage"] == 1.0
        else ("not_estimable" if not all_predictions else "partial")
    )
    return {
        "protocol": split_name,
        "status": status,
        "metrics": aggregate,
        "folds": folds,
    }


def _in_sample_result(hypothesis: str, rows: Sequence[dict]) -> dict[str, object]:
    fit = fit_hypothesis(hypothesis, rows)
    predictions = []
    for row in rows:
        prediction = predict_hypothesis(fit, row)
        if prediction is not None and math.isfinite(prediction):
            predictions.append(_prediction_record(row, prediction, "in_sample"))
    return {
        "label": "NON-DECISIONAL in-sample fit",
        "fit": fit,
        "metrics": _metrics(predictions, len(rows)),
    }


def run_evaluation(
    rows: Sequence[dict], *, scope: str = "all_precliff"
) -> dict[str, dict[str, object]]:
    protocols = {
        "leave_one_method_out": leave_one_out_splits(rows, "method"),
        "leave_one_family_out": leave_one_out_splits(rows, "family"),
        "leave_largest_model_out": leave_largest_model_out_splits(rows),
    }
    outputs = {}
    for hypothesis in ("M1", "M2", "M3"):
        outputs[hypothesis] = {
            "hypothesis": hypothesis,
            "evaluation_scope": scope,
            "n_rows": len(rows),
            "form": {
                "M1": "A * (r_storage^(-alpha) - 1); no capability scale",
                "M2": "M1 + zero-sum method offset + zero-sum family offset",
                "M3": "separate perturbative pruning/quantization branches and source-referenced distillation capacity reduction",
            }[hypothesis],
            "selection_metric": "held-out MAE only",
            "full_data_parameter_count": fit_hypothesis(hypothesis, rows).get(
                "n_params"
            ),
            "held_out": {
                name: evaluate_splits(hypothesis, rows, name, splits)
                for name, splits in protocols.items()
            },
            "non_decisional_in_sample": _in_sample_result(hypothesis, rows),
        }
    return outputs


def _write_table(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def matched_storage_pairs(
    rows: Sequence[Mapping[str, object]],
    tolerance: float = DEFAULT_STORAGE_MATCH_TOLERANCE,
) -> list[dict[str, object]]:
    """Greedily form one-to-one pruning/quantization pairs within tolerance.

    Pairing is performed separately for each model and capability.  Candidate
    edges are consumed in increasing absolute storage gap, so an observation
    cannot be duplicated merely because a tolerance window contains two grid
    points.  Exact matches are therefore always preferred.
    """
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("storage match tolerance must be finite and positive")
    grouped: dict[tuple[str, str], dict[str, list[Mapping[str, object]]]] = defaultdict(
        lambda: {"pruning": [], "quantization": []}
    )
    for row in rows:
        method = str(row["method"])
        if method not in {"pruning", "quantization"} or bool(
            row.get("is_baseline", False)
        ):
            continue
        grouped[(str(row["model"]), str(row["capability"]))][method].append(row)

    pairs: list[dict[str, object]] = []
    for (model, capability), methods in sorted(grouped.items()):
        pruning_rows = sorted(
            methods["pruning"], key=lambda row: float(row["r_storage"])
        )
        quantization_rows = sorted(
            methods["quantization"], key=lambda row: float(row["r_storage"])
        )
        candidates = []
        for p_index, pruning in enumerate(pruning_rows):
            for q_index, quantization in enumerate(quantization_rows):
                gap = abs(
                    float(pruning["r_storage"]) - float(quantization["r_storage"])
                )
                if gap < tolerance:
                    # Round only the ordering key so mathematically tied grid
                    # distances do not depend on binary floating-point noise.
                    # Prefer the less-compressed pruning point on a true tie;
                    # this is conservative for the method-gap comparison.
                    candidates.append(
                        (
                            round(gap, 12),
                            -float(pruning["r_storage"]),
                            p_index,
                            q_index,
                            gap,
                        )
                    )
        candidates.sort()
        used_pruning: set[int] = set()
        used_quantization: set[int] = set()
        for _, _, p_index, q_index, gap in candidates:
            if p_index in used_pruning or q_index in used_quantization:
                continue
            used_pruning.add(p_index)
            used_quantization.add(q_index)
            pruning = pruning_rows[p_index]
            quantization = quantization_rows[q_index]
            pruning_delta = float(pruning["delta_L_c"])
            quantization_delta = float(quantization["delta_L_c"])
            pruning_storage = float(pruning["r_storage"])
            quantization_storage = float(quantization["r_storage"])
            pruning_status = str(pruning.get("precliff_status", "unmarked"))
            quantization_status = str(quantization.get("precliff_status", "unmarked"))
            pairs.append(
                {
                    "model": model,
                    "family": str(pruning.get("family") or model_family(model)),
                    "capability": capability,
                    "pruning_r_storage": pruning_storage,
                    "quantization_r_storage": quantization_storage,
                    "delta_r_storage": gap,
                    "pruning_raw_coordinate": float(pruning["raw_coordinate"]),
                    "quantization_raw_coordinate": float(
                        quantization["raw_coordinate"]
                    ),
                    "pruning_delta_L_c": pruning_delta,
                    "quantization_delta_L_c": quantization_delta,
                    "signed_method_gap": pruning_delta - quantization_delta,
                    "absolute_gap": abs(pruning_delta - quantization_delta),
                    "sign_agreement": bool(
                        np.sign(pruning_delta) == np.sign(quantization_delta)
                    ),
                    "pruning_precliff_status": pruning_status,
                    "quantization_precliff_status": quantization_status,
                    "both_precliff": bool(
                        pruning_status == "pre_cliff"
                        and quantization_status == "pre_cliff"
                    ),
                }
            )
    return sorted(
        pairs,
        key=lambda pair: (
            str(pair["model"]),
            str(pair["capability"]),
            -float(pair["quantization_r_storage"]),
        ),
    )


def _matched_storage_pairs(
    rows: Sequence[Mapping[str, object]],
    tolerance: float = DEFAULT_STORAGE_MATCH_TOLERANCE,
) -> list[dict[str, object]]:
    """Backward-compatible private alias for tests and downstream notebooks."""
    return matched_storage_pairs(rows, tolerance)


MATCHED_AXIS_FIELDS = (
    "axis",
    "family",
    "source_model",
    "capability",
    "method_a",
    "method_b",
    "model_a",
    "model_b",
    "cost_a",
    "cost_b",
    "delta_cost",
    "r_storage_a",
    "r_storage_b",
    "r_active_a",
    "r_active_b",
    "raw_coordinate_a",
    "raw_coordinate_b",
    "delta_L_c_a",
    "delta_L_c_b",
    "delta_reference_a",
    "delta_reference_b",
    "signed_method_gap",
    "absolute_gap",
    "sign_agreement",
    "precliff_status_a",
    "precliff_status_b",
    "both_precliff",
)


def matched_axis_pairs(
    rows: Sequence[Mapping[str, object]],
    axis: str,
    tolerance: float = DEFAULT_STORAGE_MATCH_TOLERANCE,
) -> list[dict[str, object]]:
    """Match every cross-method pair on storage or active-parameter ratio.

    Distillation rows are grouped with pruning/quantization rows for their
    registered ``n0_model``.  This lets a 12B student sourced from Gemma-3 27B
    be compared with compression applied to that 27B source, rather than with
    an unrelated 12B model.  ``distill_self`` is deliberately ineligible.
    """
    if axis not in {"r_storage", "r_active"}:
        raise ValueError("matched axis must be 'r_storage' or 'r_active'")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("match tolerance must be finite and positive")

    grouped: dict[
        tuple[str, str, str], dict[str, list[Mapping[str, object]]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        method = str(row["method"])
        if method not in METHODS or bool(row.get("is_baseline", False)):
            continue
        source_model = (
            str(row["n0_model"]) if method == DISTILL_SOURCE else str(row["model"])
        )
        grouped[(str(row["family"]), source_model, str(row["capability"]))][
            method
        ].append(row)

    pairs: list[dict[str, object]] = []
    for (family, source_model, capability), by_method in sorted(grouped.items()):
        present = [method for method in METHODS if by_method.get(method)]
        for left_index, method_a in enumerate(present):
            for method_b in present[left_index + 1 :]:
                rows_a = sorted(by_method[method_a], key=lambda row: float(row[axis]))
                rows_b = sorted(by_method[method_b], key=lambda row: float(row[axis]))
                candidates = []
                for a_index, row_a in enumerate(rows_a):
                    for b_index, row_b in enumerate(rows_b):
                        gap = abs(float(row_a[axis]) - float(row_b[axis]))
                        if gap < tolerance:
                            candidates.append(
                                (
                                    round(gap, 12),
                                    -float(row_a[axis]),
                                    a_index,
                                    b_index,
                                    gap,
                                )
                            )
                candidates.sort()
                used_a: set[int] = set()
                used_b: set[int] = set()
                for _, _, a_index, b_index, gap in candidates:
                    if a_index in used_a or b_index in used_b:
                        continue
                    used_a.add(a_index)
                    used_b.add(b_index)
                    row_a = rows_a[a_index]
                    row_b = rows_b[b_index]
                    delta_a = float(row_a["delta_L_c"])
                    delta_b = float(row_b["delta_L_c"])
                    status_a = str(row_a.get("precliff_status", "unmarked"))
                    status_b = str(row_b.get("precliff_status", "unmarked"))
                    pairs.append(
                        {
                            "axis": axis,
                            "family": family,
                            "source_model": source_model,
                            "capability": capability,
                            "method_a": method_a,
                            "method_b": method_b,
                            "model_a": str(row_a["model"]),
                            "model_b": str(row_b["model"]),
                            "cost_a": float(row_a[axis]),
                            "cost_b": float(row_b[axis]),
                            "delta_cost": gap,
                            "r_storage_a": float(row_a["r_storage"]),
                            "r_storage_b": float(row_b["r_storage"]),
                            "r_active_a": float(row_a["r_active"]),
                            "r_active_b": float(row_b["r_active"]),
                            "raw_coordinate_a": float(row_a["raw_coordinate"]),
                            "raw_coordinate_b": float(row_b["raw_coordinate"]),
                            "delta_L_c_a": delta_a,
                            "delta_L_c_b": delta_b,
                            "delta_reference_a": str(row_a.get("delta_reference", "")),
                            "delta_reference_b": str(row_b.get("delta_reference", "")),
                            "signed_method_gap": delta_a - delta_b,
                            "absolute_gap": abs(delta_a - delta_b),
                            "sign_agreement": bool(
                                np.sign(delta_a) == np.sign(delta_b)
                            ),
                            "precliff_status_a": status_a,
                            "precliff_status_b": status_b,
                            "both_precliff": bool(
                                status_a == "pre_cliff" and status_b == "pre_cliff"
                            ),
                        }
                    )
    return sorted(
        pairs,
        key=lambda pair: (
            str(pair["family"]),
            str(pair["source_model"]),
            str(pair["capability"]),
            METHOD_ORDER[str(pair["method_a"])],
            METHOD_ORDER[str(pair["method_b"])],
            -float(pair["cost_b"]),
        ),
    )


def _write_matched_axis_csv(path: Path, pairs: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCHED_AXIS_FIELDS)
        writer.writeheader()
        writer.writerows(pairs)


def _write_matched_storage_outputs(
    out: Path,
    rows: Sequence[Mapping[str, object]],
    tolerance: float,
) -> dict[str, object]:
    # Retain the original pruning-vs-quantization view for its established
    # exact-half statistic and plot, while the CSV now covers every comparable
    # method and excludes the self-referenced distillation diagnostic.
    pruning_quant_pairs = matched_storage_pairs(rows, tolerance)
    pairs = matched_axis_pairs(rows, "r_storage", tolerance)
    active_pairs = matched_axis_pairs(rows, "r_active", tolerance)
    csv_path = out / "matched_storage.csv"
    active_csv_path = out / "matched_active.csv"
    _write_matched_axis_csv(csv_path, pairs)
    _write_matched_axis_csv(active_csv_path, active_pairs)
    legacy_csv_path = out / "matched_storage_pruning_vs_quant.csv"
    legacy_fields = (
        "model",
        "family",
        "capability",
        "pruning_r_storage",
        "quantization_r_storage",
        "delta_r_storage",
        "pruning_raw_coordinate",
        "quantization_raw_coordinate",
        "pruning_delta_L_c",
        "quantization_delta_L_c",
        "signed_method_gap",
        "absolute_gap",
        "sign_agreement",
        "pruning_precliff_status",
        "quantization_precliff_status",
        "both_precliff",
    )
    with legacy_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=legacy_fields)
        writer.writeheader()
        writer.writerows(pruning_quant_pairs)

    exact_half = [
        pair
        for pair in pruning_quant_pairs
        if math.isclose(float(pair["pruning_r_storage"]), 0.5, abs_tol=1e-12)
        and math.isclose(float(pair["quantization_r_storage"]), 0.5, abs_tol=1e-12)
    ]
    half_by_capability: dict[str, dict[str, object]] = {}
    for capability in CAPABILITIES:
        subset = [pair for pair in exact_half if pair["capability"] == capability]
        if not subset:
            continue
        half_by_capability[capability] = {
            "n_models": len(subset),
            "pruning_median_delta_L_c": float(
                np.median([pair["pruning_delta_L_c"] for pair in subset])
            ),
            "quantization_median_delta_L_c": float(
                np.median([pair["quantization_delta_L_c"] for pair in subset])
            ),
            "median_signed_method_gap": float(
                np.median([pair["signed_method_gap"] for pair in subset])
            ),
        }

    result: dict[str, object] = {
        "tolerance": tolerance,
        "n_pairs": len(pairs),
        "n_both_precliff_pairs": sum(bool(pair["both_precliff"]) for pair in pairs),
        "n_pruning_quantization_pairs": len(pruning_quant_pairs),
        "n_distill_source_pairs": sum(
            DISTILL_SOURCE in {str(pair["method_a"]), str(pair["method_b"])}
            for pair in pairs
        ),
        "n_active_pairs": len(active_pairs),
        "n_active_distill_source_pairs": sum(
            DISTILL_SOURCE in {str(pair["method_a"]), str(pair["method_b"])}
            for pair in active_pairs
        ),
        "nontrivial_baselines_excluded": True,
        "mean_absolute_method_gap": (
            float(np.mean([pair["absolute_gap"] for pair in pairs])) if pairs else None
        ),
        "sign_agreement": (
            float(np.mean([pair["sign_agreement"] for pair in pairs]))
            if pairs
            else None
        ),
        "exact_half_storage_n_pairs": len(exact_half),
        "exact_half_storage_by_capability": half_by_capability,
        "csv": csv_path.name,
        "active_csv": active_csv_path.name,
        "pruning_quantization_csv": legacy_csv_path.name,
        "figure": "matched_storage.png",
    }

    # Import plotting only for the full analysis path; tests and --dry-run stay light.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/scaling-down-law-v17-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 5.6))
    colors = {"math": "#2864dc", "code": "#df5b21", "qa": "#168568"}
    if pruning_quant_pairs:
        for capability in CAPABILITIES:
            subset = [
                pair for pair in pruning_quant_pairs if pair["capability"] == capability
            ]
            axis.scatter(
                [pair["pruning_delta_L_c"] for pair in subset],
                [pair["quantization_delta_L_c"] for pair in subset],
                label=capability,
                color=colors[capability],
                alpha=0.8,
                edgecolor="white",
                linewidth=0.5,
            )
        coordinates = [
            float(pair[key])
            for pair in pruning_quant_pairs
            for key in ("pruning_delta_L_c", "quantization_delta_L_c")
        ]
        low, high = min(coordinates), max(coordinates)
        padding = max((high - low) * 0.05, 0.05)
        axis.plot(
            [low - padding, high + padding],
            [low - padding, high + padding],
            "k--",
            lw=1,
            label="one-curve equality",
        )
        axis.set_xlim(low - padding, high + padding)
        axis.set_ylim(low - padding, high + padding)
    else:
        axis.text(
            0.5,
            0.5,
            "No non-baseline exact-storage matches",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.axhline(0.0, color="#999999", linewidth=0.6)
    axis.axvline(0.0, color="#999999", linewidth=0.6)
    axis.set_xlabel("Pruning $\\Delta L_c$")
    axis.set_ylabel("Quantization $\\Delta L_c$")
    axis.set_title(f"Matched-storage test ($|\\Delta r_{{storage}}|<{tolerance:g}$)")
    axis.grid(alpha=0.2)
    if pruning_quant_pairs:
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(out / str(result["figure"]), dpi=180)
    figure.savefig(out / "matched_storage_pruning_vs_quant.png", dpi=180)
    plt.close(figure)
    return result


METHOD_SELECTION_FIELDS = (
    "family",
    "source_model",
    "capability",
    "delta_L_budget",
    "cost_axis",
    "status",
    "cheapest_method",
    "cheapest_methods",
    "cost_ratio",
    "r_storage",
    "r_active",
    "raw_coordinate_name",
    "raw_coordinate",
    "predicted_delta_L_c",
    "support_models",
    "support_cells",
    "matched_methods_at_coordinate",
    "missing_methods_at_matched_coordinate",
    "feasible_methods",
    "infeasible_methods_at_budget",
    "method_options",
    "crossover",
    "evidence_basis",
    "notes",
)


def _method_coordinate_support(
    rows: Sequence[Mapping[str, object]], method: str
) -> list[dict[str, object]]:
    grouped: dict[float, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[round(float(row["r_storage"]), 15)].append(row)
    support = []
    for _, coordinate_rows in sorted(grouped.items()):
        representative = coordinate_rows[0]
        support.append(
            {
                "method": method,
                "r_storage": float(
                    np.median([row["r_storage"] for row in coordinate_rows])
                ),
                "r_active": float(
                    np.median([row["r_active"] for row in coordinate_rows])
                ),
                "raw_coordinate_name": str(representative["raw_coordinate_name"]),
                "raw_coordinate": float(
                    np.median([row["raw_coordinate"] for row in coordinate_rows])
                ),
                "support_models": len({str(row["model"]) for row in coordinate_rows}),
                "support_cells": len(coordinate_rows),
            }
        )
    return support


def build_method_selection_map(
    rows: Sequence[Mapping[str, object]],
    loss_budgets: Sequence[float] = DEFAULT_LOSS_BUDGETS,
    tolerance: float = DEFAULT_STORAGE_MATCH_TOLERANCE,
    *,
    family: str = "gemma3",
    source_model: str = "gemma3-27b",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build Gemma-3 storage and active-cost M3 maps without extrapolation.

    Pruning and quantization use all observed models in the requested family;
    distillation is restricted to ``distill_source`` rows whose ``n0_model`` is
    the registered source.  ``distill_self`` cannot enter.  Each capability
    gets an independent M3 branch per method, queried only at observed
    pre-cliff coordinates.  A separate row is emitted for the storage and
    active axes, including which methods lack matched support at the selected
    coordinate.
    """
    budgets = sorted({float(value) for value in loss_budgets})
    if not budgets or any(not math.isfinite(value) for value in budgets):
        raise ValueError("loss budgets must contain finite values")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("storage match tolerance must be finite and positive")

    scoped_rows = [
        row
        for row in rows
        if str(row["family"]) == family
        and str(row["method"]) in METHODS
        and (
            (
                str(row["method"]) == DISTILL_SOURCE
                and str(row.get("n0_model", "")) == source_model
            )
            or str(row["method"]) != DISTILL_SOURCE
        )
    ]

    fitters: dict[
        str, Callable[[Sequence[Mapping[str, object]]], dict[str, object]]
    ] = {
        "pruning": _fit_m3_pruning,
        "quantization": _fit_m3_quantization,
        DISTILL_SOURCE: _fit_m3_distillation,
    }
    candidates: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(dict)
    fit_status: dict[str, dict[str, str]] = defaultdict(dict)
    fit_details: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    method_coordinates = {
        capability: {
            axis: {
                method: sorted(
                    {
                        float(row[axis])
                        for row in scoped_rows
                        if str(row["method"]) == method
                        and str(row["capability"]) == capability
                    }
                )
                for method in METHODS
            }
            for axis in ("r_storage", "r_active")
        }
        for capability in CAPABILITIES
    }
    for capability in CAPABILITIES:
        for method in METHODS:
            subset = [
                row
                for row in scoped_rows
                if str(row["capability"]) == capability and str(row["method"]) == method
            ]
            if not subset:
                fit_status[capability][method] = "no_data"
                candidates[capability][method] = []
                continue
            fit = fitters[method](subset)
            fit_details[capability][method] = fit
            fit_status[capability][method] = str(fit.get("status"))
            method_candidates = []
            for coordinate in _method_coordinate_support(subset, method):
                prediction = _predict_m3_method(method, fit, coordinate)
                if prediction is None or not math.isfinite(prediction):
                    continue
                method_candidates.append(
                    {
                        **coordinate,
                        "predicted_delta_L_c": float(prediction),
                    }
                )
            candidates[capability][method] = method_candidates

    result_rows: list[dict[str, object]] = []
    crossovers: list[dict[str, object]] = []
    for capability in CAPABILITIES:
        for axis in ("r_storage", "r_active"):
            previous_methods = ""
            secondary_axis = "r_active" if axis == "r_storage" else "r_storage"
            for budget in budgets:
                options: dict[str, dict[str, object]] = {}
                for method in METHODS:
                    feasible = [
                        candidate
                        for candidate in candidates[capability].get(method, [])
                        if float(candidate["predicted_delta_L_c"]) <= budget
                    ]
                    if feasible:
                        options[method] = min(
                            feasible,
                            key=lambda item: (
                                float(item[axis]),
                                float(item[secondary_axis]),
                            ),
                        )

                base = {
                    "family": family,
                    "source_model": source_model,
                    "capability": capability,
                    "delta_L_budget": budget,
                    "cost_axis": axis,
                    "evidence_basis": (
                        "capability-specific M3 predictions at observed Gemma-3 "
                        "family pre-cliff coordinates only"
                    ),
                }
                if not options:
                    result_rows.append(
                        {
                            **base,
                            "status": "no_method_achieves_budget",
                            "cheapest_method": "",
                            "cheapest_methods": "",
                            "method_options": "{}",
                            "feasible_methods": "",
                            "infeasible_methods_at_budget": ";".join(METHODS),
                            "crossover": False,
                            "notes": (
                                "No fitted method-specific branch reaches this loss "
                                "budget on observed support."
                            ),
                        }
                    )
                    continue

                winning_cost = min(float(choice[axis]) for choice in options.values())
                tied_methods = [
                    method
                    for method in METHODS
                    if method in options
                    and math.isclose(
                        float(options[method][axis]),
                        winning_cost,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ]
                winner = min(
                    tied_methods,
                    key=lambda method: (
                        float(options[method][secondary_axis]),
                        METHOD_ORDER[method],
                    ),
                )
                choice = options[winner]
                matched_methods = sorted(
                    (
                        method
                        for method, coordinates in method_coordinates[capability][
                            axis
                        ].items()
                        if method != winner
                        and any(
                            abs(winning_cost - value) < tolerance
                            for value in coordinates
                        )
                    ),
                    key=lambda method: METHOD_ORDER[method],
                )
                missing_methods = [
                    method
                    for method in METHODS
                    if method != winner and method not in matched_methods
                ]
                status = (
                    "matched_all_methods"
                    if not missing_methods
                    else f"insufficient_matched_{axis.removeprefix('r_')}_support"
                )
                winner_label = ";".join(tied_methods)
                crossover = bool(previous_methods and winner_label != previous_methods)
                if crossover:
                    crossovers.append(
                        {
                            "family": family,
                            "source_model": source_model,
                            "capability": capability,
                            "cost_proxy": axis,
                            "delta_L_budget": budget,
                            "from": previous_methods,
                            "to": winner_label,
                        }
                    )
                previous_methods = winner_label

                option_summary = {
                    method: {
                        "cost_axis": axis,
                        "cost_ratio": candidate[axis],
                        "r_storage": candidate["r_storage"],
                        "r_active": candidate["r_active"],
                        "raw_coordinate_name": candidate["raw_coordinate_name"],
                        "raw_coordinate": candidate["raw_coordinate"],
                        "predicted_delta_L_c": candidate["predicted_delta_L_c"],
                    }
                    for method, candidate in sorted(
                        options.items(), key=lambda item: METHOD_ORDER[item[0]]
                    )
                }
                notes = [
                    f"Missing matched {axis} support from: "
                    + ", ".join(missing_methods)
                    if missing_methods
                    else f"All methods have matched {axis} support."
                ]
                if winner == DISTILL_SOURCE:
                    notes.append(
                        "Source-referenced distillation uses the four-point Gemma-3 "
                        "270M/1B/4B/12B ladder against 27B."
                    )
                if axis == "r_active":
                    notes.append(
                        "Only distillation reduces active parameters; unstructured "
                        "pruning and weight-only quantization remain at r_active=1."
                    )
                result_rows.append(
                    {
                        **base,
                        "status": status,
                        "cheapest_method": winner,
                        "cheapest_methods": winner_label,
                        "cost_ratio": winning_cost,
                        "r_storage": choice["r_storage"],
                        "r_active": choice["r_active"],
                        "raw_coordinate_name": choice["raw_coordinate_name"],
                        "raw_coordinate": choice["raw_coordinate"],
                        "predicted_delta_L_c": choice["predicted_delta_L_c"],
                        "support_models": choice["support_models"],
                        "support_cells": choice["support_cells"],
                        "matched_methods_at_coordinate": ";".join(matched_methods),
                        "missing_methods_at_matched_coordinate": ";".join(
                            missing_methods
                        ),
                        "feasible_methods": ";".join(
                            method for method in METHODS if method in options
                        ),
                        "infeasible_methods_at_budget": ";".join(
                            method for method in METHODS if method not in options
                        ),
                        "method_options": json.dumps(option_summary, sort_keys=True),
                        "crossover": crossover,
                        "notes": " ".join(notes),
                    }
                )

    audit = {
        "family": family,
        "source_model": source_model,
        "loss_budgets": budgets,
        "fit_status_by_capability_method": {
            capability: dict(statuses) for capability, statuses in fit_status.items()
        },
        "fits_by_capability_method": {
            capability: dict(fits) for capability, fits in fit_details.items()
        },
        "crossovers": crossovers,
        "n_no_method_cells": sum(
            row["status"] == "no_method_achieves_budget" for row in result_rows
        ),
        "n_insufficient_cross_method_support_cells": sum(
            str(row["status"]).startswith("insufficient_matched_")
            for row in result_rows
        ),
        "n_rows_by_axis": dict(
            sorted(Counter(str(row["cost_axis"]) for row in result_rows).items())
        ),
        "decisions": [dict(row) for row in result_rows],
        "evidence_basis": (
            "capability-specific M3 fit on Gemma-3-family pruning/quantization "
            "and 27B-source-referenced 270M/1B/4B/12B distillation; observed "
            "pre-cliff coordinates only"
        ),
    }
    return result_rows, audit


def _write_method_selection_map(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=METHOD_SELECTION_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def _count_table(rows: Sequence[Mapping[str, object]], key: str) -> str:
    counts = Counter(str(row[key]) for row in rows)
    return ", ".join(f"{name}={counts[name]}" for name in sorted(counts))


def _heldout_table_lines(
    outputs: Mapping[str, Mapping[str, object]], protocol: str
) -> list[str]:
    lines = [
        "| hypothesis | held-out MAE | sign accuracy | predicted / test cells | coverage | # params | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for hypothesis in ("M1", "M2", "M3"):
        result = outputs[hypothesis]
        held_out = result["held_out"]
        assert isinstance(held_out, Mapping)
        split = held_out[protocol]
        assert isinstance(split, Mapping)
        metrics = split["metrics"]
        assert isinstance(metrics, Mapping)
        lines.append(
            f"| {hypothesis} | {_format_float(metrics['mae'])} | "
            f"{_format_float(metrics['sign_accuracy'])} | "
            f"{metrics['n_cells']} / {metrics['total_test_cells']} | "
            f"{_format_float(metrics['coverage'])} | "
            f"{result['full_data_parameter_count']} | {split['status']} |"
        )
    return lines


def _shared_law_statement(
    overlap_outputs: Mapping[str, Mapping[str, object]], protocol: str
) -> str:
    values = []
    has_sign_errors = False
    for hypothesis in ("M1", "M2"):
        split = overlap_outputs[hypothesis]["held_out"][protocol]
        assert isinstance(split, Mapping)
        metrics = split["metrics"]
        assert isinstance(metrics, Mapping)
        sign_accuracy = metrics.get("sign_accuracy")
        if sign_accuracy is None:
            values.append(f"{hypothesis} not estimable")
            has_sign_errors = True
        else:
            values.append(
                f"{hypothesis} MAE {_format_float(metrics.get('mae'))}, sign accuracy {_format_float(sign_accuracy)}"
            )
            has_sign_errors = has_sign_errors or float(sign_accuracy) < 1.0
    conclusion = (
        "Neither M1 nor M2 is a usable shared law: held-out sign errors violate the law-form audit gate."
        if has_sign_errors
        else "This protocol alone does not reject M1/M2 by the sign gate; the matched-storage test remains required."
    )
    return f"{conclusion} ({'; '.join(values)}.)"


def _write_summary(
    path: Path,
    rows: Sequence[dict],
    analysis_rows: Sequence[dict],
    audit: Mapping[str, object],
    precliff_audit: Mapping[str, object],
    overlap_audit: Mapping[str, object],
    outputs: Mapping[str, Mapping[str, object]],
    overlap_outputs: Mapping[str, Mapping[str, object]],
    matched: Mapping[str, object],
    selection_audit: Mapping[str, object],
    include_distillation: bool,
) -> None:
    lines = [
        "# V17 perturbative cross-method analysis",
        "",
        "## Headline",
        "",
        "**M1 and M2 do not yield a usable shared cross-method law. M1 is not a winner.** "
        "All law fits below use only perturbative pre-cliff cells, and the shared-law "
        "claim is evaluated only where another method has nearby pre-cliff storage support. "
        "The matched-storage comparison is the primary evidence: equal storage does not make "
        "compression methods interchangeable. The practical output is the Gemma-3 M3-based "
        "method-selection map, reported separately for storage and active-parameter cost, "
        "with explicit insufficient-support flags.",
        "",
        "## Primary test: matched storage",
        "",
        f"Pairs are matched one-to-one within `|Δr_storage| < {matched['tolerance']}` for the same source model and capability. "
        f"The cross-method CSV has {matched['n_pairs']} pairs, including "
        f"{matched['n_distill_source_pairs']} involving `distill_source`; `distill_self` is excluded. "
        "This descriptive falsification uses all measured non-baseline cells; post-cliff labels are retained "
        "in the CSV, but these cells never enter a law fit.",
        "",
        "At the exact common point `r_storage=0.5` (pruning density 0.5 versus 8-bit quantization):",
        "",
        "| capability | models | pruning median ΔL | quantization median ΔL | median pruning − quantization |",
        "|---|---:|---:|---:|---:|",
    ]
    half = matched.get("exact_half_storage_by_capability", {})
    assert isinstance(half, Mapping)
    for capability in CAPABILITIES:
        value = half.get(capability)
        if not isinstance(value, Mapping):
            lines.append(f"| {capability} | 0 | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {capability} | {value['n_models']} | "
            f"{_format_float(value['pruning_median_delta_L_c'])} | "
            f"{_format_float(value['quantization_median_delta_L_c'])} | "
            f"{_format_float(value['median_signed_method_gap'])} |"
        )
    lines.extend(
        [
            "",
            "Pruning is catastrophic at the clean 0.5-storage match (medians are about "
            "+5.9 math, +8.2 code, and +2.9 QA), while 8-bit quantization is approximately "
            "zero-damage. Storage ratio alone therefore cannot define a method-invariant response law.",
            "",
            f"Across the full storage-tolerance match: {matched['n_pairs']} pairs, "
            f"{matched['n_both_precliff_pairs']} with both sides pre-cliff; mean absolute method gap "
            f"{_format_float(matched['mean_absolute_method_gap'])}; sign agreement "
            f"{_format_float(matched['sign_agreement'])}. See [{matched['csv']}]({matched['csv']}) "
            f"and [{matched['figure']}]({matched['figure']}).",
            "",
            "The active-parameter match is a separate axis. Unstructured pruning and "
            "weight-only quantization both remain at `r_active=1`; only distillation lowers "
            "active parameter count and therefore offers a direct dense-inference compute/latency "
            f"reduction. There are {matched['n_active_pairs']} active-axis pairs, but "
            f"{matched['n_active_distill_source_pairs']} include `distill_source`, so its active-cost "
            f"advantage has no matched cross-method support in these data. See [{matched['active_csv']}]({matched['active_csv']}).",
            "",
            "## Distillation reference audit",
            "",
            "The same V16 distilled checkpoints give materially different answers under the "
            "two reference choices. `distill_self` is retained only to diagnose whether SFT "
            "hurt the student relative to its own dense checkpoint. `distill_source` measures "
            "subtraction from the 27B source and is the comparable cross-method quantity.",
            "",
            "| student | capability | distill_self dL_cap | distill_source ΔL |",
            "|---|---|---:|---:|",
        ]
    )
    source_distillation_rows = [
        row for row in rows if str(row["method"]) == DISTILL_SOURCE
    ]
    for row in sorted(
        source_distillation_rows,
        key=lambda item: (str(item["model"]), str(item["capability"])),
    ):
        lines.append(
            f"| {row['model']} | {row['capability']} | "
            f"{_format_float(row['dL_cap'])} | {_format_float(row['delta_L_source_c'])} |"
        )
    lines.extend(
        [
            "",
            "The capability-specific distillation law used by the selection map is "
            "`ΔL_c = capacity_floor - α_N log(r_storage)`. Each fit now uses all four "
            "student sizes; `α_N` is the fitted size-exponent reported below.",
            "",
            "| capability | student sizes | capacity floor | size-exponent α_N |",
            "|---|---:|---:|---:|",
        ]
    )
    selection_fits = selection_audit.get("fits_by_capability_method", {})
    assert isinstance(selection_fits, Mapping)
    for capability in CAPABILITIES:
        capability_fits = selection_fits.get(capability, {})
        assert isinstance(capability_fits, Mapping)
        distillation_fit = capability_fits.get(DISTILL_SOURCE, {})
        assert isinstance(distillation_fit, Mapping)
        parameters = distillation_fit.get("parameters", {})
        assert isinstance(parameters, Mapping)
        lines.append(
            f"| {capability} | {distillation_fit.get('n_student_sizes', 0)} | "
            f"{_format_float(parameters.get('capacity_floor'))} | "
            f"{_format_float(parameters.get('size_exponent'))} |"
        )
    lines.extend(
        [
            "",
            "Source-referencing reverses the apparent sign for Gemma-3 math/code: SFT looks "
            "slightly beneficial after style residualization against each student's own dense "
            "checkpoint, yet the distilled students still have positive capability-loss gaps "
            "against the 27B source. The math/code source gaps shrink across the four-size "
            "ladder at the reported precision; QA is negative at every size, so every "
            "distilled student beats the 27B source on this QA loss. This is why "
            "`distill_self` cannot select a method.",
            "",
            "## Perturbative restriction",
            "",
            f"The cap is `ΔL_c > {precliff_audit['cap']}` nats. Within each "
            "`(model, method, capability)` trajectory, coordinates are traversed from larger "
            "to smaller `r_storage`; the first crossing and every more-compressed coordinate "
            "are post-cliff and excluded.",
            "",
            f"- Candidate non-baseline cells: {precliff_audit['total_candidate_cells']}",
            f"- Pre-cliff cells retained: {precliff_audit['total_precliff_cells']}",
            f"- Post-cliff cells excluded: {precliff_audit['total_postcliff_cells']} "
            f"({_format_float(100 * float(precliff_audit['postcliff_fraction']), 1)}%)",
            f"- Retained by method: "
            + ", ".join(
                f"{method}={precliff_audit['precliff_cells'].get(method, 0)}"
                for method in TABLE_METHODS
            ),
            "",
            "## Storage support and overlap",
            "",
            "Only source-referenced distillation enters this support audit. The self-referenced "
            "diagnostic is retained in the assembled table but excluded from comparisons.",
            "",
            "| method | pre-cliff r_storage range | distinct coordinates | overlap-aware rows |",
            "|---|---:|---:|---:|",
        ]
    )
    ranges = overlap_audit.get("ranges", {})
    comparable_counts = overlap_audit.get("comparable_rows_by_method", {})
    assert isinstance(ranges, Mapping) and isinstance(comparable_counts, Mapping)
    for method in METHODS:
        value = ranges.get(method)
        if not isinstance(value, Mapping):
            lines.append(f"| {method} | no data | 0 | 0 |")
            continue
        lines.append(
            f"| {method} | [{_format_float(value['min'], 4)}, {_format_float(value['max'], 4)}] | "
            f"{value['n_distinct_coordinates']} | {comparable_counts.get(method, 0)} |"
        )
    pairwise = overlap_audit.get("pairwise", {})
    assert isinstance(pairwise, Mapping)
    lines.extend(
        [
            "",
            f"With tolerance `{overlap_audit['tolerance']}`, {overlap_audit['n_comparable_rows']} "
            "pre-cliff rows have support from at least one other method. The continuous pruning–"
            f"quantization range intersection is `{pairwise.get('pruning__quantization', {}).get('continuous_range_intersection')}`; "
            f"the all-method intersection is `{overlap_audit['all_method_continuous_intersection']}`. "
            "A continuous range intersection is not itself an observed three-way match.",
            "",
            "## Overlap-aware shared-law test (decisional scope)",
            "",
            "These are the only held-out results used to assess M1/M2 as shared laws. M3 is shown "
            "on the same rows for context; leave-one-method-out is not estimable for M3 by construction.",
        ]
    )
    protocol_titles = {
        "leave_one_method_out": "Leave one method out",
        "leave_one_family_out": "Leave one family out",
        "leave_largest_model_out": "Leave largest model out",
    }
    for protocol, title in protocol_titles.items():
        lines.extend(["", f"### {title}", ""])
        lines.extend(_heldout_table_lines(overlap_outputs, protocol))
        lines.extend(["", _shared_law_statement(overlap_outputs, protocol)])

    lines.extend(
        [
            "",
            "## All-pre-cliff held-out diagnostics",
            "",
            "These retain every perturbative cell. They are honest held-out diagnostics, but "
            "leave-method results include storage extrapolation into regimes with no competing "
            "method and therefore do not decide cross-method unification.",
        ]
    )
    for protocol, title in protocol_titles.items():
        lines.extend(["", f"### {title}", ""])
        lines.extend(_heldout_table_lines(outputs, protocol))

    lines.extend(
        [
            "",
            "## NON-DECISIONAL in-sample fit",
            "",
            "These diagnostics use pre-cliff cells only and never select a hypothesis.",
            "",
            "| hypothesis | in-sample MAE | sign accuracy | cells | # params |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for hypothesis in ("M1", "M2", "M3"):
        result = outputs[hypothesis]["non_decisional_in_sample"]
        assert isinstance(result, Mapping)
        metrics = result["metrics"]
        fit = result["fit"]
        assert isinstance(metrics, Mapping) and isinstance(fit, Mapping)
        lines.append(
            f"| {hypothesis} | {_format_float(metrics['mae'])} | "
            f"{_format_float(metrics['sign_accuracy'])} | {metrics['n_cells']} | "
            f"{fit.get('n_params')} |"
        )

    lines.extend(
        [
            "",
            "## Method-selection map (M3-based practical result)",
            "",
            "This map is scoped to the one family with all three valid arms: Gemma-3, "
            "with Gemma-3 27B as the distillation source reference. For each capability and loss budget, "
            "method-specific M3 branches are evaluated only at observed pre-cliff coordinates. "
            "Storage and active-parameter cost are separate rows and separate decisions.",
            "",
            f"- Loss-budget grid: {selection_audit['loss_budgets']}",
            f"- No-method-achieves cells: {selection_audit['n_no_method_cells']}",
            f"- Axis-specific cells lacking all-method matched support at the selected coordinate: "
            f"{selection_audit['n_insufficient_cross_method_support_cells']}",
            f"- Detected grid crossovers: {selection_audit['crossovers'] or 'none'}",
            "- CSV: [method_selection_map.csv](method_selection_map.csv)",
            "",
            "| capability | ΔL budget | storage-cost winner (r_storage) | storage support | active-cost winner (r_active) | active support |",
            "|---|---:|---|---|---|---|",
        ]
    )
    decisions = selection_audit.get("decisions", [])
    assert isinstance(decisions, list)
    by_decision = {
        (
            str(row["capability"]),
            float(row["delta_L_budget"]),
            str(row["cost_axis"]),
        ): row
        for row in decisions
    }
    for capability in CAPABILITIES:
        for budget in selection_audit["loss_budgets"]:
            storage = by_decision.get((capability, float(budget), "r_storage"), {})
            active = by_decision.get((capability, float(budget), "r_active"), {})
            storage_choice = (
                f"{storage.get('cheapest_methods')} ({_format_float(storage.get('cost_ratio'))})"
                if storage.get("cheapest_methods")
                else "none"
            )
            active_choice = (
                f"{active.get('cheapest_methods')} ({_format_float(active.get('cost_ratio'))})"
                if active.get("cheapest_methods")
                else "none"
            )
            lines.append(
                f"| {capability} | {_format_float(budget, 2)} | {storage_choice} | "
                f"{storage.get('status', 'insufficient')} | {active_choice} | "
                f"{active.get('status', 'insufficient')} |"
            )
    lines.extend(
        [
            "",
            "The two axes must not be conflated. Weight-only quantization can reduce storage "
            "while leaving `r_active=1`; unstructured pruning in these artifacts likewise uses "
            "dense execution and leaves `r_active=1`. Distillation is the only arm that reduces "
            "active parameters, so it is the only arm here with a direct model-size route to "
            "inference speedup. Thus distillation can lose a storage-cost comparison while still "
            "winning the active-parameter/latency comparison; both views must be checked even "
            "when this coarse budget grid happens to select the same method on both axes. Any "
            "`distill_source` winner remains descriptive: it is backed by "
            "a four-point 270M/1B/4B/12B ladder, and the map flags the absence of matched support on "
            "either cost axis rather than filling gaps by interpolation.",
            "",
            "## Input and artifact audit",
            "",
            f"The assembled table has **{len(rows)} rows**; **{len(analysis_rows)} pre-cliff "
            "non-baseline rows** enter the all-pre-cliff fits.",
            "",
            f"- Methods (all cells): {_count_table(rows, 'method')}",
            f"- Families: {_count_table(rows, 'family')}",
            f"- Capabilities: {_count_table(rows, 'capability')}",
            "- Unstructured pruning: `r_storage=d`, `r_active=1` (no sparse-kernel speedup is assumed).",
            "- Quantization: `r_storage=b/16`, `r_active=1`.",
            "- Distillation: `r_storage=N_S/N_0`, `r_active=N_S/N_0`.",
            "- `distill_self` is the V16 style-residualized `dL_cap` against the student's own dense checkpoint; it answers whether SFT hurt that student and is validity-only.",
            "- `distill_source` is `L_c(distilled student) - L_c(source dense)`; it is the only distillation delta used in fits, matching, and selection.",
            "- Dense pruning/quantization anchors remain in `unification_table.csv` but are never scored.",
        ]
    )
    references = audit.get("distillation_reference_models", {})
    if include_distillation:
        lines.append(
            f"- Distillation normalization references: `{references}`. Source-referencing currently "
            "exists only for Gemma-3 students 270M, 1B, 4B, and 12B versus the Gemma-3 27B dense "
            "source: a four-point ladder, not an inferred API-teacher size."
        )
    else:
        lines.append(
            "- Distillation has fewer than two measured student sizes and is excluded from fitting."
        )
    if int(audit.get("parameter_count_fallback_rows", 0)):
        lines.append(
            f"- Count-scope warning: {audit['parameter_count_fallback_rows']} rows use an "
            "embedding-inclusive fallback count."
        )
    for note in audit.get("notes", []):
        lines.append(f"- Loader note: {note}")
    source_audit = audit.get("distillation_source_audit", {})
    if isinstance(source_audit, Mapping):
        lines.append(
            f"- Source-reference family status: `{source_audit.get('family_status', {})}`. "
            "In particular, OLMo-3 7B is flagged as a single point with no clean "
            "distilled-from-OLMo-3-32B source run; no source delta is fabricated."
        )
    lines.extend(
        [
            "",
            "Machine-readable artifacts:",
            "",
            "- `unification_table.csv`: all cells plus pre-cliff and overlap annotations.",
            "- `m1_heldout.json`, `m2_heldout.json`, `m3_heldout.json`: all-pre-cliff and nested overlap-aware held-out results.",
            "- `matched_storage.csv`: source-aware cross-method storage matches; distillation uses only `distill_source`.",
            "- `matched_active.csv`: source-aware active-parameter matches and their support gaps.",
            "- `method_selection_map.csv`: Gemma-3 capability × loss-budget × cost-axis M3 selection map.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _print_counts(rows: Sequence[Mapping[str, object]]) -> None:
    print(f"assembled table shape: ({len(rows)}, {len(TABLE_FIELDS)})")
    print(f"per method: {_count_table(rows, 'method')}")
    print(f"per family: {_count_table(rows, 'family')}")
    print(f"per capability: {_count_table(rows, 'capability')}")


def _print_results(
    outputs: Mapping[str, Mapping[str, object]],
    overlap_outputs: Mapping[str, Mapping[str, object]],
) -> None:
    for hypothesis in ("M1", "M2", "M3"):
        result = outputs[hypothesis]["non_decisional_in_sample"]
        assert isinstance(result, Mapping)
        metrics = result["metrics"]
        assert isinstance(metrics, Mapping)
        print(
            f"NON-DECISIONAL in-sample {hypothesis}: "
            f"MAE={_format_float(metrics['mae'])}, "
            f"sign-accuracy={_format_float(metrics['sign_accuracy'])}"
        )
    for protocol in (
        "leave_one_method_out",
        "leave_one_family_out",
        "leave_largest_model_out",
    ):
        print(
            f"overlap-aware {protocol}: "
            f"{_shared_law_statement(overlap_outputs, protocol)}"
        )


def _parse_loss_budgets(value: str) -> tuple[float, ...]:
    try:
        budgets = tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "loss budgets must be comma-separated numbers"
        ) from exc
    if not budgets or any(not math.isfinite(item) for item in budgets):
        raise argparse.ArgumentTypeError("loss budgets must contain finite numbers")
    return budgets


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prune-dir", type=Path, default=PRUNE_BASE)
    parser.add_argument("--quant-dir", type=Path, default=QUANT_BASE)
    parser.add_argument("--distill-dir", type=Path, default=DISTILL_BASE)
    parser.add_argument("--output-dir", type=Path, default=OUT_BASE)
    parser.add_argument(
        "--precliff-cap",
        type=float,
        default=DEFAULT_PRECLIFF_CAP,
        help="first delta_L_c above this cap starts the post-cliff regime (default: 1.0)",
    )
    parser.add_argument(
        "--storage-match-tolerance",
        type=float,
        default=DEFAULT_STORAGE_MATCH_TOLERANCE,
        help="strict absolute r_storage tolerance for overlap and matching (default: 0.03)",
    )
    parser.add_argument(
        "--loss-budgets",
        type=_parse_loss_budgets,
        default=DEFAULT_LOSS_BUDGETS,
        help="comma-separated delta_L budgets for the method-selection map",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="assemble and print counts without fitting or writing outputs",
    )
    args = parser.parse_args(argv)

    raw_rows, audit = assemble_table(args.prune_dir, args.quant_dir, args.distill_dir)
    try:
        rows, precliff_audit = apply_precliff_filter(raw_rows, args.precliff_cap)
    except ValueError as exc:
        parser.error(str(exc))
    _print_counts(rows)
    print(
        "pre-cliff retained: "
        + ", ".join(
            f"{method}={precliff_audit['precliff_cells'].get(method, 0)}"
            for method in METHODS
        )
    )
    if args.dry_run:
        print(
            "source-referenced distillation students: "
            f"{audit['n_distillation_source_students']} "
            f"({', '.join(audit['distillation_source_students']) or 'none'})"
        )
        print("dry run: no fits or output files written")
        return 0
    if not rows:
        parser.error("no input rows were assembled")

    include_distillation = int(audit["n_distillation_source_students"]) >= 2
    fitting_rows = _analysis_rows(rows, include_distillation)
    if not fitting_rows:
        parser.error("no non-baseline rows are available for fitting")
    if not include_distillation:
        print(
            "source-referenced distillation insufficient (<2 student sizes): "
            "fitting pruning + quantization only",
            file=sys.stderr,
        )

    try:
        overlap_rows, overlap_audit = overlap_aware_rows(
            fitting_rows, args.storage_match_tolerance
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not overlap_rows:
        parser.error("no pre-cliff rows have storage overlap with another method")

    outputs = run_evaluation(fitting_rows, scope="all_precliff")
    overlap_outputs = run_evaluation(overlap_rows, scope="overlap_aware_precliff")
    selection_rows, selection_audit = build_method_selection_map(
        fitting_rows, args.loss_budgets, args.storage_match_tolerance
    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    _write_table(out / "unification_table.csv", rows)
    for hypothesis, result in outputs.items():
        path = out / f"{hypothesis.lower()}_heldout.json"
        payload = {
            **result,
            "precliff_audit": precliff_audit,
            "overlap_audit": overlap_audit,
            "overlap_aware": overlap_outputs[hypothesis],
        }
        path.write_text(
            json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    matched = _write_matched_storage_outputs(out, rows, args.storage_match_tolerance)
    _write_method_selection_map(out / "method_selection_map.csv", selection_rows)
    _write_summary(
        out / "summary.md",
        rows,
        fitting_rows,
        audit,
        precliff_audit,
        overlap_audit,
        outputs,
        overlap_outputs,
        matched,
        selection_audit,
        include_distillation,
    )
    _print_results(outputs, overlap_outputs)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
