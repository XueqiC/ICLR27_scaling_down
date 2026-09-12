#!/usr/bin/env python3
"""V16: residualize V12 capability loss changes against generic-text drift.

For each existing V12 adapter, this script measures teacher-forced loss on the
usual odd-indexed capability probes and on a deterministic held-out C4 slice.
Subtracting the generic loss change from each probe loss change distinguishes
capability-specific damage from a global/style-like loss shift.

Examples:
  python3 analysis/v16_style_residual.py --model gemma3-270m \
      --adapter results/v12-distill/gemma3-270m/RUN/adapter --device cuda:0
  python3 analysis/v16_style_residual.py \
      --v12-sweep results/v12-distill --device cuda:0
  python3 analysis/v16_style_residual.py --summarize
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch

try:
    from .model_registry import require_compliant
    from .v6_capability_geometry import load_text_causal_lm, model_output_tag
    from .v12_distill import (
        CAPABILITIES,
        MAX_LEN,
        MEASUREMENT_BENCHMARKS,
        masked_causal_loss,
        measure_capability_losses,
        seed_everything,
        tokenize_text_example,
        write_json_atomic,
    )
    from .v15_accuracy_link import _load_adapter, _v12_runs, measurement_probes
except ImportError:  # direct execution: python analysis/v16_style_residual.py
    from model_registry import require_compliant
    from v6_capability_geometry import load_text_causal_lm, model_output_tag
    from v12_distill import (
        CAPABILITIES,
        MAX_LEN,
        MEASUREMENT_BENCHMARKS,
        masked_causal_loss,
        measure_capability_losses,
        seed_everything,
        tokenize_text_example,
        write_json_atomic,
    )
    from v15_accuracy_link import _load_adapter, _v12_runs, measurement_probes


ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "results/v16-style-residual"
DEFAULT_V12_ROOT = ROOT / "results/v12-distill"
SEED = 0
N_PROBE = 128
GENERIC_DATASET = "allenai/c4"
GENERIC_CONFIG = "en"
GENERIC_SPLIT = "validation"
GENERIC_SHUFFLE_BUFFER = 10_000
DEFAULT_N_GENERIC = 64
DEFAULT_GENERIC_TOKENS = MAX_LEN
INTERPRETATION_RULE = (
    "If dL_cap is near zero while dL_c is positive, the observed probe-loss "
    "increase is consistent with global/style drift rather than capability "
    "loss. If dL_cap tracks dL_c, the change is capability-specific."
)


def load_generic_documents(n_documents: int) -> list[str]:
    """Load a deterministic seed-0 streaming slice of C4 validation text."""
    if n_documents <= 0:
        raise ValueError("n_generic must be positive")
    from datasets import load_dataset

    dataset = load_dataset(
        GENERIC_DATASET,
        GENERIC_CONFIG,
        split=GENERIC_SPLIT,
        streaming=True,
    ).shuffle(seed=SEED, buffer_size=GENERIC_SHUFFLE_BUFFER)
    documents = []
    for row in dataset:
        text = str(row.get("text", "")).strip()
        if text:
            documents.append(text)
        if len(documents) == n_documents:
            break
    if len(documents) != n_documents:
        raise RuntimeError(
            f"C4 yielded {len(documents)} non-empty documents; "
            f"requested {n_documents}"
        )
    return documents


def measure_generic_loss(
    model,
    tokenizer,
    documents: Sequence[str],
    device: str,
    generic_tokens: int,
) -> tuple[float, int]:
    """Return token-weighted teacher-forced NLL on plain C4 documents."""
    if generic_tokens < 2:
        raise ValueError("generic_tokens must be at least 2")
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for text in documents:
            example = tokenize_text_example(tokenizer, text, max_len=generic_tokens)
            loss, n_tokens = masked_causal_loss(model, example, device)
            if n_tokens:
                total_loss += float(loss) * n_tokens
                total_tokens += n_tokens
    if total_tokens == 0:
        raise RuntimeError("C4 generic slice produced no causal-LM target tokens")
    return total_loss / total_tokens, total_tokens


def residual_deltas(
    dense_capability: Mapping[str, float],
    distilled_capability: Mapping[str, float],
    dense_generic: float,
    distilled_generic: float,
) -> dict[str, dict[str, float]]:
    """Compute total, generic, and capability-specific loss changes."""
    d_generic = distilled_generic - dense_generic
    return {
        capability: {
            "dL_c": distilled_capability[capability] - dense_capability[capability],
            "dL_gen": d_generic,
            "dL_cap": (
                distilled_capability[capability]
                - dense_capability[capability]
                - d_generic
            ),
        }
        for capability in CAPABILITIES
    }


def interpret_delta(delta: Mapping[str, float]) -> str:
    """Apply an explicit descriptive heuristic to one residualized delta."""
    total = float(delta["dL_c"])
    residual = float(delta["dL_cap"])
    if total <= 0:
        return "no positive probe-loss damage to attribute"
    tolerance = max(1e-3, 0.25 * abs(total))
    if abs(residual) <= tolerance:
        return "global/style-drift-like"
    if abs(residual - total) <= tolerance:
        return "capability-specific-like"
    return "mixed global and capability-specific change"


def _dense_cache_signature(
    resolved_model: str, n_generic: int, generic_tokens: int
) -> dict[str, object]:
    return {
        "resolved_model": resolved_model,
        "seed": SEED,
        "n_probe_requested": N_PROBE,
        "probe_half": "measurement (odd indices, v[1::2])",
        "generic_dataset": GENERIC_DATASET,
        "generic_config": GENERIC_CONFIG,
        "generic_split": GENERIC_SPLIT,
        "generic_shuffle_buffer": GENERIC_SHUFFLE_BUFFER,
        "n_generic": n_generic,
        "generic_tokens": generic_tokens,
    }


def _read_dense_cache(path: Path, signature: Mapping[str, object]) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if payload.get("signature") != dict(signature):
        return None
    try:
        if set(payload["L_c"]) != set(CAPABILITIES):
            return None
        if set(payload["capability_tokens"]) != set(CAPABILITIES):
            return None
        float(payload["L_gen"])
        if int(payload["generic_token_count"]) <= 0:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return payload


def _v12_metadata(adapter: Path) -> dict:
    eval_path = adapter.parent / "eval.json"
    if not eval_path.is_file():
        return {}
    try:
        return json.loads(eval_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Could not read V12 metadata {eval_path}: {exc}") from exc


def _report_text(payload: Mapping) -> str:
    lines = [
        "# V16 style-residualized capability loss",
        "",
        INTERPRETATION_RULE,
        "",
        (
            "The qualitative labels below use `near` = max(0.001 nats/token, "
            "25% of |dL_c|); the raw deltas remain the primary result."
        ),
        "",
        (
            "| capability | L_c dense | L_c distilled | dL_c | dL_gen | "
            "dL_cap | interpretation |"
        ),
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for capability in CAPABILITIES:
        delta = payload["deltas"][capability]
        lines.append(
            f"| {capability} | {payload['dense']['L_c'][capability]:.6f} | "
            f"{payload['distilled']['L_c'][capability]:.6f} | "
            f"{delta['dL_c']:+.6f} | {delta['dL_gen']:+.6f} | "
            f"{delta['dL_cap']:+.6f} | "
            f"{payload['interpretation']['by_capability'][capability]} |"
        )
    lines.extend(
        [
            "",
            f"Dense generic loss: {payload['dense']['L_gen']:.6f} nats/token.",
            (
                "Distilled generic loss: "
                f"{payload['distilled']['L_gen']:.6f} nats/token."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def evaluate_adapter(
    *,
    model_request: str,
    adapter: Path,
    device: str,
    probes: Mapping[str, Sequence[Mapping]],
    generic_documents: Sequence[str],
    n_generic: int,
    generic_tokens: int,
    student_tag: str | None = None,
    metadata: Mapping | None = None,
    output_base: Path = OUT_BASE,
) -> dict:
    """Measure one dense/adapter pair and persist its residual decomposition."""
    if not adapter.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter}")
    if len(generic_documents) != n_generic:
        raise ValueError("generic document count does not match n_generic")
    seed_everything(SEED)
    resolved_model = require_compliant(model_request)
    resolved_model = require_compliant(resolved_model)
    tag = student_tag or model_output_tag(model_request, resolved_model)
    run_name = adapter.parent.name if adapter.name == "adapter" else adapter.name
    out = output_base / tag / run_name
    dense_cache_path = output_base / tag / "dense_cache.json"
    signature = _dense_cache_signature(resolved_model, n_generic, generic_tokens)
    dense = _read_dense_cache(dense_cache_path, signature)

    model, tokenizer = load_text_causal_lm(resolved_model, torch.bfloat16)
    model.to(device).eval()
    if dense is None:
        dense_losses, dense_capability_tokens = measure_capability_losses(
            model, tokenizer, probes, device
        )
        dense_generic, dense_generic_tokens = measure_generic_loss(
            model, tokenizer, generic_documents, device, generic_tokens
        )
        dense = {
            "version": 16,
            "signature": signature,
            "L_c": dense_losses,
            "L_gen": dense_generic,
            "capability_tokens": dense_capability_tokens,
            "generic_token_count": dense_generic_tokens,
        }
        write_json_atomic(dense_cache_path, dense)
        print(f"[dense] {tag}: L_c={dense_losses}, L_gen={dense_generic:.6f}")
    else:
        print(f"[cache] {tag}: reused dense capability/generic losses")

    model = _load_adapter(model, adapter, resolved_model)
    model.to(device).eval()
    distilled_losses, distilled_capability_tokens = measure_capability_losses(
        model, tokenizer, probes, device
    )
    distilled_generic, distilled_generic_tokens = measure_generic_loss(
        model, tokenizer, generic_documents, device, generic_tokens
    )
    if distilled_capability_tokens != dense["capability_tokens"]:
        raise RuntimeError(
            "Dense and distilled capability losses used different tokens"
        )
    if distilled_generic_tokens != dense["generic_token_count"]:
        raise RuntimeError("Dense and distilled generic losses used different tokens")

    deltas = residual_deltas(
        dense["L_c"], distilled_losses, dense["L_gen"], distilled_generic
    )
    v12 = dict(metadata or _v12_metadata(adapter))
    payload = {
        "version": 16,
        "student": v12.get("student", model_request),
        "resolved_student": resolved_model,
        "student_tag": tag,
        "teacher": v12.get("teacher"),
        "recipe": v12.get("recipe"),
        "n": v12.get("n_per_domain"),
        "n_per_domain": v12.get("n_per_domain"),
        "run_name": run_name,
        "adapter": str(adapter.resolve()),
        "dtype": "bfloat16",
        "device": device,
        "seed": SEED,
        "measurement_benchmarks": MEASUREMENT_BENCHMARKS,
        "probe_source": "analysis.v6_capability_geometry.build_probes",
        "n_probe_requested": N_PROBE,
        "probe_half": "measurement (odd indices, v[1::2])",
        "measurement_samples": {
            capability: len(probes[capability]) for capability in CAPABILITIES
        },
        "generic_corpus": {
            "dataset": GENERIC_DATASET,
            "config": GENERIC_CONFIG,
            "split": GENERIC_SPLIT,
            "selection": (
                f"streaming shuffle seed={SEED}, buffer={GENERIC_SHUFFLE_BUFFER}, "
                f"first {n_generic} non-empty documents"
            ),
            "n_documents": n_generic,
            "max_tokens_per_document": generic_tokens,
        },
        "dense": {
            "L_c": dense["L_c"],
            "L_gen": dense["L_gen"],
            "capability_tokens": dense["capability_tokens"],
            "generic_tokens": dense["generic_token_count"],
        },
        "distilled": {
            "L_c": distilled_losses,
            "L_gen": distilled_generic,
            "capability_tokens": distilled_capability_tokens,
            "generic_tokens": distilled_generic_tokens,
        },
        "deltas": deltas,
        "interpretation": {
            "rule": INTERPRETATION_RULE,
            "heuristic": "near=max(0.001 nats/token, 0.25*abs(dL_c))",
            "by_capability": {
                capability: interpret_delta(deltas[capability])
                for capability in CAPABILITIES
            },
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(_report_text(payload), encoding="utf-8")
    # residual.json is the sweep completion marker and is written last.
    write_json_atomic(out / "residual.json", payload)
    print(f"[write] {out / 'residual.json'}")

    del model, tokenizer
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def run_v12_sweep(
    v12_root: Path,
    *,
    device: str,
    n_generic: int,
    generic_tokens: int,
    output_base: Path = OUT_BASE,
    dry_run: bool = False,
) -> dict[str, int]:
    """Evaluate all completed V12 adapters, skipping residual.json markers."""
    if not v12_root.is_dir():
        raise FileNotFoundError(f"V12 sweep directory does not exist: {v12_root}")
    completed = 0
    skipped = 0
    probes = None
    documents = None
    for student_dir in sorted(path for path in v12_root.iterdir() if path.is_dir()):
        runs = _v12_runs(student_dir)
        if not runs:
            continue
        model_requests = {
            str(metadata.get("student") or metadata.get("resolved_student"))
            for _, metadata in runs
        }
        if len(model_requests) != 1:
            raise ValueError(
                f"V12 student directory {student_dir} names multiple models: "
                f"{sorted(model_requests)}"
            )
        model_request = model_requests.pop()
        require_compliant(require_compliant(model_request))
        for run_dir, metadata in runs:
            marker = output_base / student_dir.name / run_dir.name / "residual.json"
            if marker.is_file():
                skipped += 1
                print(
                    f"[skip] {student_dir.name}/{run_dir.name} "
                    "(residual.json exists)"
                )
                continue
            if dry_run:
                completed += 1
                print(
                    f"[would measure] student={student_dir.name}, "
                    f"run={run_dir.name}, adapter={run_dir / 'adapter'}"
                )
                continue
            probes = probes or measurement_probes()
            documents = documents or load_generic_documents(n_generic)
            evaluate_adapter(
                model_request=model_request,
                adapter=run_dir / "adapter",
                device=device,
                probes=probes,
                generic_documents=documents,
                n_generic=n_generic,
                generic_tokens=generic_tokens,
                student_tag=student_dir.name,
                metadata=metadata,
                output_base=output_base,
            )
            completed += 1
    return {"completed": completed, "skipped": skipped}


def write_summary(output_base: Path = OUT_BASE) -> Path:
    """Build a per-capability table across all V16 residual results."""
    rows = []
    if output_base.is_dir():
        for path in sorted(output_base.glob("*/*/residual.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                for capability in CAPABILITIES:
                    delta = payload["deltas"][capability]
                    rows.append(
                        (
                            payload.get("student_tag") or payload.get("student"),
                            payload.get("teacher"),
                            payload.get("recipe"),
                            payload.get("n_per_domain"),
                            capability,
                            float(delta["dL_c"]),
                            float(delta["dL_gen"]),
                            float(delta["dL_cap"]),
                        )
                    )
            except (
                KeyError,
                TypeError,
                ValueError,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                print(f"[summary] skip malformed {path}: {exc}", file=sys.stderr)
    lines = [
        "# V16 style-residualized capability loss summary",
        "",
        INTERPRETATION_RULE,
        "",
        "| student | teacher | recipe | n | capability | dL_c | dL_gen | dL_cap |",
        "|---|---|---|---:|---|---:|---:|---:|",
    ]
    for student, teacher, recipe, n_value, capability, d_c, d_gen, d_cap in rows:
        lines.append(
            f"| {student} | {teacher or '-'} | {recipe or '-'} | "
            f"{n_value if n_value is not None else '-'} | {capability} | "
            f"{d_c:+.6f} | {d_gen:+.6f} | {d_cap:+.6f} |"
        )
    output_base.mkdir(parents=True, exist_ok=True)
    path = output_base / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def print_dry_run(
    probes: Mapping[str, Sequence[Mapping]],
    n_generic: int,
    generic_tokens: int,
) -> None:
    print(
        f"probe source: build_probes(n={N_PROBE}, seed={SEED}), "
        "measurement half v[1::2]"
    )
    for capability in CAPABILITIES:
        print(
            f"{capability}: benchmark={MEASUREMENT_BENCHMARKS[capability]}, "
            f"{len(probes[capability])} probes, n_shots=0"
        )
        print(str(probes[capability][0]["prompt"]))
    print(
        f"generic: benchmark={GENERIC_DATASET}/{GENERIC_CONFIG} "
        f"{GENERIC_SPLIT}, documents={n_generic}, "
        f"max_tokens_per_document={generic_tokens}, seed={SEED}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="gemma3-270m",
        help="model-registry tag or raw Hugging Face model id",
    )
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-generic", type=int, default=DEFAULT_N_GENERIC)
    parser.add_argument(
        "--generic-tokens",
        type=int,
        default=DEFAULT_GENERIC_TOKENS,
        help="maximum tokenizer tokens per C4 document (default: 1024)",
    )
    parser.add_argument("--v12-sweep", type=Path, default=None, metavar="DIR")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.n_generic <= 0:
        parser.error("--n-generic must be positive")
    if args.generic_tokens < 2:
        parser.error("--generic-tokens must be at least 2")
    if args.summarize:
        path = write_summary()
        print(path)
        return
    if args.v12_sweep is not None and args.adapter is not None:
        parser.error("--v12-sweep cannot be combined with --adapter")
    if not args.dry_run and args.v12_sweep is None and args.adapter is None:
        parser.error("--adapter is required unless using --v12-sweep or --dry-run")

    if args.dry_run:
        probes = measurement_probes()
        print_dry_run(probes, args.n_generic, args.generic_tokens)
        if args.v12_sweep is not None:
            result = run_v12_sweep(
                args.v12_sweep,
                device=args.device,
                n_generic=args.n_generic,
                generic_tokens=args.generic_tokens,
                dry_run=True,
            )
            print(
                f"would measure {result['completed']} run, "
                f"{result['skipped']} already complete"
            )
        else:
            require_compliant(require_compliant(args.model))
            target = args.adapter or Path("<V12 adapter>")
            print(f"[would measure] model={args.model}, adapter={target}")
        return

    if args.v12_sweep is not None:
        result = run_v12_sweep(
            args.v12_sweep,
            device=args.device,
            n_generic=args.n_generic,
            generic_tokens=args.generic_tokens,
        )
        print(
            f"sweep complete: {result['completed']} run, "
            f"{result['skipped']} skipped"
        )
        return

    probes = measurement_probes()
    documents = load_generic_documents(args.n_generic)
    evaluate_adapter(
        model_request=args.model,
        adapter=args.adapter,
        device=args.device,
        probes=probes,
        generic_documents=documents,
        n_generic=args.n_generic,
        generic_tokens=args.generic_tokens,
    )


if __name__ == "__main__":
    main()
