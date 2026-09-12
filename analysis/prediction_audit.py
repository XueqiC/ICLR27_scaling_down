"""Small CPU-only utilities shared by the V24–V26 existing-data audits."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAPABILITIES = ("math", "code", "qa")


def read_json(path):
    return json.loads(Path(path).read_text())


def provenance(paths):
    return {str(Path(p).resolve().relative_to(ROOT)): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in sorted(set(paths))}


def finite(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Nonfinite observation")
    return value


def bootstrap_means(values, groups, *, n_boot=10000, seed=240526):
    """Paired cluster bootstrap; preserve all candidates within each draw.

    Values are cells x candidates. Resample whole groups and weight each cell
    equally, including for unequal group sizes. Fits are held fixed: intervals
    describe this panel's held-out errors, not retraining/seed uncertainty.
    """
    a = np.asarray(values, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[0] != len(groups) or not len(groups) or not np.isfinite(a).all():
        raise ValueError("Need finite paired values and matching nonempty groups")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    labels = sorted(set(groups))
    indices = [np.flatnonzero(np.asarray(groups) == g) for g in labels]
    sums = np.array([a[i].sum(axis=0) for i in indices])
    sizes = np.array([len(i) for i in indices])
    draws = np.random.default_rng(seed).integers(len(labels), size=(n_boot, len(labels)))
    return sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)[:, None]


def interval(samples):
    return np.quantile(samples, [.025, .975]).tolist()


def compare_predictions(rows, predictions, reference, *, n_boot=10000, group_key="model"):
    names = list(predictions)
    y = np.array([r["observed"] for r in rows])
    errors = np.column_stack([np.abs(np.asarray(predictions[n]) - y) for n in names])
    draws = bootstrap_means(errors, [r[group_key] for r in rows], n_boot=n_boot)
    ref = names.index(reference)
    metrics = {}
    for i, name in enumerate(names):
        metrics[name] = {"n": len(rows), "mae": float(errors[:, i].mean()),
                         "mae_ci95": interval(draws[:, i]),
                         "mae_minus_reference": float((errors[:, i] - errors[:, ref]).mean()),
                         "difference_ci95": interval(draws[:, i] - draws[:, ref])}
    return metrics, errors, draws


def splits(rows, key):
    for value in sorted({r[key] for r in rows}):
        train = [i for i, r in enumerate(rows) if r[key] != value]
        test = [i for i, r in enumerate(rows) if r[key] == value]
        if not train or not test:
            raise ValueError(f"Unidentified {key} split")
        yield {"held_out": value, "train_indices": train, "test_indices": test}


def fmt(value):
    return "N/A" if value is None else f"{value:.5f}"


def with_ci(value, ci):
    return f"{fmt(value)} [{fmt(ci[0])}, {fmt(ci[1])}]"


def write_outputs(summary, report, output_dir):
    # Recheck all source hashes immediately before writing derived artifacts.
    for path, expected in summary["input_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Input changed during analysis: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output_dir / "report.md").write_text(report)


def replace_section(path, heading, next_heading, content):
    text = path.read_text()
    start, stop = text.index(heading), text.index(next_heading)
    path.write_text(text[:start] + heading + "\n\n" + content.rstrip() + "\n\n" + text[stop:])
