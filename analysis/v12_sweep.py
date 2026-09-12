#!/usr/bin/env python3
"""Sequential, resume-safe driver for V12 distillation runs.

The JSON config is a list of objects (or four-element lists) describing
``student``, ``teacher``, ``recipe``, and ``n_per_domain``.  Objects may also
set any of ``domains``, ``epochs``, ``device``, ``lr``, and ``output_suffix``.

Example config::

  [
    {"student": "gemma3-270m", "teacher": "gpt-5.6-luna",
     "recipe": "full", "n_per_domain": 600}
  ]

An existing ``eval.json`` is the completion marker and causes that run to be
skipped.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

try:
    from .model_registry import resolve_model
    from .v6_capability_geometry import model_output_tag
    from .v12_distill import (
        DOMAINS,
        OUT_BASE,
        RECIPES,
        TEACHERS,
        make_run_name,
        parse_domains,
    )
except ImportError:  # direct execution: python analysis/v12_sweep.py
    from model_registry import resolve_model
    from v6_capability_geometry import model_output_tag
    from v12_distill import (
        DOMAINS,
        OUT_BASE,
        RECIPES,
        TEACHERS,
        make_run_name,
        parse_domains,
    )


ROOT = Path(__file__).resolve().parents[1]
DISTILL_SCRIPT = ROOT / "analysis/v12_distill.py"
REQUIRED_KEYS = ("student", "teacher", "recipe", "n_per_domain")
OPTIONAL_KEYS = ("domains", "epochs", "device", "lr", "output_suffix")


def normalize_config(config: Mapping | Sequence) -> dict:
    """Normalize one JSON object or ``[student, teacher, recipe, n]`` row."""
    if isinstance(config, Mapping):
        unknown = set(config) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS)
        if unknown:
            raise ValueError(f"Unknown sweep config key(s): {sorted(unknown)}")
        missing = [key for key in REQUIRED_KEYS if key not in config]
        if missing:
            raise ValueError(f"Sweep config is missing: {', '.join(missing)}")
        normalized = dict(config)
    elif isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
        if len(config) != 4:
            raise ValueError(
                "Sweep list entries must be [student, teacher, recipe, n_per_domain]"
            )
        normalized = dict(zip(REQUIRED_KEYS, config))
    else:
        raise TypeError("Each sweep config must be an object or four-element list")

    normalized["student"] = str(normalized["student"])
    normalized["teacher"] = str(normalized["teacher"])
    normalized["recipe"] = str(normalized["recipe"])
    try:
        normalized["n_per_domain"] = int(normalized["n_per_domain"])
    except (TypeError, ValueError) as exc:
        raise ValueError("n_per_domain must be an integer") from exc
    if normalized["teacher"] not in TEACHERS:
        raise ValueError(f"Unknown teacher {normalized['teacher']!r}")
    if normalized["recipe"] not in RECIPES:
        raise ValueError(f"Unknown recipe {normalized['recipe']!r}")
    if normalized["n_per_domain"] <= 0:
        raise ValueError("n_per_domain must be positive")

    if "domains" in normalized:
        normalized["domains"] = parse_domains(normalized["domains"])
    else:
        normalized["domains"] = DOMAINS
    if "epochs" in normalized:
        normalized["epochs"] = int(normalized["epochs"])
        if normalized["epochs"] <= 0:
            raise ValueError("epochs must be positive")
    if "lr" in normalized:
        normalized["lr"] = float(normalized["lr"])
        if normalized["lr"] <= 0:
            raise ValueError("lr must be positive")
    normalized["output_suffix"] = str(normalized.get("output_suffix", ""))
    return normalized


def eval_path_for_config(config: Mapping | Sequence, output_base: Path = OUT_BASE) -> Path:
    """Return the completion-marker path without loading a model."""
    item = normalize_config(config)
    resolved = resolve_model(item["student"])
    student_tag = model_output_tag(item["student"], resolved)
    run_name = make_run_name(
        item["teacher"],
        item["recipe"],
        item["n_per_domain"],
        item["output_suffix"],
    )
    return output_base / student_tag / run_name / "eval.json"


def command_for_config(
    config: Mapping | Sequence,
    python_executable: str = sys.executable,
) -> list[str]:
    """Build one explicit V12 subprocess command."""
    item = normalize_config(config)
    command = [
        python_executable,
        str(DISTILL_SCRIPT),
        "--student",
        item["student"],
        "--teacher",
        item["teacher"],
        "--recipe",
        item["recipe"],
        "--domains",
        ",".join(item["domains"]),
        "--n-per-domain",
        str(item["n_per_domain"]),
    ]
    for key, flag in (
        ("epochs", "--epochs"),
        ("device", "--device"),
        ("lr", "--lr"),
    ):
        if key in item:
            command.extend((flag, str(item[key])))
    if item["output_suffix"]:
        command.extend(("--output-suffix", item["output_suffix"]))
    return command


def run_sweep(
    configs: Sequence[Mapping | Sequence],
    output_base: Path = OUT_BASE,
    runner: Callable[..., object] = subprocess.run,
    python_executable: str = sys.executable,
) -> dict[str, int]:
    """Run incomplete configs sequentially and return run/skip counts."""
    completed = 0
    skipped = 0
    for index, raw_config in enumerate(configs, start=1):
        config = normalize_config(raw_config)
        eval_path = eval_path_for_config(config, output_base=output_base)
        label = eval_path.parent.relative_to(output_base)
        if eval_path.is_file():
            skipped += 1
            print(f"[{index}/{len(configs)}] skip {label} (eval.json exists)")
            continue
        print(f"[{index}/{len(configs)}] run  {label}", flush=True)
        runner(
            command_for_config(config, python_executable=python_executable),
            check=True,
            cwd=ROOT,
        )
        completed += 1
    return {"completed": completed, "skipped": skipped}


def _load_configs(path: Path) -> list:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("Sweep config root must be a JSON list")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    configs = _load_configs(args.config)
    result = run_sweep(configs)
    print(
        f"sweep complete: {result['completed']} run, {result['skipped']} skipped"
    )


if __name__ == "__main__":
    main()
