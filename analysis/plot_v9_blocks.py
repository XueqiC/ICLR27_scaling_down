#!/usr/bin/env python3
"""Plot V9 capability-region similarity blocks for one or more models.

By default every directory under results/v9-capability-regions containing a
similarity.json file is included.  Use ``--models tag-a,tag-b`` to select and
order rows explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:
    from .v9_capability_regions import OUT_BASE, PROBE_REGISTRY
except ImportError:  # direct execution: python analysis/plot_v9_blocks.py
    from v9_capability_regions import OUT_BASE, PROBE_REGISTRY


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "results/figs"
METRICS = (
    ("raw_cosine", "Raw Fisher cosine"),
    ("log_cosine", "Log-Fisher cosine"),
    ("shared_component_removed_cosine", "Residual log-Fisher cosine"),
    ("top_0.1pct_jaccard", "Top-0.1% Jaccard"),
)
CAPABILITY_ORDER = {"math": 0, "code": 1, "qa": 2, "control": 3}
REGISTRY_ORDER = {spec.name: index for index, spec in enumerate(PROBE_REGISTRY)}


def _selected_model_tags(models: str | None, base: Path = OUT_BASE) -> list[str]:
    if models:
        tags = [tag.strip() for tag in models.split(",") if tag.strip()]
        if not tags:
            raise ValueError("--models did not contain a model tag")
    else:
        tags = sorted(
            path.parent.name for path in base.glob("*/similarity.json")
        )
    if not tags:
        raise FileNotFoundError(f"No similarity.json artifacts found under {base}")
    missing = [tag for tag in tags if not (base / tag / "similarity.json").exists()]
    if missing:
        raise FileNotFoundError(
            "Missing V9 similarity artifacts for: " + ", ".join(missing)
        )
    return tags


def _benchmark_order(payload: Mapping) -> list[str]:
    capabilities = payload["benchmark_capability"]
    return sorted(
        payload["benchmarks"],
        key=lambda name: (
            CAPABILITY_ORDER.get(capabilities[name], len(CAPABILITY_ORDER)),
            REGISTRY_ORDER.get(name, len(REGISTRY_ORDER)),
            name,
        ),
    )


def _matrix(payload: Mapping, metric: str, names: Sequence[str]) -> np.ndarray:
    nested = payload.get("matrices", {}).get(metric)
    if nested is None:
        raise KeyError(
            f"{payload.get('model', 'artifact')} does not contain metric {metric!r}"
        )
    matrix = np.asarray(
        [[nested[row][column] for column in names] for row in names],
        dtype=np.float64,
    )
    if matrix.shape != (len(names), len(names)) or not np.isfinite(matrix).all():
        raise ValueError(f"Metric {metric!r} is not a finite square matrix")
    return matrix


def _color_limits(matrices: Sequence[np.ndarray]) -> tuple[float, float]:
    lower = min(float(matrix.min()) for matrix in matrices)
    upper = max(float(matrix.max()) for matrix in matrices)
    if lower == upper:
        padding = max(abs(lower) * 0.05, 1e-6)
        return lower - padding, upper + padding
    return lower, upper


def _separator_positions(names: Sequence[str], capabilities: Mapping[str, str]):
    labels = [capabilities[name] for name in names]
    return [
        index - 0.5
        for index in range(1, len(labels))
        if labels[index] != labels[index - 1]
    ]


def plot_v9_blocks(
    model_tags: Sequence[str],
    base: Path = OUT_BASE,
    figure_dir: Path = FIGURE_DIR,
) -> tuple[Path, Path]:
    """Draw and save a model-by-metric grid of benchmark heatmaps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    records = []
    for tag in model_tags:
        path = base / tag / "similarity.json"
        payload = json.loads(path.read_text())
        names = _benchmark_order(payload)
        records.append(
            {
                "tag": tag,
                "payload": payload,
                "names": names,
                "matrices": {
                    metric: _matrix(payload, metric, names)
                    for metric, _ in METRICS
                },
            }
        )

    norms = {}
    for metric, _ in METRICS:
        lower, upper = _color_limits(
            [record["matrices"][metric] for record in records]
        )
        norms[metric] = Normalize(vmin=lower, vmax=upper)

    n_rows = len(records)
    figure_height = max(2.05 * n_rows + 1.1, 3.8)
    figure, axes = plt.subplots(
        n_rows,
        len(METRICS),
        figsize=(13.2, figure_height),
        squeeze=False,
    )
    column_images = []
    for row, record in enumerate(records):
        names = record["names"]
        capabilities = record["payload"]["benchmark_capability"]
        separators = _separator_positions(names, capabilities)
        for column, (metric, title) in enumerate(METRICS):
            axis = axes[row, column]
            image = axis.imshow(
                record["matrices"][metric],
                cmap="coolwarm" if "cosine" in metric else "viridis",
                norm=norms[metric],
                interpolation="nearest",
                aspect="equal",
                rasterized=True,
            )
            if row == 0:
                axis.set_title(title, fontsize=10, pad=8)
                column_images.append(image)
            axis.set_xticks(range(len(names)))
            if row == n_rows - 1:
                axis.set_xticklabels(names, rotation=55, ha="right")
            else:
                axis.set_xticklabels([])
            axis.set_yticks(range(len(names)), labels=names)
            axis.tick_params(axis="both", labelsize=6.5, length=2)
            if column == 0:
                axis.set_ylabel(record["tag"], fontsize=9, labelpad=8)
            for position in separators:
                axis.axhline(position, color="black", linewidth=1.1)
                axis.axvline(position, color="black", linewidth=1.1)
            axis.set_xlim(-0.5, len(names) - 0.5)
            axis.set_ylim(len(names) - 0.5, -0.5)

    figure.subplots_adjust(
        left=0.065,
        right=0.99,
        top=1.0 - 0.35 / figure_height,
        bottom=0.95 / figure_height,
        wspace=0.27,
        hspace=0.35,
    )
    # Manually place one horizontal scale below each column.  Reserving a
    # separate strip keeps colorbars clear of the rotated benchmark labels.
    for column, image in enumerate(column_images):
        position = axes[-1, column].get_position()
        colorbar_axis = figure.add_axes(
            [
                position.x0,
                0.12 / figure_height,
                position.width,
                0.07 / figure_height,
            ]
        )
        colorbar = figure.colorbar(
            image, cax=colorbar_axis, orientation="horizontal"
        )
        colorbar.ax.tick_params(labelsize=7, length=2)

    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / "v9_blocks.png"
    pdf_path = figure_dir / "v9_blocks.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=None,
        help="comma-separated model directory tags (default: all artifacts)",
    )
    args = parser.parse_args()
    tags = _selected_model_tags(args.models)
    png_path, pdf_path = plot_v9_blocks(tags)
    print(f"Wrote {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
