#!/usr/bin/env python3
"""Plot behavioral and geometric V11 damage curves for one model."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

try:
    from .v11_geometry_damage import (
        CAPABILITIES,
        OUT_BASE,
        model_output_tag,
        require_compliant,
    )
except ImportError:  # direct execution: python analysis/plot_v11.py
    from v11_geometry_damage import (
        CAPABILITIES,
        OUT_BASE,
        model_output_tag,
        require_compliant,
    )


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "results/figs"
GEOMETRY_STYLES = (
    ("residual_cosine", "Residual cosine", "o"),
    ("log_fisher_cosine", "Log-Fisher cosine", "s"),
    ("mass_retention", "Mass retention", "^"),
)


def _validated_curve(payload: Mapping, capability: str) -> list[dict]:
    record = payload.get("capabilities", {}).get(capability)
    if not isinstance(record, Mapping):
        raise KeyError(f"Artifact has no {capability!r} capability record")
    curve = record.get("curve")
    if not isinstance(curve, list) or not curve:
        raise ValueError(f"{capability} curve is empty")
    required = {
        "density",
        "delta_loss",
        "residual_cosine",
        "log_fisher_cosine",
        "mass_retention",
    }
    for index, row in enumerate(curve):
        missing = required - set(row)
        if missing:
            raise KeyError(
                f"{capability} curve row {index} is missing {sorted(missing)}"
            )
        for key in required:
            if not math.isfinite(float(row[key])):
                raise ValueError(
                    f"{capability} curve row {index} has non-finite {key}"
                )
    return curve


def plot_geometry_damage(
    tag: str,
    base: Path = OUT_BASE,
    figure_dir: Path = FIGURE_DIR,
) -> tuple[Path, Path]:
    """Create a two-panel behavioral/geometric plot for each capability."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifact_path = base / tag / "geometry_damage.json"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing V11 artifact: {artifact_path}")
    payload = json.loads(artifact_path.read_text())
    curves = {
        capability: _validated_curve(payload, capability)
        for capability in CAPABILITIES
    }

    figure, axes = plt.subplots(
        1,
        2 * len(CAPABILITIES),
        figsize=(18.0, 3.8),
        squeeze=False,
    )
    axes = axes[0]
    for capability_index, capability in enumerate(CAPABILITIES):
        curve = curves[capability]
        densities = [float(row["density"]) for row in curve]
        loss_damage = [float(row["delta_loss"]) for row in curve]
        loss_axis = axes[2 * capability_index]
        geometry_axis = axes[2 * capability_index + 1]

        positive_density = [
            density
            for density, damage in zip(densities, loss_damage)
            if damage > 0
        ]
        positive_damage = [damage for damage in loss_damage if damage > 0]
        if positive_damage:
            loss_axis.plot(
                positive_density,
                positive_damage,
                color="#b33b2e",
                marker="o",
                linewidth=1.8,
                markersize=4,
            )
        else:
            loss_axis.text(
                0.5,
                0.5,
                "No positive damage",
                transform=loss_axis.transAxes,
                ha="center",
                va="center",
                fontsize=9,
            )
        loss_axis.set_yscale("log")
        loss_axis.set_title(f"{capability}: behavior", fontsize=11)
        loss_axis.set_ylabel(r"$\Delta L_c$ (log scale)")
        loss_axis.set_xlabel("density")
        loss_axis.grid(True, which="both", alpha=0.22, linewidth=0.6)

        for metric, label, marker in GEOMETRY_STYLES:
            geometry_axis.plot(
                densities,
                [float(row[metric]) for row in curve],
                label=label,
                marker=marker,
                linewidth=1.5,
                markersize=3.8,
            )
        geometry_axis.axhline(0.0, color="black", alpha=0.2, linewidth=0.7)
        geometry_axis.set_title(f"{capability}: geometry", fontsize=11)
        geometry_axis.set_ylabel("retention / cosine")
        geometry_axis.set_xlabel("density")
        geometry_axis.grid(True, alpha=0.22, linewidth=0.6)
        geometry_axis.legend(fontsize=7.5, frameon=False, loc="best")

        summary = payload["capabilities"][capability].get("summary", {})
        cliff_density = summary.get("density")
        if cliff_density is not None:
            for axis in (loss_axis, geometry_axis):
                axis.axvline(
                    float(cliff_density),
                    color="#555555",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.8,
                )
        # Reversed density makes the progression toward more pruning run
        # from left to right.
        loss_axis.invert_xaxis()
        geometry_axis.invert_xaxis()

    figure.suptitle(
        f"Capability geometry damage under magnitude pruning — {tag}",
        fontsize=13,
        y=1.02,
    )
    figure.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / f"v11_geometry_{tag}.png"
    pdf_path = figure_dir / f"v11_geometry_{tag}.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="gemma3-1b",
        help="registry tag or raw Hugging Face id (used to derive artifact tag)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="explicit V11 result directory tag; overrides --model",
    )
    args = parser.parse_args()
    if args.tag:
        tag = args.tag
    else:
        resolved = require_compliant(args.model)
        tag = model_output_tag(args.model, resolved)
    png_path, pdf_path = plot_geometry_damage(tag)
    print(f"Wrote {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
