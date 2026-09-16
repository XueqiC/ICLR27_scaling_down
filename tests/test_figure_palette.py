"""Semantic palette, literal-colour lint, and rendered glyph visibility."""
import ast
import re
from pathlib import Path

import pytest

from analysis import paper_figure_style as style

ROOT = Path(__file__).resolve().parents[1]
GENERATORS = sorted((ROOT / "analysis").glob("plot_*.py")) + [
    ROOT / "analysis/v64_selection_feasible.py", ROOT / "analysis/v80_rule_addenda.py",
    ROOT / "analysis/paper_panel_exports.py",
]


def literal_colors(source):
    from matplotlib.colors import CSS4_COLORS, BASE_COLORS, TABLEAU_COLORS
    names = set(CSS4_COLORS) | set(BASE_COLORS) | set(TABLEAU_COLORS)
    # Single-letter format strings ("b", etc.) are disallowed as literal colour
    # arguments, but panel letters and data dictionary keys are not colours.
    tree = ast.parse(source)
    palette_keys = {id(n.slice) for n in ast.walk(tree) if isinstance(n, ast.Subscript)
                    and isinstance(n.value, ast.Name) and n.value.id == "PALETTE"}
    failures = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in palette_keys:
            value = node.value.lower()
            if re.fullmatch(r"#[0-9a-f]{3,8}", value) or value in names - set(BASE_COLORS):
                failures.append((node.lineno, value))
        if isinstance(node, ast.keyword) and node.arg in {
                "color", "c", "facecolor", "edgecolor", "mfc", "mec", "ecolor",
                "markerfacecolor", "markeredgecolor"}:
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                value = node.value.value
                if value != "none":
                    failures.append((node.lineno, value))
    return failures


@pytest.mark.parametrize("path", GENERATORS, ids=lambda p: p.name)
def test_generators_import_palette_without_literal_colors(path):
    text = path.read_text()
    assert "paper_figure_style import" in text
    assert not literal_colors(text), f"{path.name}: {literal_colors(text)}"


def test_lint_rejects_hex_named_grayscale_and_shorthand_colors():
    for source in ('c = "#ff0000"', 'plot(color="red")', 'plot(mfc=".5")', 'plot(c="r")'):
        assert literal_colors(source)
    assert not literal_colors('plot(color=PALETTE["math"])')


def test_author_palette_and_semantic_aliases():
    assert style.CAPABILITY_COLORS == {"math": "#2b6cb0", "code": "#d97706", "qa": "#2f855a"}
    assert style.QA_COLORS == {"2wiki_new": "#2f855a", "musique": "#68b36b", "triviaqa": "#a8d5a2"}
    assert style.BIT_COLORS == {3: "#155e75", 4: "#0e7490", 5: "#67c3d9"}
    assert style.METHOD_COLORS["prune"] == "#6b46c1"
    assert style.METHOD_COLORS["quant"] == style.METHOD_COLORS["grouped_rtn"] == "#0e7490"
    assert style.METHOD_COLORS["per_channel_rtn"] == "#9a6b2f"
    assert style.METHOD_COLORS["distill"] == "#b83280"
    assert style.METHOD_COLORS["dense"] == "#9ca3af"
    assert style.STUDENT_STYLES == {"270M": ":", "1B": "--", "4B": "-"}
    assert style.SEED_MARKERS == ("o", "s")


def test_coincident_markers_are_visible_or_bounded_and_documented():
    from analysis.paper_artifacts import pyplot
    plt = pyplot()
    style.apply_style("panel")
    fig = plt.figure(figsize=(1.8, 1.35))
    ax = style.panel_axes(fig, (1.8, 1.35))
    ax.set(xlim=(-1, 1), ylim=(-1, 1))
    for i in range(18):
        color = style.CAPABILITY_COLORS[("math", "code", "qa")[i % 3]]
        ax.plot(0, 0, marker=("o", "s", "D", "x")[i % 4], color=color,
                mfc=style.PALETTE["transparent"] if i % 4 == 2 else color)
    try:
        audit = style.prepare_figure(fig)
        pixels = style.verify_marker_pixels(fig)
        assert pixels["count"] == 18 and pixels["undocumented_hidden"] == 0
        assert pixels["fully_hidden"] > 0  # Dense overlap is now accepted.
        for r in audit:
            assert max(r["axis_displacement_fraction"]) <= .015
            assert r["visible_ink_pixels"] > 0 or r["coincident_neighbours"]
            assert r["dx_pt"] == 0  # Crowded cluster returns to true positions.
            assert r["fallback_to_true"]
        assert all(r["x"] == r["y"] == 0 for r in audit)
        circles = [r for r in audit if r["marker"] == "o"]
        squares = [r for r in audit if r["marker"] == "s"]
        assert min(r["zorder"] for r in circles) > max(r["zorder"] for r in squares)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("scale,limits,x", [("linear", (0, 14), 1),
                                            ("log", (1, 1000), 950),
                                            ("symlog", (-100, 100), .01)])
@pytest.mark.parametrize("width", [1.45, 5.5])
def test_dodge_bound_in_display_and_native_coordinates(scale, limits, x, width):
    import numpy as np
    from analysis.paper_artifacts import pyplot
    plt = pyplot()
    style.apply_style("panel")
    fig, ax = plt.subplots(figsize=(width, 1.6))
    ax.set_xscale(scale)
    ax.set(xlim=limits, ylim=(-1, 1))
    for marker in ("o", "s", "D"):
        ax.plot([x], [0], marker=marker, ls="", color=style.PALETTE["math"])
    try:
        records = style.prepare_figure(fig)
        for r in records:
            assert max(r["axis_displacement_fraction"]) <= .015 + 1e-12
            assert max(r["data_displacement_fraction"]) <= .015 + 1e-12
            assert r["y"] == 0 and r["dy_pt"] == 0
            assert r["visible_ink_pixels"] > 0 or r["coincident_neighbours"]
            point = next(p for p in ax.lines if getattr(p, "_marker_record", {}).get("id") == r["id"])
            drawn = ax.transData.inverted().transform(point.get_transform().transform(point.get_xydata()))[0]
            assert np.allclose(drawn, r["drawn_coordinate"])
        assert np.ptp([r["dx_pt"]/r["axis_range_pt"][0] for r in records]) <= .06
        assert ax.get_xlim() == limits
        assert style.prepare_figure(fig) is records
    finally:
        plt.close(fig)


def test_temp_paths_stay_outside_paper():
    import tempfile
    from analysis.paper_artifacts import MPL_CACHE, pyplot
    before = tempfile.gettempdir()
    pyplot()
    assert tempfile.gettempdir() == before
    assert not MPL_CACHE.is_relative_to(ROOT / "paper")
    assert MPL_CACHE.is_relative_to(Path(tempfile.gettempdir()))


def test_generalization_uses_capability_hue_and_segment_for_lower_pair():
    from analysis import plot_fig_generalization as gen
    from analysis.paper_artifacts import pyplot
    plt = pyplot()
    style.apply_style("panel")
    fig, ax = plt.subplots(figsize=(1.8, 1.35))
    rows = [dict(panel="A", group="Pruning: density inside range", stratum="", capability=c,
                 relation_mae=value, baseline_mae=.5, below_baseline=value < .5,
                 whisker=None, n=10) for c, value in (("math", .2), ("code", .8))]
    try:
        gen.draw_maes(ax, rows, "A", (.01, 1))
        connectors = [line for line in ax.lines if len(line.get_xdata()) == 2
                      and line.get_marker() in (None, "None", "")]
        assert [line.get_color() for line in connectors] == [style.CAPABILITY_COLORS["math"], style.PALETTE["reference"]]
        predictions = [line for line in ax.lines if line.get_marker() == "D"]
        assert [line.get_color() for line in predictions] == [style.CAPABILITY_COLORS[c] for c in ("math", "code")]
        assert all(line.get_markerfacecolor() == style.PALETTE["transparent"] for line in predictions)
    finally:
        plt.close(fig)
