import json
import re
from collections import Counter
from statistics import mean

import pytest

from analysis import plot_fig_generalization as gen
from analysis.paper_artifacts import ROOT, pyplot, output_path
from paper_generator_checks import (
    check_access, check_figures, refuses_symlink, refuses_external_io,
    refuses_output_file_symlink,
)

ROW_LABELS = {
    "a": ["Pruning, unseen densities of development states",
          "Pruning, new checkpoints inside the density range",
          "Pruning, new checkpoints outside the range",
          "Quantization, unseen group sizes",
          "Distillation, new pools, 270M student",
          "Distillation, new pools, 1B student"],
    "b": ["Pruning, new training stages of a seen size",
          "Quantization, new model state"],
}


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    # Initialize pytest's scratch root before the legacy plotting helper sets
    # tempfile.tempdir to its permitted Matplotlib cache.
    tmp_path_factory.getbasetemp()
    return gen.generate()


@pytest.fixture(scope="module")
def frozen():
    cache = {}

    def read(ref):
        path, _, ptr = ref.partition("#")
        if path not in cache:
            cache[path] = json.loads((ROOT / path).read_text())
        value = cache[path]
        for key in ptr.lstrip("/").split("/") if ptr else []:
            key = key.replace("~1", "/").replace("~0", "~")
            value = value[int(key)] if isinstance(value, list) else value[key]
        return value
    return read


def test_maes_preserve_exact_eligible_cell_coverage(generated):
    rows, audit, access = generated
    check_access(audit, access)
    check_figures("generalization")
    assert gen.format_pairs(rows) in output_path(ROOT, "figs", "generalization_mae_pairs.md").read_text()
    from analysis.paper_artifacts import Artifacts
    original = gen.cells.build(Artifacts())
    original = [r for r in original if r["measurement_interval"] is None
                and r["group"] != "Pythia: locked rule on new states"]
    cells = [c for r in rows for c in r["cells"]]
    identity = lambda r: (r["panel"], r["group"], r["capability"], r["source"])
    assert Counter(map(identity, cells)) == Counter(map(identity, original))
    assert len(cells) == 222
    assert all(r["kind"] == "mae" for r in rows)
    assert not any("locked" in r["group"] for r in rows)
    assert any("A2 frozen_prediction_error_interval" in note for note in audit.notes)


def test_maes_recompute_from_frozen_predictions_on_identical_cells(generated, frozen):
    rows, _, _ = generated
    for row in (r for r in rows if r["kind"] == "mae"):
        predicted, actual, baseline = [], [], []
        for c in row["cells"]:
            source = frozen(c["source"])
            if isinstance(source, (float, int)):
                pred, measured = source, frozen(c["measured_source"])
            elif "v46-p1-newsource" in c["source"]:
                key = f"{source['cap']}|{source['d']}"
                pred = frozen("results/v46-p1-newsource/predictions_frozen.json")["pruning"][key]["power"]
                measured = source["actual"]
            else:
                pred = source["predictions"][row["relation"]]
                measured = source.get("observed_delta_loss", source.get("dL", source.get("actual")))
            assert pred == c["predicted"] and measured == c["measured"]
            predicted.append(pred)
            actual.append(measured)
            if c["baseline_source"] is not None:
                baseline.append(frozen(c["baseline_source"]))
        assert row["n"] == len(actual)
        assert row["relation_mae"] == mean(abs(p - a) for p, a in zip(predicted, actual))
        if row["baseline"] is None:
            assert not baseline and row["baseline_mae"] is None and row["below_baseline"] is None
            assert all("v46-p1-newsource" in c["source"] or "v78-rule-confirm" in c["source"] for c in row["cells"])
        else:
            assert len(baseline) == len(actual)
            assert row["baseline_mae"] == mean(abs(b - a) for b, a in zip(baseline, actual))
            assert row["below_baseline"] == (row["relation_mae"] < row["baseline_mae"])
    # A partial V72 baseline cannot be scored against the full V46+V72 row.
    inside = [r for r in rows if r["group"] == "Pruning: density inside range"]
    assert {(r["stratum"], r["n"], r["baseline"] is None) for r in inside} == {
        ("V72", 6, False), ("V46", 1, True)}


def test_baselines_use_development_selection_not_test_oracles(generated, frozen):
    rows, _, _ = generated
    p53 = frozen(gen.P53)
    q69 = frozen(gen.Q69)
    f70 = frozen(gen.F70)
    for row in (r for r in rows if r["kind"] == "mae" and r["baseline"] is not None):
        cap = row["capability"]
        source = row["cells"][0]["source"]
        if "v53-prune-dev" in source or "v72-prune-repeat" in source:
            eligible = [r for r in p53["loso_table"] if r["subset"] == "all" and r["candidate"] != "power"]
            assert row["baseline"] == min(eligible, key=lambda r: r[f"{cap}_mae"])["candidate"]
        elif "v69-quant-confirm" in source:
            methods = set(q69["methods"]) - {row["relation"]}
            assert row["baseline"] == min(methods, key=lambda m: q69["loso"]["scores"][m][cap]["macro_mae"])
        else:
            assert source.startswith(gen.D70)
            assert row["baseline"] == f70["strongest_baseline"][row["stratum"]][cap]["method"]
        assert all(c["selection_source"] is not None for c in row["cells"])
    # Losing/neutral relations must survive; don't select against held-out errors.
    assert any(r.get("below_baseline") is False for r in rows)
    stages = {r["capability"]: r for r in rows if r["group"] == "Pythia: new stages (power)"}
    assert {c: r["baseline"] for c, r in stages.items()} == {"math": "A2", "code": "A2", "qa": "median_curve"}
    assert all(r["below_baseline"] is False for r in stages.values())


def test_whiskers_are_exact_paired_gain_intervals_never_pooled_or_mae_intervals(generated, frozen):
    rows, _, _ = generated
    interval_rows = [r for r in rows if r["kind"] == "mae" and r["whisker"] is not None]
    assert len(interval_rows) == 6
    for row in interval_rows:
        ref = row["interval_source"]
        stored = frozen(ref.rsplit("/paired_difference", 1)[0])
        assert row["n"] == stored["n_checkpoints"] == 18
        assert {c["student"] for c in row["cells"]} == {stored["student"]}
        lo, hi = frozen(ref)
        assert row["paired_gain_interval"] == [lo, hi]
        assert row["relation_mae"] == pytest.approx(stored["candidate_mae"])
        assert row["baseline_mae"] == pytest.approx(stored["baseline_mae"])
        assert row["whisker"] == [row["baseline_mae"] - hi, row["baseline_mae"] - lo]
        assert row["whisker"] != [lo, hi]
        assert ref.startswith(gen.D70) and ref.endswith("/paired_difference/ci95")
    assert all(r["paired_gain_interval"] is None and r["interval_source"] is None
               for r in rows if r["kind"] == "mae" and r["whisker"] is None)


def test_render_has_log_maes_shared_legend_and_descriptive_labels(generated):
    rows, _, _ = generated
    plt = pyplot()
    fig = gen.plot(rows, plt)
    try:
        fig.canvas.draw()
        maes = [ax for ax in fig.axes if ax.get_xscale() == "log"]
        assert len(maes) == 2
        assert maes[0].get_xlim() == maes[1].get_xlim()
        assert len(fig.subfigs) == 3
        assert [len(sub.legends) for sub in fig.subfigs] == [0, 0, 1]
        assert [t.get_text() for t in maes[0].get_yticklabels()] == ROW_LABELS["a"]
        from matplotlib.text import Text
        drawn = [t.get_text() for t in fig.findobj(Text) if t.get_visible()]
        assert not any(re.search(r"\bV\d+\b|\bBase\b|\bRel\.|\bdev\.", t) for t in drawn)
        assert [t.get_text() for t in maes[1].get_yticklabels()] == ROW_LABELS["b"]
        assert maes[0].bbox.x0 == pytest.approx(maes[1].bbox.x0)
        assert maes[0].bbox.x1 == pytest.approx(maes[1].bbox.x1)
        assert maes[0].bbox.y0 > maes[1].bbox.y1
        for panel, ax in zip("AB", maes):
            assert ax.get_xlabel() == "Mean absolute error (nats)"
            assert list(ax.get_xticks()) == [.03, .1, .3, 1.]
            assert [t.get_text() for t in ax.get_xticklabels()] == ["0.03", "0.1", "0.3", "1"]
            part = [r for r in rows if r["panel"] == panel and r["kind"] == "mae"]
            order = list(dict.fromkeys((r["group"], r["stratum"]) for r in part))
            labels = [t.get_text() for t in ax.get_yticklabels()]
            assert len(labels) == len(order)
            for r in part:
                matching = [line for line in ax.lines if line.get_marker() == "D"
                            and len(line.get_xdata()) == 1 and line.get_xdata()[0] == r["relation_mae"]]
                assert any(line.get_color() == gen.COLORS[r["capability"]] for line in matching)
        assert "unavailable" in gen.format_pairs(rows)
        assert len(gen.format_pairs(rows).splitlines()) == 26
    finally:
        plt.close(fig)


def test_exported_sizes_labels_subrows_and_coincidences(generated):
    rows, _, _ = generated
    directory = output_path(ROOT, "figs")
    limits, margins = [], []
    for panel, size in (("a", [5.5, 1.5]), ("b", [5.5, .85]), ("legend", [5.5, .3])):
        data = json.loads((directory / f"fig3_{panel}_data.json").read_text())
        assert data["size_inches"] == size
        raw = (directory / f"fig3_{panel}.pdf").read_bytes()
        box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", raw)
        assert [float(v)/72 for v in box.groups()] == pytest.approx(size)
        caption = (directory / f"fig3_{panel}_caption.txt").read_text()
        assert "hollow circle denotes the development-selected baseline" in caption
        assert "hollow diamond denotes the frozen relation" in caption
        assert "reference grey otherwise" in caption
        assert "within 2 percent" in caption
        assert "1.5% limit" in caption
        if panel == "legend":
            assert data["legend_entries"] == ["Math", "Code", "QA", "Baseline", "Relation", "Lower of pair"]
            continue
        axis = data["axes"][0]
        assert axis["row_labels"] == ROW_LABELS[panel]
        assert all("\n" not in label for label in axis["row_labels"])
        limits.append(axis["xlim"])
        margins.append(axis["rectangle"][0])
        points = data["marker_visibility"]["points"]
        assert data["marker_visibility"]["png_pixel_check"]["undocumented_hidden"] == 0
        for row in data["records"]:
            display = row["display"]
            offset = {"math": -.25, "code": 0., "qa": .25}[row["capability"]]
            assert display["subrow_offset"] == offset
            y = ROW_LABELS[panel].index(display["row_label"]) + offset
            pair = [p for p in points if p["y"] == y and p["capability"] == row["capability"]]
            assert len(pair) == (1 if row["baseline_mae"] is None else 2)
            for point in pair:
                x = row[f"{point['role']}_mae"]
                assert point["x"] == x
                assert point["drawn_coordinate"][1] == pytest.approx(y)
                assert point["dy_pt"] == 0
                assert max(point["axis_displacement_fraction"]) <= .015 + 1e-12
                assert max(point["data_displacement_fraction"]) <= .015 + 1e-12
                assert point["marker_size_pt"] == 3.8
                assert point["white_outer_edge"]
            if display["coincident_within_2_percent"]:
                assert display["relative_pair_gap"] <= .02
                circle, = [p for p in pair if p["marker"] == "o"]
                diamond, = [p for p in pair if p["marker"] == "D"]
                assert diamond["zorder"] > circle["zorder"]
                assert diamond["coincident_within_2_percent"]
                assert diamond["dx_pt"] == circle["dx_pt"]
        assert data["artist_sizes"]["marker_sizes_pt"] == ([3.8, 4.0] if panel == "a" else [3.8])
    assert margins[0] == margins[1]
    assert limits[0] == limits[1]
    coincident = [r for r in rows if r["display"]["coincident_within_2_percent"]]
    assert [(r["group"], r["capability"]) for r in coincident] == [("Quantization: new group size", "qa")]


@pytest.mark.parametrize("relation,baseline,expected", [
    (.5, .5, True), (.49, .5, True), (.51, .5, True),
    (.489, .5, False), (.511, .5, False), (.5, None, False),
])
def test_coincidence_threshold(relation, baseline, expected):
    assert gen.is_coincident(dict(relation_mae=relation, baseline_mae=baseline)) is expected


def test_export_preserves_subrow_segments_and_diamond_order_at_exact_coincidence():
    from analysis import paper_figure_style as style
    plt = pyplot()
    style.apply_style("double")
    fig = plt.figure(figsize=gen.PANEL_SIZES["a"])
    ax = style.panel_axes(fig, gen.PANEL_SIZES["a"], **gen.AXIS_MARGINS)
    rows = [dict(panel="A", group="Quantization: new group size", stratum="", capability=cap,
                 baseline_mae=.3, relation_mae=.3, below_baseline=False, whisker=None, n=1)
            for cap in gen.CAPS]
    try:
        gen.draw_maes(ax, rows, "A", (.01, 2))
        gen.prepare_mae_markers(fig)
        segments = [line for line in ax.lines if line.get_visible() and len(line.get_xdata()) == 2]
        assert [list(line.get_ydata()) for line in segments] == [[-.25, -.25], [0., 0.], [.25, .25]]
        assert all(line.get_color() == gen.GREY for line in segments)
        for cap, y in zip(gen.CAPS, [-.25, 0., .25]):
            pair = [p for p in fig._marker_audit if p["capability"] == cap]
            assert all(p["x"] == .3 and p["y"] == y and p["dx_pt"] == 0 for p in pair)
            circle, = [p for p in pair if p["marker"] == "o"]
            diamond, = [p for p in pair if p["marker"] == "D"]
            assert diamond["zorder"] > circle["zorder"]
            assert diamond["visible_ink_pixels"] > 0
    finally:
        plt.close(fig)


def test_exported_glyph_footprints_do_not_overlap_across_capabilities(generated):
    import numpy as np
    from analysis import paper_figure_style as style
    from matplotlib.backends.backend_agg import RendererAgg
    rows, _, _ = generated
    plt = pyplot()
    for panel in "ab":
        style.apply_style(gen.KINDS[panel])
        fig = plt.figure(figsize=gen.PANEL_SIZES[panel])
        try:
            ax = gen.draw_panel(fig, rows, panel)
            gen.prepare_mae_markers(fig)
            fig.canvas.draw()
            width, height = map(int, fig.bbox.size)
            renderer = RendererAgg(width, height, fig.dpi)
            rendered = []
            for point in ax.lines:
                if not getattr(point, "_palette_point", False):
                    continue
                renderer.clear()
                point.draw(renderer)
                alpha = np.asarray(renderer.buffer_rgba())[:, :, 3]
                footprint = set(np.flatnonzero(alpha > .2*255).tolist())
                record = point._marker_record
                for previous, pixels in rendered:
                    if previous["capability"] != record["capability"]:
                        assert not pixels & footprint, (previous, record)
                rendered.append((record, footprint))
            for line in ax.lines:
                if not hasattr(line, "_mae_segment"):
                    continue
                y, cap = line._mae_segment
                pair = {r["role"]: r for r, _ in rendered if r["y"] == y and r["capability"] == cap}
                assert list(line.get_ydata()) == [y, y]
                assert list(line.get_xdata()) == pytest.approx(
                    [pair[role]["drawn_coordinate"][0] for role in ("baseline", "relation")])
        finally:
            plt.close(fig)


@pytest.mark.parametrize("component", ["parent", "directory"])
def test_generalization_refuses_symlinked_output(tmp_path, component):
    refuses_symlink(gen, tmp_path, "figs", component)


@pytest.mark.parametrize("stem", ["generalization"])
def test_generalization_confines_io(tmp_path, stem):
    refuses_external_io(tmp_path)
    refuses_output_file_symlink(tmp_path, "figs", f"{stem}.pdf")
