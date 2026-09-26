"""Frozen-number, visual and I/O contracts for the additional main-text panels."""
import copy
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

from analysis import paper_panel_exports as exports
from analysis import paper_figure_style as style
from analysis import plot_fig_lr_pilot as lr
from analysis import plot_fig_pythia_responses as pythia
from analysis import plot_fig_drift as drift
from analysis import plot_fig_heterogeneity as heterogeneity
from analysis import plot_fig_selection_maps as selection
from analysis.paper_artifacts import ROOT, Artifacts, output_path
from paper_generator_checks import check_access, refuses_symlink, refuses_output_file_symlink
from test_paper_figure_style import check_artists, EXPECTED_PANEL_SIZES

GENERATORS = (lr, pythia, drift, heterogeneity, selection)
PANELS = ("lr_pilot_a", "lr_pilot_legend", "fig4_a", "fig4_b", "fig4_c", "fig5_a", "fig5_b",
          "fig4_legend", "fig5_legend", "fig8_a", "fig8_b", "fig8_legend",
          "fig7_a", "fig7_b", "fig7_c", "fig7_d", "fig7_legend")
PREVIEWS = ("lr_pilot", "pythia_responses", "drift", "heterogeneity", "selection_maps")


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    tmp_path_factory.getbasetemp()
    directory = output_path(ROOT, "figs")
    tex = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in output_path(ROOT, "tables").glob("*.tex") if p.is_file()}
    new_modules = set(sys.modules)
    saved, previews = {}, {}
    save, combine = exports.save_panel, exports.combine_panels

    def inspect(fig, stem, kind, audit, records, **kwargs):
        from matplotlib.legend import Legend
        if stem in EXPECTED_PANEL_SIZES:
            assert tuple(fig.get_size_inches()) == EXPECTED_PANEL_SIZES[stem]
        check_artists(fig)
        assert kind == ("legend" if stem.endswith("_legend") else style.kind_for_width(fig.get_size_inches()[0]))
        tick, label, legend_size = style.SIZES[kind]
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            assert ax.xaxis.label.get_fontsize() == ax.yaxis.label.get_fontsize() == label
            assert all(t.get_fontsize() == (7.5 if stem.startswith("fig8_") else tick)
                       for t in ax.get_yticklabels())
            assert all(t.get_fontsize() == (7.5 if stem.startswith("fig8_") else tick)
                       for t in ax.get_xticklabels())
            if stem.startswith("fig8_"):
                assert ax.get_xlabel() == "Loss change (nats)" and ax.get_ylabel() == ""
                assert ax.get_xscale() == "symlog" and ax.yaxis_inverted()
                assert not ax.patches
                if stem == "fig8_a":
                    assert ax.get_yticklabels()[0].get_text() == "Gemma 3 270M"
                    assert ax.get_yticklabels()[-1].get_text() == "Qwen3 4B"
                else:
                    assert not ax.get_yticklabels()
            for legend in ax.findobj(Legend):
                box = legend.get_window_extent(renderer)
                assert ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
            assert all(line.get_markersize() == 3.8
                       for line in ax.lines if line.get_marker() in ("o", "s", "^"))
            if stem.startswith("fig4_"):
                from matplotlib.collections import LineCollection, PathCollection
                lines = [c for c in ax.collections if isinstance(c, LineCollection)]
                markers = [c for c in ax.collections if isinstance(c, PathCollection)]
                assert Counter(c.get_linewidths()[0] for c in lines) == {.5: 3, 1.5: 3}
                assert len(markers) == 3 and len(ax.collections) == 9
                assert all(list(c.get_sizes()) == [3.2**2] for c in markers)
                assert len(ax.lines) == 1 and ax.lines[0].get_linewidth() == 1.1
                assert ax.get_yscale() == "symlog" and ax.yaxis.get_transform().linthresh == .1
            if stem == "lr_pilot_a":
                data_lines = [line for line in ax.lines if line.get_color() != ".55"]
                assert all(line.get_linewidth() == 1.1
                           for line in data_lines)
        for legend in fig.findobj(Legend):
            assert all(t.get_fontsize() == legend_size and t.get_weight() == "bold" for t in legend.get_texts())
        saved[stem] = list(fig.get_size_inches())
        save(fig, stem, kind, audit, records, **kwargs)

    def inspect_preview(plt, panels, size):
        fig = combine(plt, panels, size)
        check_artists(fig)
        if size[0] == 5.5:
            data_panels = [s for s in fig.subfigs if s.axes]
            assert max(s.bbox.y0 for s in data_panels)-min(s.bbox.y0 for s in data_panels) < .5
            strips = [s for s in fig.subfigs if s.legends]
            assert len(strips) == 1
            assert strips[0].bbox.y0 >= max(s.bbox.y1 for s in data_panels)-.5
        previews[len(previews)] = size
        return fig

    from unittest.mock import patch
    with patch.object(exports, "save_panel", inspect), patch.object(exports, "combine_panels", inspect_preview):
        results = {gen: gen.generate() for gen in GENERATORS}
    assert not {m for m in set(sys.modules)-new_modules if m.split(".")[0] in ("torch", "transformers", "datasets")}
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in tex.items())
    assert set(saved) == set(PANELS)
    assert len(previews) == 5
    for stem, size in saved.items():
        raw = (directory / f"{stem}.pdf").read_bytes()
        assert raw.startswith(b"%PDF-") and len(raw) > 1000
        box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", raw)
        assert [float(v)/72 for v in box.groups()] == pytest.approx(size)
        sidecar = json.loads((directory / f"{stem}_data.json").read_text())
        assert sidecar["size_inches"] == size
        assert (directory / f"{stem}_caption.txt").stat().st_size > 100
        assert (directory / f"{stem}_sources.md").stat().st_size > 100
    from PIL import Image
    for stem in PREVIEWS:
        manifest = json.loads((directory / f"{stem}_files.json").read_text())
        preview = next(f for f in manifest if f["file"].endswith(".png"))
        with Image.open(directory / preview["file"]) as image:
            # Agg truncates fractional pixels, including 2.05*220=450.999... .
            assert all(abs(pixels - inches*220) < 1.01 for pixels, inches in zip(image.size, preview["size_inches"]))
        assert (directory / f"{stem}_caption.txt").stat().st_size > 100
    for rows, audit, access in results.values():
        check_access(audit, access)
        assert not any(Path(p).suffix == ".tex" for p in access[1])
    return results


@pytest.fixture(scope="module")
def response_rows():
    # Keep these two generators independently testable without exporting the
    # unrelated figures in the broad integration fixture above.
    result = {}
    for gen in (pythia, heterogeneity):
        audit = Artifacts()
        result[gen] = gen.build(audit), audit
    return result


@pytest.fixture(scope="module")
def frozen():
    cache = {}

    def read(ref):
        path, _, pointer = ref.partition("#")
        if path not in cache:
            cache[path] = json.loads((ROOT / path).read_text())
        value = cache[path]
        for key in pointer.lstrip("/").split("/") if pointer else []:
            key = key.replace("~1", "/").replace("~0", "~")
            value = value[int(key)] if isinstance(value, list) else value[key]
        return value
    return read


def test_lr_exact_final_positive_checkpoints_and_peak_rates(generated, frozen):
    rows, _, _ = generated[lr]
    assert len(rows) == 18
    assert Counter((r["student"], r["learning_rate"]) for r in rows) == Counter(
        {(s, float(rate)): 3 for s in lr.STUDENTS for rate in lr.RATES})
    for r in rows:
        assert r["delta"] == frozen(r["delta_source"])
        assert r["learning_rate"] == float(frozen(r["learning_rate_source"]).removeprefix("lrpilot_"))
        path = ROOT / r["delta_source"].split("#")[0]
        checkpoints = [json.loads(p.read_text()) for p in path.parents[1].glob("*/eval.json")]
        assert r["updates"] == max(p["updates"] for p in checkpoints if p["processed_tokens"] > 0)
        assert r["updates"] == 8 and r["processed_tokens"] == 37903


def test_pythia_nine_states_all_frozen_cells_without_delivered_rows(response_rows, frozen):
    rows, audit = response_rows[pythia]
    assert {r["kind"] for r in rows} == {"measured"}
    manifest, register = frozen(pythia.P36), frozen(pythia.P53)
    assert pythia.P53 in audit.inputs
    expected = {f"pythia-{s}@step{step}" for s in ("160m", "410m", "1.4b") for step in (16000, 64000, 143000)}
    for panel, directory, dense in (("a", "v6-capability-geometry", "1.0"),
                                    ("b", "v10-quantization", "dense")):
        part = [r for r in rows if r["panel"] == panel]
        assert {r["state"] for r in part} == expected
        assert {r["capability"] for r in part} == set(pythia.CAPS)
        paths = [p for p in manifest["input_sha256"] if p.startswith(f"results/{directory}/")]
        assert len(paths) == 9
        expected_cells = Counter((exports.ref(path, key, cap), exports.ref(path, dense, cap))
                                 for path in paths for key in frozen(path)
                                 if not key.startswith("_") and key != "dense" for cap in pythia.CAPS)
        assert Counter((r["loss_source"], r["dense_source"]) for r in part) == expected_cells
        for path in paths:
            digest = register["dev_hashes"][path] if panel == "a" else manifest["input_sha256"][path]
            assert audit.matches_digest(digest, audit.inputs[path])
        for r in part:
            path, pointer = r["loss_source"].split("#")
            assert r["state"] == path.split("/")[-2].replace("--step", "@step")
            assert r["x"] == float(pointer.split("/")[1])
            assert r["delta"] == frozen(r["loss_source"]) - frozen(r["dense_source"])
            assert r["cohort_source"] == exports.ref(pythia.P36, "input_sha256", path)
            assert frozen(r["cohort_source"]) == manifest["input_sha256"][path]
    grouped = [r for r in rows if r["panel"] == "c"]
    expected_group_states = {f"pythia-{s}@step{step}" for s in ("160m", "410m", "1.4b")
                             for step in (16000, 143000)}
    assert len(grouped) == 54
    assert Counter((r["state"], r["bit"], r["x"]) for r in grouped) == Counter(
        {(s, b, g): 1 for s in expected_group_states for b in (3, 4, 5) for g in (64, 128, 256)})
    assert pythia.Q69 + "compare.json" not in audit.inputs
    assert audit.matches_digest(frozen(pythia.Q69 + "freeze.json")["provenance"]["develop_sha256"],
                                audit.inputs[pythia.Q69 + "develop.json"])
    for r in grouped:
        assert r["delta_source"].startswith(pythia.Q69 + "develop.json#/dev_rows/")
        assert r["delta"] == frozen(r["delta_source"])
        assert frozen(r["config_source"]) == f"b{r['bit']}_g{r['x']}"


@pytest.mark.parametrize("source,arm", [(pythia.P53, "pruning"), (pythia.P36, "quantization"),
                                        (pythia.Q69 + "freeze.json", "development")])
def test_pythia_rejects_changed_measured_input_digests(monkeypatch, source, arm):
    audit = Artifacts()
    read = audit.read

    def altered(path):
        data = read(path)
        if path == source:
            data = copy.deepcopy(data)
            if arm == "development":
                data["provenance"]["develop_sha256"] = "0" * 64
            else:
                hashes = data["dev_hashes" if arm == "pruning" else "input_sha256"]
                directory = "v6-capability-geometry" if arm == "pruning" else "v10-quantization"
                key = next(p for p in hashes if p.startswith(f"results/{directory}/"))
                hashes[key] = "0" * 64
        return data

    monkeypatch.setattr(audit, "read", altered)
    with pytest.raises(ValueError, match=f"{arm} digest mismatch"):
        pythia.build(audit)


def test_pythia_exported_rows_and_captions_match_measured_design(generated):
    rows, _, _ = generated[pythia]
    directory = output_path(ROOT, "figs")
    for panel in "abc":
        sidecar = json.loads((directory / f"fig4_{panel}_data.json").read_text())
        assert sidecar["records"] == [r for r in rows if r["panel"] == panel]
        assert {r["kind"] for r in sidecar["records"]} == {"measured"}
    for stem in ("fig4_a", "fig4_b", "fig4_c", "fig4_legend", "pythia_responses"):
        caption = " ".join((directory / f"{stem}_caption.txt").read_text().split()).lower()
        assert "state traces are 0.5 pt at alpha 0.3" in caption
        assert "median lines are 1.5 pt with 3.2-pt markers" in caption
        assert "prediction" not in caption and "band" not in caption


def test_drift_uses_saved_ratios_both_tolerances_and_no_trajectory_reads(generated, frozen):
    rows, audit, access = generated[drift]
    assert len(rows) == 216
    assert set(audit.inputs) == {drift.V96, drift.V100}
    assert not any("/trajectory/" in p for p in access[0])
    for r in rows:
        assert r["delta"] == frozen(r["delta_source"]) == r["raw_ratio"]
        assert r["x"] == frozen(r["x_source"])/1000
        assert r["tolerance"] == frozen(r["tolerance_source"])
        assert r["band"] == frozen(r["band_source"])
        assert math.isclose(r["delta"], frozen(r["effect_source"])/frozen(r["bracket_source"]), abs_tol=1e-12)
    assert {r["student"] for r in rows} == set(drift.STUDENTS)
    assert {r["tolerance"] for r in rows} == set(drift.TOLERANCES)
    for student in drift.STUDENTS:
        part = [r for r in rows if r["student"] == student]
        tight = {r["pair_id"] for r in part if r["tolerance"] == .003}
        broad = {r["pair_id"] for r in part if r["tolerance"] == .01}
        assert tight and tight < broad
    assert "zero exactly matched" in drift.CAPTION
    directory = output_path(ROOT, "figs")
    for panel in "ab":
        sidecar = json.loads((directory / f"fig5_{panel}_data.json").read_text())
        assert sidecar["records"] == [r for r in rows if r["panel"] == panel]
    assert "Both matching tolerances (1% and 0.3%) are pooled" in drift.CAPTION
    assert "sidecar keeps every pair" in drift.CAPTION


@pytest.mark.parametrize("section,field", [("fixed_budget", "raw_ratio"), ("fixed_budget", "common_T"),
                                           ("fixed_budget", "reuse_bracket"), ("raw_response_curves", "E")])
def test_missing_drift_field_stops_that_panel_without_fallback(frozen, section, field):
    summary = copy.deepcopy(frozen(drift.V100))
    if section == "fixed_budget":
        row = next(r for r in summary[section]["endpoint_ratios"] if r["student"] == "gemma3-1b")
        del row[field]
    else:
        for curve in summary[section]:
            if curve["student"] == "gemma3-1b":
                for point in curve["points"]:
                    del point[field]
    audit = Artifacts()
    audit.read = lambda p: summary if p == drift.V100 else frozen(p)
    rows, unavailable = drift.build(audit)
    assert set(unavailable) == {"a"} and field in unavailable["a"]
    assert {r["student"] for r in rows} == {"gemma3-4b"}


def test_heterogeneity_original_twelve_exact_dots_and_family_order(response_rows, frozen):
    rows, _ = response_rows[heterogeneity]
    assert len(rows) == 72
    originals = [r for r in frozen(heterogeneity.SOURCE) if r["cohort"] == "panel"]
    for panel, key in (("a", "dl07"), ("b", "dl4")):
        part = [r for r in rows if r["panel"] == panel]
        assert {r["model"] for r in part} == {r["model"] for r in originals}
        assert len(part) == 36
        ordered = sorted((r for r in part if r["capability"] == "math"), key=lambda r: r["order"])
        assert [r["series"] for r in ordered] == sorted(r["series"] for r in ordered)
        for r in part:
            assert r["delta"] == frozen(r["delta_source"])
            assert f"/{key}/" in r["delta_source"] and frozen(r["cohort_source"]) == "panel"


def test_heterogeneity_dot_positions_labels_and_aligned_rows(response_rows):
    import numpy as np
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.colors import to_rgba
    from matplotlib.markers import MarkerStyle
    from analysis.paper_artifacts import pyplot
    rows, _ = response_rows[heterogeneity]
    plt = pyplot()
    expected_labels = ["Gemma 3 270M", "Gemma 3 1B", "Gemma 3 4B", "Gemma 3 12B",
                       "Gemma 3 27B", "Gemma 4 31B", "Muse 30B", "OLMo 3 7B",
                       "OLMo 3 32B", "Qwen3 0.6B", "Qwen3 1.7B", "Qwen3 4B"]
    bounds = []
    for panel in "ab":
        size = heterogeneity.PANEL_SIZES[panel]
        assert size == EXPECTED_PANEL_SIZES[f"fig8_{panel}"]
        style.apply_style(style.kind_for_width(size[0]))
        fig = plt.figure(figsize=size)
        try:
            ax = heterogeneity.draw_panel(fig, rows, panel)
            style.prepare_figure(fig)
            check_artists(fig)
            bounds.append((ax.bbox.y0 / fig.dpi, ax.bbox.y1 / fig.dpi))
            assert ax.get_xlabel() == "Loss change (nats)" and not ax.get_ylabel()
            assert ax.get_xscale() == "symlog" and ax.yaxis_inverted()
            assert ax.xaxis.get_transform().linthresh == .1
            assert ax.xaxis.get_transform().linscale == .6
            assert list(ax.get_xticks()) == [-1, -.1, 0, .1, 1, 5]
            assert [t.get_text() for t in ax.get_xticklabels()] == ["−1", "−0.1", "0", "0.1", "1", "5"]
            assert list(ax.get_yticks()) == list(range(12))
            assert [t.get_text() for t in ax.get_yticklabels()] == (expected_labels if panel == "a" else [])
            assert all(t.get_rotation() == 0 for t in ax.get_yticklabels())
            assert not ax.patches
            dots = [c for c in ax.collections if isinstance(c, PathCollection)]
            assert len(dots) == 3
            for points, cap, marker, offset in zip(dots, heterogeneity.CAPS, ("o", "s", "^"), (-.18, 0, .18)):
                part = sorted((r for r in rows if r["panel"] == panel and r["capability"] == cap),
                              key=lambda r: r["order"])
                np.testing.assert_array_equal(points.get_offsets()[:, 0], [r["delta"] for r in part])
                np.testing.assert_allclose(points.get_offsets()[:, 1], np.arange(12) + offset)
                glyph = MarkerStyle(marker)
                np.testing.assert_array_equal(points.get_paths()[0].vertices,
                                              glyph.get_path().transformed(glyph.get_transform()).vertices)
                np.testing.assert_allclose(points.get_facecolors(), [to_rgba(heterogeneity.COLORS[cap])])
                np.testing.assert_allclose(points.get_edgecolors(), [to_rgba(style.PALETTE["white"])])
                assert list(points.get_sizes()) == [3.8**2] and list(points.get_linewidths()) == [.4]
                assert all(ax.get_xlim()[0] < r["delta"] < ax.get_xlim()[1] for r in part)
            rules = [c for c in ax.collections if isinstance(c, LineCollection)]
            assert len(rules) == 17
            spread = [c for c in rules if c.get_zorder() == 2]
            assert len(spread) == 12
            for order, bar in enumerate(spread):
                values = [r["delta"] for r in rows if r["panel"] == panel and r["order"] == order]
                assert len(values) == 3 and len(bar.get_segments()) == 1
                np.testing.assert_array_equal(bar.get_segments()[0],
                                              [[min(values), order], [max(values), order]])
                np.testing.assert_allclose(bar.get_colors(), [to_rgba(style.PALETTE["dense"])])
                assert list(bar.get_linewidths()) == [.7]
                assert bar.get_linestyles()[0][1] is None
            rules = [c for c in rules if c.get_zorder() != 2]
            assert [list(c.get_segments()[0][:, 1]) for c in rules[:-1]] == [[v, v] for v in (4.5, 5.5, 6.5, 8.5)]
            assert all(list(c.get_linewidths()) == [.5] for c in rules[:-1])
            assert list(rules[-1].get_segments()[0][:, 0]) == [0, 0]
            assert list(rules[-1].get_linewidths()) == [.7]
            assert all(line.get_visible() and line.get_alpha() <= .3 for line in ax.get_ygridlines())
            renderer = fig.canvas.get_renderer()
            for labels in (ax.get_xticklabels(), ax.get_yticklabels()):
                boxes = [t.get_window_extent(renderer) for t in labels]
                assert all(not a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i+1:])
        finally:
            plt.close(fig)
    assert bounds[0] == pytest.approx(bounds[1])
    assert "Grouped bars" not in heterogeneity.CAPTION and "rotated" not in heterogeneity.CAPTION
    assert "grey spread bar" in heterogeneity.CAPTION


@pytest.mark.parametrize("panel", "abc")
def test_pythia_state_traces_median_markers_and_scales(response_rows, panel):
    import numpy as np
    from matplotlib.collections import LineCollection, PathCollection, PolyCollection
    from matplotlib.colors import to_rgba
    from matplotlib.markers import MarkerStyle
    from analysis.paper_artifacts import pyplot
    rows, _ = response_rows[pythia]
    plt = pyplot()
    style.apply_style("panel")
    fig = plt.figure(figsize=pythia.PANEL_SIZE)
    try:
        ax = pythia.draw_panel(fig, rows, panel)
        style.prepare_figure(fig)
        check_artists(fig)
        medians = [c for c in ax.collections if isinstance(c, LineCollection) and c.get_linewidths()[0] == 1.5]
        traces = [c for c in ax.collections if isinstance(c, LineCollection) and c.get_linewidths()[0] == .5]
        markers = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(medians) == len(traces) == len(markers) == 3
        assert not any(isinstance(c, PolyCollection) for c in ax.collections)
        assert len(ax.lines) == 1 and len(ax.collections) == 9
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0, 0])
        assert ax.lines[0].get_linestyle() == "-"
        selectors = [("bit", b, pythia.BIT_COLORS[b], pythia.BIT_MARKERS[b]) for b in (3, 4, 5)] if panel == "c" else [
            ("capability", c, pythia.COLORS[c], "o") for c in pythia.CAPS]
        for line, trace, points, (field, key, color, marker) in zip(medians, traces, markers, selectors):
            measured = [r for r in rows if r["panel"] == panel and r["kind"] == "measured" and r[field] == key]
            count = 6 if panel == "c" else 9
            states = sorted({r["state"] for r in measured})
            assert len(states) == count
            by_state = {s: {r["x"]: r["delta"] for r in measured if r["state"] == s} for s in states}
            common = sorted(set.intersection(*(set(values) for values in by_state.values())))
            segments = trace.get_segments()
            assert len(segments) == count
            for segment, state in zip(segments, states):
                np.testing.assert_array_equal(segment, [[x, by_state[state][x]] for x in common])
                assert np.all(ax.get_ylim()[0] < segment[:, 1])
                assert np.all(segment[:, 1] < ax.get_ylim()[1])
            np.testing.assert_allclose(trace.get_colors(), [to_rgba(color, .3)])
            assert trace.get_alpha() == .3 and trace.get_zorder() == 2
            assert list(trace.get_linewidths()) == [.5]
            assert trace.get_linestyles()[0][1] is None
            assert len(line.get_segments()) == 1
            coordinates = line.get_segments()[0]
            assert list(coordinates[:, 0]) == common
            values = [[by_state[s][x] for s in states] for x in common]
            np.testing.assert_array_equal(coordinates[:, 1], np.median(values, axis=1))
            np.testing.assert_array_equal(points.get_offsets(), coordinates)
            glyph = MarkerStyle(marker)
            np.testing.assert_array_equal(points.get_paths()[0].vertices,
                                          glyph.get_path().transformed(glyph.get_transform()).vertices)
            np.testing.assert_allclose(line.get_colors(), [to_rgba(color)])
            np.testing.assert_allclose(points.get_facecolors(), [to_rgba(color)])
            assert list(points.get_sizes()) == [3.2**2]
            assert list(points.get_linewidths()) == [.4] and points.get_zorder() == 4
            assert list(line.get_linewidths()) == [1.5] and line.get_zorder() == 3
            assert line.get_linestyles()[0][1] is None
        assert ax.get_yscale() == "symlog" and ax.yaxis.get_transform().linthresh == .1
        assert list(ax.get_yticks()) == ([-1, 0, .1, 1, 10] if panel in "ab" else [0, .1, 1, 10])
        assert [t.get_text() for t in ax.get_yticklabels()] == (
            ["$-$1", "0", "0.1", "1", "10"] if panel in "ab" else ["0", "0.1", "1", "10"])
        if panel == "a":
            assert ax.get_ylim() == (-1.2, 12)
            assert list(medians[0].get_segments()[0][:, 0]) == [.6, .7, .8, .9, 1.]
        if panel == "b":
            assert ax.get_ylim() == (-1.2, 30)
        if panel == "c":
            assert ax.get_xscale() == "log" and ax.xaxis.get_transform().base == 2
            assert [t.get_text() for t in ax.get_xticklabels()] == ["64", "128", "256"]
            assert ax.get_ylim() == (-.025, 22)
        caption = " ".join(pythia.CAPTIONS[panel].split()).lower()
        assert "thin faint lines" in caption and "measured loss change from its own dense loss" in caption
        assert "bold line with markers" in caption and "pointwise median" in caption
        assert f"measured in all {'six' if panel == 'c' else 'nine'} states" in caption
        assert "symmetric log" in caption and "0.1-nat linear threshold" in caption
        assert "band" not in caption and "prediction" not in caption
        if panel == "c":
            assert "six development states in v69 develop.json" in caption
            assert "64, 128 and 256" in caption
    finally:
        plt.close(fig)


def test_pythia_summary_excludes_partial_cohorts_and_rejects_duplicates():
    import numpy as np
    rows = [{"state": s, "x": x, "delta": delta} for s, x, delta in
            (("one", 3, 1), ("two", 3, 9), ("one", 4, -2), ("two", 4, 0), ("one", 5, 100))]
    before = copy.deepcopy(rows)
    xs, values, median = pythia.state_summary(rows, 2)
    assert xs == [3, 4]
    np.testing.assert_array_equal(values, [[1, 9], [-2, 0]])
    np.testing.assert_array_equal(median, [5, -1])
    assert rows == before
    with pytest.raises(ValueError, match="Duplicate state"):
        pythia.state_summary(rows + [rows[0]], 2)
    with pytest.raises(ValueError, match="Expected 3 states"):
        pythia.state_summary(rows, 3)
    with pytest.raises(ValueError, match="No response coordinates shared"):
        pythia.state_summary([rows[0], rows[3]], 2)


@pytest.mark.parametrize("gen", [pythia, heterogeneity])
def test_response_figure_legends_fit_and_explain_marks(gen):
    from matplotlib.lines import Line2D
    from analysis.paper_artifacts import pyplot
    plt = pyplot()
    style.apply_style("legend")
    fig = plt.figure(figsize=(5.5, .3))
    try:
        legend = gen.draw_legend(fig)
        check_artists(fig)
        labels = {t.get_text(): h for t, h in zip(legend.get_texts(), legend.legend_handles)}
        expected = {"Math", "Code", "QA"}
        if gen is pythia:
            expected |= {"3 bit", "4 bit", "5 bit", "One state"}
            state = labels["One state"]
            assert isinstance(state, Line2D) and state.get_color() == style.PALETTE["dense"]
            assert state.get_linewidth() == .5 and state.get_linestyle() == "-"
            assert state.get_marker() in (None, "None", "none", "")
            for bit, marker in zip((3, 4, 5), ("o", "s", "^")):
                assert labels[f"{bit} bit"].get_color() == pythia.BIT_COLORS[bit]
                assert labels[f"{bit} bit"].get_marker() == marker
        for name, cap, marker in zip(("Math", "Code", "QA"), gen.CAPS, ("o", "s", "^")):
            handle = labels[name]
            assert isinstance(handle, Line2D) and handle.get_color() == gen.COLORS[cap]
            if gen is heterogeneity:
                assert handle.get_marker() == marker and handle.get_linestyle() == "None"
                assert handle.get_markeredgecolor() == style.PALETTE["white"]
                assert handle.get_markeredgewidth() == .4
        assert set(labels) == expected
        assert all(t.get_fontsize() == 8.5 for t in legend.get_texts())
    finally:
        plt.close(fig)


def test_selection_maps_match_all_v80_cells_and_oracle_marks(generated, frozen):
    rows, _, _ = generated[selection]
    assert len(rows) == 272
    assert Counter(r["panel"] for r in rows) == Counter({p: 68 for p in "abcd"})
    for r in rows:
        assert r["method"] == frozen(r["method_source"])
        assert r["method_index"] == selection.METHODS.index(r["method"])
        assert r["oracle_agreement"] == frozen(r["agreement_source"])
        assert r["method"] == frozen(r["frozen_choice_source"])["method"]
        assert r["budget"] == frozen(r["budget_source"])
    assert Counter(r["panel"] for r in rows if not r["oracle_agreement"]) == Counter({"a": 1, "b": 3, "c": 16})
    from analysis.paper_artifacts import pyplot
    from analysis.paper_figure_style import apply_style
    plt = pyplot()
    apply_style("panel")
    vertical = []
    for panel, mismatches in zip("abcd", (1, 3, 16, 0)):
        fig = plt.figure(figsize=selection.PANEL_SIZES[panel])
        try:
            ax = selection.draw_panel(fig, rows, panel)
            check_artists(fig)
            assert len(fig.axes) == 1 and len(ax.images) == 1
            assert len(ax.get_yticklabels()) == (4 if panel == "a" else 0)
            assert list(ax.get_yticks()) == [0, 1, 2, 3]
            assert ax.get_ylim() == (3.5, -.5)
            assert ax.get_xlabel() == "Storage budget (%)"
            assert [t.get_text() for t in ax.get_xticklabels()] == ["20", "100"]
            assert sum(line.get_marker() == "o" for line in ax.lines) == mismatches
            matrix = ax.images[0].get_array()
            assert matrix.shape == (4, 17)
            for r in rows:
                if r["panel"] == panel:
                    assert matrix[r["row"], r["column"]] == r["method_index"]
            vertical.append((ax.bbox.y0/fig.dpi, ax.bbox.y1/fig.dpi))
        finally:
            plt.close(fig)
    assert all(bounds == pytest.approx(vertical[0]) for bounds in vertical)
    directory = output_path(ROOT, "figs")
    for panel in "abcd":
        sidecar = json.loads((directory / f"fig7_{panel}_data.json").read_text())
        assert sidecar["records"] == [r for r in rows if r["panel"] == panel]
    manifest = json.loads((directory / "selection_maps_files.json").read_text())
    assert {f["file"] for f in manifest} == {
        "fig7_a.pdf", "fig7_b.pdf", "fig7_c.pdf", "fig7_d.pdf", "fig7_legend.pdf", "selection_maps.png"}
    assert not (directory / "fig7.pdf").exists()
    caption = " ".join((directory / "fig7_caption.txt").read_text().split())
    assert "(a) math, (b) code, (c) question answering, (d) largest loss increase across the three" in caption


@pytest.mark.parametrize("gen", GENERATORS)
@pytest.mark.parametrize("component", ("parent", "directory"))
def test_new_generators_refuse_symlinked_output(tmp_path, gen, component):
    refuses_symlink(gen, tmp_path, "figs", component)


@pytest.mark.parametrize("name", ("lr_pilot_a.pdf", "fig4_a_data.json", "drift.png", "fig8_b_caption.txt", "selection_maps_files.json"))
def test_new_output_files_refuse_symlinks(tmp_path, name):
    refuses_output_file_symlink(tmp_path, "figs", name)
