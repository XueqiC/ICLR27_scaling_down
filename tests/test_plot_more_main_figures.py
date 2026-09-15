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
from analysis import plot_fig_lr_pilot as lr
from analysis import plot_fig_pythia_responses as pythia
from analysis import plot_fig_drift as drift
from analysis import plot_fig_heterogeneity as heterogeneity
from analysis import plot_fig_selection_maps as selection
from analysis.paper_artifacts import ROOT, Artifacts, output_path
from paper_generator_checks import check_access, refuses_symlink, refuses_output_file_symlink
from test_paper_figure_style import check_artists

GENERATORS = (lr, pythia, drift, heterogeneity, selection)
PANELS = ("fig1_c", "fig4_a", "fig4_b", "fig4_c", "fig5_a", "fig5_b",
          "fig8_a", "fig8_b", "fig7_a", "fig7_b", "fig7_c", "fig7_d", "fig7_legend")
PREVIEWS = ("lr_pilot", "pythia_responses", "drift", "heterogeneity", "selection_maps")


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    tmp_path_factory.getbasetemp()
    directory = output_path(ROOT, "figs")
    tex = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in output_path(ROOT, "tables").glob("*.tex") if p.is_file()}
    new_modules = set(sys.modules)
    saved, previews = {}, {}
    save, combine = exports.save_panel, exports.combine_panels

    def inspect(fig, stem, kind, audit, records):
        from matplotlib.legend import Legend
        check_artists(fig)
        assert kind == "panel"
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            assert ax.xaxis.label.get_fontsize() == ax.yaxis.label.get_fontsize() == 15
            assert all(t.get_fontsize() == 14 for t in ax.get_xticklabels() + ax.get_yticklabels())
            for legend in ax.findobj(Legend):
                box = legend.get_window_extent(renderer)
                assert ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
            assert all(line.get_markersize() >= 7 for line in ax.lines if line.get_marker() in ("o", "s", "^"))
        for legend in fig.findobj(Legend):
            assert all(t.get_fontsize() == 13 and t.get_weight() == "bold" for t in legend.get_texts())
        saved[stem] = list(fig.get_size_inches())
        save(fig, stem, kind, audit, records)

    def inspect_preview(plt, panels, size):
        fig = combine(plt, panels, size)
        check_artists(fig)
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


def test_pythia_nine_states_all_frozen_cells_and_delivered_curves(generated, frozen):
    import numpy as np
    from analysis.plot_fig1_final import prune_curve
    rows, _, _ = generated[pythia]
    expected = {f"pythia-{s}@step{step}" for s in ("160m", "410m", "1.4b") for step in (16000, 64000, 143000)}
    for panel in "ab":
        part = [r for r in rows if r["panel"] == panel and r["kind"] == "measured"]
        assert {r["state"] for r in part} == expected
        assert {r["capability"] for r in part} == set(pythia.CAPS)
        for r in part:
            assert r["delta"] == frozen(r["loss_source"]) - frozen(r["dense_source"])
            assert r["cohort_source"].startswith(pythia.P36)
    for r in (r for r in rows if r["kind"] == "delivered"):
        assert r["method"] == ("median_curve" if r["capability"] == "qa" else "power")
        values = [float(prune_curve(frozen(pythia.P53), frozen(s), r["capability"], r["x"], r["method"])) for s in r["state_sources"]]
        assert r["state_predictions"] == values and r["delta"] == float(np.median(values))
    grouped = [r for r in rows if r["panel"] == "c"]
    assert {(r["bit"], r["x"]) for r in grouped} == {(b, g) for b in (3, 4, 5) for g in (32, 64, 128, 256, 512)}
    for r in grouped:
        assert r["delta"] == frozen(r["delta_source"])


def test_drift_uses_saved_ratios_both_tolerances_and_no_trajectory_reads(generated, frozen):
    rows, audit, access = generated[drift]
    assert len(rows) == 216
    assert set(audit.inputs) == {drift.V96, drift.V100}
    assert not any("/trajectory/" in p for p in access[0])
    for r in rows:
        assert r["delta"] == frozen(r["delta_source"]) == r["raw_ratio"]
        assert r["x"] == frozen(r["x_source"])/1000
        assert r["tolerance"] == frozen(r["tolerance_source"])
        assert math.isclose(r["delta"], frozen(r["effect_source"])/frozen(r["bracket_source"]), abs_tol=1e-12)
    assert {r["student"] for r in rows} == set(drift.STUDENTS)
    assert {r["tolerance"] for r in rows} == set(drift.TOLERANCES)
    for student in drift.STUDENTS:
        part = [r for r in rows if r["student"] == student]
        tight = {r["pair_id"] for r in part if r["tolerance"] == .003}
        broad = {r["pair_id"] for r in part if r["tolerance"] == .01}
        assert tight and tight < broad
    assert "zero exactly matched" in drift.CAPTION


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


def test_heterogeneity_original_twelve_exact_bars_and_family_order(generated, frozen):
    rows, _, _ = generated[heterogeneity]
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


@pytest.mark.parametrize("gen", GENERATORS)
@pytest.mark.parametrize("component", ("parent", "directory"))
def test_new_generators_refuse_symlinked_output(tmp_path, gen, component):
    refuses_symlink(gen, tmp_path, "figs", component)


@pytest.mark.parametrize("name", ("fig1_c.pdf", "fig4_a_data.json", "drift.png", "fig8_b_caption.txt", "selection_maps_files.json"))
def test_new_output_files_refuse_symlinks(tmp_path, name):
    refuses_output_file_symlink(tmp_path, "figs", name)
