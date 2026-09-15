import json
import re
from collections import Counter
from statistics import mean

import pytest

from analysis import plot_fig_generalization as gen
from analysis.paper_artifacts import ROOT, pyplot
from paper_generator_checks import (
    check_access, check_figures, refuses_symlink, refuses_external_io,
    refuses_output_file_symlink,
)


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


def test_both_figures_preserve_exact_cell_coverage_and_corner_bands(generated):
    rows, audit, access = generated
    check_access(audit, access)
    check_figures("generalization")
    check_figures("generalization_cells")
    assert gen.format_pairs(rows) in (ROOT / "paper/paper/figs/generalization_mae_pairs.md").read_text()
    side = (ROOT / "paper/paper/figs/generalization_cells_sources.md").read_text()
    original = json.loads(side.split("```json\n")[1].split("\n```")[0])
    cells = [c for r in rows if r["kind"] == "mae" for c in r["cells"]]
    corners = [{k: v for k, v in r.items() if k != "kind"} for r in rows if r["kind"] == "corner"]
    # Identity includes capability: V78 shares a measurement path across caps.
    identity = lambda r: (r["panel"], r["group"], r["capability"], r["source"])
    assert Counter(map(identity, cells + corners)) == Counter(map(identity, original))
    assert corners == [r for r in original if r["measurement_interval"] is not None]
    assert len(cells) == 438 and len(corners) == 12
    assert all(r["status"] == "development" for r in corners if r["student"] == "gemma3-4b")
    scope = [r for r in rows if r["panel"] == "C"]
    assert {r["group"] for r in scope} == {"2wiki_new", "musique", "triviaqa"}
    assert all(r["kind"] == "corner" and r["predicted"] == 0
               and r["candidate"] == "registered additive contrast" for r in scope)
    assert "No frozen response-law predictions" in gen.SCOPE_NOTE
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
        assert len(fig.subfigs) == 4
        assert [len(sub.legends) for sub in fig.subfigs] == [0, 0, 0, 1]
        assert [t.get_text() for t in maes[0].get_yticklabels()] == [
            "Prune in range", "Prune in, frozen", "Prune out, frozen", "Quant group", "Distill 270M", "Distill 1B"]
        from matplotlib.text import Text
        drawn = [t.get_text() for t in fig.findobj(Text) if t.get_visible()]
        assert not any(re.search(r"\bV\d+\b|\bBase\b|\bRel\.|\bdev\.", t) for t in drawn)
        assert [t.get_text() for t in maes[1].get_yticklabels()] == ["New stages", "New quant state", "Locked rule"]
        for panel, ax in zip("AB", maes):
            part = [r for r in rows if r["panel"] == panel and r["kind"] == "mae"]
            order = list(dict.fromkeys((r["group"], r["stratum"]) for r in part))
            labels = [t.get_text() for t in ax.get_yticklabels()]
            assert len(labels) == len(order)
            for r in part:
                matching = [line for line in ax.lines if line.get_marker() == gen.MARKERS[r["capability"]]
                            and len(line.get_xdata()) == 1 and line.get_xdata()[0] == r["relation_mae"]]
                assert any(line.get_color() == (gen.GREEN if r["below_baseline"] else gen.GREY) for line in matching)
        assert "unavailable" in gen.format_pairs(rows)
        assert len(gen.format_pairs(rows).splitlines()) == 29
    finally:
        plt.close(fig)


@pytest.mark.parametrize("component", ["parent", "directory"])
def test_generalization_refuses_symlinked_output(tmp_path, component):
    refuses_symlink(gen, tmp_path, "figs", component)


@pytest.mark.parametrize("stem", ["generalization", "generalization_cells"])
def test_generalization_confines_io(tmp_path, stem):
    refuses_external_io(tmp_path)
    refuses_output_file_symlink(tmp_path, "figs", f"{stem}.pdf")
