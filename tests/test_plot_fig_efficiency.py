"""CPU-only scientific aggregation and final-canvas checks for A14."""
import copy
import json
import math

import pytest

from analysis import plot_fig_efficiency as gen


@pytest.fixture(scope="module")
def data():
    audit = gen.Artifacts()
    return gen.learning_curves(audit), gen.confirmation(audit)


def test_a9_keeps_unfittable_and_partial_replicates(data):
    curves, _ = data
    assert len(curves) == 27
    for cap in gen.score.CAPS:
        part = sorted((r for r in curves if r["cap"] == cap and r["method"] == "A2"),
                      key=lambda r: r["n_measurements_per_capability"])
        assert [r["n_fitted"] for r in part] == [0, 6, 20]
        assert [r["n_failed"] for r in part] == [20, 14, 0]
        assert part[0]["median"] is part[0]["q25"] is part[0]["q75"] is None
        assert all(r["n_test_cells"] == 42 for r in part)
        assert all(len(r["replicate_sources"]) == 20 for r in part)


def test_per_state_scores_partition_the_registered_errors(data):
    _, result = data
    summary = json.loads((gen.ROOT / gen.A11 / "summary.json").read_text())
    assert result["state_order"] == ["pythia-160m@step80000", "pythia-410m@step112000",
                                     "pythia-1.4b@step48000", "pythia-1b@step48000"]
    assert len(result["records"]) == 12
    for r in result["records"]:
        cells = [c for c in summary["cells"] if (c["state"], c["capability"]) == (r["state"], r["capability"])]
        assert r["n_cells"] == len(cells) == 6
        for method in gen.score.METHODS:
            assert r["absolute_errors"][method] == [c["absolute_errors"][method] for c in cells]
            assert r["mae"][method] == pytest.approx(math.fsum(c["absolute_errors"][method] for c in cells)/6, abs=1e-15)
    for cap in gen.score.CAPS:
        for method in gen.score.METHODS:
            pooled = math.fsum(r["mae"][method] for r in result["records"] if r["capability"] == cap)/4
            assert pooled == pytest.approx(summary["by_capability"][cap]["mae"][method], abs=1e-15)
    assert [result["development_measurements_per_capability"][m] for m in gen.CONFIRM_METHODS] == [18, 36, 36]


def test_confirmation_rejects_stale_summary(monkeypatch):
    audit = gen.Artifacts()
    read = audit.read
    def stale(path):
        obj = read(path)
        if path.endswith("summary.json"):
            obj = copy.deepcopy(obj)
            obj["cells"][0]["absolute_errors"]["power_18"] += .01
        return obj
    monkeypatch.setattr(audit, "read", stale)
    with pytest.raises(ValueError, match="summary disagrees"):
        gen.confirmation(audit)


@pytest.mark.parametrize("failure", ["unpairable", "tampered"])
def test_publication_digest_mapping_still_rejects_changed_runtime(monkeypatch, failure):
    audit = gen.Artifacts()
    if failure == "unpairable":
        read = audit.read

        def without_pairs(path):
            value = read(path)
            if str(path).endswith("ANONYMIZATION_DIGESTS.json"):
                value = copy.deepcopy(value)
                for entry in value["files"].values():
                    entry["pairable"] = False
            return value

        monkeypatch.setattr(audit, "read", without_pairs)
    else:
        digest = gen.score.digest
        monkeypatch.setattr(gen.score, "digest", lambda path:
            "0" * 64 if str(path).endswith("model_registry.py") else digest(path))
    with pytest.raises(ValueError, match="Runtime/probe file changed"):
        gen.confirmation(audit)


@pytest.mark.parametrize("caps", [("math", "code"), ("qa",)])
def test_final_size_canvases_keep_labels_and_missing_fit_encoding(data, caps):
    curves, confirmation = data
    plt = gen.pyplot()
    for draw in (lambda f: gen.draw_learning(f, curves, caps),
                 lambda f: gen.draw_confirmation(f, confirmation, caps)):
        gen.apply_style("double")
        fig = plt.figure(figsize=gen.PANEL_SIZE, dpi=220)
        try:
            ax = draw(fig)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            assert not ax.get_title()
            for label in [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
                if not label.get_visible() or not label.get_text():
                    continue
                # Matplotlib may create an unused locator tick outside the view.
                if label in ax.get_yticklabels() and not ax.get_ylim()[0] <= label.get_position()[1] <= ax.get_ylim()[1]:
                    continue
                box = label.get_window_extent(renderer)
                assert box.x0 >= 0 and box.y0 >= 0
                assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
            if ax.get_xlabel().startswith("Development"):
                hollow = [l for l in ax.lines if l.get_marker() == "s" and l.get_markerfacecolor() == gen.PALETTE["white"]]
                assert len(hollow) == len(caps)  # only the partly fitted budget (18) keeps a hollow marker
                assert sorted(float(l.get_xdata()[0]) for l in hollow) == [18] * len(caps)
                assert not [l for l in ax.lines if l.get_marker() == "s" and float(l.get_xdata()[0]) == 9]
        finally:
            plt.close(fig)
    gen.apply_style("legend")
    fig = plt.figure(figsize=gen.LEGEND_SIZE, dpi=220)
    try:
        legend = gen.draw_legend(fig, caps)
        fig.canvas.draw()
        box = legend.get_window_extent(fig.canvas.get_renderer())
        assert box.x0 >= 0 and box.y0 >= 0
        assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
    finally:
        plt.close(fig)
