import pytest

from analysis import plot_fig_generalization_cells as gen
from paper_generator_checks import check_access, check_figures, refuses_symlink, refuses_external_io, refuses_output_file_symlink


def test_generalization_preserves_prediction_status_and_uncertainty():
    rows,audit,access=gen.generate();check_access(audit,access);check_figures("generalization_cells")
    assert {r["panel"] for r in rows}==set("ABC")
    assert {r["group"] for r in rows if r["panel"]=="C"}=={"2wiki_new","musique","triviaqa"}
    assert {"Pruning: density inside range","Pruning: density outside range"}<={r["group"] for r in rows}
    assert any(r["status"]=="development" for r in rows)
    for r in rows:
        if r.get("student")=="gemma3-4b": assert r["status"]=="development"
        assert r["residual"]==r["predicted"]-r["measured"]
        if r["measurement_interval"] is None:
            assert r["within"] is None and r["residual_interval"] is None
        else:
            lo,hi=r["measurement_interval"]
            assert r["within"]==(lo<=r["predicted"]<=hi)
            assert r["residual_interval"]==[r["predicted"]-hi,r["predicted"]-lo]
        if r["panel"]=="C":
            assert r["predicted"]==0 and r["candidate"]=="registered additive contrast"
    # Metadata needed to disclose unavailable evidence cannot silently vanish.
    assert any("No measurement intervals" in n for n in audit.notes)
    assert all(any(source.startswith("results/"+prefix) for source in audit.inputs) for prefix in
               ("v69-quant-confirm","v70-distill-confirm","v78-rule-confirm","v93-confirm-inputs","v46-p1-newsource","v47-p2-register","v99-scope","a5-corner-second-difference"))


@pytest.mark.parametrize("component",["parent","directory"])
def test_generalization_refuses_symlinked_output(tmp_path,component):
    refuses_symlink(gen,tmp_path,"figs",component)


def test_generalization_confines_io(tmp_path):
    refuses_external_io(tmp_path)
    refuses_output_file_symlink(tmp_path,"figs","generalization_cells.pdf")
