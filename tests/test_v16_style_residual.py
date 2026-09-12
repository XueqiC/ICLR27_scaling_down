import json

import pytest

from analysis import v16_style_residual as residual


def test_residual_deltas_remove_shared_generic_shift():
    dense = {"math": 1.0, "code": 2.0, "qa": 3.0}
    distilled = {"math": 1.2, "code": 2.5, "qa": 3.1}

    deltas = residual.residual_deltas(dense, distilled, 4.0, 4.2)

    assert deltas["math"] == pytest.approx({"dL_c": 0.2, "dL_gen": 0.2, "dL_cap": 0.0})
    assert deltas["code"] == pytest.approx({"dL_c": 0.5, "dL_gen": 0.2, "dL_cap": 0.3})
    assert deltas["qa"] == pytest.approx({"dL_c": 0.1, "dL_gen": 0.2, "dL_cap": -0.1})


def test_interpretation_distinguishes_global_and_capability_change():
    assert (
        residual.interpret_delta({"dL_c": 0.2, "dL_gen": 0.2, "dL_cap": 0.0})
        == "global/style-drift-like"
    )
    assert (
        residual.interpret_delta({"dL_c": 0.2, "dL_gen": 0.0, "dL_cap": 0.2})
        == "capability-specific-like"
    )


def test_summary_writes_one_row_per_capability(tmp_path):
    run = tmp_path / "student/run"
    run.mkdir(parents=True)
    payload = {
        "student_tag": "student",
        "teacher": "teacher",
        "recipe": "full",
        "n_per_domain": 64,
        "deltas": {
            capability: {"dL_c": 0.2, "dL_gen": 0.1, "dL_cap": 0.1}
            for capability in residual.CAPABILITIES
        },
    }
    (run / "residual.json").write_text(json.dumps(payload), encoding="utf-8")

    summary = residual.write_summary(tmp_path)
    text = summary.read_text(encoding="utf-8")

    assert text.count("| student | teacher | full | 64 |") == 3
    assert all(f"| {capability} |" in text for capability in residual.CAPABILITIES)
