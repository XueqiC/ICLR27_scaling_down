from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from analysis import v22_distill_decomp as v22


def test_accounting_preserves_signed_training_effect_and_rejects_nonfinite():
    terms = v22.decompose(2.0, 1.7, 1.0)
    assert terms["baseline_gap"] == 1.0
    assert terms["distillation_gain"] == pytest.approx(-0.3)
    assert terms["source_referenced_gap"] == pytest.approx(0.7)
    assert terms["identity_error"] == pytest.approx(0, abs=1e-12)
    with pytest.raises(ValueError, match="non-finite"):
        v22.decompose(float("nan"), 1, 2)


def test_reference_correction_is_exact_coordinate_change():
    alpha, floor, q_ratio, g_loss, q_loss = 0.2, -0.1, 0.15, 0.6, 0.9
    correction = v22.reference_correction(alpha, g_loss, q_loss)
    q_prediction = floor - alpha * math.log(q_ratio)
    common_prediction = floor - alpha * math.log(q_ratio * 4 / 27)
    assert q_prediction + correction["known_correction"] == pytest.approx(
        common_prediction + g_loss - q_loss
    )


def test_existing_pairs_complete_separate_and_reproduce_v21():
    results = v22.ROOT / "results"
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in v22.input_paths(results)}
    summary = v22.build_summary(results)
    rows = summary["rows"]
    # New completed D-ladder cells may be added without changing the V22 audit.
    # Verify complete per-capability coverage of the input inventory instead of
    # freezing the older 23-run snapshot.
    expected = {
        12: set((results / "v12-distill").glob("*/*/eval.json")),
        16: set((results / "v16-style-residual").glob("*/*/residual.json")),
    }
    assert summary["counts"] == {f"v{v}_runs": len(paths) for v, paths in expected.items()}
    expected_cells = {(f"results/{p.relative_to(results).as_posix()}", cap)
                      for paths in expected.values() for p in paths for cap in v22.CAPS}
    assert len(rows) == len(expected_cells)
    assert {(r["loss_path"], r["capability"]) for r in rows} == expected_cells
    for r in rows:
        assert r["source_referenced_gap"] == pytest.approx(r["baseline_gap"] + r["distillation_gain"])
    paired = [r for r in rows if r["student"] == "gemma3-4b" and r["run"] == "gpt-5.6-luna_full_600" and r["capability"] == "qa"]
    assert len(paired) == 2
    assert paired[0]["distilled_loss"] != paired[1]["distilled_loss"]
    fits = summary["c21_audit"]["component_fits"]
    for cap in v22.CAPS:
        params = {k: v["shared"]["fit"]["parameters"][cap] for k, v in fits.items()}
        for term in ("alpha", "floor"):
            assert params["source_referenced_gap"][term] == pytest.approx(
                params["baseline_gap"][term] + params["distillation_gain"][term]
            )
    total = fits["source_referenced_gap"]
    assert total["rotating_intercept"]["metrics"]["mae"] == pytest.approx(0.10637868388712933)
    assert total["rotating_intercept"]["metrics"]["sign_correct"] == 6
    assert fits["distillation_gain"]["rotating_intercept"]["metrics"]["mae"] > total["rotating_intercept"]["metrics"]["mae"]
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}


def test_held_out_student_outcome_cannot_change_its_prediction():
    rows = v22.build_summary(v22.ROOT / "results")["c21_audit"]["rows"]
    perturbed = copy.deepcopy(rows)
    for r in perturbed:
        if r["student"] == "Qwen3-1.7B":
            r["distillation_gain"] += 10
            r["source_referenced_gap"] += 10
    first, second = v22.component_fits(rows), v22.component_fits(perturbed)
    for component in first:
        def predictions(fit):
            return [r["predicted"] for r in fit[component]["rotating_intercept"]["records"]
                    if r["held_out_model"] == "Qwen3-1.7B"]
        assert predictions(first) == pytest.approx(predictions(second))
        for fold in first[component]["rotating_intercept"]["folds"]:
            assert set(fold["train_row_ids"]).isdisjoint(fold["test_row_ids"])
            assert len(fold["qwen_intercepts"]) == 3


def test_loader_fails_on_inconsistent_recorded_gain(tmp_path):
    results = v22.ROOT / "results"
    paths = [Path("v12-distill/gemma3-4b/gpt-5.6-luna_full_600/eval.json"),
             Path("v6-capability-geometry/gemma3-27b/prune_losses.json")]
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((results / relative).read_bytes())
    path = tmp_path / paths[0]
    payload = json.loads(path.read_text())
    payload["delta"]["math"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="recorded gain disagrees"):
        v22.load_decompositions(tmp_path)


def test_dry_run_lists_inputs_without_output_or_gpu_import(tmp_path):
    output = tmp_path / "absent"
    program = (
        "import sys; from analysis import v22_distill_decomp as v; "
        f"sys.argv=['v22','--dry-run','--output-dir',{str(output)!r}]; "
        "v.main(); assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", program], cwd=v22.ROOT, check=True, capture_output=True, text=True)
    inputs = v22.input_paths(v22.ROOT / "results")
    assert f"{len(inputs)} inputs" in result.stdout
    assert all(str(path) in result.stdout for path in inputs)
    assert "v16-style-residual/gemma3-4b" in result.stdout
    assert not output.exists()


def test_report_update_is_idempotent_and_preserves_other_sections(tmp_path):
    path = tmp_path / "report.md"
    path.write_text("# Original\n\nUnrelated pruning result.\n")
    v22.update_law_report(path, "## Audit\nTest.\n")
    once = path.read_text()
    v22.update_law_report(path, "## Audit\nTest.\n")
    assert path.read_text() == once
    assert "Unrelated pruning result." in once
