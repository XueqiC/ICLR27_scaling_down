import hashlib
import json
from pathlib import Path

import pytest
import torch

from analysis import eval_records as records
from analysis.tiny_test_model import TinyModel, TinyTokenizer, PROBES
from analysis import v10_quantization as v10, v12_distill as v12

FIXTURES = Path(__file__).parent / "fixtures"


def test_real_legacy_fixture_is_unchanged_and_reader_accepts_it():
    path = FIXTURES / "v12_eval_legacy.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "1bae1161f09faddd59da022582ce2bde014a68163846ffd7b0ca3cd663d12843"
    payload = records.read_eval(path)
    assert records.KEY not in payload
    assert payload == json.loads(path.read_text())


def test_synthetic_extension_of_real_fixture_recomputes_to_1e9():
    old = records.read_eval(FIXTURES / "v12_eval_legacy.json")
    new = records.read_eval(FIXTURES / "v12_eval_with_synthetic_records.json")
    assert {key: value for key, value in new.items() if key != records.KEY} == old
    assert "synthetic" in new[records.KEY]["provenance"]
    for name, rows in new[records.KEY]["evaluations"].items():
        aggregate = records.aggregate_records(rows)
        for cap in old[name]:
            assert aggregate[cap] == pytest.approx(old[name][cap], rel=0, abs=1e-9)
            assert sum(r["scored_token_count"] for r in rows if r["capability"] == cap) == old["measurement_tokens"][cap]


def test_reader_detects_record_aggregate_mismatch(tmp_path):
    payload = json.loads((FIXTURES / "v12_eval_with_synthetic_records.json").read_text())
    payload[records.KEY]["evaluations"]["dense"][0]["summed_nll"] += 1
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="mismatch"):
        records.read_eval(path)


def test_actual_records_reproduce_both_original_evaluators():
    torch.set_num_threads(1)
    model, tokenizer = TinyModel(), TinyTokenizer()
    old_quant = v10._measure_capability_losses(model, tokenizer, PROBES, "cpu")
    old_distill = v12.measure_capability_losses(model, tokenizer, PROBES, "cpu")
    quant = records.measure_capability_losses(model, tokenizer, PROBES, "cpu", distribution="j")
    distill, counts = records.measure_distillation(model, tokenizer, PROBES, "cpu", distribution="j")
    assert dict(quant) == old_quant == dict(distill) == old_distill[0]
    assert counts == old_distill[1]
    assert records.aggregate_records(quant.records) == old_quant
    assert records.aggregate_records(distill.records) == old_distill[0]
    assert quant.records == distill.records
    assert all(row["distribution"] == "j" for row in quant.records)
    assert quant.records[2]["reference_byte_count"] == 3
    for row in quant.records:
        assert len(row["prompt_reference_sha256"]) == 64


def test_distillation_adapter_writes_additive_records_and_restores(tmp_path):
    model, tokenizer = TinyModel(), TinyTokenizer()
    original_measure, original_json = v12.measure_capability_losses, v12.json
    with records.recording_evaluation("heldout"):
        losses, counts = v12.measure_capability_losses(model, tokenizer, PROBES, "cpu")
        for name in ("final", "trajectory", "baseline"):
            payload = {"version": 12, "dense": losses, "post_training": losses,
                       "capability_losses": losses, "measurement_tokens": counts}
            path = tmp_path / name / "eval.json"
            v12.write_json_atomic(path, payload)
            actual = records.read_eval(path)
            assert {key: actual[key] for key in payload} == payload
            assert set(actual[records.KEY]["evaluations"]) == {"dense", "post_training", "capability_losses"}
    assert v12.measure_capability_losses is original_measure and v12.json is original_json


def test_quantization_runner_writes_records_and_preserves_old_keys(tmp_path, monkeypatch):
    model, tokenizer = TinyModel(), TinyTokenizer()
    monkeypatch.setattr(v10, "load_text_causal_lm", lambda *args: (model, tokenizer))
    monkeypatch.setattr(v10, "build_probes", lambda *a, **kw: {c: rows * 2 for c, rows in PROBES.items()})
    monkeypatch.setattr(v10, "language_weight_parameters", lambda m: [("weight", m.weight)])
    monkeypatch.setattr(v10, "write_report", lambda *args: None)
    expected_weight = model.weight.detach().clone()
    with records.recording_evaluation("heldout"):
        v10.run_quantization("tiny", "cpu", "fp32", 4, [3], tmp_path, reference_device="cpu")
    actual = records.read_eval(tmp_path / "quant_losses.json")
    assert set(actual) == {"dense", "3"}
    assert set(actual["dense"][records.KEY]["evaluations"]) == {"dense", "3"}
    assert torch.equal(model.weight, expected_weight)
    with records.recording_evaluation("heldout"):
        v10.run_quantization("tiny", "cpu", "fp32", 4, [4], tmp_path, reference_device="cpu")
    resumed = records.read_eval(tmp_path / "quant_losses.json")
    assert set(resumed["dense"][records.KEY]["evaluations"]) == {"dense", "3", "4"}
    assert resumed["dense"][records.KEY]["evaluations"]["3"] == actual["dense"][records.KEY]["evaluations"]["3"]


def test_v10_addition_is_accepted_by_existing_v17_reader(tmp_path, monkeypatch):
    from analysis import v17_unification as v17
    loss = records.measure_capability_losses(TinyModel(), TinyTokenizer(), PROBES, "cpu")
    payload = records.with_records({"dense": loss, "3": loss}, placement="dense")
    path = tmp_path / "gemma3-4b" / "quant_losses.json"
    path.parent.mkdir()
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(v17, "PRUNE_BASE", tmp_path / "no_pruning_metadata")
    monkeypatch.setattr(v17, "_model_count", lambda *args: (4e9, "fixture"))
    rows, notes = v17._load_quantization_rows(tmp_path)
    assert len(rows) == 6


def test_adapter_restores_after_exception():
    original = v12.measure_capability_losses
    with pytest.raises(RuntimeError), records.recording_evaluation():
        raise RuntimeError("fixture")
    assert v12.measure_capability_losses is original
