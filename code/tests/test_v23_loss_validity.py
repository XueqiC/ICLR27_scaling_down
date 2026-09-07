import json
import math
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from analysis import v6_capability_geometry as geometry
from analysis import v23_loss_validity as validity


@pytest.fixture
def dataset_rows(monkeypatch):
    rows = {}
    for key, (dataset, _, _) in validity.BENCHMARKS.items():
        rows[dataset] = [
            {"problem": f"math {i}", "solution": f"solution {i}", "answer": f"{i}",
             "text": f"code {i}", "code": f"def answer(): return {i}",
             "question": f"question {i}", "context": {"title": ["Fact"], "sentences": [["One."]]},
             "prompt": f"def problem_{i}():\n", "canonical_solution": f"    return {i}\n",
             "task_id": f"HumanEval/{i}"}
            for i in range(12)
        ]
        if key == "math_gsm8k":
            for i, row in enumerate(rows[dataset]):
                row["answer"] = f"Compute {i}.\n#### {i}"
    calls = []
    module = ModuleType("datasets")

    def load_dataset(dataset, *args, split):
        calls.append((dataset, args, split))
        return rows[dataset]

    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    return rows, calls


def test_secondary_opt_in_preserves_original_sampling_order_and_behavior(dataset_rows):
    _, calls = dataset_rows
    original = geometry.build_probes(8, seed=5)
    assert list(original) == ["math", "code", "qa"]
    assert len(calls) == 3
    extended = geometry.build_probes(8, seed=5, include_secondary=True)
    assert list(extended) == list(validity.BENCHMARKS)
    assert {key: extended[key] for key in original} == original
    rng = np.random.default_rng(5)
    expected_math = rng.choice(12, 8, replace=False)
    expected_code = rng.choice(12, 8, replace=False)
    expected_qa = rng.choice(12, 8, replace=False)
    assert [p["prompt"] for p in original["math"]] == [
        f"Problem: math {i}\nSolution:" for i in expected_math]
    assert [p["completion"] for p in original["code"]] == [
        f"def answer(): return {i}" for i in expected_code]
    assert [p["answer"] for p in original["qa"]] == [str(i) for i in expected_qa]
    assert validity.measurement_probes(8, 5) == {k: v[1::2] for k, v in extended.items()}
    assert all(call[2] in ("test", "validation") for call in calls)


def test_gsm8k_extracts_final_answer_but_scores_entire_worked_solution():
    row = {"question": "How many?", "answer": "Work.\n#### 1,234"}
    probe = geometry.secondary_probe("math_gsm8k", row)
    assert probe == {"prompt": "Problem: How many?\nSolution:",
                     "completion": " Work.\n#### 1,234", "answer": "1,234"}
    with pytest.raises(ValueError, match="delimiter"):
        geometry.secondary_probe("math_gsm8k", {**row, "answer": "missing"})


def test_humaneval_target_is_canonical_continuation_without_tests():
    row = {"prompt": 'def one():\n    """Return one."""\n',
           "canonical_solution": "    return 1\n", "test": "secret harness",
           "task_id": "HumanEval/0"}
    probe = geometry.secondary_probe("code_humaneval", row)
    assert probe["prompt"] == row["prompt"]
    assert probe["completion"] == row["canonical_solution"]
    assert "secret harness" not in json.dumps(probe)
    assert probe["completion"] not in probe["prompt"]


@pytest.mark.parametrize("context", [
    {"title": ["France", "Capital"], "sentences": [["Country."], ["Paris.", "City."]]},
    [["France", ["Country."]], ["Capital", ["Paris.", "City."]]],
])
def test_hotpotqa_context_and_short_answer(context):
    probe = geometry.secondary_probe("qa_hotpotqa", {
        "context": context, "question": "Where?", "answer": "Paris"})
    assert probe["prompt"] == ("Context:\nFrance: Country.\nCapital: Paris. City."
                               "\n\nQuestion: Where?\nAnswer:")
    assert probe["completion"] == " Paris"
    assert probe["answer"] == "Paris"


class DigitTokenizer:
    name_or_path = "gemma-test"
    bos_token_id = 0

    def __call__(self, text, *, return_tensors, truncation, max_length, add_special_tokens):
        ids = ([0] if add_special_tokens else []) + [int(c) for c in text]
        return SimpleNamespace(input_ids=torch.tensor([ids[:max_length]], dtype=torch.long))


class FixedLogitsModel(torch.nn.Module):
    def forward(self, input_ids, **kwargs):
        self.last_ids = input_ids
        logits = torch.zeros((*input_ids.shape, 4))
        logits[:, 0, 1] = -10  # prompt CE must be excluded
        logits[:, 1:, 2] = math.log(3)
        return SimpleNamespace(logits=logits)


def test_teacher_forced_target_mask_bos_truncation_and_length_normalization():
    model = FixedLogitsModel()
    total, count = geometry.completion_loss(model, DigitTokenizer(), "1", "23333", "cpu", max_len=4)
    assert model.last_ids.tolist() == [[0, 1, 2, 3]]
    assert count == 2
    # Target probabilities: p(2)=3/6, p(3)=1/6. No prompt CE, target BOS or EOS.
    assert float(total) == pytest.approx(-math.log(0.5) - math.log(1 / 6))
    measured = validity.measure_losses(model, DigitTokenizer(), {
        "math": [{"prompt": "1", "completion": "23333"}]}, "cpu", max_len=4)
    assert measured["math"]["L_c"] == pytest.approx(float(total) / 2)


def test_example_mean_is_not_corpus_token_weighting():
    def loss(*args, **kwargs):
        return (1.0, 1) if args[3] == "short" else (27.0, 9)

    measured = validity.measure_losses(torch.nn.Linear(1, 1), None, {
        "math": [{"prompt": "p", "completion": "short"},
                 {"prompt": "p", "completion": "long"}]}, "cpu", loss_function=loss)
    assert measured["math"]["L_c"] == 2.0  # (1/1 + 27/9) / 2
    assert measured["math"]["token_weighted_L_c"] == 2.8
    assert measured["math"]["n_tokens"] == 10


@pytest.mark.parametrize("bad", [(0.0, 0), (float("nan"), 1)])
def test_empty_or_nonfinite_target_loss_fails(bad):
    with pytest.raises(RuntimeError, match="empty/nonfinite"):
        validity.measure_losses(torch.nn.Linear(1, 1), None, {
            "math": [{"prompt": "p", "completion": "x"}]}, "cpu",
            loss_function=lambda *a, **kw: bad)


def test_dry_run_cli_prints_all_probes_and_loads_no_model(tmp_path, monkeypatch, dataset_rows, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run must not load a model or seed CUDA")

    monkeypatch.setattr(validity, "load_text_causal_lm", forbidden)
    monkeypatch.setattr(validity, "seed_everything", forbidden)
    out = tmp_path / "results"
    validity.main(["--model", "gemma3-270m", "--dry-run", "--n-probe", "8",
                   "--prune-density", "0.9", "0.8", "--quant-bits", "8", "4",
                   "--output-base", str(out)])
    text = capsys.readouterr().out
    for name in validity.NAMES.values():
        assert name in text
    assert text.count("4 measurement probes") == 6
    assert text.count("[would measure]") == 5
    assert "google/gemma-3-270m" in text
    assert not out.exists()


def test_guard_checks_resolved_model_and_adapter_metadata(tmp_path, monkeypatch):
    from analysis import model_registry

    monkeypatch.delenv("SDL_ALLOW_PRC", raising=False)
    monkeypatch.setitem(model_registry.MODEL_REGISTRY, "safe-looking", {"hf_id": "Qwen/Qwen3-4B"})
    with pytest.raises(RuntimeError, match="prohibits"):
        validity.resolve_source("safe-looking", None, None)
    (tmp_path / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "Qwen/Qwen3-4B"}))
    with pytest.raises(RuntimeError, match="prohibits"):
        validity.resolve_source("gemma3-270m", None, tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    with pytest.raises(RuntimeError, match="prohibits"):
        validity.resolve_source("gemma3-270m", str(tmp_path), None)


def test_each_cell_reloads_v12_delta_and_does_not_accumulate_compression(tmp_path, monkeypatch):
    loaded, seen = [], []
    torch.save(torch.tensor([[0.25]]), tmp_path / "weight.pt")
    (tmp_path / "delta_manifest.json").write_text(json.dumps({
        "format": "v12-full-finetune-delta-v1", "base_model": "source",
        "parameters": [{"name": "weight", "file": "weight.pt"}],
    }))

    def load(source, dtype):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(1)
        loaded.append(source)
        return model, SimpleNamespace(name_or_path="same-tokenizer")

    def prune(model, density, seed):
        seen.append(float(model.weight.item()))
        model.weight.data.mul_(density)
        return 0.5

    def quantize(model, bits):
        seen.append(float(model.weight.item()))

    monkeypatch.setattr(validity, "load_text_causal_lm", load)
    monkeypatch.setattr(validity, "seed_everything", lambda seed: None)
    monkeypatch.setattr(validity, "apply_global_magnitude_pruning", prune)
    monkeypatch.setattr(validity, "_apply_quantization", quantize)
    monkeypatch.setattr(validity, "measure_losses", lambda *a: {})
    for cell in validity.compression_cells([0.9, 0.8], [4]):
        validity.evaluate_cell("source", "source", tmp_path, cell, {}, "cpu", 1024)
    assert loaded == ["source"] * 4
    assert seen == [1.25, 1.25, 1.25]


def _fake_measurement(cell, probes):
    data = {}
    for i, key in enumerate(probes):
        base = 1.0 + i
        delta = cell["severity"] * (2 if key in geometry.SECONDARY_BENCHMARKS else 1)
        data[key] = {"L_c": base + delta, "token_weighted_L_c": base + delta,
                     "items": [{"n_tokens": 3}], "n_tokens": 3, "n_samples": 1}
    return data, {"tokenizer": "fixed", "pruning_threshold": None}


def test_sweep_json_has_paired_deltas_and_summary_is_read_only(tmp_path, monkeypatch, dataset_rows):
    monkeypatch.setattr(validity, "evaluate_cell",
                        lambda source, resolved, adapter, cell, probes, *a: _fake_measurement(cell, probes))
    out = tmp_path / "results"
    paths = validity.run_sweep(model_request="gemma3-270m", prune_density=[0.9, 0.8, 0.7, 0.6],
                              n_probe=8, output_base=out, device="cpu")
    assert len(paths) == 5
    before = {path: path.read_bytes() for path in paths}
    payload = json.loads(paths[-1].read_text())
    for pair in payload["capabilities"].values():
        assert pair["primary"]["delta_L_c"] == pytest.approx(0.4)
        assert pair["secondary"]["delta_L_c"] == pytest.approx(0.8)
    assert payload["protocol"]["loss_definition"] == validity.LOSS_DEFINITION
    report = validity.write_summary(out, tmp_path / "LOSS_VALIDITY.md")
    text = report.read_text()
    assert "Compression trajectories: 1" in text
    assert "| math | 4 | 1 | 1 | 1 | 1 (6) | 1 |" in text
    assert "| math | ok | 4 / 2 |" in text
    assert "dense_prune-d0.6" in text
    assert {path: path.read_bytes() for path in paths} == before


def test_summary_separates_protocols_and_reports_missing_data(tmp_path, monkeypatch, dataset_rows):
    report = validity.write_summary(tmp_path / "empty", tmp_path / "empty.md")
    assert "PENDING" in report.read_text()
    monkeypatch.setattr(validity, "evaluate_cell",
                        lambda source, resolved, adapter, cell, probes, *a: _fake_measurement(cell, probes))
    validity.run_sweep(model_request="gemma3-270m", prune_density=[0.9], n_probe=8,
                       output_base=tmp_path / "data", device="cpu")
    validity.run_sweep(model_request="gemma3-270m", prune_density=[0.8], n_probe=10,
                       output_base=tmp_path / "data", device="cpu")
    report = validity.write_summary(tmp_path / "data", tmp_path / "separate.md")
    assert "Compression trajectories: 2" in report.read_text()


def test_agreement_ignores_scales_and_handles_inversions_ties_and_small_panels():
    stats = validity.response_agreement([-1, 0, 1, 3], [-2, 0, 2, 6])
    for key in ("pearson", "spearman", "kendall_tau", "ordering_agreement", "sign_agreement"):
        assert stats[key] == pytest.approx(1)
    reversed_stats = validity.response_agreement([1, 2, 3], [-1, -2, -3])
    assert reversed_stats["spearman"] == -1
    assert reversed_stats["ordering_agreement"] == reversed_stats["sign_agreement"] == 0
    assert validity.response_agreement([1, 1, 1], [2, 2, 2])["pearson"] is None
    assert validity.response_agreement([1, 2], [2, 4])["spearman"] is None
    assert validity.response_agreement([], [])["sign_agreement"] is None


def test_law_transfer_holds_secondary_test_cells_out_and_censors_cliffs():
    severity = np.array([0.1, 0.2, 0.3, 0.4])
    primary = severity ** 1.5
    secondary = primary * 2
    fit = validity.law_transfer(severity, primary, secondary)
    assert fit["status"] == "ok"
    assert fit["heldout_relative_mae"] < 0.03
    perturbed = secondary.copy()
    perturbed[2:] *= -1
    changed = validity.law_transfer(severity, primary, perturbed)
    assert changed["gamma"] == fit["gamma"]
    assert changed["secondary_scale"] == fit["secondary_scale"]
    assert changed["heldout_predicted"] == fit["heldout_predicted"]
    assert changed["heldout_mae"] > fit["heldout_mae"]
    cliff = validity.law_transfer([*severity, 0.5, 0.6], [*primary, 1.1, 0.1],
                                  [*secondary, 0.8, 0.2])
    assert cliff["n_eligible"] == 4  # do not re-admit a stronger post-cliff cell
    assert validity.law_transfer(severity[:3], primary[:3], secondary[:3])["status"].startswith("insufficient")


@pytest.mark.parametrize("densities,bits", [([0], None), ([float("nan")], None),
                                          ([0.9, 0.9], None), (None, [1])])
def test_invalid_compression_ladder_rejected(densities, bits):
    with pytest.raises(ValueError):
        validity.compression_cells(densities, bits)
