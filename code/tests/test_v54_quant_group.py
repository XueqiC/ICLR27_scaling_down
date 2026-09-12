import json
from pathlib import Path

import pytest
import torch

from analysis import v54_quant_group as quant
from analysis.tiny_test_model import TinyModel, TinyTokenizer, PROBES
from analysis.eval_records import KEY, read_eval


def independent(weight, bits, size, mode):
    """Straightforward unpadded row/group loop, independent of production."""
    output = weight.clone()
    steps, zeros, clipped, rms, counts, errors = [], [], [], [], [], []
    for row in range(weight.shape[0]):
        for start in range(0, weight.shape[1], size or weight.shape[1]):
            x = weight[row, start:start + (size or weight.shape[1])]
            if mode == "symmetric":
                lo, hi = -(2**(bits-1)-1), 2**(bits-1)-1
                step = x.abs().max() / hi
                divisor = step if step > 0 else torch.ones_like(step)
                zero = torch.zeros_like(step)
                code = torch.round(x / divisor)
                q = code.clamp(lo, hi) * step
            else:
                x32 = x.float()
                lo, hi = 0, 2**bits - 1
                minimum = min(float(x32.min()), 0.0)
                maximum = max(float(x32.max()), 0.0)
                # Model FP32 subtraction before division exactly, independently.
                step = (torch.tensor(maximum) - torch.tensor(minimum)) / hi
                divisor = step if step > 0 else torch.ones_like(step)
                zero = torch.round(torch.tensor(-minimum) / divisor).clamp(lo, hi)
                code = torch.round(x32 / divisor) + zero
                q = ((code.clamp(lo, hi) - zero) * step).to(x.dtype)
            output[row, start:start + len(x)] = q
            steps.append(float(step)); zeros.append(float(zero))
            num = int(((code < lo) | (code > hi)).sum())
            clipped.append(num / len(x)); counts.append(len(x))
            squared = float((q.double() - x.double()).square().sum())
            rms.append((squared / len(x)) ** 0.5); errors.append(squared)
    return output, steps, zeros, clipped, rms, counts, errors


def test_default_reproduces_pre_edit_realized_tensors_exactly():
    fixtures = torch.load(Path(__file__).parent / "fixtures/v54_symmetric_before.pt", weights_only=True)
    for fixture in fixtures:
        weight, bits, size = (fixture[key] for key in ("weight", "bits", "group_size"))
        before = weight.clone()
        result = quant.fake_quantize_grouped(weight, bits, size)
        assert torch.equal(result, fixture["expected"])
        assert torch.equal(quant.fake_quantize_grouped(weight, bits, size, "symmetric"), result)
        assert torch.equal(weight, before)


@pytest.mark.parametrize("mode", ["symmetric", "asymmetric"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bits", [2, 3, 8])
@pytest.mark.parametrize("size", [0, 3, 9])
def test_modes_and_group_statistics_against_independent_reference(mode, dtype, bits, size):
    weight = torch.tensor([[0, 0, 0, 0, 0], [-2.1, -.01, .31, 1.2, 3.1],
                           [1, 2, 3, 4, 5], [-5, -4, -3, -2, -1], [2, 2, 2, 2, 2]], dtype=dtype)
    expected, steps, zeros, clipped, rms, counts, errors = independent(weight, bits, size, mode)
    actual = quant.fake_quantize_grouped(weight, bits, size, mode)
    assert torch.equal(actual, expected)
    stats = quant.quantization_statistics(weight, actual, bits, size, mode)
    for name, values in (("step", steps), ("zero_point", zeros),
                         ("fraction_clipped_per_group", clipped), ("rms_rounding_error_per_group", rms)):
        x = torch.tensor(values, dtype=torch.float64)
        expected_stats = {"min": float(x.min()), "max": float(x.max()),
                          "mean": float(x.mean()), "std": float(x.std(unbiased=False))}
        assert stats[name] == pytest.approx(expected_stats, abs=1e-8)
    assert stats["n_groups"] == len(counts) and stats["n_weights"] == weight.numel()
    assert stats["fraction_weights_clipped"] == pytest.approx(sum(x*n for x, n in zip(clipped, counts)) / sum(counts))
    assert stats["rms_rounding_error"] == pytest.approx((sum(errors) / sum(counts))**.5)
    assert actual.dtype == dtype and actual.device.type == "cpu"


@pytest.mark.parametrize("mode", ["symmetric", "asymmetric"])
def test_run_writes_stats_records_restores_weights_and_resumes(tmp_path, monkeypatch, mode):
    torch.set_num_threads(1)
    model, tokenizer = TinyModel(), TinyTokenizer()
    original = model.weight.detach().clone()
    monkeypatch.setattr(quant, "require_compliant", lambda tag: "tiny")
    monkeypatch.setattr(quant, "resolve_model_and_revision", lambda tag: ("tiny", "87"))
    monkeypatch.setattr(quant, "load_text_causal_lm", lambda *a: (model, tokenizer))
    monkeypatch.setattr(quant, "build_probes", lambda *a, **kw: {cap: rows * 2 for cap, rows in PROBES.items()})
    monkeypatch.setattr(quant, "language_weight_parameters", lambda m: [("weight", m.weight)])
    kwargs = dict(model_tag="tiny@87", configs=[(3, 4)], device="cpu", reference_device="cpu",
                  model_dtype="fp32", n_probe=4, out=tmp_path, mode=mode)
    quant.run_quantization(**kwargs)
    path = tmp_path / "quant_group_losses.json"
    payload = read_eval(path)
    assert set(payload["dense"]) == set(PROBES) == set(payload["b3_g4"])
    assert set(payload[KEY]["evaluations"]) == {"dense", "b3_g4"}
    stats = payload["_meta"]["configs"]["b3_g4"]["quantization_statistics"]["weight"]
    assert stats["mode"] == mode and stats["step"]["max"] > 0
    assert torch.equal(model.weight, original)
    first = path.read_bytes()
    monkeypatch.setattr(quant, "load_text_causal_lm", lambda *a: pytest.fail("completed run must not load"))
    quant.run_quantization(**kwargs)
    assert path.read_bytes() == first
    with pytest.raises(ValueError, match="differs"):
        quant.run_quantization(**{**kwargs, "mode": "asymmetric" if mode == "symmetric" else "symmetric"})


def test_clipping_counts_actual_clamps_not_padding():
    weight = torch.tensor([[-.5, .5, 0.]])
    result = quant.fake_quantize_grouped(weight, 2, 2, "asymmetric")
    stats = quant.quantization_statistics(weight, result, 2, 2, "asymmetric")
    assert stats["fraction_weights_clipped"] == pytest.approx(1/3)
    assert stats["fraction_clipped_per_group"]["mean"] == pytest.approx(.25)


def test_invalid_mode_and_asymmetric_zero_groups():
    weight = torch.zeros(2, 5)
    assert torch.equal(quant.fake_quantize_grouped(weight, 3, 3, "asymmetric"), weight)
    with pytest.raises(ValueError, match="mode"):
        quant.fake_quantize_grouped(weight, 3, 3, "other")
