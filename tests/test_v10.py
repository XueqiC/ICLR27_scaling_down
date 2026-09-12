import numpy as np
import torch

from analysis import v10_quantization as quantization


def test_per_output_channel_fake_quant_has_known_symmetric_mapping():
    weight = torch.tensor(
        [[-2.0, -0.7, 0.7, 2.0], [0.0, 1.0, 2.0, 3.0]],
        dtype=torch.float32,
    )

    actual = quantization.fake_quantize_per_output_channel(weight, bits=3)
    expected = torch.tensor(
        [[-2.0, -2.0 / 3.0, 2.0 / 3.0, 2.0], [0.0, 1.0, 2.0, 3.0]],
        dtype=torch.float32,
    )

    torch.testing.assert_close(actual, expected)
    assert actual.dtype == weight.dtype
    assert actual.device == weight.device


def test_per_output_channel_fake_quant_is_close_and_idempotent_at_high_bits():
    weight = torch.tensor(
        [[-1.2345, -0.125, 0.3333, 1.1111], [0.001, -0.2, 0.9, 2.3456]],
        dtype=torch.float32,
    )

    quantized = quantization.fake_quantize_per_output_channel(weight, bits=16)
    requantized = quantization.fake_quantize_per_output_channel(
        quantized, bits=16
    )
    per_row_error_bound = (
        weight.abs().amax(dim=1, keepdim=True) / (2 * (2**15 - 1))
    )

    assert torch.all((quantized - weight).abs() <= per_row_error_bound + 1e-7)
    torch.testing.assert_close(requantized, quantized, rtol=0, atol=1e-7)


def test_per_output_channel_fake_quant_clamps_codes_and_handles_zero_rows():
    weight = torch.tensor(
        [[-100.0, -1.0, 0.0, 99.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    bits = 3

    quantized = quantization.fake_quantize_per_output_channel(weight, bits)
    scale = weight.abs().amax(dim=1, keepdim=True) / (2 ** (bits - 1) - 1)
    nonzero_codes = quantized[0] / scale[0]

    assert float(nonzero_codes.min()) >= -3
    assert float(nonzero_codes.max()) <= 3
    assert quantized[0, 0] == -100
    torch.testing.assert_close(quantized[1], torch.zeros(4))
    assert torch.isfinite(quantized).all()


def test_exponential_fit_matches_numpy_least_squares_reference():
    bits = np.array([8.0, 6.0, 4.0, 3.0])
    deltas = 2.75 * np.power(5.0, -bits) * np.array([1.02, 0.98, 1.01, 0.99])
    design = np.column_stack((np.ones_like(bits), bits))
    expected_intercept, expected_slope = np.linalg.lstsq(
        design, np.log(deltas), rcond=None
    )[0]
    fitted = design @ np.array([expected_intercept, expected_slope])
    expected_r2 = 1.0 - np.sum((np.log(deltas) - fitted) ** 2) / np.sum(
        (np.log(deltas) - np.log(deltas).mean()) ** 2
    )

    actual = quantization.fit_exponential_decay(bits, deltas)

    assert actual["n_points"] == 4
    np.testing.assert_allclose(actual["intercept"], expected_intercept)
    np.testing.assert_allclose(actual["slope"], expected_slope)
    np.testing.assert_allclose(actual["q"], np.exp(expected_intercept))
    np.testing.assert_allclose(actual["base"], np.exp(-expected_slope))
    np.testing.assert_allclose(actual["r2"], expected_r2)


def test_exponential_fit_uses_only_positive_finite_damage():
    actual = quantization.fit_exponential_decay(
        [8, 7, 6, 5, 4], [4.0**-8, -1.0, np.nan, 4.0**-5, 0.0]
    )

    assert actual["n_points"] == 2
    np.testing.assert_allclose(actual["base"], 4.0)
    np.testing.assert_allclose(actual["r2"], 1.0)
