"""Offline, CPU fixtures and independent derivative checks for V87."""
from types import SimpleNamespace

import torch

try:
    from .descriptor_bv import describe, score_sample
    from .tiny_test_model import PROBES, TinyModel, TinyTokenizer
    from .v6_capability_geometry import completion_loss
except ImportError:
    from descriptor_bv import describe, score_sample
    from tiny_test_model import PROBES, TinyModel, TinyTokenizer
    from v6_capability_geometry import completion_loss


def derivative_checks():
    model, tokenizer = TinyModel(), TinyTokenizer()
    sample = PROBES["math"][1]
    measured = score_sample(model, tokenizer, sample, "math", 0, "selftest")
    # Capture the final logits through the SAME frozen loss scorer. The numeric
    # oracle evaluates CE in float64 so subtraction error does not dominate.
    class Capture:
        def __call__(self, **kwargs):
            self.z = model(**kwargs).logits.detach().double()
            return SimpleNamespace(logits=self.z)

    capture = Capture()
    _, count = completion_loss(capture, tokenizer, sample["prompt"], sample["completion"], "cpu")
    z = capture.z

    def scored_ce(value):
        fixed = lambda **kwargs: SimpleNamespace(logits=value)
        loss, n = completion_loss(fixed, tokenizer, sample["prompt"], sample["completion"], "cpu")
        assert n == count
        return loss / n

    eps = 1e-3
    slope = float((scored_ce((1 - eps) * z) - scored_ce((1 + eps) * z)) / (2 * eps))
    hessian = torch.autograd.functional.hessian(scored_ce, z).reshape(z.numel(), z.numel())
    trace = float(hessian.diagonal().sum())
    baseline = scored_ce(z)
    trace_fd = 0.0
    for index in range(z.numel()):
        perturbation = torch.zeros_like(z)
        perturbation.reshape(-1)[index] = eps
        trace_fd += float((scored_ce(z + perturbation) + scored_ce(z - perturbation) - 2 * baseline) / eps**2)
    b_error = abs(measured["B"] - slope) / abs(slope)
    v_error = abs(measured["V"] - trace) / abs(trace)
    v_fd_error = abs(measured["V"] - trace_fd) / abs(trace_fd)
    assert b_error < 1e-4
    assert v_error < 2e-3 and v_fd_error < 2e-3
    return {"B": measured["B"], "dCE_d_shrinkage": slope,
            "negative_dCE_d_shrinkage": -slope, "B_relative_error": b_error,
            "B_tolerance": 1e-4, "V": measured["V"], "hessian_trace": trace,
            "V_relative_error": v_error, "hessian_trace_central_difference": trace_fd,
            "V_finite_difference_relative_error": v_fd_error, "V_tolerance": 2e-3,
            "finite_difference_step": eps, "oracle_dtype": "torch.float64",
            "sign_note": "Requested B formula equals +dCE/d shrinkage, not its negative."}


def selftest():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        payload = describe(TinyModel(), TinyTokenizer(), PROBES, "selftest",
                           model_id="random-tiny-transition-softcap", revision="seed-87")
        payload["selftest"] = derivative_checks()
        payload["selftest"]["status"] = "passed"
        return payload
    finally:
        torch.set_num_threads(before)
