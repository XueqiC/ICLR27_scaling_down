"""A17 accuracy comparison: the pruning-law fit, the link and the scoring on synthetic panels (CPU only)."""
import math

from analysis import a17_accuracy_comparison as a17


def synthetic_panel(alpha=1.5, p0=1.0):
    panel = {}
    for i, m in enumerate(a17.MODELS):
        a0 = 0.5 + 0.1 * i
        rows = {"1.0": {c: {"accuracy": a0, "loss": 1.0 + 0.1 * i} for c in a17.CAPS}}
        for d in a17.DENSITIES:
            rows[str(d)] = {c: {"accuracy": a0 * p0 * d ** alpha, "loss": 1.0 + 0.1 * i + (1 - d)} for c in a17.CAPS}
        panel[m] = rows
    return panel


def test_alpha_is_recovered_on_an_exact_law_panel():
    panel = synthetic_panel(alpha=1.5)
    alpha = a17.fit_alpha(panel, list(a17.MODELS[:3]), "math")
    assert math.isclose(alpha, 1.5, abs_tol=1e-9)
    assert math.isclose(a17.law_predict(panel, a17.MODELS[3], "math", 0.7, alpha, False), 0.8 * 0.7 ** 1.5, abs_tol=1e-9)


def test_calibration_absorbs_a_constant_factor():
    panel = synthetic_panel(alpha=1.5, p0=0.9)
    alpha = a17.fit_alpha(panel, list(a17.MODELS[:3]), "code")
    assert math.isclose(alpha, 1.5, abs_tol=1e-9)
    held = a17.MODELS[3]
    assert not math.isclose(a17.law_predict(panel, held, "code", 0.6, alpha, False), panel[held]["0.6"]["code"]["accuracy"], abs_tol=1e-6)
    assert math.isclose(a17.law_predict(panel, held, "code", 0.6, alpha, True), panel[held]["0.6"]["code"]["accuracy"], abs_tol=1e-9)


def test_logistic_link_fits_a_logistic_relation():
    panel = synthetic_panel()
    for m in a17.MODELS:
        for k, row in panel[m].items():
            for c in a17.CAPS:
                row[c]["accuracy"] = 1 / (1 + math.exp(-(2.0 - 1.5 * row[c]["loss"])))
    link = a17.fit_link(panel, list(a17.MODELS[:3]), "qa")
    assert math.isclose(link[0], 2.0, abs_tol=1e-4) and math.isclose(link[1], -1.5, abs_tol=1e-4)
    assert math.isclose(a17.link_apply(link, 1.0), 1 / (1 + math.exp(-0.5)), abs_tol=1e-4)
