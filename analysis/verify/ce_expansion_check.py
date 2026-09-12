"""Numerical check of the CE logit-displacement identities used by the next round.

Run: python3 analysis/verify/ce_expansion_check.py
CPU only, numpy only, no model weights. Verifies, for cross-entropy on the reference token:

  dL = (E_p[r] - r_y) + 0.5 * Var_p(r) + O(||r||^3)
  B  = z_y - sum_v p_v z_v   equals   d/deps of L((1-eps) z) at eps = 0
  V  = 1 - sum_v p_v^2       equals   trace of the Hessian of L wrt z
"""
from __future__ import annotations
import numpy as np

VOCAB = 5000
DRAWS = 200


def ce(z: np.ndarray, y: int) -> float:
    z = z - z.max()
    return float(-(z[y] - np.log(np.exp(z).sum())))


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"{'displacement':>14s} {'exact dL':>12s} {'2nd order':>12s} {'1st only':>12s} {'rel err':>9s}")
    for scale in (1e-3, 0.03, 0.3, 1.5):
        exact, second, first = [], [], []
        for _ in range(DRAWS):
            z = rng.normal(0.0, 3.0, VOCAB)
            y = int(rng.integers(VOCAB))
            p = softmax(z)
            r = rng.normal(0.0, scale, VOCAB)
            e = ce(z + r, y) - ce(z, y)
            f = float(p @ r - r[y])
            s = f + 0.5 * float(p @ (r * r) - (p @ r) ** 2)
            exact.append(e); first.append(f); second.append(s)
        exact_a, second_a, first_a = map(np.array, (exact, second, first))
        rel = np.abs(second_a - exact_a).mean() / max(np.abs(exact_a).mean(), 1e-12)
        print(f"{scale:14.4g} {exact_a.mean():12.6f} {second_a.mean():12.6f} {first_a.mean():12.6f} {rel:9.4f}")
        if scale <= 0.03:
            assert rel < 1e-3, f"second-order expansion inaccurate at scale {scale}: {rel}"

    z = rng.normal(0.0, 3.0, VOCAB)
    y = int(rng.integers(VOCAB))
    p = softmax(z)
    b = float(z[y] - p @ z)
    v = float(1.0 - p @ p)
    h = 1e-5
    finite = (ce((1 - h) * z, y) - ce((1 + h) * z, y)) / (2 * h)
    hess_trace = float(np.trace(np.diag(p) - np.outer(p, p)))
    print(f"\nB analytic {b:.8f}  finite difference {finite:.8f}  relative {abs(b - finite) / abs(b):.2e}")
    print(f"V analytic {v:.8f}  trace(Hessian)    {hess_trace:.8f}")
    assert abs(b - finite) / abs(b) < 1e-6
    assert abs(v - hess_trace) < 1e-12
    print("\nall identities verified")


if __name__ == "__main__":
    main()
