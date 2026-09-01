# V6 report — Qwen/Qwen3-4B

params in prune scope: 4,022,272,000

| d | pred D_math | meas ΔL_math | pred D_code | meas ΔL_code | pred D_qa | meas ΔL_qa |
|---|---|---|---|---|---|---|
| 0.9 | 0.00052 | +0.005 | 0.00107 | +0.018 | 0.629 | -0.002 |
| 0.8 | 0.00262 | +0.005 | 0.00555 | -0.010 | 3.3 | -0.293 |
| 0.7 | 0.00754 | -0.003 | 0.017 | -0.088 | 8.47 | -0.793 |
| 0.6 | 0.0166 | +0.145 | 0.0387 | -0.001 | 16.6 | -1.760 |
| 0.55 | 0.0232 | +0.486 | 0.0549 | +0.254 | 22.4 | -1.958 |
| 0.5 | 0.0318 | +1.308 | 0.0756 | +0.904 | 29.8 | -1.476 |
| 0.45 | 0.0429 | +2.720 | 0.103 | +2.666 | 38.7 | -0.553 |
| 0.4 | 0.0552 | +4.998 | 0.133 | +5.827 | 49 | +1.369 |
| 0.35 | 0.0717 | +9.342 | 0.172 | +10.039 | 61.9 | +4.482 |
| 0.3 | 0.0919 | +12.343 | 0.224 | +11.948 | 77.6 | +6.841 |

pooled Spearman(pred, meas) = 0.010
  math: Spearman = 0.952, tr_F = 1.47e+03
  code: Spearman = 0.915, tr_F = 3.28e+03
  qa: Spearman = 0.479, tr_F = 1.31e+06

spectrum concentration: top-1% magnitude-coordinate mass share
  math: 16.4% of Fisher-weight mass in top 1% coordinates
  code: 14.0% of Fisher-weight mass in top 1% coordinates
  qa: 21.9% of Fisher-weight mass in top 1% coordinates