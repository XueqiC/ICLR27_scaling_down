# V6 report — Qwen/Qwen3-1.7B

params in prune scope: 1,720,451,072

| d | pred D_math | meas ΔL_math | pred D_code | meas ΔL_code | pred D_qa | meas ΔL_qa |
|---|---|---|---|---|---|---|
| 0.9 | 0.00196 | -0.011 | 0.00426 | -0.023 | 0.327 | -0.050 |
| 0.8 | 0.00885 | +0.033 | 0.018 | -0.006 | 1.52 | -0.181 |
| 0.7 | 0.023 | +0.241 | 0.0471 | +0.191 | 3.86 | -0.509 |
| 0.6 | 0.046 | +1.200 | 0.0947 | +0.932 | 7.81 | -0.793 |
| 0.55 | 0.0629 | +2.101 | 0.131 | +1.848 | 10.7 | -0.531 |
| 0.5 | 0.0903 | +3.598 | 0.189 | +3.843 | 15 | +1.491 |
| 0.45 | 0.12 | +5.678 | 0.251 | +6.201 | 19.1 | +3.692 |
| 0.4 | 0.16 | +10.445 | 0.338 | +9.581 | 25.2 | +7.669 |
| 0.35 | 0.192 | +12.465 | 0.408 | +12.483 | 31 | +9.175 |
| 0.3 | 0.235 | +13.132 | 0.506 | +12.977 | 38.7 | +9.076 |

pooled Spearman(pred, meas) = 0.239
  math: Spearman = 1.000, tr_F = 2.94e+03
  code: Spearman = 1.000, tr_F = 6.8e+03
  qa: Spearman = 0.758, tr_F = 5.09e+05

spectrum concentration: top-1% magnitude-coordinate mass share
  math: 12.7% of Fisher-weight mass in top 1% coordinates
  code: 12.9% of Fisher-weight mass in top 1% coordinates
  qa: 15.3% of Fisher-weight mass in top 1% coordinates