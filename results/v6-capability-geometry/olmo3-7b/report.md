# V6 report — allenai/Olmo-3-1025-7B

params in prune scope: 7,297,482,752

| d | pred D_math | meas ΔL_math | pred D_code | meas ΔL_code | pred D_qa | meas ΔL_qa |
|---|---|---|---|---|---|---|
| 0.9 | 1.13e-05 | -0.001 | 2.26e-05 | -0.001 | 0.00496 | +0.007 |
| 0.8 | 0.000145 | +0.007 | 0.000348 | +0.015 | 0.0618 | +0.120 |
| 0.7 | 0.000525 | +0.027 | 0.00127 | +0.026 | 0.219 | +0.139 |
| 0.6 | 0.00127 | +0.075 | 0.00303 | +0.055 | 0.529 | +0.073 |
| 0.55 | 0.00184 | +0.122 | 0.00456 | +0.087 | 0.763 | -0.021 |
| 0.5 | 0.00255 | +0.217 | 0.00625 | +0.167 | 1.06 | -0.011 |
| 0.45 | 0.00351 | +0.381 | 0.00892 | +0.307 | 1.46 | +0.004 |
| 0.4 | 0.00469 | +0.676 | 0.0114 | +0.678 | 1.95 | +0.181 |
| 0.35 | 0.00606 | +1.252 | 0.0147 | +1.529 | 2.53 | +0.650 |
| 0.3 | 0.00789 | +2.623 | 0.0195 | +3.074 | 3.31 | +1.875 |

pooled Spearman(pred, meas) = 0.329
  math: Spearman = 1.000, tr_F = 81.6
  code: Spearman = 1.000, tr_F = 202
  qa: Spearman = 0.455, tr_F = 3.52e+04

spectrum concentration: top-1% magnitude-coordinate mass share
  math: 10.4% of Fisher-weight mass in top 1% coordinates
  code: 2.4% of Fisher-weight mass in top 1% coordinates
  qa: 10.9% of Fisher-weight mass in top 1% coordinates