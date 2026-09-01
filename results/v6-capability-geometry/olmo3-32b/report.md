# V6 report — allenai/Olmo-3-1125-32B

params in prune scope: 32,232,468,480

| d | pred D_math | meas ΔL_math | pred D_code | meas ΔL_code | pred D_qa | meas ΔL_qa |
|---|---|---|---|---|---|---|
| 0.9 | 6.58e-06 | +0.000 | 1.83e-05 | +0.001 | 0.00296 | -0.007 |
| 0.8 | 6.93e-05 | +0.003 | 0.00022 | +0.011 | 0.0303 | +0.003 |
| 0.7 | 0.000256 | +0.015 | 0.000738 | +0.032 | 0.112 | +0.020 |
| 0.6 | 0.000647 | +0.042 | 0.00182 | +0.062 | 0.282 | +0.055 |
| 0.55 | 0.000911 | +0.060 | 0.00266 | +0.093 | 0.404 | -0.008 |
| 0.5 | 0.00129 | +0.091 | 0.00389 | +0.310 | 0.573 | +0.002 |
| 0.45 | 0.00175 | +0.147 | 0.00529 | +0.476 | 0.778 | -0.077 |
| 0.4 | 0.00234 | +0.238 | 0.00718 | +0.652 | 1.05 | -0.100 |
| 0.35 | 0.00307 | +0.369 | 0.00946 | +0.790 | 1.39 | -0.132 |
| 0.3 | 0.00393 | +0.627 | 0.0123 | +1.066 | 1.79 | -0.239 |

pooled Spearman(pred, meas) = -0.185
  math: Spearman = 1.000, tr_F = 43.2
  code: Spearman = 1.000, tr_F = 139
  qa: Spearman = -0.806, tr_F = 1.99e+04

spectrum concentration: top-1% magnitude-coordinate mass share
  math: 9.5% of Fisher-weight mass in top 1% coordinates
  code: 14.7% of Fisher-weight mass in top 1% coordinates
  qa: 18.5% of Fisher-weight mass in top 1% coordinates