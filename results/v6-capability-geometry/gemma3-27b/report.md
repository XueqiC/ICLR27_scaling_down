# V6 report — google/gemma-3-27b-pt

params in prune scope: 27,007,991,808

| d | pred D_math | meas ΔL_math | pred D_code | meas ΔL_code | pred D_qa | meas ΔL_qa |
|---|---|---|---|---|---|---|
| 0.9 | 6.91e-05 | +0.007 | 0.000168 | +0.015 | 0.162 | +0.146 |
| 0.8 | 0.000377 | +0.877 | 0.0009 | +0.792 | 0.838 | +0.006 |
| 0.7 | 0.00102 | +4.041 | 0.00243 | +4.691 | 2.25 | +3.589 |
| 0.6 | 0.00211 | +8.249 | 0.005 | +7.913 | 4.61 | +9.124 |
| 0.55 | 0.00287 | +11.589 | 0.00681 | +11.262 | 6.29 | +11.232 |
| 0.5 | 0.00375 | +15.213 | 0.00894 | +13.851 | 8.18 | +12.582 |
| 0.45 | 0.0048 | +16.030 | 0.0114 | +12.235 | 10.5 | +14.687 |
| 0.4 | 0.00608 | +17.058 | 0.0143 | +16.791 | 13.4 | +14.136 |
| 0.35 | 0.00744 | +24.957 | 0.0176 | +20.099 | 16.7 | +18.739 |
| 0.3 | 0.00902 | +18.223 | 0.0212 | +16.576 | 20.2 | +13.904 |

pooled Spearman(pred, meas) = 0.385
  math: Spearman = 0.988, tr_F = 673
  code: Spearman = 0.952, tr_F = 1.61e+03
  qa: Spearman = 0.903, tr_F = 1.51e+06

spectrum concentration: top-1% magnitude-coordinate mass share
  math: 33.9% of Fisher-weight mass in top 1% coordinates
  code: 49.4% of Fisher-weight mass in top 1% coordinates
  qa: 35.9% of Fisher-weight mass in top 1% coordinates