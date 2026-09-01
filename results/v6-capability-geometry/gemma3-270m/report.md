# V6 report — google/gemma-3-270m

params in prune scope: 268,042,240

| d | pred D_math | meas ΔL_math | pred D_code | meas ΔL_code | pred D_qa | meas ΔL_qa |
|---|---|---|---|---|---|---|
| 0.9 | 0.00135 | +0.100 | 0.00293 | +0.069 | 0.469 | +0.206 |
| 0.8 | 0.00743 | +0.914 | 0.0164 | +0.961 | 2.54 | +0.925 |
| 0.7 | 0.0189 | +3.495 | 0.0422 | +4.044 | 6.25 | +2.543 |
| 0.6 | 0.036 | +6.557 | 0.0813 | +8.080 | 11.6 | +7.366 |
| 0.55 | 0.0472 | +7.189 | 0.107 | +8.501 | 14.9 | +7.208 |
| 0.5 | 0.0595 | +11.210 | 0.134 | +11.105 | 18.3 | +10.214 |
| 0.45 | 0.0743 | +16.809 | 0.167 | +18.871 | 22.7 | +11.257 |
| 0.4 | 0.0907 | +20.934 | 0.204 | +22.161 | 27.3 | +13.250 |
| 0.35 | 0.109 | +21.788 | 0.245 | +20.438 | 32.2 | +16.424 |
| 0.3 | 0.132 | +20.728 | 0.293 | +17.854 | 38.3 | +21.600 |

pooled Spearman(pred, meas) = 0.433
  math: Spearman = 0.964, tr_F = 1.02e+03
  code: Spearman = 0.915, tr_F = 2.25e+03
  qa: Spearman = 0.988, tr_F = 3.38e+05

spectrum concentration: top-1% magnitude-coordinate mass share
  math: 27.0% of Fisher-weight mass in top 1% coordinates
  code: 29.7% of Fisher-weight mass in top 1% coordinates
  qa: 25.8% of Fisher-weight mass in top 1% coordinates