# V6 report — google/gemma-3-1b-pt

params in prune scope: 999,751,680

| d | pred D_math | meas ΔL_math | pred D_code | meas ΔL_code | pred D_qa | meas ΔL_qa |
|---|---|---|---|---|---|---|
| 0.9 | 0.000424 | +0.044 | 0.000747 | +0.010 | 0.197 | +0.195 |
| 0.8 | 0.00238 | +0.346 | 0.00438 | +0.219 | 1.08 | +0.679 |
| 0.7 | 0.00643 | +1.307 | 0.012 | +1.227 | 2.85 | +0.827 |
| 0.6 | 0.013 | +4.140 | 0.0245 | +5.520 | 5.55 | +2.647 |
| 0.55 | 0.0172 | +7.856 | 0.0324 | +7.901 | 7.42 | +5.735 |
| 0.5 | 0.0221 | +13.784 | 0.0417 | +11.897 | 9.6 | +9.339 |
| 0.45 | 0.0275 | +14.967 | 0.0518 | +15.169 | 11.6 | +11.028 |
| 0.4 | 0.0337 | +12.071 | 0.0637 | +13.615 | 14 | +9.070 |
| 0.35 | 0.0402 | +12.109 | 0.0755 | +14.116 | 16.4 | +10.161 |
| 0.3 | 0.0477 | +13.573 | 0.0892 | +13.349 | 19.2 | +12.957 |

pooled Spearman(pred, meas) = 0.346
  math: Spearman = 0.818, tr_F = 847
  code: Spearman = 0.891, tr_F = 1.55e+03
  qa: Spearman = 0.939, tr_F = 3.75e+05

spectrum concentration: top-1% magnitude-coordinate mass share
  math: 26.7% of Fisher-weight mass in top 1% coordinates
  code: 18.8% of Fisher-weight mass in top 1% coordinates
  qa: 20.0% of Fisher-weight mass in top 1% coordinates