# V6 report — meta-models/Muse-Glimmer-30B

params in prune scope: 27,853,389,824

| d | pred D_math | meas ΔL_math | pred D_code | meas ΔL_code | pred D_qa | meas ΔL_qa |
|---|---|---|---|---|---|---|
| 0.9 | 0.000114 | +0.006 | 0.000269 | -0.006 | 0.0653 | +0.020 |
| 0.8 | 0.000702 | +0.033 | 0.00177 | +0.023 | 0.424 | +0.110 |
| 0.7 | 0.00201 | +0.173 | 0.00515 | +0.136 | 1.24 | +0.197 |
| 0.6 | 0.00426 | +0.802 | 0.0111 | +0.918 | 2.67 | +0.089 |
| 0.55 | 0.00591 | +1.895 | 0.0155 | +2.200 | 3.7 | +0.485 |
| 0.5 | 0.00778 | +3.534 | 0.0205 | +4.207 | 4.9 | +3.002 |
| 0.45 | 0.0101 | +5.390 | 0.0269 | +6.460 | 6.39 | +5.322 |
| 0.4 | 0.0129 | +7.864 | 0.0344 | +8.637 | 8.18 | +6.688 |
| 0.35 | 0.0163 | +12.007 | 0.0436 | +11.097 | 10.4 | +8.147 |
| 0.3 | 0.0205 | +13.346 | 0.0547 | +12.111 | 13 | +9.531 |

pooled Spearman(pred, meas) = 0.443
  math: Spearman = 1.000, tr_F = 2.49e+03
  code: Spearman = 1.000, tr_F = 6.38e+03
  qa: Spearman = 0.964, tr_F = 1.53e+06

spectrum concentration: top-1% magnitude-coordinate mass share
  math: 5.5% of Fisher-weight mass in top 1% coordinates
  code: 13.8% of Fisher-weight mass in top 1% coordinates
  qa: 10.1% of Fisher-weight mass in top 1% coordinates