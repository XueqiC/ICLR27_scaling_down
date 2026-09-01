# V6 report — Qwen/Qwen3-0.6B

params in prune scope: 595,984,384

| d | pred D_math | meas ΔL_math | pred D_code | meas ΔL_code | pred D_qa | meas ΔL_qa |
|---|---|---|---|---|---|---|
| 0.9 | 0.000707 | -0.004 | 0.00246 | +0.002 | 0.181 | -0.102 |
| 0.8 | 0.00314 | +0.085 | 0.0104 | +0.104 | 0.804 | -0.173 |
| 0.7 | 0.00781 | +0.502 | 0.0251 | +0.550 | 1.99 | -0.248 |
| 0.6 | 0.0153 | +1.743 | 0.049 | +2.124 | 3.81 | -0.016 |
| 0.55 | 0.02 | +2.744 | 0.0659 | +3.450 | 5 | +0.864 |
| 0.5 | 0.0263 | +4.678 | 0.0867 | +5.389 | 6.56 | +2.768 |
| 0.45 | 0.0336 | +8.118 | 0.113 | +8.268 | 8.44 | +6.196 |
| 0.4 | 0.0424 | +10.780 | 0.141 | +10.163 | 10.5 | +7.199 |
| 0.35 | 0.0522 | +12.421 | 0.174 | +13.008 | 13 | +9.864 |
| 0.3 | 0.0644 | +15.217 | 0.218 | +13.901 | 16.2 | +10.320 |

pooled Spearman(pred, meas) = 0.261
  math: Spearman = 1.000, tr_F = 1.09e+03
  code: Spearman = 1.000, tr_F = 3.8e+03
  qa: Spearman = 0.952, tr_F = 2.79e+05

spectrum concentration: top-1% prune-order mass share
  math: 58.4% of Fisher-weight mass in top 1% coordinates
  code: 65.4% of Fisher-weight mass in top 1% coordinates
  qa: 63.1% of Fisher-weight mass in top 1% coordinates