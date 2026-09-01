# V6 report — Qwen/Qwen3-0.6B

params in prune scope: 595,984,384

| d | pred D_math | meas ΔL_math | pred D_code | meas ΔL_code | pred D_qa | meas ΔL_qa |
|---|---|---|---|---|---|---|
| 0.9 | 0.000735 | -0.004 | 0.00238 | +0.001 | 0.212 | -0.134 |
| 0.8 | 0.00352 | +0.085 | 0.011 | +0.100 | 1.01 | -0.137 |
| 0.7 | 0.00919 | +0.492 | 0.0279 | +0.563 | 2.63 | -0.258 |
| 0.6 | 0.019 | +1.714 | 0.0574 | +2.125 | 5.3 | -0.053 |
| 0.55 | 0.0256 | +2.712 | 0.0779 | +3.446 | 7.14 | +0.897 |
| 0.5 | 0.0341 | +4.718 | 0.105 | +5.393 | 9.52 | +2.628 |
| 0.45 | 0.0447 | +8.128 | 0.138 | +8.277 | 12.5 | +6.217 |
| 0.4 | 0.057 | +10.905 | 0.175 | +10.258 | 15.8 | +7.286 |
| 0.35 | 0.0715 | +12.410 | 0.22 | +13.055 | 19.8 | +9.979 |
| 0.3 | 0.0893 | +15.071 | 0.279 | +13.736 | 25.1 | +10.728 |

pooled Spearman(pred, meas) = 0.269
  math: Spearman = 1.000, tr_F = 1.03e+03
  code: Spearman = 1.000, tr_F = 3.4e+03
  qa: Spearman = 0.952, tr_F = 2.99e+05

spectrum concentration: top-1% magnitude-coordinate mass share
  math: 8.8% of Fisher-weight mass in top 1% coordinates
  code: 10.3% of Fisher-weight mass in top 1% coordinates
  qa: 11.2% of Fisher-weight mass in top 1% coordinates