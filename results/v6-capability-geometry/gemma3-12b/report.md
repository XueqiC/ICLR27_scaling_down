# V6 report — google/gemma-3-12b-pt

params in prune scope: 11,765,268,480

| d | pred D_math | meas ΔL_math | pred D_code | meas ΔL_code | pred D_qa | meas ΔL_qa |
|---|---|---|---|---|---|---|
| 0.9 | 8.06e-05 | +0.016 | 0.000221 | +0.016 | 0.13 | +0.075 |
| 0.8 | 0.000462 | +0.113 | 0.00128 | +0.096 | 0.703 | +0.389 |
| 0.7 | 0.00129 | +0.684 | 0.0036 | +0.460 | 1.91 | +0.539 |
| 0.6 | 0.00268 | +4.165 | 0.00754 | +1.952 | 4.08 | +7.381 |
| 0.55 | 0.00361 | +10.934 | 0.0102 | +8.900 | 5.47 | +8.303 |
| 0.5 | 0.00474 | +17.103 | 0.0134 | +19.275 | 7.12 | +15.684 |
| 0.45 | 0.0061 | +20.281 | 0.0173 | +22.167 | 9.14 | +9.927 |
| 0.4 | 0.00758 | +19.698 | 0.0217 | +14.657 | 11.3 | +17.411 |
| 0.35 | 0.00939 | +22.578 | 0.0268 | +17.788 | 13.8 | +25.009 |
| 0.3 | 0.0113 | +17.359 | 0.0321 | +16.123 | 16.5 | +14.832 |

pooled Spearman(pred, meas) = 0.393
  math: Spearman = 0.915, tr_F = 568
  code: Spearman = 0.806, tr_F = 1.57e+03
  qa: Spearman = 0.903, tr_F = 8.46e+05

spectrum concentration: top-1% magnitude-coordinate mass share
  math: 36.2% of Fisher-weight mass in top 1% coordinates
  code: 43.1% of Fisher-weight mass in top 1% coordinates
  qa: 21.8% of Fisher-weight mass in top 1% coordinates