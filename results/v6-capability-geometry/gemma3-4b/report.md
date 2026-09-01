# V6 report — google/gemma-3-4b-pt

params in prune scope: 3,879,895,040

| d | pred D_math | meas ΔL_math | pred D_code | meas ΔL_code | pred D_qa | meas ΔL_qa |
|---|---|---|---|---|---|---|
| 0.9 | 0.000106 | +0.020 | 0.000276 | +0.002 | 0.0974 | +0.053 |
| 0.8 | 0.000604 | +0.153 | 0.0016 | +0.114 | 0.522 | +0.222 |
| 0.7 | 0.00169 | +0.831 | 0.00449 | +1.474 | 1.34 | +0.241 |
| 0.6 | 0.00349 | +2.214 | 0.00934 | +3.159 | 2.64 | +0.065 |
| 0.55 | 0.00476 | +4.257 | 0.0127 | +5.037 | 3.54 | +1.189 |
| 0.5 | 0.00628 | +7.002 | 0.0169 | +11.063 | 4.64 | +2.837 |
| 0.45 | 0.008 | +12.471 | 0.0214 | +14.704 | 5.9 | +6.497 |
| 0.4 | 0.01 | +19.774 | 0.0267 | +10.497 | 7.27 | +18.923 |
| 0.35 | 0.0124 | +27.760 | 0.0329 | +28.967 | 8.86 | +23.736 |
| 0.3 | 0.0151 | +20.611 | 0.04 | +16.172 | 10.8 | +16.631 |

pooled Spearman(pred, meas) = 0.371
  math: Spearman = 0.988, tr_F = 490
  code: Spearman = 0.952, tr_F = 1.28e+03
  qa: Spearman = 0.927, tr_F = 3.84e+05

spectrum concentration: top-1% magnitude-coordinate mass share
  math: 27.9% of Fisher-weight mass in top 1% coordinates
  code: 27.8% of Fisher-weight mass in top 1% coordinates
  qa: 25.0% of Fisher-weight mass in top 1% coordinates