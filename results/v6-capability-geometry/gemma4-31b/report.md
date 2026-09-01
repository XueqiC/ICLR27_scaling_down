# V6 report — google/gemma-4-31B

params in prune scope: 30,696,013,824

| d | pred D_math | meas ΔL_math | pred D_code | meas ΔL_code | pred D_qa | meas ΔL_qa |
|---|---|---|---|---|---|---|
| 0.9 | 0.000711 | +0.006 | 0.000641 | +0.006 | 10 | +0.219 |
| 0.8 | 0.00453 | +0.055 | 0.0039 | +0.086 | 66.8 | +0.635 |
| 0.7 | 0.0131 | +0.679 | 0.0108 | +0.768 | 193 | +2.889 |
| 0.6 | 0.0281 | +12.566 | 0.0226 | +9.903 | 416 | +15.284 |
| 0.55 | 0.0388 | +15.491 | 0.031 | +14.071 | 574 | +13.835 |
| 0.5 | 0.0513 | +16.058 | 0.041 | +16.440 | 750 | +14.709 |
| 0.45 | 0.0671 | +17.715 | 0.055 | +16.662 | 998 | +15.225 |
| 0.4 | 0.0852 | +18.217 | 0.0688 | +17.492 | 1.26e+03 | +15.798 |
| 0.35 | 0.106 | +20.000 | 0.0841 | +20.918 | 1.56e+03 | +15.808 |
| 0.3 | 0.13 | +20.922 | 0.102 | +19.373 | 1.9e+03 | +15.435 |

pooled Spearman(pred, meas) = 0.402
  math: Spearman = 1.000, tr_F = 6.42e+03
  code: Spearman = 0.988, tr_F = 5.37e+03
  qa: Spearman = 0.891, tr_F = 9.4e+07

spectrum concentration: top-1% magnitude-coordinate mass share
  math: 26.8% of Fisher-weight mass in top 1% coordinates
  code: 21.9% of Fisher-weight mass in top 1% coordinates
  qa: 27.9% of Fisher-weight mass in top 1% coordinates