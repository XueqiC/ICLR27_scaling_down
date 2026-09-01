# V11 geometry-damage report — Qwen/Qwen3-1.7B

Even-indexed V6 probes estimate diagonal empirical Fishers; the disjoint odd-indexed halves measure completion loss. `Delta L_c` is relative to the dense capability loss.

## math

Dense measurement loss: 0.979398

| d | Delta L_c | residual cosine | log cosine | mass retention |
|---:|---:|---:|---:|---:|
| 0.90 | -0.012477 | 0.9668 | 0.9993 | 0.2087 |
| 0.80 | +0.022318 | 0.9411 | 0.9989 | 0.2241 |
| 0.70 | +0.238662 | 0.9260 | 0.9986 | 0.2186 |
| 0.60 | +1.220477 | 0.8973 | 0.9977 | 0.2212 |
| 0.55 | +2.136634 | 0.8751 | 0.9968 | 0.2068 |
| 0.50 | +3.618606 | 0.8448 | 0.9942 | 0.1726 |
| 0.45 | +6.063646 | 0.7786 | 0.9849 | 0.1534 |
| 0.40 | +10.957221 | 0.4618 | 0.9535 | 0.1203 |
| 0.35 | +12.647551 | 0.4844 | 0.9350 | 0.1468 |
| 0.30 | +13.468512 | 0.5618 | 0.9168 | 0.1280 |

## code

Dense measurement loss: 1.780850

| d | Delta L_c | residual cosine | log cosine | mass retention |
|---:|---:|---:|---:|---:|
| 0.90 | -0.047202 | 0.9504 | 0.9996 | 0.3117 |
| 0.80 | +0.030007 | 0.8770 | 0.9991 | 0.2845 |
| 0.70 | +0.267701 | 0.8354 | 0.9987 | 0.2691 |
| 0.60 | +0.938638 | 0.6286 | 0.9976 | 0.2406 |
| 0.55 | +1.872893 | 0.5422 | 0.9968 | 0.2328 |
| 0.50 | +3.703978 | 0.5662 | 0.9931 | 0.1673 |
| 0.45 | +6.094066 | 0.3868 | 0.9817 | 0.1500 |
| 0.40 | +9.690492 | 0.3507 | 0.9557 | 0.1482 |
| 0.35 | +12.351989 | 0.0858 | 0.9285 | 0.1342 |
| 0.30 | +13.006743 | 0.1273 | 0.9010 | 0.0960 |

## qa

Dense measurement loss: 6.448608

| d | Delta L_c | residual cosine | log cosine | mass retention |
|---:|---:|---:|---:|---:|
| 0.90 | -0.080444 | 0.9885 | 0.9994 | 0.1490 |
| 0.80 | -0.340698 | 0.9724 | 0.9984 | 0.1312 |
| 0.70 | -0.683716 | 0.9604 | 0.9976 | 0.1470 |
| 0.60 | -0.708374 | 0.9309 | 0.9960 | 0.1461 |
| 0.55 | -0.507690 | 0.9071 | 0.9948 | 0.1349 |
| 0.50 | +1.559204 | 0.8941 | 0.9924 | 0.1309 |
| 0.45 | +3.591919 | 0.8086 | 0.9842 | 0.1127 |
| 0.40 | +8.231079 | 0.5619 | 0.9668 | 0.0864 |
| 0.35 | +9.159302 | 0.3541 | 0.9598 | 0.1322 |
| 0.30 | +8.494751 | 0.5097 | 0.9346 | 0.1278 |

## Behavioral cliffs and geometric shape

The behavioral cliff is the first listed density, moving from dense toward sparse, where `Delta L_c > 1.0`. Candidate critical values are the three geometric metrics at that density.

A pre-cliff curve is labeled smooth when it has a clear decline, no rebound larger than 0.03, and no single density step accounts for more than 75% of its cumulative decline. The dense baseline for each geometric metric is 1.0.

| capability | behavioral cliff d | residual before cliff | log-Fisher before cliff | mass before cliff | candidate values at cliff (residual / log / mass) |
|---|---:|---|---|---|---|
| math | 0.60 | smooth decline | flat / no clear decline | step-like decline | 0.8973 / 0.9977 / 0.2212 |
| code | 0.55 | smooth decline | flat / no clear decline | step-like decline | 0.5422 / 0.9968 / 0.2328 |
| qa | 0.50 | smooth decline | flat / no clear decline | step-like decline | 0.8941 / 0.9924 / 0.1309 |
