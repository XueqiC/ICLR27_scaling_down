# V11 geometry-damage report — google/gemma-3-4b-pt

Even-indexed V6 probes estimate diagonal empirical Fishers; the disjoint odd-indexed halves measure completion loss. `Delta L_c` is relative to the dense capability loss.

## math

Dense measurement loss: 0.713892

| d | Delta L_c | residual cosine | log cosine | mass retention |
|---:|---:|---:|---:|---:|
| 0.90 | +0.014309 | 0.9951 | 0.9999 | 0.1835 |
| 0.80 | +0.141471 | 0.9793 | 0.9996 | 0.1685 |
| 0.70 | +0.855629 | 0.9501 | 0.9967 | 0.0995 |
| 0.60 | +2.178159 | 0.9090 | 0.9920 | 0.0598 |
| 0.55 | +4.352683 | 0.6241 | 0.9772 | 0.0205 |
| 0.50 | +6.585507 | 0.5971 | 0.9530 | 0.0072 |
| 0.45 | +11.799649 | 0.1911 | 0.9186 | 0.0059 |
| 0.40 | +19.330241 | -0.0619 | 0.8997 | 0.0044 |
| 0.35 | +27.607071 | 0.4138 | 0.9155 | 0.0037 |
| 0.30 | +20.299147 | 0.1962 | 0.9107 | 0.0033 |

## code

Dense measurement loss: 0.842967

| d | Delta L_c | residual cosine | log cosine | mass retention |
|---:|---:|---:|---:|---:|
| 0.90 | +0.012084 | 0.9844 | 0.9999 | 0.2348 |
| 0.80 | +0.147570 | 0.9277 | 0.9993 | 0.2128 |
| 0.70 | +1.790281 | -0.3379 | 0.9853 | 0.0877 |
| 0.60 | +3.342967 | -0.3994 | 0.9814 | 0.0546 |
| 0.55 | +5.062404 | 0.1883 | 0.9787 | 0.0372 |
| 0.50 | +11.078772 | -0.4344 | 0.9337 | 0.0173 |
| 0.45 | +13.890537 | 0.0120 | 0.9217 | 0.0079 |
| 0.40 | +10.402046 | 0.6663 | 0.9424 | 0.0075 |
| 0.35 | +29.379028 | 0.5648 | 0.9330 | 0.0060 |
| 0.30 | +16.150384 | 0.5494 | 0.9327 | 0.0045 |

## qa

Dense measurement loss: 4.671552

| d | Delta L_c | residual cosine | log cosine | mass retention |
|---:|---:|---:|---:|---:|
| 0.90 | +0.065083 | 0.9958 | 0.9995 | 0.1464 |
| 0.80 | +0.278151 | 0.9842 | 0.9984 | 0.1294 |
| 0.70 | +0.439631 | 0.9212 | 0.9963 | 0.0976 |
| 0.60 | +0.343427 | 0.7441 | 0.9950 | 0.0677 |
| 0.55 | +1.456547 | 0.5824 | 0.9930 | 0.0350 |
| 0.50 | +3.414450 | -0.1054 | 0.9868 | 0.0182 |
| 0.45 | +6.662642 | 0.1164 | 0.9700 | 0.0079 |
| 0.40 | +19.760266 | 0.6868 | 0.9444 | 0.0071 |
| 0.35 | +23.987539 | 0.7221 | 0.9444 | 0.0072 |
| 0.30 | +17.831547 | 0.6294 | 0.9471 | 0.0043 |

## Behavioral cliffs and geometric shape

The behavioral cliff is the first listed density, moving from dense toward sparse, where `Delta L_c > 1.0`. Candidate critical values are the three geometric metrics at that density.

A pre-cliff curve is labeled smooth when it has a clear decline, no rebound larger than 0.03, and no single density step accounts for more than 75% of its cumulative decline. The dense baseline for each geometric metric is 1.0.

| capability | behavioral cliff d | residual before cliff | log-Fisher before cliff | mass before cliff | candidate values at cliff (residual / log / mass) |
|---|---:|---|---|---|---|
| math | 0.60 | smooth decline | flat / no clear decline | step-like decline | 0.9090 / 0.9920 / 0.0598 |
| code | 0.70 | step-like decline | flat / no clear decline | step-like decline | -0.3379 / 0.9853 / 0.0877 |
| qa | 0.55 | smooth decline | flat / no clear decline | step-like decline | 0.5824 / 0.9930 / 0.0350 |
