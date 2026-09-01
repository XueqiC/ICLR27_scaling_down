# V11 geometry-damage report — google/gemma-3-1b-pt

Even-indexed V6 probes estimate diagonal empirical Fishers; the disjoint odd-indexed halves measure completion loss. `Delta L_c` is relative to the dense capability loss.

## math

Dense measurement loss: 1.192860

| d | Delta L_c | residual cosine | log cosine | mass retention |
|---:|---:|---:|---:|---:|
| 0.90 | +0.036547 | 0.9922 | 0.9999 | 0.1492 |
| 0.80 | +0.332215 | 0.9614 | 0.9992 | 0.1255 |
| 0.70 | +1.266017 | 0.9308 | 0.9976 | 0.1019 |
| 0.60 | +3.925370 | 0.8470 | 0.9898 | 0.0417 |
| 0.55 | +7.607040 | 0.4189 | 0.9748 | 0.0238 |
| 0.50 | +13.795606 | 0.1669 | 0.9492 | 0.0227 |
| 0.45 | +15.435024 | 0.1545 | 0.9410 | 0.0126 |
| 0.40 | +12.616067 | 0.3926 | 0.9404 | 0.0149 |
| 0.35 | +12.464613 | 0.6154 | 0.9477 | 0.0102 |
| 0.30 | +14.927501 | 0.5556 | 0.9182 | 0.0081 |

## code

Dense measurement loss: 1.205754

| d | Delta L_c | residual cosine | log cosine | mass retention |
|---:|---:|---:|---:|---:|
| 0.90 | +0.010486 | 0.9837 | 0.9999 | 0.1674 |
| 0.80 | +0.225448 | 0.9196 | 0.9991 | 0.1410 |
| 0.70 | +1.228261 | 0.7764 | 0.9964 | 0.0955 |
| 0.60 | +5.518031 | -0.1217 | 0.9822 | 0.0249 |
| 0.55 | +7.691944 | 0.4496 | 0.9796 | 0.0239 |
| 0.50 | +9.719054 | 0.5463 | 0.9635 | 0.0147 |
| 0.45 | +14.984527 | 0.2316 | 0.9487 | 0.0091 |
| 0.40 | +12.461765 | 0.2965 | 0.9456 | 0.0105 |
| 0.35 | +13.808056 | 0.0304 | 0.9332 | 0.0068 |
| 0.30 | +12.570205 | 0.5389 | 0.9225 | 0.0077 |

## qa

Dense measurement loss: 5.089941

| d | Delta L_c | residual cosine | log cosine | mass retention |
|---:|---:|---:|---:|---:|
| 0.90 | +0.195248 | 0.9949 | 0.9997 | 0.0731 |
| 0.80 | +0.709130 | 0.9749 | 0.9984 | 0.0723 |
| 0.70 | +0.941309 | 0.9451 | 0.9975 | 0.0688 |
| 0.60 | +2.559595 | 0.7043 | 0.9951 | 0.0433 |
| 0.55 | +5.596785 | 0.5812 | 0.9924 | 0.0230 |
| 0.50 | +8.724367 | 0.5796 | 0.9827 | 0.0114 |
| 0.45 | +11.573799 | 0.2593 | 0.9813 | 0.0075 |
| 0.40 | +9.304688 | 0.5444 | 0.9700 | 0.0072 |
| 0.35 | +11.114605 | 0.5624 | 0.9561 | 0.0037 |
| 0.30 | +12.724109 | 0.7440 | 0.9026 | 0.0052 |

## Behavioral cliffs and geometric shape

The behavioral cliff is the first listed density, moving from dense toward sparse, where `Delta L_c > 1.0`. Candidate critical values are the three geometric metrics at that density.

A pre-cliff curve is labeled smooth when it has a clear decline, no rebound larger than 0.03, and no single density step accounts for more than 75% of its cumulative decline. The dense baseline for each geometric metric is 1.0.

| capability | behavioral cliff d | residual before cliff | log-Fisher before cliff | mass before cliff | candidate values at cliff (residual / log / mass) |
|---|---:|---|---|---|---|
| math | 0.70 | step-like decline | flat / no clear decline | step-like decline | 0.9308 / 0.9976 / 0.1019 |
| code | 0.70 | step-like decline | flat / no clear decline | step-like decline | 0.7764 / 0.9964 / 0.0955 |
| qa | 0.60 | smooth decline | flat / no clear decline | step-like decline | 0.7043 / 0.9951 / 0.0433 |
