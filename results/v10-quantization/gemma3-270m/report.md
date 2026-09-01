# V10 quantization report — google/gemma-3-270m

All language weight matrices use symmetric round-to-nearest fake quantization with one scale per output channel. Losses use the odd-indexed measurement half of each V6 probe set.

## Dense losses

| capability | dense loss |
|---|---:|
| math | 1.315995 |
| code | 1.190296 |
| qa | 5.275741 |

## Quantization loss damage

Damage is `Delta L_c(b) = L_c(b) - L_c(dense)`.

| bits | Delta L_math | Delta L_code | Delta L_qa |
|---:|---:|---:|---:|
| 8 | +0.001762 | +0.008415 | +0.004544 |
| 6 | +0.086187 | +0.045837 | +0.082202 |
| 4 | +1.993067 | +2.354324 | +0.876650 |
| 3 | +26.145136 | +27.939908 | +25.926948 |

## Exponential fit check

Least-squares fit of `ln(Delta L_c)` against bit-width, using only positive damage. The proposed `4^(-b)` law predicts slope `-ln(4) = -1.3863` and fitted base `4`.

| capability | positive points | q_c | slope | fitted base | R^2 |
|---|---:|---:|---:|---:|---:|
| math | 4 | 5282.7727 | -1.8632 | 6.4446 | 0.9944 |
| code | 4 | 2056.5300 | -1.6281 | 5.0940 | 0.9588 |
| qa | 4 | 1484.4516 | -1.6181 | 5.0435 | 0.9591 |
