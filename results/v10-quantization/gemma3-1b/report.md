# V10 quantization report — google/gemma-3-1b-pt

All language weight matrices use symmetric round-to-nearest fake quantization with one scale per output channel. Losses use the odd-indexed measurement half of each V6 probe set.

## Dense losses

| capability | dense loss |
|---|---:|
| math | 1.221056 |
| code | 1.060812 |
| qa | 5.567138 |

## Quantization loss damage

Damage is `Delta L_c(b) = L_c(b) - L_c(dense)`.

| bits | Delta L_math | Delta L_code | Delta L_qa |
|---:|---:|---:|---:|
| 8 | +0.001042 | -0.000803 | +0.014629 |
| 6 | +0.034664 | +0.021424 | +0.066858 |
| 4 | +0.947227 | +0.592338 | +0.884058 |
| 3 | +15.326754 | +14.124308 | +9.799272 |

## Exponential fit check

Least-squares fit of `ln(Delta L_c)` against bit-width, using only positive damage. The proposed `4^(-b)` law predicts slope `-ln(4) = -1.3863` and fitted base `4`.

| capability | positive points | q_c | slope | fitted base | R^2 |
|---|---:|---:|---:|---:|---:|
| math | 4 | 2676.6053 | -1.8630 | 6.4429 | 0.9912 |
| code | 3 | 4871.4031 | -2.0917 | 8.0989 | 0.9690 |
| qa | 4 | 229.8960 | -1.2629 | 3.5357 | 0.9524 |
