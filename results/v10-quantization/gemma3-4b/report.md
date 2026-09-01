# V10 quantization report — google/gemma-3-4b-pt

All language weight matrices use symmetric round-to-nearest fake quantization with one scale per output channel. Losses use the odd-indexed measurement half of each V6 probe set.

## Dense losses

| capability | dense loss |
|---|---:|
| math | 0.740472 |
| code | 0.802259 |
| qa | 5.474228 |

## Quantization loss damage

Damage is `Delta L_c(b) = L_c(b) - L_c(dense)`.

| bits | Delta L_math | Delta L_code | Delta L_qa |
|---:|---:|---:|---:|
| 8 | +0.002207 | +0.000775 | -0.010707 |
| 6 | +0.020818 | +0.015556 | -0.020605 |
| 4 | +0.723178 | +0.500803 | -0.265314 |
| 3 | +21.734145 | +19.235385 | +15.303660 |

## Exponential fit check

Least-squares fit of `ln(Delta L_c)` against bit-width, using only positive damage. The proposed `4^(-b)` law predicts slope `-ln(4) = -1.3863` and fitted base `4`.

| capability | positive points | q_c | slope | fitted base | R^2 |
|---|---:|---:|---:|---:|---:|
| math | 4 | 1869.5050 | -1.7794 | 5.9262 | 0.9550 |
| code | 4 | 2744.3777 | -1.9395 | 6.9554 | 0.9678 |
| qa | 1 | n/a | n/a | n/a | n/a |
