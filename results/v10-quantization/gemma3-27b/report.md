# V10 quantization report — google/gemma-3-27b-pt

All language weight matrices use symmetric round-to-nearest fake quantization with one scale per output channel. Losses use the odd-indexed measurement half of each V6 probe set.

## Dense losses

| capability | dense loss |
|---|---:|
| math | 0.601881 |
| code | 0.733946 |
| qa | 5.786432 |

## Quantization loss damage

Damage is `Delta L_c(b) = L_c(b) - L_c(dense)`.

| bits | Delta L_math | Delta L_code | Delta L_qa |
|---:|---:|---:|---:|
| 8 | +0.001184 | -0.001827 | -0.019843 |
| 6 | +0.009395 | +0.008913 | -0.041444 |
| 4 | +0.327512 | +0.226556 | -0.665105 |
| 3 | +18.455855 | +17.150465 | +16.100520 |

## Exponential fit check

Least-squares fit of `ln(Delta L_c)` against bit-width, using only positive damage. The proposed `4^(-b)` law predicts slope `-ln(4) = -1.3863` and fitted base `4`.

| capability | positive points | q_c | slope | fitted base | R^2 |
|---|---:|---:|---:|---:|---:|
| math | 4 | 1448.7885 | -1.8439 | 6.3210 | 0.9309 |
| code | 3 | 10335.1577 | -2.3918 | 10.9328 | 0.9272 |
| qa | 1 | n/a | n/a | n/a | n/a |
