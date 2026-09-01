# V10 quantization report — allenai/Olmo-3-1025-7B

All language weight matrices use symmetric round-to-nearest fake quantization with one scale per output channel. Losses use the odd-indexed measurement half of each V6 probe set.

## Dense losses

| capability | dense loss |
|---|---:|
| math | 0.635576 |
| code | 1.129559 |
| qa | 5.030065 |

## Quantization loss damage

Damage is `Delta L_c(b) = L_c(b) - L_c(dense)`.

| bits | Delta L_math | Delta L_code | Delta L_qa |
|---:|---:|---:|---:|
| 8 | -0.000183 | +0.000181 | -0.005248 |
| 6 | -0.000574 | +0.002822 | -0.004372 |
| 4 | +0.158706 | +0.208973 | -0.071095 |
| 3 | +4.696021 | +4.633104 | +2.758103 |

## Exponential fit check

Least-squares fit of `ln(Delta L_c)` against bit-width, using only positive damage. The proposed `4^(-b)` law predicts slope `-ln(4) = -1.3863` and fitted base `4`.

| capability | positive points | q_c | slope | fitted base | R^2 |
|---|---:|---:|---:|---:|---:|
| math | 2 | 121658.2349 | -3.3874 | 29.5895 | 1.0000 |
| code | 4 | 995.6227 | -2.0064 | 7.4362 | 0.9763 |
| qa | 1 | n/a | n/a | n/a | n/a |
