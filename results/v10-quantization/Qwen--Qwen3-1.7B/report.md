# V10 quantization report — Qwen/Qwen3-1.7B

All language weight matrices use symmetric round-to-nearest fake quantization with one scale per output channel. Losses use the odd-indexed measurement half of each V6 probe set.

## Dense losses

| capability | dense loss |
|---|---:|
| math | 1.008093 |
| code | 1.602210 |
| qa | 6.388722 |

## Quantization loss damage

Damage is `Delta L_c(b) = L_c(b) - L_c(dense)`.

| bits | Delta L_math | Delta L_code | Delta L_qa |
|---:|---:|---:|---:|
| 8 | -0.006114 | -0.009679 | -0.058464 |
| 6 | -0.005218 | -0.020081 | +0.033095 |
| 4 | +0.400211 | +0.161514 | -0.213878 |
| 3 | +9.530886 | +9.456154 | +8.101822 |

## Exponential fit check

Least-squares fit of `ln(Delta L_c)` against bit-width, using only positive damage. The proposed `4^(-b)` law predicts slope `-ln(4) = -1.3863` and fitted base `4`.

| capability | positive points | q_c | slope | fitted base | R^2 |
|---|---:|---:|---:|---:|---:|
| math | 2 | 128725.9794 | -3.1703 | 23.8147 | 1.0000 |
| code | 2 | 1897700.0679 | -4.0698 | 58.5470 | 1.0000 |
| qa | 2 | 1983.3655 | -1.8335 | 6.2557 | 1.0000 |
