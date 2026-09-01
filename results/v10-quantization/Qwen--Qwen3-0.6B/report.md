# V10 quantization report — Qwen/Qwen3-0.6B

All language weight matrices use symmetric round-to-nearest fake quantization with one scale per output channel. Losses use the odd-indexed measurement half of each V6 probe set.

## Dense losses

| capability | dense loss |
|---|---:|
| math | 1.004238 |
| code | 1.591086 |
| qa | 5.610182 |

## Quantization loss damage

Damage is `Delta L_c(b) = L_c(b) - L_c(dense)`.

| bits | Delta L_math | Delta L_code | Delta L_qa |
|---:|---:|---:|---:|
| 8 | +0.001167 | -0.000578 | +0.000173 |
| 6 | +0.017697 | +0.023837 | -0.018220 |
| 4 | +0.559541 | +0.729125 | +0.213388 |
| 3 | +8.827976 | +10.825556 | +6.610298 |

## Exponential fit check

Least-squares fit of `ln(Delta L_c)` against bit-width, using only positive damage. The proposed `4^(-b)` law predicts slope `-ln(4) = -1.3863` and fitted base `4`.

| capability | positive points | q_c | slope | fitted base | R^2 |
|---|---:|---:|---:|---:|---:|
| math | 4 | 968.3796 | -1.7473 | 5.7390 | 0.9822 |
| code | 3 | 3219.9195 | -1.9925 | 7.3335 | 0.9852 |
| qa | 3 | 1488.7502 | -2.0157 | 7.5060 | 0.9820 |
