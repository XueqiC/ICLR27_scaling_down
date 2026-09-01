# V3: measurement-layer audit (8B-teacher campaign)

## (c) trained-half vs transfer-half split

### math: trained (gsm8k) vs transfer (math500)
| recipe | n | Δacc trained | Δacc transfer | trained−transfer [90% CI] | 读法 |
|---|---|---|---|---|---|
| M0 | 12 | +0.005 | +0.028 | -0.023 [-0.060,+0.013] | 无分裂 |
| M1 | 12 | -0.578 | -0.405 | -0.173 [-0.210,-0.140] | 迁移半区偏高 |
| M2 | 12 | -0.012 | +0.017 | -0.028 [-0.072,+0.013] | 无分裂 |
| M3 | 12 | -0.013 | +0.015 | -0.028 [-0.073,+0.017] | 无分裂 |

### qa: trained (hotpotqa) vs transfer (2wiki)
| recipe | n | Δacc trained | Δacc transfer | trained−transfer [90% CI] | 读法 |
|---|---|---|---|---|---|
| M0 | 12 | +0.055 | -0.052 | +0.107 [+0.073,+0.140] | 训练半区偏高 |
| M1 | 12 | +0.090 | -0.043 | +0.133 [+0.060,+0.202] | 训练半区偏高 |
| M2 | 12 | +0.198 | +0.127 | +0.072 [+0.028,+0.115] | 训练半区偏高 |
| M3 | 12 | -0.000 | -0.025 | +0.025 [-0.018,+0.067] | 无分裂 |

### code: trained (humaneval) vs transfer (mbpp)
| recipe | n | Δacc trained | Δacc transfer | trained−transfer [90% CI] | 读法 |
|---|---|---|---|---|---|
| M0 | 12 | -0.016 | -0.013 | -0.003 [-0.051,+0.041] | 无分裂 |
| M1 | 12 | -0.035 | -0.018 | -0.017 [-0.059,+0.023] | 无分裂 |
| M2 | 12 | +0.016 | +0.002 | +0.014 [-0.012,+0.042] | 无分裂 |
| M3 | 12 | -0.024 | -0.008 | -0.016 [-0.050,+0.019] | 无分裂 |

## (a) 1PL vs 2PL and (b) DIF

### math: 52 respondents × 100 items | 2PL a: median 1.62, IQR [0.97,2.12], SD(log a) 0.52 | ΔAIC(2PL−1PL) -348 | Spearman(θ1,θ2) 0.994 | DIF flags |γ|>1,z>2.58: 0/100 (0 trained-half)

### qa: 52 respondents × 100 items | 2PL a: median 1.29, IQR [0.87,1.58], SD(log a) 0.51 | ΔAIC(2PL−1PL) -196 | Spearman(θ1,θ2) 0.981 | DIF flags |γ|>1,z>2.58: 0/100 (0 trained-half)

### code: 52 respondents × 81 items | 2PL a: median 1.54, IQR [1.23,1.64], SD(log a) 0.60 | ΔAIC(2PL−1PL) -260 | Spearman(θ1,θ2) 0.987 | DIF flags |γ|>1,z>2.58: 0/81 (0 trained-half)
