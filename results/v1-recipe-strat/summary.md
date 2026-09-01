# V1: recipe-stratified anchor + transfer law (2026-08-25)

Data: locked grid-fill IRT (frozen difficulties 2026-07-28). Noise floor = prior per-domain seed SD.

## math (noise floor 0.193)

### r=1 anchor by recipe (law demands mean Δθ = 0)
| recipe | n | mean Δθ | SE | |mean|≤2·SE |
|---|---|---|---|---|
| M0 | 12 | -0.126 | 0.132 | PASS |
| M1 | 12 | -3.223 | 0.327 | FAIL |
| M2 | 12 | -0.136 | 0.075 | PASS |
| M3 | 9 | -0.437 | 0.145 | FAIL |
| pooled (prior) | 45 | -1.017 | 0.224 | FAIL |
| pooled minus M1 | 33 | -0.214 | 0.070 | FAIL |

### η per recipe + leave-one-size-out extrapolation
| recipe | n | η [90% CI] | R² | LOO-RMSE | ×floor |
|---|---|---|---|---|---|
| M0 | 90 | +0.046 [+0.019,+0.076] | 0.038 | 0.367 | 1.9× |
| M1 | 90 | -1.030 [-1.194,-0.886] | 0.221 | 2.881 | 14.9× |
| M2 | 72 | +0.064 [+0.038,+0.093] | 0.080 | 0.342 | 1.8× |
| M3 | 54 | +0.030 [+0.005,+0.060] | 0.019 | 0.353 | 1.8× |
| pooled | 306 | -0.258 [-0.351,-0.176] | 0.047 | 1.739 | 9.0× |
| pooled-noM1 | 216 | +0.048 [+0.032,+0.064] | 0.044 | 0.355 | 1.8× |

## code (noise floor 0.137)

### r=1 anchor by recipe (law demands mean Δθ = 0)
| recipe | n | mean Δθ | SE | |mean|≤2·SE |
|---|---|---|---|---|
| M0 | 12 | +0.129 | 0.096 | PASS |
| M1 | 12 | +0.076 | 0.111 | PASS |
| M2 | 12 | +0.169 | 0.080 | FAIL |
| M3 | 9 | -0.047 | 0.071 | PASS |
| pooled (prior) | 45 | +0.090 | 0.047 | PASS |
| pooled minus M1 | 33 | +0.096 | 0.051 | PASS |

### η per recipe + leave-one-size-out extrapolation
| recipe | n | η [90% CI] | R² | LOO-RMSE | ×floor |
|---|---|---|---|---|---|
| M0 | 90 | -0.050 [-0.088,-0.018] | 0.060 | 0.471 | 3.4× |
| M1 | 90 | -0.059 [-0.100,-0.022] | 0.065 | 0.522 | 3.8× |
| M2 | 72 | -0.050 [-0.088,-0.017] | 0.069 | 0.398 | 2.9× |
| M3 | 54 | -0.072 [-0.115,-0.034] | 0.154 | 0.301 | 2.2× |
| pooled | 306 | -0.055 [-0.075,-0.036] | 0.071 | 0.441 | 3.2× |
| pooled-noM1 | 216 | -0.054 [-0.075,-0.034] | 0.075 | 0.406 | 3.0× |

## qa (noise floor 0.117)

### r=1 anchor by recipe (law demands mean Δθ = 0)
| recipe | n | mean Δθ | SE | |mean|≤2·SE |
|---|---|---|---|---|
| M0 | 12 | +0.076 | 0.087 | PASS |
| M1 | 12 | -1.051 | 0.088 | FAIL |
| M2 | 12 | +0.028 | 0.070 | PASS |
| M3 | 9 | +0.017 | 0.129 | PASS |
| pooled (prior) | 45 | -0.249 | 0.085 | FAIL |
| pooled minus M1 | 33 | +0.043 | 0.052 | PASS |

### η per recipe + leave-one-size-out extrapolation
| recipe | n | η [90% CI] | R² | LOO-RMSE | ×floor |
|---|---|---|---|---|---|
| M0 | 90 | +0.094 [+0.060,+0.120] | 0.193 | 0.563 | 4.8× |
| M1 | 90 | +0.186 [+0.139,+0.222] | 0.197 | 1.008 | 8.6× |
| M2 | 72 | +0.430 [+0.404,+0.459] | 0.913 | 0.289 | 2.5× |
| M3 | 54 | +0.091 [+0.052,+0.123] | 0.242 | 0.380 | 3.2× |
| pooled | 306 | +0.192 [+0.167,+0.216] | 0.310 | 0.712 | 6.1× |
| pooled-noM1 | 216 | +0.195 [+0.166,+0.227] | 0.419 | 0.553 | 4.7× |
