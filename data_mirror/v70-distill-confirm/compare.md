# V70 distillation confirmation

Each student × capability: **6 pools × 3 dependent budgets = 18 points**. The same six pool samples (data seeds 31..36) are reused across students. Training seed is fixed at 0.

Paired difference is baseline absolute error minus selected absolute error, so positive values favor the selected form. Intervals use 5,000 pool-cluster bootstrap resamples, seed 0, keeping all three budgets together and sharing draws across students/capabilities. They describe resampling of these six observed pools; they do not estimate training-seed variability. Pools can overlap in source examples (Jaccard values are registered).

Predictions are the stored planned-T values. Actual token counts and overshoot are recorded in compare.json. The strongest baseline was fixed using development LOCO; no confirmation-based selection occurs.

| Student | Cap. | Selected | Frozen baseline | MAE | Baseline MAE | Paired difference [95% CI] |
|---|---|---|---|---:|---:|---:|
| gemma3-270m | math | E | T-only | 0.074254 | 0.065444 | -0.008810 [-0.009782, -0.008105] |
| gemma3-270m | code | E | E-only | 0.019252 | 0.023006 | 0.003754 [0.002534, 0.004966] |
| gemma3-270m | qa | joint | E-only | 0.514982 | 0.609664 | 0.094682 [0.077442, 0.109336] |
| gemma3-1b | math | E | surface:L0 | 0.056897 | 0.033487 | -0.023410 [-0.023537, -0.023278] |
| gemma3-1b | code | E | E-only | 0.048525 | 0.053212 | 0.004687 [0.004158, 0.004969] |
| gemma3-1b | qa | joint | surface:logN | 0.463357 | 0.450152 | -0.013205 [-0.019849, -0.009775] |
