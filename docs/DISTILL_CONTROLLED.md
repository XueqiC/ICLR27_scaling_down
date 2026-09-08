# Distillation controlled-panel source-transfer test (P2)

Panel: pythia-{160m,410m,1.4b} x step{16000,64000,143000} = 9 cells, run `gpt-5.6-luna_full_600_lora`.
Endpoint delta_c = L_c(S_KD) - L_c(S0). Predictor: linear in {N0,L0,[D0]}, no config indicators.
Question: does D0 add held-out predictive value for delta beyond {N0,L0}, and does the model
beat a constant-delta baseline? (parallel to pruning/quant v36).

## math
- **leave-one-size-out**: noD0 MAE=0.033, +D0 MAE=0.016, baseline(mean_base) MAE=0.038 -> D0✓beats-noD0, ✓beats-baseline
- **leave-one-step-out**: noD0 MAE=0.026, +D0 MAE=0.014, baseline(med_base) MAE=0.031 -> D0✓beats-noD0, ✓beats-baseline

## code
- **leave-one-size-out**: noD0 MAE=0.124, +D0 MAE=0.131, baseline(med_base) MAE=0.080 -> D0✗beats-noD0, ✗beats-baseline
- **leave-one-step-out**: noD0 MAE=0.044, +D0 MAE=0.057, baseline(med_base) MAE=0.061 -> D0✗beats-noD0, ✓beats-baseline

## qa
- **leave-one-size-out**: noD0 MAE=0.294, +D0 MAE=0.367, baseline(med_base) MAE=0.188 -> D0✗beats-noD0, ✗beats-baseline
- **leave-one-step-out**: noD0 MAE=0.265, +D0 MAE=0.263, baseline(mean_base) MAE=0.159 -> D0✓beats-noD0, ✗beats-baseline


## Raw δ across the panel (why math is predictable and code/qa are not)

δ_c by step, per size (teacher gpt-5.6-luna, full/600/LoRA/2ep):

| size | δ_math (16k→64k→143k) | δ_code | δ_qa |
|------|-----------------------|--------|------|
| 160m | 0.023 → 0.033 → 0.038 | 0.128 → 0.098 → 0.078 | −0.437 → −0.495 → **+0.012** |
| 410m | 0.036 → 0.056 → 0.083 | 0.115 → 0.149 → 0.178 | −0.565 → −0.667 → −0.555 |
| 1.4b | 0.064 → 0.102 → 0.120 | 0.189 → 0.254 → 0.321 | −0.428 → −0.623 → −0.768 |

- **math**: δ grows smoothly & monotonically with training step in EVERY size → a clean D0 signal →
  the {N0,L0,D0} predictor beats a constant (MAE ~halved, both splits).
- **code**: direction is inconsistent across sizes (160m decreases, 410m/1.4b increase) → no clean
  cross-cell D0 law → a per-capability constant is as good or better; D0 does not help.
- **qa**: strongly negative (KD improves QA) but noisy, with a sign-flipped outlier (160m@143k = +0.012)
  → constant baseline wins; the structured predictor cannot generalize.

## Honest headline (P2)

> On the controlled LoRA panel, distillation admits a training-state (D0) source-transfer predictor
> ONLY for math (D0 beats both the no-D0 model and the constant baseline, though on a small absolute
> scale ~0.01–0.04 nat). For code and qa the response is best predicted by a per-capability constant;
> D0 adds nothing. Distillation therefore gives a WEAKER, capability-specific source-transfer
> relationship than pruning (which generalized prospectively across all three capabilities). n=9 cells,
> point MAE only (no CI); single distillation setting (fixed teacher/pool/E/recipe).
