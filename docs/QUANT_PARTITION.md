# Quantization on the round-6 panel: partition prediction on new source-states

Quantization has NO unseen-bit axis (integer bits 8/6/4/3 are all measured; nothing between 3 and 4),
so unlike pruning it cannot test an unseen compression strength. The panel test is therefore
source-transfer per bit on NEW source-states (pythia-{160m,410m,1.4b}@step96000, none in the fit grid),
reported split into the smooth region (>=4 bit) and the int3 collapse region, candidate = frozen
{N0,L0,D0} config-indicator predictor vs per-bit median baseline. |error| MAE:

| source | cap | >=4bit cand/base | int3 cand/base |
|--------|-----|------------------|----------------|
| 160m@96k | math | 0.060 / 0.195 ✓ | 2.721 / 2.142 ✗ |
| 160m@96k | code | 0.219 / 0.196 ✗ | 4.773 / 0.647 ✗ |
| 160m@96k | qa   | 0.059 / 0.337 ✓ | 1.752 / 2.761 ✓ |
| 410m@96k | math | 0.022 / 0.046 ✓ | 0.108 / 0.891 ✓ |
| 410m@96k | code | 0.030 / 0.067 ✓ | 0.492 / 0.384 ✗ |
| 410m@96k | qa   | 0.331 / 0.023 ✗ | 4.275 / 0.583 ✗ |
| 1.4b@96k | math | 0.005 / 0.011 ✓ | 1.634 / 2.769 ✓ |
| 1.4b@96k | code | 0.064 / 0.018 ✗ | 0.625 / 1.497 ✓ |
| 1.4b@96k | qa   | 0.128 / 0.041 ✗ | 0.112 / 1.838 ✓ |

## Read
- **>=4 bit (smooth region): almost nothing to predict.** Both candidate and baseline errors are tiny
  (~0.005–0.33 nat); "wins" are marginal and mixed. math is the most consistent small candidate edge;
  code/qa are noise-level either way. There is no meaningful smooth-region damage law to identify here.
- **int3 (collapse region): magnitude prediction is UNSTABLE.** The candidate beats the baseline on some
  source-states (410m/1.4b math, 1.4b code/qa) and fails badly on others (160m-code cand 4.77, 410m-qa 4.28).
  Source-conditioned amplitude helps where int3 damage tracks dense loss and misfires where it doesn't.
- **Verdict**: unlike pruning (C30, a clean unseen-strength law), quantization gets NO clean second-axis
  result — no unseen bit to predict, and the only large signal (int3) is an unstable collapse whose
  magnitude is not reliably predicted. This is what "predicting entry into the large-damage region" looks
  like: the RISK of int3 collapse is real and universal, but its MAGNITUDE is not a smooth predictable law.
  Confirms C28 on a third independent new source. RTN definition unchanged throughout.
