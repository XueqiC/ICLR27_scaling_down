# FIGURE_SPEC — core figures for the rewritten manuscript (generated from numeric result files only)

All scripts live in `analysis/` (mirrored to `paper/analysis/`), run on CPU in seconds, write PDF+PNG to
`paper/figs/`, and print the exact data files and selection rules they used. Style: matplotlib, bold Nimbus Roman /
Times-metric fonts (see `analysis/plot_v9_blocks_appendix.py` for the font setup), font size >= 8 pt at final size,
one consistent colour per method/candidate across all figures, capability order math / code / QA, improvement
direction "positive = candidate better" everywhere. No smoothing that is not in the data; measured points are markers,
frozen predictions are lines/hollow markers; never draw a continuous line through integer bit-widths as if it were a
validated law (dashed thin guide lines only).

## Fig 1 `figs/predictive_responses.pdf` — three arms: measured capability response vs frozen prediction
Three columns (pruning | quantization | distillation), rows = capability (math, code, QA); y = ΔL in nats/token
(native units; one source per column so no cross-tokenizer axis).
- Pruning column: source = Pythia-1B@step112000 (in-range, late stage); x = density d (0.9 … 0.55). Measured ΔL from
  `results/v49-p1v2/compare_pythia-1b@step112000.json` → protocols.A.rows_prune (fields cap, d, actual, and per-candidate
  {pred, signed, abs} for power, A2, A1, cont, strength_only, median_curve, zero). Plot: measured (filled markers),
  frozen predictions of the headline candidate "power" and of A2 (lines), strongest simple baseline median_curve (dotted).
  Shade the extrapolated density 0.55 region. Also overlay, in a lighter colour, the 6.9B@112000 measured and A2/median
  predictions from `compare_pythia-6.9b@step112000.json` so the size-extrapolation failure is visible (legend: "6.9B").
- Quantization column: same two sources; x = integer bits {8,6,5,4,3} (categorical axis, measured markers only, thin
  dashed guide); from protocols.A.rows_quant (fields cap, bit, actual, rule, full, noD0, per_bit_mean, per_bit_median,
  zero). Show measured, "full" frozen prediction (hollow markers), per-bit median (dotted). Mark the 5-bit point as
  "rule (interp of 4/6)".
- Distillation column: student Gemma-3-1B; x = reuse count E = T/D_U (log scale); y = δ_c = L_c(S_KD) − L_c(S_0).
  Data: `results/v41-distill-newpool/summary.json` (by_cap[c].fits for constant/T/E/2D with fitted parameters;
  endpoint_E_range, u225_E_range) and the per-run records under `results/v12-distill/gemma3-1b/` (run dirs whose names
  contain `_75_`, `_225_`, `_600_` with `eval.json` / `train_log.json`; the dense student loss is in eval.json). If the
  summary lacks per-run points, compute δ_c per run from eval.json (student KD loss − dense loss) and E from train_log
  (tokens_seen / unique_data_pool_tokens). Plot endpoint runs (U75, U600) as filled markers, U225 test runs as hollow
  markers, the frozen constant and E-only predictions as lines over the E range, and annotate the QA bias at the two
  budgets. If any of these quantities cannot be recovered from files, leave that panel element out and print why.
- Caption text must be emitted by the script into `figs/predictive_responses_caption.txt` listing the exact files.

## Fig 2 `figs/input_form_gain.pdf` — paired improvements, three panels, zero line
Data: `results/v52-prediction-tables/groups.json` (per group: cand{cap}, same{cap}=(name, mae), simple{cap}=(name, mae),
improvement{cap}, origin, axis, method, test) and `results/v36b-input-comparison/summary.json` →
results[arm][cap].comparison.input_pairs (keys combined_vs_L0 = adding D0 given {N0,L0}; combined_vs_D0 = adding L0;
each with improvement and improvement_ci95) and .core_table[N0_D0_L0].improvement_ci95 (vs strongest simple baseline).
- Panel A "full vs no-D0": one point per (arm, capability) on the leave-one-step-out panel (v36b, two arms) and the
  distillation panel (`results/v39-distill-controlled/summary.json` by_cap[c].step: mae_noD0 − mae_D0, no CI) and the
  frozen prospectives where a no-D0 line exists (v46, v49 groups: same-input no-D0 minus candidate). Error bars only
  where a CI exists (v36b); otherwise plain points; label n (number of independent sources) next to each point.
- Panel B "source-conditioned vs source-free": candidate vs strongest simple baseline per group and capability
  (improvement field), grouped by test, colour by method, marker by prediction origin (P vs L).
- Panel C "shared power vs A2 / A1 (same input)": v42 by_cap mae (cand − A2, cand − A1) and v46 / v49 pruning groups
  (candidate power minus the same-input best), per capability.
- Show the sign convention on the axis label; do not pool different panels into one column.

## Fig 3 `figs/transfer_limits.pdf` — transfer range matrix
Data: `results/v49-p1v2/compare_pythia-*.json` (protocols A and B; summary.prune.interp_0.9-0.6 / extrap_0.55 per
candidate; summary.quant_ge4 and summary.quant.int3 per candidate) plus `results/v46-p1-newsource/compare.json` (1B@96k)
and `results/v41-distill-newpool/summary.json` (unseen pool).
Layout: facets = protocol A | protocol B; rows = source (1B@32k, 1B@96k [A only], 1B@112k, 6.9B@32k, 6.9B@112k);
columns = regime (prune interp, prune extrap d=0.55, quant >=4-bit, quant int3). Cell value = improvement of the
headline candidate (power for pruning, full for quantization) over the strongest source-free baseline, MAE difference in
nats, printed in the cell together with the candidate MAE in small type; colour diverging around zero. Cells not
evaluated (e.g. 1B@96k under protocol B) are hatched grey and labelled "not evaluated". A separate small panel for
distillation: unseen pool U225 per capability (E-only vs constant).

## Fig 4 `figs/measurement_support.pdf` (or a compact table if data are thin)
From `paper/docs/LOSS_VALIDITY.md` source files `results/v26-loss-validity-pred/` and `results/v23-loss-validity/`
(inspect their JSON): the prediction gain of primary → secondary benchmark within the same capability versus across
capabilities (MAE of same-capability vs cross-capability predictors vs zero/mean baselines). If the JSON supports it,
draw one panel per capability; if not, write `paper/tables/measurement_support.tex` with the numbers and say so.

## Deliverables and checks
- `analysis/plot_fig1_responses.py`, `plot_fig2_gains.py`, `plot_fig3_transfer.py`, `plot_fig4_measurement.py`; each
  idempotent, each printing the files read; PDF + PNG at 300 dpi; fonts embedded.
- Do not modify any existing result JSON, TeX file, or register. Do not run anything on GPU.
- Write `paper/docs/FIGURE_BUILD_NOTES.md`: for each figure the data files, the selection rules, and anything that could
  not be plotted and why.
