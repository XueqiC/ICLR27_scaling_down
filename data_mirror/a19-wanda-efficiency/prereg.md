# A19 — measurement-efficiency confirmation under a second pruning criterion (Wanda)

Status: design fixed 2026-09-23 from the fourth review ("Wanda 应测试测量效率"); the user allowed
GPU work on two hosts in parallel on 2026-09-23 10:04 EDT. Runner: `analysis/a19_wanda_efficiency.py`.

## Question

Under a different pruning algorithm, does the same compact structure, refitted, still predict unseen
states and densities from half the development density measurements about as well as the full-grid
per-density regression? The Pythia confirmation (A11) and the OLMo-2 replication (A18) both used
global magnitude pruning; this experiment changes only the pruning criterion.

## Pruning criterion

Wanda (Sun et al., 2023) as implemented in `analysis/p2_wanda_panel.py`: every weight of the block
linears is scored by its magnitude times the L2 norm of its input activation over the calibration set,
and the scoring is applied per output row at the target retained density. Calibration: the first 128
documents of the C4 English validation split with at least 512 tokens, truncated to 512 tokens, the
same set for every state. Activation norms are computed once per state and reused for every density
of that state; their GPU seconds are recorded as part of the state's cost (they are the cost that
magnitude pruning does not pay).

## Family and states (the A11 design, unchanged)

- Development (9): Pythia 160M, 410M and 1.4B at steps 16000, 64000 and 143000, the nine-state panel
  of the Pythia development. Full grid: densities 0.9, 0.8, 0.7, 0.6 → 36 configurations per
  capability. Reduced grid: the same nine states at 0.9 and 0.7 → 18.
- Test (4): the A11 states, 160M at step 80000, 410M at step 112000, 1.4B at step 48000 and 1B at
  step 48000, in no development fit, at the A11 densities 0.9, 0.85, 0.8, 0.75, 0.7, 0.65 (three
  seen and three unseen densities), pruned only after the freeze.
- Every checkpoint is a public EleutherAI revision; weights are read from the local cache or
  downloaded on the shared cluster (no restricted model is involved).

## Predictors (all refitted on Wanda measurements, same labels, same tuning budget)

Compact power form (Eq. 3, five parameters per capability), quadratic strength form, strength-only
curve, median density curve, regularised per-density regression with linear interpolation, zero
change. Estimator for every linear-in-feature form: ridge on the standardized inputs with the weight
chosen from {0, 1e-3, 1e-2, 1e-1, 1} by leave-one-state-out on the development grid, as in A18.
Diagnostic only: the A11 magnitude-pruning coefficients (power_18) applied to the Wanda measurements
without refitting, which separates structure reuse from coefficient transfer.

## Endpoint and cost

Capability loss on the V6 probes (odd half, 64 per capability) and on the A18 new item set
(64 per capability, `results/a18-second-family/items.json`), per-item loss sums and token counts
stored; GPU seconds per state for loading, calibration, each pruned evaluation and the dense anchor.
Reporting as in A18: per-state errors, paired item bootstrap with states fixed, reduced-grid compact
form against full-grid regression as the primary comparison; the reduced grid's share of GPU seconds
and evaluation tokens with the calibration pass counted once per state.

## Order of operations, one freeze

`--items` (reuses A18 items) → `--run` on the nine development states → `--freeze` → `--run` on the
four test states (refused before freeze.json exists) → `--score` → `--table`. Any change after the
test belongs to the next round. Development states may run on two hosts in parallel; the freeze is
written once from the assembled development measurements.

## Pre-specified reading

Support for the claim "the measurement saving does not depend on the pruning criterion": the
reduced-grid compact form within 0.020 nats of the full-grid regression on mathematics and code over
the four test states (the tolerance the A11 confirmation reached). A failure locates the dependence of
the structure on the pruning algorithm and is reported as such; it does not license a claim that the
shared power form covers the pruning family.
