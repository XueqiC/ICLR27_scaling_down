# A18 — second-family measurement-efficiency confirmation (family confirmed 2026-09-22 19:05 EDT)

Status: design fixed on 2026-09-22 from the advisor's plan; family and checkpoint list confirmed by the
user ("好的", 19:05 EDT) and frozen below; the measurement runner is `analysis/a18_second_family.py`.

## Question

Does a compact pruning relation fitted on half the development density measurements predict new
model states of a second family within the tolerance the Pythia confirmation reached, against the
same competitors at the same labels and tuning budget? Primary target: the reproducibility of the
measurement advantage of the compact structure (refit on the second family). Secondary diagnostic:
zero-calibration transfer of the Pythia coefficients to the second family.

## Family (proposed): OLMo-2

- `allenai/OLMo-2-0425-1B` and `allenai/OLMo-2-1124-7B`, stage-1 intermediate checkpoints; each
  revision name carries the token count (e.g. `stage1-step10000-tokens21B`), so D0 is confirmable
  and N0 is the released architecture. Weights differ across revisions by construction.
- Development states (6): 1B at stage1-step190000-tokens399B, stage1-step950000-tokens1993B,
  stage1-step1720000-tokens3608B; 7B at stage1-step93000-tokens391B, stage1-step464000-tokens1947B,
  stage1-step836000-tokens3507B (10, 50 and 90 percent of each family's stage-1 token range).
- Test states (4): 1B at stage1-step570000-tokens1196B and stage1-step1340000-tokens2811B; 7B at
  stage1-step278000-tokens1167B and stage1-step650000-tokens2727B (30 and 70 percent, inside the
  development range of each size), used in no fit.
- Every checkpoint's safetensors shards are hashed before measurement (`hashes.json`); two states with
  identical weights abort the run.

## Design

- Full development grid: 4 densities per state, {0.9, 0.8, 0.7, 0.6} → 24 configurations.
- Reduced grid: the same 6 states at 2 densities, {0.9, 0.7} (as in the Pythia confirmation) → 12.
- Test: the 4 held-out states at 3 unseen densities fixed now, {0.85, 0.75, 0.65}.
- Forms, all with the same labels and the same tuning budget: compact power form (Eq. 3, five
  parameters per capability), same-complexity quadratic strength form, regularised per-density
  regression with linear interpolation between densities, median density curve.
- Estimator, fixed before the freeze (2026-09-22 15:45 EDT, after the three 1B development states
  had been measured and before any 7B or test state was fitted): every linear-in-feature form (power,
  quadratic, per-density) is fitted by ridge on the standardized inputs, with the ridge weight chosen
  from {0, 1e-3, 1e-2, 1e-1, 1} by leave-one-state-out on the development grid, once per form,
  capability and grid; the exponent and curvature grids are those of the Pythia development. With
  six states and a four-input amplitude, an unregularised fit is nearly exactly determined, and a dry
  run on the three 1B states alone gave code predictions of tens of nats; the rule is the same for
  every form, so it favours none.
- Endpoint: capability loss on the existing V6 probes (odd half, 64 per capability) and on a new
  independent item set per capability, disjoint from every probe, sample, teacher trace and few-shot
  exemplar used so far; per-item loss and token counts are saved for every cell.
- Cost accounting: evaluation tokens and GPU seconds per state and configuration, dense anchors
  included; "half the development configuration measurements" is 12 against 24 configurations, and
  the recorded times say what fraction of compute that is.
- Reporting: per-state errors (configurations are not independent models); item bootstrap only for
  evaluation-sample uncertainty; the reduced-grid compact form against the full-grid regression on
  the 4 test states as the primary comparison; zero-calibration Pythia coefficients as a diagnostic.
- One freeze, one confirmation: forms and coefficients are fixed on the development grid before any
  test state is pruned; any change after the test belongs to the next round.

## Item sets

`items.json`: 64 new items per capability (seed 2027) from MATH-500 test, MBPP test and 2Wiki validation,
excluding every V6 probe, the V71 register and its exclusions, the 384 P1 items, the 600 teacher-trace
questions and the 4 few-shot exemplars.

## Order of operations

`--hash` → `--run` on the six development states → `--freeze` (writes freeze.json and its sha256) →
`--run` on the four test states (refused before the freeze) → `--score`.
