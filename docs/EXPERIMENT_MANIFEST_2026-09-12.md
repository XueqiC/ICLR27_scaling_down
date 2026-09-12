# Manifest: development and confirmation, round of 2026-09-12

Written so that what counts as development and what counts as confirmation is fixed before
the next measurement, and so that an already unblinded artifact can never be re-labelled as
an independent test later.

## Status of every panel now in use

| Panel | States or runs | Status | Why |
|---|---|---|---|
| Pythia controlled panel, 9 states (160M, 410M, 1.4B x 16k, 64k, 143k) | measured, in every fit | **development, unblinded** | the fitting set for pruning and quantization |
| Pythia 1B and 6.9B at further steps | measured, first as frozen prospective tests, then folded into the V53 refit | **development, unblinded since the refit** | may not be reused as an independent test |
| 410M@48k, 1.4B@112k, 6.9B@80k | measured, held out of fits | **spent confirmation** | already scored; not available again |
| 2.8B, three revision labels | one learned state, measured twice | **spent, and counts as one state** | 387 of 388 learned tensors byte-identical; the remaining tensor differs by a signed zero |
| 160M@32k, 410M@32k, 1.4B@32k, 1B@64k | measured under the locked selection rule | **spent confirmation** | the V78 panel |
| V88 displacement cells | 5 states x 7 configurations | **diagnostic on development states** | decomposes a known response; carries no predictive claim |
| Gemma-3 students 270M, 1B | many pools and budgets | **development** | the distillation fitting set |
| Gemma-3 4B | trained at several pools | **development from now on** | it has been observed repeatedly; it is no longer a fresh student |
| Gemma-3 12B | one run, U=600, full fine-tune, no trajectory checkpoints | **unused for distillation laws** | different training mode; not comparable to the LoRA panel |

## Never measured, and therefore available

Cached and unmeasured: pythia-2.8b@step64000, which is the same learned state as the other
2.8B labels and so buys nothing new. Every other cached Pythia revision has been measured on
at least one arm. **A genuinely new source state requires downloading a revision that is not
in the cache**, which is cheap for 160M through 1.4B. SmolLM2-360M and SmolLM2-1.7B have
never been touched by this project, so they remain available as a cross-family challenge.

Unmeasured combinations on measured states, which are new configurations rather than new
states: 410M@48k and 6.9B@80k have pruning only; 1.4B@112k has no per-channel quantization
and no distillation; 1B and 6.9B lack grouped quantization except 1B@96k.

## Teacher traces

600 distinct prompts per domain, two teachers over the same prompts, so 1200 distinct
(prompt, teacher) traces per domain and no new API call is needed for a 1:3:9 pool ladder.
Pools already spent: U = 16, 75, 150, 200, 225, 300, 375, 450, 600 with the data seeds
recorded in the register. A fresh ladder of U = 66 / 198 / 594 avoids every spent size under
the current independent-draw protocol; U = 46 / 138 / 414 does so with strictly disjoint
rungs, which needs a partition step that does not exist yet.

## Rules for the next round

1. Registration before measurement, in `docs/prereg/`, naming the cells, the quantities, the
   splits and the pass rule. An amendment is allowed only before the first cell is measured,
   and it must keep the original text visible.
2. Predictions and their inputs are hashed at freeze time. A frozen file is never edited; a
   correction is a new version with a ledger entry.
3. Development results may enter a later fit. Once they do, they stop being available as
   confirmation, and the ledger says so on the entry that consumed them.
4. Budget: at most half of a granted GPU allowance on new development measurement, at least
   thirty percent reserved for confirmation, the rest for pilots and re-measurement.
5. Every reported interval names its clustering unit and its number of clusters.
