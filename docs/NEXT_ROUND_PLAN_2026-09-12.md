# Next round: assessment of the 2026-09-12 research plan, and what I propose instead

Status: planning document. No new GPU measurement has been taken for it. Written after the
round-v10 close-out (paper commit `ddcc3de` in the paper repository, nine-page main text).
The advisor's plan is `docs/inbox/scaling_down_laws_next_research_plan_2026-09-12.md` in
spirit; this file records which parts I accept, which I would change, and why.

## 1. Where the paper actually stands

The delivered relations are, per arm, a compact pruning form inside a tested density range, a
same-input piecewise interpolation for grouped quantization on seen states, and two frozen
distillation forms; everywhere else a source-free reference (median development curve or
per-configuration median) is as good or better. Round v10 made the main table say this
plainly: per test it names the frozen candidate, the strongest of all predictors frozen in
that round, and the delivered rule with the time it was fixed.

The honest summary of the science is narrower than the paper's framing can afford to be for
long: **we have not found pre-compression inputs that predict the size of capability damage
across source models.** `N_0`, `D_0` and the dense capability loss `L_{0,c}` support
interpolation inside a fitted configuration range on states we have already measured, and
they stop helping the moment the source changes. Every arm shows it, and §4.3 says so.

That is the gap the next round should attack, and it is a mechanism question, not a
regression-tuning question.

## 2. My main disagreement with the plan: test the mechanism, not just the descriptors

The plan proposes two dense descriptors as extra regression inputs,

```
B_{c,j} = E[ z_y - sum_v p_v z_v ]        (first-order CE response to logit shrinkage)
V_{c,j} = E[ 1 - sum_v p_v^2 ]            (trace of the CE Hessian in logit space)
```

and correctly flags that "compression acts like logit shrinkage plus isotropic noise" is a
hypothesis, not a result. I accept the descriptors. I want to change the experiment that
uses them, because the hypothesis itself is directly measurable and, if it holds, it gives a
**coefficient-free** prediction rather than one more fitted input.

### 2.1 The exact identity we should build on

For a scored reference token with final logits `z`, `p = softmax(z)`, reference index `y`,
and a logit displacement `r = z_compressed - z_dense`, the cross-entropy change is exactly

```
dL = ( E_p[r] - r_y )  +  1/2 * Var_p(r)  +  O(||r||^3)
```

where `E_p[r] = sum_v p_v r_v` and `Var_p(r) = sum_v p_v r_v^2 - (sum_v p_v r_v)^2`. No
isotropy assumption is needed. The plan's two descriptors are the special case
`r = -eps * z`, which gives `dL = eps * B + 1/2 * eps^2 * Var_p(z)`, and isotropic noise of
variance `sigma^2` gives the `1/2 * sigma^2 * V` term.

I verified these numerically before proposing them (vocabulary 5000, 200 random draws per
case; the check is `analysis/verify/ce_expansion_check.py`, which runs on CPU in a second):

| displacement scale | mean exact dL | second-order prediction | first-order only | mean abs. rel. error of second order |
|---|---|---|---|---|
| 1e-3 | -0.000043 | -0.000043 | -0.000043 | 0.00% |
| 0.03 | 0.001961 | 0.001961 | 0.001545 | 0.01% |
| 0.3 | 0.015545 | 0.015335 | -0.026416 | 0.56% |
| 1.5 | 0.980181 | 1.010875 | -0.029574 | 11.2% |

The `B` identity matches a central finite difference of the shrinkage response to 1.5e-12
relative, and `V` equals `trace(diag(p) - p p^T)` exactly. Two facts matter for the paper:
the first-order term alone gets the **sign** wrong once displacements are non-trivial, so
damage is carried by the variance term; and the expansion degrades smoothly with displacement
size, which is a principled account of the "mild / non-trivial / aggressive" damage regimes
the plan asks us to separate. Our compact forms work in a limited range because the expansion
does.

Two implementation facts settled while preparing this (codex V87, CPU only, 199 tests):
the identity is `B = +dCE/deps` under `z -> (1-eps) z`, and the project's loss aggregation is
**pooled tokens** (`total_loss / total_tokens`, `analysis/v10_quantization.py:179`,
`analysis/v12_distill.py:523`), not an equal per-sample mean, so `B` and `V` are aggregated
the same way, with a per-byte variant under the same rule. Finite-difference agreement on a
tiny CPU model: `B` to 2.42e-7, `V` against the Hessian trace to 8.02e-8.

### 2.1b Why this is not the descriptor line we already closed

The project has already run a descriptor line and closed it as a negative, and the new round
must not reopen it under a new name. What was tried, and its outcome:

| Family | What it was | Where | Outcome |
|---|---|---|---|
| Diagonal capability Fisher, tr(F), capability spectra, deleted mass | backward passes on the dense model | `analysis/v6_capability_geometry.py:531` | normalised spectra collapse across capabilities, pooled Spearman 0.346; no capability discrimination |
| Top-1% Fisher-mass concentration | pre-registered prunability predictor | ledger C13 | retired as falsified |
| Signed first-order alignment `g . dw` | sign of the damage | ledger C6 | the one mechanism win: cell-level sign accuracy 67.6%, CI [58.3, 75.7] |
| Second-order weight-space Taylor `1st + 1/2 quad` | magnitude of the damage | `analysis/v6b_alignment.py:120` | fails by orders of magnitude at and after the cliff (predicted -0.005 against measured +2.214) |
| V2 layerwise retention statistics | signed pruning amplitude | `docs/PRUNE_DESCRIPTORS.md` | decisively worse than metadata alone (math 2.07 against 0.87) |
| V3 dense activation moments | signed pruning amplitude | `docs/PRUNE_DESCRIPTORS_V3.md` | also worse, and gets math and code signs wrong on the held-out source |

Three differences make the present proposal a different experiment rather than a rerun.
First, `B` and `V` are not heuristic summaries chosen because they sound relevant; they are
the first and second derivatives of the quantity we predict, so they enter through an
identity and not through a fit. Second, the failed families were computed on the twelve
heterogeneous models and have zero coverage on the Pythia states where every frozen test
lives, and most of them need backward passes, which puts them outside the paper's own input
budget; `B` and `V` are forward-only and can be produced on the Pythia panel in the same pass
that measures the loss. Third, the weight-space second-order expansion failed because the
weight perturbation is large; the expansion proposed here is in the **measured logit
displacement**, which removes that approximation entirely. Neither quantity has ever been
computed in this project: the full-vocabulary distribution `p` is never materialised, and the
only `entropy` in the code base is over layer indices, not over the vocabulary.

The cheapest insertion point is the single loss primitive that feeds the pruning,
quantization, distillation and measurement-check paths, where the logits are already in scope
one line before they are reduced (`analysis/v6_capability_geometry.py:471`). Hooking there
also fixes the per-example-loss gap that blocks item-level intervals today (ledger C51;
`analysis/v75_distill_audit.py:244` records the blocker).

### 2.2 Three tests, in order, each with its own failure mode

- **T1 (does the expansion hold?)** For already-measured (state, configuration) pairs,
  recompute the compressed model deterministically, run paired dense and compressed forwards
  over the same scored tokens, reduce `E_p[r] - r_y` and `1/2 Var_p(r)` online, and compare
  their sum with the measured `dL`. No fitting, no free coefficients. Failure tells us the
  predictable regime ends where the perturbation stops being small, and we report the
  boundary in displacement units instead of guessing at density or bit-width.
- **T2 (is the displacement structured?)** Decompose `r` into a shrinkage component along
  `z` (`eps_hat = -<r,z>/||z||^2`) and a residual, and ask whether `eps_hat * B + 1/2 *
  sigma_hat^2 * V` reproduces T1's prediction. This is the plan's hypothesis, stated as a
  measurement. If it fails, `B` and `V` stay useful as summaries of sensitivity but the
  "shrinkage plus noise" story is dropped, with evidence.
- **T3 (the actual scaling-down law)** If T1/T2 hold, the prediction problem factorises into
  a derived response law in `(eps, sigma^2)` and a map from configuration and source metadata
  to `(eps, sigma^2)`. `eps` and `sigma^2` are properties of the weight perturbation, not of
  the capability, so this map has a real chance of transferring across sources where `dL`
  did not. T3 is where the frozen new-state confirmation goes.

This design is strictly stronger than adding `B` and `V` to a regression: it can fail
informatively at three separate places, and its success would be a mechanism result rather
than a fit. It also answers the reviewer question the current paper cannot: *why* do the
compact forms have the ranges they have.

### 2.2b What each tier is allowed to claim, and its measurement budget

T1 and T2 need the compressed model's logits, so on their own they are diagnostics, not
predictions, and the paper must label them that way. Their value is that they factor the
problem into pieces with different costs, which gives a new and honestly cheaper budget tier:

| Tier | What you must measure | What it predicts | Status in the plan |
|---|---|---|---|
| Diagnostic | dense and compressed logits on the scored probe tokens | decomposes the measured loss change; no free coefficients | T1, T2 |
| Capability-free measurement | `B`, `V` once per capability before compression, plus the displacement statistics `(eps, sigma^2)` from one forward pass of the compressed model on unlabelled text | every capability's loss change without evaluating any capability | the practically useful claim, testable this round |
| Pre-compression | configuration and source metadata only | `(eps, sigma^2)`, hence the loss change | T3, the actual scaling-down law |

The middle tier is worth stating as a target in its own right: it would mean a practitioner
predicts capability damage from one cheap unlabelled forward pass instead of running every
capability benchmark. Whether the displacement statistics estimated on generic text transfer
to the capability probe distributions is an open question and one of the things T2 measures.

### 2.3 Cost, and why this goes first

T1 and T2 need paired forwards on models we already have, on probe sets we already use, for
configurations we have already measured. Pruning uses a stored quantile threshold and
quantization is round-to-nearest, so both transforms are exactly reproducible
(`analysis/v6_capability_geometry.py:289`, `analysis/v54_quant_group.py:89`). Nothing is
trained. Estimated 1 to 2 GPU-h for a dozen state-configuration pairs across three
capabilities, and it is the cheapest experiment in the plan by an order of magnitude.

## 3. Package-by-package verdict

| Plan item | Verdict | Reason |
|---|---|---|
| A1-A3 dense descriptors as regression inputs | Accept, restructured as T1/T2/T3 above | The hypothesis is measurable; a coefficient-free test beats another fitted input |
| A4 asymmetric grouped RTN control | Accept, small | Current quantizer is symmetric RTN only (`v54_quant_group.py:193`); one controlled variation tests whether nominal (b,g) omits operational information |
| A5 new-state confirmation | Accept | Needs weight-identity-distinct states that entered no fit; see the state inventory section |
| B1 units and roles | Accept as written | Matches what the paper already does; keep teacher, `S_0` and deployment reference `M_0` separate |
| B2 learning-rate pilot before size conclusions | Accept, and it is the precondition for any 4B claim | Otherwise an optimisation artefact gets reported as a size law |
| B3 18-trajectory crossed matrix | Reduce and gate | This is the expensive item; run the retrospective intervention analysis on the existing 25 trajectories first, on CPU, and spend training budget only if the signal exists |
| B4 two-term saturation-plus-reuse candidate | Accept, with `p` pre-registered | Identifiability of `tau` must be checked before any optimum-budget claim |
| B5 intervention-effect evaluation | Accept, and promote | Predicting the effect of doubling `T` or enlarging `D_U` is a better target than average MAE, and most of it is computable on existing data today |
| B6 per-example records and a second distribution | Accept | Fixes a real gap: per-example losses are not stored, so no item-level intervals exist |
| C1-C3 three-way selection panel | Accept, after A and B | The point is to remove the dependence on one historical KD candidate at 1B@64k, which round v10 showed carries the math and code margin |
| MoE, accuracy grids, new teacher API calls, several new compression algorithms | Drop, as the plan says | Out of scope for this round |

## 4. Resource audit (measured 2026-09-12, 16:40 EDT)

| Item | State |
|---|---|
| rai GPU 0, RTX 6000 Ada, `GPU-83e71df1` | 42.9 of 49.1 GB used; a root vLLM plus a labmate's training |
| rai GPU 1, RTX PRO 6000 Blackwell, `GPU-97762062` | 49.1 of 97.9 GB used, 37% util, labmate's vLLM and training |
| rai GPU 2, RTX 6000 Ada, `GPU-20b20454` (our designated card) | 46.8 of 49.1 GB used, 70% util; our own tc-alignment vLLM (15h45m) plus a labmate's vLLM. About 2.3 GB free |
| rai GPU 3, A100 80GB, `GPU-8b270cf8` | 63.9 of 81.9 GB used; our tc-alignment processes |
| rai GPU 4, RTX PRO 6000 Blackwell, `GPU-aaebd5af` | 50.8 of 97.9 GB used, 29% util, labmate |
| HiPerGator | Tunnel down: `Permission denied (keyboard-interactive)` on `ssh hpg`. Needs autossh restarted on the Mac |
| Disk | 13 TB of 14 TB used, 522 GB free, 97%; `results/` is 52 GB |
| GPU environment | System `python3` with torch 2.10.0+cu128, CUDA available; the project `.venv` has no torch and is for CPU analysis |

Consequences for planning. There is no free GPU on rai right now, and the card we are
supposed to use has 2.3 GB free. The two Blackwell cards have about 47 GB free each but
break fp32 cuBLAS, so any use of them needs a bf16 forward path with float32 reductions,
which is exactly what the descriptor script is written to do; that is worth one smoke test
before relying on it. The historical 72 GPU-h figure is a total, not a new allowance.

## 5. Proposed order of work, with gates

**Phase 0, no GPU, running now.** Descriptor implementation with finite-difference tests,
per-example evaluation records, asymmetric grouped RTN option (codex V87). Then the
retrospective intervention analysis of the existing distillation trajectories, the
never-measured state inventory, and the development/confirmation manifest. Paper side: add
the "dense statistics" row to the information-budget table (`tab:budgets`) so the current
claims state their budget, and keep every old prediction as it stands.

**Gate 0.** If the retrospective intervention analysis shows no predictable effect of
doubling `T` or enlarging `D_U` on existing trajectories, the 18-trajectory matrix does not
get trained; we would be buying variance.

**Phase 1, 1 to 2 GPU-h.** T1 and T2 on already-measured state-configuration pairs, plus
`B`/`V` extraction on the Pythia panel. Pre-registered success criterion for continuing:
the derived second-order prediction must beat the strongest same-information baseline by at
least 15% relative on development states, and the shrinkage-plus-noise reduction must stay
within the noise of the exact expansion.

**Gate 1.** If T1 fails outside a narrow displacement range, we publish the boundary and
stop the descriptor branch rather than adding polynomial terms.

**Phase 2, needs the user's go and a free card.** T3 frozen confirmation on at least four
weight-identity-distinct, never-measured source states across at least three sizes; the
learning-rate pilot (about 4 GPU-h); then the reduced distillation matrix if Gate 0 passed;
then the three-way selection panel.

**Budget rule, as the plan asks.** At most half of whatever GPU-h the user grants goes to new
development experiments, at least 30% is reserved for frozen confirmation, and the remainder
covers pilots and re-measurement. Manifests are written before any new response is unblinded.

## 6. What I will not do

Rename earlier descriptor work as new (the Fisher, alignment and geometry experiments stay
where they are, with their recorded outcomes). Re-label already unblinded tests as
independent. Treat repeated Pythia revisions as separate source states. Fit a form and then
call the fitted coefficients a mechanism. Extend a pool's nominal size into a claim about
training information actually delivered. Report budget cells as independent samples.
