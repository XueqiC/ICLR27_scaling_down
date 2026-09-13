# Response to the review, and the first experiments of the new round

Everything the review asked for in items one to three is done, plus the reproduction request.
Two new experiments ran on the workstation and both produced decision-relevant results. A
third, the controlled distillation matrix, started tonight. This document is the consolidated
report; the ledger entries are C63 to C66.

---

## 1. Corrections, and what survived them

Four statements were narrower than I had written them. The measurements never changed; the
readings did. Ledger C63, with C60 and C62 flagged to be read alongside it.

**The shrinkage-plus-noise hypothesis was not falsified.** The registered predictor, after its
pre-measurement amendment, carries only the shrinkage term, and the noise reading was
deferred by that same amendment. What the experiment rules out is the *pure shrinkage*
account. The review was right to insist on the distinction, and the follow-up below shows the
noise half is in fact supported.

**"Damage is carried by the variance" was a median hiding cancellation.** The two terms have
opposite signs in 81 of 105 cells. Per cell the variance term is larger for math in 35 of 35
and code in 35 of 35, but for QA in only 16 of 35. Seventeen cells have a *negative* measured
change, which a non-negative variance term cannot produce; in all seventeen the first-order
term is negative and carries it.

**Two boundaries were understated.** The displacement comes from the compressed model, so the
decomposition is post-hoc, not a prediction from pre-compression information, and no
relabelling makes it one. And the shrinkage coefficient used an uncentred projection.

**Code was not shown to be unpredictable.** It did not clear the pre-registered investment
gate, which is a different statement. The same reuse-only form cuts the error from 0.0687 to
0.0423 on the budget target and from 0.1195 to 0.0667 on the pool target, 38% and 44%
reductions with intervals above zero. It failed because on the first target the improvement,
0.0264, is smaller than the interval width, 0.0297. Also corrected: the two gate baselines
coincide, since a constant fitted to the response and then differenced is identically zero;
none of the seven candidates carried a student-state input; the historical pool-change target
mixes schedules and budgets matched only to about ten percent; and the time constant is
unidentified under the size holdout but has interior intervals in some 270M code folds.

---

## 2. The two checks on the displacement experiment (ledger C64)

**Shift invariance.** Adding seven to every logit leaves probabilities and loss untouched and
moved the old coefficient by 85%. The projection is now the p-weighted covariance,
`eps = -Cov_p(r,z)/Var_p(z)`, which moves by 1e-6 under the same shift and leaves a residual
with zero covariance against the logits, so the cross term vanishes. A unit test asserts both,
and asserts that the old coefficient fails invariance.

Re-running all 105 cells on the well-defined quantity improves the one-scalar summary and does
not change the verdict: 83.8% median relative error in the mild regime against 98.0% before,
still 36 times the exact account's error where the threshold was 1.5. Sign agreement rises
from 66% to 85%.

**What the noise term carries.** This is the part that vindicates the review's caution.

| Regime | Shrinkage | Residual first order | Noise, the variance term | Sum |
|---|---:|---:|---:|---:|
| mild | 2.7% | -9.4% | **88.6%** | 99.1% |
| moderate | 2.8% | 2.1% | 72.0% | 94.5% |
| severe | 3.7% | -2.6% | 59.2% | 61.2% |

Per capability over mild and moderate cells: math 3.2 / -3.9 / 84.5, code -0.3 / -19.8 / 88.6,
QA 91.0 / 61.4 / -94.0. For math and code, 84 to 89 percent of the response in the predictable
regime is the noise term. The shrinkage half fails because shrinkage is genuinely a small
share, about three percent, not because it was badly estimated. QA is structurally different
and needs separate treatment.

---

## 3. The intervention controls (V89b)

On the same-protocol subset, where compared trajectories share student, training mode, LoRA
settings, schedule kind and run generation, and budgets match to two percent rather than ten,
1163 pool-change pairs fall to 379. On that subset **the QA signal disappears as well**: math,
code and QA are all absent. Two student-conditioned controls, one scaling the leading
coefficient by log parameter count and one by the student's own initial loss, reproduce the
QA signal on the full set and lose it on the subset.

The report states, as asked, that a constant fitted to the response and then differenced is
identically zero, so there is one no-intervention baseline rather than two.

The conclusion is not that these relations are unpredictable. It is that the historical
trajectories are exhausted as evidence: once the protocol is controlled, nothing survives, and
only a controlled experiment can settle it.

---

## 4. New experiment: the learning-rate protocol pilot (ledger C66)

Six short trajectories, Gemma-3-1B and 4B, one fixed development pool, identical settings,
learning rate the only variable. Change from each student's own initial checkpoint, positive
means worse.

| Student | Rate | math | code | QA |
|---|---|---:|---:|---:|
| 1B | 5e-5 | **+0.003** | **+0.007** | -0.791 |
| 1B | 1e-4 | +0.051 | +0.052 | -1.312 |
| 1B | 2e-4 | +0.061 | +0.102 | -1.136 |
| 4B | 5e-5 | +0.058 | +0.022 | -1.095 |
| 4B | 1e-4 | +0.102 | +0.072 | -1.340 |
| 4B | 2e-4 | +0.142 | +0.154 | -1.180 |

Two readings, and they point in opposite directions, which is why the pilot was worth running.

**Protocol sensitivity is large.** Math and code degrade monotonically with the learning rate,
and at 5e-5 the 1B student is essentially unharmed while still gaining 0.79 on QA. Every
earlier distillation result used the 1e-4 default, so the *size* of the reported math and code
cost is a property of that choice as much as of distillation.

**The size effect survives protocol matching.** At every rate the 4B student degrades more
than the 1B on both capabilities, by roughly a factor of two on math. So the earlier
size-transfer failure is not explained away by the learning rate, though its magnitude is. QA
improves for both sizes at every rate and peaks at 1e-4 for both, so that gain is close to
size-independent.

**Decision taken:** the controlled matrix fixes one rate, 1e-4, for every student. It keeps
the 41 existing trajectories comparable and it is where the QA effect is largest. The pilot is
the sensitivity record, and any statement about math or code damage now names the rate.

---

## 5. New experiment: can pre-compression information predict across sources? (ledger C65)

The dense statistics were measured first: one forward pass per state and capability on the
nine-state development panel, 27 of 27, giving the reference-token logit margin, the trace of
the loss Hessian in logit space, and the logit variance. That third quantity is new this
round, because the displacement experiment showed it is the one that appears in the shrinkage
term.

Then a development study with leave-one-source-state-out, three strictly separated input
budgets, low-degree-of-freedom candidates, and the reading rule fixed before the numbers.

**Under the registered rule the statistics help in 3 of 9 arm-and-capability pairs**, all
through ridge: pruning math 0.084 [0.049, 0.122], pruning QA 0.109 [0.062, 0.157], grouped
quantization math 0.352 [0.257, 0.437]. Ordinary least squares gains stay inside their
intervals everywhere.

**In 9 of 9 pairs the statistics-augmented predictors still lose to the source-free median
curve.** Clearing an incremental input rule inside the parametric family is not the same as
beating what the paper already delivers. Accuracy at naming the most fragile capability also
falls when the statistics are added, from 58.3% to 41.7% for pruning and from 38.9% to 18.5%
for grouped quantization.

**The bar this round set is therefore not met**, and the reserved confirmation states should
not be spent on this branch as it stands. Their dense inputs were extracted anyway, twelve
forward passes, since those are pre-compression quantities that touch no response and keep the
option open after a freeze.

---

## 6. Manuscript and reproduction

The hindsight ranking is now labelled a diagnostic whose minimum selection favours by
construction, with the registered comparison named as primary, in the main table caption and
in both places the text refers to it. Nine pages hold.

Reproduction went from zero to eight of nine named generators running from a plain checkout.
The two capability-conditioning audits now rebuild once their inputs are mirrored and one more
digest check is routed through the anonymisation-aware comparison; one of them additionally
needs ripgrep. The one that does not run hits a genuine conflict: two experiments recorded
different states of the same grouped-quantization file, so the repository ships both and the
README says how to swap them.

One integrity point is worth stating plainly. A published frozen file cannot keep the original
bytes, so its digest check must fail. The map that reconciles this now pairs a published file
with a frozen digest **only** when anonymising the working-tree original reproduces the
published bytes exactly, 28 of 97 entries. Differences from an earlier state of an artifact are
recorded but never accepted, so a check that fails locally still fails on the published copy.
Without that restriction the map would have quietly turned the checks into decoration.

---

## 7. What is running, and what I would do next

**Running now:** the controlled distillation matrix on the A100, three students by three
independent-data rungs of 66, 198 and 594 traces per domain, two pool seeds, four budget
checkpoints at 25k, 50k, 100k and 200k supervised tokens, at the fixed rate. Eighteen
trajectories, roughly ten to twelve hours, resumable.

**Idle:** the Blackwell card, which cannot train because the stack reaches an fp32 matrix
product its cuBLAS refuses. It takes forward-only work well and did the dense statistics.
Forcing the adapters to bf16 would fix it but edits a file whose digest frozen records name,
so I have not done it unilaterally.

**My recommendation on the pruning and quantization branch:** hold the confirmation panel. The
statistics buy something inside the parametric family and nothing against the simple
source-free curve, which is the comparison that decides whether the paper's claim can be
upgraded. Spending four reserved states and two external families on that would most likely
buy a null result at full price.

**Where the remaining budget is worth putting:** the distillation matrix now under way, since
it is the only route left after the historical trajectories were exhausted, and then a frozen
confirmation of whatever it establishes, with the three-way selection rule frozen in the same
act rather than after the responses are seen.
