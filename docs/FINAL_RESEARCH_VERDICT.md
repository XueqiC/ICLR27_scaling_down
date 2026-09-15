# Final research verdict for the predictive round

Written 2026-09-15 after the last corner trajectory finished, and revised the same night after the
advisor's reading of the corner result. Every number is in the ledger, C73 to C78, with the artifact
that produced it. This version supersedes the first; the first overstated what the corner test shows.

## The question

Whether a small set of pre-compression observables predicts the conditional loss of unseen
compression and distillation configurations well enough to support a resource decision that can be
checked.

## The verdict

**No new predictive relation cleared the bar.** Three permitted structures, one parameter set each
producing the response and both interventions, lost to the strongest baseline on every capability,
target and grouped hold-out, including both pre-declared primary targets. The confirmation package
never launched, and the larger student, the second family and the data-requirement relation, all
conditional on it, did not run. One secondary cell replicates across all three held-out students
against a pre-specified baseline and is recorded as the only candidate that would have been worth
confirming. Whether the launch verdict changes when the gate is read against the development-stage
baseline rather than the post-hoc strongest one is being checked (C78); the per-fold records read so
far say it does not.

**On the corner test, the primary QA check failed to reject additivity in the budget-and-reuse
region measured, on the two students measured; the frozen interaction candidate clearly overestimated
the second difference there.** That is the whole of what the test shows. Failing to reject is not
establishing: the measured statistic is about −0.21 to −0.24 nats on both students, inside the
registered band but not shown to lie inside any pre-defined equivalence margin, so no claim of
practical additivity is made. It does not exclude an interaction, and it does not license the
inference that the fitted structures failed because of the form of their individual terms. What the
test does separate, cleanly, is two things the earlier falsification had run together: the specific
implication of the logarithmic reuse form fails, and additivity as such was not rejected.

**The result does not generalise across capabilities.** On math and code the 4B student's statistic
falls outside its band while the 1B's does not, and TriviaQA shows the same shape on the fresh-sample
check. These readouts were registered as marginal, underpowered, or resting on two seed pairs. They
stand as scope statements and as leads, and they are the reason the QA finding is stated for QA only.

## What the paper says now

The systematic-investigation framing carries the round's evidence. Three findings have content:
within-configuration prediction and cross-source transfer are problems of different difficulty, and
adding source descriptors did not stably improve transfer within the inputs and models tested; the
distillation data response depends strongly on student and evaluation distribution, with the reuse
cost reproducing on three QA distributions and the net gain confined to 2Wiki; and curve fit,
structural interpretation and decision validity each need their own verification, since a rejected
log form is not a rejected additive class, an accurate second-order displacement account is not a
pre-compression predictor, and a low-regret selector is not thereby shown to beat quant-only across
references.

## What would reopen experiments

A new information source or a mechanism hypothesis, taken up as a new research stage with its own
justification. The remaining 41 GPU-hours are not a reason on their own, and another reuse function
is not a new hypothesis.
