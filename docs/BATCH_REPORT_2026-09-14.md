# This batch, end to end: the falsification, the scope, and where the law can still be built

Four things ran since the last report: an audit of the proposed joint response form, four new
trajectories trained to make one specific test possible, an extension of the displacement
diagnostic to a third compression family, and an evaluation of every controlled endpoint on
two external QA distributions. Ledger entries C68 to C72.

The headline is that the proposed functional family is falsified on its own terms, and that
the effect it was meant to describe is real, large, and now measured across three
distributions and three student sizes.

---

## 1. Two of my own statements needed narrowing first

**"Not predictable" became "did not meet the pre-registered threshold."** For the
budget-doubling target on code, a one-parameter reuse form cuts the error by 43% with an
interval clear of zero and misses the bar by 0.0012 nats. That is development evidence worth
an independent confirmation, and reporting it as a null was wrong.

**The claim that the noise half of the mechanism hypothesis holds was withdrawn.** What the
displacement experiment measures is the variance of the component orthogonal to the logits.
A large orthogonal residual variance says the loss change lives there. It does not establish
that the residual is random, and it is not a predictor, because computing it needs the
compressed model. What stands is that the shrinkage share is about three percent, so the
pure-shrinkage account fails on magnitude.

## 2. The budget-only anomaly was exactly what the review suspected

With a student and a budget both fixed, a predictor that depends only on the budget must
predict exactly zero for a change of pool. The small non-zero gains reported earlier
reconstruct, 100 percent of them, from the residual budget mismatch allowed by the matching
tolerance. At identical budgets both the prediction and its gain are exactly zero.

The same audit found a real gap in the logs: per-domain supervised-token totals are not
recorded, only totals and per-domain example counts, so per-domain token shares cannot be
reported for these runs. That is stated rather than approximated with example counts.

## 3. The proposed joint form is falsified, not under-powered

The form was
`delta = (a + a'z) log(1 + T/T_ref) + (b + b'z) log(1 + E)`, four coefficients per capability,
zero response at zero budget, one fitted parameter set producing the response and both
interventions. It is not algebraically equivalent to any candidate tested before, across
fourteen comparisons.

It did not meet the registered threshold for either student descriptor or any capability. More
importantly, its own falsifiable implication fails. At fixed student and budget the predicted
pool-change effect is a budget-independent coefficient times a bracket, so the measured effect
divided by that bracket should be flat. It is not.

**Four new trajectories were trained to test this properly.** The matrix had no pair of pools
whose checkpoints land at the same supervised budget, so the test could only be run
approximately. The new runs, 1B and 4B at 132 traces per domain with two seeds and fourteen
checkpoints each instead of four, provide matched pairs at one percent and, for some
contrasts, at 0.3 percent, with residuals of 22 to 64 supervised tokens.

**The drift survives at both tolerances:**

| Tolerance | math | code | QA |
|---|---|---|---|
| 1% | 0.261 [0.105, 0.426] | 0.168 [0.083, 0.283] | 3.99 [1.80, 6.19] |
| 0.3% | 0.348 [0.096, 0.612] | 0.168 [0.072, 0.229] | 2.34 [0.39, 4.04] |

A single interaction term of the form the plan allowed does not remove it, and buys no
held-out gain.

**One piece of structure is worth keeping.** The drift concentrates in the high-reuse
contrasts and nearly vanishes between the two low-reuse pools: 0.038 for math and 0.102 for
code there, against 0.277 and 0.202 in the high-reuse contrast. The response looks close to
separable at low reuse and clearly is not at high reuse, which is where the largest effects
are. Any replacement candidate has to let the reuse effect grow with the budget, and I am not
improvising one: the next form gets written down before it is fitted.

## 4. The scope of the effect, which decides how a data requirement may be stated

Every controlled endpoint was scored on a fresh 2Wiki sample, MuSiQue and TriviaQA, using the
sampling and scoring of the earlier scope check so the numbers are comparable. Change from
each trajectory's own initial student, averaged over the two seeds; positive means worse.

| Student | reuse 11.7 | reuse 5.8 | reuse 3.9 | reuse 1.3 |
|---|---|---|---|---|
| 270M, 2Wiki / MuSiQue / TriviaQA | +0.60 / +0.63 / +0.90 | — | -0.92 / +0.06 / +0.40 | -1.16 / -0.01 / +0.36 |
| 1B | +2.62 / +1.99 / +0.99 | -0.22 / +0.66 / +0.48 | -0.77 / +0.48 / +0.36 | -1.10 / +0.23 / +0.26 |
| 4B | +5.77 / +3.01 / +2.49 | +2.40 / +1.43 / +1.43 | +0.06 / +0.69 / +1.29 | -1.00 / +0.28 / +0.56 |

**The cost of over-reuse transfers; the benefit does not.** All three distributions worsen at
high reuse and improve monotonically as reuse falls, so "reusing a small pool hurts QA" is not
a 2Wiki artefact. But only 2Wiki crosses into a gain. MuSiQue flattens near zero and TriviaQA
stays positive at every reuse level measured, for every student.

**The requirement rises with student size, and rises further off 2Wiki.** The 4B student is
damaged roughly twice as much as the 1B at matched reuse and needs the lowest reuse measured
before it gains anything on 2Wiki, while still losing on both external sets there.

So a data requirement of the form *how much independent data keeps the QA loss increase below
a threshold* is supportable across distributions, and a claim that distillation improves QA
remains a 2Wiki statement. That completes the earlier finding that the gain does not transfer,
by showing that the dependence on reuse does.

## 5. The mechanism boundary now covers three compression families

The displacement diagnostic grew from 105 to 339 cells, adding the development states the
first sweep missed, intermediate pruning densities, two more group sizes, and a third family,
per-channel quantization. Median relative error of the coefficient-free second-order account:

| Family | Configurations and error |
|---|---|
| pruning | d=0.9 0.9%, 0.85 4.3%, 0.8 5.3%, 0.75 11.1%, 0.7 16.0%, 0.65 28.2%, 0.6 44.4% |
| grouped quantization | 5 bit 1.6-2.7% at every group size, 4 bit 3.3% to 9.2% by group size, 3 bit 35.9% |
| per-channel quantization | 8 bit 2.5%, 6 bit 3.0%, 5 bit 4.2%, 4 bit 23.8%, 3 bit 53.9% |

The boundary is the same in all three families and tracks the damage magnitude rather than the
family or the knob. Grouped four-bit with fine groups sits at 0.06 nats and 3.3% error while
per-channel four-bit sits at 0.24 nats and 23.8%: the same curve through two different knobs.

## 6. Things that went wrong, and what was done

**Two tool guards fired in the wrong place and were fixed properly.** The 4B adapters carry no
base-model name, since peft leaves that null when the base is loaded locally, and the
evaluation refused to score them as a mismatch. Absent and mismatched are now distinguished, a
real mismatch stays fatal, and the output records which case applied. Changing the tool
changed its protocol record, so it then refused to write into the existing output directory,
which is correct, and the rerun went to a new subdirectory.

**Test fixtures had been written into the results tree** by the suite and were moved out rather
than left to contaminate it.

**Noted and not changed:** those refusals exit zero where they should exit non-zero. Fixing it
edits a file whose digest frozen records name, so it waits for a moment when that is being
done deliberately.

**Idle time:** the training finished overnight and I did not notice for several hours. That is
on me; the lanes now have explicit waiters.

## 7. Where the law can still be built

**Established this batch.** The reuse effect is real, large, monotone, size-dependent, and its
cost is distribution-general. The mechanism account of compression damage holds across three
families with a boundary in damage magnitude.

**Falsified this batch.** The proposed joint family, on its own implication, at two matching
tolerances.

**The open question is unchanged in shape but sharper in content.** A data-requirement relation
needs a form whose reuse term strengthens with budget, and the region where the current family
fails is exactly the high-reuse region where the effects are largest. The cheapest next step is
not more training: it is to write down one such form, check it against the matrix and the four
critical trajectories that already exist, and only then spend the confirmation package.

Nothing is running. Both workstation cards and Loni are free.
