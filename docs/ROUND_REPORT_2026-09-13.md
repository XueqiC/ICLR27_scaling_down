# The new round, end to end: what was run, how, and what came out

Everything the plan asked for in its first phase has finished. Two machines were used: the
shared workstation and the Loni cluster. Five things were measured, three of them new
experiments, and one of them produced the strongest positive result this project has had.
Ledger entries C63 to C67.

---

## 1. The controlled distillation matrix, and what it shows

**Why it existed.** The historical trajectories had been exhausted as evidence: once the
protocol was controlled, no relation survived on them. Only a controlled experiment could
separate the training budget from the amount of independent data.

**What was run.** Eighteen trajectories under one protocol: students of 270M, 1B and 4B
parameters; independent-data pools of 66, 198 and 594 traces per domain, a 1:3:9 ladder that
collides with no pool used before; two pool seeds; LoRA at a learning rate fixed by the pilot
below; identical schedule, stopping budget and evaluation in every cell; four checkpoints per
trajectory, ending near 200k supervised tokens. Three lanes ran concurrently on one A100,
which took the wall time from about 32 hours to about 10.

**The effect, at a fixed budget of roughly 200k supervised tokens.** Positive means worse.

| Student | small pool, reuse ~12x | middle pool, reuse ~4x | large pool, reuse ~1.3x |
|---|---|---|---|
| 270M math / code / QA | +0.18 / +0.30 / **+0.29** | +0.09 / +0.18 / -0.87 | +0.08 / +0.14 / -0.97 |
| 1B math / code / QA | +0.30 / +0.38 / **+2.55** | +0.12 / +0.15 / -0.70 | +0.09 / +0.09 / -0.91 |
| 4B math / code / QA | +0.56 / +0.40 / **+4.85** | +0.20 / +0.24 / -0.18 | +0.12 / +0.14 / -1.17 |

Three readings, all new:

**Reuse is expensive, and the price is paid mostly by QA.** Spending the same budget on a
small pool reused about twelve times costs QA +0.29 nats at 270M, +2.55 at 1B and +4.85 at
4B. Spending it on nine times more independent data turns that into a gain of about one nat
for every student. Math and code damage falls monotonically as well, by a factor of five on
the 4B student.

**The reuse penalty grows with student size.** At matched reuse the QA cost rises from +0.29
to +2.55 to +4.85 across the three sizes. This reinterprets the earlier finding that the 4B
student "fails to transfer": those runs all sat in the high-reuse regime, where larger
students degrade fastest.

**This is the first time the two axes were separated under control.** Everything before it
varied budget and pool together, or varied the protocol at the same time.

## 2. Is the intervention predictable? Registered answer: not yet, and the two targets differ

The rule was fixed before the numbers: the same candidate must beat the single
no-intervention baseline on both targets, under leave-one-student-out, by more than the full
width of its interval. Seven candidates, all low degrees of freedom, no searching. A constant
fitted to the response and then differenced is identically zero, so zero and the constant are
one baseline, not two.

**Verdict: the pre-registered threshold was not met for math, code or QA.** That is the
correct phrasing: not met is not the same as unpredictable, and the detail below matters
more than the verdict.

| Capability | Target | Best candidate | MAE against baseline | Gain [95% CI] | Interval width |
|---|---|---|---|---|---|
| code | double the budget | reuse-only | 0.0337 vs 0.0590, **43% lower** | +0.0253 [+0.0123, +0.0388] | 0.0265 |
| QA | double the budget | student-scaled | 0.5330 vs 0.5944, 10% lower | +0.0614 [+0.0126, +0.1105] | 0.0979 |
| math | double the budget | student-scaled | 0.0426 vs 0.0449, 5% lower | +0.0023 [-0.0090, +0.0135] | 0.0224 |
| code | more independent data | student-scaled | 0.0622 vs 0.0713, 13% lower | +0.0091 [+0.0001, +0.0169] | 0.0167 |
| math | more independent data | budget-only | no improvement | -0.0005 | 0.0012 |
| QA | more independent data | budget-only | no improvement | +0.0013 | 0.0020 |

For code, a one-parameter reuse form cuts the error of predicting a budget doubling by 43%,
with an interval well clear of zero, and misses the registered bar by 0.0012 nats. That is a
near miss on a strict rule, not an absence of signal, and it must not be reported as one.

The real negative is the second target. **Nothing registered predicts what changing the
independent data does, for any capability**, even though that intervention produces the
largest effects in the whole dataset. Leave-one-pool-seed-out gives the same picture, so this
is not a failure to cross students; the relation itself is missing.

## 3. The learning-rate pilot that fixed the protocol

Six short trajectories, 1B and 4B, one fixed pool, everything identical except the rate.

| Student | 5e-5 | 1e-4 | 2e-4 |
|---|---|---|---|
| 1B math / code | **+0.003 / +0.007** | +0.051 / +0.052 | +0.061 / +0.102 |
| 4B math / code | +0.058 / +0.022 | +0.102 / +0.072 | +0.142 / +0.154 |
| QA, 1B / 4B | -0.79 / -1.10 | -1.31 / -1.34 | -1.14 / -1.18 |

Math and code damage is monotone in the learning rate, and at the lowest rate the 1B student
is essentially unharmed while still gaining 0.79 on QA. So the *magnitude* of the damage
reported in earlier work is a property of the 1e-4 default as much as of distillation. The
size ordering survives matching, though: at every rate the 4B degrades about twice as much as
the 1B on math. The matrix therefore fixes one rate, 1e-4, for every student, and any claim
about math or code damage now names it.

## 4. Can pre-compression information predict response across models?

Measured first: the dense statistics for every development state, one forward pass per state
and capability, giving the reference-token logit margin, the trace of the loss Hessian in
logit space, and the logit variance. The third is new this round, because the displacement
experiment showed it is the quantity that appears in the shrinkage term.

Then a development study with three strictly separated input budgets, leave-one-source-out,
and a rule fixed in advance.

**The statistics help in 4 of 9 arm-and-capability pairs**, all through ridge: pruning math, pruning QA, grouped-quantization math, and grouped-quantization QA, each with an interval above zero. The completed panel has 459 primary response rows and 9 states in every arm.

**In 9 of 9 they still lose to the source-free median curve.** Clearing an incremental rule
inside a parametric family is not the same as beating what the paper already delivers, and
accuracy at naming the most fragile capability falls for pruning (58.3% to 41.7%) but rises for grouped quantization (40.7% to 49.4%).

The bar this round set is therefore not met, and I recommend holding the reserved confirmation
panel rather than spending four Pythia states and two external models on a likely null. Their
dense inputs are already extracted, which measures no response and keeps the option open.

## 5. Filling a hole in the development panel, on Loni

The input study reported that grouped quantization had no step-64k states at all. Loni turned
out to be a good fit for this: fifty two-GPU nodes, a work filesystem, and, unlike the other
cluster, outbound network from compute nodes, with transformers and peft at exactly the
versions the workstation runs. One job measured three states by nine configurations under the
existing protocol, and the results were transferred back and verified. The panel is now a
complete nine states by nine configurations, and the input study above was rerun on it. The
conclusion did not change.

The matrix deliberately stayed on one machine. Splitting it across two torch versions would
have blended the size effect with an environment difference, which is what the round exists to
separate.

## 6. Corrections I had to make, including two of my own errors

**Four readings were narrowed after review** (C63): the pure-shrinkage account was falsified
rather than the whole shrinkage-plus-noise hypothesis; "damage is carried by the variance" was
a median hiding cancellation, true per cell for math and code but not for QA; the displacement
decomposition is post-hoc, not a pre-compression prediction; and code had not been shown
unpredictable, only shown not to clear an investment gate.

**Two checks then vindicated the review** (C64): the old shrinkage coefficient moved 85% when
a constant was added to every logit, so the projection is now shift-invariant, and with it the
noise term is shown to carry 84 to 89 percent of the response for math and code. The noise
half of the hypothesis stands; only the shrinkage half fell.

**Two errors in the first matrix launch were mine** (C67). The trajectory flag counts
processed tokens, not supervised tokens, so the registered budget grid landed at a third of
its intended values; and with no stop point every run continued to the epoch limit, so larger
pools silently trained longer, which breaks the fixed-budget design. Projected cost of the
wrong version was 120 hours. I stopped it, corrected the specification, kept the five finished
cells as data rather than deleting them, and relaunched.

## 7. Where this leaves the round

**Established.** A large, monotone, size-dependent cost of reusing a small pool, measured
under control for the first time; a protocol dependence that reframes earlier damage numbers;
a mechanism account of compression damage as second-order noise in logit space, with a
boundary stated in configuration units.

**Not established.** That any of it clears the registered bar. The budget axis is close for
code, an error cut of 43% with an interval excluding zero, which is development evidence worth
confirming rather than a null; the independent-data axis is not close for anything; and the
pre-compression statistics do not beat a source-free curve.

**What I would do next, in order.** Widen the matrix along the axis that failed, since the
independent-data intervention is both the largest effect and the least predictable, before
committing anything to a frozen test. Keep the reserved confirmation panel unspent until a
candidate relation clears a development bar. And when a freeze does happen, freeze the
selection rule in the same act rather than after the responses are seen.

Nothing is running. The A100 and Loni are both free.
