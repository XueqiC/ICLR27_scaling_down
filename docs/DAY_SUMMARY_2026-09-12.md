# 2026-09-12: what was done, how, and what came out

Two experiments ran today, both designed as gates rather than as confirmations, and neither
hypothesis cleared its gate. That is the useful outcome, and it is narrower than it first
looks: one rules out a *pure shrinkage* account of compression, the other withholds funding
from a large training matrix without showing the underlying relation is unpredictable. A
review of the first version of this report corrected both readings, and the corrections are
folded in below and recorded as ledger C63. The rest of the day went to the manuscript round the advisor asked for, and to
turning the code repository into something a reviewer can actually run behind the anonymous
link.

---

## 1. Manuscript round v10: separating three things the paper had been mixing

**The problem.** Reading the compiled PDF, the advisor found that the paper used one label for
three different objects: the candidate that was frozen before a test, the rule finally
delivered for that arm, and the rule locked for the selection experiment. The main table and
Figure 2 scored whichever of them was convenient.

**What I did.** The main table and Figure 2 now carry three explicit columns per test: the
frozen candidate registered before measurement, the strongest of *all* predictors frozen in
that round, and the delivered rule with the time it was fixed. The quantization rows revert
to the frozen development selection rather than the retrospectively chosen interpolation. The
older U=375 distillation context moved to the appendix. Section 5 now cites the locked rule
as coded, with a table of where it departs from the delivered predictors.

**What fell out of it.** Ranking the distillation confirmation against *every* form frozen
alongside it, rather than against the single baseline designated during development, shows
that the selected forms are the hindsight best for no student and no capability. The
registered gain over the designated baseline stands, and both facts are now in the abstract,
Section 3.4, Section 4.1 and Appendix D. Ledger C59.

The paper holds at nine pages of main text, references starting on page ten with no spill,
and no undefined references.

---

## 2. V88: does the logit displacement explain the damage?

**The idea.** The advisor's next-round plan proposed two dense descriptors as extra regression
inputs, on the hypothesis that pruning and quantization act like shrinking the logits plus
isotropic noise. Rather than feed them to another regression, I tested the hypothesis
directly, because the underlying identity is exact and needs no fitted coefficients:

```
dL = (E_p[r] - r_y) + 0.5*Var_p(r) + O(||r||^3),   r = z_compressed - z_dense
```

**Registration first.** Cells, damage regimes, pass rules and the stop rule were committed
before any measurement. While implementing it, the delegated implementer caught that my
registered shrinkage predictor dropped the shrinkage's own quadratic term. The objection was
right, the complete decomposition was derived and verified to 1.5e-14 on fixtures, and the
registration was amended before the first cell was measured, with the original text left
visible.

**Method.** Five weight-identity-distinct Pythia states, seven configurations each, three
capabilities: 105 non-zero cells. Dense and compressed models held in memory together and
scored in the same run, so no comparison crosses hardware; bfloat16 weights, float32
vocabulary reductions; probe identity matching the frozen protocol. Ten GPU-minutes against a
registered two-hour cap, on the released GPU selected by UUID.

**Result, part one: the local account passes.**

| Regime | Cells | Median relative error | Sign agreement |
|---|---:|---:|---:|
| mild, \|dL\| < 0.1 | 41 | 2.3% | 97.6% |
| moderate, 0.1 to 0.5 | 31 | 15.3% | 93.5% |
| severe, \|dL\| > 0.5 | 33 | 38.8% | 100% |

The registered threshold was 20% and 90% in the mild regime. Accuracy degrades monotonically
with displacement size, which converts into a boundary stated in the units the paper already
uses: pruning at density 0.9 gives 0.9% error, 0.8 gives 6.9%, 0.7 gives 18.4%, 0.6 gives
45.1%; five-bit quantization gives 1.5%, four-bit 8.3%, three-bit 32.4%. The paper reports
these ranges empirically; this is a mechanism for them.

**Result, part two: the pure-shrinkage account fails.** Summarising the whole displacement by
one scalar gives 98% median relative error, 42 times the T1 error against a 1.5 threshold,
with the sign wrong on a third of the cells. This rules out pure shrinkage. It does not rule
out shrinkage plus noise, because the amended predictor carries only the shrinkage term and
the noise reading was deferred by the same amendment.

**Which term carries the damage, corrected.** A median across cells hides cancellation: the
two terms have opposite signs in 81 of 105 cells. Per cell the variance term is the larger one
for math in 35 of 35 and code in 35 of 35, but for QA in only 16 of 35. Seventeen cells have a
negative measured change, which a non-negative variance term cannot produce; in all seventeen
the first-order term is negative and outweighs it. Two boundaries also need stating: the
displacement comes from the compressed model, so this is a post-hoc decomposition rather than
a prediction from pre-compression inputs, and the shrinkage coefficient is an uncentred
projection that a constant shift of all logits would change even though the loss does not.

**Engineering note.** The delivered reducer ran one forward pass per scored token, 12,643 of
them for one capability. Replacing that with one causal forward per sample plus a batched
reduction, verified against the old path to 5.9e-4, is why the sweep took ten minutes.

Ledger C60; details in `results/v88-displacement/FINDINGS.md`.

---

## 3. V89: is a training intervention predictable from what we already have?

**The question that decides a lot of GPU time.** The next-round plan proposes an
eighteen-trajectory crossed distillation matrix. Before funding it, the honest test is whether
the *effect of an intervention* is predictable on the trajectories that already exist.

**Method.** 89 run directories inspected, 41 usable, 48 dropped with reasons recorded. Two
targets: the change in response when the budget roughly doubles inside a trajectory, 134
pairs; and the change when the pool changes at a matched budget, 1163 pairs. Seven candidates
fixed before fitting, each fitted to the response and then differenced, so the intervention
effect is never fitted directly. Leave-one-student-size-out and leave-one-pool-seed-out, with
a 5000-draw bootstrap clustered on whole trajectories. The reading rule was fixed in advance:
signal counts only if the same candidate beats both zero and the per-capability constant on
both targets, by more than the interval width.

**Verdict, corrected.** QA on 2Wiki clears. Math does not. **Code did not clear the gate, but
it is not shown to be unpredictable**: the same reuse-only form cuts the error from 0.0687 to
0.0423 on the budget target and from 0.1195 to 0.0667 on the pool target, improvements of 38%
and 44% with intervals above zero. It failed only because on the first target the improvement,
0.0264, is smaller than the interval width, 0.0297. Two further qualifications belong here:
the two baselines are really one, since a constant fitted to the response and then differenced
is identically zero; and none of the seven candidates carries a student-state input, so what
was tested is whether a relation in budget, pool and reuse alone transfers across sizes.

**Identifiability.** The saturation time constant is not identified under the size holdout,
where every profile interval touches a boundary of the observed token range. Some 270M code
pool-seed folds do give interior intervals, so the earlier blanket statement was wrong. Either
way no asymptotic floor and no optimal-budget claim is made.

**Consequence.** The eighteen-trajectory matrix stays unfunded on this evidence. Code and QA
remain the primary distillation hypotheses for the next round, math keeps being measured
because evaluating all three capabilities after one training run costs little, and the
historical pool-change target cannot settle a strictly controlled matrix anyway, since it
mixes schedules, run generations and budgets matched only to about ten percent. Ledger C62,
corrected by C63.

---

## 4. Repository work behind the anonymous link

The anonymous review service mirrors the code repository verbatim, so the repository itself,
not a separate snapshot, is what reviewers read. Three things were wrong with that.

**Identity.** The author's name appeared in 29 files, home and cluster paths in 20, plus the
institution, the HPC system, and both repository URLs. All are rewritten. A local
model-eligibility policy that named the institution and a country of origin is now a neutral
restricted-origin policy with the environment variable renamed; behaviour is unchanged and
tests pass. Author-side tooling, advisor correspondence and fifteen internal planning notes,
including the unpublished next-round plan, are no longer tracked there. An independent
deny-list scanner covering contents, gzipped contents, file names and git metadata reports no
hits across 1301 files, and a structural validator parses every Python, shell and JSON file so
that a scrub can never ship broken code. It caught two real breakages.

**Runnability.** The published layout put the code one directory down, so every script
resolved its root incorrectly and found no inputs; the inputs themselves live under different
names in the mirror, with the large summaries gzipped. A bootstrap script now reconstructs the
input tree, creates the directories generators write into, and provides the compatibility
paths that self-mirroring scripts expect. Measured on a fresh clone, seven of nine named
generators regenerate their table. The two that do not are stopped by the project's own
integrity check, which is doing its job, and the README says exactly that.

**Provenance under anonymisation.** A published frozen file cannot keep the original bytes,
so its own digest check must fail. Rather than weaken the checks, a recorded map pairs each
published file with the digest the frozen record names and gives the reason, and the
comparison is widened by exactly that set and nothing else. Stale digests from earlier freezes
are deliberately not paired, so a check that fails locally still fails on the published copy.

The abstract now ends with the anonymous repository link.

---

## 5. Housekeeping worth knowing

**Test suite:** 702 passing, 6 failing, 5 skipped. Three failures fixed today came from the
threading layer torch sets in the parent leaking into spawned children, and one from a
case-sensitive assertion. Of the six left, three are the pre-existing failures recorded on
2026-09-08, one wants a measurement that was never taken, and two correctly report that the
shared evaluation path changed when per-sample records were added.

**New infrastructure, all CPU-tested:** per-sample evaluation records, which unblock the
item-level intervals the paper currently cannot compute; the dense descriptors with
finite-difference checks; an asymmetric grouped-quantization mode with the symmetric default
proved byte-identical on saved cases.

**A fact worth carrying:** the project's loss aggregation is pooled tokens, not an equal
per-sample mean. Anything new must aggregate the same way.

---

## 6. Where this leaves the next round

Package A is not retired; its pure-shrinkage form is. The next checks are a shift-invariant,
centred projection, an explicit accounting of the residual first-order and noise terms, and
only then a judgement on whether the noise part of the hypothesis still earns measurement.
For math and code the variance term is the larger one in every cell, which makes it the
candidate quantity to try to predict from configuration and pre-compression information on
states that entered no fit, under its own registration.

Package B keeps code and QA as primary hypotheses and keeps measuring math; what it loses is
the large crossed matrix, pending a same-protocol recomputation and a low-degree-of-freedom
student-conditioned control. The short learning-rate pilot on 1B and 4B stays worth running,
since it separates an optimisation artefact from a size effect. Package C, the three-way
selection panel, is untouched by today's results and still rests on the known weakness that
the current math and code margin comes almost entirely from one state with historical
distillation candidates.

No GPU work is running. Nothing is pending on my side.
