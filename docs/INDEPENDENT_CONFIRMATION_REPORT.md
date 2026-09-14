# Stage B: the confirmation package did not launch, and why that is the result rather than a gap

The round reserved its largest single expense for one independent confirmation: sixteen
trajectories, two students by two data rungs by four fresh pool seeds, with predictions, the
candidate set and the selection rule all frozen before any response was measured. It was never
started. This report records the decision and the evidence behind it, because an unspent package is
a finding about the candidate relations, not an omission.

## The three conditions, and which failed

The plan fixed three conditions, all of which had to hold before any confirmation training began.

**One: at least one primary target beats the strongest baseline on a meaningful hold-out, with the
gain not driven by a single trajectory. FAILED.** Across three capabilities, three targets and four
grouped hold-outs, no structure beat the strongest of the six baselines anywhere. Both pre-declared
primary targets lost: code under budget doubling on all four hold-outs, and QA on the fresh 2Wiki
sample under a data change on three of four folds with intervals clear of zero, the fourth
indistinguishable from zero.

**Two: the candidate predicts stably inside the intended confirmation region rather than relying on
a parameter exploding at a boundary. NOT REACHED.** The curvature exponent is interior for every
capability on the full development fit, which is the good case, but fold-to-fold ranges are wide and
one fold in twelve reaches a boundary for three capabilities. This condition was never the binding
one, because the first had already failed.

**Three: the planned confirmation configurations separate candidate from baseline instead of both
predicting near zero. NOT EVALUATED.** There was no candidate left to separate.

## What was spent, and what remains

| Item | GPU-hours |
|---|---:|
| Development work before this round | 28.0 |
| Stage A, entirely CPU | 0.0 |
| Four corner trajectories authorised on the structural finding | about 3 |
| **Remaining of the round's 72** | **about 41** |

The confirmation package would have cost about nine physical GPU-hours, not the thirty-eight the
plan estimated, as the measured wall times recorded separately show. Affordability was therefore
never the reason it went unspent. It went unspent because nothing had earned it.

## The one result that is not nothing

Against the baseline selected inside the training folds rather than the strongest-of-all comparator,
the additive log form beats the fitted mean-effect baseline for code under a data change on every
held-out student, with all three intervals above zero. That is a replicated development signal on a
secondary target, and it is the only candidate in this round that would be worth confirming if the
round continued. It is recorded here so that a future confirmation has a pre-existing, dated
statement of what it would be testing, rather than a target chosen after the fact.

## What a future confirmation would have to do differently

The development set cannot separate an additive law from an interacting one, because it contains no
complete rectangle in budget-by-reuse space. Any future confirmation design should carry that
constraint in its construction: pools and budgets chosen so that at least one rectangle exists by
design, rather than a grid that happens to give each trajectory its own budget.
