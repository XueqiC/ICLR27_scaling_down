# The four-corner second difference: what it tests, what it can resolve, and what it cannot

Registered 2026-09-14, before any corner trajectory was trained and before the pool configurations
were finalised. Nothing below was chosen after seeing a corner measurement, because none exists.

## The statistic

For a fixed student, with corners at two budgets and two reuse levels,

    I = [delta(T2, E_lo) - delta(T2, E_hi)] - [delta(T1, E_lo) - delta(T1, E_hi)]

is exactly zero for any structure that adds a budget term to a reuse term, whatever curvature the
reuse term carries, because the budget term cancels in each bracket and the reuse term cancels
between them. A non-zero I falsifies additivity without fitting anything. This is the only fit-free
discriminator available, and it has never been computed in this project: the development set
contains no complete rectangle, as A2 established and I verified independently.

## Why pool B enters twice, and why that costs precision

Reuse is the budget over the pool size, so the four corners force a pool assignment: the reference
pool A serves the high-reuse corner at the low budget, the new pool B serves the low-reuse corner at
the low budget AND the high-reuse corner at the high budget, and the new pool C serves the low-reuse
corner at the high budget. In

    I = delta_3 - delta_4 - delta_1 + delta_2

the two pool-B corners are delta_1 and delta_4, and both enter with a minus sign. A constant offset
attached to a particular pool draw therefore contributes to I at double weight instead of
cancelling. That is a structural property of the design, not an implementation choice, and it is the
main reason the statistic is noisier than a single response measurement.

## Measured noise and the resulting power

The development set contains two pool seeds for every matrix configuration, which measures the
pool-draw noise directly. Taking the median absolute difference between the two seeds at matched
budgets, and propagating it through the four-corner combination above:

| Capability | Matched seed pairs | Noise on one response | Noise on I | Predicted additive-vs-interaction disagreement | Ratio |
|---|---:|---:|---:|---:|---:|
| code | 30 | 0.0127 | 0.0254 | 0.0144 | 0.57 |
| math | 30 | 0.0044 | 0.0088 | 0.0215 | 2.44 |
| QA | 36 | 0.0928 | 0.1856 | 0.8711 | 4.69 |

**Registered in advance: QA is the powered readout, math is marginal, and code is underpowered.**
The predicted disagreement for code is smaller than the noise on the statistic, so a null result on
code will be reported as an experiment that could not resolve the effect, never as evidence for
additivity. Doubling the seeds would only bring code to a ratio of about 0.8 and would cost the
entire trajectory budget, so it is not worth buying.

## What is being run

Four new development trajectories, the cap the round allows: two pools times two students, the 1B
and the 4B, one seed each. The two students give two independent evaluations of the same structural
question and preserve the size axis, which a single student with replicated seeds would lose while
still leaving code underpowered.

## The decision rule, fixed now

- **QA**: if the measured I is separated from zero by more than twice the noise figure above, and
  the two students agree in sign, additivity is rejected and the paper states that any law in this
  regime needs a reuse term whose strength depends on the budget.
- **QA**: if I is within that band on both students, additivity survives its one fit-free test, and
  the failure of the three fitted structures is attributed to the form of the individual terms or to
  irreducible variation rather than to a missing interaction.
- **math**: reported with the same rule and labelled marginal.
- **code**: reported with its interval and labelled underpowered, whatever the outcome.
- Disagreement between the two students is reported as a size-dependent interaction, not averaged
  away.

Residual budget and reuse mismatches at the corners are reported as measured, and any contamination
they imply is computed from the local budget slope between adjacent checkpoints on the same
trajectory, which is an empirical interpolation over about one optimizer step and is never the
structure under test.

---

## Amendment, same day, still before any corner was trained

The table above pooled the three QA distributions into one capability. Separating them changes which
measurement is the primary readout, so the choice is restated here rather than left to be made after
the fact.

| Distribution | Matched seed pairs behind the noise figure | Noise on I | Predicted disagreement | Ratio |
|---|---:|---:|---:|---:|
| QA probe, 2Wiki, measured in the trajectory | 30 | 0.1608 | 1.5546 | **9.67** |
| MATH-500 probe | 30 | 0.0088 | 0.0215 | 2.44 |
| MBPP probe | 30 | 0.0254 | 0.0144 | 0.57 |
| 2Wiki, fresh sample | 2 | 0.3823 | 0.8711 | 2.28 |
| TriviaQA | 2 | 0.1440 | 0.2321 | 1.61 |
| MuSiQue | 2 | 0.8360 | 0.3658 | 0.44 |

**The primary readout is the 2Wiki QA probe the trainer already measures at every checkpoint**, on a
ratio of about ten to one, with its noise estimated from thirty matched seed pairs. MATH-500 stays
marginal and MBPP stays underpowered, exactly as registered above.

The three fresh-sample distributions are demoted to a secondary check, for a reason that is about
evidence rather than preference: they were only ever measured at trajectory endpoints, so their
noise figures rest on two matched seed pairs each and are not trustworthy enough to carry a
falsification. They will still be measured at the corner checkpoints and reported, labelled as
secondary with the weakness of their noise estimate stated.

One practical consequence: the primary statistic needs no evaluation beyond what training already
writes, including at the existing corner, which has training-probe measurements but no fresh-sample
measurements at that checkpoint.
