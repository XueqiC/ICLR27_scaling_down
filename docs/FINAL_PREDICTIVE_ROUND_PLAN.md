# Final predictive round: data splits, candidates, budget, and the conditions for each stage

Written at the start of execution, before Stage A produced anything. It fixes what counts as
development data, which function family may be opened, what the primary targets are, and what
has to be true before any confirmation training starts.

## What this round decides

Whether a small set of pre-compression observables predicts the conditional loss of unseen
compression and distillation configurations well enough to support a resource decision that can
be checked. Three questions, in order: is there a parsimonious relation that predicts an
intervention rather than fitting a curve after the fact; does it carry to new data
configurations, budgets, larger students or another family; and does a decision built on it stop
depending on one historical distillation candidate.

## Data, as actually found

| Set | Contents | Role |
|---|---|---|
| Controlled matrix | 18 trajectories: 270M, 1B, 4B x pools of 66, 198, 594 traces per domain x two pool seeds, four budget checkpoints, one fixed LoRA protocol at learning rate 1e-4 | core development |
| Critical region | 4 trajectories: 1B and 4B x 132 traces per domain x two seeds, fourteen checkpoints | core development, already unblinded, therefore not usable as confirmation |
| Historical | 88 heterogeneous trajectories under mixed protocols and schedules | external diagnostic only, written to a separate file so a fit cannot mix them in silently |
| Archived | 5 trajectories from the discarded first launch, wrong budget units and no stop condition | excluded by assertion, kept on disk |

Already unblinded and therefore unavailable as a confirmation configuration: the 132-trace pool,
the 4B student as a development student, and every budget checkpoint in the tables above.

## Endpoint and budget

The endpoint is the pooled supervised-token cross-entropy on a fixed evaluation distribution,
`L_{c,j}`, with the response `delta = L(S_KD) - L(S_0)` measured against each trajectory's own
initial student. The two interventions are the budget change `I_T` and the independent-data
change `I_U` at matched budget. One parameter set must produce the response and both
interventions; three separately fitted curves are not a joint relation.

Inputs stay at the K0 budget: metadata, the operating configuration, and a dense loss anchor
whose measurement cost is reported. No post-compression loss, logits, gradients or post-hoc
rescaling. Any K1 calibration appears as one auxiliary column, never as the headline.

## The one function family this round may open

With `u = log(1 + T/T_ref)`, `v = log(1 + E)`, `T_ref = 100000` supervised tokens fixed, and `z`
the training-fold-standardised `log N_S`, with the student's own initial loss as the one
pre-declared alternative descriptor:

- `F_log = (a + a'z) u + (b + b'z) v`, four coefficients per capability, the family already
  falsified on its own flatness implication, kept as the reference point.
- `F_curv = (a + a'z) u + (b + b'z) h_p(E)`, five parameters, with
  `h_p(E) = ((1+E)^p - 1)/p` for non-zero `p` and `log(1+E)` at `p = 0`, implemented as
  `expm1(p*log1p(E))/p` with a stable limit near zero. `p = 0` recovers the old form exactly and
  `p = 1` is a linear reuse term. Coefficients may take either sign.
- `F_int = (a + a'z) u + (b + b'z) v + k uv`, five coefficients, reusing the earlier run when the
  inputs, protocol and folds are identical rather than renaming it as new evidence.

Nothing else. No free curvature plus interaction plus threshold plus floor. If confirmation
fails, no further family is invented in this round.

## Baselines that the candidate must beat

Zero response and zero intervention; a constant response on positive budgets, with its
differenced form stated as identical to zero intervention; budget-only and reuse-only forms; the
non-zero mean intervention effect estimated inside the training fold; a same-input low-order
response surface on a fixed basis with the same fold-internal regularisation; and empirical
interpolation where a support region genuinely exists. Parameter counts, effective degrees of
freedom, fitting units, information used and calibration cost are reported for each.

## Hold-outs

Four grouped splits, run separately: leave one entire data rung out; leave one entire student
out; leave the largest training budget out and fit only on earlier checkpoints; leave a shared
pool seed out across all students and nested pools. Random cells and neighbouring checkpoints do
not substitute. Candidate selection happens inside the training folds; the outer error is
reported separately.

## Primary targets, fixed now

- `code x I_T`, because a budget-response signal already exists in development and the question
  is whether it reproduces independently.
- `QA(2Wiki) x I_U`, because it carries the largest data effect and the question is whether any
  response shape captures it.

Math and the external QA distributions are reported in full as secondary. These are fixed before
any Stage A number is read.

## Conditions for starting confirmation

All three must hold: at least one primary target beats the strongest baseline on a meaningful
hold-out, with the gain not driven by a single trajectory; the candidate predicts stably inside
the intended confirmation region rather than relying on a parameter exploding at the boundary;
and the planned confirmation configurations actually separate candidate from baseline instead of
both predicting near zero. Effect sizes, intervals and risks are published whether or not the
old "gain exceeds the whole interval width" rule is met, since that rule is a historical verdict
rather than the launch criterion here.

## Budget

The round default is 72 physical GPU-hours, counted per device actually occupied, with
concurrent lanes on one card counted once. Spent so far in this round, by that accounting:

| Item | GPU-h |
|---|---|
| controlled matrix, three lanes on one A100 | 10.0 |
| critical region, two lanes on one A100 | 9.0 |
| learning-rate pilot | 1.5 |
| displacement lanes across three cards | 3.0 |
| dense statistics | 0.5 |
| scope evaluation | 2.5 |
| cluster jobs | 1.5 |
| **total** | **28.0** |

That leaves 44 hours. A confirmation package of 16 trajectories, two students by two data rungs
by four fresh pool seeds, costs about 38 hours at the current per-trajectory rates, roughly 1.4
hours for a 1B run and 3.3 for a 4B run to 200k supervised tokens. That is affordable only if
Stage A stays on CPU and the larger-student and second-family packages are treated as
conditional on what remains. The 24-trajectory version at six seeds would cost about 56 hours and
does not fit; the reduction to four seeds is therefore declared here, before freezing, as a
precision limit rather than chosen later.

## Stage order and stopping

Stage A is CPU work: the canonical development table, then the three-structure comparison with
the baselines and hold-outs above, then a decision table. Stage B is one independent confirmation
on new pool seeds, with predictions, selection and candidate sets frozen before any response is
measured. Stage C is conditional: the larger student and the second family, in that priority,
and only if the confirmation passes. If every new candidate still loses to a strong simple
baseline, the analytic search stops, one independent confirmation of the existing code candidate
may still run, and the paper closes as a systematic investigation with the strength of the new
relations stated accurately.

## Reporting

At the end of every stage, and at least every four hours while a stage is running, using the
template the instruction specifies: the scientific question, the inputs and protocol, the
pre-fixed candidate and baselines, completion status and resource use, the result table, what was
supported and what was excluded, the paper and ledger updates, and the next step with its
remaining budget and stopping condition.
