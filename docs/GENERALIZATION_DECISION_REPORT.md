# Stage C: the conditional generalization packages, and the condition that did not arrive

Stage C held three items, each explicitly conditional on the independent confirmation passing: a
held-out larger student at twelve billion parameters, a second model family, and a data-requirement
relation derived from a confirmed response model. The confirmation did not run, so none of them did.
This report states what each would have tested and what would have to be true first, so that the
work is specified rather than merely postponed.

## The larger student

Four trajectories at twelve billion parameters, two pool sizes by two seeds, testing whether the
size term predicts a student outside the three development sizes rather than interpolating between
them. Two things must be said precisely whenever this is described. Twelve billion has historical
experiments in this project, so the accurate phrasing is a large-size held-out prediction under this
round's protocol and configurations, not an unseen model. And the four-billion student is now a
development student and can no longer be called an unseen size.

Precondition: a response model that has survived independent confirmation on new pool seeds. Without
it, a twelve-billion run measures a number without testing a prediction.

## The second family

A cross-family panel testing whether the relations are properties of the compression and training
process or of one model family. Precondition is the same, with one addition: the development and
confirmation protocol must be reproducible in the second family without changing the learning rate
schedule or the supervised-token accounting, since a family difference blended with a protocol
difference answers nothing.

## The data requirement

The delivery the round was aiming at:

    D_U_min(N_S, T, tau) = inf { D_U : predicted response(N_S, T, D_U) <= tau }

answering how much independent supervised data a given student needs, at a given training budget, to
keep the increase in a conditional loss below a threshold. The forward check was to be two data
budgets either side of the predicted requirement, testing whether the constraint holds, rather than
reporting an aggregate error.

Precondition: a confirmed response model. Deriving a requirement from a relation that loses to a
low-order response surface on every hold-out would produce a number with no support, and the plan
was explicit that if the data does not support a clear boundary, the continuous response and its
uncertainty are reported instead of a manufactured threshold law.

## What the round can say about generalization without these packages

Two statements survive from earlier work in this round and remain the honest scope claims. The cost
of over-reusing a small pool transfers across all three QA distributions measured and grows with
student size. The gain from distillation does not transfer: it is a 2Wiki statement, with MuSiQue
flattening near zero and TriviaQA positive at every reuse level for every student. A data
requirement phrased as keeping a loss increase below a threshold is therefore supportable in form
across distributions, while a claim that distillation improves question answering is not.
