# Correction to the Stage B cost estimate, and the fresh pool seeds

Written 2026-09-14, while the Stage A2 structure comparison was still running and before any of
its output was read. The timing matters: nothing here is a reaction to a fitted result.

## The estimate in the round plan was wrong

The plan priced a confirmation trajectory at roughly 1.4 GPU-hours for a 1B student and 3.3 for a
4B one, put a sixteen-trajectory package at about 38 hours, declared a twenty-four-trajectory
package at six seeds unaffordable against the 44 hours remaining, and fixed the seed count at four
on that basis.

The recorded wall time of the twenty-two core development trajectories says otherwise. Every
figure below is `wall_time_seconds` from the run's own training log, not a reconstruction.

| Student | Trajectories | Hours per trajectory, matrix | Hours per trajectory, dense critical region |
|---|---:|---:|---:|
| 270M | 6 | 1.29 - 1.51 | not run |
| 1B | 8 | 1.33 - 1.58 | 2.00 - 2.04 |
| 4B | 8 | 1.07 - 1.66 | 2.08 - 2.12 |

The 4B student costs about the same as the 1B under this LoRA protocol, not 2.4 times as much.
The whole development set is 35.5 lane-hours, which is why three concurrent lanes on one A100
delivered it in the 10 physical GPU-hours already charged to the round.

## What that does to the package sizes

Counting the way the round counts, per device occupied with concurrent lanes on one card counted
once, and assuming five checkpoints rather than fourteen:

| Package | Lane-hours | Physical GPU-h at three lanes |
|---|---:|---:|
| 16 trajectories, four fresh seeds | about 27 | about 9 |
| 24 trajectories, six fresh seeds | about 40 | about 14 |

Both fit in the 44 hours remaining. The stated reason for limiting the confirmation to four seeds
does not survive its own arithmetic.

## What I am doing about it

The seed count is a precision limit, not a bar the candidate has to clear, and raising it before
any response is measured cannot bias the test in either direction. The decision is therefore
deferred to the freeze itself and will be recorded there with this correction cited, rather than
being quietly enlarged now or defended on a cost figure known to be false.

## Fresh pool seeds

Every data seed that appears in any recorded run: 0, 1, 2, 3, 11, 12, 13, 21, 22, 23, 31, 32, 33,
34, 35, 36, 41, 42, 51, 52, 91. The confirmation package draws its pools with seeds **60, 61, 62,
63**, extended to 64 and 65 if the six-seed version is chosen. A fresh seed means a fresh draw
from the same trace corpus; it does not mean a pool disjoint from every earlier pool, and no claim
in the confirmation depends on disjointness.
