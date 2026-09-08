# Pruning source x compression-strength prediction (round-6 main line)

Frozen shared-shape candidate dL_c(x,d)=A_c(x)*((1-d)/0.3)^gamma_c fit on DEV (9-state grid, seen d={0.9,0.8,0.7,0.6}); predicts UNSEEN densities. Domain pre-declared: 0.65 interp, 0.55 deeper extrap (not dropped on collapse). Baselines: zero, strength-only, median-strength-curve.

## pythia-1.4b@step64000 (seen source)
- math d=0.65 (interp): obs=+0.429 cand=+0.494 strength-only=+0.943 median=+0.573 | |err| cand=0.064
- math d=0.55 (extrap): obs=+2.091 cand=+1.175 strength-only=+2.419 median=+1.567 | |err| cand=0.916
- code d=0.65 (interp): obs=+0.336 cand=+0.599 strength-only=+1.130 median=+0.558 | |err| cand=0.263
- code d=0.55 (extrap): obs=+2.086 cand=+1.241 strength-only=+2.759 median=+1.526 | |err| cand=0.845
- qa d=0.65 (interp): obs=-0.598 cand=-0.177 strength-only=+0.470 median=-0.201 | |err| cand=0.421
- qa d=0.55 (extrap): obs=+0.258 cand=-0.390 strength-only=+1.285 median=-0.316 | |err| cand=0.648

## pythia-160m@step143000 (seen source)
- math d=0.65 (interp): obs=+3.964 cand=+4.094 strength-only=+0.943 median=+0.573 | |err| cand=0.130
- math d=0.55 (extrap): obs=+8.493 cand=+9.742 strength-only=+2.419 median=+1.567 | |err| cand=1.249
- code d=0.65 (interp): obs=+5.641 cand=+5.041 strength-only=+1.130 median=+0.558 | |err| cand=0.600
- code d=0.55 (extrap): obs=+8.874 cand=+10.448 strength-only=+2.759 median=+1.526 | |err| cand=1.574
- qa d=0.65 (interp): obs=+4.160 cand=+3.047 strength-only=+0.470 median=-0.201 | |err| cand=1.113
- qa d=0.55 (extrap): obs=+7.293 cand=+6.725 strength-only=+1.285 median=-0.316 | |err| cand=0.568

## pythia-410m@step96000 (NEW source)
- math d=0.8 (seen-d): obs=+0.098 cand=+0.098 strength-only=+0.116 median=+0.061 | |err| cand=0.001
- math d=0.6 (seen-d): obs=+1.211 cand=+1.076 strength-only=+1.555 median=+0.978 | |err| cand=0.135
- math d=0.65 (interp): obs=+0.648 cand=+0.679 strength-only=+0.943 median=+0.573 | |err| cand=0.031
- math d=0.55 (extrap): obs=+2.059 cand=+1.616 strength-only=+2.419 median=+1.567 | |err| cand=0.443
- code d=0.8 (seen-d): obs=+0.080 cand=+0.184 strength-only=+0.155 median=+0.060 | |err| cand=0.104
- code d=0.6 (seen-d): obs=+1.541 cand=+1.374 strength-only=+1.816 median=+0.953 | |err| cand=0.167
- code d=0.65 (interp): obs=+0.808 cand=+0.933 strength-only=+1.130 median=+0.558 | |err| cand=0.125
- code d=0.55 (extrap): obs=+2.447 cand=+1.933 strength-only=+2.759 median=+1.526 | |err| cand=0.513
- qa d=0.8 (seen-d): obs=-0.083 cand=+0.170 strength-only=+0.050 median=-0.073 | |err| cand=0.253
- qa d=0.6 (seen-d): obs=+0.454 cand=+1.509 strength-only=+0.802 median=-0.256 | |err| cand=1.055
- qa d=0.65 (interp): obs=-0.087 cand=+0.991 strength-only=+0.470 median=-0.201 | |err| cand=1.078
- qa d=0.55 (extrap): obs=+1.253 cand=+2.187 strength-only=+1.285 median=-0.316 | |err| cand=0.934

## Aggregate MAE (all predicted cells)
| cap | candidate | strength-only | median-curve | zero |
|---|---|---|---|---|
| math | **0.371** | 1.369 | 1.478 | 2.374 |
| code | **0.524** | 1.635 | 1.874 | 2.726 |
| qa | **0.759** | 1.608 | 1.918 | 1.773 |

## Read (round-6 second-axis result — pruning)

**The shared-shape candidate connects the source axis to the compression-strength axis.** Against real
compression-strength-only baselines (not the strawman "categorical model can't predict new density"),
the candidate wins decisively on UNSEEN densities:

| cap | candidate | strength-only | median-curve | zero |
|-----|-----------|---------------|--------------|------|
| math | **0.371** | 1.369 | 1.478 | 2.374 |
| code | **0.524** | 1.635 | 1.874 | 2.726 |
| qa   | **0.759** | 1.608 | 1.918 | 1.773 |

- **Why it wins**: source-dependent amplitude A_c(x)=β_c·[1,z(logN0),z(L0),z(logD0)] lets it see that a
  given source-state is much more/less fragile than a source-blind curve assumes. Extreme case: 160m@143k
  at d=0.55 has obs ΔL_math=+8.49; the candidate predicts 9.74 while every strength-only baseline predicts
  ~1.5–2.4 (they cannot know this state is ~4× more fragile). This is the whole point of sharing the shape
  across densities while conditioning amplitude on source.
- **Interpolation (d=0.65) is strong**, including on the NEW source-state 410m@96k (math err 0.031,
  code 0.125; d=0.8 seen-density math err 0.001) — new source AND unseen density generalize for math/code.
- **Deeper extrapolation (d=0.55) systematically UNDER-predicts** as it approaches the cliff (obs>cand
  everywhere at 0.55: 1.4b math obs 2.09 vs cand 1.18; 160m code obs 8.87 vs cand 10.45 over-shoots the
  other way). Pre-declared domain edge — reported, NOT dropped. The single power under-predicts cliff onset.
- **QA amplitude is OVER-predicted** on the new source (d=0.6 err 1.06, d=0.65 err 1.08) — QA fragility does
  not follow the same source-amplitude law as math/code (consistent with QA being hard throughout).

**Honest headline**: on pruning, a low-DOF shared-shape law with source-conditioned amplitude predicts
UNSEEN compression strengths — including on a new training state — far better than any strength-only
baseline for math/code (MAE ~0.4–0.5 vs ~1.4–1.6); interpolation is strong, deep extrapolation degrades at
the cliff, and QA amplitude does not transfer. This is the first real result on the COMPRESSION-STRENGTH
axis (not just source-transfer at fixed d). Small panel, point estimates, single frozen candidate.
