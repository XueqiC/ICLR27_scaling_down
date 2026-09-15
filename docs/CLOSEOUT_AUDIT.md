# A7 closeout audit

Status: **COMPLETE**. CPU only; existing artifacts only; no training, new measurement, fitting, resampling, or new candidate.

All losses, contrasts, errors, and registered noise are in native-token nats. Displayed decimals are rounded; summary.json retains full precision.

## 1. The gate baseline

Frozen pointwise error intervals; folds overlap and are not independent replications. QA I_U is the stored 1% tolerance proxy, not an exact fixed-budget contrast. When inner folds were insufficient, A2 used its registered fallback settings. Outer-oracle intervals do not cover the post-hoc baseline selection.

### Development-stage baseline

| Primary target | Split | Held | Primary F | INNER baseline | MAE F | MAE baseline | Gain [frozen 95% interval] | Excludes 0? | Positive and clear? |
|---|---|---|---|---|---|---|---|---|---|
| MBPP I_T | data_rung | 66 | F_curv | mean_effect | 0.065018 | 0.062378 | -0.002640 [-0.004233, -0.001054] | yes | no |
| MBPP I_T | data_rung | 132 | F_log | reuse_only | 0.023339 | 0.020340 | -0.002999 [-0.004077, -0.001097] | yes | no |
| MBPP I_T | data_rung | 198 | F_int | surface | 0.028744 | 0.034769 | 0.006025 [0.003962, 0.007231] | yes | yes |
| MBPP I_T | data_rung | 594 | F_log | reuse_only | 0.036985 | 0.032618 | -0.004367 [-0.007480, -0.001547] | yes | no |
| MBPP I_T | student | gemma3-1b | F_log | mean_effect | 0.043361 | 0.037078 | -0.006283 [-0.011983, 0.002571] | no | no |
| MBPP I_T | student | gemma3-270m | F_log | mean_effect | 0.031690 | 0.025288 | -0.006403 [-0.012263, 0.000142] | no | no |
| MBPP I_T | student | gemma3-4b | F_log | mean_effect | 0.032049 | 0.033367 | 0.001318 [-0.002899, 0.006456] | no | no |
| MBPP I_T | largest_budget | T>=150000 | F_log | mean_effect | 0.045698 | 0.039847 | -0.005850 [-0.013540, 0.003310] | no | no |
| MBPP I_T | pool_seed | 41 | F_int | mean_effect | 0.034413 | 0.030749 | -0.003663 [-0.010354, 0.005024] | no | no |
| MBPP I_T | pool_seed | 42 | F_int | mean_effect | 0.040232 | 0.041235 | 0.001003 [-0.003644, 0.006668] | no | no |
| MBPP I_T | pool_seed | 51 | F_curv | surface | 0.029475 | 0.022646 | -0.006828 [-0.008032, -0.005634] | yes | no |
| MBPP I_T | pool_seed | 52 | F_curv | reuse_only | 0.031282 | 0.021592 | -0.009690 [-0.012657, -0.006747] | yes | no |
| fresh 2wiki_new I_U | data_rung | 66 | F_curv | zero | 11.123465 | 3.374658 | -7.748807 [-12.492520, -1.110775] | yes | no |
| fresh 2wiki_new I_U | data_rung | 132 | F_log | surface | 1.116584 | 0.868818 | -0.247766 [-0.465527, -0.026220] | yes | no |
| fresh 2wiki_new I_U | data_rung | 198 | F_log | mean_effect | 1.447093 | 1.760991 | 0.313898 [-0.421751, 1.084052] | no | no |
| fresh 2wiki_new I_U | data_rung | 594 | F_int | surface | 1.365336 | 1.185901 | -0.179435 [-0.324912, -0.002791] | yes | no |
| fresh 2wiki_new I_U | student | gemma3-1b | F_log | mean_effect | 1.083579 | 1.328124 | 0.244545 [-0.359963, 0.896271] | no | no |
| fresh 2wiki_new I_U | student | gemma3-270m | F_log | mean_effect | 1.631196 | 1.798219 | 0.167024 [-0.576797, 1.456098] | no | no |
| fresh 2wiki_new I_U | student | gemma3-4b | F_log | mean_effect | 1.983881 | 2.456583 | 0.472702 [-0.221918, 1.419711] | no | no |
| fresh 2wiki_new I_U | largest_budget | T>=150000 | NA | NA | NA | NA | NA [NA] | NA | unscorable |
| fresh 2wiki_new I_U | pool_seed | 41 | F_log | surface | 1.509319 | 1.191950 | -0.317369 [-0.737907, 0.272595] | no | no |
| fresh 2wiki_new I_U | pool_seed | 42 | F_log | mean_effect | 1.681465 | 1.620374 | -0.061091 [-0.612046, 0.553478] | no | no |
| fresh 2wiki_new I_U | pool_seed | 51 | F_curv | surface | 0.820272 | 0.733709 | -0.086563 [-0.412246, 0.121798] | no | no |
| fresh 2wiki_new I_U | pool_seed | 52 | F_curv | surface | 0.644252 | 0.632173 | -0.012079 [-0.350044, 0.284937] | no | no |

### DIAGNOSTIC — post-hoc OUTER-score oracle

This baseline was selected using the outer score and cannot authorize launch. Its frozen interval is conditional on that choice.

| Primary target | Split | Held | Primary F | OUTER oracle | MAE F | MAE baseline | Gain [frozen 95% interval] | Excludes 0? | Positive and clear? |
|---|---|---|---|---|---|---|---|---|---|
| MBPP I_T | data_rung | 66 | F_curv | reuse_only | 0.065018 | 0.042639 | -0.022379 [-0.026614, -0.016586] | yes | no |
| MBPP I_T | data_rung | 132 | F_log | mean_effect | 0.023339 | 0.017058 | -0.006281 [-0.009976, -0.001794] | yes | no |
| MBPP I_T | data_rung | 198 | F_int | mean_effect | 0.028744 | 0.021759 | -0.006984 [-0.011732, 0.002071] | no | no |
| MBPP I_T | data_rung | 594 | F_log | constant | 0.036985 | 0.027936 | -0.009049 [-0.014845, -0.002650] | yes | no |
| MBPP I_T | student | gemma3-1b | F_log | surface | 0.043361 | 0.036332 | -0.007028 [-0.010202, -0.003910] | yes | no |
| MBPP I_T | student | gemma3-270m | F_log | mean_effect | 0.031690 | 0.025288 | -0.006403 [-0.012263, 0.000142] | no | no |
| MBPP I_T | student | gemma3-4b | F_log | surface | 0.032049 | 0.026808 | -0.005241 [-0.011149, -0.000534] | yes | no |
| MBPP I_T | largest_budget | T>=150000 | F_log | surface | 0.045698 | 0.030334 | -0.015364 [-0.022170, -0.007146] | yes | no |
| MBPP I_T | pool_seed | 41 | F_int | mean_effect | 0.034413 | 0.030749 | -0.003663 [-0.010354, 0.005024] | no | no |
| MBPP I_T | pool_seed | 42 | F_int | surface | 0.040232 | 0.027748 | -0.012484 [-0.018209, -0.003185] | yes | no |
| MBPP I_T | pool_seed | 51 | F_curv | mean_effect | 0.029475 | 0.018010 | -0.011465 [-0.019240, -0.003749] | yes | no |
| MBPP I_T | pool_seed | 52 | F_curv | mean_effect | 0.031282 | 0.015967 | -0.015315 [-0.019394, -0.011267] | yes | no |
| fresh 2wiki_new I_U | data_rung | 66 | F_curv | surface | 11.123465 | 1.731052 | -9.392413 [-13.668355, -2.900561] | yes | no |
| fresh 2wiki_new I_U | data_rung | 132 | F_log | surface | 1.116584 | 0.868818 | -0.247766 [-0.465527, -0.026220] | yes | no |
| fresh 2wiki_new I_U | data_rung | 198 | F_log | surface | 1.447093 | 1.341085 | -0.106009 [-0.259771, 0.118976] | no | no |
| fresh 2wiki_new I_U | data_rung | 594 | F_int | surface | 1.365336 | 1.185901 | -0.179435 [-0.324912, -0.002791] | yes | no |
| fresh 2wiki_new I_U | student | gemma3-1b | F_log | surface | 1.083579 | 0.801168 | -0.282411 [-0.529095, -0.056688] | yes | no |
| fresh 2wiki_new I_U | student | gemma3-270m | F_log | reuse_only | 1.631196 | 0.530535 | -1.100661 [-1.691140, -0.467254] | yes | no |
| fresh 2wiki_new I_U | student | gemma3-4b | F_log | surface | 1.983881 | 1.636580 | -0.347301 [-0.535274, -0.106974] | yes | no |
| fresh 2wiki_new I_U | largest_budget | T>=150000 | NA | NA | NA | NA | NA [NA] | NA | unscorable |
| fresh 2wiki_new I_U | pool_seed | 41 | F_log | surface | 1.509319 | 1.191950 | -0.317369 [-0.737907, 0.272595] | no | no |
| fresh 2wiki_new I_U | pool_seed | 42 | F_log | surface | 1.681465 | 1.541160 | -0.140304 [-0.382086, 0.213377] | no | no |
| fresh 2wiki_new I_U | pool_seed | 51 | F_curv | surface | 0.820272 | 0.733709 | -0.086563 [-0.412246, 0.121798] | no | no |
| fresh 2wiki_new I_U | pool_seed | 52 | F_curv | surface | 0.644252 | 0.632173 | -0.012079 [-0.350044, 0.284937] | no | no |

code × I_T (MBPP): the launch verdict remains not met with the development-stage baseline; 3/12 scorable folds have positive gain, 1/12 have positive gain and an interval clear of zero (0 unscorable), and 0/4 scorable splits have a clear positive gain.

QA × I_U (fresh 2wiki_new, 1% proxy): the launch verdict remains not met with the development-stage baseline; 4/11 scorable folds have positive gain, 0/11 have positive gain and an interval clear of zero (1 unscorable), and 0/3 scorable splits have a clear positive gain.

Split-level gains [frozen intervals], read directly from A2's result metrics:

- code / data_rung: -0.000695 [-0.0027685519898134913, 0.0014252791551698217].
- code / student: -0.003789 [-0.007868208553274535, 0.00022226416848877812].
- code / largest_budget: -0.005850 [-0.013539954496465475, 0.0033099379032058627].
- code / pool_seed: -0.002914 [-0.00683359294172371, 0.001447717444031196].
- qa / data_rung: -2.583019 [-4.694708758112539, -0.4530269927368718].
- qa / student: 0.294757 [-0.16134951340890344, 0.9415688796111975].
- qa / largest_budget: NA (unscorable).
- qa / pool_seed: -0.162440 [-0.45509910502601963, 0.19519656469878338].

## 2. Parameter accounting

For gemma3-4b, 4.300B minus 0.671B equals 3.629B only when the 0.420B vision/projector deduction is omitted; the round used 3.209B TEXT-DECODER NON-EMBEDDING parameters. These rounded values are calculated from the exact file counts below. The artifact combines vision tower and multimodal projector into one deduction.

| Student | Total | − tied token embedding | − (vision tower + projector) | = TEXT-DECODER NON-EMBEDDING | Revision |
|---|---|---|---|---|---|
| gemma3-270m | 268,098,176 | 167,772,160 | 0 | 100,326,016 | 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 |
| gemma3-1b | 999,885,952 | 301,989,888 | 0 | 697,896,064 | fcf18a2a879aab110ca39f8bffbccd5d49d8eb29 |
| gemma3-4b | 4,300,079,472 | 671,252,480 | 419,816,304 | 3,209,010,688 | cc012e0a6d0787b4adcc0fa2c4da74402494554d |
| gemma3-12b | 12,187,325,040 | 1,006,878,720 | 421,290,864 | 10,759,155,456 | 295efb63d01a7017928f273a94ebb86105c9526f |

## 3. One training-seed pair

gemma3-1b, pool B, seeds 0 and 1: **one run pair**, 14 shared checkpoints. Signed difference = delta(seed 1) − delta(seed 0).

Descriptive only: one pair documents the signed seed differences along this pool-B trajectory and at its two corner checkpoints. The 14 checkpoints are correlated observations of one pair, not 14 independent replicates. One pair cannot estimate a general training-seed noise distribution, establish interval coverage, or establish that the registered band is generally conservative. No sigma is calculated from these points; A5's derived sigma summaries are not reused.

| Update | Supervised tokens | 2Wiki QA probe difference | MATH-500 difference | MBPP difference | Corner checkpoint |
|---|---|---|---|---|---|
| 33 | 46348 | 0.002770 | 0.000322 | -0.001163 | — |
| 34 | 47630 | 0.006816 | 0.000663 | -0.001273 | — |
| 35 | 49189 | 0.013042 | -0.000189 | -0.001246 | — |
| 36 | 50558 | 0.010427 | 0.001118 | -0.002574 | corner 1 |
| 37 | 52086 | 0.016901 | 0.001818 | -0.001578 | — |
| 38 | 54152 | 0.016777 | 0.002103 | -0.001633 | — |
| 39 | 55468 | 0.016216 | 0.001800 | 0.000083 | — |
| 71 | 101095 | 0.086685 | 0.004357 | 0.000111 | — |
| 72 | 102704 | 0.095586 | 0.005891 | -0.000664 | — |
| 73 | 103832 | 0.106543 | 0.005607 | 0.000775 | — |
| 74 | 105450 | 0.116814 | 0.006838 | 0.000996 | corner 4 |
| 75 | 106128 | 0.102870 | 0.006421 | 0.003709 | — |
| 76 | 107466 | 0.105547 | 0.007084 | 0.004014 | — |
| 77 | 108709 | 0.101189 | 0.010324 | 0.006864 | — |

- **Update 36 (corner 1)**: 2Wiki QA probe 0.010427; MATH-500 0.001118; MBPP -0.002574.
- **Update 74 (corner 4)**: 2Wiki QA probe 0.116814; MATH-500 0.006838; MBPP 0.000996.

## 4. Frozen predictions at achieved corners

The achieved coordinates are an approximate rectangle. The additive structures predict exactly zero on a matched rectangle, but their frozen predictions at the four unequal achieved coordinates include mismatch terms. The zero column is the registered rectangle reference, not their exact achieved-coordinate evaluation. No coordinates are projected, averaged, or interpolated.

### Derivation from A2's stored form

u = log(1 + T/100000), v = log(1 + E), z = (descriptor − mean)/scale. The descriptor, mean, scale, lambda, and coefficients are the stored full-development values. A2 saves coefficients in the original design-column units after undoing ridge column scaling; no further coefficient standardisation is applied.

Write C(q) = q3 − q4 − q1 + q2, A = a + a′z, and B = b + b′z. The stored forms give:

- F_log: I = A C(u) + B C(v).
- F_curv: I = A C(u) + B C(h_p(E)), with h_p(E) = ((1+E)^p−1)/p and h_0(E) = v.
- F_int: I = A C(u) + B C(v) + k C(uv).

On an exact rectangle, C(u) = C(v) = C(h_p(E)) = 0, so F_log and F_curv give exactly 0 and F_int gives k (u_high−u_low)(v_low−v_high), independent of A, B, and z. At the achieved coordinates all four stored coordinates enter C directly; the small A and B mismatch terms remain. The interaction coefficient k has no student modifier in A2's form.

- gemma3-1b: corner 1 (B): T=50558, E=1.429161013116; corner 2 (A): T=50563, E=2.980957434265; corner 3 (C): T=105433, E=1.429289916764; corner 4 (B): T=105450, E=2.980834464043.
- gemma3-4b: corner 1 (B): T=50558, E=1.429161013116; corner 2 (A): T=50563, E=2.980957434265; corner 3 (C): T=105433, E=1.429289916764; corner 4 (B): T=105450, E=2.980834464043.

Each prediction cell gives **I / absolute error**. The zero reference applies to both additive structures on a matched rectangle. Registered quoted disagreement is copied from A5, not used as a coefficient or a prediction.

| Student | Readout | F_int achieved I / error | F_log achieved I / error | F_curv achieved I / error | Additive rectangle 0 / error | Measured I | Registered noise on I | Registered quoted disagreement |
|---|---|---|---|---|---|---|---|---|
| gemma3-1b | 2Wiki QA probe | -0.488928 / 0.245526 | 0.000174 / 0.243575 | 0.000148 / 0.243549 | 0 / 0.243401 | -0.243401 | 0.160800 | 1.554600 |
| gemma3-1b | MATH-500 | -0.003187 / 0.010764 | 0.000003 / 0.007574 | -0.000001 / 0.007578 | 0 / 0.007577 | 0.007577 | 0.008800 | 0.021500 |
| gemma3-1b | MBPP | -0.003424 / 0.010554 | 0.000003 / 0.013981 | 0.000001 / 0.013979 | 0 / 0.013978 | -0.013978 | 0.025400 | 0.014400 |
| gemma3-4b | 2Wiki QA probe | -0.488849 / 0.276511 | 0.000245 / 0.212584 | 0.000179 / 0.212518 | 0 / 0.212338 | -0.212338 | 0.160800 | 1.554600 |
| gemma3-4b | MATH-500 | -0.003179 / 0.033673 | 0.000010 / 0.036862 | 0.000003 / 0.036855 | 0 / 0.036852 | -0.036852 | 0.008800 | 0.021500 |
| gemma3-4b | MBPP | -0.003424 / 0.091184 | 0.000004 / 0.094612 | 0.000001 / 0.094609 | 0 / 0.094608 | -0.094608 | 0.025400 | 0.014400 |

### Frozen parameter provenance

- gemma3-1b / 2Wiki QA probe: `parameter_intervals[4].fits.F_int.fit`, fit `54d390e63b462d4644b3`; descriptor=log_parameters, lambda=0.0001, mean=20.225581625016378, scale=1.4180612558071819, descriptor value=20.363580744195005, z=0.09731534418100686; [a,a′,b,b′,k]=[-3.1410703331501093, -0.47865274569564903, -1.2045320494714542, 0.5871917491468696, 3.185144947346786]; k C(uv)=-0.488989228641, A C(u)+B C(v)=6.15865652367e-05.
- gemma3-1b / MATH-500: `parameter_intervals[1].fits.F_int.fit`, fit `f6dcf391766f12f4df18`; descriptor=initial_loss, lambda=0.01, mean=1.0926245138152246, scale=0.25212106823136698, descriptor value=1.2205068949840885, z=0.5072260801763242; [a,a′,b,b′,k]=[0.03776588947415002, 0.009702590545040234, 0.06956177856864044, -0.041333646924034746, 0.02077201994034371]; k C(uv)=-0.00318895817171, A C(u)+B C(v)=1.96513841394e-06.
- gemma3-1b / MBPP: `parameter_intervals[0].fits.F_int.fit`, fit `245505687a5c1bdfd70b`; descriptor=initial_loss, lambda=0.01, mean=1.0175856214939474, scale=0.16173007606526607, descriptor value=1.0600918954827281, z=0.26282232113479825; [a,a′,b,b′,k]=[0.06611035139599394, -0.004433951424927834, 0.06935622601785665, -0.007422578617359822, 0.022320976939089823]; k C(uv)=-0.00342675685923, A C(u)+B C(v)=2.4416026872e-06.
- gemma3-4b / 2Wiki QA probe: `parameter_intervals[4].fits.F_int.fit`, fit `54d390e63b462d4644b3`; descriptor=log_parameters, lambda=0.0001, mean=20.225581625016378, scale=1.4180612558071819, descriptor value=21.889228529701182, z=1.173184090512247; [a,a′,b,b′,k]=[-3.1410703331501093, -0.47865274569564903, -1.2045320494714542, 0.5871917491468696, 3.185144947346786]; k C(uv)=-0.488989228641, A C(u)+B C(v)=0.000140134794028.
- gemma3-4b / MATH-500: `parameter_intervals[1].fits.F_int.fit`, fit `f6dcf391766f12f4df18`; descriptor=initial_loss, lambda=0.01, mean=1.0926245138152246, scale=0.25212106823136698, descriptor value=0.74044362782239737, z=-1.3968720998343429; [a,a′,b,b′,k]=[0.03776588947415002, 0.009702590545040234, 0.06956177856864044, -0.041333646924034746, 0.02077201994034371]; k C(uv)=-0.00318895817171, A C(u)+B C(v)=9.48780326308e-06.
- gemma3-4b / MBPP: `parameter_intervals[0].fits.F_int.fit`, fit `245505687a5c1bdfd70b`; descriptor=initial_loss, lambda=0.01, mean=1.0175856214939474, scale=0.16173007606526607, descriptor value=0.8017050487156776, z=-1.3348202018476223; [a,a′,b,b′,k]=[0.06611035139599394, -0.004433951424927834, 0.06935622601785665, -0.007422578617359822, 0.022320976939089823]; k C(uv)=-0.00342675685923, A C(u)+B C(v)=3.08624867223e-06.

All three frozen fit records, standardizers, decompositions, and errors are retained in summary.json. The registered quoted disagreement is not reproduced by the stored full-development F_int evaluations at these achieved coordinates; this audit does not infer how that quoted number was obtained.

For the 2Wiki QA probe, both measured values are inside twice the registered noise: failed to reject additivity. This does not identify why the fitted structures miss the measurements. MATH-500 is marginal and has opposite measured signs across students; MBPP remains underpowered. Students are reported separately. The registered bands are not calibrated confidence intervals.

## Reproduction and input integrity

```bash
python -B -m pytest -q tests/test_a7_closeout_audit.py
python -B analysis/a7_closeout_audit.py
```

The audit command prints all four numbered checks and their tables after writing the reports. A failed check or a stopped prediction row gives a non-zero exit status. Inputs are hashed before and after the audit; A2/A3/A5 artifacts are never written.

- `results/a2-curvature-interaction/summary.json`: `aed1934bf0c81333e200678f08567d018575fb3fcef7e2ecfcd881c95d3f820f`
- `results/a2-curvature-interaction/nonembedding_counts.json`: `f79883a113b84af0fbee0f4b8437feff29ae359f16747e7c9dbad90c4fea6213`
- `results/a3-corner-pools/plan.json`: `fc5fb5a7623d502e25bd8a5ef31a85226a3ff3944edf408bb6968ce8149dd505`
- `results/a5-training-seed-noise/summary.json`: `dbf05ffd475fae3bcb7074b8e39ba8b931f7d7d8047ec3abda79d298791075f8`
- `results/a5-corner-second-difference/summary.json`: `05de90f078ea37d760f256eb9b9cfe37bea734827b2c8099e25f66a689a1bc75`
