# V63 quantization identifiability — retrospective (R)

CPU-only audit. The delivered V55 predictions are unchanged. All controls are retrospective (R); none is selected or substituted using test outcomes. Signed response and MAE are in nats. Fits use exactly 24 development cells per capability; frozen tests contain 12 bit, 18 granularity, and 3 joint cells per capability.

## Identifiability and rank / effective degrees of freedom

With u3=-1.160964047443681 and u5=1.160964047443681, u²=(u3+u5)u−u3u5=1.347837519456814 on development rows. Maximum scalar residual: 0.000e+00. The delivered design has 20 coefficients, rank 16, and four null directions per capability. Those directions produce a nonzero polynomial at b=4 (residual -1.344064074).

Four coefficient directions are unidentifiable from the dev responses. Ridge uniquely chooses coefficients but does not identify bit curvature. The delivered and no-u^2 designs span the same dev column space; applying isotropic ridge separately to their coefficients changes the penalty. For u3+u5=0 and c=-u3*u5, the delivered penalty on the combined constant phi block is lambda/(1+c^2), versus lambda without u^2. The null polynomial is nonzero at b=4, so its prediction is representation-dependent.

Effective df = tr[X (XᵀX + λI)⁻¹ Xᵀ], checked independently against Σ sᵢ²/(sᵢ²+λ). λ=0.001 penalizes every coefficient, including intercepts. Median is nonlinear in responses, so this linear-smoother df is not assigned to it; the constant zero predictor has df=0.

| Capability | Control | Rows | Coefficients | Rank | Nullity | λ | Effective df |
| --- | --- | --- | --- | --- | --- | --- | --- |
| math | Delivered (20) | 24 | 20 | 16 | 4 | 0.001 | 15.978983355 |
| math | Without u^2 (16) | 24 | 16 | 16 | 0 | 0.001 | 15.974228146 |
| math | Bit-only (8) | 24 | 8 | 8 | 0 | 0.001 | 7.987114073 |
| math | Bit+group additive (12) | 24 | 12 | 12 | 0 | 0.001 | 11.979722408 |
| code | Delivered (20) | 24 | 20 | 16 | 4 | 0.001 | 15.980115555 |
| code | Without u^2 (16) | 24 | 16 | 16 | 0 | 0.001 | 15.975615489 |
| code | Bit-only (8) | 24 | 8 | 8 | 0 | 0.001 | 7.987807745 |
| code | Bit+group additive (12) | 24 | 12 | 12 | 0 | 0.001 | 11.980813679 |
| qa | Delivered (20) | 24 | 20 | 16 | 4 | 0.001 | 15.921972288 |
| qa | Without u^2 (16) | 24 | 16 | 16 | 0 | 0.001 | 15.904522055 |
| qa | Bit-only (8) | 24 | 8 | 8 | 0 | 0.001 | 7.952261027 |
| qa | Bit+group additive (12) | 24 | 12 | 12 | 0 | 0.001 | 11.924937278 |

## Standardization

Natural logs of N0 and D0; population mean/std pooled over training response rows and capabilities, as in v53. Constant feature scale=1. Refit on training states in every LOSO fold. u_center is the training mean of log2(qmax); v=log2(g/128). Only phi covariates are z-scored; configuration factors are as specified.

| Raw feature | Center | Population scale |
| --- | --- | --- |
| ln N0 | 19.565165103220309 | 1.0841753790076554 |
| L0c | 2.7458963962102487 | 1.6234865401697796 |
| ln D0 | 25.331562746990869 | 1.0951279540100636 |

Shared across capabilities and controls. u_center=2.7459265481648374; qmax=2^(b−1)−1; u=log2(qmax)−u_center; v=log2(g/128). Only the three raw phi features are z-scored. No constant features. Each full-fit diagnostic includes these constants in summary.json; each LOSO fold exports its own training-only constants.

## Frozen test MAE

### Bit test (12 cells/capability)

States: pythia-160m@step16000, pythia-160m@step143000, pythia-410m@step16000, pythia-410m@step143000, pythia-1.4b@step16000, pythia-1.4b@step143000. Configurations: b4_g64, b4_g256.

| Control | Math | Code | QA |
| --- | --- | --- | --- |
| Delivered (20) | 0.193432707 | 0.213606392 | 0.436358923 |
| Without u^2 (16) | 1.299633030 | 1.386332072 | 1.944117911 |
| Bit-only (8) | 1.299633030 | 1.386332072 | 1.944117911 |
| Bit+group additive (12) | 1.299633030 | 1.386332072 | 1.944117911 |
| V55 median | 0.553264240 | 0.725764643 | 0.537138516 |
| Zero | 0.449911348 | 0.531831868 | 0.480486078 |

### Granularity test (18 cells/capability)

States: pythia-160m@step16000, pythia-160m@step143000, pythia-410m@step16000, pythia-410m@step143000, pythia-1.4b@step16000, pythia-1.4b@step143000. Configurations: b3_g128, b4_g128, b5_g128.

| Control | Math | Code | QA |
| --- | --- | --- | --- |
| Delivered (20) | 0.218938065 | 0.319233704 | 0.985740974 |
| Without u^2 (16) | 0.609232729 | 0.718320610 | 1.475905643 |
| Bit-only (8) | 0.609232729 | 0.718320610 | 1.475905643 |
| Bit+group additive (12) | 0.609232729 | 0.718320610 | 1.475905643 |
| V55 median | 1.122512430 | 1.221895401 | 1.220163822 |
| Zero | 1.238664127 | 1.345654069 | 1.238961172 |

### Joint test (3 cells/capability)

States: pythia-1b@step96000. Configurations: b3_g128, b4_g128, b5_g128.

| Control | Math | Code | QA |
| --- | --- | --- | --- |
| Delivered (20) | 0.344966600 | 0.096861168 | 0.268850664 |
| Without u^2 (16) | 0.527068107 | 0.222480133 | 0.319682470 |
| Bit-only (8) | 0.527068107 | 0.222480133 | 0.319682470 |
| Bit+group additive (12) | 0.527068107 | 0.222480133 | 0.319682470 |
| V55 median | 0.151543962 | 0.269793422 | 0.301725425 |
| Zero | 0.299698120 | 0.338503288 | 0.103938926 |

The three controls without u^2 have equal test MAEs to numerical precision. Their group blocks are orthogonal to the shared [phi,u*phi] blocks on the balanced development grid. At g=128, v=0, so their granularity/joint predictions coincide. On the bit test their predictions differ; symmetric group contributions cancel in paired absolute errors for these measured responses. Equal MAEs do not mean identical predictors or establish that group effects are unnecessary.

## Development LOSO by state

Six folds, five training states (20 cells/capability) and one held state (4 cells/capability). Standardization and median anchors are refit using each fold's training rows. Pooled MAE covers 24 held-out cells/capability; equal fold sizes make this also the mean of the six fold MAEs.

| Control | Math | Code | QA |
| --- | --- | --- | --- |
| Delivered (20) | 0.913503470 | 1.909849626 | 4.371232976 |
| Without u^2 (16) | 0.980353008 | 1.901381411 | 4.364868549 |
| Bit-only (8) | 1.062993355 | 1.901381411 | 4.364868549 |
| Bit+group additive (12) | 1.099844046 | 1.974519331 | 4.376835118 |
| V55 median | 1.693907531 | 1.758183286 | 1.776515714 |
| Zero | 1.821478948 | 1.929113680 | 1.795834076 |

| Held-out state | Control | Math | Code | QA |
| --- | --- | --- | --- | --- |
| pythia-160m@step16000 | Delivered (20) | 0.440513690 | 1.440183628 | 1.995066371 |
| pythia-160m@step16000 | Without u^2 (16) | 0.469394409 | 1.456011188 | 2.144020159 |
| pythia-160m@step16000 | Bit-only (8) | 0.469394409 | 1.456011188 | 2.144020159 |
| pythia-160m@step16000 | Bit+group additive (12) | 0.506696312 | 1.511444027 | 2.144020159 |
| pythia-160m@step16000 | V55 median | 0.216300522 | 0.260458759 | 0.114068441 |
| pythia-160m@step16000 | Zero | 0.350826544 | 0.489452104 | 0.272850820 |
| pythia-160m@step143000 | Delivered (20) | 4.058081116 | 7.338620514 | 7.986104270 |
| pythia-160m@step143000 | Without u^2 (16) | 4.457625197 | 7.266758094 | 7.981653413 |
| pythia-160m@step143000 | Bit-only (8) | 4.457625197 | 7.266758094 | 7.981653413 |
| pythia-160m@step143000 | Bit+group additive (12) | 4.635728136 | 7.279776948 | 7.981653413 |
| pythia-160m@step143000 | V55 median | 8.028009571 | 7.856489185 | 8.818723265 |
| pythia-160m@step143000 | Zero | 8.378974531 | 8.345094485 | 8.979740970 |
| pythia-410m@step16000 | Delivered (20) | 0.344567215 | 0.715379097 | 6.397970819 |
| pythia-410m@step16000 | Without u^2 (16) | 0.342761031 | 0.711712189 | 6.183620039 |
| pythia-410m@step16000 | Bit-only (8) | 0.342761031 | 0.711712189 | 6.183620039 |
| pythia-410m@step16000 | Bit+group additive (12) | 0.367193064 | 0.728331434 | 6.183620039 |
| pythia-410m@step16000 | V55 median | 0.351587835 | 0.491160566 | 0.198201343 |
| pythia-410m@step16000 | Zero | 0.215539231 | 0.258824578 | 0.103270556 |
| pythia-410m@step143000 | Delivered (20) | 0.091465991 | 0.857768469 | 1.840606949 |
| pythia-410m@step143000 | Without u^2 (16) | 0.091465991 | 0.826555579 | 1.875883610 |
| pythia-410m@step143000 | Bit-only (8) | 0.390423555 | 0.826555579 | 1.875883610 |
| pythia-410m@step143000 | Bit+group additive (12) | 0.372567824 | 0.986900184 | 1.947683029 |
| pythia-410m@step143000 | V55 median | 0.924849719 | 1.087978369 | 1.001009981 |
| pythia-410m@step143000 | Zero | 1.275814680 | 1.576583670 | 1.168978731 |
| pythia-1.4b@step16000 | Delivered (20) | 0.458168789 | 0.110362828 | 5.765730683 |
| pythia-1.4b@step16000 | Without u^2 (16) | 0.427927187 | 0.146555551 | 5.756749283 |
| pythia-1.4b@step16000 | Bit-only (8) | 0.427927187 | 0.146555551 | 5.756749283 |
| pythia-1.4b@step16000 | Bit+group additive (12) | 0.444064935 | 0.146555551 | 5.756749283 |
| pythia-1.4b@step16000 | V55 median | 0.426535435 | 0.592554077 | 0.305370722 |
| pythia-1.4b@step16000 | Zero | 0.140591632 | 0.156509983 | 0.029690471 |
| pythia-1.4b@step143000 | Delivered (20) | 0.088224017 | 0.996783218 | 2.241918767 |
| pythia-1.4b@step143000 | Without u^2 (16) | 0.092944233 | 1.000695867 | 2.247284787 |
| pythia-1.4b@step143000 | Bit-only (8) | 0.289828749 | 1.000695867 | 2.247284787 |
| pythia-1.4b@step143000 | Bit+group additive (12) | 0.272814008 | 1.194107840 | 2.247284787 |
| pythia-1.4b@step143000 | V55 median | 0.216162106 | 0.260458759 | 0.221720532 |
| pythia-1.4b@step143000 | Zero | 0.567127066 | 0.748217257 | 0.220472909 |

## Delivered-design singular values

Descending, including all four near-zero singular values; rank tolerance is max(X.shape) × machine epsilon × largest singular value. Full spectra for all controls and LOSO folds are in summary.json.

math (rank tolerance 5.375673565e-14):

```text
 1  1.00874506099157344e+01
 2  8.22192091256854773e+00
 3  8.22192091256853885e+00
 4  6.97802495161100023e+00
 5  6.97802495161099934e+00
 6  6.01054353662016094e+00
 7  5.68753905190669506e+00
 8  5.68753905190669151e+00
 9  5.68753905190666842e+00
10  5.68753905190666131e+00
11  4.89897948556636553e+00
12  4.89897948556634155e+00
13  6.19720223979896123e-01
14  4.28693368936931662e-01
15  4.28693368936930885e-01
16  3.69256369205289758e-01
17  1.32467935024720907e-15
18  6.60441343429750471e-16
19  5.03502117476329253e-16
20  3.13814144173820657e-16
```

code (rank tolerance 5.405312589e-14):

```text
 1  1.01430682333697870e+01
 2  8.22192091256854773e+00
 3  8.22192091256853708e+00
 4  7.01649861351227422e+00
 5  7.01649861351227155e+00
 6  6.04368294518839910e+00
 7  5.68753905190669506e+00
 8  5.68753905190667908e+00
 9  5.68753905190667641e+00
10  5.68753905190666131e+00
11  4.89897948556636553e+00
12  4.89897948556634244e+00
13  6.37481537419734012e-01
14  4.40979812077950839e-01
15  4.40979812077950506e-01
16  3.79839335291166758e-01
17  1.28998609034133321e-15
18  7.31073040050723668e-16
19  2.73739198379096617e-16
20  1.07689924963516361e-16
```

qa (rank tolerance 7.537768876e-14):

```text
 1  1.41446221246867765e+01
 2  9.78458581201398836e+00
 3  9.78458581201398658e+00
 4  8.42798347938387948e+00
 5  8.22192091256855839e+00
 6  8.22192091256854596e+00
 7  5.68753905190668885e+00
 8  5.68753905190668707e+00
 9  5.68753905190668174e+00
10  5.68753905190667997e+00
11  4.89897948556636553e+00
12  4.89897948556635665e+00
13  3.17233422211383675e-01
14  2.19447194467550150e-01
15  2.19447194467549483e-01
16  1.89021524784294098e-01
17  1.11048805796663842e-15
18  6.50721226891147221e-16
19  5.00063398007488483e-16
20  1.18102380819531350e-16
```

## Reproduction and provenance

All nine delivered test MAEs reproduce V55 compare to absolute tolerance 1e-06. Maximum delivered MAE difference: 0.000e+00; maximum frozen prediction difference: 0.000e+00. Median/zero test MAEs and all six delivered/median/zero LOSO folds also reproduce. Every ridge solve agrees with independent augmented least squares; trace df agrees with SVD df. Full input hashes, coefficients, per-cell predictions/errors, and fold diagnostics are exported in summary.json.

Test features use the frozen V55 state inputs, including the recorded joint dense fallback. Test targets are read from the exact measurement files named by V55 compare, subtracting each file's own dense reference. All seven measurement file hashes match compare; all 72 development rows and 99 test responses match their V55 records. Protected V54/V55 file hashes are unchanged.

Reproduce: `python -B analysis/v63_quant_identifiability.py`. The default output includes `quant_ident.tex`; `--write-paper` additionally writes the two paper destinations when authorized.
