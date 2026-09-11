# V70 development

25 trajectories = 12 old dev + 9 old test + 4 held-out-4B; four checkpoints each (100 points).

Per capability, lowest leave-one-trajectory-out (LOCO) MAE, equal weight per trajectory and per checkpoint within trajectory. All 12 candidates within 0.02 nats (inclusive) of the minimum tie: choose fewer fitted parameters, then lower MAE, then declared candidate order. F2 counts four coefficients plus one fitted discrete T_star. Refit coefficients, descriptor references, column scaling and F2 T_star inside each training fold; T_star minimizes training SSE per capability on V56's fixed five-value grid.

Per student and capability, lowest development LOCO MAE among constant, intercept+T-only, intercept+E-only, and same-input surfaces with L0/logN. Exact ties: fewer parameters, then declared baseline order. Frozen before confirmation; no baseline selection or refitting on confirmation errors.

T_ref=35000; D_ref=median training-point D_U; N_ref=geometric mean of unique training-student total meta-device parameter counts (V50 convention, not training-manifest or nominal sizes). L0 and logN use population mean/std with equal weight per unique training student. Recompute within each fold.

| Capability | Selected | LOCO MAE | Parameters |
|---|---|---:|---:|
| math | E | 0.076370 | 1 |
| code | E | 0.048024 | 1 |
| qa | joint | 0.598470 | 3 |

| Method | Math LOCO MAE | Code LOCO MAE | QA LOCO MAE |
|---|---:|---:|---:|
| constant | 0.092414 | 0.081957 | 1.119922 |
| constant+src | 0.090676 | 0.083311 | 1.140889 |
| T | 0.088423 | 0.065260 | 1.371135 |
| T+src | 0.089251 | 0.065780 | 1.409866 |
| E | 0.076370 | 0.048024 | 1.523435 |
| E+src | 0.061438 | 0.045514 | 1.538881 |
| joint | 0.074314 | 0.040756 | 0.598470 |
| joint+src | 0.071204 | 0.038450 | 0.611653 |
| F1:L0 | 0.062324 | 0.044996 | 1.527618 |
| F1:logN | 0.061437 | 0.045513 | 1.538881 |
| F2:L0 | 0.064326 | 0.043234 | 0.592068 |
| F2:logN | 0.063939 | 0.043528 | 0.620735 |
| T-only | 0.089226 | 0.065011 | 1.031297 |
| E-only | 0.078796 | 0.045803 | 0.681547 |
| surface:L0 | 0.074204 | 0.043994 | 0.686484 |
| surface:logN | 0.074124 | 0.044745 | 0.692731 |

Frozen comparison baselines (development LOCO within student):

- gemma3-270m: math: T-only; code: E-only; qa: E-only
- gemma3-1b: math: surface:L0; code: E-only; qa: surface:logN
