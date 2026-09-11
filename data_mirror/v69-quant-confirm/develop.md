# V69 quantization development

54 unblinded state/config cells; 162 capability responses. Six whole-state LOSO folds, 45 training and 9 held-out cells per fold/capability.

Per capability: minimize LOSO macro MAE (equal weight per held-out state). Candidates within 0.02 nats of the minimum, inclusive, tie; choose the fewest coefficients, then lower MAE, then declared candidate order. All six candidates, including median and zero, are eligible. No test-based selection.

V55 population mean/std of [log(N0), L0c, log(D0)], pooled over training rows and capabilities; constant scale=1. Refit on five training states in each LOSO fold. u=log2(qmax)-training mean(log2(qmax)); v=log2(g/128). N0 counts transformer matrices excluding embeddings/head (V55), while V54 quantization includes embeddings/head. D0=step*2097152.

Piecewise bilinear interpolation in x=log2(qmax) and v=log2(g/128) on the 3x3 measured grid. For g<64 use g=64,128; for g>256 use g=128,256, linearly extrapolating in v at the same b. Floor the extrapolated predicted signed dL at zero: max(0, dL_extrapolated). Interior predictions remain signed. Apply this identical rule to same-input interpolation and per-config medians. No bit extrapolation; only b=3,4,5 is in the confirmation protocol.

Ridge 1e-3 penalizes all regression coefficients including intercepts. Median has nine scalar anchors (no ridge); zero has no coefficients. Condition numbers below are for the unregularized design, not the normal matrix. Median diagnostics describe its configuration indicator design.

| Candidate | Coefficients | Math macro MAE | Code macro MAE | QA macro MAE |
|---|---:|---:|---:|---:|
| 2-D surface | 20 | 0.612132 | 1.476641 | 3.233241 |
| Bilinear | 16 | 0.842964 | 1.562056 | 3.601410 |
| Piecewise interpolation | 36 | 0.889909 | 1.437934 | 3.204602 |
| Bit only | 8 | 0.873761 | 1.560169 | 3.601410 |
| Per-config median | 9 | 1.216184 | 1.295991 | 1.319762 |
| Zero | 0 | 1.322415 | 1.424120 | 1.317910 |

Selections: math: low_order_2d; code: median; qa: zero.

| Capability | Candidate | Full rank / coefficients | Condition number | Fold rank range | Fold condition range |
|---|---|---:|---:|---:|---:|
| math | 2-D surface | 20 / 20 | 51.7739 | 20–20 | 47.4966–446.8980 |
| math | Bilinear | 16 / 16 | 21.0211 | 16–16 | 19.2845–181.4485 |
| math | Piecewise interpolation | 36 / 36 | 16.2774 | 36–36 | 14.9327–140.5023 |
| math | Bit only | 8 / 8 | 17.1637 | 8–8 | 15.7457–148.1521 |
| math | Per-config median | 9 / 9 | 1.0000 | 9–9 | 1.0000–1.0000 |
| math | Zero | 0 / 0 | not applicable (fixed zero) | 0–0 | See JSON (undefined or rank deficient) |
| code | 2-D surface | 20 / 20 | 50.6089 | 20–20 | 45.9039–301.7243 |
| code | Bilinear | 16 / 16 | 20.5481 | 16–16 | 18.6378–122.5054 |
| code | Piecewise interpolation | 36 / 36 | 15.9112 | 36–36 | 14.4320–94.8605 |
| code | Bit only | 8 / 8 | 16.7775 | 8–8 | 15.2177–100.0253 |
| code | Per-config median | 9 / 9 | 1.0000 | 9–9 | 1.0000–1.0000 |
| code | Zero | 0 / 0 | not applicable (fixed zero) | 0–0 | See JSON (undefined or rank deficient) |
| qa | 2-D surface | 20 / 20 | 141.8200 | 20–20 | 126.3760–234.0861 |
| qa | Bilinear | 16 / 16 | 57.5814 | 16–16 | 51.3109–95.0431 |
| qa | Piecewise interpolation | 36 / 36 | 44.5874 | 36–36 | 39.7319–73.5954 |
| qa | Bit only | 8 / 8 | 47.0150 | 8–8 | 41.8952–77.6024 |
| qa | Per-config median | 9 / 9 | 1.0000 | 9–9 | 1.0000–1.0000 |
| qa | Zero | 0 / 0 | not applicable (fixed zero) | 0–0 | See JSON (undefined or rank deficient) |

Three measured bit levels remove V55's two-level u² aliasing; ranks and condition numbers are measured rather than assumed. Full singular spectra and every fold's coefficients/scaler are in develop.json.

V54 zero-pads each last contiguous input group, computes absmax using real entries plus neutral zeros, then trims. qmax=2**(b-1)-1. Grouping is along input width, not vocabulary/output width; embeddings and LM head have input width hidden_size.

All confirmation input widths (1024/4096 and 2048/8192) divide by 32 and 512: no partial groups. The development-only 160M width 768 would have 256 real tail entries and 256 padding zeros at g=512; its MLP input width 3072 divides exactly.

Confirmation panel: two development states and the new 1.4B@112000 state, each at b=3/4/5 × g=32/512 (18 cells), plus the new state at g=128 (3 cells). Every candidate is frozen for all 21 cells. No V54 sentinel is created or changed.
