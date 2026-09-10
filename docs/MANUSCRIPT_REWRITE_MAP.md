# MANUSCRIPT_REWRITE_MAP — S0 fact check for the 2026-09-09 rewrite directive

Written 2026-09-09 17:50 EDT. Scope: writing + integration of existing data; no new GPU work. Running P2-v2 lanes are
untouched; their results are handled after a stated data cut-off (see §5).

## 0. Versions recorded at S0

| item | value |
|---|---|
| submission repo | `paper/` (its own git; remote github.com/XueqiC/ICLR27_scaling_down, branch `main`); HEAD at S0 start **8ff7e67** (working tree clean) |
| analysis repo | project root (branch `master`, HEAD a4fb160, 38 uncommitted paths: scripts/state; `results/` IS tracked there) |
| baseline PDF kept | `deliverables/scaling_down_law_draft_2026-09-09e.pdf` (build of paper HEAD c217ed8, 22 pp.) — the PDF the directive reviewed, plus main-table rows/legend added after review |
| data cut-off for this rewrite | all result files present at 2026-09-09 17:30 EDT; P2-v2 (v50) results arriving later are NOT integrated in this round |

Frozen predictions and their provenance (original freeze, not the mirror copy):

| test | frozen artifact (authoritative) | freeze commit | sha256 (first 16) | measurement after freeze |
|---|---|---|---|---|
| v38 quant/prune source transfer to step 96k (160M, 1.4B) | results/v38-prospective/register.json (analysis repo) | 4e0bf2c 2026-09-08 12:36 (analysis repo); register carries frozen_fit_sha256 | e60fcd9ba0a2573c | C28 |
| v40 pruning strength axis (d=0.65/0.55, new source 410M@96k) | results/v40-prune-strength/register.json | 20542e2 2026-09-08 17:17 (analysis repo) | 2080fb5ddf606f3a | C30 |
| v41 distillation unseen pool U225 (4 candidates) | results/v41-distill-newpool/summary.json (predictions section) | 0bbbaa8 2026-09-08 22:28 scaffold with predictions (analysis repo); result 84182be 09-09 01:50 | 36f2b4ae7790d34d | C31 |
| v46 Pythia-1B@96k (d=0.65/0.55; int4/int3) | data_mirror/v46-p1-newsource/predictions_frozen.json (paper repo) | b1bf631 2026-09-09 15:19 (paper repo) | c33b03950f0a649d | C32 |
| v49 P1-v2 1B@32k/112k | data_mirror/v49-p1v2/predictions_pythia-1b--step*.json | cd9ed8f 2026-09-09 15:48 (paper repo) | 42a890efab89 / e02ddeef0656 | C33 (measured 15:47–16:12) |
| v49 P1-v2 6.9B@32k/112k | data_mirror/v49-p1v2/predictions_pythia-6.9b--step*.json | 950312f 2026-09-09 15:55:58 (paper repo) | cc24dacc8797 / 86cc12db7a77 | C34 (measured from 16:12:04) |
| v47 P2-v2 pools (dev/test registration) | data_mirror/v47-p2-register/register.json | mirrored 1889db7 2026-09-09 15:23:25 -0 (paper repo) | 88472817365fea4e | running; out of scope |
| v36b / v42 / v43 / v44 | summary.json in data_mirror/<dir> | retrospective analyses (LOO or post-test baselines); no freeze | — | C21, C30, C30b, C31b |

Mirror rule: `results/<dir>` (analysis repo) is authoritative for measurements; `paper/data_mirror/<dir>` is a byte copy with
`@`→`--` renames for Overleaf; `paper/code/analysis/` mirrors the generating scripts. Where a number appears in a
document AND a table, the table generated from JSON is authoritative; prose must follow it.

## 1. Path and claim map

| paper location (09e) | actual TeX | claim id | measurement / frozen prediction | generator | current problem | action |
|---|---|---|---|---|---|---|
| Abstract | main.tex 99–128 | A1–A6 | — | hand | states invariant ordering only implicitly; lacks 3-level structure | rewrite (S2) |
| Intro finding 1 "invariant ordering" | intro.tex 66–72; con.tex 17–19; experiment.tex caption 70–74 | H1 | results/v6-capability-geometry/<model>/prune_losses.json | none (table hand-typed) | FALSE for Gemma-3/4, Muse, OLMo-7B (see §2) | generate table from JSON; family-specific wording |
| Intro finding 4 "int3 collapse in nine models" | intro.tex 86–88; experiment.tex 81–84 | H1b | results/v10-quantization/<model>/quant_losses.json | none | cohort is 14 models; OLMo-3-32B int3 = +0.4–1.2 nats | generate cohort table; bounded wording |
| Table tab:grid | experiment.tex 45–76 | H1/H8 | same as above | hand-typed | violates "generated from result files" | new script → tables/panel_prune.tex |
| Main table tab:main | tables/main_prediction.tex ← analysis/v45_main_table.py ← results/v45-main-table/main_table.json | H2/H3 | v36b, v42, v38+v40, v39, v41, v46, v49 | v45_main_table.py | R overloaded; rotated overfull table; baseline columns without values | split into 2a (source axis) / 2b (configuration axis) with provenance columns (S1) |
| 1B@96k int3 prose | experiment.tex 222–225 | H3 | results/v46-p1-newsource/compare.json | v46_p1_compare.py | prose swaps no-D0 (2.46) and per-bit median (0.620); table is right | fix prose |
| tab:p1v2 best column | tables/p1v2.tex ← v49_p1v2_table.py | H3 | v49 compare files | v49_p1v2_table.py | best = median-curve but no median column | add median column (S1) |
| Related work: Sengupta / Zhou | related.tex 15–20 | H4 | — | — | unverified claims | verification doc (agent) → rewrite |
| Methods §3.3 candidate templates | method.tex 52–96 | H5 | — | — | retired KD template, unfitted quant candidates, "future work" | replace with actually-tested predictors |
| Appendix "leaving only basic parameters" | appendix.tex 208–210 | H5 | no mapping/coefficient file exists | — | unsupported | delete |
| Recovery "asymptotic residual" | appendix.tex 250–272; method.tex 86–96; intro.tex 90 | H5 | results/v13? (recovery) | — | 16M-token max budget called asymptote | reword as fitted plateau under recipe |
| Quantizer text | appendix.tex 46–54 | H6 | analysis/v10_quantization.py fake_quantize_per_output_channel | — | "2^{b−1}−1 levels" wrong; (7/15)^2 stated as generic consecutive-bit ratio | correct (see §2) |
| 6.9B conclusion | experiment.tex 250–269 | H7 | v49 compare | v49 | protocol B exceptions under-reported in prose | per protocol × capability × regime wording |
| Counts | intro 63,75,88; experiment 20,29,81; appendix 103,138; main 121 | H8 | result dirs | none | 12/four families/seven models/nine models/ten densities inconsistent | generate counts (S1) |

## 2. Hard-issue verification (from raw result files, 2026-09-09)

**H1 (ordering).** Per model, pre-cliff densities (cliff = largest d with ΔL_math ≥ 1) where QA has the SMALLEST ΔL:
Qwen3-0.6B 3/3, 1.7B 6/6, 4B 5/5, 8B 4/4, 14B 3/4; OLMo-3-32B 7/10; Gemma-3-270M 0/2, 1B 0/5, 4B 1/3, 12B 0/3, 27B 1/2;
Gemma-4-31B 0/3; Muse-30B 1/4; OLMo-3-7B 4/8. In Gemma-3 (all sizes), Gemma-4-31B, Muse and OLMo-3-7B, QA is the MOST
damaged capability at d = 0.9/0.8 (OLMo-3-7B and Muse differences are ≤ 0.15 nats, near noise). Verdict: "invariant
ordering" is false; correct statement = ordering is family-specific (QA least damaged and improving in Qwen3 and
OLMo-3-32B; QA most damaged at mild densities in Gemma-3/4 and Muse).

**H1b (int3 cohort).** 14 non-Pythia models have int3: Qwen3 {0.6B 8.8/10.8/6.6, 1.7B 9.5/9.5/8.1, 4B 10.7/11.9/6.0,
8B 12.9/13.6/9.5, 14B 9.3/10.8/6.5}, Gemma-3 {270M 26/28/26, 1B 15/14/10, 4B 22/19/15, 12B 20/23/21, 27B 18/17/16},
Gemma-4-31B 20/20/15, Muse-30B 11/11/8, OLMo-3-7B 4.7/4.6/2.8, **OLMo-3-32B 0.68/1.22/0.38** (math/code/QA nats).
Verdict: 12 of 14 collapse (≥ 4.6 nats); OLMo-3-7B is intermediate; OLMo-3-32B does not collapse. "Nine models" is stale.

**H2 (labels).** Confirmed: tables/main_prediction.tex uses R for both "frozen candidate lost" (added 62684cb) and
"closeout baseline"; appendix.tex 108–113 defines R by timing. Fix: separate columns prediction_origin / baseline_origin /
selection_origin / outcome (S1).

**H3 (numbers).** results/v46-p1-newsource/compare.json int3 MAE: full 0.397, no-D0 2.46, per-bit median 0.620, zero 4.87.
Table row correct; prose (experiment.tex 222–225) swapped → fix. tab:p1v2 lacks median-curve column → fix. Main table
overfull: replace sidewaystable with two normal tables.

**H4 (novelty).** Verification of Sengupta 2025, Pruning Laws, P² Law, task-specific distillation laws, Kumar 2024,
Zhou 2025, Busbridge 2025, Frantar 2023, Panferov 2025 delegated; output docs/RELATED_WORK_VERIFICATION.md. Until then the
two unsupported sentences in related.tex are treated as unverified.

**H5 (methods).** method.tex still lists D1–D4 dense candidates, M1/M2/M3 hypotheses, quant candidates
(4^{-b}, exponential, polynomial, effective-parameter), KD template (capacity floor, teacher vector, data power law,
coverage factor), recovery law with r_{m,c} "asymptotic". Actually tested: pruning config-indicator OLS {N0,L0,D0} (v36b),
shared power A_c(x)((1−d)/0.3)^γ (v40) vs A2 per-density regression + interpolation and A1 γ=1 (v42, v46, v49) and a
continuous two-term polynomial (v49); quantization config-indicator per bit (v38/v40/v44/v46/v49) with per-bit
mean/median/no-D0/zero baselines and learned exponential vs 4^{-b} (leave-one-bit-out, v10 shape); distillation linear
{N0,L0,D0} vs constant (v39), constant/T/E/joint on pools (v41). appendix.tex 208–210 ("leaving only basic parameters")
has no mapping file → delete.

**H6 (quantizer).** Code: qmax = 2^{b−1}−1; integers clamped to [−qmax, qmax] (2^b − 1 values); scale = per-output-channel
max|w|/qmax; fake-quant, weights only, activations bf16, dense restored between bits. Step-size ratio 4→5 bit = 7/15;
squared 0.218; general (qmax(b)/qmax(b+1))^2. Appendix sentence "uses 2^{b−1}−1 levels ... step-size ratio between
consecutive bit-widths is (7/15)^2" → rewrite; squared-step ratio ≠ loss ratio without further assumptions.

**H7 (6.9B).** Protocol B exceptions in v49 compare: 6.9B@112k int3 full 1.294 < mean 3.77 / median 4.14; 6.9B@112k
pruning interp A2 0.106 vs median 0.095 / strength-only 0.111 (on par); code-capability lead for A2 under B. Prose must keep
protocol × capability × regime structure.

**H8 (counts, from result dirs).** Pruning heterogeneous panel: 12 models (Qwen3 0.6/1.7/4B, Gemma-3 270M/1B/4B/12B/27B,
Gemma-4-31B, Muse-30B, OLMo-3 7B/32B) + 2 prospective Qwen3 8B/14B (4 densities). Quantization: 14 models (same 12 + Qwen3
8B/14B) + Pythia. Geometry block figures: 10 models (no Gemma-3-27B, OLMo-3-32B; Gemma-4-31B has similarity.json but no
figure) — "seven models, three families" is stale. Distillation heterogeneous: Gemma-3 270M/1B/4B/12B, OLMo-3-7B, Qwen3
0.6B/1.7B/4B students. Family count: 4 families if Gemma-3 and Gemma-4 are one family (Qwen3, Gemma, Muse, OLMo-3), 5
series otherwise → adopt "five model series from four families" consistently. Densities: 10 for most models, 15–16 for
Qwen3-1.7B/Gemma-3-1B (infill), 4 for prospectives. Pythia-2.8B: excluded because the step revisions loaded identical
weights (hash evidence in v36 guard) — do not generalize to other revisions.

## 3. Authoritative sources for duplicated quantities
- Panel pruning ΔL → prune_losses.json (v6) → new tables/panel_prune.tex (S1).
- Quantization ΔL → quant_losses.json (v10) → new tables/panel_quant.tex (S1).
- Prediction comparisons → results/v45-main-table/main_table.json (rebuilt with provenance columns) → tables 2a/2b.
- P1-v2 per-stage → results/v49-p1v2/p1v2_table.json → tables/p1v2.tex (appendix).

## 4. Things NOT verified in S0
- ICLR 2027 page limit / template rules: not checked online in this round (template file iclr2027_conference.sty in
  repo; no page count enforced by us). Recorded as unverified.
- Related-work claims: pending docs/RELATED_WORK_VERIFICATION.md.
- Per-byte normalization data: existence of per-sample logprob + byte counts to be checked in S1 (MEASUREMENT_AUDIT.md).

## 5. Data cut-off
Any P2-v2 (v50) result landing during this rewrite is recorded in RESULTS_LEDGER as a separate entry and integrated only
after the rewrite is delivered, under a new PDF version tag.
