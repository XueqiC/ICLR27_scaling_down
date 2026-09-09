# REWRITE_SUMMARY — 2026-09-09 manuscript rewrite (S0–S4)

Start: paper HEAD 8ff7e67 (PDF 09e reviewed by the directive). Data cut-off: result files present at 17:30 EDT; the
P2-v2 distillation matrix (v50) is running and is not in this version. No GPU, training, or teacher-API work was run.
Frozen predictions and pre-registration files were not modified (hashes in docs/MANUSCRIPT_REWRITE_MAP.md and
docs/CLAIM_EVIDENCE_MATRIX.md).

## Problem → change → evidence → remaining limit

1. **Invariant capability ordering / "QA most robust"** → replaced by a series-specific statement (QA least damaged and
   improving in Qwen3 and OLMo-3-32B; most damaged at mild densities in Gemma-3/4 and Muse; OLMo-3-7B within noise).
   Evidence: per-model pre-cliff ranks computed from prune_losses.json (v51_panel_tables.py → Table 2, counts.json).
   Limit: OLMo-3-7B and Muse differences ≤ 0.15 nats, near the 0.025-nat noise screen.
2. **"Int3 collapses every model (nine models)"** → 12 of 14 collapse (≥ 4.6 nats), OLMo-3-7B intermediate, OLMo-3-32B
   does not. Evidence: quant_losses.json (Appendix Table 5). Limit: collapse threshold (4 nats) declared in the script.
3. **Prospective label R decided by outcome** → four-part origin code (prediction origin · same-input baseline origin ·
   simple baseline origin · selection); a failed prospective keeps P; retrospective baselines are marked R; the only
   post-test selection (distillation per-capability reading) is labeled. Evidence: v52_prediction_tables.py (Table 3;
   Appendix Tables 6–8), method §4.3, appendix A.4.
4. **Swapped numbers and missing baseline values** → 1B@96k int3: full 0.397 / no-D0 2.46 / per-bit median 0.620 (text
   fixed); Table 9 now carries the median-curve column; the rotated overfull main table is replaced by a one-row-per-test
   table (Table 3) with candidate / strongest baseline per capability and three-line provenance tables in the appendix.
5. **Related work** → Sengupta et al. rewritten for the 2026 v2 (Pruning Laws: rolling hold-out, zero/one-shot transfer,
   per-task exponents; no source-state or training-history input; accuracy endpoint); Dettmers 2023 moved to the
   downstream-metric sentence; P² law credited for source-state inputs; Zhou et al. described with their held-out
   validation; Ghita et al. 2026 added. Evidence: docs/RELATED_WORK_VERIFICATION.md (verbatim quotes, arXiv versions).
6. **Methods listing untested templates** (D1–D4 dense candidates, M1–M3 hypotheses, KD capacity/teacher/data template,
   geometry reduced form, "basic-input law by substitution") → replaced by the predictors actually fit and frozen, with
   parameter counts, development sets, and register paths (§4.2; appendix A.4). The unsupported "leaving only basic
   parameters" sentence was deleted; the mechanism form is labeled a non-K0 comparator.
7. **Quantizer description** → qmax = 2^(b−1)−1, integers clamped to ±qmax (2^b−1 values), per-output-channel absmax
   scale; 4→5-bit squared step ratio (7/15)^2 stated as that case, general ratio (qmax(b)/qmax(b+1))^2; squared step ratio
   is not a loss ratio without further assumptions (appendix A.2, verified against v10_quantization.py).
8. **6.9B conclusion** → reported per protocol × capability × regime; protocol B exceptions kept (6.9B@112k int3 full 1.29
   vs 3.8–4.1; interpolation on par with source-free curves). Evidence: v49 compare files (Table 9, Fig. 2).
9. **D0 increment wording** → with {N0,L0} present, adding D0 has an interval above zero in one cell (quantization code;
   v36b); adding the dense anchor L0 to {N0,D0} clears zero in four of six. The earlier "4 of 12" came from the v36
   size+step split and is cited as such. Evidence: v36b input_pairs (Appendix Fig. 14A).
10. **Counts and wording** → 12 heterogeneous models + 2 prospectives, 14 quantization models, 10 geometry models from 5
    series / 4 families, generated macros (tables/counts.tex); recovery "asymptotic residual" → fitted plateau at 16M
    tokens; "resolves", "decisive", "free lunch", "left to future work" removed or bounded; Pythia-2.8B statement limited
    to the hash evidence.
11. **Loss definition and measurement boundaries** → token-weighted CE stated once with T_c; 64 measurement probes; native
    units; capability specificity as a prediction test (math/QA supported, code not detected: v26, Fig. 3); QA loss vs
    accuracy audit (v19); reference-condition variants (appendix A.1).
12. **Structure** → Problem and measurement (§3, with the information-budget table) → Study design and predictors (§4)
    → Results with one template per arm (§5.2–5.4) → Cross-arm synthesis and limitations (§5.5). Abstract rewritten
    (problem, design, three-arm evidence, limits, contribution); three contributions; selector as discussion only.
13. **Figures** → generated from result files: Fig. 1 measured responses vs frozen predictions (three arms, one source
    each); Fig. 2 transfer matrix by source × protocol × regime (+ distillation pool panel); Appendix Fig. 14 paired
    gains (full vs no-D0; source-conditioned vs source-free; power vs A1/A2); Appendix Fig. 3 measurement support.
    Scripts analysis/plot_fig1_responses.py … plot_fig4_measurement.py; docs/FIGURE_BUILD_NOTES.md lists every file read.

## Claims that changed (old → new → evidence)
- "QA most robust, invariant ordering" → series-specific ordering → v51 panel.json.
- "int3 collapses each of nine models" → 12/14 collapse; OLMo-3-32B does not → quant_losses.json.
- "source-conditioned predictors beat source-free curves for in-range sizes" → beat at seen densities and at interpolated
  densities on the development sources; match at a new in-range size (pooled stages; A2 best at 112k, median at 32k) →
  v42, v46, v49 (Table 3).
- "D0 improves in 4/12 cells (CI)" → 1/6 on the step-out split used in the table; adding L0 is the interval-supported
  input → v36b input_pairs.
- "Sengupta et al. report no held-out validation and no capability stratification" → true for v1 only; v2 has both →
  RELATED_WORK_VERIFICATION.md.
- "full wins at both bits on 1B@96k" (with swapped baselines) → int4 mixed, int3 ahead on aggregate with math-positive,
  code-negative pattern → v46 compare.json.

## Remaining scientific questions (max 5)
1. Why the source regression over-predicts QA at 6.9B: the QA response of larger sources does not follow the ≤1.4B size
   trend. Needs new measurements (more sizes between 1.4B and 6.9B, or a larger development panel). Experiment.
2. Whether any input carries the QA response at all (dense QA loss and tokens do not). Needs new inputs or panels.
   Experiment.
3. Whether the collapse-regime (int3) gain at in-range sizes survives more sources and other quantizers (GPTQ). Experiment
   (P4 in the plan, unfunded this round).
4. Whether the distillation reuse-count effect survives the absolute-exposure protocol that separates pool size from
   sampling protocol (P2-v2 matrix, running now; results to be integrated under a new data cut-off). Experiment in flight.
5. Code specificity of the endpoint: the secondary-benchmark test did not detect a within-capability gain for code.
   Partly text (state as boundary, done), partly a measurement question (more code benchmarks). Mixed.

## CPU reproduction (submission repo; requires only the mirrored JSON under data_mirror/ and the scripts under code/analysis/)
    python3 code/analysis/v51_panel_tables.py          # Tables 2, 5, counts.tex   (reads results/v6-*, v10-*)
    python3 code/analysis/v52_prediction_tables.py     # Tables 3, 6, 7, 8         (reads v36b, v38, v39, v41, v42, v44, v46, v49)
    python3 code/analysis/v49_p1v2_table.py            # Table 9
    python3 code/analysis/plot_fig1_responses.py; plot_fig2_gains.py; plot_fig3_transfer.py; plot_fig4_measurement.py
    tectonic main.tex                                  # or pdflatex+bibtex
Scripts resolve results/ relative to the repository root; in the submission repo the same files are under data_mirror/
with '@' replaced by '--' in file names (a one-line path map in code/analysis/README if needed). Model measurements
(prune_losses.json, quant_losses.json, eval.json) are not regenerable without weights and GPU; they are shipped as data.

## Not verified / not done in this round
- ICLR 2027 page limit and template rules were not checked online; the main text is ~12 pages including two figures
  and the panel and prediction tables, which likely exceeds the limit and needs a further cut once the rule is known.
- No new experiments; the P2-v2 matrix and any GPTQ comparison are future data.
- Per-byte normalization across tokenizers is stated as a limit, not applied (no per-sample byte counts for all panels).
