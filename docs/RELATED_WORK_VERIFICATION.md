# Related-work verification against original papers (2026-09-09)

Scope: the five items requested (Sengupta "Compression Laws"; "Pruning Laws"; P2 Law; task-specific
distillation laws; Kumar / Zhou / Busbridge / Frantar / Panferov). Sources: arXiv abstract, HTML and PDF pages
only (no blogs). For 2504.04342 v1/v2 and 2502.08606 v2 the PDFs were downloaded and text-extracted locally
(`pdftotext`) so the quotes below are verbatim from the full text; other papers were read from arXiv HTML.
Dettmers and Zettlemoyer 2023 is added at the end because `related.tex` cites it in the same sentence as Kumar
et al. and the citation is mis-scoped.

Headline finding: arXiv 2504.04342 is ONE paper with two titles. v1 (6 Apr 2025) = "Compression Laws for Large
Language Models"; v2 (27 Aug 2026, "Accepted at EMNLP 2026") = "Pruning Laws for Large Language Models". The
sentence in `related.tex` describing `sengupta2025compression` is true only of v1 and is contradicted by the
version a reviewer will now see on arXiv.

## (a) Summary table

| paper | arXiv id / version / date | endpoint | source inputs | training-history control | calibration budget on target | held-out axes | methods covered | validated range | capability stratification |
|---|---|---|---|---|---|---|---|---|---|
| Sengupta et al., Compression Laws (v1) | 2504.04342v1, 6 Apr 2025 | intrinsic: test cross-entropy on WikiText2/PTB/Alpaca; extrinsic: *average* zero-shot accuracy over PIQA, WinoGrande, HellaSwag, ARC-e, ARC-c, MMLU | L0 (uncompressed score), compression ratio r, RFT tokens D | none (no pretraining-token or checkpoint axis) | n/a: law fit on all points (adj. R^2, F-stat) | none found (no unseen ratio / model / RFT-size prediction in text) | structured pruning: random-PruneNet (calibration-free), SliceGPT (calibration-based); RFT 1M-25M tokens | 8 models: Qwen-2.5 0.5B-14B, LLaMA-3.2-1B/3B, LLaMA-3-8B; r 10-90% | no: law fit on averaged accuracy; per-task values appear only in a radar figure |
| Sengupta et al., Pruning Laws (v2, same id) | 2504.04342v2, 27 Aug 2026, EMNLP 2026 | per-task score: accuracy (reasoning, QA) and 1/log(ppl) (LM), plus an "Average" | L0 (unpruned per-task score), pruning ratio r; no size/token term | none ("regardless of model size") | zero-shot: 0 target points; one-shot: 1 pruned point re-estimates P0 | rolling hold-out over ratios (train <= r, test > r); zero-shot to unseen models (LLaMA-3.1, Phi-3) and unseen pruners; one-shot | SparseGPT (unstructured), LaCo (depth), SliceGPT (width); zero/one-shot on SlimGPT, LLM-Pruner, SVD-LLM, PruneNet, ShortGPT; RFT refit (WikiText-2, LoRA r=8) | 10 dense 1.3B-30B (OPT, LLaMA-2/3.x, Phi-3-mini) + GPT-OSS-20B MoE; r 10-90%; avg RMSE < 7% | yes: separate (alpha, P0) per task and per category (reasoning / QA / LM) |
| Chen et al., P2 Law | 2411.10272v3, 26 May 2025, ACL 2025 main | aggregate post-training loss during continual pretraining (SlimPajama) | N0 (pre-pruning size), D (post-training tokens), rho (pruning rate), L0 (pre-pruning loss) | partial: L0 summarises the source; no pretraining-token axis | fit on 80% of checkpoints / smaller sizes / lower rates, then predict | tokens (last 20% of checkpoints), size (Qwen-2.5-0.5B+1.5B -> 3B), pruning rate (0.15, 0.25 -> 0.35) | depth, width, 2:4 pruning + post-training | Llama-3.2-1B/3B, Llama-3.1-8B, Qwen-2.5-0.5B/1.5B/3B; rho 0.15-0.35; 0.5-1B tokens | none (single aggregate loss) |
| Ghita et al., Task-Specific LLM Distillation | 2606.24747v2, 23 Aug 2026 (v1 23 Jun 2026) | in-domain macro-F1, gold-label NLL, Brier; general: MMLU, MMLU-Pro | dataset size, compression ratio, supervision format, pruning schedule | none | n/a | no parametric law and no extrapolation test; 3 seeds; fixed held-out test set (10k) | iterative structured pruning (depth+width) + logit KD or LoRA; label-only vs blended CoT | teacher Qwen3-32B; students 79/58/37/16% of teacher; ~6k-400k examples | in-domain vs general benchmarks only; no per-capability law |
| Kumar et al., Scaling Laws for Precision | 2411.04330v2, 30 Nov 2024, ICLR 2025 | validation loss (Dolma v1.7) | N, D, training precisions (Pw, Pa, Pkv), post-train precision Ppost | yes: D is an explicit input; PTQ damage grows with D | n/a ("fit and plot on the same data") | size/tokens: fit 30-220M, validate up to 1.7B / 26B tokens | GPTQ PTQ (+2 others, App. F); low-precision training / QAT | N 30-220M (fit), D 1.5-26B, 3-16 bits | none ("without downstream model evaluations") |
| Zhou et al., Task-Stratified PTQ laws | 2508.18609v4, 21 Apr 2026 (v1 26 Aug 2025), Findings of ACL 2026 | -ln(baseline-adjusted accuracy) per stratum (memorisation / application / reasoning) and general | N, bit-width B, calibration-set size Cb, group size G | none (no FP16 baseline, no pretraining tokens) | fit on Qwen3 0.6B-14B; no per-model calibration | model size (Qwen3-32B held out); cross-family Llama-3 1B/3B/8B (42 configs) | GPTQ weight-only PTQ; 3/4-bit grid over Cb, G; 8-bit fixed; 2-bit excluded | 293 Qwen3 configs + 42 Llama-3; MAE 0.03-0.09 | yes: three knowledge strata with separate exponents; qualitative mechanism (FFN key-value lookup); no recovery-training axis |
| Busbridge et al., Distillation Scaling Laws | 2502.08606v2, 25 Jul 2025, ICML 2025 | student validation cross-entropy w.r.t. data (C4) | N_S, D_S, teacher loss L_T (which summarises N_T, D_T) | yes: teacher tokens via L_T; student tokens D_S | n/a | extrapolation weaker -> stronger models (<= 1% relative error) | logit KD during pretraining | students/teachers 143M-12.6B; up to 512B tokens | no law on downstream; App. E.1 shows downstream accuracy only as a proxy check of cross-entropy |
| Frantar et al., Sparsely-Connected FMs | 2309.08520v1, 15 Sep 2023, ICLR 2024 | validation loss (T5/C4 MLM; ViT/JFT-4B) | sparsity S, non-zero params N, data D | yes: D; also pruning from pretrained checkpoints vs from scratch | n/a | size: extrapolate to 2.3B T5 at 75% sparsity (~6.75x non-zero params) | gradual magnitude sparse training, n:m, from pretrained dense | ViT <= 42.4M nz, T5 <= 85M nz (fit); S = 50/75/87.5% | none (pretraining loss only) |
| Panferov et al., Unified Compressed Representations | 2506.01863v1, 2 Jun 2025 | C4 validation loss | N, D, capacity rho(R) from Gaussian MSE | D fixed at 100 tokens/param | n/a | new formats predicted from GMSE; fit MSE reported (Table 1); no larger-model extrapolation stated | QAT scalar/vector quantisation, unstructured/2:4 sparsity, hybrids (training-time, not PTQ) | 30-200M Llama-style; 1-8 bit | none |
| Dettmers & Zettlemoyer, k-bit Inference Scaling Laws | 2212.09720v2, 28 Feb 2023, ICML 2023 | zero-shot accuracy vs total model bits | model size, bit precision, block size, data type | none | n/a | none stated in abstract | zero-shot PTQ 3-8 bit | 19M-176B (BLOOM, OPT, NeoX/Pythia, GPT-2) | no (accuracy law, not loss law) |

## (b) Supporting quotes (verbatim, short)

### 1. Sengupta et al. 2025, Compression Laws (v1) -- https://arxiv.org/abs/2504.04342v1
- Endpoints: "methods for estimating the intrinsic (e.g., test cross-entropy loss) and extrinsic (e.g., zero-shot test accuracy) performance of LLMs post-compression."
- Extrinsic is an average: "the extrinsic performance (average zero-shot accuracy)."
- Inputs and methods: "compressed using both calibration-free ... and calibration-based ... structured pruning methods, with compression ratios ranging from 10% to 90% and recovery fine-tuning token sizes varying from 1M to 25M."
- Fit criterion (no held-out): Table 1, "Higher adjusted R2 and F-statistics indicate better goodness-of-fit for the functional form L = f(L0, r, D)". A full-text search for predict / unseen / held-out / extrapolat / generaliz / cross-valid returns no sentence in which the fitted law predicts a configuration outside the fitting set (the only "predictability" hit refers to inference speed-up curves).

### 2. Sengupta et al. 2026, Pruning Laws (v2, same arXiv id) -- https://arxiv.org/abs/2504.04342v2
- Endpoint and task groups: "The tasks span three categories: Reasoning: PIQA, WinoGrande, HellaSwag, ARC-e and ARC-c. Question-answering: CoQA. Language modeling: WikiText and LAMBADA." / "For language modeling tasks, we compute perplexity ... we report the inverse log-perplexity (i.e., 1/log(ppl)) to map the metric to a (0,1) range."
- Form and inputs: "L(L0, r) = L0 P0 (1 - r)^alpha" -- "This connects the pruned model's performance L to its unpruned baseline L0 and the retention ratio (1 - r)". No model-size or token term appears in the law; "regardless of model size, performance consistently collapses ... once pruning ratios exceed 70-80%".
- Rolling hold-out: "for each pruning ratio r in [20%, 90%], we train the pruning laws on pruning ratios {10%, ..., r%} and test the pruning laws on {r%+10%, ... 90%}. Finally, we calculate the average root mean square error (RMSE)".
- Zero-/one-shot transfer: "zero-shot extrapolation, where we use the fitted parametric functions to test on unseen data, and one-shot extrapolation, where we use the alpha coefficient from the fitted parametric function, but the bias term P0 is re-estimated from pruning the model on a single pruning ratio." / "When applied zero-shot to unseen architectures like LLaMA-3.1 and Phi-3, our laws predict performance with low error (0.09-0.12)."
- Task sensitivity: "reasoning tasks are remarkably resilient ... (alpha ~ 0.22) ... QA tasks are highly fragile (alpha ~ 2.42) ... Language modeling falls in between (alpha ~ 0.73)."
- Recovery: "Table 4 reports the pruning law coefficients fitted on models after they have undergone recovery fine-tuning on the WikiText-2 dataset".
- Range: "our empirical validation is conducted on models ranging from 1.3B to 30B parameters, together with a single 20B mixture-of-experts model."

### 3. Chen et al. 2025, P2 Law -- https://arxiv.org/abs/2411.10272 (v3)
- Abstract: "a scaling law identifying four key factors for predicting the pruned model's post-training loss: the model size before pruning, the number of post-training tokens, the pruning rate, and the model's loss before pruning."
- Held-out: "the first 80% of the checkpoints recorded during each training process are used to fit the P2 Law, and the remaining 20% for validation." / "fit the P2 Law using all checkpoints from Qwen-2.5-0.5B and Qwen-2.5-1.5B, and subsequently validate it with the actual checkpoints of Qwen-2.5-3B." / "fit ... at lower pruning rates (0.15 and 0.25) and then validate it ... at a higher pruning rate of 0.35."
- Methods: "depth pruning, width pruning, 2:4 semi-structured pruning" on Llama-3 and Qwen-2.5 series; post-training on SlimPajama. No per-task evaluation.

### 4. Ghita, Desai, Boier 2026, Scaling Laws for Task-Specific LLM Distillation -- https://arxiv.org/abs/2606.24747 (v2)
- Scope: "This paper derives empirical scaling laws for domain-specific LLM compression, quantifying how in-domain and general-knowledge performance scale with dataset size, compression ratio, supervision format, and iterative pruning schedule."
- Finding: "In-domain task quality degrades predictably under compression while general-knowledge benchmarks collapse well before the same point"; "blended chain-of-thought supervision loss that stabilizes KL-divergence distillation over reasoning traces."
- Method: "Starting from the teacher, we reduce model size by a constant step size in percentage points ... to produce intermediate student checkpoints." Teacher "Qwen3-32B"; students at 79/58/37/16% of the teacher.
- No closed-form law: results are curves and tables (Figures 4-9, Tables 1-3); no fitted exponents, R^2, or extrapolation test are reported (HTML read; no equation found).

### 5a. Kumar et al. 2024, Scaling Laws for Precision -- https://arxiv.org/abs/2411.04330 (v2)
- Abstract: "the degradation introduced by post-training quantization increases as models are trained on more data, eventually making additional pretraining data actively harmful." / "We fit on over 465 pretraining runs and validate our predictions on model sizes up to 1.7B parameters trained on up to 26B tokens."
- Setup: "N in [30, 60, 110, 220] million parameters (non-embedding) and D in [1.5, 3, 6, 13, 26] billion tokens." / "use GPTQ to post-train quantize them, replicating our findings with two other methods in Appendix F."
- Validation style: "we fit and plot on the same data, as is standard in scaling laws." (App. C)
- Downstream: "Third, we only consider loss scaling without downstream model evaluations." (Conclusion and Limitations)

### 5b. Zhou et al. 2025/2026, Task-Stratified Knowledge Scaling Laws for PTQ LLMs -- https://arxiv.org/abs/2508.18609 (v4)
- Abstract: "By stratifying capabilities into memorization, application, and reasoning, we develop a framework that unifies model size, bit-width, and fine-grained factors: group size and calibration set size. Validated on 293 diverse PTQ configurations, our framework demonstrates strong fit and cross-architecture consistency."
- Held-out: "Qwen3-32B is reserved to validate the extrapolation of our proposed laws." Fit sizes "0.6B, 1.7B, 4B, 8B, and 14B". Cross-family: "extend the evaluation to the Llama-3 family (1B, 3B, 8B) ... a representative subset of 42 configurations".
- Method: GPTQ only; grid "Cb in {8, 32, 128, 1024}, G in {32, 64, 128, 1024}" at 3/4-bit. No fine-tuning / QAT / recovery axis found in text.
- Mechanism (qualitative only): "We attribute this to KM's reliance on precise activation alignment to trigger Key-Value pairs in FFN layers." No Hessian/Fisher/curvature term enters the law.

### 5c. Busbridge et al. 2025, Distillation Scaling Laws -- https://arxiv.org/abs/2502.08606 (v2)
- Endpoint: "By cross-entropy, we always mean with respect to data, not the teacher." / "In all settings, we optimize for and predict validation cross-entropy." (App. E.1)
- Form: "One of our main contributions is that the student loss follows a broken power law, where the transition between the two ..." (App.)
- Validation: "fit the observations at the level of <= 1% relative prediction error, including when extrapolated from weaker to stronger models (see Figure 5b)."
- Downstream: "To confirm that the validation cross-entropy is a good proxy for the downstream evaluation ... in Figure 35 we show evaluations ... on downstream evaluation tasks. ARC Easy, ARC Challenge, HellaSwag, Piqa, Sciq, WinoGrande and Lambada OpenAI are zero-shot tasks. TriviaQA and WebQS are one-shot tasks." / Limitations: "Our performance over standard English language downstream tasks closely follows cross-entropy, however, C4 is not as well suited ... to probe aspects like reasoning performance."
- Range: "Models have sizes which range from 143M to 12.6B parameters, and we allow the teacher to be smaller or larger than the student."

### 5d. Frantar et al. 2023, Scaling Laws for Sparsely-Connected Foundation Models -- https://arxiv.org/abs/2309.08520 (v1)
- Form: "L(S,N,D) = (aS (1-S)^bS + cS) (1/N)^bN + (aD/D)^bD + c"; "sparsity affects each model size in a similar way, i.e., as a multiplicative constant to the size scaling."
- Extrapolation: "we evaluate extrapolation performance by pruning a 2.3 billion parameter model to 75% sparsity. This constitutes an ~6.75x larger target number of non-zero parameters than the maximum in our fitting data ... the prediction of our fitted scaling law is quite close to the actual validation loss."
- Downstream: "We deliberately consider the pretraining loss and infinite data setting to assess the effectiveness of sparsity in its most challenging ... yet also most useful application".
- Pretrained start: "sparsifying from scratch requires 4.90x, 4.27x, and 2.45x more data for 50%, 75%, and 87.5% sparsity respectively to match pruning from pretrained checkpoints".

### 5e. Panferov et al. 2025, Unified Scaling Laws for Compressed Representations -- https://arxiv.org/abs/2506.01863 (v1)
- Abstract: "there exists a simple 'capacity' metric -- based on the representation's ability to fit random Gaussian data -- which can robustly predict parameter efficiency across multiple compressed representations."
- Setup: "we pretrained decoder-only Transformers following the Llama architecture for 30M, 50M, 100M and 200M non-embedding parameters." / "We follow standard quantization-aware training (QAT) methods, combined with various levels of unstructured weight sparsity."
- Capacity: "rho(R) is a simple parametric function of the MSE of the representation R when fitting random Gaussian data". No downstream evaluation found in text.

### Extra. Dettmers & Zettlemoyer 2023, The case for 4-bit precision -- https://arxiv.org/abs/2212.09720 (v2)
- Abstract: "developing inference scaling laws of zero-shot performance in Large Language Models (LLMs) to determine the bit-precision and model size that maximizes zero-shot performance." / "4-bit precision is almost universally optimal for total model bits and zero-shot accuracy."

## (c) Statements in related.tex not supported by the originals, with suggested wording

C1. `related.tex` par. 1: "Empirical claims of compression laws with downstream metrics exist \citep{sengupta2025compression} but report no held-out predictive validation and no capability stratification."
- Status: TRUE for v1 only (no out-of-sample prediction found; law fit on averaged accuracy). FALSE for the current arXiv version v2 (retitled "Pruning Laws for Large Language Models", EMNLP 2026): rolling hold-out over pruning ratios, zero-shot transfer to unseen models (LLaMA-3.1, Phi-3) and unseen pruners, one-shot calibration, average extrapolation error < 7%, and per-task / per-category coefficients. A reviewer opening 2504.04342 sees v2.
- Also imprecise: v1's endpoints are test cross-entropy AND average zero-shot accuracy, so "with downstream metrics" understates it, and v1 includes a recovery-fine-tuning token axis D.
- Suggested wording: "Sengupta et al. (2025) fit compression laws on test loss and averaged zero-shot accuracy against compression ratio and recovery tokens, evaluated by in-sample goodness of fit only; the revised version (Pruning Laws, EMNLP 2026) adds rolling hold-out over pruning ratios and zero-/one-shot transfer to unseen models with per-task exponents. Both take the unpruned score and the pruning ratio as the only inputs: no source-state or training-history term, an accuracy (or inverse log-perplexity) endpoint rather than a per-capability loss, and no mechanism."
- Bib action: `references.bib` entry `sengupta2025compression` should either pin v1 ("arXiv:2504.04342v1") if the v1 content is what is described, or be updated to the v2 title "Pruning Laws for Large Language Models" (EMNLP 2026) with the claim revised as above. Note the repo's own `docs/P0_RELATED_CONTRAST.md` already lists the v2 title under the same id.

C2. `related.tex` par. 1: "task-stratified PTQ laws \citep{zhou2025task} are the closest antecedent, limited to quantization without a mechanism or recovery axis."
- Status: MOSTLY SUPPORTED (GPTQ weight-only, no fine-tuning / recovery axis, no mechanistic term in the law). Two caveats: (i) the paper does offer a qualitative mechanism (FFN key-value lookup for memorisation), so "without a mechanism" should read "without a mechanistic term" or "with only a post-hoc explanation"; (ii) it DOES validate by extrapolation (held-out Qwen3-32B; cross-family Llama-3), so the closing contrast "Our framework ... validates them by extrapolation" is not a differentiator against Zhou et al.
- Suggested wording: "task-stratified PTQ laws (Zhou et al.) are the closest antecedent: an accuracy endpoint on one quantiser (GPTQ) with model size, bit-width, group size and calibration-set size as inputs, validated by one held-out larger model; they carry no source-state input, no recovery axis, and only a post-hoc explanation of the strata."

C3. `related.tex` par. 1: "precision as a saturating effective-parameter count with a post-training degradation term \citep{kumar2024scaling, dettmers2023case}" under the heading "laws of pretraining loss".
- Status: NOT SUPPORTED for Dettmers & Zettlemoyer: their law is zero-shot accuracy versus total model bits; it has no effective-parameter form, no post-training degradation term, and is not a loss law. Kumar et al. alone supports the clause.
- Suggested wording: cite `kumar2024scaling` alone for that clause and move `dettmers2023case` to the "downstream metrics exist" sentence, e.g. "k-bit inference scaling of zero-shot accuracy \citep{dettmers2023case}".

C4. `related.tex` par. 1: "None of these validate their forms on downstream capabilities."
- Status: DEFENSIBLE but loose. Busbridge et al. do report downstream accuracy (App. E.1) as a proxy check of cross-entropy, without fitting the law to it; `muralidharan2024compact` and `xia2023sheared` propose no law at all, so "their forms" does not apply to them.
- Suggested wording: "None of these fit or validate their functional forms on per-capability downstream measurements; Busbridge et al. report downstream accuracy only as a proxy check of cross-entropy (their App. E.1)."

C5. `related.tex` par. 1: "pruned models admit post-training recovery laws \citep{zhang2024scaling}".
- Status: SUPPORTED. Note for the intro/method: P2 Law already uses the pre-pruning loss L0 and size N0 as inputs and validates on held-out tokens / size / pruning rate, so "source-state conditioning helps" must not be claimed as new (consistent with `docs/P0_RELATED_CONTRAST.md`). Bib key `zhang2024scaling` renders as "Chen et al., 2025" (first author Xiaodong Chen); harmless but confusing.

C6. `related.tex` par. 1: "sparse/quantized formats unify under a single capacity scalar \citep{panferov2025unified}".
- Status: SUPPORTED, with a scope note: Panferov et al. train models in the compressed format (QAT / sparse training, 30-200M); it is not a post-training compression law. If the paragraph is meant to be about PTQ/pruning of a trained model, add "for training in compressed formats".

C7. Verified as SUPPORTED: "sparsity acts as an effective-parameter multiplier \citep{frantar2023scaling}" (multiplicative constant on the size term); "distillation obeys a broken power law in teacher loss \citep{busbridge2025distillation}" (verbatim in the paper).

## (d) Could not verify / limits of this check
- Busbridge App. E.1 and the "broken power law" sentence were read from the locally extracted PDF text; the arXiv HTML page is truncated before the appendix.
- Ghita et al. (2606.24747): the claim "no parametric law is fitted" rests on an HTML read of v2; no equation, R^2 or extrapolation statement was found, but the 24-page PDF was not text-extracted. Treat as "none found", not "none exists".
- Panferov et al.: no larger-model extrapolation or downstream evaluation was found in the HTML; the paper's appendix was not separately searched.
- P2 Law: the text does not state whether the endpoint is training or validation loss on SlimPajama.
- Sengupta v1: the "no held-out prediction" finding is a full-text keyword search of the extracted PDF (predict / unseen / held-out / extrapolat / generaliz / cross-valid); figures were not inspected.
- No OpenReview reviews were consulted; venue attributions come from the arXiv comment fields (EMNLP 2026, ACL 2025, Findings ACL 2026, ICML 2025, ICLR 2025/2024).
