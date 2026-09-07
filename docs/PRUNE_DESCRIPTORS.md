# Pruning descriptors and signed amplitude prediction

Advisor development audit. GPU extraction has not been run by this implementation task. Numeric results below use only the versions and source JSON listed in the audit artifact.

## Definition and fold discipline

We import V28's shape and shape fitter: ΔL_c(d)=a_c((1-d)/0.3)^gamma_c. Both fitted labels and predictions are signed; no log amplitude, absolute-value target, or clipping. Each outer model holdout refits shared gamma_c on the other development models at d=0.9/0.8/0.7/0.6. The held-out label is the least-squares projection of its curve onto that fixed source shape, used only for scoring. Consequently the estimand is amplitude MAE, not density-cell loss MAE.

V1 uses log(N0/1e9), family, dense L_c. V2 adds group (a); V3 adds group (b). V28's unchanged ridge helper supplies lambda=1 and an unpenalized intercept. Added columns use an exact Schur-complement extension of the same joint ridge objective. The fixed feature allowlist, nonconstant-column selection, scaling and family vocabulary use source rows only. Unseen families get V28's zero correction. No feature/hyperparameter is chosen from outer-fold outcomes. All versions use identical model folds; missing requested descriptors stop the audit rather than change the cohort.

The 12 development models are the V28 metadata whitelist. Qwen3-8B is excluded from every development fold and scored separately after fitting all 12. Its existing curve is retrospective advisor evidence, not a new prospective test. The registry resolves Qwen3-8B to Qwen/Qwen3-8B; the legacy loss JSON does not establish Base versus post-trained checkpoint identity. Extraction records its actual HF ID/revision; this limits interpretation of that comparison.

95% intervals use paired whole-model bootstrap draws with fits held fixed. Positive gain means comparator MAE minus candidate MAE. These intervals describe this panel, not retraining or measurement uncertainty. Sign agreement uses sign(a), with exact zero as a third class; negative-only agreement is also saved in JSON.

## Measurement and cost

Group (a) imports V6's deterministic proportional sample (seed 0, requested size 2,000,000) and quantiles. Retention counts use the exact strict abs(w)>threshold comparison in the original weight dtype. Ties and sampling mean achieved density can differ from requested density. We never mask or change the dense weights. Threshold scope includes embeddings/head as in V6; N0 excludes them and counts only 2-D language matrices as in V28.

Layers are decoder blocks (layers.N/blocks.N/h.N); other matrix-owning modules, including embeddings/head, are separate layers. Each density has equally weighted layer-retention mean, population variance, entropy, skew, max, min and Gini. The same statistics describe sampled absolute weights. Entropy is the natural-log Shannon entropy of normalized nonnegative mass; zero mass has entropy/Gini zero, constant vectors have skew zero. Raw counts, layer sizes, thresholds and achieved densities are retained for inspection.

Group (b) hooks dense decoder-block outputs (matrix owners for other architectures), excluding embeddings/head. It streams per-layer mean/variance of absolute activations, then summarizes each with the seven statistics. The default budget is 12 unpadded, non-generating forwards: first four even-index probes per capability from V6 build_probes(128, seed=0), round robin. Direct prompt/completion tokenization, prompt BOS only, each half truncated to 256 tokens. No backward pass or compressed loss is needed. Protocol, probe hash, input-token count, forward count and elapsed time are saved; mismatched extraction protocols are rejected.

| Version | Incremental measurement beyond existing dense L_c |
|---|---|
| V1 | Metadata and existing dense L_c; 0 extra forwards |
| V2 | V1 + dense weights only; 0 extra forwards; bytes/load/descriptor time recorded |
| V3 | V2 + measured N dense forwards (default N=12); activation time/tokens recorded separately |

The weight loader reuses V6 checkpoint-integrity checks without its unconditional sanity forward. Weight-only extraction does not load a tokenizer or probe dataset. Existing compatible features are reused; a later --with-activations call adds group (b), recording reload cost. Existing prune_losses.json is read only. Schema version 1 stores model, HF ID/revision, dtype, N0, family, optional dense_L_c, weights, activations and cost.

## Results

Computed versions: V1, V2. Bootstrap resamples: 10000.

Pending GPU descriptors: V3. No descriptor improvement is claimed before extraction.

| Capability | Predictor | Signed a MAE [95% CI] | Gain over V1 | Gain over zero | Gain over mean | Sign agreement [95% CI] |
|---|---|---|---|---|---|---|
| math | V1 | 0.87342 [0.47097, 1.38854] | 0.00000 [0.00000, 0.00000] | 0.28290 [-0.13168, 0.69918] | 0.20215 [-0.13778, 0.50435] | 1.00000 [1.00000, 1.00000] |
| math | V2 | 2.07263 [1.21617, 3.11078] | -1.19921 [-2.30364, -0.25247] | -0.91631 [-2.32679, 0.28352] | -0.99706 [-1.94188, -0.23823] | 0.80000 [0.50000, 1.00000] |
| math | zero | 1.15631 [0.51975, 1.88940] | -0.28290 [-0.69918, 0.13168] | 0.00000 [0.00000, 0.00000] | -0.08075 [-0.66526, 0.53057] | 0.00000 [0.00000, 0.00000] |
| math | population_mean | 1.07557 [0.70104, 1.49541] | -0.20215 [-0.50435, 0.13778] | 0.08075 [-0.53057, 0.66526] | 0.00000 [0.00000, 0.00000] | 1.00000 [1.00000, 1.00000] |
| code | V1 | 0.90071 [0.40097, 1.50082] | 0.00000 [0.00000, 0.00000] | 0.35725 [-0.17142, 0.94024] | 0.33249 [-0.19621, 0.79400] | 0.90000 [0.70000, 1.00000] |
| code | V2 | 2.15079 [1.21170, 3.21245] | -1.25008 [-2.47159, -0.22177] | -0.89283 [-2.37639, 0.43399] | -0.91759 [-1.87824, -0.12468] | 0.90000 [0.70000, 1.00000] |
| code | zero | 1.25795 [0.53097, 2.10268] | -0.35725 [-0.94024, 0.17142] | 0.00000 [0.00000, 0.00000] | -0.02476 [-0.68328, 0.64392] | 0.00000 [0.00000, 0.00000] |
| code | population_mean | 1.23319 [0.78803, 1.69049] | -0.33249 [-0.79400, 0.19621] | 0.02476 [-0.64392, 0.68328] | 0.00000 [0.00000, 0.00000] | 0.90000 [0.70000, 1.00000] |
| qa | V1 | 0.88284 [0.48346, 1.32185] | 0.00000 [0.00000, 0.00000] | -0.01849 [-0.49564, 0.45601] | 0.22090 [-0.19315, 0.59795] | 0.90000 [0.70000, 1.00000] |
| qa | V2 | 1.84449 [1.09655, 2.76421] | -0.96165 [-1.94164, -0.11733] | -0.98014 [-2.11774, 0.11303] | -0.74074 [-1.63608, 0.08909] | 0.60000 [0.30000, 0.90000] |
| qa | zero | 0.86435 [0.31448, 1.49722] | 0.01849 [-0.45601, 0.49564] | 0.00000 [0.00000, 0.00000] | 0.23940 [-0.18774, 0.64573] | 0.00000 [0.00000, 0.00000] |
| qa | population_mean | 1.10374 [0.76563, 1.43779] | -0.22090 [-0.59795, 0.19315] | -0.23940 [-0.64573, 0.18774] | 0.00000 [0.00000, 0.00000] | 0.70000 [0.40000, 1.00000] |

### Qwen3-8B separate retrospective check

One model: report individual signed labels/predictions and correctness; no population CI.

| Capability | Observed signed a | Predictor | Predicted signed a | Correct sign |
|---|---|---|---|---|
| math | 0.05773 | V1 | 0.53112 | True |
| math | 0.05773 | V2 | 0.22543 | True |
| math | 0.05773 | zero | 0.00000 | False |
| math | 0.05773 | population_mean | 1.18400 | True |
| code | 0.10974 | V1 | 0.29224 | True |
| code | 0.10974 | V2 | 0.29191 | True |
| code | 0.10974 | zero | 0.00000 | False |
| code | 0.10974 | population_mean | 1.28526 | True |
| qa | -0.31718 | V1 | -0.11769 | True |
| qa | -0.31718 | V2 | -0.76462 | True |
| qa | -0.31718 | zero | 0.00000 | False |
| qa | -0.31718 | population_mean | 0.72049 | False |

## GPU commands (emitted only; not executed)

Run from the repository root in an allocated GPU environment with the checkpoint available. Gemma and OLMo are hpg-eligible. Muse is also non-PRC and allowed by the existing registry/hpg wrapper. The existing hpg wrapper uses its tf5 environment for Gemma4/Muse. Qwen commands are rai-only: SDL_ALLOW_PRC=1 must never be set on hpg. These are 12 dev models plus the separate Qwen3-8B target. Commands include the optional 12-forward group so all three versions can be compared. For weights only, omit --with-activations (the budget flags have no effect without it).

```bash
python analysis/v32_prune_descriptors.py extract --model gemma3-270m --device cuda:0 --dtype bf16 --with-activations --activation-forwards 12 --activation-max-length 512
python analysis/v32_prune_descriptors.py extract --model gemma3-1b --device cuda:0 --dtype bf16 --with-activations --activation-forwards 12 --activation-max-length 512
python analysis/v32_prune_descriptors.py extract --model gemma3-4b --device cuda:0 --dtype bf16 --with-activations --activation-forwards 12 --activation-max-length 512
python analysis/v32_prune_descriptors.py extract --model gemma3-12b --device cuda:0 --dtype bf16 --with-activations --activation-forwards 12 --activation-max-length 512
python analysis/v32_prune_descriptors.py extract --model gemma3-27b --device cuda:0 --dtype bf16 --with-activations --activation-forwards 12 --activation-max-length 512
python analysis/v32_prune_descriptors.py extract --model gemma4-31b --device cuda:0 --dtype bf16 --with-activations --activation-forwards 12 --activation-max-length 512
python analysis/v32_prune_descriptors.py extract --model muse-30b --device cuda:0 --dtype bf16 --with-activations --activation-forwards 12 --activation-max-length 512
python analysis/v32_prune_descriptors.py extract --model olmo3-7b --device cuda:0 --dtype bf16 --with-activations --activation-forwards 12 --activation-max-length 512
python analysis/v32_prune_descriptors.py extract --model olmo3-32b --device cuda:0 --dtype bf16 --with-activations --activation-forwards 12 --activation-max-length 512
SDL_ALLOW_PRC=1 python analysis/v32_prune_descriptors.py extract --model Qwen3-0.6B --device cuda:0 --dtype bf16 --with-activations --activation-forwards 12 --activation-max-length 512
SDL_ALLOW_PRC=1 python analysis/v32_prune_descriptors.py extract --model Qwen3-1.7B --device cuda:0 --dtype bf16 --with-activations --activation-forwards 12 --activation-max-length 512
SDL_ALLOW_PRC=1 python analysis/v32_prune_descriptors.py extract --model Qwen3-4B --device cuda:0 --dtype bf16 --with-activations --activation-forwards 12 --activation-max-length 512
SDL_ALLOW_PRC=1 python analysis/v32_prune_descriptors.py extract --model Qwen3-8B --device cuda:0 --dtype bf16 --with-activations --activation-forwards 12 --activation-max-length 512
```

After extraction, run the CPU audit:

```bash
python analysis/v32_prune_descriptors.py predict --versions V1 V2 V3 --n-boot 10000
```

Before extraction, the existing-data baseline can be reproduced with:

```bash
python analysis/v32_prune_descriptors.py predict --versions V1 --n-boot 10000
```

Machine-readable folds, labels, predictions, costs and input SHA256 hashes are written to results/v32-descriptors/prediction/summary.json; the same report is saved there as report.md.
