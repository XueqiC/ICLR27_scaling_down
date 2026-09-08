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

Computed versions: V1, V2, V3. Bootstrap resamples: 10000.

| Capability | Predictor | Signed a MAE [95% CI] | Gain over V1 | Gain over zero | Gain over mean | Sign agreement [95% CI] |
|---|---|---|---|---|---|---|
| math | V1 | 0.82373 [0.45884, 1.27483] | 0.00000 [0.00000, 0.00000] | 0.25399 [-0.14870, 0.65802] | 0.20580 [-0.10892, 0.46836] | 1.00000 [1.00000, 1.00000] |
| math | V2 | 1.67390 [1.07318, 2.32237] | -0.85018 [-1.65222, -0.15734] | -0.59618 [-1.57234, 0.28179] | -0.64438 [-1.28677, -0.12464] | 0.81818 [0.54545, 1.00000] |
| math | V3 | 1.76425 [0.75890, 3.06885] | -0.94053 [-2.39810, 0.09490] | -0.68653 [-2.14744, 0.37571] | -0.73473 [-2.15664, 0.35624] | 0.90909 [0.72727, 1.00000] |
| math | zero | 1.07772 [0.48395, 1.76082] | -0.25399 [-0.65802, 0.14870] | 0.00000 [0.00000, 0.00000] | -0.04820 [-0.55710, 0.48373] | 0.00000 [0.00000, 0.00000] |
| math | population_mean | 1.02952 [0.68446, 1.43094] | -0.20580 [-0.46836, 0.10892] | 0.04820 [-0.48373, 0.55710] | 0.00000 [0.00000, 0.00000] | 1.00000 [1.00000, 1.00000] |
| code | V1 | 0.83013 [0.38200, 1.35764] | 0.00000 [0.00000, 0.00000] | 0.34307 [-0.12357, 0.86512] | 0.34458 [-0.11385, 0.74459] | 0.90909 [0.72727, 1.00000] |
| code | V2 | 1.64040 [0.96662, 2.30627] | -0.81026 [-1.62693, -0.10651] | -0.46719 [-1.46980, 0.50137] | -0.46569 [-1.10886, 0.09292] | 0.90909 [0.72727, 1.00000] |
| code | V3 | 2.08585 [0.84773, 3.71890] | -1.25572 [-2.98346, -0.12989] | -0.91265 [-2.76183, 0.48915] | -0.91114 [-2.65021, 0.43536] | 0.90909 [0.72727, 1.00000] |
| code | zero | 1.17320 [0.49816, 1.94347] | -0.34307 [-0.86512, 0.12357] | 0.00000 [0.00000, 0.00000] | 0.00151 [-0.56376, 0.57257] | 0.00000 [0.00000, 0.00000] |
| code | population_mean | 1.17471 [0.77476, 1.62354] | -0.34458 [-0.74459, 0.11385] | -0.00151 [-0.57257, 0.56376] | 0.00000 [0.00000, 0.00000] | 0.90909 [0.72727, 1.00000] |
| qa | V1 | 0.84527 [0.48179, 1.23916] | 0.00000 [0.00000, 0.00000] | -0.05515 [-0.50522, 0.38213] | 0.19761 [-0.17067, 0.53324] | 0.90909 [0.72727, 1.00000] |
| qa | V2 | 1.65022 [1.15139, 2.19256] | -0.80494 [-1.49237, -0.14870] | -0.86009 [-1.66445, -0.01128] | -0.60733 [-1.20886, 0.00593] | 0.63636 [0.36364, 0.90909] |
| qa | V3 | 1.42484 [0.75751, 2.25287] | -0.57957 [-1.56345, 0.11414] | -0.63471 [-1.63989, 0.13962] | -0.38196 [-1.21931, 0.21336] | 0.72727 [0.45455, 1.00000] |
| qa | zero | 0.79013 [0.26733, 1.38306] | 0.05515 [-0.38213, 0.50522] | 0.00000 [0.00000, 0.00000] | 0.25276 [-0.10189, 0.58701] | 0.00000 [0.00000, 0.00000] |
| qa | population_mean | 1.04289 [0.73143, 1.36904] | -0.19761 [-0.53324, 0.17067] | -0.25276 [-0.58701, 0.10189] | 0.00000 [0.00000, 0.00000] | 0.72727 [0.45455, 1.00000] |

### Qwen3-8B separate retrospective check

One model: report individual signed labels/predictions and correctness; no population CI.

| Capability | Observed signed a | Predictor | Predicted signed a | Correct sign |
|---|---|---|---|---|
| math | 0.05767 | V1 | 0.47271 | True |
| math | 0.05767 | V2 | 0.02572 | True |
| math | 0.05767 | V3 | -0.77229 | False |
| math | 0.05767 | zero | 0.00000 | False |
| math | 0.05767 | population_mean | 1.10331 | True |
| code | 0.10957 | V1 | 0.26603 | True |
| code | 0.10957 | V2 | -0.21027 | False |
| code | 0.10957 | V3 | -1.11523 | False |
| code | 0.10957 | zero | 0.00000 | False |
| code | 0.10957 | population_mean | 1.19870 | True |
| qa | -0.31731 | V1 | -0.16058 | True |
| qa | -0.31731 | V2 | -0.57294 | True |
| qa | -0.31731 | V3 | -1.21436 | True |
| qa | -0.31731 | zero | 0.00000 | False |
| qa | -0.31731 | population_mean | 0.65914 | False |

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
