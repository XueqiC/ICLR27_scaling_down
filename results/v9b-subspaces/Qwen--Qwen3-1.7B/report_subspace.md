# V9b gradient-subspace report — Qwen/Qwen3-1.7B

Each benchmark subspace uses up to the top 8 right singular vectors of its centered per-sample projected-gradient matrix. Similarity is the mean squared cosine of the principal angles.

## Benchmarks

| benchmark | capability | samples | effective rank |
|---|---|---:|---:|
| gsm8k | math | 64 | 8 |
| math500 | math | 64 | 8 |
| svamp | math | 64 | 8 |
| humaneval | code | 64 | 8 |
| mbpp | code | 64 | 8 |
| 2wiki | qa | 64 | 8 |
| hotpotqa | qa | 64 | 8 |
| triviaqa | qa | 64 | 8 |
| c4 | control | 61 | 8 |

## Principal-angle similarity

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.0911 | 0.0836 | 0.0192 | 0.0278 | 0.0193 | 0.0282 | 0.0134 | 0.0073 |
| math500 | 0.0911 | 1.0000 | 0.0824 | 0.0312 | 0.0474 | 0.0240 | 0.0373 | 0.0097 | 0.0102 |
| svamp | 0.0836 | 0.0824 | 1.0000 | 0.0108 | 0.0173 | 0.0107 | 0.0141 | 0.0149 | 0.0060 |
| humaneval | 0.0192 | 0.0312 | 0.0108 | 1.0000 | 0.0457 | 0.0165 | 0.0204 | 0.0075 | 0.0071 |
| mbpp | 0.0278 | 0.0474 | 0.0173 | 0.0457 | 1.0000 | 0.0228 | 0.0293 | 0.0128 | 0.0101 |
| 2wiki | 0.0193 | 0.0240 | 0.0107 | 0.0165 | 0.0228 | 1.0000 | 0.0566 | 0.0103 | 0.0114 |
| hotpotqa | 0.0282 | 0.0373 | 0.0141 | 0.0204 | 0.0293 | 0.0566 | 1.0000 | 0.0125 | 0.0111 |
| triviaqa | 0.0134 | 0.0097 | 0.0149 | 0.0075 | 0.0128 | 0.0103 | 0.0125 | 1.0000 | 0.0041 |
| c4 | 0.0073 | 0.0102 | 0.0060 | 0.0071 | 0.0101 | 0.0114 | 0.0111 | 0.0041 | 1.0000 |

## Block-structure summary

The block score is `(mean within - mean cross) / std(all off-diagonal pairs)`, with C4 included only among cross-capability pairs.

| mean within | mean cross | std all pairs | block score |
|---:|---:|---:|---:|
| 0.0546 | 0.0173 | 0.0221 | 1.6893 |

## Joint sample-level clustering

The joint centering and SVD include every benchmark sample. C4 scores are then excluded, and the remaining samples in the scores in the top 10 components are clustered with k-means (k=3).

Adjusted Rand index versus capability: **0.0039**.

Cluster sizes: [465, 8, 39]; inertia: 6636.25.
