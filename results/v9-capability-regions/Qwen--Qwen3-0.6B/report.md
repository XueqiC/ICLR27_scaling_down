# V9 capability-region report — Qwen/Qwen3-0.6B

Benchmarks are ordered by capability (math, code, QA, control). The shared-component-removed metric subtracts `log(S)`, where `S` is the coordinatewise geometric mean over every benchmark, including C4.

## Benchmarks

| benchmark | capability | Fisher samples |
|---|---|---:|
| gsm8k | math | 64 |
| math500 | math | 64 |
| svamp | math | 64 |
| humaneval | code | 64 |
| mbpp | code | 64 |
| 2wiki | qa | 64 |
| hotpotqa | qa | 64 |
| triviaqa | qa | 64 |
| c4 | control | 61 |

## Raw Fisher cosine

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.7667 | 0.6125 | 0.5820 | 0.6527 | 0.7381 | 0.6508 | 0.7199 | 0.7782 |
| math500 | 0.7667 | 1.0000 | 0.5480 | 0.5701 | 0.7388 | 0.6875 | 0.5475 | 0.6551 | 0.8290 |
| svamp | 0.6125 | 0.5480 | 1.0000 | 0.4185 | 0.4610 | 0.5841 | 0.4783 | 0.5436 | 0.5530 |
| humaneval | 0.5820 | 0.5701 | 0.4185 | 1.0000 | 0.5687 | 0.5028 | 0.4298 | 0.4989 | 0.5572 |
| mbpp | 0.6527 | 0.7388 | 0.4610 | 0.5687 | 1.0000 | 0.5930 | 0.4605 | 0.5591 | 0.7727 |
| 2wiki | 0.7381 | 0.6875 | 0.5841 | 0.5028 | 0.5930 | 1.0000 | 0.6577 | 0.6119 | 0.6840 |
| hotpotqa | 0.6508 | 0.5475 | 0.4783 | 0.4298 | 0.4605 | 0.6577 | 1.0000 | 0.6025 | 0.5596 |
| triviaqa | 0.7199 | 0.6551 | 0.5436 | 0.4989 | 0.5591 | 0.6119 | 0.6025 | 1.0000 | 0.6655 |
| c4 | 0.7782 | 0.8290 | 0.5530 | 0.5572 | 0.7727 | 0.6840 | 0.5596 | 0.6655 | 1.0000 |

## Log-Fisher cosine

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.9980 | 0.9846 | 0.9969 | 0.9965 | 0.9763 | 0.9727 | 0.9707 | 0.9944 |
| math500 | 0.9980 | 1.0000 | 0.9809 | 0.9971 | 0.9967 | 0.9732 | 0.9692 | 0.9668 | 0.9943 |
| svamp | 0.9846 | 0.9809 | 1.0000 | 0.9844 | 0.9857 | 0.9898 | 0.9888 | 0.9909 | 0.9813 |
| humaneval | 0.9969 | 0.9971 | 0.9844 | 1.0000 | 0.9981 | 0.9771 | 0.9735 | 0.9720 | 0.9932 |
| mbpp | 0.9965 | 0.9967 | 0.9857 | 0.9981 | 1.0000 | 0.9788 | 0.9753 | 0.9741 | 0.9936 |
| 2wiki | 0.9763 | 0.9732 | 0.9898 | 0.9771 | 0.9788 | 1.0000 | 0.9934 | 0.9915 | 0.9794 |
| hotpotqa | 0.9727 | 0.9692 | 0.9888 | 0.9735 | 0.9753 | 0.9934 | 1.0000 | 0.9918 | 0.9761 |
| triviaqa | 0.9707 | 0.9668 | 0.9909 | 0.9720 | 0.9741 | 0.9915 | 0.9918 | 1.0000 | 0.9727 |
| c4 | 0.9944 | 0.9943 | 0.9813 | 0.9932 | 0.9936 | 0.9794 | 0.9761 | 0.9727 | 1.0000 |

## Shared-component-removed cosine

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.8910 | -0.5691 | 0.7752 | 0.7227 | -0.8470 | -0.8555 | -0.8645 | 0.5129 |
| math500 | 0.8910 | 1.0000 | -0.6600 | 0.8229 | 0.7822 | -0.8566 | -0.8732 | -0.8908 | 0.5577 |
| svamp | -0.5691 | -0.6600 | 1.0000 | -0.6012 | -0.5832 | 0.4838 | 0.5074 | 0.6338 | -0.6895 |
| humaneval | 0.7752 | 0.8229 | -0.6012 | 1.0000 | 0.8422 | -0.8193 | -0.8353 | -0.8225 | 0.4070 |
| mbpp | 0.7227 | 0.7822 | -0.5832 | 0.8422 | 1.0000 | -0.7930 | -0.8123 | -0.7868 | 0.3954 |
| 2wiki | -0.8470 | -0.8566 | 0.4838 | -0.8193 | -0.7930 | 1.0000 | 0.8037 | 0.7608 | -0.5064 |
| hotpotqa | -0.8555 | -0.8732 | 0.5074 | -0.8353 | -0.8123 | 0.8037 | 1.0000 | 0.7837 | -0.5253 |
| triviaqa | -0.8645 | -0.8908 | 0.6338 | -0.8225 | -0.7868 | 0.7608 | 0.7837 | 1.0000 | -0.6119 |
| c4 | 0.5129 | 0.5577 | -0.6895 | 0.4070 | 0.3954 | -0.5064 | -0.5253 | -0.6119 | 1.0000 |

## Top-0.1% coordinate Jaccard

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.5280 | 0.3752 | 0.3491 | 0.3321 | 0.3773 | 0.3383 | 0.3209 | 0.4028 |
| math500 | 0.5280 | 1.0000 | 0.3156 | 0.3765 | 0.3552 | 0.3551 | 0.3000 | 0.2731 | 0.3822 |
| svamp | 0.3752 | 0.3156 | 1.0000 | 0.2409 | 0.2509 | 0.2846 | 0.2845 | 0.3177 | 0.2884 |
| humaneval | 0.3491 | 0.3765 | 0.2409 | 1.0000 | 0.4756 | 0.2765 | 0.2522 | 0.2300 | 0.2927 |
| mbpp | 0.3321 | 0.3552 | 0.2509 | 0.4756 | 1.0000 | 0.2787 | 0.2492 | 0.2318 | 0.2957 |
| 2wiki | 0.3773 | 0.3551 | 0.2846 | 0.2765 | 0.2787 | 1.0000 | 0.3728 | 0.2827 | 0.4324 |
| hotpotqa | 0.3383 | 0.3000 | 0.2845 | 0.2522 | 0.2492 | 0.3728 | 1.0000 | 0.3239 | 0.3370 |
| triviaqa | 0.3209 | 0.2731 | 0.3177 | 0.2300 | 0.2318 | 0.2827 | 0.3239 | 1.0000 | 0.2878 |
| c4 | 0.4028 | 0.3822 | 0.2884 | 0.2927 | 0.2957 | 0.4324 | 0.3370 | 0.2878 | 1.0000 |

## Block-structure summary

Within-capability pairs exclude C4 because control has one benchmark; cross-capability pairs include all C4 comparisons. The block score is `(mean within - mean cross) / std(all off-diagonal pairs)`.

| metric | mean within | mean cross | std all pairs | block score | block structure? |
|---|---:|---:|---:|---:|---|
| Raw Fisher cosine | 0.6240 | 0.6025 | 0.1024 | 0.2104 | yes |
| Log-Fisher cosine | 0.9912 | 0.9825 | 0.0098 | 0.8880 | yes |
| Shared-component-removed cosine | 0.4075 | -0.2405 | 0.7140 | 0.9074 | yes |
| Top-0.1% coordinate Jaccard | 0.3820 | 0.3101 | 0.0660 | 1.0889 | yes |

Metrics showing positive block structure: Raw Fisher cosine, Log-Fisher cosine, Shared-component-removed cosine, Top-0.1% coordinate Jaccard.
Strongest standardized block structure: Top-0.1% coordinate Jaccard (block score 1.0889).
