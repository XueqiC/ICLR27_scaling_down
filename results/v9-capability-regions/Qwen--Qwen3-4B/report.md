# V9 capability-region report — Qwen/Qwen3-4B

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
| gsm8k | 1.0000 | 0.8768 | 0.7840 | 0.5359 | 0.6271 | 0.7121 | 0.7349 | 0.5610 | 0.4935 |
| math500 | 0.8768 | 1.0000 | 0.7208 | 0.5532 | 0.6578 | 0.7359 | 0.7746 | 0.5875 | 0.5368 |
| svamp | 0.7840 | 0.7208 | 1.0000 | 0.5176 | 0.5743 | 0.6524 | 0.6677 | 0.6297 | 0.4306 |
| humaneval | 0.5359 | 0.5532 | 0.5176 | 1.0000 | 0.6615 | 0.5167 | 0.5282 | 0.4807 | 0.3623 |
| mbpp | 0.6271 | 0.6578 | 0.5743 | 0.6615 | 1.0000 | 0.5995 | 0.6372 | 0.5166 | 0.4443 |
| 2wiki | 0.7121 | 0.7359 | 0.6524 | 0.5167 | 0.5995 | 1.0000 | 0.7834 | 0.5758 | 0.4846 |
| hotpotqa | 0.7349 | 0.7746 | 0.6677 | 0.5282 | 0.6372 | 0.7834 | 1.0000 | 0.6072 | 0.5143 |
| triviaqa | 0.5610 | 0.5875 | 0.6297 | 0.4807 | 0.5166 | 0.5758 | 0.6072 | 1.0000 | 0.3955 |
| c4 | 0.4935 | 0.5368 | 0.4306 | 0.3623 | 0.4443 | 0.4846 | 0.5143 | 0.3955 | 1.0000 |

## Log-Fisher cosine

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.9981 | 0.9657 | 0.9976 | 0.9973 | 0.9819 | 0.9818 | 0.9812 | 0.9962 |
| math500 | 0.9981 | 1.0000 | 0.9585 | 0.9976 | 0.9974 | 0.9781 | 0.9781 | 0.9771 | 0.9964 |
| svamp | 0.9657 | 0.9585 | 1.0000 | 0.9649 | 0.9645 | 0.9867 | 0.9857 | 0.9891 | 0.9583 |
| humaneval | 0.9976 | 0.9976 | 0.9649 | 1.0000 | 0.9983 | 0.9817 | 0.9816 | 0.9812 | 0.9955 |
| mbpp | 0.9973 | 0.9974 | 0.9645 | 0.9983 | 1.0000 | 0.9821 | 0.9820 | 0.9815 | 0.9958 |
| 2wiki | 0.9819 | 0.9781 | 0.9867 | 0.9817 | 0.9821 | 1.0000 | 0.9950 | 0.9939 | 0.9808 |
| hotpotqa | 0.9818 | 0.9781 | 0.9857 | 0.9816 | 0.9820 | 0.9950 | 1.0000 | 0.9935 | 0.9813 |
| triviaqa | 0.9812 | 0.9771 | 0.9891 | 0.9812 | 0.9815 | 0.9939 | 0.9935 | 1.0000 | 0.9791 |
| c4 | 0.9962 | 0.9964 | 0.9583 | 0.9955 | 0.9958 | 0.9808 | 0.9813 | 0.9791 | 1.0000 |

## Shared-component-removed cosine

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.9258 | -0.8734 | 0.8586 | 0.8396 | -0.9057 | -0.9025 | -0.8989 | 0.8000 |
| math500 | 0.9258 | 1.0000 | -0.9271 | 0.8937 | 0.8858 | -0.9155 | -0.9110 | -0.9198 | 0.8375 |
| svamp | -0.8734 | -0.9271 | 1.0000 | -0.8736 | -0.8853 | 0.8465 | 0.8306 | 0.8787 | -0.9121 |
| humaneval | 0.8586 | 0.8937 | -0.8736 | 1.0000 | 0.8932 | -0.8926 | -0.8923 | -0.8809 | 0.7537 |
| mbpp | 0.8396 | 0.8858 | -0.8853 | 0.8932 | 1.0000 | -0.8836 | -0.8787 | -0.8739 | 0.7708 |
| 2wiki | -0.9057 | -0.9155 | 0.8465 | -0.8926 | -0.8836 | 1.0000 | 0.8747 | 0.8459 | -0.8018 |
| hotpotqa | -0.9025 | -0.9110 | 0.8306 | -0.8923 | -0.8787 | 0.8747 | 1.0000 | 0.8354 | -0.7765 |
| triviaqa | -0.8989 | -0.9198 | 0.8787 | -0.8809 | -0.8739 | 0.8459 | 0.8354 | 1.0000 | -0.8340 |
| c4 | 0.8000 | 0.8375 | -0.9121 | 0.7537 | 0.7708 | -0.8018 | -0.7765 | -0.8340 | 1.0000 |

## Top-0.1% coordinate Jaccard

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.3983 | 0.3649 | 0.2768 | 0.2592 | 0.2659 | 0.2447 | 0.2230 | 0.2633 |
| math500 | 0.3983 | 1.0000 | 0.2691 | 0.2919 | 0.2761 | 0.2631 | 0.2398 | 0.1981 | 0.2629 |
| svamp | 0.3649 | 0.2691 | 1.0000 | 0.2061 | 0.1964 | 0.2031 | 0.1903 | 0.2071 | 0.2006 |
| humaneval | 0.2768 | 0.2919 | 0.2061 | 1.0000 | 0.3350 | 0.2098 | 0.1956 | 0.1772 | 0.2099 |
| mbpp | 0.2592 | 0.2761 | 0.1964 | 0.3350 | 1.0000 | 0.2023 | 0.1906 | 0.1665 | 0.2108 |
| 2wiki | 0.2659 | 0.2631 | 0.2031 | 0.2098 | 0.2023 | 1.0000 | 0.3251 | 0.2134 | 0.2711 |
| hotpotqa | 0.2447 | 0.2398 | 0.1903 | 0.1956 | 0.1906 | 0.3251 | 1.0000 | 0.2042 | 0.2691 |
| triviaqa | 0.2230 | 0.1981 | 0.2071 | 0.1772 | 0.1665 | 0.2134 | 0.2042 | 1.0000 | 0.2063 |
| c4 | 0.2633 | 0.2629 | 0.2006 | 0.2099 | 0.2108 | 0.2711 | 0.2691 | 0.2063 | 1.0000 |

## Block-structure summary

Within-capability pairs exclude C4 because control has one benchmark; cross-capability pairs include all C4 comparisons. The block score is `(mean within - mean cross) / std(all off-diagonal pairs)`.

| metric | mean within | mean cross | std all pairs | block score | block structure? |
|---|---:|---:|---:|---:|---|
| Raw Fisher cosine | 0.7157 | 0.5677 | 0.1169 | 1.2657 | yes |
| Log-Fisher cosine | 0.9861 | 0.9839 | 0.0114 | 0.1991 | yes |
| Shared-component-removed cosine | 0.3678 | -0.2291 | 0.8606 | 0.6935 | yes |
| Top-0.1% coordinate Jaccard | 0.3014 | 0.2268 | 0.0527 | 1.4153 | yes |

Metrics showing positive block structure: Raw Fisher cosine, Log-Fisher cosine, Shared-component-removed cosine, Top-0.1% coordinate Jaccard.
Strongest standardized block structure: Top-0.1% coordinate Jaccard (block score 1.4153).
