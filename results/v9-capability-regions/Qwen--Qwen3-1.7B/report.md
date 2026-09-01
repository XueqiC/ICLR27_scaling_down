# V9 capability-region report — Qwen/Qwen3-1.7B

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
| gsm8k | 1.0000 | 0.8547 | 0.8153 | 0.8092 | 0.7270 | 0.5833 | 0.7265 | 0.7033 | 0.7325 |
| math500 | 0.8547 | 1.0000 | 0.7489 | 0.8282 | 0.7266 | 0.5824 | 0.7778 | 0.7165 | 0.7653 |
| svamp | 0.8153 | 0.7489 | 1.0000 | 0.7973 | 0.7186 | 0.6247 | 0.6901 | 0.7321 | 0.7303 |
| humaneval | 0.8092 | 0.8282 | 0.7973 | 1.0000 | 0.7969 | 0.6559 | 0.7866 | 0.7803 | 0.8279 |
| mbpp | 0.7270 | 0.7266 | 0.7186 | 0.7969 | 1.0000 | 0.5708 | 0.7170 | 0.6506 | 0.7512 |
| 2wiki | 0.5833 | 0.5824 | 0.6247 | 0.6559 | 0.5708 | 1.0000 | 0.6520 | 0.6062 | 0.6494 |
| hotpotqa | 0.7265 | 0.7778 | 0.6901 | 0.7866 | 0.7170 | 0.6520 | 1.0000 | 0.7088 | 0.8276 |
| triviaqa | 0.7033 | 0.7165 | 0.7321 | 0.7803 | 0.6506 | 0.6062 | 0.7088 | 1.0000 | 0.7408 |
| c4 | 0.7325 | 0.7653 | 0.7303 | 0.8279 | 0.7512 | 0.6494 | 0.8276 | 0.7408 | 1.0000 |

## Log-Fisher cosine

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.9968 | 0.9727 | 0.9960 | 0.9958 | 0.9822 | 0.9807 | 0.9745 | 0.9925 |
| math500 | 0.9968 | 1.0000 | 0.9621 | 0.9968 | 0.9966 | 0.9769 | 0.9750 | 0.9660 | 0.9945 |
| svamp | 0.9727 | 0.9621 | 1.0000 | 0.9654 | 0.9662 | 0.9829 | 0.9838 | 0.9899 | 0.9551 |
| humaneval | 0.9960 | 0.9968 | 0.9654 | 1.0000 | 0.9977 | 0.9786 | 0.9770 | 0.9692 | 0.9929 |
| mbpp | 0.9958 | 0.9966 | 0.9662 | 0.9977 | 1.0000 | 0.9808 | 0.9791 | 0.9711 | 0.9939 |
| 2wiki | 0.9822 | 0.9769 | 0.9829 | 0.9786 | 0.9808 | 1.0000 | 0.9939 | 0.9899 | 0.9781 |
| hotpotqa | 0.9807 | 0.9750 | 0.9838 | 0.9770 | 0.9791 | 0.9939 | 1.0000 | 0.9902 | 0.9761 |
| triviaqa | 0.9745 | 0.9660 | 0.9899 | 0.9692 | 0.9711 | 0.9899 | 0.9902 | 1.0000 | 0.9641 |
| c4 | 0.9925 | 0.9945 | 0.9551 | 0.9929 | 0.9939 | 0.9781 | 0.9761 | 0.9641 | 1.0000 |

## Shared-component-removed cosine

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.8194 | -0.6303 | 0.7244 | 0.6642 | -0.8026 | -0.8107 | -0.7851 | 0.5764 |
| math500 | 0.8194 | 1.0000 | -0.8186 | 0.8355 | 0.8098 | -0.8321 | -0.8515 | -0.8931 | 0.7428 |
| svamp | -0.6303 | -0.8186 | 1.0000 | -0.7694 | -0.7975 | 0.6146 | 0.6517 | 0.8118 | -0.8749 |
| humaneval | 0.7244 | 0.8355 | -0.7694 | 1.0000 | 0.8495 | -0.8164 | -0.8301 | -0.8428 | 0.6435 |
| mbpp | 0.6642 | 0.8098 | -0.7975 | 0.8495 | 1.0000 | -0.7770 | -0.7953 | -0.8346 | 0.6830 |
| 2wiki | -0.8026 | -0.8321 | 0.6146 | -0.8164 | -0.7770 | 1.0000 | 0.7844 | 0.7411 | -0.6110 |
| hotpotqa | -0.8107 | -0.8515 | 0.6517 | -0.8301 | -0.7953 | 0.7844 | 1.0000 | 0.7606 | -0.6433 |
| triviaqa | -0.7851 | -0.8931 | 0.8118 | -0.8428 | -0.8346 | 0.7411 | 0.7606 | 1.0000 | -0.7986 |
| c4 | 0.5764 | 0.7428 | -0.8749 | 0.6435 | 0.6830 | -0.6110 | -0.6433 | -0.7986 | 1.0000 |

## Top-0.1% coordinate Jaccard

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.4566 | 0.3960 | 0.2799 | 0.2666 | 0.2322 | 0.2662 | 0.2469 | 0.2646 |
| math500 | 0.4566 | 1.0000 | 0.3250 | 0.3154 | 0.3047 | 0.2322 | 0.2733 | 0.2207 | 0.2773 |
| svamp | 0.3960 | 0.3250 | 1.0000 | 0.2065 | 0.1983 | 0.1826 | 0.2057 | 0.2452 | 0.1976 |
| humaneval | 0.2799 | 0.3154 | 0.2065 | 1.0000 | 0.4090 | 0.1872 | 0.2162 | 0.1764 | 0.2268 |
| mbpp | 0.2666 | 0.3047 | 0.1983 | 0.4090 | 1.0000 | 0.1877 | 0.2156 | 0.1660 | 0.2358 |
| 2wiki | 0.2322 | 0.2322 | 0.1826 | 0.1872 | 0.1877 | 1.0000 | 0.3339 | 0.1933 | 0.2782 |
| hotpotqa | 0.2662 | 0.2733 | 0.2057 | 0.2162 | 0.2156 | 0.3339 | 1.0000 | 0.2171 | 0.3223 |
| triviaqa | 0.2469 | 0.2207 | 0.2452 | 0.1764 | 0.1660 | 0.1933 | 0.2171 | 1.0000 | 0.1955 |
| c4 | 0.2646 | 0.2773 | 0.1976 | 0.2268 | 0.2358 | 0.2782 | 0.3223 | 0.1955 | 1.0000 |

## Block-structure summary

Within-capability pairs exclude C4 because control has one benchmark; cross-capability pairs include all C4 comparisons. The block score is `(mean within - mean cross) / std(all off-diagonal pairs)`.

| metric | mean within | mean cross | std all pairs | block score | block structure? |
|---|---:|---:|---:|---:|---|
| Raw Fisher cosine | 0.7404 | 0.7217 | 0.0743 | 0.2512 | yes |
| Log-Fisher cosine | 0.9862 | 0.9804 | 0.0117 | 0.4923 | yes |
| Shared-component-removed cosine | 0.3580 | -0.2279 | 0.7607 | 0.7702 | yes |
| Top-0.1% coordinate Jaccard | 0.3330 | 0.2353 | 0.0674 | 1.4496 | yes |

Metrics showing positive block structure: Raw Fisher cosine, Log-Fisher cosine, Shared-component-removed cosine, Top-0.1% coordinate Jaccard.
Strongest standardized block structure: Top-0.1% coordinate Jaccard (block score 1.4496).
