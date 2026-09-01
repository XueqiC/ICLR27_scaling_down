# V9 capability-region report — allenai/Olmo-3-1025-7B

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
| gsm8k | 1.0000 | 0.3221 | 0.0935 | 0.1939 | 0.0460 | 0.0582 | 0.0407 | 0.0603 | 0.3769 |
| math500 | 0.3221 | 1.0000 | 0.0952 | 0.0418 | 0.0574 | 0.0648 | 0.0275 | 0.0435 | 0.2364 |
| svamp | 0.0935 | 0.0952 | 1.0000 | 0.0100 | 0.0089 | 0.0200 | 0.0127 | 0.0461 | 0.0223 |
| humaneval | 0.1939 | 0.0418 | 0.0100 | 1.0000 | 0.0602 | 0.0031 | 0.0015 | 0.0014 | 0.0482 |
| mbpp | 0.0460 | 0.0574 | 0.0089 | 0.0602 | 1.0000 | 0.0116 | 0.0027 | 0.0037 | 0.0149 |
| 2wiki | 0.0582 | 0.0648 | 0.0200 | 0.0031 | 0.0116 | 1.0000 | 0.1163 | 0.0119 | 0.0390 |
| hotpotqa | 0.0407 | 0.0275 | 0.0127 | 0.0015 | 0.0027 | 0.1163 | 1.0000 | 0.0080 | 0.0299 |
| triviaqa | 0.0603 | 0.0435 | 0.0461 | 0.0014 | 0.0037 | 0.0119 | 0.0080 | 1.0000 | 0.0157 |
| c4 | 0.3769 | 0.2364 | 0.0223 | 0.0482 | 0.0149 | 0.0390 | 0.0299 | 0.0157 | 1.0000 |

## Log-Fisher cosine

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.9982 | 0.9847 | 0.9975 | 0.9976 | 0.9862 | 0.9864 | 0.9869 | 0.9970 |
| math500 | 0.9982 | 1.0000 | 0.9784 | 0.9983 | 0.9983 | 0.9822 | 0.9824 | 0.9826 | 0.9974 |
| svamp | 0.9847 | 0.9784 | 1.0000 | 0.9795 | 0.9804 | 0.9933 | 0.9932 | 0.9938 | 0.9807 |
| humaneval | 0.9975 | 0.9983 | 0.9795 | 1.0000 | 0.9986 | 0.9830 | 0.9832 | 0.9838 | 0.9968 |
| mbpp | 0.9976 | 0.9983 | 0.9804 | 0.9986 | 1.0000 | 0.9847 | 0.9848 | 0.9848 | 0.9974 |
| 2wiki | 0.9862 | 0.9822 | 0.9933 | 0.9830 | 0.9847 | 1.0000 | 0.9972 | 0.9953 | 0.9863 |
| hotpotqa | 0.9864 | 0.9824 | 0.9932 | 0.9832 | 0.9848 | 0.9972 | 1.0000 | 0.9955 | 0.9868 |
| triviaqa | 0.9869 | 0.9826 | 0.9938 | 0.9838 | 0.9848 | 0.9953 | 0.9955 | 1.0000 | 0.9864 |
| c4 | 0.9970 | 0.9974 | 0.9807 | 0.9968 | 0.9974 | 0.9863 | 0.9868 | 0.9864 | 1.0000 |

## Shared-component-removed cosine

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.8899 | -0.7758 | 0.8253 | 0.8075 | -0.8940 | -0.8964 | -0.8618 | 0.7052 |
| math500 | 0.8899 | 1.0000 | -0.9079 | 0.9108 | 0.9096 | -0.9319 | -0.9337 | -0.9182 | 0.8161 |
| svamp | -0.7758 | -0.9079 | 1.0000 | -0.8842 | -0.9023 | 0.8386 | 0.8346 | 0.8414 | -0.8761 |
| humaneval | 0.8253 | 0.9108 | -0.8842 | 1.0000 | 0.9165 | -0.9160 | -0.9190 | -0.8868 | 0.7573 |
| mbpp | 0.8075 | 0.9096 | -0.9023 | 0.9165 | 1.0000 | -0.9007 | -0.9030 | -0.8947 | 0.7819 |
| 2wiki | -0.8940 | -0.9319 | 0.8386 | -0.9160 | -0.9007 | 1.0000 | 0.9203 | 0.8550 | -0.7966 |
| hotpotqa | -0.8964 | -0.9337 | 0.8346 | -0.9190 | -0.9030 | 0.9203 | 1.0000 | 0.8597 | -0.7833 |
| triviaqa | -0.8618 | -0.9182 | 0.8414 | -0.8868 | -0.8947 | 0.8550 | 0.8597 | 1.0000 | -0.7987 |
| c4 | 0.7052 | 0.8161 | -0.8761 | 0.7573 | 0.7819 | -0.7966 | -0.7833 | -0.7987 | 1.0000 |

## Top-0.1% coordinate Jaccard

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.2850 | 0.3273 | 0.1885 | 0.1681 | 0.1698 | 0.1753 | 0.1858 | 0.1767 |
| math500 | 0.2850 | 1.0000 | 0.2244 | 0.2317 | 0.2057 | 0.1998 | 0.2019 | 0.1842 | 0.2234 |
| svamp | 0.3273 | 0.2244 | 1.0000 | 0.1577 | 0.1491 | 0.1571 | 0.1605 | 0.1751 | 0.1454 |
| humaneval | 0.1885 | 0.2317 | 0.1577 | 1.0000 | 0.2925 | 0.1743 | 0.1730 | 0.1529 | 0.1810 |
| mbpp | 0.1681 | 0.2057 | 0.1491 | 0.2925 | 1.0000 | 0.1700 | 0.1693 | 0.1406 | 0.1789 |
| 2wiki | 0.1698 | 0.1998 | 0.1571 | 0.1743 | 0.1700 | 1.0000 | 0.2786 | 0.1637 | 0.1929 |
| hotpotqa | 0.1753 | 0.2019 | 0.1605 | 0.1730 | 0.1693 | 0.2786 | 1.0000 | 0.1694 | 0.1979 |
| triviaqa | 0.1858 | 0.1842 | 0.1751 | 0.1529 | 0.1406 | 0.1637 | 0.1694 | 1.0000 | 0.1541 |
| c4 | 0.1767 | 0.2234 | 0.1454 | 0.1810 | 0.1789 | 0.1929 | 0.1979 | 0.1541 | 1.0000 |

## Block-structure summary

Within-capability pairs exclude C4 because control has one benchmark; cross-capability pairs include all C4 comparisons. The block score is `(mean within - mean cross) / std(all off-diagonal pairs)`.

| metric | mean within | mean cross | std all pairs | block score | block structure? |
|---|---:|---:|---:|---:|---|
| Raw Fisher cosine | 0.1010 | 0.0531 | 0.0859 | 0.5584 | yes |
| Log-Fisher cosine | 0.9926 | 0.9887 | 0.0068 | 0.5757 | yes |
| Shared-component-removed cosine | 0.3940 | -0.2369 | 0.8568 | 0.7363 | yes |
| Top-0.1% coordinate Jaccard | 0.2487 | 0.1773 | 0.0431 | 1.6585 | yes |

Metrics showing positive block structure: Raw Fisher cosine, Log-Fisher cosine, Shared-component-removed cosine, Top-0.1% coordinate Jaccard.
Strongest standardized block structure: Top-0.1% coordinate Jaccard (block score 1.6585).
