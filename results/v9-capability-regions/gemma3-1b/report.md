# V9 capability-region report — google/gemma-3-1b-pt

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
| gsm8k | 1.0000 | 0.8104 | 0.4614 | 0.5017 | 0.4748 | 0.5585 | 0.5536 | 0.3910 | 0.6037 |
| math500 | 0.8104 | 1.0000 | 0.5242 | 0.6588 | 0.5591 | 0.7172 | 0.7286 | 0.4215 | 0.6879 |
| svamp | 0.4614 | 0.5242 | 1.0000 | 0.5399 | 0.4111 | 0.6142 | 0.6259 | 0.2371 | 0.3524 |
| humaneval | 0.5017 | 0.6588 | 0.5399 | 1.0000 | 0.6156 | 0.7702 | 0.7839 | 0.2974 | 0.4388 |
| mbpp | 0.4748 | 0.5591 | 0.4111 | 0.6156 | 1.0000 | 0.5085 | 0.5126 | 0.2797 | 0.3915 |
| 2wiki | 0.5585 | 0.7172 | 0.6142 | 0.7702 | 0.5085 | 1.0000 | 0.9566 | 0.3264 | 0.4837 |
| hotpotqa | 0.5536 | 0.7286 | 0.6259 | 0.7839 | 0.5126 | 0.9566 | 1.0000 | 0.3327 | 0.5135 |
| triviaqa | 0.3910 | 0.4215 | 0.2371 | 0.2974 | 0.2797 | 0.3264 | 0.3327 | 1.0000 | 0.3606 |
| c4 | 0.6037 | 0.6879 | 0.3524 | 0.4388 | 0.3915 | 0.4837 | 0.5135 | 0.3606 | 1.0000 |

## Log-Fisher cosine

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.9986 | 0.9895 | 0.9974 | 0.9977 | 0.9743 | 0.9758 | 0.9809 | 0.9958 |
| math500 | 0.9986 | 1.0000 | 0.9885 | 0.9977 | 0.9982 | 0.9737 | 0.9753 | 0.9807 | 0.9959 |
| svamp | 0.9895 | 0.9885 | 1.0000 | 0.9921 | 0.9909 | 0.9911 | 0.9912 | 0.9934 | 0.9905 |
| humaneval | 0.9974 | 0.9977 | 0.9921 | 1.0000 | 0.9990 | 0.9799 | 0.9810 | 0.9855 | 0.9958 |
| mbpp | 0.9977 | 0.9982 | 0.9909 | 0.9990 | 1.0000 | 0.9779 | 0.9792 | 0.9841 | 0.9960 |
| 2wiki | 0.9743 | 0.9737 | 0.9911 | 0.9799 | 0.9779 | 1.0000 | 0.9958 | 0.9947 | 0.9809 |
| hotpotqa | 0.9758 | 0.9753 | 0.9912 | 0.9810 | 0.9792 | 0.9958 | 1.0000 | 0.9950 | 0.9827 |
| triviaqa | 0.9809 | 0.9807 | 0.9934 | 0.9855 | 0.9841 | 0.9947 | 0.9950 | 1.0000 | 0.9866 |
| c4 | 0.9958 | 0.9959 | 0.9905 | 0.9958 | 0.9960 | 0.9809 | 0.9827 | 0.9866 | 1.0000 |

## Shared-component-removed cosine

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.8988 | -0.4970 | 0.7575 | 0.8058 | -0.8910 | -0.8875 | -0.8567 | 0.5069 |
| math500 | 0.8988 | 1.0000 | -0.5762 | 0.7928 | 0.8543 | -0.9056 | -0.8959 | -0.8542 | 0.5248 |
| svamp | -0.4970 | -0.5762 | 1.0000 | -0.4937 | -0.5564 | 0.4847 | 0.4458 | 0.4326 | -0.5776 |
| humaneval | 0.7575 | 0.7928 | -0.4937 | 1.0000 | 0.8795 | -0.8195 | -0.8293 | -0.7962 | 0.3449 |
| mbpp | 0.8058 | 0.8543 | -0.5564 | 0.8795 | 1.0000 | -0.8690 | -0.8672 | -0.8232 | 0.4281 |
| 2wiki | -0.8910 | -0.9056 | 0.4847 | -0.8195 | -0.8690 | 1.0000 | 0.8598 | 0.7968 | -0.5744 |
| hotpotqa | -0.8875 | -0.8959 | 0.4458 | -0.8293 | -0.8672 | 0.8598 | 1.0000 | 0.7906 | -0.5273 |
| triviaqa | -0.8567 | -0.8542 | 0.4326 | -0.7962 | -0.8232 | 0.7968 | 0.7906 | 1.0000 | -0.5202 |
| c4 | 0.5069 | 0.5248 | -0.5776 | 0.3449 | 0.4281 | -0.5744 | -0.5273 | -0.5202 | 1.0000 |

## Top-0.1% coordinate Jaccard

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.3435 | 0.1843 | 0.2012 | 0.2040 | 0.2021 | 0.1873 | 0.1560 | 0.2161 |
| math500 | 0.3435 | 1.0000 | 0.1966 | 0.2548 | 0.2648 | 0.2144 | 0.2013 | 0.1615 | 0.2400 |
| svamp | 0.1843 | 0.1966 | 1.0000 | 0.1525 | 0.1591 | 0.1561 | 0.1487 | 0.1391 | 0.1599 |
| humaneval | 0.2012 | 0.2548 | 0.1525 | 1.0000 | 0.4102 | 0.1740 | 0.1619 | 0.1292 | 0.1833 |
| mbpp | 0.2040 | 0.2648 | 0.1591 | 0.4102 | 1.0000 | 0.1757 | 0.1692 | 0.1385 | 0.1930 |
| 2wiki | 0.2021 | 0.2144 | 0.1561 | 0.1740 | 0.1757 | 1.0000 | 0.2978 | 0.1818 | 0.2472 |
| hotpotqa | 0.1873 | 0.2013 | 0.1487 | 0.1619 | 0.1692 | 0.2978 | 1.0000 | 0.1859 | 0.2454 |
| triviaqa | 0.1560 | 0.1615 | 0.1391 | 0.1292 | 0.1385 | 0.1818 | 0.1859 | 1.0000 | 0.1914 |
| c4 | 0.2161 | 0.2400 | 0.1599 | 0.1833 | 0.1930 | 0.2472 | 0.2454 | 0.1914 | 1.0000 |

## Block-structure summary

Within-capability pairs exclude C4 because control has one benchmark; cross-capability pairs include all C4 comparisons. The block score is `(mean within - mean cross) / std(all off-diagonal pairs)`.

| metric | mean within | mean cross | std all pairs | block score | block structure? |
|---|---:|---:|---:|---:|---|
| Raw Fisher cosine | 0.5753 | 0.5165 | 0.1631 | 0.3609 | yes |
| Log-Fisher cosine | 0.9945 | 0.9870 | 0.0080 | 0.9372 | yes |
| Shared-component-removed cosine | 0.4503 | -0.2471 | 0.7142 | 0.9766 | yes |
| Top-0.1% coordinate Jaccard | 0.2572 | 0.1872 | 0.0575 | 1.2176 | yes |

Metrics showing positive block structure: Raw Fisher cosine, Log-Fisher cosine, Shared-component-removed cosine, Top-0.1% coordinate Jaccard.
Strongest standardized block structure: Top-0.1% coordinate Jaccard (block score 1.2176).
