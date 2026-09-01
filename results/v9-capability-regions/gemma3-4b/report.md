# V9 capability-region report — google/gemma-3-4b-pt

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
| gsm8k | 1.0000 | 0.8769 | 0.5092 | 0.2380 | 0.3511 | 0.0734 | 0.5446 | 0.5813 | 0.8111 |
| math500 | 0.8769 | 1.0000 | 0.4698 | 0.2523 | 0.3541 | 0.0703 | 0.5416 | 0.5548 | 0.8197 |
| svamp | 0.5092 | 0.4698 | 1.0000 | 0.1334 | 0.1991 | 0.0449 | 0.3311 | 0.3527 | 0.3952 |
| humaneval | 0.2380 | 0.2523 | 0.1334 | 1.0000 | 0.3722 | 0.0196 | 0.1594 | 0.1649 | 0.2595 |
| mbpp | 0.3511 | 0.3541 | 0.1991 | 0.3722 | 1.0000 | 0.0341 | 0.2348 | 0.2739 | 0.3405 |
| 2wiki | 0.0734 | 0.0703 | 0.0449 | 0.0196 | 0.0341 | 1.0000 | 0.0968 | 0.0698 | 0.0654 |
| hotpotqa | 0.5446 | 0.5416 | 0.3311 | 0.1594 | 0.2348 | 0.0968 | 1.0000 | 0.3993 | 0.4844 |
| triviaqa | 0.5813 | 0.5548 | 0.3527 | 0.1649 | 0.2739 | 0.0698 | 0.3993 | 1.0000 | 0.5083 |
| c4 | 0.8111 | 0.8197 | 0.3952 | 0.2595 | 0.3405 | 0.0654 | 0.4844 | 0.5083 | 1.0000 |

## Log-Fisher cosine

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.9989 | 0.9859 | 0.9978 | 0.9972 | 0.9613 | 0.9721 | 0.9795 | 0.9948 |
| math500 | 0.9989 | 1.0000 | 0.9859 | 0.9981 | 0.9977 | 0.9622 | 0.9731 | 0.9804 | 0.9952 |
| svamp | 0.9859 | 0.9859 | 1.0000 | 0.9882 | 0.9893 | 0.9853 | 0.9901 | 0.9930 | 0.9889 |
| humaneval | 0.9978 | 0.9981 | 0.9882 | 1.0000 | 0.9985 | 0.9660 | 0.9760 | 0.9829 | 0.9951 |
| mbpp | 0.9972 | 0.9977 | 0.9893 | 0.9985 | 1.0000 | 0.9688 | 0.9789 | 0.9852 | 0.9959 |
| 2wiki | 0.9613 | 0.9622 | 0.9853 | 0.9660 | 0.9688 | 1.0000 | 0.9932 | 0.9909 | 0.9742 |
| hotpotqa | 0.9721 | 0.9731 | 0.9901 | 0.9760 | 0.9789 | 0.9932 | 1.0000 | 0.9948 | 0.9835 |
| triviaqa | 0.9795 | 0.9804 | 0.9930 | 0.9829 | 0.9852 | 0.9909 | 0.9948 | 1.0000 | 0.9883 |
| c4 | 0.9948 | 0.9952 | 0.9889 | 0.9951 | 0.9959 | 0.9742 | 0.9835 | 0.9883 | 1.0000 |

## Shared-component-removed cosine

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.9500 | -0.6730 | 0.8892 | 0.8458 | -0.9354 | -0.9342 | -0.9029 | 0.5117 |
| math500 | 0.9500 | 1.0000 | -0.7106 | 0.8998 | 0.8667 | -0.9390 | -0.9313 | -0.8980 | 0.5156 |
| svamp | -0.6730 | -0.7106 | 1.0000 | -0.6610 | -0.6788 | 0.6441 | 0.6152 | 0.5969 | -0.6593 |
| humaneval | 0.8892 | 0.8998 | -0.6610 | 1.0000 | 0.8911 | -0.9101 | -0.9133 | -0.8773 | 0.4249 |
| mbpp | 0.8458 | 0.8667 | -0.6788 | 0.8911 | 1.0000 | -0.8943 | -0.8754 | -0.8395 | 0.4420 |
| 2wiki | -0.9354 | -0.9390 | 0.6441 | -0.9101 | -0.8943 | 1.0000 | 0.9011 | 0.8517 | -0.5539 |
| hotpotqa | -0.9342 | -0.9313 | 0.6152 | -0.9133 | -0.8754 | 0.9011 | 1.0000 | 0.8553 | -0.5033 |
| triviaqa | -0.9029 | -0.8980 | 0.5969 | -0.8773 | -0.8395 | 0.8517 | 0.8553 | 1.0000 | -0.5030 |
| c4 | 0.5117 | 0.5156 | -0.6593 | 0.4249 | 0.4420 | -0.5539 | -0.5033 | -0.5030 | 1.0000 |

## Top-0.1% coordinate Jaccard

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.3334 | 0.2591 | 0.1963 | 0.1725 | 0.0756 | 0.1511 | 0.1361 | 0.1904 |
| math500 | 0.3334 | 1.0000 | 0.2031 | 0.2114 | 0.1806 | 0.0739 | 0.1403 | 0.1263 | 0.1801 |
| svamp | 0.2591 | 0.2031 | 1.0000 | 0.1454 | 0.1342 | 0.0624 | 0.1169 | 0.1313 | 0.1342 |
| humaneval | 0.1963 | 0.2114 | 0.1454 | 1.0000 | 0.2782 | 0.0629 | 0.1154 | 0.1076 | 0.1421 |
| mbpp | 0.1725 | 0.1806 | 0.1342 | 0.2782 | 1.0000 | 0.0631 | 0.1176 | 0.1101 | 0.1381 |
| 2wiki | 0.0756 | 0.0739 | 0.0624 | 0.0629 | 0.0631 | 1.0000 | 0.0998 | 0.0787 | 0.0884 |
| hotpotqa | 0.1511 | 0.1403 | 0.1169 | 0.1154 | 0.1176 | 0.0998 | 1.0000 | 0.1389 | 0.1661 |
| triviaqa | 0.1361 | 0.1263 | 0.1313 | 0.1076 | 0.1101 | 0.0787 | 0.1389 | 1.0000 | 0.1417 |
| c4 | 0.1904 | 0.1801 | 0.1342 | 0.1421 | 0.1381 | 0.0884 | 0.1661 | 0.1417 | 1.0000 |

## Block-structure summary

Within-capability pairs exclude C4 because control has one benchmark; cross-capability pairs include all C4 comparisons. The block score is `(mean within - mean cross) / std(all off-diagonal pairs)`.

| metric | mean within | mean cross | std all pairs | block score | block structure? |
|---|---:|---:|---:|---:|---|
| Raw Fisher cosine | 0.3991 | 0.3170 | 0.2254 | 0.3643 | yes |
| Log-Fisher cosine | 0.9926 | 0.9841 | 0.0106 | 0.7962 | yes |
| Shared-component-removed cosine | 0.4379 | -0.2468 | 0.7736 | 0.8851 | yes |
| Top-0.1% coordinate Jaccard | 0.1987 | 0.1315 | 0.0597 | 1.1273 | yes |

Metrics showing positive block structure: Raw Fisher cosine, Log-Fisher cosine, Shared-component-removed cosine, Top-0.1% coordinate Jaccard.
Strongest standardized block structure: Top-0.1% coordinate Jaccard (block score 1.1273).
