# V9 capability-region report — google/gemma-3-270m

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
| gsm8k | 1.0000 | 0.6923 | 0.1691 | 0.7748 | 0.5879 | 0.7444 | 0.7197 | 0.3428 | 0.7063 |
| math500 | 0.6923 | 1.0000 | 0.2839 | 0.7330 | 0.7688 | 0.6855 | 0.7056 | 0.5331 | 0.6917 |
| svamp | 0.1691 | 0.2839 | 1.0000 | 0.2079 | 0.2770 | 0.2138 | 0.2173 | 0.2820 | 0.2788 |
| humaneval | 0.7748 | 0.7330 | 0.2079 | 1.0000 | 0.6387 | 0.7916 | 0.7729 | 0.4236 | 0.6109 |
| mbpp | 0.5879 | 0.7688 | 0.2770 | 0.6387 | 1.0000 | 0.5350 | 0.5596 | 0.5546 | 0.7009 |
| 2wiki | 0.7444 | 0.6855 | 0.2138 | 0.7916 | 0.5350 | 1.0000 | 0.9536 | 0.3909 | 0.6136 |
| hotpotqa | 0.7197 | 0.7056 | 0.2173 | 0.7729 | 0.5596 | 0.9536 | 1.0000 | 0.4226 | 0.6393 |
| triviaqa | 0.3428 | 0.5331 | 0.2820 | 0.4236 | 0.5546 | 0.3909 | 0.4226 | 1.0000 | 0.6501 |
| c4 | 0.7063 | 0.6917 | 0.2788 | 0.6109 | 0.7009 | 0.6136 | 0.6393 | 0.6501 | 1.0000 |

## Log-Fisher cosine

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.9988 | 0.9948 | 0.9979 | 0.9982 | 0.9857 | 0.9865 | 0.9888 | 0.9958 |
| math500 | 0.9988 | 1.0000 | 0.9938 | 0.9978 | 0.9983 | 0.9845 | 0.9853 | 0.9879 | 0.9955 |
| svamp | 0.9948 | 0.9938 | 1.0000 | 0.9960 | 0.9954 | 0.9931 | 0.9932 | 0.9948 | 0.9927 |
| humaneval | 0.9979 | 0.9978 | 0.9960 | 1.0000 | 0.9992 | 0.9883 | 0.9888 | 0.9910 | 0.9949 |
| mbpp | 0.9982 | 0.9983 | 0.9954 | 0.9992 | 1.0000 | 0.9872 | 0.9877 | 0.9901 | 0.9952 |
| 2wiki | 0.9857 | 0.9845 | 0.9931 | 0.9883 | 0.9872 | 1.0000 | 0.9956 | 0.9951 | 0.9874 |
| hotpotqa | 0.9865 | 0.9853 | 0.9932 | 0.9888 | 0.9877 | 0.9956 | 1.0000 | 0.9950 | 0.9887 |
| triviaqa | 0.9888 | 0.9879 | 0.9948 | 0.9910 | 0.9901 | 0.9951 | 0.9950 | 1.0000 | 0.9903 |
| c4 | 0.9958 | 0.9955 | 0.9927 | 0.9949 | 0.9952 | 0.9874 | 0.9887 | 0.9903 | 1.0000 |

## Shared-component-removed cosine

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.8077 | -0.1730 | 0.5651 | 0.6625 | -0.7747 | -0.7570 | -0.7263 | 0.2391 |
| math500 | 0.8077 | 1.0000 | -0.2704 | 0.6184 | 0.7211 | -0.7881 | -0.7741 | -0.7242 | 0.2413 |
| svamp | -0.1730 | -0.2704 | 1.0000 | -0.1151 | -0.1935 | 0.1247 | 0.0837 | 0.1269 | -0.4718 |
| humaneval | 0.5651 | 0.6184 | -0.1151 | 1.0000 | 0.8031 | -0.6523 | -0.6758 | -0.6261 | -0.0428 |
| mbpp | 0.6625 | 0.7211 | -0.1935 | 0.8031 | 1.0000 | -0.7272 | -0.7417 | -0.6861 | 0.0616 |
| 2wiki | -0.7747 | -0.7881 | 0.1247 | -0.6523 | -0.7272 | 1.0000 | 0.6643 | 0.5812 | -0.3393 |
| hotpotqa | -0.7570 | -0.7741 | 0.0837 | -0.6758 | -0.7417 | 0.6643 | 1.0000 | 0.5573 | -0.2636 |
| triviaqa | -0.7263 | -0.7242 | 0.1269 | -0.6261 | -0.6861 | 0.5812 | 0.5573 | 1.0000 | -0.2739 |
| c4 | 0.2391 | 0.2413 | -0.4718 | -0.0428 | 0.0616 | -0.3393 | -0.2636 | -0.2739 | 1.0000 |

## Top-0.1% coordinate Jaccard

| benchmark | gsm8k | math500 | svamp | humaneval | mbpp | 2wiki | hotpotqa | triviaqa | c4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 1.0000 | 0.4562 | 0.2538 | 0.3247 | 0.3226 | 0.2977 | 0.2909 | 0.2697 | 0.3259 |
| math500 | 0.4562 | 1.0000 | 0.2569 | 0.3634 | 0.3687 | 0.3180 | 0.3153 | 0.2906 | 0.3581 |
| svamp | 0.2538 | 0.2569 | 1.0000 | 0.2113 | 0.2204 | 0.2100 | 0.2155 | 0.2352 | 0.2289 |
| humaneval | 0.3247 | 0.3634 | 0.2113 | 1.0000 | 0.4668 | 0.2679 | 0.2634 | 0.2442 | 0.2921 |
| mbpp | 0.3226 | 0.3687 | 0.2204 | 0.4668 | 1.0000 | 0.2753 | 0.2763 | 0.2578 | 0.3066 |
| 2wiki | 0.2977 | 0.3180 | 0.2100 | 0.2679 | 0.2753 | 1.0000 | 0.3856 | 0.2913 | 0.3437 |
| hotpotqa | 0.2909 | 0.3153 | 0.2155 | 0.2634 | 0.2763 | 0.3856 | 1.0000 | 0.3061 | 0.3574 |
| triviaqa | 0.2697 | 0.2906 | 0.2352 | 0.2442 | 0.2578 | 0.2913 | 0.3061 | 1.0000 | 0.3287 |
| c4 | 0.3259 | 0.3581 | 0.2289 | 0.2921 | 0.3066 | 0.3437 | 0.3574 | 0.3287 | 1.0000 |

## Block-structure summary

Within-capability pairs exclude C4 because control has one benchmark; cross-capability pairs include all C4 comparisons. The block score is `(mean within - mean cross) / std(all off-diagonal pairs)`.

| metric | mean within | mean cross | std all pairs | block score | block structure? |
|---|---:|---:|---:|---:|---|
| Raw Fisher cosine | 0.5073 | 0.5628 | 0.2049 | -0.2711 | no |
| Log-Fisher cosine | 0.9960 | 0.9916 | 0.0043 | 1.0240 | yes |
| Shared-component-removed cosine | 0.4243 | -0.2382 | 0.5447 | 1.2163 | yes |
| Top-0.1% coordinate Jaccard | 0.3452 | 0.2890 | 0.0607 | 0.9277 | yes |

Metrics showing positive block structure: Log-Fisher cosine, Shared-component-removed cosine, Top-0.1% coordinate Jaccard.
Strongest standardized block structure: Shared-component-removed cosine (block score 1.2163).
