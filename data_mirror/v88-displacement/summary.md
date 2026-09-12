# V88 displacement summary

one pooled (state, config, capability) measurement, equally weighted.

mild |dL|<0.1; moderate 0.1<=|dL|<=0.5; severe |dL|>0.5

abs(pred-measured)/abs(measured); zero measured values excluded and counted

The stipulated shrinkage prediction omits 0.5*eps_hat**2*Var_p(z). For pure shrinkage its difference from the Taylor polynomial is exactly that term. Also sigma2_hat is a p-weighted residual variance, not the coordinate variance of isotropic noise: before projection E[Var_p(eta)] = sigma_coordinate**2*V. No fitted coefficients or silent rescaling.

| Regime | N | Prediction | MAE | Median relative error | Sign agreement |
|---|---:|---|---:|---:|---:|
| mild | 41 | second_order_prediction | 0.0123532 | 0.0230939 | 0.97561 |
| mild | 41 | shrinkage_prediction | 0.019669 | 0.371073 | 0.853659 |
| moderate | 31 | second_order_prediction | 0.0793517 | 0.152649 | 0.935484 |
| moderate | 31 | shrinkage_prediction | 0.211366 | 0.560745 | 0.83871 |
| severe | 33 | second_order_prediction | 1.16718 | 0.388361 | 1 |
| severe | 33 | shrinkage_prediction | 1.57229 | 0.568614 | 0.939394 |

Top-16 |r| share of p-weighted r² (larger shares indicate concentration; not an isotropy proof).
Zero displacement has share 0; nonzero-token means and pooled mass ratios are also reported.

| Regime | Mean token share | Median token share | Mean nonzero share | Mean mass share |
|---|---:|---:|---:|---:|
| mild | 0.0243281 | 0.0126975 | 0.0243281 | 0.082478 |
| moderate | 0.0949432 | 0.0420318 | 0.0949432 | 0.208709 |
| severe | 0.202374 | 0.207876 | 0.202374 | 0.313434 |

## Frozen reproduction checks

Differences are recomputed minus frozen; no frozen value is corrected.

| State | Config | Capability | Recomputed dL | Frozen dL | Difference | Probe hash match |
|---|---|---|---:|---:|---:|---|
| pythia-1.4b--step143000 | b3_g128 | math | 0.935529 | 0.931543 | 0.00398603 | True |
| pythia-1.4b--step143000 | b3_g128 | code | 1.25621 | 1.23716 | 0.0190477 | True |
| pythia-1.4b--step143000 | b3_g128 | qa | -0.00460057 | 0.059975 | -0.0645756 | True |
| pythia-1.4b--step143000 | b4_g128 | math | 0.0964697 | 0.0965356 | -6.59192e-05 | True |
| pythia-1.4b--step143000 | b4_g128 | code | 0.121881 | 0.127763 | -0.00588187 | True |
| pythia-1.4b--step143000 | b4_g128 | qa | -0.0344529 | 0.0254575 | -0.0599104 | True |
| pythia-1.4b--step143000 | b5_g128 | math | 0.0197346 | 0.0188642 | 0.000870391 | True |
| pythia-1.4b--step143000 | b5_g128 | code | 0.0115985 | 0.0155099 | -0.00391134 | True |
| pythia-1.4b--step143000 | b5_g128 | qa | -0.100991 | -0.0198135 | -0.0811776 | True |
| pythia-1.4b--step143000 | prune_d0.6 | math | 1.1509 | 1.14965 | 0.00124844 | None |
| pythia-1.4b--step143000 | prune_d0.6 | code | 0.982752 | 0.982767 | -1.45468e-05 | None |
| pythia-1.4b--step143000 | prune_d0.6 | qa | -0.458766 | -0.445194 | -0.0135727 | None |
| pythia-1.4b--step143000 | prune_d0.7 | math | 0.295469 | 0.294807 | 0.000661994 | None |
| pythia-1.4b--step143000 | prune_d0.7 | code | 0.178344 | 0.174055 | 0.00428873 | None |
| pythia-1.4b--step143000 | prune_d0.7 | qa | -0.437904 | -0.413944 | -0.0239601 | None |
| pythia-1.4b--step143000 | prune_d0.8 | math | 0.0710756 | 0.070632 | 0.00044366 | None |
| pythia-1.4b--step143000 | prune_d0.8 | code | 0.0520918 | 0.0515213 | 0.000570538 | None |
| pythia-1.4b--step143000 | prune_d0.8 | qa | -0.11102 | -0.063807 | -0.0472134 | None |
| pythia-1.4b--step143000 | prune_d0.9 | math | 0.0108023 | 0.0101044 | 0.000697845 | None |
| pythia-1.4b--step143000 | prune_d0.9 | code | 0.0209292 | 0.0210958 | -0.000166558 | None |
| pythia-1.4b--step143000 | prune_d0.9 | qa | 0.0154806 | 0.0282201 | -0.0127395 | None |
| pythia-1.4b--step16000 | b3_g128 | math | 0.276855 | 0.277486 | -0.000630767 | True |
| pythia-1.4b--step16000 | b3_g128 | code | 0.292639 | 0.291419 | 0.00121996 | True |
| pythia-1.4b--step16000 | b3_g128 | qa | -0.0367648 | -0.038914 | 0.00214922 | True |
| pythia-1.4b--step16000 | b4_g128 | math | 0.0254904 | 0.02622 | -0.000729601 | True |
| pythia-1.4b--step16000 | b4_g128 | code | 0.0169804 | 0.0156287 | 0.00135169 | True |
| pythia-1.4b--step16000 | b4_g128 | qa | -0.0586801 | -0.0622624 | 0.00358229 | True |
| pythia-1.4b--step16000 | b5_g128 | math | 0.00616365 | 0.00599146 | 0.000172196 | True |
| pythia-1.4b--step16000 | b5_g128 | code | 0.00709539 | 0.00629903 | 0.000796368 | True |
| pythia-1.4b--step16000 | b5_g128 | qa | -0.0158275 | -0.0177044 | 0.00187691 | True |
| pythia-1.4b--step16000 | prune_d0.6 | math | 0.36569 | 0.36538 | 0.000310073 | None |
| pythia-1.4b--step16000 | prune_d0.6 | code | 0.353986 | 0.353993 | -7.79556e-06 | None |
| pythia-1.4b--step16000 | prune_d0.6 | qa | -0.312428 | -0.319926 | 0.00749836 | None |
| pythia-1.4b--step16000 | prune_d0.7 | math | 0.0786802 | 0.0781065 | 0.000573702 | None |
| pythia-1.4b--step16000 | prune_d0.7 | code | 0.0498361 | 0.0495603 | 0.000275861 | None |
| pythia-1.4b--step16000 | prune_d0.7 | qa | -0.138924 | -0.141694 | 0.00277066 | None |
| pythia-1.4b--step16000 | prune_d0.8 | math | 0.0152269 | 0.0150479 | 0.000179074 | None |
| pythia-1.4b--step16000 | prune_d0.8 | code | 0.0133095 | 0.0125386 | 0.000770899 | None |
| pythia-1.4b--step16000 | prune_d0.8 | qa | -0.000499468 | -0.00249525 | 0.00199578 | None |
| pythia-1.4b--step16000 | prune_d0.9 | math | 0.0010946 | 0.00104801 | 4.65876e-05 | None |
| pythia-1.4b--step16000 | prune_d0.9 | code | 0.00109669 | 5.94248e-05 | 0.00103727 | None |
| pythia-1.4b--step16000 | prune_d0.9 | qa | 0.0098657 | 0.00986217 | 3.53186e-06 | None |
| pythia-160m--step143000 | b3_g128 | math | 14.4718 | 14.5132 | -0.041341 | True |
| pythia-160m--step143000 | b3_g128 | code | 14.6051 | 14.6168 | -0.0116347 | True |
| pythia-160m--step143000 | b3_g128 | qa | 15.8162 | 15.5379 | 0.278323 | True |
| pythia-160m--step143000 | b4_g128 | math | 2.17351 | 2.19157 | -0.0180544 | True |
| pythia-160m--step143000 | b4_g128 | code | 2.79098 | 2.79813 | -0.00715566 | True |
| pythia-160m--step143000 | b4_g128 | qa | 2.9137 | 2.71542 | 0.19828 | True |
| pythia-160m--step143000 | b5_g128 | math | 0.413958 | 0.413668 | 0.000290785 | True |
| pythia-160m--step143000 | b5_g128 | code | 0.458797 | 0.469753 | -0.0109558 | True |
| pythia-160m--step143000 | b5_g128 | qa | 0.613352 | 0.549548 | 0.0638038 | True |
| pythia-160m--step143000 | prune_d0.6 | math | 6.42342 | 6.46413 | -0.0407087 | None |
| pythia-160m--step143000 | prune_d0.6 | code | 7.21844 | 7.24643 | -0.0279985 | None |
| pythia-160m--step143000 | prune_d0.6 | qa | 6.24238 | 6.10658 | 0.135801 | None |
| pythia-160m--step143000 | prune_d0.7 | math | 2.38354 | 2.38808 | -0.00454475 | None |
| pythia-160m--step143000 | prune_d0.7 | code | 3.39381 | 3.38026 | 0.0135466 | None |
| pythia-160m--step143000 | prune_d0.7 | qa | 3.17756 | 2.81 | 0.367554 | None |
| pythia-160m--step143000 | prune_d0.8 | math | 0.714065 | 0.733845 | -0.0197795 | None |
| pythia-160m--step143000 | prune_d0.8 | code | 1.09034 | 1.08444 | 0.00589859 | None |
| pythia-160m--step143000 | prune_d0.8 | qa | 1.01739 | 0.952471 | 0.0649149 | None |
| pythia-160m--step143000 | prune_d0.9 | math | 0.125652 | 0.118801 | 0.00685098 | None |
| pythia-160m--step143000 | prune_d0.9 | code | 0.205053 | 0.237937 | -0.032884 | None |
| pythia-160m--step143000 | prune_d0.9 | qa | 0.218913 | 0.220651 | -0.00173776 | None |
| pythia-160m--step16000 | b3_g128 | math | 0.656814 | 0.656885 | -7.08293e-05 | True |
| pythia-160m--step16000 | b3_g128 | code | 0.858933 | 0.858272 | 0.000661201 | True |
| pythia-160m--step16000 | b3_g128 | qa | 0.500669 | 0.504396 | -0.00372714 | True |
| pythia-160m--step16000 | b4_g128 | math | 0.0688343 | 0.0683778 | 0.000456517 | True |
| pythia-160m--step16000 | b4_g128 | code | 0.0792701 | 0.0781436 | 0.00112657 | True |
| pythia-160m--step16000 | b4_g128 | qa | 0.0822517 | 0.0813926 | 0.000859126 | True |
| pythia-160m--step16000 | b5_g128 | math | 0.00991569 | 0.00945187 | 0.000463822 | True |
| pythia-160m--step16000 | b5_g128 | code | 0.023358 | 0.0226408 | 0.000717164 | True |
| pythia-160m--step16000 | b5_g128 | qa | 0.00973817 | 0.0104563 | -0.000718102 | True |
| pythia-160m--step16000 | prune_d0.6 | math | 0.872722 | 0.872103 | 0.000618572 | None |
| pythia-160m--step16000 | prune_d0.6 | code | 1.13548 | 1.13365 | 0.00183724 | None |
| pythia-160m--step16000 | prune_d0.6 | qa | 0.354467 | 0.348473 | 0.00599427 | None |
| pythia-160m--step16000 | prune_d0.7 | math | 0.257757 | 0.257059 | 0.000698164 | None |
| pythia-160m--step16000 | prune_d0.7 | code | 0.279683 | 0.277989 | 0.00169395 | None |
| pythia-160m--step16000 | prune_d0.7 | qa | -0.171312 | -0.173628 | 0.00231588 | None |
| pythia-160m--step16000 | prune_d0.8 | math | 0.0743093 | 0.0732619 | 0.00104741 | None |
| pythia-160m--step16000 | prune_d0.8 | code | 0.0759747 | 0.0756477 | 0.000326974 | None |
| pythia-160m--step16000 | prune_d0.8 | qa | -0.0240612 | -0.0271804 | 0.00311917 | None |
| pythia-160m--step16000 | prune_d0.9 | math | 0.00872648 | 0.00887843 | -0.000151952 | None |
| pythia-160m--step16000 | prune_d0.9 | code | 0.0134233 | 0.0136083 | -0.00018502 | None |
| pythia-160m--step16000 | prune_d0.9 | qa | 0.00739536 | 0.0120901 | -0.00469471 | None |
| pythia-410m--step143000 | b3_g128 | math | 2.23923 | 2.24009 | -0.000862299 | True |
| pythia-410m--step143000 | b3_g128 | code | 2.71804 | 2.68451 | 0.0335298 | True |
| pythia-410m--step143000 | b3_g128 | qa | 2.0905 | 2.12161 | -0.0311176 | True |
| pythia-410m--step143000 | b4_g128 | math | 0.288086 | 0.29459 | -0.00650359 | True |
| pythia-410m--step143000 | b4_g128 | code | 0.309521 | 0.30687 | 0.00265158 | True |
| pythia-410m--step143000 | b4_g128 | qa | 0.277541 | 0.298301 | -0.0207603 | True |
| pythia-410m--step143000 | b5_g128 | math | 0.0619393 | 0.0704145 | -0.00847512 | True |
| pythia-410m--step143000 | b5_g128 | code | 0.121239 | 0.104528 | 0.0167106 | True |
| pythia-410m--step143000 | b5_g128 | qa | -0.0241507 | -0.0139615 | -0.0101892 | True |
| pythia-410m--step143000 | prune_d0.6 | math | 1.65797 | 1.66246 | -0.0044942 | None |
| pythia-410m--step143000 | prune_d0.6 | code | 2.33925 | 2.3055 | 0.0337481 | None |
| pythia-410m--step143000 | prune_d0.6 | qa | 1.2681 | 1.25677 | 0.0113321 | None |
| pythia-410m--step143000 | prune_d0.7 | math | 0.60474 | 0.612809 | -0.00806993 | None |
| pythia-410m--step143000 | prune_d0.7 | code | 0.806344 | 0.806038 | 0.000306395 | None |
| pythia-410m--step143000 | prune_d0.7 | qa | 0.176239 | 0.127733 | 0.0485058 | None |
| pythia-410m--step143000 | prune_d0.8 | math | 0.197922 | 0.19665 | 0.00127197 | None |
| pythia-410m--step143000 | prune_d0.8 | code | 0.224451 | 0.216128 | 0.0083234 | None |
| pythia-410m--step143000 | prune_d0.8 | qa | -0.177456 | -0.166825 | -0.0106306 | None |
| pythia-410m--step143000 | prune_d0.9 | math | 0.0317758 | 0.0341098 | -0.00233401 | None |
| pythia-410m--step143000 | prune_d0.9 | code | 0.0182019 | 0.00956739 | 0.00863448 | None |
| pythia-410m--step143000 | prune_d0.9 | qa | -0.103259 | -0.115375 | 0.0121164 | None |

V6 probe hashes and realized thresholds may be unavailable. Forward shapes and CE arithmetic differ; see summary.json for source hashes and protocol metadata.
