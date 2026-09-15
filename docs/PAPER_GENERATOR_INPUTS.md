# Frozen inputs for the paper generators

The five generators are `final_prediction_table`, `plot_fig_generalization`,
`plot_fig_generalization_cells`, `plot_fig_explanation`, and `plot_fig_responses_v2`.
The inventory also traces `a7_closeout_audit` and `a5_corner_second_difference`.

The trace calls the existing loaders against the frozen source tree. It includes
the V86 comparison/prediction loaders, confirmation identity checks, the canonical
A1 CSV/metadata reader, V88 configuration glob, and both students’ A5 planned
corner checkpoints, their update-zero baselines, and exact V99 scope corners.
A5 reads 14 distinct `v12-distill` evaluation files; no weights or training logs
are needed. All V53 files and the three V46 files are included for completeness.

`explicit_directory_inventory` identifies those additional directory artifacts.
Existing V88 aggregate mirrors retain the same consumed `per_capability` values;
their omitted per-sample arrays are not read by these generators.

There are 268 result artifacts and two documentation inputs below. Bootstrap
expands gzip and writes both `--step` and `@step` filename spellings. The new A2
summary exceeds 5 MB and uses deterministic gzip, with timestamp zero.

## Unavailable optional input

`results/v47-p2-register/freeze.json` does not exist in the frozen source tree.
It is not fabricated. Both loaders retain their explicit unavailable-input note.
The refreshed V47 register contains `v5_confirm`; the corresponding available
confirmation freeze is `results/v70-distill-confirm/freeze.json`.

## Complete inventory

| Materialised input | Published source | Consumers |
|---|---|---|
| `docs/RESULTS_LEDGER.md` | `docs/RESULTS_LEDGER.md` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `docs/prereg/corner_second_difference_prereg.md` | `docs/prereg/corner_second_difference_prereg.md` | `a5_corner_second_difference` |
| `results/a1-development-table/development_table.csv` | `data_mirror/a1-development-table/development_table.csv` | `plot_fig_responses_v2` |
| `results/a1-development-table/row_metadata.csv` | `data_mirror/a1-development-table/row_metadata.csv` | `plot_fig_responses_v2` |
| `results/a1-development-table/summary.json` | `data_mirror/a1-development-table/summary.json` | `plot_fig_responses_v2` |
| `results/a2-curvature-interaction/nonembedding_counts.json` | `data_mirror/a2-curvature-interaction/nonembedding_counts.json` | `a7_closeout_audit` |
| `results/a2-curvature-interaction/summary.json` | `data_mirror/a2-curvature-interaction/summary.json.gz` | `a7_closeout_audit`, `final_prediction_table`, `plot_fig_explanation` |
| `results/a3-corner-pools/plan.json` | `data_mirror/a3-corner-pools/plan.json` | `a5_corner_second_difference`, `a7_closeout_audit` |
| `results/a5-corner-second-difference/summary.json` | `data_mirror/a5-corner-second-difference/summary.json` | `a7_closeout_audit`, `final_prediction_table`, `plot_fig_explanation`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/a5-training-seed-noise/summary.json` | `data_mirror/a5-training-seed-noise/summary.json` | `a7_closeout_audit` |
| `results/a7-closeout-audit/summary.json` | `data_mirror/a7-closeout-audit/summary.json` | `final_prediction_table`, `plot_fig_explanation`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000000/eval.json` | `data_mirror/v12-distill/gemma3-1b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000000/eval.json` | `a5_corner_second_difference` |
| `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000036/eval.json` | `data_mirror/v12-distill/gemma3-1b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000036/eval.json` | `a5_corner_second_difference` |
| `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000074/eval.json` | `data_mirror/v12-distill/gemma3-1b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000074/eval.json` | `a5_corner_second_difference` |
| `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234/trajectory/update-00000000/eval.json` | `data_mirror/v12-distill/gemma3-1b/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234/trajectory/update-00000000/eval.json` | `a5_corner_second_difference` |
| `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234/trajectory/update-00000074/eval.json` | `data_mirror/v12-distill/gemma3-1b/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234/trajectory/update-00000074/eval.json` | `a5_corner_second_difference` |
| `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_66_matrix2_lora_dseed41/trajectory/update-00000000/eval.json` | `data_mirror/v12-distill/gemma3-1b/gpt-5.6-luna_full_66_matrix2_lora_dseed41/trajectory/update-00000000/eval.json` | `a5_corner_second_difference` |
| `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_66_matrix2_lora_dseed41/trajectory/update-00000038/eval.json` | `data_mirror/v12-distill/gemma3-1b/gpt-5.6-luna_full_66_matrix2_lora_dseed41/trajectory/update-00000038/eval.json` | `a5_corner_second_difference` |
| `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000000/eval.json` | `data_mirror/v12-distill/gemma3-4b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000000/eval.json` | `a5_corner_second_difference` |
| `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000036/eval.json` | `data_mirror/v12-distill/gemma3-4b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000036/eval.json` | `a5_corner_second_difference` |
| `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000074/eval.json` | `data_mirror/v12-distill/gemma3-4b/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101/trajectory/update-00000074/eval.json` | `a5_corner_second_difference` |
| `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234/trajectory/update-00000000/eval.json` | `data_mirror/v12-distill/gemma3-4b/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234/trajectory/update-00000000/eval.json` | `a5_corner_second_difference` |
| `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234/trajectory/update-00000074/eval.json` | `data_mirror/v12-distill/gemma3-4b/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234/trajectory/update-00000074/eval.json` | `a5_corner_second_difference` |
| `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_66_matrix2_lora_dseed41/trajectory/update-00000000/eval.json` | `data_mirror/v12-distill/gemma3-4b/gpt-5.6-luna_full_66_matrix2_lora_dseed41/trajectory/update-00000000/eval.json` | `a5_corner_second_difference` |
| `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_66_matrix2_lora_dseed41/trajectory/update-00000038/eval.json` | `data_mirror/v12-distill/gemma3-4b/gpt-5.6-luna_full_66_matrix2_lora_dseed41/trajectory/update-00000038/eval.json` | `a5_corner_second_difference` |
| `results/v40-prune-strength/register.json` | `data_mirror/v40-prune-strength/register.json` | `final_prediction_table` |
| `results/v46-p1-newsource/compare.json` | `data_mirror/v46-p1-newsource/compare.json` | `explicit_directory_inventory`, `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v46-p1-newsource/dense.json` | `data_mirror/v46-p1-newsource/dense.json` | `explicit_directory_inventory` |
| `results/v46-p1-newsource/predictions_frozen.json` | `data_mirror/v46-p1-newsource/predictions_frozen.json` | `explicit_directory_inventory`, `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v47-p2-register/register.json` | `data_mirror/v47-p2-register/register.json` | `explicit_directory_inventory`, `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v50-p2v2/compare_test.json` | `data_mirror/v50-p2v2/compare_test.json` | `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v53-prune-dev/compare_pythia-1.4b@step112000.json` | `data_mirror/v53-prune-dev/compare_pythia-1.4b--step112000.json` | `explicit_directory_inventory`, `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v53-prune-dev/compare_pythia-410m@step48000.json` | `data_mirror/v53-prune-dev/compare_pythia-410m--step48000.json` | `explicit_directory_inventory`, `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v53-prune-dev/compare_pythia-6.9b@step80000.json` | `data_mirror/v53-prune-dev/compare_pythia-6.9b--step80000.json` | `explicit_directory_inventory`, `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v53-prune-dev/dense_dir_pythia-1.4b@step112000/dense.json` | `data_mirror/v53-prune-dev/dense_dir_pythia-1.4b--step112000/dense.json` | `explicit_directory_inventory` |
| `results/v53-prune-dev/dense_dir_pythia-410m@step48000/dense.json` | `data_mirror/v53-prune-dev/dense_dir_pythia-410m--step48000/dense.json` | `explicit_directory_inventory` |
| `results/v53-prune-dev/dense_dir_pythia-6.9b@step80000/dense.json` | `data_mirror/v53-prune-dev/dense_dir_pythia-6.9b--step80000/dense.json` | `explicit_directory_inventory` |
| `results/v53-prune-dev/dense_pythia-1.4b@step112000.json` | `data_mirror/v53-prune-dev/dense_pythia-1.4b--step112000.json` | `explicit_directory_inventory` |
| `results/v53-prune-dev/dense_pythia-410m@step48000.json` | `data_mirror/v53-prune-dev/dense_pythia-410m--step48000.json` | `explicit_directory_inventory` |
| `results/v53-prune-dev/dense_pythia-6.9b@step80000.json` | `data_mirror/v53-prune-dev/dense_pythia-6.9b--step80000.json` | `explicit_directory_inventory` |
| `results/v53-prune-dev/loso_table.csv` | `data_mirror/v53-prune-dev/loso_table.csv` | `explicit_directory_inventory` |
| `results/v53-prune-dev/predictions_pythia-1.4b@step112000.json` | `data_mirror/v53-prune-dev/predictions_pythia-1.4b--step112000.json` | `explicit_directory_inventory`, `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v53-prune-dev/predictions_pythia-410m@step48000.json` | `data_mirror/v53-prune-dev/predictions_pythia-410m--step48000.json` | `explicit_directory_inventory`, `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v53-prune-dev/predictions_pythia-6.9b@step80000.json` | `data_mirror/v53-prune-dev/predictions_pythia-6.9b--step80000.json` | `explicit_directory_inventory`, `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v53-prune-dev/register.json` | `data_mirror/v53-prune-dev/register.json` | `explicit_directory_inventory`, `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v55-quant-group/compare.json` | `data_mirror/v55-quant-group/compare.json` | `final_prediction_table` |
| `results/v55-quant-group/register.json` | `data_mirror/v55-quant-group/register.json` | `final_prediction_table` |
| `results/v69-quant-confirm/compare.json` | `data_mirror/v69-quant-confirm/compare.json` | `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v69-quant-confirm/develop.json` | `data_mirror/v69-quant-confirm/develop.json` | `final_prediction_table`, `plot_fig_generalization` |
| `results/v69-quant-confirm/freeze.json` | `data_mirror/v69-quant-confirm/freeze.json` | `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v70-distill-confirm/compare.json` | `data_mirror/v70-distill-confirm/compare.json` | `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v70-distill-confirm/freeze.json` | `data_mirror/v70-distill-confirm/freeze.json` | `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v72-prune-repeat/compare.json` | `data_mirror/v72-prune-repeat/compare.json` | `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v72-prune-repeat/freeze.json` | `data_mirror/v72-prune-repeat/freeze.json` | `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/compare.json` | `data_mirror/v78-rule-confirm/compare.json` | `final_prediction_table` |
| `results/v78-rule-confirm/freeze.json` | `data_mirror/v78-rule-confirm/freeze.json` | `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1.4b--step32000/prune__d0.6.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1.4b--step32000/prune__d0.6.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1.4b--step32000/prune__d0.7.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1.4b--step32000/prune__d0.7.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1.4b--step32000/prune__d0.8.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1.4b--step32000/prune__d0.8.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1.4b--step32000/prune__d0.9.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1.4b--step32000/prune__d0.9.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__b3_g128.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__b3_g128.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__b3_g256.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__b3_g256.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__b3_g64.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__b3_g64.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__b4_g128.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__b4_g128.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__b4_g256.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__b4_g256.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__b4_g64.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__b4_g64.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__b5_g128.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__b5_g128.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__b5_g256.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__b5_g256.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__b5_g64.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__b5_g64.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__channel_b3.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__channel_b3.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__channel_b4.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__channel_b4.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__channel_b5.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__channel_b5.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__channel_b6.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__channel_b6.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__channel_b8.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1.4b--step32000/quant__channel_b8.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-160m--step32000/prune__d0.6.json` | `data_mirror/v78-rule-confirm/measurements/pythia-160m--step32000/prune__d0.6.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-160m--step32000/prune__d0.7.json` | `data_mirror/v78-rule-confirm/measurements/pythia-160m--step32000/prune__d0.7.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-160m--step32000/prune__d0.8.json` | `data_mirror/v78-rule-confirm/measurements/pythia-160m--step32000/prune__d0.8.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-160m--step32000/prune__d0.9.json` | `data_mirror/v78-rule-confirm/measurements/pythia-160m--step32000/prune__d0.9.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-160m--step32000/quant__b3_g128.json` | `data_mirror/v78-rule-confirm/measurements/pythia-160m--step32000/quant__b3_g128.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-160m--step32000/quant__b3_g256.json` | `data_mirror/v78-rule-confirm/measurements/pythia-160m--step32000/quant__b3_g256.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-160m--step32000/quant__b3_g64.json` | `data_mirror/v78-rule-confirm/measurements/pythia-160m--step32000/quant__b3_g64.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-160m--step32000/quant__b4_g128.json` | `data_mirror/v78-rule-confirm/measurements/pythia-160m--step32000/quant__b4_g128.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-160m--step32000/quant__b4_g256.json` | `data_mirror/v78-rule-confirm/measurements/pythia-160m--step32000/quant__b4_g256.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-160m--step32000/quant__b4_g64.json` | `data_mirror/v78-rule-confirm/measurements/pythia-160m--step32000/quant__b4_g64.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-160m--step32000/quant__b5_g128.json` | `data_mirror/v78-rule-confirm/measurements/pythia-160m--step32000/quant__b5_g128.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-160m--step32000/quant__b5_g256.json` | `data_mirror/v78-rule-confirm/measurements/pythia-160m--step32000/quant__b5_g256.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-160m--step32000/quant__b5_g64.json` | `data_mirror/v78-rule-confirm/measurements/pythia-160m--step32000/quant__b5_g64.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-160m--step32000/quant__channel_b3.json` | `data_mirror/v78-rule-confirm/measurements/pythia-160m--step32000/quant__channel_b3.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-160m--step32000/quant__channel_b4.json` | `data_mirror/v78-rule-confirm/measurements/pythia-160m--step32000/quant__channel_b4.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-160m--step32000/quant__channel_b5.json` | `data_mirror/v78-rule-confirm/measurements/pythia-160m--step32000/quant__channel_b5.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-160m--step32000/quant__channel_b6.json` | `data_mirror/v78-rule-confirm/measurements/pythia-160m--step32000/quant__channel_b6.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-160m--step32000/quant__channel_b8.json` | `data_mirror/v78-rule-confirm/measurements/pythia-160m--step32000/quant__channel_b8.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1b--step64000/prune__d0.6.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1b--step64000/prune__d0.6.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1b--step64000/prune__d0.7.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1b--step64000/prune__d0.7.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1b--step64000/prune__d0.8.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1b--step64000/prune__d0.8.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1b--step64000/prune__d0.9.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1b--step64000/prune__d0.9.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1b--step64000/quant__b3_g128.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1b--step64000/quant__b3_g128.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1b--step64000/quant__b3_g256.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1b--step64000/quant__b3_g256.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1b--step64000/quant__b3_g64.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1b--step64000/quant__b3_g64.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1b--step64000/quant__b4_g128.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1b--step64000/quant__b4_g128.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1b--step64000/quant__b4_g256.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1b--step64000/quant__b4_g256.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1b--step64000/quant__b4_g64.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1b--step64000/quant__b4_g64.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1b--step64000/quant__b5_g128.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1b--step64000/quant__b5_g128.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1b--step64000/quant__b5_g256.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1b--step64000/quant__b5_g256.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1b--step64000/quant__b5_g64.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1b--step64000/quant__b5_g64.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1b--step64000/quant__channel_b3.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1b--step64000/quant__channel_b3.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1b--step64000/quant__channel_b4.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1b--step64000/quant__channel_b4.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1b--step64000/quant__channel_b5.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1b--step64000/quant__channel_b5.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1b--step64000/quant__channel_b6.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1b--step64000/quant__channel_b6.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-1b--step64000/quant__channel_b8.json` | `data_mirror/v78-rule-confirm/measurements/pythia-1b--step64000/quant__channel_b8.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-410m--step32000/prune__d0.6.json` | `data_mirror/v78-rule-confirm/measurements/pythia-410m--step32000/prune__d0.6.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-410m--step32000/prune__d0.7.json` | `data_mirror/v78-rule-confirm/measurements/pythia-410m--step32000/prune__d0.7.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-410m--step32000/prune__d0.8.json` | `data_mirror/v78-rule-confirm/measurements/pythia-410m--step32000/prune__d0.8.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-410m--step32000/prune__d0.9.json` | `data_mirror/v78-rule-confirm/measurements/pythia-410m--step32000/prune__d0.9.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-410m--step32000/quant__b3_g128.json` | `data_mirror/v78-rule-confirm/measurements/pythia-410m--step32000/quant__b3_g128.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-410m--step32000/quant__b3_g256.json` | `data_mirror/v78-rule-confirm/measurements/pythia-410m--step32000/quant__b3_g256.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-410m--step32000/quant__b3_g64.json` | `data_mirror/v78-rule-confirm/measurements/pythia-410m--step32000/quant__b3_g64.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-410m--step32000/quant__b4_g128.json` | `data_mirror/v78-rule-confirm/measurements/pythia-410m--step32000/quant__b4_g128.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-410m--step32000/quant__b4_g256.json` | `data_mirror/v78-rule-confirm/measurements/pythia-410m--step32000/quant__b4_g256.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-410m--step32000/quant__b4_g64.json` | `data_mirror/v78-rule-confirm/measurements/pythia-410m--step32000/quant__b4_g64.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-410m--step32000/quant__b5_g128.json` | `data_mirror/v78-rule-confirm/measurements/pythia-410m--step32000/quant__b5_g128.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-410m--step32000/quant__b5_g256.json` | `data_mirror/v78-rule-confirm/measurements/pythia-410m--step32000/quant__b5_g256.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-410m--step32000/quant__b5_g64.json` | `data_mirror/v78-rule-confirm/measurements/pythia-410m--step32000/quant__b5_g64.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-410m--step32000/quant__channel_b3.json` | `data_mirror/v78-rule-confirm/measurements/pythia-410m--step32000/quant__channel_b3.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-410m--step32000/quant__channel_b4.json` | `data_mirror/v78-rule-confirm/measurements/pythia-410m--step32000/quant__channel_b4.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-410m--step32000/quant__channel_b5.json` | `data_mirror/v78-rule-confirm/measurements/pythia-410m--step32000/quant__channel_b5.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-410m--step32000/quant__channel_b6.json` | `data_mirror/v78-rule-confirm/measurements/pythia-410m--step32000/quant__channel_b6.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v78-rule-confirm/measurements/pythia-410m--step32000/quant__channel_b8.json` | `data_mirror/v78-rule-confirm/measurements/pythia-410m--step32000/quant__channel_b8.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v88-displacement-uncentred-run/summary.json` | `data_mirror/v88-displacement-uncentred-run/summary.json` | `plot_fig_explanation` |
| `results/v88-displacement/FINDINGS.md` | `data_mirror/v88-displacement/FINDINGS.md` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step143000/b3_g0.json` | `data_mirror/v88-displacement/pythia-1.4b--step143000/b3_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step143000/b3_g128.json` | `data_mirror/v88-displacement/pythia-1.4b--step143000/b3_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step143000/b4_g0.json` | `data_mirror/v88-displacement/pythia-1.4b--step143000/b4_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step143000/b4_g128.json` | `data_mirror/v88-displacement/pythia-1.4b--step143000/b4_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step143000/b5_g0.json` | `data_mirror/v88-displacement/pythia-1.4b--step143000/b5_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step143000/b5_g128.json` | `data_mirror/v88-displacement/pythia-1.4b--step143000/b5_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step143000/b6_g0.json` | `data_mirror/v88-displacement/pythia-1.4b--step143000/b6_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step143000/b8_g0.json` | `data_mirror/v88-displacement/pythia-1.4b--step143000/b8_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step143000/prune_d0.6.json` | `data_mirror/v88-displacement/pythia-1.4b--step143000/prune_d0.6.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step143000/prune_d0.65.json` | `data_mirror/v88-displacement/pythia-1.4b--step143000/prune_d0.65.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step143000/prune_d0.7.json` | `data_mirror/v88-displacement/pythia-1.4b--step143000/prune_d0.7.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step143000/prune_d0.75.json` | `data_mirror/v88-displacement/pythia-1.4b--step143000/prune_d0.75.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step143000/prune_d0.8.json` | `data_mirror/v88-displacement/pythia-1.4b--step143000/prune_d0.8.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step143000/prune_d0.85.json` | `data_mirror/v88-displacement/pythia-1.4b--step143000/prune_d0.85.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step143000/prune_d0.9.json` | `data_mirror/v88-displacement/pythia-1.4b--step143000/prune_d0.9.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step16000/b3_g0.json` | `data_mirror/v88-displacement/pythia-1.4b--step16000/b3_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step16000/b3_g128.json` | `data_mirror/v88-displacement/pythia-1.4b--step16000/b3_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step16000/b4_g0.json` | `data_mirror/v88-displacement/pythia-1.4b--step16000/b4_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step16000/b4_g128.json` | `data_mirror/v88-displacement/pythia-1.4b--step16000/b4_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step16000/b4_g32.json` | `data_mirror/v88-displacement/pythia-1.4b--step16000/b4_g32.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step16000/b4_g512.json` | `data_mirror/v88-displacement/pythia-1.4b--step16000/b4_g512.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step16000/b5_g0.json` | `data_mirror/v88-displacement/pythia-1.4b--step16000/b5_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step16000/b5_g128.json` | `data_mirror/v88-displacement/pythia-1.4b--step16000/b5_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step16000/b6_g0.json` | `data_mirror/v88-displacement/pythia-1.4b--step16000/b6_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step16000/b8_g0.json` | `data_mirror/v88-displacement/pythia-1.4b--step16000/b8_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step16000/prune_d0.6.json` | `data_mirror/v88-displacement/pythia-1.4b--step16000/prune_d0.6.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step16000/prune_d0.65.json` | `data_mirror/v88-displacement/pythia-1.4b--step16000/prune_d0.65.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step16000/prune_d0.7.json` | `data_mirror/v88-displacement/pythia-1.4b--step16000/prune_d0.7.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step16000/prune_d0.75.json` | `data_mirror/v88-displacement/pythia-1.4b--step16000/prune_d0.75.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step16000/prune_d0.8.json` | `data_mirror/v88-displacement/pythia-1.4b--step16000/prune_d0.8.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step16000/prune_d0.85.json` | `data_mirror/v88-displacement/pythia-1.4b--step16000/prune_d0.85.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step16000/prune_d0.9.json` | `data_mirror/v88-displacement/pythia-1.4b--step16000/prune_d0.9.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step64000/b3_g128.json` | `data_mirror/v88-displacement/pythia-1.4b--step64000/b3_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step64000/b4_g0.json` | `data_mirror/v88-displacement/pythia-1.4b--step64000/b4_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step64000/b4_g128.json` | `data_mirror/v88-displacement/pythia-1.4b--step64000/b4_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step64000/b5_g128.json` | `data_mirror/v88-displacement/pythia-1.4b--step64000/b5_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step64000/b6_g0.json` | `data_mirror/v88-displacement/pythia-1.4b--step64000/b6_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step64000/b8_g0.json` | `data_mirror/v88-displacement/pythia-1.4b--step64000/b8_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step64000/prune_d0.6.json` | `data_mirror/v88-displacement/pythia-1.4b--step64000/prune_d0.6.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step64000/prune_d0.7.json` | `data_mirror/v88-displacement/pythia-1.4b--step64000/prune_d0.7.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step64000/prune_d0.8.json` | `data_mirror/v88-displacement/pythia-1.4b--step64000/prune_d0.8.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-1.4b--step64000/prune_d0.9.json` | `data_mirror/v88-displacement/pythia-1.4b--step64000/prune_d0.9.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step143000/b3_g0.json` | `data_mirror/v88-displacement/pythia-160m--step143000/b3_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step143000/b3_g128.json` | `data_mirror/v88-displacement/pythia-160m--step143000/b3_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step143000/b4_g0.json` | `data_mirror/v88-displacement/pythia-160m--step143000/b4_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step143000/b4_g128.json` | `data_mirror/v88-displacement/pythia-160m--step143000/b4_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step143000/b5_g0.json` | `data_mirror/v88-displacement/pythia-160m--step143000/b5_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step143000/b5_g128.json` | `data_mirror/v88-displacement/pythia-160m--step143000/b5_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step143000/b6_g0.json` | `data_mirror/v88-displacement/pythia-160m--step143000/b6_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step143000/b8_g0.json` | `data_mirror/v88-displacement/pythia-160m--step143000/b8_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step143000/prune_d0.6.json` | `data_mirror/v88-displacement/pythia-160m--step143000/prune_d0.6.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step143000/prune_d0.65.json` | `data_mirror/v88-displacement/pythia-160m--step143000/prune_d0.65.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step143000/prune_d0.7.json` | `data_mirror/v88-displacement/pythia-160m--step143000/prune_d0.7.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step143000/prune_d0.75.json` | `data_mirror/v88-displacement/pythia-160m--step143000/prune_d0.75.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step143000/prune_d0.8.json` | `data_mirror/v88-displacement/pythia-160m--step143000/prune_d0.8.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step143000/prune_d0.85.json` | `data_mirror/v88-displacement/pythia-160m--step143000/prune_d0.85.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step143000/prune_d0.9.json` | `data_mirror/v88-displacement/pythia-160m--step143000/prune_d0.9.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step16000/b3_g0.json` | `data_mirror/v88-displacement/pythia-160m--step16000/b3_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step16000/b3_g128.json` | `data_mirror/v88-displacement/pythia-160m--step16000/b3_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step16000/b4_g0.json` | `data_mirror/v88-displacement/pythia-160m--step16000/b4_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step16000/b4_g128.json` | `data_mirror/v88-displacement/pythia-160m--step16000/b4_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step16000/b4_g32.json` | `data_mirror/v88-displacement/pythia-160m--step16000/b4_g32.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step16000/b4_g512.json` | `data_mirror/v88-displacement/pythia-160m--step16000/b4_g512.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step16000/b5_g0.json` | `data_mirror/v88-displacement/pythia-160m--step16000/b5_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step16000/b5_g128.json` | `data_mirror/v88-displacement/pythia-160m--step16000/b5_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step16000/b6_g0.json` | `data_mirror/v88-displacement/pythia-160m--step16000/b6_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step16000/b8_g0.json` | `data_mirror/v88-displacement/pythia-160m--step16000/b8_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step16000/prune_d0.6.json` | `data_mirror/v88-displacement/pythia-160m--step16000/prune_d0.6.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step16000/prune_d0.65.json` | `data_mirror/v88-displacement/pythia-160m--step16000/prune_d0.65.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step16000/prune_d0.7.json` | `data_mirror/v88-displacement/pythia-160m--step16000/prune_d0.7.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step16000/prune_d0.75.json` | `data_mirror/v88-displacement/pythia-160m--step16000/prune_d0.75.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step16000/prune_d0.8.json` | `data_mirror/v88-displacement/pythia-160m--step16000/prune_d0.8.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step16000/prune_d0.85.json` | `data_mirror/v88-displacement/pythia-160m--step16000/prune_d0.85.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step16000/prune_d0.9.json` | `data_mirror/v88-displacement/pythia-160m--step16000/prune_d0.9.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step64000/b3_g128.json` | `data_mirror/v88-displacement/pythia-160m--step64000/b3_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step64000/b4_g0.json` | `data_mirror/v88-displacement/pythia-160m--step64000/b4_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step64000/b4_g128.json` | `data_mirror/v88-displacement/pythia-160m--step64000/b4_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step64000/b5_g128.json` | `data_mirror/v88-displacement/pythia-160m--step64000/b5_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step64000/b6_g0.json` | `data_mirror/v88-displacement/pythia-160m--step64000/b6_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step64000/b8_g0.json` | `data_mirror/v88-displacement/pythia-160m--step64000/b8_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step64000/prune_d0.6.json` | `data_mirror/v88-displacement/pythia-160m--step64000/prune_d0.6.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step64000/prune_d0.7.json` | `data_mirror/v88-displacement/pythia-160m--step64000/prune_d0.7.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step64000/prune_d0.8.json` | `data_mirror/v88-displacement/pythia-160m--step64000/prune_d0.8.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-160m--step64000/prune_d0.9.json` | `data_mirror/v88-displacement/pythia-160m--step64000/prune_d0.9.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step143000/b3_g0.json` | `data_mirror/v88-displacement/pythia-410m--step143000/b3_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step143000/b3_g128.json` | `data_mirror/v88-displacement/pythia-410m--step143000/b3_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step143000/b4_g0.json` | `data_mirror/v88-displacement/pythia-410m--step143000/b4_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step143000/b4_g128.json` | `data_mirror/v88-displacement/pythia-410m--step143000/b4_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step143000/b5_g0.json` | `data_mirror/v88-displacement/pythia-410m--step143000/b5_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step143000/b5_g128.json` | `data_mirror/v88-displacement/pythia-410m--step143000/b5_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step143000/b6_g0.json` | `data_mirror/v88-displacement/pythia-410m--step143000/b6_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step143000/b8_g0.json` | `data_mirror/v88-displacement/pythia-410m--step143000/b8_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step143000/prune_d0.6.json` | `data_mirror/v88-displacement/pythia-410m--step143000/prune_d0.6.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step143000/prune_d0.65.json` | `data_mirror/v88-displacement/pythia-410m--step143000/prune_d0.65.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step143000/prune_d0.7.json` | `data_mirror/v88-displacement/pythia-410m--step143000/prune_d0.7.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step143000/prune_d0.75.json` | `data_mirror/v88-displacement/pythia-410m--step143000/prune_d0.75.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step143000/prune_d0.8.json` | `data_mirror/v88-displacement/pythia-410m--step143000/prune_d0.8.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step143000/prune_d0.85.json` | `data_mirror/v88-displacement/pythia-410m--step143000/prune_d0.85.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step143000/prune_d0.9.json` | `data_mirror/v88-displacement/pythia-410m--step143000/prune_d0.9.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step16000/b3_g128.json` | `data_mirror/v88-displacement/pythia-410m--step16000/b3_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step16000/b4_g0.json` | `data_mirror/v88-displacement/pythia-410m--step16000/b4_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step16000/b4_g128.json` | `data_mirror/v88-displacement/pythia-410m--step16000/b4_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step16000/b5_g128.json` | `data_mirror/v88-displacement/pythia-410m--step16000/b5_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step16000/b6_g0.json` | `data_mirror/v88-displacement/pythia-410m--step16000/b6_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step16000/b8_g0.json` | `data_mirror/v88-displacement/pythia-410m--step16000/b8_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step16000/prune_d0.6.json` | `data_mirror/v88-displacement/pythia-410m--step16000/prune_d0.6.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step16000/prune_d0.7.json` | `data_mirror/v88-displacement/pythia-410m--step16000/prune_d0.7.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step16000/prune_d0.8.json` | `data_mirror/v88-displacement/pythia-410m--step16000/prune_d0.8.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step16000/prune_d0.9.json` | `data_mirror/v88-displacement/pythia-410m--step16000/prune_d0.9.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step64000/b3_g128.json` | `data_mirror/v88-displacement/pythia-410m--step64000/b3_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step64000/b4_g0.json` | `data_mirror/v88-displacement/pythia-410m--step64000/b4_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step64000/b4_g128.json` | `data_mirror/v88-displacement/pythia-410m--step64000/b4_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step64000/b5_g128.json` | `data_mirror/v88-displacement/pythia-410m--step64000/b5_g128.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step64000/b6_g0.json` | `data_mirror/v88-displacement/pythia-410m--step64000/b6_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step64000/b8_g0.json` | `data_mirror/v88-displacement/pythia-410m--step64000/b8_g0.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step64000/prune_d0.6.json` | `data_mirror/v88-displacement/pythia-410m--step64000/prune_d0.6.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step64000/prune_d0.7.json` | `data_mirror/v88-displacement/pythia-410m--step64000/prune_d0.7.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step64000/prune_d0.8.json` | `data_mirror/v88-displacement/pythia-410m--step64000/prune_d0.8.json` | `plot_fig_explanation` |
| `results/v88-displacement/pythia-410m--step64000/prune_d0.9.json` | `data_mirror/v88-displacement/pythia-410m--step64000/prune_d0.9.json` | `plot_fig_explanation` |
| `results/v93-confirm-inputs/pythia-1.4b--step32000/code/descriptor_bv.json` | `data_mirror/v93-confirm-inputs/pythia-1.4b--step32000/code/descriptor_bv.json` | `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v93-confirm-inputs/pythia-1.4b--step32000/math/descriptor_bv.json` | `data_mirror/v93-confirm-inputs/pythia-1.4b--step32000/math/descriptor_bv.json` | `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v93-confirm-inputs/pythia-1.4b--step32000/qa/descriptor_bv.json` | `data_mirror/v93-confirm-inputs/pythia-1.4b--step32000/qa/descriptor_bv.json` | `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v93-confirm-inputs/pythia-160m--step32000/code/descriptor_bv.json` | `data_mirror/v93-confirm-inputs/pythia-160m--step32000/code/descriptor_bv.json` | `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v93-confirm-inputs/pythia-160m--step32000/math/descriptor_bv.json` | `data_mirror/v93-confirm-inputs/pythia-160m--step32000/math/descriptor_bv.json` | `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v93-confirm-inputs/pythia-160m--step32000/qa/descriptor_bv.json` | `data_mirror/v93-confirm-inputs/pythia-160m--step32000/qa/descriptor_bv.json` | `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v93-confirm-inputs/pythia-1b--step64000/code/descriptor_bv.json` | `data_mirror/v93-confirm-inputs/pythia-1b--step64000/code/descriptor_bv.json` | `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v93-confirm-inputs/pythia-1b--step64000/math/descriptor_bv.json` | `data_mirror/v93-confirm-inputs/pythia-1b--step64000/math/descriptor_bv.json` | `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v93-confirm-inputs/pythia-1b--step64000/qa/descriptor_bv.json` | `data_mirror/v93-confirm-inputs/pythia-1b--step64000/qa/descriptor_bv.json` | `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v93-confirm-inputs/pythia-410m--step32000/code/descriptor_bv.json` | `data_mirror/v93-confirm-inputs/pythia-410m--step32000/code/descriptor_bv.json` | `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v93-confirm-inputs/pythia-410m--step32000/math/descriptor_bv.json` | `data_mirror/v93-confirm-inputs/pythia-410m--step32000/math/descriptor_bv.json` | `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v93-confirm-inputs/pythia-410m--step32000/qa/descriptor_bv.json` | `data_mirror/v93-confirm-inputs/pythia-410m--step32000/qa/descriptor_bv.json` | `final_prediction_table`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v99-scope/a5-corners-gemma3-1b-poolA/gpt-5.6-luna_full_66_matrix2_lora_dseed41-01b39086b0f5e684/update-00000038.json` | `data_mirror/v99-scope/a5-corners-gemma3-1b-poolA/gpt-5.6-luna_full_66_matrix2_lora_dseed41-01b39086b0f5e684/update-00000038.json` | `a5_corner_second_difference`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v99-scope/a5-corners-gemma3-1b-poolB/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101-f8defe2cc5d54af5/update-00000036.json` | `data_mirror/v99-scope/a5-corners-gemma3-1b-poolB/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101-f8defe2cc5d54af5/update-00000036.json` | `a5_corner_second_difference`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v99-scope/a5-corners-gemma3-1b-poolB/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101-f8defe2cc5d54af5/update-00000074.json` | `data_mirror/v99-scope/a5-corners-gemma3-1b-poolB/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101-f8defe2cc5d54af5/update-00000074.json` | `a5_corner_second_difference`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v99-scope/a5-corners-gemma3-1b-poolC/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234-98e5d57052842e92/update-00000074.json` | `data_mirror/v99-scope/a5-corners-gemma3-1b-poolC/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234-98e5d57052842e92/update-00000074.json` | `a5_corner_second_difference`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v99-scope/a5-corners-gemma3-4b-poolA/gpt-5.6-luna_full_66_matrix2_lora_dseed41-0e02d1102f506ae9/update-00000038.json` | `data_mirror/v99-scope/a5-corners-gemma3-4b-poolA/gpt-5.6-luna_full_66_matrix2_lora_dseed41-0e02d1102f506ae9/update-00000038.json` | `a5_corner_second_difference`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v99-scope/a5-corners-gemma3-4b-poolB/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101-15fadb650795b560/update-00000036.json` | `data_mirror/v99-scope/a5-corners-gemma3-4b-poolB/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101-15fadb650795b560/update-00000036.json` | `a5_corner_second_difference`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v99-scope/a5-corners-gemma3-4b-poolB/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101-15fadb650795b560/update-00000074.json` | `data_mirror/v99-scope/a5-corners-gemma3-4b-poolB/gpt-5.6-luna_full_130_a3b_corners_1_4_poolB_lora_dseed101-15fadb650795b560/update-00000074.json` | `a5_corner_second_difference`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v99-scope/a5-corners-gemma3-4b-poolC/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234-2da4fe4688510efb/update-00000074.json` | `data_mirror/v99-scope/a5-corners-gemma3-4b-poolC/gpt-5.6-luna_full_279_a3b_corners_3_poolC_lora_dseed234-2da4fe4688510efb/update-00000074.json` | `a5_corner_second_difference`, `plot_fig_generalization`, `plot_fig_generalization_cells` |
| `results/v99-scope/protocol.json` | `data_mirror/v99-scope/protocol.json` | `plot_fig_generalization`, `plot_fig_generalization_cells` |
