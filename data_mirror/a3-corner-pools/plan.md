# A3b corrected corner pool plan

Status: **PASSED — within all four 0.5% limits**.

CPU only; local tokenizer files only; no training, GPU work, or model weights loaded.

## Corrected objective

Corner 2 is the existing A measurement: D_A=16962, T1_A=50563, E_2=50563/16962. The replayed boundaries determine ideal D_B=T2_B×16962/50563 and D_C=D_B×T2_C/T1_B. Actual D values are token-counted candidate pools; no doubled-pool target is imposed.

| mismatch | definition | achieved % | limit % |
|---|---|---:|---:|
| budget_low | abs(T1_B−50563)/50563 | 0.009888654 | 0.5 |
| budget_high | abs(T2_C−T2_B)/min(T2_C,T2_B) | 0.016123984 | 0.5 |
| reuse_low | abs(E_1−E_3)/min(E_1,E_3) | 0.009019533 | 0.5 |
| reuse_high | abs(E_2−E_4)/min(E_2,E_4) | 0.004125362 | 0.5 |

Worst mismatch: **0.016123984%**. Conservative budget contrast min(T2_B,T2_C)/max(50563,T1_B)=2.085180863; reuse contrast E_2/E_1=2.085809371; both must be ≥1.8.

## Boundary-grid search

All four recorded pool totals (16962, 17104, 34503, 34639), trainer pool hashes, and every logged update boundary matched independently for both students before search.

Candidate space: B n/domain 124–140; C 248–280; seeds 70–269 excluding every observed/previously used seed and reserved 60–63. B and C also use distinct seeds.
Every pool's full schedule grid is replayed. Targets must have three completed boundaries on each side; domain-share drift against A is a hard ≤2 percentage point constraint on every domain.
Search B first, then C conditional on achieved B. Complete all B pairs whose two known residuals can improve the incumbent; this certifies the minimum of all four residuals over the enumerated grids. Rank by worst mismatch, then maximum B/C domain drift. The B table profiles each B against its best conditional C; the C table freezes the chosen B. Rows represent distinct pools.

### Best pools: B (with its best conditional C)

3366 pools; 486538 boundaries; 2073 pools excluded by drift; 0 by recorded hash.

| rank | n/seed | D | T1_B / T2_B / T2_C | budget low % | budget high % | reuse low % | reuse high % | worst % | max drift pp | paired n/seed |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 130/101 | 35376 | 50558 / 105450 / 105433 | 0.009888654 | 0.016123984 | 0.009019533 | 0.004125362 | 0.016123984 | 1.764213 | 279/234 |
| 2 | 127/130 | 33948 | 50537 / 101176 / 101164 | 0.051421000 | 0.011861927 | 0.061042613 | 0.021292578 | 0.061042613 | 1.822346 | 256/167 |
| 3 | 127/194 | 33925 | 50552 / 101050 / 101109 | 0.021755038 | 0.058386937 | 0.031899003 | 0.078160275 | 0.078160275 | 1.950215 | 251/196 |
| 4 | 131/246 | 34660 | 50607 / 103412 / 103367 | 0.087020153 | 0.043534203 | 0.031874838 | 0.089058597 | 0.089058597 | 0.238622 | 266/124 |
| 5 | 133/240 | 35172 | 50606 / 104745 / 104658 | 0.085042422 | 0.083127902 | 0.037082544 | 0.096648888 | 0.096648888 | 0.751950 | 267/151 |

### Best pools: C conditional on chosen B

6534 pools; 935500 boundaries; 4115 pools excluded by drift; 0 by recorded hash.

| rank | n/seed | D | T1_B / T2_B / T2_C | budget low % | budget high % | reuse low % | reuse high % | worst % | max drift pp | paired n/seed |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 279/234 | 73766 | 50558 / 105450 / 105433 | 0.009888654 | 0.016123984 | 0.009019533 | 0.004125362 | 0.016123984 | 1.764213 | 130/101 |
| 2 | 275/128 | 73754 | 50558 / 105450 / 105417 | 0.009888654 | 0.031304249 | 0.010111952 | 0.004125362 | 0.031304249 | 1.566890 | 130/101 |
| 3 | 279/224 | 73843 | 50558 / 105450 / 105496 | 0.009888654 | 0.043622570 | 0.035581151 | 0.004125362 | 0.043622570 | 1.169356 | 130/101 |
| 4 | 276/240 | 73776 | 50558 / 105450 / 105395 | 0.009888654 | 0.052184639 | 0.040592916 | 0.004125362 | 0.052184639 | 1.145240 | 130/101 |
| 5 | 279/262 | 73755 | 50558 / 105450 / 105372 | 0.009888654 | 0.074023460 | 0.033946895 | 0.004125362 | 0.074023460 | 1.227315 | 130/101 |

Completed 67 B boundary pairs and 68 conditional C searches; all exclusions above the incumbent use a provable lower bound, not the 0.5% acceptance limit.

## Chosen pools and achieved corners

Both chosen trainer data_pool_sha256 values are distinct and **absent from every one of 32 recorded hashes** (963 JSON sources, including archives). Inventory was checked again after search.

Pool A: n/domain=66, seed=41, D=16962; data_pool_sha256 `6113a4e84a49e12a44245ed90fab0005118c94c914c98facac510dff030ec88e`.
Pool B: n/domain=130, seed=101, D=35376; data_pool_sha256 `c642c3ddb0f5becd6673776285286bbe753215e4eeb6954d4da2899551f0faa6`.
Pool C: n/domain=279, seed=234, D=73766; data_pool_sha256 `e4f04fd52f939b44a32db062936f28d4f7c8b51caf2ad0504bb8a66867b78f25`.

Boundary-derived ideal D_B=1788642900/50563 (35374.540672); ideal D_C conditional on actual B=1864898904/25279 (73772.653349).

| corner | pool | achieved supervised T | exact E=T/D | status |
|---:|---|---:|---|---|
| 1 | B | 50558 | 25279/17688 | predicted completed update |
| 2 | A | 50563 | 50563/16962 | existing; do not retrain |
| 3 | C | 105433 | 105433/73766 | predicted completed update |
| 4 | B | 105450 | 17575/5896 | predicted completed update |

### Domain shares

| pool | domain | selected / prepared | supervised tokens | share % | drift pp |
|---|---|---:|---:|---:|---:|
| A | math | 66 / 66 | 6608 | 38.957670 | +0.000000 |
| A | qa | 66 / 65 | 820 | 4.834336 | +0.000000 |
| A | code | 66 / 66 | 9534 | 56.207994 | +0.000000 |
| B | math | 130 / 130 | 14134 | 39.953641 | +0.995971 |
| B | qa | 130 / 129 | 1763 | 4.983605 | +0.149269 |
| B | code | 130 / 128 | 19479 | 55.062754 | -1.145240 |
| C | math | 279 / 279 | 29944 | 40.593227 | +1.635557 |
| C | qa | 279 / 275 | 3661 | 4.962991 | +0.128656 |
| C | code | 279 / 274 | 40161 | 54.443782 | -1.764213 |

## Residual sensitivity of the second difference

Second difference: **L4−L3−L2+L1**, in native-token nats. Adjacent checkpoints in each existing run give s=(loss_next−loss_prev)/(T_next−T_prev), using pairs bracketing the relevant achieved budget. The table reports the largest absolute effect across these measured local slopes for each student/distribution.

| residual | signed equivalent supervised displacement | contribution |
|---|---:|---|
| budget_low | -5/1 | +s×δT |
| budget_high | -17/1 | −s×δT |
| reuse_low | 7645/804 | −s×δT |
| reuse_high | -1118/257 | +s×δT |

Reuse-low displacement is T2_C−D_C×T1_B/D_B; reuse-high displacement is T2_B−D_B×50563/16962.

| student | distribution | budget low nats | budget high nats | reuse low nats | reuse high nats |
|---|---|---:|---:|---:|---:|
| gemma3-1b | training_probe:MBPP:d5f45bdd0f442e0e | 0.00001951 | 0.00003331 | 0.00001863 | 0.00000852 |
| gemma3-1b | training_probe:MATH-500:c76472bee07727fe | 0.00001090 | 0.00002486 | 0.00001390 | 0.00000636 |
| gemma3-1b | training_probe:2WikiMultihopQA:35e3fde33c8b6e91 | 0.00010310 | 0.00043685 | 0.00024435 | 0.00011179 |
| gemma3-4b | training_probe:MBPP:d5f45bdd0f442e0e | 0.00001531 | 0.00002346 | 0.00001312 | 0.00000600 |
| gemma3-4b | training_probe:MATH-500:c76472bee07727fe | 0.00001728 | 0.00004990 | 0.00002791 | 0.00001277 |
| gemma3-4b | training_probe:2WikiMultihopQA:35e3fde33c8b6e91 | 0.00030553 | 0.00063272 | 0.00035390 | 0.00016191 |

Slopes follow fixed-pool trajectories, where T and E both change. These are local budget-equivalent sensitivities, not identified fixed-T reuse effects or rigorous bias bounds. The four edge corrections are dependent: do not sum them. Dense new ladders will measure local slopes on the chosen pools; reuse-only contamination still needs a surface assumption or matched-budget evidence.

All slope endpoints, losses, source paths, signed effects, and slope ranges are recorded in plan.json. Slopes are diagnostics only and never enter pool selection.

## Processed thresholds and dense local ladders

Frozen condition at `analysis/v12_distill.py:887`: `next_milestone < len(milestones) and accounting.processed >= milestones[next_milestone]`. --trajectory-tokens carries processed prompt+completion tokens. Each threshold below equals the processed total at its selected completed update, so it selects that boundary exactly. All supervised totals are replay predictions.

Corner 2 was requested at 170000 processed tokens and landed at update 38: 171860 processed / 50563 supervised, for both students.
Each target includes updates −3, −2, −1, 0, +1, +2, +3. The final three updates supply the upper side of the ladder; --stop-after-trajectory stops at the last ladder boundary. Intermediate probes preserve training state.
--schedule-tokens 678000 keeps the processed-token schedule horizon: ceil(678000/(pool processed tokens/ceil(prepared examples/16))) updates. --epochs 60 is retained; with schedule_tokens, the effective epoch limit is derived from that horizon.

## Four future trajectory commands

Two new trajectories per student (B for corners 1/4, C for corner 3), four total: the round cap. Commands are emitted only; none was executed. The CUDA device occurs only in these future-training command strings. The plan passes the specified tolerances.

### gemma3-1b

Reuse corner 2: `results/v12-distill/gemma3-1b/gpt-5.6-luna_full_66_matrix2_lora_dseed41/trajectory/update-00000038/eval.json`; no new A trajectory.

Pool B, corners [1, 4]; schedule 145 updates, warmup 4 updates, effective epoch limit 6.

```bash
python3 analysis/v12_distill.py --student gemma3-1b --teacher gpt-5.6-luna --recipe full --domains math,qa,code --n-per-domain 130 --data-seed 101 --seed 0 --trajectory-tokens 156194 161056 166153 170151 175006 181735 185746 338946 342526 346889 350853 352149 356758 362175 --stop-after-trajectory --schedule-tokens 678000 --epochs 60 --lr 1e-4 --training-mode lora --save-trajectory --output-suffix a3b_corners_1_4_poolB --device cuda
```

Pool C, corners [3]; schedule 142 updates, warmup 4 updates, effective epoch limit 3.

```bash
python3 analysis/v12_distill.py --student gemma3-1b --teacher gpt-5.6-luna --recipe full --domains math,qa,code --n-per-domain 279 --data-seed 234 --seed 0 --trajectory-tokens 339553 344629 348717 352731 357055 362785 366835 --stop-after-trajectory --schedule-tokens 678000 --epochs 60 --lr 1e-4 --training-mode lora --save-trajectory --output-suffix a3b_corners_3_poolC --device cuda
```

### gemma3-4b

Reuse corner 2: `results/v12-distill/gemma3-4b/gpt-5.6-luna_full_66_matrix2_lora_dseed41/trajectory/update-00000038/eval.json`; no new A trajectory.

Pool B, corners [1, 4]; schedule 145 updates, warmup 4 updates, effective epoch limit 6.

```bash
python3 analysis/v12_distill.py --student gemma3-4b --teacher gpt-5.6-luna --recipe full --domains math,qa,code --n-per-domain 130 --data-seed 101 --seed 0 --trajectory-tokens 156194 161056 166153 170151 175006 181735 185746 338946 342526 346889 350853 352149 356758 362175 --stop-after-trajectory --schedule-tokens 678000 --epochs 60 --lr 1e-4 --training-mode lora --save-trajectory --output-suffix a3b_corners_1_4_poolB --device cuda
```

Pool C, corners [3]; schedule 142 updates, warmup 4 updates, effective epoch limit 3.

```bash
python3 analysis/v12_distill.py --student gemma3-4b --teacher gpt-5.6-luna --recipe full --domains math,qa,code --n-per-domain 279 --data-seed 234 --seed 0 --trajectory-tokens 339553 344629 348717 352731 357055 362785 366835 --stop-after-trajectory --schedule-tokens 678000 --epochs 60 --lr 1e-4 --training-mode lora --save-trajectory --output-suffix a3b_corners_3_poolC --device cuda
```

### Requested boundaries (identical for both students)

Pool B:

| update | processed threshold | predicted supervised total | target-relative update |
|---:|---:|---:|---|
| 33 | 156194 | 46348 | corner 1: -3 |
| 34 | 161056 | 47630 | corner 1: -2 |
| 35 | 166153 | 49189 | corner 1: -1 |
| 36 | 170151 | 50558 | corner 1: +0 |
| 37 | 175006 | 52086 | corner 1: +1 |
| 38 | 181735 | 54152 | corner 1: +2 |
| 39 | 185746 | 55468 | corner 1: +3 |
| 71 | 338946 | 101095 | corner 4: -3 |
| 72 | 342526 | 102704 | corner 4: -2 |
| 73 | 346889 | 103832 | corner 4: -1 |
| 74 | 350853 | 105450 | corner 4: +0 |
| 75 | 352149 | 106128 | corner 4: +1 |
| 76 | 356758 | 107466 | corner 4: +2 |
| 77 | 362175 | 108709 | corner 4: +3 |

Pool C:

| update | processed threshold | predicted supervised total | target-relative update |
|---:|---:|---:|---|
| 71 | 339553 | 100894 | corner 3: -3 |
| 72 | 344629 | 102849 | corner 3: -2 |
| 73 | 348717 | 104204 | corner 3: -1 |
| 74 | 352731 | 105433 | corner 3: +0 |
| 75 | 357055 | 106917 | corner 3: +1 |
| 76 | 362785 | 108477 | corner 3: +2 |
| 77 | 366835 | 110140 | corner 3: +3 |

Use the same evaluation distributions at all four corners, including 2wiki_new. The existing corner-2 training-probe measurement is reused; its missing scope-QA evaluation remains necessary for that channel. This plan does not count unmeasured responses as observations.

Trainer remains byte-identical to A1: SHA256 `2d1d78b3562583a38fd15cda5f61ed9fbec7563ace7d44d9ddad7bd22225bc3a`.
Failure to meet any fixed 0.5% limit produces a failed, non-launchable plan and non-zero exit; counter/hash verification failures abort before search.
