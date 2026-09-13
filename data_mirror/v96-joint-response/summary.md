V96 joint response audit — CPU only, saved losses only

Part A comes first: the directory assertion passed (18 matrix2 trajectories selected; all five discarded _matrix_lora_dseed41/42 directories excluded, with no contents read).

100% of V95 T-only nonzero predictions and MAE gains/losses on I2 arise from residual budget mismatch; its T-only F has no other moving input.

V95 used max(T1,T2)/min(T1,T2)<=1.10, not exact equality. Every T-only effect reconstructs as a*[log(1+T2/100000)-log(1+T1/100000)]. At identical T the prediction and its MAE gain over zero are EXACTLY ZERO. No nonzero-budget exactly matched pool pair exists in these 18 trajectories.

Not recorded: logs contain total completion tokens and per-domain example counts, but no per-domain supervised-token totals or per-example token ledger. Example proportions and evaluation measurement_tokens are not training token shares.

Pool contents change with size as well as seed; pools are independently sampled, not nested. Actual schedule lengths vary 143–153 optimizer updates (warmup=4); checkpoint update, epoch and LR also vary. These are in every endpoint accounting record. T-only has none of these as inputs, so they do not explain its nonzero predictions, but they limit causal interpretation of measured effects.

Reused V95 T-only pool-change gains (zero MAE minus T-only MAE; 100% attributable to residual T mismatch):

| Split | Capability | Gain [95% CI] | Explained | Fixed-T prediction/gain |
|---|---|---|---|---|
| leave_one_pool_seed_out | code | -0.000485566 [-0.00119561, 0.000199575] (clusters=18) | 100% | 0 / 0 |
| leave_one_pool_seed_out | math | -0.000392074 [-0.000928582, 0.000216625] (clusters=18) | 100% | 0 / 0 |
| leave_one_pool_seed_out | qa | 0.00148298 [0.000573332, 0.00259027] (clusters=18) | 100% | 0 / 0 |
| leave_one_student_out | code | -0.000483693 [-0.00115798, 0.000196609] (clusters=18) | 100% | 0 / 0 |
| leave_one_student_out | math | -0.000450698 [-0.00104571, 0.00015864] (clusters=18) | 100% | 0 / 0 |
| leave_one_student_out | qa | 0.00129404 [0.000314846, 0.00231886] (clusters=18) | 100% | 0 / 0 |

The counterfactual fixed-T prediction is algebraic, not a measured fixed-T effect. No exact nonzero-T pool observations exist. The decomposition explains model gains, not how much of the observed loss difference is causally due to budget.

Part A: recorded pool-contrast accounting

Checkpoint is ordinal, not a nominal supervised-token target. Shares are math/code/qa; NA means not recorded. Ratio is T2/T1. Every seed remains explicit.

Pool size contrasts

| Capability | Student | Checkpoint | Contrast | Seeds | T1 | T2 | T2/T1 | D_U1 | D_U2 | Shares1 m/c/q | Shares2 m/c/q | <=1.10 max/min |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|---|
| math | gemma3-270m | 1 | small->middle | 41->41 | 25460 | 24623 | 0.96712490 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 2 | small->middle | 41->41 | 50563 | 49836 | 0.98562190 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 3 | small->middle | 41->41 | 100114 | 101139 | 1.01023833 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 4 | small->middle | 41->41 | 199166 | 198736 | 0.99784100 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 1 | small->large | 41->41 | 25460 | 27919 | 1.09658288 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 2 | small->large | 41->41 | 50563 | 53525 | 1.05858038 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 3 | small->large | 41->41 | 100114 | 104539 | 1.04419961 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 4 | small->large | 41->41 | 199166 | 198722 | 0.99777070 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 1 | middle->large | 41->41 | 24623 | 27919 | 1.13385859 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | False |
| math | gemma3-270m | 2 | middle->large | 41->41 | 49836 | 53525 | 1.07402279 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 3 | middle->large | 41->41 | 101139 | 104539 | 1.03361710 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 4 | middle->large | 41->41 | 198736 | 198722 | 0.99992955 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 1 | small->middle | 42->42 | 24822 | 25396 | 1.02312465 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 2 | small->middle | 42->42 | 50869 | 50541 | 0.99355207 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 3 | small->middle | 42->42 | 99576 | 101373 | 1.01804652 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 4 | small->middle | 42->42 | 199673 | 199848 | 1.00087643 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 1 | small->large | 42->42 | 24822 | 24067 | 0.96958343 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 2 | small->large | 42->42 | 50869 | 50038 | 0.98366392 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 3 | small->large | 42->42 | 99576 | 102110 | 1.02544790 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 4 | small->large | 42->42 | 199673 | 201946 | 1.01138361 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 1 | middle->large | 42->42 | 25396 | 24067 | 0.94766892 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 2 | middle->large | 42->42 | 50541 | 50038 | 0.99004768 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 3 | middle->large | 42->42 | 101373 | 102110 | 1.00727018 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 4 | middle->large | 42->42 | 199848 | 201946 | 1.01049798 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 1 | small->middle | 41->41 | 25460 | 24623 | 0.96712490 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 2 | small->middle | 41->41 | 50563 | 49836 | 0.98562190 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 3 | small->middle | 41->41 | 100114 | 101139 | 1.01023833 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 4 | small->middle | 41->41 | 199166 | 198736 | 0.99784100 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 1 | small->large | 41->41 | 25460 | 27919 | 1.09658288 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 2 | small->large | 41->41 | 50563 | 53525 | 1.05858038 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 3 | small->large | 41->41 | 100114 | 104539 | 1.04419961 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 4 | small->large | 41->41 | 199166 | 198722 | 0.99777070 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 1 | middle->large | 41->41 | 24623 | 27919 | 1.13385859 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | False |
| math | gemma3-1b | 2 | middle->large | 41->41 | 49836 | 53525 | 1.07402279 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 3 | middle->large | 41->41 | 101139 | 104539 | 1.03361710 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 4 | middle->large | 41->41 | 198736 | 198722 | 0.99992955 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 1 | small->middle | 42->42 | 24822 | 25396 | 1.02312465 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 2 | small->middle | 42->42 | 50869 | 50541 | 0.99355207 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 3 | small->middle | 42->42 | 99576 | 101373 | 1.01804652 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 4 | small->middle | 42->42 | 199673 | 199848 | 1.00087643 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 1 | small->large | 42->42 | 24822 | 24067 | 0.96958343 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 2 | small->large | 42->42 | 50869 | 50038 | 0.98366392 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 3 | small->large | 42->42 | 99576 | 102110 | 1.02544790 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 4 | small->large | 42->42 | 199673 | 201946 | 1.01138361 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 1 | middle->large | 42->42 | 25396 | 24067 | 0.94766892 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 2 | middle->large | 42->42 | 50541 | 50038 | 0.99004768 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 3 | middle->large | 42->42 | 101373 | 102110 | 1.00727018 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 4 | middle->large | 42->42 | 199848 | 201946 | 1.01049798 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 1 | small->middle | 41->41 | 25460 | 24623 | 0.96712490 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 2 | small->middle | 41->41 | 50563 | 49836 | 0.98562190 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 3 | small->middle | 41->41 | 100114 | 101139 | 1.01023833 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 4 | small->middle | 41->41 | 199166 | 198736 | 0.99784100 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 1 | small->large | 41->41 | 25460 | 27919 | 1.09658288 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 2 | small->large | 41->41 | 50563 | 53525 | 1.05858038 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 3 | small->large | 41->41 | 100114 | 104539 | 1.04419961 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 4 | small->large | 41->41 | 199166 | 198722 | 0.99777070 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 1 | middle->large | 41->41 | 24623 | 27919 | 1.13385859 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | False |
| math | gemma3-4b | 2 | middle->large | 41->41 | 49836 | 53525 | 1.07402279 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 3 | middle->large | 41->41 | 101139 | 104539 | 1.03361710 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 4 | middle->large | 41->41 | 198736 | 198722 | 0.99992955 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 1 | small->middle | 42->42 | 24822 | 25396 | 1.02312465 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 2 | small->middle | 42->42 | 50869 | 50541 | 0.99355207 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 3 | small->middle | 42->42 | 99576 | 101373 | 1.01804652 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 4 | small->middle | 42->42 | 199673 | 199848 | 1.00087643 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 1 | small->large | 42->42 | 24822 | 24067 | 0.96958343 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 2 | small->large | 42->42 | 50869 | 50038 | 0.98366392 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 3 | small->large | 42->42 | 99576 | 102110 | 1.02544790 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 4 | small->large | 42->42 | 199673 | 201946 | 1.01138361 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 1 | middle->large | 42->42 | 25396 | 24067 | 0.94766892 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 2 | middle->large | 42->42 | 50541 | 50038 | 0.99004768 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 3 | middle->large | 42->42 | 101373 | 102110 | 1.00727018 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 4 | middle->large | 42->42 | 199848 | 201946 | 1.01049798 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 1 | small->middle | 41->41 | 25460 | 24623 | 0.96712490 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 2 | small->middle | 41->41 | 50563 | 49836 | 0.98562190 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 3 | small->middle | 41->41 | 100114 | 101139 | 1.01023833 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 4 | small->middle | 41->41 | 199166 | 198736 | 0.99784100 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 1 | small->large | 41->41 | 25460 | 27919 | 1.09658288 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 2 | small->large | 41->41 | 50563 | 53525 | 1.05858038 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 3 | small->large | 41->41 | 100114 | 104539 | 1.04419961 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 4 | small->large | 41->41 | 199166 | 198722 | 0.99777070 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 1 | middle->large | 41->41 | 24623 | 27919 | 1.13385859 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | False |
| code | gemma3-270m | 2 | middle->large | 41->41 | 49836 | 53525 | 1.07402279 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 3 | middle->large | 41->41 | 101139 | 104539 | 1.03361710 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 4 | middle->large | 41->41 | 198736 | 198722 | 0.99992955 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 1 | small->middle | 42->42 | 24822 | 25396 | 1.02312465 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 2 | small->middle | 42->42 | 50869 | 50541 | 0.99355207 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 3 | small->middle | 42->42 | 99576 | 101373 | 1.01804652 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 4 | small->middle | 42->42 | 199673 | 199848 | 1.00087643 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 1 | small->large | 42->42 | 24822 | 24067 | 0.96958343 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 2 | small->large | 42->42 | 50869 | 50038 | 0.98366392 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 3 | small->large | 42->42 | 99576 | 102110 | 1.02544790 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 4 | small->large | 42->42 | 199673 | 201946 | 1.01138361 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 1 | middle->large | 42->42 | 25396 | 24067 | 0.94766892 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 2 | middle->large | 42->42 | 50541 | 50038 | 0.99004768 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 3 | middle->large | 42->42 | 101373 | 102110 | 1.00727018 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 4 | middle->large | 42->42 | 199848 | 201946 | 1.01049798 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 1 | small->middle | 41->41 | 25460 | 24623 | 0.96712490 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 2 | small->middle | 41->41 | 50563 | 49836 | 0.98562190 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 3 | small->middle | 41->41 | 100114 | 101139 | 1.01023833 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 4 | small->middle | 41->41 | 199166 | 198736 | 0.99784100 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 1 | small->large | 41->41 | 25460 | 27919 | 1.09658288 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 2 | small->large | 41->41 | 50563 | 53525 | 1.05858038 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 3 | small->large | 41->41 | 100114 | 104539 | 1.04419961 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 4 | small->large | 41->41 | 199166 | 198722 | 0.99777070 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 1 | middle->large | 41->41 | 24623 | 27919 | 1.13385859 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | False |
| code | gemma3-1b | 2 | middle->large | 41->41 | 49836 | 53525 | 1.07402279 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 3 | middle->large | 41->41 | 101139 | 104539 | 1.03361710 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 4 | middle->large | 41->41 | 198736 | 198722 | 0.99992955 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 1 | small->middle | 42->42 | 24822 | 25396 | 1.02312465 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 2 | small->middle | 42->42 | 50869 | 50541 | 0.99355207 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 3 | small->middle | 42->42 | 99576 | 101373 | 1.01804652 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 4 | small->middle | 42->42 | 199673 | 199848 | 1.00087643 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 1 | small->large | 42->42 | 24822 | 24067 | 0.96958343 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 2 | small->large | 42->42 | 50869 | 50038 | 0.98366392 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 3 | small->large | 42->42 | 99576 | 102110 | 1.02544790 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 4 | small->large | 42->42 | 199673 | 201946 | 1.01138361 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 1 | middle->large | 42->42 | 25396 | 24067 | 0.94766892 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 2 | middle->large | 42->42 | 50541 | 50038 | 0.99004768 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 3 | middle->large | 42->42 | 101373 | 102110 | 1.00727018 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 4 | middle->large | 42->42 | 199848 | 201946 | 1.01049798 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 1 | small->middle | 41->41 | 25460 | 24623 | 0.96712490 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 2 | small->middle | 41->41 | 50563 | 49836 | 0.98562190 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 3 | small->middle | 41->41 | 100114 | 101139 | 1.01023833 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 4 | small->middle | 41->41 | 199166 | 198736 | 0.99784100 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 1 | small->large | 41->41 | 25460 | 27919 | 1.09658288 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 2 | small->large | 41->41 | 50563 | 53525 | 1.05858038 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 3 | small->large | 41->41 | 100114 | 104539 | 1.04419961 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 4 | small->large | 41->41 | 199166 | 198722 | 0.99777070 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 1 | middle->large | 41->41 | 24623 | 27919 | 1.13385859 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | False |
| code | gemma3-4b | 2 | middle->large | 41->41 | 49836 | 53525 | 1.07402279 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 3 | middle->large | 41->41 | 101139 | 104539 | 1.03361710 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 4 | middle->large | 41->41 | 198736 | 198722 | 0.99992955 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 1 | small->middle | 42->42 | 24822 | 25396 | 1.02312465 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 2 | small->middle | 42->42 | 50869 | 50541 | 0.99355207 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 3 | small->middle | 42->42 | 99576 | 101373 | 1.01804652 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 4 | small->middle | 42->42 | 199673 | 199848 | 1.00087643 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 1 | small->large | 42->42 | 24822 | 24067 | 0.96958343 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 2 | small->large | 42->42 | 50869 | 50038 | 0.98366392 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 3 | small->large | 42->42 | 99576 | 102110 | 1.02544790 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 4 | small->large | 42->42 | 199673 | 201946 | 1.01138361 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 1 | middle->large | 42->42 | 25396 | 24067 | 0.94766892 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 2 | middle->large | 42->42 | 50541 | 50038 | 0.99004768 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 3 | middle->large | 42->42 | 101373 | 102110 | 1.00727018 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 4 | middle->large | 42->42 | 199848 | 201946 | 1.01049798 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 1 | small->middle | 41->41 | 25460 | 24623 | 0.96712490 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 2 | small->middle | 41->41 | 50563 | 49836 | 0.98562190 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 3 | small->middle | 41->41 | 100114 | 101139 | 1.01023833 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 4 | small->middle | 41->41 | 199166 | 198736 | 0.99784100 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 1 | small->large | 41->41 | 25460 | 27919 | 1.09658288 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 2 | small->large | 41->41 | 50563 | 53525 | 1.05858038 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 3 | small->large | 41->41 | 100114 | 104539 | 1.04419961 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 4 | small->large | 41->41 | 199166 | 198722 | 0.99777070 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 1 | middle->large | 41->41 | 24623 | 27919 | 1.13385859 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | False |
| qa | gemma3-270m | 2 | middle->large | 41->41 | 49836 | 53525 | 1.07402279 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 3 | middle->large | 41->41 | 101139 | 104539 | 1.03361710 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 4 | middle->large | 41->41 | 198736 | 198722 | 0.99992955 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 1 | small->middle | 42->42 | 24822 | 25396 | 1.02312465 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 2 | small->middle | 42->42 | 50869 | 50541 | 0.99355207 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 3 | small->middle | 42->42 | 99576 | 101373 | 1.01804652 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 4 | small->middle | 42->42 | 199673 | 199848 | 1.00087643 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 1 | small->large | 42->42 | 24822 | 24067 | 0.96958343 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 2 | small->large | 42->42 | 50869 | 50038 | 0.98366392 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 3 | small->large | 42->42 | 99576 | 102110 | 1.02544790 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 4 | small->large | 42->42 | 199673 | 201946 | 1.01138361 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 1 | middle->large | 42->42 | 25396 | 24067 | 0.94766892 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 2 | middle->large | 42->42 | 50541 | 50038 | 0.99004768 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 3 | middle->large | 42->42 | 101373 | 102110 | 1.00727018 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 4 | middle->large | 42->42 | 199848 | 201946 | 1.01049798 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 1 | small->middle | 41->41 | 25460 | 24623 | 0.96712490 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 2 | small->middle | 41->41 | 50563 | 49836 | 0.98562190 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 3 | small->middle | 41->41 | 100114 | 101139 | 1.01023833 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 4 | small->middle | 41->41 | 199166 | 198736 | 0.99784100 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 1 | small->large | 41->41 | 25460 | 27919 | 1.09658288 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 2 | small->large | 41->41 | 50563 | 53525 | 1.05858038 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 3 | small->large | 41->41 | 100114 | 104539 | 1.04419961 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 4 | small->large | 41->41 | 199166 | 198722 | 0.99777070 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 1 | middle->large | 41->41 | 24623 | 27919 | 1.13385859 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | False |
| qa | gemma3-1b | 2 | middle->large | 41->41 | 49836 | 53525 | 1.07402279 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 3 | middle->large | 41->41 | 101139 | 104539 | 1.03361710 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 4 | middle->large | 41->41 | 198736 | 198722 | 0.99992955 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 1 | small->middle | 42->42 | 24822 | 25396 | 1.02312465 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 2 | small->middle | 42->42 | 50869 | 50541 | 0.99355207 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 3 | small->middle | 42->42 | 99576 | 101373 | 1.01804652 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 4 | small->middle | 42->42 | 199673 | 199848 | 1.00087643 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 1 | small->large | 42->42 | 24822 | 24067 | 0.96958343 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 2 | small->large | 42->42 | 50869 | 50038 | 0.98366392 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 3 | small->large | 42->42 | 99576 | 102110 | 1.02544790 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 4 | small->large | 42->42 | 199673 | 201946 | 1.01138361 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 1 | middle->large | 42->42 | 25396 | 24067 | 0.94766892 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 2 | middle->large | 42->42 | 50541 | 50038 | 0.99004768 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 3 | middle->large | 42->42 | 101373 | 102110 | 1.00727018 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 4 | middle->large | 42->42 | 199848 | 201946 | 1.01049798 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 1 | small->middle | 41->41 | 25460 | 24623 | 0.96712490 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 2 | small->middle | 41->41 | 50563 | 49836 | 0.98562190 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 3 | small->middle | 41->41 | 100114 | 101139 | 1.01023833 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 4 | small->middle | 41->41 | 199166 | 198736 | 0.99784100 | 16962 | 51065 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 1 | small->large | 41->41 | 25460 | 27919 | 1.09658288 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 2 | small->large | 41->41 | 50563 | 53525 | 1.05858038 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 3 | small->large | 41->41 | 100114 | 104539 | 1.04419961 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 4 | small->large | 41->41 | 199166 | 198722 | 0.99777070 | 16962 | 156826 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 1 | middle->large | 41->41 | 24623 | 27919 | 1.13385859 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | False |
| qa | gemma3-4b | 2 | middle->large | 41->41 | 49836 | 53525 | 1.07402279 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 3 | middle->large | 41->41 | 101139 | 104539 | 1.03361710 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 4 | middle->large | 41->41 | 198736 | 198722 | 0.99992955 | 51065 | 156826 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 1 | small->middle | 42->42 | 24822 | 25396 | 1.02312465 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 2 | small->middle | 42->42 | 50869 | 50541 | 0.99355207 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 3 | small->middle | 42->42 | 99576 | 101373 | 1.01804652 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 4 | small->middle | 42->42 | 199673 | 199848 | 1.00087643 | 17104 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 1 | small->large | 42->42 | 24822 | 24067 | 0.96958343 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 2 | small->large | 42->42 | 50869 | 50038 | 0.98366392 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 3 | small->large | 42->42 | 99576 | 102110 | 1.02544790 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 4 | small->large | 42->42 | 199673 | 201946 | 1.01138361 | 17104 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 1 | middle->large | 42->42 | 25396 | 24067 | 0.94766892 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 2 | middle->large | 42->42 | 50541 | 50038 | 0.99004768 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 3 | middle->large | 42->42 | 101373 | 102110 | 1.00727018 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 4 | middle->large | 42->42 | 199848 | 201946 | 1.01049798 | 51812 | 156653 | NA/NA/NA | NA/NA/NA | True |

Pool seed contrasts

| Capability | Student | Checkpoint | Contrast | Seeds | T1 | T2 | T2/T1 | D_U1 | D_U2 | Shares1 m/c/q | Shares2 m/c/q | <=1.10 max/min |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|---|
| math | gemma3-270m | 1 | small:seed41->42 | 41->42 | 25460 | 24822 | 0.97494108 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 2 | small:seed41->42 | 41->42 | 50563 | 50869 | 1.00605186 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 3 | small:seed41->42 | 41->42 | 100114 | 99576 | 0.99462613 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 4 | small:seed41->42 | 41->42 | 199166 | 199673 | 1.00254562 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 1 | middle:seed41->42 | 41->42 | 24623 | 25396 | 1.03139341 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 2 | middle:seed41->42 | 41->42 | 49836 | 50541 | 1.01414640 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 3 | middle:seed41->42 | 41->42 | 101139 | 101373 | 1.00231365 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 4 | middle:seed41->42 | 41->42 | 198736 | 199848 | 1.00559536 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 1 | large:seed41->42 | 41->42 | 27919 | 24067 | 0.86202944 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | False |
| math | gemma3-270m | 2 | large:seed41->42 | 41->42 | 53525 | 50038 | 0.93485287 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 3 | large:seed41->42 | 41->42 | 104539 | 102110 | 0.97676465 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-270m | 4 | large:seed41->42 | 41->42 | 198722 | 201946 | 1.01622367 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 1 | small:seed41->42 | 41->42 | 25460 | 24822 | 0.97494108 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 2 | small:seed41->42 | 41->42 | 50563 | 50869 | 1.00605186 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 3 | small:seed41->42 | 41->42 | 100114 | 99576 | 0.99462613 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 4 | small:seed41->42 | 41->42 | 199166 | 199673 | 1.00254562 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 1 | middle:seed41->42 | 41->42 | 24623 | 25396 | 1.03139341 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 2 | middle:seed41->42 | 41->42 | 49836 | 50541 | 1.01414640 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 3 | middle:seed41->42 | 41->42 | 101139 | 101373 | 1.00231365 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 4 | middle:seed41->42 | 41->42 | 198736 | 199848 | 1.00559536 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 1 | large:seed41->42 | 41->42 | 27919 | 24067 | 0.86202944 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | False |
| math | gemma3-1b | 2 | large:seed41->42 | 41->42 | 53525 | 50038 | 0.93485287 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 3 | large:seed41->42 | 41->42 | 104539 | 102110 | 0.97676465 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-1b | 4 | large:seed41->42 | 41->42 | 198722 | 201946 | 1.01622367 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 1 | small:seed41->42 | 41->42 | 25460 | 24822 | 0.97494108 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 2 | small:seed41->42 | 41->42 | 50563 | 50869 | 1.00605186 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 3 | small:seed41->42 | 41->42 | 100114 | 99576 | 0.99462613 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 4 | small:seed41->42 | 41->42 | 199166 | 199673 | 1.00254562 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 1 | middle:seed41->42 | 41->42 | 24623 | 25396 | 1.03139341 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 2 | middle:seed41->42 | 41->42 | 49836 | 50541 | 1.01414640 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 3 | middle:seed41->42 | 41->42 | 101139 | 101373 | 1.00231365 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 4 | middle:seed41->42 | 41->42 | 198736 | 199848 | 1.00559536 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 1 | large:seed41->42 | 41->42 | 27919 | 24067 | 0.86202944 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | False |
| math | gemma3-4b | 2 | large:seed41->42 | 41->42 | 53525 | 50038 | 0.93485287 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 3 | large:seed41->42 | 41->42 | 104539 | 102110 | 0.97676465 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| math | gemma3-4b | 4 | large:seed41->42 | 41->42 | 198722 | 201946 | 1.01622367 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 1 | small:seed41->42 | 41->42 | 25460 | 24822 | 0.97494108 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 2 | small:seed41->42 | 41->42 | 50563 | 50869 | 1.00605186 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 3 | small:seed41->42 | 41->42 | 100114 | 99576 | 0.99462613 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 4 | small:seed41->42 | 41->42 | 199166 | 199673 | 1.00254562 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 1 | middle:seed41->42 | 41->42 | 24623 | 25396 | 1.03139341 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 2 | middle:seed41->42 | 41->42 | 49836 | 50541 | 1.01414640 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 3 | middle:seed41->42 | 41->42 | 101139 | 101373 | 1.00231365 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 4 | middle:seed41->42 | 41->42 | 198736 | 199848 | 1.00559536 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 1 | large:seed41->42 | 41->42 | 27919 | 24067 | 0.86202944 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | False |
| code | gemma3-270m | 2 | large:seed41->42 | 41->42 | 53525 | 50038 | 0.93485287 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 3 | large:seed41->42 | 41->42 | 104539 | 102110 | 0.97676465 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-270m | 4 | large:seed41->42 | 41->42 | 198722 | 201946 | 1.01622367 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 1 | small:seed41->42 | 41->42 | 25460 | 24822 | 0.97494108 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 2 | small:seed41->42 | 41->42 | 50563 | 50869 | 1.00605186 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 3 | small:seed41->42 | 41->42 | 100114 | 99576 | 0.99462613 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 4 | small:seed41->42 | 41->42 | 199166 | 199673 | 1.00254562 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 1 | middle:seed41->42 | 41->42 | 24623 | 25396 | 1.03139341 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 2 | middle:seed41->42 | 41->42 | 49836 | 50541 | 1.01414640 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 3 | middle:seed41->42 | 41->42 | 101139 | 101373 | 1.00231365 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 4 | middle:seed41->42 | 41->42 | 198736 | 199848 | 1.00559536 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 1 | large:seed41->42 | 41->42 | 27919 | 24067 | 0.86202944 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | False |
| code | gemma3-1b | 2 | large:seed41->42 | 41->42 | 53525 | 50038 | 0.93485287 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 3 | large:seed41->42 | 41->42 | 104539 | 102110 | 0.97676465 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-1b | 4 | large:seed41->42 | 41->42 | 198722 | 201946 | 1.01622367 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 1 | small:seed41->42 | 41->42 | 25460 | 24822 | 0.97494108 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 2 | small:seed41->42 | 41->42 | 50563 | 50869 | 1.00605186 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 3 | small:seed41->42 | 41->42 | 100114 | 99576 | 0.99462613 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 4 | small:seed41->42 | 41->42 | 199166 | 199673 | 1.00254562 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 1 | middle:seed41->42 | 41->42 | 24623 | 25396 | 1.03139341 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 2 | middle:seed41->42 | 41->42 | 49836 | 50541 | 1.01414640 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 3 | middle:seed41->42 | 41->42 | 101139 | 101373 | 1.00231365 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 4 | middle:seed41->42 | 41->42 | 198736 | 199848 | 1.00559536 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 1 | large:seed41->42 | 41->42 | 27919 | 24067 | 0.86202944 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | False |
| code | gemma3-4b | 2 | large:seed41->42 | 41->42 | 53525 | 50038 | 0.93485287 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 3 | large:seed41->42 | 41->42 | 104539 | 102110 | 0.97676465 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| code | gemma3-4b | 4 | large:seed41->42 | 41->42 | 198722 | 201946 | 1.01622367 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 1 | small:seed41->42 | 41->42 | 25460 | 24822 | 0.97494108 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 2 | small:seed41->42 | 41->42 | 50563 | 50869 | 1.00605186 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 3 | small:seed41->42 | 41->42 | 100114 | 99576 | 0.99462613 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 4 | small:seed41->42 | 41->42 | 199166 | 199673 | 1.00254562 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 1 | middle:seed41->42 | 41->42 | 24623 | 25396 | 1.03139341 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 2 | middle:seed41->42 | 41->42 | 49836 | 50541 | 1.01414640 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 3 | middle:seed41->42 | 41->42 | 101139 | 101373 | 1.00231365 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 4 | middle:seed41->42 | 41->42 | 198736 | 199848 | 1.00559536 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 1 | large:seed41->42 | 41->42 | 27919 | 24067 | 0.86202944 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | False |
| qa | gemma3-270m | 2 | large:seed41->42 | 41->42 | 53525 | 50038 | 0.93485287 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 3 | large:seed41->42 | 41->42 | 104539 | 102110 | 0.97676465 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-270m | 4 | large:seed41->42 | 41->42 | 198722 | 201946 | 1.01622367 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 1 | small:seed41->42 | 41->42 | 25460 | 24822 | 0.97494108 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 2 | small:seed41->42 | 41->42 | 50563 | 50869 | 1.00605186 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 3 | small:seed41->42 | 41->42 | 100114 | 99576 | 0.99462613 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 4 | small:seed41->42 | 41->42 | 199166 | 199673 | 1.00254562 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 1 | middle:seed41->42 | 41->42 | 24623 | 25396 | 1.03139341 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 2 | middle:seed41->42 | 41->42 | 49836 | 50541 | 1.01414640 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 3 | middle:seed41->42 | 41->42 | 101139 | 101373 | 1.00231365 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 4 | middle:seed41->42 | 41->42 | 198736 | 199848 | 1.00559536 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 1 | large:seed41->42 | 41->42 | 27919 | 24067 | 0.86202944 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | False |
| qa | gemma3-1b | 2 | large:seed41->42 | 41->42 | 53525 | 50038 | 0.93485287 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 3 | large:seed41->42 | 41->42 | 104539 | 102110 | 0.97676465 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-1b | 4 | large:seed41->42 | 41->42 | 198722 | 201946 | 1.01622367 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 1 | small:seed41->42 | 41->42 | 25460 | 24822 | 0.97494108 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 2 | small:seed41->42 | 41->42 | 50563 | 50869 | 1.00605186 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 3 | small:seed41->42 | 41->42 | 100114 | 99576 | 0.99462613 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 4 | small:seed41->42 | 41->42 | 199166 | 199673 | 1.00254562 | 16962 | 17104 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 1 | middle:seed41->42 | 41->42 | 24623 | 25396 | 1.03139341 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 2 | middle:seed41->42 | 41->42 | 49836 | 50541 | 1.01414640 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 3 | middle:seed41->42 | 41->42 | 101139 | 101373 | 1.00231365 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 4 | middle:seed41->42 | 41->42 | 198736 | 199848 | 1.00559536 | 51065 | 51812 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 1 | large:seed41->42 | 41->42 | 27919 | 24067 | 0.86202944 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | False |
| qa | gemma3-4b | 2 | large:seed41->42 | 41->42 | 53525 | 50038 | 0.93485287 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 3 | large:seed41->42 | 41->42 | 104539 | 102110 | 0.97676465 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |
| qa | gemma3-4b | 4 | large:seed41->42 | 41->42 | 198722 | 201946 | 1.01622367 | 156826 | 156653 | NA/NA/NA | NA/NA/NA | True |


Part B: independent baselines

A per-capability constant fitted to the response and then differenced is identically zero; it cannot serve as a second independent baseline.

Per-capability, per-target arithmetic mean of intervention effects whose BOTH endpoints are on the training fold only. It is not forced to zero. For response scoring the mean is the training checkpoint response. This benchmark directly estimates an effect and is not presented as one fitted response law.

Separate same-input second-order surfaces for each descriptor: [t,E,z*t,z*E,t^2,t*E,E^2], t=T/T_ref. Seven OLS coefficients, no intercept; zero at zero budget. Fit only checkpoint responses, then difference the same fitted F.

Training-fold mean-effect estimates (each capability and target fitted separately):

| Split | Held out | Capability | Target | Mean effect | Training pairs | Training trajectories |
|---|---|---|---|---:|---:|---:|
| leave_one_student_out | gemma3-1b | math | I2_size | -0.066554056 | 46 | 12 |
| leave_one_student_out | gemma3-1b | math | response | 0.14021814 | 48 | 12 |
| leave_one_student_out | gemma3-1b | math | I1 | 0.029373927 | 36 | 12 |
| leave_one_student_out | gemma3-1b | math | I2_seed | 0.012933261 | 22 | 12 |
| leave_one_student_out | gemma3-1b | code | I2_size | -0.060252556 | 46 | 12 |
| leave_one_student_out | gemma3-1b | code | response | 0.15038428 | 48 | 12 |
| leave_one_student_out | gemma3-1b | code | I1 | 0.055849264 | 36 | 12 |
| leave_one_student_out | gemma3-1b | code | I2_seed | 0.012776542 | 22 | 12 |
| leave_one_student_out | gemma3-1b | qa | I2_size | -0.99615126 | 46 | 12 |
| leave_one_student_out | gemma3-1b | qa | response | -0.65288715 | 48 | 12 |
| leave_one_student_out | gemma3-1b | qa | I1 | 0.50959011 | 36 | 12 |
| leave_one_student_out | gemma3-1b | qa | I2_seed | -0.0068560983 | 22 | 12 |
| leave_one_student_out | gemma3-270m | math | I2_size | -0.07798283 | 46 | 12 |
| leave_one_student_out | gemma3-270m | math | response | 0.14467983 | 48 | 12 |
| leave_one_student_out | gemma3-270m | math | I1 | 0.045306475 | 36 | 12 |
| leave_one_student_out | gemma3-270m | math | I2_seed | 0.017290843 | 22 | 12 |
| leave_one_student_out | gemma3-270m | code | I2_size | -0.071575822 | 46 | 12 |
| leave_one_student_out | gemma3-270m | code | response | 0.14134236 | 48 | 12 |
| leave_one_student_out | gemma3-270m | code | I1 | 0.060087282 | 36 | 12 |
| leave_one_student_out | gemma3-270m | code | I2_seed | 0.026423726 | 22 | 12 |
| leave_one_student_out | gemma3-270m | qa | I2_size | -1.2643042 | 46 | 12 |
| leave_one_student_out | gemma3-270m | qa | response | -0.61526161 | 48 | 12 |
| leave_one_student_out | gemma3-270m | qa | I1 | 0.69792358 | 36 | 12 |
| leave_one_student_out | gemma3-270m | qa | I2_seed | 0.058261273 | 22 | 12 |
| leave_one_student_out | gemma3-4b | math | I2_size | -0.036554452 | 46 | 12 |
| leave_one_student_out | gemma3-4b | math | response | 0.099126446 | 48 | 12 |
| leave_one_student_out | gemma3-4b | math | I1 | 0.024603896 | 36 | 12 |
| leave_one_student_out | gemma3-4b | math | I2_seed | 0.0025322363 | 22 | 12 |
| leave_one_student_out | gemma3-4b | code | I2_size | -0.064089758 | 46 | 12 |
| leave_one_student_out | gemma3-4b | code | response | 0.13039171 | 48 | 12 |
| leave_one_student_out | gemma3-4b | code | I1 | 0.055560169 | 36 | 12 |
| leave_one_student_out | gemma3-4b | code | I2_seed | 0.020194611 | 22 | 12 |
| leave_one_student_out | gemma3-4b | qa | I2_size | -0.64414624 | 46 | 12 |
| leave_one_student_out | gemma3-4b | qa | response | -0.84737612 | 48 | 12 |
| leave_one_student_out | gemma3-4b | qa | I1 | 0.36420983 | 36 | 12 |
| leave_one_student_out | gemma3-4b | qa | I2_seed | 0.055830655 | 22 | 12 |
| leave_one_pool_seed_out | 41 | math | I2_size | -0.065562333 | 36 | 9 |
| leave_one_pool_seed_out | 41 | math | response | 0.13270082 | 36 | 9 |
| leave_one_pool_seed_out | 41 | math | I1 | 0.035190517 | 27 | 9 |
| leave_one_pool_seed_out | 41 | code | I2_size | -0.085686079 | 36 | 9 |
| leave_one_pool_seed_out | 41 | code | response | 0.14958235 | 36 | 9 |
| leave_one_pool_seed_out | 41 | code | I1 | 0.062771668 | 27 | 9 |
| leave_one_pool_seed_out | 41 | qa | I2_size | -1.0458894 | 36 | 9 |
| leave_one_pool_seed_out | 41 | qa | response | -0.69216191 | 36 | 9 |
| leave_one_pool_seed_out | 41 | qa | I1 | 0.53998358 | 27 | 9 |
| leave_one_pool_seed_out | 42 | math | I2_size | -0.05469263 | 33 | 9 |
| leave_one_pool_seed_out | 42 | math | response | 0.12331545 | 36 | 9 |
| leave_one_pool_seed_out | 42 | math | I1 | 0.030999015 | 27 | 9 |
| leave_one_pool_seed_out | 42 | code | I2_size | -0.043073282 | 33 | 9 |
| leave_one_pool_seed_out | 42 | code | response | 0.13182988 | 36 | 9 |
| leave_one_pool_seed_out | 42 | code | I1 | 0.051559476 | 27 | 9 |
| leave_one_pool_seed_out | 42 | qa | I2_size | -0.88344916 | 33 | 9 |
| leave_one_pool_seed_out | 42 | qa | response | -0.71818801 | 36 | 9 |
| leave_one_pool_seed_out | 42 | qa | I1 | 0.5078321 | 27 | 9 |

Part C: equivalence check before fits

Joint basis is [x,z*x,y,z*y], x=log(1+T/T_ref), y=log(1+T/D_U). V95 conditioned basis was [1,x,log(D_U/D_ref),z*x]. At T=0 the joint is zero for every D_U; the V95 family is a+q*log(D_U/D_ref), forcing a=q=0 to obey that constraint. This leaves only [x,z*x], which cannot span the joint y and z*y terms. log(1+T/D_U)=log(T+D_U)-log(D_U), not log(T)-log(D_U). The one-term T/reuse forms are proper submodels, not equivalent four-parameter families.

All 14 comparisons (two descriptors x seven V95 candidates) have unequal function-space witness ranks; no earlier candidate is a renamed joint form.

F = (a+a_prime*z)*log(1+T/T_ref)+(b+b_prime*z)*log(1+T/D_U); T_ref=100000 fixed; four coefficients per capability in order a,a_prime,b,b_prime. No intercept, fitted scale, floor, threshold, or extra time constant.

delta = saved capability loss minus the same trajectory's own update-0 loss; negative means loss reduction.

Training-fold checkpoint mean/population std, separately per capability; own initial capability loss is a permitted held-out input, never post-training loss.

One unweighted per-capability OLS fit on all training checkpoints including zero. F predicts response; F(T_high)-F(T_low) predicts registered about-doubling; F(T2,D_U2,z)-F(T1,D_U1,z) predicts observed pool contrast. Exact doubling and exact-T counterfactual methods use the same coefficients, without intervention fits.

The same candidate must have zero-baseline MAE improvement strictly greater than the FULL paired 95% improvement-CI width on BOTH I1 and I2_size under pooled leave-one-student-out. Equality fails; seed holdout cannot rescue failure. Response, seed-only and additional-baseline comparisons are separate diagnostics.

Joint candidate decisions:

| Capability | Descriptor | Registered I1 + I2_size decision | Also clears both vs mean | Also clears both vs surface |
|---|---|---|---|---|
| math | log_parameters | did not meet the pre-registered threshold | False | False |
| math | initial_loss | did not meet the pre-registered threshold | False | False |
| code | log_parameters | did not meet the pre-registered threshold | False | False |
| code | initial_loss | did not meet the pre-registered threshold | False | False |
| qa | log_parameters | did not meet the pre-registered threshold | False | False |
| qa | initial_loss | did not meet the pre-registered threshold | False | False |

Evaluation: 5000 paired percentile bootstrap draws clustered on whole trajectories. Cluster counts accompany every interval. Refit-in-bootstrap=false, matching V95; intervals condition on cross-fitted predictions. All fold metrics, coefficients, endpoint predictions and training IDs are in summary.json. Response scoring excludes the trivial zero-budget anchors; fitting includes them.

Each seed41->42 contrast crosses the two leave-one-pool-seed-out folds. Neither fold contains both held-out endpoints, and neither training fold contains a seed contrast. Scoring it with an in-training endpoint would leak; report unavailable, with zero eligible secondary seed-only pairs.

leave_one_student_out, descriptor=log_parameters; pooled held-out predictions; targets remain separate

| Capability | Target | Candidate | Pairs | MAE [95% CI] | Bias [95% CI] | Gain vs zero [95% CI] | Gain vs mean [95% CI] | Gain vs surface [95% CI] |
|---|---|---|---:|---|---|---|---|---|
| code | I1 | zero | 54 | 0.0590416 [0.0440249, 0.0762806] (clusters=18) | -0.0571656 [-0.0751231, -0.0410726] (clusters=18) | 0 [0, 0] (clusters=18) | -0.0228733 [-0.0350512, -0.0104853] (clusters=18) | — |
| code | I1 | T_only | 54 | 0.0431818 [0.0351822, 0.0516429] (clusters=18) | 0.00722578 [-0.010802, 0.0234442] (clusters=18) | 0.0158598 [4.08711e-05, 0.0322022] (clusters=18) | -0.00701355 [-0.0125079, -0.00105652] (clusters=18) | — |
| code | I1 | joint_log_parameters | 54 | 0.035956 [0.0286866, 0.0441598] (clusters=18) | -0.0122744 [-0.0254255, -7.67222e-05] (clusters=18) | 0.0230856 [0.00980935, 0.0368516] (clusters=18) | 0.000212269 [-0.00434604, 0.00490504] (clusters=18) | -0.000245997 [-0.00846102, 0.00745636] (clusters=18) |
| code | I1 | surface_log_parameters | 54 | 0.03571 [0.031134, 0.0405069] (clusters=18) | -0.0142697 [-0.0250517, -0.00437118] (clusters=18) | 0.0233316 [0.00729792, 0.0409701] (clusters=18) | 0.000458266 [-0.0077109, 0.00940453] (clusters=18) | — |
| code | I1 | mean_effect | 54 | 0.0361683 [0.0280913, 0.0462324] (clusters=18) | 2.56996e-18 [-0.0182759, 0.0162295] (clusters=18) | 0.0228733 [0.0104853, 0.0350512] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| code | I2_seed | zero | 33 | 0.0321601 [0.0105256, 0.0739708] (clusters=18) | -0.0197983 [-0.0700987, 0.0121512] (clusters=18) | 0 [0, 0] (clusters=18) | 0.00216677 [-0.0121907, 0.0186326] (clusters=18) | — |
| code | I2_seed | T_only | 33 | 0.0318226 [0.00994955, 0.074019] (clusters=18) | -0.0200502 [-0.0703416, 0.0104845] (clusters=18) | 0.00033751 [-0.000245395, 0.0013924] (clusters=18) | 0.00250428 [-0.0123084, 0.0195508] (clusters=18) | — |
| code | I2_seed | joint_log_parameters | 33 | 0.0323533 [0.0101215, 0.07478] (clusters=18) | -0.0204396 [-0.0709692, 0.0116302] (clusters=18) | -0.000193269 [-0.000908945, 0.000650636] (clusters=18) | 0.0019735 [-0.0130527, 0.018781] (clusters=18) | -0.000356601 [-0.0018289, 0.000544569] (clusters=18) |
| code | I2_seed | surface_log_parameters | 33 | 0.0319967 [0.00935209, 0.0750508] (clusters=18) | -0.0211517 [-0.0714956, 0.00991002] (clusters=18) | 0.000163332 [-0.00136689, 0.00222381] (clusters=18) | 0.0023301 [-0.0135248, 0.020389] (clusters=18) | — |
| code | I2_seed | mean_effect | 33 | 0.0343268 [0.0191161, 0.0686647] (clusters=18) | -2.52323e-18 [-0.0539912, 0.0317772] (clusters=18) | -0.00216677 [-0.0186326, 0.0121907] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| code | I2_size | zero | 69 | 0.0713177 [0.0412525, 0.107236] (clusters=18) | 0.065306 [0.0358652, 0.103237] (clusters=18) | 0 [0, 0] (clusters=18) | -0.0041739 [-0.0164009, 0.0117413] (clusters=18) | — |
| code | I2_size | T_only | 69 | 0.0718014 [0.0418805, 0.10797] (clusters=18) | 0.0663626 [0.0368843, 0.104087] (clusters=18) | -0.000483693 [-0.00115798, 0.000196609] (clusters=18) | -0.00465759 [-0.0170094, 0.0116081] (clusters=18) | — |
| code | I2_size | joint_log_parameters | 69 | 0.0569946 [0.0382302, 0.0797166] (clusters=18) | -0.0262028 [-0.0548922, 0.0162746] (clusters=18) | 0.014323 [-0.00559995, 0.0327352] (clusters=18) | 0.0101491 [-0.00500403, 0.0206802] (clusters=18) | -0.0103433 [-0.0222099, 0.00164086] (clusters=18) |
| code | I2_size | surface_log_parameters | 69 | 0.0466514 [0.0284113, 0.0669353] (clusters=18) | -0.0252871 [-0.057189, 0.0211853] (clusters=18) | 0.0246663 [0.000207263, 0.0522523] (clusters=18) | 0.0204924 [-0.00161509, 0.0412268] (clusters=18) | — |
| code | I2_size | mean_effect | 69 | 0.0671438 [0.044743, 0.0953459] (clusters=18) | 1.60902e-17 [-0.0298831, 0.0406183] (clusters=18) | 0.0041739 [-0.0117413, 0.0164009] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| code | response | zero | 72 | 0.140706 [0.119756, 0.164009] (clusters=18) | -0.140706 [-0.164009, -0.119756] (clusters=18) | 0 [0, 0] (clusters=18) | -0.0740354 [-0.0867891, -0.0594286] (clusters=18) | — |
| code | response | T_only | 72 | 0.0491246 [0.0372515, 0.0626421] (clusters=18) | -0.005728 [-0.0302924, 0.0168267] (clusters=18) | 0.0915815 [0.0722908, 0.108412] (clusters=18) | 0.0175462 [0.00607174, 0.0285351] (clusters=18) | — |
| code | response | joint_log_parameters | 72 | 0.0569472 [0.0488418, 0.0659231] (clusters=18) | -0.0242701 [-0.0453118, -0.00219879] (clusters=18) | 0.0837589 [0.062429, 0.106602] (clusters=18) | 0.00972354 [-0.0107327, 0.0293004] (clusters=18) | -0.00680813 [-0.012578, -0.00107241] (clusters=18) |
| code | response | surface_log_parameters | 72 | 0.0501391 [0.0419186, 0.0585426] (clusters=18) | -0.0209332 [-0.0416853, 0.00109601] (clusters=18) | 0.090567 [0.0667734, 0.117452] (clusters=18) | 0.0165317 [-0.00573683, 0.0396095] (clusters=18) | — |
| code | response | mean_effect | 72 | 0.0666707 [0.0501123, 0.0851932] (clusters=18) | -1.54198e-18 [-0.0245835, 0.0227773] (clusters=18) | 0.0740354 [0.0594286, 0.0867891] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| math | I1 | zero | 54 | 0.0448632 [0.0260747, 0.0670939] (clusters=18) | -0.0330948 [-0.0577697, -0.0124391] (clusters=18) | 0 [0, 0] (clusters=18) | 0.00162981 [-0.00826732, 0.0123644] (clusters=18) | — |
| math | I1 | T_only | 54 | 0.0546497 [0.0427067, 0.0684622] (clusters=18) | 0.0222117 [-0.00397832, 0.0444031] (clusters=18) | -0.00978649 [-0.0261641, 0.00665095] (clusters=18) | -0.00815668 [-0.0164189, 0.000701382] (clusters=18) | — |
| math | I1 | joint_log_parameters | 54 | 0.0396584 [0.0271786, 0.0548263] (clusters=18) | 0.00357919 [-0.0154342, 0.0199799] (clusters=18) | 0.0052048 [-0.00450334, 0.0156797] (clusters=18) | 0.00683461 [-0.00223275, 0.0161209] (clusters=18) | 0.00512064 [-0.00497519, 0.014936] (clusters=18) |
| math | I1 | surface_log_parameters | 54 | 0.044779 [0.0373816, 0.0531242] (clusters=18) | -0.00231519 [-0.0144856, 0.00907381] (clusters=18) | 8.41568e-05 [-0.0163542, 0.0180399] (clusters=18) | 0.00171397 [-0.00994893, 0.0146884] (clusters=18) | — |
| math | I1 | mean_effect | 54 | 0.046493 [0.0313206, 0.0652531] (clusters=18) | 2.05597e-18 [-0.0264243, 0.0225421] (clusters=18) | -0.00162981 [-0.0123644, 0.00826732] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| math | I2_seed | zero | 33 | 0.0179464 [0.00702373, 0.0389625] (clusters=18) | -0.0109188 [-0.0364653, 0.00461102] (clusters=18) | 0 [0, 0] (clusters=18) | 0.00300284 [-0.00521231, 0.0114373] (clusters=18) | — |
| math | I2_seed | T_only | 33 | 0.0174143 [0.00600333, 0.0388051] (clusters=18) | -0.0111351 [-0.0366576, 0.00387304] (clusters=18) | 0.000532056 [-0.000136502, 0.0019224] (clusters=18) | 0.0035349 [-0.00522059, 0.0127307] (clusters=18) | — |
| math | I2_seed | joint_log_parameters | 33 | 0.017856 [0.00633241, 0.0394991] (clusters=18) | -0.0112904 [-0.0371816, 0.00428588] (clusters=18) | 9.04038e-05 [-0.000733716, 0.00110278] (clusters=18) | 0.00309325 [-0.00589108, 0.0120239] (clusters=18) | 0.000365014 [-0.000910984, 0.000985087] (clusters=18) |
| math | I2_seed | surface_log_parameters | 33 | 0.018221 [0.00680658, 0.0400895] (clusters=18) | -0.0120283 [-0.0380539, 0.00343613] (clusters=18) | -0.00027461 [-0.00136302, 0.00128681] (clusters=18) | 0.00272823 [-0.00650401, 0.0125147] (clusters=18) | — |
| math | I2_seed | mean_effect | 33 | 0.0209492 [0.0120359, 0.0389308] (clusters=18) | 0 [-0.0310536, 0.020514] (clusters=18) | -0.00300284 [-0.0114373, 0.00521231] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| math | I2_size | zero | 69 | 0.0687308 [0.030717, 0.11597] (clusters=18) | 0.0603638 [0.0250761, 0.104693] (clusters=18) | 0 [0, 0] (clusters=18) | 0.0105924 [-0.00549658, 0.0309872] (clusters=18) | — |
| math | I2_size | T_only | 69 | 0.0691815 [0.0312778, 0.116242] (clusters=18) | 0.0612713 [0.0261349, 0.10535] (clusters=18) | -0.000450698 [-0.00104571, 0.00015864] (clusters=18) | 0.0101417 [-0.00575349, 0.0305454] (clusters=18) | — |
| math | I2_size | joint_log_parameters | 69 | 0.0725838 [0.0463552, 0.0998743] (clusters=18) | 0.01578 [-0.0311648, 0.0561299] (clusters=18) | -0.00385305 [-0.0258028, 0.0188139] (clusters=18) | 0.0067393 [-0.0101834, 0.0231] (clusters=18) | -0.0205935 [-0.0391383, 0.000897239] (clusters=18) |
| math | I2_size | surface_log_parameters | 69 | 0.0519904 [0.0298179, 0.0723739] (clusters=18) | 0.0214186 [-0.0184725, 0.059636] (clusters=18) | 0.0167404 [-0.0242457, 0.0563583] (clusters=18) | 0.0273328 [-0.00706383, 0.0552963] (clusters=18) | — |
| math | I2_size | mean_effect | 69 | 0.0793231 [0.0514828, 0.114074] (clusters=18) | 8.04509e-19 [-0.0461183, 0.0590368] (clusters=18) | -0.0105924 [-0.0309872, 0.00549658] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| math | response | zero | 72 | 0.128008 [0.103076, 0.159011] (clusters=18) | -0.128008 [-0.159011, -0.103076] (clusters=18) | 0 [0, 0] (clusters=18) | -0.0578041 [-0.0734746, -0.0421643] (clusters=18) | — |
| math | response | T_only | 72 | 0.0688941 [0.0503875, 0.0929077] (clusters=18) | -0.0120738 [-0.0496877, 0.0194507] (clusters=18) | 0.059114 [0.0421703, 0.0757663] (clusters=18) | 0.00130989 [-0.0091751, 0.013067] (clusters=18) | — |
| math | response | joint_log_parameters | 72 | 0.0669673 [0.0522363, 0.0829223] (clusters=18) | -0.0401913 [-0.0673177, -0.0122015] (clusters=18) | 0.0610409 [0.0373249, 0.0867955] (clusters=18) | 0.00323672 [-0.0178921, 0.0231008] (clusters=18) | -0.00939156 [-0.013438, -0.00464834] (clusters=18) |
| math | response | surface_log_parameters | 72 | 0.0575757 [0.0439844, 0.0728485] (clusters=18) | -0.034124 [-0.0590257, -0.00919302] (clusters=18) | 0.0704324 [0.0474927, 0.0958772] (clusters=18) | 0.0126283 [-0.00870505, 0.0327725] (clusters=18) | — |
| math | response | mean_effect | 72 | 0.070204 [0.052727, 0.0922336] (clusters=18) | -1.07938e-17 [-0.0379515, 0.0318456] (clusters=18) | 0.0578041 [0.0421643, 0.0734746] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| qa | I1 | zero | 54 | 0.594364 [0.330725, 0.91514] (clusters=18) | -0.523908 [-0.867916, -0.240006] (clusters=18) | 0 [0, 0] (clusters=18) | 0.0498533 [-0.0864976, 0.197534] (clusters=18) | — |
| qa | I1 | T_only | 54 | 0.682625 [0.396015, 1.03375] (clusters=18) | -0.659395 [-1.0166, -0.364339] (clusters=18) | -0.0882605 [-0.121333, -0.0558594] (clusters=18) | -0.0384072 [-0.195486, 0.131667] (clusters=18) | — |
| qa | I1 | joint_log_parameters | 54 | 0.793214 [0.570234, 1.0636] (clusters=18) | -0.785011 [-1.05734, -0.559861] (clusters=18) | -0.198849 [-0.274771, -0.12526] (clusters=18) | -0.148996 [-0.271918, -0.0236189] (clusters=18) | -0.152613 [-0.3645, 0.0275431] (clusters=18) |
| qa | I1 | surface_log_parameters | 54 | 0.640601 [0.525528, 0.766354] (clusters=18) | -0.272267 [-0.38478, -0.185256] (clusters=18) | -0.0462367 [-0.277695, 0.220952] (clusters=18) | 0.00361662 [-0.179039, 0.204657] (clusters=18) | — |
| qa | I1 | mean_effect | 54 | 0.644218 [0.452732, 0.87775] (clusters=18) | 6.5791e-17 [-0.375255, 0.314311] (clusters=18) | -0.0498533 [-0.197534, 0.0864976] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| qa | I2_seed | zero | 33 | 0.182307 [0.0564873, 0.339532] (clusters=18) | -0.0357453 [-0.259613, 0.158766] (clusters=18) | 0 [0, 0] (clusters=18) | 0.0123943 [-0.0275612, 0.0479796] (clusters=18) | — |
| qa | I2_seed | T_only | 33 | 0.182862 [0.0568769, 0.339309] (clusters=18) | -0.0352153 [-0.259118, 0.160325] (clusters=18) | -0.000555392 [-0.00214356, 0.000539433] (clusters=18) | 0.0118389 [-0.0277832, 0.0480587] (clusters=18) | — |
| qa | I2_seed | joint_log_parameters | 33 | 0.18575 [0.0583491, 0.34231] (clusters=18) | -0.0378165 [-0.266467, 0.162453] (clusters=18) | -0.00344285 [-0.0103045, 0.00292111] (clusters=18) | 0.00895144 [-0.0280148, 0.0467537] (clusters=18) | 0.011825 [-0.000271212, 0.0313031] (clusters=18) |
| qa | I2_seed | surface_log_parameters | 33 | 0.197575 [0.0659115, 0.346825] (clusters=18) | -0.0305373 [-0.273453, 0.179617] (clusters=18) | -0.0152678 [-0.0375188, -0.00137306] (clusters=18) | -0.00287352 [-0.038344, 0.0348991] (clusters=18) | — |
| qa | I2_seed | mean_effect | 33 | 0.194701 [0.0648072, 0.347939] (clusters=18) | 1.34572e-17 [-0.229478, 0.210262] (clusters=18) | -0.0123943 [-0.0479796, 0.0275612] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| qa | I2_size | zero | 69 | 1.00354 [0.395669, 1.67609] (clusters=18) | 0.968201 [0.359174, 1.63662] (clusters=18) | 0 [0, 0] (clusters=18) | 0.17362 [-0.103937, 0.548031] (clusters=18) | — |
| qa | I2_size | T_only | 69 | 1.00225 [0.394751, 1.67397] (clusters=18) | 0.965977 [0.357686, 1.63537] (clusters=18) | 0.00129404 [0.000314846, 0.00231886] (clusters=18) | 0.174914 [-0.102599, 0.549183] (clusters=18) | — |
| qa | I2_size | joint_log_parameters | 69 | 0.911681 [0.550401, 1.35088] (clusters=18) | 0.351396 [-0.107753, 0.75671] (clusters=18) | 0.0918633 [-0.202758, 0.37506] (clusters=18) | 0.265484 [0.0766635, 0.51245] (clusters=18) | -0.439486 [-0.725497, -0.152056] (clusters=18) |
| qa | I2_size | surface_log_parameters | 69 | 0.472196 [0.244315, 0.703715] (clusters=18) | 0.115366 [-0.225439, 0.44461] (clusters=18) | 0.531349 [-0.0299415, 1.08815] (clusters=18) | 0.70497 [0.357323, 1.02602] (clusters=18) | — |
| qa | I2_size | mean_effect | 69 | 1.17717 [0.819321, 1.64362] (clusters=18) | -1.28722e-16 [-0.762009, 0.86979] (clusters=18) | -0.17362 [-0.548031, 0.103937] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| qa | response | zero | 72 | 1.24699 [1.07051, 1.45651] (clusters=18) | 0.705175 [0.33567, 1.01329] (clusters=18) | 0 [0, 0] (clusters=18) | -0.461771 [-0.587958, -0.325908] (clusters=18) | — |
| qa | response | T_only | 72 | 1.12378 [0.904636, 1.39353] (clusters=18) | 0.421166 [0.0225362, 0.749508] (clusters=18) | 0.123211 [0.0303052, 0.214771] (clusters=18) | -0.338559 [-0.394213, -0.280658] (clusters=18) | — |
| qa | response | joint_log_parameters | 72 | 1.00919 [0.769196, 1.29197] (clusters=18) | 0.306151 [0.0839886, 0.519656] (clusters=18) | 0.237804 [0.0742365, 0.398428] (clusters=18) | -0.223966 [-0.315024, -0.135995] (clusters=18) | -0.451575 [-0.625993, -0.294451] (clusters=18) |
| qa | response | surface_log_parameters | 72 | 0.557613 [0.446529, 0.692439] (clusters=18) | 0.0178593 [-0.153562, 0.180207] (clusters=18) | 0.689379 [0.531956, 0.830621] (clusters=18) | 0.227608 [0.0499926, 0.425707] (clusters=18) | — |
| qa | response | mean_effect | 72 | 0.785221 [0.541146, 1.0835] (clusters=18) | 8.63507e-17 [-0.389589, 0.321431] (clusters=18) | 0.461771 [0.325908, 0.587958] (clusters=18) | 0 [0, 0] (clusters=18) | — |

leave_one_student_out, descriptor=initial_loss; pooled held-out predictions; targets remain separate

| Capability | Target | Candidate | Pairs | MAE [95% CI] | Bias [95% CI] | Gain vs zero [95% CI] | Gain vs mean [95% CI] | Gain vs surface [95% CI] |
|---|---|---|---:|---|---|---|---|---|
| code | I1 | zero | 54 | 0.0590416 [0.0440249, 0.0762806] (clusters=18) | -0.0571656 [-0.0751231, -0.0410726] (clusters=18) | 0 [0, 0] (clusters=18) | -0.0228733 [-0.0350512, -0.0104853] (clusters=18) | — |
| code | I1 | T_only | 54 | 0.0431818 [0.0351822, 0.0516429] (clusters=18) | 0.00722578 [-0.010802, 0.0234442] (clusters=18) | 0.0158598 [4.08711e-05, 0.0322022] (clusters=18) | -0.00701355 [-0.0125079, -0.00105652] (clusters=18) | — |
| code | I1 | joint_initial_loss | 54 | 0.0354732 [0.0278959, 0.0441634] (clusters=18) | -0.0118026 [-0.0246394, 7.20831e-05] (clusters=18) | 0.0235684 [0.0108147, 0.0364731] (clusters=18) | 0.000695053 [-0.0036364, 0.00523316] (clusters=18) | 0.00050922 [-0.00738208, 0.00792395] (clusters=18) |
| code | I1 | surface_initial_loss | 54 | 0.0359824 [0.0306735, 0.0416295] (clusters=18) | -0.0127683 [-0.0247043, -0.0016142] (clusters=18) | 0.0230592 [0.00806434, 0.0396614] (clusters=18) | 0.000185833 [-0.00721246, 0.00812633] (clusters=18) | — |
| code | I1 | mean_effect | 54 | 0.0361683 [0.0280913, 0.0462324] (clusters=18) | 2.56996e-18 [-0.0182759, 0.0162295] (clusters=18) | 0.0228733 [0.0104853, 0.0350512] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| code | I2_seed | zero | 33 | 0.0321601 [0.0105256, 0.0739708] (clusters=18) | -0.0197983 [-0.0700987, 0.0121512] (clusters=18) | 0 [0, 0] (clusters=18) | 0.00216677 [-0.0121907, 0.0186326] (clusters=18) | — |
| code | I2_seed | T_only | 33 | 0.0318226 [0.00994955, 0.074019] (clusters=18) | -0.0200502 [-0.0703416, 0.0104845] (clusters=18) | 0.00033751 [-0.000245395, 0.0013924] (clusters=18) | 0.00250428 [-0.0123084, 0.0195508] (clusters=18) | — |
| code | I2_seed | joint_initial_loss | 33 | 0.032376 [0.0100728, 0.0748752] (clusters=18) | -0.020477 [-0.0710605, 0.012066] (clusters=18) | -0.00021592 [-0.00101992, 0.000697456] (clusters=18) | 0.00195085 [-0.0131721, 0.0189321] (clusters=18) | -0.000331532 [-0.00189455, 0.000663307] (clusters=18) |
| code | I2_seed | surface_initial_loss | 33 | 0.0320445 [0.00928348, 0.0750621] (clusters=18) | -0.0212153 [-0.0718392, 0.010247] (clusters=18) | 0.000115613 [-0.00161997, 0.00236075] (clusters=18) | 0.00228238 [-0.0137566, 0.0205471] (clusters=18) | — |
| code | I2_seed | mean_effect | 33 | 0.0343268 [0.0191161, 0.0686647] (clusters=18) | -2.52323e-18 [-0.0539912, 0.0317772] (clusters=18) | -0.00216677 [-0.0186326, 0.0121907] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| code | I2_size | zero | 69 | 0.0713177 [0.0412525, 0.107236] (clusters=18) | 0.065306 [0.0358652, 0.103237] (clusters=18) | 0 [0, 0] (clusters=18) | -0.0041739 [-0.0164009, 0.0117413] (clusters=18) | — |
| code | I2_size | T_only | 69 | 0.0718014 [0.0418805, 0.10797] (clusters=18) | 0.0663626 [0.0368843, 0.104087] (clusters=18) | -0.000483693 [-0.00115798, 0.000196609] (clusters=18) | -0.00465759 [-0.0170094, 0.0116081] (clusters=18) | — |
| code | I2_size | joint_initial_loss | 69 | 0.0612014 [0.0390382, 0.089524] (clusters=18) | -0.033226 [-0.0721847, 0.0170296] (clusters=18) | 0.0101163 [-0.012339, 0.0297526] (clusters=18) | 0.0059424 [-0.0121609, 0.0175685] (clusters=18) | -0.00476863 [-0.0185784, 0.0133196] (clusters=18) |
| code | I2_size | surface_initial_loss | 69 | 0.0564328 [0.0285318, 0.0926774] (clusters=18) | -0.0342384 [-0.0834259, 0.0219083] (clusters=18) | 0.0148849 [-0.0177362, 0.0413898] (clusters=18) | 0.010711 [-0.019395, 0.031789] (clusters=18) | — |
| code | I2_size | mean_effect | 69 | 0.0671438 [0.044743, 0.0953459] (clusters=18) | 1.60902e-17 [-0.0298831, 0.0406183] (clusters=18) | 0.0041739 [-0.0117413, 0.0164009] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| code | response | zero | 72 | 0.140706 [0.119756, 0.164009] (clusters=18) | -0.140706 [-0.164009, -0.119756] (clusters=18) | 0 [0, 0] (clusters=18) | -0.0740354 [-0.0867891, -0.0594286] (clusters=18) | — |
| code | response | T_only | 72 | 0.0491246 [0.0372515, 0.0626421] (clusters=18) | -0.005728 [-0.0302924, 0.0168267] (clusters=18) | 0.0915815 [0.0722908, 0.108412] (clusters=18) | 0.0175462 [0.00607174, 0.0285351] (clusters=18) | — |
| code | response | joint_initial_loss | 72 | 0.0541443 [0.0428628, 0.066533] (clusters=18) | -0.0215789 [-0.0433762, 0.000510304] (clusters=18) | 0.0865618 [0.0635939, 0.110226] (clusters=18) | 0.0125264 [-0.0110667, 0.0330995] (clusters=18) | -0.00443802 [-0.0128863, 0.00289777] (clusters=18) |
| code | response | surface_initial_loss | 72 | 0.0497063 [0.038016, 0.0625574] (clusters=18) | -0.0185255 [-0.0413789, 0.00474329] (clusters=18) | 0.0909998 [0.0661154, 0.119024] (clusters=18) | 0.0169644 [-0.008649, 0.0417715] (clusters=18) | — |
| code | response | mean_effect | 72 | 0.0666707 [0.0501123, 0.0851932] (clusters=18) | -1.54198e-18 [-0.0245835, 0.0227773] (clusters=18) | 0.0740354 [0.0594286, 0.0867891] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| math | I1 | zero | 54 | 0.0448632 [0.0260747, 0.0670939] (clusters=18) | -0.0330948 [-0.0577697, -0.0124391] (clusters=18) | 0 [0, 0] (clusters=18) | 0.00162981 [-0.00826732, 0.0123644] (clusters=18) | — |
| math | I1 | T_only | 54 | 0.0546497 [0.0427067, 0.0684622] (clusters=18) | 0.0222117 [-0.00397832, 0.0444031] (clusters=18) | -0.00978649 [-0.0261641, 0.00665095] (clusters=18) | -0.00815668 [-0.0164189, 0.000701382] (clusters=18) | — |
| math | I1 | joint_initial_loss | 54 | 0.0448691 [0.0335484, 0.0575268] (clusters=18) | 0.0201538 [0.00358066, 0.035809] (clusters=18) | -5.96556e-06 [-0.0137509, 0.0140457] (clusters=18) | 0.00162385 [-0.00817544, 0.0114492] (clusters=18) | 5.09337e-05 [-0.00910843, 0.00848086] (clusters=18) |
| math | I1 | surface_initial_loss | 54 | 0.0449201 [0.0381416, 0.0533951] (clusters=18) | 0.023523 [0.00964265, 0.0382769] (clusters=18) | -5.68992e-05 [-0.0155213, 0.0174707] (clusters=18) | 0.00157291 [-0.00976885, 0.0148628] (clusters=18) | — |
| math | I1 | mean_effect | 54 | 0.046493 [0.0313206, 0.0652531] (clusters=18) | 2.05597e-18 [-0.0264243, 0.0225421] (clusters=18) | -0.00162981 [-0.0123644, 0.00826732] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| math | I2_seed | zero | 33 | 0.0179464 [0.00702373, 0.0389625] (clusters=18) | -0.0109188 [-0.0364653, 0.00461102] (clusters=18) | 0 [0, 0] (clusters=18) | 0.00300284 [-0.00521231, 0.0114373] (clusters=18) | — |
| math | I2_seed | T_only | 33 | 0.0174143 [0.00600333, 0.0388051] (clusters=18) | -0.0111351 [-0.0366576, 0.00387304] (clusters=18) | 0.000532056 [-0.000136502, 0.0019224] (clusters=18) | 0.0035349 [-0.00522059, 0.0127307] (clusters=18) | — |
| math | I2_seed | joint_initial_loss | 33 | 0.0179884 [0.00655929, 0.0400573] (clusters=18) | -0.0115728 [-0.0378536, 0.00401673] (clusters=18) | -4.19938e-05 [-0.0011339, 0.00108202] (clusters=18) | 0.00296085 [-0.00593068, 0.0120626] (clusters=18) | 0.000508214 [-0.000810893, 0.00140857] (clusters=18) |
| math | I2_seed | surface_initial_loss | 33 | 0.0184966 [0.00680838, 0.0413408] (clusters=18) | -0.0124355 [-0.0392646, 0.00293745] (clusters=18) | -0.000550207 [-0.00237601, 0.00133675] (clusters=18) | 0.00245264 [-0.0066493, 0.0125763] (clusters=18) | — |
| math | I2_seed | mean_effect | 33 | 0.0209492 [0.0120359, 0.0389308] (clusters=18) | 0 [-0.0310536, 0.020514] (clusters=18) | -0.00300284 [-0.0114373, 0.00521231] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| math | I2_size | zero | 69 | 0.0687308 [0.030717, 0.11597] (clusters=18) | 0.0603638 [0.0250761, 0.104693] (clusters=18) | 0 [0, 0] (clusters=18) | 0.0105924 [-0.00549658, 0.0309872] (clusters=18) | — |
| math | I2_size | T_only | 69 | 0.0691815 [0.0312778, 0.116242] (clusters=18) | 0.0612713 [0.0261349, 0.10535] (clusters=18) | -0.000450698 [-0.00104571, 0.00015864] (clusters=18) | 0.0101417 [-0.00575349, 0.0305454] (clusters=18) | — |
| math | I2_size | joint_initial_loss | 69 | 0.0619762 [0.0327788, 0.0990319] (clusters=18) | -0.027051 [-0.0565748, -0.00348275] (clusters=18) | 0.00675462 [-0.00967486, 0.0213598] (clusters=18) | 0.017347 [-0.000956363, 0.0308596] (clusters=18) | -0.0117611 [-0.0280412, 0.00904265] (clusters=18) |
| math | I2_size | surface_initial_loss | 69 | 0.050215 [0.0194356, 0.100309] (clusters=18) | -0.0348122 [-0.095004, 0.00558154] (clusters=18) | 0.0185157 [-0.00473226, 0.0417973] (clusters=18) | 0.0291081 [0.00041656, 0.0498305] (clusters=18) | — |
| math | I2_size | mean_effect | 69 | 0.0793231 [0.0514828, 0.114074] (clusters=18) | 8.04509e-19 [-0.0461183, 0.0590368] (clusters=18) | -0.0105924 [-0.0309872, 0.00549658] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| math | response | zero | 72 | 0.128008 [0.103076, 0.159011] (clusters=18) | -0.128008 [-0.159011, -0.103076] (clusters=18) | 0 [0, 0] (clusters=18) | -0.0578041 [-0.0734746, -0.0421643] (clusters=18) | — |
| math | response | T_only | 72 | 0.0688941 [0.0503875, 0.0929077] (clusters=18) | -0.0120738 [-0.0496877, 0.0194507] (clusters=18) | 0.059114 [0.0421703, 0.0757663] (clusters=18) | 0.00130989 [-0.0091751, 0.013067] (clusters=18) | — |
| math | response | joint_initial_loss | 72 | 0.044911 [0.033906, 0.0568031] (clusters=18) | 0.00498869 [-0.00793278, 0.0194994] (clusters=18) | 0.0830971 [0.0646398, 0.106139] (clusters=18) | 0.025293 [0.00716845, 0.0437928] (clusters=18) | -0.00468879 [-0.0132262, 0.00502786] (clusters=18) |
| math | response | surface_initial_loss | 72 | 0.0402223 [0.0276075, 0.0564241] (clusters=18) | 0.00764577 [-0.0092979, 0.0284113] (clusters=18) | 0.0877859 [0.072699, 0.106564] (clusters=18) | 0.0299817 [0.0159116, 0.0450958] (clusters=18) | — |
| math | response | mean_effect | 72 | 0.070204 [0.052727, 0.0922336] (clusters=18) | -1.07938e-17 [-0.0379515, 0.0318456] (clusters=18) | 0.0578041 [0.0421643, 0.0734746] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| qa | I1 | zero | 54 | 0.594364 [0.330725, 0.91514] (clusters=18) | -0.523908 [-0.867916, -0.240006] (clusters=18) | 0 [0, 0] (clusters=18) | 0.0498533 [-0.0864976, 0.197534] (clusters=18) | — |
| qa | I1 | T_only | 54 | 0.682625 [0.396015, 1.03375] (clusters=18) | -0.659395 [-1.0166, -0.364339] (clusters=18) | -0.0882605 [-0.121333, -0.0558594] (clusters=18) | -0.0384072 [-0.195486, 0.131667] (clusters=18) | — |
| qa | I1 | joint_initial_loss | 54 | 0.703445 [0.454859, 1.01026] (clusters=18) | -0.524055 [-0.880701, -0.217045] (clusters=18) | -0.109081 [-0.170299, -0.0445174] (clusters=18) | -0.0592275 [-0.172438, 0.0663323] (clusters=18) | 0.22917 [-0.140568, 0.651385] (clusters=18) |
| qa | I1 | surface_initial_loss | 54 | 0.932616 [0.624358, 1.30516] (clusters=18) | 0.193026 [-0.2911, 0.724382] (clusters=18) | -0.338251 [-0.746469, 0.0222431] (clusters=18) | -0.288398 [-0.726994, 0.0502792] (clusters=18) | — |
| qa | I1 | mean_effect | 54 | 0.644218 [0.452732, 0.87775] (clusters=18) | 6.5791e-17 [-0.375255, 0.314311] (clusters=18) | -0.0498533 [-0.197534, 0.0864976] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| qa | I2_seed | zero | 33 | 0.182307 [0.0564873, 0.339532] (clusters=18) | -0.0357453 [-0.259613, 0.158766] (clusters=18) | 0 [0, 0] (clusters=18) | 0.0123943 [-0.0275612, 0.0479796] (clusters=18) | — |
| qa | I2_seed | T_only | 33 | 0.182862 [0.0568769, 0.339309] (clusters=18) | -0.0352153 [-0.259118, 0.160325] (clusters=18) | -0.000555392 [-0.00214356, 0.000539433] (clusters=18) | 0.0118389 [-0.0277832, 0.0480587] (clusters=18) | — |
| qa | I2_seed | joint_initial_loss | 33 | 0.194521 [0.0700153, 0.345831] (clusters=18) | -0.0458226 [-0.268197, 0.162391] (clusters=18) | -0.0122144 [-0.024557, -0.00138314] (clusters=18) | 0.000179894 [-0.0511363, 0.0403254] (clusters=18) | 0.0134889 [-0.00322068, 0.0323776] (clusters=18) |
| qa | I2_seed | surface_initial_loss | 33 | 0.20801 [0.0738857, 0.351476] (clusters=18) | -0.0409732 [-0.277601, 0.181371] (clusters=18) | -0.0257033 [-0.0492505, -0.00290449] (clusters=18) | -0.013309 [-0.0713208, 0.0344503] (clusters=18) | — |
| qa | I2_seed | mean_effect | 33 | 0.194701 [0.0648072, 0.347939] (clusters=18) | 1.34572e-17 [-0.229478, 0.210262] (clusters=18) | -0.0123943 [-0.0479796, 0.0275612] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| qa | I2_size | zero | 69 | 1.00354 [0.395669, 1.67609] (clusters=18) | 0.968201 [0.359174, 1.63662] (clusters=18) | 0 [0, 0] (clusters=18) | 0.17362 [-0.103937, 0.548031] (clusters=18) | — |
| qa | I2_size | T_only | 69 | 1.00225 [0.394751, 1.67397] (clusters=18) | 0.965977 [0.357686, 1.63537] (clusters=18) | 0.00129404 [0.000314846, 0.00231886] (clusters=18) | 0.174914 [-0.102599, 0.549183] (clusters=18) | — |
| qa | I2_size | joint_initial_loss | 69 | 1.90731 [1.19592, 2.84741] (clusters=18) | -1.0279 [-2.6122, 0.572516] (clusters=18) | -0.903764 [-2.20587, 0.0545793] (clusters=18) | -0.730144 [-1.85338, 0.0217303] (clusters=18) | 0.0473207 [-0.449889, 0.567122] (clusters=18) |
| qa | I2_size | surface_initial_loss | 69 | 1.95463 [0.988107, 3.15064] (clusters=18) | -1.33896 [-3.03209, 0.332385] (clusters=18) | -0.951085 [-2.54241, 0.348046] (clusters=18) | -0.777464 [-2.22109, 0.385476] (clusters=18) | — |
| qa | I2_size | mean_effect | 69 | 1.17717 [0.819321, 1.64362] (clusters=18) | -1.28722e-16 [-0.762009, 0.86979] (clusters=18) | -0.17362 [-0.548031, 0.103937] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| qa | response | zero | 72 | 1.24699 [1.07051, 1.45651] (clusters=18) | 0.705175 [0.33567, 1.01329] (clusters=18) | 0 [0, 0] (clusters=18) | -0.461771 [-0.587958, -0.325908] (clusters=18) | — |
| qa | response | T_only | 72 | 1.12378 [0.904636, 1.39353] (clusters=18) | 0.421166 [0.0225362, 0.749508] (clusters=18) | 0.123211 [0.0303052, 0.214771] (clusters=18) | -0.338559 [-0.394213, -0.280658] (clusters=18) | — |
| qa | response | joint_initial_loss | 72 | 1.69685 [1.17685, 2.26932] (clusters=18) | 1.18816 [0.523058, 1.88584] (clusters=18) | -0.449859 [-1.08136, 0.095844] (clusters=18) | -0.91163 [-1.49439, -0.415015] (clusters=18) | -0.410094 [-0.620982, -0.216159] (clusters=18) |
| qa | response | surface_initial_loss | 72 | 1.28676 [0.757992, 1.89852] (clusters=18) | 0.768544 [0.107277, 1.51331] (clusters=18) | -0.0397658 [-0.735594, 0.529266] (clusters=18) | -0.501536 [-1.131, 0.0140734] (clusters=18) | — |
| qa | response | mean_effect | 72 | 0.785221 [0.541146, 1.0835] (clusters=18) | 8.63507e-17 [-0.389589, 0.321431] (clusters=18) | 0.461771 [0.325908, 0.587958] (clusters=18) | 0 [0, 0] (clusters=18) | — |

leave_one_pool_seed_out, descriptor=log_parameters; pooled held-out predictions; targets remain separate

| Capability | Target | Candidate | Pairs | MAE [95% CI] | Bias [95% CI] | Gain vs zero [95% CI] | Gain vs mean [95% CI] | Gain vs surface [95% CI] |
|---|---|---|---:|---|---|---|---|---|
| code | I1 | zero | 54 | 0.0590416 [0.0440249, 0.0762806] (clusters=18) | -0.0571656 [-0.0751231, -0.0410726] (clusters=18) | 0 [0, 0] (clusters=18) | -0.0224564 [-0.034248, -0.0101572] (clusters=18) | — |
| code | I1 | T_only | 54 | 0.0430603 [0.0347284, 0.0521809] (clusters=18) | 0.00718919 [-0.0112127, 0.0234023] (clusters=18) | 0.0159813 [0.000339592, 0.0320954] (clusters=18) | -0.0064751 [-0.012097, -0.000393759] (clusters=18) | — |
| code | I1 | joint_log_parameters | 54 | 0.036129 [0.0289042, 0.0447283] (clusters=18) | 0.000361614 [-0.0141754, 0.0122528] (clusters=18) | 0.0229126 [0.0085793, 0.0378324] (clusters=18) | 0.000456235 [-0.00435585, 0.00546659] (clusters=18) | -0.00466636 [-0.0103625, 0.00185361] (clusters=18) |
| code | I1 | surface_log_parameters | 54 | 0.0314626 [0.0248026, 0.0390272] (clusters=18) | -0.00151497 [-0.0136864, 0.00919247] (clusters=18) | 0.027579 [0.0154924, 0.0409703] (clusters=18) | 0.0051226 [-0.00162198, 0.0119129] (clusters=18) | — |
| code | I1 | mean_effect | 54 | 0.0365852 [0.0279822, 0.0472591] (clusters=18) | -3.08395e-18 [-0.0186464, 0.0165448] (clusters=18) | 0.0224564 [0.0101572, 0.034248] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| code | I2_size | zero | 69 | 0.0713177 [0.0412525, 0.107236] (clusters=18) | 0.065306 [0.0358652, 0.103237] (clusters=18) | 0 [0, 0] (clusters=18) | -0.00218747 [-0.0159687, 0.0177207] (clusters=18) | — |
| code | I2_size | T_only | 69 | 0.0718032 [0.0418856, 0.107955] (clusters=18) | 0.0664157 [0.0369495, 0.104089] (clusters=18) | -0.000485566 [-0.00119561, 0.000199575] (clusters=18) | -0.00267304 [-0.016599, 0.0175069] (clusters=18) | — |
| code | I2_size | joint_log_parameters | 69 | 0.0632061 [0.0390019, 0.0889521] (clusters=18) | -0.0126394 [-0.0630541, 0.0396871] (clusters=18) | 0.00811157 [-0.0157244, 0.027495] (clusters=18) | 0.0059241 [-0.00779353, 0.0148882] (clusters=18) | -0.011782 [-0.0209795, -0.00153831] (clusters=18) |
| code | I2_size | surface_log_parameters | 69 | 0.0514241 [0.0293581, 0.0746739] (clusters=18) | -0.0046948 [-0.0536885, 0.0464652] (clusters=18) | 0.0198936 [-0.00610749, 0.0435895] (clusters=18) | 0.0177061 [0.00237563, 0.0337185] (clusters=18) | — |
| code | I2_size | mean_effect | 69 | 0.0691302 [0.0460806, 0.0964477] (clusters=18) | 0.00185273 [-0.0404264, 0.0526605] (clusters=18) | 0.00218747 [-0.0177207, 0.0159687] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| code | response | zero | 72 | 0.140706 [0.119756, 0.164009] (clusters=18) | -0.140706 [-0.164009, -0.119756] (clusters=18) | 0 [0, 0] (clusters=18) | -0.0756392 [-0.0870631, -0.0628082] (clusters=18) | — |
| code | response | T_only | 72 | 0.0484663 [0.0360857, 0.0633457] (clusters=18) | -0.00568956 [-0.0305967, 0.0164456] (clusters=18) | 0.0922398 [0.0736741, 0.108856] (clusters=18) | 0.0166006 [0.00444059, 0.0281878] (clusters=18) | — |
| code | response | joint_log_parameters | 72 | 0.0427089 [0.0323855, 0.0544788] (clusters=18) | -0.000972561 [-0.0202488, 0.0191389] (clusters=18) | 0.0979972 [0.0790562, 0.1171] (clusters=18) | 0.022358 [0.00886655, 0.0355876] (clusters=18) | -0.00713732 [-0.0108012, -0.00371935] (clusters=18) |
| code | response | surface_log_parameters | 72 | 0.0355716 [0.0246284, 0.0480222] (clusters=18) | -0.000121869 [-0.0185928, 0.01938] (clusters=18) | 0.105135 [0.087024, 0.123] (clusters=18) | 0.0294953 [0.0175024, 0.0409038] (clusters=18) | — |
| code | response | mean_effect | 72 | 0.0650669 [0.0491161, 0.0836933] (clusters=18) | -4.62593e-18 [-0.0245081, 0.0216686] (clusters=18) | 0.0756392 [0.0628082, 0.0870631] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| math | I1 | zero | 54 | 0.0448632 [0.0260747, 0.0670939] (clusters=18) | -0.0330948 [-0.0577697, -0.0124391] (clusters=18) | 0 [0, 0] (clusters=18) | 0.000259988 [-0.00870192, 0.00958689] (clusters=18) | — |
| math | I1 | T_only | 54 | 0.0528202 [0.0418134, 0.0657523] (clusters=18) | 0.0221938 [-0.00254388, 0.0429017] (clusters=18) | -0.00795706 [-0.0242772, 0.00811163] (clusters=18) | -0.00769708 [-0.0156794, 0.000741362] (clusters=18) | — |
| math | I1 | joint_log_parameters | 54 | 0.0423786 [0.0317411, 0.0558771] (clusters=18) | 0.0153439 [-0.00201766, 0.0299251] (clusters=18) | 0.00248453 [-0.00951351, 0.0146564] (clusters=18) | 0.00274451 [-0.00375471, 0.00932053] (clusters=18) | -0.00365078 [-0.0129912, 0.00464352] (clusters=18) |
| math | I1 | surface_log_parameters | 54 | 0.0387279 [0.031879, 0.0460071] (clusters=18) | 0.0141263 [0.00614721, 0.0213992] (clusters=18) | 0.0061353 [-0.0117534, 0.0257646] (clusters=18) | 0.00639529 [-0.00561803, 0.0206359] (clusters=18) | — |
| math | I1 | mean_effect | 54 | 0.0451232 [0.0306738, 0.0631102] (clusters=18) | 6.16791e-18 [-0.0246436, 0.0207086] (clusters=18) | -0.000259988 [-0.00958689, 0.00870192] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| math | I2_size | zero | 69 | 0.0687308 [0.030717, 0.11597] (clusters=18) | 0.0603638 [0.0250761, 0.104693] (clusters=18) | 0 [0, 0] (clusters=18) | 0.00688149 [-0.00554901, 0.0232408] (clusters=18) | — |
| math | I2_size | T_only | 69 | 0.0691228 [0.0311787, 0.116242] (clusters=18) | 0.0612973 [0.0261336, 0.105473] (clusters=18) | -0.000392074 [-0.000928582, 0.000216625] (clusters=18) | 0.00648941 [-0.00591117, 0.0228429] (clusters=18) | — |
| math | I2_size | joint_log_parameters | 69 | 0.0605617 [0.0314292, 0.0936802] (clusters=18) | -0.0174932 [-0.0455794, 0.00878862] (clusters=18) | 0.00816903 [-0.00868159, 0.0249674] (clusters=18) | 0.0150505 [0.00421582, 0.0255099] (clusters=18) | -0.0310004 [-0.050022, -0.0124579] (clusters=18) |
| math | I2_size | surface_log_parameters | 69 | 0.0295614 [0.0151016, 0.0465939] (clusters=18) | -0.00493355 [-0.0317894, 0.016816] (clusters=18) | 0.0391694 [0.0127759, 0.0716611] (clusters=18) | 0.0460509 [0.0260941, 0.0692147] (clusters=18) | — |
| math | I2_size | mean_effect | 69 | 0.0756123 [0.0465933, 0.112949] (clusters=18) | 0.000472596 [-0.035238, 0.0468206] (clusters=18) | -0.00688149 [-0.0232408, 0.00554901] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| math | response | zero | 72 | 0.128008 [0.103076, 0.159011] (clusters=18) | -0.128008 [-0.159011, -0.103076] (clusters=18) | 0 [0, 0] (clusters=18) | -0.0699421 [-0.0861965, -0.0545254] (clusters=18) | — |
| math | response | T_only | 72 | 0.0586277 [0.0423957, 0.0801577] (clusters=18) | -0.012055 [-0.0435017, 0.0131972] (clusters=18) | 0.0693804 [0.0525526, 0.0855624] (clusters=18) | -0.000561679 [-0.0125875, 0.011324] (clusters=18) | — |
| math | response | joint_log_parameters | 72 | 0.0439462 [0.0356249, 0.0531713] (clusters=18) | -0.00739309 [-0.022502, 0.0079159] (clusters=18) | 0.0840619 [0.064782, 0.107033] (clusters=18) | 0.0141198 [-0.00110064, 0.0296759] (clusters=18) | -0.010902 [-0.0154624, -0.00621813] (clusters=18) |
| math | response | surface_log_parameters | 72 | 0.0330442 [0.026254, 0.0409004] (clusters=18) | -0.00739845 [-0.0210535, 0.00614167] (clusters=18) | 0.0949639 [0.0743324, 0.120231] (clusters=18) | 0.0250218 [0.00965883, 0.0420341] (clusters=18) | — |
| math | response | mean_effect | 72 | 0.058066 [0.0407695, 0.0792694] (clusters=18) | -7.70988e-18 [-0.0312172, 0.0251931] (clusters=18) | 0.0699421 [0.0545254, 0.0861965] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| qa | I1 | zero | 54 | 0.594364 [0.330725, 0.91514] (clusters=18) | -0.523908 [-0.867916, -0.240006] (clusters=18) | 0 [0, 0] (clusters=18) | 0.0222677 [-0.101531, 0.147473] (clusters=18) | — |
| qa | I1 | T_only | 54 | 0.687036 [0.4135, 1.02001] (clusters=18) | -0.659471 [-1.00248, -0.374937] (clusters=18) | -0.0926714 [-0.112424, -0.0705672] (clusters=18) | -0.0704036 [-0.20762, 0.0672375] (clusters=18) | — |
| qa | I1 | joint_log_parameters | 54 | 0.745716 [0.529678, 1.008] (clusters=18) | -0.732169 [-0.996394, -0.515422] (clusters=18) | -0.151352 [-0.21744, -0.0814808] (clusters=18) | -0.129084 [-0.226823, -0.0301986] (clusters=18) | -0.105961 [-0.328341, 0.076961] (clusters=18) |
| qa | I1 | surface_log_parameters | 54 | 0.639755 [0.527629, 0.773631] (clusters=18) | -0.192355 [-0.301691, -0.105836] (clusters=18) | -0.0453907 [-0.263539, 0.219373] (clusters=18) | -0.023123 [-0.181878, 0.168531] (clusters=18) | — |
| qa | I1 | mean_effect | 54 | 0.616632 [0.436249, 0.842772] (clusters=18) | -1.31582e-16 [-0.342486, 0.284538] (clusters=18) | -0.0222677 [-0.147473, 0.101531] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| qa | I2_size | zero | 69 | 1.00354 [0.395669, 1.67609] (clusters=18) | 0.968201 [0.359174, 1.63662] (clusters=18) | 0 [0, 0] (clusters=18) | 0.109425 [-0.115607, 0.421323] (clusters=18) | — |
| qa | I2_size | T_only | 69 | 1.00206 [0.394189, 1.67448] (clusters=18) | 0.966089 [0.35707, 1.63574] (clusters=18) | 0.00148298 [0.000573332, 0.00259027] (clusters=18) | 0.110908 [-0.114332, 0.423415] (clusters=18) | — |
| qa | I2_size | joint_log_parameters | 69 | 0.810456 [0.394529, 1.30399] (clusters=18) | 0.128697 [-0.285609, 0.523354] (clusters=18) | 0.193088 [-0.07022, 0.421478] (clusters=18) | 0.302513 [0.158947, 0.48532] (clusters=18) | -0.370223 [-0.622499, -0.139529] (clusters=18) |
| qa | I2_size | surface_log_parameters | 69 | 0.440233 [0.213581, 0.719674] (clusters=18) | -0.0634607 [-0.442572, 0.227246] (clusters=18) | 0.563311 [0.145906, 0.993556] (clusters=18) | 0.672736 [0.435539, 0.942582] (clusters=18) | — |
| qa | I2_size | mean_effect | 69 | 1.11297 [0.712605, 1.62715] (clusters=18) | 0.00706262 [-0.611611, 0.696128] (clusters=18) | -0.109425 [-0.421323, 0.115607] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| qa | response | zero | 72 | 1.24699 [1.07051, 1.45651] (clusters=18) | 0.705175 [0.33567, 1.01329] (clusters=18) | 0 [0, 0] (clusters=18) | -0.467857 [-0.594716, -0.333484] (clusters=18) | — |
| qa | response | T_only | 72 | 1.09481 [0.879672, 1.35822] (clusters=18) | 0.421246 [0.0531427, 0.72777] (clusters=18) | 0.152182 [0.0716601, 0.225615] (clusters=18) | -0.315675 [-0.370611, -0.25855] (clusters=18) | — |
| qa | response | joint_log_parameters | 72 | 0.987725 [0.763977, 1.25119] (clusters=18) | 0.470883 [0.30956, 0.635589] (clusters=18) | 0.259267 [0.1128, 0.400905] (clusters=18) | -0.20859 [-0.297993, -0.120622] (clusters=18) | -0.468473 [-0.611403, -0.337759] (clusters=18) |
| qa | response | surface_log_parameters | 72 | 0.519253 [0.406951, 0.65519] (clusters=18) | 0.144733 [0.0283687, 0.270808] (clusters=18) | 0.727739 [0.620886, 0.831979] (clusters=18) | 0.259883 [0.113431, 0.437143] (clusters=18) | — |
| qa | response | mean_effect | 72 | 0.779135 [0.530135, 1.08172] (clusters=18) | -2.46716e-17 [-0.368605, 0.306965] (clusters=18) | 0.467857 [0.333484, 0.594716] (clusters=18) | 0 [0, 0] (clusters=18) | — |

leave_one_pool_seed_out, descriptor=initial_loss; pooled held-out predictions; targets remain separate

| Capability | Target | Candidate | Pairs | MAE [95% CI] | Bias [95% CI] | Gain vs zero [95% CI] | Gain vs mean [95% CI] | Gain vs surface [95% CI] |
|---|---|---|---:|---|---|---|---|---|
| code | I1 | zero | 54 | 0.0590416 [0.0440249, 0.0762806] (clusters=18) | -0.0571656 [-0.0751231, -0.0410726] (clusters=18) | 0 [0, 0] (clusters=18) | -0.0224564 [-0.034248, -0.0101572] (clusters=18) | — |
| code | I1 | T_only | 54 | 0.0430603 [0.0347284, 0.0521809] (clusters=18) | 0.00718919 [-0.0112127, 0.0234023] (clusters=18) | 0.0159813 [0.000339592, 0.0320954] (clusters=18) | -0.0064751 [-0.012097, -0.000393759] (clusters=18) | — |
| code | I1 | joint_initial_loss | 54 | 0.0360029 [0.0287543, 0.0446578] (clusters=18) | 0.000361614 [-0.0141381, 0.0122486] (clusters=18) | 0.0230387 [0.00847604, 0.0379602] (clusters=18) | 0.000582362 [-0.00425621, 0.00562082] (clusters=18) | -0.00462529 [-0.0103335, 0.00181666] (clusters=18) |
| code | I1 | surface_initial_loss | 54 | 0.0313776 [0.0247262, 0.0392316] (clusters=18) | -0.00151497 [-0.0139677, 0.00912772] (clusters=18) | 0.027664 [0.0154436, 0.0411004] (clusters=18) | 0.00520765 [-0.00175613, 0.0122098] (clusters=18) | — |
| code | I1 | mean_effect | 54 | 0.0365852 [0.0279822, 0.0472591] (clusters=18) | -3.08395e-18 [-0.0186464, 0.0165448] (clusters=18) | 0.0224564 [0.0101572, 0.034248] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| code | I2_size | zero | 69 | 0.0713177 [0.0412525, 0.107236] (clusters=18) | 0.065306 [0.0358652, 0.103237] (clusters=18) | 0 [0, 0] (clusters=18) | -0.00218747 [-0.0159687, 0.0177207] (clusters=18) | — |
| code | I2_size | T_only | 69 | 0.0718032 [0.0418856, 0.107955] (clusters=18) | 0.0664157 [0.0369495, 0.104089] (clusters=18) | -0.000485566 [-0.00119561, 0.000199575] (clusters=18) | -0.00267304 [-0.016599, 0.0175069] (clusters=18) | — |
| code | I2_size | joint_initial_loss | 69 | 0.0633659 [0.0400553, 0.0876791] (clusters=18) | -0.0126394 [-0.0608483, 0.0401943] (clusters=18) | 0.00795177 [-0.0152375, 0.0269133] (clusters=18) | 0.0057643 [-0.0088034, 0.0146448] (clusters=18) | -0.011928 [-0.0210266, -0.00187868] (clusters=18) |
| code | I2_size | surface_initial_loss | 69 | 0.0514379 [0.0299685, 0.0748807] (clusters=18) | -0.0046948 [-0.0522038, 0.0474936] (clusters=18) | 0.0198798 [-0.00524017, 0.0437073] (clusters=18) | 0.0176923 [0.00102963, 0.0341609] (clusters=18) | — |
| code | I2_size | mean_effect | 69 | 0.0691302 [0.0460806, 0.0964477] (clusters=18) | 0.00185273 [-0.0404264, 0.0526605] (clusters=18) | 0.00218747 [-0.0177207, 0.0159687] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| code | response | zero | 72 | 0.140706 [0.119756, 0.164009] (clusters=18) | -0.140706 [-0.164009, -0.119756] (clusters=18) | 0 [0, 0] (clusters=18) | -0.0756392 [-0.0870631, -0.0628082] (clusters=18) | — |
| code | response | T_only | 72 | 0.0484663 [0.0360857, 0.0633457] (clusters=18) | -0.00568956 [-0.0305967, 0.0164456] (clusters=18) | 0.0922398 [0.0736741, 0.108856] (clusters=18) | 0.0166006 [0.00444059, 0.0281878] (clusters=18) | — |
| code | response | joint_initial_loss | 72 | 0.0424352 [0.032669, 0.0535436] (clusters=18) | -0.000972561 [-0.0198443, 0.01854] (clusters=18) | 0.098271 [0.0798717, 0.117261] (clusters=18) | 0.0226317 [0.009366, 0.0358271] (clusters=18) | -0.00715392 [-0.010532, -0.00393633] (clusters=18) |
| code | response | surface_initial_loss | 72 | 0.0352812 [0.0248146, 0.047196] (clusters=18) | -0.000121869 [-0.018106, 0.0185709] (clusters=18) | 0.105425 [0.088032, 0.123304] (clusters=18) | 0.0297857 [0.0181859, 0.0412645] (clusters=18) | — |
| code | response | mean_effect | 72 | 0.0650669 [0.0491161, 0.0836933] (clusters=18) | -4.62593e-18 [-0.0245081, 0.0216686] (clusters=18) | 0.0756392 [0.0628082, 0.0870631] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| math | I1 | zero | 54 | 0.0448632 [0.0260747, 0.0670939] (clusters=18) | -0.0330948 [-0.0577697, -0.0124391] (clusters=18) | 0 [0, 0] (clusters=18) | 0.000259988 [-0.00870192, 0.00958689] (clusters=18) | — |
| math | I1 | T_only | 54 | 0.0528202 [0.0418134, 0.0657523] (clusters=18) | 0.0221938 [-0.00254388, 0.0429017] (clusters=18) | -0.00795706 [-0.0242772, 0.00811163] (clusters=18) | -0.00769708 [-0.0156794, 0.000741362] (clusters=18) | — |
| math | I1 | joint_initial_loss | 54 | 0.0429119 [0.0319509, 0.0561539] (clusters=18) | 0.0153439 [-0.00256562, 0.0307644] (clusters=18) | 0.00195125 [-0.0105919, 0.0140437] (clusters=18) | 0.00221124 [-0.00413583, 0.00868956] (clusters=18) | -0.00366037 [-0.0129568, 0.00468788] (clusters=18) |
| math | I1 | surface_initial_loss | 54 | 0.0392516 [0.0328567, 0.0458688] (clusters=18) | 0.0141263 [0.00445081, 0.0233329] (clusters=18) | 0.00561162 [-0.0120948, 0.0255668] (clusters=18) | 0.00587161 [-0.00639471, 0.0206253] (clusters=18) | — |
| math | I1 | mean_effect | 54 | 0.0451232 [0.0306738, 0.0631102] (clusters=18) | 6.16791e-18 [-0.0246436, 0.0207086] (clusters=18) | -0.000259988 [-0.00958689, 0.00870192] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| math | I2_size | zero | 69 | 0.0687308 [0.030717, 0.11597] (clusters=18) | 0.0603638 [0.0250761, 0.104693] (clusters=18) | 0 [0, 0] (clusters=18) | 0.00688149 [-0.00554901, 0.0232408] (clusters=18) | — |
| math | I2_size | T_only | 69 | 0.0691228 [0.0311787, 0.116242] (clusters=18) | 0.0612973 [0.0261336, 0.105473] (clusters=18) | -0.000392074 [-0.000928582, 0.000216625] (clusters=18) | 0.00648941 [-0.00591117, 0.0228429] (clusters=18) | — |
| math | I2_size | joint_initial_loss | 69 | 0.0592433 [0.0308761, 0.0942795] (clusters=18) | -0.0174932 [-0.046901, 0.00661239] (clusters=18) | 0.00948749 [-0.00535341, 0.0261878] (clusters=18) | 0.016369 [0.00551656, 0.0254304] (clusters=18) | -0.0306257 [-0.0484943, -0.0128364] (clusters=18) |
| math | I2_size | surface_initial_loss | 69 | 0.0286176 [0.0146841, 0.0469827] (clusters=18) | -0.00493355 [-0.033226, 0.0134554] (clusters=18) | 0.0401132 [0.01507, 0.0716415] (clusters=18) | 0.0469947 [0.0277413, 0.0689715] (clusters=18) | — |
| math | I2_size | mean_effect | 69 | 0.0756123 [0.0465933, 0.112949] (clusters=18) | 0.000472596 [-0.035238, 0.0468206] (clusters=18) | -0.00688149 [-0.0232408, 0.00554901] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| math | response | zero | 72 | 0.128008 [0.103076, 0.159011] (clusters=18) | -0.128008 [-0.159011, -0.103076] (clusters=18) | 0 [0, 0] (clusters=18) | -0.0699421 [-0.0861965, -0.0545254] (clusters=18) | — |
| math | response | T_only | 72 | 0.0586277 [0.0423957, 0.0801577] (clusters=18) | -0.012055 [-0.0435017, 0.0131972] (clusters=18) | 0.0693804 [0.0525526, 0.0855624] (clusters=18) | -0.000561679 [-0.0125875, 0.011324] (clusters=18) | — |
| math | response | joint_initial_loss | 72 | 0.0407041 [0.0316409, 0.0505834] (clusters=18) | -0.00739309 [-0.0191489, 0.00523414] (clusters=18) | 0.087304 [0.0679971, 0.110557] (clusters=18) | 0.0173619 [0.000904063, 0.0341202] (clusters=18) | -0.0105931 [-0.0154718, -0.00560673] (clusters=18) |
| math | response | surface_initial_loss | 72 | 0.030111 [0.0233478, 0.0378644] (clusters=18) | -0.00739845 [-0.0173084, 0.00269682] (clusters=18) | 0.0978971 [0.0779993, 0.122844] (clusters=18) | 0.027955 [0.011742, 0.0459323] (clusters=18) | — |
| math | response | mean_effect | 72 | 0.058066 [0.0407695, 0.0792694] (clusters=18) | -7.70988e-18 [-0.0312172, 0.0251931] (clusters=18) | 0.0699421 [0.0545254, 0.0861965] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| qa | I1 | zero | 54 | 0.594364 [0.330725, 0.91514] (clusters=18) | -0.523908 [-0.867916, -0.240006] (clusters=18) | 0 [0, 0] (clusters=18) | 0.0222677 [-0.101531, 0.147473] (clusters=18) | — |
| qa | I1 | T_only | 54 | 0.687036 [0.4135, 1.02001] (clusters=18) | -0.659471 [-1.00248, -0.374937] (clusters=18) | -0.0926714 [-0.112424, -0.0705672] (clusters=18) | -0.0704036 [-0.20762, 0.0672375] (clusters=18) | — |
| qa | I1 | joint_initial_loss | 54 | 0.741538 [0.510651, 1.02396] (clusters=18) | -0.732169 [-1.01693, -0.501736] (clusters=18) | -0.147173 [-0.198239, -0.0955354] (clusters=18) | -0.124905 [-0.221628, -0.0256859] (clusters=18) | -0.0696735 [-0.265835, 0.100528] (clusters=18) |
| qa | I1 | surface_initial_loss | 54 | 0.671864 [0.545188, 0.815196] (clusters=18) | -0.192355 [-0.369509, -0.0517843] (clusters=18) | -0.0774997 [-0.274237, 0.148554] (clusters=18) | -0.055232 [-0.198564, 0.0976542] (clusters=18) | — |
| qa | I1 | mean_effect | 54 | 0.616632 [0.436249, 0.842772] (clusters=18) | -1.31582e-16 [-0.342486, 0.284538] (clusters=18) | -0.0222677 [-0.147473, 0.101531] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| qa | I2_size | zero | 69 | 1.00354 [0.395669, 1.67609] (clusters=18) | 0.968201 [0.359174, 1.63662] (clusters=18) | 0 [0, 0] (clusters=18) | 0.109425 [-0.115607, 0.421323] (clusters=18) | — |
| qa | I2_size | T_only | 69 | 1.00206 [0.394189, 1.67448] (clusters=18) | 0.966089 [0.35707, 1.63574] (clusters=18) | 0.00148298 [0.000573332, 0.00259027] (clusters=18) | 0.110908 [-0.114332, 0.423415] (clusters=18) | — |
| qa | I2_size | joint_initial_loss | 69 | 0.842506 [0.387098, 1.37517] (clusters=18) | 0.128697 [-0.377544, 0.723418] (clusters=18) | 0.161038 [-0.0851396, 0.373201] (clusters=18) | 0.270463 [0.138509, 0.41703] (clusters=18) | -0.328316 [-0.619532, -0.0625577] (clusters=18) |
| qa | I2_size | surface_initial_loss | 69 | 0.51419 [0.212803, 0.913844] (clusters=18) | -0.0634607 [-0.566682, 0.480917] (clusters=18) | 0.489354 [0.045706, 0.96566] (clusters=18) | 0.598779 [0.268545, 0.912986] (clusters=18) | — |
| qa | I2_size | mean_effect | 69 | 1.11297 [0.712605, 1.62715] (clusters=18) | 0.00706262 [-0.611611, 0.696128] (clusters=18) | -0.109425 [-0.421323, 0.115607] (clusters=18) | 0 [0, 0] (clusters=18) | — |
| qa | response | zero | 72 | 1.24699 [1.07051, 1.45651] (clusters=18) | 0.705175 [0.33567, 1.01329] (clusters=18) | 0 [0, 0] (clusters=18) | -0.467857 [-0.594716, -0.333484] (clusters=18) | — |
| qa | response | T_only | 72 | 1.09481 [0.879672, 1.35822] (clusters=18) | 0.421246 [0.0531427, 0.72777] (clusters=18) | 0.152182 [0.0716601, 0.225615] (clusters=18) | -0.315675 [-0.370611, -0.25855] (clusters=18) | — |
| qa | response | joint_initial_loss | 72 | 1.02569 [0.777507, 1.31639] (clusters=18) | 0.470883 [0.248747, 0.668015] (clusters=18) | 0.221298 [0.0635605, 0.375685] (clusters=18) | -0.246559 [-0.331937, -0.168276] (clusters=18) | -0.448289 [-0.584202, -0.322421] (clusters=18) |
| qa | response | surface_initial_loss | 72 | 0.577406 [0.426118, 0.761812] (clusters=18) | 0.144733 [-0.0417078, 0.324539] (clusters=18) | 0.669586 [0.536761, 0.787551] (clusters=18) | 0.201729 [0.0654477, 0.361115] (clusters=18) | — |
| qa | response | mean_effect | 72 | 0.779135 [0.530135, 1.08172] (clusters=18) | -2.46716e-17 [-0.368605, 0.306965] (clusters=18) | 0.467857 [0.333484, 0.594716] (clusters=18) | 0 [0, 0] (clusters=18) | — |

Part D: falsifiable implication and observation limit

At fixed student and T, F(pool2)-F(pool1) = (b+b_prime*z)*[log(1+T/D_U2)-log(1+T/D_U1)]. Thus the measured effect divided by that bracket should be flat in budget. A controlled systematic drift would falsify this implication.

Measured delta2-delta1 divided by log(1+T2/D_U2)-log(1+T1/D_U1). T2 differs from T1: these are observed-endpoint diagnostics, not the exact fixed-T falsification statistic.

Part D: measured observed-endpoint ratios (T1 != T2; approximate diagnostic)

Exact fixed-T ratio table: unavailable (zero nonzero-budget matched-T pairs). These intervals resample trajectories; clusters<6 are weak, and a single seed dyad gives a degenerate interval. Size rows average the two seed-specific ratios; seed rows never enter that average.

Pool size contrasts

| Capability | Student | Contrast | Checkpoint | Measured effect / observed reuse bracket [95% CI] | Eligible near-match pairs / all pairs |
|---|---|---|---:|---|---|
| code | gemma3-1b | middle->large | 1 | -0.0552184 [-0.0739946, -0.0364422] (clusters=4) | 1/2 |
| code | gemma3-1b | middle->large | 2 | -0.0116461 [-0.0170761, -0.00621607] (clusters=4) | 2/2 |
| code | gemma3-1b | middle->large | 3 | 0.072407 [0.058183, 0.086631] (clusters=4) | 2/2 |
| code | gemma3-1b | middle->large | 4 | 0.074523 [0.0563995, 0.0926465] (clusters=4) | 2/2 |
| code | gemma3-1b | small->large | 1 | -0.00928275 [-0.0173536, -0.00121195] (clusters=4) | 2/2 |
| code | gemma3-1b | small->large | 2 | 0.0116264 [-0.00887973, 0.0321326] (clusters=4) | 2/2 |
| code | gemma3-1b | small->large | 3 | 0.100556 [0.0428903, 0.158221] (clusters=4) | 2/2 |
| code | gemma3-1b | small->large | 4 | 0.163694 [0.0886318, 0.238756] (clusters=4) | 2/2 |
| code | gemma3-1b | small->middle | 1 | 0.0122096 [0.00751278, 0.0169064] (clusters=4) | 2/2 |
| code | gemma3-1b | small->middle | 2 | 0.025097 [-0.0103529, 0.0605468] (clusters=4) | 2/2 |
| code | gemma3-1b | small->middle | 3 | 0.120185 [0.0323032, 0.208067] (clusters=4) | 2/2 |
| code | gemma3-1b | small->middle | 4 | 0.233942 [0.114525, 0.353359] (clusters=4) | 2/2 |
| code | gemma3-270m | middle->large | 1 | -0.0470411 [-0.0653177, -0.0287645] (clusters=4) | 1/2 |
| code | gemma3-270m | middle->large | 2 | 0.0301798 [-0.00657332, 0.0669329] (clusters=4) | 2/2 |
| code | gemma3-270m | middle->large | 3 | 0.00356157 [-0.00647536, 0.0135985] (clusters=4) | 2/2 |
| code | gemma3-270m | middle->large | 4 | 0.0522811 [0.0431904, 0.0613719] (clusters=4) | 2/2 |
| code | gemma3-270m | small->large | 1 | 0.00716034 [-0.00213243, 0.0164531] (clusters=4) | 2/2 |
| code | gemma3-270m | small->large | 2 | 0.0342441 [0.0304757, 0.0380124] (clusters=4) | 2/2 |
| code | gemma3-270m | small->large | 3 | 0.0697935 [0.0568754, 0.0827116] (clusters=4) | 2/2 |
| code | gemma3-270m | small->large | 4 | 0.0918627 [0.0673397, 0.116386] (clusters=4) | 2/2 |
| code | gemma3-270m | small->middle | 1 | 0.0326574 [0.0256069, 0.0397078] (clusters=4) | 2/2 |
| code | gemma3-270m | small->middle | 2 | 0.0360475 [0.00942455, 0.0626704] (clusters=4) | 2/2 |
| code | gemma3-270m | small->middle | 3 | 0.115783 [0.100733, 0.130833] (clusters=4) | 2/2 |
| code | gemma3-270m | small->middle | 4 | 0.123138 [0.0867392, 0.159537] (clusters=4) | 2/2 |
| code | gemma3-4b | middle->large | 1 | -0.0503326 [-0.125212, 0.0245471] (clusters=4) | 1/2 |
| code | gemma3-4b | middle->large | 2 | -0.0255264 [-0.0417645, -0.00928838] (clusters=4) | 2/2 |
| code | gemma3-4b | middle->large | 3 | 0.0699375 [0.0624682, 0.0774068] (clusters=4) | 2/2 |
| code | gemma3-4b | middle->large | 4 | 0.128613 [0.0858409, 0.171385] (clusters=4) | 2/2 |
| code | gemma3-4b | small->large | 1 | -0.0191975 [-0.0487517, 0.0103566] (clusters=4) | 2/2 |
| code | gemma3-4b | small->large | 2 | -0.00559595 [-0.0372745, 0.0260826] (clusters=4) | 2/2 |
| code | gemma3-4b | small->large | 3 | 0.102397 [0.0857221, 0.119071] (clusters=4) | 2/2 |
| code | gemma3-4b | small->large | 4 | 0.149824 [0.120314, 0.179334] (clusters=4) | 2/2 |
| code | gemma3-4b | small->middle | 1 | -0.00606278 [-0.0151843, 0.00305872] (clusters=4) | 2/2 |
| code | gemma3-4b | small->middle | 2 | 0.00625344 [-0.0527522, 0.0652591] (clusters=4) | 2/2 |
| code | gemma3-4b | small->middle | 3 | 0.124951 [0.101821, 0.14808] (clusters=4) | 2/2 |
| code | gemma3-4b | small->middle | 4 | 0.166787 [0.148006, 0.185568] (clusters=4) | 2/2 |
| math | gemma3-1b | middle->large | 1 | -0.069148 [-0.112987, -0.0253091] (clusters=4) | 1/2 |
| math | gemma3-1b | middle->large | 2 | -0.017105 [-0.0191012, -0.0151088] (clusters=4) | 2/2 |
| math | gemma3-1b | middle->large | 3 | 0.00922673 [0.00289995, 0.0155535] (clusters=4) | 2/2 |
| math | gemma3-1b | middle->large | 4 | 0.0368818 [0.0302407, 0.043523] (clusters=4) | 2/2 |
| math | gemma3-1b | small->large | 1 | -0.023799 [-0.0421693, -0.00542875] (clusters=4) | 2/2 |
| math | gemma3-1b | small->large | 2 | -0.00630049 [-0.0122059, -0.000395129] (clusters=4) | 2/2 |
| math | gemma3-1b | small->large | 3 | 0.0596654 [0.050106, 0.0692249] (clusters=4) | 2/2 |
| math | gemma3-1b | small->large | 4 | 0.121926 [0.111928, 0.131924] (clusters=4) | 2/2 |
| math | gemma3-1b | small->middle | 1 | -0.00314192 [-0.0110792, 0.00479539] (clusters=4) | 2/2 |
| math | gemma3-1b | small->middle | 2 | -9.71035e-05 [-0.0106004, 0.0104062] (clusters=4) | 2/2 |
| math | gemma3-1b | small->middle | 3 | 0.0946905 [0.0827867, 0.106594] (clusters=4) | 2/2 |
| math | gemma3-1b | small->middle | 4 | 0.18928 [0.16688, 0.211681] (clusters=4) | 2/2 |
| math | gemma3-270m | middle->large | 1 | -0.0717238 [-0.119915, -0.023533] (clusters=4) | 1/2 |
| math | gemma3-270m | middle->large | 2 | -0.00306056 [-0.0133486, 0.00722747] (clusters=4) | 2/2 |
| math | gemma3-270m | middle->large | 3 | 0.0151966 [0.0014011, 0.028992] (clusters=4) | 2/2 |
| math | gemma3-270m | middle->large | 4 | 0.0187556 [0.00820212, 0.0293091] (clusters=4) | 2/2 |
| math | gemma3-270m | small->large | 1 | -0.0273539 [-0.0394268, -0.0152809] (clusters=4) | 2/2 |
| math | gemma3-270m | small->large | 2 | 0.000717513 [-0.00182103, 0.00325606] (clusters=4) | 2/2 |
| math | gemma3-270m | small->large | 3 | 0.0437374 [0.0432205, 0.0442543] (clusters=4) | 2/2 |
| math | gemma3-270m | small->large | 4 | 0.0554691 [0.0531169, 0.0578212] (clusters=4) | 2/2 |
| math | gemma3-270m | small->middle | 1 | -0.00756418 [-0.011037, -0.00409135] (clusters=4) | 2/2 |
| math | gemma3-270m | small->middle | 2 | 0.00269669 [-0.00704584, 0.0124392] (clusters=4) | 2/2 |
| math | gemma3-270m | small->middle | 3 | 0.0635244 [0.0531273, 0.0739215] (clusters=4) | 2/2 |
| math | gemma3-270m | small->middle | 4 | 0.0846915 [0.0801852, 0.0891978] (clusters=4) | 2/2 |
| math | gemma3-4b | middle->large | 1 | -0.0270416 [-0.133193, 0.0791094] (clusters=4) | 1/2 |
| math | gemma3-4b | middle->large | 2 | -0.0162694 [-0.0187257, -0.0138131] (clusters=4) | 2/2 |
| math | gemma3-4b | middle->large | 3 | 0.0426522 [0.0233791, 0.0619254] (clusters=4) | 2/2 |
| math | gemma3-4b | middle->large | 4 | 0.116018 [0.10265, 0.129385] (clusters=4) | 2/2 |
| math | gemma3-4b | small->large | 1 | -0.0378485 [-0.0537809, -0.0219161] (clusters=4) | 2/2 |
| math | gemma3-4b | small->large | 2 | 0.00876688 [-0.0127108, 0.0302446] (clusters=4) | 2/2 |
| math | gemma3-4b | small->large | 3 | 0.130684 [0.122939, 0.13843] (clusters=4) | 2/2 |
| math | gemma3-4b | small->large | 4 | 0.259893 [0.227148, 0.292637] (clusters=4) | 2/2 |
| math | gemma3-4b | small->middle | 1 | -0.0463949 [-0.0738718, -0.018918] (clusters=4) | 2/2 |
| math | gemma3-4b | small->middle | 2 | 0.02321 [-0.0121012, 0.0585211] (clusters=4) | 2/2 |
| math | gemma3-4b | small->middle | 3 | 0.191857 [0.165179, 0.218535] (clusters=4) | 2/2 |
| math | gemma3-4b | small->middle | 4 | 0.373923 [0.327161, 0.420686] (clusters=4) | 2/2 |
| qa | gemma3-1b | middle->large | 1 | -0.295271 [-0.568898, -0.0216449] (clusters=4) | 1/2 |
| qa | gemma3-1b | middle->large | 2 | 0.156305 [-0.0465999, 0.359211] (clusters=4) | 2/2 |
| qa | gemma3-1b | middle->large | 3 | 0.521923 [0.434644, 0.609202] (clusters=4) | 2/2 |
| qa | gemma3-1b | middle->large | 4 | 0.275127 [0.101587, 0.448666] (clusters=4) | 2/2 |
| qa | gemma3-1b | small->large | 1 | -0.0880949 [-0.236898, 0.0607083] (clusters=4) | 2/2 |
| qa | gemma3-1b | small->large | 2 | 0.334863 [0.196043, 0.473683] (clusters=4) | 2/2 |
| qa | gemma3-1b | small->large | 3 | 1.02303 [0.866611, 1.17945] (clusters=4) | 2/2 |
| qa | gemma3-1b | small->large | 4 | 2.01339 [1.83579, 2.19099] (clusters=4) | 2/2 |
| qa | gemma3-1b | small->middle | 1 | 0.00595824 [-0.0911447, 0.103061] (clusters=4) | 2/2 |
| qa | gemma3-1b | small->middle | 2 | 0.435009 [0.330237, 0.539782] (clusters=4) | 2/2 |
| qa | gemma3-1b | small->middle | 3 | 1.37108 [1.16566, 1.57649] (clusters=4) | 2/2 |
| qa | gemma3-1b | small->middle | 4 | 3.39326 [3.22891, 3.55761] (clusters=4) | 2/2 |
| qa | gemma3-270m | middle->large | 1 | -0.23721 [-0.330931, -0.143488] (clusters=4) | 1/2 |
| qa | gemma3-270m | middle->large | 2 | 0.156504 [0.128391, 0.184618] (clusters=4) | 2/2 |
| qa | gemma3-270m | middle->large | 3 | 0.179008 [0.0788121, 0.279204] (clusters=4) | 2/2 |
| qa | gemma3-270m | middle->large | 4 | 0.126785 [0.0248505, 0.228719] (clusters=4) | 2/2 |
| qa | gemma3-270m | small->large | 1 | -0.0460362 [-0.130315, 0.0382421] (clusters=4) | 2/2 |
| qa | gemma3-270m | small->large | 2 | 0.124948 [0.0770783, 0.172818] (clusters=4) | 2/2 |
| qa | gemma3-270m | small->large | 3 | 0.551524 [0.476739, 0.62631] (clusters=4) | 2/2 |
| qa | gemma3-270m | small->large | 4 | 0.731959 [0.649781, 0.814137] (clusters=4) | 2/2 |
| qa | gemma3-270m | small->middle | 1 | 0.0447312 [-0.0422407, 0.131703] (clusters=4) | 2/2 |
| qa | gemma3-270m | small->middle | 2 | 0.107352 [0.0487001, 0.166004] (clusters=4) | 2/2 |
| qa | gemma3-270m | small->middle | 3 | 0.810105 [0.752223, 0.867987] (clusters=4) | 2/2 |
| qa | gemma3-270m | small->middle | 4 | 1.21256 [1.1518, 1.27332] (clusters=4) | 2/2 |
| qa | gemma3-4b | middle->large | 1 | -0.34402 [-0.578113, -0.109927] (clusters=4) | 1/2 |
| qa | gemma3-4b | middle->large | 2 | -0.342898 [-0.646774, -0.0390212] (clusters=4) | 2/2 |
| qa | gemma3-4b | middle->large | 3 | -0.157482 [-0.515207, 0.200243] (clusters=4) | 2/2 |
| qa | gemma3-4b | middle->large | 4 | 1.30829 [1.26057, 1.35601] (clusters=4) | 2/2 |
| qa | gemma3-4b | small->large | 1 | -0.114771 [-0.23326, 0.00371683] (clusters=4) | 2/2 |
| qa | gemma3-4b | small->large | 2 | 0.300716 [0.273579, 0.327852] (clusters=4) | 2/2 |
| qa | gemma3-4b | small->large | 3 | 2.11065 [2.06587, 2.15542] (clusters=4) | 2/2 |
| qa | gemma3-4b | small->large | 4 | 3.50551 [2.89711, 4.1139] (clusters=4) | 2/2 |
| qa | gemma3-4b | small->middle | 1 | -0.00985086 [-0.0818636, 0.0621619] (clusters=4) | 2/2 |
| qa | gemma3-4b | small->middle | 2 | 0.661136 [0.539693, 0.782579] (clusters=4) | 2/2 |
| qa | gemma3-4b | small->middle | 3 | 3.68599 [3.50899, 3.86298] (clusters=4) | 2/2 |
| qa | gemma3-4b | small->middle | 4 | 5.24443 [4.21178, 6.27709] (clusters=4) | 2/2 |

Pool seed contrasts

| Capability | Student | Contrast | Checkpoint | Measured effect / observed reuse bracket [95% CI] | Eligible near-match pairs / all pairs |
|---|---|---|---:|---|---|
| code | gemma3-1b | large:seed41->42 | 1 | 0.11112 [0.11112, 0.11112] (clusters=2) | 0/1 |
| code | gemma3-1b | large:seed41->42 | 2 | 0.614203 [0.614203, 0.614203] (clusters=2) | 1/1 |
| code | gemma3-1b | large:seed41->42 | 3 | 2.34145 [2.34145, 2.34145] (clusters=2) | 1/1 |
| code | gemma3-1b | large:seed41->42 | 4 | -1.81885 [-1.81885, -1.81885] (clusters=2) | 1/1 |
| code | gemma3-1b | middle:seed41->42 | 1 | 0.996478 [0.996478, 0.996478] (clusters=2) | 1/1 |
| code | gemma3-1b | middle:seed41->42 | 2 | 62.1606 [62.1606, 62.1606] (clusters=2) | 1/1 |
| code | gemma3-1b | middle:seed41->42 | 3 | 0.522972 [0.522972, 0.522972] (clusters=2) | 1/1 |
| code | gemma3-1b | middle:seed41->42 | 4 | -1.23442 [-1.23442, -1.23442] (clusters=2) | 1/1 |
| code | gemma3-1b | small:seed41->42 | 1 | -0.488919 [-0.488919, -0.488919] (clusters=2) | 1/1 |
| code | gemma3-1b | small:seed41->42 | 2 | -20.2923 [-20.2923, -20.2923] (clusters=2) | 1/1 |
| code | gemma3-1b | small:seed41->42 | 3 | -12.1624 [-12.1624, -12.1624] (clusters=2) | 1/1 |
| code | gemma3-1b | small:seed41->42 | 4 | -44.5901 [-44.5901, -44.5901] (clusters=2) | 1/1 |
| code | gemma3-270m | large:seed41->42 | 1 | -0.268541 [-0.268541, -0.268541] (clusters=2) | 0/1 |
| code | gemma3-270m | large:seed41->42 | 2 | 0.720216 [0.720216, 0.720216] (clusters=2) | 1/1 |
| code | gemma3-270m | large:seed41->42 | 3 | 1.53609 [1.53609, 1.53609] (clusters=2) | 1/1 |
| code | gemma3-270m | large:seed41->42 | 4 | -1.73278 [-1.73278, -1.73278] (clusters=2) | 1/1 |
| code | gemma3-270m | middle:seed41->42 | 1 | 2.47312 [2.47312, 2.47312] (clusters=2) | 1/1 |
| code | gemma3-270m | middle:seed41->42 | 2 | -75.4891 [-75.4891, -75.4891] (clusters=2) | 1/1 |
| code | gemma3-270m | middle:seed41->42 | 3 | 0.246105 [0.246105, 0.246105] (clusters=2) | 1/1 |
| code | gemma3-270m | middle:seed41->42 | 4 | 0.529593 [0.529593, 0.529593] (clusters=2) | 1/1 |
| code | gemma3-270m | small:seed41->42 | 1 | -0.976461 [-0.976461, -0.976461] (clusters=2) | 1/1 |
| code | gemma3-270m | small:seed41->42 | 2 | 11.3662 [11.3662, 11.3662] (clusters=2) | 1/1 |
| code | gemma3-270m | small:seed41->42 | 3 | -1.94523 [-1.94523, -1.94523] (clusters=2) | 1/1 |
| code | gemma3-270m | small:seed41->42 | 4 | -12.4023 [-12.4023, -12.4023] (clusters=2) | 1/1 |
| code | gemma3-4b | large:seed41->42 | 1 | 0.838693 [0.838693, 0.838693] (clusters=2) | 0/1 |
| code | gemma3-4b | large:seed41->42 | 2 | 1.12744 [1.12744, 1.12744] (clusters=2) | 1/1 |
| code | gemma3-4b | large:seed41->42 | 3 | -0.0901753 [-0.0901753, -0.0901753] (clusters=2) | 1/1 |
| code | gemma3-4b | large:seed41->42 | 4 | -1.93934 [-1.93934, -1.93934] (clusters=2) | 1/1 |
| code | gemma3-4b | middle:seed41->42 | 1 | 3.26308 [3.26308, 3.26308] (clusters=2) | 1/1 |
| code | gemma3-4b | middle:seed41->42 | 2 | 135.527 [135.527, 135.527] (clusters=2) | 1/1 |
| code | gemma3-4b | middle:seed41->42 | 3 | -1.17925 [-1.17925, -1.17925] (clusters=2) | 1/1 |
| code | gemma3-4b | middle:seed41->42 | 4 | -6.21883 [-6.21883, -6.21883] (clusters=2) | 1/1 |
| code | gemma3-4b | small:seed41->42 | 1 | -1.34143 [-1.34143, -1.34143] (clusters=2) | 1/1 |
| code | gemma3-4b | small:seed41->42 | 2 | -29.4431 [-29.4431, -29.4431] (clusters=2) | 1/1 |
| code | gemma3-4b | small:seed41->42 | 3 | -4.08169 [-4.08169, -4.08169] (clusters=2) | 1/1 |
| code | gemma3-4b | small:seed41->42 | 4 | -15.0777 [-15.0777, -15.0777] (clusters=2) | 1/1 |
| math | gemma3-1b | large:seed41->42 | 1 | 0.735099 [0.735099, 0.735099] (clusters=2) | 0/1 |
| math | gemma3-1b | large:seed41->42 | 2 | 0.26947 [0.26947, 0.26947] (clusters=2) | 1/1 |
| math | gemma3-1b | large:seed41->42 | 3 | 1.02994 [1.02994, 1.02994] (clusters=2) | 1/1 |
| math | gemma3-1b | large:seed41->42 | 4 | 1.14263 [1.14263, 1.14263] (clusters=2) | 1/1 |
| math | gemma3-1b | middle:seed41->42 | 1 | 0.763203 [0.763203, 0.763203] (clusters=2) | 1/1 |
| math | gemma3-1b | middle:seed41->42 | 2 | 26.799 [26.799, 26.799] (clusters=2) | 1/1 |
| math | gemma3-1b | middle:seed41->42 | 3 | 0.222222 [0.222222, 0.222222] (clusters=2) | 1/1 |
| math | gemma3-1b | middle:seed41->42 | 4 | -0.0426383 [-0.0426383, -0.0426383] (clusters=2) | 1/1 |
| math | gemma3-1b | small:seed41->42 | 1 | -0.610745 [-0.610745, -0.610745] (clusters=2) | 1/1 |
| math | gemma3-1b | small:seed41->42 | 2 | -4.87801 [-4.87801, -4.87801] (clusters=2) | 1/1 |
| math | gemma3-1b | small:seed41->42 | 3 | -1.51862 [-1.51862, -1.51862] (clusters=2) | 1/1 |
| math | gemma3-1b | small:seed41->42 | 4 | -8.16101 [-8.16101, -8.16101] (clusters=2) | 1/1 |
| math | gemma3-270m | large:seed41->42 | 1 | 0.0280641 [0.0280641, 0.0280641] (clusters=2) | 0/1 |
| math | gemma3-270m | large:seed41->42 | 2 | 0.686343 [0.686343, 0.686343] (clusters=2) | 1/1 |
| math | gemma3-270m | large:seed41->42 | 3 | 1.72578 [1.72578, 1.72578] (clusters=2) | 1/1 |
| math | gemma3-270m | large:seed41->42 | 4 | -0.643957 [-0.643957, -0.643957] (clusters=2) | 1/1 |
| math | gemma3-270m | middle:seed41->42 | 1 | 3.90435 [3.90435, 3.90435] (clusters=2) | 1/1 |
| math | gemma3-270m | middle:seed41->42 | 2 | 13.6416 [13.6416, 13.6416] (clusters=2) | 1/1 |
| math | gemma3-270m | middle:seed41->42 | 3 | -0.0865495 [-0.0865495, -0.0865495] (clusters=2) | 1/1 |
| math | gemma3-270m | middle:seed41->42 | 4 | -1.34044 [-1.34044, -1.34044] (clusters=2) | 1/1 |
| math | gemma3-270m | small:seed41->42 | 1 | -0.874647 [-0.874647, -0.874647] (clusters=2) | 1/1 |
| math | gemma3-270m | small:seed41->42 | 2 | 9.76701 [9.76701, 9.76701] (clusters=2) | 1/1 |
| math | gemma3-270m | small:seed41->42 | 3 | 1.44592 [1.44592, 1.44592] (clusters=2) | 1/1 |
| math | gemma3-270m | small:seed41->42 | 4 | -0.195155 [-0.195155, -0.195155] (clusters=2) | 1/1 |
| math | gemma3-4b | large:seed41->42 | 1 | 0.309611 [0.309611, 0.309611] (clusters=2) | 0/1 |
| math | gemma3-4b | large:seed41->42 | 2 | 0.674827 [0.674827, 0.674827] (clusters=2) | 1/1 |
| math | gemma3-4b | large:seed41->42 | 3 | -3.13237 [-3.13237, -3.13237] (clusters=2) | 1/1 |
| math | gemma3-4b | large:seed41->42 | 4 | 0.730342 [0.730342, 0.730342] (clusters=2) | 1/1 |
| math | gemma3-4b | middle:seed41->42 | 1 | 8.2751 [8.2751, 8.2751] (clusters=2) | 1/1 |
| math | gemma3-4b | middle:seed41->42 | 2 | 56.7057 [56.7057, 56.7057] (clusters=2) | 1/1 |
| math | gemma3-4b | middle:seed41->42 | 3 | -0.678361 [-0.678361, -0.678361] (clusters=2) | 1/1 |
| math | gemma3-4b | middle:seed41->42 | 4 | -3.57895 [-3.57895, -3.57895] (clusters=2) | 1/1 |
| math | gemma3-4b | small:seed41->42 | 1 | -0.870405 [-0.870405, -0.870405] (clusters=2) | 1/1 |
| math | gemma3-4b | small:seed41->42 | 2 | -20.9238 [-20.9238, -20.9238] (clusters=2) | 1/1 |
| math | gemma3-4b | small:seed41->42 | 3 | -4.22224 [-4.22224, -4.22224] (clusters=2) | 1/1 |
| math | gemma3-4b | small:seed41->42 | 4 | -21.6764 [-21.6764, -21.6764] (clusters=2) | 1/1 |
| qa | gemma3-1b | large:seed41->42 | 1 | 4.46268 [4.46268, 4.46268] (clusters=2) | 0/1 |
| qa | gemma3-1b | large:seed41->42 | 2 | 9.53318 [9.53318, 9.53318] (clusters=2) | 1/1 |
| qa | gemma3-1b | large:seed41->42 | 3 | 1.20983 [1.20983, 1.20983] (clusters=2) | 1/1 |
| qa | gemma3-1b | large:seed41->42 | 4 | -6.36817 [-6.36817, -6.36817] (clusters=2) | 1/1 |
| qa | gemma3-1b | middle:seed41->42 | 1 | 5.92203 [5.92203, 5.92203] (clusters=2) | 1/1 |
| qa | gemma3-1b | middle:seed41->42 | 2 | -26.5274 [-26.5274, -26.5274] (clusters=2) | 1/1 |
| qa | gemma3-1b | middle:seed41->42 | 3 | -11.262 [-11.262, -11.262] (clusters=2) | 1/1 |
| qa | gemma3-1b | middle:seed41->42 | 4 | -27.8497 [-27.8497, -27.8497] (clusters=2) | 1/1 |
| qa | gemma3-1b | small:seed41->42 | 1 | -6.50457 [-6.50457, -6.50457] (clusters=2) | 1/1 |
| qa | gemma3-1b | small:seed41->42 | 2 | -88.2781 [-88.2781, -88.2781] (clusters=2) | 1/1 |
| qa | gemma3-1b | small:seed41->42 | 3 | -36.7136 [-36.7136, -36.7136] (clusters=2) | 1/1 |
| qa | gemma3-1b | small:seed41->42 | 4 | -97.2049 [-97.2049, -97.2049] (clusters=2) | 1/1 |
| qa | gemma3-270m | large:seed41->42 | 1 | 3.44816 [3.44816, 3.44816] (clusters=2) | 0/1 |
| qa | gemma3-270m | large:seed41->42 | 2 | 3.35307 [3.35307, 3.35307] (clusters=2) | 1/1 |
| qa | gemma3-270m | large:seed41->42 | 3 | 13.8327 [13.8327, 13.8327] (clusters=2) | 1/1 |
| qa | gemma3-270m | large:seed41->42 | 4 | -16.3301 [-16.3301, -16.3301] (clusters=2) | 1/1 |
| qa | gemma3-270m | middle:seed41->42 | 1 | -6.13104 [-6.13104, -6.13104] (clusters=2) | 1/1 |
| qa | gemma3-270m | middle:seed41->42 | 2 | 129.454 [129.454, 129.454] (clusters=2) | 1/1 |
| qa | gemma3-270m | middle:seed41->42 | 3 | 0.791798 [0.791798, 0.791798] (clusters=2) | 1/1 |
| qa | gemma3-270m | middle:seed41->42 | 4 | 0.648075 [0.648075, 0.648075] (clusters=2) | 1/1 |
| qa | gemma3-270m | small:seed41->42 | 1 | -2.72572 [-2.72572, -2.72572] (clusters=2) | 1/1 |
| qa | gemma3-270m | small:seed41->42 | 2 | -29.8954 [-29.8954, -29.8954] (clusters=2) | 1/1 |
| qa | gemma3-270m | small:seed41->42 | 3 | -7.47546 [-7.47546, -7.47546] (clusters=2) | 1/1 |
| qa | gemma3-270m | small:seed41->42 | 4 | -21.351 [-21.351, -21.351] (clusters=2) | 1/1 |
| qa | gemma3-4b | large:seed41->42 | 1 | 3.68617 [3.68617, 3.68617] (clusters=2) | 0/1 |
| qa | gemma3-4b | large:seed41->42 | 2 | 6.58505 [6.58505, 6.58505] (clusters=2) | 1/1 |
| qa | gemma3-4b | large:seed41->42 | 3 | -2.50359 [-2.50359, -2.50359] (clusters=2) | 1/1 |
| qa | gemma3-4b | large:seed41->42 | 4 | -60.746 [-60.746, -60.746] (clusters=2) | 1/1 |
| qa | gemma3-4b | middle:seed41->42 | 1 | 5.13243 [5.13243, 5.13243] (clusters=2) | 1/1 |
| qa | gemma3-4b | middle:seed41->42 | 2 | -539.037 [-539.037, -539.037] (clusters=2) | 1/1 |
| qa | gemma3-4b | middle:seed41->42 | 3 | 48.661 [48.661, 48.661] (clusters=2) | 1/1 |
| qa | gemma3-4b | middle:seed41->42 | 4 | 75.3256 [75.3256, 75.3256] (clusters=2) | 1/1 |
| qa | gemma3-4b | small:seed41->42 | 1 | -5.03949 [-5.03949, -5.03949] (clusters=2) | 1/1 |
| qa | gemma3-4b | small:seed41->42 | 2 | 25.7794 [25.7794, 25.7794] (clusters=2) | 1/1 |
| qa | gemma3-4b | small:seed41->42 | 3 | 9.4505 [9.4505, 9.4505] (clusters=2) | 1/1 |
| qa | gemma3-4b | small:seed41->42 | 4 | -272.153 [-272.153, -272.153] (clusters=2) | 1/1 |


OLS within trajectory-dyad slope versus log(1+(T1+T2)/(2*T_ref)); dyad centering controls student, seed and pool-contrast intercepts. Exactly 5000 whole-trajectory draws; bootstrap sign-tail probability is descriptive, not an exact null p-value. No multiple-comparison claim.

Separately for each joint descriptor, subtract F(T2,D_U2,z)-F(T1,D_U2,z) from the measured effect, divide by the common-T1 bracket, then subtract b+b'*z. This is model-dependent mismatch adjustment, not an independent fixed-T measurement. All F parameters come from the leave-one-student-out training fold.

Slopes pooled across students with separate trajectory-dyad intercepts (individual-student slopes also saved in JSON):

| Kind | Capability | Contrast | Raw ratio slope [95% CI] | Sign-tail probability | Adjusted residual slope: log parameters [95% CI] | Adjusted residual slope: initial loss [95% CI] |
|---|---|---|---|---:|---|---|
| seed | code | large:seed41->42 | -2.36258 [-3.46998, -1.70464] (clusters=6) | 0.00039992 | -9.26373 [-54.8675, 74.6946] (clusters=6) | -8.51541 [-64.5006, 85.8582] (clusters=6) |
| seed | code | middle:seed41->42 | -23.2864 [-72.2901, 33.0993] (clusters=6) | 0.591882 | -0.197108 [-3.50424, 3.87685] (clusters=6) | -0.202354 [-3.48542, 3.83303] (clusters=6) |
| seed | code | small:seed41->42 | -21.8738 [-43.4437, -3.4244] (clusters=6) | 0.00039992 | -15.7479 [-34.607, -4.49054] (clusters=6) | -15.7357 [-34.6199, -4.47523] (clusters=6) |
| seed | math | large:seed41->42 | -0.133327 [-0.683682, 0.737991] (clusters=6) | 0.732254 | 42.2837 [-8.5804, 79.8206] (clusters=6) | 36.7777 [-17.49, 84.2103] (clusters=6) |
| seed | math | middle:seed41->42 | -20.5516 [-37.4682, -11.2213] (clusters=6) | 0.00039992 | 2.50565 [0.128277, 4.63113] (clusters=6) | 2.42305 [0.198137, 4.46041] (clusters=6) |
| seed | math | small:seed41->42 | -8.54004 [-15.2165, -3.58102] (clusters=6) | 0.00039992 | -4.72776 [-12.5147, 2.17832] (clusters=6) | -4.60901 [-12.2719, 2.39338] (clusters=6) |
| seed | qa | large:seed41->42 | -36.8527 [-75.3448, -15.2539] (clusters=6) | 0.00039992 | 384.863 [-162.167, 920.734] (clusters=6) | 408.095 [-191.462, 956.872] (clusters=6) |
| seed | qa | middle:seed41->42 | 86.0099 [-53.0866, 337.907] (clusters=6) | 0.935013 | 17.9043 [-14.7955, 74.3334] (clusters=6) | 16.2521 [-16.0858, 74.8964] (clusters=6) |
| seed | qa | small:seed41->42 | -132.046 [-314.327, -9.79924] (clusters=6) | 0.00039992 | -85.7476 [-195.833, -7.63865] (clusters=6) | -82.802 [-196.928, -0.0842203] (clusters=6) |
| size | code | middle->large | 0.152543 [0.0766007, 0.225732] (clusters=12) | 0.00039992 | 0.146761 [0.0760451, 0.21087] (clusters=12) | 0.146874 [0.0765801, 0.216055] (clusters=12) |
| size | code | small->large | 0.172633 [0.083168, 0.278799] (clusters=12) | 0.00039992 | 0.171457 [0.0825302, 0.279675] (clusters=12) | 0.171481 [0.0816601, 0.279783] (clusters=12) |
| size | code | small->middle | 0.200656 [0.0821204, 0.381885] (clusters=12) | 0.00039992 | 0.201325 [0.0830044, 0.3802] (clusters=12) | 0.201342 [0.083488, 0.380222] (clusters=12) |
| size | math | middle->large | 0.123935 [0.0600476, 0.246585] (clusters=12) | 0.00039992 | 0.116729 [0.0637716, 0.224617] (clusters=12) | 0.116023 [0.0654206, 0.219142] (clusters=12) |
| size | math | small->large | 0.207171 [0.0916955, 0.356339] (clusters=12) | 0.00039992 | 0.205888 [0.0916162, 0.355709] (clusters=12) | 0.205767 [0.0918935, 0.356068] (clusters=12) |
| size | math | small->middle | 0.281163 [0.114241, 0.526018] (clusters=12) | 0.00039992 | 0.281629 [0.114187, 0.524713] (clusters=12) | 0.281977 [0.114202, 0.52472] (clusters=12) |
| size | qa | middle->large | 0.943657 [0.298197, 2.21542] (clusters=12) | 0.00039992 | 0.964331 [0.314482, 2.31256] (clusters=12) | 0.969472 [0.309388, 2.2981] (clusters=12) |
| size | qa | small->large | 2.57689 [0.928049, 4.63131] (clusters=12) | 0.00039992 | 2.57685 [0.917008, 4.60743] (clusters=12) | 2.5783 [0.917923, 4.60777] (clusters=12) |
| size | qa | small->middle | 3.92145 [1.43778, 7.11221] (clusters=12) | 0.00039992 | 3.91427 [1.43569, 7.09796] (clusters=12) | 3.92055 [1.43473, 7.09863] (clusters=12) |

Exact fixed-T support/falsification cannot be identified from these saved checkpoints. A nonzero observed-endpoint slope diagnoses drift plus possible budget/composition/schedule confounding; a CI containing zero does not establish flatness.

Conditional interaction gate: Before extension fitting: raw and both descriptor-adjusted residual slope intervals exclude zero in one common direction for all three size contrasts, both with all ordinal checkpoints and with only registered near-matches; slope signs must replicate separately in each pool seed. This tests stability of an observed residual, not exact fixed-T causality.

No nonzero-budget exact-T pool pairs and no recorded domain shares. A stable observed residual permits the requested one-term descriptive refit, with the exact-T limitation retained. Its post-diagnostic selection cannot revise the frozen four-coefficient registered decisions.

Observed-residual gate, math: pooled and eligible-only intervals agree=True; direction replicates in each seed=True.
Observed-residual gate, code: pooled and eligible-only intervals agree=True; direction replicates in each seed=True.
Observed-residual gate, qa: pooled and eligible-only intervals agree=False; direction replicates in each seed=True.

The observed-residual gate passed for code, math. Refit only F_joint + k*log(1+T/T_ref)*log(1+E) (five coefficients per capability).

Conditional exploratory refit after full-data residual diagnostics. Each F is fit on its training fold only, but gate selection used the diagnostic data; these comparisons cannot change the original registered decisions or establish exact fixed-T falsification.

Residual before/after comparison uses identical registered near-matched endpoints; both descriptors and size/seed targets remain separate.

| Kind | Capability | Contrast | Descriptor | Before residual slope [95% CI] | After residual slope [95% CI] | Interpretation |
|---|---|---|---|---|---|---|
| seed | code | large:seed41->42 | log_parameters | 13.2677 [-10.2904, 33.8904] (clusters=6) | 13.363 [-10.2692, 33.9803] (clusters=6) | Residual interval includes zero; this does not prove flatness |
| seed | code | large:seed41->42 | initial_loss | 13.7193 [-9.77992, 41.1671] (clusters=6) | 13.8146 [-9.75876, 41.257] (clusters=6) | Residual interval includes zero; this does not prove flatness |
| seed | code | middle:seed41->42 | log_parameters | -0.197108 [-3.50424, 3.87685] (clusters=6) | -0.186951 [-3.49466, 3.89549] (clusters=6) | Residual interval includes zero; this does not prove flatness |
| seed | code | middle:seed41->42 | initial_loss | -0.202354 [-3.48542, 3.83303] (clusters=6) | -0.192197 [-3.47584, 3.85166] (clusters=6) | Residual interval includes zero; this does not prove flatness |
| seed | code | small:seed41->42 | log_parameters | -15.7479 [-34.607, -4.49054] (clusters=6) | -15.7743 [-34.6129, -4.51541] (clusters=6) | Drift remains detectable |
| seed | code | small:seed41->42 | initial_loss | -15.7357 [-34.6199, -4.47523] (clusters=6) | -15.762 [-34.6258, -4.50009] (clusters=6) | Drift remains detectable |
| seed | math | large:seed41->42 | log_parameters | 39.7856 [34.5387, 46.8264] (clusters=6) | 39.8735 [34.585, 46.8161] (clusters=6) | Drift remains detectable |
| seed | math | large:seed41->42 | initial_loss | 35.7894 [31.4515, 38.1698] (clusters=6) | 35.8773 [31.6791, 38.1594] (clusters=6) | Drift remains detectable |
| seed | math | middle:seed41->42 | log_parameters | 2.50565 [0.128277, 4.63113] (clusters=6) | 2.51502 [0.133215, 4.63002] (clusters=6) | Drift remains detectable |
| seed | math | middle:seed41->42 | initial_loss | 2.42305 [0.198137, 4.46041] (clusters=6) | 2.43241 [0.203074, 4.4593] (clusters=6) | Drift remains detectable |
| seed | math | small:seed41->42 | log_parameters | -4.72776 [-12.5147, 2.17832] (clusters=6) | -4.75207 [-12.5118, 2.11536] (clusters=6) | Residual interval includes zero; this does not prove flatness |
| seed | math | small:seed41->42 | initial_loss | -4.60901 [-12.2719, 2.39338] (clusters=6) | -4.63331 [-12.2691, 2.33041] (clusters=6) | Residual interval includes zero; this does not prove flatness |
| size | code | middle->large | log_parameters | 0.128828 [0.0685279, 0.199516] (clusters=12) | 0.106348 [0.0274269, 0.18049] (clusters=12) | Drift remains detectable |
| size | code | middle->large | initial_loss | 0.128811 [0.0694435, 0.197298] (clusters=12) | 0.106331 [0.0282525, 0.1784] (clusters=12) | Drift remains detectable |
| size | code | small->large | log_parameters | 0.171457 [0.0825302, 0.279675] (clusters=12) | 0.149111 [0.0414331, 0.273016] (clusters=12) | Drift remains detectable |
| size | code | small->large | initial_loss | 0.171481 [0.0816601, 0.279783] (clusters=12) | 0.149136 [0.040563, 0.273124] (clusters=12) | Drift remains detectable |
| size | code | small->middle | log_parameters | 0.201325 [0.0830044, 0.3802] (clusters=12) | 0.178875 [0.047257, 0.372522] (clusters=12) | Drift remains detectable |
| size | code | small->middle | initial_loss | 0.201342 [0.083488, 0.380222] (clusters=12) | 0.178892 [0.0476996, 0.372535] (clusters=12) | Drift remains detectable |
| size | math | middle->large | log_parameters | 0.0789767 [0.0358319, 0.149479] (clusters=12) | 0.0582466 [-0.0170784, 0.150585] (clusters=12) | Residual interval includes zero; this does not prove flatness |
| size | math | middle->large | initial_loss | 0.0790862 [0.0331425, 0.145065] (clusters=12) | 0.0583562 [-0.0189364, 0.146171] (clusters=12) | Residual interval includes zero; this does not prove flatness |
| size | math | small->large | log_parameters | 0.205888 [0.0916162, 0.355709] (clusters=12) | 0.185281 [0.0383408, 0.358135] (clusters=12) | Drift remains detectable |
| size | math | small->large | initial_loss | 0.205767 [0.0918935, 0.356068] (clusters=12) | 0.185161 [0.038618, 0.358493] (clusters=12) | Drift remains detectable |
| size | math | small->middle | log_parameters | 0.281629 [0.114187, 0.524713] (clusters=12) | 0.260926 [0.0605332, 0.527059] (clusters=12) | Drift remains detectable |
| size | math | small->middle | initial_loss | 0.281977 [0.114202, 0.52472] (clusters=12) | 0.261274 [0.0607538, 0.527075] (clusters=12) | Drift remains detectable |

Conditional extension evaluation, descriptor=log_parameters; pooled held-out predictions

| Split | Capability | Target | MAE [95% CI] | Gain vs four-term joint [95% CI] | Gain vs zero [95% CI] | Gain vs training mean [95% CI] |
|---|---|---|---|---|---|---|
| leave_one_pool_seed_out | code | I1 | 0.0400488 [0.0330509, 0.0482297] (clusters=18) | -0.00391979 [-0.00605174, -0.0018728] (clusters=18) | 0.0189928 [0.00473824, 0.033731] (clusters=18) | -0.00346356 [-0.0092142, 0.00243142] (clusters=18) |
| leave_one_pool_seed_out | code | I2_size | 0.0639552 [0.0398022, 0.0898814] (clusters=18) | -0.000749083 [-0.00199753, 0.000508187] (clusters=18) | 0.00736249 [-0.0173271, 0.0272633] (clusters=18) | 0.00517502 [-0.00925883, 0.0142568] (clusters=18) |
| leave_one_pool_seed_out | code | response | 0.0441127 [0.0346295, 0.0548727] (clusters=18) | -0.00140375 [-0.00314446, 0.000276961] (clusters=18) | 0.0965935 [0.0776801, 0.115914] (clusters=18) | 0.0209542 [0.00709458, 0.0346525] (clusters=18) |
| leave_one_pool_seed_out | math | I1 | 0.0423775 [0.0326916, 0.0546882] (clusters=18) | 1.12786e-06 [-0.00218051, 0.00253756] (clusters=18) | 0.00248565 [-0.0109581, 0.0164453] (clusters=18) | 0.00274564 [-0.00559112, 0.0114373] (clusters=18) |
| leave_one_pool_seed_out | math | I2_size | 0.0572885 [0.0284686, 0.0899292] (clusters=18) | 0.00327324 [0.000671724, 0.0057793] (clusters=18) | 0.0114423 [-0.00746968, 0.0288908] (clusters=18) | 0.0183238 [0.00660123, 0.028518] (clusters=18) |
| leave_one_pool_seed_out | math | response | 0.0444134 [0.0368837, 0.0530156] (clusters=18) | -0.000467152 [-0.00243169, 0.00177558] (clusters=18) | 0.0835948 [0.063043, 0.107846] (clusters=18) | 0.0136527 [-0.00248507, 0.0303998] (clusters=18) |
| leave_one_student_out | code | I1 | 0.0361799 [0.0296193, 0.0436651] (clusters=18) | -0.000223861 [-0.00267024, 0.00239401] (clusters=18) | 0.0228618 [0.00847673, 0.038054] (clusters=18) | -1.15919e-05 [-0.0054833, 0.00569157] (clusters=18) |
| leave_one_student_out | code | I2_seed | 0.0323819 [0.0101765, 0.0747637] (clusters=18) | -2.85748e-05 [-0.000111405, 1.26835e-05] (clusters=18) | -0.000221844 [-0.000906639, 0.000613983] (clusters=18) | 0.00194492 [-0.0130392, 0.0186889] (clusters=18) |
| leave_one_student_out | code | I2_size | 0.0553239 [0.0364769, 0.078468] (clusters=18) | 0.00167074 [-0.000674533, 0.00428325] (clusters=18) | 0.0159938 [-0.00585027, 0.0360095] (clusters=18) | 0.0118199 [-0.00499133, 0.0242311] (clusters=18) |
| leave_one_student_out | code | response | 0.0572226 [0.0482492, 0.0670144] (clusters=18) | -0.000275355 [-0.00228418, 0.00177759] (clusters=18) | 0.0834836 [0.0606569, 0.108089] (clusters=18) | 0.00944819 [-0.0127194, 0.0309051] (clusters=18) |
| leave_one_student_out | math | I1 | 0.0393247 [0.0274694, 0.0544447] (clusters=18) | 0.000333637 [-0.00197938, 0.00299923] (clusters=18) | 0.00553843 [-0.00487009, 0.0167499] (clusters=18) | 0.00716825 [-0.00125451, 0.0156007] (clusters=18) |
| leave_one_student_out | math | I2_seed | 0.0178613 [0.00631116, 0.0395051] (clusters=18) | -5.32084e-06 [-0.000119319, 7.68707e-05] (clusters=18) | 8.5083e-05 [-0.000725375, 0.00109792] (clusters=18) | 0.00308793 [-0.00586418, 0.0119598] (clusters=18) |
| leave_one_student_out | math | I2_size | 0.0716294 [0.045131, 0.0996884] (clusters=18) | 0.000954432 [-0.000306887, 0.00222341] (clusters=18) | -0.00289862 [-0.0245417, 0.0187826] (clusters=18) | 0.00769373 [-0.00891857, 0.0244539] (clusters=18) |
| leave_one_student_out | math | response | 0.0689127 [0.05343, 0.0856507] (clusters=18) | -0.00194545 [-0.00375499, -0.000260952] (clusters=18) | 0.0590954 [0.0341153, 0.0859121] (clusters=18) | 0.00129128 [-0.0207503, 0.0220234] (clusters=18) |

Conditional extension evaluation, descriptor=initial_loss; pooled held-out predictions

| Split | Capability | Target | MAE [95% CI] | Gain vs four-term joint [95% CI] | Gain vs zero [95% CI] | Gain vs training mean [95% CI] |
|---|---|---|---|---|---|---|
| leave_one_pool_seed_out | code | I1 | 0.0399079 [0.0327158, 0.0482702] (clusters=18) | -0.00390506 [-0.00593589, -0.00193023] (clusters=18) | 0.0191337 [0.0046437, 0.0340796] (clusters=18) | -0.00332269 [-0.0090419, 0.00257788] (clusters=18) |
| leave_one_pool_seed_out | code | I2_size | 0.0639554 [0.0403238, 0.0884824] (clusters=18) | -0.000589466 [-0.0019927, 0.00129849] (clusters=18) | 0.00736231 [-0.0165964, 0.0268332] (clusters=18) | 0.00517483 [-0.0101849, 0.0140465] (clusters=18) |
| leave_one_pool_seed_out | code | response | 0.0434726 [0.0342678, 0.0537262] (clusters=18) | -0.00103747 [-0.00290682, 0.000706443] (clusters=18) | 0.0972335 [0.078516, 0.116689] (clusters=18) | 0.0215943 [0.00793656, 0.0354525] (clusters=18) |
| leave_one_pool_seed_out | math | I1 | 0.0431146 [0.0330384, 0.0551161] (clusters=18) | -0.000202673 [-0.00242466, 0.00239189] (clusters=18) | 0.00174858 [-0.0120892, 0.0157045] (clusters=18) | 0.00200857 [-0.0060978, 0.0107836] (clusters=18) |
| leave_one_pool_seed_out | math | I2_size | 0.0563242 [0.0292847, 0.0901129] (clusters=18) | 0.00291913 [0.000192442, 0.00552121] (clusters=18) | 0.0124066 [-0.0046148, 0.0293815] (clusters=18) | 0.0192881 [0.00778311, 0.0281069] (clusters=18) |
| leave_one_pool_seed_out | math | response | 0.0411869 [0.0331274, 0.0501242] (clusters=18) | -0.000482782 [-0.00291361, 0.00243428] (clusters=18) | 0.0868212 [0.0659312, 0.111329] (clusters=18) | 0.0168791 [-0.000599753, 0.0349919] (clusters=18) |
| leave_one_student_out | code | I1 | 0.0361186 [0.0291458, 0.044022] (clusters=18) | -0.000645411 [-0.00290335, 0.00157884] (clusters=18) | 0.022923 [0.00939401, 0.0371287] (clusters=18) | 4.96415e-05 [-0.00519583, 0.00560305] (clusters=18) |
| leave_one_student_out | code | I2_seed | 0.0324046 [0.0101135, 0.0748589] (clusters=18) | -2.85748e-05 [-0.000111405, 1.26835e-05] (clusters=18) | -0.000244494 [-0.00101556, 0.00064229] (clusters=18) | 0.00192227 [-0.0131597, 0.0188261] (clusters=18) |
| leave_one_student_out | code | I2_size | 0.0595558 [0.0370365, 0.0885972] (clusters=18) | 0.00164558 [-0.000868825, 0.0047304] (clusters=18) | 0.0117619 [-0.0129004, 0.0332655] (clusters=18) | 0.00758798 [-0.0124122, 0.0211523] (clusters=18) |
| leave_one_student_out | code | response | 0.0546304 [0.0433418, 0.0675291] (clusters=18) | -0.000486066 [-0.00296068, 0.00244331] (clusters=18) | 0.0860757 [0.0616705, 0.111425] (clusters=18) | 0.0120404 [-0.0127268, 0.0341527] (clusters=18) |
| leave_one_student_out | math | I1 | 0.0474449 [0.0366041, 0.059437] (clusters=18) | -0.00257575 [-0.00496697, -0.000366336] (clusters=18) | -0.00258171 [-0.0168841, 0.0122083] (clusters=18) | -0.000951897 [-0.0107513, 0.00910385] (clusters=18) |
| leave_one_student_out | math | I2_seed | 0.0179937 [0.00656586, 0.0400536] (clusters=18) | -5.32084e-06 [-0.000119319, 7.68707e-05] (clusters=18) | -4.73147e-05 [-0.00113785, 0.00106048] (clusters=18) | 0.00295553 [-0.00592867, 0.0119951] (clusters=18) |
| leave_one_student_out | math | I2_size | 0.0604376 [0.0300264, 0.0988903] (clusters=18) | 0.00153851 [-0.000863229, 0.00508277] (clusters=18) | 0.00829313 [-0.00950179, 0.0225357] (clusters=18) | 0.0188855 [-5.36108e-05, 0.0321326] (clusters=18) |
| leave_one_student_out | math | response | 0.0481401 [0.0383197, 0.0589565] (clusters=18) | -0.00322909 [-0.00596646, -0.000803901] (clusters=18) | 0.079868 [0.0605192, 0.103565] (clusters=18) | 0.0220639 [0.00425074, 0.0409963] (clusters=18) |

Data limitations are explicit: requested per-domain token shares and the exact fixed-T measured falsification statistic cannot be recovered from the allowed saved accounting. No new training, inference, tokenization, external trajectory, or interpolated measurement was used.
