# V67 MuSiQue QA

Completed 34/34 evaluations. Losses are native-token nats; delta = checkpoint minus the student's dense loss (negative is improvement).
MuSiQue answerable dev: 128 references, probe seed 0, supporting context, V6 QA template and 1024-token limit (512 per span), batch size 1.
2Wiki primary QA (64 measurement examples) is read from checkpoint eval.json; dense uses that file's dense.qa. No accuracy or generated references.
T = eval.json completion_tokens_seen; E = T / registered D_U_completion (V50 convention). Raw processed-token exposure is retained in measurements.json.
Trajectory snapshots from the same run are dependent measurements, not independent seeds.

| Student | Run | Checkpoint | T | E | MuSiQue | Δ MuSiQue | 2Wiki | Δ 2Wiki | Tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemma3-270m | dense | dense | 0 | 0.0000 | 2.019008 | +0.000000 | 5.274807 | +0.000000 | 582 |
| gemma3-270m | gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | update-00000199 | 283327 | 2.8474 | 2.173191 | +0.154183 | 4.340575 | -0.934232 | 582 |
| gemma3-270m | gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | update-00000200 | 285998 | 2.8778 | 2.099059 | +0.080051 | 4.599477 | -0.675330 | 582 |
| gemma3-270m | gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | update-00000197 | 289585 | 2.8246 | 2.175600 | +0.156592 | 4.602029 | -0.672778 | 582 |
| gemma3-270m | gpt-5.6-luna_full_450_p2v2_lora_dseed11 | update-00000199 | 280329 | 2.3599 | 2.189601 | +0.170593 | 4.624626 | -0.650181 | 582 |
| gemma3-270m | gpt-5.6-luna_full_450_p2v2_lora_dseed12 | update-00000199 | 280508 | 2.3803 | 2.110717 | +0.091710 | 4.615662 | -0.659145 | 582 |
| gemma3-270m | gpt-5.6-luna_full_450_p2v2_lora_dseed13 | update-00000198 | 282988 | 2.3543 | 2.205246 | +0.186238 | 4.595493 | -0.679314 | 582 |
| gemma3-270m | gpt-5.6-luna_full_75_p2v2_lora_dseed11 | update-00000196 | 280140 | 14.0000 | 3.743080 | +1.724072 | 7.825635 | +2.550828 | 582 |
| gemma3-270m | gpt-5.6-luna_full_75_p2v2_lora_dseed12 | update-00000200 | 277507 | 14.2912 | 3.714320 | +1.695312 | 8.627988 | +3.353181 | 582 |
| gemma3-270m | gpt-5.6-luna_full_75_p2v2_lora_dseed13 | update-00000197 | 288977 | 14.0608 | 3.360633 | +1.341626 | 7.249875 | +1.975068 | 582 |
| gemma3-270m | gpt-5.6-luna_full_75_p2v2rep_lora_dseed11_seed1 | update-00000196 | 280140 | 14.0000 | 3.908596 | +1.889588 | 8.011703 | +2.736896 | 582 |
| gemma3-1b | dense | dense | 0 | 0.0000 | 1.615832 | +0.000000 | 5.567791 | +0.000000 | 582 |
| gemma3-1b | gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | update-00000199 | 283327 | 2.8474 | 2.181084 | +0.565252 | 4.967131 | -0.600660 | 582 |
| gemma3-1b | gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | update-00000200 | 285998 | 2.8778 | 2.134289 | +0.518457 | 4.996825 | -0.570966 | 582 |
| gemma3-1b | gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | update-00000197 | 289585 | 2.8246 | 2.207535 | +0.591703 | 5.060601 | -0.507190 | 582 |
| gemma3-1b | gpt-5.6-luna_full_450_p2v2_lora_dseed11 | update-00000199 | 280329 | 2.3599 | 1.852177 | +0.236345 | 4.849633 | -0.718159 | 582 |
| gemma3-1b | gpt-5.6-luna_full_450_p2v2_lora_dseed12 | update-00000199 | 280508 | 2.3803 | 2.061983 | +0.446151 | 4.975566 | -0.592225 | 582 |
| gemma3-1b | gpt-5.6-luna_full_450_p2v2_lora_dseed13 | update-00000198 | 282988 | 2.3543 | 1.989456 | +0.373624 | 4.990662 | -0.577129 | 582 |
| gemma3-1b | gpt-5.6-luna_full_450_p2v2rep_lora_dseed11_seed1 | update-00000199 | 280329 | 2.3599 | 1.840428 | +0.224596 | 4.938683 | -0.629109 | 582 |
| gemma3-1b | gpt-5.6-luna_full_75_p2v2_lora_dseed11 | update-00000196 | 280140 | 14.0000 | 4.983351 | +3.367520 | 10.170443 | +4.602652 | 582 |
| gemma3-1b | gpt-5.6-luna_full_75_p2v2_lora_dseed12 | update-00000200 | 277507 | 14.2912 | 5.009668 | +3.393836 | 10.377615 | +4.809823 | 582 |
| gemma3-1b | gpt-5.6-luna_full_75_p2v2_lora_dseed13 | update-00000197 | 288977 | 14.0608 | 4.871212 | +3.255380 | 10.841260 | +5.273469 | 582 |
| gemma3-1b | gpt-5.6-luna_full_75_p2v2_lora_dseed11 | update-00000025 | 36542 | 1.8262 | 1.782015 | +0.166183 | 4.238733 | -1.329059 | 582 |
| gemma3-1b | gpt-5.6-luna_full_75_p2v2_lora_dseed11 | update-00000049 | 69387 | 3.4676 | 2.062916 | +0.447084 | 4.882283 | -0.685508 | 582 |
| gemma3-1b | gpt-5.6-luna_full_450_p2v2_lora_dseed11 | update-00000026 | 36203 | 0.3048 | 1.879551 | +0.263719 | 4.105578 | -1.462214 | 582 |
| gemma3-1b | gpt-5.6-luna_full_450_p2v2_lora_dseed11 | update-00000050 | 69539 | 0.5854 | 2.019545 | +0.403713 | 4.165214 | -1.402577 | 582 |
| gemma3-4b | dense | dense | 0 | 0.0000 | 1.485761 | +0.000000 | 5.477154 | +0.000000 | 582 |
| gemma3-4b | gpt-5.6-luna_full_375_p2v2test_lora_dseed21 | update-00000199 | 283327 | 2.8474 | 2.414532 | +0.928771 | 4.880167 | -0.596987 | 582 |
| gemma3-4b | gpt-5.6-luna_full_375_p2v2test_lora_dseed22 | update-00000200 | 285998 | 2.8778 | 2.433607 | +0.947846 | 5.293202 | -0.183952 | 582 |
| gemma3-4b | gpt-5.6-luna_full_375_p2v2test_lora_dseed23 | update-00000197 | 289585 | 2.8246 | 2.131349 | +0.645588 | 5.051295 | -0.425859 | 582 |
| gemma3-4b | gpt-5.6-luna_full_450_p2v2test_lora_dseed11 | update-00000199 | 280329 | 2.3599 | 1.889835 | +0.404074 | 4.659861 | -0.817293 | 582 |
| gemma3-4b | gpt-5.6-luna_full_450_p2v2test_lora_dseed12 | update-00000199 | 280508 | 2.3803 | 2.195319 | +0.709558 | 5.182395 | -0.294758 | 582 |
| gemma3-4b | gpt-5.6-luna_full_75_p2v2test_lora_dseed11 | update-00000196 | 280140 | 14.0000 | 5.171965 | +3.686204 | 12.207918 | +6.730764 | 582 |
| gemma3-4b | gpt-5.6-luna_full_75_p2v2test_lora_dseed12 | update-00000200 | 277507 | 14.2912 | 4.611716 | +3.125955 | 10.354084 | +4.876930 | 582 |
