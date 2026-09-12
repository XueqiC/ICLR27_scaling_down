#!/bin/bash
# After the V71 waiter (PID 3512566) finishes, run V72 on the Ada: dense -> freeze (CPU) -> pruned cells -> compare.
cd .
while kill -0 3512566 2>/dev/null; do sleep 60; done
ADA=GPU-20b20454-ae9f-7860-6801-430d68842a27
echo "[V72 $(date +%H:%M:%S)] dense start" >> logs/v5_author_chain.log
CUDA_VISIBLE_DEVICES=$ADA bash scripts/run_v72_measure.sh > logs/v72_dense.log 2>&1; echo "[V72 $(date +%H:%M:%S)] dense rc=$?" >> logs/v5_author_chain.log
python3 analysis/v72_prune_repeat.py freeze > logs/v72_freeze.log 2>&1; echo "[V72 $(date +%H:%M:%S)] freeze rc=$?" >> logs/v5_author_chain.log
( cd paper && mkdir -p data_mirror/v72-prune-repeat && cp ../results/v72-prune-repeat/freeze.json data_mirror/v72-prune-repeat/ 2>/dev/null && git add data_mirror/v72-prune-repeat && git commit -q -m "V72 pruning repeat: frozen predictions for 2.8B@16k/143k x d 0.85/0.75/0.65 (before pruned measurement)

" && git push -q origin main && echo "[V72 $(date +%H:%M:%S)] freeze committed $(git rev-parse --short HEAD)" >> ../logs/v5_author_chain.log )
CUDA_VISIBLE_DEVICES=$ADA bash scripts/run_v72_measure.sh > logs/v72_measure.log 2>&1; echo "[V72 $(date +%H:%M:%S)] measure rc=$?" >> logs/v5_author_chain.log
python3 analysis/v72_prune_repeat.py compare > logs/v72_compare.log 2>&1; echo "[V72 $(date +%H:%M:%S)] compare rc=$?" >> logs/v5_author_chain.log
