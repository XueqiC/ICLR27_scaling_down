#!/bin/bash
# After both V70 lanes exit, run V71 (QA scope: 8 states x 3 sets x 256) on the Ada by UUID, then the V70 compare on CPU.
cd .
while kill -0 3356758 2>/dev/null || kill -0 3357480 2>/dev/null; do sleep 60; done
echo "[V70 $(date +%H:%M:%S)] both lanes exited; running compare" >> logs/v5_author_chain.log
python3 analysis/v70_distill_confirm.py compare > logs/v70_compare.log 2>&1; echo "[V70 $(date +%H:%M:%S)] compare rc=$?" >> logs/v5_author_chain.log
echo "[V71 $(date +%H:%M:%S)] start on Ada" >> logs/v5_author_chain.log
V67_ALLOW_ONLINE=1 CUDA_VISIBLE_DEVICES=GPU-20b20454-ae9f-7860-6801-430d68842a27 python3 analysis/v71_qa_scope.py > logs/v71_qa_scope.log 2>&1
echo "[V71 $(date +%H:%M:%S)] rc=$?" >> logs/v5_author_chain.log
