#!/bin/bash
#SBATCH --job-name=r1-acc-olmo2
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# E5 (rebuttal, pre-registered notes/rebuttal/prereg_r1_accuracy.md): OLMo-2 accuracy-endpoint measurement efficiency.
# No PRC/Qwen model is involved. Dense development pilots first, then every state.
set -euo pipefail
PROJ=/blue/fsu-compsci-dept/xc25.fsu/hq/scaling-down-law
cd "$PROJ"
export HF_HOME=$PROJ/hf-cache HF_HUB_CACHE=$PROJ/hf-cache/hub HF_DATASETS_CACHE=$PROJ/hf-cache/datasets
export XDG_CACHE_HOME=$PROJ/.cache TMPDIR=$PROJ/tmp
mkdir -p "$HF_HOME/datasets" "$HF_HOME/hub" "$XDG_CACHE_HOME" "$TMPDIR" logs results/r1-accuracy-efficiency/measurements
source .venv-tf5/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} PYTHONUNBUFFERED=1
echo "=== R1 accuracy node $(hostname) $(date) ==="
for s in olmo2-1b@399B olmo2-1b@1993B olmo2-1b@3608B olmo2-7b@391B olmo2-7b@1947B olmo2-7b@3507B; do
  python -B analysis/r1_accuracy_efficiency.py --measure --state "$s" --dense-only --batch-size 32 --device cuda:0
done
for s in olmo2-1b@399B olmo2-1b@1993B olmo2-1b@3608B olmo2-1b@1196B olmo2-1b@2811B olmo2-7b@391B olmo2-7b@1947B olmo2-7b@3507B olmo2-7b@1167B olmo2-7b@2727B; do
  python -B analysis/r1_accuracy_efficiency.py --measure --state "$s" --batch-size 32 --device cuda:0
done
echo "=== DONE $(date) ==="
