#!/bin/bash
#SBATCH --job-name=sdl-v10s
#SBATCH --partition=the cluster-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=230gb
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# Usage: sbatch -A <account> -q <qos> --export=ALL,MODEL=gemma3-12b scripts/job_cluster_v9.slurm
set -e
MODEL=${MODEL:-gemma3-4b}
MODEL_LC=${MODEL,,}
if [[ "$MODEL_LC" =~ qwen|deepseek|glm|yi-|baichuan|internlm|minimax|moonshot|kimi ]]; then
    echo "ERROR: refusing restricted-origin model '$MODEL' under the shared cluster policy." >&2
    exit 1
fi
PROJ=$CLUSTER_PROJECT/scaling-down-law
cd "$PROJ"
mkdir -p logs
if [[ "$MODEL_LC" =~ gemma4|muse ]]; then
    export VENV_FLAVOR=tf5; VENV=.venv-tf5
else
    export VENV_FLAVOR=main; VENV=.venv
fi
bash scripts/cluster_bootstrap_env.sh
source "$VENV/bin/activate"
export HF_HOME=$PROJ/hf-cache
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-14}
echo "=== V9 on $MODEL, node $(hostname) ==="
BITS_ARG=$(echo "${BITSU:-8_6_4_3}" | tr _ ,)
python analysis/v10_quantization.py \
    --model "$MODEL" --device cuda:0 --model-dtype bf16 \
    --n-probe ${NPROBE:-128} --bits "$BITS_ARG" ${OUTBASE:+--output-base "$OUTBASE"}
echo "=== DONE ==="
