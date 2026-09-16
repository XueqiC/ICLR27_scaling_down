#!/usr/bin/env bash
# Supplied quant64k.slurm conventions; submit from $CLUSTER_PROJECT/sdl.
#SBATCH --job-name=a11-prune
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=$CLUSTER_PROJECT/sdl/a11-prune-%j.out
#SBATCH --error=$CLUSTER_PROJECT/sdl/a11-prune-%j.err

set -euo pipefail
export HF_HOME=$CLUSTER_PROJECT/hf-cache
export PYTHONPATH=$CLUSTER_PROJECT/sdl
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
cd $CLUSTER_PROJECT/sdl

# Honor an explicit cluster interpreter, otherwise use the existing project venv
# when present, or the Python environment inherited by sbatch.
if [[ -n "${A11_PYTHON:-}" ]]; then
  a11_python="$A11_PYTHON"
elif [[ -x $CLUSTER_PROJECT/sdl/.venv/bin/python ]]; then
  a11_python=$CLUSTER_PROJECT/sdl/.venv/bin/python
else
  a11_python=python3
fi

# Keep Slurm's CUDA_VISIBLE_DEVICES; cuda:0 is the one GPU allocated to this job.
# Checkpoint weights/tokenizers/configs resolve via HF_HOME at the exact step
# revisions. Probes are local frozen files; no dataset downloads are needed.
"$a11_python" -B analysis/a11_score.py --verify-only
for state in pythia-160m@step80000 pythia-410m@step112000 \
             pythia-1.4b@step48000 pythia-1b@step48000; do
  "$a11_python" -B analysis/a11_measure.py --state "$state" \
    --device cuda:0 --model-dtype bf16 --n-probe 128 \
    --probe-seed 0 --pruning-seed 0 --densities 0.9,0.85,0.8,0.75,0.7,0.65
done
"$a11_python" -B analysis/a11_score.py
