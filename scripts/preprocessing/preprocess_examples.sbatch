#!/bin/bash
#SBATCH --job-name=nucaliby_cache
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-99
#SBATCH --output=slurm_logs/preprocess_examples_%A_%a.out
#SBATCH --error=slurm_logs/preprocess_examples_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
: "${NUCALIBY_SOLUBLE_DATA_DIR:?set NUCALIBY_SOLUBLE_DATA_DIR to the output store}"
mkdir -p slurm_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHONPATH=. python -m nucaliby.preprocessing.preprocess_examples \
    out_dir="$NUCALIBY_SOLUBLE_DATA_DIR" \
    num_workers=30 \
    num_shards=100 \
    shard_id=${SLURM_ARRAY_TASK_ID}
