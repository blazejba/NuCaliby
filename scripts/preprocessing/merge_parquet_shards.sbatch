#!/bin/bash
#SBATCH --job-name=nucaliby_merge
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/merge_metadata_%j.out
#SBATCH --error=slurm_logs/merge_metadata_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
: "${NUCALIBY_SOLUBLE_DATA_DIR:?set NUCALIBY_SOLUBLE_DATA_DIR to the output store}"
mkdir -p slurm_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHONPATH=. python -m nucaliby.preprocessing.merge_parquet_shards \
    --out_dir "$NUCALIBY_SOLUBLE_DATA_DIR"
