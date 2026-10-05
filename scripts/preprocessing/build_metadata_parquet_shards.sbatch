#!/bin/bash
#SBATCH --job-name=nucaliby_metadata
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-99
#SBATCH --output=slurm_logs/build_metadata_%A_%a.out
#SBATCH --error=slurm_logs/build_metadata_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
: "${PDB_MIRROR_PATH:?set PDB_MIRROR_PATH to the RCSB mmCIF mirror}"
: "${NUCALIBY_SOLUBLE_DATA_DIR:?set NUCALIBY_SOLUBLE_DATA_DIR to the output store}"
mkdir -p slurm_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHONPATH=. python -m nucaliby.preprocessing.build_metadata_parquet_shards \
    mmcif_dir="$PDB_MIRROR_PATH" \
    out_dir="$NUCALIBY_SOLUBLE_DATA_DIR" \
    num_workers=30 \
    num_shards=100 \
    shard_id=${SLURM_ARRAY_TASK_ID}
