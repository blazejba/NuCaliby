#!/bin/bash
#SBATCH --job-name=nucaliby_cluster
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/cluster_sequences_%j.out
#SBATCH --error=slurm_logs/cluster_sequences_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
: "${NUCALIBY_SOLUBLE_DATA_DIR:?set NUCALIBY_SOLUBLE_DATA_DIR to the output store}"
mkdir -p slurm_logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHONPATH=. python -m nucaliby.preprocessing.cluster_sequences \
    pdb_path="$NUCALIBY_SOLUBLE_DATA_DIR"
