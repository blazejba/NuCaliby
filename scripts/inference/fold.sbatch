#!/bin/bash
#SBATCH --job-name=nucaliby_fold
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/fold_%j.out
#SBATCH --error=logs/fold_%j.err
#
# Fold designs with ColabFold. Arguments are forwarded to scripts/inference/fold.py.
#
#   sbatch scripts/inference/fold.sbatch --designs results/nt/designs.csv --out results/nt/fold --skip-existing
#
# Never run two of these against the same --out concurrently: they share the
# FASTA list that colabfold_batch reads.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs
PYTHONPATH=. python scripts/inference/fold.py "$@"
