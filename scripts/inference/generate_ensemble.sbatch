#!/bin/bash
#SBATCH --job-name=nucaliby_ensemble
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ensemble_%j.out
#SBATCH --error=logs/ensemble_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs
PYTHON_BIN=${NUCALIBY_PYTHON:-python}
PYTHONPATH=. "$PYTHON_BIN" scripts/inference/generate_ensemble.py "$@"
