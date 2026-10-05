#!/bin/bash
#SBATCH --job-name=nucaliby_design
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/design_%j.out
#SBATCH --error=logs/design_%j.err
#
# Sample designs. All arguments are forwarded to scripts/inference/design.py.
#
#   sbatch scripts/inference/design.sbatch --ckpt <ckpt> --test-ids assets/test_pdb_ids.txt \
#       --alphabet nt --guide stop:weight=100 --out results/nt/designs.csv
#
# The evaluation set fits comfortably in one job.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs
PYTHONPATH=. python scripts/inference/design.py "$@"
