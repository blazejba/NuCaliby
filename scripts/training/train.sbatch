#!/bin/bash
#SBATCH --job-name=nucaliby_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=1000G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#
# Arguments are forwarded to nucaliby.training.train.
#
#   sbatch scripts/training/train.sbatch --data-root /path/to/store --out-dir runs/nucaliby
#

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs
PYTHONPATH=. torchrun --standalone --nproc_per_node=4 --module nucaliby.training.train "$@"
