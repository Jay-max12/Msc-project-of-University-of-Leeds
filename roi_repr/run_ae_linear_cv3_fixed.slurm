#!/bin/bash
#SBATCH -J ae_lin_cv3
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 48:00:00

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_BIN="${HOME}/.conda/envs/myenv/bin/python"
CV_SPLIT_FILE="${PROJECT_ROOT}/GBCU/cv_splits/cv3_seed42_binary.json"
OUTPUT_DIR="outputs/roi_repr/binary/ae_linear_cv3_fixed"

cd "${PROJECT_ROOT}"
mkdir -p logs

"${PYTHON_BIN}" -m roi_repr.train \
  --method ae \
  --task_mode binary \
  --classifier_head linear \
  --embedding_dim 128 \
  --lambda_cls 1.0 \
  --lambda_recon 0.3 \
  --cv_split_file "${CV_SPLIT_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.005 \
  --n_cv_folds 3 \
  --seed 42
