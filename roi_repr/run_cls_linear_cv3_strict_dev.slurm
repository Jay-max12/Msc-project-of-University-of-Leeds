#!/bin/bash
#SBATCH -J cls_strict_dev
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 48:00:00

set -eo pipefail

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_BIN="${HOME}/.conda/envs/myenv/bin/python"
CV_SPLIT_FILE="${PROJECT_ROOT}/GBCU/cv_splits/cv3_seed42_binary.json"
OUTPUT_DIR="outputs/roi_repr/binary/cls_linear_cv3_strict_dev"

cd "${PROJECT_ROOT}"
mkdir -p logs

"${PYTHON_BIN}" -m roi_repr.train \
  --method cls \
  --task_mode binary \
  --classifier_head linear \
  --embedding_dim 128 \
  --lambda_cls 1.0 \
  --cv_split_file "${CV_SPLIT_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.005 \
  --n_cv_folds 3 \
  --seed 42 \
  --strict_determinism \
  --no-evaluate_test \
  --min_epochs_before_early_stop 1
