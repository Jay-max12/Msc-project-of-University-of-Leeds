#!/bin/bash
#SBATCH -J cls_cv3_fix
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 48:00:00

# Pure-classification baseline using the exact fixed CV3 manifest shared by AE and YOLO.

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CV_SPLIT_FILE="${PROJECT_ROOT}/GBCU/cv_splits/cv3_seed42_binary.json"
OUTPUT_DIR="outputs/roi_repr/binary/cls_cv3_fixed"
PYTHON_BIN="${HOME}/.conda/envs/myenv/bin/python"

cd "${PROJECT_ROOT}"
mkdir -p logs

if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="$(command -v python)"
fi

if [ -f "${OUTPUT_DIR}/cv_results.json" ]; then
  echo "Already complete, skipping: ${OUTPUT_DIR}/cv_results.json"
  exit 0
fi

"${PYTHON_BIN}" -m roi_repr.train \
  --method cls \
  --task_mode binary \
  --embedding_dim 128 \
  --lambda_cls 1.0 \
  --train_hflip_prob 0 \
  --cv_split_file "${CV_SPLIT_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.005 \
  --n_cv_folds 3 \
  --seed 42
