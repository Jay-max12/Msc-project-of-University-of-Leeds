#!/bin/bash
#SBATCH -J roi_repr
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 48:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
METHOD="${METHOD:-all}"          # cls | ae | vae | siamese | triplet | all
TASK_MODE="${TASK_MODE:-multiclass}"  # multiclass | binary

mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

# 按需激活环境（正式训练务必取消注释）：
# source ~/.bashrc
# conda activate myenv

python -m pip install -q -r requirements.txt

python -m roi_repr.train \
  --method "${METHOD}" \
  --task_mode "${TASK_MODE}" \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.005 \
  --n_cv_folds 3 \
  --seed 42
