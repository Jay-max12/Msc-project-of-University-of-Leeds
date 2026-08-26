#!/bin/bash
#SBATCH -J roi_repr
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 48:00:00

# 用法示例：
#   cd GBCNET_Plus && SKIP_PIP_INSTALL=1 sbatch roi_repr/train_roi_repr.slurm
#   METHOD=ae EMBEDDING_DIM=128 LAMBDA_CLS=1.0 LAMBDA_RECON=0.3 \
#     OUTPUT_DIR=outputs/roi_repr/binary/ae_cv3_hflip sbatch roi_repr/train_roi_repr.slurm

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
METHOD="${METHOD:-ae}"
TASK_MODE="${TASK_MODE:-binary}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
LAMBDA_CLS="${LAMBDA_CLS:-1.0}"
LAMBDA_RECON="${LAMBDA_RECON:-0.3}"
TRAIN_HFLIP_PROB="${TRAIN_HFLIP_PROB:-0}"
TRAIN_ROTATION_DEGREES="${TRAIN_ROTATION_DEGREES:-0}"
TRAIN_JITTER_BRIGHTNESS="${TRAIN_JITTER_BRIGHTNESS:-0}"
TRAIN_JITTER_CONTRAST="${TRAIN_JITTER_CONTRAST:-0}"
N_CV_FOLDS="${N_CV_FOLDS:-3}"
CV_SPLIT_FILE="${CV_SPLIT_FILE:-${PROJECT_ROOT}/GBCU/cv_splits/cv3_seed42_binary.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/roi_repr/binary/ae_cv3}"

mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

MYENV_PYTHON="${HOME}/.conda/envs/myenv/bin/python"
if [ -x "${MYENV_PYTHON}" ]; then
  PYTHON_BIN="${MYENV_PYTHON}"
elif [ -n "${PYTHON_BIN:-}" ] && [ -x "${PYTHON_BIN}" ]; then
  :
else
  PYTHON_BIN="$(command -v python)"
fi

if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
  "${PYTHON_BIN}" -m pip install -r "${PROJECT_ROOT}/requirements.txt"
fi

if [ ! -f "${CV_SPLIT_FILE}" ]; then
  echo "CV split manifest missing, generating: ${CV_SPLIT_FILE}"
  "${PYTHON_BIN}" -m roi_repr.data.cv_splits \
    --dataset_root "${PROJECT_ROOT}/GBCU" \
    --n_cv_folds "${N_CV_FOLDS}" \
    --seed 42 \
    --stratify binary \
    --output "${CV_SPLIT_FILE}"
fi

if [ -f "${OUTPUT_DIR}/cv_results.json" ]; then
  echo "Already complete, skipping: ${OUTPUT_DIR}/cv_results.json"
  exit 0
fi

echo "=== roi_repr train ==="
echo "python=${PYTHON_BIN}"
echo "method=${METHOD} task_mode=${TASK_MODE}"
echo "embedding_dim=${EMBEDDING_DIM} lambda_cls=${LAMBDA_CLS} lambda_recon=${LAMBDA_RECON}"
echo "train_hflip_prob=${TRAIN_HFLIP_PROB} rotation=${TRAIN_ROTATION_DEGREES}"
echo "jitter brightness=${TRAIN_JITTER_BRIGHTNESS} contrast=${TRAIN_JITTER_CONTRAST}"
echo "n_cv_folds=${N_CV_FOLDS} cv_split_file=${CV_SPLIT_FILE}"
echo "output_dir=${OUTPUT_DIR}"

"${PYTHON_BIN}" -m roi_repr.train \
  --method "${METHOD}" \
  --task_mode "${TASK_MODE}" \
  --embedding_dim "${EMBEDDING_DIM}" \
  --lambda_cls "${LAMBDA_CLS}" \
  --lambda_recon "${LAMBDA_RECON}" \
  --train_hflip_prob "${TRAIN_HFLIP_PROB}" \
  --train_rotation_degrees "${TRAIN_ROTATION_DEGREES}" \
  --train_jitter_brightness "${TRAIN_JITTER_BRIGHTNESS}" \
  --train_jitter_contrast "${TRAIN_JITTER_CONTRAST}" \
  --cv_split_file "${CV_SPLIT_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.005 \
  --n_cv_folds "${N_CV_FOLDS}" \
  --seed 42

echo "=== Done: ${OUTPUT_DIR} ==="
