#!/bin/bash
#SBATCH -J ae_cv3
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 48:00:00

# AE baseline with fixed CV3 splits (shared manifest with YOLO one-step):
#   emb=128, lambda_cls=1.0, lambda_recon=0.3, no train augmentation
#
# Submit:
#   SKIP_PIP_INSTALL=1 sbatch roi_repr/run_ae_cv3.slurm

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
METHOD="${METHOD:-ae}"
TASK_MODE="${TASK_MODE:-binary}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
LAMBDA_CLS="${LAMBDA_CLS:-1.0}"
LAMBDA_RECON="${LAMBDA_RECON:-0.3}"
N_CV_FOLDS="${N_CV_FOLDS:-3}"
CV_SPLIT_FILE="${CV_SPLIT_FILE:-${PROJECT_ROOT}/GBCU/cv_splits/cv3_seed42_binary.json}"
TRAIN_HFLIP_PROB="${TRAIN_HFLIP_PROB:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/roi_repr/binary/ae_cv3_fixed}"

mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

MYENV_PYTHON="${HOME}/.conda/envs/myenv/bin/python"
if [ -x "${MYENV_PYTHON}" ]; then
  PYTHON_BIN="${MYENV_PYTHON}"
else
  PYTHON_BIN="$(command -v python)"
fi

if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
  "${PYTHON_BIN}" -m pip install -q -r "${PROJECT_ROOT}/requirements.txt"
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

echo "=== AE ${N_CV_FOLDS}-fold CV (fixed splits) ==="
echo "python=${PYTHON_BIN}"
echo "embedding_dim=${EMBEDDING_DIM} lambda_cls=${LAMBDA_CLS} lambda_recon=${LAMBDA_RECON}"
echo "train_hflip_prob=${TRAIN_HFLIP_PROB}"
echo "cv_split_file=${CV_SPLIT_FILE}"
echo "output_dir=${OUTPUT_DIR}"

"${PYTHON_BIN}" -m roi_repr.train \
  --method "${METHOD}" \
  --task_mode "${TASK_MODE}" \
  --embedding_dim "${EMBEDDING_DIM}" \
  --lambda_cls "${LAMBDA_CLS}" \
  --lambda_recon "${LAMBDA_RECON}" \
  --train_hflip_prob "${TRAIN_HFLIP_PROB}" \
  --cv_split_file "${CV_SPLIT_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.005 \
  --n_cv_folds "${N_CV_FOLDS}" \
  --seed 42

echo "=== Done: ${OUTPUT_DIR} ==="
