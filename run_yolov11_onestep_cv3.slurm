#!/bin/bash
#SBATCH -J yolo_cv3
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 48:00:00

# YOLOv11 one-step with fixed CV3 splits (shared manifest with roi_repr)
#
# Submit:
#   SKIP_PIP_INSTALL=1 sbatch run_yolov11_onestep_cv3.slurm

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/GBCU}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/yolov11_onestep_cv3}"
CV_SPLIT_FILE="${CV_SPLIT_FILE:-${DATASET_ROOT}/cv_splits/cv3_seed42_binary.json}"
N_CV_FOLDS="${N_CV_FOLDS:-3}"
MODEL="${MODEL:-yolo11n.pt}"
SEED="${SEED:-42}"

MYENV_PYTHON="${HOME}/.conda/envs/myenv/bin/python"
if [ -x "${MYENV_PYTHON}" ]; then
  PYTHON_BIN="${MYENV_PYTHON}"
else
  PYTHON_BIN="$(command -v python)"
fi

if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
  "${PYTHON_BIN}" -m pip install ultralytics
fi

if [ ! -f "${CV_SPLIT_FILE}" ]; then
  echo "CV split manifest missing, generating: ${CV_SPLIT_FILE}"
  "${PYTHON_BIN}" -m roi_repr.data.cv_splits \
    --dataset_root "${DATASET_ROOT}" \
    --n_cv_folds "${N_CV_FOLDS}" \
    --seed "${SEED}" \
    --stratify binary \
    --output "${CV_SPLIT_FILE}"
fi

if [ -f "${OUTPUT_DIR}/cv_results.json" ]; then
  echo "Already complete, skipping: ${OUTPUT_DIR}/cv_results.json"
  exit 0
fi

"${PYTHON_BIN}" train_yolov11_onestep.py \
  --dataset_root "${DATASET_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --model "${MODEL}" \
  --n_cv_folds "${N_CV_FOLDS}" \
  --cv_split_file "${CV_SPLIT_FILE}" \
  --seed "${SEED}"

echo "=== Done: ${OUTPUT_DIR} ==="
