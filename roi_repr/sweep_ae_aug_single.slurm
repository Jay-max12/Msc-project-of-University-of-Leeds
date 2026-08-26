#!/bin/bash
#SBATCH -J ae_aug
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 48:00:00
#SBATCH --array=0-1

# Single-factor AE augmentation (baseline emb128 cls1.0 recon0.3 unchanged):
#   [0] hflip p=0.5 only
#   [1] ColorJitter brightness=0.05 contrast=0.05 only
#
# Submit:
#   SKIP_PIP_INSTALL=1 bash roi_repr/submit_ae_aug_single.sh

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
METHOD="${METHOD:-ae}"
TASK_MODE="${TASK_MODE:-binary}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
LAMBDA_CLS="${LAMBDA_CLS:-1.0}"
LAMBDA_RECON="${LAMBDA_RECON:-0.3}"
SWEEP_ROOT="${SWEEP_ROOT:-outputs/roi_repr/binary/ae_aug_single}"

IDX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

case "${IDX}" in
  0)
    EXP_TAG="hflip"
    TRAIN_HFLIP_PROB="0.5"
    TRAIN_ROTATION_DEGREES="0"
    TRAIN_JITTER_BRIGHTNESS="0"
    TRAIN_JITTER_CONTRAST="0"
    ;;
  1)
    EXP_TAG="intensity5"
    TRAIN_HFLIP_PROB="0"
    TRAIN_ROTATION_DEGREES="0"
    TRAIN_JITTER_BRIGHTNESS="0.05"
    TRAIN_JITTER_CONTRAST="0.05"
    ;;
  *)
    echo "Unknown array task id: ${IDX}"
    exit 1
    ;;
esac

OUTPUT_DIR="${SWEEP_ROOT}/${EXP_TAG}"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/${SWEEP_ROOT}"
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
  "${PYTHON_BIN}" -m pip install -q -r "${PROJECT_ROOT}/requirements.txt"
fi

if [ -f "${OUTPUT_DIR}/cv_results.json" ]; then
  echo "Already complete, skipping: ${OUTPUT_DIR}/cv_results.json"
  exit 0
fi

echo "=== AE aug single-factor task ${IDX} (${EXP_TAG}) ==="
echo "python=${PYTHON_BIN}"
echo "embedding_dim=${EMBEDDING_DIM} lambda_cls=${LAMBDA_CLS} lambda_recon=${LAMBDA_RECON}"
echo "hflip=${TRAIN_HFLIP_PROB} rotation=${TRAIN_ROTATION_DEGREES}"
echo "jitter brightness=${TRAIN_JITTER_BRIGHTNESS} contrast=${TRAIN_JITTER_CONTRAST}"
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
  --output_dir "${OUTPUT_DIR}" \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.005 \
  --n_cv_folds 3 \
  --seed 42

echo "=== Done: ${OUTPUT_DIR} ==="
