#!/bin/bash
#SBATCH -J ae_imp2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 48:00:00
#SBATCH --array=0-2

# AE improvement round 2 (baseline: emb=128, cls=1.0, recon=0.3):
#   4) malignant class weight = 1.5
#   5) backbone lr=0.001, head/decoder lr=0.005
#   6) freeze backbone for first 5 epochs
#
# Submit:
#   SKIP_PIP_INSTALL=1 bash roi_repr/submit_ae_improve_round2.sh

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
METHOD="${METHOD:-ae}"
TASK_MODE="${TASK_MODE:-binary}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
LAMBDA_CLS="${LAMBDA_CLS:-1.0}"
LAMBDA_RECON="${LAMBDA_RECON:-0.3}"
SWEEP_ROOT="${SWEEP_ROOT:-outputs/roi_repr/binary/ae_improve_round2}"

IDX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

case "${IDX}" in
  0)
    EXP_TAG="exp4_mal_weight1p5"
    EXTRA_ARGS=(--malignant_class_weight 1.5)
    ;;
  1)
    EXP_TAG="exp5_lr_bb0p001_hd0p005"
    EXTRA_ARGS=(--lr_backbone 0.001 --lr_head 0.005)
    ;;
  2)
    EXP_TAG="exp6_freeze_bb5"
    EXTRA_ARGS=(--freeze_backbone_epochs 5)
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
  if [ -f "${HOME}/.bashrc" ]; then
    set +u
    # shellcheck disable=SC1090
    source "${HOME}/.bashrc"
    set -u
  fi
  if command -v conda >/dev/null 2>&1; then
    conda activate myenv
  fi
  PYTHON_BIN="$(command -v python)"
fi

if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
  "${PYTHON_BIN}" -m pip install -q -r "${PROJECT_ROOT}/requirements.txt"
fi

if [ -f "${OUTPUT_DIR}/cv_results.json" ]; then
  echo "Already complete, skipping: ${OUTPUT_DIR}/cv_results.json"
  exit 0
fi

echo "=== AE improve round2 task ${IDX} (${EXP_TAG}) ==="
echo "python=${PYTHON_BIN}"
echo "embedding_dim=${EMBEDDING_DIM} lambda_cls=${LAMBDA_CLS} lambda_recon=${LAMBDA_RECON}"
echo "extra_args=${EXTRA_ARGS[*]}"
echo "output_dir=${OUTPUT_DIR}"

"${PYTHON_BIN}" -m roi_repr.train \
  --method "${METHOD}" \
  --task_mode "${TASK_MODE}" \
  --embedding_dim "${EMBEDDING_DIM}" \
  --lambda_cls "${LAMBDA_CLS}" \
  --lambda_recon "${LAMBDA_RECON}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.005 \
  --n_cv_folds 3 \
  --seed 42 \
  "${EXTRA_ARGS[@]}"

echo "=== Done: ${OUTPUT_DIR} ==="
