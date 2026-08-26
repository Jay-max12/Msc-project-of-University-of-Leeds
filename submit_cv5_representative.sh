#!/usr/bin/env bash
# Submit representative 5-fold CV experiments:
#   1) AE baseline (emb128, cls1.0, recon0.3)
#   2) YOLOv11 one-step (detect_cls)
#
# Usage:
#   cd GBCNET_Plus
#   SKIP_PIP_INSTALL=1 bash submit_cv5_representative.sh

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs
export SKIP_PIP_INSTALL="${SKIP_PIP_INSTALL:-1}"

AE_JOB="$(sbatch --parsable roi_repr/run_ae_cv5.slurm)"
YOLO_JOB="$(sbatch --parsable run_yolov11_onestep_cv5.slurm)"

echo "Submitted representative CV5 jobs:"
echo "  AE baseline:  ${AE_JOB}  -> outputs/roi_repr/binary/ae_cv5/"
echo "  YOLO one-step: ${YOLO_JOB} -> outputs/yolov11_onestep_cv5/"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/ae_cv5_${AE_JOB}.out"
echo "  tail -f logs/yolo_cv5_${YOLO_JOB}.out"
