#!/usr/bin/env bash
# Submit single-factor AE augmentation experiments (hflip, intensity).
#
# Baseline unchanged: emb=128, cls=1.0, recon=0.3
#   [0] RandomHorizontalFlip p=0.5
#   [1] ColorJitter brightness=0.05 contrast=0.05
#
# Usage:
#   cd GBCNET_Plus
#   SKIP_PIP_INSTALL=1 bash roi_repr/submit_ae_aug_single.sh

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs

export SKIP_PIP_INSTALL="${SKIP_PIP_INSTALL:-1}"
export SWEEP_ROOT="${SWEEP_ROOT:-outputs/roi_repr/binary/ae_aug_single}"

JOB_ID="$(sbatch --parsable roi_repr/sweep_ae_aug_single.slurm)"
echo "Submitted Slurm array job: ${JOB_ID}"
echo "Tasks:"
echo "  [0] hflip p=0.5  -> ${SWEEP_ROOT}/hflip/"
echo "  [1] intensity5   -> ${SWEEP_ROOT}/intensity5/"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/ae_aug_${JOB_ID}_*.out"
