#!/usr/bin/env bash
# Submit 9 AE runs (3 embedding dims x 3 lambda_cls) as a Slurm array job.
#
# Usage:
#   cd GBCNET_Plus
#   SKIP_PIP_INSTALL=1 bash roi_repr/submit_ae_sweep.sh
#
# Optional overrides:
#   SWEEP_ROOT=outputs/roi_repr/binary/ae_sweep_night_v2 bash roi_repr/submit_ae_sweep.sh

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p logs

export SKIP_PIP_INSTALL="${SKIP_PIP_INSTALL:-1}"
export SWEEP_ROOT="${SWEEP_ROOT:-outputs/roi_repr/binary/ae_sweep_night}"

JOB_ID="$(sbatch --parsable roi_repr/sweep_ae_hyperparams.slurm)"
echo "Submitted Slurm array job: ${JOB_ID}"
echo "Tasks: 9 (emb=64,128,256 x lambda_cls=1.0,2.0,3.0; lambda_recon=0.3)"
echo "Outputs: ${SWEEP_ROOT}/emb<dim>_cls<cls>_recon0p3/"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/ae_sweep_${JOB_ID}_*.out"
echo ""
echo "Morning summary:"
echo "  python roi_repr/summarize_ae_sweep.py --sweep_root ${SWEEP_ROOT}"
