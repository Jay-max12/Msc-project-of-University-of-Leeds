# GBCNET_Plus

Research code for an MSc project investigating whether auxiliary representation-learning objectives improve gallbladder malignancy classification from ultrasound images. The repository contains reproducible training and evaluation pipelines for ROI classification, autoencoder-assisted learning, metric learning, DETR/Faster R-CNN ROI detection, and YOLO one-step detection-classification baselines.

This repository is an academic software deliverable. It is not a medical device and must not be used for diagnosis or clinical decision-making.

## Research scope

The main experiment compares five methods using a shared ResNet-50 feature extractor and classification path:

| Method | Auxiliary objective |
| --- | --- |
| `cls` | Classification only |
| `ae` | ROI reconstruction |
| `vae` | Reconstruction and KL divergence |
| `siamese` | Contrastive learning |
| `triplet` | Triplet metric learning |

Additional baselines evaluate one-step YOLO detection-classification, DETR and Faster R-CNN ROI detection, and two-stage detector-classifier pipelines.

## Repository contents

- `roi_repr/`: primary representation-learning implementation and experiment launchers.
- `detr_roi/`: DETR ROI detector implementation.
- `train_yolov11_onestep.py`: one-step YOLO training and evaluation.
- `evaluate_fasterrcnn_resnet_pipeline.py`: two-stage detector-classifier evaluation.
- `*.slurm`: reproducible Leeds HPC job configurations with portable project paths.
- `docs/`: data access, development, project layout, deliverables, and experiment provenance.
- `YOLO_roi_gradcam_explain.ipynb`: output-free interpretability notebook.

Dataset files, annotations, split manifests, model weights, logs, generated figures, and full experiment outputs are intentionally excluded.

## Setup

The recorded experiment environment used Python 3.9.23, PyTorch 2.6.0 with CUDA 12.4, and the package versions in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CUDA system, install the PyTorch wheel appropriate for the local CUDA driver before installing the remaining requirements if the default wheel is unsuitable.

## Data preparation

The GBCU dataset is not included. Obtain it directly from the dataset owners and comply with their licence agreement. After authorised download, arrange the private local data as described in [docs/DATA.md](docs/DATA.md). The entire `GBCU/` directory is ignored by Git.

Generate a deterministic three-fold manifest locally:

```bash
python -m roi_repr.data.cv_splits \
  --dataset_root GBCU \
  --n_cv_folds 3 \
  --seed 42 \
  --stratify binary \
  --output GBCU/cv_splits/cv3_seed42_binary.json
```

The public dataset does not provide patient identifiers. Consequently, ordinary generated folds are image-level stratified folds and must not be described as patient-level cross-validation.

## Running the main experiment

```bash
python -m roi_repr.train \
  --method all \
  --task_mode binary \
  --n_cv_folds 3 \
  --seed 42 \
  --cv_split_file GBCU/cv_splits/cv3_seed42_binary.json
```

On a Slurm cluster, submit jobs from the repository root so that `SLURM_SUBMIT_DIR` resolves correctly:

```bash
mkdir -p logs
SKIP_PIP_INSTALL=1 sbatch roi_repr/run_ae_cv3.slurm
SKIP_PIP_INSTALL=1 sbatch run_yolov11_onestep_cv3.slurm
```

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for entry points and verification, and [docs/EXPERIMENT_PROVENANCE.md](docs/EXPERIMENT_PROVENANCE.md) for the experiment audit.

## Evaluation protocol

- Model selection uses only the corresponding training-fold validation partition.
- The official test split is reserved for final evaluation and is not a validation set.
- Report internal fold validation and official test performance separately.
- Accuracy should be accompanied by balanced accuracy, malignant sensitivity, malignant specificity, and a confusion matrix.
- Historical experiments with known test-set leakage are documented as legacy results and must not be presented as unbiased estimates.

## Dataset citation

Any work using GBCU must cite:

> S. Basu, M. Gupta, P. Rana, P. Gupta and C. Arora, “Surpassing the Human Accuracy: Detecting Gallbladder Cancer from USG Images with Curriculum Learning,” CVPR, 2022. DOI: 10.1109/CVPR52688.2022.02022.

## Licence and reuse

No open-source licence is granted for this repository. See [LICENSE](LICENSE). Third-party libraries, pretrained models, and the GBCU dataset remain subject to their own licences and terms.
