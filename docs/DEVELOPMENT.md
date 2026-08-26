# Developer documentation

## Architecture

The project has three main paths:

1. `roi_repr`: crops ground-truth or predicted ROIs, extracts ResNet-50 features, learns a 128-dimensional representation, and predicts the image class. AE, VAE, Siamese, and Triplet variants add an auxiliary objective while retaining the same classification path.
2. `detr_roi` and `evaluate_fasterrcnn_resnet_pipeline.py`: train or evaluate ROI detectors and pass selected regions to a classifier.
3. `train_yolov11_onestep.py`: trains an end-to-end detector whose ROI classes are fused into an image-level prediction.

Configuration for the main representation experiments lives in `roi_repr/config.py`. Shared data and CV logic lives in `roi_repr/data/`; model components live in `roi_repr/models/`; training and metrics live in `roi_repr/training/`.

## Principal entry points

| Command | Purpose |
| --- | --- |
| `python -m roi_repr.data.cv_splits ...` | Generate a local deterministic CV manifest |
| `python -m roi_repr.train ...` | Run representation-learning experiments |
| `python train_yolov11_onestep.py ...` | Run one-step YOLO experiments |
| `python -m detr_roi.train ...` | Train the DETR ROI detector |
| `python -m detr_roi.evaluate ...` | Evaluate a DETR checkpoint |
| `python evaluate_fasterrcnn_resnet_pipeline.py ...` | Evaluate two-stage detection/classification |
| `python -m roi_repr.eval_yolo_roi_pipeline ...` | Evaluate YOLO ROI plus representation classifier |

Use `python <entry-point> --help` to inspect the complete command-line interface.

## Output policy

All generated artefacts belong under `outputs/` or `logs/`; both are ignored by Git. Checkpoints and export formats (`.pt`, `.pth`, `.ckpt`, `.onnx`, `.engine`) are also ignored globally.

Every experiment should record:

- code commit;
- split protocol and seed;
- full command or Slurm script;
- package/CUDA environment;
- training hyperparameters;
- fold-wise metrics and selection rule;
- whether the official test split was evaluated.

Never select a model, threshold, architecture, or hyperparameter using official test performance.

## Slurm portability

Slurm files derive `PROJECT_ROOT` from an explicit environment override, `SLURM_SUBMIT_DIR`, or the current directory. Submit from the repository root:

```bash
cd /path/to/GBCNET_Plus
mkdir -p logs
SKIP_PIP_INSTALL=1 sbatch <job-file>.slurm
```

Set `PROJECT_ROOT=/absolute/path/to/GBCNET_Plus` when submitting from another directory. Site-specific partition, GPU, memory, and time directives may need adjustment.

## Verification before release

The following checks do not require the private dataset:

```bash
python -m compileall -q detr_roi roi_repr *.py
for file in *.slurm roi_repr/*.slurm; do bash -n "$file"; done
python -m json.tool YOLO_roi_gradcam_explain.ipynb >/dev/null
python -m json.tool roi_repr/experiments.ipynb >/dev/null
```

Before publishing, also run:

```bash
git status --short
git ls-files | grep -E '(^GBCU/|similarity_review|\.(pt|pth|ckpt|onnx|engine)$)' && exit 1 || true
```

Full training and numerical reproduction require authorised GBCU data and suitable GPU resources.
