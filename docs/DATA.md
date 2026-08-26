# Data access and local layout

## Why the data are absent

The Gallbladder Cancer Ultrasound (GBCU) dataset is governed by a separate usage agreement. That agreement prohibits redistribution, transmission, publication, copying, or dissemination of the dataset, in whole or in part, without prior written approval. It also requires acknowledgement of the source paper.

For that reason, this public repository does not contain:

- ultrasound images;
- `bbox_annot.json` or `roi_pred.json`;
- the official `train.txt` or `test.txt` files;
- generated cross-validation manifests containing image identifiers or labels;
- contact sheets, similarity-review images, or per-image predictions;
- checkpoints trained on the dataset.

Researchers must request or download the dataset from its owners and independently accept the applicable terms. Do not obtain the data from a fork of this repository.

## Expected private directory structure

After authorised access, create the following untracked layout:

```text
GBCU/
├── imgs/
│   └── <authorised image files>
├── bbox_annot.json
├── roi_pred.json
├── train.txt
├── test.txt
└── cv_splits/
```

The training code expects each split line to contain an image filename and integer class label separated by a comma. Annotation JSON files must retain the schema supplied by the dataset owners.

The repository-wide `.gitignore` excludes `GBCU/`. Before every commit, verify this remains true:

```bash
git check-ignore -v GBCU/train.txt
git status --short
```

## Cross-validation

The deterministic manifest generator reads the authorised local split and annotations:

```bash
python -m roi_repr.data.cv_splits \
  --dataset_root GBCU \
  --n_cv_folds 3 \
  --seed 42 \
  --stratify binary \
  --output GBCU/cv_splits/cv3_seed42_binary.json
```

The public GBCU release does not include patient identifiers. A manifest produced by the command above is therefore image-level stratified cross-validation, not verified patient-level cross-validation. If independently authorised patient/group identifiers are available, use a group-aware split and document the grouping source.

## Required citation

S. Basu, M. Gupta, P. Rana, P. Gupta and C. Arora, “Surpassing the Human Accuracy: Detecting Gallbladder Cancer from USG Images with Curriculum Learning,” IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 20854–20864. DOI: 10.1109/CVPR52688.2022.02022.
