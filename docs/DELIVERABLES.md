# Project deliverables

## 1. Software product

The software product is the collection of training, evaluation, interpretability, and HPC execution code in this repository. It supports the MSc investigation of auxiliary representation learning and detector-classifier alternatives for gallbladder malignancy classification from ultrasound images.

The software is research code, not a deployable clinical system or medical device.

## 2. Source-code repository

The public GitHub repository contains:

- source code required to understand and rerun the experiments;
- dependency versions recorded from the experimental environment;
- portable Slurm launchers;
- output-free notebooks;
- data acquisition and layout instructions;
- developer and reproducibility documentation;
- an experiment provenance audit.

It intentionally excludes restricted data and metadata, trained weights, raw experiment outputs, scheduler logs, secrets, local environments, caches, and temporary review artefacts.

## 3. Developer documentation

Developer documentation is provided by:

- `README.md` for scope, setup, and primary usage;
- `docs/DATA.md` for authorised data preparation and restrictions;
- `docs/DEVELOPMENT.md` for architecture, entry points, outputs, and verification;
- `docs/PROJECT_LAYOUT.md` for code organisation;
- `docs/DETR_ROI.md` and `roi_repr/README.md` for subsystem details;
- `docs/EXPERIMENT_PROVENANCE.md` for protocol and historical experiment audit.

## 4. MSc project report

The final MSc report is a separate assessed deliverable and is not stored in this public repository by default. It should be submitted through the University’s required assessment channel. A public copy should only be added if University policy, supervisor approval, copyright permissions, and dataset confidentiality requirements all permit publication.
