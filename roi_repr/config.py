"""Shared hyper-parameters for all five representation-learning methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Tuple

MethodName = Literal["cls", "ae", "vae", "siamese", "triplet"]
TaskMode = Literal["multiclass", "binary"]
ReconWeightSchedule = Literal["fixed", "linear_decay", "two_stage"]

RAW_CLASS_TAGS = {0: "normal", 1: "benign", 2: "malignant"}
BINARY_CLASS_TAGS = {0: "non_malignant", 1: "malignant"}


@dataclass
class ReprConfig:
    # paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    dataset_root: Path | None = None
    output_dir: Path | None = None

    # task
    method: MethodName = "cls"
    task_mode: TaskMode = "multiclass"

    # architecture (fixed across methods)
    embedding_dim: int = 128
    classifier_head: Literal["linear", "mlp"] = "mlp"
    classifier_hidden_dim: int = 64
    classifier_dropout: float = 0.3
    classifier_norm: Literal["batchnorm", "layernorm"] = "batchnorm"
    classifier_activation: Literal["relu", "gelu"] = "relu"
    roi_size: int = 224
    roi_resize: int = 256
    pretrained: bool = True
    train_hflip_prob: float = 0.0
    train_rotation_degrees: float = 0.0
    train_jitter_brightness: float = 0.0
    train_jitter_contrast: float = 0.0

    # training
    seed: int = 42
    strict_determinism: bool = True
    epochs: int = 100
    batch_size: int = 16
    num_workers: int = 4
    lr: float = 0.005
    lr_backbone: float | None = None
    lr_head: float | None = None
    momentum: float = 0.9
    weight_decay: float = 5e-4
    step_size: int = 5
    gamma: float = 0.9
    freeze_backbone_epochs: int = 0
    malignant_class_weight: float = 1.0

    # split / CV
    n_cv_folds: int = 3  # 在官方 train 上做分层 K-fold；test 固定 hold-out
    cv_split_file: Path | None = None  # 默认 GBCU/cv_splits/cv{n}_seed{seed}_{stratify}.json
    evaluate_test: bool = False

    # early stopping & checkpoint（best 仅从第 min_epochs_before_early_stop 轮起保存）
    min_epochs_before_early_stop: int = 1
    early_stopping_patience: int = 15

    # auxiliary loss weights
    lambda_cls: float = 1.0
    lambda_recon: float = 0.3
    lambda_recon_schedule: ReconWeightSchedule = "fixed"
    stage2_start_epoch: int = 51
    beta_kl: float = 1e-5
    lambda_contrastive: float = 0.5
    lambda_triplet: float = 0.5
    margin: float = 1.0
    cls_on_both_images: bool = True  # siamese: CE on both views

    def __post_init__(self) -> None:
        if self.dataset_root is None:
            self.dataset_root = self.project_root / "GBCU"
        if self.output_dir is None:
            self.output_dir = (
                self.project_root
                / "outputs"
                / "roi_repr"
                / self.task_mode
                / f"{self.method}_cv{self.n_cv_folds}"
            )

    def fold_output_dir(self, fold: int) -> Path:
        return self.output_dir / f"fold{fold}"

    def recon_weight_for_epoch(self, epoch: int) -> float:
        """Return the reconstruction-loss weight used for a 1-based epoch."""
        if self.lambda_recon_schedule == "fixed":
            return float(self.lambda_recon)
        if self.lambda_recon_schedule == "two_stage":
            return float(self.lambda_recon) if epoch < self.stage2_start_epoch else 0.0
        if self.lambda_recon_schedule != "linear_decay":
            raise ValueError(f"Unsupported lambda_recon_schedule: {self.lambda_recon_schedule}")
        if self.epochs <= 1:
            return 0.0
        progress = (epoch - 1) / (self.epochs - 1)
        return float(self.lambda_recon) * max(0.0, 1.0 - progress)

    @property
    def checkpoint_start_epoch(self) -> int:
        if self.lambda_recon_schedule == "two_stage":
            return max(self.min_epochs_before_early_stop, self.stage2_start_epoch)
        return self.min_epochs_before_early_stop

    @property
    def num_classes(self) -> int:
        return 2 if self.task_mode == "binary" else 3

    @property
    def malignant_id(self) -> int:
        return 1 if self.task_mode == "binary" else 2

    def map_label(self, raw_cls: int) -> int:
        raw_cls = int(raw_cls)
        if self.task_mode == "binary":
            return 1 if raw_cls == 2 else 0
        return raw_cls

    def class_tag(self, cls_id: int) -> str:
        if self.task_mode == "binary":
            return BINARY_CLASS_TAGS.get(int(cls_id), str(cls_id))
        return RAW_CLASS_TAGS.get(int(cls_id), str(cls_id))

    def imagenet_norm(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
