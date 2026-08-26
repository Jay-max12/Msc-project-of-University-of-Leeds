"""CLI entry point for unified ROI representation learning experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from roi_repr.config import ReprConfig, MethodName
from roi_repr.training.trainer import run_all_methods, run_experiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("GBCU ROI representation learning (5-method comparison)")
    p.add_argument("--method", type=str, default="cls", choices=["cls", "ae", "vae", "siamese", "triplet", "all"])
    p.add_argument("--task_mode", type=str, default="multiclass", choices=["multiclass", "binary"])
    p.add_argument("--dataset_root", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--n_cv_folds", type=int, default=3, help="Stratified K-fold on train split")
    p.add_argument(
        "--evaluate_test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate held-out test after CV (default: disabled during development).",
    )
    p.add_argument(
        "--cv_split_file",
        type=str,
        default=None,
        help="Path to fixed CV split JSON (default: dataset_root/cv_splits/cv{n}_seed{seed}_{stratify}.json)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--strict_determinism",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require deterministic PyTorch/CUDA algorithms (default: enabled).",
    )
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument(
        "--min_epochs_before_early_stop",
        type=int,
        default=1,
        help="First epoch eligible for checkpoint selection and early stopping.",
    )
    p.add_argument(
        "--classifier_head",
        type=str,
        default="mlp",
        choices=["linear", "mlp"],
        help="Classifier attached to the shared embedding; mlp preserves the current default.",
    )
    p.add_argument(
        "--lambda_cls",
        type=float,
        default=1.0,
        help="Weight for classification CE loss (AE/VAE/cls/siamese/triplet)",
    )
    p.add_argument(
        "--lambda_recon",
        type=float,
        default=0.3,
        help="Weight for AE/VAE reconstruction L1 loss",
    )
    p.add_argument(
        "--lambda_recon_schedule",
        type=str,
        default="fixed",
        choices=["fixed", "linear_decay", "two_stage"],
        help="Reconstruction schedule: fixed, linear decay, or hard switch to 0 in stage 2.",
    )
    p.add_argument(
        "--stage2_start_epoch",
        type=int,
        default=51,
        help="First classification-only epoch for the two_stage reconstruction schedule.",
    )
    p.add_argument(
        "--malignant_class_weight",
        type=float,
        default=1.0,
        help="CrossEntropy weight for the malignant class (binary task)",
    )
    p.add_argument(
        "--lr_backbone",
        type=float,
        default=None,
        help="Backbone learning rate; use with --lr_head for differential LR",
    )
    p.add_argument(
        "--lr_head",
        type=float,
        default=None,
        help="Learning rate for embedding head / classifier / decoder",
    )
    p.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=0,
        help="Freeze ResNet50 backbone for the first N epochs",
    )
    p.add_argument(
        "--train_hflip_prob",
        type=float,
        default=0.0,
        help="Train-time horizontal flip probability (0 to disable)",
    )
    p.add_argument(
        "--train_rotation_degrees",
        type=float,
        default=0.0,
        help="Train-time random rotation max degrees (e.g. 5 for ±5°)",
    )
    p.add_argument(
        "--train_jitter_brightness",
        type=float,
        default=0.0,
        help="Train-time ColorJitter brightness (e.g. 0.05 for ±5%%)",
    )
    p.add_argument(
        "--train_jitter_contrast",
        type=float,
        default=0.0,
        help="Train-time ColorJitter contrast (e.g. 0.05 for ±5%%)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = {
        "method": "cls" if args.method == "all" else args.method,
        "task_mode": args.task_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "n_cv_folds": args.n_cv_folds,
        "evaluate_test": args.evaluate_test,
        "seed": args.seed,
        "strict_determinism": args.strict_determinism,
        "embedding_dim": args.embedding_dim,
        "min_epochs_before_early_stop": args.min_epochs_before_early_stop,
        "classifier_head": args.classifier_head,
        "lambda_cls": args.lambda_cls,
        "lambda_recon": args.lambda_recon,
        "lambda_recon_schedule": args.lambda_recon_schedule,
        "stage2_start_epoch": args.stage2_start_epoch,
        "malignant_class_weight": args.malignant_class_weight,
        "lr_backbone": args.lr_backbone,
        "lr_head": args.lr_head,
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "train_hflip_prob": args.train_hflip_prob,
        "train_rotation_degrees": args.train_rotation_degrees,
        "train_jitter_brightness": args.train_jitter_brightness,
        "train_jitter_contrast": args.train_jitter_contrast,
    }
    if args.dataset_root:
        kwargs["dataset_root"] = Path(args.dataset_root)
    if args.cv_split_file:
        kwargs["cv_split_file"] = Path(args.cv_split_file)
    if args.output_dir:
        kwargs["output_dir"] = Path(args.output_dir)

    base_cfg = ReprConfig(**kwargs)
    base_cfg.__post_init__()

    if args.method == "all":
        run_all_methods(base_cfg)
    else:
        run_experiment(base_cfg)


if __name__ == "__main__":
    main()
