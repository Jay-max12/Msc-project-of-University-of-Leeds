"""Unified training loop for all five methods (3-fold CV)."""

from __future__ import annotations

import json
import os
import random
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from roi_repr.config import ReprConfig, MethodName
from roi_repr.data.splits import (
    FoldSplits,
    build_cv_folds,
    make_single_loader,
    make_train_loader,
    print_cv_summary,
)
from roi_repr.losses.losses import classification_loss, compute_batch_loss, contrastive_loss
from roi_repr.models.repr_model import RoiReprModel
from roi_repr.training.metrics import checkpoint_score, eval_split, format_metrics_line

_METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "mal_sens",
    "mal_spec",
    "select",
)


def set_seed(seed: int, strict_determinism: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = strict_determinism
    torch.use_deterministic_algorithms(strict_determinism)
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


def set_backbone_trainable(model: RoiReprModel, trainable: bool) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = trainable


def set_decoder_trainable(model: RoiReprModel, trainable: bool) -> None:
    if model.decoder is None:
        return
    for param in model.decoder.parameters():
        param.requires_grad = trainable


def build_cls_criterion(cfg: ReprConfig, device: torch.device) -> nn.Module:
    if cfg.malignant_class_weight != 1.0:
        weights = torch.ones(cfg.num_classes, dtype=torch.float32, device=device)
        weights[cfg.malignant_id] = float(cfg.malignant_class_weight)
        return nn.CrossEntropyLoss(weight=weights)
    return nn.CrossEntropyLoss()


def build_optimizer(cfg: ReprConfig, model: RoiReprModel) -> SGD:
    if cfg.lr_backbone is not None and cfg.lr_head is not None:
        head_params = (
            list(model.embedding_head.parameters())
            + list(model.classifier.parameters())
        )
        if model.decoder is not None:
            head_params += list(model.decoder.parameters())
        if model.fc_logvar is not None:
            head_params += list(model.fc_logvar.parameters())
        return SGD(
            [
                {"params": model.backbone.parameters(), "lr": cfg.lr_backbone},
                {"params": head_params, "lr": cfg.lr_head},
            ],
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    return SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )


def train_one_epoch(
    cfg: ReprConfig,
    model: RoiReprModel,
    loader,
    aux_ds,
    optimizer: SGD,
    cls_crit: nn.Module,
    triplet_crit: nn.Module,
    device: torch.device,
    lambda_recon: float,
) -> Tuple[float, float, float]:
    model.train()
    if lambda_recon == 0.0 and model.decoder is not None:
        model.decoder.eval()
    totals, cls_losses, metric_losses = [], [], []

    if cfg.method == "siamese":
        assert aux_ds is not None
        for img1, img2, pair_label, lbl1, lbl2 in loader:
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            pair_label = pair_label.to(device, non_blocking=True)
            lbl1 = lbl1.to(device, non_blocking=True).long()
            lbl2 = lbl2.to(device, non_blocking=True).long()
            out1 = model(img1)
            out2 = model(img2)
            l_cls = classification_loss(out1["logits"], lbl1, cls_crit)
            if cfg.cls_on_both_images:
                l_cls = 0.5 * (l_cls + classification_loss(out2["logits"], lbl2, cls_crit))
            l_metric = contrastive_loss(out1["z"], out2["z"], pair_label, margin=cfg.margin)
            loss = cfg.lambda_cls * l_cls + cfg.lambda_contrastive * l_metric
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals.append(float(loss.item()))
            cls_losses.append(float(l_cls.item()))
            metric_losses.append(float(l_metric.item()))
        return _mean3(totals, cls_losses, metric_losses)

    if cfg.method == "triplet":
        for anchor, positive, negative, lbl in loader:
            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True).long()
            out_a = model(anchor)
            z_p = model.forward_embedding(positive)
            z_n = model.forward_embedding(negative)
            l_cls = classification_loss(out_a["logits"], lbl, cls_crit)
            l_metric = triplet_crit(out_a["z"], z_p, z_n)
            loss = cfg.lambda_cls * l_cls + cfg.lambda_triplet * l_metric
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals.append(float(loss.item()))
            cls_losses.append(float(l_cls.item()))
            metric_losses.append(float(l_metric.item()))
        return _mean3(totals, cls_losses, metric_losses)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        out = model(x)
        loss, parts = compute_batch_loss(
            cfg, cfg.method, out, y, x, cls_crit, lambda_recon=lambda_recon
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totals.append(parts["total"])
        cls_losses.append(parts["cls"])
        metric_losses.append(parts.get("recon", 0.0) + parts.get("kl", 0.0))
    return _mean3(totals, cls_losses, metric_losses)


def _mean3(a: List[float], b: List[float], c: List[float]) -> Tuple[float, float, float]:
    return (
        float(np.mean(a)) if a else 0.0,
        float(np.mean(b)) if b else 0.0,
        float(np.mean(c)) if c else 0.0,
    )


def run_single_fold(cfg: ReprConfig, splits: FoldSplits) -> dict:
    fold = int(splits["fold"])
    fold_seed = int(cfg.seed) + fold
    set_seed(fold_seed, cfg.strict_determinism)
    print(
        f"[Reproducibility] fold_seed={fold_seed} strict_determinism={cfg.strict_determinism} "
        f"deterministic_algorithms={torch.are_deterministic_algorithms_enabled()} "
        f"cudnn_benchmark={torch.backends.cudnn.benchmark}"
    )
    out_dir = cfg.fold_output_dir(fold)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, aux_ds = make_train_loader(cfg, splits["train"], loader_seed=fold_seed)
    fold_val_loader = make_single_loader(
        cfg, splits["val"], shuffle=False, loader_seed=fold_seed + 10_000
    )
    test_loader = None
    if cfg.evaluate_test:
        test_loader = make_single_loader(
            cfg, splits["test"], shuffle=False, loader_seed=fold_seed + 20_000
        )

    model = RoiReprModel(cfg).to(device)
    optimizer = build_optimizer(cfg, model)
    scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    cls_crit = build_cls_criterion(cfg, device)
    triplet_crit = nn.TripletMarginLoss(margin=cfg.margin, p=2)

    backbone_frozen = False
    if cfg.freeze_backbone_epochs > 0:
        set_backbone_trainable(model, False)
        backbone_frozen = True
        print(f"[Freeze] backbone frozen for first {cfg.freeze_backbone_epochs} epoch(s)")

    best_score = -1.0
    best_epoch = -1
    best_state = None
    best_fold_val = None
    patience = 0
    history: List[dict] = []

    print(f"\n===== [{cfg.method.upper()}] fold={fold} TASK_MODE={cfg.task_mode} =====")
    for epoch in range(1, cfg.epochs + 1):
        lambda_recon = cfg.recon_weight_for_epoch(epoch)
        stage = "classification_only" if lambda_recon == 0.0 else "joint_ae"
        if cfg.lambda_recon_schedule == "two_stage" and epoch == cfg.stage2_start_epoch:
            set_decoder_trainable(model, False)
            print(
                f"[Stage2] epoch={epoch}: reconstruction loss disabled; decoder frozen; "
                "encoder and classifier remain trainable"
            )
        if backbone_frozen and epoch > cfg.freeze_backbone_epochs:
            set_backbone_trainable(model, True)
            backbone_frozen = False
            print(f"[Freeze] backbone unfrozen at epoch {epoch}")

        if aux_ds is not None and hasattr(aux_ds, "set_epoch"):
            aux_ds.set_epoch(epoch)

        train_total, train_cls, train_aux = train_one_epoch(
            cfg,
            model,
            train_loader,
            aux_ds,
            optimizer,
            cls_crit,
            triplet_crit,
            device,
            lambda_recon,
        )
        fold_val_m = eval_split(
            cfg, model, fold_val_loader, cls_crit, device, lambda_recon=lambda_recon
        )
        score = checkpoint_score(cfg, fold_val_m)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_total": train_total,
                "train_cls": train_cls,
                "train_aux": train_aux,
                "lambda_recon": lambda_recon,
                "stage": stage,
                "fold_val_acc": fold_val_m["accuracy"],
                "fold_val_bal_acc": fold_val_m["balanced_accuracy"],
                "fold_val_sens": fold_val_m["mal_sens"],
                "fold_val_spec": fold_val_m["mal_spec"],
                "fold_val_select": fold_val_m["select"],
            }
        )

        ckpt_tag = ""
        if epoch >= cfg.checkpoint_start_epoch:
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_fold_val = {k: v for k, v in fold_val_m.items() if k not in ("y_true", "y_pred")}
                patience = 0
                ckpt_tag = " *best*"
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "method": cfg.method,
                        "task_mode": cfg.task_mode,
                        "fold": fold,
                        "epoch": best_epoch,
                        "lambda_recon": lambda_recon,
                        "lambda_recon_schedule": cfg.lambda_recon_schedule,
                        "stage2_start_epoch": cfg.stage2_start_epoch,
                        "fold_val_metrics": best_fold_val,
                    },
                    ckpt_dir / "best_model.pth",
                )
            else:
                patience += 1

        print(
            f"[Epoch {epoch:03d}/{cfg.epochs}] loss={train_total:.4f} cls={train_cls:.4f} "
            f"aux={train_aux:.4f} lambda_recon={lambda_recon:.6f} | "
            f"fold_val_acc={fold_val_m['accuracy']:.4f} sens={fold_val_m['mal_sens']:.4f} "
            f"spec={fold_val_m['mal_spec']:.4f} bal={fold_val_m['balanced_accuracy']:.4f} "
            f"select={fold_val_m['select']:.4f}{ckpt_tag}"
        )

        if epoch >= cfg.checkpoint_start_epoch and patience >= cfg.early_stopping_patience:
            print(
                f"[EarlyStop] fold={fold} stopped_at_epoch={epoch} best_epoch={best_epoch} "
                f"best_fold_val_select={best_score:.4f} "
                f"(best checkpoint only saved from epoch {cfg.checkpoint_start_epoch}+)"
            )
            break

    if best_state is None:
        raise RuntimeError(
            f"fold {fold}: no checkpoint saved (need training to reach epoch {cfg.checkpoint_start_epoch})"
        )
    model.load_state_dict(best_state)

    print(
        f"\n----- fold={fold} best checkpoint (epoch {best_epoch}) fold-val metrics -----\n"
        f"{format_metrics_line('[FOLD_VAL] ', best_fold_val or {})}"
    )

    test_m = None
    report = None
    cm = None
    if cfg.evaluate_test:
        assert test_loader is not None
        test_m = eval_split(
            cfg,
            model,
            test_loader,
            cls_crit,
            device,
            lambda_recon=cfg.recon_weight_for_epoch(best_epoch),
        )
        y_true, y_pred = test_m.pop("y_true"), test_m.pop("y_pred")
        target_names = [cfg.class_tag(i) for i in range(cfg.num_classes)]
        report = classification_report(
            y_true, y_pred, target_names=target_names, digits=4, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=list(range(cfg.num_classes)))

        print(f"\n===== TEST [{cfg.method}] fold={fold} TASK_MODE={cfg.task_mode} =====")
        print(format_metrics_line("[TEST] ", test_m))
        print(f"\nPer-class report:\n{report}")
        print(f"Confusion matrix:\n{cm}")
    else:
        print("[TEST LOCKED] held-out test evaluation disabled for development run")

    result = {
        "method": cfg.method,
        "task_mode": cfg.task_mode,
        "fold": fold,
        "fold_seed": fold_seed,
        "strict_determinism": cfg.strict_determinism,
        "best_epoch": best_epoch,
        "best_fold_val_select": best_score,
        "best_fold_val_metrics": best_fold_val,
        "lambda_recon_initial": cfg.lambda_recon,
        "lambda_recon_schedule": cfg.lambda_recon_schedule,
        "stage2_start_epoch": cfg.stage2_start_epoch,
        "test_metrics": test_m,
        "classification_report": report,
        "confusion_matrix": cm.tolist() if cm is not None else None,
        "history": history,
    }
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def summarize_cv(fold_results: List[dict]) -> dict:
    summary: Dict[str, dict] = {}
    for key in _METRIC_KEYS:
        vals = [float(r["best_fold_val_metrics"][key]) for r in fold_results]
        summary[f"fold_val_{key}_mean"] = float(np.mean(vals))
        summary[f"fold_val_{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    has_test = all(r.get("test_metrics") is not None for r in fold_results)
    if has_test:
        for key in _METRIC_KEYS:
            vals = [float(r["test_metrics"][key]) for r in fold_results]
            summary[f"test_{key}_mean"] = float(np.mean(vals))
            summary[f"test_{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    summary["n_folds"] = len(fold_results)
    summary["test_evaluated"] = has_test
    summary["folds"] = [
        {
            "fold": r["fold"],
            "best_epoch": r["best_epoch"],
            "best_fold_val_select": r["best_fold_val_select"],
            **(
                {f"test_{k}": r["test_metrics"][k] for k in _METRIC_KEYS}
                if r.get("test_metrics") is not None
                else {}
            ),
        }
        for r in fold_results
    ]
    return summary


def print_cv_aggregate(method: str, task_mode: str, summary: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"[CV SUMMARY] method={method} task_mode={task_mode} n_folds={summary['n_folds']}")
    print(
        f"  fold_val_acc      {summary['fold_val_accuracy_mean']:.4f} ± "
        f"{summary['fold_val_accuracy_std']:.4f}"
    )
    print(
        f"  fold_val_select   {summary['fold_val_select_mean']:.4f} ± "
        f"{summary['fold_val_select_std']:.4f}"
    )
    if summary["test_evaluated"]:
        print(
            f"  test_select       {summary['test_select_mean']:.4f} ± "
            f"{summary['test_select_std']:.4f}"
        )
    else:
        print("  test              LOCKED (not evaluated)")
    for row in summary["folds"]:
        line = (
            f"    fold {row['fold']}: best_ep={row['best_epoch']} "
            f"fold_val_select={row['best_fold_val_select']:.4f}"
        )
        if summary["test_evaluated"]:
            line += f" test_select={row['test_select']:.4f}"
        print(line)
    print(f"{'=' * 60}\n")


def run_experiment(cfg: ReprConfig) -> dict:
    set_seed(cfg.seed, cfg.strict_determinism)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    folds = build_cv_folds(cfg)
    print_cv_summary(cfg, folds)

    fold_results = [run_single_fold(cfg, fd) for fd in folds]
    cv_summary = summarize_cv(fold_results)
    print_cv_aggregate(cfg.method, cfg.task_mode, cv_summary)

    out = {
        "method": cfg.method,
        "task_mode": cfg.task_mode,
        "n_cv_folds": cfg.n_cv_folds,
        "seed": cfg.seed,
        "fold_seeds": [int(cfg.seed) + i for i in range(cfg.n_cv_folds)],
        "strict_determinism": cfg.strict_determinism,
        "evaluate_test": cfg.evaluate_test,
        "lambda_recon_initial": cfg.lambda_recon,
        "lambda_recon_schedule": cfg.lambda_recon_schedule,
        "stage2_start_epoch": cfg.stage2_start_epoch,
        "cv_summary": cv_summary,
        "fold_results": fold_results,
    }
    with (cfg.output_dir / "cv_results.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def run_all_methods(base_cfg: ReprConfig, methods: Optional[List[MethodName]] = None) -> Dict[str, dict]:
    methods = methods or ["cls", "ae", "vae", "siamese", "triplet"]
    all_results: Dict[str, dict] = {}
    for m in methods:
        cfg = replace(base_cfg, method=m, output_dir=None)
        cfg.__post_init__()
        all_results[m] = run_experiment(cfg)

    summary_path = (
        base_cfg.project_root
        / "outputs"
        / "roi_repr"
        / base_cfg.task_mode
        / f"all_methods_cv{base_cfg.n_cv_folds}.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {m: r["cv_summary"] for m, r in all_results.items()},
            f,
            indent=2,
            default=str,
        )
    print(f"[Saved] {summary_path}")
    return all_results
