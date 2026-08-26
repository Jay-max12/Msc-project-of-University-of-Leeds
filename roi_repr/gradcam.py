"""Grad-CAM explainability for YOLO-predicted ROI + roi_repr classifier pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from roi_repr.config import ReprConfig
from roi_repr.eval_multi_roi import fuse_binary_any_positive
from roi_repr.eval_yolo_roi_pipeline import FUSION_RULE
from train_resnet50_roi_classifier import load_bbox_annotations, sanitize_box_xyxy
from vae_yolo_roi import list_pred_boxes_for_image

CaseType = Literal["TP", "TN", "FP", "FN"]
NoBoxFallback = Literal["full_image", "non_malignant", "skip"]
ALL_METHODS: Tuple[str, ...] = ("ae", "cls", "vae", "siamese", "triplet")


def binary_case_type(gt_class: int, pred_class: int) -> CaseType:
    gt_pos = int(gt_class) == 1
    pred_pos = int(pred_class) == 1
    if gt_pos and pred_pos:
        return "TP"
    if not gt_pos and not pred_pos:
        return "TN"
    if not gt_pos and pred_pos:
        return "FP"
    return "FN"


CASE_COLORS = {
    "TP": "#2ecc71",
    "TN": "#3498db",
    "FP": "#e67e22",
    "FN": "#e74c3c",
}

# Radiologist GT boxes in bbox_annot.json (display-only overlay).
GT_BOX_STYLES: Dict[str, Dict[str, object]] = {
    "nml": {"color": "#00bcd4", "linestyle": "--", "linewidth": 1.8, "desc": "normal ROI"},
    "abn": {"color": "#009688", "linestyle": "--", "linewidth": 1.8, "desc": "abnormal ROI"},
    "malg": {"color": "#ffeb3b", "linestyle": "-", "linewidth": 2.2, "desc": "malignancy"},
    "bmt": {"color": "#42a5f5", "linestyle": "-", "linewidth": 2.2, "desc": "benign thickening"},
    "stn": {"color": "#bdbdbd", "linestyle": ":", "linewidth": 2.0, "desc": "stone"},
}
GT_LABEL_ALIASES = {"mal": "malg", "ben": "bmt", "normal": "nml", "malignant": "malg", "benign": "bmt"}
# Preview overlay: gallbladder ROI shell + malignant lesion only (no stone / bmt).
GT_GALLBLADDER_MALIGNANT_LABELS = frozenset({"nml", "abn", "malg"})


def normalize_gt_label(label: str) -> str:
    key = str(label).lower().strip()
    return GT_LABEL_ALIASES.get(key, key)


def parse_gt_boxes(
    raw_bbs: Sequence[Sequence],
    *,
    include_labels: frozenset[str] | set[str] | None = None,
) -> List[Dict]:
    """Parse bbox_annot.json ``bbs`` entries into drawable GT box dicts."""
    out: List[Dict] = []
    for item in raw_bbs:
        if len(item) != 2:
            continue
        label, box = item
        if not isinstance(box, list) or len(box) != 4:
            continue
        norm = normalize_gt_label(str(label))
        if include_labels is not None and norm not in include_labels:
            continue
        style = GT_BOX_STYLES.get(norm, {"color": "#ffffff", "linestyle": "-", "linewidth": 1.5, "desc": norm})
        out.append(
            {
                "label": norm,
                "box_xyxy": [float(v) for v in box],
                "color": style["color"],
                "linestyle": style["linestyle"],
                "linewidth": style["linewidth"],
                "desc": style.get("desc", norm),
            }
        )
    return out


def get_gt_boxes_for_image(
    bbox_annot: Dict,
    image_name: str,
    *,
    include_labels: frozenset[str] | set[str] | None = None,
) -> List[Dict]:
    if image_name not in bbox_annot:
        return []
    return parse_gt_boxes(bbox_annot[image_name].get("bbs", []), include_labels=include_labels)


def draw_pred_gt_boxes_on_ax(
    ax: plt.Axes,
    record: Dict,
    gt_boxes: Sequence[Dict],
    *,
    cfg: ReprConfig,
) -> None:
    """Draw YOLO predicted ROI boxes (red/green) and GT annot boxes (by label color)."""
    for roi in record.get("roi_details", []):
        x0, y0, x1, y1 = roi["box_xyxy"]
        is_fallback = str(roi.get("source", "")) == "full_image_fallback"
        color = "#2ecc71" if is_fallback else "#ff3333"
        ax.add_patch(
            mpatches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=2.0,
                edgecolor=color,
                facecolor="none",
            )
        )
        conf = float(roi.get("conf", 0.0))
        pred_tag = cfg.class_tag(int(roi["pred"]))
        ax.text(
            x0 + 2,
            max(y0 - 3, 6),
            f"YOLO conf={conf:.2f}\n{pred_tag} P(mal)={roi['mal_prob']:.2f}",
            color=color,
            fontsize=7,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.55, edgecolor="none"),
        )

    for gt in gt_boxes:
        x0, y0, x1, y1 = gt["box_xyxy"]
        ax.add_patch(
            mpatches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=float(gt["linewidth"]),
                edgecolor=str(gt["color"]),
                linestyle=str(gt["linestyle"]),
                facecolor="none",
            )
        )
        ax.text(
            x1 - 2,
            y1 - 2,
            str(gt["label"]),
            color=str(gt["color"]),
            fontsize=7,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.45, edgecolor="none"),
        )


def render_pred_gt_gradcam_on_ax(
    ax: plt.Axes,
    image: Image.Image,
    record: Dict,
    gt_boxes: Sequence[Dict],
    cfg: ReprConfig,
    *,
    method: str,
    fold: int,
    alpha: float = 0.45,
) -> None:
    overlay = build_overlay_rgb(image, record, alpha=alpha)
    ax.imshow(overlay)
    draw_pred_gt_boxes_on_ax(ax, record, gt_boxes, cfg=cfg)

    case = record["case_type"]
    case_color = CASE_COLORS.get(case, "#ffffff")
    gt_tag = cfg.class_tag(record["gt_class"])
    pred_tag = cfg.class_tag(record["pred_class"])
    gt_labels = ", ".join(sorted({g["label"] for g in gt_boxes})) or "none"
    ax.set_title(
        f"{record['image_name']} | {method.upper()} fold={fold}\n"
        f"img GT={gt_tag} Fused={pred_tag} [{case}] | annot: {gt_labels}",
        fontsize=9,
        fontweight="bold",
        color=case_color,
    )
    ax.set_axis_off()


def render_pred_gt_gradcam_preview(
    images_records_gt: Sequence[Tuple[Image.Image, Dict, List[Dict]]],
    cfg: ReprConfig,
    *,
    method: str = "ae",
    fold: int = 0,
) -> plt.Figure:
    """Side-by-side preview: Grad-CAM + YOLO pred + GT annot boxes (no file save)."""
    n = len(images_records_gt)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 6.0), dpi=110)
    if n == 1:
        axes = [axes]

    used_gt_labels = set()
    for ax, (image, record, gt_boxes) in zip(axes, images_records_gt):
        render_pred_gt_gradcam_on_ax(ax, image, record, gt_boxes, cfg, method=method, fold=fold)
        used_gt_labels.update(g["label"] for g in gt_boxes)

    legend_handles = [
        mpatches.Patch(edgecolor="#ff3333", facecolor="none", linewidth=2, label="YOLO pred ROI"),
        mpatches.Patch(edgecolor="#2ecc71", facecolor="none", linewidth=2, label="YOLO full-image fallback"),
    ]
    for label in sorted(used_gt_labels):
        style = GT_BOX_STYLES.get(label, {"color": "#ffffff", "linestyle": "-", "desc": label})
        legend_handles.append(
            mpatches.Patch(
                edgecolor=str(style["color"]),
                facecolor="none",
                linewidth=2,
                linestyle=str(style["linestyle"]),
                label=f"GT {label} ({style.get('desc', label)})",
            )
        )
    fig.legend(handles=legend_handles, loc="lower center", ncol=min(7, len(legend_handles)), fontsize=8, framealpha=0.85)
    fig.suptitle(
        f"{method.upper()} Grad-CAM vs radiologist annotations (YOLO pipeline, display only)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.patch.set_facecolor("#1a1a1a")
    plt.tight_layout()
    return fig


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """ResNet50 layer4 inside RoiReprModel.backbone Sequential."""
    return model.backbone[7]


class RoiGradCAM:
    """Grad-CAM on RoiReprModel backbone layer4 for a single ROI tensor."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.target_layer = get_gradcam_target_layer(model)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: List = []

    def _save_activation(self, _module, _inputs, output) -> None:
        self._activations = output

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0]

    def attach(self) -> None:
        self.detach()
        self._handles.append(self.target_layer.register_forward_hook(self._save_activation))
        self._handles.append(self.target_layer.register_full_backward_hook(self._save_gradient))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._activations = None
        self._gradients = None

    def __call__(self, x: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        was_training = self.model.training
        self.model.eval()

        out = self.model(x)
        logits = out["logits"]
        score = logits[0, int(target_class)]
        self.model.zero_grad(set_to_none=True)
        score.backward()

        assert self._activations is not None and self._gradients is not None
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

        if was_training:
            self.model.train()
        return cam_np


def _denorm_roi_tensor(tensor: torch.Tensor, cfg: ReprConfig) -> np.ndarray:
    mean, std = cfg.imagenet_norm()
    mean_t = torch.tensor(mean, device=tensor.device).view(3, 1, 1)
    std_t = torch.tensor(std, device=tensor.device).view(3, 1, 1)
    img = tensor.squeeze(0) * std_t + mean_t
    img = img.clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
    return img


def map_cam_to_crop(cam: np.ndarray, crop_w: int, crop_h: int, cfg: ReprConfig) -> np.ndarray:
    """Map 7x7 CAM (on 224x224 center crop) back to original crop pixel size."""
    cam_t = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0)
    cam_224 = F.interpolate(cam_t, size=(cfg.roi_size, cfg.roi_size), mode="bilinear", align_corners=False)
    pad = (cfg.roi_resize - cfg.roi_size) // 2
    cam_256 = F.pad(cam_224, (pad, pad, pad, pad))
    cam_crop = F.interpolate(cam_256, size=(crop_h, crop_w), mode="bilinear", align_corners=False)
    out = cam_crop.squeeze().numpy()
    return (out - out.min()) / (out.max() - out.min() + 1e-8)


def overlay_heatmap(
    base_rgb: np.ndarray,
    heatmap: np.ndarray,
    box_xyxy: Sequence[float],
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    out = base_rgb.copy()
    x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, out.shape[1]), min(y1, out.shape[0])
    if x1 <= x0 or y1 <= y0:
        return out

    region = out[y0:y1, x0:x1]
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.shape[0] != region.shape[0] or hm.shape[1] != region.shape[1]:
        hm_img = Image.fromarray((hm * 255).astype(np.uint8))
        hm_img = hm_img.resize((region.shape[1], region.shape[0]), Image.BILINEAR)
        hm = np.asarray(hm_img, dtype=np.float32) / 255.0

    colored = plt.cm.jet(hm)[..., :3]
    blended = (1.0 - alpha) * region.astype(np.float32) / 255.0 + alpha * colored
    out[y0:y1, x0:x1] = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return out


@torch.no_grad()
def predict_roi_detailed(
    model: nn.Module,
    transform: transforms.Compose,
    image: Image.Image,
    box: Sequence[float],
    device: torch.device,
    cfg: ReprConfig,
) -> Dict:
    w, h = image.size
    x0, y0, x1, y1 = sanitize_box_xyxy(box, width=w, height=h)
    crop = image.crop((x0, y0, x1, y1))
    tensor = transform(crop).unsqueeze(0).to(device)
    logits = model(tensor)["logits"]
    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred = int(torch.argmax(logits, dim=1).item())
    mal_prob = float(probs[cfg.malignant_id].item())
    return {
        "box_xyxy": [float(x0), float(y0), float(x1), float(y1)],
        "pred": pred,
        "mal_prob": mal_prob,
        "probs": probs.detach().cpu().numpy().tolist(),
        "crop": crop,
        "tensor": tensor,
        "crop_size": (int(x1 - x0), int(y1 - y0)),
    }


def explain_image(
    image: Image.Image,
    image_name: str,
    gt_class: int,
    *,
    image_path: Path,
    yolo_model,
    classifier: nn.Module,
    gradcam: RoiGradCAM,
    transform: transforms.Compose,
    cfg: ReprConfig,
    device: torch.device,
    eval_score_threshold: float = 0.5,
    nms_iou_threshold: float = 0.5,
    max_detections: int = 5,
    imgsz: int = 800,
    no_box_fallback: NoBoxFallback = "full_image",
    gradcam_target: Literal["pred", "malignant"] = "pred",
) -> Dict:
    w, h = image.size
    gt_class = int(gt_class)

    box_items = list_pred_boxes_for_image(
        yolo_model,
        Path(image_path),
        eval_score_threshold=eval_score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
        imgsz=imgsz,
    )
    fallback_used = False
    if not box_items:
        if no_box_fallback == "skip":
            return {"skipped": True, "image_name": image_name}
        if no_box_fallback == "full_image":
            box_items = [
                {
                    "box_xyxy": [0.0, 0.0, float(w), float(h)],
                    "conf": 0.0,
                    "source": "full_image_fallback",
                }
            ]
            fallback_used = True

    roi_details: List[Dict] = []
    for entry in box_items:
        detail = predict_roi_detailed(
            classifier,
            transform,
            image,
            entry["box_xyxy"],
            device,
            cfg,
        )
        target_cls = detail["pred"] if gradcam_target == "pred" else cfg.malignant_id
        cam = gradcam(detail["tensor"], target_class=target_cls)
        crop_w, crop_h = detail["crop_size"]
        cam_crop = map_cam_to_crop(cam, crop_w, crop_h, cfg)
        roi_details.append(
            {
                **entry,
                "pred": detail["pred"],
                "mal_prob": detail["mal_prob"],
                "probs": detail["probs"],
                "box_xyxy": detail["box_xyxy"],
                "cam": cam,
                "cam_crop": cam_crop,
                "crop": detail["crop"],
                "tensor": detail["tensor"],
                "gradcam_target_class": int(target_cls),
            }
        )

    roi_preds = [int(r["pred"]) for r in roi_details]
    if cfg.task_mode == "binary":
        fused = fuse_binary_any_positive(roi_preds, malignant_id=cfg.malignant_id) if roi_preds else 0
    else:
        fused = int(max(roi_preds, key=lambda p: (p == cfg.malignant_id, p))) if roi_preds else 0

    case = binary_case_type(gt_class, fused)
    return {
        "image_name": image_name,
        "gt_class": gt_class,
        "pred_class": fused,
        "case_type": case,
        "fallback_used": fallback_used,
        "roi_details": roi_details,
        "fusion_rule": FUSION_RULE,
        "skipped": False,
    }


def build_overlay_rgb(
    image: Image.Image,
    record: Dict,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    """Main-panel overlay (heatmap on full image) for compact multi-method grids."""
    base = np.asarray(image.convert("RGB"))
    overlay = base.copy()
    for roi in record.get("roi_details", []):
        overlay = overlay_heatmap(overlay, roi["cam_crop"], roi["box_xyxy"], alpha=alpha)
    return overlay


def compare_methods_on_sample(
    image: Image.Image,
    image_name: str,
    gt_class: int,
    image_path: Path,
    methods: Sequence[str],
    *,
    loader,
    yolo_model,
    device: torch.device,
    eval_score_threshold: float = 0.5,
    nms_iou_threshold: float = 0.5,
    max_detections: int = 5,
    imgsz: int = 800,
    no_box_fallback: NoBoxFallback = "full_image",
    gradcam_target: Literal["pred", "malignant"] = "pred",
) -> Tuple[Dict[str, Dict], Dict[str, ReprConfig]]:
    """Run explain_image for each method on the same image (shared YOLO boxes per call)."""
    method_records: Dict[str, Dict] = {}
    method_cfgs: Dict[str, ReprConfig] = {}
    for method in methods:
        m_cfg, m_model, m_transform, m_gradcam = loader(method)
        m_gradcam.attach()
        try:
            rec = explain_image(
                image,
                image_name,
                gt_class,
                image_path=image_path,
                yolo_model=yolo_model,
                classifier=m_model,
                gradcam=m_gradcam,
                transform=m_transform,
                cfg=m_cfg,
                device=device,
                eval_score_threshold=eval_score_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections=max_detections,
                imgsz=imgsz,
                no_box_fallback=no_box_fallback,
                gradcam_target=gradcam_target,
            )
            method_records[method] = rec
            method_cfgs[method] = m_cfg
        finally:
            m_gradcam.detach()
    return method_records, method_cfgs


def render_multi_method_comparison_figure(
    image: Image.Image,
    method_records: Dict[str, Dict],
    method_cfgs: Dict[str, ReprConfig],
    *,
    image_name: str,
    gt_class: int,
    fold: int,
    alpha: float = 0.45,
) -> plt.Figure:
    """One row: same sample, all methods side-by-side (compact overlay panels)."""
    methods = list(method_records.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 5.2), dpi=120)
    if n == 1:
        axes = [axes]

    gt_tag = method_cfgs[methods[0]].class_tag(gt_class)
    for ax, method in zip(axes, methods):
        rec = method_records[method]
        m_cfg = method_cfgs[method]
        overlay = build_overlay_rgb(image, rec, alpha=alpha)
        ax.imshow(overlay)
        ax.set_axis_off()

        for roi in rec.get("roi_details", []):
            x0, y0, x1, y1 = roi["box_xyxy"]
            is_fallback = str(roi.get("source", "")) == "full_image_fallback"
            color = "#2ecc71" if is_fallback else "#ff3333"
            rect = mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, linewidth=1.8, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)

        pred_tag = m_cfg.class_tag(rec["pred_class"])
        case = rec["case_type"]
        case_color = CASE_COLORS.get(case, "#ffffff")
        ax.set_title(
            f"{method.upper()}\nFused={pred_tag} [{case}]",
            fontsize=10,
            fontweight="bold",
            color=case_color,
        )

    fig.suptitle(
        f"{image_name}  |  GT={gt_tag}  |  fold={fold}  |  5-method Grad-CAM",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.patch.set_facecolor("#1a1a1a")
    plt.tight_layout()
    return fig


def save_multi_method_comparison_figure(
    image: Image.Image,
    method_records: Dict[str, Dict],
    method_cfgs: Dict[str, ReprConfig],
    out_path: Path,
    *,
    image_name: str,
    gt_class: int,
    fold: int,
) -> Path:
    fig = render_multi_method_comparison_figure(
        image,
        method_records,
        method_cfgs,
        image_name=image_name,
        gt_class=gt_class,
        fold=fold,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def render_samples_methods_grid(
    batch_results: Sequence[Dict],
    methods: Sequence[str],
    imgs_dir: Path,
    *,
    fold: int,
    class_tag_fn,
    alpha: float = 0.45,
) -> plt.Figure:
    """N samples x M methods grid; methods order left-to-right (AE first)."""
    n_samples = len(batch_results)
    n_methods = len(methods)
    fig, axes = plt.subplots(
        n_samples, n_methods, figsize=(4.2 * n_methods, 4.0 * n_samples), dpi=120
    )
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_methods == 1:
        axes = np.expand_dims(axes, axis=1)

    for row, item in enumerate(batch_results):
        image = Image.open(imgs_dir / item["image_name"]).convert("RGB")
        gt_tag = class_tag_fn(item["gt_class"])
        for col, method in enumerate(methods):
            ax = axes[row, col]
            rec = item["method_records"][method]
            m_cfg = item["method_cfgs"][method]
            overlay = build_overlay_rgb(image, rec, alpha=alpha)
            ax.imshow(overlay)
            ax.set_axis_off()

            for roi in rec.get("roi_details", []):
                x0, y0, x1, y1 = roi["box_xyxy"]
                is_fallback = str(roi.get("source", "")) == "full_image_fallback"
                color = "#2ecc71" if is_fallback else "#ff3333"
                rect = mpatches.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0, linewidth=1.5, edgecolor=color, facecolor="none"
                )
                ax.add_patch(rect)

            if row == 0:
                ax.set_title(method.upper(), fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{item['image_name']}\nGT={gt_tag}", fontsize=9)

            pred_tag = m_cfg.class_tag(rec["pred_class"])
            case = rec["case_type"]
            ax.text(
                0.02,
                0.98,
                f"{pred_tag}\n[{case}]",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
            )

    fig.suptitle(
        f"{n_samples} samples x {n_methods} methods Grad-CAM (fold={fold}, AE=left)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.patch.set_facecolor("#1a1a1a")
    plt.tight_layout()
    return fig


def save_samples_methods_grid(
    batch_results: Sequence[Dict],
    methods: Sequence[str],
    imgs_dir: Path,
    out_path: Path,
    *,
    fold: int,
    class_tag_fn,
) -> Path:
    fig = render_samples_methods_grid(
        batch_results, methods, imgs_dir, fold=fold, class_tag_fn=class_tag_fn
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=120)
    plt.close(fig)
    return out_path


def run_multi_method_comparison_batch(
    samples: Sequence[Tuple[str, int]],
    methods: Sequence[str],
    *,
    loader,
    yolo_model,
    device: torch.device,
    imgs_dir: Path,
    output_dir: Path,
    fold: int,
    map_label_fn,
    class_tag_fn=None,
    max_samples: int | None = None,
    eval_score_threshold: float = 0.5,
    nms_iou_threshold: float = 0.5,
    max_detections: int = 5,
    imgsz: int = 800,
    no_box_fallback: NoBoxFallback = "full_image",
    gradcam_target: Literal["pred", "malignant"] = "pred",
) -> Dict:
    """Run comparison batch.

    Output layout under ``output_dir``:
      panels/{image_stem}_{method}.jpg   # N_samples * N_methods individual figures
      summary_{N}x{M}_grid.jpg           # one combined grid for notebook inspection
    """
    output_dir = Path(output_dir)
    panels_dir = output_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    if class_tag_fn is None:
        class_tag_fn = lambda cls_id: str(cls_id)

    results: List[Dict] = []
    for i, (image_name, raw_cls) in enumerate(samples):
        if max_samples is not None and i >= max_samples:
            break
        image_path = imgs_dir / image_name
        if not image_path.is_file():
            continue

        image = Image.open(image_path).convert("RGB")
        gt_class = int(map_label_fn(raw_cls))
        method_records, method_cfgs = compare_methods_on_sample(
            image,
            image_name,
            gt_class,
            image_path,
            methods,
            loader=loader,
            yolo_model=yolo_model,
            device=device,
            eval_score_threshold=eval_score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
            imgsz=imgsz,
            no_box_fallback=no_box_fallback,
            gradcam_target=gradcam_target,
        )

        panel_paths: Dict[str, str] = {}
        stem = Path(image_name).stem
        for method in methods:
            rec = method_records[method]
            m_cfg = method_cfgs[method]
            panel_path = panels_dir / f"{stem}_{method}.jpg"
            save_explanation_figure(image, rec, panel_path, m_cfg, method=method, fold=fold)
            panel_paths[method] = str(panel_path)

        results.append(
            {
                "image_name": image_name,
                "gt_class": gt_class,
                "method_records": method_records,
                "method_cfgs": method_cfgs,
                "panel_paths": panel_paths,
            }
        )

    n_samples = len(results)
    n_methods = len(methods)
    summary_path = output_dir / f"summary_{n_samples}x{n_methods}_grid.jpg"
    if results:
        save_samples_methods_grid(
            results,
            methods,
            imgs_dir,
            summary_path,
            fold=fold,
            class_tag_fn=class_tag_fn,
        )

    return {
        "results": results,
        "panels_dir": str(panels_dir),
        "summary_grid_path": str(summary_path) if results else None,
        "n_samples": n_samples,
        "n_methods": n_methods,
        "n_panels": n_samples * n_methods,
    }


def render_explanation_figure(
    image: Image.Image,
    record: Dict,
    cfg: ReprConfig,
    *,
    method: str,
    fold: int,
    alpha: float = 0.45,
) -> plt.Figure:
    base = np.asarray(image.convert("RGB"))
    overlay = base.copy()
    roi_details = record["roi_details"]

    for roi in roi_details:
        overlay = overlay_heatmap(overlay, roi["cam_crop"], roi["box_xyxy"], alpha=alpha)

    n_roi = len(roi_details)
    n_cols = max(n_roi, 1)
    fig_h = 4.8 + 2.2 * (1 if n_roi else 0)
    fig = plt.figure(figsize=(max(12, 3.2 * n_cols), fig_h), dpi=120)
    gs = fig.add_gridspec(2, n_cols, height_ratios=[3.0, 1.4] if n_roi else [1.0, 0.0], hspace=0.28, wspace=0.12)

    ax_main = fig.add_subplot(gs[0, :])
    ax_main.imshow(overlay)
    ax_main.set_axis_off()

    case = record["case_type"]
    case_color = CASE_COLORS.get(case, "#ffffff")
    gt_tag = cfg.class_tag(record["gt_class"])
    pred_tag = cfg.class_tag(record["pred_class"])
    title = (
        f"{record['image_name']}  |  method={method.upper()} fold={fold}  |  "
        f"GT={gt_tag}  Fused={pred_tag}  [{case}]"
    )
    ax_main.set_title(title, fontsize=11, fontweight="bold", color=case_color, pad=10)

    for idx, roi in enumerate(roi_details):
        x0, y0, x1, y1 = roi["box_xyxy"]
        is_fallback = str(roi.get("source", "")) == "full_image_fallback"
        box_color = "#2ecc71" if is_fallback else "#ff3333"
        rect = mpatches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=2.0,
            edgecolor=box_color,
            facecolor="none",
        )
        ax_main.add_patch(rect)

        conf = float(roi.get("conf", 0.0))
        pred_tag = cfg.class_tag(int(roi["pred"]))
        label = (
            f"#{idx + 1} conf={conf:.2f}\n"
            f"pred={pred_tag}\n"
            f"P(mal)={roi['mal_prob']:.3f}"
        )
        ax_main.text(
            x0 + 3,
            max(y0 - 4, 8),
            label,
            color=box_color,
            fontsize=8,
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55, edgecolor="none"),
        )

    footer = (
        f"fusion: {record['fusion_rule']}"
        + ("  |  full_image_fallback" if record.get("fallback_used") else "")
        + f"  |  Grad-CAM target: class {roi_details[0]['gradcam_target_class']}" if roi_details else ""
    )
    fig.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=9, color="#dddddd")

    if n_roi:
        for idx, roi in enumerate(roi_details):
            ax = fig.add_subplot(gs[1, idx])
            crop_rgb = np.asarray(roi["crop"].convert("RGB"))
            crop_hm = plt.cm.jet(roi["cam_crop"])[..., :3]
            ax.imshow(0.55 * crop_rgb / 255.0 + 0.45 * crop_hm)
            ax.set_title(
                f"ROI #{idx + 1}: {cfg.class_tag(roi['pred'])} | P(mal)={roi['mal_prob']:.3f}",
                fontsize=9,
            )
            ax.set_axis_off()

    fig.patch.set_facecolor("#1a1a1a")
    return fig


def save_explanation_figure(
    image: Image.Image,
    record: Dict,
    out_path: Path,
    cfg: ReprConfig,
    *,
    method: str,
    fold: int,
) -> Path:
    fig = render_explanation_figure(image, record, cfg, method=method, fold=fold)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def run_explanation_batch(
    samples: Sequence[Tuple[str, int]],
    *,
    cfg: ReprConfig,
    classifier: nn.Module,
    yolo_model,
    device: torch.device,
    output_dir: Path,
    fold: int,
    imgs_dir: Path,
    max_images: int | None = None,
    case_filter: Sequence[CaseType] | None = None,
    eval_score_threshold: float = 0.5,
    nms_iou_threshold: float = 0.5,
    max_detections: int = 5,
    imgsz: int = 800,
    no_box_fallback: NoBoxFallback = "full_image",
    gradcam_target: Literal["pred", "malignant"] = "pred",
) -> List[Dict]:
    from roi_repr.data.splits import build_roi_transform

    transform = build_roi_transform(cfg)
    gradcam = RoiGradCAM(classifier)
    gradcam.attach()

    output_dir = Path(output_dir)
    records: List[Dict] = []
    try:
        for i, (image_name, raw_cls) in enumerate(samples):
            if max_images is not None and i >= max_images:
                break

            image_path = imgs_dir / image_name
            if not image_path.is_file():
                continue

            image = Image.open(image_path).convert("RGB")
            gt_class = int(cfg.map_label(raw_cls))

            record = explain_image(
                image,
                image_name,
                gt_class,
                image_path=image_path,
                yolo_model=yolo_model,
                classifier=classifier,
                gradcam=gradcam,
                transform=transform,
                cfg=cfg,
                device=device,
                eval_score_threshold=eval_score_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections=max_detections,
                imgsz=imgsz,
                no_box_fallback=no_box_fallback,
                gradcam_target=gradcam_target,
            )
            if record.get("skipped"):
                continue

            if case_filter and record["case_type"] not in case_filter:
                continue

            out_name = f"{Path(image_name).stem}_{record['case_type']}_{cfg.method}.jpg"
            out_path = output_dir / out_name
            save_explanation_figure(
                image,
                record,
                out_path,
                cfg,
                method=cfg.method,
                fold=fold,
            )
            record["vis_path"] = str(out_path)
            records.append(record)
    finally:
        gradcam.detach()

    return records
