# GBCNET_Plus 项目布局说明

## 实验重心：ROI 表征学习 → 分类

**主目录**：[`roi_repr/`](../roi_repr/README.md)

五种对照实验共享同一分类路径（ResNet50 → 128-d → Linear classifier），仅损失与采样不同：

| 方法 | 说明 |
|------|------|
| `cls` | 纯分类基线 |
| `ae` | 自编码器 + 分类 |
| `vae` | 变分自编码器 + 分类 |
| `siamese` | 对比学习 + 分类 |
| `triplet` | 三元组 + 分类 |

```bash
python -m roi_repr.train --method all --task_mode multiclass

# 集群提交
sbatch train_roi_repr.slurm
```

---

## 目录分工

```
GBCNET_Plus/
├── GBCU/                          # 数据集（imgs, bbox, train/test split）
├── roi_repr/                      # ★ 表征学习五方法 + 消融 + Grad-CAM
│   ├── train.py                   # 五方法训练入口
│   ├── gradcam.py                 # Grad-CAM 可视化模块
│   ├── eval_oracle_roi_ablation.py
│   ├── eval_yolo_roi_pipeline.py
│   └── eval_multi_roi.py
├── detr_roi/                      # DETR ROI 检测（独立基线）
├── train_yolov11_onestep.py       # YOLO 一步检测+二分类
├── vae_yolo_roi.py                # YOLO ROI 裁剪工具（pipeline / Grad-CAM 共用）
├── train_resnet50_roi_classifier.py  # ROI 工具函数（被 roi_repr 引用，非独立训练入口）
├── YOLO_roi_gradcam_explain.ipynb # Grad-CAM 可视化 notebook
├── evaluate_fasterrcnn_resnet_pipeline.py  # 检测→分类 两阶段 pipeline
└── outputs/
    ├── roi_repr/                  # ★ 五方法 + 消融实验结果
    ├── yolo_roi_gradcam/          # Grad-CAM 可视化输出
    ├── yolo_roi_repr_pipeline_single_maxbox/  # YOLO→五方法 pipeline 评估
    ├── yolov11_onestep_run1/      # YOLO 检测权重
    └── detr_roi_run1/             # DETR 检测权重
```

---

## 关键 Slurm 脚本

| 脚本 | 用途 |
|------|------|
| `train_roi_repr.slurm` | 五方法表征学习训练 |
| `run_yolo_roi_pipeline.slurm` | YOLO ROI → 五方法 pipeline 评估 |
| `run_oracle_roi_ablation.slurm` | Oracle ROI 消融实验 |
| `run_yolov11_onestep.slurm` | YOLO 一步检测训练 |

---

## 数据与 ROI

当前表征学习实验默认使用 **GT ROI**（`bbox_annot.json`，nml/abn 优先）。
YOLO 预测 ROI 用于 pipeline 评估与 Grad-CAM，不应与 GT ROI 结果直接混比。
