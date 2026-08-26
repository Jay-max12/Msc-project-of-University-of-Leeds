# ROI Representation Learning（统一表征学习实验）

本目录集中 **五种对照实验**，共享同一分类路径，仅损失函数与训练样本组织方式不同：

| 方法 | 额外模块 | 损失 |
|------|----------|------|
| `cls` | 无 | CE |
| `ae` | Decoder | CE + Reconstruction |
| `vae` | Decoder + KL | CE + Reconstruction + KL |
| `siamese` | Pair sampling | CE + Contrastive |
| `triplet` | Triplet sampling | CE + Triplet |

## 统一架构

```
ROI image (GT bbox crop)
  → Resize(256) → CenterCrop(224) → ImageNet Normalize
  → ResNet50 backbone (ImageNet pretrained)
  → Global feature 2048-d
  → Embedding head: Linear(2048 → 128)  →  z
  → Classifier head: Linear(128 → num_classes)
```

AE/VAE 在 `z` 上额外接 **共享 Decoder** 做重建；Siamese/Triplet 在 `z` 上做度量学习。

## 目录结构

```
roi_repr/
  config.py          # 全局超参与实验常量
  data/              # 数据集、划分、pair/triplet 采样
  models/            # 共享 backbone / heads / 五种模型
  losses/            # CE、recon、KL、contrastive、triplet
  training/          # 训练循环、评估、早停、checkpoint
  train.py           # CLI 入口：python -m roi_repr.train --method cls
  experiments.ipynb  # Notebook 入口（对比五种方法）
  train_roi_repr.slurm   # 本目录内 slurm 副本
```

## 与 GBCNET_Plus 根目录的关系

| 位置 | 用途 | 状态 |
|------|------|------|
| `roi_repr/` | **表征学习五方法对照（主实验）** | 当前维护 |
| `GBCU/` | 数据集 | 共用 |
| `detr_roi/` | DETR 检测 | 独立子项目 |
| `train_yolov11_onestep.py` | YOLO 一步检测+分类 | 独立基线 |
| `train_resnet50_roi_classifier.py` | 旧 ResNet50 三分类 | 遗留，可被 `roi_repr --method cls` 替代 |
| `VAE_latent_classification*.ipynb` | 旧 ResNet18 VAE 实验 | 遗留，架构不一致 |
| `Siamese_Triplet_*.ipynb` | 旧 Siamese/Triplet | 遗留，请用本目录 |

**建议**：新实验只改 `roi_repr/`；检测与 pipeline 评估脚本保留在根目录。

## 快速开始

```bash
cd /path/to/GBCNET_Plus

# 单方法
python -m roi_repr.train --method cls --task_mode multiclass
python -m roi_repr.train --method vae --task_mode multiclass
# 二分类（normal+benign → non_malignant, malignant → malignant）
python -m roi_repr.train --method all --task_mode binary

# 跑全部五种（依次训练，3-fold CV）
python -m roi_repr.train --method all --task_mode multiclass
```

输出：`outputs/roi_repr/<task_mode>/<method>_cv3/fold{0,1,2}/` + `cv_results.json`

## 在集群上提交（Slurm）

本地 login 节点适合调试；**正式训练（尤其 `--method all` 五种依次跑）应提交 GPU 作业**。

项目根目录已提供 slurm 脚本（与其他实验一致）：

```bash
cd /path/to/GBCNET_Plus

# 五种方法全部跑（默认 multiclass，约需数小时）
sbatch train_roi_repr.slurm

# 只跑单一方法
METHOD=cls sbatch train_roi_repr.slurm
METHOD=vae sbatch train_roi_repr.slurm

# 二分类主实验
TASK_MODE=binary METHOD=all sbatch train_roi_repr.slurm
```

日志：`GBCNET_Plus/logs/roi_repr_<jobid>.out` / `.err`

> Slurm 的 `#SBATCH -o/-e` 必须用**绝对路径**；相对路径 `logs/...` 会在计算节点 spool 目录（如 `/var/spool/slurmd/`）下建目录失败。

本地调试（需 GPU）：

```bash
python -m roi_repr.train --method all --task_mode multiclass
```

## 严格一致项

- ROI：GT `bbox_annot.json`（nml/abn 优先）
- 划分：**3-fold 分层 CV** 在官方 `train` 上；`test` 固定 hold-out（无独立 validation 集）
- 预处理、ResNet50 预训练、embedding=128、分类头结构
- 优化器 SGD、评估指标 acc / bal_acc / mal_sens / mal_spec / select
- 选模：每折 fold_val `select = 0.5×(sens+spec)`；**best checkpoint 仅从第 30 epoch 起保存**
- 早停：第 30 epoch 后 patience=15
