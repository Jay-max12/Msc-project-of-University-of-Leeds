# 旧实验结果溯源审计报告

> **Public-release note:** This is a historical protocol audit. Referenced checkpoints,
> raw outputs, logs, and the fixed split manifest are intentionally absent from the
> public repository because they are generated artefacts or contain information from
> the separately licensed GBCU dataset. Recreate authorised local manifests using the
> instructions in [`DATA.md`](DATA.md).

**审计日期：** 2026-07-25
**审计范围：** `outputs/roi_repr/binary/{ae,cls,vae,siamese,triplet}_cv3`、旧 YOLO one-step、旧 YOLO+AE two-step
**方法：** 仅阅读代码、输出 JSON、Slurm 日志、checkpoint 权重结构、历史对话记录；**未修改代码、未重新训练**

---

## 执行摘要

| 实验组 | 完成时间 | Job ID | 核心结论 |
|--------|----------|--------|----------|
| 五方法 `*_cv3`（GT ROI） | 2026-06-25 16:20–17:47 | **未知**（无 Slurm 日志） | 一次 `--method all` 顺序跑完；**线性分类头**、**无增强**；划分算法与现 manifest 一致但**当时未落盘** |
| `ae_cv3_fixed`（对照） | 2026-07-24 15:55 | **6824103** | **MLP 分类头**、无增强、**显式固定 manifest** |
| 旧 YOLO one-step (`run1`) | 2026-05-11 | **未知** | 非 CV；**训练时 val=官方 test.txt**（存在 test 泄漏） |
| 旧 YOLO+AE two-step | 2026-07-07 | **未知** | 单 YOLO `run1` + 旧 `*_cv3` 线性头分类器 |
| 新 YOLO one-step cv3 | 2026-07-25 15:28 | **6851445** | 3-fold CV + 固定 manifest；与 `ae_cv3_fixed` 协议对齐 |
| 新 YOLO+AE two-step | 2026-07-25 15:36 | **6852178** | per-fold YOLO + `ae_cv3_fixed` 分类器 |

**最重要差异：** 旧 `*_cv3` 使用 `Linear(128→2)` 分类头；`ae_cv3_fixed` 使用 `MLP(128→64→2)` 分类头（checkpoint 已核实）。旧 YOLO `run1` 在训练阶段把官方 test 当作 Ultralytics 的 val 集。

---

## 1. 五方法旧结果（`outputs/roi_repr/binary/*_cv3`）

以下五项目共享同一训练协议，仅 `method` 与损失项不同。汇总见 `outputs/roi_repr/binary/all_methods_cv3.json`（2026-06-25 17:47，与 `triplet_cv3` 同时写入，佐证为单次 `--method all` 批量实验）。

### 1.1 各方法 Test 指标（3-fold 均值 ± 标准差）

| 方法 | test_acc | test_bal_acc | test_mal_sens | test_mal_spec | 结果目录 |
|------|----------|--------------|---------------|---------------|----------|
| **cls** | 0.781 ± 0.033 | 0.749 ± 0.043 | 0.643 ± 0.086 | 0.854 ± 0.031 | `outputs/roi_repr/binary/cls_cv3/` |
| **ae** | 0.828 ± 0.033 | 0.807 ± 0.027 | 0.738 ± 0.041 | 0.875 ± 0.054 | `outputs/roi_repr/binary/ae_cv3/` |
| **vae** | 0.798 ± 0.021 | 0.765 ± 0.037 | 0.659 ± 0.096 | 0.871 ± 0.031 | `outputs/roi_repr/binary/vae_cv3/` |
| **siamese** | 0.798 ± 0.029 | 0.731 ± 0.042 | 0.516 ± 0.084 | 0.946 ± 0.007 | `outputs/roi_repr/binary/siamese_cv3/` |
| **triplet** | 0.762 ± 0.038 | 0.675 ± 0.042 | 0.397 ± 0.069 | 0.954 ± 0.038 | `outputs/roi_repr/binary/triplet_cv3/` |

每 fold 明细：`outputs/roi_repr/binary/<method>_cv3/fold{0,1,2}/results.json`
汇总：`outputs/roi_repr/binary/<method>_cv3/cv_results.json`

### 1.2 代码版本与训练入口

| 项目 | 内容 |
|------|------|
| **推断入口** | `python -m roi_repr.train --method all --task_mode binary`（见 `roi_repr/README.md` 与 `all_methods_cv3.json` 时间戳链） |
| **备选入口** | 根目录 `train_roi_repr.slurm`（默认 `METHOD=all`，但默认 `TASK_MODE=multiclass`，与本次 binary 结果不完全吻合，故更可能是直接 CLI） |
| **当时代码特征** | `ClassifierHead` 为 **单层线性** `128→num_classes`（checkpoint 中键名为 `classifier.fc.*`，已与 `ae_cv3_fixed` 的 `classifier.net.*` 对比核实） |
| **conda 环境** | `~/.conda/envs/myenv`（后续有日志的作业均用此环境，旧实验推断相同） |
| **Job ID** | **无留存日志**。`logs/` 中最早 roi_repr 日志为 2026-07-23 的 `roi_repr_6797734`（`ae_cv3_hflip`），不包含本次五方法运行 |

### 1.3 Train / Validation / Test 划分

```
GBCU 官方划分
├── train.txt  → 1133 张（有 GT bbox）
└── test.txt   → 122 张（固定 hold-out，各 fold 共用）

在 train 上做 3-fold 分层 CV（stratify=binary, seed=42）：
  fold k: train (~755) + fold_val (~378)
  test: 122（始终不参与训练与 checkpoint 选择）
```

- **fold 规模**（与 `ae_cv3_fixed` 日志一致）：fold0 755/378，fold1 755/378，fold2 756/377，test 122
- **分层标签**：`normal+benign → non_malignant(0)`，`malignant → malignant(1)`

### 1.4 是否使用固定 `cv_split` manifest

| 问题 | 结论 |
|------|------|
| 当时是否读取 `GBCU/cv_splits/cv3_seed42_binary.json`？ | **未能从日志证实**。该文件 **物理创建时间为 2026-07-24 15:04**，晚于旧实验一个月 |
| 划分是否与现 manifest 一致？ | **是**。用当前 `generate_cv_split_manifest(seed=42)` 重新生成，fold0 train 与 test 样本集合与 manifest **完全一致** |
| 语义 | 旧实验使用 **相同确定性 StratifiedKFold 算法**，但当时 manifest **可能仅在内存/运行时生成、未持久化**；2026-07-24 起才固化为共享 JSON |

### 1.5 Seed

- **全局 seed = 42**（`ReprConfig.seed`，CLI 默认 `--seed 42`）
- CV 划分：`StratifiedKFold(..., random_state=42)`
- PyTorch / NumPy / Python random 在 `run_experiment()` 开头 `set_seed(42)`

### 1.6 数据增强

- **训练：无**（`Resize(256) → CenterCrop(224) → Normalize`，确定性变换）
- **fold_val / test：无**
- 历史对话（2026-07-23）明确记录：在引入 `train_hflip_prob` 之前，**全部 roi_repr 实验均无随机增强**
- 后续对照：`ae_cv3_hflip`（Job **6797734**）才首次引入 `hflip p=0.5`，且输出目录为 `ae_cv3_hflip`，**不是**本报告审计的 `ae_cv3`

### 1.7 Embedding、损失权重、学习率

| 超参 | cls | ae | vae | siamese | triplet |
|------|-----|----|----|---------|---------|
| backbone | ResNet50 (ImageNet pretrained) | 同左 | 同左 | 同左 | 同左 |
| embedding_dim | **128** | 128 | 128 | 128 | 128 |
| 分类头 | **Linear(128→2)** | 同左 | 同左 | 同左 | 同左 |
| 主损失 | CE | CE + 0.3·L1(recon) | CE + 0.3·L1(recon) + **1e-5·KL** | CE + **0.5·contrastive** | CE + **0.5·triplet** |
| λ_cls | 1.0（当时无 CLI，等价于不加权 CE） | 1.0 | 1.0 | 1.0 | 1.0 |
| margin (siamese/triplet) | — | — | — | 1.0 | 1.0 |
| lr | **0.005** (SGD) | 0.005 | 0.005 | 0.005 | 0.005 |
| momentum / weight_decay | 0.9 / 5e-4 | 同左 | 同左 | 同左 | 同左 |
| lr scheduler | StepLR(step=5, γ=0.9) | 同左 | 同左 | 同左 | 同左 |
| batch_size / epochs | 16 / 100 | 同左 | 同左 | 同左 | 同左 |
| ROI 输入 | GT bbox crop, roi_resize=256, roi_size=224 | 同左 | 同左 | 同左 | 同左 |

### 1.8 Early stopping 与 checkpoint 选择

（`roi_repr/training/trainer.py`，旧版与现版规则相同）

1. 每 epoch 在 **fold_val** 上评估
2. 选模分数：`select = 0.5 × (mal_sens + mal_spec)`（`checkpoint_score()`）
3. **仅从 epoch ≥ 30** 起保存 `best_model.pth`
4. **patience = 15**：连续 15 epoch 无提升则 early stop
5. 用 best checkpoint 在 **test** 上做一次评估（test **不参与**选模）

### 1.9 Test 是否参与模型选择或调参

- **roi_repr 五方法：否**。test 仅最终报告；选模只看 fold_val
- 五方法之间 **无** 基于 test 的超参搜索；超参均为 `ReprConfig` 默认值
- **注意**：旧结果 **不能** 与之后基于 test 反馈做的 AE 超参扫描（`ae_cv3_cls*` 等）混为一谈

### 1.10 与 `ae_cv3_fixed` 的差异

| 维度 | 旧 `ae_cv3` 等 | `ae_cv3_fixed` (Job **6824103**) |
|------|----------------|----------------------------------|
| 分类头 | Linear `128→2` | MLP `128→64→BN→ReLU→Dropout(0.3)→2` |
| cv manifest | 算法同、**未落盘** | 显式 `--cv_split_file=GBCU/cv_splits/cv3_seed42_binary.json` |
| 数据增强 | 无 | 无 (`train_hflip_prob=0`) |
| λ_cls / λ_recon | 1.0 / 0.3 | 1.0 / 0.3 |
| 入口脚本 | 推断 CLI `--method all` | `roi_repr/run_ae_cv3.slurm` |
| test_bal_acc (AE) | **0.807 ± 0.027** | **0.731 ± 0.015** |

**AE test 指标不可直接与 `ae_cv3_fixed` 比高低**，分类头结构变更会显著改变决策边界。

---

## 2. 各方法分项（路径与 Job ID）

### 2.1 `ae_cv3`

- **结果：** `outputs/roi_repr/binary/ae_cv3/cv_results.json`（2026-06-25 16:56）
- **Checkpoint 证据：** `fold0/checkpoints/best_model.pth` → `classifier.fc.weight` shape `(2, 128)`
- **Job ID：** 未知

### 2.2 `cls_cv3`

- **结果：** `outputs/roi_repr/binary/cls_cv3/cv_results.json`（2026-06-25 16:20）
- **Checkpoint 证据：** `classifier.fc.*`（线性头）
- **Job ID：** 未知

### 2.3 `vae_cv3`

- **结果：** `outputs/roi_repr/binary/vae_cv3/cv_results.json`（2026-06-25 17:29）
- **额外损失：** `beta_kl=1e-5`
- **Job ID：** 未知

### 2.4 `siamese_cv3`

- **结果：** `outputs/roi_repr/binary/siamese_cv3/cv_results.json`（2026-06-25 17:36）
- **额外：** `lambda_contrastive=0.5`, `margin=1.0`, `cls_on_both_images=True`
- **Job ID：** 未知

### 2.5 `triplet_cv3`

- **结果：** `outputs/roi_repr/binary/triplet_cv3/cv_results.json`（2026-06-25 17:47）
- **额外：** `lambda_triplet=0.5`, `margin=1.0`
- **Job ID：** 未知

---

## 3. 旧 YOLO One-Step 结果

### 3.1 标识与输出

| 项目 | 值 |
|------|-----|
| **主目录** | `outputs/yolov11_onestep_run1/` |
| **汇总** | `outputs/yolov11_onestep_run1/pipeline_summary.json`（2026-05-11） |
| **配置** | `outputs/yolov11_onestep_run1/run_config.json` |
| **权重** | `outputs/yolov11_onestep_run1/weights/best.pt` |
| **训练入口** | `python train_yolov11_onestep.py`（默认 `n_cv_folds=1`，走 `run_legacy_experiment`） |
| **Job ID** | 未知 |

### 3.2 划分协议（legacy 模式）

```python
# train_yolov11_onestep.py → prepare_yolo_dataset()
train_items = train.txt      # 官方 train
val_items   = test.txt       # ⚠️ 官方 test 用作 YOLO 训练期 val
test_items  = test.txt       # 最终评估也是 test
```

- **关键问题：** Ultralytics 根据 **val** 指标保存 `best.pt`，因此 **官方 test 参与了 YOLO 模型选择**
- **无** 3-fold CV；与 roi_repr 的 fold_val 协议 **不对齐**

### 3.3 超参与增强

| 项目 | 值 |
|------|-----|
| model | yolo11n.pt |
| task_mode | detect_cls（2 类：non_malignant / malignant） |
| epochs / batch / imgsz | 100 / 16 / 800 |
| lr / momentum / wd | 0.005 / 0.9 / 5e-4 |
| seed | 42 |
| 增强 | **仅 hflip=0.5**；mosaic/mixup/scale/rotation=0 |
| checkpoint | Ultralytics 内置 best（`patience=100` ≈ 无 early stop） |

### 3.4 Test 指标（122 张官方 test）

| 指标 | 检测器 | 分类器 |
|------|--------|--------|
| 主要数字 | mIoU=0.533, P=0.949, R=0.770 | acc=**0.844**, mal_sens=**0.881**, mal_spec=0.825 |

### 3.5 与 `yolov11_onestep_cv3` 对比

| 维度 | 旧 `run1` | 新 `yolov11_onestep_cv3` (Job **6851445**) |
|------|-----------|-------------------------------------------|
| CV | 无 | 3-fold + 固定 manifest |
| 训练期 val | **test.txt（泄漏）** | 各 fold 的 fold_val |
| test 选模 | **是（间接）** | 否 |
| cls test_acc 均值 | 0.844（单次） | **0.787 ± 0.020** |
| 失败作业 | — | Job **6824104**、**6837297**（Python 3.9 不支持 `Path \| str` 类型注解，后修复） |

---

## 4. 旧 YOLO + AE Two-Step 结果

### 4.1 标识与输出

| 项目 | 值 |
|------|-----|
| **主汇总** | `outputs/yolo_roi_repr_pipeline_single_maxbox/all_methods_summary_test.json`（2026-07-07） |
| **评估入口** | `python -m roi_repr.eval_yolo_roi_pipeline`（推断；见 [`PROJECT_LAYOUT.md`](PROJECT_LAYOUT.md)） |
| **YOLO 权重** | **单个** `outputs/yolov11_onestep_run1/weights/best.pt`（非 per-fold） |
| **分类器权重** | 旧 `outputs/roi_repr/binary/<method>_cv3/fold{k}/checkpoints/best_model.pth` |
| **Job ID** | 未知 |

### 4.2 协议要点

1. **Stage 1 检测：** 旧 YOLO `run1`（test 泄漏训练）
2. **Stage 2 分类：** 在 **GT ROI** 上训练的旧 `*_cv3` 线性头模型
3. **推理：** YOLO 检框 → `largest_area` 单 ROI → 分类器；无框则 `full_image` fallback
4. **评估：** 官方 **test** 集；按 3-fold 分别加载对应 fold 分类器，但 **共用同一个 YOLO**
5. **fold 与 YOLO 不匹配：** 检测器未按 fold 训练，却按 fold 报告分类器 — **协议不完整**

### 4.3 AE two-step 指标（旧）

来源：`all_methods_summary_test.json` → methods.ae.cv_summary

| 指标 | 均值 ± 标准差 |
|------|---------------|
| accuracy | 0.795 ± — |
| balanced_accuracy | **0.774** |
| malignant_sensitivity | 0.706 |
| malignant_specificity | 0.842 |

（五方法均有对应 `outputs/yolo_roi_repr_pipeline_single_maxbox/<method>_cv3/` 子目录）

### 4.4 与新 two-step 对比

| 维度 | 旧 `yolo_roi_repr_pipeline_single_maxbox` | 新 `fixed_cv3/yolo_ae_two_step` (Job **6852178**) |
|------|---------------------------------------------|---------------------------------------------------|
| YOLO | 单个 `run1` | **per-fold** `yolov11_onestep_cv3/fold{k}` |
| 分类器 | 旧 `ae_cv3`（线性头） | `ae_cv3_fixed`（MLP 头） |
| manifest | 无显式记录 | `cv3_seed42_binary.json` |
| AE bal_acc | ~0.774 | **0.746 ± 0.036** |

---

## 5. 可公平比较性矩阵

### ✅ 可以公平比较

| 比较组 | 理由 |
|--------|------|
| **旧五方法之间**（cls / ae / vae / siamese / triplet `@ *_cv3`） | 同一日期、同一入口、同一划分算法、同一线性头、同一无增强协议、同一选模规则 |
| **`ae_cv3_fixed` vs `cls_cv3_fixed`** | 同 manifest、同 MLP 头时代、同无增强（Job 6824103 / 6851444） |
| **`yolov11_onestep_cv3` vs `fixed_cv3/yolo_ae_two_step`** | 共享 manifest 与 per-fold YOLO |
| **旧五方法 GT ROI test 指标之间排序** | 可用于方法学相对比较（在旧协议下） |

### ❌ 不应公平比较（或需显式注明差异）

| 比较 | 原因 |
|------|------|
| **旧 `ae_cv3` vs `ae_cv3_fixed`** | 线性头 vs MLP 头；指标差异可能来自架构而非方法本身 |
| **旧 `cls_cv3` vs `cls_cv3_fixed`** | 同上 |
| **旧五方法 vs `ae_cv3_fixed`** | 分类头世代不同 |
| **旧 GT ROI `*_cv3` vs 旧 YOLO `run1`** | 输入不同（GT crop vs 全图检测）；YOLO 有 test 泄漏 |
| **旧 YOLO `run1` vs `yolov11_onestep_cv3`** | test 是否参与选模不同；单次 vs 3-fold CV |
| **旧 two-step vs 新 `fixed_cv3/yolo_ae_two_step`** | YOLO 单模型 vs per-fold；线性头 vs MLP；旧 YOLO 有泄漏 |
| **旧 `ae_cv3` vs 旧 two-step AE** | 前者 GT ROI pipeline，后者 YOLO 框 + 旧分类器；任务难度不同 |
| **任何 roi_repr 结果 vs `ae_cv3_hflip` 等增强实验** | 增强策略不同 |

---

## 6. 证据索引

### 6.1 结果文件

```
outputs/roi_repr/binary/all_methods_cv3.json
outputs/roi_repr/binary/{cls,ae,vae,siamese,triplet}_cv3/cv_results.json
outputs/roi_repr/binary/{method}_cv3/fold{0,1,2}/results.json
outputs/roi_repr/binary/{method}_cv3/fold{0,1,2}/checkpoints/best_model.pth
outputs/yolov11_onestep_run1/pipeline_summary.json
outputs/yolov11_onestep_run1/run_config.json
outputs/yolo_roi_repr_pipeline_single_maxbox/all_methods_summary_test.json
```

### 6.2 日志与 Job ID

| Job ID | 日志文件 | 用途（非本次旧五方法，但为相关对照） |
|--------|----------|--------------------------------------|
| **未知** | — | 2026-06-25 五方法 `*_cv3` |
| **6797734** | `logs/roi_repr_6797734.out` | `ae_cv3_hflip`（增强消融，非旧 baseline） |
| **6824103** | `logs/ae_cv3_6824103.out` | `ae_cv3_fixed` |
| **6851444** | `logs/cls_cv3_fixed_6851444.out` | `cls_cv3_fixed` |
| **6824104** | `logs/yolo_cv3_6824104.err` | YOLO cv3 失败（类型注解） |
| **6837297** | `logs/yolo_cv3_6837297.err` | YOLO cv3 失败（同上） |
| **6851445** | `logs/yolo_cv3_6851445.out` | `yolov11_onestep_cv3` 成功 |
| **6852178** | `logs/yolo_roi_pipeline_6852178.out` | 新 two-step pipeline |

### 6.3 代码与配置

- 训练入口：`roi_repr/train.py` → `roi_repr/training/trainer.py`
- 默认 Slurm：`train_roi_repr.slurm`（根目录）、`roi_repr/train_roi_repr.slurm`
- 固定协议 Slurm：`roi_repr/run_ae_cv3.slurm`、`run_yolov11_onestep_cv3.slurm`、`run_yolo_roi_pipeline.slurm`
- CV manifest：`GBCU/cv_splits/cv3_seed42_binary.json`（2026-07-24 落盘）
- 分类头：`roi_repr/models/heads.py`
- YOLO legacy val=test：`train_yolov11_onestep.py` → `prepare_yolo_dataset()`

### 6.4 历史对话

- [MLP 头与 cls_cv3 对照](a417b9c5-c2b3-454b-99e7-27bc49efc7e0)：确认旧 `cls_cv3` 为线性头；记录无增强事实；记录 fixed cv3 改造时间线

---

## 7. 建议（审计结论，非执行项）

1. **论文/报告中引用旧五方法数字时**，应注明：**线性分类头、无数据增强、CV 划分算法与现 manifest 一致但未当时落盘**。
2. **与 `ae_cv3_fixed` 对比时**，应视为 **协议变更后的重跑**，不可当作同一实验的微调。
3. **旧 YOLO `run1` 数字** 应标注 **test 用作训练期 validation**；不宜与严格 hold-out 的 `yolov11_onestep_cv3` 直接比。
4. **旧 two-step** 应视为探索性 pipeline；正式对比请用 `outputs/fixed_cv3/yolo_ae_two_step/`。
5. 若需复现旧五方法 exact 数字，除恢复 **线性 `ClassifierHead`** 外，其余条件（seed=42、无增强、λ_recon=0.3 等）已与现配置对齐。

---

*本报告由 Cursor _agent 根据仓库内可追溯材料自动生成。*
