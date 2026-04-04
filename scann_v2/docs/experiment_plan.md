# SCANN v2 实验改进方案

## 1. 文档目标

本文档基于当前仓库的真实实现，重新定义为达到实验要求需要补齐的工程项与实验路线。

当前阶段的总体原则如下：

- 先固化可复现的 baseline，再开展后续改进实验。
- 以 v2 FITS 数据链路为主，不再把 v1 PNG 兼容链路作为主实验对象。
- 当前主任务应定义为 `real/bogus` 二分类与 v2 全图检测。
- `asteroid`、`supernova`、`variable_star`、`satellite_trail`、`noise`、`diffraction_spike`、`cmos_condensation`、`corresponding` 这 8 个细分类型目前更适合作为误差分析和统计维度，而不是直接作为主训练任务。
- 量化和 TurboQuant 相关实验是下一阶段重点，但仓库当前还没有对应实现，需要先补实验基础设施。

> 说明  
> 本文档不再沿用原先以 `PolarQuant` 为中心的假设性设计，而是改为贴合 SCANN v2 当前代码结构的可执行路线图。

## 2. 当前项目现状

| 模块 | 当前状态 | 对实验的意义 |
| --- | --- | --- |
| 数据预处理 | 已有 `dataset_preprocess_service.py`、`scann_dataset.db`、`task_artifacts`，支持新旧图配对、对齐裁剪、任务化管理 | 可以作为 v2 实验数据主链路 |
| 标注体系 | 已支持 `real/bogus` 大类与 8 个细分 `detail_type` | 可以做二分类训练，也可以做细分类统计与误差分析 |
| 分类训练 | `training_worker.py` 已支持 `ResNet18`、`ResNet34`、`ResNet50`、`ViT_B_16` | 可直接开展分类 baseline 实验 |
| 检测训练 | `SCANNDetector` 已支持 v2 dense detection 训练 | 可作为全图检测 baseline |
| 推理 | 已有 patch 分类与 dense full-image detection 推理路径 | 可以做离线推理评测与效率测试 |
| 量化 | 仓库内暂无 PTQ/QAT/低比特推理实现 | 量化实验需要新增模块与脚本 |
| TurboQuant | 仓库内暂无相关实现 | 属于下一阶段预研与实验方向 |

## 3. 当前与实验要求之间的主要差距

### 3.1 任务定义与原草案不一致

原草案默认直接做 8 类分类，但当前训练代码并不是 8 分类训练，而是：

- 分类链路以 `real/bogus` 二分类为主；
- 细分类型通过 `detail_type -> label` 映射参与统计；
- 检测链路当前主要服务于全图目标检测，而不是细分类别识别。

因此，正式实验应拆成两条主线：

1. `real/bogus` 分类 baseline 与量化实验。
2. v2 dense detection baseline 与后续 Transformer 量化迁移实验。

### 3.2 当前划分方式存在数据泄漏风险

这是当前最需要优先修正的问题。

- `training_worker.py` 会先把标注框转成 patch 样本，再对 patch 样本做随机切分。
- 这样会导致同一张图、同一任务甚至同一旧图复用链路上的 patch 同时进入训练集和验证集。
- `fits_annotation_backend.py` 与 `triplet_backend.py` 的导出拆分也只是按顺序切分，不是按任务组、天区组或旧图复用组切分。

对本项目而言，以下分组维度必须纳入正式实验划分规则：

- `task_id`
- `field_key`
- `old_asset_id` 或等价旧图复用组

结论：正式实验必须在 patch 提取之前先完成分组划分，并把划分结果固化为清单文件。

### 3.3 可复现性还不够

目前代码里虽然有 `seed` 概念，但正式训练与导出链路尚未形成完整的可复现约束，至少还缺：

- 固定 train/val/test 清单；
- 固定随机种子；
- 固定输入尺寸、patch 生成规则、增强策略；
- 固定 backbone、epoch、优化器和阈值搜索策略；
- 固定实验日志字段和结果落盘格式。

### 3.4 检测任务评测还不完整

当前 dense detection 训练主要看 `val_loss`，但正式实验应至少补齐：

- `precision`
- `recall`
- `F1` 或 `F2`
- `AP50` 或等价检测指标
- 推理延迟、吞吐、显存/内存占用

### 3.5 量化与 TurboQuant 尚无工程入口

当前仓库没有以下内容：

- 量化模型导出脚本；
- PTQ/QAT 配置；
- 低比特评测脚本；
- 量化后 latency / throughput / model size 统计脚本；
- 面向 Transformer 模块的 TurboQuant 类实验入口。

因此，量化与 TurboQuant 不是“直接跑实验”，而是“先补实验框架，再逐步做实验”。

## 4. 正式实验目标重定义

### 4.1 主实验任务

| 实验主线 | 任务定义 | 当前可行性 | 是否应立即开展 |
| --- | --- | --- | --- |
| 分类 baseline | v2 数据导出的三通道 patch，做 `real/bogus` 二分类 | 高 | 是 |
| 检测 baseline | v2 FITS 全图做 dense detection | 中 | 是 |
| 细分类型统计 | 8 个 `detail_type` 的数据分布、误差分布、召回分布 | 高 | 是 |
| 8 类直接分类 | 把 8 个 `detail_type` 直接作为训练标签 | 低 | 否，作为后续扩展 |
| 量化实验 | 分类模型与检测模型压缩和加速 | 低 | 是，但需先补基础设施 |
| TurboQuant 相关实验 | 面向 Transformer 模块的低比特实验 | 低 | 是，作为第二阶段重点 |

### 4.2 统一数据规则

所有正式实验必须统一遵守以下规则：

1. 只使用 v2 标准化数据链路。
2. 统一以 `scann_dataset.db` 和导出的标注文档作为事实来源。
3. 划分顺序必须是“先按任务分组切分，再做 patch 提取”。
4. 同一个 `field_key` 与共享同一 `old_asset_id` 的任务不能跨 train/val/test。
5. 推荐固定比例为 `70 / 15 / 15`，并生成可复用的 split manifest。
6. 分类、检测、量化实验必须使用同一批 train/val/test 划分。
7. 不得在同一张总表里混用 v1 PNG 数据和 v2 FITS 数据。

## 5. 为达到实验要求必须完成的改进项

| 优先级 | 改进项 | 目的 |
| --- | --- | --- |
| P0 | 新增分组切分脚本，输出 `train/val/test` manifest | 解决数据泄漏与可复现问题 |
| P0 | 新增统一实验配置文件 | 固化 backbone、输入尺寸、增强、seed、epoch |
| P0 | 新增分类评测脚本 | 输出 `accuracy`、`precision`、`recall`、`F1/F2`、`PR-AUC`、混淆矩阵 |
| P0 | 新增检测评测脚本 | 输出 `precision`、`recall`、`F1/F2`、`AP50`、漏检/误检统计 |
| P0 | 新增效率评测脚本 | 统一统计 latency、throughput、模型大小、显存/内存 |
| P1 | 新增量化实验入口 | 支持 FP32 / INT8 / INT4 等对照实验 |
| P1 | 新增量化结果落盘格式 | 让 baseline 与量化实验可横向比较 |
| P1 | 新增 TurboQuant 实验入口 | 面向 ViT 与 Transformer 编码器做低比特实验 |
| P2 | 新增细分类型误差分析脚本 | 输出 8 类细分标签的召回/误检分布 |

## 6. 建议的实验路线

### 6.1 第一阶段：可复现 baseline

目标：先把“当前仓库已经支持的实验”做扎实。

#### A. 分类 baseline

| 项目 | 设计 |
| --- | --- |
| 任务 | `real/bogus` 二分类 |
| 输入 | v2 对齐后的 `new/old` 图像生成的三通道 patch |
| 模型 | `ResNet18`、`ResNet34`、`ResNet50`、`ViT_B_16` |
| 主指标 | `recall`、`F2`、`PR-AUC` |
| 辅指标 | `accuracy`、`precision`、混淆矩阵 |
| 目标 | 得到当前项目最强可复现分类 baseline |

说明：

- 当前代码的训练逻辑天然偏向高召回，因此正式报告中应把 `recall` 和 `F2` 放在主指标位置。
- `detail_type` 用于统计不同子类上的误差分布，而不是直接替代主标签。

#### B. 检测 baseline

| 项目 | 设计 |
| --- | --- |
| 任务 | v2 FITS 全图 dense detection |
| 输入 | `[diff, new, old]` 三通道全图 |
| 模型 | `SCANNDetector` |
| 主指标 | `recall`、`F2`、`AP50` |
| 辅指标 | `precision`、推理时间、显存占用 |
| 目标 | 建立后续 Transformer 量化实验的全图检测基线 |

说明：

- 目前 dense detection 已能训练，但正式实验还需增加检测评测脚本。
- 如果第一阶段时间有限，应优先保证分类 baseline 完整，再补检测 baseline。

### 6.2 第二阶段：标准量化实验

目标：建立项目自己的量化对照组。

推荐实施顺序如下：

1. `ResNet18` 分类模型 PTQ INT8。
2. `ResNet34/ResNet50` 分类模型 PTQ INT8。
3. `ViT_B_16` 分类模型量化实验。
4. `SCANNDetector` 的局部量化或模块级量化实验。

推荐对照组：

| 对照项 | 说明 |
| --- | --- |
| FP32 | 原始模型基线 |
| INT8 | 标准量化 baseline |
| INT6 | 可选，作为低比特过渡 |
| INT4 | 低比特重点对照 |

量化实验必须统一记录：

- `recall`
- `F2`
- `PR-AUC` 或 `AP50`
- 模型文件大小
- 单张推理延迟
- 批量吞吐
- 峰值显存/内存

### 6.3 第三阶段：TurboQuant 相关实验

目标：把 TurboQuant 类低比特实验优先放在项目中最适合的 Transformer 模块上验证可行性。

推荐优先级如下：

1. `ViT_B_16` 分类模型。
2. `SCANNDetector.encoder`。
3. `SCANNDetector` 全模型或 `encoder + heads` 联合实验。

原因：

- `ViT_B_16` 输入固定、评测简单、最适合先验证量化策略是否可行。
- `SCANNDetector` 也包含 Transformer 编码器，但检测评测链更复杂，应在分类验证稳定后再迁移。

TurboQuant 相关实验建议做以下消融：

| 消融方向 | 说明 |
| --- | --- |
| 量化位置消融 | 只量化 `QKV/Linear`、只量化 `MLP`、量化整个 encoder |
| bit-width 消融 | `INT8 / INT6 / INT4` |
| 模块范围消融 | `head only`、`encoder only`、`full model` |
| 任务迁移消融 | 先分类后检测，比较同一策略跨任务稳定性 |

推荐判定标准：

- 若 `recall` 和 `F2` 下降可控，同时 latency 和模型大小明显改善，则该量化策略可以进入下一轮检测实验。
- 若分类任务都无法保持稳定，则不建议直接迁移到 `SCANNDetector`。

## 7. 建议的结果输出规范

每次实验至少输出一份结构化结果文件，推荐字段如下：

```text
run_id
task_type
dataset_version
split_manifest
model_name
backbone
quant_method
bit_width
seed
input_size
patch_size
epochs
precision
recall
f1
f2
pr_auc
ap50
latency_ms
throughput_fps
model_size_mb
peak_memory_mb
notes
```

推荐输出物：

- `results/*.csv`：结构化结果总表
- `results/*.json`：配置与环境快照
- `plots/*.png`：PR 曲线、混淆矩阵、效率对比图
- `manifests/*.json`：train/val/test 划分文件

## 8. 建议的目录补充

结合当前仓库结构，建议新增以下实验目录：

```text
experiments/
|-- manifests/
|-- configs/
|-- results/
|-- plots/
`-- benchmarks/

scripts/
`-- experiments/
    |-- build_split_manifest.py
    |-- train_classifier.py
    |-- eval_classifier.py
    |-- eval_detector.py
    |-- quantize_model.py
    `-- benchmark_inference.py
```

## 9. 近期执行顺序建议

### P0：本周应先完成

1. 固化 v2 数据集 split manifest。
2. 修正训练/验证划分，避免 patch 级随机泄漏。
3. 跑通分类 baseline：`ResNet18/34/50/ViT_B_16`。
4. 补齐统一结果落盘格式。

### P1：随后进入

1. 加入标准量化 baseline。
2. 做 `ResNet` 与 `ViT_B_16` 的 FP32 vs INT8/INT4 对照。
3. 补齐 latency、throughput、模型大小和峰值显存统计。

### P2：量化稳定后开展

1. 开始 TurboQuant 相关实验。
2. 先在 `ViT_B_16` 上做模块级消融。
3. 稳定后迁移到 `SCANNDetector.encoder`。

## 10. 与本文档直接相关的实现入口

当前与本实验方案最相关的代码位置如下：

- `src/scann/services/dataset_preprocess_service.py`
- `src/scann/core/dataset_storage.py`
- `src/scann/core/annotation_models.py`
- `src/scann/ai/dataset.py`
- `src/scann/ai/training_worker.py`
- `src/scann/ai/model.py`
- `src/scann/ai/inference.py`

如果后续实验设计发生变化，应优先更新本文件，再调整训练、评测与量化脚本。

## 11. ViT Full-Resolution Packed-KV 路线补充

截至 2026-04-04，旧数据集实验框架已经补齐一条可运行的 `ViT-B/16` 与 `ViT-H/14` full-resolution packed-KV attention 压缩路线，并完成一轮面向现有 `224 x 224` checkpoint 的重构版兼容复测，当前能力边界如下：

- 支持 `vit_b_16` 与 `vit_h_14` 的 full-resolution square 输入。
- 支持 `PackedKV4Bit` 的真实 `uint8` 打包存储，而不是逻辑量化后立即恢复全精度。
- 支持 blockwise dequantization 与 streaming attention，不再要求整张 attention matrix 常驻内存。
- 支持 `all / first_n / last_n / middle / explicit_indices` 五类 layer selector。
- 支持 `K only`、`V only`、`K/V both` 三类压缩目标。
- 支持 `preserve_cls_token` 与 `qjl_sign_norm` residual correction 作为可选增强分支。
- 已修复 benchmark 串行运行导致的显存峰值污染。
- 已修复 packed-KV 路径先生成整块 dense `K/V` 再压缩的问题，改为按 token block 投影与压缩。
- 已支持导出并重新加载压缩 checkpoint 版本，包括 `custom_int8_weight_only` 与 `packed_int4_weight_only`。

当前推荐配置文件：

- `scann_v2/experiments/configs/legacy_vit_b16_fullres_baseline.json`
- `scann_v2/experiments/configs/legacy_vit_b16_fullres_k4_stream.json`
- `scann_v2/experiments/configs/legacy_vit_b16_fullres_k4_stream_qjl.json`
- `scann_v2/experiments/configs/legacy_vit_b16_fullres_ablation.json`
- `scann_v2/experiments/configs/legacy_vit_h14_fullres_baseline.json`
- `scann_v2/experiments/configs/legacy_vit_h14_fullres_k4_stream.json`
- `scann_v2/experiments/configs/legacy_vit_h14_fullres_k4_stream_qjl.json`
- `scann_v2/experiments/configs/legacy_vit_h14_fullres_ablation.json`

当前推荐脚本入口：

- 单 checkpoint benchmark：`scann_v2/scripts/experiments/benchmark_legacy_checkpoint.py`
- 全模块消融矩阵：`scann_v2/scripts/experiments/benchmark_vit_attention_ablation.py`
- 压缩 checkpoint 导出：`scann_v2/scripts/experiments/export_legacy_compressed_checkpoint.py`
- 轻量回归检查：`scann_v2/scripts/experiments/validate_vit_attention_workflow.py`

当前结果字段补充：

- 单次 benchmark / summary 结果已补充：
  - `peak_gpu_memory_attention_only_mb`
  - `packed_kv_size_mb`
  - `token_count`
  - `num_patched_layers`
  - `residual_mode`
  - `qjl_dim`
  - `streaming_enabled`
  - `materialize_attention_matrix`
- 全模块消融结果已补充：
  - `layer_scope`
  - `kv_target`
  - `cls_policy`
  - `precision_mode`
  - `streaming_enabled`
  - `materialize_attention_matrix`
  - `residual_mode`
  - `qjl_dim`

2026-04-04 兼容复测摘要：

- 使用 `legacy_vit_b16_pretrained_gpu_best.pt` 与 `legacy_vit_b16_exp8_refactored_ablation.json` 对重构版路径做 `224 x 224` 复测。
- `baseline_dense` 与 `all_layers_kv_4bit` 的测试集 F1 都约为 `0.9496`。
- 修复统计口径后，`all_layers_kv_4bit` 的真实 GPU 峰值显存约为 `466.29 MB`，与 baseline 的 `465.66 MB` 基本持平。
- 已导出两组压缩 checkpoint：
  - `custom_int8_weight_only`：约 `145.34 MB`
  - `packed_int4_weight_only`：约 `114.96 MB`

当前推荐先跑顺序：

1. `legacy_vit_b16_fullres_baseline.json`
2. `legacy_vit_b16_fullres_k4_stream.json`
3. `legacy_vit_b16_fullres_k4_stream_qjl.json`
4. `legacy_vit_b16_fullres_ablation.json`
5. 在 ViT-B 路线稳定后再切换 `vit_h_14`

当前明确不在第一阶段内的能力：

- 非 square 的 torchvision ViT 位置编码插值正式支持
- 3-bit 真正 bit-pack
- 训练态 attention dropout 精确等价支持
- 自定义 CUDA kernel
