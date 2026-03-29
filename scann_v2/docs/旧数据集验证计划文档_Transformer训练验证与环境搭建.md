# 旧数据集验证计划文档
## 项目名称
基于旧版天文告警裁块数据集的 Transformer 训练验证与实验环境搭建计划

---

## 1. 文档目的
本计划用于在新数据集和细粒度多分类标注完成前，复用原 CNN 时代旧数据集，快速完成以下目标：

1. 验证 Transformer / ViT 类模型在现有业务数据上的可训练性与可收敛性。
2. 搭建统一的数据读取、预处理、训练、验证、测试与日志记录环境。
3. 形成可复用的实验脚手架，为后续切换到新 8 类数据集做准备。
4. 为后续加入量化、压缩、PolarQuant-inspired 模块预留接口。

本阶段目标是“技术路线验证”和“实验环境搭建”，不是最终论文主实验。

---

## 2. 数据集说明

### 2.1 数据来源
使用原 CNN 项目中已经分割好的裁块数据。

### 2.2 数据形式
每个样本由一张对齐图组成：
- new
- old
- diff

单张裁块尺寸：
- 240 x 80 像素

### 2.3 标签体系
当前仅保留二分类标签：
- Real（真目标）
- Bogus（伪目标）

### 2.4 数据规模
当前总量约：
- 5245 张（2943真+2302假）

统计以下信息并固化到数据说明文件：
- 总样本数
- Real 样本数
- Bogus 样本数
- 类别比例
- 是否存在重复样本
- 是否存在同一观测目标跨集合泄漏风险

---

## 3. 本阶段研究问题

旧数据集阶段需要回答的核心问题如下：

1. Transformer 模型能否在 240 x 80 的小尺寸天文裁块上稳定收敛？
2. new / old / diff 三路信息采用哪种融合方式更合适？
3. Transformer 相比原 CNN 基线是否具有性能优势或至少可比？
4. 在旧数据集上，标准量化或轻量压缩模块是否能顺利插入？
5. 整套实验环境是否可迁移到后续新数据集？

---

## 4. 阶段目标与交付物

### 4.1 阶段目标
完成一个可运行的旧数据集验证版本，至少支持以下能力：
- 数据加载
- 训练/验证/测试
- 多模型切换
- 指标记录
- 混淆矩阵输出
- 日志保存
- 实验配置化

### 4.2 阶段交付物
应至少输出以下文件或目录：
- 数据集清单文件
- 训练配置文件
- 模型定义文件
- 训练脚本
- 评估脚本
- 实验日志
- 结果表格
- 可视化图表

---

## 5. 数据组织规范

### 5.1 推荐目录结构
```text
dataset/
├── train/
│   ├── real/
│   └── bogus/
├── val/
│   ├── real/
│   └── bogus/
├── test/
│   ├── real/
│   └── bogus/
```

### 5.2 推荐数据划分
固定划分，不允许不同实验更换划分：
- Train: 70%
- Val: 15%
- Test: 15%

### 5.3 必须检查的事项
在正式训练前必须完成：
1. 去重检查
2. 数据泄漏检查
3. 类别比例统计
4. 图像读取完整性检查
5. new/old/diff 三路对应关系检查

---

## 6. 输入方案设计

由于原始数据为三路图像，本阶段需要验证多种输入方式。

### 6.1 输入方案 A：三路拼为 3 通道
将：
- new -> channel 1
- old -> channel 2
- diff -> channel 3

形成一个 80 x 80 x 3 输入张量。

用途：
- 最快验证
- 最容易接入预训练视觉模型

优先级：
- 最高，必须实现

### 6.2 输入方案 B：仅使用 diff
只输入 diff 图像，作为单通道或复制成 3 通道。

用途：
- 验证 diff 单独的有效性
- 作为对照组

优先级：
- 高

### 6.3 输入方案 C：双路组合
例如：
- diff + new
- diff + old

用途：
- 验证不同信息组合对分类的影响

优先级：
- 中

### 6.4 输入方案 D：多分支结构
分别编码 new / old / diff，再做特征融合。

用途：
- 为后续更复杂模型预留路线

优先级：
- 后续可选，不作为第一阶段必须项

---

## 7. 预处理方案

### 7.1 基础预处理
必须统一：
- 读取方式一致
- 像素归一化方式一致
- 缺失值处理一致

### 7.2 归一化方案
建议支持以下两种：
1. min-max 归一化
2. z-score 标准化

如果历史项目已有固定做法，应优先保持一致。

### 7.3 尺寸处理
原始尺寸为 240 x 80，建议支持以下三种策略：
1. 保持原尺寸直接输入
2. resize 到统一尺寸，例如 224 x 224
3. padding 后 resize

说明：
- 标准 ViT 更适合规则尺寸
- 但过度拉伸可能改变目标形态
- 因此必须做输入尺寸对比实验

### 7.4 数据增强
建议支持但不要过强，避免破坏天文告警结构：
- Horizontal flip（谨慎）
- Vertical flip（谨慎）
- 小角度旋转
- 轻微高斯噪声
- 亮度/对比度微调

必须避免：
- 大幅随机裁剪
- 强颜色增强
- 改变真实物理结构的几何变换

---

## 8. 模型验证范围

本阶段不建议直接上 ViT-Huge，应优先使用轻量模型完成技术验证。

### 8.1 基线模型
#### Model-B1：原 CNN 基线
目的：
- 复现旧模型性能
- 作为 Transformer 对照

#### Model-B2：简单 ResNet / ConvNet
目的：
- 提供一个现代卷积基线

### 8.2 Transformer 候选模型
#### Model-T1：ViT-Tiny / ViT-Small
目的：
- 验证标准 ViT 可行性

#### Model-T2：DeiT-Tiny / DeiT-Small
目的：
- 在小数据条件下更稳定

#### Model-T3：Swin-Tiny
目的：
- 验证分层 Transformer 对小尺寸局部结构的适应性

#### Model-T4：ConvStem + Transformer
目的：
- 验证卷积前端是否能提升小数据和小目标稳定性

### 8.3 第一阶段建议优先级
建议实现顺序：
1. 原 CNN
2. DeiT-Tiny 或 ViT-Tiny
3. Swin-Tiny
4. ConvStem Transformer

---

## 9. 实验设计总览

本阶段实验按以下顺序推进。

### Exp-1：数据链路验证实验
目的：
- 验证数据读取与标签正确性
- 验证训练流程无报错

完成状态：
- 已完成（2026-03-29）

实际执行：
- 基于固定 manifest `scann_v2/experiments/manifests/legacy_v1_manifest.json` 进行数据链路检查
- 随机抽样生成 8 个旧数据集三联图样本可视化
- 使用 `legacy_resnet18_smoke.json` 完成 3 epoch 冒烟训练

结果记录：
- 数据成功读入，train / val / test 抽样 tensor shape 均为 `(3, 224, 224)`
- 抽样检查未发现 shape 错误
- 抽样检查中 `label` 与 manifest 内 `label_name` 不一致数量为 `0`
- smoke train loss 下降：`0.5052 -> 0.2756 -> 0.2011`
- smoke val loss 下降：`0.8280 -> 0.2250 -> 0.2088`
- 训练、验证、测试流程均已跑通，无报错

输出文件：
- `scann_v2/experiments/results/smoke_test_log.txt`
- `scann_v2/experiments/plots/sample_visualization.png`
- `scann_v2/experiments/plots/legacy_resnet18_smoke_learning_curves.png`

---

### Exp-2：CNN 基线复现实验
目的：
- 获得旧数据集上的可靠基准性能

完成状态：
- 已完成 3 次重复实验与均值统计（2026-03-29）

实际执行：
- 使用 `ResNet18` 作为接近原 CNN 路线的现代卷积基线
- 固定使用 manifest 划分：`train=3671 / val=786 / test=786`
- 使用 `new_old_diff` 作为三路 3 通道输入，图像尺寸 `224`
- 固定训练配置，仅改变随机种子，完成 `seed=42 / 43 / 44` 三轮训练

当前结果：
- Accuracy：`0.9402 ± 0.0079`
- Precision：`0.9270 ± 0.0227`
- Recall：`0.9382 ± 0.0102`
- F1-score：`0.9324 ± 0.0079`
- ROC-AUC：`0.9877 ± 0.0013`
- 三轮单独 F1：`0.9343 / 0.9391 / 0.9237`
- Confusion Matrix：已汇总为三轮并排对照图
- 参数量：`11,177,538`
- 当前环境下单张平均推理时间：`0.247 ms / image`

输出文件：
- `scann_v2/experiments/results/baseline_cnn_results.csv`
- `scann_v2/experiments/plots/baseline_cnn_confusion_matrix.png`
- `scann_v2/experiments/results/legacy_resnet18_baseline_summary.json`
- `scann_v2/experiments/results/legacy_resnet18_baseline_seed43_summary.json`
- `scann_v2/experiments/results/legacy_resnet18_baseline_seed44_summary.json`

---

### Exp-3：Transformer 初始可行性实验
目的：
- 验证 Transformer 是否能在旧数据上稳定收敛

完成状态：
- 已完成初始可行性验证

实际执行：
- 当前先使用 `ViT_B_16` 进行旧数据集 Transformer 验证
- 输入方案采用 `new_old_diff`，与方案 A 的三路 3 通道语义一致
- 已完成两组对照：
  - scratch：`pretrained=False`
  - pretrained：`pretrained=True`

当前结果：
- `ViT_B_16 scratch`
  - Accuracy：`0.9071`
  - Precision：`0.8931`
  - Recall：`0.8957`
  - F1-score：`0.8944`
  - ROC-AUC：`0.9703`
  - 收敛轮次：`50 epoch`，最佳轮次 `40`
- `ViT_B_16 pretrained`
  - Accuracy：`0.9542`
  - Precision：`0.9402`
  - Recall：`0.9565`
  - F1-score：`0.9483`
  - ROC-AUC：`0.9902`
  - 收敛轮次：`25 epoch` 早停，最佳轮次 `15`
- 结论：Transformer 在旧数据上可以稳定收敛；从零训练明显弱于 CNN，但加载预训练权重后已超过当前 `ResNet18` 基线
- 过拟合判断：scratch 版本泛化较弱；pretrained 版本收敛更快、验证与测试表现更稳定

输出文件：
- `scann_v2/experiments/results/transformer_baseline_results.csv`
- `scann_v2/experiments/plots/transformer_learning_curve.png`
- `scann_v2/experiments/results/legacy_vit_b16_pretrained_gpu_summary.json`

---

### Exp-4：输入融合方式对比实验
目的：
- 找到最适合旧数据集的输入组织方式

实验组：
1. new + old + diff 三通道
2. 仅 diff
3. diff + new
4. diff + old

保持：
- 同一模型
- 同一训练设置
- 同一数据划分

记录指标：
- Accuracy
- Recall
- F1-score
- ROC-AUC

输出：
- input_fusion_comparison.csv

---

### Exp-5：输入尺寸与预处理对比实验
目的：
- 找到适合 Transformer 的输入尺寸策略

实验组建议：
1. 原始 240 x 80
2. resize 到 224 x 224
3. padding 到近似方形后 resize
4. 不同归一化方式对比

记录指标：
- Accuracy
- F1-score
- 收敛速度
- 显存占用

输出：
- preprocessing_comparison.csv

---

### Exp-6：预训练与非预训练对比实验
目的：
- 验证迁移学习在旧数据小样本场景中的必要性

实验组：
1. pretrained = True
2. pretrained = False

记录指标：
- Accuracy
- Recall
- F1-score
- 最终 loss
- 收敛轮次

输出：
- pretrained_vs_scratch.csv

---

### Exp-7：模型规模对比实验
目的：
- 找到适合旧数据规模的模型容量

实验组：
1. Tiny
2. Small
3. Base（如资源允许）

记录指标：
- Accuracy
- F1-score
- 参数量
- GPU 占用
- 训练时间
- 过拟合程度

输出：
- model_scale_comparison.csv

---

### Exp-8：轻量量化/压缩可插拔验证实验
目的：
- 为后续 PolarQuant-inspired 方法预验证接口可用性

第一阶段不要求完整实现复杂压缩，只需验证以下两类：
1. 标准 INT8 / INT4 推理量化
2. 自定义量化层接口可插入

记录指标：
- 是否可正常运行
- 推理精度变化
- GPU/CPU 推理时间
- 显存占用变化

输出：
- quantization_smoke_results.csv

---

## 10. 实验优先级

### P0（必须完成）
1. 数据链路验证
2. CNN 基线复现
3. Transformer 初始可行性
4. 输入融合对比

### P1（建议完成）
5. 输入尺寸与预处理对比
6. 预训练与非预训练对比
7. 模型规模对比

### P2（后续衔接）
8. 轻量量化/压缩可插拔验证

---

## 11. 统一训练配置要求

为了保证结果可比，以下项目必须统一：

- 固定随机种子
- 固定数据划分
- 固定 epoch 数
- 固定 early stopping 策略
- 固定评价脚本
- 固定最佳模型保存规则

建议统一记录：
- seed
- batch size
- learning rate
- optimizer
- scheduler
- image size
- augmentation 配置
- pretrained 开关
- model name

---

## 12. 建议训练配置初稿

以下为建议初始配置，可根据显存调整。

### CNN / Transformer 统一建议
- Optimizer: AdamW
- Initial LR: 1e-4 ~ 3e-4
- Weight Decay: 1e-4
- Batch Size: 16 / 32
- Epoch: 50 ~ 100
- Early Stopping Patience: 10
- Loss: CrossEntropyLoss
- Scheduler: CosineAnnealing 或 StepLR

对于类别不均衡情况，建议支持：
- class weight
或
- focal loss（后续可选）

---

## 13. 指标与记录规范

### 13.1 主指标
本项目以二分类为主，建议以下指标全部记录：
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

其中建议重点关注：
- Recall
- F1-score

原因：
- 天文告警任务通常更关心漏检情况
- 单纯 Accuracy 容易受类别比例影响

### 13.2 辅助指标
- Train loss
- Val loss
- 参数量
- 单张推理时间
- 每 epoch 训练时间
- GPU peak memory

### 13.3 必须产出的可视化
- Train/Val loss 曲线
- ROC 曲线
- Confusion Matrix
- 不同输入方案柱状图
- 不同模型对比表

---

## 14. 实验日志格式规范

每次实验必须保存一条结构化记录，建议 CSV 字段如下：

```csv
experiment_name,model_name,input_mode,image_size,pretrained,seed,batch_size,lr,epochs,best_val_acc,test_acc,precision,recall,f1,roc_auc,params,peak_gpu_memory,inference_time
```

同时保存完整配置文件，例如：
- config_exp3_vit_tiny.yaml

---

## 15. 推荐工程目录结构

```text
project/
├── configs/
│   ├── data/
│   ├── model/
│   └── experiment/
├── datasets/
│   ├── legacy_binary_dataset.py
│   └── transforms.py
├── models/
│   ├── cnn_baseline.py
│   ├── vit_wrapper.py
│   ├── deit_wrapper.py
│   └── swin_wrapper.py
├── trainers/
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── experiments/
│   ├── exp1_smoke_test.py
│   ├── exp2_cnn_baseline.py
│   ├── exp3_transformer_baseline.py
│   ├── exp4_input_fusion.py
│   ├── exp5_preprocessing.py
│   ├── exp6_pretrained_vs_scratch.py
│   ├── exp7_model_scale.py
│   └── exp8_quant_smoke.py
├── logs/
├── checkpoints/
├── results/
│   ├── tables/
│   └── figures/
└── README.md
```

---

## 16. 实现要求清单（交给 IDE/实现人员）

实现人员需要保证以下能力：

### 数据层
- 支持 CSV 索引读取
- 支持 new/old/diff 三路配对加载
- 支持多种输入模式切换
- 支持 train/val/test 固定划分

### 模型层
- 支持 CNN 基线
- 支持至少一种 ViT/DeiT
- 支持模型配置化加载
- 支持预训练权重开关

### 训练层
- 支持单卡训练
- 支持日志打印
- 支持 TensorBoard 或 CSV Logger
- 支持保存 best model
- 支持 early stopping

### 评估层
- 输出主指标
- 输出混淆矩阵
- 输出 ROC 曲线
- 输出结果表格

### 工程层
- 支持 YAML 配置驱动
- 支持命令行指定实验名
- 支持随机种子设置
- 支持自动保存实验目录

---

## 17. 本阶段判定标准

若满足以下条件，则认为旧数据集验证阶段成功：

1. 旧数据集可稳定训练 CNN 和至少一种 Transformer 模型。
2. Transformer 模型在验证集和测试集上可得到稳定结果。
3. 已完成至少一种输入融合方式对比。
4. 已明确旧数据集上推荐的输入方式和预处理策略。
5. 实验环境可直接迁移到新数据集，仅需替换数据读取层和分类头。
6. 已预留量化/压缩接口，可进入下一阶段方法实验。

---

## 18. 下一阶段衔接说明

当新数据集和 8 类标注完成后，本阶段产物将直接复用：
- 数据加载框架
- 模型训练框架
- 日志与结果记录框架
- Transformer 基线
- 预处理策略
- 输入融合策略

后续新增内容主要包括：
- 多分类头替换
- 新类别不平衡处理
- 更高分辨率输入支持
- PolarQuant-inspired 压缩模块正式接入
- 面向论文的主实验扩展

---

## 19. 建议执行顺序（简版）

第 1 周：
- 完成数据清点、索引生成、冒烟测试

第 2 周：
- 复现 CNN 基线
- 跑通 Transformer baseline

第 3 周：
- 完成输入融合对比
- 完成输入尺寸与预处理对比

第 4 周：
- 完成预训练对比与模型规模对比
- 预留量化模块接口

---

## 20. 一句话总结
旧数据集阶段的核心任务不是追求最终论文结果，而是：
“尽快验证 Transformer 在现有业务裁块数据上的可行性，并搭建一套能直接迁移到新数据集的实验环境。”
