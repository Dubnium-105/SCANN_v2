# SCANN v2 全图检测（Vision Transformer）架构设计

> 实现状态（2026-03-13）：
> - ✅ `InferenceEngine.detect_dense_full_image()` 已接入并用于 dense 推理解码
> - ✅ `DetectionPipeline` 已支持 `patch | full_image | hybrid` 三种模式
> - ✅ `hybrid` 回退路径与日志已落地（模型不可用/异常/低置信回退）
> - ✅ 训练链路已支持 dense detection 最小闭环（heatmap+bbox 监督，focal+bbox 损失）

## 1. 目标与约束

- 目标：在 v2 的新旧 FITS 对图上进行**全图候选体检测**，减少仅靠滑窗分类带来的召回损失。
- 约束：
  - 推理显存上限约 8GB（消费级 GPU）
  - 兼容现有管线 `DetectionPipeline`
  - 保持可回退到当前 patch 分类路径
  - 输入以单通道 FITS 为主，需支持多通道拼接策略

## 2. 总体架构

采用 **ViT 编码器 + 密集检测头** 的两阶段轻量实现：

1. 输入构建：`new`、`old`、`diff=abs(new-old)` 形成 3 通道张量
2. Patch Embedding：`Conv2d(kernel=stride=16)` 映射到 token 序列
3. ViT Encoder：多层自注意力建模全局上下文
4. Dense Head：
   - `bbox_head` 预测 `[dx, dy, w, h]`
   - `heatmap_head` 预测中心置信度
5. 解码后处理：top-k + 阈值过滤 + NMS + 已知天体排除

> 当前代码中的 `SCANNDetector` 已按该形态实现为骨架：
> - `forward_dense()` 输出 `[B,5,Hp,Wp]`
> - `forward()` 保持兼容，返回全局池化后的 `[B,5]`

## 3. 模块拆分建议（对齐现有目录）

### 3.1 `src/scann/ai/model.py`

- `SCANNDetector`
  - 输入适配：`in_channels -> 3`
  - 编码器：`patch_embed + TransformerEncoder + LayerNorm`
  - 检测头：`bbox_head + heatmap_head`
  - 辅助接口：`estimate_memory_mb()`

### 3.2 `src/scann/ai/inference.py`

全图检测接口（已实现）

- `detect_dense_full_image(new_data, old_data, score_threshold, top_k, iou_threshold) -> list[Detection]`
  - 预处理（归一化、通道拼接）
  - 调用 `model.forward_dense()`
  - 解码 dense 输出并执行 NMS

### 3.3 `src/scann/services/detection_pipeline.py`

管线分支建议：

- 若模型支持 `forward_dense`：
  - 优先走全图检测分支
- 否则：
  - 保留现有 `CV 候选 + patch 分类 + 滑窗补召回` 逻辑

这样可以实现渐进替换，不影响旧模型与历史数据。

## 4. 训练设计

## 4.1 标签表达

面向全图检测，推荐热力图中心点 + 尺寸回归：

- `heatmap`: 每类目标中心点高斯热力图（单类可 1 通道）
- `bbox`: 对中心点回归 `(w, h)` 或 `(l,t,r,b)`
- 可选 `offset`: 亚像素偏移，提升定位精度

## 4.2 损失函数

- `L = λ1 * focal_loss(heatmap) + λ2 * IoU/GIoU(bbox) + λ3 * L1(offset)`
- 建议初始权重：`λ1=1.0, λ2=2.0, λ3=1.0`

## 4.3 数据增强

- 几何：随机翻转、90°旋转
- 光度：对比度拉伸、噪声扰动（贴近 FITS 背景）
- 天文特化：模拟条带、热像素、不同 seeing 的 PSF 模糊

## 5. 推理策略与性能

## 5.1 推理流程

1. 对齐（沿用现有 `align`）
2. 构造 3 通道输入
3. 全图前向（必要时 tiled inference）
4. 阈值筛选 + NMS
5. exclusion service 排除已知天体

## 5.2 推理显存控制

- 默认 patch size=16，降低 token 数
- 启用 AMP（混合精度）
- 对超大图（如 4k）采用 overlap tile（例如 1024/1536）
- 批次大小对全图检测通常固定 1

## 6. 与现有 v2 的兼容策略

- 模型层：保留 `SCANNClassifier`（v1/v2 patch 分类兼容）
- 管线层：新增 full-image 分支，不删除旧分支
- 配置层：增加 `detection_mode = patch|full_image|hybrid`
- 回滚策略：若 full-image 模型加载失败，自动回退 patch 模式

## 6.1 运行示例与关键参数

### 配置示例（`scann_v2_config.json`）

```json
{
  "detection_mode": "hybrid",
  "hybrid_primary_mode": "full_image",
  "hybrid_low_confidence": 0.5,
  "ai_confidence": 0.5
}
```

### 推荐模式选择

- `patch`：兼容优先，完全沿用旧链路
- `full_image`：优先全图 dense 检测，适合显存充足和召回优先场景
- `hybrid`：生产默认建议，先走主模式，失败/低置信自动回退

### dense 推理解码参数（接口级）

- `score_threshold`：热力图置信筛选阈值
- `top_k`：解码保留候选上限
- `iou_threshold`：NMS 重叠抑制阈值

## 7. 里程碑建议

1. M1（已具备骨架）：`SCANNDetector` ViT 密集输出 + 内存估算
2. M2：`InferenceEngine.detect_dense_full_image()` 与 dense 解码
3. M3：`DetectionPipeline` 接入 full-image/hybrid 分支
4. M4：训练脚本与评估指标（Recall@IoU、F1、漏检率）
5. M5：UI 与日志展示全图检测中间信息（可选）

## 8. 风险与缓解

- 小目标极稀疏，正负样本极不平衡 → 使用 focal loss + hard negative mining
- FITS 噪声域变化大 → 强化天文噪声增强与域归一化
- token 数随分辨率平方增长 → tile + overlap + half precision

## 8.1 已知限制（当前版本）

- 当前主路径按单类 real 候选检测设计，多类别检测头未纳入本次范围
- 对 4K 及以上图像，`full_image` 模式可能出现显存占用与时延波动，建议优先 `hybrid`
- 小目标在低信噪比背景中仍可能漏检，需通过阈值调优与数据增强改善召回

## 9. 配套执行文档

- 实现计划：`vit_implementation_plan.md`
- 提交检查单：`vit_commit_checklist.md`

---

该设计遵循当前 SCANN v2 分层架构，可在不破坏现有功能的前提下逐步切换到 ViT 全图检测。